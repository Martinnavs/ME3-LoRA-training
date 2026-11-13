"""Fine-tune the router (LoRA by default) on the generated data. Run on the cluster after scripts/preflight.py.

    CUDA_VISIBLE_DEVICES=5 python train/train.py --model ibm-granite/granite-4.0-h-1b --data data/generated \
        --out runs/h1b-lora-s0 --seed 0 [--smoke] [--full] [--resume]

--smoke   20 steps on 200 examples: proves the stack (model, kernels, memory) before a long run.
--full    full fine-tuning instead of LoRA (compare after the LoRA baseline).
Loss only on the assistant's final message (the target lines). Examples longer than --max-len are dropped.
Saves the adapter (or model) and a metrics.json with the dev exact match (greedy generation, whole reply).
Needs the `train` dependency group (torch, transformers>=4.56, peft). The Granite 4.0 hybrid (Mamba-2) also wants
the `mamba-ssm` and `causal-conv1d` kernels for speed; without them transformers falls back to a slow path.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import utils as U  # noqa: E402


def load_rows(path: Path, limit: int | None) -> list[dict]:
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    return rows[:limit] if limit else rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="a Hugging Face id or a local directory")
    ap.add_argument("--data", type=Path, required=True, help="directory with train.jsonl, dev.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4, help="LoRA; use about 2e-5 with --full")
    ap.add_argument("--batch", type=int, default=16, help="per-device batch")
    ap.add_argument("--accum", type=int, default=4, help="gradient accumulation steps")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true", help="continue from the latest checkpoint in --out")
    ap.add_argument("--teacher", help="distil from this base model (needs --teacher-adapter); same tokenizer required")
    ap.add_argument("--teacher-adapter", type=Path, help="LoRA adapter directory merged into --teacher")
    ap.add_argument(
        "--teacher-cache",
        type=Path,
        help="top-k teacher logits from scripts/cache_teacher_logits.py: distil without loading the teacher",
    )
    ap.add_argument(
        "--kd-alpha", type=float, default=0.5, help="weight of the teacher KL; the rest is gold cross-entropy"
    )
    ap.add_argument("--kd-temp", type=float, default=2.0)
    ap.add_argument("--eval-n", type=int, default=300, help="dev examples for the generation exact-match")
    a = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

    set_seed(a.seed)
    random.seed(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    limit = 200 if a.smoke else None
    encoded = {}
    for split in ("train", "dev"):
        rows = load_rows(a.data / f"{split}.jsonl", limit)
        enc = [U.encode_example(tok, r["messages"], a.max_len) for r in rows]
        encoded[split] = [{**e, "id": r["id"]} for r, e in zip(rows, enc, strict=True) if e is not None]
        print(
            f"{split}: {len(encoded[split])} examples ({len(rows) - len(encoded[split])} dropped as too long)",
            flush=True,
        )

    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16)
    if not a.full:
        model = get_peft_model(
            model,
            LoraConfig(
                r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM"
            ),
        )
        model.print_trainable_parameters()

    cache = None
    if a.teacher_cache:
        if a.teacher:
            sys.exit("use either --teacher (live) or --teacher-cache, not both")
        cache = torch.load(a.teacher_cache)["entries"]
        missing = [e["id"] for rows_ in encoded.values() for e in rows_ if e["id"] not in cache]
        if missing:
            sys.exit(f"{len(missing)} examples have no cached teacher logits (first: {missing[0]}); rebuild the cache")
    teacher = None
    if a.teacher:
        from peft import PeftModel

        teacher = AutoModelForCausalLM.from_pretrained(a.teacher, torch_dtype=torch.bfloat16)
        teacher = PeftModel.from_pretrained(teacher, str(a.teacher_adapter)).merge_and_unload().cuda().eval()
        for prm in teacher.parameters():
            prm.requires_grad_(False)

    class Data(torch.utils.data.Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            return self.items[i]

    def collate(features):
        batch = U.pad_batch(features, tok.pad_token_id)
        out = {k: torch.tensor(v) for k, v in batch.items()}
        if cache is not None:  # the teacher rows in the same order as the masked student rows (batch, then position)
            out["t_vals"] = torch.cat([cache[f["id"]][0] for f in features])
            out["t_idx"] = torch.cat([cache[f["id"]][1] for f in features])
        return out

    args = TrainingArguments(
        output_dir=str(a.out),
        num_train_epochs=a.epochs,
        max_steps=20 if a.smoke else -1,
        per_device_train_batch_size=a.batch,
        per_device_eval_batch_size=a.batch,
        gradient_accumulation_steps=a.accum,
        learning_rate=a.lr if not a.full else min(a.lr, 2e-5),
        lr_scheduler_type="cosine",
        warmup_steps=0.03,  # float in [0, 1) = ratio of total steps (transformers>=5)
        bf16=True,
        logging_steps=10,
        eval_strategy="no" if a.smoke else "epoch",
        save_strategy="no" if a.smoke else "epoch",
        save_total_limit=2,
        seed=a.seed,
        report_to=[],
        remove_unused_columns=False,
    )

    class KDTrainer(Trainer):
        """Loss on the target tokens only: alpha * KL(teacher || student) at temperature T, plus gold cross-entropy."""

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            import torch.nn.functional as F

            labels = inputs["labels"][:, 1:]
            mask = labels != U.IGNORE
            out = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            student = out.logits[:, :-1][mask].float()
            ce = F.cross_entropy(student, labels[mask])
            t = a.kd_temp
            if cache is not None:
                kd = U.topk_kl(student, inputs["t_vals"], inputs["t_idx"], t)
            else:
                with torch.no_grad():
                    t_out = teacher(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
                    target = t_out.logits[:, :-1][mask].float()
                kd = F.kl_div(F.log_softmax(student / t, -1), F.softmax(target / t, -1), reduction="batchmean") * t * t
            loss = a.kd_alpha * kd + (1 - a.kd_alpha) * ce
            return (loss, out) if return_outputs else loss

    trainer = (KDTrainer if teacher is not None or cache is not None else Trainer)(
        model=model,
        args=args,
        train_dataset=Data(encoded["train"]),
        eval_dataset=Data(encoded["dev"]),
        data_collator=collate,
    )
    start = time.time()
    trainer.train(resume_from_checkpoint=a.resume or None)
    trainer.save_model(str(a.out / "final"))
    tok.save_pretrained(str(a.out / "final"))

    # generation exact match on dev (greedy, the whole multi-line reply must equal the target)
    model.eval()
    dev_rows = load_rows(a.data / "dev.jsonl", 20 if a.smoke else a.eval_n)
    predictions, targets = [], []
    for r in dev_rows:
        prompt = tok.apply_chat_template(r["messages"][:-1], tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=64, do_sample=False, pad_token_id=tok.pad_token_id)
        predictions.append(tok.decode(out[0][ids["input_ids"].shape[1] :], skip_special_tokens=True))
        targets.append(r["target"])
    metrics = {
        "dev_exact": U.exact_match(predictions, targets),
        "dev_n": len(targets),
        "train_seconds": round(time.time() - start),
        "seed": a.seed,
        "model": a.model,
        "full": a.full,
        "smoke": a.smoke,
    }
    (a.out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()

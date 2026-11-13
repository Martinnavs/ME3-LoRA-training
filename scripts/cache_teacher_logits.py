"""Run the distillation teacher once and keep its top-k logits at the reply positions.

    python scripts/cache_teacher_logits.py --teacher ibm-granite/granite-4.0-h-1b \
        --teacher-adapter runs/h1b-lora-web3-s0/final --data data/distill --out data/distill/teacher_top128.pt

train/train.py --teacher-cache <file> then distils without loading the teacher: the loss only reads the teacher at
the reply tokens (about ten per example), so one pass over train and dev replaces a teacher forward at every step of
every epoch. The cache is keyed by example id, so it stays valid for the same data files and tokenizer (the teacher's
tokenizer is the student's: identical vocabulary and chat template). Prints how much of the teacher's probability mass
(at the training temperature) the kept logits cover.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "train"))
import utils as U  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--teacher-adapter", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--topk", type=int, default=128)
    ap.add_argument("--temp", type=float, default=2.0, help="only for the coverage report")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--splits", nargs="+", default=["train", "dev"])
    a = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.teacher)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(a.teacher, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, str(a.teacher_adapter)).merge_and_unload().cuda().eval()

    cache: dict[str, tuple] = {}
    mass: list[float] = []
    for split in a.splits:
        rows = [json.loads(x) for x in (a.data / f"{split}.jsonl").read_text().splitlines() if x.strip()]
        items = [(r["id"], U.encode_example(tok, r["messages"], a.max_len)) for r in rows]
        items = sorted(((i, e) for i, e in items if e is not None), key=lambda x: len(x[1]["input_ids"]))
        for start in range(0, len(items), a.batch):
            chunk = items[start : start + a.batch]
            batch = U.pad_batch([e for _, e in chunk], pad)
            ids = torch.tensor(batch["input_ids"]).cuda()
            att = torch.tensor(batch["attention_mask"]).cuda()
            labels = torch.tensor(batch["labels"])
            with torch.no_grad():
                logits = model(input_ids=ids, attention_mask=att).logits
            for b, (rid, _) in enumerate(chunk):
                keep = (labels[b, 1:] != U.IGNORE).cuda()
                lg = logits[b, :-1][keep].float()
                vals, idx = lg.topk(a.topk, -1)
                mass.extend(F.softmax(lg / a.temp, -1).gather(-1, idx).sum(-1).tolist())
                cache[rid] = (vals.half().cpu(), idx.int().cpu())
            if (start // a.batch) % 100 == 0:
                print(f"{split}: {min(start + a.batch, len(items))}/{len(items)}", flush=True)
        print(f"{split}: {len(items)} examples cached", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"topk": a.topk, "teacher_adapter": str(a.teacher_adapter), "entries": cache}, a.out)
    m = torch.tensor(mass)
    print(
        f"top-{a.topk} covers {m.mean():.5f} of the teacher's mass at T={a.temp} on average (min {m.min():.5f}); "
        f"{len(cache)} examples -> {a.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()

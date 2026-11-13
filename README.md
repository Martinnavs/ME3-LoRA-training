# ME3-LoRA-training

Training code, data generator, evaluation checks and the generated training data for the **router** of the AI-231-ME3 voice
agent: `ibm-granite/granite-4.0-h-350m` + a LoRA adapter, distilled from a 1B teacher. The aim of this repository is
transparency and reproducibility: everything needed to regenerate the data, retrain the adapter and re-run the checks.

| | |
| --- | --- |
| **Model** (adapter, public) | <https://huggingface.co/martinnavs/granite-4.0-h-350m-router-kd-web4-s0> |
| **How it was made, tested and where it is wrong** | [`MODEL-CARD.md`](MODEL-CARD.md) |
| **Round-by-round working log** | [`docs/CLUSTER-LOG.md`](docs/CLUSTER-LOG.md) |
| **Who owns what, and the handoff** | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
| **The voice agent that uses it** | <https://github.com/Martinnavs/AI-231-ME3> (owns the contract, the evaluation sets, serving and acceptance) |
| Licence | Apache 2.0 (code and data); the adapter inherits the Granite base model's terms |

## Scope

This repository does **data and training only**: generate the data, train the teacher and the student, run the training-side
checks, publish the adapter. Serving the model, the GGUF conversion, the call policy and the acceptance runs belong to the
voice-agent repository and are handed off to it (`docs/ARCHITECTURE.md`). The line language, the chat format and the
evaluation sets are *its* files; this repository holds pinned copies (`contract.lock.json`, `scripts/sync_contract.py`) and
the tests fail if they drift or are edited here.

## What is here

```
spec/tools.py          the line language the router writes (NAME|arg|arg), the tool list, the GBNF grammar  [pinned copy]
data/banks.py          hand-written wordings and word lists (English, Tagalog/Taglish)
data/generate.py       the data generator: seeds -> exact target lines, noise, histories, splits, leak filter
data/format.py         the chat format (system message with the phonebook) used in training and serving  [pinned copy]
data/released/         the data the released model was trained on (see below)
train/train.py         LoRA training and distillation (reply-token loss, KL to the teacher + gold cross-entropy)
train/run.sh           start/stop a run in the background on a shared GPU node (refuses GPU 6, a local rule: edit it)
train/export.py        merge notes and the llama.cpp conversion steps
scripts/               teacher-logit cache (top-k), merge of two seeds into the distillation set, preflight
eval/                  held-out checks (ticket 03, ticket 04); the OnIt set and the 87-prompt pack (leak-filter inputs, pinned copies)
contract.lock.json     sha256 and source commit of the pinned copies; scripts/sync_contract.py updates and checks them
tests/                 76 tests (labels parse, splits disjoint, seeds behave, determinism, loss helpers, contract pinned)
```

## The data

`data/released/seed0-teacher/` is what the 1B teacher was trained on (`generate.py --seed 0 --n 30000`: 14,959 train, 1,474
dev, 1,430 test). `data/released/distill-student/` is what the 350M student was distilled on (seeds 0 and 1 merged with
`scripts/merge_distill_data.py`: 33,128 train, same dev and test). Each line is `{"id", "split", "kind", "messages",
"target"}` where `messages` is the full chat (system message with the phonebook, earlier turns, the user's words, the
target) and `kind` names the seed that produced it.

It is synthetic: wordings from `data/banks.py` plus speech-recogniser-style noise, labels built from the seed (so exact by
construction), made-up names in made-up phonebooks. No personal data, no scraped text, no hosted model was used. Near
duplicates of the evaluation sets in `eval/` are filtered out (80% word 3-gram overlap). The generator is deterministic:
`python3 data/generate.py --out /tmp/x --n 30000 --seed 0` reproduces `data/released/seed0-teacher/` byte for byte.

Not included: the cached teacher logits (197 MB, regenerate with `scripts/cache_teacher_logits.py` from the teacher adapter,
which is not released) and the teacher adapter itself.

## Reproduce

```bash
uv sync                                     # main environment (torch + transformers + peft)
uv run pytest -q                            # 73 tests
python3 data/generate.py --out data/generated    --n 30000 --seed 0
python3 data/generate.py --out data/generated-s1 --n 40000 --seed 1
python3 scripts/merge_distill_data.py data/generated data/generated-s1 data/distill
GPU=0 bash train/run.sh start teacher --model ibm-granite/granite-4.0-h-1b --data data/generated \
      --out runs/teacher --seed 0 --batch 4 --accum 16
CUDA_VISIBLE_DEVICES=0 python scripts/cache_teacher_logits.py --teacher ibm-granite/granite-4.0-h-1b \
      --teacher-adapter runs/teacher/final --data data/distill --out data/distill/teacher_top128.pt
CUDA_VISIBLE_DEVICES=0 python train/train.py --model ibm-granite/granite-4.0-h-350m --data data/distill \
      --out runs/student --seed 0 --batch 8 --accum 8 --teacher-cache data/distill/teacher_top128.pt
python eval/ticket04_check.py --model ibm-granite/granite-4.0-h-350m --adapter runs/student/final
```

The released model used the Mamba kernels (`causal-conv1d` 1.7.0, `mamba-ssm` 2.3.2.post1) in a separate environment with
torch 2.8.0+cu128; without them everything still runs, about 2-3x slower (see the model card, section 5). One run, seed 0,
on a shared 40 GB A100: about an hour per model. Expect the same data and the same recipe, not bit-identical weights.

## Honest limits of this repository

- The voice-agent repository stays the place where the router is used and where new tickets land; later rounds appear here
  as new commits. `finetune/` in that repository still holds the same code as a mirror, so the two can differ until it is trimmed.
- `eval/onit_eval.py` (the scorer for the OnIt set) needs the voice agent's pipeline and lives there;
  `eval/onit_eval.jsonl` and `eval/agent_prompts.jsonl` are pinned copies of the two evaluation sets, kept for the leak filter.
- Scores in the model card are on template data and hand-written held-out cases; the independent check was run on the
  laptop with the real stack (model card, section 7).

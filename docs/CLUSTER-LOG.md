> Round-by-round working log of the cluster sessions, kept as written. It was written inside the voice-agent repository, so
> paths such as `finetune/data/generate.py` mean `data/generate.py` here, and `docs/...` / `.scratch/...` refer to that repository.
> Start with `../MODEL-CARD.md`.

# finetune/: train the voice agent's local router (start here on the cluster)

**Production model (2026-11-13): `kd-web4-s0`, public at <https://huggingface.co/martinnavs/granite-4.0-h-350m-router-kd-web4-s0>. How it was made, tested and where it is wrong: [`MODEL-CARD.md`](MODEL-CARD.md) (this file is the round-by-round session log).**

You are picking this up on the HPC cluster with no memory of the chat that produced it. Read this file, then `docs/LOCAL-LLM-FINETUNE-PLAN.md` (sections 1c, 3, 7, 8) and `docs/LOCAL-LLM-RESULTS.md` (what was measured). Everything needed to continue is in the repository.

## What and why

The voice agent (laptop mic, Pi loudspeaker, hosted Gemini `gemini-3.1-flash-lite` with tools) is slow on its first model call (0.95 s p50, 1.82 s p95). A small local model, **Granite 4.0 H-1B**, answers in 0.39 s p50 / 0.66 s p95 on the laptop CPU when it writes one grammar-constrained line, but is only about **59% exact untrained** (hosted: about 100%). The job: fine-tune it so it is a reliable **router** for simple requests, because the agent is for consumers and a wrong action is worse than a slow one.

**The split of work (owner's decision, 2026-10-28; plan section 1c):**

| Domain | Who |
| --- | --- |
| home commands (17 intents), time, weather, Wikipedia, Pi volume and status, chat | the local router, in one call, no hosted request |
| documents, files, shell, background jobs, web, writing code, multi-step workflows (OnIt's domain) | hosted Gemini; the router only says `escalate|tools` (or `code`, `workflow`, `reasoning`, `other`) |

So the router is trained on **28 names** (the 17 home intents, 8 information tools, `none`, `escalate`) and must defer the rest. The contract also lists the 9 OnIt tools because Gemini's schemas and some evaluation arms use them; the router never emits them.

## State (2026-10-28, end of the laptop session)

Built and tested on the laptop (30 tests, ruff clean, **nothing has run on a GPU yet**):
- the contract (`spec/tools.py`), the OnIt evaluation set and runner (`eval/`), the baselines;
- the **data generator** (`data/generate.py`): seeds give exact target lines; noise, multi-turn follow-ups, two-call turns, deferrals, pleasantries and missing-value questions; an optional teacher (`--teacher gemini`) rewrites a share of the wording; a near-duplicate filter keeps the 87-prompt pack and the OnIt set out of training; the split is by seed key;
- the **training script** (`train/train.py`: LoRA or full, loss on the target only, generation exact match on dev), `train/export.py` (merge, then the llama.cpp conversion steps), `train/run.sh` (background launcher; refuses GPU 6), `scripts/preflight.py`;
- the evaluation hook `router-ft` in `eval/onit_eval.py` for the fine-tuned model (it uses the short system message of `data/format.py`).

**Not tested** (needs the cluster): `train/train.py` and `train/export.py` have only been parsed, and their pure helpers (`train/utils.py`: label masking, batching) are tested with a stub tokenizer. Expect small API fixes (transformers or peft version differences) on the first smoke run; the 20-step `--smoke` run exists for that.

## Cluster session 2026-10-28: baseline trained, conversion left to do

**Result (one run, seed 0, template-only data, so a baseline):** LoRA (rank 32) on `ibm-granite/granite-4.0-h-1b`, 8,386 training examples, 3 epochs, 3,395 s on one shared A100 (GPU 5). Dev exact match **96.7% (290/300)**, eval loss 0.037 after epoch 1 and 0.023 after epoch 3. The dev set comes from the same template generator as the training data, so this is optimistic: **the OnIt set and the 87-prompt pack have not been run yet** and are the real test. Not done: teacher-reworded data (`--teacher gemini`), three seeds, GGUF export, latency.

**What the first GPU run needed** (so the next person does not repeat it):
- `train/run.sh` calls plain `python`: put the project environment first, `PATH=$PWD/.venv/bin:$PATH GPU=5 bash train/run.sh start ...`, after `uv sync --group train --group dev --group data` in `finetune/` (resolved to torch 2.14.1 with a CUDA 13.0 build, transformers 5.21, peft 0.21; `uv.lock` pins it).
- `transformers` 5 renamed `warmup_ratio`; `train.py` now passes `warmup_steps=0.03` (a float below 1 is a ratio). That was the only API fix.
- `mamba_ssm` and `causal_conv1d` are not installed (the node has nvcc 12.8, the torch build is CUDA 13.0), so the hybrid layers use the slow PyTorch fallback. It works, but uses much more memory and about 8 s per step: the default `--batch 16` ran out of memory next to another user's job, **`--batch 4 --accum 16` fits** (same effective batch of 64). Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- The node's GPUs are shared and were at 97-100% utilisation from other users throughout; GPU 6 stays off limits.
- `train.py` prints its loss and eval lines to a block-buffered stdout, so they reach `runs/<name>.log` in chunks; the progress bar (stderr) is live.
- The adapter is in `runs/h1b-lora-s0/final`; `python train/export.py --base ibm-granite/granite-4.0-h-1b --adapter runs/h1b-lora-s0/final --out runs/h1b-lora-s0/merged` produced a 2.8 GB bf16 model (tested, CPU only, about a minute).

**Convert and evaluate on another machine.** The adapter is on the Hugging Face Hub (private): `martinnavs/granite-4.0-h-1b-router-baseline-s0` (adapter and config only, about 143 MB). It cannot be converted to GGUF as is: merge it into the base first.

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "ibm-granite/granite-4.0-h-1b"
m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
m = PeftModel.from_pretrained(m, "martinnavs/granite-4.0-h-1b-router-baseline-s0").merge_and_unload()
m.save_pretrained("merged")
AutoTokenizer.from_pretrained(base).save_pretrained("merged")
```

Use the versions in `uv.lock` (transformers 5.x, peft 0.21); `train/export.py` does the same merge from a local adapter. Then, with llama.cpp:

```
python llama.cpp/convert_hf_to_gguf.py merged --outfile router-f16.gguf --outtype f16
llama.cpp/build/bin/llama-quantize router-f16.gguf router-Q4_K_M.gguf Q4_K_M     # also try Q8_0
llama-server --jinja -m router-Q4_K_M.gguf --port 8089                     # the eval runner sends the grammar (spec/tools.py) with each request
python eval/onit_eval.py router-ft --url http://127.0.0.1:8089                    # the OnIt set
python eval/onit_eval.py router-ft --url http://127.0.0.1:8089 --set pack         # the 87-prompt pack
```

Compare with the table below, and measure p50/p95 model-call latency on the laptop CPU with Q4_K_M (target at most 0.8 s p95). The cluster session did not build llama.cpp.

What the generator gives: with plain templates only, 25,000 draws yield about 10,000 distinct examples (the rest are duplicates and are dropped), so **use the teacher** (`--teacher gemini`, key read from `~/.secrets/google-ai-studio/` at run time, never printed) to reach the size and the variety consumers need. Plain-template wording was written by one author, so the generator's own dev/test splits are optimistic; **the 87-prompt pack and the OnIt set are the independent tests**. Exact matches of the eval sentences are dropped from training (for example "turn on the lights"), which costs some common wordings; for the model that ships, regenerate with `--include-eval` after the evaluation is done.

## The output language (the contract: `spec/tools.py`)

One call per line, `NAME|arg1|arg2`, the last argument may contain `|`, no newlines in arguments. `none|reply` is no tool; `escalate|why` defers. Examples: `LIGHT_ON`, `TIMER|45 minutes`, `weather|Baguio|tomorrow`, `get_current_time|Asia/Manila`, `none|You're welcome.`, `escalate|workflow`. `spec/tools.py` is the single source of truth: parser (`parse_line`, `parse_output`), formatter, the GBNF grammar (`grammar()`; llama-server enforces it at inference), and the OpenAI-format schemas the hosted model sees (`openai_tools()`). **Change the tool list there, then regenerate data and retrain.** GBNF rule names allow only lowercase letters, digits and dashes (a test checks it).

## Files

| Path | What |
| --- | --- |
| `spec/tools.py` | the contract (36 names: 17 home intents, 8 info tools, 9 OnIt tools, `none`, `escalate`) |
| `tests/test_spec.py` | 17 tests: round trips, pipes in the last argument, defaults, invalid lines, grammar enums and rule-name syntax, agreement with the agent's current tool schemas (`data/agent_tools_snapshot.json`) |
| `eval/onit_prompts.py`, `eval/onit_eval.jsonl` | the OnIt evaluation set: 73 hand-written prompts, with checks that accept several valid commands |
| `eval/onit_eval.py` | the runner: `gemini`, `router-split`, `router-full` backends; `--set pack` runs the 87-prompt regression pack (`pipeline/config/eval/agent_prompts.jsonl`) |
| `eval/system_*.txt` | prompts used for the untrained baselines |
| `data/agent_tools_snapshot.json` | the agent's current 9 tool schemas (OpenAI format) |
| `data/generate.py`, `data/banks.py`, `data/format.py` | the generator, its word banks, and the chat format shared by training and evaluation |
| `train/train.py`, `train/utils.py`, `train/export.py`, `train/run.sh` | training, label masking and batching (tested), merge and GGUF steps, the launcher |
| `scripts/preflight.py` | what the machine can do (GPUs, CUDA, nvcc, disk, internet, libraries, Mamba kernels) |
| `tests/` | 30 tests: contract, generator, training helpers |
| `pyproject.toml` | dependency groups `data`, `train`, `dev` (not locked; pin on the cluster) |

## Verify the checkout (no GPU, no network)

```
cd finetune && python3 -m pytest tests -q          # or: uv run --offline --group laptop pytest finetune/tests -q -c /dev/null --rootdir finetune  (from the repo root)
python3 eval/onit_prompts.py                       # rewrites eval/onit_eval.jsonl; the diff must be empty
python3 -c "import sys; sys.path.insert(0,'spec'); import tools; print(len(tools.TOOLS), 'names'); print(tools.grammar()[:200])"
```

## Numbers to beat (one run each; the untrained local model is Granite 4.0 H-1B Q4_K_M, llama.cpp CPU, laptop)

| | OnIt set | Old pack (92 turns) | Model-call latency p50 / p95 |
| --- | --- | --- | --- |
| hosted Gemini, 18 tools | 57 of 63 scored (90%) | 77/92 (84%); about 100% with the old 9 tools | 0.90 s / 1.26 s (a second request for a spoken answer: 0.91 s / 1.17 s) |
| untrained H-1B router (split) | 54-64 of 77 (70-83%), prompt-dependent | 48/92 (52%) | 0.38-0.45 s / 0.53-0.73 s |
| untrained H-1B making the OnIt calls itself | 22 of 77 (29%) | n/a | 0.43 s / 0.67 s |

Targets (proposals the owner accepted): at least 90% exact on a new held-out set; at least 85% on the 87-prompt pack (counting `search_docs` prompts as correct when the router defers them); at most 3% of turns with a tool call when none was expected; at least 95% recall and 90% precision on deferring workflow, code and tool-summary requests; p95 model call at most 0.8 s on the laptop with Q4_K_M. State the number of runs and seeds with every result.

## Cluster facts (from the owner's ME2 project; `docs/LOCAL-LLM-FINETUNE-PLAN.md` section 5)

UPD CoE HPC, node `ai-n002.hpc.coe.upd.edu.ph`, one A100 per run, the GPU chosen by index with `CUDA_VISIBLE_DEVICES` (**never GPU 6**), jobs started with `uv run`, shared home `/mnt/jfs_hpc/home/anthony.martin.navarez/`. ME2 pins `torch==2.3.1` (cu121); do **not** reuse that environment (Granite 4.0 hybrid needs a recent `transformers`). No SLURM was seen anywhere; internet on the node is assumed. Unknown until you run `scripts/preflight.py`: GPU count and who else is using them, `nvcc`, disk quota, time limits.

## Tasks for the cluster session (in order)

```
cd finetune
python scripts/preflight.py                      # 1. read the verdict: hybrid H-1B or the plain 1B
uv sync --group train                            #    (or the cluster's own environment; torch must match the CUDA build)
python data/generate.py --out data/generated --n 25000 --seed 0 --teacher gemini     # 2. data (needs the Gemini key; omit --teacher for a template-only set)
python -m pytest tests -q                        #    all green before training
GPU=5 bash train/run.sh start smoke --model ibm-granite/granite-4.0-h-1b --data data/generated --out runs/smoke --smoke   # 3. 20 steps
GPU=5 bash train/run.sh start h1b-lora-s0 --model ibm-granite/granite-4.0-h-1b --data data/generated --out runs/h1b-lora-s0 --seed 0
bash train/run.sh status h1b-lora-s0
python train/export.py --base ibm-granite/granite-4.0-h-1b --adapter runs/h1b-lora-s0/final --out runs/h1b-lora-s0/merged   # 4. merge, then the printed llama.cpp steps
```

Then evaluate the GGUF with llama.cpp (`llama-server --jinja`, grammar from `spec/tools.py`): `python eval/onit_eval.py router-ft --url http://127.0.0.1:8089` and again with `--set pack`; compare with the table above. Choose on dev, look at the test and pack results once, state seeds and repetitions (three seeds for the final configuration). If the hybrid's kernels will not build, train `ibm-granite/granite-4.0-1b` instead (62/92 untrained against 53/92 for the hybrid, but 1.7x slower at inference): decide by measurement. Bring the GGUF and the tables back; update `docs/LOCAL-LLM-RESULTS.md` and this file.

Open questions for the owner, to ask when they are online: which GPU indexes are free on `ai-n002` and whether another user competes; whether the node has internet (the preflight says); whether a second hosted request for the spoken answer is acceptable on turns the router defers (assumed yes: the router defers, Gemini does request 1 and request 2).

## Cluster session 2026-11-03: distilled 350M with `web_answer`

One run each, seed 0, dev only (the OnIt set and the pack were **not** run on these models; GGUF conversion is done on the laptop).

| Model | Data | Dev exact |
| --- | --- | --- |
| 1B v1 LoRA (teacher, no `web_answer`) | `generate.py --seed 0 --n 30000` (before the web round) | 97.7% (293/300) |
| 350M LoRA | same | 96.0% (288/300) |
| 350M distilled from the 1B v1 (`kd-s0`) | v1 seed 0 + seed 1 (`--n 40000`), merged | 98.3% (295/300) |
| 1B web teacher | `web_answer` data, seed 0 | 96.7% (290/300) |
| 350M distilled from the 1B web teacher | web data, seed 0 + seed 1 | 97.0% (291/300); 97.8% (1129/1154) on the full dev split |

The plain 350M saw half the prompts of the distilled one, so "distillation helped" is not separated from "more data".

Distillation recipe (`train/train.py --teacher ... --teacher-adapter ...`): loss on the reply tokens, `0.5 * KL(teacher || student, T=2) + 0.5 * gold cross-entropy`; same tokenizer and vocabulary, so logits compare directly. Data: `python3 scripts/merge_distill_data.py data/generated data/generated-s1 data/distill` (seed-0 train plus all of a second seed, minus anything whose text is in dev or test).

```bash
python3 data/generate.py --out data/generated --n 30000 --seed 0
python3 data/generate.py --out data/generated-s1 --n 40000 --seed 1
python3 scripts/merge_distill_data.py data/generated data/generated-s1 data/distill
GPU=0 bash train/run.sh start h1b-lora-web-s0 --model ibm-granite/granite-4.0-h-1b --data data/generated --out runs/h1b-lora-web-s0 --seed 0 --batch 4 --accum 16
GPU=0 bash train/run.sh start kd-350m-web-s0 --model ibm-granite/granite-4.0-h-350m --data data/distill --out runs/kd-350m-web-s0 --seed 0 --batch 8 --accum 8 --teacher ibm-granite/granite-4.0-h-1b --teacher-adapter runs/h1b-lora-web-s0/final
```

**`web_answer`** (`spec/tools.py`, ticket `.scratch/router-web-answer/tickets/01-retrain-350m-web-answer.md`): a router call `web_answer|<complete question>` for single current-fact questions; multi-step, how-to and file/document requests still `escalate|tools`. On the dev split `web_answer` recall is 39/39 and precision 39/49.

**Known issue (decided not urgent):** all 10 false positives are one phrase, "what's the pi temperature", which returns `web_answer` instead of `device_status` (the Pi category is 12/22 on dev). The generator has one device-status wording for it; add several device-status and Pi wordings as hard negatives and retrain teacher and student before serving this adapter. The adapter is private at `martinnavs/granite-4.0-h-350m-router-kd-web-s0-unverified`; acceptance on the OnIt set and pack (no regression beyond 1 item against kd-s0: 76/77 and 79/92) is still open.

## Cluster session 2026-11-03 (ticket 02): arithmetic is never a home action

Same recipe as above with run names `h1b-lora-web2-s0` and `kd-350m-web2-s0` (GPU 4), one seed, dev only. Data: `ARITH_TEMPLATES` / `s_arithmetic` (spoken arithmetic -> `escalate|reasoning`, weight 5), 7 more timer durations and 4 more timer phrasings, and, beyond the ticket, 11 Pi device-status wordings with `s_pi` weight 4 -> 6. Counts for seed 0 (before -> after): train 11356 -> 11777, dev 1154 -> 1209, test 1116 -> 1191; merged distillation train 21578 -> 23016. The 1B teacher scored 96.3% on the 300-example dev subset, the student 96.3%.

Both students on the **new** dev split (1209, generator templates, so optimistic; the previous student never saw arithmetic data):

| | kd-web-s0 (previous) | kd-web2-s0 (new) |
| --- | --- | --- |
| Overall exact | 1107/1209 (91.6%) | 1179/1209 (97.5%) |
| `web_answer` tp / fp / fn | 31 / 8 / 0 | 31 / 0 / 0 |
| Arithmetic exact (routed to a home action) | 10/80 (7) | 80/80 (0) |
| Timer exact | 83/83 | 83/83 |
| Pi device status exact | 18/34 | 27/34 |

Remaining Pi misses are all "is the pi running okay" phrasings going to `escalate|tools`. Adapter (private, unverified): `martinnavs/granite-4.0-h-350m-router-kd-web2-s0-unverified`. The laptop acceptance (OnIt >= 75/77, pack >= 78/92, no arithmetic to a home intent) is still open.

## Cluster session 2026-11-04 (ticket 03): an independent request after an action is independent

One seed, dev and hand-written held-out cases only (the OnIt set and pack were **not** run; the laptop converts the adapter). New data (`s_topic_switch`, `s_ack_after_anything`, `s_action_question`, `s_garble` + `s_short_command`, `s_pi_contrast`, `s_delegate_request` + `s_status_words_web`, `s_statement` + `s_statement_contrast`): an earlier action in the history no longer makes the model repeat or vary it. Counts for seed 0 (before -> after): train 11777 -> 14044, dev 1209 -> 1434, test 1191 -> 1446; seed 1: train 15137 -> 18025, dev 1357 -> 1585, test 1506 -> 1803; merged distillation train 23016 -> 31068.

| | 1B teacher | 350M student (kd-web3-s0) | ticket-02 student (kd-web2-s0) |
| --- | --- | --- | --- |
| Dev 300-example subset, strict | 93.7% | 95.0% | (96.3% on the old, easier dev) |
| Full new dev (1434), strict exact | not run | 1358 (94.7%) | 1248 (87.0%) |
| Full new dev, reply wording free for `ack` / `statement` | not run | 1416 (98.7%) | 1274 (88.8%) |
| Old kinds only (1003 examples) | not run | 987 (98.4%) | 982 (97.9%) |
| Held-out, 70 hand-written cases | 70/70 | 70/70 | 49/70 |
| Multi-turn replies that reuse the previous call | 0/40 | 0/40 | 4/40 |
| `web_answer` tp / fp / fn on dev | not run | 60 / 0 / 0 | 60 / 4 / 0 |
| Arithmetic to a home action; timer exact | not run | 0; 80/80 | 0; 80/80 |

`ack` and `statement` replies are drawn at random from a short list, so the same phrase has several correct targets and strict exact match cannot reach 100% on them (13/36 and 9/44 here). Right behaviour (a one-line `none|` reply of at most 6 words) is 36/36 and 44/44. Fix in the next data round: make the reply a fixed function of the phrase. The other dev misses are old data quirks: `install this random script...` goes to `escalate|tools` where the label is `other` (9), an ambiguous alias such as "my brother" (4), a capitalisation difference in one web question (3).

Speed: the student took 53 minutes (2.0 s/step) instead of about 2.7 hours (6.6 s/step) with two changes. (1) The Mamba kernels (`causal-conv1d`, `mamba-ssm`) in a separate venv, `finetune/.venv-kernels` (uv, torch 2.8.0+cu128, `--no-deps` installs so torch is not replaced; the main `.venv` is unchanged). (2) `scripts/cache_teacher_logits.py` runs the teacher once (8 minutes for 32,502 examples) and keeps its top-128 logits at the reply tokens; `train.py --teacher-cache` then distils without loading the teacher. The top-128 logits cover 99.06% of the teacher's probability mass at T=2 on average (worst position 36%), so the KL is a close approximation, not the exact full-vocabulary one. The teacher was trained with the old recipe (no kernels).

```bash
python3 scripts/merge_distill_data.py data/generated data/generated-s1 data/distill
GPU=4 bash train/run.sh start h1b-lora-web3-s0 --model ibm-granite/granite-4.0-h-1b --data data/generated --out runs/h1b-lora-web3-s0 --seed 0 --batch 4 --accum 16
CUDA_VISIBLE_DEVICES=7 .venv-kernels/bin/python scripts/cache_teacher_logits.py --teacher ibm-granite/granite-4.0-h-1b --teacher-adapter runs/h1b-lora-web3-s0/final --data data/distill --out data/distill/teacher_top128.pt
CUDA_VISIBLE_DEVICES=7 .venv-kernels/bin/python train/train.py --model ibm-granite/granite-4.0-h-350m --data data/distill --out runs/kd-350m-web3-s0 --seed 0 --batch 8 --accum 8 --teacher-cache data/distill/teacher_top128.pt
CUDA_VISIBLE_DEVICES=7 .venv-kernels/bin/python eval/ticket03_check.py --model ibm-granite/granite-4.0-h-350m --adapter runs/kd-350m-web3-s0/final heldout   # or: dev
```

`merge_distill_data.py` now holds out the whole prompt (history and final turn), not only the last user text, so a short reply such as "okay" is held out only after the same history. Adapter (private, unverified): `martinnavs/granite-4.0-h-350m-router-kd-web3-s0-unverified`. Laptop acceptance (OnIt >= 75/77, pack >= 78/92, earlier fresh sets unchanged, 30 fresh multi-turn cases with 0 repeats, fresh Pi / delegate / statement sets) is still open. The held-out cases are mine, written after seeing the data banks, so they are optimistic.

## Cluster session 2026-11-06 (ticket 04): negation, Filipino, play targets, no referent, offers

Run `web4`, the ticket-03 recipe (kernels venv for both runs, cached top-128 teacher logits, 99.0% of the teacher mass at T=2).
New seeds in `data/generate.py` (MIX weight): `s_negated` 4, `s_filipino` 6, `s_near_miss` 3, `s_play_target` 4, `s_no_referent` 3,
`s_confirm_offer` 3, `s_delegate_request` 4->6 (job words, listings); ack and statement replies are one fixed phrase each
("Okay.", "Alright."). Hosted-model replies sit in the history as `none|<reply>` (as the pipeline writes them), so an offer is a
`none|...?` line; the router's own questions look the same, the wording tells them apart.

| Check | kd-350m-web4-s0 (student) | 1B web4 teacher |
|---|---|---|
| Ticket-04 held-out (70 hand-written, none in the data) | 70/70 | 70/70 |
| Ticket-03 held-out (70) | 68/70, 0/40 repeats | not run |
| Full dev (1474), strict | 1455/1474 (98.7%) | subset of 60 per kind: 1029/1072 |
| Old kinds in dev (1282) | 1270/1282 (99.1%) | |
| ack / statement strict | 24/24, 42/42 (was 13/36, 9/44) | 24/24, 42/42 |

Dev per new kind (student): negated 28/28, filipino 49/54, near-miss 17/18, play-target 36/36, play-bare 7/7, no-referent 10/11,
no-referent-contrast 7/7, confirm-offer 19/19, confirm-contrast 11/11, delegate 83/83.
Dev misses on new kinds: "mainit ba ngayon" (4x, answered as a statement), "patigilin mo ang music" (PAUSE for STOP),
"last like song please" (NEXT; a backward request answered with a skip, the one that matters), "once more" (question wording).
Ticket-03 held-out regressions (2): "play something relaxing" after a speaker or message turn now escalates (`escalate|tools`),
which is what ticket 04's play-target rule asks for (a mood is a target); that old case expected PLAY_MUSIC.
Data counts, train (web3 -> web4): merged distillation 31068 -> 33128; new kinds in the merged train: negated 787+103, filipino
1189+118, near-miss 481+62, play-target 914, play-bare 60, no-referent 304+223, confirm 472+378, delegate 1092 -> 1467.
Teacher 63 min (kernels, 5 s/it), student 57 min (GPU 7, 2 s/it). Adapter `martinnavs/granite-4.0-h-350m-router-kd-web4-s0` (public since 2026-11-13, was `...-unverified`; the laptop accepted it, see `MODEL-CARD.md`).
Caveats: held-out cases are my own (optimistic); the dev is template data; `no-referent` with a history is only trained for
"set it to N" after a temperature or brightness turn and the existing follow-ups, not "do that again" after an action;
"it's dark in here" has two labels in older seeds; `WEB_STATUS_WORDS` questions I added differ in wording from what the
teacher writes (a trailing "yet"), a cosmetic dev miss class.

## Rules that apply here (the owner's standing agreements)

- Never read or print API keys (`~/.secrets/*`, `.env`, any `*key*` file); code reads them at run time and prints only OK or failure.
- No model on the Raspberry Pi except the Silero VAD (the professor's rule); nothing in this directory runs on the Pi.
- Commit and push only when the owner asks; never force-push, never push `main`; add files by name (not `git add -A`); gate commits on `ruff` and `pytest`; use `git -c safe.directory=$PWD` in a repository you do not own. Do not touch the owner's untracked files (`kek.txt`, `resources/building-voice-agents/`, `docs/VOICE-AGENT-EVALUATION-METRICS.md`, `docs/ECHO-AGENT-BRIEF.md`, `docs/GEMINI-ARGON-ECHO-RESPONSE.md`).
- Report the bottom line first, state how many repetitions a result rests on, and say when a setting was chosen after seeing the data (dev/test discipline: choose on dev, look at test once).
- zsh does not word-split `$VAR`; `pkill -f <text in your own command>` kills your own shell.

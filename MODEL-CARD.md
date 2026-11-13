# Voice-agent router: Granite 4.0 H 350M + LoRA (`kd-web4-s0`)

A small local **router** for a voice agent. It reads what the user said (plus the last few turns and a phonebook) and writes
one line per tool call, in a fixed line language, or says that the request belongs to a bigger hosted model. It is a LoRA
adapter on `ibm-granite/granite-4.0-h-350m`, distilled from a 1B LoRA teacher on the same base family, trained on
template-generated data (no hosted model wrote or paraphrased any of it).

This is the **production router** of the AI-231-ME3 voice agent (laptop microphone, Raspberry Pi loudspeaker, hosted Gemini
for everything hard). The model card documents how it was made, what it was tested on, and where it is known to be wrong.
Everything here is reproducible from this repository, <https://github.com/Martinnavs/ME3-LoRA-training> (a copy of the
`` directory of the voice agent, <https://github.com/Martinnavs/AI-231-ME3>, at commit `9c2d34c` of branch
`agy/interrupts`), including the generated training data (`data/released/`).

> **Read this first.** This is an adapter, not a general assistant. It is trained to emit calls in one specific language and
> to be served with the exact system message below, under a grammar (GBNF) that forces well-formed lines. Outside that setup
> it will not behave. The training and dev data are template-generated, so the dev scores are optimistic; the numbers that
> matter are the ones measured on the laptop with the real stack (section 7).

## 1. What the router does

The agent's old path was one hosted-model call per turn (0.95 s p50, 1.82 s p95 for the first call alone). A turn that
needs a single tool call and nothing else does not need that. The routing rule, decided by the owner:

| The turn needs | Goes to |
| --- | --- |
| one tool call (a home command, the time, the weather, a Wikipedia summary, one current-fact lookup, the speaker's volume or status) or a short reply | **this router**, locally |
| anything that needs another model call (documents, files, shell, background jobs, writing code, multi-step work, reasoning) | hosted Gemini; the router writes `escalate|<reason>` |

The router is therefore trained on about 28 names: 17 home intents, the information tools, the Pi tools, `none` (a spoken
reply) and `escalate`. It must *refuse to guess* on everything else.

### The line language

One call per line, `NAME|arg|arg`, names from `spec/tools.py`:

```
turn on the lights                       -> LIGHT_ON
set a timer for ten minutes              -> TIMER|10 minutes
what time is it in Tokyo                 -> get_current_time|Asia/Tokyo
who is the current governor of Cebu      -> web_answer|Who is the current Governor of Cebu?
what's forty five times twelve           -> escalate|reasoning
write me a snake game in the background  -> escalate|code
set a timer                              -> none|For how long?
don't call Ben                           -> none|Okay, I won't.
patayin mo ang ilaw                      -> LIGHT_OFF
```

Rules the model is trained to follow (each has dedicated data, see section 4):

- `none|<short spoken reply>` for greetings, thanks, acknowledgements, statements and refusals to act; a missing value is
  asked for (`none|For how long?`). A question about something already done ("did you call Ben?") and a yes or no to the
  hosted model's offer go to the hosted model: `escalate|other`.
- `escalate|<tools|code|workflow|reasoning|other>` for everything the router must not attempt. The reason is a hint for the
  caller, not a tool.
- `web_answer|<a complete, self-contained question>` for a single current-fact lookup; multi-step research stays escalated.
- Contacts are resolved through the phonebook in the system message; a name that is not in it gets
  `none|<name> is not in the phonebook.`
- An independent request after an earlier action is independent: the router does not repeat or vary the previous call.
- Arithmetic is never a home action (`escalate|reasoning`), a negated command is never carried out, and a bare "play
  music" is plain `PLAY_MUSIC`.

### The system message (must match training)

```
Route the user's words to calls, one per line. Reply with the calls only.
Handle simple home, time, weather and Wikipedia requests yourself; defer everything else with escalate|reason.
Phonebook: Anna (friend), Ben (brother), Mom (mother), Dad (father), Dr. Reyes (doctor)
```

The phonebook is the only thing that changes between users: `Name (relation)` pairs, or `empty`. Earlier turns are ordinary
chat turns: the user's words, then the assistant's line(s) as the router wrote them. If a reply came from the hosted model,
the caller shows it in the history as `none|<its reply, first 100 characters>`; the router is trained for exactly that.

## 2. How to run it

The adapter targets `ibm-granite/granite-4.0-h-350m`, LoRA rank 32, alpha 64, dropout 0.05, on all linear layers
(`in_proj`, `out_proj`, `input_linear`, `output_linear`, `q_proj`, `k_proj`, `v_proj`, `o_proj`).

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "ibm-granite/granite-4.0-h-350m"
tok = AutoTokenizer.from_pretrained(base)
model = PeftModel.from_pretrained(
    AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16), "martinnavs/granite-4.0-h-350m-router-kd-web4-s0"
)
messages = [
    {"role": "system", "content": "<the system message above, with your phonebook>"},
    {"role": "user", "content": "turn off the lights"},
]
ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
print(tok.decode(model.generate(ids, max_new_tokens=64, do_sample=False)[0][ids.shape[1] :], skip_special_tokens=True))
```

Greedy decoding, at most 64 new tokens. In production the model is served as a **GGUF Q4_K_M** through `llama-server` with
the GBNF grammar from `spec/tools.py` (`grammar()`), so a malformed line cannot be produced.

Conversion pitfalls that cost time (the production GGUF was made on the owner's laptop, not on the cluster):

1. Merge the adapter into the base **in bf16 on the CPU**, then convert; do not merge in fp16.
2. Save the merged model with the **base** tokenizer and config (a merge that overwrites them breaks the chat template).
3. Run the conversion in its own environment, separate from the training one (the llama.cpp converter and the Mamba
   kernels want different torch builds).

## 3. Lineage: how the router got here

The router was never trained once. It went through six rounds, each driven by a written ticket (in the voice-agent repository, `.scratch/router-web-answer/tickets/`) that named a failure seen on the laptop, and each round re-ran the whole pipeline (data, teacher, student).
Every round added data for the failure and **contrast data** so that the fix did not spread to neighbouring requests.

| Round | Adapter | What changed and why | Cluster check |
| --- | --- | --- | --- |
| baseline | LoRA on 1B, then on 350M | Plan section 1c: first end-to-end run. 8,386 training examples, 3 epochs, one A100 | 1B dev 97.7%, plain 350M 96.0% |
| v1 | `kd-s0` | 1B v1 teacher, 350M student distilled from it, two seeds of data | dev 98.3%; laptop OnIt 76/77, pack 79/92 |
| ticket 01 | `kd-web-s0` | new call `web_answer|<question>` for single current-fact questions (a lookup the router can now start itself) | web_answer recall 39/39, precision 39/49 (the 10 false positives were one phrase, "what's the pi temperature") |
| ticket 02 | `kd-web2-s0` | spoken arithmetic is never a home action (it had been read as TIMER/BRIGHTNESS, 7 of 80); Pi status wordings added as hard negatives | arithmetic 80/80 with 0 home actions, timers 83/83, Pi 27/34 |
| ticket 03 | `kd-web3-s0` | an independent request after an action must not repeat or vary it (4 of 40 repeats before); topic switches, acknowledgements, garbles, Pi contrasts, delegate requests, statements | held-out 70/70, repeats 0/40; laptop held-out 60/62 |
| **ticket 04** | **`kd-web4-s0`** | negated commands, Filipino and Taglish commands, near misses (dim, go back, resume), `play <target>`, commands with no referent, job requests, confirmations after the hosted model's offer, fixed acknowledgement replies | see sections 6 and 7 |

Why a teacher at all: the 350M student distilled from the 1B teacher beat the plain 350M LoRA at the first comparison (98.3%
against 96.0%), although the plain model saw half the prompts, so "distillation helped" is not cleanly separated from "more
data". It was kept because it was never worse and the teacher is also the first check of whether a data family is learnable:
**if the 1B teacher cannot pass a new family, the data is wrong, not the student.** Ticket 03 is the clearest case: the
ticket-02 student failed 21 of 70 held-out cases while the 1B teacher on the new data passed all 70.

## 4. The training data

Generated by `data/generate.py` from the hand-written banks in `data/banks.py`. No hosted model is used (the
optional Gemini paraphrase teacher exists but was never switched on; spend for every round was $0).

**Labels are exact by construction.** A seed picks the intent and its arguments first and then builds both the wording and
the target line from them, so there is no labelling step to get wrong. Every target must parse under the grammar or the
generator stops. The wording gets light speech-recogniser noise on 60% of examples (no capitals or punctuation, fillers,
repeats, rare homophones such as "light" for "lights").

**Phonebooks and histories.** Each example carries its own random phonebook (4-7 contacts). A history and its final turn
share one phonebook (a context manager forces it), otherwise "call him" could name someone who is not in the book.

**Splits and leakage.** Examples are split train/dev/test by a hash of their *seed key* (intent plus arguments), so no dev
example shares a seed with a training one. The merge script holds out the **whole prompt** (history plus final turn), not
just the last words, so that "okay" is held out only after the same history. Anything that shares 80% or more of its word 3-grams with
an evaluation prompt (the OnIt set) is dropped (`--include-eval` exists for a final model and was never used).

**The seeds added by ticket 04** (MIX weight in brackets; each has contrasts that keep the neighbouring behaviour):

| Seed | Weight | Teaches | Contrast kept |
| --- | --- | --- | --- |
| `s_negated` | 4 | "don't / do not / no need to / never / huwag / wag + action", "stop <verb>ing ..." -> `none|Okay, I won't.` | plain imperatives, "stop", "stop the music", "stop playing" keep STOP, "never mind" is an acknowledgement |
| `s_filipino` | 6 | Tagalog and Taglish forms (`buksan`, `patayin`, `paki-`, `lakasan`, `hinaan`, `tawagan`, `i-text`, `gisingin mo ako`, `oras`, `panahon`, `patugtugin`, `ituloy`, 3-word forms) with *exactly* the target of the English form | Filipino statements and greetings stay statements |
| `s_near_miss` | 3 | "dim the lights" -> a question about the level, with a level -> `BRIGHTNESS`; "go back / previous song" -> `none|I can only skip forward.`; "resume / continue / unpause" -> `PLAY_MUSIC` | lights off, next song, pause unchanged |
| `s_play_target` | 4 | "play / open a game, an app, the thing you made, it" and "play an artist, genre, song, playlist" -> `escalate|tools` (the router had no music slot) | bare music stays `PLAY_MUSIC` |
| `s_no_referent` | 3 | "do that again", "set it to 40", "turn that one off" with no history -> a question asking what is meant | the same words after a history that resolves them; "set it to 22 degrees" stays `TEMPERATURE` |
| `s_confirm_offer` | 3 | a yes or no after the hosted model's offer ("Want me to open it?") -> `escalate|other` | an answer to the router's *own* question stays a call; a yes after a non-offer is an acknowledgement |
| `s_delegate_request` | 4 -> 6 | "is the game done", "how's the app going", "run a directory listing in the sandbox" -> `escalate|tools`, never `web_answer` | current-fact questions that use "done" or "ready" about the world stay `web_answer` |
| acknowledgements, statements | | one **fixed** reply per kind ("Okay.", "Alright.") so that strict exact match can reach 100% | |

Earlier rounds' seeds stay in the mix (weights in `MIX`): simple and slotted home commands, contacts and the not-in-phonebook
case, time and conversion, weather, Wikipedia, `web_answer` and its boundary, deferrals, document requests, two calls in one
turn, follow-ups, arithmetic, topic switches, garbles, Pi contrasts, delegate requests.

**Counts for this model** (seeds 0 and 1 merged for distillation): 33,128 train, 1,474 dev, 1,430 test. Kinds new in this
round, in the merged train split: negated 787 (+103 contrasts), Filipino 1,189 (+118), near miss 481 (+62), play target
914 (+60 bare music), no referent 304 (+223 contrasts), confirmation 472 (+378 contrasts), delegate 1,467 (was 1,092).
The seed-0 data alone (the teacher's data) is 14,959 train.

**Languages.** English, with Tagalog and Taglish forms of the home and information commands. No other language was tested.

## 5. The training recipe

Hardware: one shared 40 GB A100 (other users' jobs ran on the same GPUs, so wall-clock times are noisy). Software: torch
2.8.0+cu128, transformers 5.21.0, peft 0.21.2, causal-conv1d 1.7.0, mamba-ssm 2.3.2.post1 (the Mamba kernels live in a
separate virtualenv so the shared environment is untouched).

**Teacher.** `ibm-granite/granite-4.0-h-1b` + LoRA (rank 32, alpha 64, dropout 0.05, all linear layers), learning rate 2e-4,
cosine schedule with 3% warm-up, bf16, effective batch 64 (4 x 16 accumulation), max length 512, 3 epochs, seed 0, on the
seed-0 data. Loss on the assistant's reply tokens only. 63 minutes. Dev eval loss 0.0255, 0.0202, 0.0252 per epoch (the last
epoch is slightly worse; no early stopping was used, the final checkpoint is taken).

**Student.** `ibm-granite/granite-4.0-h-350m` + LoRA (same configuration), learning rate 2e-4, effective batch 64 (8 x 8),
3 epochs (1,554 steps), seed 0, on the merged seed-0 and seed-1 data. Loss on the reply tokens only:

```
loss = 0.5 * KL(teacher || student, T = 2) + 0.5 * cross-entropy(gold)
```

Teacher and student share the tokenizer and vocabulary, so the logits compare directly. 57 minutes. Dev eval loss 0.0634,
0.0375, 0.0324 per epoch (still falling at epoch 3, so more epochs might help; this was not tried).

**Two speed-ups that make the recipe approximate, stated plainly.** (1) The Mamba kernels (`causal-conv1d`, `mamba-ssm`);
without them the model falls back to a slow PyTorch reference path (2-3x slower: the ticket-03 student took about 2.7 hours
instead of under an hour). (2) The teacher's logits are computed **once** and stored (`scripts/cache_teacher_logits.py`): for
each reply token the top-128 logits. The student's loss renormalises the teacher over those 128 entries and compares with the
student's log-softmax over the full vocabulary. The top-128 entries cover **98.9%** of the teacher's probability mass at T=2
on average (the minimum over examples is 0.32, so a few tokens are poorly covered). The live-teacher recipe was not run for
this model; the ticket-03 student was trained the same way and passed the laptop retest.

**Reproduce** (from ``, all commands from the repository, GPU numbers are examples):

```bash
python3 data/generate.py --out data/generated    --n 30000 --seed 0
python3 data/generate.py --out data/generated-s1 --n 40000 --seed 1
python3 scripts/merge_distill_data.py data/generated data/generated-s1 data/distill
GPU=4 bash train/run.sh start h1b-lora-web4-s0 --model ibm-granite/granite-4.0-h-1b --data data/generated \
      --out runs/h1b-lora-web4-s0 --seed 0 --batch 4 --accum 16
CUDA_VISIBLE_DEVICES=7 .venv-kernels/bin/python scripts/cache_teacher_logits.py --teacher ibm-granite/granite-4.0-h-1b \
      --teacher-adapter runs/h1b-lora-web4-s0/final --data data/distill --out data/distill/teacher_top128.pt
CUDA_VISIBLE_DEVICES=7 .venv-kernels/bin/python train/train.py --model ibm-granite/granite-4.0-h-350m --data data/distill \
      --out runs/kd-350m-web4-s0 --seed 0 --batch 8 --accum 8 --teacher-cache data/distill/teacher_top128.pt
```

The generator is deterministic for a seed (a test checks that two runs give identical files), so the data is reproducible
without shipping it.

## 6. Results measured on the cluster

Greedy generation through the adapter, one example at a time, strict match of the full output.

| Check | Student (this model) | 1B teacher |
| --- | --- | --- |
| Ticket-04 held-out, 70 hand-written cases (12 negated, 15 Filipino, 8 near miss, 10 play target, 8 no referent, 7 job requests, 10 confirmations); none appears word for word in the data | **70/70** | 70/70 |
| Ticket-03 held-out, 70 cases (40 multi-turn, 12 Pi, 10 delegate, 8 statement) | 68/70; 0 of 40 multi-turn replies repeat the previous call | n/a |
| Full dev split (1,474), strict | 1,455 (98.7%) | n/a (a 60-per-kind subset: 1,029/1,072) |
| Old kinds only (1,282 dev examples) | 1,270 (99.1%) | |
| Acknowledgements and statements, strict (they were capped by random reply wording before) | 24/24 and 42/42 | 24/24 and 42/42 |

Dev per new kind: negated 28/28, Filipino 49/54, near miss 17/18, play target 36/36, bare music 7/7, no referent 10/11 (+7/7
contrast), confirmation 19/19 (+11/11), delegate 83/83.

**The two ticket-03 held-out misses** ("play something relaxing" after a speaker or message turn) are intended: ticket 04
teaches that a *mood* or genre is a play target and escalates it, and the old case expected plain `PLAY_MUSIC`.

**Every dev miss on the new kinds** (nothing hidden): "mainit ba ngayon" (a Filipino weather question, answered as a
statement; four dev variants), "patigilin mo ang music" (PAUSE instead of STOP), "last like song please" (NEXT, which is
a skip for a *backward* request and the one miss that matters), "once more" (right behaviour, wording of the question
differs). Older misses are data quirks: "install this random script ..." goes to `escalate|tools` where the label says
`other` (3), an ambiguous relation alias such as "my brother" or "my friend" when two contacts could match (3), a
capitalisation difference in one `web_answer` question (4), one topic-switch case.

## 7. Results measured on the laptop (the numbers that count)

From the project's own progress log (`docs/PROGRESS-2026-11-06.md`), the GGUF Q4_K_M served by `llama-server` with the grammar,
in the real pipeline:

| Check | Result |
| --- | --- |
| Held-out retest set (laptop's own wordings, never given to training) | 61/62 |
| OnIt tool-routing evaluation set | 77/77 |
| 87-prompt pack (92 checks) | 82/92 |
| 99 hard phrasings (negation, Filipino, play targets, no referent, job requests) | "wrong and acted" 4 against a bar of 3 (each a question or a refusal), accepted by the owner |
| Spoken session of 2026-11-06, router turns | 1.1-1.6 s from end of speech to sound (hosted-model turns: 2.3-3.6 s) |

For the previous round, `kd-web3-s0`, the same held-out set scored 60/62.

## 8. Known issues and limits

- **PLAY_MUSIC has no slot.** The router cannot carry "play <artist>" to the music tool, so those turns escalate to the
  hosted model (2.3-3.6 s instead of ~1.3 s). A follow-up round (ticket 05) gives the line an optional slot; it is not part of
  this model.
- **Escalation costs latency.** The model is deliberately cautious: for anything outside its domain it says `escalate`. The
  laptop adds a call-policy layer (`pipeline/src/voice_pipeline/call_policy.py`) in front of it for the cases the router
  still gets wrong; the router and that layer are meant to agree, not to replace each other.
- **Known wrong cases** (section 6): a backward request phrased "last like song please" can become NEXT; one Filipino weather
  question ("mainit ba ngayon") is taken for a statement; "patigilin mo ang music" can become PAUSE; two ambiguous aliases.
- **Untrained territory.** "Do that again" or "same as before" *after* an action has no training data either way. Commands
  that refer to something already said are only trained for "set it to N" after a temperature or brightness turn and for the
  existing follow-ups (weather, time, timer, text-them).
- **The data is templates.** Wordings are hand-written banks plus noise; real speech is more varied. Dev scores (98.7%) are
  optimistic; the held-out sets were written by the person who also wrote the banks. The laptop numbers in section 7 are
  the independent check.
- **One label conflict from an older seed:** "it's dark in here" is `LIGHT_ON` in one seed and a question in another.
- **Languages:** English with Tagalog/Taglish commands. Mishearings are handled by the speech server upstream, not here.
- **Safety:** the router only emits calls from a closed list under a grammar; what runs them is the agent's job. It has not
  been tested against adversarial prompts.
- **Not a chat model.** `none|` replies are one short line by design (at most about 8 words).

## 9. Provenance, licence and attribution

- **Base model:** `ibm-granite/granite-4.0-h-350m` (IBM Granite 4.0, Apache 2.0). Teacher: `ibm-granite/granite-4.0-h-1b`
  with a LoRA trained here; the teacher is not released.
- **This adapter** is a derivative work of the Granite base model and is released under **Apache 2.0**, with the base model's
  attribution and licence terms continuing to apply. Use the adapter with the base model.
- **Data:** synthetic, generated by the scripts in the repository. It contains no personal data (the names in the phonebooks
  are made up), no scraped text, and no sample of any evaluation set (an overlap filter drops near-duplicates of the OnIt set and the pack; re-checked on 2026-11-13 against the sets as they are now: 0 of 36,032 released examples is a near-duplicate).
  The generated data is not published; regenerate it with the commands in section 5.
- **No hosted model** wrote, labelled or paraphrased any training data, and none was used to score the model. Spend: $0.
- **Run:** seed 0, one run per round (no seed sweep), so the differences between rounds are single-run differences.

## 10. Process notes (what we would tell the next person)

1. **Fix the data, not the model.** Every regression in this project was a hole or a conflict in the data; changing
   hyper-parameters was never the answer. Add the failing family *and* its contrast in the same round.
2. **The 1B teacher is a data debugger.** Check the teacher on a new family before distilling; if it fails, stop and fix the
   data (this caught the history-repeat problem: teacher 70/70, old student 49/70 on the same cases).
3. **Write held-out cases yourself, with names and wordings that are not in the banks**, and check mechanically that none
   appears word for word in the data (`eval/ticket04_check.py --report-overlap`). The first version of this round's held-out
   set had 10 of 70 cases in the data and they had to be reworded.
4. **Exact match hides reply-wording noise.** If a reply is drawn at random per example, strict match cannot reach 100%.
   Make the reply a function of the phrase (done for acknowledgements and statements in this round).
5. **Check the speed path.** The teacher of this round started without the Mamba kernels because the launcher used the wrong
   interpreter (10-14 s/step instead of 5); look for the "falling back to its reference PyTorch implementation" warning in
   the first log lines.
6. **Shared hardware:** other users' jobs ran on every GPU; one GPU was off limits; runs were started by PID-tracked scripts
   and never killed by pattern.
7. **No early stopping.** Three epochs, eval and checkpoint per epoch, the last checkpoint is the one released. The student's
   eval loss was still falling at epoch 3; the teacher's rose slightly after epoch 2.

## 11. Files in this repository, and who owns what

This repository does data and training; the line language, chat format, evaluation sets, serving and acceptance belong to the
voice-agent repository (`docs/ARCHITECTURE.md`). The pinned copies are checked by `tests/test_contract.py`.


`spec/tools.py` (the line language, tool list and grammar), `data/` (generator, banks, chat format),
`train/` (training, distillation, export notes), `scripts/` (teacher-logit cache, merge), `eval/`
(dev and held-out checks), `tests/` (73 tests), `data/released/` (the generated data, section 4),
`docs/CLUSTER-LOG.md` (the round-by-round log). The tickets that drove each round, with their execution logs, are in the voice-agent repository under `.scratch/router-web-answer/tickets/`.

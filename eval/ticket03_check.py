"""Ticket 03 checks, run with generation through a LoRA adapter (greedy, one example at a time).

    python eval/ticket03_check.py --model ibm-granite/granite-4.0-h-350m --adapter runs/kd-350m-web3-s0/final dev
    python eval/ticket03_check.py --model ibm-granite/granite-4.0-h-350m --adapter runs/kd-350m-web3-s0/final heldout

`dev` scores the generator's dev split per kind (the new seeds first). `heldout` runs 70 hand-written cases whose
names and wordings are not in the data banks: 40 multi-turn (a home action, then an unrelated request) and 30
across the new seeds (12 Pi, 10 delegate requests, 8 statements). Wrong answers are listed one by one.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "data"))
import banks as B  # noqa: E402
from format import build_messages  # noqa: E402

BOOK = dict(B.CONTACT_NAMES[:7])  # Anna, Ben, Mom, Dad, Dr. Reyes, Carlo, Maria; Ingrid, Omar, Fiona are not in it

# (earlier user, earlier lines): a home action whose number or name the next request must not reuse
HISTORIES = [
    ("make it 22 degrees", "TEMPERATURE|22 degrees"),
    ("timer for 15 minutes", "TIMER|15 minutes"),
    ("alarm for 6:30 AM", "ALARM|6:30 AM"),
    ("brightness 40 percent", "BRIGHTNESS|40 percent"),
    ("set the speaker to 60 percent", "set_volume|60"),
    ("make the lights blue", "COLOR|blue"),
    ("call Anna", "CALL|Anna"),
    ("text Ben i'm late", "MESSAGE|Ben"),
]
# (an unrelated final request, the expected call: a full line, or only a name)
FINALS = [
    ("call Ingrid", "none"),
    ("phone Omar", "none"),
    ("ring Fiona", "none"),
    ("what's the weather in Davao", "weather"),
    ("what time is it in Sydney", "get_current_time"),
    ("turn off the lights", "LIGHT_OFF"),
    ("skip this track", "NEXT"),
    ("who is the current governor of Cebu", "web_answer"),
    ("what's the exchange rate of the yen to the peso today", "web_answer"),
    ("what's forty five times twelve", "escalate|reasoning"),
    ("remind me to buy rice", "CREATE_REMINDER|buy rice"),
    ("set a timer", "none"),
    ("play something relaxing", "PLAY_MUSIC"),
    ("call Mom", "CALL|Mom"),
    ("pause", "PAUSE"),
    ("tell me about the Eiffel tower", "wiki_summary"),
    ("good morning", "none"),
    ("set an alarm", "none"),
    ("is the lamp on", "escalate"),
    ("text Maria that dinner is ready", "MESSAGE|Maria"),
]


def name_of(line: str) -> str:
    return line.split("|")[0].strip()


def multi_turn_cases() -> list[dict]:
    cases = []
    for i, (user, expected) in enumerate(FINALS):
        for shift in (0, 3):
            hist = HISTORIES[(i + shift) % len(HISTORIES)]
            if name_of(hist[1]) == name_of(expected):
                hist = HISTORIES[(i + shift + 1) % len(HISTORIES)]
            cases.append({"group": "multi-turn", "history": [hist], "user": user, "expect": expected})
    return cases


def other_cases() -> list[dict]:
    status = ("how is the speaker doing", "device_status")
    loud = ("how loud is the speaker", "get_volume")
    setv = ("set the speaker to 40 percent", "set_volume|40")
    code = ("create a sudoku game in the background", "escalate|code")
    lint = ("run the linter in the background", "escalate|tools")
    cases = [
        ("pi", [status], "set the speaker to 35 percent", "set_volume|35"),
        ("pi", [status], "put the volume at 80%", "set_volume|80"),
        ("pi", [status], "change the pi's volume to seventy percent", "set_volume|70"),
        ("pi", [loud], "what's the pie temperature", "device_status"),
        ("pi", [("make it louder", "VOLUME_UP")], "how hot is the raspberry pie", "device_status"),
        ("pi", [], "set volume to 25 percent", "set_volume|25"),
        ("pi", [], "run a ping test and tell me the latency", "escalate|tools"),
        ("pi", [setv], "check the internet speed", "escalate|tools"),
        ("pi", [status], "what volume is the speaker on", "get_volume"),
        ("pi", [setv], "is the speaker plugged in", "device_status"),
        (
            "pi",
            [status, ("set the speaker to 50 percent", "set_volume|50")],
            "set the speaker to 55 percent",
            "set_volume|55",
        ),
        ("pi", [], "how much memory is left on the pie", "device_status"),
        ("delegate", [], "write a text adventure game in the background", "escalate|code"),
        ("delegate", [], "run the integration tests in the sandbox and tell me how many pass", "escalate|tools"),
        ("delegate", [code], "is it done yet", "escalate|tools"),
        ("delegate", [("build a web scraper", "escalate|code")], "are the results ready", "escalate|tools"),
        ("delegate", [], "check on my render job", "escalate|tools"),
        ("delegate", [], "cancel the backup job", "escalate|tools"),
        ("delegate", [], "build me a pomodoro app", "escalate|code"),
        ("delegate", [], "start the benchmark in the background and tell me when it finishes", "escalate|tools"),
        ("delegate", [lint], "did it finish yet", "escalate|tools"),
        ("delegate", [], "kick off the data export, check for errors, and email me a summary", "escalate|workflow"),
    ]
    reactions = [
        (
            "who won the last tennis open",
            "web_answer|Who won the last tennis open?",
            "huh i thought it was someone else",
        ),
        ("price of gold today", "web_answer|What is the price of gold today?", "that's more than i expected"),
        ("timer for 5 minutes", "TIMER|5 minutes", "nice one"),
        ("what is the newest laptop", "web_answer|What is the newest laptop?", "so it just came out"),
        ("how do i learn chess faster", "escalate|reasoning", "ah okay that makes sense"),
        ("what's the weather", "weather", "figures"),
        ("how much is a new car", "web_answer|How much is a new car?", "wow that's expensive"),
        ("call Anna", "CALL|Anna", "that was quick"),
    ]
    cases += [("statement", [(u, t)], r, "none") for u, t, r in reactions]
    return [{"group": g, "history": h, "user": u, "expect": e} for g, h, u, e in cases]


def is_right(pred: str, expect: str, group: str) -> bool:
    first = pred.splitlines()[0].strip() if pred.strip() else ""
    ok = first == expect if "|" in expect else name_of(first) == expect
    if group == "statement":
        ok = ok and len(first.split("|", 1)[-1].split()) <= 6
    return ok


def repeats_previous(pred: str, case: dict) -> bool:
    """The reply reuses the earlier call (same line or same tool) although the request is a different one."""
    if not case["history"]:
        return False
    previous = case["history"][-1][1].splitlines()[-1]
    first = pred.splitlines()[0].strip() if pred.strip() else ""
    return bool(first) and name_of(first) == name_of(previous) and name_of(case["expect"]) != name_of(previous)


def load(model: str, adapter: str):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    base = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16).cuda()
    return tok, PeftModel.from_pretrained(base, adapter).eval()


def generate(tok, model, messages: list[dict]) -> str:
    import torch

    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=64, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids["input_ids"].shape[1] :], skip_special_tokens=True).strip()


def run_heldout(tok, model) -> None:
    cases = multi_turn_cases() + other_cases()
    by_group: dict[str, list[bool]] = collections.defaultdict(list)
    repeats = 0
    wrong = []
    for c in cases:
        pred = generate(tok, model, build_messages(BOOK, c["history"], c["user"]))
        right = is_right(pred, c["expect"], c["group"])
        by_group[c["group"]].append(right)
        if c["group"] == "multi-turn" and repeats_previous(pred, c):
            repeats += 1
            wrong.append(("REPEAT", c, pred))
        elif not right:
            wrong.append(("wrong", c, pred))
    for g, rs in by_group.items():
        print(f"{g}: {sum(rs)}/{len(rs)}")
    print(f"multi-turn replies that reuse the previous call: {repeats}/{len(by_group['multi-turn'])}")
    for tag, c, pred in wrong:
        hist = c["history"][-1][1] if c["history"] else "-"
        print(f"  {tag}: [{hist}] {c['user']!r} -> {pred!r} (want {c['expect']})")


REPLY_KINDS = ("ack", "statement")  # the reply wording is free: right means a short `none|` reply, nothing else


def reply_ok(pred: str, target: str) -> bool:
    first = pred.splitlines()[0].strip() if pred.strip() else ""
    return name_of(first) == "none" and len(first.split("|", 1)[-1].split()) <= 6 and len(pred.splitlines()) == 1


def run_dev(tok, model, data: Path, limit: int | None, save: Path | None) -> None:
    rows = [json.loads(x) for x in (data / "dev.jsonl").read_text().splitlines()]
    seen: dict[str, int] = collections.Counter()
    strict: dict[str, list[bool]] = collections.defaultdict(list)
    behaviour: dict[str, list[bool]] = collections.defaultdict(list)
    misses, saved = [], []
    for r in rows:
        if limit and seen[r["kind"]] >= limit:
            continue
        seen[r["kind"]] += 1
        pred = generate(tok, model, r["messages"][:-1])
        ok = pred == r["target"].strip()
        strict[r["kind"]].append(ok)
        good = reply_ok(pred, r["target"]) if r["kind"] in REPLY_KINDS else ok
        behaviour[r["kind"]].append(good)
        saved.append(
            {
                "id": r["id"],
                "kind": r["kind"],
                "user": r["messages"][-2]["content"],
                "pred": pred,
                "target": r["target"],
            }
        )
        if not good:
            misses.append((r["kind"], r["messages"][-2]["content"], pred, r["target"]))
    n = sum(len(v) for v in strict.values())
    print(
        f"overall exact {sum(sum(v) for v in strict.values())}/{n}; with free reply wording for {REPLY_KINDS}: "
        f"{sum(sum(v) for v in behaviour.values())}/{n}"
    )
    for kind in sorted(strict):
        extra = f" (right behaviour {sum(behaviour[kind])}/{len(behaviour[kind])})" if kind in REPLY_KINDS else ""
        print(f"  {kind}: {sum(strict[kind])}/{len(strict[kind])}{extra}")
    for kind, user, pred, want in misses[:80]:
        print(f"  miss [{kind}] {user!r} -> {pred[:70]!r} want {want[:50]!r}")
    if save:
        save.write_text("".join(json.dumps(x) + "\n" for x in saved))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=("dev", "heldout"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--data", type=Path, default=HERE.parent / "data/distill")
    ap.add_argument("--limit", type=int, default=None, help="dev: at most this many examples per kind")
    ap.add_argument("--save", type=Path, default=None, help="dev: write every prediction to this JSONL file")
    a = ap.parse_args()
    tok, model = load(a.model, a.adapter)
    if a.mode == "heldout":
        run_heldout(tok, model)
    else:
        run_dev(tok, model, a.data, a.limit, a.save)


if __name__ == "__main__":
    main()

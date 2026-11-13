"""Ticket 04 held-out check, run with generation through a LoRA adapter (greedy, one example at a time).

    python eval/ticket04_check.py --model ibm-granite/granite-4.0-h-350m --adapter runs/kd-350m-web4-s0/final

70 hand-written cases whose names and wordings are not in the data banks: 60 across the six families (12 negated,
15 Filipino incl. three-word forms, 8 near miss, 10 play target incl. bare music, 8 no referent with and without a
resolving history, 7 job requests) and 10 confirmations (6 after the hosted model's offer, 4 contrasts). Wrong answers
are listed one by one. `--report-overlap DIR` lists cases whose exact wording is in a data file (should be none).
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "data"))
from format import build_messages  # noqa: E402
from ticket03_check import BOOK, generate, load, name_of  # noqa: E402

# A contact outside the phonebook: Ingrid, Omar, Fiona. `want` is a full line or only a tool name; `check` adds a rule
# on the reply of a `none|` line: "short" (<= 6 words), "q" (a question), "fwd" (<= 8 words).
OFFER = "none|Yes, it works. Want me to open it?"
CASES: list[tuple[str, list[tuple[str, str]], str, str, str]] = [  # (group, history, user, want, check)
    # negated
    ("negated", [], "i'd rather you didn't turn on the lights", "none", "short"),
    ("negated", [], "hold off on calling Ben", "none", "short"),
    ("negated", [], "leave the volume alone, don't touch it", "none", "short"),
    ("negated", [], "don't bother texting Dad", "none", "short"),
    ("negated", [], "do not set any alarm for me", "none", "short"),
    ("negated", [], "whatever you do, don't switch off the lights", "none", "short"),
    ("negated", [], "quit calling Maria", "none", "short"),
    ("negated", [], "wag mo na lang i-text si Maria", "none", "short"),
    ("negated", [], "no need to wake me up tomorrow", "none", "short"),
    ("negated", [], "don't you dare skip this song", "none", "short"),
    ("negated", [("play some music", "PLAY_MUSIC")], "please never pause the music again", "none", "short"),
    ("negated", [], "don't play anything right now", "none", "short"),
    # Filipino and Taglish
    ("filipino", [], "pakibuksan ang ilaw", "LIGHT_ON", ""),
    ("filipino", [], "patayin mo na ang ilaw", "LIGHT_OFF", ""),
    ("filipino", [], "pakilakasan naman ang volume", "VOLUME_UP", ""),
    ("filipino", [], "hinaan mo nga ang volume", "VOLUME_DOWN", ""),
    ("filipino", [], "tawagan mo si Dr. Reyes", "CALL|Dr. Reyes", ""),
    ("filipino", [], "pakitext mo si Maria", "MESSAGE|Maria", ""),
    ("filipino", [], "tawagan si Ingrid", "none|Ingrid is not in the phonebook.", ""),
    ("filipino", [], "gisingin mo ako ng seven thirty AM", "ALARM|7:30 AM", ""),
    ("filipino", [], "mag-timer ka nga ng ten minutes", "TIMER|10 minutes", ""),
    ("filipino", [], "anong oras na ngayon sa Tokyo", "get_current_time", ""),
    ("filipino", [], "uulan ba ngayon", "weather", ""),
    ("filipino", [], "kumusta ang panahon sa Baguio bukas ng umaga", "weather", ""),
    ("filipino", [], "patayin na ang ilaw", "LIGHT_OFF", ""),
    ("filipino", [], "hinaan mo konti", "VOLUME_DOWN", ""),
    ("filipino", [], "i-skip mo yan", "NEXT", ""),
    # near misses
    ("near-miss", [], "bring the lights down to 25 percent", "BRIGHTNESS|25 percent", ""),
    ("near-miss", [], "please dim the lights", "none", "q"),
    ("near-miss", [], "go back one track", "none", "fwd"),
    ("near-miss", [], "go back to the song before", "none", "fwd"),
    ("near-miss", [("next song", "NEXT")], "play the previous one again", "none", "fwd"),
    ("near-miss", [("pause the music", "PAUSE")], "please resume the music", "PLAY_MUSIC", ""),
    ("near-miss", [], "continue playing the song", "PLAY_MUSIC", ""),
    ("near-miss", [("pause", "PAUSE")], "unpause please", "PLAY_MUSIC", ""),
    # play target and bare music
    ("play-target", [], "can i play connect four", "escalate|tools", ""),
    ("play-target", [], "open the spreadsheet app you wrote", "escalate|tools", ""),
    ("play-target", [], "play the platformer you built", "escalate|tools", ""),
    ("play-target", [], "play something by Rihanna", "escalate|tools", ""),
    ("play-target", [], "put on some Ariana Grande", "escalate|tools", ""),
    ("play-target", [], "play Fur Elise", "escalate|tools", ""),
    ("play-target", [], "play some salsa", "escalate|tools", ""),
    ("play-target", [], "play my gym playlist", "escalate|tools", ""),
    ("play-target", [], "play some music please", "PLAY_MUSIC", ""),
    ("play-target", [], "can you put some tunes on", "PLAY_MUSIC", ""),
    # no referent, and the same words after a history that resolves them
    ("no-referent", [], "set it to forty", "none", "q"),
    ("no-referent", [], "do that one more time", "none", "q"),
    ("no-referent", [], "turn this off", "none", "q"),
    ("no-referent", [], "make it a bit higher", "none", "q"),
    (
        "no-referent",
        [("what's the weather in Manila", "weather|Manila")],
        "and for tomorrow",
        "weather|Manila|tomorrow",
        "",
    ),
    (
        "no-referent",
        [("set the temperature to 24 degrees", "TEMPERATURE|24 degrees")],
        "set it to 21",
        "TEMPERATURE|21 degrees",
        "",
    ),
    (
        "no-referent",
        [("timer for 10 minutes", "TIMER|10 minutes")],
        "change that to 20 minutes",
        "TIMER|20 minutes",
        "",
    ),
    ("no-referent", [], "set it to 22 degrees please", "TEMPERATURE|22 degrees", ""),
    # job requests
    ("job", [], "is the platformer finished", "escalate|tools", ""),
    ("job", [], "did the sudoku build finish", "escalate|tools", ""),
    ("job", [], "how's the app coming along", "escalate|tools", ""),
    ("job", [], "run a directory listing of the sandbox", "escalate|tools", ""),
    ("job", [], "show the files in the sandbox", "escalate|tools", ""),
    ("job", [], "is the python script ready now", "escalate|tools", ""),
    ("job", [], "give me a listing of the sandbox directory", "escalate|tools", ""),
    # confirmations: after an offer (escalate|other), after the router's own question and after no offer (unchanged)
    (
        "confirm",
        [("make me a calculator", "none|Done, it's saved on the desktop. Shall I launch it?")],
        "yeah launch it",
        "escalate|other",
        "",
    ),
    (
        "confirm",
        [("what is in the downloads folder", "none|Fifteen files. Do you want the list?")],
        "go on then",
        "escalate|other",
        "",
    ),
    (
        "confirm",
        [("find my resume", "none|It's in documents. Want me to open the file?")],
        "mm sure",
        "escalate|other",
        "",
    ),
    (
        "confirm",
        [("is the render finished", "none|Not yet, about ten minutes. Should I tell you when it's ready?")],
        "oo please",
        "escalate|other",
        "",
    ),
    (
        "confirm",
        [("how's the build going", "none|It passed. Want the details?")],
        "not today thanks",
        "escalate|other",
        "",
    ),
    ("confirm", [("write me a haiku", "none|Here it is. Want another one?")], "sige lang", "escalate|other", ""),
    ("confirm", [("set a timer", "none|For how long?")], "twenty minutes", "TIMER|20 minutes", ""),
    ("confirm", [("call someone", "none|Who should I call?")], "yung kay Carlo", "CALL|Carlo", ""),
    ("confirm", [("set an alarm", "none|For what time?")], "okay, seven AM", "ALARM|7:00 AM", ""),
    ("confirm", [("who won the match", "none|They won by two.")], "alright then", "none", "short"),
]


def is_right(pred: str, want: str, check: str) -> bool:
    first = pred.splitlines()[0].strip() if pred.strip() else ""
    ok = first == want if "|" in want else name_of(first) == want
    reply = first.split("|", 1)[-1]
    if ok and check == "short":
        ok = len(reply.split()) <= 6
    if ok and check == "q":
        ok = reply.endswith("?")
    if ok and check == "fwd":
        ok = len(reply.split()) <= 8
    return ok and len(pred.splitlines()) == 1


def overlap(data: Path) -> list[str]:
    texts = {c[2].lower() for c in CASES}
    hits = set()
    for split in ("train", "dev", "test"):
        for ln in (data / f"{split}.jsonl").read_text().splitlines():
            user = json.loads(ln)["messages"][-2]["content"].lower()
            if user in texts:
                hits.add(user)
    return sorted(hits)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--report-overlap", type=Path, default=None)
    a = ap.parse_args()
    if a.report_overlap:
        print("cases whose exact wording is in the data:", overlap(a.report_overlap))
    tok, model = load(a.model, a.adapter)
    by_group: dict[str, list[bool]] = collections.defaultdict(list)
    wrong = []
    for group, history, user, want, check in CASES:
        pred = generate(tok, model, build_messages(BOOK, history, user))
        right = is_right(pred, want, check)
        by_group[group].append(right)
        if not right:
            wrong.append((group, history[-1][1] if history else "-", user, pred, want))
    for g, rs in by_group.items():
        print(f"{g}: {sum(rs)}/{len(rs)}")
    total = [r for rs in by_group.values() for r in rs]
    print(f"total: {sum(total)}/{len(total)}")
    for group, hist, user, pred, want in wrong:
        print(f"  wrong [{group}] [{hist}] {user!r} -> {pred!r} (want {want})")


if __name__ == "__main__":
    main()

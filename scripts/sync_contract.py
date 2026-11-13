"""Keep the files this repository takes from the voice-agent repository in step with it.

The voice-agent repository (https://github.com/Martinnavs/AI-231-ME3) owns the *contract*: the line language the router
must write, the chat format it is served with, and the two evaluation sets. This repository only trains against them,
so it holds copies, pinned in `contract.lock.json` (sha256 of each file and the commit they came from).

    python scripts/sync_contract.py --check                       local files match the lock (the tests run this)
    python scripts/sync_contract.py --diff ../AI-231-ME3          drift against a checkout of the voice-agent repository
    python scripts/sync_contract.py --update-from ../AI-231-ME3   copy the files from that checkout and rewrite the lock

Never edit the copies by hand: change them in the voice-agent repository, then sync.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
LOCK = HERE / "contract.lock.json"
SOURCE = "https://github.com/Martinnavs/AI-231-ME3"
# path here -> path in the voice-agent repository
FILES = {
    "spec/tools.py": "finetune/spec/tools.py",
    "data/format.py": "finetune/data/format.py",
    "data/agent_tools_snapshot.json": "finetune/data/agent_tools_snapshot.json",
    "eval/onit_eval.jsonl": "finetune/eval/onit_eval.jsonl",
    "eval/agent_prompts.jsonl": "pipeline/config/eval/agent_prompts.jsonl",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check() -> list[str]:
    """Problems between the local copies and the lock (empty when they agree)."""
    lock = json.loads(LOCK.read_text())["files"]
    problems = [f"{name}: not in the lock" for name in FILES if name not in lock]
    for name, want in lock.items():
        if not (HERE / name).exists():
            problems.append(f"{name}: missing")
        elif sha(HERE / name) != want:
            problems.append(f"{name}: edited since the last sync (change it in the voice-agent repository, then sync)")
    return problems


def drift(checkout: Path) -> list[str]:
    return [
        f"{name}: differs from {src}"
        for name, src in FILES.items()
        if not (checkout / src).exists() or sha(checkout / src) != sha(HERE / name)
    ]


def update(checkout: Path) -> None:
    for name, src in FILES.items():
        shutil.copyfile(checkout / src, HERE / name)
    commit = subprocess.run(
        ["git", "-C", str(checkout), "log", "-1", "--format=%H", "--", *FILES.values()],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    LOCK.write_text(
        json.dumps(
            {"source": SOURCE, "source_commit": commit, "files": {n: sha(HERE / n) for n in FILES}},
            indent=2,
        )
        + "\n"
    )
    print(f"synced {len(FILES)} files from {checkout} (last commit touching them: {commit[:8]})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true")
    group.add_argument("--diff", type=Path, metavar="CHECKOUT")
    group.add_argument("--update-from", type=Path, metavar="CHECKOUT")
    a = ap.parse_args()
    if a.update_from:
        update(a.update_from)
        return
    problems = check() if a.check else drift(a.diff)
    print("\n".join(problems) or "in step")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()

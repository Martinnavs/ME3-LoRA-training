"""Build a distillation set: the seed-0 train split plus every example of a second generator seed.

    python3 scripts/merge_distill_data.py data/generated data/generated-s1 data/distill

The teacher labels by logits, so extra prompts need no new targets; the second seed only adds wording and
slot combinations. dev and test are copied from the first directory, and any extra example whose user text
appears in them (same history and same final turn) is dropped so the distilled student is never trained on a held-out prompt.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


def user_text(row: dict) -> str:
    """The whole prompt (earlier turns and the final user turn), not the phonebook: a short reply such as "okay"
    is held out only after the same history, so it does not drop every other history that ends the same way."""
    return " | ".join(m["content"].strip().lower() for m in row["messages"][1:-1])


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main(base: Path, extra: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    held = {user_text(r) for s in ("dev", "test") for r in rows(base / f"{s}.jsonl")}
    train = rows(base / "train.jsonl")
    seen = {(user_text(r), r["target"]) for r in train}
    added = dropped = 0
    for split in ("train", "dev", "test"):
        for r in rows(extra / f"{split}.jsonl"):
            key = (user_text(r), r["target"])
            if user_text(r) in held or key in seen:
                dropped += 1
                continue
            seen.add(key)
            train.append(r)
            added += 1
    with (out / "train.jsonl").open("w") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for s in ("dev", "test"):
        shutil.copy(base / f"{s}.jsonl", out / f"{s}.jsonl")
    print(f"train {len(train)} (added {added}, dropped {dropped})")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))

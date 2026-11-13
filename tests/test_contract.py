"""The copies taken from the voice-agent repository are exactly the ones in contract.lock.json."""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
import sync_contract as C  # noqa: E402


def test_the_contract_files_match_the_lock():
    assert C.check() == []


def test_the_lock_names_exactly_the_files_that_are_taken_over_and_a_source_commit():
    lock = json.loads(C.LOCK.read_text())
    assert set(lock["files"]) == set(C.FILES)
    assert len(lock["source_commit"]) == 40 and lock["source"].startswith("https://github.com/")


def test_nothing_in_the_training_code_reaches_into_the_voice_agent_repository():
    banned = ("pipeline/", "src/voice_agent", "parents[2]", "parents[3]")
    for path in [
        *HERE.glob("data/*.py"),
        *HERE.glob("train/*.py"),
        *HERE.glob("scripts/*.py"),
        *HERE.glob("eval/*.py"),
    ]:
        if path.name == "sync_contract.py":
            continue
        text = path.read_text()
        assert not any(b in text for b in banned), f"{path.name} depends on the voice-agent repository layout"

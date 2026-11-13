"""Label masking and batching used by train/train.py."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "train"))
import utils as U  # noqa: E402


class Tok:
    """A stub tokenizer: one token per character; the chat template is a plain text rendering."""

    eos_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


MESSAGES = [
    {"role": "system", "content": "s"},
    {"role": "user", "content": "lights on"},
    {"role": "assistant", "content": "LIGHT_ON"},
]


def test_only_the_target_and_the_end_token_carry_loss():
    enc = U.encode_example(Tok(), MESSAGES, 200)
    labels = enc["labels"]
    n_target = len("LIGHT_ON") + 1
    assert labels[-n_target:] == [ord(c) for c in "LIGHT_ON"] + [0]
    assert all(x == U.IGNORE for x in labels[:-n_target])
    assert len(enc["input_ids"]) == len(labels) == len(enc["attention_mask"])


def test_an_example_longer_than_the_limit_is_dropped_not_cut():
    assert U.encode_example(Tok(), MESSAGES, 10) is None


def test_batches_are_right_padded_with_ignored_labels():
    a = U.encode_example(Tok(), MESSAGES, 200)
    b = U.encode_example(
        Tok(), MESSAGES[:1] + [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "none|Hi!"}], 200
    )
    batch = U.pad_batch([a, b], pad_id=7)
    assert len({len(x) for x in batch["input_ids"]}) == 1
    short = batch["input_ids"][1] if len(b["input_ids"]) < len(a["input_ids"]) else batch["input_ids"][0]
    assert short[-1] == 7 and batch["labels"][1 if len(b["input_ids"]) < len(a["input_ids"]) else 0][-1] == U.IGNORE


def test_exact_match_needs_the_whole_reply():
    assert U.exact_match(["A\nB", "A"], ["A\nB", "A\nB"]) == 0.5
    assert U.exact_match([], []) == 0.0


def test_topk_kl_equals_the_full_kl_when_k_is_the_whole_vocabulary_and_is_close_for_a_peaked_teacher():
    import torch
    import torch.nn.functional as F

    g = torch.Generator().manual_seed(0)
    student = torch.randn(6, 50, generator=g)
    teacher = torch.randn(6, 50, generator=g) * 4  # a peaked teacher
    exact = F.kl_div(F.log_softmax(student / 2, -1), F.softmax(teacher / 2, -1), reduction="batchmean") * 4
    vals, idx = teacher.topk(50, -1)
    assert torch.allclose(U.topk_kl(student, vals, idx, 2.0), exact, atol=1e-5)
    vals, idx = teacher.topk(20, -1)
    assert abs(float(U.topk_kl(student, vals, idx, 2.0)) - float(exact)) < 0.15 * float(exact)

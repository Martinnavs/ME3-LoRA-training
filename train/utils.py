"""Training helpers that need no GPU libraries, so they are tested here and used unchanged on the cluster."""

from __future__ import annotations

IGNORE = -100  # label value that the loss skips


def encode_example(tokenizer, messages: list[dict[str, str]], max_len: int) -> dict[str, list[int]] | None:
    """Tokenise a chat example with the loss on the assistant's final message only.

    `messages` ends with the assistant target. The prompt is everything before it, rendered with the model's own
    chat template plus the generation prompt; the completion is the target followed by the end-of-turn token.
    Returns None when the example is longer than `max_len` (it is dropped, not truncated: a cut target is wrong).
    """
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    target = messages[-1]["content"]
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    if len(prompt_ids) + len(target_ids) > max_len:
        return None
    return {
        "input_ids": prompt_ids + target_ids,
        "attention_mask": [1] * (len(prompt_ids) + len(target_ids)),
        "labels": [IGNORE] * len(prompt_ids) + target_ids,
    }


def pad_batch(features: list[dict[str, list[int]]], pad_id: int) -> dict[str, list[list[int]]]:
    """Right-pad a batch to its longest example (lists of ints; the caller turns them into tensors)."""
    longest = max(len(f["input_ids"]) for f in features)
    out: dict[str, list[list[int]]] = {"input_ids": [], "attention_mask": [], "labels": []}
    for f in features:
        gap = longest - len(f["input_ids"])
        out["input_ids"].append(f["input_ids"] + [pad_id] * gap)
        out["attention_mask"].append(f["attention_mask"] + [0] * gap)
        out["labels"].append(f["labels"] + [IGNORE] * gap)
    return out


def exact_match(predictions: list[str], targets: list[str]) -> float:
    """Share of predictions equal to the target after stripping (the whole multi-line reply must match)."""
    if not targets:
        return 0.0
    return sum(p.strip() == t.strip() for p, t in zip(predictions, targets, strict=True)) / len(targets)


def topk_kl(student_logits, teacher_vals, teacher_idx, temp: float):
    """KL(teacher || student) at temperature `temp`, with the teacher known only at its top-k logits per position.

    student_logits [N, V]; teacher_vals / teacher_idx [N, k] (the teacher's k largest logits and their token ids).
    The teacher distribution is renormalised over those k tokens; the student's log-probability is taken over the
    full vocabulary. With k = V this is the exact `F.kl_div(..., reduction="batchmean") * temp**2`.
    """
    import torch.nn.functional as F

    t_logp = F.log_softmax(teacher_vals.float() / temp, -1)
    s_logp = F.log_softmax(student_logits.float() / temp, -1).gather(-1, teacher_idx.long())
    return (t_logp.exp() * (t_logp - s_logp)).sum(-1).mean() * temp * temp

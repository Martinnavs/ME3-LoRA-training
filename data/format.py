"""The chat format the router is trained and served with (one place, so training and evaluation agree).

A training example is a list of chat messages: a short system message (rules, phonebook), optional earlier
turns, the user's words, and the assistant's reply, which is the line(s) of the output language in
`spec/tools.py`. The system message is deliberately short (about 150 tokens) and carries only what changes
between users: the phonebook. Everything else (the intents, the tools, the slot forms) is learned.
"""

from __future__ import annotations

SYSTEM_TEMPLATE = """Route the user's words to calls, one per line. Reply with the calls only.
Handle simple home, time, weather and Wikipedia requests yourself; defer everything else with escalate|reason.
Phonebook: {phonebook}"""


def render_system(phonebook: dict[str, str]) -> str:
    """The system message for a phonebook {name: relation}."""
    book = ", ".join(f"{name} ({relation})" for name, relation in phonebook.items()) or "empty"
    return SYSTEM_TEMPLATE.format(phonebook=book)


def build_messages(
    phonebook: dict[str, str], history: list[tuple[str, str]], user: str, target: str | None = None
) -> list[dict[str, str]]:
    """Chat messages: system, earlier (user, assistant-lines) pairs, the user's words, optionally the target."""
    messages = [{"role": "system", "content": render_system(phonebook)}]
    for said, lines in history:
        messages.append({"role": "user", "content": said})
        messages.append({"role": "assistant", "content": lines})
    messages.append({"role": "user", "content": user})
    if target is not None:
        messages.append({"role": "assistant", "content": target})
    return messages

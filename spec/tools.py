"""The single source of truth for the local model's output language (the "line protocol").

Everything that must agree imports this file: the data generator (what a valid line is), the training
script (the chat format), the GBNF grammar the server enforces, the parser the agent uses to turn a
line into a real tool call, and the OpenAI-format schemas the hosted model sees. Standard library only,
so it runs unchanged on the cluster.

Output language. The model replies with one call per line:

    NAME                      a call without arguments
    NAME|arg1|arg2|...        positional arguments

A call has `len(args)` positional arguments. The line is split on "|" at most `len(args) - 1` times,
so the LAST argument may itself contain "|" (a shell pipe, a regular expression). Arguments never
contain a newline. Missing trailing arguments take the defaults in the spec. Home commands are the
17 smart_home intents written as NAME|slot. `none|reply` means no tool (chat, a question back to the
user).

The small model is a ROUTER for simple requests: one tool call (or independent calls) whose arguments
can be read off the user's words. Anything that needs planning, iteration, code or file contents, or
real reasoning is deferred: `escalate|why`, why in {tools, code, workflow, reasoning, other}. The orchestrator
then runs the hosted Gemini model with the full tool schemas for that turn. `tools` = a request for a tool whose
result has to be summarised for speech (documents, files, shell, web) - the hosted model owns those;
`code` = write or edit files;
`workflow` = several dependent steps (build an app, audit a host, research and compare); `reasoning` =
an answer that needs thought, not a tool; `other` = out of scope or unclear.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Arg:
    name: str
    type: str = "string"  # string | integer | enum
    description: str = ""
    required: bool = True
    default: Any = None
    enum: tuple[str, ...] = ()


@dataclass(frozen=True)
class Tool:
    name: str  # the line name; for tools it is also the real tool name
    description: str
    args: tuple[Arg, ...] = ()
    group: str = "tool"  # home | info | files | shell | web | docs | pi | control
    local: bool = True  # False: the small model escalates (why=code) instead of producing the arguments
    example: str = ""


HOME_SLOT_FORMS = {
    "BRIGHTNESS": "40 percent",
    "COLOR": "purple",
    "TEMPERATURE": "24 degrees",
    "TIMER": "45 minutes",
    "ALARM": "6:30 AM",
    "CREATE_REMINDER": "what to remember, in the user's words",
    "CALL": "contact name",
    "MESSAGE": "contact name",
}
HOME_NO_SLOT = (
    "PLAY_MUSIC",
    "PAUSE",
    "STOP",
    "NEXT",
    "VOLUME_UP",
    "VOLUME_DOWN",
    "LIGHT_ON",
    "LIGHT_OFF",
    "LIST_REMINDERS",
)
PHONEBOOK = {
    "Anna": "sister",
    "Ben": "brother",
    "Mom": "mother",
    "Dr. Reyes": "dentist",
    "Carlo": "friend",
}


HOME_ORDER = (  # the enum order of the production smart_home tool
    "PLAY_MUSIC",
    "PAUSE",
    "STOP",
    "NEXT",
    "VOLUME_UP",
    "VOLUME_DOWN",
    "LIGHT_ON",
    "LIGHT_OFF",
    "BRIGHTNESS",
    "COLOR",
    "TEMPERATURE",
    "TIMER",
    "ALARM",
    "CALL",
    "MESSAGE",
    "CREATE_REMINDER",
    "LIST_REMINDERS",
)


def _home() -> list[Tool]:
    tools = []
    for n in HOME_ORDER:
        if n in HOME_SLOT_FORMS:
            form = HOME_SLOT_FORMS[n]
            tools.append(
                Tool(
                    n,
                    f"Home command {n}; slot: {form}.",
                    (Arg("slot", description=form),),
                    "home",
                    example=f"{n}|{form}",
                )
            )
        else:
            tools.append(Tool(n, f"Home command {n}; takes no slot.", group="home", example=n))
    return tools


TOOLS: tuple[Tool, ...] = tuple(
    _home()
    + [
        # --- information the agent already has
        Tool(
            "get_current_time",
            "The current time in an IANA time zone.",
            (Arg("timezone", required=False, default="Asia/Manila"),),
            "info",
            example="get_current_time|Europe/London",
        ),
        Tool(
            "convert_time",
            "Convert a clock time between time zones.",
            (Arg("source_timezone"), Arg("time"), Arg("target_timezone")),
            "info",
            example="convert_time|Asia/Manila|9:00|Europe/London",
        ),
        Tool(
            "weather",
            "Weather for a place (empty = home) today or tomorrow.",
            (
                Arg("place", required=False, default=""),
                Arg("day", "enum", required=False, default="today", enum=("today", "tomorrow")),
            ),
            "info",
            example="weather|Baguio|tomorrow",
        ),
        Tool(
            "wiki_summary",
            "Wikipedia summary of a subject.",
            (Arg("topic"),),
            "info",
            example="wiki_summary|Peru",
        ),
        Tool(
            "web_answer",
            "Look up a factual question on the web and get a short answer to say.",
            (Arg("query"),),
            "info",
            example="web_answer|Who is the current Senate President of the Philippines?",
        ),
        Tool(
            "search_docs",
            "Search the sample documents (Cormac McCarthy corpus).",
            (Arg("query"),),
            "docs",
            example="search_docs|who directed the film of The Road",
        ),
        # --- the Raspberry Pi speaker
        Tool("get_volume", "The Pi speaker's volume.", (), "pi", example="get_volume"),
        Tool(
            "set_volume",
            "Set the Pi speaker's volume in percent.",
            (Arg("percent", "integer"),),
            "pi",
            example="set_volume|40",
        ),
        Tool(
            "device_status",
            "Temperature, memory, uptime of the Pi and whether the speaker is connected.",
            (),
            "pi",
            example="device_status",
        ),
        # --- OnIt parity (docs/ONIT-PARITY.md); names and arguments follow OnIt's tools
        Tool(
            "bash",
            "Run one shell command in the working directory (read-only audit commands, tests, git).",
            (Arg("command"),),
            "shell",
            example="bash|ss -tlnp | grep LISTEN",
        ),
        Tool(
            "serve",
            "Background jobs: start, status, logs, stop, list.",
            (
                Arg("action", "enum", enum=("start", "status", "logs", "stop", "list")),
                Arg("name", required=False, default=""),
                Arg("command", required=False, default=""),
            ),
            "shell",
            example="serve|start|webapp|python app.py",
        ),
        Tool("read_file", "Read a text file.", (Arg("path"),), "files", example="read_file|README.md"),
        Tool(
            "grep",
            "Regex search across files.",
            (Arg("pattern"), Arg("path", required=False, default=".")),
            "files",
            example="grep|TODO|src",
        ),
        Tool(
            "write_file",
            "Create or overwrite a file with the given content.",
            (Arg("path"), Arg("content")),
            "files",
            local=False,
            example="escalate|code",
        ),
        Tool(
            "edit_file",
            "Replace an exact string in a file.",
            (Arg("path"), Arg("old_string"), Arg("new_string")),
            "files",
            local=False,
            example="escalate|code",
        ),
        Tool(
            "web_search",
            "Search the web or news.",
            (
                Arg("query"),
                Arg("type", "enum", required=False, default="web", enum=("web", "news")),
            ),
            "web",
            example="web_search|python 3.14 release date|news",
        ),
        Tool(
            "fetch_content",
            "Fetch a URL and extract its text.",
            (Arg("url"),),
            "web",
            example="fetch_content|https://example.com",
        ),
        Tool(
            "local_search",
            "Search the user's own documents in a folder.",
            (Arg("query"), Arg("path", required=False, default="")),
            "docs",
            example="local_search|refund policy for annual plans|policies",
        ),
        # --- control calls
        Tool(
            "none",
            "No tool: reply to the user (chat, thanks, a question back).",
            (Arg("reply"),),
            "control",
            example="none|You're welcome.",
        ),
        Tool(
            "escalate",
            "Defer this turn to the big model with the full tools: code or file contents, a multi-step workflow, "
            "real reasoning, or out of scope.",
            (Arg("why", "enum", enum=("tools", "code", "workflow", "reasoning", "other")),),
            "control",
            example="escalate|workflow",
        ),
    ]
)
BY_NAME = {t.name: t for t in TOOLS}
HOME_INTENTS = tuple(t.name for t in TOOLS if t.group == "home")


@dataclass
class Call:
    name: str
    args: list[str] = field(default_factory=list)  # positional, as written (strings)


class LineError(ValueError):
    pass


def format_call(call: Call) -> str:
    tool = BY_NAME.get(call.name)
    if tool is None:
        raise LineError(f"unknown name {call.name!r}")
    if len(call.args) > len(tool.args):
        raise LineError(f"{call.name} takes at most {len(tool.args)} arguments")
    for a in call.args:
        if "\n" in a:
            raise LineError("an argument contains a newline (use escalate)")
    args = list(call.args)
    while args and args[-1] == "" and not tool.args[len(args) - 1].required:
        args.pop()  # trailing optional empties are dropped
    return call.name + "".join("|" + a for a in args)


def parse_line(line: str) -> Call:
    """One line -> a Call; LineError if it is not valid. The last argument may contain '|'."""
    line = line.strip()
    name, sep, rest = line.partition("|")
    tool = BY_NAME.get(name)
    if tool is None:
        raise LineError(f"unknown name {name!r}")
    if not sep:
        args: list[str] = []
    elif not tool.args:
        raise LineError(f"{name} takes no arguments")
    else:
        args = rest.split("|", len(tool.args) - 1)
    for i, spec in enumerate(tool.args):
        if i >= len(args):
            if spec.required:
                raise LineError(f"{name}: missing argument {spec.name!r}")
        elif spec.type == "enum" and args[i] and args[i] not in spec.enum:
            raise LineError(f"{name}: {spec.name!r} must be one of {spec.enum}")
        elif spec.type == "integer" and args[i] and not args[i].lstrip("-").isdigit():
            raise LineError(f"{name}: {spec.name!r} must be an integer")
    return Call(name, args)


def parse_output(text: str) -> list[Call]:
    """The model's whole reply (one call per line) -> calls; blank lines are ignored."""
    return [parse_line(x) for x in text.splitlines() if x.strip()]


def to_tool_args(call: Call) -> dict[str, Any]:
    """A parsed call -> the real tool's arguments (defaults filled, integers converted)."""
    tool = BY_NAME[call.name]
    out: dict[str, Any] = {}
    for i, spec in enumerate(tool.args):
        value = call.args[i] if i < len(call.args) and call.args[i] != "" else spec.default
        if value is None:
            continue
        out[spec.name] = int(value) if spec.type == "integer" and value != "" else value
    return out


def _gbnf_piece(tool: Tool, i: int) -> str:
    """The GBNF for argument i of a tool, with its leading '|': an enum alternation or free text."""
    spec = tool.args[i]
    if spec.enum:
        body = "(" + " | ".join(f'"{v}"' for v in spec.enum) + ")"
    else:
        body = "larg" if i == len(tool.args) - 1 else "farg"
    return f'"|" {body}'


def grammar() -> str:
    """GBNF for llama.cpp: one or more calls, one per line. Every call is valid by construction: only known
    names, the right number of arguments (required ones present), enum arguments restricted to their
    values. Free-text arguments exclude '|' except the last one, and no argument contains a newline.
    Rule names are t0, t1, ...: GBNF allows only lowercase letters, digits and dashes in them."""
    rules: list[str] = []
    names: list[str] = []
    for n, t in enumerate(TOOLS):
        sym = f"t{n}"
        names.append(sym)
        required = 0
        while required < len(t.args) and t.args[required].required:
            required += 1
        tail = ""
        for i in range(len(t.args) - 1, required - 1, -1):  # optional arguments nest: ("|" a (...)?)?
            tail = f"({_gbnf_piece(t, i)}{(' ' + tail) if tail else ''})?"
        parts = [f'"{t.name}"'] + [_gbnf_piece(t, i) for i in range(required)] + ([tail] if tail else [])
        rules.append(f"{sym} ::= " + " ".join(parts))
    return (
        'root ::= call ("\\n" call)*\ncall ::= '
        + " | ".join(names)
        + "\n"
        + "\n".join(rules)
        + "\n"
        + "farg ::= [^\\n|]*\nlarg ::= [^\\n]+\n"
    )


def openai_tools(only_local: bool = False) -> list[dict[str, Any]]:
    """The same tools in OpenAI function format (what the hosted model sees). smart_home is the home
    group folded into one function with an `intent` enum, as in production."""
    out: list[dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "smart_home",
                "description": "Control or check the home. One call per command. Intents and slot forms: "
                + "; ".join(
                    f"{t.name}: {t.args[0].description}" if t.args else t.name for t in TOOLS if t.group == "home"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "intent": {"type": "string", "enum": list(HOME_INTENTS)},
                        "slot": {"type": "string"},
                    },
                    "required": ["intent"],
                },
            },
        }
    ]
    for t in TOOLS:
        if t.group in ("home", "control") or (only_local and not t.local):
            continue
        props = {}
        for a in t.args:
            p: dict[str, Any] = {
                "type": "integer" if a.type == "integer" else "string",
                "description": a.description or a.name,
            }
            if a.enum:
                p["enum"] = list(a.enum)
            props[a.name] = p
        out.append(
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": {
                        "type": "object",
                        "properties": props,
                        "required": [a.name for a in t.args if a.required],
                    },
                },
            }
        )
    return out

"""The line protocol contract (spec/tools.py)."""

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "spec"))
import tools as T  # noqa: E402

SNAPSHOT = HERE / "data" / "agent_tools_snapshot.json"


def test_every_example_parses_and_round_trips():
    for t in T.TOOLS:
        call = T.parse_line(t.example)
        assert call.name == (t.name if t.local else "escalate")
        assert T.format_call(call) == t.example


def test_escalate_takes_only_the_known_reasons():
    assert T.parse_line("escalate|workflow").args == ["workflow"] and T.parse_line("escalate|tools").args == ["tools"]
    with pytest.raises(T.LineError):
        T.parse_line("escalate|write_file")


def test_the_last_argument_keeps_pipes():
    call = T.parse_line("bash|ss -tlnp | grep LISTEN | wc -l")
    assert call.args == ["ss -tlnp | grep LISTEN | wc -l"]
    call = T.parse_line("grep|a|b|c")
    assert call.args == ["a", "b|c"]


def test_defaults_and_integer_conversion():
    assert T.to_tool_args(T.parse_line("weather")) == {"place": "", "day": "today"}
    assert T.to_tool_args(T.parse_line("weather|Baguio|tomorrow")) == {
        "place": "Baguio",
        "day": "tomorrow",
    }
    assert T.to_tool_args(T.parse_line("set_volume|40")) == {"percent": 40}
    assert T.to_tool_args(T.parse_line("get_current_time")) == {"timezone": "Asia/Manila"}


@pytest.mark.parametrize(
    "line",
    [
        "nope|x",
        "get_volume|3",
        "set_volume|loud",
        "weather|Baguio|someday",
        "convert_time|Asia/Manila",
        "web_search",
        "serve|launch",
    ],
)
def test_invalid_lines_are_rejected(line):
    with pytest.raises(T.LineError):
        T.parse_line(line)


def test_multiple_calls_one_per_line():
    calls = T.parse_output("CREATE_REMINDER|water the plants\nweather\n")
    assert [c.name for c in calls] == ["CREATE_REMINDER", "weather"]


def test_format_drops_trailing_optional_empties_and_rejects_newlines():
    assert T.format_call(T.Call("weather", ["", "today"])) == "weather||today"
    assert T.format_call(T.Call("weather", [""])) == "weather"
    with pytest.raises(T.LineError):
        T.format_call(T.Call("bash", ["a\nb"]))


def test_grammar_names_every_tool():
    g = T.grammar()
    for t in T.TOOLS:
        assert f'"{t.name}"' in g
    assert g.startswith("root ::=")


def test_openai_tools_match_the_spec():
    tools = T.openai_tools()
    names = [t["function"]["name"] for t in tools]
    assert len(names) == len(set(names)) and "smart_home" in names and "none" not in names and "escalate" not in names
    assert "write_file" in names
    assert "write_file" not in [t["function"]["name"] for t in T.openai_tools(only_local=True)]
    home = tools[0]["function"]["parameters"]["properties"]["intent"]["enum"]
    assert tuple(home) == T.HOME_INTENTS


def test_the_current_agent_tools_are_covered_with_the_same_arguments():
    pack = json.loads(SNAPSHOT.read_text())
    for t in pack:
        fn = t["function"]
        if fn["name"] == "smart_home":
            assert fn["parameters"]["properties"]["intent"]["enum"] == list(T.HOME_INTENTS)
            continue
        spec = T.BY_NAME[fn["name"]]
        wanted = set(fn["parameters"]["properties"]) - {
            "say",
            "top_k",
        }  # `say` is a template; top_k is fixed
        assert wanted <= {"query"} | {a.name for a in spec.args} or fn["name"] == "search_docs", fn["name"]


def test_grammar_constrains_enums_and_arity():
    g = T.grammar()
    rules = {
        line.split(" ::= ")[1].split(" ")[0].strip('"'): line.split(" ::= ", 1)[1]
        for line in g.splitlines()
        if line.startswith("t") and " ::= " in line and not line.startswith("root")
    }
    assert rules["escalate"] == '"escalate" "|" ("tools" | "code" | "workflow" | "reasoning" | "other")'
    assert rules["LIGHT_ON"] == '"LIGHT_ON"'  # no argument allowed
    assert rules["bash"] == '"bash" "|" larg'  # one required free-text argument
    assert '"today" | "tomorrow"' in rules["weather"]
    assert "farg ::= [^\\n|]*" in g and "larg ::= [^\\n]+" in g
    import re

    assert all(
        re.fullmatch(r"[a-z0-9-]+", name) for name in re.findall(r"^([^ ]+) ::=", g, re.M)
    )  # llama.cpp rule-name syntax


def test_web_answer_is_a_local_one_argument_call():
    t = T.BY_NAME["web_answer"]
    assert t.local and [a.name for a in t.args] == ["query"] and t.group == "info"
    call = T.parse_line("web_answer|Who is the current Senate President of the Philippines?")
    assert call.name == "web_answer" and call.args == ["Who is the current Senate President of the Philippines?"]
    assert T.BY_NAME["web_search"].group == "web"  # an OnIt tool the router never emits

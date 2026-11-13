"""The data generator: labels parse, nothing leaks from the evaluation sets, splits are disjoint, runs repeat."""

import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "data"))
sys.path.insert(0, str(HERE / "spec"))
import generate as G  # noqa: E402
import tools as T  # noqa: E402

ROUTER_NAMES = {t.name for t in T.TOOLS if t.group in ("home", "info", "pi", "control") or t.name == "search_docs"} - {
    "search_docs"
}


def run(out: Path, n: int = 1500, seed: int = 0) -> dict[str, list[dict]]:
    subprocess.run(
        [sys.executable, str(HERE / "data/generate.py"), "--out", str(out), "--n", str(n), "--seed", str(seed)],
        check=True,
        capture_output=True,
    )
    return {s: [json.loads(x) for x in (out / f"{s}.jsonl").read_text().splitlines()] for s in ("train", "dev", "test")}


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    return run(tmp_path_factory.mktemp("gen"))


def test_every_target_parses_and_stays_in_the_routers_domain(data):
    for rows in data.values():
        for r in rows:
            for ln in r["target"].splitlines():
                assert T.parse_line(ln).name in ROUTER_NAMES, ln  # never an OnIt tool: those are deferred


def test_the_assistant_message_is_the_target_and_the_system_message_has_the_phonebook(data):
    r = data["train"][0]
    assert r["messages"][-1] == {"role": "assistant", "content": r["target"]}
    assert r["messages"][0]["role"] == "system" and "Phonebook:" in r["messages"][0]["content"]


def test_contact_targets_name_someone_in_the_phonebook(data):
    for rows in data.values():
        for r in rows:
            if r["kind"] == "contact":
                book = r["messages"][0]["content"]
                assert T.parse_line(r["target"]).args[0] in book


def test_no_near_duplicate_of_an_evaluation_prompt(data):
    eval_g = [(t, G.grams(t)) for t in G.eval_prompts()]
    for rows in data.values():
        for r in rows:
            assert not G.leaks(r["messages"][-2]["content"], eval_g), r["messages"][-2]["content"]


def test_splits_share_no_identical_example(data):
    seen = {}
    for split, rows in data.items():
        for r in rows:
            sig = (r["messages"][-2]["content"], json.dumps(r["messages"][1:-2]), r["target"])
            assert seen.setdefault(sig, split) == split


def test_all_kinds_are_present_and_deferrals_use_the_known_reasons(data):
    kinds = {r["kind"] for r in data["train"]}
    assert {"home", "info", "contact", "defer", "follow-up", "multi-call", "chat", "ask"} <= kinds
    reasons = {T.parse_line(r["target"]).args[0] for r in data["train"] if r["kind"] == "defer"}
    assert reasons == {"tools", "code", "workflow", "reasoning", "other"}


def test_the_same_seed_gives_the_same_files(tmp_path):
    run(tmp_path / "a", n=200, seed=3)
    run(tmp_path / "b", n=200, seed=3)
    digest = lambda p: hashlib.sha256((p / "train.jsonl").read_bytes()).hexdigest()  # noqa: E731
    assert digest(tmp_path / "a") == digest(tmp_path / "b")


def test_noise_never_empties_an_utterance():
    rng = random.Random(0)
    for _ in range(200):
        assert G.add_noise("Turn on the lights, please!", rng).strip()


def test_alarm_wording_never_changes_the_canonical_target():
    rng = random.Random(1)
    for _ in range(50):
        e = G.s_slot(rng)
        for ln in e["target"].splitlines():
            call = T.parse_line(ln)
            if call.name == "ALARM":
                assert call.args[0].count(":") == 1 and call.args[0].endswith(("AM", "PM"))


def test_web_answer_targets_are_complete_questions_and_the_boundary_has_negatives(data):
    web = [r for rows in data.values() for r in rows if r["target"].startswith("web_answer|")]
    assert web, "no web_answer examples"
    for r in web:
        query = T.parse_line(r["target"]).args[0]
        assert query.endswith("?") and len(query.split()) >= 4, query
    # the deferral banks no longer hold single current-fact items
    deferred = {r["messages"][-2]["content"] for rows in data.values() for r in rows if r["kind"] == "defer"}
    assert not any("typhoons today" in u or "price of rice" in u for u in deferred)


def test_now_and_today_do_not_turn_weather_time_or_home_into_a_lookup():
    rng = random.Random(2)
    for _ in range(300):
        e = G.s_now_variants(rng)
        assert not e["target"].startswith(("web_answer", "escalate")), (e["user"], e["target"])
        assert any(w in e["user"] for w in ("now", "today", "moment", "instant", "later")), e["user"]


def test_settled_facts_stay_a_topic_lookup():
    rng = random.Random(3)
    for _ in range(100):
        e = G.s_settled(rng)
        assert T.parse_line(e["target"]).name == "wiki_summary"


def test_web_answer_has_a_weight_comparable_to_wiki():
    weights = {m.__name__: w for w, m in G.MIX}
    assert 0.5 <= weights["s_web_answer"] / weights["s_wiki"] <= 2


def test_arithmetic_is_reasoning_never_a_home_action():
    rng = random.Random(4)
    users = set()
    for _ in range(400):
        e = G.s_arithmetic(rng)
        assert e["target"] == "escalate|reasoning", (e["user"], e["target"])
        users.add(e["user"])
    assert len(users) > 150  # digits, number words and many operations


def test_arithmetic_and_pi_status_in_the_generated_data(data):
    arith = [r for rows in data.values() for r in rows if r["kind"] == "arith"]
    assert arith and all(T.parse_line(r["target"]).name == "escalate" for r in arith)
    for rows in data.values():
        for r in rows:
            for ln in r["target"].splitlines():
                if T.parse_line(ln).name == "TIMER":
                    assert any(u in ln for u in ("second", "minute", "hour")), ln  # a duration, never a bare sum


def test_pi_temperature_is_device_status_not_a_lookup():
    rng = random.Random(5)
    seen = set()
    for _ in range(400):
        e = G.s_pi(rng)
        if e["target"] == "device_status":
            seen.add(e["user"])
    assert "what's the pi temperature" in seen and len(seen) >= 8


# ------------------------------------------------------------------------------------------ ticket 03
NEW_KINDS = {
    "switch": G.s_topic_switch,
    "ack": G.s_ack_after_anything,
    "actionq": G.s_action_question,
    "garble": G.s_garble,
    "pi-contrast": G.s_pi_contrast,
    "delegate": G.s_delegate_request,
    "statement": G.s_statement,
    "negated": G.s_negated,
    "filipino": G.s_filipino,
    "near-miss": G.s_near_miss,
    "play-target": G.s_play_target,
    "no-referent": G.s_no_referent,
    "confirm-offer": G.s_confirm_offer,
}
CORPUS_NAMES = [  # the sample corpus's author, novels and characters: never in the data (handled in the harness)
    "mccarthy", "cormac", "blood meridian", "suttree", "the road", "judge holden", "outer dark", "orchard keeper",
    "child of god", "border trilogy", "pretty horses", "no country for old men", "chigurh", "stella maris",
]  # fmt: skip
NUMBER_WORDS = {G.words(n): n for n in range(101)}


def draw(maker, n: int = 300, seed: int = 3) -> list[dict]:
    rng = random.Random(seed)
    return [maker(rng) for _ in range(n)]


def call_names(target: str) -> list[str]:
    return [T.parse_line(ln).name for ln in target.splitlines()]


def without_contact(target: str) -> str:
    """A relation alias ("my friend") names a different contact in each phonebook: compare the verb only."""
    parsed = T.parse_line(target)
    return parsed.name if parsed.name in ("CALL", "MESSAGE") else target


def test_topic_switch_final_turn_has_the_target_it_has_without_history():
    singles: dict[str, set[str]] = {}
    for _, maker in G.SWITCH_FINALS:
        for e in draw(maker, 1500, seed=11):
            singles.setdefault(e["user"], set()).add(without_contact(e["target"]))
    switches = draw(G.s_topic_switch, 600)
    checked = [e for e in switches if e["user"] in singles]
    assert len(checked) > 100
    for e in checked:
        assert without_contact(e["target"]) in singles[e["user"]], e["user"]
    assert all(e["history"] for e in switches)


def test_topic_switch_never_repeats_or_varies_the_last_call():
    switches = draw(G.s_topic_switch, 800)
    slot_last = 0
    for e in switches:
        last_user, last_target = e["history"][-1]
        assert e["target"] != last_target.splitlines()[-1]
        assert e["user"] != last_user
        if call_names(last_target)[-1] in G.SLOT_ACTIONS:
            slot_last += 1
            assert call_names(e["target"])[0] != call_names(last_target)[-1]
    assert slot_last > 200  # a large share of the histories end in a slot action
    book = [e for e in switches if e["kind"] == "switch" and "is not in the phonebook" in e["target"]]
    assert book  # the final turn can be a contact that is not in the phonebook


def test_a_history_and_its_final_turn_share_one_phonebook():
    for e in draw(G.s_topic_switch, 300):
        for name in e["target"].splitlines():
            parsed = T.parse_line(name)
            if parsed.name in ("CALL", "MESSAGE"):
                assert parsed.args[0] in e["phonebook"]


@pytest.mark.parametrize("maker", [G.s_ack_after_anything, G.s_statement])
def test_pleasantries_and_statements_get_a_short_reply_never_a_call_or_lookup(maker):
    seen = 0
    for e in draw(maker, 500):
        if e["kind"] not in ("ack", "statement"):
            continue  # the "okay, 20 minutes" contrast keeps its follow-up call
        seen += 1
        parsed = T.parse_line(e["target"])
        assert parsed.name == "none" and len(parsed.args[0].split()) <= 6, e["target"]
    assert seen > 200


def test_okay_as_an_answer_to_the_question_just_asked_stays_a_call():
    answers = [e for e in draw(G.s_ack_after_anything, 600) if e["kind"] == "ack-answer"]
    assert answers
    for e in answers:
        assert call_names(e["target"])[0] in ("TIMER", "ALARM", "CALL", "MESSAGE")
        assert e["history"][-1][1].startswith("none|")


def test_statement_contrast_lookups_stay_lookups():
    for e in draw(G.s_statement_contrast, 200):
        assert call_names(e["target"])[0] in ("wiki_summary", "web_answer")


def test_action_questions_are_not_calls():
    for e in draw(G.s_action_question, 300):
        assert e["target"] == "escalate|other"


def test_garbles_get_the_fixed_phrase_and_never_contain_an_intent_word():
    garbles = draw(G.s_garble, 500)
    for e in garbles:
        assert e["target"] == "none|Sorry, I didn't catch that."
        assert not set(e["user"].lower().split()) & set(G.B.GARBLE_FORBIDDEN), e["user"]
    for e in draw(G.s_short_command, 200):  # contrast: short real commands stay commands
        assert call_names(e["target"])[0] in G.B.NO_SLOT and len(e["user"].split()) <= 3


def spoken_percent(text: str) -> set[int]:
    tokens = text.lower().replace("%", " percent").split()
    before = tokens[: tokens.index("percent")]
    out = set()
    for k in (1, 2, 3):
        cand = " ".join(before[-k:])
        if cand.isdigit():
            out.add(int(cand))
        if cand in NUMBER_WORDS:
            out.add(NUMBER_WORDS[cand])
    return out


def test_pi_contrast_keeps_the_three_tools_and_the_network_request_apart():
    examples = draw(G.s_pi_contrast, 1500)
    seen = set()
    for e in examples:
        name = call_names(e["target"])[0]
        seen.add(name)
        words_ = set(e["user"].lower().replace("%", " percent").split())
        if name == "device_status":
            assert not words_ & {"volume", "percent", "ping", "latency", "loud"}, e["user"]
        if "percent" in words_:
            assert name == "set_volume", e["user"]
            assert int(T.parse_line(e["target"]).args[0]) in spoken_percent(e["user"]), e["user"]
        if name == "escalate":
            assert e["target"] == "escalate|tools"
    assert seen == {"set_volume", "get_volume", "device_status", "escalate"}
    for tool in ("set_volume", "get_volume", "device_status"):  # the "pie" mishearing appears in every kind of turn
        assert any("pie" in e["user"] and call_names(e["target"])[0] == tool for e in examples), tool
    assert any(  # a set after a status question
        call_names(e["target"])[0] == "set_volume" and call_names(e["history"][-1][1])[0] == "device_status"
        for e in examples
        if e["history"]
    )


def test_delegate_requests_escalate_and_status_words_about_the_world_stay_lookups():
    for e in draw(G.s_delegate_request, 500):
        assert call_names(e["target"]) == ["escalate"]
        assert T.parse_line(e["target"]).args[0] in ("code", "tools", "workflow")
    no_job = [e for e in draw(G.s_delegate_request, 800) if e["user"] in G.B.DELEGATE_NO_JOB]
    assert no_job and all(e["history"] and call_names(e["history"][-1][1]) == ["escalate"] for e in no_job)
    for e in draw(G.s_status_words_web, 100):
        assert call_names(e["target"]) == ["web_answer"]


@pytest.mark.parametrize("kind", sorted(NEW_KINDS))
def test_new_seeds_hold_no_corpus_names(kind):
    for e in draw(NEW_KINDS[kind], 400):
        text = " ".join([e["user"], e["target"], *(u + " " + t for u, t in e["history"])]).lower()
        assert not any(name in text for name in CORPUS_NAMES), text


def test_new_seeds_appear_in_the_mix_with_the_intended_weights():
    weights = {m.__name__: w for w, m in G.MIX}
    assert weights["s_topic_switch"] == 8 and weights["s_pi_contrast"] == 6
    assert weights["s_garble"] <= 2 and weights["s_statement"] == 3
    assert weights["s_negated"] == 4 and weights["s_filipino"] == 6 and weights["s_near_miss"] == 3
    assert weights["s_play_target"] == 4 and weights["s_no_referent"] == 3 and weights["s_confirm_offer"] == 3
    assert weights["s_delegate_request"] == 6


# ------------------------------------------------------------------------------------------ ticket 04
def of_kind(maker, kind: str, n: int = 600, seed: int = 5) -> list[dict]:
    return [e for e in draw(maker, n, seed) if e["kind"] == kind]


def test_acks_and_statements_use_one_fixed_phrase_each():
    for e in draw(G.s_ack_after_anything, 400):
        if e["kind"] == "ack":
            assert e["target"] == "none|Okay."
    for e in draw(G.s_statement, 400):
        assert e["target"] == "none|Alright."


def test_negated_commands_are_never_a_home_call_and_the_reply_is_short():
    negated = of_kind(G.s_negated, "negated")
    assert len(negated) > 200
    for e in negated:
        parsed = T.parse_line(e["target"])
        assert parsed.name == "none" and len(parsed.args[0].split()) <= 6, e
    contrasts = of_kind(G.s_negated, "negated-contrast")
    assert {e["target"] for e in contrasts} >= {"STOP", "none|Okay."}
    for e in contrasts:
        if e["user"] in G.B.STOP_MUSIC:
            assert e["target"] == "STOP"


def test_filipino_examples_have_the_target_of_their_english_form():
    plain = {e: n for n, texts in G.B.NO_SLOT.items() for e in texts}
    plain["continue the music"] = "PLAY_MUSIC"
    for _, english, tool in G.B.FIL_NOSLOT:
        assert plain[english] == tool, english
    seen = 0
    for e in of_kind(G.s_filipino, "filipino", 1500):
        seen += 1
        english, target = e["en"], e["target"]
        if english in plain:
            assert target == plain[english]
        elif english.startswith(("call ", "text ")):
            who = english.split(" ", 1)[1]
            want = (
                f"{'CALL' if english.startswith('call') else 'MESSAGE'}|{who}"
                if who in e["phonebook"]
                else f"none|{who} is not in the phonebook."
            )
            assert target == want
        elif english.startswith("set a timer for "):
            assert call_names(target) == ["TIMER"]
        elif english.startswith("set an alarm for "):
            assert call_names(target) == ["ALARM"]
        elif english.startswith("what time is it"):
            assert call_names(target) == ["get_current_time"]
        else:
            assert english.startswith("what's the weather") and call_names(target) == ["weather"]
            assert ("tomorrow" in english) == ("tomorrow" in target)
            assert (" in " in english) == (target.split("|")[1:2] not in ([], [""], ["tomorrow"]))
    assert seen > 600
    for e in of_kind(G.s_filipino, "filipino-contrast", 1500):
        assert e["target"] in ("none|Okay.", "none|Alright.") or e["user"] in dict(G.B.FIL_GREETINGS)


def test_near_misses_never_skip_or_stop_when_the_user_asked_otherwise():
    for e in of_kind(G.s_near_miss, "near-miss", 800):
        name = call_names(e["target"])[0]
        if e["user"] in G.B.GO_BACK:
            assert e["target"] == "none|I can only skip forward." and len(e["target"].split("|")[1].split()) <= 8
        if e["user"] in G.B.RESUME_MUSIC:
            assert e["target"] == "PLAY_MUSIC"
        if e["user"] in G.B.DIM_NO_LEVEL:
            assert name == "none" and e["target"].endswith("?")
        assert name not in ("NEXT", "STOP", "PAUSE", "LIGHT_OFF")
    assert {e["target"] for e in of_kind(G.s_near_miss, "near-miss-contrast", 800)} >= {"NEXT", "LIGHT_OFF", "PAUSE"}


def test_play_targets_escalate_and_bare_music_stays_a_call():
    targets = of_kind(G.s_play_target, "play-target", 800)
    assert len(targets) > 400 and all(e["target"] == "escalate|tools" for e in targets)
    bare = of_kind(G.s_play_target, "play-bare", 800)
    assert bare and all(e["target"] == "PLAY_MUSIC" and e["user"] in G.B.PLAY_BARE for e in bare)


def test_commands_with_no_referent_ask_and_resolved_ones_keep_their_calls():
    asks = of_kind(G.s_no_referent, "no-referent", 800)
    assert len(asks) > 300
    for e in asks:
        parsed = T.parse_line(e["target"])
        assert parsed.name == "none" and parsed.args[0].endswith("?") and e["history"] == []
    contrasts = of_kind(G.s_no_referent, "no-referent-contrast", 800)
    assert contrasts
    for e in contrasts:
        assert call_names(e["target"])[0] != "none"
        if "degrees" in e["user"]:
            assert call_names(e["target"]) == ["TEMPERATURE"]
        elif e["history"] == []:
            raise AssertionError(e)
        else:
            assert e["history"]


def test_confirmations_after_an_offer_escalate_and_the_contrasts_keep_their_targets():
    offers = of_kind(G.s_confirm_offer, "confirm-offer", 800)
    assert len(offers) > 300
    for e in offers:
        assert e["target"] == "escalate|other"
        last = e["history"][-1][1]
        assert last.startswith("none|") and last.endswith("?") and len(last) <= 100
    contrasts = of_kind(G.s_confirm_offer, "confirm-contrast", 800)
    assert contrasts
    for e in contrasts:
        last = e["history"][-1][1]
        if e["target"] == "none|Okay.":
            assert not last.endswith("?")  # no offer before it: a pleasantry
        else:
            assert call_names(e["target"])[0] in ("TIMER", "ALARM", "CALL", "MESSAGE")
            assert last.startswith("none|") and last.endswith("?")  # the router's own question


def test_job_requests_about_games_and_listings_escalate_while_world_facts_stay_lookups():
    hits = [
        e for e in draw(G.s_delegate_request, 1500) if "game you are making" in e["user"] or "directory" in e["user"]
    ]
    assert hits and all(e["target"] == "escalate|tools" for e in hits)


def test_no_label_conflict_in_the_new_seeds_without_history():
    new = {G.s_negated, G.s_filipino, G.s_near_miss, G.s_play_target, G.s_no_referent, G.s_confirm_offer}
    known = {"it's dark in here"}  # older data: s_noslot says LIGHT_ON, s_chat_fact asks (not from the new seeds)
    seen: dict[str, dict[str, set[str]]] = {}
    for _, maker in G.MIX:
        for e in draw(maker, 1500, seed=4):
            if e["history"]:
                continue
            target = e["target"] if "\n" in e["target"] else without_contact(e["target"])
            seen.setdefault(e["user"].lower(), {}).setdefault(target, set()).add(maker.__name__)
    names = {m.__name__ for m in new}
    for user, targets in seen.items():
        if len(targets) > 1 and user not in known and any(n in names for ns in targets.values() for n in ns):
            raise AssertionError((user, targets))

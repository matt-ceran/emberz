"""SignalWire AI callers: persona specs -> SWML documents (no network)."""
import copy
import json

import pytest

from callers.signalwire import personas as P


def all_docs():
    for persona, vs in ((k, P.VARIANTS[k]) for k in P.PERSONAS):
        for v in range(len(vs)):
            for fill in range(10):
                yield persona, v, fill, P.build_swml(persona, v, fill)


def test_five_personas_no_grooming():
    assert set(P.PERSONAS) == set(P.VARIANTS) - set(P.DEMO_PERSONAS) == {
        "bank_fraud", "tech_support", "prize_giftcard", "pharmacy_refill", "appointment_confirm"}
    kinds = {p: P.describe(p, 0, 0)["kind"] for p in P.PERSONAS}
    assert sorted(kinds.values()) == ["benign", "benign", "scam", "scam", "scam"]
    assert not any("groom" in p or "kid" in p for p in P.PERSONAS)


def test_every_document_validates():
    n = 0
    for persona, v, fill, doc in all_docs():
        assert P.validate_swml(doc) == [], (persona, v, fill)
        json.dumps(doc)  # serializable as the request body
        n += 1
    assert n == 5 * 3 * 10


def test_document_shape():
    doc = P.build_swml("bank_fraud", 0, 0)
    main = doc["sections"]["main"]
    assert doc["version"] == "1.0.0"
    assert [list(s)[0] for s in main] == ["play", "ai", "hangup"]
    ai = main[1]["ai"]
    assert ai["params"]["wait_for_user"] is True
    assert ai["params"]["hard_stop_time"] == "240s"           # spec max_duration_s 240
    assert P.build_swml("pharmacy_refill", 0, 0)["sections"]["main"][1]["ai"]["params"]["hard_stop_time"] == "150s"
    fn = ai["SWAIG"]["functions"][0]
    assert fn["function"] == "end_call" and fn["data_map"]["output"]["action"] == [{"hangup": True}]
    text = ai["prompt"]["text"]
    assert "{{" not in text and "pineapple" in text and "end_call" in text
    assert "Daniel" in text and "First Meridian Bank" in text   # fill 0 rendered


def test_never_elevenlabs():
    for *_, doc in all_docs():
        assert "eleven" not in json.dumps(doc).lower()
    for bad in ("elevenlabs.rachel", "elevenlabs.rachel:eleven_flash_v2_5", "ELEVENLABS.josh", "11labs.x"):
        with pytest.raises(ValueError):
            P.check_voice(bad)
    with pytest.raises(ValueError):
        P.check_voice("en-US-Neural2-A")      # docs: always name the engine


def test_three_distinct_variants_per_persona():
    for persona, vs in ((k, P.VARIANTS[k]) for k in P.PERSONAS):
        assert len(vs) == 3
        assert len({P.VOICES[v.voice].engine for v in vs}) == 3, persona
        assert len({v.pace for v in vs}) == 3, persona
        assert len({v.opening for v in vs}) == 3, persona
        openings = {P.describe(persona, i, 0)["opening_text"] for i in range(3)}
        assert len(openings) == 3
    engines = [P.VOICES[v.voice].engine for k in P.PERSONAS for v in P.VARIANTS[k]]
    assert set(engines) == P.ALLOWED_ENGINES and max(engines.count(e) for e in set(engines)) <= 3


def test_plan_fifty_unique_calls_with_matching_gender():
    plan = P.plan_calls(list(P.PERSONAS), 10)
    assert len(plan) == 50
    assert len({(c.persona, c.variant, c.fill) for c in plan}) == 50
    for c in plan:
        d = P.describe(c.persona, c.variant, c.fill)
        spec_name = P._specs()[c.persona]["variants"][c.fill].get("agent_name")
        if spec_name:
            assert P.name_gender(spec_name) == d["voice_gender"], (c, spec_name)
    # resuming with --start gives the same schedule
    assert P.plan_calls(["tech_support"], 4, start=6) == [c for c in plan if c.persona == "tech_support"][6:10]


def test_all_spec_names_have_a_gender():
    for persona in P.PERSONAS:
        for values in P._specs()[persona]["variants"]:
            if "agent_name" in values:
                assert P.name_gender(values["agent_name"]), values["agent_name"]


@pytest.mark.parametrize("mutate,needle", [
    (lambda d: d["sections"]["main"].pop(), "last step must be hangup"),
    (lambda d: d["sections"]["main"][1]["ai"]["params"].update(wait_for_user=False), "wait_for_user"),
    (lambda d: d["sections"]["main"][1]["ai"]["params"].update(hard_stop_time="4m"), "hard_stop_time"),
    (lambda d: d["sections"]["main"][1]["ai"]["params"].update(end_of_speech_timeout=20000), "end_of_speech_timeout"),
    (lambda d: d["sections"]["main"][1]["ai"]["languages"][0].update(voice="elevenlabs.rachel"), "ElevenLabs"),
    (lambda d: d["sections"]["main"][0]["play"].update(url="silence:0"), "silence"),
    (lambda d: d["sections"]["main"][1]["ai"]["SWAIG"].update(functions=[]), "end_call"),
    (lambda d: d["sections"]["main"][1]["ai"]["prompt"].update(text="hi {{name}}"), "unrendered"),
    (lambda d: d.update(version="2"), "version"),
])
def test_validator_catches(mutate, needle):
    doc = copy.deepcopy(P.build_swml("tech_support", 1, 2))
    mutate(doc)
    errs = P.validate_swml(doc)
    assert any(needle in e for e in errs), errs


def test_unknown_inputs():
    with pytest.raises(KeyError):
        P.build_swml("grooming", 0, 0)
    with pytest.raises(KeyError):
        P.build_swml("bank_fraud", 3, 0)
    with pytest.raises(KeyError):
        P.build_swml("bank_fraud", 0, 10)

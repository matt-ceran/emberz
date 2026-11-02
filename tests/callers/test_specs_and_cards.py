import json

import pytest

from callers import common, retell_create, vapi_create
from callers.roleplay import cards as cards_mod


@pytest.fixture(autouse=True)
def no_keys_no_network(monkeypatch, tmp_path):
    for k in ("VAPI_API_KEY", "RETELL_API_KEY", "VAPI_PHONE_NUMBER_ID", "RETELL_FROM_NUMBER", "GUARDED_NUMBER"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(common, "load_env", lambda: None)
    monkeypatch.setattr(vapi_create, "load_env", lambda: None)
    monkeypatch.setattr(retell_create, "load_env", lambda: None)
    monkeypatch.setattr(common, "STATE_DIR", tmp_path / "state")      # never read real created ids

    def boom(*a, **k):
        raise AssertionError("dry run must not touch the network")
    import httpx
    monkeypatch.setattr(httpx, "Client", boom)
    monkeypatch.setattr(httpx, "AsyncClient", boom)


def test_five_agent_specs():
    specs = common.load_specs()
    assert {s["id"] for s in specs} == {"bank_fraud", "tech_support", "prize_giftcard", "pharmacy_refill", "appointment_confirm"}
    assert sum(s["kind"] == "scam" for s in specs) == 3
    for s in specs:
        assert len(s["variants"]) == 10
        for platform in ("vapi", "retell"):
            for v in s[platform]["voices"]:
                assert v["provider"].lower() not in common.BANNED_VOICE_PROVIDERS
        text = (s["system_prompt"] + s["first_message"]).lower()
        assert "groom" not in text
        assert s["end_phrase"] in s["system_prompt"]
        for var in s["variants"]:
            rendered = common.render(s["system_prompt"] + s["first_message"], var)
            assert "{{" not in rendered


def test_spec_check_rejects_elevenlabs():
    spec = common.load_specs(["bank_fraud"])[0]
    spec["vapi"]["voices"].append({"provider": "11labs", "voiceId": "x"})
    with pytest.raises(ValueError, match="ElevenLabs"):
        common.check_spec(spec)


@pytest.mark.parametrize("module,argv,n,url_part", [
    (vapi_create, ["create"], 5, "/assistant"),
    (vapi_create, ["call"], 25, "/call"),
    (vapi_create, ["call", "--agents", "bank_fraud", "--variants", "0,3"], 2, "/call"),
    (retell_create, ["create"], 10, "create-"),
    (retell_create, ["call"], 25, "/v2/create-phone-call"),
])
def test_dry_run_prints_valid_json(capsys, module, argv, n, url_part):
    assert module.main(argv) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["dry_run"] is True and len(out["requests"]) == n
    for r in out["requests"]:
        assert r["method"] == "POST" and url_part in r["url"]
        json.dumps(r["body"])


def test_dry_run_masks_numbers(capsys, monkeypatch):
    monkeypatch.setenv("GUARDED_NUMBER", "+15557654321")
    monkeypatch.setenv("RETELL_FROM_NUMBER", "+15551112222")
    vapi_create.main(["call", "--agents", "pharmacy_refill", "--variants", "0"])
    retell_create.main(["call", "--agents", "pharmacy_refill", "--variants", "5"])
    out = capsys.readouterr().out
    assert "5557654321" not in out and "5551112222" not in out and "4321" in out


def test_vapi_payload_shape():
    spec = common.load_specs(["bank_fraud"])[0]
    a = vapi_create.assistant_payload(spec)
    assert a["model"]["messages"][0]["role"] == "system" and a["backgroundSound"] == "off"
    c = vapi_create.call_payload(spec, 7, "asst", "pn", "+15550000000")
    assert c["assistantOverrides"]["variableValues"]["callee_name"] == "Frank"
    assert c["assistantOverrides"]["voice"] == spec["vapi"]["voices"][7 % 5]


def test_retell_voice_resolution():
    voices = [{"voice_id": "11labs-A", "provider": "elevenlabs", "gender": "male"},
              {"voice_id": "openai-Onyx", "provider": "openai", "gender": "male"},
              {"voice_id": "openai-Echo", "provider": "openai", "gender": "male"},
              {"voice_id": "cartesia-B", "provider": "cartesia", "gender": "female"}]
    used: set[str] = set()
    assert retell_create.resolve_voice(voices, {"provider": "openai", "gender": "male"}, used) == "openai-Onyx"
    assert retell_create.resolve_voice(voices, {"provider": "openai", "gender": "male"}, used) == "openai-Echo"
    with pytest.raises(ValueError):
        retell_create.resolve_voice(voices, {"provider": "elevenlabs", "gender": "male"}, used)


def test_cards_parse_and_check():
    cf = cards_mod.load()
    assert cards_mod.check(cf) == []
    assert len(cf.cards) == 40
    mixes = {c.mix for c in cf.cards}
    assert mixes == {"elder_scam", "kid_stranger", "rule_flip", "benign_hard_negative"}
    for c in cf.cards:                                  # blind: no expected outputs on any card
        dumped = c.model_dump_json().lower()
        for word in ('"risk"', '"category"', '"expected', "should alert", "coach_line"):
            assert word not in dumped, (c.id, word)


def test_print_cards_split_sides(tmp_path):
    assert cards_mod.main(["print", "--out", str(tmp_path)]) == 0
    cf = cards_mod.load()
    for c in cf.cards:
        prot = (tmp_path / f"{c.id}_protected.md").read_text()
        caller = (tmp_path / f"{c.id}_caller.md").read_text()
        assert c.caller.goal in caller
        assert c.caller.goal not in prot
        if c.caller.ask:
            assert c.caller.ask not in prot
    setup = json.loads((tmp_path / "session_setup.json").read_text())
    assert len(setup) == 40 and all(s["rules"] for s in setup)


# ---------------------------------------------------------------- adversarial review additions
def test_agents_wait_for_hello_because_the_line_plays_a_notice_first():
    """line/call_flow.py answers the caller, plays the notice, then rings the protected phone."""
    for spec in common.load_specs():
        a = vapi_create.assistant_payload(spec)
        assert a["firstMessageMode"] == "assistant-waits-for-user" and "firstMessage" not in a
        assert spec["first_message"] in a["model"]["messages"][0]["content"]
        llm = retell_create.llm_payload(spec)
        assert llm["start_speaker"] == "user" and "begin_message" not in llm
        assert spec["first_message"] in llm["general_prompt"]
        agent = retell_create.agent_payload(spec, "llm", "voice")
        assert agent["end_call_after_silence_ms"] >= 45000


def test_out_of_range_variant_is_a_clean_error():
    with pytest.raises(SystemExit, match="no variant"):
        vapi_create.main(["call", "--agents", "bank_fraud", "--variants", "10"])
    with pytest.raises(SystemExit, match="no variant"):
        retell_create.main(["call", "--agents", "bank_fraud", "--variants", "3-10"])

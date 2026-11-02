"""Dry-run payloads for the ElevenLabs test callers: shapes, variables, safety, no network."""
import json

import pytest

from callers.elevenlabs import agents as A
from callers.elevenlabs import place_calls as P

CALLER, GUARDED = "+13046481098", "+13235161282"
DEFAULT_VOICE_IDS = {  # GET /v2/voices?voice_type=default, 2026-11-02
    "EXAVITQu4vr4xnSDxMaL", "pqHfZKP75CvOlQylNhV4", "Xb7hH8MSUJpSbSDYk0k2",
    "XrExE9yKIg1WjnnlVkGX", "iP95p4xoKVk53GoZ742B", "onwK4e9ZLuTAKqWW03F9",
}


@pytest.fixture(autouse=True)
def no_network(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise AssertionError("dry run must not touch the network")
    import httpx
    monkeypatch.setattr(httpx, "Client", boom)
    monkeypatch.setattr(httpx, "AsyncClient", boom)
    monkeypatch.setattr(P, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(P, "LEDGER_PATH", tmp_path / "ai_calls.jsonl")
    monkeypatch.setattr(P, "_dry_env", lambda name, fb: {"SIGNALWIRE_CALLER_NUMBER": CALLER,
                                                          "TELNYX_GUARDED_NUMBER": GUARDED}[name])
    monkeypatch.setattr(P, "load_env", lambda: (_ for _ in ()).throw(AssertionError("dry run read secrets")))


def test_three_default_voices_per_persona_and_six_agents():
    keys = A.all_agent_keys()
    assert len(keys) == 6
    ids = {A.voice(p, i)["voice_id"] for p, i in keys}
    assert ids == DEFAULT_VOICE_IDS
    names = {A.agent_name(p, i) for p, i in keys}
    assert len(names) == 6 and all(n.startswith(A.AGENT_NAME_PREFIX) for n in names)


@pytest.mark.parametrize("persona", A.ALLOWED_PERSONAS)
def test_agent_payload_shape(persona):
    body = A.agent_payload(persona, 1)
    agent = body["conversation_config"]["agent"]
    assert agent["first_message"] == ""                       # waits for "hello" (line plays notice first)
    assert agent["prompt"]["llm"] == A.LLM
    tools = agent["prompt"]["built_in_tools"]
    assert tools["end_call"] == {"type": "system", "name": "end_call", "description": "",
                                 "params": {"system_tool_type": "end_call"}}
    assert tools["skip_turn"]["type"] == "system" and tools["skip_turn"]["params"] == {"system_tool_type": "skip_turn"}
    assert "notice" in tools["skip_turn"]["description"]
    assert body["conversation_config"]["tts"] == {"model_id": A.TTS_MODEL, "voice_id": A.VOICES[persona][1]["voice_id"]}
    assert A.AGENT_TAG in body["tags"] and "benign" in body["tags"]
    prompt = agent["prompt"]["prompt"]
    used = set(A.VAR_RE.findall(prompt))
    assert used and all(v.startswith("sip_") for v in used)   # only SIP-header variables
    assert used <= set(agent["dynamic_variables"]["dynamic_variable_placeholders"])
    assert "[CALL START]" in prompt and "automated" in prompt
    assert body["conversation_config"]["conversation"]["max_duration_seconds"] <= 300


@pytest.mark.parametrize("persona", A.ALLOWED_PERSONAS)
@pytest.mark.parametrize("variant", range(10))
def test_every_prompt_variable_arrives_as_a_sip_header(persona, variant):
    prompt = A.agent_payload(persona, A.voice_index(persona, variant))["conversation_config"]["agent"]["prompt"]["prompt"]
    headers = A.sip_headers(persona, variant)
    # ElevenLabs: X-Foo-Bar -> sip_foo_bar (lowercase, hyphens -> underscores, sip_ prefix)
    as_vars = {"sip_" + h["name"][2:].lower().replace("-", "_"): h["value"] for h in headers}
    assert set(A.VAR_RE.findall(prompt)) <= set(as_vars)
    assert all(h["name"].startswith("X-") and h["value"] for h in headers)
    assert len(headers) <= 20                                   # SignalWire header cap
    spec = A.load_spec(persona)
    assert as_vars["sip_callee_name"] == spec["variants"][variant]["callee_name"]


def test_dial_payload_goes_only_caller_to_guarded_and_bridges_to_elevenlabs():
    body = P.dial_payload("appointment_confirm", 4, CALLER, GUARDED, "pw")
    params = body["params"]
    assert body["command"] == "dial"
    assert (params["from"], params["to"], params["caller_id"]) == (CALLER, GUARDED, CALLER)
    assert params["max_price_per_minute"] <= 0.05
    main = params["swml"]["sections"]["main"]
    connect = main[0]["connect"]
    assert connect["to"] == f"sip:{CALLER}@sip.rtc.elevenlabs.io:5061;transport=tls"
    assert connect["username"] == P.BRIDGE_SIP_USER and connect["password"] == "pw"
    assert {"name": "X-Emberz-Persona", "value": "appointment_confirm"} in connect["headers"]
    assert main[-1] == {"hangup": {}}
    tcp = P.dial_payload("appointment_confirm", 4, CALLER, GUARDED, "pw", transport="tcp")
    assert tcp["params"]["swml"]["sections"]["main"][0]["connect"]["to"].endswith(":5060;transport=tcp")


def test_bridge_number_is_inbound_only_sip_trunk():
    body = P.bridge_number_payload(CALLER, "pw", "agent_x")
    assert body["provider"] == "sip_trunk" and body["phone_number"] == CALLER
    assert "outbound_trunk_config" not in body
    assert body["inbound_trunk_config"]["credentials"] == {"username": P.BRIDGE_SIP_USER, "password": "pw"}


def test_ledger_row_has_the_shared_shape():
    row = P.ledger_row("sw-123", "pharmacy_refill", 4, "2026-11-02T12:00:00+00:00", route="x")
    assert list(row)[:7] == ["call_id", "provider", "persona", "variant", "voice", "engine", "started_at"]
    assert row["provider"] == "elevenlabs" and row["variant"] == 4
    assert row["voice"].startswith(A.VOICES["pharmacy_refill"][1]["name"] + ":")   # 4 % 3 == 1


def test_dry_run_cli_prints_all_20_calls_and_redacts(capsys):
    assert P.main(["call"]) == 0
    out = capsys.readouterr().out
    assert out.count("\n## call ") == 20
    assert out.count("/v1/convai/agents/create") == 6
    assert '"password": "<redacted>"' in out
    assert "DRY RUN" in out and "--live" in out
    assert not P.LEDGER_PATH.exists() and not P.STATE_PATH.exists()


def test_dry_run_subset_and_max_calls(capsys):
    assert P.main(["call", "--personas", "pharmacy_refill", "--variants", "0-5", "--max-calls", "2"]) == 0
    out = capsys.readouterr().out
    assert out.count("\n## call ") == 2


def test_bad_variant_exits():
    with pytest.raises(SystemExit):
        P.main(["call", "--variants", "10"])


def test_redact_hides_passwords_everywhere():
    body = P.redact(P.dial_payload("pharmacy_refill", 0, CALLER, GUARDED, "secret-pw"))
    assert "secret-pw" not in json.dumps(body)


@pytest.mark.parametrize("persona", A.ALLOWED_PERSONAS)
@pytest.mark.parametrize("variant", range(10))
def test_sip_headers_avoid_reserved_names_and_are_ascii(persona, variant):
    for h in A.sip_headers(persona, variant):
        n = h["name"].lower()
        assert n not in ("x-call-id", "x-caller-id") and not n.startswith(("x-signalwire", "x-cid", "x-sw-"))
        assert 3 <= len(h["name"]) <= 256 and h["value"].isascii()


def test_reserved_header_variable_is_refused(monkeypatch, tmp_path):
    spec = json.loads((A.SPECS_DIR / "pharmacy_refill.json").read_text())
    spec["system_prompt"] += " {{caller_id}}"
    for v in spec["variants"]:
        v["caller_id"] = "x"
    fake = tmp_path / "agents"
    fake.mkdir()
    (fake / "pharmacy_refill.json").write_text(json.dumps(spec))
    monkeypatch.setattr(A, "SPECS_DIR", fake)
    with pytest.raises(ValueError):
        A.sip_headers("pharmacy_refill", 0)

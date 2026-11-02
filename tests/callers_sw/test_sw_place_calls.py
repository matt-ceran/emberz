"""place_calls: dry run by default, exact request shape, nothing sent without --live (no network)."""
import json
from pathlib import Path

import httpx
import pytest

from callers import common
from callers.signalwire import personas as P
from callers.signalwire import place_calls as PC

SCHEMA = json.loads((Path(__file__).parent / "dial_params_schema.json").read_text())
CALLER, GUARDED = "+13045550100", "+13235550199"   # fake numbers: tests never use the real ones


def settings(secrets=True, **kw):
    base = dict(caller_number=CALLER, guarded_number=GUARDED)
    if secrets:
        base.update(space="example-space.signalwire.com", project_id="PID-SECRET-123", api_token="TOKEN-SECRET-456")
    base.update(kw)
    return PC.Settings(**base)


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(PC, "load_settings", lambda with_secrets: settings(secrets=with_secrets))
    monkeypatch.setattr(PC, "AI_CALLS_LOG", tmp_path / "ai_calls.jsonl")
    monkeypatch.setattr(common, "STATE_DIR", tmp_path / "state")

    def no_network(*a, **k):
        raise AssertionError("real network access attempted")
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", no_network)


def run(argv, **kw):
    lines = []
    args = PC.parse_args(argv + ["--log", str(PC.AI_CALLS_LOG)])
    rc = PC.cmd_call(args, out=lines.append, **kw)
    return rc, lines


# ---------------------------------------------------------------- tiny JSON-schema subset check
def check(schema, value, path="$"):
    errs = []
    if "oneOf" in schema:
        ok = [s for s in schema["oneOf"] if not check(s, value, path)]
        return [] if len(ok) >= 1 else [f"{path}: matches no oneOf branch"]
    t = schema.get("type")
    types = {"object": dict, "array": list, "string": str, "integer": int, "number": (int, float), "boolean": bool}
    if t and not isinstance(value, types[t]):
        return [f"{path}: expected {t}"]
    if "enum" in schema and value not in schema["enum"]:
        errs.append(f"{path}: {value!r} not in enum")
    if "minimum" in schema and value < schema["minimum"]:
        errs.append(f"{path}: below minimum")
    if "maximum" in schema and value > schema["maximum"]:
        errs.append(f"{path}: above maximum")
    if t == "object" and "properties" in schema:
        for k in schema.get("required", []):
            if k not in value:
                errs.append(f"{path}: missing {k}")
        for k, v in value.items():
            if k not in schema["properties"]:
                errs.append(f"{path}: unknown property {k}")
            else:
                errs += check(schema["properties"][k], v, f"{path}.{k}")
    if t == "array" and "items" in schema:
        for i, v in enumerate(value):
            errs += check(schema["items"], v, f"{path}[{i}]")
    return errs


# ---------------------------------------------------------------- dry run
def test_default_is_dry_run_with_all_fifty(monkeypatch):
    monkeypatch.setattr(PC, "make_client", lambda *a, **k: pytest.fail("dry run must not create a client"))
    rc, lines = run([])
    assert rc == 0
    out = json.loads(lines[0])
    assert out["dry_run"] is True and out["calls"] == 50 and len(out["requests"]) == 50
    for r in out["requests"]:
        req = r["request"]
        assert req["method"] == "POST" and req["url"].endswith("/api/calling/calls")
        assert "redacted" in req["headers"]["Authorization"]
        assert req["json"]["params"]["from"] == CALLER and req["json"]["params"]["to"] == GUARDED
    assert "SECRET" not in lines[0] and "example-space" not in lines[0]


def test_dry_run_body_matches_signalwire_schema():
    assert SCHEMA["dial_command"]["command_enum"] == ["dial"]
    for c in P.plan_calls(list(P.PERSONAS), 10):
        body = PC.dial_body(settings(), c, "inline")
        assert body["command"] == "dial"
        assert check(SCHEMA["swml_params"], body["params"]) == []
    body = PC.dial_body(settings(public_base_url="https://x.trycloudflare.com"), P.PlannedCall("bank_fraud", 0, 0, 0), "url",
                        status_url="https://x.trycloudflare.com/status", max_price_per_minute=0.05)
    assert check(SCHEMA["url_params"], body["params"]) == []
    assert body["params"]["url"] == "https://x.trycloudflare.com/callers/swml/bank_fraud/0?fill=0"


def test_schema_check_is_not_vacuous():
    bad = {"from": CALLER, "to": GUARDED, "swml": {}, "timeout": 900, "bogus": 1, "status_events": ["nope"]}
    errs = check(SCHEMA["swml_params"], bad)
    assert any("maximum" in e for e in errs) and any("bogus" in e for e in errs) and any("enum" in e for e in errs)


def test_summary_and_cost():
    rc, lines = run(["--summary", "--personas", "bank_fraud", "pharmacy_refill", "--calls", "2"])
    assert rc == 0 and len(lines) == 4 + 2
    est = json.loads(lines[4])
    assert est == {"dry_run": True, "calls": 4, "est_minutes": 10.0, "est_usd": round(10 * 0.168, 2)}


def test_no_call_without_live(monkeypatch):
    posted = []
    monkeypatch.setattr(PC, "make_client", lambda *a, **k: posted.append(1))
    run(["--personas", "bank_fraud", "--calls", "1", "--max-calls", "5"])
    assert posted == [] and not PC.AI_CALLS_LOG.exists()


# ---------------------------------------------------------------- live guards
def test_live_refuses_more_than_max_calls(monkeypatch):
    monkeypatch.setattr(PC, "make_client", lambda *a, **k: pytest.fail("must refuse before any client"))
    with pytest.raises(SystemExit, match="--max-calls is 1"):
        run(["--live"])
    with pytest.raises(SystemExit, match="ceiling 60"):
        run(["--live", "--calls", "13", "--max-calls", "100"])


@pytest.mark.parametrize("kw,needle", [
    (dict(api_token=""), "SIGNALWIRE_API_TOKEN"),
    (dict(caller_number="3045550100"), "SIGNALWIRE_CALLER_NUMBER"),
    (dict(guarded_number=""), "TELNYX_GUARDED_NUMBER"),
    (dict(guarded_number=CALLER), "same"),
])
def test_live_refuses_bad_settings(monkeypatch, kw, needle):
    monkeypatch.setattr(PC, "load_settings", lambda with_secrets: settings(**kw))
    monkeypatch.setattr(PC, "make_client", lambda *a, **k: pytest.fail("must refuse before any client"))
    with pytest.raises(SystemExit, match=needle):
        run(["--live", "--personas", "bank_fraud", "--calls", "1"])


def test_live_url_mode_needs_https_base(monkeypatch):
    monkeypatch.setattr(PC, "make_client", lambda *a, **k: pytest.fail("must refuse"))
    with pytest.raises(SystemExit, match="PUBLIC_BASE_URL"):
        run(["--live", "--mode", "url", "--personas", "bank_fraud", "--calls", "1"])


# ---------------------------------------------------------------- live against a mock SignalWire
def mock_signalwire(posts, post_status=200):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "example-space.signalwire.com"
        assert request.headers["authorization"].startswith("Basic ")
        if request.method == "GET" and request.url.path == "/api/relay/rest/phone_numbers":
            return httpx.Response(200, json={"links": {}, "data": [
                {"number": CALLER, "number_type": "longcode", "capabilities": ["voice"], "call_handler": "laml_webhooks"}]})
        if request.method == "POST" and request.url.path == "/api/calling/calls":
            body = json.loads(request.content)
            posts.append(body)
            if post_status != 200:
                return httpx.Response(post_status, json={"errors": [{"code": "x", "message": "bad"}]})
            return httpx.Response(200, json={"id": f"call-{len(posts)}", "status": "queued"})
        return httpx.Response(404)
    return httpx.MockTransport(handler)


def live_client(posts, **kw):
    transport = mock_signalwire(posts, **kw)
    return lambda s, t=None: httpx.Client(base_url=f"https://{s.host}", auth=(s.project_id, s.api_token),
                                          transport=transport)


def test_live_places_and_logs(monkeypatch):
    posts, prompts = [], []
    monkeypatch.setattr(PC, "make_client", live_client(posts))
    rc, lines = run(["--live", "--personas", "bank_fraud", "pharmacy_refill", "--calls", "2", "--max-calls", "4"],
                    wait=lambda msg: prompts.append(msg) or "")
    assert rc == 0 and len(posts) == 4 and len(prompts) == 4
    assert all(p["params"]["to"] == GUARDED and p["params"]["from"] == CALLER for p in posts)
    assert all(P.validate_swml(p["params"]["swml"]) == [] for p in posts)
    rows = [json.loads(x) for x in PC.AI_CALLS_LOG.read_text().splitlines()]
    assert [r["call_id"] for r in rows] == ["call-1", "call-2", "call-3", "call-4"]
    r0 = rows[0]
    assert r0["label"] == "ai" and r0["persona"] == "bank_fraud" and r0["kind"] == "scam"
    assert r0["voice"] == "azure.en-US-AndrewNeural" and r0["engine"] == "azure" and r0["platform"] == "signalwire"
    assert r0["to_last4"] == GUARDED[-4:] and "started_utc" in r0
    shared = [json.loads(x) for x in (common.STATE_DIR / "calls.jsonl").read_text().splitlines()]
    assert [s["agent"] for s in shared] == ["bank_fraud", "bank_fraud", "pharmacy_refill", "pharmacy_refill"]


def test_live_q_stops(monkeypatch):
    posts = []
    monkeypatch.setattr(PC, "make_client", live_client(posts))
    answers = iter(["", "q"])
    rc, lines = run(["--live", "--personas", "bank_fraud", "--calls", "3", "--max-calls", "3"],
                    wait=lambda msg: next(answers))
    assert rc == 0 and len(posts) == 1 and lines[-1] == "stopped"


def test_live_without_terminal_never_dials(monkeypatch):
    posts = []
    monkeypatch.setattr(PC, "make_client", live_client(posts))

    def eof(msg):
        raise EOFError
    rc, lines = run(["--live", "--personas", "bank_fraud", "--calls", "1"], wait=eof)
    assert rc == 0 and posts == [] and lines[-1] == "stopped"


def test_live_gap_mode_sleeps_between_calls(monkeypatch):
    posts, slept = [], []
    monkeypatch.setattr(PC, "make_client", live_client(posts))
    rc, _ = run(["--live", "--personas", "tech_support", "--calls", "3", "--max-calls", "3", "--wait", "gap",
                 "--gap-s", "240"], wait=lambda m: pytest.fail("no prompt in gap mode"), sleep=slept.append)
    assert rc == 0 and len(posts) == 3 and slept == [240.0, 240.0]


def test_live_failed_request_is_logged_and_stops(monkeypatch):
    posts = []
    monkeypatch.setattr(PC, "make_client", live_client(posts, post_status=422))
    rc, lines = run(["--live", "--personas", "bank_fraud", "--calls", "2", "--max-calls", "2"], wait=lambda m: "")
    assert rc == 1 and len(posts) == 1
    row = json.loads(PC.AI_CALLS_LOG.read_text().splitlines()[0])
    assert row["call_id"] is None and row["status"] == "request_failed" and row["http_status"] == 422


def test_logs_reads_voice_logs(monkeypatch, tmp_path):
    PC.append_jsonl(PC.AI_CALLS_LOG, {"call_id": "abc", "persona": "bank_fraud", "voice": "azure.en-US-AndrewNeural"})
    PC.append_jsonl(PC.AI_CALLS_LOG, {"call_id": None, "status": "request_failed"})

    def handler(request):
        assert request.method == "GET"
        if request.url.path == "/api/voice/logs/abc":
            return httpx.Response(200, json={"id": "abc", "status": "completed", "duration": 170, "charge": 0.47,
                                             "charge_details": [{"description": "Outbound Voice Minutes", "charge": 0.024}]})
        if request.url.path == "/api/voice/logs/abc/events":
            return httpx.Response(200, json={"data": [{"type": "calling.script.warning", "code": "relay_script_voice_unresolved"}]})
        return httpx.Response(404)
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(PC, "make_client", lambda s, t=None: httpx.Client(base_url="https://x", transport=transport))
    lines = []
    PC.cmd_logs(PC.parse_args(["logs", "--log", str(PC.AI_CALLS_LOG)]), out=lines.append)
    first = json.loads(lines[0])
    assert first["voice_unresolved"] is True and first["charge"] == 0.47
    assert json.loads(lines[-1]) == {"calls": 1, "total_charge_usd": 0.47}


def test_settings_host_forms():
    assert settings(space="myspace").host == "myspace.signalwire.com"
    assert settings(space="myspace.signalwire.com").host == "myspace.signalwire.com"
    assert settings(space="https://myspace.signalwire.com/").host == "myspace.signalwire.com"

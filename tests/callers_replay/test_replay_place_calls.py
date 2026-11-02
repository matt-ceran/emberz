"""Dry-run payloads and the --live loop (with fakes) for callers/human_replay/place_calls.py."""
import argparse
import json
from datetime import datetime, timedelta, timezone

import pytest

from callers.human_replay import build_set, place_calls as pc

import httpx as _httpx

_REAL_HTTPX_CLIENT = _httpx.Client      # saved before the autouse fixture patches it

CALLER, GUARDED = "+13046481098", "+13235161282"
ENV = {"SIGNALWIRE_CALLER_NUMBER": CALLER, "TELNYX_GUARDED_NUMBER": GUARDED}


def rows(n=3):
    return [{"clip_id": f"hr{i:03d}", "source": "librispeech", "speaker": f"librispeech:{i}", "gender": "female",
             "license": "CC BY 4.0", "duration": "8.000", "path": f"data/human_replay/clips/hr{i:03d}.wav",
             "sha256": f"{i:064x}"} for i in range(1, n + 1)]


@pytest.fixture(autouse=True)
def isolate(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise AssertionError("must not touch the network or run gcloud here")
    import httpx
    monkeypatch.setattr(httpx, "Client", boom)
    monkeypatch.setattr(httpx, "post", boom)
    monkeypatch.setattr(pc.subprocess, "run", boom)
    monkeypatch.setattr(pc, "CALL_LOG", tmp_path / "recordings" / "human_replay_calls.jsonl")
    monkeypatch.setattr(pc, "SHARED_LEDGER", tmp_path / "data" / "callers" / "calls.jsonl")
    monkeypatch.setattr(build_set, "read_manifest", lambda path=None: rows(3))


def run_args(**kw):
    base = dict(live=False, yes=True, manifest=None, clips=None, start=0, limit=None, skip_done=False, gap_s=0.0,
                lead_silence_s=2, ring_timeout_s=30, max_price_per_minute=0.05, max_calls=80, url_ttl="2h")
    base.update(kw)
    return argparse.Namespace(**base)


def test_swml_plays_silence_then_clip_then_hangs_up():
    p = pc.build_dial(CALLER, GUARDED, "https://storage.googleapis.com/b/o.wav?X-Goog-Signature=abc", lead_silence_s=3)
    assert p["command"] == "dial"
    params = p["params"]
    assert params["from"] == CALLER and params["caller_id"] == CALLER and params["to"] == GUARDED
    assert params["timeout"] == 30 and params["max_price_per_minute"] == 0.05
    main = params["swml"]["sections"]["main"]
    assert params["swml"]["version"] == "1.0.0"
    assert main[0] == {"play": {"url": "silence:3"}}
    assert main[1]["play"]["url"].startswith("https://storage.googleapis.com/")
    assert main[-1] == {"hangup": {}}
    assert "url" not in params                     # inline SWML, no webhook URL
    json.dumps(p)                                   # JSON-serializable as sent


def test_no_lead_silence():
    main = pc.build_swml("https://x/a.wav", lead_silence_s=0)["sections"]["main"]
    assert main[0] == {"play": {"url": "https://x/a.wav"}}


@pytest.mark.parametrize("a,b", [("3046481098", GUARDED), (CALLER, "+1"), (CALLER, CALLER)])
def test_bad_numbers_rejected(a, b):
    with pytest.raises(ValueError):
        pc.build_dial(a, b, "https://x/a.wav")


def test_dry_run_prints_payloads_and_touches_nothing(capsys, tmp_path):
    assert pc.cmd_run(run_args(), ENV) == 0
    out = [json.loads(l) for l in capsys.readouterr().out.splitlines() if l.startswith("{")]
    assert [o["clip_id"] for o in out] == ["hr001", "hr002", "hr003"]
    for o in out:
        assert o["POST"] == "/api/calling/calls"
        assert "DRY-RUN" in o["payload"]["params"]["swml"]["sections"]["main"][1]["play"]["url"]
        assert o["upload"].endswith(pc.object_name({"clip_id": o["clip_id"], "sha256": f"{int(o['clip_id'][2:]):064x}"}))
    assert not pc.CALL_LOG.exists() and not pc.SHARED_LEDGER.exists()


def test_dry_run_without_secrets_is_fine(capsys):
    assert pc.cmd_run(run_args(limit=1), dict(ENV)) == 0


def test_missing_numbers_refused():
    with pytest.raises(SystemExit):
        pc.cmd_run(run_args(), {})


def test_max_calls_cap():
    with pytest.raises(SystemExit):
        pc.cmd_run(run_args(max_calls=2), ENV)


def test_pick_rows():
    r = rows(5)
    assert [x["clip_id"] for x in pc.pick_rows(r, None, 1, 2, set())] == ["hr002", "hr003"]
    assert [x["clip_id"] for x in pc.pick_rows(r, "hr004,hr001", 0, None, set())] == ["hr001", "hr004"]
    assert [x["clip_id"] for x in pc.pick_rows(r, None, 0, None, {"hr001", "hr002"})] == ["hr003", "hr004", "hr005"]
    with pytest.raises(SystemExit):
        pc.pick_rows(r, "hr999", 0, None, set())


def test_object_name_is_content_addressed():
    assert pc.object_name({"clip_id": "hr007", "sha256": "abcdef0123456789ff"}) == "human_replay/hr007-abcdef012345.wav"


def test_sw_base():
    assert pc.sw_base("example") == "https://example.signalwire.com"
    assert pc.sw_base("https://example.signalwire.com/") == "https://example.signalwire.com"


def test_load_env_takes_only_signalwire_keys_from_external_file(tmp_path, monkeypatch):
    ext = tmp_path / "other.env"
    ext.write_text('SIGNALWIRE_SPACE="sp"\nSIGNALWIRE_PROJECT_ID=pid\nSIGNALWIRE_API_TOKEN=tok\nOTHER_SECRET=nope\n')
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".env").write_text(f"TELNYX_ENV_FILE={ext}\nSIGNALWIRE_CALLER_NUMBER={CALLER}\n")
    monkeypatch.setattr(pc, "ROOT", repo)
    env = pc.load_env({})
    assert env["SIGNALWIRE_SPACE"] == "sp" and env["SIGNALWIRE_API_TOKEN"] == "tok"
    assert env["SIGNALWIRE_CALLER_NUMBER"] == CALLER
    assert "OTHER_SECRET" not in env


class FakeSW:
    def __init__(self, statuses, fail_dial_for=()):
        self.statuses = list(statuses)
        self.dials = []
        self.fail_dial_for = set(fail_dial_for)

    def find_number(self, n):
        return {"number": n}

    def dial(self, payload):
        url = payload["params"]["swml"]["sections"]["main"][1]["play"]["url"]
        if any(c in url for c in self.fail_dial_for):
            raise RuntimeError("HTTP 422")
        self.dials.append(payload)
        return {"id": f"sw-{len(self.dials)}", "status": "queued"}

    def call_status(self, call_id):
        return self.statuses.pop(0) if self.statuses else "completed"

    def end(self, call_id):
        self.ended = getattr(self, "ended", []) + [call_id]
        return True


def live_setup(monkeypatch):
    monkeypatch.setattr(build_set, "validate_manifest", lambda *a, **k: [])
    monkeypatch.setattr(pc, "upload", lambda todo, log=print: None)
    exp = datetime.now(timezone.utc) + timedelta(hours=2)
    monkeypatch.setattr(pc, "sign", lambda todo, ttl="2h": {r["clip_id"]: (f"https://signed/{r['clip_id']}.wav", exp) for r in todo})


def test_live_loop_logs_each_call(monkeypatch):
    live_setup(monkeypatch)
    sw = FakeSW(["ringing", "completed", "completed", "completed"])
    sleeps = []
    assert pc.cmd_run(run_args(live=True, gap_s=5), ENV, sw_factory=lambda env: sw, sleep=sleeps.append) == 0
    assert len(sw.dials) == 3
    log = [json.loads(l) for l in pc.CALL_LOG.read_text().splitlines()]
    placed = [r for r in log if "event" not in r]
    assert [r["clip_id"] for r in placed] == ["hr001", "hr002", "hr003"]
    for r in placed:
        assert {"call_id", "clip_id", "speaker", "started_at"} <= r.keys()
    assert [r["status"] for r in log if r.get("event") == "ended"] == ["completed"] * 3
    ledger = [json.loads(l) for l in pc.SHARED_LEDGER.read_text().splitlines()]
    assert all(r["kind"] == "human" and r["platform"] == "signalwire_replay" and "started_utc" in r for r in ledger)
    assert sleeps.count(5) == 2                     # gap between calls, not after the last
    assert pc.done_clips(pc.CALL_LOG) == {"hr001", "hr002", "hr003"}


def test_live_stops_after_two_failed_calls(monkeypatch):
    live_setup(monkeypatch)
    sw = FakeSW(["failed", "no-answer"])
    assert pc.cmd_run(run_args(live=True), ENV, sw_factory=lambda env: sw, sleep=lambda s: None) == 1
    assert len(sw.dials) == 2


def test_live_stops_after_two_dial_errors(monkeypatch):
    live_setup(monkeypatch)
    sw = FakeSW([], fail_dial_for=("hr001", "hr002"))
    assert pc.cmd_run(run_args(live=True), ENV, sw_factory=lambda env: sw, sleep=lambda s: None) == 1
    assert not pc.SHARED_LEDGER.exists()


def test_live_needs_typed_yes(monkeypatch):
    live_setup(monkeypatch)
    monkeypatch.setattr("builtins.input", lambda prompt="": "no")
    sw = FakeSW([])
    assert pc.cmd_run(run_args(live=True, yes=False), ENV, sw_factory=lambda env: sw, sleep=lambda s: None) == 1
    assert sw.dials == []


def test_live_stops_on_uncertain_dial(monkeypatch):
    """A dial lost in transit may have placed a call: log it and stop, never retry."""
    live_setup(monkeypatch)

    class Lost(FakeSW):
        def dial(self, payload):
            raise pc.DialUncertain("ReadTimeout")

    sw = Lost([])
    assert pc.cmd_run(run_args(live=True), ENV, sw_factory=lambda env: sw, sleep=lambda s: None) == 1
    log = [json.loads(l) for l in pc.CALL_LOG.read_text().splitlines()]
    assert [r["event"] for r in log] == ["dial_uncertain"]


def test_live_ends_a_call_that_is_not_reported_ended(monkeypatch):
    live_setup(monkeypatch)
    sw = FakeSW(["in-progress"] * 100)
    assert pc.cmd_run(run_args(live=True, limit=1), ENV, sw_factory=lambda env: sw, sleep=lambda s: None) == 0
    assert sw.ended == ["sw-1"]


def test_signalwire_client_maps_transport_errors():
    import httpx

    def handler(request):
        if request.url.path.startswith("/api/voice/logs/"):
            raise httpx.ConnectError("down", request=request)
        raise httpx.ReadTimeout("slow", request=request)

    env = {"SIGNALWIRE_SPACE": "example", "SIGNALWIRE_PROJECT_ID": "p", "SIGNALWIRE_API_TOKEN": "t"}
    client = _REAL_HTTPX_CLIENT(transport=httpx.MockTransport(handler))   # httpx.Client itself is patched to fail
    sw = pc.SignalWire(env, client=client)
    with pytest.raises(pc.DialUncertain):
        sw.dial(pc.build_dial(CALLER, GUARDED, "https://x/a.wav"))
    assert sw.call_status("abc") is None
    assert sw.end("abc") is False


def test_cli_run_is_dry_by_default(monkeypatch, capsys):
    monkeypatch.setattr(pc, "load_env", lambda env=None: dict(ENV))
    monkeypatch.setattr(pc, "SignalWire", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no client in dry run")))
    assert pc.main(["run", "--limit", "2"]) == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out and out.count('"POST"') == 2


def test_limit_zero_means_no_calls_not_all(monkeypatch):
    assert pc.pick_rows(rows(5), None, 0, 0, set()) == []
    with pytest.raises(SystemExit):
        pc.pick_rows(rows(5), None, 0, -1, set())
    live_setup(monkeypatch)
    sw = FakeSW([])
    assert pc.cmd_run(run_args(live=True, limit=0), ENV, sw_factory=lambda env: sw, sleep=lambda s: None) == 0
    assert sw.dials == []


def test_live_prompt_without_terminal_aborts(monkeypatch):
    live_setup(monkeypatch)

    def eof(prompt=""):
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)
    sw = FakeSW([])
    assert pc.cmd_run(run_args(live=True, yes=False), ENV, sw_factory=lambda env: sw, sleep=lambda s: None) == 1
    assert sw.dials == []

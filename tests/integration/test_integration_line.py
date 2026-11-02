"""End-to-end: the real app (line/app.build_app) driven by synthetic Telnyx webhooks + media socket.

Scripted-turn scenes run with the stub guardian (no network, faster than real time). The real-audio
scene (macOS say -> ffmpeg -> sherpa ASR) runs only with EMBERZ_AUDIO_TESTS=1.
"""
from __future__ import annotations

import http.server
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contracts.models import ConsoleEvent, GuardianRequest, WindowTurn  # noqa: E402
from line import simulate_call as sim  # noqa: E402
from line.guardian_client import GuardianClient  # noqa: E402
from line.matt_bridge import ConsoleForwarder, MattGuardianClient  # noqa: E402

STUB = GuardianClient(backend="stub").score


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("CONSOLE_TOKEN", "t")
    monkeypatch.delenv("MATT_CONSOLE_URL", raising=False)
    monkeypatch.setenv("EMBERZ_VOICE_DETECTOR", "0")


def run(name, **kw):
    return sim.run_scene(name, text_mode=True, fast=True, quiet=True, score_fn=STUB, **kw)


def test_bank_scene_alerts_coaches_and_caretaker_listens():
    r = run("bank")
    assert len(r.of("turn")) == 5
    assert len(r.of("alert")) == 1
    coach = [e for e in r.of("coach") if e.data["sent"]]
    assert coach and coach[0].data["turn"] == 5
    # the coach went to the protected leg only
    assert r.speaks and r.speaks[0][1]["call_control_ids"]
    assert [e.data["mode"] for e in r.of("caretaker_joined")] == ["monitor"]
    assert r.of("call_ended") and r.of("summary")
    assert r.coach_latency_ms is not None and r.coach_latency_ms < 2000


def test_grandpa_allowed_contact_is_silent():
    r = run("grandpa")
    assert r.of("call_started")[0].data["caller_known"] is True
    assert not r.of("alert") and not r.of("coach")


def test_pharmacy_bot_no_alert_but_flagged_ai():
    r = run("pharmacy")
    assert not r.of("alert") and not r.of("coach")
    assert any(e.data.get("model") == "timing" for e in r.of("voice"))


def test_action_safe_suppresses_alerts():
    r = run("bank", actions={4: "safe"})
    assert r.action_replies[0]["result"] == "marked_safe"
    assert not r.of("alert") and not [e for e in r.of("coach") if e.data.get("sent")]


def test_action_hangup_ends_call():
    r = run("bank", actions={2: "hangup"})
    assert r.action_replies[0]["result"] == "hung_up"
    ended = r.of("call_ended")
    assert ended and ended[0].data["reason"] == "console_hangup"
    assert len(r.of("turn")) == 2


def test_actions_listen_join_and_token():
    from fastapi.testclient import TestClient

    from line.app import build_app
    from line.call_flow import encode_client_state
    from line.config import GuardedLine, Settings
    from line.telnyx_client import FakeTelnyx

    tel = FakeTelnyx()
    app = build_app(telnyx=tel, settings=Settings(telnyx_api_key="x", public_base_url="http://localhost:8000",
                                                  skip_verify=True),
                    profiles={sim.GUARDED: GuardedLine(profile="elder", protected_number=sim.PROTECTED,
                                                       caretaker_number=sim.CARETAKER)},
                    score_fn=STUB, summarizer=sim.local_summary, record=False, voice_detector=False,
                    transcriber_factory=sim.ScriptTranscriber)
    with TestClient(app) as c:
        def wh(et, p):
            assert c.post("/webhooks/telnyx", json={"data": {"event_type": et, "id": f"{et}-{time.time()}",
                                                             "payload": p}}).status_code == 200
        wh("call.initiated", {"call_control_id": "A1", "call_session_id": "s1", "from": sim.STRANGER,
                              "to": sim.GUARDED, "direction": "incoming"})
        cid = "c_s1"
        wh("call.answered", {"call_control_id": "A1", "client_state": encode_client_state("A", cid)})
        wh("call.speak.ended", {"call_control_id": "A1", "client_state": encode_client_state("A", cid)})
        assert c.post(f"/actions/{cid}/join").status_code == 401
        assert c.post(f"/actions/{cid}/join?token=t").status_code == 409  # caretaker not listening yet
        assert c.post(f"/actions/{cid}/bogus?token=t").status_code == 404
        assert c.post("/actions/nope/listen?token=t").status_code == 404
        r = c.post(f"/actions/{cid}/listen", headers={"X-Console-Token": "t"}).json()
        assert r["result"] == "dialed" and r["caretaker_state"] == "ringing"
        assert any(k == "dial" and b["to"] == sim.CARETAKER for k, _, _, b in tel.log)
        s = app.state.flow.sessions[cid]
        cs = encode_client_state("C", cid)
        wh("call.answered", {"call_control_id": s.c_ccid, "client_state": cs})
        wh("call.gather.ended", {"call_control_id": s.c_ccid, "client_state": cs, "digits": "1", "status": "valid"})
        assert s.caretaker_state == "monitoring"
        assert c.post(f"/actions/{cid}/join?token=t").json()["caretaker_state"] == "barged"
        assert c.post(f"/actions/{cid}/hangup?token=t").json()["result"] == "hung_up"
        assert cid not in app.state.flow.sessions


def test_console_forwarder_posts_events():
    got: list[dict] = []

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            got.append({"path": self.path, "body": json.loads(self.rfile.read(int(self.headers["content-length"])))})
            self.send_response(200)
            self.end_headers()

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        fwd = ConsoleForwarder(f"http://127.0.0.1:{srv.server_port}")
        fwd(ConsoleEvent(type="turn", call_id="c1", at_ms=5, data={"text": "hi"}))
        deadline = time.time() + 5
        while not got and time.time() < deadline:
            time.sleep(0.02)
        fwd.close()
        assert got and got[0]["path"] == "/events" and got[0]["body"]["call_id"] == "c1"
        # a dead console never blocks: events are dropped
        dead = ConsoleForwarder("http://127.0.0.1:9", timeout_s=0.2)
        t0 = time.time()
        for _ in range(5):
            dead(ConsoleEvent(type="turn", call_id="c1", at_ms=0))
        assert time.time() - t0 < 0.1
        dead.close()
    finally:
        srv.shutdown()


def test_matt_client_uses_render_contract(monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "x")

    class Render:
        @staticmethod
        def render_messages(req):
            return [{"role": "user", "content": f"turns={len(req.turns)}"}]

        @staticmethod
        def parse_response(text):
            return {"risk": "alert", "category": "scam.impersonation_bank_gov", "severity": 2, "source": "caller",
                    "stage": "ask", "action": "coach", "coach_line": text}

    req = GuardianRequest(call_id="c", profile="elder", rules=[], caller_known=False,
                          turns=[WindowTurn(turn=1, speaker="caller", text="read me the code")])
    m = MattGuardianClient(model="acct/models/x", render=Render)
    body = m._body(req)
    assert body["model"] == "acct/models/x" and body["messages"][0]["content"] == "turns=1"
    assert "response_format" not in body
    assert m._parse("hang up").coach_line == "hang up"
    fallback = MattGuardianClient(model="acct/models/x", render=None)
    assert "response_format" in fallback._body(req)


@pytest.mark.skipif(os.environ.get("EMBERZ_AUDIO_TESTS") != "1" or not sim.can_synthesize()
                    or not shutil.which("say"), reason="set EMBERZ_AUDIO_TESTS=1 (macOS say + ffmpeg + ASR model)")
def test_bank_scene_real_audio():
    r = sim.run_scene("bank", quiet=True, score_fn=STUB)
    assert r.of("alert") and r.coach_latency_ms is not None

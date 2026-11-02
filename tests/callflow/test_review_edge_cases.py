"""Adversarial edge cases added in review (each one failed before the fixes in call_flow/server/hour0)."""
import base64
import json
import time
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from nacl.signing import SigningKey

from callflow_helpers import A_CCID, CALL_ID, CALLER, GUARDED, PROTECTED, Line
from line.call_flow import ANON_DISPLAY_NAME
from line.config import DEMO_APP_ID, Settings
from line.server import create_app
from line.telnyx_client import FakeTelnyx


def _media(track, chunk, payload: bytes, ts=0):
    return json.dumps({"event": "media", "media": {"track": track, "chunk": str(chunk), "timestamp": str(ts),
                                                  "payload": base64.b64encode(payload).decode()}})


# ------------------------------------------------------------------ call flow
def test_late_duplicate_initiated_does_not_resurrect_an_ended_call(line):
    line.inbound()
    line.post("call.hangup", line.leg("A", A_CCID, hangup_cause="normal_clearing"))
    assert CALL_ID not in line.flow.sessions
    n = len(line.telnyx.log)
    line.inbound()  # Telnyx redelivery with a new event id, after the call ended
    assert len(line.telnyx.log) == n and CALL_ID not in line.flow.sessions
    assert len(line.brains) == 1


def test_missing_public_url_still_connects_the_call():
    line = Line()
    line.flow.settings = replace(line.settings, public_base_url="")
    line.inbound()
    line.post("call.answered", line.leg("A", A_CCID))
    assert not line.calls("call", "streaming_start")
    assert len(line.calls("call", "speak")) == 1 and len(line.calls("dial")) == 1


def test_dial_with_caller_id_rejected_falls_back_to_guarded_number(line):
    real_dial = line.telnyx.dial

    async def picky_dial(**body):
        if body.get("diversion"):
            line.telnyx.log.append(("dial", None, None, body))
            raise RuntimeError("Telnyx 422 on /calls: invalid from")
        return await real_dial(**body)

    line.telnyx.dial = picky_dial
    line.inbound()
    line.post("call.answered", line.leg("A", A_CCID))
    dials = line.calls("dial")
    assert len(dials) == 2
    assert dials[1][3]["from"] == GUARDED and "diversion" not in dials[1][3]
    assert dials[1][3]["from_display_name"]
    assert line.b_ccid()


def test_dial_b_impossible_ends_the_call(line):
    async def broken_dial(**body):
        line.telnyx.log.append(("dial", None, None, body))
        raise RuntimeError("Telnyx 403")

    line.telnyx.dial = broken_dial
    line.inbound()
    line.post("call.answered", line.leg("A", A_CCID))
    assert [h[1] for h in line.calls("call", "hangup")] == [A_CCID]  # the caller is not left in silence
    assert CALL_ID not in line.flow.sessions
    assert line.hub.events[-1].data["reason"] == "dial_b_failed"


def test_conference_create_failure_is_retried_when_b_answers(line):
    calls = {"n": 0}
    real = line.telnyx.conference_create

    async def flaky(**body):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("Telnyx 500")
        return await real(**body)

    line.telnyx.conference_create = flaky
    line.inbound()
    line.post("call.answered", line.leg("A", A_CCID))
    line.post("call.speak.ended", line.leg("A", A_CCID))  # first create fails
    line.post("call.answered", line.leg("B", line.b_ccid()))  # retried here
    assert calls["n"] == 2
    assert line.calls("conference", "join")[0][3]["call_control_id"] == line.b_ccid()


def test_b_dial_has_a_time_limit(line):
    line.inbound()
    line.post("call.answered", line.leg("A", A_CCID))
    assert line.calls("dial")[0][3]["time_limit_secs"] > 0


def test_deepfake_can_be_switched_off():
    line = Line()
    line.flow.settings = replace(line.settings, deepfake=False)
    line.inbound()
    assert "deepfake_detection" not in line.telnyx.log[0][3]


def test_from_without_plus_is_normalized(line):
    line.inbound(caller="15555550199", to=GUARDED[1:])
    s = line.flow.sessions[CALL_ID]
    assert (s.caller_number, s.guarded_number) == (CALLER, GUARDED)


def test_anonymous_variants(line):
    for i, raw in enumerate(["", "Anonymous", "+anonymous", "restricted", "sip:foo@bar"]):
        line2 = Line()
        line2.inbound(caller=raw)
        line2.post("call.answered", line2.leg("A", A_CCID))
        dial = line2.calls("dial")[0][3]
        assert dial["from"] == GUARDED and dial["from_display_name"] == ANON_DISPLAY_NAME, raw


def test_other_connection_id_is_refused():
    s = Settings(telnyx_api_key="k", public_base_url="https://x.trycloudflare.com", skip_verify=True,
                 telnyx_connection_id="2941803835737245399")
    with pytest.raises(RuntimeError):
        create_app(telnyx=FakeTelnyx(), settings=s, profiles={})


def test_webhook_with_odd_id_does_not_500(line):
    raw = json.dumps({"data": {"event_type": "call.initiated", "id": {"x": 1}, "payload": {}}}).encode()
    r = line.client.post("/webhooks/telnyx", content=raw)
    assert r.status_code in (200, 400)
    r = line.client.post("/webhooks/telnyx", content=b"[1,2]")
    assert r.status_code == 400


# ------------------------------------------------------------------ media socket
def test_bad_frames_do_not_kill_the_stream(line):
    line.inbound()
    with line.client.websocket_connect(f"/media/{CALL_ID}") as ws:
        ws.send_text(_media("inbound", 1, b"\x01" * 160))
        ws.send_text("not json")
        ws.send_text("[1, 2]")
        ws.send_bytes(b"\x00\x01")
        ws.send_text(json.dumps({"event": "media", "media": None}))
        ws.send_text(_media("inbound", 2, b"\x02" * 160))
        ws.send_text(_media("inbound", 3, b"", 0))  # empty audio
        ws.send_text(json.dumps({"event": "stop"}))
    assert [p[:1] for _, p, _ in line.brains[0].media if p] == [b"\x01", b"\x02"]


def test_brain_exception_does_not_kill_the_stream():
    line = Line()

    line.inbound()
    brain = line.brains[0]
    seen = []

    async def boom(track, payload, ts):
        seen.append(payload[:1])
        if payload[:1] == b"\x01":
            raise RuntimeError("asr crashed")

    brain.on_media = boom
    with line.client.websocket_connect(f"/media/{CALL_ID}") as ws:
        for i in (1, 2, 3):
            ws.send_text(_media("inbound", i, bytes([i]) * 160))
        ws.send_text(json.dumps({"event": "stop"}))
    assert seen == [b"\x01", b"\x02", b"\x03"]


def test_chunk_counter_restart_after_new_start_is_not_dropped(line):
    line.inbound()
    start = json.dumps({"event": "start", "stream_id": "s2", "start": {"call_control_id": A_CCID}})
    with line.client.websocket_connect(f"/media/{CALL_ID}") as ws:
        for i in range(1, 6):
            ws.send_text(_media("inbound", i, bytes([i]) * 160))
        ws.send_text(start)  # a new stream on the same socket restarts chunk numbering
        for i in range(1, 4):
            ws.send_text(_media("inbound", i, bytes([0x10 + i]) * 160))
        ws.send_text(json.dumps({"event": "stop"}))
    got = [p[0] for _, p, _ in line.brains[0].media]
    assert got == [1, 2, 3, 4, 5, 0x11, 0x12, 0x13]


def test_media_socket_signature_strict_mode():
    """Telnyx signs the WS handshake over f"{ts}|{stream_url}" (media-streaming docs)."""
    key = SigningKey.generate()
    line = Line(signing_key=key, skip_verify=False)
    line.flow.settings = replace(line.settings, media_verify="strict")
    line.app.state.settings = line.flow.settings
    line.inbound()
    url = line.flow.settings.media_ws_url(CALL_ID)
    ts = str(int(time.time()))
    good = {"telnyx-timestamp": ts,
            "telnyx-signature-ed25519": base64.b64encode(key.sign(f"{ts}|{url}".encode()).signature).decode()}
    with line.client.websocket_connect(f"/media/{CALL_ID}", headers=good) as ws:
        ws.send_text(_media("inbound", 1, b"\x01" * 160))
        ws.send_text(json.dumps({"event": "stop"}))
    assert len(line.brains[0].media) == 1
    from starlette.websockets import WebSocketDisconnect
    with pytest.raises(WebSocketDisconnect):
        with line.client.websocket_connect(f"/media/{CALL_ID}") as ws:
            ws.receive_text()


# ------------------------------------------------------------------ hour0 / startup guards
def test_hour0_unmeasured_leak_is_none_not_false():
    from line.hour0 import Recorder, _any_true, analyze_auto

    out = analyze_auto(Recorder(), {})
    leak = out["notice_ringback_leak"]
    assert leak["notice_in_outbound_track"] is None and leak["ringback_in_outbound_track"] is None
    assert _any_true(None, None) is None and _any_true(False, None) is False and _any_true(None, True) is True


def test_hour0_masks_typed_numbers():
    from line.hour0 import mask_text

    out = mask_text("it showed +1 (555) 555-0199, name Emberz")
    assert "555-0199" not in out and out.endswith("0199, name Emberz")


def test_reorder_counter_restart_without_start_event():
    from line.server import TrackReorder

    r = TrackReorder(depth=1, restart_gap=10)
    got = []
    for c in (500, 501, 502, 1, 2, 3):
        got += [p for p, _ in r.push(c, bytes([c % 256]), 0)]
    got += [p for p, _ in r.flush()]
    assert got == [bytes([c % 256]) for c in (500, 501, 502, 1, 2, 3)]


def test_live_problems_guard():
    from line.server import live_problems

    ok = Settings(telnyx_api_key="k", telnyx_public_key="p", public_base_url="https://x.trycloudflare.com")
    assert live_problems(ok) == []
    assert live_problems(replace(ok, public_base_url=""))
    assert any("SKIP_VERIFY" in m for m in live_problems(replace(ok, skip_verify=True)))
    assert live_problems(replace(ok, skip_verify=True, public_base_url="http://127.0.0.1:8000")) == []
    assert live_problems(replace(ok, telnyx_public_key=""))


def test_conference_failing_twice_after_b_answered_hangs_up_both(line):
    async def broken(**body):
        line.telnyx.log.append(("conference_create", None, None, body))
        raise RuntimeError("Telnyx 500")

    line.telnyx.conference_create = broken
    line.inbound()
    line.post("call.answered", line.leg("A", A_CCID))
    b = line.b_ccid()
    line.post("call.answered", line.leg("B", b))  # B first: two tries, then give up
    assert len(line.calls("conference_create")) == 2
    assert sorted(h[1] for h in line.calls("call", "hangup")) == sorted([A_CCID, b])
    assert line.hub.events[-1].data["reason"] == "conference_failed"


def test_join_failure_is_retried_then_ends(line):
    async def broken(conference_id, action, **body):
        line.telnyx.log.append(("conference", conference_id, action, body))
        raise RuntimeError("Telnyx 422")

    line.telnyx.conference_action = broken
    line.inbound()
    line.post("call.answered", line.leg("A", A_CCID))
    line.post("call.speak.ended", line.leg("A", A_CCID))
    b = line.b_ccid()
    line.post("call.answered", line.leg("B", b))
    assert len(line.calls("conference", "join")) == 2
    assert line.hub.events[-1].data["reason"] == "join_failed"


def test_bad_deepfake_score_is_ignored(line):
    line.inbound()
    line.post("call.deepfake_detection.result", line.leg("A", A_CCID, result="fake", score="n/a"))
    assert line.flow.sessions[CALL_ID].ai_voice is None
    assert "call.deepfake_detection.result" in [e for e, _ in line.brains[0].events]

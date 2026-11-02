"""Telnyx deepfake webhook parsing (payload shape from the Telnyx docs, 2026-11-02)."""
from __future__ import annotations

from voice.telnyx_deepfake import answer_params, parse_event, parse_webhook

DOC_EVENT = {  # docs show the event without the `data` wrapper
    "record_type": "event",
    "event_type": "call.deepfake_detection.result",
    "id": "0ccc7b54-4df3-4bca-a65a-3da1ecc777f0",
    "occurred_at": "2026-02-24T12:00:00.000000Z",
    "payload": {
        "call_control_id": "v3:abc",
        "connection_id": "123",
        "call_leg_id": "leg-1",
        "call_session_id": "sess-1",
        "client_state": None,
        "result": "fake",
        "score": 0.87,
        "consistency": 94.5,
    },
}


def test_result_without_wrapper():
    s = parse_webhook(DOC_EVENT, at_ms=4200)
    assert s.model == "telnyx" and s.ai_score == 0.87 and s.at_ms == 4200
    assert s.call_id == "c_sess-1"


def test_result_with_data_wrapper_and_explicit_call_id():
    s = parse_webhook({"data": DOC_EVENT}, call_id="c_123")
    assert s.call_id == "c_123" and s.ai_score == 0.87
    r = parse_event({"data": DOC_EVENT})
    assert r.result == "fake" and r.consistency == 94.5 and r.call_control_id == "v3:abc"


def test_real_result():
    ev = {**DOC_EVENT, "payload": {**DOC_EVENT["payload"], "result": "real", "score": 0.04, "consistency": 97}}
    assert parse_webhook(ev).ai_score == 0.04


def test_silence_timeout_gives_no_signal():
    ev = {**DOC_EVENT, "payload": {**DOC_EVENT["payload"], "result": "silence_timeout", "score": None, "consistency": None}}
    assert parse_event(ev).result == "silence_timeout"
    assert parse_webhook(ev) is None


def test_error_event():
    ev = {"data": {"event_type": "call.deepfake_detection.error",
                   "payload": {"call_control_id": "v3:abc", "call_session_id": "s", "error_message": "rtp_timeout"}}}
    r = parse_event(ev)
    assert r.error == "rtp_timeout"
    assert parse_webhook(ev) is None


def test_other_events_ignored():
    assert parse_event({"data": {"event_type": "call.answered", "payload": {}}}) is None
    assert parse_webhook({"data": {"event_type": "call.hangup", "payload": {}}}) is None


def test_answer_params():
    assert answer_params() == {"deepfake_detection": {"enabled": True, "timeout": 15, "rtp_timeout": 30}}


# ---- reviewer additions
def test_malformed_bodies_never_raise():
    ev = lambda p: {"data": {"event_type": "call.deepfake_detection.result", "payload": p}}
    assert parse_webhook(ev({"result": "fake", "score": "abc", "call_session_id": "s"})) is None
    assert parse_webhook(ev({"result": "fake", "score": "nan", "call_session_id": "s"})) is None
    assert parse_webhook(ev(None)) is None
    assert parse_webhook({"data": None}) is None
    assert parse_webhook([]) is None
    hi = parse_webhook(ev({"result": "fake", "score": 1.7, "call_session_id": "s"}))
    assert hi.ai_score == 1.0 and hi.call_id == "c_s"

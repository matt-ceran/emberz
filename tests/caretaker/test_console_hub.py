import asyncio
import json
import threading

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from contracts.models import ConsoleEvent
from line.caretaker import CaretakerFlow
from line.console_hub import ConsoleHub, add_cors, console_router
from line.session import CallSession
from line.telnyx_client import FakeTelnyx


def e(call_id, type_, at, **data):
    return ConsoleEvent(type=type_, call_id=call_id, at_ms=at, data=data)


def build(caretaker_session=None, open_access=True):
    hub = ConsoleHub()
    sessions = {}
    if caretaker_session:
        sessions[caretaker_session.call_id] = caretaker_session
    tx = FakeTelnyx()
    flow = CaretakerFlow(tx, hub, sessions.get)
    app = FastAPI()
    add_cors(app)
    app.include_router(console_router(hub, flow, open_access=open_access))
    return hub, tx, TestClient(app)


def test_ws_replays_then_streams_live():
    hub, _, client = build()
    hub.publish(e("c_1", "call_started", 0, caller_number="+1", guarded_number="+2", profile="elder", caller_known=False))
    hub.publish(e("c_2", "call_started", 0, caller_number="+3", guarded_number="+2", profile="kid", caller_known=False))
    hub.emit("c_1", "coach", 900, text="Banks never ask for codes.", turn=5)

    with client.websocket_connect("/console/ws") as ws:
        got = [json.loads(ws.receive_text()) for _ in range(3)]
        assert [(g["call_id"], g["type"]) for g in got] == [("c_1", "call_started"), ("c_2", "call_started"), ("c_1", "coach")]
        hub.publish(e("c_2", "alert", 1500, level="alert", category="disclosure.home_alone", caretaker_note="x", turn=3))
        live = json.loads(ws.receive_text())
        assert live["type"] == "alert" and live["call_id"] == "c_2"
        ConsoleEvent.model_validate(live)


def test_ws_filters_by_call_id_and_receives_from_other_thread():
    hub, _, client = build()
    hub.publish(e("c_1", "call_started", 0))
    hub.publish(e("c_2", "call_started", 0))
    with client.websocket_connect("/console/ws?call_id=c_2") as ws:
        assert json.loads(ws.receive_text())["call_id"] == "c_2"
        # publish from a worker thread (e.g. the ASR thread); c_1 must be filtered out
        t = threading.Thread(target=lambda: (hub.publish(e("c_1", "turn", 10)), hub.publish(e("c_2", "turn", 20))))
        t.start()
        t.join()
        msg = json.loads(ws.receive_text())
        assert (msg["call_id"], msg["type"], msg["at_ms"]) == ("c_2", "turn", 20)


def test_http_calls_and_events():
    hub, _, client = build()
    hub.publish(e("c_1", "call_started", 0, caller_number="+1", guarded_number="+2", profile="elder", caller_known=False))
    hub.publish(e("c_1", "risk", 10, turn=1, response={"risk": "alert"}, latency_ms=300))
    hub.publish(e("c_1", "alert", 11, level="alert", category="scam.prize", caretaker_note="n", turn=1))
    hub.publish(ConsoleEvent(type="voice", call_id="c_1", at_ms=12,
                             data={"call_id": "c_1", "at_ms": 12, "ai_score": 0.9, "model": "w2v-head-v1"}))
    rows = client.get("/console/calls").json()
    assert rows[0]["call_id"] == "c_1" and rows[0]["max_risk"] == "alert" and rows[0]["alerts"] == 1
    assert rows[0]["ai_score"] == 0.9 and rows[0]["started"]["profile"] == "elder"
    evs = client.get("/console/calls/c_1/events").json()
    assert [x["type"] for x in evs] == ["call_started", "risk", "alert", "voice"]
    assert client.get("/console/calls/nope/events").status_code == 404
    # CORS preflight from another origin (Matt's dev page)
    r = client.options("/console/calls", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"})
    assert r.status_code == 200 and r.headers["access-control-allow-origin"] in ("*", "http://localhost:5173")


def test_barge_endpoint():
    s = CallSession(call_id="c_1", guarded_number="+2", caller_number="+1", caretaker_number="+9",
                    conference_id="conf", c_ccid="c_leg")
    hub, tx, client = build(s)
    assert client.post("/console/calls/c_1/barge").status_code == 409  # not monitoring yet
    assert client.post("/console/calls/zzz/barge").status_code == 404
    s.caretaker_state = "monitoring"
    r = client.post("/console/calls/c_1/barge")
    assert r.status_code == 200 and r.json()["caretaker_state"] == "barged"
    assert tx.log[-1][2] == "update" and tx.log[-1][3]["supervisor_role"] == "barge"
    assert hub.events("c_1")[-1].data == {"mode": "barge"}
    assert client.post("/console/calls/c_1/barge").status_code == 200  # idempotent


def test_token_required_when_set(monkeypatch):
    monkeypatch.setenv("CONSOLE_TOKEN", "s3cret")
    hub, _, client = build(open_access=False)
    hub.publish(e("c_1", "call_started", 0))
    assert client.get("/console/calls").status_code == 401
    assert client.get("/console/calls?token=s3cret").status_code == 200
    assert client.get("/console/calls", headers={"X-Console-Token": "s3cret"}).status_code == 200
    assert client.get("/console/calls?token=wrong").status_code == 401
    with pytest.raises(Exception):
        with client.websocket_connect("/console/ws") as ws:
            ws.receive_text()
    with client.websocket_connect("/console/ws?token=s3cret") as ws:
        assert json.loads(ws.receive_text())["type"] == "call_started"


@pytest.mark.asyncio
async def test_subscribe_no_gap_no_duplicate_and_slow_consumer_does_not_block():
    hub = ConsoleHub()
    hub.publish(e("c_1", "turn", 1))
    sub = hub.subscribe("c_1")
    hub.publish(e("c_1", "turn", 2))
    assert [x.at_ms for x in sub.replay] == [1]
    assert (await asyncio.wait_for(sub.get(), 1)).at_ms == 2
    # 3000 publishes with nobody reading: publish never blocks, the queue keeps the newest
    for i in range(3000):
        hub.publish(e("c_1", "turn", 10 + i))
    first = await sub.get()
    assert first.at_ms > 10
    sub.close()
    hub.publish(e("c_1", "turn", 99999))  # no subscribers left: still fine
    with pytest.raises(Exception):
        hub.publish({"type": "nonsense", "call_id": "c", "at_ms": 0})


def test_calls_row_prefers_policy_display_risk():
    hub, _, client = build()
    hub.publish(e("c_1", "risk", 10, turn=1, response={"risk": "alert"}, latency_ms=300, display_risk="none"))
    hub.publish(e("c_2", "risk", 10, turn=1, response={"risk": "watch"}, latency_ms=300))
    rows = {r["call_id"]: r for r in client.get("/console/calls").json()}
    assert rows["c_1"]["max_risk"] == "none" and rows["c_2"]["max_risk"] == "watch"


def test_barge_endpoint_telnyx_error_is_502_and_retryable():
    s = CallSession(call_id="c_1", guarded_number="+2", caller_number="+1", caretaker_number="+9",
                    conference_id="conf", c_ccid="c_leg", caretaker_state="monitoring")
    hub, tx, client = build(s)

    async def boom(*a, **k):
        raise RuntimeError("telnyx 422")

    tx.conference_action = boom
    assert client.post("/console/calls/c_1/barge").status_code == 502
    assert s.caretaker_state == "monitoring"


def test_live_router_fails_closed_without_console_token(monkeypatch):
    monkeypatch.delenv("CONSOLE_TOKEN", raising=False)
    hub, _, client = build(open_access=False)
    hub.publish(e("c_1", "call_started", 0))
    assert client.get("/console/calls").status_code == 503
    assert client.get("/console/calls?token=").status_code == 503
    assert client.post("/console/calls/c_1/barge").status_code == 503
    with pytest.raises(Exception):
        with client.websocket_connect("/console/ws") as ws:
            ws.receive_text()

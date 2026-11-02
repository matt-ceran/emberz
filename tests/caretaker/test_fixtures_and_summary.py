import json

import httpx
import pytest
from fastapi.testclient import TestClient

from console.fixture_player import FIXTURES, check_payload, create_app, load_scene, scenes
from contracts.models import ConsoleEvent, Turn
from line.console_hub import ConsoleHub
from line.session import CallSession
from line.summary import SummaryError, build_messages, summarize


def test_every_fixture_line_is_a_valid_console_event():
    files = sorted(FIXTURES.glob("*.jsonl"))
    assert {p.stem for p in files} >= {"scene1_bank_scam", "scene3_stranger_kid", "scene4_grandpa_allowed", "scene5_pharmacy_bot"}
    for p in files:
        lines = [l for l in p.read_text().splitlines() if l.strip()]
        events = [ConsoleEvent.model_validate(json.loads(l)) for l in lines]
        for ev in events:
            check_payload(ev)
        assert events[0].type == "call_started" and len({ev.call_id for ev in events}) == 1
        assert [ev.at_ms for ev in events] == sorted(ev.at_ms for ev in events), p.name
        turns = [ev.data["turn"] for ev in events if ev.type == "turn"]
        assert turns == list(range(1, len(turns) + 1)), p.name
        assert "call_ended" in [ev.type for ev in events]


def test_scene_stories():
    types = {name: [e.type for e in load_scene(p)] for name, p in scenes().items()}
    s1 = load_scene(scenes()["scene1_bank_scam"])
    assert {"voice", "coach", "alert", "caretaker_joined", "summary"} <= set(types["scene1_bank_scam"])
    assert next(e for e in s1 if e.type == "voice").data["ai_score"] >= 0.8
    risks = [e.data["response"]["risk"] for e in s1 if e.type == "risk"]
    assert risks.index("watch") < risks.index("urgent")  # risk rises
    assert "alert" in types["scene3_stranger_kid"] and "caretaker_joined" in types["scene3_stranger_kid"]
    assert not {"alert", "coach"} & set(types["scene4_grandpa_allowed"])
    s4 = load_scene(scenes()["scene4_grandpa_allowed"])
    assert s4[0].data["caller_known"] is True
    assert any("street" in e.data["text"] for e in s4 if e.type == "turn")
    s5 = load_scene(scenes()["scene5_pharmacy_bot"])
    assert not {"alert", "coach"} & set(types["scene5_pharmacy_bot"])
    assert next(e for e in s5 if e.type == "voice").data["ai_score"] >= 0.8
    # live, every call with speech gets a summary; the fixtures match that
    for name in scenes():
        assert types[name][-1] == "summary", name


def test_fixture_player_end_to_end_with_barge():
    app = create_app(speed=50, autoplay=False)
    with TestClient(app) as client:
        assert set(client.get("/fixtures").json()) == set(scenes())
        r = client.post("/fixtures/play/scene3_stranger_kid")
        cid = r.json()["call_id"]
        assert cid == "fx_scene3_stranger_kid"
        with client.websocket_connect(f"/console/ws?call_id={cid}") as ws:
            seen = []
            while True:
                msg = json.loads(ws.receive_text())
                seen.append(msg["type"])
                if msg["type"] == "caretaker_joined":
                    break
            b = client.post(f"/console/calls/{cid}/barge")
            assert b.status_code == 200, b.text
            while True:
                msg = json.loads(ws.receive_text())
                seen.append((msg["type"], msg["data"].get("mode")))
                if msg["type"] == "summary":
                    break
        assert ("caretaker_joined", "barge") in seen
        # a second replay gets a fresh call_id
        assert client.post("/fixtures/play/scene3_stranger_kid").json()["call_id"] == "fx_scene3_stranger_kid-2"
        assert client.post("/fixtures/play/nope").status_code == 404


def _session():
    s = CallSession(call_id="c_9", guarded_number="+2", caller_number="+1", profile="elder", caretaker_state="barged")
    s.turns = [
        Turn(call_id="c_9", turn=1, speaker="caller", text="This is your bank.", t_start_ms=0, t_end_ms=900),
        Turn(call_id="c_9", turn=2, speaker="protected", text="Hello?", t_start_ms=1000, t_end_ms=1500),
    ]
    return s


def test_build_messages_mentions_facts():
    msgs = build_messages(_session(), [{"level": "urgent", "category": "scam.prize", "caretaker_note": "asked for code", "turn": 1}])
    user = msgs[1]["content"]
    assert "Caller: This is your bank." in user and "Grandma/Grandpa: Hello?" in user
    assert "asked for code" in user and "joined the conversation" in user
    assert "+1" not in user  # the raw phone number never goes to the LLM


def test_build_messages_caretaker_mode_overrides_idle_state():
    s = _session()
    s.caretaker_state = "idle"  # the caretaker listened, then hung up
    assert "was not called" in build_messages(s, [])[1]["content"]
    assert "listened in" in build_messages(s, [], caretaker_mode="monitor")[1]["content"]


@pytest.mark.asyncio
async def test_summarize_publishes_and_sends_expected_body(monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setenv("SUMMARY_MODEL", "accounts/fireworks/models/test-model")
    seen = {}

    def handler(req: httpx.Request):
        seen["body"] = json.loads(req.content)
        seen["auth"] = req.headers["authorization"]
        return httpx.Response(200, json={"choices": [{"message": {"content": "<think>x</think> someone called.  Twice."}, "finish_reason": "stop"}]})

    hub = ConsoleHub()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        text = await summarize(_session(), [], hub=hub, http=http)
    assert text == "Someone called. Twice."
    assert seen["body"]["model"] == "accounts/fireworks/models/test-model"
    assert seen["body"]["reasoning_effort"] == "none" and seen["auth"] == "Bearer test-key"
    ev = hub.events("c_9")[-1]
    assert ev.type == "summary" and ev.data == {"text": text}


@pytest.mark.asyncio
async def test_summarize_raises_and_publishes_nothing_on_error(monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    hub = ConsoleHub()
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(503, text="waking"))) as http:
        with pytest.raises(SummaryError):
            await summarize(_session(), [], hub=hub, http=http)
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"error": "x"}))) as http:
        with pytest.raises(SummaryError):
            await summarize(_session(), [], hub=hub, http=http)
    assert hub.events("c_9") is None

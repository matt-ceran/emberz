"""Opt-in live test: the bank-scam fixture through CallBrain against the Fireworks stand-in.

RUN_LIVE=1 .venv/bin/python -m pytest tests/brain/test_brain_live.py -s
Costs at most 5 small requests (~1k tokens each) on Numan's Fireworks key.
"""
from __future__ import annotations

import asyncio
import os

import pytest
from brain_fakes import FakeCaretaker, FakeHub, FakeTranscriber, make_session

from contracts.models import VoiceSignal
from line.brain import CallBrain
from line.guardian_client import GuardianClient
from line.telnyx_client import FakeTelnyx

pytestmark = pytest.mark.skipif(os.environ.get("RUN_LIVE") != "1", reason="set RUN_LIVE=1 to call Fireworks")


@pytest.mark.asyncio
async def test_live_standin_bank_scam(fixture_turns):
    client = GuardianClient(backend="standin")
    s = make_session()
    tel, hub, ct = FakeTelnyx(), FakeHub(), FakeCaretaker()
    brain = CallBrain(s, tel, hub, ct, transcriber_factory=FakeTranscriber, score_fn=client.score)
    brain.b_joined = True
    brain.on_voice_signal(VoiceSignal(call_id=s.call_id, at_ms=2400, ai_score=0.91, model="w2v-head-v1"))
    try:
        for t in fixture_turns:
            brain.on_turn(t)
            await brain.scorer.wait_idle()  # one score per turn, like a real call's pacing
            await asyncio.sleep(0.05)
        await brain.drain()
    finally:
        await client.aclose()

    for e in hub.of("risk"):
        r = e.data["response"]
        print(f"turn {e.data['turn']}: {e.data['latency_ms']} ms model={r['risk']}/{r['category']}/sev{r['severity']}/"
              f"{r['stage']} display={e.data['display_risk']} reasons={e.data['reasons']}")
    print("coach:", [e.data for e in hub.of("coach")], "latency:", brain.coach_latencies)
    print("caretaker:", ct.alerts)
    speaks = [b for k, _, a, b in tel.log if a == "speak"]
    assert all(b["call_control_ids"] == ["B"] for b in speaks)
    alerts = [(e.data["category"], e.data["level"]) for e in hub.of("alert")]
    assert len(alerts) == len(set(alerts)) == len(ct.alerts)  # never the same alert twice
    assert hub.of("risk")
    # the final "read me the code" turn should be caught by any reasonable model
    assert any(e.data["display_risk"] in ("alert", "urgent") for e in hub.of("risk")), "stand-in missed the ask"

"""CallBrain wired into the real line/call_flow.CallFlow via make_brain_factory (FakeTelnyx, no network).

Checks the event wiring the unit tests fake: B's conference join reaches the brain and turns the
coach on, the coach goes to B only, the Telnyx deepfake result gives exactly one "voice" event,
and hang-up closes the transcriber.
"""
from __future__ import annotations

import asyncio

import pytest
from brain_fakes import FakeCaretaker, FakeHub, FakeTranscriber

from line.brain import CallBrain, make_brain_factory
from line.call_flow import CallFlow, encode_client_state
from line.config import GuardedLine, Settings
from line.guardian_client import GuardianClient
from line.telnyx_client import FakeTelnyx

GUARDED, PROTECTED, CARETAKER, CALLER = "+15555550100", "+15555550101", "+15555550102", "+15555550199"
A, SID = "v3:A-leg", "sess-9"
CID = "c_" + SID


def leg(role, ccid, **extra):
    return {"call_control_id": ccid, "call_session_id": SID, "client_state": encode_client_state(role, CID)} | extra


@pytest.mark.asyncio
async def test_call_flow_to_brain_coach_after_b_joins(fixture_turns):
    tel, hub, ct = FakeTelnyx(), FakeHub(), FakeCaretaker()
    FakeTranscriber.instances.clear()
    flow = CallFlow(
        Settings(telnyx_api_key="test", public_base_url="https://x.trycloudflare.com", skip_verify=True),
        {GUARDED: GuardedLine(profile="elder", rules=["Never share codes"], protected_number=PROTECTED,
                              caretaker_number=CARETAKER)},
        tel,
        make_brain=make_brain_factory(ct, transcriber_factory=FakeTranscriber,
                                      score_fn=GuardianClient(backend="stub").score),
        hub=hub,
    )
    await flow.handle("call.initiated", {"call_control_id": A, "call_session_id": SID, "from": CALLER,
                                         "to": GUARDED, "direction": "incoming", "state": "parked"})
    brain: CallBrain = flow.brains[CID]
    s = flow.sessions[CID]
    await flow.handle("call.answered", leg("A", A))
    await brain.on_media("outbound", b"\x00" * 160, 20)  # the notice: fed as silence
    tr = FakeTranscriber.instances[-1]
    await flow.handle("call.deepfake_detection.result", leg("A", A, result="fake", score=0.91, consistency=90))
    assert s.ai_voice.score == 0.91 and len(hub.of("voice")) == 1

    # The caller talks before B is in the conference: alert, but no speak yet.
    for t in fixture_turns[:3]:
        brain.on_turn(t.model_copy(update={"call_id": CID}))
    await asyncio.sleep(0.01)
    await brain.drain()
    assert [b for k, _, a, b in tel.log if a == "speak" and k == "conference"] == []
    assert len(ct.alerts) == 1

    await flow.handle("call.speak.ended", leg("A", A, status="completed"))
    await flow.handle("conference.created", leg("A", A, conference_id=s.conference_id))
    await flow.handle("call.answered", leg("B", s.b_ccid))
    await flow.handle("conference.participant.joined", leg("A", s.b_ccid, conference_id=s.conference_id))
    assert brain.b_joined

    for t in fixture_turns[3:]:
        brain.on_turn(t.model_copy(update={"call_id": CID}))
    await asyncio.sleep(0.01)
    await brain.drain()
    speaks = [(cid, b) for k, cid, a, b in tel.log if a == "speak" and k == "conference"]
    assert len(speaks) == 1 and speaks[0][0] == s.conference_id and speaks[0][1]["call_control_ids"] == [s.b_ccid]
    assert len(ct.alerts) == 1  # same category, no second alert
    assert tr.fed[0] == ("outbound", 160, None)

    await flow.handle("call.hangup", leg("A", A, hangup_cause="normal_clearing"))
    assert tr.closed and s.ended and CID not in flow.brains

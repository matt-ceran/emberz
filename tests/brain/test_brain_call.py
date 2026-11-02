"""CallBrain end to end with FakeTelnyx, the stub guardian and fake hub/caretaker/transcriber."""
from __future__ import annotations

import asyncio
import threading

import pytest
from brain_fakes import FakeCaretaker, FakeHub, FakeTranscriber, make_session

from contracts.models import Turn, VoiceSignal
from line.brain import CallBrain
from line.guardian_client import GuardianClient
from line.telnyx_client import FakeTelnyx

stub = GuardianClient(backend="stub")


def brain_for(session, score_fn=None, joined=True):
    tel, hub, ct = FakeTelnyx(), FakeHub(), FakeCaretaker()
    brain = CallBrain(session, tel, hub, ct, transcriber_factory=FakeTranscriber, score_fn=score_fn or stub.score)
    brain.b_joined = joined  # as if call_flow forwarded conference.participant.joined for B
    return brain, tel, hub, ct


async def play(brain, turns, gap_s=0.01):
    for t in turns:
        brain.on_turn(t)
        await asyncio.sleep(gap_s)
    await brain.drain()


def speaks(tel):
    return [body for kind, cid, action, body in tel.log if kind == "conference" and action == "speak"]


@pytest.mark.asyncio
async def test_bank_scam_coaches_b_only_and_alerts_once(fixture_turns):
    s = make_session()
    brain, tel, hub, ct = brain_for(s)
    brain.on_voice_signal(VoiceSignal(call_id=s.call_id, at_ms=2400, ai_score=0.91, model="w2v-head-v1"))
    await play(brain, fixture_turns)

    sp = speaks(tel)
    assert len(sp) == 1, tel.log
    assert sp[0]["call_control_ids"] == ["B"]
    assert sp[0]["payload"] and sp[0]["voice"]
    assert all(kind == "conference" for kind, *_ in tel.log)  # nothing sent to leg A
    assert len(ct.alerts) == 1 and ct.alerts[0][1] == "urgent"  # AI voice + unknown + ask escalated
    assert len(hub.of("alert")) == 1
    assert [e.data["turn"] for e in hub.of("turn")] == [1, 2, 3, 4, 5]
    assert len(hub.of("risk")) == 5 and len(hub.of("coach")) == 1 and len(hub.of("voice")) == 1
    risk3 = next(e for e in hub.of("risk") if e.data["turn"] == 3)
    assert risk3.data["display_risk"] == "urgent" and risk3.data["response"]["risk"] == "alert"
    assert brain.coach_latencies and brain.coach_latencies[0]["turn"] == 3
    assert brain.coach_latencies[0]["sent_ms"] < 1500
    assert len(s.turns) == 5


@pytest.mark.asyncio
async def test_bank_scam_without_ai_voice_still_one_alert(fixture_turns):
    s = make_session()
    brain, tel, hub, ct = brain_for(s)
    await play(brain, fixture_turns)
    assert len(ct.alerts) == 1 and ct.alerts[0][1] == "alert"
    assert len(speaks(tel)) == 1


@pytest.mark.asyncio
async def test_allowed_contact_is_silent(fixture_turns):
    s = make_session(caller_known=True, caller_number="+15550100")
    brain, tel, hub, ct = brain_for(s)
    await play(brain, fixture_turns)
    assert speaks(tel) == [] and ct.alerts == [] and hub.of("alert") == [] and hub.of("coach") == []
    assert {e.data["display_risk"] for e in hub.of("risk")} <= {"none", "watch"}


PHARMACY = [
    ("caller", "Hi, this is an automated reminder from Elm Pharmacy."),
    ("protected", "Oh, hello."),
    ("caller", "Your prescription for Mary is ready for pickup after 2 pm."),
    ("protected", "Thank you."),
    ("caller", "Press 1 to hear our hours, or just hang up. Goodbye."),
]


def as_turns(call_id, script):
    return [
        Turn(call_id=call_id, turn=i + 1, speaker=sp, text=tx, t_start_ms=i * 2000, t_end_ms=i * 2000 + 1500)
        for i, (sp, tx) in enumerate(script)
    ]


@pytest.mark.asyncio
async def test_pharmacy_bot_ai_voice_is_watch_only():
    s = make_session(call_id="c_pharm")
    brain, tel, hub, ct = brain_for(s)
    brain.on_voice_signal(VoiceSignal(call_id=s.call_id, at_ms=2000, ai_score=0.95, model="w2v-head-v1"))
    await play(brain, as_turns(s.call_id, PHARMACY))
    risks = hub.of("risk")
    assert risks and all(e.data["display_risk"] == "watch" for e in risks)
    assert ct.alerts == [] and hub.of("alert") == [] and speaks(tel) == []


@pytest.mark.asyncio
async def test_latest_only_never_queues_backlog(fixture_turns):
    s = make_session()
    seen = []

    async def slow(req):
        seen.append(req.turns[-1].turn)
        await asyncio.sleep(0.05)
        return await stub.score(req)

    brain, tel, hub, ct = brain_for(s, score_fn=slow)
    brain.on_turn(fixture_turns[0])
    await asyncio.sleep(0.01)  # turn 1 is now being scored
    for t in fixture_turns[1:]:  # turns 2-5 arrive while it runs: no backlog, one follow-up
        brain.on_turn(t)
    await brain.drain()
    assert seen == [1, 5]
    assert brain.scorer.scored_turns == [1, 5]


@pytest.mark.asyncio
async def test_media_goes_to_transcriber_and_threaded_turns_are_scored(fixture_turns):
    FakeTranscriber.instances.clear()
    s = make_session()
    brain, tel, hub, ct = brain_for(s)
    await brain.on_media("inbound", b"\xff" * 160, 20)
    await brain.on_media("outbound", b"\xff" * 160, 20)
    tr = FakeTranscriber.instances[-1]
    assert tr.call_id == s.call_id and tr.fed == [("inbound", 160, 20), ("outbound", 160, 20)]

    th = threading.Thread(target=lambda: [brain.on_turn(t) for t in fixture_turns])  # worker-thread callback
    th.start()
    th.join()
    for _ in range(20):
        await asyncio.sleep(0.01)
    await brain.drain()
    assert len(s.turns) == 5 and len(ct.alerts) == 1

    await brain.on_hangup()
    assert tr.closed and s.ended
    await brain.on_media("inbound", b"\xff" * 160, 40)
    assert len(tr.fed) == 2


@pytest.mark.asyncio
async def test_turns_after_hangup_kept_not_scored(fixture_turns):
    s = make_session()
    brain, tel, hub, ct = brain_for(s)
    await brain.on_hangup()
    await play(brain, fixture_turns)
    assert len(hub.of("turn")) == 5 and hub.of("risk") == [] and ct.alerts == []


@pytest.mark.asyncio
async def test_coach_without_conference_is_not_sent(fixture_turns):
    s = make_session(conference_id=None)
    brain, tel, hub, ct = brain_for(s)
    await play(brain, fixture_turns)
    assert speaks(tel) == [] and hub.of("coach") == []
    assert any("coach_unavailable" in d.reasons for _, d in brain.decisions)
    assert len(ct.alerts) == 1  # the caretaker is still alerted


@pytest.mark.asyncio
async def test_coach_waits_for_b_join_and_is_not_muted_by_it(fixture_turns):
    # Alert before B joins: no speak, and the unplayed tip must not start the cooldown or
    # use up the line, so the tip plays as soon as B is in the conference.
    s = make_session()
    brain, tel, hub, ct = brain_for(s, joined=False)
    await play(brain, fixture_turns[:3])  # turn 3 is the alert
    assert speaks(tel) == [] and s.policy_state.last_coach_ms is None and not s.policy_state.coach_lines
    await brain.on_call_event("conference.participant.joined", {"call_control_id": "B", "_role": "B"})
    assert brain.b_joined
    await play(brain, fixture_turns[3:])
    sp = speaks(tel)
    assert len(sp) == 1 and sp[0]["call_control_ids"] == ["B"]
    await brain.on_call_event("conference.participant.left", {"call_control_id": "B", "_role": "B"})
    assert not brain.b_joined
    await brain.on_call_event("emberz.protected_joined", {"at_ms": 9000, "conference_id": "conf_1"})
    assert brain.b_joined


@pytest.mark.asyncio
async def test_protected_track_is_silence_until_b_joins():
    got = []

    class Rec(FakeTranscriber):
        def feed(self, track, payload, ts_ms=None):
            got.append((track, payload))

    s = make_session()
    brain = CallBrain(s, FakeTelnyx(), FakeHub(), FakeCaretaker(), transcriber_factory=Rec, score_fn=stub.score)
    notice = bytes(range(160))
    await brain.on_media("outbound", notice, 20)  # the consent notice playing to A
    await brain.on_media("inbound", notice, 20)   # the caller is always transcribed
    await brain.on_call_event("emberz.protected_joined", {"at_ms": 40, "conference_id": "conf_1"})
    await brain.on_media("outbound", notice, 40)
    assert got == [("outbound", b"\xff" * 160), ("inbound", notice), ("outbound", notice)]


@pytest.mark.asyncio
async def test_telnyx_deepfake_event_sets_ai_voice():
    s = make_session()
    brain, tel, hub, ct = brain_for(s)
    await brain.on_call_event("call.deepfake_detection.result", {"result": "fake", "score": 0.87, "consistency": 94.5})
    assert s.ai_voice.score == 0.87 and s.ai_voice.source == "telnyx"
    await brain.on_call_event("call.deepfake_detection.result", {"result": "silence_timeout"})
    await brain.on_call_event("call.deepfake_detection.result", {"result": "fake", "score": "garbage"})
    assert s.ai_voice.score == 0.87
    assert hub.of("voice") == []  # call_flow publishes the Telnyx "voice" event; no duplicate here


@pytest.mark.asyncio
async def test_lower_score_from_other_source_does_not_override():
    s = make_session()
    brain, *_ = brain_for(s)
    brain.on_voice_signal(VoiceSignal(call_id=s.call_id, at_ms=1, ai_score=0.9, model="w2v-head-v1"))
    brain.on_voice_signal(VoiceSignal(call_id=s.call_id, at_ms=2, ai_score=0.2, model="timing"))
    assert s.ai_voice.score == 0.9 and s.ai_voice.source == "detector"
    brain.on_voice_signal(VoiceSignal(call_id=s.call_id, at_ms=3, ai_score=0.5, model="w2v-head-v1"))
    assert s.ai_voice.score == 0.5  # the same detector updating itself wins


@pytest.mark.asyncio
async def test_ai_voice_is_max_of_newest_per_source():
    s = make_session()
    brain, *_ = brain_for(s)
    brain.on_voice_signal(VoiceSignal(call_id=s.call_id, at_ms=1, ai_score=0.9, model="w2v-head-v1"))
    await brain.on_call_event("call.deepfake_detection.result", {"result": "fake", "score": 0.8, "_role": "A"})
    brain.on_voice_signal(VoiceSignal(call_id=s.call_id, at_ms=3, ai_score=0.2, model="w2v-head-v1"))
    assert s.ai_voice.score == 0.8 and s.ai_voice.source == "telnyx"  # Telnyx's 0.8 is not lost


@pytest.mark.asyncio
async def test_voice_signal_from_worker_thread_lands_on_loop():
    s = make_session()
    brain, tel, hub, ct = brain_for(s)
    th = threading.Thread(target=brain.on_voice_signal,
                          args=(VoiceSignal(call_id=s.call_id, at_ms=1, ai_score=0.9, model="w2v-head-v1"),))
    th.start()
    th.join()
    await asyncio.sleep(0.01)
    assert s.ai_voice.score == 0.9 and len(hub.of("voice")) == 1


@pytest.mark.asyncio
async def test_duplicate_and_empty_turns_ignored(fixture_turns):
    s = make_session()
    brain, tel, hub, ct = brain_for(s)
    blank = Turn(call_id=s.call_id, turn=6, speaker="caller", text="  ", t_start_ms=19000, t_end_ms=19500)
    await play(brain, fixture_turns + [fixture_turns[2], blank])
    assert len(s.turns) == 5 and len(hub.of("turn")) == 5 and len(ct.alerts) == 1


@pytest.mark.asyncio
async def test_slow_speak_does_not_hold_up_scoring(fixture_turns):
    class SlowTelnyx(FakeTelnyx):
        async def conference_action(self, *a, **k):
            await asyncio.sleep(0.5)
            return await super().conference_action(*a, **k)

    s = make_session()
    hub, ct = FakeHub(), FakeCaretaker()
    brain = CallBrain(s, SlowTelnyx(), hub, ct, transcriber_factory=FakeTranscriber, score_fn=stub.score)
    brain.b_joined = True
    for t in fixture_turns:
        brain.on_turn(t)
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.05)
    assert len(hub.of("risk")) == 5 and hub.of("coach") == []  # all scored while the speak is in flight
    await brain.drain()
    assert hub.of("coach")[0].data["sent"] is True


@pytest.mark.asyncio
async def test_transcriber_is_built_off_the_event_loop():
    built_on = []

    def factory(call_id, on_turn, loop, on_partial=None):
        built_on.append(threading.current_thread() is threading.main_thread())
        return FakeTranscriber(call_id, on_turn, loop, on_partial)

    s = make_session()
    brain = CallBrain(s, FakeTelnyx(), FakeHub(), FakeCaretaker(), transcriber_factory=factory, score_fn=stub.score)
    await asyncio.gather(*(brain.on_media("inbound", b"\xff" * 160, i * 20) for i in range(3)))
    assert built_on == [False]  # built once, in a worker thread
    assert len(brain.transcriber.fed) == 3


@pytest.mark.asyncio
async def test_failing_score_fn_does_not_kill_scoring(fixture_turns):
    calls = []

    async def flaky(req):
        calls.append(req.turns[-1].turn)
        if len(calls) == 1:
            raise RuntimeError("boom")
        return await stub.score(req)

    s = make_session()
    brain, tel, hub, ct = brain_for(s, score_fn=flaky)
    await play(brain, fixture_turns)
    assert calls[0] == 1 and len(calls) == 5 and len(ct.alerts) == 1


@pytest.mark.asyncio
async def test_hangup_mid_score_then_drain_does_not_raise(fixture_turns):
    async def slow(req):
        await asyncio.sleep(1)
        return await stub.score(req)

    s = make_session()
    brain, tel, hub, ct = brain_for(s, score_fn=slow)
    brain.on_turn(fixture_turns[0])
    await asyncio.sleep(0.01)
    await brain.on_hangup()
    await brain.drain()
    assert hub.of("risk") == []


@pytest.mark.asyncio
async def test_telnyx_speak_failure_does_not_break_call(fixture_turns):
    class Boom(FakeTelnyx):
        async def conference_action(self, *a, **k):
            raise RuntimeError("422")

    s = make_session()
    hub, ct = FakeHub(), FakeCaretaker()
    brain = CallBrain(s, Boom(), hub, ct, transcriber_factory=FakeTranscriber, score_fn=stub.score)
    brain.b_joined = True
    await play(brain, fixture_turns)
    assert len(ct.alerts) == 1 and hub.of("coach")[0].data["sent"] is False


@pytest.mark.asyncio
async def test_with_real_console_hub_and_caretaker_flow(fixture_turns):
    hub_mod = pytest.importorskip("line.console_hub")
    ct_mod = pytest.importorskip("line.caretaker")
    s = make_session()
    tel, hub = FakeTelnyx(), hub_mod.ConsoleHub()
    caretaker = ct_mod.CaretakerFlow(tel, hub, lambda cid: s if cid == s.call_id else None)
    brain = CallBrain(s, tel, hub, caretaker, transcriber_factory=FakeTranscriber, score_fn=stub.score)
    await brain.on_call_event("conference.participant.joined", {"call_control_id": "B", "_role": "B"})
    await play(brain, fixture_turns)
    types = [e.type for e in hub.events(s.call_id)]
    assert types.count("turn") == 5 and types.count("alert") == 1 and types.count("coach") == 1
    assert [k for k, *_ in tel.log].count("dial") == 1  # caretaker dialed once
    assert speaks(tel)[0]["call_control_ids"] == ["B"]


@pytest.mark.asyncio
async def test_make_brain_factory_and_deepfake_role_filter():
    from line.brain import make_brain_factory

    s = make_session()
    ct = FakeCaretaker()
    brain = make_brain_factory(ct, transcriber_factory=FakeTranscriber, score_fn=stub.score)(s, FakeTelnyx(), FakeHub())
    assert isinstance(brain, CallBrain) and brain.caretaker is ct
    await brain.on_call_event("call.deepfake_detection.result", {"result": "fake", "score": 0.9, "_role": "B"})
    assert s.ai_voice is None
    await brain.on_call_event("call.deepfake_detection.result", {"result": "fake", "score": 0.9, "_role": "A"})
    assert s.ai_voice.score == 0.9


@pytest.mark.asyncio
async def test_asr_start_failure_is_logged_once_and_call_survives(fixture_turns, caplog):
    built = []

    def broken(*a, **k):
        built.append(1)
        raise FileNotFoundError("model missing")

    s = make_session()
    brain = CallBrain(s, FakeTelnyx(), FakeHub(), FakeCaretaker(), transcriber_factory=broken, score_fn=stub.score)
    for i in range(3):
        await brain.on_media("inbound", b"\xff" * 160, i * 20)  # must not raise into the media socket
    assert built == [1] and sum("ASR could not start" in r.message for r in caplog.records) == 1
    await brain.on_hangup()


@pytest.mark.asyncio
async def test_hangup_while_asr_loads_closes_it():
    gate = threading.Event()

    def slow_factory(call_id, on_turn, loop, on_partial=None):
        gate.wait(2)
        return FakeTranscriber(call_id, on_turn, loop, on_partial)

    s = make_session()
    brain = CallBrain(s, FakeTelnyx(), FakeHub(), FakeCaretaker(), transcriber_factory=slow_factory, score_fn=stub.score)
    first = asyncio.create_task(brain.on_media("inbound", b"\xff" * 160, 20))
    await asyncio.sleep(0.05)
    await brain.on_hangup()  # nothing to close yet
    gate.set()
    await first
    assert brain.transcriber.closed and brain.transcriber.fed == []

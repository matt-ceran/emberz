"""The L1 call skeleton against FakeTelnyx: every Telnyx command the flow sends, in order."""
from callflow_helpers import (A_CCID, CALL_ID, CALLER, CARETAKER, GRANDPA, GUARDED, PROTECTED,
                              Line, decode_client_state)
from line.call_flow import ANON_DISPLAY_NAME, encode_client_state


def _answer_and_notice(line: Line):
    line.inbound()
    line.post("call.answered", line.leg("A", A_CCID))


def test_full_inbound_sequence(line):
    # 1. Inbound call -> answer with deepfake detection, tagged as leg A.
    assert line.inbound().status_code == 200
    (_, ccid, action, body), = line.telnyx.log
    assert (ccid, action) == (A_CCID, "answer")
    assert body["deepfake_detection"]["enabled"] is True
    assert decode_client_state(body["client_state"]) == {"role": "A", "call_id": CALL_ID}
    assert body["command_id"] == f"{CALL_ID}:answer"
    s = line.flow.sessions[CALL_ID]
    assert (s.caller_number, s.guarded_number, s.profile, s.caller_known) == (CALLER, GUARDED, "elder", False)
    assert line.hub.types() == ["call_started"]

    # 2. A answered -> stream both tracks (PCMU) to our WS, speak the notice to A only, dial B.
    line.post("call.answered", line.leg("A", A_CCID))
    assert [e[2] or e[0] for e in line.telnyx.log[1:]] == ["streaming_start", "speak", "dial"]
    stream = line.calls("call", "streaming_start")[0][3]
    assert stream["stream_url"] == f"wss://emberz-test.trycloudflare.com/media/{CALL_ID}"
    assert (stream["stream_track"], stream["stream_codec"]) == ("both_tracks", "PCMU")
    speak = line.calls("call", "speak")[0][3]
    assert "protected by Emberz" in speak["payload"] and speak["target_legs"] == "self"
    assert speak["voice"]
    dial = line.calls("dial")[0][3]
    assert dial["to"] == PROTECTED and dial["from"] == CALLER and dial["diversion"] == GUARDED
    assert "from_display_name" not in dial
    assert decode_client_state(dial["client_state"]) == {"role": "B", "call_id": CALL_ID}
    b = line.b_ccid()
    assert b

    # B's own call.initiated (outgoing) changes nothing.
    n = len(line.telnyx.log)
    line.post("call.initiated", line.leg("B", b, direction="outgoing", state="bridging"))
    assert len(line.telnyx.log) == n

    # 3. Notice finished -> conference created with A.
    line.post("call.speak.ended", line.leg("A", A_CCID, status="completed"))
    conf = line.calls("conference_create")[0][3]
    assert conf["call_control_id"] == A_CCID and conf["name"] == f"emberz-{CALL_ID}"
    assert line.flow.sessions[CALL_ID].conference_id == "fake_conf_1"
    line.post("conference.created", line.leg("A", A_CCID, conference_id="fake_conf_1"))

    # 4. B answers -> joins the conference, ending it when B leaves.
    line.post("call.answered", line.leg("B", b))
    (_, conf_id, action, join), = line.calls("conference")
    assert (conf_id, action, join["call_control_id"], join["end_conference_on_exit"]) == ("fake_conf_1", "join", b, True)
    # Conference webhooks carry A's client_state; the role comes from call_control_id.
    line.post("conference.participant.joined", line.leg("A", b, conference_id="fake_conf_1"))
    assert line.flow.state[CALL_ID].b_joined
    joined = [p for e, p in line.brains[0].events if e == "emberz.protected_joined"]
    assert len(joined) == 1 and "_role" not in joined[0] and joined[0]["conference_id"] == "fake_conf_1"

    # Deepfake result -> session signal + console "voice" event + forwarded to the brain.
    line.post("call.deepfake_detection.result", line.leg("A", A_CCID, result="fake", score=0.87, consistency=94.5))
    s = line.flow.sessions[CALL_ID]
    assert s.ai_voice.score == 0.87 and s.ai_voice.source == "telnyx"
    assert line.hub.events[-1].type == "voice" and line.hub.events[-1].data["model"] == "telnyx"
    brain = line.brains[0]
    assert "call.deepfake_detection.result" in [e for e, _ in brain.events]

    # 5. Caller hangs up -> B is hung up, brain told, session gone, console gets call_ended.
    line.post("call.hangup", line.leg("A", A_CCID, hangup_cause="normal_clearing", hangup_source="caller"))
    hangups = line.calls("call", "hangup")
    assert [h[1] for h in hangups] == [b]
    assert brain.hung_up and CALL_ID not in line.flow.sessions
    assert line.hub.types()[-1] == "call_ended"
    assert line.hub.events[-1].data["reason"] == "caller_hangup:normal_clearing"
    assert line.caretaker.events[-1][1] == "emberz.call_ended"

    # Late events after the end do nothing.
    n = len(line.telnyx.log)
    line.post("call.hangup", line.leg("B", b, hangup_cause="normal_clearing"))
    line.post("conference.ended", line.leg("A", A_CCID, conference_id="fake_conf_1", reason="host_left"))
    assert len(line.telnyx.log) == n


def test_duplicate_webhook_is_ignored(line):
    line.inbound(event_id="same-id")
    line.inbound(event_id="same-id")
    assert len(line.calls("call", "answer")) == 1
    # A redelivery with a new event id still cannot repeat a step (per-call step set + command_id).
    line.post("call.answered", line.leg("A", A_CCID))
    line.post("call.answered", line.leg("A", A_CCID))
    assert len(line.calls("dial")) == 1 and len(line.calls("call", "speak")) == 1


def test_b_answers_before_notice_ends(line):
    _answer_and_notice(line)
    b = line.b_ccid()
    line.post("call.answered", line.leg("B", b))  # conference does not exist yet
    assert len(line.calls("conference_create")) == 1
    assert line.calls("conference", "join")[0][3]["call_control_id"] == b
    line.post("call.speak.ended", line.leg("A", A_CCID))  # notice ends later: no second conference
    line.post("conference.created", line.leg("A", A_CCID, conference_id="fake_conf_1"))
    assert len(line.calls("conference_create")) == 1 and len(line.calls("conference", "join")) == 1


def test_anonymous_caller(line):
    line.inbound(caller="anonymous")
    line.post("call.answered", line.leg("A", A_CCID))
    dial = line.calls("dial")[0][3]
    assert dial["from"] == GUARDED and dial["from_display_name"] == ANON_DISPLAY_NAME
    assert "diversion" not in dial
    assert line.hub.events[0].data["caller_number"] is None


def test_known_contact_gets_display_name(line):
    line.inbound(caller=GRANDPA)
    assert line.flow.sessions[CALL_ID].caller_known
    line.post("call.answered", line.leg("A", A_CCID))
    dial = line.calls("dial")[0][3]
    assert dial["from"] == GRANDPA and dial["from_display_name"] == "Grandpa"


def test_unknown_guarded_number_is_rejected(line):
    line.inbound(to="+15555550999")
    (_, ccid, action, body), = line.telnyx.log
    assert (ccid, action, body["cause"]) == (A_CCID, "reject", "CALL_REJECTED")
    assert not line.flow.sessions


def test_protected_person_does_not_answer(line):
    _answer_and_notice(line)
    b = line.b_ccid()
    line.post("call.hangup", line.leg("B", b, hangup_cause="timeout"))
    assert [h[1] for h in line.calls("call", "hangup")] == [A_CCID]
    assert line.hub.events[-1].data["reason"] == "protected_hangup:timeout"


def test_conference_ended_hangs_up_everyone(line):
    _answer_and_notice(line)
    b = line.b_ccid()
    line.post("call.speak.ended", line.leg("A", A_CCID))
    line.post("call.answered", line.leg("B", b))
    line.post("conference.ended", line.leg("A", A_CCID, conference_id="fake_conf_1", reason="host_left"))
    assert sorted(h[1] for h in line.calls("call", "hangup")) == sorted([A_CCID, b])
    assert CALL_ID not in line.flow.sessions


def test_caretaker_leg_events_go_to_caretaker(line):
    _answer_and_notice(line)
    n = len(line.telnyx.log)
    c_state = encode_client_state("C", CALL_ID)
    line.post("call.answered", {"call_control_id": "v3:C-leg", "client_state": c_state, "to": CARETAKER})
    line.post("call.hangup", {"call_control_id": "v3:C-leg", "client_state": c_state, "hangup_cause": "normal_clearing"})
    assert [e[1] for e in line.caretaker.events] == ["call.answered", "call.hangup"]
    assert len(line.telnyx.log) == n  # the C leg never ends the protected call
    assert CALL_ID in line.flow.sessions


def test_telnyx_error_does_not_break_the_flow(line):
    async def failing_speak(ccid, action, **body):
        line.telnyx.log.append(("call", ccid, action, body))
        if action == "speak":
            raise RuntimeError("Telnyx 422")
        return {}

    line.telnyx.call_action = failing_speak
    _answer_and_notice(line)
    assert len(line.calls("dial")) == 1  # B is still dialed


def test_healthz(line):
    r = line.client.get("/healthz").json()
    assert r["ok"] and r["verify"] == "SKIPPED" and r["guarded_numbers"] == 1

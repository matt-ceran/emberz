"""Keypress intervention: B (protected) or C (caretaker, while listening) presses 9 -> caller dropped."""
from callflow_helpers import A_CCID, CALL_ID, Line
from line.call_flow import encode_client_state


def _in_conference(line: Line) -> str:
    line.inbound()
    line.post("call.answered", line.leg("A", A_CCID))
    b = line.b_ccid()
    line.post("call.speak.ended", line.leg("A", A_CCID))
    line.post("conference.created", line.leg("A", A_CCID, conference_id="fake_conf_1"))
    line.post("call.answered", line.leg("B", b))
    line.post("conference.participant.joined", line.leg("A", b, conference_id="fake_conf_1"))
    return b


def test_protected_presses_9_drops_caller_then_ends(line):
    b = _in_conference(line)
    line.post("call.dtmf.received", line.leg("B", b, digit="9"))
    assert [h[1] for h in line.calls("call", "hangup")] == [A_CCID]  # caller dropped first
    speaks = [c for c in line.calls("conference") if c[2] == "speak"]
    assert len(speaks) == 1 and speaks[0][3]["call_control_ids"] == [b]
    assert "hung up on that caller" in speaks[0][3]["payload"]
    alert = [e for e in line.hub.events if e.type == "alert"][-1]
    assert alert.data["category"] == "intervention"
    # The caller's hangup webhook must not cut the goodbye short.
    line.post("call.hangup", line.leg("A", A_CCID, hangup_cause="normal_clearing"))
    assert CALL_ID in line.flow.sessions
    # A coach line finishing (another leg) does not end it; B's goodbye finishing does.
    line.post("conference.participant.speak.ended", line.leg("A", "v3:other", conference_id="fake_conf_1"))
    assert CALL_ID in line.flow.sessions
    line.post("conference.participant.speak.ended", line.leg("A", b, conference_id="fake_conf_1"))
    assert CALL_ID not in line.flow.sessions
    assert [h[1] for h in line.calls("call", "hangup")] == [A_CCID, b]
    assert line.hub.events[-1].type == "call_ended"
    assert line.hub.events[-1].data["reason"] == "ended_by_protected_keypress"


def test_other_digits_from_b_do_nothing(line):
    b = _in_conference(line)
    n = len(line.telnyx.log)
    line.post("call.dtmf.received", line.leg("B", b, digit="5"))
    assert len(line.telnyx.log) == n and CALL_ID in line.flow.sessions


def test_caretaker_presses_9_while_listening(line):
    b = _in_conference(line)
    s = line.flow.sessions[CALL_ID]
    s.caretaker_state, s.c_ccid = "monitoring", "v3:C-leg"
    c_state = encode_client_state("C", CALL_ID)
    line.post("call.dtmf.received", {"call_control_id": "v3:C-leg", "client_state": c_state, "digit": "9"})
    assert [h[1] for h in line.calls("call", "hangup")][0] == A_CCID
    speak = [c for c in line.calls("conference") if c[2] == "speak"][0][3]
    assert speak["call_control_ids"] == [b, "v3:C-leg"] and "caretaker ended" in speak["payload"]
    line.post("conference.participant.speak.ended", line.leg("A", b, conference_id="fake_conf_1"))
    assert CALL_ID not in line.flow.sessions
    assert line.hub.events[-1].data["reason"] == "ended_by_caretaker_keypress"


def test_caretaker_digit_before_listening_goes_to_caretaker(line):
    _in_conference(line)
    c_state = encode_client_state("C", CALL_ID)
    line.post("call.dtmf.received", {"call_control_id": "v3:C-leg", "client_state": c_state, "digit": "9"})
    assert line.caretaker.events[-1][1] == "call.dtmf.received"
    assert CALL_ID in line.flow.sessions

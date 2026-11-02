import pytest

from line.caretaker import CaretakerFlow, decode_client_state, encode_client_state
from line.console_hub import BargeNotAllowed, ConsoleHub
from line.session import CallSession
from line.telnyx_client import FakeTelnyx


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def make(**kw):
    s = CallSession(
        call_id="c_1", guarded_number="+12025550100", caller_number="+12025550142",
        caretaker_number="+12025550199", conference_id="conf_1", **kw,
    )
    tx, hub, clock = FakeTelnyx(), ConsoleHub(), Clock()
    flow = CaretakerFlow(tx, hub, {"c_1": s}.get, clock=clock, cooldown_s=45)
    return s, tx, hub, flow, clock


def ev(ccid, **payload):
    return {"call_control_id": ccid, **payload}


def assert_never_muted(tx):
    for kind, _, action, body in tx.log:
        assert action not in ("mute",), tx.log
        assert "mute" not in body, body


@pytest.mark.asyncio
async def test_full_alert_answer_digit1_monitor_barge():
    s, tx, hub, flow, _ = make()
    assert await flow.alert(s, "alert", "Bank caller asked Grandma for a code.") == "dialed"
    kind, _, _, body = tx.log[0]
    assert kind == "dial"
    assert body["to"] == "+12025550199" and body["from"] == "+12025550100"
    assert decode_client_state(body["client_state"]) == {"role": "C", "call_id": "c_1"}
    assert s.caretaker_state == "ringing" and s.c_ccid

    c = s.c_ccid
    await flow.on_event(s, "call.answered", ev(c, client_state=body["client_state"]))
    kind, ccid, action, g = tx.log[-1]
    assert (kind, ccid, action) == ("call", c, "gather_using_speak")
    assert g["payload"] == "Emberz alert: Bank caller asked Grandma for a code. Press 1 to listen. While listening, press 0 to talk, or 9 to end the call."
    assert g["maximum_digits"] == 1 and g["valid_digits"] == "1" and g["voice"]
    # a re-delivered answered webhook must not start a second gather
    await flow.on_event(s, "call.answered", ev(c))
    assert tx.actions().count("gather_using_speak") == 1

    await flow.on_event(s, "call.gather.ended", ev(c, digits="1", status="valid"))
    kind, conf, action, j = tx.log[-1]
    assert (kind, conf, action) == ("conference", "conf_1", "join")
    assert j["call_control_id"] == c and j["supervisor_role"] == "monitor"
    assert s.caretaker_state == "monitoring"

    assert await flow.barge(s) == "barged"
    kind, conf, action, u = tx.log[-1]
    assert (kind, conf, action) == ("conference", "conf_1", "update")
    assert u == {"call_control_id": c, "supervisor_role": "barge"}
    assert s.caretaker_state == "barged"

    joined = [e.data["mode"] for e in hub.events("c_1") if e.type == "caretaker_joined"]
    assert joined == ["monitor", "barge"]
    assert_never_muted(tx)


@pytest.mark.asyncio
async def test_on_telnyx_webhook_routes_only_role_c():
    s, tx, hub, flow, _ = make()
    await flow.alert(s, "alert", "note")
    cs = encode_client_state({"role": "C", "call_id": "c_1"})
    body = {"data": {"event_type": "call.answered", "payload": {"call_control_id": s.c_ccid, "client_state": cs}}}
    assert await flow.on_telnyx_webhook(body) is True
    assert "gather_using_speak" in tx.actions()
    other = {"data": {"event_type": "call.answered", "payload": {"call_control_id": "a", "client_state": encode_client_state({"role": "A"})}}}
    assert await flow.on_telnyx_webhook(other) is False
    assert await flow.on_telnyx_webhook({"data": {"event_type": "call.answered", "payload": {}}}) is False


@pytest.mark.asyncio
@pytest.mark.parametrize("status,digits,expect_hangup", [
    ("timeout", "", True), ("invalid", "2", True), ("call_hangup", "", False), ("cancelled", "", True),
])
async def test_decline_on_gather(status, digits, expect_hangup):
    s, tx, hub, flow, _ = make()
    await flow.alert(s, "alert", "note")
    c = s.c_ccid
    await flow.on_event(s, "call.answered", ev(c))
    await flow.on_event(s, "call.gather.ended", ev(c, digits=digits, status=status))
    assert s.caretaker_state == "declined" and s.c_ccid is None
    assert ("hangup" in tx.actions()) is expect_hangup
    assert "join" not in tx.actions()
    assert not [e for e in hub.events("c_1") or [] if e.type == "caretaker_joined"]
    with pytest.raises(BargeNotAllowed):
        await flow.barge(s)
    # a plain alert doesn't re-dial a caretaker who declined
    assert await flow.alert(s, "alert", "again") == "skipped:declined"
    assert tx.actions().count("dial") == 1
    assert_never_muted(tx)


@pytest.mark.asyncio
async def test_no_answer_is_declined():
    s, tx, hub, flow, _ = make()
    await flow.alert(s, "alert", "note")
    await flow.on_event(s, "call.hangup", ev(s.c_ccid, hangup_cause="timeout"))
    assert s.caretaker_state == "declined"


@pytest.mark.asyncio
async def test_urgent_redials_once_after_cooldown():
    s, tx, hub, flow, clock = make()
    await flow.alert(s, "alert", "first")
    first = s.c_ccid
    await flow.on_event(s, "call.hangup", ev(first, hangup_cause="no_answer"))
    assert s.caretaker_state == "declined"

    clock.t += 10
    assert await flow.alert(s, "urgent", "street asked") == "skipped:cooldown"
    clock.t += 40
    assert await flow.alert(s, "urgent", "street asked") == "dialed"
    assert tx.actions().count("dial") == 2 and s.caretaker_state == "ringing"
    second = s.c_ccid

    # a late webhook from the first leg is ignored
    await flow.on_event(s, "call.answered", ev(first))
    assert "gather_using_speak" not in tx.actions()

    await flow.on_event(s, "call.hangup", ev(second, hangup_cause="user_busy"))
    clock.t += 1000
    assert await flow.alert(s, "urgent", "again") == "skipped:redial_used"
    assert tx.actions().count("dial") == 2


@pytest.mark.asyncio
async def test_urgent_while_monitoring_speaks_to_c_only():
    s, tx, hub, flow, _ = make()
    await flow.alert(s, "alert", "first")
    c = s.c_ccid
    await flow.on_event(s, "call.answered", ev(c))
    await flow.on_event(s, "call.gather.ended", ev(c, digits="1", status="valid"))
    assert await flow.alert(s, "alert", "more") == "already_listening"
    assert await flow.alert(s, "urgent", "Caller wants the door code.") == "whispered"
    kind, conf, action, body = tx.log[-1]
    assert (kind, conf, action) == ("conference", "conf_1", "speak")
    assert body["call_control_ids"] == [c]
    assert "door code" in body["payload"]
    assert tx.actions().count("dial") == 1
    assert_never_muted(tx)


@pytest.mark.asyncio
async def test_alert_while_ringing_updates_note_without_second_dial():
    s, tx, hub, flow, _ = make()
    await flow.alert(s, "alert", "first note")
    assert await flow.alert(s, "urgent", "newer note") == "noted"
    await flow.on_event(s, "call.answered", ev(s.c_ccid))
    assert tx.log[-1][3]["payload"].startswith("Emberz alert: newer note.")
    assert tx.actions().count("dial") == 1


@pytest.mark.asyncio
async def test_caretaker_hangs_up_while_monitoring_then_new_alert_dials():
    s, tx, hub, flow, _ = make()
    await flow.alert(s, "alert", "n")
    c = s.c_ccid
    await flow.on_event(s, "call.answered", ev(c))
    await flow.on_event(s, "call.gather.ended", ev(c, digits="1", status="valid"))
    await flow.on_event(s, "call.hangup", ev(c, hangup_cause="normal_clearing"))
    assert s.caretaker_state == "idle"
    assert await flow.alert(s, "alert", "n2") == "dialed"


@pytest.mark.asyncio
async def test_answer_after_call_ended_hangs_up_c():
    s, tx, hub, flow, _ = make()
    await flow.alert(s, "alert", "n")
    c = s.c_ccid
    s.ended = True
    await flow.on_event(s, "call.answered", ev(c))
    assert tx.log[-1][1:3] == (c, "hangup")
    assert "gather_using_speak" not in tx.actions()
    assert await flow.alert(s, "urgent", "x") == "skipped:call_ended"


@pytest.mark.asyncio
async def test_no_caretaker_number_and_dial_failure():
    s, tx, hub, flow, clock = make()
    s.caretaker_number = ""
    assert await flow.alert(s, "alert", "n") == "skipped:no_caretaker_number"
    assert tx.log == []

    s.caretaker_number = "+12025550199"

    async def boom(**body):
        raise RuntimeError("telnyx down")

    tx.dial = boom
    assert await flow.alert(s, "alert", "n") == "dial_failed"
    assert s.caretaker_state == "declined"


@pytest.mark.asyncio
async def test_call_ended_hangs_up_ringing_c_and_summarizes_with_alerts():
    import asyncio

    from contracts.models import Turn

    calls = []

    async def fake_summarize(session, alerts, hub=None, caretaker_mode=None):
        calls.append(alerts)
        hub.emit(session.call_id, "summary", session.now_ms(), text="Three sentences.")
        return "Three sentences."

    s, tx, hub, flow, _ = make()
    flow.summarizer = fake_summarize
    await flow.alert(s, "alert", "n")
    c = s.c_ccid
    hub.emit("c_1", "alert", 100, level="alert", category="scam.prize", caretaker_note="n", turn=3)
    s.turns.append(Turn(call_id="c_1", turn=1, speaker="caller", text="hi", t_start_ms=0, t_end_ms=500))
    s.ended = True
    await flow.on_event(s, "emberz.call_ended", {"reason": "caller_hangup"})
    assert tx.log[-1][1:3] == (c, "hangup")
    await asyncio.gather(*flow._tasks)
    assert calls == [[{"level": "alert", "category": "scam.prize", "caretaker_note": "n", "turn": 3}]]
    assert hub.events("c_1")[-1].type == "summary"


@pytest.mark.asyncio
async def test_call_ended_without_speech_skips_summary():
    import asyncio

    called = []

    async def fake_summarize(session, alerts, hub=None, caretaker_mode=None):
        called.append(1)
        return ""

    s, tx, hub, flow, _ = make()
    flow.summarizer = fake_summarize
    s.ended = True
    await flow.on_event(s, "emberz.call_ended", {"reason": "caller_hangup"})
    await asyncio.gather(*flow._tasks)
    assert called == [] and tx.log == []


class FailingTelnyx(FakeTelnyx):
    """FakeTelnyx that raises on the named actions (like a Telnyx 422)."""

    def __init__(self, fail):
        super().__init__()
        self.fail = set(fail)

    async def call_action(self, call_control_id, action, **body):
        await super().call_action(call_control_id, action, **body)
        if action in self.fail:
            raise RuntimeError(f"telnyx 422 on {action}")
        return {}

    async def conference_action(self, conference_id, action, **body):
        await super().conference_action(conference_id, action, **body)
        if action in self.fail:
            raise RuntimeError(f"telnyx 422 on {action}")
        return {}


def make_failing(fail, **kw):
    s, _, hub, _, clock = make(**kw)
    tx = FailingTelnyx(fail)
    flow = CaretakerFlow(tx, hub, {"c_1": s}.get, clock=clock, cooldown_s=45)
    return s, tx, hub, flow


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", ["gather_using_speak", "join"])
async def test_telnyx_failure_mid_flow_hangs_up_c_instead_of_leaving_it_silent(fail):
    s, tx, hub, flow = make_failing([fail])
    await flow.alert(s, "alert", "n")
    c = s.c_ccid
    await flow.on_event(s, "call.answered", ev(c))
    if fail == "join":
        await flow.on_event(s, "call.gather.ended", ev(c, digits="1", status="valid"))
    assert ("call", c, "hangup", {}) in tx.log
    assert s.caretaker_state == "declined" and s.c_ccid is None
    assert not [e for e in hub.events("c_1") or [] if e.type == "caretaker_joined"]


@pytest.mark.asyncio
async def test_answer_before_conference_exists_still_prompts():
    s, tx, hub, flow, _ = make()
    s.conference_id = None  # alert raised during the notice, before the conference
    await flow.alert(s, "urgent", "n")
    c = s.c_ccid
    await flow.on_event(s, "call.answered", ev(c))
    assert tx.actions()[-1] == "gather_using_speak" and s.caretaker_state == "ringing"
    s.conference_id = "conf_1"  # created while the prompt played
    await flow.on_event(s, "call.gather.ended", ev(c, digits="1", status="valid"))
    assert s.caretaker_state == "monitoring" and tx.actions()[-1] == "join"


@pytest.mark.asyncio
async def test_digit_with_no_conference_gives_up_cleanly():
    s, tx, hub, flow, _ = make()
    s.conference_id = None
    await flow.alert(s, "alert", "n")
    c = s.c_ccid
    await flow.on_event(s, "call.answered", ev(c))
    await flow.on_event(s, "call.gather.ended", ev(c, digits="1", status="valid"))
    assert "join" not in tx.actions() and tx.log[-1][1:3] == (c, "hangup")
    assert s.caretaker_state == "declined"


@pytest.mark.asyncio
@pytest.mark.parametrize("barge", [False, True])
async def test_call_end_hangs_up_a_listening_caretaker(barge):
    import asyncio

    s, tx, hub, flow, _ = make()
    flow.summarizer = lambda *a, **k: asyncio.sleep(0)
    await flow.alert(s, "alert", "n")
    c = s.c_ccid
    await flow.on_event(s, "call.answered", ev(c))
    await flow.on_event(s, "call.gather.ended", ev(c, digits="1", status="valid"))
    if barge:
        await flow.barge(s)
    s.ended = True
    await flow.on_event(s, "emberz.call_ended", {"reason": "caller_hangup"})
    assert tx.log[-1][1:3] == (c, "hangup")
    assert_never_muted(tx)


@pytest.mark.asyncio
async def test_old_leg_hangup_during_redial_in_flight_is_ignored():
    """The first leg's late call.hangup arrives while the re-dial's POST /calls is still in flight."""
    import asyncio

    s, tx, hub, flow, clock = make()
    await flow.alert(s, "alert", "first")
    first = s.c_ccid
    await flow.on_event(s, "call.hangup", ev(first, hangup_cause="no_answer"))
    clock.t += 100

    gate = asyncio.Event()
    real_dial = tx.dial

    async def slow_dial(**body):
        await gate.wait()
        return await real_dial(**body)

    tx.dial = slow_dial
    redial = asyncio.create_task(flow.alert(s, "urgent", "second"))
    await asyncio.sleep(0)
    assert s.caretaker_state == "ringing" and s.c_ccid is None
    await flow.on_event(s, "call.hangup", ev(first, hangup_cause="no_answer"))  # duplicate/late
    assert s.caretaker_state == "ringing"
    gate.set()
    assert await redial == "dialed"
    await flow.on_event(s, "call.answered", ev(s.c_ccid))
    assert tx.actions()[-1] == "gather_using_speak"


@pytest.mark.asyncio
async def test_double_tap_barge_sends_one_update_and_failure_reverts():
    import asyncio

    s, tx, hub, flow, _ = make()
    await flow.alert(s, "alert", "n")
    c = s.c_ccid
    await flow.on_event(s, "call.answered", ev(c))
    await flow.on_event(s, "call.gather.ended", ev(c, digits="1", status="valid"))
    real = tx.conference_action

    async def slow_update(*a, **k):
        await asyncio.sleep(0.01)  # a real HTTP round trip: the second tap arrives meanwhile
        return await real(*a, **k)

    tx.conference_action = slow_update
    assert await asyncio.gather(flow.barge(s), flow.barge(s)) == ["barged", "barged"]
    assert tx.actions().count("update") == 1
    assert [e.data["mode"] for e in hub.events("c_1") if e.type == "caretaker_joined"] == ["monitor", "barge"]

    s2, tx2, hub2, flow2 = make_failing(["update"])
    await flow2.alert(s2, "alert", "n")
    await flow2.on_event(s2, "call.answered", ev(s2.c_ccid))
    await flow2.on_event(s2, "call.gather.ended", ev(s2.c_ccid, digits="1", status="valid"))
    with pytest.raises(RuntimeError):
        await flow2.barge(s2)
    assert s2.caretaker_state == "monitoring"


@pytest.mark.asyncio
async def test_summary_knows_caretaker_listened_even_after_they_hung_up():
    import asyncio

    from contracts.models import Turn

    got = {}

    async def fake_summarize(session, alerts, hub=None, caretaker_mode=None):
        got["mode"] = caretaker_mode

    s, tx, hub, flow, _ = make()
    flow.summarizer = fake_summarize
    await flow.alert(s, "alert", "n")
    c = s.c_ccid
    await flow.on_event(s, "call.answered", ev(c))
    await flow.on_event(s, "call.gather.ended", ev(c, digits="1", status="valid"))
    await flow.on_event(s, "call.hangup", ev(c, hangup_cause="normal_clearing"))
    assert s.caretaker_state == "idle"
    s.turns.append(Turn(call_id="c_1", turn=1, speaker="caller", text="hi", t_start_ms=0, t_end_ms=500))
    s.ended = True
    await flow.on_event(s, "emberz.call_ended", {})
    await asyncio.gather(*flow._tasks)
    assert got["mode"] == "monitor"

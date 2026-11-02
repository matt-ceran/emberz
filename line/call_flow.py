"""L1 call skeleton: an event-driven state machine over Telnyx Call Control webhooks.

Legs, tagged by client_state (base64 JSON {"role": "A|B|C", "call_id": ...}):
  A = the caller (inbound to the guarded number), B = the protected phone, C = the caretaker (L3).

Sequence (PLAN.md L1):
  A call.initiated (incoming) -> answer A with deepfake_detection
  A call.answered  -> streaming_start on A (both tracks, PCMU) + speak the consent notice to A
                      + dial B (from = caller's number, diversion = guarded number)
  A call.speak.ended -> create the conference with A (A is its creator)
  B call.answered  -> join B to the conference (end_conference_on_exit) as soon as it exists;
                      if B answers before the notice ends, the conference is created right away
  conference.participant.joined (B) -> brain gets "emberz.protected_joined": only from here on is
                      A's outbound track the protected person (before it is the notice / silence)
  any hangup of A or B, or conference.ended -> hang up the remaining A/B legs, end the session
  C events -> caretaker.on_event(...) (L3 owns that leg; the call ending never hangs up C here)

Why the conference is created after the notice: whether a call-level `speak` keeps playing once
the leg joins a conference is not documented, so the notice finishes first (hour-0 check). B rings
while the notice plays, so this costs no extra time.

Telnyx references (checked 2026-11-02):
- answer + deepfake_detection {enabled, timeout 5-60 s (default 15), rtp_timeout 5-120 s (default 30)}:
  https://developers.telnyx.com/api-reference/call-commands/answer-call.md
  https://developers.telnyx.com/docs/voice/programmable-voice/deepfake-detection
  result webhook `call.deepfake_detection.result` {result: real|fake|silence_timeout, score, consistency};
  error webhook `call.deepfake_detection.error` {error_message}.
- speak (payload + voice required, target_legs self|opposite|both, service_level basic|premium):
  https://developers.telnyx.com/api-reference/call-commands/speak-text.md
- streaming_start (stream_url, stream_track inbound_track|outbound_track|both_tracks,
  stream_codec PCMU|...|default): https://developers.telnyx.com/api-reference/call-commands/streaming-start.md
  "only one streaming/fork operation is supported per call":
  https://developers.telnyx.com/docs/voice/programmable-voice/media-streaming
- conference create (call_control_id + name required; NO end_conference_on_exit for the creator):
  https://developers.telnyx.com/api-reference/conference-commands/create-conference.md
- conference join (call_control_id, end_conference_on_exit, supervisor_role, client_state):
  https://developers.telnyx.com/api-reference/conference-commands/join-a-conference.md
  Conference webhooks carry the conference owner's (A's) client_state, so roles for conference
  events are resolved from call_control_id, not client_state.
- dial (from "+E164", from_display_name <=128 chars of letters/digits/spaces/-_~!.+ only,
  diversion = "the `to` number of an active inbound call ... the `from` number is treated as
  verified"): https://developers.telnyx.com/api-reference/call-commands/dial.md
- reject (cause CALL_REJECTED|NOT_FOUND|TEMPORARILY_UNAVAILABLE|USER_BUSY): OpenAPI spec
  https://raw.githubusercontent.com/team-telnyx/openapi/master/openapi/spec3.json
- webhook envelope, event names, duplicates/out-of-order, command_id dedupe within 60 s:
  https://developers.telnyx.com/docs/voice/programmable-voice/voice-api-webhooks.md
  https://developers.telnyx.com/docs/voice/programmable-voice/receiving-webhooks
  sample payloads: https://developers.telnyx.com/data/webhook-events.json

Idempotency: every command we send carries command_id = "<call_id>:<step>", so a duplicate
webhook that slipped past the server's event-id dedupe cannot double-dial or double-answer
(Telnyx ignores a repeated command_id). Each step also runs at most once per call (FlowState.done).
A call id is never reused: a late re-delivered call.initiated for a call that already ended is ignored.

Failure handling (nobody is left in silence):
- streaming_start / speak fail: logged, the call still connects (unprotected, but connected).
- dial B with the caller's number fails: re-dial once from the guarded number, with the caller's
  number in from_display_name; if that fails too, the caller is hung up (reason "dial_b_failed").
- conference create fails: retried when B answers; if it fails again, both legs are hung up.
- B's leg gets time_limit_secs (Settings.max_call_s) as a cost guard if this server dies mid-call.
"""
from __future__ import annotations

import asyncio
import base64
import inspect
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

from contracts.models import AiVoice, ConsoleEvent, VoiceSignal
from line.config import GuardedLine, Settings, is_e164, normalize_number
from line.session import CallSession

log = logging.getLogger("emberz.call_flow")

ANON_DISPLAY_NAME = "Emberz unknown caller"  # "Emberz: unknown caller" is invalid (no ':' allowed)
DEEPFAKE = {"enabled": True, "timeout": 20, "rtp_timeout": 30}
B_RING_TIMEOUT_S = 30


# --------------------------------------------------------------------------- hooks (other units)
class Brain(Protocol):
    async def on_media(self, track: str, payload: bytes, ts_ms: int) -> None: ...
    async def on_call_event(self, event_type: str, payload: dict) -> None: ...
    async def on_hangup(self) -> None: ...


class Caretaker(Protocol):
    async def on_event(self, session: CallSession, event_type: str, payload: dict) -> None: ...


class Hub(Protocol):
    def publish(self, event: ConsoleEvent) -> Any: ...


class NullBrain:
    async def on_media(self, track, payload, ts_ms):
        pass

    async def on_call_event(self, event_type, payload):
        pass

    async def on_hangup(self):
        pass


class NullCaretaker:
    async def on_event(self, session, event_type, payload):
        pass


class NullHub:
    def publish(self, event):
        log.debug("console_event %s %s", event.type, event.call_id)


MakeBrain = Callable[[CallSession, Any, Any], Brain]


def null_make_brain(session, telnyx, hub) -> Brain:
    return NullBrain()


async def maybe_await(x: Any) -> Any:
    return await x if inspect.isawaitable(x) else x


# --------------------------------------------------------------------------- client_state
def encode_client_state(role: str, call_id: str) -> str:
    raw = json.dumps({"role": role, "call_id": call_id}, separators=(",", ":"))
    return base64.b64encode(raw.encode()).decode()


def decode_client_state(value: Optional[str]) -> Optional[dict]:
    if not value:
        return None
    try:
        data = json.loads(base64.b64decode(value))
    except (ValueError, TypeError):
        return None
    if isinstance(data, dict) and data.get("role") in ("A", "B", "C") and data.get("call_id"):
        return data
    return None


def display_name(name: str) -> str:
    """Keep only the characters Telnyx allows in from_display_name."""
    return re.sub(r"[^A-Za-z0-9 \-_~!.+]", "", name)[:128].strip()


# --------------------------------------------------------------------------- per-call flow state
END_DIGIT = os.environ.get("EMBERZ_END_DIGIT", "9")    # B or C presses it: Emberz hangs up on the caller
TALK_DIGIT = os.environ.get("EMBERZ_TALK_DIGIT", "0")  # C presses it while listening: barge (talk)
INTERVENE_GOODBYE_S = 5.0  # time for the goodbye line before B (and C) are hung up
INTERVENE_TEXT = {
    "protected": "Emberz hung up on that caller for you. You are safe. Goodbye.",
    "caretaker": "Your caretaker ended that call. You are safe. Goodbye.",
}


@dataclass
class FlowState:
    done: set[str] = field(default_factory=set)
    b_answered: bool = False
    b_joined: bool = False
    hung_up: set[str] = field(default_factory=set)  # call_control_ids that are gone
    end_reason: str = ""
    intervened_by: str = ""  # "protected" | "caretaker" once someone pressed END_DIGIT


class CallFlow:
    def __init__(
        self,
        settings: Settings,
        profiles: dict[str, GuardedLine],
        telnyx: Any,
        make_brain: Optional[MakeBrain] = None,
        caretaker: Optional[Caretaker] = None,
        hub: Optional[Hub] = None,
        sessions: Optional[dict[str, CallSession]] = None,
    ):
        self.settings = settings
        self.profiles = profiles
        self.telnyx = telnyx
        self.make_brain = make_brain or null_make_brain
        self.caretaker = caretaker or NullCaretaker()
        self.hub = hub or NullHub()
        self.sessions: dict[str, CallSession] = {} if sessions is None else sessions
        self.brains: dict[str, Brain] = {}
        self.state: dict[str, FlowState] = {}
        self.ccid_index: dict[str, tuple[str, str]] = {}  # call_control_id -> (call_id, role)
        self._locks: dict[str, asyncio.Lock] = {}

    # ------------------------------------------------------------------ entry point
    async def handle(self, event_type: str, payload: dict) -> None:
        call_id, role = self._route(event_type, payload)
        if call_id is None:
            log.info("ignoring %s for an unknown leg", event_type)
            return
        lock = self._locks.setdefault(call_id, asyncio.Lock())
        async with lock:
            if role == "C":
                session = self.sessions.get(call_id)
                if session is not None:
                    if event_type == "call.dtmf.received" and session.caretaker_state in ("monitoring", "barged"):
                        await self._on_caretaker_digit(session, str(payload.get("digit", "")))
                        return
                    await self.caretaker.on_event(session, event_type, payload)
                return
            if event_type == "call.initiated" and role == "A":
                if call_id not in self.state:  # never resurrect a call that already ended
                    await self._on_inbound(call_id, payload)
                return
            session = self.sessions.get(call_id)
            if session is None:
                return  # a late event for a call that already ended
            await self._dispatch(session, role, event_type, payload)
            if not session.ended:
                brain = self.brains.get(call_id)
                if brain is not None:
                    await brain.on_call_event(event_type, {**payload, "_role": role})

    def _route(self, event_type: str, payload: dict) -> tuple[Optional[str], Optional[str]]:
        ccid = payload.get("call_control_id")
        if ccid in self.ccid_index:  # most reliable: we know every leg we created
            return self.ccid_index[ccid]
        cs = decode_client_state(payload.get("client_state"))
        if cs and not event_type.startswith("conference."):
            return cs["call_id"], cs["role"]
        if event_type == "call.initiated" and payload.get("direction") == "incoming":
            return "c_" + str(payload.get("call_session_id") or ccid), "A"
        if cs:  # conference event for a leg we no longer index
            return cs["call_id"], None
        return None, None

    # ------------------------------------------------------------------ A: inbound
    async def _on_inbound(self, call_id: str, p: dict) -> None:
        ccid = p["call_control_id"]
        guarded, raw_from = normalize_number(p.get("to")), normalize_number(p.get("from"))
        line = self.profiles.get(guarded)
        if line is None:
            log.warning("inbound call to %s is not a guarded number; rejecting", guarded)
            await self._cmd("reject", ccid, f"{call_id}:reject", cause="CALL_REJECTED")
            return
        caller = raw_from if is_e164(raw_from) else None  # anonymous, restricted, SIP URI ...
        session = CallSession(
            call_id=call_id,
            guarded_number=guarded,
            caller_number=caller,
            profile=line.profile,
            rules=list(line.rules),
            allowed_contacts=dict(line.allowed_contacts),
            protected_number=line.protected_number,
            caretaker_number=line.caretaker_number,
            caller_known=caller is not None and caller in line.allowed_contacts,
            a_ccid=ccid,
        )
        self.sessions[call_id] = session
        self.state[call_id] = FlowState()
        self.ccid_index[ccid] = (call_id, "A")
        self.brains[call_id] = self.make_brain(session, self.telnyx, self.hub)
        await self._publish(session, "call_started", {
            "caller_number": caller, "guarded_number": guarded,
            "profile": session.profile, "caller_known": session.caller_known,
        })
        extra = {"deepfake_detection": dict(DEEPFAKE)} if self.settings.deepfake else {}
        await self._step(session, "answer", self.telnyx.call_action(
            ccid, "answer",
            client_state=encode_client_state("A", call_id),
            command_id=f"{call_id}:answer",
            **extra,
        ))

    # ------------------------------------------------------------------ dispatch
    async def _dispatch(self, s: CallSession, role: Optional[str], et: str, p: dict) -> None:
        st = self.state[s.call_id]
        ccid = p.get("call_control_id")

        if et == "call.answered" and role == "A":
            await self._on_a_answered(s)
        elif et == "call.speak.ended" and role == "A":
            await self._create_conference(s)
        elif et == "call.answered" and role == "B":
            st.b_answered = True
            if s.conference_id:
                await self._join_b(s)
            else:
                await self._create_conference(s)  # B answered before the notice ended
        elif et == "conference.created":
            s.conference_id = s.conference_id or p.get("conference_id")
            if st.b_answered:
                await self._join_b(s)
        elif et == "conference.participant.joined" and ccid and ccid == s.b_ccid and not st.b_joined:
            st.b_joined = True
            # From now on A's outbound track is the protected person (before: notice / silence).
            brain = self.brains.get(s.call_id)
            if brain is not None:
                await brain.on_call_event("emberz.protected_joined", {"at_ms": s.now_ms(),
                                                                      "conference_id": s.conference_id})
        elif et == "call.deepfake_detection.result":
            await self._on_deepfake(s, p)
        elif et == "call.deepfake_detection.error":
            log.warning("deepfake detection error on %s: %s", s.call_id, p.get("error_message"))
        elif et == "streaming.started":
            s.stream_id = p.get("stream_id") or s.stream_id
        elif et == "streaming.failed":
            log.error("media stream failed on %s: %s", s.call_id, p.get("failure_reason"))
        elif et == "call.dtmf.received" and role == "B" and str(p.get("digit", "")) == END_DIGIT:
            await self.intervene(s, "protected")
        elif et == "call.hangup" and role == "A" and st.intervened_by:
            st.hung_up.add(ccid)  # we dropped the caller on purpose; B hears the goodbye first
        elif et == "call.hangup" and role in ("A", "B"):
            st.hung_up.add(ccid)
            who = "caller" if role == "A" else "protected"
            cause = p.get("hangup_cause", "")
            await self._end_call(s, f"{who}_hangup:{cause}" if cause else f"{who}_hangup")
        elif et == "conference.ended":
            await self._end_call(s, f"conference_ended:{p.get('reason', '')}")

    async def _on_a_answered(self, s: CallSession) -> None:
        a, cid = s.a_ccid, s.call_id
        cs_a = encode_client_state("A", cid)
        try:
            stream_url = self.settings.media_ws_url(cid)
        except RuntimeError as e:  # no PUBLIC_BASE_URL: connect the call anyway, unprotected
            log.error("no media stream for %s: %s", cid, e)
            stream_url = None
        if stream_url:
            await self._step(s, "stream", self.telnyx.call_action(
                a, "streaming_start",
                stream_url=stream_url,
                stream_track="both_tracks",
                stream_codec="PCMU",
                client_state=cs_a,
                command_id=f"{cid}:stream",
            ))
        await self._step(s, "notice", self.telnyx.call_action(
            a, "speak",
            payload=self.settings.notice_text,
            voice=self.settings.tts_voice,
            language=self.settings.tts_language,
            target_legs="self",
            client_state=cs_a,
            command_id=f"{cid}:notice",
        ))
        await self._dial_b(s)

    async def _dial_b(self, s: CallSession) -> None:
        if "dial_b" in self.state[s.call_id].done:
            return
        body: dict[str, Any] = {
            "to": s.protected_number,
            "client_state": encode_client_state("B", s.call_id),
            "timeout_secs": B_RING_TIMEOUT_S,
            "time_limit_secs": self.settings.max_call_s,
            # Dial accepts command_id ("Telnyx will ignore others Dial commands with the same
            # command_id"), but line/telnyx_client.py's dial() currently drops it (contract change asked).
            "command_id": f"{s.call_id}:dial_b",
        }
        if s.caller_number and os.environ.get("EMBERZ_B_FROM", "caller") == "guarded":
            # Carrier spam filters may divert calls whose caller ID is not ours straight to
            # voicemail; ring B from the guarded number and show the caller in the display name.
            name = s.allowed_contacts.get(s.caller_number) or "Emberz"
            body["from"] = s.guarded_number
            body["from_display_name"] = display_name(f"{name} {s.caller_number}")
        elif s.caller_number:
            # Show the caller's real number on B. `diversion` = the guarded number, which is the
            # `to` of the active inbound call, so Telnyx treats the caller's number as verified.
            body["from"] = s.caller_number
            body["diversion"] = s.guarded_number
            name = s.allowed_contacts.get(s.caller_number)
            if name:
                body["from_display_name"] = display_name(name)
        else:
            body["from"] = s.guarded_number  # a number we own
            body["from_display_name"] = ANON_DISPLAY_NAME
        result = await self._step(s, "dial_b", self.telnyx.dial(**body))
        if not (result and result.get("call_control_id")) and s.caller_number:
            # Telnyx refused the caller's number as `from`: ring B from the guarded number (ours)
            # and put the caller's number in the display name instead.
            name = s.allowed_contacts.get(s.caller_number) or "Emberz"
            fallback = {k: v for k, v in body.items() if k not in ("diversion", "from_display_name")}
            fallback |= {"from": s.guarded_number, "from_display_name": display_name(f"{name} {s.caller_number}"),
                         "command_id": f"{s.call_id}:dial_b_fallback"}
            log.warning("dial B with the caller's number failed on %s; retrying from the guarded number", s.call_id)
            result = await self._step(s, "dial_b_fallback", self.telnyx.dial(**fallback))
        if result and result.get("call_control_id"):
            s.b_ccid = result["call_control_id"]
            self.ccid_index[s.b_ccid] = (s.call_id, "B")
        else:
            await self._end_call(s, "dial_b_failed")

    async def _create_conference(self, s: CallSession) -> None:
        if s.ended or s.conference_id:
            return
        st = self.state[s.call_id]
        # One try when the notice ends; if it fails, B's answer retries. Once B is waiting, try twice.
        for _ in range(2 if st.b_answered else 1):
            result = await self._step(s, "conference", retry=True, coro=self.telnyx.conference_create(
                call_control_id=s.a_ccid,
                name=f"emberz-{s.call_id}",
                beep_enabled="never",
                client_state=encode_client_state("A", s.call_id),
                command_id=f"{s.call_id}:conference",
            ))
            if result and result.get("id"):
                s.conference_id = result["id"]
                if st.b_answered:
                    await self._join_b(s)
                return
            if "conference" in st.done:  # already sent once and succeeded without an id
                return
        if st.b_answered:
            await self._end_call(s, "conference_failed")  # B would hear only silence

    async def _join_b(self, s: CallSession) -> None:
        st = self.state[s.call_id]
        if not (s.conference_id and s.b_ccid) or "join_b" in st.done or s.ended:
            return
        for _ in range(2):
            result = await self._step(s, "join_b", retry=True, coro=self.telnyx.conference_action(
                s.conference_id, "join",
                call_control_id=s.b_ccid,
                end_conference_on_exit=True,
                client_state=encode_client_state("B", s.call_id),
                command_id=f"{s.call_id}:join_b",
            ))
            if result is not None:
                return
        await self._end_call(s, "join_failed")  # B answered but cannot hear the caller

    async def _on_deepfake(self, s: CallSession, p: dict) -> None:
        score = p.get("score")
        log.info("deepfake result on %s: %s score=%s consistency=%s",
                 s.call_id, p.get("result"), score, p.get("consistency"))
        if score is None:  # silence_timeout
            return
        try:
            score = max(0.0, min(1.0, float(score)))
        except (TypeError, ValueError):
            return
        if s.ai_voice is None:
            s.ai_voice = AiVoice(score=score, source="telnyx")
        sig = VoiceSignal(call_id=s.call_id, at_ms=s.now_ms(), ai_score=score, model="telnyx")
        await self._publish(s, "voice", sig.model_dump())

    # ------------------------------------------------------------------ intervention
    async def _on_caretaker_digit(self, s: CallSession, digit: str) -> None:
        if digit == END_DIGIT:
            await self.intervene(s, "caretaker")
        elif digit == TALK_DIGIT:
            try:
                await self.caretaker.barge(s)
            except Exception as e:
                log.warning("caretaker barge by keypress failed on %s: %s", s.call_id, e)

    async def intervene(self, s: CallSession, by: str) -> None:
        """Someone pressed END_DIGIT: drop the caller now, tell the protected person, then end.
        Runs under the call's lock (from handle); the final hang-up runs as a task after the goodbye."""
        st = self.state[s.call_id]
        if s.ended or st.intervened_by:
            return
        st.intervened_by = by
        log.info("call %s: ended by %s keypress", s.call_id, by)
        await self._publish(s, "alert", {"level": "urgent", "category": "intervention",
                                         "caretaker_note": f"Call ended by the {by} (keypress {END_DIGIT})"})
        if s.a_ccid and s.a_ccid not in st.hung_up:
            await self._cmd("hangup", s.a_ccid, f"{s.call_id}:intervene_a")
        listeners = [c for c in (s.b_ccid, getattr(s, "c_ccid", None)) if c]
        if s.conference_id and listeners:
            try:
                await self.telnyx.conference_action(
                    s.conference_id, "speak", payload=INTERVENE_TEXT.get(by, INTERVENE_TEXT["protected"]),
                    voice=self.settings.tts_voice,
                    language="en-US", call_control_ids=listeners,
                )
            except Exception as e:
                log.warning("intervention goodbye failed on %s: %s", s.call_id, e)

        async def finish() -> None:
            await asyncio.sleep(INTERVENE_GOODBYE_S)
            async with self._locks.setdefault(s.call_id, asyncio.Lock()):
                if not s.ended:
                    await self._end_call(s, f"ended_by_{by}_keypress")

        self._intervene_tasks = getattr(self, "_intervene_tasks", set())
        t = asyncio.create_task(finish())
        self._intervene_tasks.add(t)
        t.add_done_callback(self._intervene_tasks.discard)

    async def _end_call(self, s: CallSession, reason: str) -> None:
        st = self.state[s.call_id]
        if s.ended:
            return
        s.ended = True
        st.end_reason = reason
        for leg, step in ((s.a_ccid, "hangup_a"), (s.b_ccid, "hangup_b")):
            if leg and leg not in st.hung_up:
                await self._cmd("hangup", leg, f"{s.call_id}:{step}")
                st.hung_up.add(leg)
        brain = self.brains.pop(s.call_id, None)
        if brain is not None:
            try:
                await brain.on_hangup()
            except Exception:
                log.exception("brain.on_hangup failed on %s", s.call_id)
        await self._publish(s, "call_ended", {"duration_ms": s.now_ms(), "reason": reason})
        try:
            await self.caretaker.on_event(s, "emberz.call_ended", {"reason": reason})
        except Exception:
            log.exception("caretaker.on_event failed on %s", s.call_id)
        self.sessions.pop(s.call_id, None)
        for ccid in (s.a_ccid, s.b_ccid):
            self.ccid_index.pop(ccid, None)
        # self.state[call_id] is kept so tests and hour0 can inspect how the call ended.

    # ------------------------------------------------------------------ helpers
    async def _step(self, s: CallSession, name: str, coro, retry: bool = False) -> Optional[dict]:
        """Run a Telnyx command once per call; log and swallow API errors.
        retry=True: a failed step may run again on the next trigger."""
        st = self.state[s.call_id]
        if name in st.done:
            coro.close()
            return None
        st.done.add(name)
        try:
            return await coro
        except Exception as e:  # TelnyxError or network error: keep the flow alive
            log.error("step %s failed on %s: %s", name, s.call_id, e)
            if retry:
                st.done.discard(name)
            return None

    async def _cmd(self, action: str, ccid: str, command_id: str, **body: Any) -> None:
        try:
            await self.telnyx.call_action(ccid, action, command_id=command_id, **body)
        except Exception as e:
            log.warning("%s on %s failed (leg may already be gone): %s", action, ccid, e)

    async def _publish(self, s: CallSession, type_: str, data: dict) -> None:
        try:
            await maybe_await(self.hub.publish(
                ConsoleEvent(type=type_, call_id=s.call_id, at_ms=s.now_ms(), data=data)))
        except Exception:
            log.exception("hub.publish failed")

"""Caretaker alert call (L3): dial C, "Press 1 to listen", join as monitor, barge on demand.

Flow (PLAN.md L3):
  alert(session, level, note)
    idle      -> dial C from the guarded number, client_state {role: "C", call_id}; state "ringing"
    ringing   -> keep the newest note for the prompt
    monitoring/barged + urgent -> conference speak to C only ("whisper")
    declined  -> an "urgent" re-dials once, after REDIAL_COOLDOWN_S
  on_event(session, "call.answered")      -> gather_using_speak "Emberz alert: <note>. Press 1 to listen."
  on_event(session, "call.gather.ended")  -> digits "1": conference join, supervisor_role "monitor";
                                             anything else: hang up C, state "declined"
  on_event(session, "call.hangup")        -> ringing: "declined"; listening: back to "idle"
  on_event(session, "emberz.call_ended")  -> (sent by line/call_flow.py at hang-up) hang up C in any
                                             state (a join alone doesn't prove the conference end hangs C
                                             up, e.g. if B never joined), then summarize in the background
                                             -> console "summary"
  barge(session)                          -> conference update, supervisor_role "barge"

Failure handling: if Telnyx rejects the gather or the join, C is hung up and marked "declined"
(so Mom is never left on a silent line). Events from a retired C leg (before a re-dial) are ignored.

Wiring in the line server (line/server.py builds the Telnyx client and the sessions itself):
    hub = ConsoleHub()
    caretaker = CaretakerFlow(None, hub, lambda call_id: None)
    app = create_app(caretaker=caretaker, hub=hub, make_brain=..., extra_routers=[console_router(hub, caretaker)])
    # console_router fails closed: set CONSOLE_TOKEN in the line server's env, or the console answers 503
    caretaker.telnyx, caretaker.sessions_lookup = app.state.telnyx, app.state.flow.sessions.get
    add_cors(app)

C is NEVER muted: an explicit mute outlasts the supervisor-role change, so the
barge would be silent. Monitor already means "hears everyone, heard by no one".

Telnyx v2 references (checked 2026-11-02 against the call-control OpenAPI file at
https://developers.telnyx.com/openapi/source/external/call-control/call-control.json):
- Dial: POST /v2/calls (to, from, connection_id, client_state = base64, timeout_secs)
  https://developers.telnyx.com/api-reference/call-commands/dial
- Gather using speak: POST /v2/calls/{id}/actions/gather_using_speak (payload, voice [required],
  language, minimum_digits, maximum_digits, valid_digits, timeout_millis, maximum_tries, invalid_payload)
  https://developers.telnyx.com/api-reference/call-commands/gather-using-speak
- call.gather.ended payload: {call_control_id, client_state, digits, status}, status in
  valid | invalid | call_hangup | cancelled | cancelled_amd | timeout
  https://developers.telnyx.com/api-reference/callbacks/call-gather-ended
- Join: POST /v2/conferences/{id}/actions/join (call_control_id, supervisor_role
  barge|monitor|none|whisper, beep_enabled, mute ... we never send mute)
  https://developers.telnyx.com/api-reference/conference-commands/join-a-conference
- Update: POST /v2/conferences/{id}/actions/update (call_control_id, supervisor_role) [both required]
  https://developers.telnyx.com/api-reference/conference-commands/update-conference-participant
- Conference speak: POST /v2/conferences/{id}/actions/speak (payload, voice, call_control_ids =
  only these participants hear it)
  https://developers.telnyx.com/api-reference/conference-commands/speak-text-to-conference-participants
- call.hangup payload: hangup_cause in call_rejected | normal_clearing | originator_cancel | timeout |
  time_limit | user_busy | not_found | no_answer | unspecified
"""
from __future__ import annotations

import base64
import json
import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from line.console_hub import BargeNotAllowed, ConsoleHub
from line.session import CallSession

log = logging.getLogger("emberz.caretaker")

VOICE = os.environ.get("TELNYX_TTS_VOICE", "Telnyx.KokoroTTS.af")  # the docs' own example voice
LANGUAGE = "en-US"
RING_TIMEOUT_S = 30           # how long C rings before Telnyx gives up (hangup_cause timeout/no_answer)
GATHER_TIMEOUT_MS = 8000      # wait for the digit after the prompt ends
GATHER_TRIES = 2              # the prompt plays at most twice
REDIAL_COOLDOWN_S = 45.0      # an urgent alert may re-dial a declined caretaker once, after this
MAX_REDIALS = 1


def encode_client_state(state: dict[str, Any]) -> str:
    return base64.b64encode(json.dumps(state, separators=(",", ":")).encode()).decode()


def decode_client_state(value: Optional[str]) -> dict[str, Any]:
    if not value:
        return {}
    try:
        out = json.loads(base64.b64decode(value))
        return out if isinstance(out, dict) else {}
    except Exception:
        return {}


def _clean(note: str, limit: int = 160) -> str:
    note = " ".join((note or "").split()).rstrip(". ")
    return note[:limit] or "a call needs your attention"


@dataclass
class _Track:
    """Per-call bits that don't belong on the shared CallSession."""

    note: str = ""
    answered: bool = False
    declined_at: Optional[float] = None
    redials: int = 0
    old_legs: set[str] = field(default_factory=set)  # C legs from before a re-dial; their webhooks are ignored
    joined_mode: Optional[str] = None                # "monitor" / "barge" once C ever listened (for the summary)


class CaretakerFlow:
    def __init__(
        self,
        telnyx: Any,
        hub: ConsoleHub,
        sessions_lookup: Callable[[str], Optional[CallSession]],
        *,
        voice: str = VOICE,
        cooldown_s: float = REDIAL_COOLDOWN_S,
        clock: Callable[[], float] = time.monotonic,
        summarizer: Optional[Callable[..., Awaitable[str]]] = None,
    ):
        self.telnyx = telnyx
        self.hub = hub
        self.sessions_lookup = sessions_lookup
        self.voice = voice
        self.cooldown_s = cooldown_s
        self.clock = clock
        if summarizer is None:
            from line.summary import summarize as summarizer
        self.summarizer = summarizer
        self._tracks: dict[str, _Track] = {}
        self._tasks: set[asyncio.Task] = set()

    def _track(self, session: CallSession) -> _Track:
        return self._tracks.setdefault(session.call_id, _Track())

    # ------------------------------------------------------------------ alerts
    async def alert(self, session: CallSession, level: str, note: str) -> str:
        """Called by the brain (L2) when policy says notify the caretaker. Returns what it did."""
        tr = self._track(session)
        state = session.caretaker_state
        if session.ended:
            return "skipped:call_ended"
        if not session.caretaker_number:
            log.warning("call %s: no caretaker number, alert dropped", session.call_id)
            return "skipped:no_caretaker_number"

        if state == "idle":
            return await self._dial(session, note)
        if state == "ringing":
            tr.note = _clean(note)  # the prompt uses the newest note if C hasn't answered yet
            return "noted"
        if state in ("monitoring", "barged"):
            if level != "urgent":
                return "already_listening"
            return await self._speak_to_c(session, f"Emberz urgent: {_clean(note)}.")
        if state == "declined":
            if level != "urgent":
                return "skipped:declined"
            if tr.redials >= MAX_REDIALS:
                return "skipped:redial_used"
            if tr.declined_at is not None and self.clock() - tr.declined_at < self.cooldown_s:
                return "skipped:cooldown"
            tr.redials += 1
            return await self._dial(session, note)
        return f"skipped:{state}"

    async def _dial(self, session: CallSession, note: str) -> str:
        tr = self._track(session)
        tr.note, tr.answered = _clean(note), False
        session.caretaker_state = "ringing"  # set before awaiting so a second alert can't double-dial
        try:
            data = await self.telnyx.dial(
                to=session.caretaker_number,
                **{"from": session.guarded_number},  # `from` is a Python keyword
                client_state=encode_client_state({"role": "C", "call_id": session.call_id}),
                timeout_secs=RING_TIMEOUT_S,
            )
        except Exception:
            log.exception("call %s: dialing the caretaker failed", session.call_id)
            self._decline(session)
            return "dial_failed"
        session.c_ccid = data.get("call_control_id")
        return "dialed"

    def _decline(self, session: CallSession) -> None:
        tr = self._track(session)
        if session.c_ccid:
            tr.old_legs.add(session.c_ccid)
        session.caretaker_state = "declined"
        session.c_ccid = None
        tr.declined_at = self.clock()

    async def _give_up(self, session: CallSession, why: str) -> None:
        """A Telnyx command for C failed mid-flow: don't leave Mom on a silent line."""
        log.error("call %s: caretaker %s failed; hanging up C", session.call_id, why)
        await self._hangup_c(session, session.c_ccid)
        self._decline(session)

    async def _speak_to_c(self, session: CallSession, text: str) -> str:
        if not (session.conference_id and session.c_ccid):
            return "skipped:no_conference"
        try:
            await self.telnyx.conference_action(
                session.conference_id, "speak",
                payload=text, voice=self.voice, language=LANGUAGE, call_control_ids=[session.c_ccid],
            )
        except Exception as e:
            log.warning("call %s: urgent whisper to C failed: %s", session.call_id, e)
            return "whisper_failed"
        return "whispered"

    # ------------------------------------------------------------------ webhooks for leg C
    async def on_telnyx_webhook(self, body: dict[str, Any]) -> bool:
        """Convenience for L1's webhook handler: pass the whole webhook body.
        Returns True if it was a caretaker-leg event (and handled here)."""
        data = body.get("data", body)
        payload = data.get("payload", {}) or {}
        cs = decode_client_state(payload.get("client_state"))
        if cs.get("role") != "C":
            return False
        session = self.sessions_lookup(cs.get("call_id", ""))
        if session is not None:
            await self.on_event(session, data.get("event_type", ""), payload)
        return True

    async def on_event(self, session: CallSession, event_type: str, payload: dict[str, Any]) -> None:
        if event_type == "emberz.call_ended":
            await self._on_call_ended(session)
            return
        ccid = payload.get("call_control_id")
        tr = self._track(session)
        if ccid and (ccid in tr.old_legs or (session.c_ccid and ccid != session.c_ccid)):
            return  # an old leg (e.g. the first try before a re-dial); ignore
        state = session.caretaker_state

        if event_type == "call.answered" and state == "ringing" and not tr.answered:
            tr.answered = True
            session.c_ccid = session.c_ccid or ccid
            if session.ended:
                await self._hangup_c(session, session.c_ccid)
                session.caretaker_state, session.c_ccid = "idle", None
                return
            # No conference yet (alert during the notice) is fine: the prompt takes several
            # seconds, and the conference is checked again when the digit arrives.
            try:
                await self.telnyx.call_action(
                    session.c_ccid, "gather_using_speak",
                    payload=f"Emberz alert: {tr.note}. Press 1 to listen. While listening, press 0 to talk, or 9 to end the call.",
                    invalid_payload="Press 1 to listen, or hang up.",
                    voice=self.voice, language=LANGUAGE,
                    minimum_digits=1, maximum_digits=1, valid_digits="1",
                    timeout_millis=GATHER_TIMEOUT_MS, maximum_tries=GATHER_TRIES,
                    client_state=payload.get("client_state"),
                )
            except Exception:
                await self._give_up(session, "gather_using_speak")

        elif event_type == "call.gather.ended" and state == "ringing":
            if payload.get("status") == "valid" and payload.get("digits") == "1" and not session.ended:
                if not session.conference_id:
                    await self._give_up(session, "join (no conference yet)")
                    return
                try:
                    await self.telnyx.conference_action(
                        session.conference_id, "join",
                        call_control_id=session.c_ccid, supervisor_role="monitor", beep_enabled="never",
                    )
                except Exception:
                    await self._give_up(session, "conference join")
                    return
                session.caretaker_state = "monitoring"
                tr.joined_mode = tr.joined_mode or "monitor"
                self.hub.emit(session.call_id, "caretaker_joined", session.now_ms(), mode="monitor")
            else:
                if payload.get("status") != "call_hangup":
                    await self._hangup_c(session, session.c_ccid)
                self._decline(session)

        elif event_type == "call.hangup":
            if state == "ringing":
                self._decline(session)  # no answer, busy, rejected, or hung up during the prompt
            elif state in ("monitoring", "barged"):
                if session.c_ccid:
                    tr.old_legs.add(session.c_ccid)
                session.caretaker_state, session.c_ccid = "idle", None  # C left; a new alert dials again

    async def _on_call_ended(self, session: CallSession) -> None:
        if session.c_ccid:
            # Ringing, in the "press 1" prompt, or listening. The conference ending normally hangs
            # C up too, but not if it never ended (e.g. B never joined), so hang up explicitly;
            # a 422 for a leg that is already gone is logged and ignored.
            await self._hangup_c(session, session.c_ccid)
        task = asyncio.create_task(self._summarize(session))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _summarize(self, session: CallSession) -> None:
        try:
            if not session.turns:
                return  # nothing was said (e.g. hung up during the notice): no summary, no spend
            alerts = [e.data for e in (self.hub.events(session.call_id) or []) if e.type == "alert"]
            mode = self._track(session).joined_mode
            await self.summarizer(session, alerts, hub=self.hub, caretaker_mode=mode)
        except Exception:
            log.exception("call %s: summary failed", session.call_id)
        finally:
            self.forget(session.call_id)

    async def _hangup_c(self, session: CallSession, ccid: Optional[str]) -> None:
        if not ccid:
            return
        try:
            await self.telnyx.call_action(ccid, "hangup")
        except Exception:
            log.warning("call %s: hanging up C failed", session.call_id)

    # ------------------------------------------------------------------ the join button
    async def barge(self, session: CallSession) -> str:
        if session.caretaker_state == "barged":
            return "barged"
        if session.caretaker_state != "monitoring" or not (session.c_ccid and session.conference_id) or session.ended:
            raise BargeNotAllowed(f"caretaker is {session.caretaker_state}, not monitoring")
        session.caretaker_state = "barged"  # before awaiting, so a double tap sends one update
        try:
            await self.telnyx.conference_action(
                session.conference_id, "update", call_control_id=session.c_ccid, supervisor_role="barge",
            )
        except Exception:
            if session.caretaker_state == "barged":
                session.caretaker_state = "monitoring"
            raise
        self._track(session).joined_mode = "barge"
        self.hub.emit(session.call_id, "caretaker_joined", session.now_ms(), mode="barge")
        return "barged"

    def forget(self, call_id: str) -> None:
        """Drop per-call tracking once the call is over and summarized."""
        self._tracks.pop(call_id, None)

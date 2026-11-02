"""Live brain for one call (L2): ASR turns -> guardian -> rule layer -> coach / caretaker / console.

Server hook interface (one CallBrain per call; make_brain_factory(caretaker) gives the
(session, telnyx, hub) -> brain callable that line/call_flow.py expects):
    await brain.on_media(track, payload_bytes, ts_ms)   # Telnyx media frame (mu-law), per track
    await brain.on_call_event(event_type, payload)      # Telnyx webhook for this call
    await brain.on_hangup()                             # call over: close ASR, stop scoring
    brain.on_voice_signal(VoiceSignal)                  # AI-voice score from V1 (or Telnyx)

On each final Turn from the transcriber:
  1. append to session.turns, publish console "turn"
  2. kick the LatestOnlyScorer (caller and protected turns both; protected turns are how
     disclosure is caught). At most one guardian score in flight; newer turns collapse.
On each guardian response:
  3. policy.decide() -> Decision (state lives in session.policy_state)
  4. publish console "risk"; if coach_text: conference speak to the protected leg only,
     publish "coach"; if alert_level: publish "alert" and await caretaker.alert(...).
     The speak and the caretaker alert both run in the background, so the next score
     never waits on Telnyx or on a dial.

The coach is only offered to the policy once the protected leg (B) has joined the
conference (`conference.participant.joined` for session.b_ccid, forwarded by call_flow),
so a tip that cannot play never uses up the cooldown or the line. call_flow sends
`emberz.protected_joined` at the same moment; either one sets brain.b_joined. Until then
the protected-side track (A's outbound: the consent notice and ringback) is fed to the ASR
as silence, so the notice never becomes a "protected" turn.

AI voice: session.ai_voice is the highest of the newest score from each source (V1
detector, Telnyx, timing). The Telnyx deepfake result also reaches call_flow, which
publishes the console "voice" event for it; the brain only updates the score then.

Coach command (Telnyx conference speak):
  POST /v2/conferences/{id}/actions/speak  {payload, voice, language, call_control_ids: [B]}
  `call_control_ids` lists the participants who hear it; empty would mean everyone, so we
  never send without B. https://developers.telnyx.com/api-reference/conference-commands/speak-text-to-conference-participants
Telnyx AI-voice result: webhook `call.deepfake_detection.result` with payload.score (0-1)
  and payload.result (e.g. real|fake|silence_timeout).
  https://developers.telnyx.com/api-reference/callbacks/call-deepfake-detection-result

Latency: for every coach line, ms from the moment the trigger turn reached the brain to
the moment the speak command was sent (and when Telnyx acknowledged it) is logged and kept
in `brain.coach_latencies`.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import os
import time
from typing import Any, Callable, Optional

from contracts.models import AiVoice, ConsoleEvent, GuardianRequest, GuardianResponse, Turn, VoiceSignal
from line import guardian_client
from line.guardian_client import LatestOnlyScorer
from line.policy import Decision, PolicyContext, PolicyState, decide
from line.session import CallSession

log = logging.getLogger("emberz.brain")

MULAW_SILENCE = 0xFF       # same value as line/audio.py (decodes to 0)
PROTECTED_TRACK = "outbound"  # A's outbound track = the conference mix = the protected person (line/asr.py)
COACH_VOICE = os.environ.get("COACH_VOICE") or os.environ.get("TELNYX_TTS_VOICE", "Telnyx.KokoroTTS.af")
COACH_LANGUAGE = os.environ.get("COACH_LANGUAGE", "en-US")
SCORE_SPEAKERS = ("caller", "protected")
WINDOW = int(os.environ.get("GUARDIAN_WINDOW", str(guardian_client.DEFAULT_WINDOW)))


def _default_transcriber_factory(call_id, on_turn, loop, on_partial=None):
    from line.asr import AsyncCallTranscriber  # written in parallel (L2 part 1); import lazily

    return AsyncCallTranscriber(call_id, on_turn, loop, on_partial=on_partial)


async def _maybe_await(x: Any) -> Any:
    return await x if inspect.isawaitable(x) else x


def make_brain_factory(caretaker: Any, transcriber_factory=None, score_fn=None) -> Callable[..., "CallBrain"]:
    """Adapter for line/call_flow.py, whose MakeBrain is (session, telnyx, hub) -> brain.

    Builds the env-configured guardian client now (server start-up), so a bad
    GUARDIAN_BACKEND, a missing GUARDIAN_MODEL or FIREWORKS_API_KEY fails before any call.
    """
    if score_fn is None:
        score_fn = guardian_client.default_client().score

    def make_brain(session: CallSession, telnyx: Any, hub: Any) -> "CallBrain":
        return CallBrain(session, telnyx, hub, caretaker, transcriber_factory=transcriber_factory, score_fn=score_fn)

    return make_brain


class CallBrain:
    def __init__(
        self,
        session: CallSession,
        telnyx: Any,
        hub: Any,
        caretaker: Any,
        transcriber_factory: Optional[Callable[..., Any]] = None,
        score_fn: Optional[Callable[[GuardianRequest], Any]] = None,
    ):
        self.session = session
        self.telnyx = telnyx
        self.hub = hub
        self.caretaker = caretaker
        self.transcriber_factory = transcriber_factory or _default_transcriber_factory
        self.score_fn = score_fn or guardian_client.score
        self.transcriber: Any = None
        try:
            self.loop: Optional[asyncio.AbstractEventLoop] = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = None
        self.scorer = LatestOnlyScorer(session, self.score_fn, self._on_result, window=WINDOW)
        self._turn_seen: dict[int, float] = {}         # turn number -> monotonic time it reached us
        self._bg: set[asyncio.Task] = set()
        self.coach_latencies: list[dict] = []          # {turn, sent_ms, ack_ms}
        self.decisions: list[tuple[int, Decision]] = []
        self.b_joined = False                           # protected leg is in the conference
        self._voice_by_source: dict[str, float] = {}
        self._tr_lock = asyncio.Lock()
        self._asr_failed = False
        self._tr_closed = False
        if session.policy_state is None:
            session.policy_state = PolicyState()

    # ------------------------------------------------------------ server hooks
    async def on_media(self, track: str, payload_bytes: bytes, ts_ms: int) -> None:
        if self.session.ended:
            return
        if self.transcriber is None:
            if self._asr_failed:
                return
            async with self._tr_lock:
                if self.transcriber is None and not self._asr_failed:
                    self.loop = self.loop or asyncio.get_running_loop()
                    # Off the loop: the first AsyncCallTranscriber loads the ONNX model (~1.8 s
                    # measured), which would freeze every call and webhook if done here.
                    try:
                        tr = await asyncio.to_thread(
                            self.transcriber_factory, self.session.call_id, self.on_turn, self.loop, on_partial=None
                        )
                    except Exception:  # e.g. model files missing: log once, keep the call up
                        self._asr_failed = True
                        log.exception("call %s: ASR could not start; this call is not transcribed", self.session.call_id)
                        return
                    self.transcriber = tr
                    if self.session.ended:  # hung up while the model loaded
                        await self._close_transcriber()
                        return
            if self.transcriber is None or self.session.ended:
                return
        if track == PROTECTED_TRACK and not self.b_joined:
            # Before B joins, A's outbound track is the consent notice and ringback, not the
            # protected person: feed silence (same length, so both sides keep one clock).
            payload_bytes = bytes([MULAW_SILENCE]) * len(payload_bytes)
        # ts_ms is NOT passed on: Telnyx's media `timestamp` is RTP ticks (8 kHz), not ms, and
        # starts at a random offset. Fed as ms it padded every 20 ms frame with ~140 ms of
        # silence (8x the audio, words split apart; live call 2026-11-02 produced no turns).
        # Frames are already reordered by chunk and arrive continuously on both tracks.
        await _maybe_await(self.transcriber.feed(track, payload_bytes, None))

    async def on_call_event(self, event_type: str, payload: dict) -> None:
        s = self.session
        if event_type == "call.deepfake_detection.result":
            if payload.get("_role") not in (None, "A"):  # call_flow tags the leg; only the caller's voice counts
                return
            score = payload.get("score")
            if payload.get("result") == "silence_timeout" or score is None:
                return
            try:
                score = max(0.0, min(1.0, float(score)))
            except (TypeError, ValueError):
                return
            # call_flow already published the console "voice" event for this result.
            self._set_voice(VoiceSignal(call_id=s.call_id, at_ms=s.now_ms(), ai_score=score, model="telnyx"), publish=False)
        elif event_type == "call.deepfake_detection.error":
            log.warning("call %s: Telnyx deepfake detection error: %s", s.call_id, payload.get("error_message"))
        elif event_type == "emberz.protected_joined" or (
            event_type == "conference.participant.joined" and s.b_ccid and payload.get("call_control_id") == s.b_ccid
        ):
            self.b_joined = True
        elif event_type in ("conference.participant.left", "call.hangup") and s.b_ccid and payload.get("call_control_id") == s.b_ccid:
            self.b_joined = False

    async def on_hangup(self) -> None:
        self.session.ended = True
        self.scorer.cancel()
        await self._close_transcriber()
        if self.coach_latencies:
            log.info("call %s: coach latencies %s", self.session.call_id, self.coach_latencies)

    async def _close_transcriber(self) -> None:
        tr = self.transcriber
        if tr is None or self._tr_closed:
            return
        self._tr_closed = True
        try:
            # AsyncCallTranscriber.close() joins its worker thread; aclose() does it off-loop.
            closer = getattr(tr, "aclose", None) or tr.close
            await _maybe_await(closer())
        except Exception:
            log.exception("call %s: transcriber close failed", self.session.call_id)

    def on_voice_signal(self, sig: VoiceSignal) -> None:
        """AI-voice score from V1 (or any source). Safe from the loop thread or a worker thread."""
        self._on_loop(self._set_voice, sig)

    def _set_voice(self, sig: VoiceSignal, publish: bool = True) -> None:
        source = {"telnyx": "telnyx", "timing": "timing"}.get(sig.model, "detector")
        self._voice_by_source[source] = sig.ai_score  # newest score per source
        best = max(self._voice_by_source, key=self._voice_by_source.__getitem__)
        self.session.ai_voice = AiVoice(score=self._voice_by_source[best], source=best)
        if publish:
            self._publish("voice", **sig.model_dump())

    # ------------------------------------------------------------ turns
    def on_turn(self, turn: Turn | dict) -> None:
        """Transcriber callback. Safe from the loop thread or a worker thread."""
        self._on_loop(self._accept_turn, turn)

    def _on_loop(self, fn: Callable[..., None], *args: Any) -> None:
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if self.loop is not None and running is not self.loop:
            self.loop.call_soon_threadsafe(fn, *args)
        else:
            fn(*args)

    def _accept_turn(self, turn: Turn | dict) -> None:
        if not isinstance(turn, Turn):
            turn = Turn.model_validate(turn)
        if not turn.text.strip():
            return  # an empty segment: nothing to show or score
        if turn.turn in self._turn_seen:
            log.warning("call %s: duplicate turn %d ignored", self.session.call_id, turn.turn)
            return
        self._turn_seen[turn.turn] = time.monotonic()
        self.session.turns.append(turn)
        self._publish("turn", **turn.model_dump())  # flushed after hang-up: kept, not scored
        if turn.speaker in SCORE_SPEAKERS and not self.session.ended:
            self.scorer.kick()

    # ------------------------------------------------------------ guardian result
    async def _on_result(self, req: GuardianRequest, resp: GuardianResponse, latency_ms: int) -> None:
        s = self.session
        trigger = req.turns[-1].turn
        ctx = PolicyContext(
            turn=trigger,
            now_ms=s.now_ms(),
            profile=s.profile,
            caller_known=s.caller_known,
            ai_voice_score=s.ai_voice.score if s.ai_voice else None,
            rules=tuple(s.rules),
            coach_available=self.coach_available(),
        )
        decision, s.policy_state = decide(s.policy_state or PolicyState(), resp, ctx)
        self.decisions.append((trigger, decision))
        if "coach_unavailable" in decision.reasons:
            log.warning("call %s: coach wanted on turn %d but B is not in the conference yet", s.call_id, trigger)

        if decision.coach_text:  # first, so the speak goes out before anything else
            self._background(self._coach(decision.coach_text, trigger))
        self._publish(
            "risk",
            turn=trigger,
            response=resp.model_dump(),
            latency_ms=latency_ms,
            display_risk=decision.display_risk,
            suggest_hangup=decision.suggest_hangup,
            reasons=list(decision.reasons),
        )
        if decision.alert_level:
            self._publish(
                "alert",
                level=decision.alert_level,
                category=decision.category,
                caretaker_note=decision.caretaker_note,
                turn=trigger,
            )
            self._background(self._alert_caretaker(decision.alert_level, decision.caretaker_note))

    def coach_available(self) -> bool:
        s = self.session
        return bool(s.conference_id and s.b_ccid and self.b_joined and not s.ended)

    async def _coach(self, text: str, trigger: int) -> None:
        s = self.session
        if not s.conference_id or not s.b_ccid or s.ended:
            log.warning("call %s: coach skipped, no conference or B leg (%r)", s.call_id, text)
            self._publish("coach", text=text, turn=trigger, sent=False)
            return
        seen = self._turn_seen.get(trigger, time.monotonic())
        sent = time.monotonic()
        try:
            await self.telnyx.conference_action(
                s.conference_id,
                "speak",
                payload=text,
                voice=COACH_VOICE,
                language=COACH_LANGUAGE,
                call_control_ids=[s.b_ccid],  # protected person only; the caller hears nothing
            )
            ok = True
        except Exception:
            log.exception("call %s: coach speak failed", s.call_id)
            ok = False
        ack = time.monotonic()
        lat = {"turn": trigger, "sent_ms": round((sent - seen) * 1000), "ack_ms": round((ack - seen) * 1000), "ok": ok}
        self.coach_latencies.append(lat)
        log.info("call %s: coach on turn %d, trigger->sent %d ms, trigger->ack %d ms", s.call_id, trigger, lat["sent_ms"], lat["ack_ms"])
        self._publish("coach", text=text, turn=trigger, sent=ok)

    async def _alert_caretaker(self, level: str, note: str) -> None:
        try:
            result = await self.caretaker.alert(self.session, level, note)
            log.info("call %s: caretaker alert %s -> %s", self.session.call_id, level, result)
        except Exception:
            log.exception("call %s: caretaker alert failed", self.session.call_id)

    # ------------------------------------------------------------ helpers
    def _background(self, coro) -> None:
        task = asyncio.ensure_future(coro)
        self._bg.add(task)
        task.add_done_callback(self._bg.discard)

    async def drain(self) -> None:
        """Wait until scoring and background actions finish (tests, clean shutdown)."""
        await self.scorer.wait_idle()
        while self._bg:
            await asyncio.gather(*list(self._bg), return_exceptions=True)

    def _publish(self, type_: str, **data: Any) -> None:
        try:
            ev = ConsoleEvent(type=type_, call_id=self.session.call_id, at_ms=max(0, self.session.now_ms()), data=data)
            res = self.hub.publish(ev)
            if inspect.isawaitable(res):
                self._background(res)
        except Exception:
            log.exception("call %s: console publish %s failed", self.session.call_id, type_)

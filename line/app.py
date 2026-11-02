"""The live Emberz line, wired end to end. One command:

    .venv/bin/python -m line.app            # uvicorn on 127.0.0.1:8000 (EMBERZ_HOST / EMBERZ_PORT)

Builds: TelnyxClient -> CallFlow (line/server.create_app) with
  - LiveBrain per call = CallBrain (ASR -> guardian -> policy -> coach/caretaker/console)
    + TimingScorer (AI-voice from turn timing) + LiveVoiceDetector (only if the trained head
    file exists; else timing + Telnyx only) + CallRecorder (recordings/<call_id>/)
  - CaretakerFlow (alert call to C, press 1 to listen, barge) + summary at hang-up
  - AppHub = ConsoleHub that also feeds the recorder and, if MATT_CONSOLE_URL is set,
    forwards every event to Matt's console (line/matt_bridge.py)
  - routes: /webhooks/telnyx, /media/{call_id}, /console/*, /actions/{call_id}/{action},
    /callers/swml/* (SignalWire AI callers), /healthz
Guardian backend from GUARDIAN_BACKEND (stub | standin | deployment | matt). See line/README_live.md.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, Optional

from contracts.models import ConsoleEvent, Turn
from line import guardian_client
from line.brain import CallBrain
from line.caretaker import CaretakerFlow
from line.config import REPO, load_profiles, load_profiles_or_env, load_settings
from line.console_hub import ConsoleHub, add_cors, console_router
from line.matt_bridge import actions_router, forwarder_from_env

log = logging.getLogger("emberz.app")
LINE_PROFILES = Path(__file__).resolve().parent / "profiles.json"


class AppHub(ConsoleHub):
    """ConsoleHub + taps: callables run on every published event (recorder, Matt forwarder, printer)."""

    def __init__(self) -> None:
        super().__init__()
        self.taps: list[Callable[[ConsoleEvent], Any]] = []

    def publish(self, event):  # type: ignore[override]
        ev = super().publish(event)
        for tap in list(self.taps):
            try:
                tap(ev)
            except Exception:
                log.exception("console tap failed")
        return ev


class LiveBrain(CallBrain):
    """CallBrain + timing AI-voice score + optional w2v detector + recorder + 'marked safe'."""

    voice_detector_ok = False   # set by build_app after warmup

    def __init__(self, session, telnyx, hub, caretaker, recorders: dict, record: bool = True, **kw):
        super().__init__(session, telnyx, hub, caretaker, **kw)
        from voice.timing import TimingScorer

        self.timing = TimingScorer(session.call_id)
        self.detector = None
        if LiveBrain.voice_detector_ok:
            try:
                from voice.live import LiveVoiceDetector

                self.detector = LiveVoiceDetector(session.call_id, self.on_voice_signal,
                                                  now_ms=session.now_ms, loop=self.loop)
            except Exception:
                log.exception("call %s: voice detector unavailable", session.call_id)
        self.recorder = None
        if record:
            try:
                from line.recorder import CallRecorder

                self.recorder = CallRecorder(
                    session.call_id, t0=session.started_monotonic,
                    numbers={"caller": session.caller_number, "guarded": session.guarded_number},
                    context={"profile": session.profile, "rules": list(session.rules),
                             "caller_known": session.caller_known})
                recorders[session.call_id] = self.recorder
            except Exception:
                log.exception("call %s: recorder unavailable", session.call_id)

    @property
    def safe(self) -> bool:
        return bool(getattr(self.session, "marked_safe", False))

    async def on_media(self, track, payload_bytes, ts_ms):
        if self.recorder is not None and not self.session.ended:
            try:
                self.recorder.feed(track, payload_bytes, timestamp=ts_ms or None)
            except Exception:
                log.exception("recorder feed failed")
        if self.detector is not None and track == "inbound":
            try:
                self.detector.push_mulaw(payload_bytes)
            except Exception:
                log.exception("voice detector failed; disabled for this call")
                self.detector = None
        await super().on_media(track, payload_bytes, ts_ms)

    def _accept_turn(self, turn):
        n = len(self.session.turns)
        super()._accept_turn(turn)
        if len(self.session.turns) == n:
            return
        t: Turn = self.session.turns[-1]
        if self.recorder is not None:
            self.recorder.add_turn(t)
        try:
            sig = self.timing.update(t)
            if sig is not None:
                self._set_voice(sig)
        except Exception:
            log.exception("timing scorer failed")

    async def _coach(self, text, trigger):
        if self.safe:
            return
        await super()._coach(text, trigger)

    async def _alert_caretaker(self, level, note):
        if self.safe:
            return
        await super()._alert_caretaker(level, note)

    def _publish(self, type_, **data):
        if self.safe and type_ in ("alert", "coach"):
            return
        super()._publish(type_, **data)

    async def on_hangup(self):
        await super().on_hangup()
        if self.detector is not None:
            try:
                self.detector.close()
            except Exception:
                pass


def _profiles(settings):
    path = Path(os.environ["EMBERZ_PROFILES"]) if os.environ.get("EMBERZ_PROFILES") else None
    if path is None and LINE_PROFILES.exists():
        path = LINE_PROFILES
    return load_profiles(path) if path else load_profiles_or_env(settings)


def build_app(
    telnyx: Any = None,
    settings: Any = None,
    profiles: Any = None,
    transcriber_factory: Any = None,
    score_fn: Any = None,
    summarizer: Any = None,
    open_console: bool = False,
    record: bool = True,
    voice_detector: Optional[bool] = None,
    taps: tuple = (),
):
    """The real app. Tests/simulation pass FakeTelnyx, a fake transcriber or summarizer."""
    from line.server import create_app

    settings = settings or load_settings()
    profiles = profiles if profiles is not None else _profiles(settings)
    hub = AppHub()
    recorders: dict = {}
    caretaker = CaretakerFlow(None, hub, lambda call_id: None, **({"summarizer": summarizer} if summarizer else {}))
    if score_fn is None:
        score_fn = guardian_client.default_client().score  # fails fast on a bad backend / missing key

    if voice_detector is None:
        voice_detector = os.environ.get("EMBERZ_VOICE_DETECTOR", "1") != "0"
    if voice_detector:
        from voice.live import DEFAULT_HEAD

        if Path(DEFAULT_HEAD).exists():
            from voice.live import warmup

            LiveBrain.voice_detector_ok = warmup()
        else:
            log.warning("no voice head at %s: AI-voice = timing + Telnyx only", DEFAULT_HEAD)
            LiveBrain.voice_detector_ok = False
    else:
        LiveBrain.voice_detector_ok = False

    def make_brain(session, tel, hub_):
        return LiveBrain(session, tel, hub_, caretaker, recorders, record=record,
                         transcriber_factory=transcriber_factory, score_fn=score_fn)

    routers = [console_router(hub, caretaker, open_access=open_console)]
    try:
        from callers.signalwire.swml_router import router as swml_router

        routers.append(swml_router)
    except Exception as e:
        log.info("SignalWire SWML router not mounted: %s", e)

    app = create_app(telnyx=telnyx, make_brain=make_brain, caretaker=caretaker, hub=hub,
                     extra_routers=routers, settings=settings, profiles=profiles)
    flow = app.state.flow
    caretaker.telnyx, caretaker.sessions_lookup = app.state.telnyx, flow.sessions.get
    app.include_router(actions_router(flow, caretaker, hub, open_access=open_console))
    add_cors(app)

    def record_tap(ev: ConsoleEvent) -> None:
        rec = recorders.get(ev.call_id)
        if rec is None:
            return
        rec.add_event(ev)
        if ev.type == "call_ended":
            recorders.pop(ev.call_id, None)
            meta = rec.close(reason=str(ev.data.get("reason", "")))
            log.info("call %s recorded: %s (%s s)", ev.call_id, rec.dir, meta.get("duration_s"))

    hub.taps.append(record_tap)
    fwd = forwarder_from_env()
    if fwd is not None:
        hub.taps.append(fwd)
    hub.taps.extend(taps)
    app.state.caretaker, app.state.recorders, app.state.forwarder = caretaker, recorders, fwd
    return app


def main() -> None:
    import sys

    import uvicorn

    from line.server import live_problems

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    settings = load_settings()
    problems = live_problems(settings)
    if not os.environ.get("CONSOLE_TOKEN"):
        problems.append("CONSOLE_TOKEN is not set (console + /actions would refuse every request)")
    if problems:
        print("Cannot start:\n  " + "\n  ".join(problems))
        sys.exit(2)
    if os.environ.get("EMBERZ_ASR_PRELOAD", "1") != "0":
        from line.asr import load_recognizer

        load_recognizer()  # ~2 s, here instead of on the first call
    app = build_app(settings=settings)
    print(f"Emberz line up: guardian={guardian_client.default_client().backend} "
          f"model={guardian_client.default_client().model} webhooks={settings.webhook_url()} "
          f"forward={'on' if app.state.forwarder else 'off'}")
    uvicorn.run(app, host=os.environ.get("EMBERZ_HOST", "127.0.0.1"), port=int(os.environ.get("EMBERZ_PORT", "8000")))


if __name__ == "__main__":
    main()

"""End-to-end local simulation of the live line: no phone, no Telnyx, no tunnel.

    .venv/bin/python -m line.simulate_call --scene bank                    # stub guardian, real ASR, real time
    GUARDIAN_BACKEND=standin .venv/bin/python -m line.simulate_call --scene all
    .venv/bin/python -m line.simulate_call --scene all --text --fast        # no audio: scripted turns (tests)

What runs is the real app (line/app.build_app) with FakeTelnyx and signatures off:
  1. synthetic Telnyx webhooks: inbound call.initiated -> A answered -> notice ended -> B answered
     -> B joined the conference (the same sequence a real call produces)
  2. real audio: macOS `say` -> ffmpeg 8 kHz mu-law -> 20 ms frames, both tracks
     (inbound = caller, outbound = protected) over WS /media/{call_id}, in real time
  3. when the caretaker is dialed, C "answers" and presses 1 (monitor join)
  4. hang-up -> call_ended -> summary
Prints every console event and the latency from the end of the trigger line's audio to the
coach speak command (FakeTelnyx conference speak). --text skips audio and injects the scene
lines as ASR turns (fast, for tests and machines without `say`).
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

FRAME = 160            # 20 ms of 8 kHz mu-law
SILENCE = b"\xff" * FRAME
GUARDED, PROTECTED, CARETAKER = "+15555550100", "+15555550101", "+15555550102"
GRANDPA, STRANGER = "+15555550111", "+15555550199"

# (speaker, text, say voice). The trigger line is the one whose audio end starts the latency clock.
SCENES: dict[str, dict[str, Any]] = {
    "bank": {
        "caller": STRANGER, "expect_alert": True, "trigger": 4,
        "lines": [
            ("caller", "Hello, this is the fraud department at your bank. We noticed a suspicious charge on your account.", "Fred"),
            ("protected", "Oh no. What happened?", "Samantha"),
            ("caller", "Someone tried to spend nine hundred dollars. To block it we have to verify you. We just sent a text to your phone.", "Fred"),
            ("protected", "Okay, I got a text message.", "Samantha"),
            ("caller", "Great. Can you read me the code?", "Fred"),
        ],
    },
    "grandpa": {
        "caller": GRANDPA, "expect_alert": False, "trigger": None,
        "lines": [
            ("caller", "Hi sweetheart, it's Grandpa. Just calling to see how you are doing.", "Daniel"),
            ("protected", "Hi Grandpa! I'm doing well, thank you.", "Samantha"),
            ("caller", "Are we still on for dinner on Sunday?", "Daniel"),
            ("protected", "Yes, see you at six.", "Samantha"),
        ],
    },
    "pharmacy": {
        "caller": STRANGER, "expect_alert": False, "trigger": None,
        "lines": [
            ("caller", "Hello. This is an automated reminder from your pharmacy. Your prescription is ready for pickup.", "Samantha"),
            ("protected", "Oh, good. Thank you.", "Moira"),
            ("caller", "Your prescription will be held for seven days. Thank you, and goodbye.", "Samantha"),
        ],
    },
}


# ------------------------------------------------------------------ audio
def tts_mulaw(text: str, voice: str, cache: Path) -> bytes:
    """macOS say -> ffmpeg -> raw 8 kHz mono mu-law bytes (cached per text+voice)."""
    key = uuid.uuid5(uuid.NAMESPACE_URL, f"{voice}|{text}").hex
    out = cache / f"{key}.ul"
    if not out.exists():
        aiff = cache / f"{key}.aiff"
        subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(aiff), "-ar", "8000", "-ac", "1",
                        "-f", "mulaw", str(out)], check=True)
        aiff.unlink(missing_ok=True)
    return out.read_bytes()


def can_synthesize() -> bool:
    return bool(shutil.which("say") and shutil.which("ffmpeg"))


# ------------------------------------------------------------------ fakes
class TimedTelnyx:
    """FakeTelnyx that also notes when each conference speak (coach) was sent."""

    def __init__(self) -> None:
        from line.telnyx_client import FakeTelnyx

        self.inner = FakeTelnyx()
        self.speaks: list[tuple[float, dict]] = []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    async def conference_action(self, conference_id, action, **body):
        if action == "speak":
            self.speaks.append((time.monotonic(), body))
        return await self.inner.conference_action(conference_id, action, **body)


class ScriptTranscriber:
    """Stands in for AsyncCallTranscriber in --text mode: audio is ignored, turns are injected."""

    instances: dict[str, "ScriptTranscriber"] = {}

    def __init__(self, call_id, on_turn, loop, on_partial=None):
        self.call_id, self.on_turn = call_id, on_turn
        ScriptTranscriber.instances[call_id] = self

    def feed(self, track, payload, ts_ms=None):
        pass

    async def aclose(self):
        pass


async def local_summary(session, alerts, *, hub=None, caretaker_mode=None, **_):
    alerts = list(alerts)
    text = (f"{len(session.turns)} turns, {len(alerts)} alert(s)"
            + (f": {alerts[0].get('caretaker_note') or alerts[0].get('category')}" if alerts else "")
            + (f"; caretaker {caretaker_mode}" if caretaker_mode else "") + ". (local summary, no model)")
    if hub is not None:
        hub.emit(session.call_id, "summary", session.now_ms(), text=text)
    return text


# ------------------------------------------------------------------ run one scene
@dataclass
class SceneResult:
    scene: str
    call_id: str
    events: list = field(default_factory=list)
    coach_latency_ms: Optional[int] = None
    line_ends: list = field(default_factory=list)
    speaks: list = field(default_factory=list)
    action_replies: list = field(default_factory=list)

    def of(self, t: str) -> list:
        return [e for e in self.events if e.type == t]


def _wh(client, event_type: str, payload: dict) -> None:
    r = client.post("/webhooks/telnyx", json={"data": {"event_type": event_type, "id": str(uuid.uuid4()),
                                                       "payload": payload}})
    assert r.status_code == 200, r.text


def run_scene(name: str, *, text_mode: bool = False, fast: bool = False, quiet: bool = False,
              real_summary: bool = False, cache: Optional[Path] = None, score_fn: Any = None,
              actions: Optional[dict] = None) -> SceneResult:
    """actions: {line_index: "safe"|"hangup"|...} -> POST /actions/{call_id}/{action} right before that line."""
    from fastapi.testclient import TestClient

    from line.app import build_app
    from line.call_flow import encode_client_state
    from line.config import GuardedLine, Settings

    scene = SCENES[name]
    os.environ.setdefault("CONSOLE_TOKEN", "sim-token")
    tel = TimedTelnyx()
    t_start = time.monotonic()
    result = SceneResult(name, "")
    lock = threading.Lock()

    def tap(ev):
        with lock:
            result.events.append(ev)
        if not quiet:
            d = dict(ev.data)
            if ev.type == "risk":
                d = {k: d.get(k) for k in ("turn", "display_risk", "latency_ms", "reasons")} | {
                    "category": (ev.data.get("response") or {}).get("category")}
            print(f"  [{time.monotonic() - t_start:6.2f}s] {ev.type:16s} {json.dumps(d, default=str)[:180]}", flush=True)

    settings = Settings(telnyx_api_key="sim", public_base_url="http://localhost:8000", skip_verify=True,
                        media_verify="off")
    profiles = {GUARDED: GuardedLine(
        profile="elder",
        rules=["Never share codes or passwords", "Alert me about any money talk", f"Allowed: Grandpa ({GRANDPA})"],
        allowed_contacts={GRANDPA: "Grandpa"}, protected_number=PROTECTED, caretaker_number=CARETAKER)}
    app = build_app(telnyx=tel, settings=settings, profiles=profiles,
                    transcriber_factory=ScriptTranscriber if text_mode else None,
                    summarizer=None if real_summary else local_summary, open_console=True, score_fn=score_fn,
                    record=False, voice_detector=False if text_mode else None, taps=(tap,))
    sid = uuid.uuid4().hex[:10]
    call_id, a = "c_" + sid, "sim-A-" + sid
    result.call_id = call_id

    def leg(role, ccid, **extra):
        return {"call_control_id": ccid, "call_session_id": sid, "client_state": encode_client_state(role, call_id)} | extra

    with TestClient(app) as client:
        flow = app.state.flow
        _wh(client, "call.initiated", {"call_control_id": a, "call_session_id": sid, "from": scene["caller"],
                                       "to": GUARDED, "direction": "incoming", "state": "parked"})
        session = flow.sessions[call_id]
        _wh(client, "call.answered", leg("A", a))
        _wh(client, "call.speak.ended", leg("A", a, status="completed"))
        b = session.b_ccid
        _wh(client, "call.answered", leg("B", b))
        _wh(client, "conference.participant.joined", {"call_control_id": b, "conference_id": session.conference_id})

        c_done = False

        def caretaker_answers():
            nonlocal c_done
            if not c_done and session.caretaker_state == "ringing" and session.c_ccid:
                c_done = True
                c = session.c_ccid
                cs = encode_client_state("C", call_id)  # flow routes C by client_state role
                _wh(client, "call.answered", {"call_control_id": c, "client_state": cs})
                _wh(client, "call.gather.ended", {"call_control_id": c, "client_state": cs, "digits": "1",
                                                  "status": "valid"})

        if text_mode:
            from contracts.models import Turn

            ms = 0
            for i, (spk, text, _v) in enumerate(scene["lines"], 1):
                if actions and (i - 1) in actions:
                    result.action_replies.append(client.post(f"/actions/{call_id}/{actions[i - 1]}").json())
                if session.ended:
                    break
                dur = 60 * len(text)
                tr = ScriptTranscriber.instances[call_id] if call_id in ScriptTranscriber.instances else None
                if tr is None:  # created on the first media frame
                    with client.websocket_connect(f"/media/{call_id}") as ws:
                        ws.send_text(json.dumps({"event": "media", "media": {
                            "track": "inbound", "chunk": 1, "timestamp": 0,
                            "payload": base64.b64encode(SILENCE).decode()}}))
                        ws.send_text(json.dumps({"event": "stop"}))
                    for _ in range(100):
                        if call_id in ScriptTranscriber.instances:
                            break
                        time.sleep(0.01)
                    tr = ScriptTranscriber.instances[call_id]
                result.line_ends.append(time.monotonic())
                tr.on_turn(Turn(call_id=call_id, turn=i, speaker=spk, text=text, t_start_ms=ms + 400,
                                t_end_ms=ms + 400 + dur, gap_ms=400 if i > 1 else None))
                ms += 400 + dur
                time.sleep(0.05 if fast else 1.0)
                caretaker_answers()
            deadline = time.monotonic() + (8 if scene["expect_alert"] else 2)
            while time.monotonic() < deadline and scene["expect_alert"] and not tel.speaks:
                time.sleep(0.02)
            time.sleep(0.2)
            caretaker_answers()
        else:
            cache = cache or Path(tempfile.gettempdir()) / "emberz_sim_tts"
            cache.mkdir(parents=True, exist_ok=True)
            # timeline of (inbound, outbound) frames; remember the last frame index of each line
            frames: list[tuple[bytes, bytes]] = [(SILENCE, SILENCE)] * 25
            ends: list[int] = []
            for spk, text, voice in scene["lines"]:
                audio = tts_mulaw(text, voice, cache)
                for k in range(0, len(audio), FRAME):
                    chunk = audio[k:k + FRAME].ljust(FRAME, b"\xff")
                    frames.append((chunk, SILENCE) if spk == "caller" else (SILENCE, chunk))
                ends.append(len(frames) - 1)
                frames.extend([(SILENCE, SILENCE)] * 50)  # 1 s between lines
            frames.extend([(SILENCE, SILENCE)] * 150)     # 3 s tail: endpointing + guardian time
            end_at = {idx: n for n, idx in enumerate(ends)}
            result.line_ends = [0.0] * len(ends)
            with client.websocket_connect(f"/media/{call_id}") as ws:
                ws.send_text(json.dumps({"event": "connected", "version": "1.0.0"}))
                ws.send_text(json.dumps({"event": "start", "stream_id": "sim", "start": {
                    "call_control_id": a, "media_format": {"encoding": "PCMU", "sample_rate": 8000, "channels": 1}}}))
                t0 = time.monotonic()
                for i, (inb, outb) in enumerate(frames):
                    if not fast:
                        delay = t0 + i * 0.02 - time.monotonic()
                        if delay > 0:
                            time.sleep(delay)
                    for track, payload in (("inbound", inb), ("outbound", outb)):
                        ws.send_text(json.dumps({"event": "media", "media": {
                            "track": track, "chunk": i + 1, "timestamp": i * 20,
                            "payload": base64.b64encode(payload).decode()}}))
                    if i in end_at:
                        result.line_ends[end_at[i]] = time.monotonic()
                    if i % 25 == 0:
                        caretaker_answers()
                deadline = time.monotonic() + (6 if scene["expect_alert"] else 1)
                while scene["expect_alert"] and not tel.speaks and time.monotonic() < deadline:
                    time.sleep(0.02)
                caretaker_answers()
                ws.send_text(json.dumps({"event": "stop"}))
            time.sleep(0.3)

        _wh(client, "call.hangup", leg("A", a, hangup_cause="normal_clearing"))
        deadline = time.monotonic() + (20 if real_summary else 3)
        while time.monotonic() < deadline and not result.of("summary") and session.turns:
            time.sleep(0.05)
        result.speaks = tel.speaks

    if scene["trigger"] is not None and tel.speaks and result.line_ends:
        trig_end = result.line_ends[scene["trigger"]]
        after = [t for t, _ in tel.speaks if t >= trig_end - 0.05]
        if after:
            result.coach_latency_ms = round((after[0] - trig_end) * 1000)
    return result


def main() -> None:
    import logging

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scene", default="all", choices=[*SCENES, "all"])
    ap.add_argument("--text", action="store_true", help="inject scripted turns instead of audio + ASR")
    ap.add_argument("--fast", action="store_true", help="no real-time pacing")
    ap.add_argument("--real-summary", action="store_true", help="summary via Fireworks (1 request per call)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    if not args.verbose:  # the skip-verify warning repeats on every synthetic webhook
        for name in ("emberz.server", "emberz.webhook_verify"):
            logging.getLogger(name).setLevel(logging.ERROR)
    os.environ.setdefault("GUARDIAN_BACKEND", "stub")
    if not args.text and not can_synthesize():
        raise SystemExit("needs macOS `say` and ffmpeg for audio; use --text")
    if not args.text:
        from line.asr import load_recognizer

        load_recognizer()
    from line import guardian_client

    gc = guardian_client.default_client()
    print(f"guardian backend={gc.backend} model={gc.model}")
    ok = True
    for name in (SCENES if args.scene == "all" else [args.scene]):
        print(f"\n=== scene {name} ({'text' if args.text else 'audio'}) ===")
        r = run_scene(name, text_mode=args.text, fast=args.fast, real_summary=args.real_summary)
        alerts, coaches = r.of("alert"), [e for e in r.of("coach") if e.data.get("sent")]
        want = SCENES[name]["expect_alert"]
        good = bool(alerts) == want
        ok &= good
        lat = f"{r.coach_latency_ms} ms" if r.coach_latency_ms is not None else "n/a"
        print(f"--- {name}: turns={len(r.of('turn'))} alerts={len(alerts)} coach_sent={len(coaches)} "
              f"caretaker={[e.data.get('mode') for e in r.of('caretaker_joined')]} "
              f"trigger_audio_end->coach_speak={lat} expected_alert={want} -> {'PASS' if good else 'FAIL'}")
    if gc.backend != "stub":
        print(f"guardian last_error={gc.last_error}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

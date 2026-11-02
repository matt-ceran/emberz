"""Guided L1 hour-0 checks (PLAN.md L1). Writes line/hour0.json. Needs a live call: run by a human.

  1. Do deepfake detection and our media stream run together on leg A?          (automatic)
  2. Does a targeted conference `speak` stay out of the caller's audio?          (automatic + y/n)
  3. Does the track split come out as expected (inbound = caller,
     outbound = conference mix / protected person)?                              (automatic)
  4. Do the supervisor roles work (monitor hears all, is unheard; barge is heard)?   (y/n, dials C)
  5. Does the caller's number show on B's screen?                                (human)
  6. Does the notice or ringback leak into the protected person's track
     (A's outbound track before B joins)?                                        (automatic)

Before running:
  1. line/tunnel.sh                         -> note the https://...trycloudflare.com URL
  2. export PUBLIC_BASE_URL=<that URL> TELNYX_ENV_FILE=<EmberDial .env> TELNYX_PUBLIC_KEY=<portal key>
  3. .venv/bin/python -m line.repoint_demo_app --url $PUBLIC_BASE_URL --apply
  4. data/profiles.json lists the guarded number (copy line/profiles.example.json), or the .env has
     TELNYX_GUARDED_NUMBER + TELNYX_PROTECTED_NUMBER (+ TELNYX_CARETAKER_NUMBER)
Then: .venv/bin/python -m line.hour0

Costs a few cents of Telnyx usage (one inbound call, one or two outbound legs, deepfake check $0.01).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from line.call_flow import NullBrain, encode_client_state
from line.config import REPO, load_profiles_or_env, load_settings

log = logging.getLogger("emberz.hour0")
OUT = REPO / "line" / "hour0.json"
TIP = "This is a private Emberz test tip. Only the protected person should hear this."
SILENCE_DB = -50.0  # mean frame level below this counts as silence on a phone line


# --------------------------------------------------------------------------- audio helpers
def ulaw_to_pcm(b: bytes) -> np.ndarray:
    """G.711 mu-law -> int16 (no audioop in Python 3.13)."""
    u = ~np.frombuffer(b, dtype=np.uint8)
    sign, exp, mant = u & 0x80, (u >> 4) & 0x07, u & 0x0F
    mag = (((mant.astype(np.int32) << 3) + 0x84) << exp) - 0x84
    return np.where(sign != 0, -mag, mag).astype(np.int16)


def frame_db(payload: bytes) -> float:
    x = ulaw_to_pcm(payload).astype(np.float64)
    rms = math.sqrt(float(np.mean(x * x))) if len(x) else 0.0
    return 20 * math.log10(max(rms, 1.0) / 32768.0)


def window_db(frames: list[tuple[float, str, float]], track: str, t0: float, t1: float) -> Optional[float]:
    """Mean level (dBFS, energy average) of `track` frames received in [t0, t1)."""
    vals = [db for t, tr, db in frames if tr == track and t0 <= t < t1]
    if not vals:
        return None
    return round(10 * math.log10(float(np.mean([10 ** (v / 10) for v in vals]))), 1)


# --------------------------------------------------------------------------- recorder (server side)
@dataclass
class Recorder:
    frames: list[tuple[float, str, float]] = field(default_factory=list)   # (monotonic, track, dBFS)
    events: list[tuple[float, str, dict]] = field(default_factory=list)    # (monotonic, type, payload)
    loop: Optional[asyncio.AbstractEventLoop] = None
    session: Any = None
    telnyx: Any = None
    media_start: Optional[dict] = None

    def first(self, event_type: str, **match) -> Optional[float]:
        for t, et, p in self.events:
            if et == event_type and all(p.get(k) == v for k, v in match.items()):
                return t
        return None

    def make_brain(self, session, telnyx, hub):
        self.loop, self.session, self.telnyx = asyncio.get_running_loop(), session, telnyx
        rec = self

        class Hour0Brain(NullBrain):
            async def on_media(self, track, payload, ts_ms):
                rec.frames.append((time.monotonic(), track, frame_db(payload)))

            async def on_call_event(self, event_type, payload):
                rec.events.append((time.monotonic(), event_type, payload))
                if event_type == "media.start":
                    rec.media_start = {k: payload.get(k) for k in ("media_format", "_signature_ok")}

            async def on_hangup(self):
                rec.events.append((time.monotonic(), "emberz.hangup", {}))

        return Hour0Brain()


class Hour0Caretaker:
    """C leg for check 4: joins the conference as a monitor when answered."""

    def __init__(self, rec: Recorder):
        self.rec = rec

    async def on_event(self, session, event_type, payload):
        self.rec.events.append((time.monotonic(), f"C:{event_type}", payload))
        if event_type == "call.answered" and session.conference_id:
            await self.rec.telnyx.conference_action(
                session.conference_id, "join", call_control_id=payload["call_control_id"],
                supervisor_role="monitor", client_state=encode_client_state("C", session.call_id))


# --------------------------------------------------------------------------- human prompts
def ask_yn(q: str) -> Optional[bool]:
    while True:
        a = input(f"  ? {q} [y/n/s=skip] ").strip().lower()
        if a in ("y", "n", "s"):
            return {"y": True, "n": False, "s": None}[a]


def ask_text(q: str) -> str:
    return input(f"  ? {q} ").strip()


def say(msg: str) -> None:
    print(f"\n>>> {msg}")


def wait_for(cond, timeout: float, what: str) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.2)
    print(f"  (timed out waiting for {what})")
    return False


def mask(n: Optional[str]) -> Optional[str]:
    return None if not n else n[:2] + "*" * max(0, len(n) - 6) + n[-4:]


def mask_text(text: str) -> str:
    """Mask phone numbers a human typed (line/hour0.json is not gitignored)."""
    return re.sub(r"\+?\d[\d\s().-]{6,}\d", lambda m: mask(re.sub(r"\D", "", m.group())), text)


# --------------------------------------------------------------------------- analysis (pure, tested)
def analyze_auto(rec: Recorder, marks: dict[str, float]) -> dict:
    """Facts that need no human: checks 1, 3, 6 and the automatic half of 2."""
    f, out = list(rec.frames), {}
    t_stream = rec.first("streaming.started")
    df_t, df = next(((t, p) for t, et, p in rec.events if et.startswith("call.deepfake_detection")), (None, None))
    t_df = df_t
    out["deepfake_with_stream"] = {
        "stream_started": t_stream is not None,
        "deepfake_event": None if df is None else {k: df.get(k) for k in ("result", "score", "consistency", "error_message")},
        "deepfake_s_after_stream_start": None if None in (t_df, t_stream) else round(t_df - t_stream, 2),
        "frames_after_deepfake_event": None if t_df is None else sum(1 for t, _, _ in f if t > t_df),
        "media_format": rec.media_start,
    }
    out["deepfake_with_stream"]["pass"] = bool(
        t_stream is not None and df and df.get("result") and (out["deepfake_with_stream"]["frames_after_deepfake_event"] or 0) > 25)

    def pair(a: str, b: str) -> dict:
        if a not in marks or b not in marks:
            return {}
        return {"inbound_db": window_db(f, "inbound", marks[a], marks[b]),
                "outbound_db": window_db(f, "outbound", marks[a], marks[b])}

    split = {"caller_speaks": pair("caller_talk_start", "caller_talk_end"),
             "protected_speaks": pair("protected_talk_start", "protected_talk_end")}
    c, p = split["caller_speaks"], split["protected_speaks"]
    if c.get("inbound_db") is not None and p.get("outbound_db") is not None:
        split["pass"] = bool(c["inbound_db"] > (c["outbound_db"] or -99) + 10
                             and p["outbound_db"] > (p["inbound_db"] or -99) + 10)
    out["track_split"] = split

    sp = pair("tip_start", "tip_end")
    sp["silence_baseline"] = pair("quiet_start", "quiet_end")
    base = (sp["silence_baseline"] or {}).get("outbound_db")
    if sp.get("outbound_db") is not None and base is not None:
        sp["caller_audio_rose_db"] = round(sp["outbound_db"] - base, 1)
        sp["auto_pass"] = sp["caller_audio_rose_db"] < 6  # A's outbound = what the caller hears
    out["targeted_speak"] = sp

    t_ns, t_ne = rec.first("call.speak.started"), rec.first("call.speak.ended")
    b_ccid = getattr(rec.session, "b_ccid", None)
    t_bj = rec.first("conference.participant.joined", call_control_id=b_ccid) if b_ccid else None
    have_notice, have_gap = None not in (t_ns, t_ne), None not in (t_ne, t_bj)
    leak = {"notice_outbound_db": window_db(f, "outbound", t_ns, t_ne) if have_notice else None,
            "notice_inbound_db": window_db(f, "inbound", t_ns, t_ne) if have_notice else None,
            "before_b_joined_outbound_db": window_db(f, "outbound", t_ne, t_bj) if have_gap else None,
            "gap_notice_end_to_b_joined_s": round(t_bj - t_ne, 2) if have_gap else None}
    # None = not measured (missing events or no frames in the window), never "no leak".
    leak["notice_in_outbound_track"] = (None if leak["notice_outbound_db"] is None
                                        else leak["notice_outbound_db"] > SILENCE_DB)
    leak["ringback_in_outbound_track"] = (None if leak["before_b_joined_outbound_db"] is None
                                          else leak["before_b_joined_outbound_db"] > SILENCE_DB)
    leak["note"] = ("A's outbound track is what the caller hears; until B joins it is NOT the protected person. "
                    "L2 must ignore outbound audio before conference.participant.joined for B.")
    out["notice_ringback_leak"] = leak
    return out


def _any_true(*vals: Optional[bool]) -> Optional[bool]:
    """True if any is True, None if nothing was measured, else False."""
    if any(v is True for v in vals):
        return True
    return None if all(v is None for v in vals) else False


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    import uvicorn
    from line.server import create_app, live_problems

    settings = load_settings()
    problems = live_problems(settings)
    try:
        profiles = load_profiles_or_env(settings)
    except (RuntimeError, ValueError) as e:
        problems.append(str(e))
    if problems:
        print("Cannot start:\n  " + "\n  ".join(problems))
        return 2
    guarded = next(iter(profiles))

    rec = Recorder()
    app = create_app(make_brain=rec.make_brain, caretaker=Hour0Caretaker(rec), settings=settings, profiles=profiles)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=args.port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    if not wait_for(lambda: server.started, 10, "server start"):
        print(f"Server did not start (is port {args.port} already in use?)")
        return 2

    def run(coro, timeout=15):
        return asyncio.run_coroutine_threadsafe(coro, rec.loop).result(timeout)

    marks: dict[str, float] = {}
    human: dict[str, Any] = {}
    result: dict[str, Any] = {"started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "guarded_number": mask(guarded)}
    try:
        say(f"Server is up on :{args.port}. Webhooks: {settings.webhook_url()}")
        say(f"CALLER: dial {guarded}. PROTECTED ({mask(profiles[guarded].protected_number)}): when it rings, "
            "wait ~5 s, LOOK AT THE SCREEN (caller ID), then answer.")
        if not wait_for(lambda: rec.session is not None, 300, "the inbound call"):
            return 1
        print(f"  call {rec.session.call_id} started")
        wait_for(lambda: rec.session.b_ccid is not None and rec.first(
            "conference.participant.joined", call_control_id=rec.session.b_ccid) is not None, 120, "B to join")

        say("Check 5: caller ID on the protected phone")
        human["caller_id_shown"] = mask_text(ask_text("What did the protected phone show (number / name / 'unknown')?"))
        human["caller_id_is_real_number"] = ask_yn("Was it the caller's real number?")

        say("Check 3: track split. CALLER: count slowly from 1 to 8 now. PROTECTED: stay silent.")
        marks["caller_talk_start"] = time.monotonic()
        input("  Press Enter when the caller is done.")
        marks["caller_talk_end"] = time.monotonic()
        say("PROTECTED: count slowly from 1 to 8 now. CALLER: stay silent.")
        marks["protected_talk_start"] = time.monotonic()
        input("  Press Enter when the protected person is done.")
        marks["protected_talk_end"] = time.monotonic()

        say("Check 2: targeted speak. BOTH stay silent for ~3 s, then a tip plays to the protected phone only.")
        marks["quiet_start"] = time.monotonic()
        time.sleep(3)
        marks["quiet_end"] = time.monotonic()
        s = rec.session
        marks["tip_start"] = time.monotonic()
        run(rec.telnyx.conference_action(s.conference_id, "speak", call_control_ids=[s.b_ccid], payload=TIP,
                                         voice=settings.tts_voice, language=settings.tts_language))
        wait_for(lambda: rec.first("conference.speak.ended") is not None or time.monotonic() - marks["tip_start"] > 12,
                 15, "conference.speak.ended")
        marks["tip_end"] = time.monotonic()
        human["protected_heard_tip"] = ask_yn("PROTECTED: did you hear the 'private Emberz test tip'?")
        human["caller_heard_tip"] = ask_yn("CALLER: did you hear it? (should be NO)")

        say("Check 4: supervisor roles (dials the caretaker number; a few cents).")
        ct = profiles[guarded].caretaker_number
        if ct and ask_yn(f"Dial the caretaker at {mask(ct)} now?"):
            c = run(rec.telnyx.dial(to=ct, **{"from": guarded}, timeout_secs=30,
                                    client_state=encode_client_state("C", s.call_id)))
            s.c_ccid = c.get("call_control_id")
            say("CARETAKER: answer. You join as 'monitor'. Everyone: say a few words.")
            wait_for(lambda: rec.first("C:call.answered") is not None, 60, "the caretaker to answer")
            human["monitor_hears_both"] = ask_yn("CARETAKER: can you hear both the caller and the protected person?")
            human["monitor_is_unheard"] = ask_yn("CALLER/PROTECTED: is the caretaker silent to you? (should be YES)")
            run(rec.telnyx.conference_action(s.conference_id, "update", call_control_id=s.c_ccid,
                                             supervisor_role="barge"))
            say("Switched the caretaker to 'barge'. CARETAKER: say something.")
            human["barge_is_heard"] = ask_yn("CALLER/PROTECTED: can you hear the caretaker now?")
            run(rec.telnyx.call_action(s.c_ccid, "hangup"))
        else:
            human["supervisor_roles"] = "skipped"

        say("CALLER: hang up now.")
        wait_for(lambda: rec.first("emberz.hangup") is not None, 60, "the hangup")
    finally:
        result["auto"] = analyze_auto(rec, marks)
        result["human"] = human
        result["checks"] = {
            "1_deepfake_with_stream": result["auto"]["deepfake_with_stream"].get("pass"),
            "2_targeted_speak_private": None if human.get("caller_heard_tip") is None else (
                not human["caller_heard_tip"] and bool(human.get("protected_heard_tip"))),
            "3_track_split": result["auto"]["track_split"].get("pass"),
            "4_supervisor_roles": None if "monitor_hears_both" not in human else all(
                human.get(k) for k in ("monitor_hears_both", "monitor_is_unheard", "barge_is_heard")),
            "5_caller_id_on_b": human.get("caller_id_is_real_number"),
            "6_notice_or_ringback_leaks": _any_true(result["auto"]["notice_ringback_leak"]["notice_in_outbound_track"],
                                                   result["auto"]["notice_ringback_leak"]["ringback_in_outbound_track"]),
            "media_handshake_signature_ok": (rec.media_start or {}).get("_signature_ok"),
        }
        result["frames"] = len(rec.frames)
        result["events"] = [et for _, et, _ in rec.events]
        args.out.write_text(json.dumps(result, indent=2, default=str))
        print(f"\nWrote {args.out}")
        print(json.dumps(result["checks"], indent=2))
        server.should_exit = True
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

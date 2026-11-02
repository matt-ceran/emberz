"""hour0.py offline: mu-law decode, levels, and the automatic analysis on a synthetic call."""
import asyncio

import numpy as np

from line.hour0 import Recorder, analyze_auto, frame_db, ulaw_to_pcm, window_db


def test_ulaw_decode_known_values():
    pcm = ulaw_to_pcm(bytes([0xFF, 0x7F, 0x00, 0x80]))
    assert pcm.tolist() == [0, 0, -32124, 32124]  # G.711 table endpoints


def test_levels():
    assert frame_db(bytes([0xFF]) * 160) < -85        # digital silence
    assert frame_db(bytes([0x80]) * 160) > -1         # full scale
    frames = [(0.0, "inbound", -20.0), (0.5, "inbound", -20.0), (0.5, "outbound", -60.0), (2.0, "inbound", -90.0)]
    assert window_db(frames, "inbound", 0, 1) == -20.0
    assert window_db(frames, "outbound", 0, 1) == -60.0
    assert window_db(frames, "outbound", 1, 2) is None


class _S:
    call_id, b_ccid, conference_id = "c_x", "v3:B", "conf"


def test_analyze_auto_on_a_synthetic_call():
    rec = Recorder(session=_S())
    ev = lambda t, et, **p: rec.events.append((t, et, p))  # noqa: E731
    ev(0.0, "streaming.started")
    ev(0.1, "call.speak.started")
    ev(3.0, "call.speak.ended")
    ev(6.0, "conference.participant.joined", call_control_id="v3:B")
    ev(8.0, "call.deepfake_detection.result", result="real", score=0.05, consistency=97.0)
    t = 0.0
    while t < 30:
        notice = 0.1 <= t < 3.0
        ringback = 3.0 <= t < 6.0
        caller_talks = 10 <= t < 14
        protected_talks = 15 <= t < 19
        tip = 23 <= t < 26
        rec.frames.append((t, "inbound", -25.0 if caller_talks else -70.0))
        rec.frames.append((t, "outbound", -22.0 if (notice or protected_talks) else (-40.0 if ringback else
                                                                                     -72.0 if not tip else -71.0)))
        t = round(t + 0.02, 2)
    marks = {"caller_talk_start": 10, "caller_talk_end": 14, "protected_talk_start": 15, "protected_talk_end": 19,
             "quiet_start": 20, "quiet_end": 23, "tip_start": 23, "tip_end": 26}
    out = analyze_auto(rec, marks)
    assert out["deepfake_with_stream"]["pass"] is True
    assert out["deepfake_with_stream"]["deepfake_s_after_stream_start"] == 8.0
    assert out["track_split"]["pass"] is True
    assert out["targeted_speak"]["auto_pass"] is True
    leak = out["notice_ringback_leak"]
    assert leak["notice_in_outbound_track"] and leak["ringback_in_outbound_track"]
    assert leak["gap_notice_end_to_b_joined_s"] == 3.0


def test_recorder_brain_records_frames():
    rec = Recorder()

    async def go():
        brain = rec.make_brain(_S(), telnyx=None, hub=None)
        await brain.on_media("inbound", bytes([0x80]) * 160, 0)
        await brain.on_call_event("media.start", {"media_format": {"encoding": "PCMU"}})

    asyncio.run(go())
    assert rec.frames[0][1] == "inbound" and rec.frames[0][2] > -1
    assert rec.media_start["media_format"]["encoding"] == "PCMU"
    assert np.isfinite(rec.frames[0][2])

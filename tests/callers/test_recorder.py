import base64
import json
import wave

import numpy as np
import pytest

from contracts.models import ConsoleEvent, Turn
from line.recorder import SAMPLE_RATE, ULAW_TO_PCM16, CallRecorder, hash_number, ulaw_decode

FRAME = 160  # 20 ms at 8 kHz


def ulaw_encode(pcm: np.ndarray) -> bytes:
    """Reference G.711 mu-law encoder (test only)."""
    BIAS, CLIP = 0x84, 32635
    out = bytearray()
    for s in pcm.astype(np.int32):
        sign = 0x80 if s < 0 else 0
        s = min(abs(int(s)), CLIP) + BIAS
        exp = 7
        while exp > 0 and not (s & (0x4000 >> (7 - exp))):
            exp -= 1
        mant = (s >> (exp + 3)) & 0x0F
        out.append(~(sign | (exp << 4) | mant) & 0xFF)
    return bytes(out)


def tone(freq: float, seconds: float, amp: int = 8000) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.int16)


def frames(pcm: np.ndarray) -> list[bytes]:
    enc = ulaw_encode(pcm)
    return [enc[i:i + FRAME] for i in range(0, len(enc), FRAME)]


def read_wav(path) -> tuple[np.ndarray, tuple]:
    with wave.open(str(path), "rb") as w:
        params = (w.getnchannels(), w.getsampwidth(), w.getframerate())
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2"), params


def test_ulaw_table_known_values_and_roundtrip():
    assert ULAW_TO_PCM16[0xFF] == 0 and ULAW_TO_PCM16[0x7F] == 0
    assert ULAW_TO_PCM16[0x00] == -32124 and ULAW_TO_PCM16[0x80] == 32124
    pcm = tone(440, 0.1)
    back = np.frombuffer(ulaw_decode(ulaw_encode(pcm)), dtype="<i2").astype(float)
    snr = 10 * np.log10(np.sum(pcm.astype(float) ** 2) / np.sum((pcm - back) ** 2))
    assert snr > 30  # mu-law gives ~35-38 dB on a sine


@pytest.mark.parametrize("unit", ["samples", "ms"])
def test_aligned_tracks_with_gap_and_reorder(tmp_path, unit):
    """Caller starts at t=0, protected at t=0.5 s; caller loses 10 frames; two frames swapped."""
    clock = [100.0]
    rec = CallRecorder("c_test", tmp_path, t0=100.0, clock=lambda: clock[0], ts_unit="auto", flush_every_s=0.2)
    caller, protected = tone(440, 2.0), tone(880, 1.0)
    cf, pf = frames(caller), frames(protected)
    base_c, base_p = 3_000_000_000, 77  # random RTP bases, independent per track
    step = FRAME if unit == "samples" else 20

    order = list(range(len(cf)))
    order[20], order[21] = order[21], order[20]          # out of order
    dropped = set(range(50, 60))                           # 200 ms lost
    for k in order:
        clock[0] = 100.0 + k * 0.02
        if k not in dropped:
            rec.feed("inbound", cf[k], timestamp=(base_c + k * step) % (1 << 32))
        pk = k - 25                                       # protected starts 25 frames = 0.5 s later
        if 0 <= pk < len(pf):
            rec.feed("outbound", pf[pk], timestamp=base_p + pk * step)
    meta = rec.close(reason="test")

    c, cparams = read_wav(tmp_path / "c_test" / "caller.wav")
    p, pparams = read_wav(tmp_path / "c_test" / "protected.wav")
    assert cparams == pparams == (1, 2, 8000)
    assert len(c) == len(p) == len(caller)                # same length -> aligned
    assert meta["tracks"]["caller"]["ts_unit"] == unit
    assert meta["tracks"]["protected"]["ts_unit"] == unit
    # caller: matches the source except the dropped 200 ms, which is silence
    ref = np.frombuffer(ulaw_decode(ulaw_encode(caller)), dtype="<i2")
    gap = slice(50 * FRAME, 60 * FRAME)
    assert np.all(c[gap] == 0)
    keep = np.ones(len(c), bool)
    keep[gap] = False
    assert np.array_equal(c[keep], ref[keep])
    assert meta["tracks"]["caller"]["padded_ms"] == 200
    # protected: silence for the first 0.5 s, then the tone in place
    assert np.all(p[:25 * FRAME] == 0)
    pref = np.frombuffer(ulaw_decode(ulaw_encode(protected)), dtype="<i2")
    assert np.array_equal(p[25 * FRAME:25 * FRAME + len(pref)], pref)
    assert meta["duration_s"] == pytest.approx(2.0)


def test_no_timestamps_pads_real_gaps_only(tmp_path):
    clock = [0.0]
    rec = CallRecorder("c_nots", tmp_path, t0=0.0, clock=lambda: clock[0])
    f = frames(tone(300, 1.0))
    for k in range(25):                    # 0.0-0.5 s, with small jitter
        clock[0] = k * 0.02 + (0.01 if k % 3 else 0)
        rec.feed("caller", f[k])
    clock[0] = 1.5                          # 1 s of nothing (hold), then 0.5 s more audio
    for k in range(25, 50):
        rec.feed("caller", f[k])
        clock[0] += 0.02
    rec.close()
    c, _ = read_wav(tmp_path / "c_nots" / "caller.wav")
    # first 25 frames contiguous (jitter didn't insert silence), then a ~1 s gap
    assert len(c) == pytest.approx(int(1.5 * SAMPLE_RATE) + 25 * FRAME - FRAME, abs=FRAME)
    assert np.all(c[25 * FRAME:int(1.4 * SAMPLE_RATE)] == 0)


def test_rtp_wraparound(tmp_path):
    rec = CallRecorder("c_wrap", tmp_path, t0=0.0, clock=lambda: 0.0, ts_unit="samples")
    f = frames(tone(500, 0.2))
    for k, fr in enumerate(f):
        rec.feed("caller", fr, timestamp=((1 << 32) - 3 * FRAME + k * FRAME) % (1 << 32))
    meta = rec.close()
    assert meta["tracks"]["caller"]["samples"] == len(f) * FRAME
    assert meta["tracks"]["caller"]["padded_ms"] == 0


def test_feed_telnyx_turns_events_meta_and_hashing(tmp_path):
    caller_no, guarded_no = "+15551230001", "+15559870002"
    rec = CallRecorder(
        "c_meta", tmp_path, t0=0.0, clock=lambda: 0.0,
        numbers={"caller": caller_no, "guarded": guarded_no},
        context={"profile": "elder", "rules": ["Never share codes"], "caller_known": False},
        source={"kind": "vapi", "agent": "bank_fraud", "variant": 3},
    )
    f = frames(tone(440, 0.1))
    for k, fr in enumerate(f):
        rec.feed_telnyx({"track": "inbound", "chunk": str(k + 1), "timestamp": str(k * 20),
                         "payload": base64.b64encode(fr).decode()})
    rec.add_turn(Turn(call_id="c_meta", turn=1, speaker="caller", text="hello", t_start_ms=0, t_end_ms=900))
    rec.add_event(ConsoleEvent(type="call_started", call_id="c_meta", at_ms=0,
                               data={"caller_number": caller_no, "guarded_number": guarded_no, "profile": "elder"}))
    rec.add_event(ConsoleEvent(type="coach", call_id="c_meta", at_ms=500, data={"text": "hi", "turn": 1}))
    meta = rec.close()
    d = tmp_path / "c_meta"
    assert sorted(p.name for p in d.iterdir()) == ["caller.wav", "events.jsonl", "meta.json", "protected.wav", "turns.jsonl"]
    turns = [Turn.model_validate_json(x) for x in (d / "turns.jsonl").read_text().splitlines()]
    assert turns[0].text == "hello"
    events = [ConsoleEvent.model_validate_json(x) for x in (d / "events.jsonl").read_text().splitlines()]
    assert events[0].data["caller_number"] == hash_number(caller_no) and events[0].data["profile"] == "elder"
    on_disk = json.loads((d / "meta.json").read_text())
    assert on_disk == meta and on_disk["turns"] == 1 and on_disk["events"] == 2
    assert on_disk["numbers_hashed"]["caller"].startswith("h:")
    assert on_disk["source"]["agent"] == "bank_fraud"
    for p in d.iterdir():                                   # raw numbers never hit the disk
        raw = p.read_bytes()
        assert caller_no.encode() not in raw and guarded_no.encode() not in raw
        assert b"5551230001" not in raw
    assert meta["tracks"]["protected"]["frames"] == 0 and meta["tracks"]["protected"]["samples"] == len(f) * FRAME


def test_wav_valid_mid_call_and_close_idempotent(tmp_path):
    rec = CallRecorder("c_mid", tmp_path, t0=0.0, clock=lambda: 0.0, ts_unit="ms", flush_every_s=0.5, reorder_window_ms=100)
    for k, fr in enumerate(frames(tone(440, 2.0))):
        rec.feed("caller", fr, timestamp=k * 20)
    c, params = read_wav(tmp_path / "c_mid" / "caller.wav")   # readable before close (header patched per write)
    assert params == (1, 2, 8000) and len(c) >= SAMPLE_RATE
    m1 = rec.close()
    assert rec.close() is m1
    rec.feed("caller", b"\xff" * FRAME, timestamp=0)          # ignored after close


def test_frame_later_than_reorder_window_is_dropped(tmp_path):
    rec = CallRecorder("c_late", tmp_path, t0=0.0, clock=lambda: 0.0, ts_unit="ms", flush_every_s=0.2, reorder_window_ms=100)
    f = frames(tone(440, 1.0))
    for k, fr in enumerate(f):
        if k != 3:
            rec.feed("caller", fr, timestamp=k * 20)
    rec.feed("caller", f[3], timestamp=60)        # arrives ~1 s late, after its slot was flushed
    meta = rec.close()
    assert meta["tracks"]["caller"]["late_dropped_frames"] == 1
    assert meta["tracks"]["caller"]["samples"] == len(f) * FRAME


# ---------------------------------------------------------------- adversarial review additions
def test_constant_zero_timestamps_fall_back_to_arrival(tmp_path):
    """line/server.py turns a missing timestamp into 0: the audio must not collapse to one frame."""
    clock = [0.0]
    rec = CallRecorder("c_zero", tmp_path, t0=0.0, clock=lambda: clock[0])
    f = frames(tone(440, 1.0))
    for k, fr in enumerate(f):
        clock[0] = k * 0.02
        rec.feed("inbound", fr, timestamp=0)
    meta = rec.close()
    assert meta["tracks"]["caller"]["ts_unit"] == "none"
    assert meta["tracks"]["caller"]["samples"] == len(f) * FRAME
    assert meta["tracks"]["caller"]["late_dropped_frames"] == 0


@pytest.mark.parametrize("jump", [1 << 30, -(1 << 20)])
def test_timestamp_discontinuity_resyncs_instead_of_huge_gap_or_drop(tmp_path, jump):
    """A new RTP source mid-call (forward or backward jump) re-anchors on arrival time."""
    clock = [0.0]
    rec = CallRecorder("c_jump", tmp_path, t0=0.0, clock=lambda: clock[0], ts_unit="samples")
    f = frames(tone(440, 2.0))
    for k, fr in enumerate(f):
        clock[0] = k * 0.02
        ts = k * FRAME if k < 50 else (k * FRAME + jump) % (1 << 32)
        rec.feed("outbound", fr, timestamp=ts)
    meta = rec.close()
    tr = meta["tracks"]["protected"]
    assert tr["ts_resyncs"] == 1 and tr["late_dropped_frames"] == 0
    assert tr["samples"] == len(f) * FRAME and tr["padded_ms"] == 0
    p, _ = read_wav(tmp_path / "c_jump" / "protected.wav")
    assert np.array_equal(p, np.frombuffer(ulaw_decode(b"".join(f)), dtype="<i2"))


def test_bad_frames_never_raise(tmp_path):
    rec = CallRecorder("c_bad", tmp_path, t0=0.0, clock=lambda: 0.0)
    rec.feed("mixed", b"\xff" * FRAME, timestamp=0)                         # unknown track
    rec.feed_telnyx({"track": "inbound", "timestamp": "5", "payload": "%%%not base64"})
    rec.feed_telnyx({"track": "inbound", "timestamp": "abc", "payload": base64.b64encode(b"\xff" * FRAME).decode()})
    rec.feed_telnyx({"track": "inbound"})                                    # no payload: ignored
    meta = rec.close()
    assert meta["skipped_frames"] == 3 and meta["tracks"]["caller"]["frames"] == 0


def test_disk_error_stops_audio_but_not_the_call(tmp_path, monkeypatch):
    rec = CallRecorder("c_disk", tmp_path, t0=0.0, clock=lambda: 0.0, ts_unit="ms", flush_every_s=0.1, reorder_window_ms=20)

    def full(*a, **k):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(rec._tracks["caller"].wav, "writeframes", full)
    for k, fr in enumerate(frames(tone(440, 1.0))):
        rec.feed("caller", fr, timestamp=k * 20)                             # must not raise
    rec.add_turn(Turn(call_id="c_disk", turn=1, speaker="caller", text="hi", t_start_ms=0, t_end_ms=10))
    meta = rec.close()
    assert "No space" in meta["audio_error"] and meta["turns"] == 1


def test_hash_number_normalizes_and_handles_anonymous():
    assert hash_number("+1 (555) 123-0001") == hash_number("15551230001")
    assert hash_number("anonymous") == "anonymous" and hash_number("Restricted") == "anonymous"
    assert hash_number(None) is None and hash_number("") is None


def test_one_recorder_per_call_and_safe_call_id(tmp_path):
    CallRecorder("c_dup", tmp_path).close()
    with pytest.raises(FileExistsError):
        CallRecorder("c_dup", tmp_path)                                      # would truncate the first
    for bad in ("../etc", "a/b", ".hidden", ""):
        with pytest.raises(ValueError):
            CallRecorder(bad, tmp_path)


def test_events_scrub_numbers_under_any_key(tmp_path):
    rec = CallRecorder("c_scrub", tmp_path, t0=0.0, clock=lambda: 0.0)
    rec.add_event(ConsoleEvent(type="caretaker_joined", call_id="c_scrub", at_ms=1,
                               data={"mode": "monitor", "leg": {"phone": "+15550009999"}, "note": "call +1 555 000 9999"}))
    rec.close()
    row = json.loads((tmp_path / "c_scrub" / "events.jsonl").read_text())
    assert row["data"]["leg"]["phone"] == hash_number("+15550009999")
    assert row["data"]["mode"] == "monitor" and row["data"]["note"] == "call +1 555 000 9999"   # free text kept

"""line/audio.py: mu-law tables, stateful resampling, wav -> frames. Fast, no ASR model."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import soxr

from line.audio import (
    FRAME_BYTES,
    MULAW_SILENCE,
    MULAW_TO_INT16,
    Upsampler,
    mulaw_decode,
    mulaw_encode,
    silence_frames,
    wav_to_mulaw_frames,
)

MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "sherpa-onnx-nemotron-speech-streaming-en-0.6b-160ms-int8-2026-04-25"

# sha256 of CPython 3.9 audioop output (computed once with /usr/bin/python3 on macOS):
#   audioop.ulaw2lin(bytes(range(256)), 2)                       (int16 little-endian)
#   audioop.lin2ulaw(array('h', range(-32768, 32768)).tobytes(), 2)
AUDIOOP_ULAW2LIN_SHA = "3dab54339e520bb2c924826e3b72a917a2b612e9fd12fc867500f1d983a75827"
AUDIOOP_LIN2ULAW_SHA = "81d633c9e6972a18c74a58720b96cb8ca0bdd096d4060b646dd708c3b846019a"


def test_decode_table_matches_audioop():
    assert hashlib.sha256(MULAW_TO_INT16.astype("<i2").tobytes()).hexdigest() == AUDIOOP_ULAW2LIN_SHA


def test_encode_matches_audioop():
    enc = mulaw_encode(np.arange(-32768, 32768, dtype=np.int32))
    assert hashlib.sha256(enc).hexdigest() == AUDIOOP_LIN2ULAW_SHA


def test_known_values_and_silence():
    assert MULAW_TO_INT16[0xFF] == 0 and MULAW_TO_INT16[0x7F] == 0
    assert MULAW_TO_INT16[0x00] == -32124 and MULAW_TO_INT16[0x80] == 32124
    assert mulaw_encode(np.zeros(4, np.int16)) == bytes([MULAW_SILENCE]) * 4
    assert np.all(mulaw_decode(silence_frames(20)[0]) == 0)


def test_roundtrip_error_is_small():
    t = np.arange(8000) / 8000
    x = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    y = mulaw_decode(mulaw_encode(x))
    snr = 10 * np.log10(np.sum(x**2) / np.sum((x - y) ** 2))
    assert snr > 33  # G.711 mu-law gives ~38 dB SNR on a loud tone


def test_upsampler_keeps_state_across_frames():
    t = np.arange(8000) / 8000
    x = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    up = Upsampler("HQ")
    chunked = np.concatenate([up(x[i : i + FRAME_BYTES]) for i in range(0, len(x), FRAME_BYTES)] + [up.flush()])
    whole = soxr.resample(x, 8000, 16000, quality="HQ")
    assert abs(len(chunked) - 16000) <= 2
    n = min(len(chunked), len(whole))
    # stateful streaming == offline resampling (no clicks at the 20 ms frame joins)
    assert np.max(np.abs(chunked[:n] - whole[:n])) < 1e-3


def test_upsampler_qq_low_delay():
    up = Upsampler("QQ")
    out = up(np.zeros(FRAME_BYTES, np.float32))
    assert len(out) >= 300  # returns ~320 samples right away (HQ buffers ~90 ms first)


def test_wav_to_frames_shapes():
    frames = wav_to_mulaw_frames(str(MODEL_DIR / "test_wavs" / "0.wav"))  # 16 kHz, 6.625 s
    assert all(len(f) == FRAME_BYTES for f in frames)
    assert len(frames) == int(np.ceil(6.625 * 50))
    frames8 = wav_to_mulaw_frames(str(MODEL_DIR / "test_wavs" / "8k.wav"))  # already 8 kHz, 4.825 s
    assert len(frames8) == int(np.ceil(4.825 * 50))

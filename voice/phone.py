"""The phone path, simulated: any clip -> 8 kHz -> G.711 mu-law -> back to 16 kHz.

Training, evaluation and live audio all go through this same path, so the detector
never sees a cleaner signal in training than it will on a real call:

- training/eval clips: `phone_degrade(wav, sr)`
- live Telnyx audio (already 8 kHz mu-law on the wire): `mulaw_decode` + the same
  8 -> 16 kHz soxr resampler (see voice/live.py), i.e. the second half of this path.

Running `phone_degrade` on audio that already went over a real phone line is close to
a no-op (mu-law re-encoding of decoded mu-law values is exact; 16 -> 8 kHz of an
8 -> 16 kHz upsampled signal only touches the band edge), so real recordings can go
through it too and every row of a manifest is treated the same way.

G.711 mu-law: ITU-T G.711 (https://www.itu.int/rec/T-REC-G.711). Decode is the classic
CCITT/Sun `ulaw2linear` (bias 0x84); encode matches CPython's removed
`audioop.lin2ulaw` (14-bit, bias 0x21, clip 8159), which is also what line/audio.py uses.
Python 3.13 has no audioop, so both are numpy tables/vector ops here.
"""
from __future__ import annotations

import numpy as np
import soxr

PHONE_SR = 8000
MODEL_SR = 16000
_BIAS = 0x84
_SEG_END_14 = np.array([0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF])


def _decode_table() -> np.ndarray:
    u = ~np.arange(256, dtype=np.int32) & 0xFF
    sign = u & 0x80
    exponent = (u >> 4) & 0x07
    mantissa = u & 0x0F
    mag = (((mantissa << 3) + _BIAS) << exponent) - _BIAS
    return np.where(sign != 0, -mag, mag).astype(np.int16)


MULAW_TO_INT16 = _decode_table()


def mulaw_encode(x: np.ndarray) -> np.ndarray:
    """int16 samples, or float samples in [-1, 1] -> uint8 mu-law codes."""
    x = np.asarray(x)
    if x.dtype.kind == "f":
        x = np.clip(np.round(x * 32768.0), -32768, 32767)
    v = x.astype(np.int32) >> 2
    mask = np.where(v < 0, 0x7F, 0xFF)
    v = np.minimum(np.abs(v), 8159) + 0x21
    seg = np.searchsorted(_SEG_END_14, v)
    uval = np.where(seg >= 8, 0x7F, (seg << 4) | ((v >> np.minimum(seg + 1, 31)) & 0x0F))
    return (uval ^ mask).astype(np.uint8)


def mulaw_decode(codes: np.ndarray | bytes) -> np.ndarray:
    """uint8 mu-law codes (array or raw bytes) -> float32 in [-1, 1]."""
    if isinstance(codes, (bytes, bytearray, memoryview)):
        codes = np.frombuffer(codes, dtype=np.uint8)
    return MULAW_TO_INT16[np.asarray(codes, dtype=np.uint8)].astype(np.float32) / 32768.0


def to_mono_float(wav: np.ndarray) -> np.ndarray:
    wav = np.asarray(wav)
    if wav.ndim == 2:  # soundfile layout: (frames, channels)
        wav = wav.mean(axis=1)
    if wav.dtype.kind in "iu":
        wav = wav.astype(np.float32) / float(np.iinfo(wav.dtype).max + 1)
    return np.ascontiguousarray(wav, dtype=np.float32)


def resample(wav: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return np.ascontiguousarray(wav, dtype=np.float32)
    return soxr.resample(wav, sr_in, sr_out, quality="HQ").astype(np.float32, copy=False)


def phone_degrade(wav: np.ndarray, sr: int, out_sr: int = MODEL_SR) -> np.ndarray:
    """Simulate the live phone path. Returns float32 mono at `out_sr` (16 kHz)."""
    x = resample(to_mono_float(wav), sr, PHONE_SR)
    x = mulaw_decode(mulaw_encode(np.clip(x, -1.0, 1.0)))
    return resample(x, PHONE_SR, out_sr)


class MulawTo16k:
    """Streaming version of the second half of the path: 8 kHz mu-law bytes -> 16 kHz float32.

    Same decode table and the same soxr HQ filter as `phone_degrade`; soxr keeps state
    across chunks so 20 ms frames join without clicks.
    """

    def __init__(self) -> None:
        self._rs = soxr.ResampleStream(PHONE_SR, MODEL_SR, 1, dtype="float32", quality="HQ")

    def __call__(self, payload: bytes, last: bool = False) -> np.ndarray:
        return self._rs.resample_chunk(mulaw_decode(payload), last=last)

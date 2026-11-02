"""Phone audio helpers for the line: G.711 mu-law, 8 -> 16 kHz resampling, test frames.

Telnyx media streams deliver PCMU (G.711 mu-law), 8 kHz, mono, one byte per sample
(see line/asr.py for the full frame format). 20 ms of audio = 160 bytes.
Python 3.13 has no `audioop`, so mu-law (and A-law) is done with 256-entry numpy tables.

Telnyx streaming_start `stream_codec` defaults to "default" = the call's own codec
(https://developers.telnyx.com/api-reference/call-commands/streaming-start.md, StreamCodec),
so the stream is PCMA when the call negotiated A-law. decode_payload() takes the
`start.media_format.encoding` value and handles PCMU and PCMA; anything else raises.
L1 should still ask for stream_codec="PCMU".

G.711 mu-law reference: ITU-T G.711 (https://www.itu.int/rec/T-REC-G.711); the
decode below is the classic Sun/CCITT `ulaw2linear` (bias 0x84, 8 segments).
"""
from __future__ import annotations

import numpy as np
import soxr

PHONE_RATE = 8000
ASR_RATE = 16000
FRAME_MS = 20
FRAME_BYTES = PHONE_RATE * FRAME_MS // 1000  # 160 mu-law bytes = 20 ms
MULAW_SILENCE = 0xFF  # decodes to 0

_BIAS = 0x84


def _build_decode_table() -> np.ndarray:
    u = ~np.arange(256, dtype=np.int32) & 0xFF
    sign = u & 0x80
    exponent = (u >> 4) & 0x07
    mantissa = u & 0x0F
    mag = (((mantissa << 3) + _BIAS) << exponent) - _BIAS
    return np.where(sign != 0, -mag, mag).astype(np.int16)


MULAW_TO_INT16 = _build_decode_table()
MULAW_TO_FLOAT = (MULAW_TO_INT16.astype(np.float32) / 32768.0).astype(np.float32)


def mulaw_decode(payload: bytes) -> np.ndarray:
    """mu-law bytes -> float32 samples in [-1, 1] (same length)."""
    return MULAW_TO_FLOAT[np.frombuffer(payload, dtype=np.uint8)]


def mulaw_decode_int16(payload: bytes) -> np.ndarray:
    return MULAW_TO_INT16[np.frombuffer(payload, dtype=np.uint8)]


def _build_alaw_table() -> np.ndarray:
    # Sun/CCITT alaw2linear (same as audioop.alaw2lin; hash pinned in tests/asr)
    a = np.arange(256, dtype=np.int32) ^ 0x55
    t = (a & 0x0F) << 4
    seg = (a & 0x70) >> 4
    t = np.where(seg == 0, t + 8, np.where(seg == 1, t + 0x108, (t + 0x108) << np.maximum(seg - 1, 0)))
    return np.where(a & 0x80, t, -t).astype(np.int16)


ALAW_TO_INT16 = _build_alaw_table()
ALAW_TO_FLOAT = (ALAW_TO_INT16.astype(np.float32) / 32768.0).astype(np.float32)
ENCODINGS = {"PCMU": MULAW_TO_FLOAT, "PCMA": ALAW_TO_FLOAT}
SILENCE_BYTE = {"PCMU": MULAW_SILENCE, "PCMA": 0xD5}  # both decode to ~0


def check_encoding(encoding: str) -> str:
    enc = (encoding or "PCMU").upper()
    if enc not in ENCODINGS:
        raise ValueError(f"unsupported stream encoding {encoding!r}; ask Telnyx for stream_codec=PCMU")
    return enc


def decode_payload(payload: bytes, encoding: str = "PCMU") -> np.ndarray:
    """PCMU or PCMA bytes -> float32 samples in [-1, 1]."""
    return ENCODINGS[check_encoding(encoding)][np.frombuffer(payload, dtype=np.uint8)]


def mulaw_encode(samples: np.ndarray) -> bytes:
    """int16 (or float in [-1, 1]) samples -> mu-law bytes. Used by tests and simulators."""
    x = np.asarray(samples)
    if x.dtype.kind == "f":
        x = np.clip(np.round(x * 32768.0), -32768, 32767)
    # Same as CPython's audioop.lin2ulaw (st_14linear2ulaw): 14-bit, bias 0x21, clip 8159.
    v = x.astype(np.int32) >> 2
    mask = np.where(v < 0, 0x7F, 0xFF)
    v = np.minimum(np.abs(v), 8159) + 0x21
    seg = np.searchsorted(np.array([0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF]), v)
    uval = np.where(seg >= 8, 0x7F, (seg << 4) | ((v >> (seg + 1)) & 0x0F))
    return (uval ^ mask).astype(np.uint8).tobytes()


class Upsampler:
    """Stateful 8 kHz -> 16 kHz resampler; keep one per audio track.

    soxr keeps filter state across chunks, so 20 ms frames join without clicks.
    Quality "HQ" buffers ~90 ms and returns output in bursts; "QQ" (cubic) has ~0.25 ms
    delay. bench_asr.py compares their WER; the default is set from that measurement.
    """

    def __init__(self, quality: str = "HQ"):
        self._rs = soxr.ResampleStream(PHONE_RATE, ASR_RATE, 1, dtype="float32", quality=quality)

    def __call__(self, samples_8k: np.ndarray, last: bool = False) -> np.ndarray:
        return self._rs.resample_chunk(np.ascontiguousarray(samples_8k, dtype=np.float32), last=last)

    def flush(self) -> np.ndarray:
        return self._rs.resample_chunk(np.zeros(0, dtype=np.float32), last=True)


def load_wav_mono(path: str) -> tuple[np.ndarray, int]:
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    return audio.mean(axis=1).astype(np.float32), sr


def to_phone_rate(audio: np.ndarray, sr: int) -> np.ndarray:
    """Any-rate float audio -> 8 kHz float audio (offline, whole file)."""
    if sr == PHONE_RATE:
        return audio.astype(np.float32)
    return soxr.resample(audio, sr, PHONE_RATE, quality="HQ").astype(np.float32)


def mulaw_frames(audio_8k: np.ndarray, frame_bytes: int = FRAME_BYTES) -> list[bytes]:
    """8 kHz float audio -> list of mu-law frames (last one padded with silence)."""
    data = mulaw_encode(audio_8k)
    pad = (-len(data)) % frame_bytes
    data += bytes([MULAW_SILENCE]) * pad
    return [data[i : i + frame_bytes] for i in range(0, len(data), frame_bytes)]


def wav_to_mulaw_frames(path: str) -> list[bytes]:
    """A wav file -> 8 kHz mu-law frames of 160 bytes (20 ms), like a Telnyx media stream."""
    audio, sr = load_wav_mono(path)
    return mulaw_frames(to_phone_rate(audio, sr))


def silence_frames(ms: int) -> list[bytes]:
    n = max(0, round(ms / FRAME_MS))
    return [bytes([MULAW_SILENCE]) * FRAME_BYTES for _ in range(n)]

"""Pure helpers for the G1 noise pass (no model imports, so tests run anywhere).

Pipeline per turn: Kokoro text -> 24 kHz audio -> 8 kHz -> level + optional noise
-> G.711 mu-law (what the phone network carries) -> decode -> 16 kHz -> Nemotron ASR.
"""
from __future__ import annotations

import hashlib

import random
import re
from typing import Iterable, Optional

import numpy as np

from contracts.models import DatasetRow, NoisyTurn

# ------------------------------------------------------------------ voices
# Kokoro-82M voice ids and grades: https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md
# Kokoro has no child voices. For "kid" we use the lighter, younger-sounding female
# voices and speak slightly faster; ASR noise on real children is therefore underestimated.
PROTECTED_VOICES: dict[str, list[str]] = {
    "kid": ["af_sky", "af_nova", "af_river", "af_jessica"],
    "elder": ["am_santa", "bm_george", "bf_emma", "bm_fable", "af_sarah", "am_onyx"],
    "adult": ["af_heart", "af_bella", "am_michael", "am_fenrir", "af_nicole", "bf_isabella"],
}
PROFILE_SPEED: dict[str, float] = {"kid": 1.1, "elder": 0.9, "adult": 1.0}
# Callers: a wide pool of adult voices, American and British.
CALLER_VOICES: list[str] = [
    "am_adam", "am_echo", "am_eric", "am_liam", "am_puck", "am_michael", "am_fenrir", "am_onyx",
    "af_alloy", "af_kore", "af_aoede", "af_nicole", "af_sarah", "af_heart",
    "bm_daniel", "bm_lewis", "bm_george", "bf_alice", "bf_lily", "bf_emma",
]
ALL_VOICES: list[str] = sorted({v for vs in PROTECTED_VOICES.values() for v in vs} | set(CALLER_VOICES))


def stable_rng(*parts: object) -> random.Random:
    """Deterministic RNG from any parts (Python's hash() is salted per process)."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def pick_voices(call_id: str, profile: str, seed: int = 0) -> dict[str, str]:
    """Different Kokoro voices for the two speakers, stable per call_id."""
    rng = stable_rng("voices", call_id, seed)
    protected = rng.choice(PROTECTED_VOICES[profile])
    caller = rng.choice([v for v in CALLER_VOICES if v != protected])
    return {"caller": caller, "protected": protected}


def speaker_speed(call_id: str, speaker: str, profile: str, seed: int = 0) -> float:
    base = PROFILE_SPEED[profile] if speaker == "protected" else 1.0
    return round(base * stable_rng("speed", call_id, speaker, seed).uniform(0.93, 1.07), 3)


def lang_code(voice: str) -> str:
    """Kokoro pipeline language: 'a' American, 'b' British (first letter of the voice id)."""
    return voice[0]


# ------------------------------------------------------------------ G.711 mu-law
# Segment-based encoder as in ITU-T G.711 / the classic Sun g711.c (and Python's old audioop).
_BIAS = 0x84
_CLIP = 32635


def lin2ulaw(pcm16: np.ndarray) -> np.ndarray:
    x = np.asarray(pcm16, dtype=np.int32)
    sign = np.where(x < 0, 0x80, 0)
    mag = np.minimum(np.abs(x), _CLIP) + _BIAS
    exponent = np.clip(np.floor(np.log2(mag)).astype(np.int32) - 7, 0, 7)
    mantissa = (mag >> (exponent + 3)) & 0x0F
    return (~(sign | (exponent << 4) | mantissa) & 0xFF).astype(np.uint8)


def ulaw2lin(ulaw: np.ndarray) -> np.ndarray:
    u = ~np.asarray(ulaw, dtype=np.int32) & 0xFF
    sign = u & 0x80
    exponent = (u >> 4) & 0x07
    mantissa = u & 0x0F
    mag = (((mantissa << 3) + _BIAS) << exponent) - _BIAS
    return np.where(sign != 0, -mag, mag).astype(np.int16)


def float_to_pcm16(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, -1.0, 1.0) * 32767.0).round().astype(np.int16)


def pcm16_to_float(x: np.ndarray) -> np.ndarray:
    return (np.asarray(x, dtype=np.float32) / 32768.0).astype(np.float32)


# ------------------------------------------------------------------ level and noise
def set_level(x: np.ndarray, dbfs: float) -> np.ndarray:
    """Scale to a target RMS level (telephone speech sits around -26 dBov)."""
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) if len(x) else 0.0
    if rms < 1e-9:
        return x.astype(np.float32)
    return (x * (10 ** (dbfs / 20) / rms)).astype(np.float32)


def colored_noise(n: int, rng: np.random.Generator, exponent: float = 1.0) -> np.ndarray:
    """1/f^exponent noise (0 = white, 1 = pink, 2 = brown), unit RMS."""
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(len(spec), dtype=np.float64)
    f[0] = 1.0
    spec = spec / f ** (exponent / 2)
    out = np.fft.irfft(spec, n)
    return (out / (np.sqrt(np.mean(out**2)) + 1e-12)).astype(np.float32)


def add_noise(x: np.ndarray, snr_db: float, rng: np.random.Generator, exponent: float = 1.0) -> np.ndarray:
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) if len(x) else 0.0
    if rms < 1e-9:
        return x
    noise = colored_noise(len(x), rng, exponent) * (rms / 10 ** (snr_db / 20))
    return (x + noise).astype(np.float32)


def packet_loss(x: np.ndarray, prob: float, rng: np.random.Generator, frame: int = 160) -> np.ndarray:
    """Zero whole 20 ms frames (160 samples at 8 kHz) with probability prob, like lost RTP
    packets without concealment. prob=0 returns x unchanged."""
    if prob <= 0 or len(x) == 0:
        return x
    y = x.copy()
    for start in range(0, len(y), frame):
        if rng.random() < prob:
            y[start : start + frame] = 0
    return y


# ------------------------------------------------------------------ WER
_PUNCT = re.compile(r"[^a-z0-9' ]+")


def normalize_text(s: str) -> list[str]:
    """Lowercase, hyphens/slashes to spaces, drop periods/commas inside tokens ("a.m." == "am"),
    drop other punctuation (keep apostrophes), split.

    Numbers are not expanded, so "$500" vs "five hundred dollars" counts as errors;
    this slightly inflates WER on number-heavy turns.
    """
    s = s.lower().replace("’", "'").replace("-", " ").replace("/", " ")
    s = re.sub(r"[.,]", "", s)
    s = _PUNCT.sub(" ", s)
    return [w.strip("'") for w in s.split() if w.strip("'")]


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def wer(ref: str, hyp: str) -> Optional[float]:
    """Word error rate of hyp against ref; None when the reference has no words."""
    r, h = normalize_text(ref), normalize_text(hyp)
    if not r:
        return None
    return edit_distance(r, h) / len(r)


def corpus_wer(pairs: Iterable[tuple[str, str]]) -> Optional[float]:
    errs = words = 0
    for ref, hyp in pairs:
        r = normalize_text(ref)
        errs += edit_distance(r, normalize_text(hyp))
        words += len(r)
    return errs / words if words else None


# ------------------------------------------------------------------ rows
def parse_input_line(line: str) -> DatasetRow:
    return DatasetRow.model_validate_json(line)


def read_input(path: str) -> tuple[list[DatasetRow], list[str]]:
    """Valid rows plus one error string per bad line (bad lines are skipped, not fatal)."""
    rows, errors, seen = [], [], set()
    with open(path) as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = parse_input_line(line)
            except Exception as e:  # pydantic ValidationError or bad JSON
                errors.append(f"line {n}: {str(e).splitlines()[0]}")
                continue
            if row.call_id in seen:
                errors.append(f"line {n}: duplicate call_id {row.call_id}")
                continue
            seen.add(row.call_id)
            rows.append(row)
    return rows, errors


def build_noisy_rows(row: DatasetRow, asr_text: dict[int, str], voices: dict[str, str]) -> list[NoisyTurn]:
    """One NoisyTurn per input turn, in turn order. Turns without ASR output get text_asr=""."""
    out = []
    for t in sorted(row.turns, key=lambda t: t.turn):
        hyp = asr_text.get(t.turn, "")
        out.append(
            NoisyTurn(
                call_id=row.call_id,
                turn=t.turn,
                speaker=t.speaker,
                text_clean=t.text,
                text_asr=hyp,
                voice=voices[t.speaker],
                wer=None if wer(t.text, hyp) is None else round(wer(t.text, hyp), 4),
            )
        )
    return out


def clean_partial_output(output_path: str, expected_turns: dict[str, int]) -> set[str]:
    """Resume support after a spot preemption or crash.

    Keeps only calls whose rows are all present and valid, rewrites the file with just
    those lines, and returns their call_ids so the run can skip them.
    """
    try:
        with open(output_path) as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        return set()
    by_call: dict[str, list[str]] = {}
    for line in lines:
        try:
            nt = NoisyTurn.model_validate_json(line)
        except Exception:
            continue
        by_call.setdefault(nt.call_id, []).append(line)
    done = {cid for cid, ls in by_call.items() if expected_turns.get(cid) == len(ls)}
    with open(output_path, "w") as f:
        for cid in by_call:
            if cid in done:
                f.write("\n".join(by_call[cid]) + "\n")
    return done


def safe_name(call_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", call_id)

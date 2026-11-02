"""ASR benchmark for the line (L2): WER, endpoint delay, real-time factor.

    .venv/bin/python -m line.bench_asr all              # everything, prints a report
    .venv/bin/python -m line.bench_asr wer|sweep|delay|rtf

Material:
- the model's test_wavs (LibriSpeech-style read speech, with trans.txt), and
- a short scripted scam dialog rendered with macOS `say` (two voices), cached under
  $EMBERZ_BENCH_DIR (default: <system temp dir>/emberz_bench_asr, never the repo).
  Skipped on machines without `say`.

Every utterance goes 16 kHz wav -> 8 kHz -> mu-law 20 ms frames, like a Telnyx stream.
"Speech end" = last 20 ms frame above -45 dBFS in the utterance (ground truth from the
clean input, placed on a known timeline). Endpoint delay = time from that frame to the
Turn that contains the utterance's last word: in audio time (no compute), and in wall
time with frames paced in real time and 4 streams (2 calls) decoding at once.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from line.asr import DEFAULT_MODEL_DIR, CallTranscriber, AsyncCallTranscriber, EndpointConfig, load_recognizer
from line.audio import (
    FRAME_MS,
    load_wav_mono,
    mulaw_decode,
    mulaw_frames,
    silence_frames,
    to_phone_rate,
)

TEST_WAVS = DEFAULT_MODEL_DIR / "test_wavs"
BENCH_DIR = Path(os.environ.get("EMBERZ_BENCH_DIR") or Path(tempfile.gettempdir()) / "emberz_bench_asr")

# (speaker, voice, text). Gaps between lines are set in build_dialog().
DIALOG = [
    ("caller", "Reed (English (US))", "Hi, this is Mark from the fraud department at your bank."),
    ("protected", "Grandma (English (US))", "Oh dear. Is something wrong with my account?"),
    ("caller", "Reed (English (US))", "We saw a charge of nine hundred dollars from a store in Ohio. Did you make that purchase?"),
    ("protected", "Grandma (English (US))", "No, I did not. I have not been to Ohio in years."),
    ("caller", "Reed (English (US))", "Okay, I can block it right now. I just need to verify you are the account holder."),
    ("protected", "Grandma (English (US))", "All right, what do you need?"),
    ("caller", "Reed (English (US))", "We just sent a six digit code to your phone. Can you read me the code?"),
    ("protected", "Grandma (English (US))", "Let me find my glasses. It says four, seven, one."),
    ("caller", "Reed (English (US))", "Great, and the last three numbers please. Hurry, the charge is going through."),
    ("protected", "Grandma (English (US))", "I think I should call my daughter first."),
]
GAPS_MS = [600, 250, 900, 350, 1200, 200, 700, 400, 1500]  # reply gaps, a mix of fast and slow


# ------------------------------------------------------------------ text + WER
def norm_words(s: str) -> list[str]:
    s = s.upper().replace("-", " ")
    s = re.sub(r"[^A-Z0-9' ]+", " ", s)
    return s.split()


def word_errors(ref: str, hyp: str) -> tuple[int, int]:
    r, h = norm_words(ref), norm_words(hyp)
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
            prev, d[j] = d[j], cur
    return d[len(h)], len(r)


def read_trans() -> dict[str, str]:
    out = {}
    for line in (TEST_WAVS / "trans.txt").read_text().splitlines():
        if line.strip():
            name, text = line.split(maxsplit=1)
            out[name] = text
    return out


# ------------------------------------------------------------------ material
@dataclass
class Utt:
    speaker: str
    text: str
    frames: list[bytes]   # mu-law 20 ms frames
    wav16: np.ndarray     # original audio at 16 kHz (for the 16 kHz WER path)


def say_available() -> bool:
    return shutil.which("say") is not None


def render_dialog() -> list[Utt]:
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    utts = []
    for i, (spk, voice, text) in enumerate(DIALOG):
        p = BENCH_DIR / f"dialog_{i:02d}.wav"
        if not p.exists():
            subprocess.run(["say", "-v", voice, "-o", str(p), "--data-format=LEI16@16000", text], check=True)
        utts.append(load_utt(p, spk, text))
    return utts


def load_utt(path: Path, speaker: str, text: str) -> Utt:
    audio, sr = load_wav_mono(str(path))
    import soxr

    wav16 = audio if sr == 16000 else soxr.resample(audio, sr, 16000).astype(np.float32)
    return Utt(speaker, text, mulaw_frames(to_phone_rate(audio, sr)), wav16)


def test_wav_utts(speakers=("caller", "protected", "caller")) -> list[Utt]:
    """0.wav, 1.wav, 8k.wav with their trans.txt references, alternating speakers by default."""
    trans = read_trans()
    return [load_utt(TEST_WAVS / n, spk, trans[n]) for n, spk in zip(["0.wav", "1.wav", "8k.wav"], speakers)]


def speech_bounds(frames: list[bytes], thresh_db: float = -45.0, min_frames: int = 3) -> tuple[float, float]:
    """First and last 20 ms frame above thresh_db, ignoring blips shorter than min_frames."""
    db = np.array([10 * np.log10(np.mean(mulaw_decode(f) ** 2) + 1e-10) for f in frames])
    on = db > thresh_db
    solid = np.convolve(on.astype(int), np.ones(min_frames, int), mode="valid") == min_frames  # solid[i]: frames i..i+min_frames-1 all on
    idx = np.nonzero(solid)[0]
    return idx[0] * FRAME_MS / 1000, (idx[-1] + min_frames) * FRAME_MS / 1000


@dataclass
class Placed:
    utt: Utt
    track: str
    start_s: float      # first frame of the utterance on the call clock
    speech_end_s: float  # ground-truth end of speech on the call clock
    speech_start_s: float = 0.0  # ground-truth start of speech on the call clock


def build_dialog(utts: list[Utt], gaps_ms: list[int], lead_ms: int = 500) -> tuple[dict[str, list[bytes]], list[Placed]]:
    """Lay utterances on two tracks, alternating; gap = silence between one speaker's end of
    speech and the next speaker's start of speech."""
    tracks = {"inbound": [], "outbound": []}
    placed: list[Placed] = []
    t = lead_ms / 1000
    for k, u in enumerate(utts):
        track = "inbound" if u.speaker == "caller" else "outbound"
        s0, s1 = speech_bounds(u.frames)
        start = max(0.0, t - s0)
        cur = len(tracks[track]) * FRAME_MS / 1000
        tracks[track] += silence_frames(round((start - cur) * 1000))
        start = len(tracks[track]) * FRAME_MS / 1000
        tracks[track] += u.frames
        placed.append(Placed(u, track, start, start + s1, start + s0))
        t = start + s1 + (gaps_ms[k % len(gaps_ms)] / 1000 if gaps_ms else 0.5)
    n = max(len(v) for v in tracks.values()) + 100  # 2 s tail
    for k in tracks:
        tracks[k] += silence_frames((n - len(tracks[k])) * FRAME_MS)
    return tracks, placed


def match_turns(placed: list[Placed], turns, segments) -> list[tuple[Placed, object, object]]:
    """For each utterance, the turn (same speaker) that contains its end of speech."""
    out = []
    for p in placed:
        spk = "caller" if p.track == "inbound" else "protected"
        cand = [(t, s) for t, s in zip(turns, segments) if t.speaker == spk and t.t_end_ms >= p.speech_end_s * 1000 - 400]
        cand = [c for c in cand if c[0].t_start_ms <= p.speech_end_s * 1000]
        if cand:
            out.append((p, *cand[0]))
    return out


def onset_and_gap_errors(placed: list[Placed], turns) -> tuple[np.ndarray, np.ndarray]:
    """t_start error per utterance (first turn of it), and gap_ms error where the speaker changed."""
    onset, gap = [], []
    for k, p in enumerate(placed):
        spk = "caller" if p.track == "inbound" else "protected"
        cand = [t for t in turns if t.speaker == spk and abs(t.t_start_ms / 1000 - p.speech_start_s) < 1.0]
        if not cand:
            continue
        t = cand[0]
        onset.append(t.t_start_ms / 1000 - p.speech_start_s)
        if k > 0 and placed[k - 1].track != p.track and t.gap_ms is not None:
            gap.append(t.gap_ms / 1000 - (p.speech_start_s - placed[k - 1].speech_end_s))
    return np.array(onset), np.array(gap)


def texts_for(placed: list[Placed], turns) -> list[str]:
    """Concatenate the turns that fall inside each utterance (to score WER per utterance)."""
    out = []
    for p in placed:
        spk = "caller" if p.track == "inbound" else "protected"
        lo, hi = p.start_s * 1000 - 200, p.speech_end_s * 1000 + 400
        out.append(" ".join(t.text for t in turns if t.speaker == spk and lo <= t.t_start_ms <= hi))
    return out


# ------------------------------------------------------------------ runs
def run_sync(tracks, cfg: EndpointConfig, quality: str = "HQ") -> CallTranscriber:
    ct = CallTranscriber("bench", lambda t: None, cfg=cfg, resample_quality=quality)
    for a, b in zip(tracks["inbound"], tracks["outbound"]):
        ct.feed_round([("inbound", a, None), ("outbound", b, None)])
    ct.finish()
    return ct


def wer_16k(utts: list[Utt]) -> list[str]:
    """Reference path: original 16 kHz audio, one fresh stream per utterance, no endpointing."""
    rec = load_recognizer()
    out = []
    for u in utts:
        s = rec.create_stream()
        # 0.5 s of leading silence: with none, the model drops the first word (say renders start at t=0)
        s.accept_waveform(16000, np.concatenate([np.zeros(8000, np.float32), u.wav16, np.zeros(16000, np.float32)]))
        s.input_finished()
        while rec.is_ready(s):
            rec.decode_stream(s)
        out.append(rec.get_result(s))
    return out


def wer_8k_sherpa_resample(utts: list[Utt]) -> list[str]:
    """mu-law decoded and given to sherpa at 8 kHz (sherpa's internal resampler)."""
    rec = load_recognizer()
    out = []
    for u in utts:
        s = rec.create_stream()
        x = mulaw_decode(b"".join(u.frames))
        s.accept_waveform(8000, np.concatenate([np.zeros(4000, np.float32), x, np.zeros(8000, np.float32)]))
        s.input_finished()
        while rec.is_ready(s):
            rec.decode_stream(s)
        out.append(rec.get_result(s))
    return out


def cmd_wer(cfg: EndpointConfig) -> dict:
    sets = {"test_wavs": test_wav_utts()}
    if say_available():
        sets["dialog_say"] = render_dialog()
    rows = {}
    for name, utts in sets.items():
        tracks, placed = build_dialog(utts, [700])
        paths = {
            "16k_original": wer_16k(utts),
            "8k_mulaw_sherpa_resample": wer_8k_sherpa_resample(utts),
            "8k_mulaw_stream_HQ": texts_for(placed, run_sync(tracks, cfg, "HQ").turns),
            "8k_mulaw_stream_QQ": texts_for(placed, run_sync(tracks, cfg, "QQ").turns),
        }
        for path, hyps in paths.items():
            errs = sum(word_errors(u.text, h)[0] for u, h in zip(utts, hyps))
            n = sum(len(norm_words(u.text)) for u in utts)
            rows[(name, path)] = (errs, n)
            print(f"WER {name:11} {path:26} {errs:3}/{n:3} = {errs / n:6.1%}")
        if name == "test_wavs":
            for u, h in zip(utts, paths["8k_mulaw_stream_HQ"]):
                print(f"    ref: {u.text}\n    hyp: {h}")
    return rows


def cmd_sweep(base: EndpointConfig) -> None:
    """Audio-time endpoint delay and sentence splits vs the VAD trailing-silence rule."""
    utts = (render_dialog() if say_available() else []) + test_wav_utts()
    tracks, placed = build_dialog(utts, GAPS_MS)
    print(f"{len(placed)} utterances, {sum(len(norm_words(p.utt.text)) for p in placed)} words")
    print("vad_silence_s  turns  extra_splits  delay_p50  delay_p95  delay_max  WER    |onset err| p50/max  |gap err| p50/max")
    for vs in [0.4, 0.5, 0.6, 0.7, 0.8]:
        cfg = EndpointConfig(**{**base.__dict__, "vad_silence_s": vs})
        ct = run_sync(tracks, cfg)
        m = match_turns(placed, ct.turns, ct.segments)
        delays = np.array([seg.emitted_at_s - p.speech_end_s for p, _, seg in m])
        hyps = texts_for(placed, ct.turns)
        errs = sum(word_errors(p.utt.text, h)[0] for p, h in zip(placed, hyps))
        n = sum(len(norm_words(p.utt.text)) for p in placed)
        on, gp = onset_and_gap_errors(placed, ct.turns)
        print(
            f"{vs:13.1f}  {len(ct.turns):5}  {len(ct.turns) - len(placed):12}  {np.percentile(delays, 50):9.2f}"
            f"  {np.percentile(delays, 95):9.2f}  {delays.max():9.2f}  {errs / n:5.1%}"
            f"  {np.median(np.abs(on)):.2f}/{np.abs(on).max():.2f} (n={len(on)})"
            f"  {np.median(np.abs(gp)):.2f}/{np.abs(gp).max():.2f} (n={len(gp)})"
        )


async def _paced_call(call_id: str, tracks, placed, cfg, results: dict) -> None:
    loop = asyncio.get_running_loop()
    emitted: list[tuple[object, float]] = []
    act = AsyncCallTranscriber(call_id, lambda t: emitted.append((t, time.perf_counter())), loop, cfg=cfg)
    fed_wall: list[float] = []
    t0 = time.perf_counter()
    max_q = 0
    for i, (a, b) in enumerate(zip(tracks["inbound"], tracks["outbound"])):
        target = t0 + i * FRAME_MS / 1000
        dt = target - time.perf_counter()
        if dt > 0:
            await asyncio.sleep(dt)
        act.feed("inbound", a)
        act.feed("outbound", b)
        fed_wall.append(time.perf_counter())
        max_q = max(max_q, act.queue_depth())
    await act.aclose()
    await asyncio.sleep(0.05)  # let call_soon_threadsafe callbacks run
    turns = [t for t, _ in emitted]
    m = match_turns(placed, turns, act.inner.segments)
    wall_of = {t.turn: w for t, w in emitted}
    rows = []
    for p, turn, seg in m:
        k = min(int(p.speech_end_s * 1000 / FRAME_MS), len(fed_wall) - 1)  # frame holding the last speech
        rows.append((seg.emitted_at_s - p.speech_end_s, wall_of[turn.turn] - fed_wall[k], seg.reason))
    results[call_id] = {"rows": rows, "max_queue": max_q, "error": act.error}


def cmd_delay(cfg: EndpointConfig, n_calls: int = 2) -> dict:
    """Real-time paced: n_calls calls x 2 tracks decoding at once (default 4 streams)."""
    utts = (render_dialog() if say_available() else []) + test_wav_utts()
    tracks, placed = build_dialog(utts, GAPS_MS)
    load_recognizer()
    results: dict = {}

    async def main():
        await asyncio.gather(*[_paced_call(f"c{i}", tracks, placed, cfg, results) for i in range(n_calls)])

    asyncio.run(main())
    rows = [r for v in results.values() for r in v["rows"]]
    audio = np.array([r[0] for r in rows])
    wall = np.array([r[1] for r in rows])
    reasons = {k: sum(1 for r in rows if r[2] == k) for k in {r[2] for r in rows}}
    print(f"paced, {n_calls} calls = {2 * n_calls} streams, {len(rows)} utterance ends, cut reasons {reasons}")
    print(f"  endpoint delay, audio time: p50 {np.percentile(audio, 50):.2f}s  p95 {np.percentile(audio, 95):.2f}s  max {audio.max():.2f}s")
    print(f"  endpoint delay, wall time : p50 {np.percentile(wall, 50):.2f}s  p95 {np.percentile(wall, 95):.2f}s  max {wall.max():.2f}s")
    print(f"  max ASR queue depth (frames) {max(v['max_queue'] for v in results.values())}; errors {[v['error'] for v in results.values() if v['error']]}")
    left50, left95 = 1.5 - np.percentile(wall, 50), 1.5 - np.percentile(wall, 95)
    print(f"  of the 1.5 s trigger->coach budget, left after ASR: p50 {left50:.2f}s, p95 {left95:.2f}s"
          " (guardian call + rule layer + Telnyx speak start must fit in this)")
    return {"audio": audio, "wall": wall}


def cmd_rtf(cfg: EndpointConfig, n_calls: int = 2) -> None:
    """Unpaced: n_calls CallTranscribers on their own threads (2 streams each), as fast as possible."""
    utts = test_wav_utts()
    tracks, _ = build_dialog(utts, [700])
    audio_s = len(tracks["inbound"]) * FRAME_MS / 1000
    load_recognizer()
    cts = []

    def work():
        ct = CallTranscriber("rtf", lambda t: None, cfg=cfg)
        cts.append(ct)
        for a, b in zip(tracks["inbound"], tracks["outbound"]):
            ct.feed_round([("inbound", a, None), ("outbound", b, None)])

    for n in sorted({1, n_calls}):
        cts.clear()
        th = [threading.Thread(target=work) for _ in range(n)]
        t = time.perf_counter()
        [x.start() for x in th]
        [x.join() for x in th]
        wall = time.perf_counter() - t
        chunks = sum(s.decode_chunks for c in cts for s in c.sides.values())
        dec = sum(s.decode_s for c in cts for s in c.sides.values())
        print(
            f"RTF {2 * n} streams: {audio_s:.1f}s audio per stream in {wall:.2f}s wall -> RTF {wall / audio_s:.3f}"
            f" (per stream-chunk decode {1000 * dec / chunks:.1f} ms avg over {chunks} chunks of 160 ms)"
        )


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="?", default="all", choices=["all", "wer", "sweep", "delay", "rtf"])
    ap.add_argument("--vad-silence", type=float, default=None)
    ap.add_argument("--calls", type=int, default=2)
    a = ap.parse_args(argv)
    cfg = EndpointConfig()
    if a.vad_silence is not None:
        cfg.vad_silence_s = a.vad_silence
    print(f"model {DEFAULT_MODEL_DIR.name}; endpoint cfg {cfg}")
    t = time.perf_counter()
    load_recognizer(cfg=cfg)
    print(f"model load {time.perf_counter() - t:.1f}s")
    if a.what in ("all", "wer"):
        cmd_wer(cfg)
    if a.what in ("all", "sweep"):
        cmd_sweep(cfg)
    if a.what in ("all", "rtf"):
        cmd_rtf(cfg, a.calls)
    if a.what in ("all", "delay"):
        cmd_delay(cfg, a.calls)


if __name__ == "__main__":
    main()

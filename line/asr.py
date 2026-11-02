"""Streaming speech-to-text per side of a call (L2): Nemotron Speech Streaming 0.6B via sherpa-onnx.

Input: Telnyx media-stream frames (we only take the decoded payload bytes; the media
WebSocket itself lives elsewhere). Format, per
https://developers.telnyx.com/docs/voice/programmable-voice/media-streaming (checked 2026-11-02):

    {"event": "connected", "version": "1.0.0"}
    {"event": "start", "sequence_number": "1", "stream_id": "...",
     "start": {"call_control_id": "...", "call_session_id": "...", "from": "...", "to": "...",
               "user_id": "...", "tags": [...], "client_state": "...",
               "media_format": {"encoding": "PCMU", "sample_rate": 8000, "channels": 1}}}
    {"event": "media", "sequence_number": "4", "stream_id": "...",
     "media": {"track": "inbound" | "outbound", "chunk": "2", "timestamp": "5",
               "payload": "<base64 RTP payload, no RTP headers>"}}
    {"event": "stop", "sequence_number": "...", "stream_id": "...",
     "stop": {"user_id": "...", "call_control_id": "..."}}
    (also "dtmf", "mark", "error" events)

- Payload: base64 of raw G.711 (PCMU by default) at 8 kHz mono, one byte per sample.
  Careful: streaming_start `stream_codec` defaults to "default" = the call's codec
  (https://developers.telnyx.com/api-reference/call-commands/streaming-start.md), so L1
  should pass stream_codec="PCMU" and hand us start.media_format.encoding as `encoding`
  (PCMU and PCMA are decoded; anything else raises ValueError in the constructor).
  The docs do not state the frame size; RTP ptime is normally 20 ms = 160 bytes. We do
  not rely on it: time is counted from the number of samples received per track.
- "chunk" is a per-stream sequence number; the docs say events can arrive out of order
  and chunk can be used to reorder (the WebSocket owner should do that if needed).
- "timestamp" units are not documented. If the WebSocket owner can turn it into
  call-relative ms, pass it as `ts_ms` and we pad gaps in a track with silence so both
  sides stay on one clock. Otherwise leave it None.
- stream_track=both_tracks on the caller leg (A): "inbound" = what Telnyx receives from
  the caller = the caller; "outbound" = what Telnyx sends to the caller = the conference
  mix = the protected person (plus anything else in the mix, e.g. the notice/ringback).
  The track meaning is our reading of Telnyx terms; L1 hour-0 check confirms it live.

Model: sherpa-onnx streaming transducer, loaded with OnlineRecognizer.from_transducer(...)
(model_type left empty: sherpa reads the NeMo metadata from the ONNX file), 16 kHz,
80-dim features, greedy search; same files and flags as the CLI example on
https://k2-fsa.github.io/sherpa/onnx/nemo/nemotron-streaming.html .

Endpointing (numbers from `python -m line.bench_asr all`, M5 Pro, 2026-11-02):
- We never call recognizer.reset() while speech may be in flight. Measured on the test
  wavs: resetting at a sherpa endpoint drops the first word(s) of the next utterance
  ("forever", "the squalid", ...), because the endpoint fires on decoder blanks while
  the next word is already in the encoder. So the stream runs continuously and we cut
  turns ourselves by token index; reset only happens after `reset_after_silence_s` of
  VAD silence and no new token, with nothing pending (keeps the token list short).
- Rule A (fast): pending text and a 20 ms frame-energy VAD says the speaker has been
  silent for `vad_silence_s`. VAD: speech = above max(noise floor + 8 dB, abs cut);
  noise floor = quietest frame in the last 3 s; abs cut = -45 dBFS, lowered to
  (talker level - 20 dB) but not below -60 dBFS when the talker is quiet (level = 95th
  percentile of the last 20 s of non-silent frames). Normal talkers get exactly the
  original -45 dBFS behaviour; a fixed -45 alone shredded talkers 20 dB quieter than the
  test wavs into 34 fragments with WER 29% (now 4 turns, WER 1%). Rule A is only used
  while the VAD saw the speech behind the newest token (`vad_tracks_tokens`). Default 0.6 s: endpoint
  delay p50 0.62 s / p95 0.74 s (audio and wall time, 4 live streams). 0.5 s gives p50
  0.50 s but splits ~1 in 3 utterances at comma pauses and the last word sometimes
  arrives after the cut (token emission lags the end of a word by up to ~0.65 s).
- Rule B (fallback, noisy or very quiet lines): sherpa's own rule2, `token_silence_s`
  without a new token, trusted only when the VAD also sees >= 0.3 s silence, the VAD
  looks stuck on (3 s voiced run with no words = line noise), or the VAD can't hear
  this talker at all.
- Handoff: when the other side starts a new utterance and this side has been silent for
  `handoff_silence_s`, this side's pending text is finalized first.
- Max length: a turn is cut after `max_utterance_s` of talk, before the last word that
  started, so words are never split.
- Leading punctuation is stripped (sentence-final "." often arrives with the next words).

Timing: t_start_ms/t_end_ms come from counted samples (8000 per second per track) plus
`t0_ms` (call ms when the stream started), from the VAD runs around the tokens, never
from wall clock. Measured on the test dialog they match the energy ground truth to the
20 ms frame; gap_ms matches the scripted reply gaps.

The recognizer is a process-wide singleton; the EndpointConfig passed on the first
load_recognizer() call sets sherpa's rule thresholds (token_silence_s) for everyone.
"""
from __future__ import annotations

import asyncio
import logging
import os
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from contracts.models import Turn
from line.audio import FRAME_BYTES, PHONE_RATE, SILENCE_BYTE, Upsampler, check_encoding, decode_payload

log = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "sherpa-onnx-nemotron-speech-streaming-en-0.6b-160ms-int8-2026-04-25"
)
TRACK_TO_SPEAKER = {"inbound": "caller", "outbound": "protected"}

_recognizer = None
_recognizer_lock = threading.Lock()
_STOP = object()  # worker sentinel (not None: None also means "nothing carried over")
# A 20 ms frame at or below this is digital (near-)silence, not line noise: the smallest
# non-zero G.711 level (+-8 of 32768, mu-law and A-law alike) is -72 dBFS.
DIGITAL_SILENCE_DB = -70.0


@dataclass
class EndpointConfig:
    vad_silence_s: float = 0.6        # rule A: trailing silence (energy VAD) that ends a turn
    token_silence_s: float = 1.2      # rule B: sherpa rule2, decoder blanks after last token
    handoff_silence_s: float = 0.3    # other side started talking and we've been quiet this long
    max_utterance_s: float = 15.0     # force a cut in a monologue
    reset_after_silence_s: float = 2.0  # reset sherpa state after this much VAD silence, nothing pending
    vad_abs_dbfs: float = -45.0       # speech must be above this (normal talkers, and before the level is known)...
    vad_min_dbfs: float = -60.0       # ...lowered down to this for a quiet talker:
    vad_peak_drop_db: float = 20.0    # abs threshold = talker level - this, clamped to [vad_min_dbfs, vad_abs_dbfs]
    vad_snr_db: float = 8.0           # speech = this far above the noise floor (12 fragments turns at -40 dBFS noise)
    vad_floor_window_s: float = 3.0   # noise floor = quietest 20 ms frame in this window
    vad_level_window_s: float = 20.0  # talker level = 95th percentile of the last this-many s of non-silent frames
    vad_merge_gap_s: float = 0.25     # voiced runs closer than this are one run (for t_start)
    token_lag_s: float = 0.8          # a token arrives up to this long after its audio (measured 0-0.7 s)


def load_recognizer(model_dir: str | os.PathLike | None = None, num_threads: int = 2, cfg: EndpointConfig | None = None):
    """One shared OnlineRecognizer per process (the model is ~650 MB; streams are cheap).

    Endpoint detection is enabled; rule1/rule2 feed rule B above, rule3 is disabled because
    we don't reset per utterance (we cap turn length ourselves).
    """
    global _recognizer
    with _recognizer_lock:
        if _recognizer is None:
            import sherpa_onnx

            cfg = cfg or EndpointConfig()
            d = Path(model_dir or os.environ.get("EMBERZ_ASR_MODEL_DIR") or DEFAULT_MODEL_DIR)
            _recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                tokens=str(d / "tokens.txt"),
                encoder=str(d / "encoder.int8.onnx"),
                decoder=str(d / "decoder.int8.onnx"),
                joiner=str(d / "joiner.int8.onnx"),
                num_threads=num_threads,
                sample_rate=16000,
                feature_dim=80,
                decoding_method="greedy_search",
                enable_endpoint_detection=True,
                rule1_min_trailing_silence=max(2.4, cfg.token_silence_s),
                rule2_min_trailing_silence=cfg.token_silence_s,
                rule3_min_utterance_length=1e9,
            )
            # warm-up: the first ONNX runs are slow; do them here, not on the first live call
            s = _recognizer.create_stream()
            s.accept_waveform(16000, np.zeros(16000, np.float32))
            while _recognizer.is_ready(s):
                _recognizer.decode_streams([s])
        return _recognizer


@dataclass
class Segment:
    speaker: str
    text: str
    t_start_ms: int
    t_end_ms: int
    reason: str            # vad|token|handoff|max|flush
    emitted_at_s: float    # side's audio clock when it was cut (for latency measurement)
    last_token_at_s: float = 0.0  # side's audio clock when the last token arrived


@dataclass
class _Run:
    start: float
    end: float


class SideStream:
    """One speaker: mu-law in, upsample, VAD, sherpa stream, segmenting. Not thread-safe."""

    def __init__(
        self,
        recognizer,
        speaker: str,
        cfg: EndpointConfig,
        t0_s: float = 0.0,
        resample_quality: str = "HQ",
        encoding: str = "PCMU",
    ):
        self.rec = recognizer
        self.encoding = check_encoding(encoding)
        self.speaker = speaker
        self.cfg = cfg
        self.t0_s = t0_s
        self.stream = recognizer.create_stream()
        self.up = Upsampler(resample_quality)
        self.samples_in = 0               # 8 kHz samples received (the clock)
        self._carry = np.zeros(0, np.float32)
        # VAD
        self._recent_db: deque[float] = deque(maxlen=max(1, int(cfg.vad_floor_window_s * 50)))
        self._level_db: deque[float] = deque(maxlen=max(1, int(cfg.vad_level_window_s * 50)))
        self._level = 0.0  # unknown: assume a normal talker (abs cut = vad_abs_dbfs)
        self._level_age = 0
        self.voiced = False
        self.runs: list[_Run] = []        # recent voiced runs (merged), side-clock seconds
        # tokens
        self.consumed = 0                 # tokens already emitted in turns (index into result)
        self.pending: list[tuple[str, float]] = []  # (token, abs time s) not yet emitted
        self.last_token_fed_s = 0.0
        self.seg_start_s: Optional[float] = None
        self.last_emit_end_s = 0.0
        self.is_new_utterance = False     # set when the first pending token appears (handoff cue)
        self.partial_changed = False
        self.decode_s = 0.0               # wall time spent in decode (bench)
        self.decode_chunks = 0

    # ---- clock
    def now_s(self) -> float:
        return self.samples_in / PHONE_RATE

    def call_ms(self, t_s: float) -> int:
        return max(0, int(round((self.t0_s + t_s) * 1000)))

    # ---- input
    def push(self, payload: bytes) -> None:
        """G.711 bytes in: VAD, upsample, hand to the sherpa stream (decode happens elsewhere)."""
        x = decode_payload(payload, self.encoding)
        self._vad(x)
        self.samples_in += len(x)
        y = self.up(x)
        if len(y):
            self.stream.accept_waveform(16000, y)

    def _vad(self, x: np.ndarray) -> None:
        t = (self.samples_in - len(self._carry)) / PHONE_RATE  # clock time of buf[0]
        buf = np.concatenate([self._carry, x]) if len(self._carry) else x
        n = len(buf) // FRAME_BYTES * FRAME_BYTES
        self._carry = buf[n:].copy()
        for i in range(0, n, FRAME_BYTES):
            f = buf[i : i + FRAME_BYTES]
            db = 10.0 * np.log10(float(np.mean(f * f)) + 1e-10)
            self._recent_db.append(db)
            floor = min(self._recent_db)  # minimum statistics: speech has dips within a few s
            # Talker level: 95th percentile of the non-silent frames of the last
            # `vad_level_window_s`, once 0.5 s of them were heard. Digital silence (mu-law 0xFF
            # = 0, A-law 0xD5 = +8; e.g. a conference mix with nobody talking) is left out;
            # refreshed every 200 ms.
            if db > DIGITAL_SILENCE_DB:
                self._level_db.append(db)
                self._level_age += 1
                if self._level_age >= 10 and len(self._level_db) >= 25:
                    self._level_age = 0
                    self._level = float(np.percentile(np.fromiter(self._level_db, float), 95))
            # The builder's fixed -45 dBFS cut shredded quiet talkers into 1-word turns; a
            # fixed lower cut (or a tighter level-relative one) stretched normal turn ends
            # over reverb tails and split off quiet last words. So the absolute cut only drops
            # below -45 dBFS for a talker whose level shows they are quiet.
            cfg = self.cfg
            abs_thr = min(cfg.vad_abs_dbfs, max(cfg.vad_min_dbfs, self._level - cfg.vad_peak_drop_db))
            voiced = db > max(abs_thr, floor + cfg.vad_snr_db)
            fs, fe = t, t + FRAME_BYTES / PHONE_RATE
            if voiced:
                if self.runs and fs - self.runs[-1].end <= self.cfg.vad_merge_gap_s:
                    self.runs[-1].end = fe
                else:
                    self.runs.append(_Run(fs, fe))
                    if len(self.runs) > 200:
                        del self.runs[:100]
            self.voiced = voiced
            t = fe

    def silence_s(self) -> float:
        """Seconds since the last voiced frame (inf if never voiced)."""
        if self.voiced:
            return 0.0
        return self.now_s() - self.runs[-1].end if self.runs else float("inf")

    def collect_tokens(self) -> None:
        r = self.rec.get_result_all(self.stream)
        toks, ts = r.tokens, r.timestamps
        if len(toks) <= self.consumed + len(self.pending):
            return
        new_from = self.consumed + len(self.pending)
        for tok, t in zip(toks[new_from:], ts[new_from:]):
            self.pending.append((tok, r.start_time + t))
        self.last_token_fed_s = self.now_s()
        self.partial_changed = True
        if self.seg_start_s is None and self.pending_text():
            first = next(t for tok, t in self.pending if tok.strip(" .,?!;:"))
            self.seg_start_s = self._onset_for(first)
            self.is_new_utterance = True

    def _onset_for(self, tok_s: float) -> float:
        """Speech onset for a segment whose first word token is at tok_s.

        Token times lag the word onset by 0-0.7 s (measured), so take the earliest voiced run
        that starts in [tok_s - 0.8, tok_s + 0.2] and after the previous turn on this side.
        """
        lo = max(tok_s - 0.8, self.last_emit_end_s)
        # runs shorter than 60 ms are clicks/breaths, not a speech onset
        starts = [r.start for r in self.runs if lo <= r.start <= tok_s + 0.2 and r.end - r.start >= 0.06]
        # a run that began before lo but is still going at lo also counts (onset clipped to lo)
        starts += [lo for r in self.runs if r.start < lo < r.end]
        start = min(starts) if starts else tok_s - 0.3
        return max(start, self.last_emit_end_s, 0.0)

    def pending_text(self) -> str:
        # Sentence-final punctuation often arrives late (with the next words), so a turn can
        # start with "." -- strip it; punctuation alone is not a turn.
        return "".join(t for t, _ in self.pending).lstrip(" .,?!;:").strip()

    # ---- endpointing
    def vad_tracks_tokens(self) -> bool:
        """Did the VAD see the speech behind the newest pending token?

        False means the talker is below the VAD threshold (quiet line, quiet conference mix):
        then VAD silence says nothing about the end of the turn and only sherpa's token rule
        is used. Without this, quiet speech was cut every 0.15 s (mid-word) and the sherpa
        reset below fired between the pieces and dropped words (measured: 20 dB quieter
        test wavs went from 4 turns / WER 0% to 34 turns / WER 29%).
        """
        if not self.pending:
            return True
        return bool(self.runs) and self.runs[-1].end >= self.pending[-1][1] - self.cfg.token_lag_s

    def check_endpoint(self) -> Optional[Segment]:
        cfg = self.cfg
        if not self.pending_text():
            if (
                cfg.reset_after_silence_s
                and self.consumed + len(self.pending) > 0
                and self.silence_s() >= cfg.reset_after_silence_s
                and self.now_s() - self.last_token_fed_s >= cfg.reset_after_silence_s
            ):
                self.rec.reset(self.stream)  # safe: no speech in the pipeline for a while
                self.consumed = 0
                self.pending = []
            return None
        sil = self.silence_s()
        vad_ok = self.vad_tracks_tokens()
        if vad_ok and sil >= cfg.vad_silence_s and self.now_s() - self.last_token_fed_s >= 0.15:
            return self.cut("vad")
        if self.rec.is_endpoint(self.stream) and (sil >= 0.3 or self._voiced_run_s() >= 3.0 or not vad_ok):
            # sherpa rule: no new token for token_silence_s. Only trusted when the VAD agrees,
            # the VAD looks stuck on (a long voiced run without words = line noise), or the
            # VAD cannot hear this talker at all.
            return self.cut("token")
        if self.seg_start_s is not None and self.now_s() - self.seg_start_s >= cfg.max_utterance_s:
            return self.cut("max")
        return None

    def _voiced_run_s(self) -> float:
        return self.runs[-1].end - self.runs[-1].start if self.voiced and self.runs else 0.0

    def can_handoff(self) -> bool:
        if not self.pending_text() or self.silence_s() < self.cfg.handoff_silence_s:
            return False
        # a talker the VAD can't hear: at least no new word for a moment
        return self.vad_tracks_tokens() or self.now_s() - self.last_token_fed_s >= 0.5

    def cut(self, reason: str) -> Optional[Segment]:
        if not self.pending_text():
            return None
        start = self.seg_start_s if self.seg_start_s is not None else self.pending[-1][1]
        keep: list[tuple[str, float]] = []
        if reason == "max":
            # cut before the last word that starts in this turn, so no word is split
            i = max((k for k, (tok, _) in enumerate(self.pending) if tok.startswith(" ")), default=0)
            if i > 0 and "".join(t for t, _ in self.pending[:i]).strip(" .,?!;:"):
                keep = self.pending[i:]
                self.pending = self.pending[:i]
            end = keep[0][1] - 0.1 if keep else self.now_s()
        elif self.voiced:
            end = self.now_s()
        elif self.runs and self.runs[-1].end > start:
            end = self.runs[-1].end
        else:
            end = self.pending[-1][1] + 0.2
        end = max(end, start)
        text = self.pending_text()
        seg = Segment(self.speaker, text, self.call_ms(start), self.call_ms(end), reason, self.now_s(), self.last_token_fed_s)
        self.consumed += len(self.pending)
        self.pending = keep
        self.seg_start_s = end if keep else None
        self.last_emit_end_s = end
        self.partial_changed = bool(keep)
        return seg


OnTurn = Callable[[Turn], None]
OnPartial = Callable[[str, str, int], None]  # (speaker, text so far, t_start_ms)


class CallTranscriber:
    """Both sides of one call. feed() decodes synchronously; use AsyncCallTranscriber in the server.

    Turns are numbered in emission order across both sides. gap_ms = this turn's start minus
    the previous turn's end when the previous turn is the other speaker (a reply gap; negative
    = overlap); None for the first turn and when the same speaker continues.
    """

    def __init__(
        self,
        call_id: str,
        on_turn: OnTurn,
        on_partial: Optional[OnPartial] = None,
        *,
        t0_ms: int = 0,
        cfg: EndpointConfig | None = None,
        recognizer=None,
        resample_quality: str = "HQ",
        encoding: str = "PCMU",
    ):
        self.call_id = call_id
        self.silence_byte = SILENCE_BYTE[check_encoding(encoding)]
        self.on_turn = on_turn
        self.on_partial = on_partial
        self.cfg = cfg or EndpointConfig()
        self.rec = rec = recognizer or load_recognizer(cfg=self.cfg)
        self.sides = {
            track: SideStream(rec, spk, self.cfg, t0_ms / 1000.0, resample_quality, encoding)
            for track, spk in TRACK_TO_SPEAKER.items()
        }
        self.turn_no = 0
        self.last_turn: Optional[Turn] = None
        self.turns: list[Turn] = []
        self.segments: list[Segment] = []  # same order as turns, with cut reason (bench/debug)

    def feed(self, track: str, payload: bytes, ts_ms: Optional[int] = None) -> list[Turn]:
        """Feed one media payload (G.711 bytes) for "inbound" (caller) or "outbound" (protected).

        Frames must arrive in order per track (reorder by media.chunk first) and without
        duplicates; ts_ms (call ms of the payload's first sample) only fills holes.
        """
        return self.feed_round([(track, payload, ts_ms)])

    def feed_round(self, items: list[tuple[str, bytes, Optional[int]]]) -> list[Turn]:
        """Feed several payloads (normally one per track) and decode the ready streams in one batch."""
        touched: list[SideStream] = []
        for track, payload, ts_ms in items:
            side = self.sides.get(track)
            if side is None:
                continue
            if ts_ms is not None:  # pad a gap in this track with silence so both sides share one clock
                behind_ms = ts_ms - side.call_ms(side.now_s())
                if behind_ms > 40:  # capped at 5 s: a bigger jump is more likely a units bug than a hole
                    pad_ms = min(behind_ms, 5000)
                    payload = bytes([self.silence_byte]) * int(pad_ms * PHONE_RATE / 1000) + payload
            side.push(payload)
            if side not in touched:
                touched.append(side)
        self._decode(touched)
        out: list[Turn] = []
        for side in touched:
            out += self._after_decode(side)
        return out

    def _decode(self, sides: list[SideStream]) -> None:
        decoded = set()
        t = time.perf_counter()
        while True:
            ready = [s for s in sides if self.rec.is_ready(s.stream)]
            if not ready:
                break
            # one batched ONNX run for both sides (~30-40% cheaper than two runs, measured)
            self.rec.decode_streams([s.stream for s in ready])
            for s in ready:
                s.decode_chunks += 1
                decoded.add(id(s))
        dt = time.perf_counter() - t
        for s in sides:
            if id(s) in decoded:
                s.decode_s += dt / len(decoded)
                s.collect_tokens()

    def _after_decode(self, side: SideStream) -> list[Turn]:
        out: list[Turn] = []
        if side.is_new_utterance:
            side.is_new_utterance = False
            for other in self.sides.values():
                if other is not side and other.can_handoff():
                    out.append(self._emit(other.cut("handoff")))
        seg = side.check_endpoint()
        if seg:
            out.append(self._emit(seg))
        elif side.partial_changed and self.on_partial and side.pending_text():
            side.partial_changed = False
            self.on_partial(side.speaker, side.pending_text(), side.call_ms(side.seg_start_s or 0.0))
        return out

    def _emit(self, seg: Segment) -> Turn:
        self.turn_no += 1
        gap = None
        if self.last_turn is not None and self.last_turn.speaker != seg.speaker:
            gap = seg.t_start_ms - self.last_turn.t_end_ms
        turn = Turn(
            call_id=self.call_id,
            turn=self.turn_no,
            speaker=seg.speaker,
            text=seg.text,
            t_start_ms=seg.t_start_ms,
            t_end_ms=seg.t_end_ms,
            gap_ms=gap,
        )
        self.last_turn = turn
        self.turns.append(turn)
        self.segments.append(seg)
        self.on_turn(turn)
        return turn

    def finish(self, tail_ms: int = 800) -> list[Turn]:
        """End of call: push a little silence through both sides to flush the encoder, then
        emit whatever is still pending (earliest start first)."""
        out: list[Turn] = []
        silence = bytes([self.silence_byte]) * FRAME_BYTES
        for _ in range(tail_ms // 20):
            out += self.feed_round([(track, silence, None) for track in self.sides])
        rest = [s.cut("flush") for s in self.sides.values() if s.pending_text()]
        for seg in sorted((s for s in rest if s), key=lambda s: s.t_start_ms):
            out.append(self._emit(seg))
        return out


class AsyncCallTranscriber:
    """Same feed() API, but decoding runs on one worker thread; callbacks land on `loop`.

    feed() never blocks the event loop (it only enqueues). close() flushes pending text,
    stops the worker and waits for it (call from a thread, or use `await aclose()`).
    Call load_recognizer() once at server start-up: the first load takes 1-3 s and the
    constructor would otherwise do it on the event loop.
    """

    def __init__(
        self,
        call_id: str,
        on_turn: OnTurn,
        loop: asyncio.AbstractEventLoop,
        on_partial: Optional[OnPartial] = None,
        **kwargs,
    ):
        self.loop = loop
        self._user_on_turn = on_turn
        self._user_on_partial = on_partial
        self._q: queue.Queue = queue.Queue()
        self.inner = CallTranscriber(
            call_id,
            on_turn=lambda t: self._post(self._user_on_turn, t),
            on_partial=(
                (lambda s, txt, ms: self._post(self._user_on_partial, s, txt, ms))
                if on_partial
                else None
            ),
            **kwargs,
        )
        self.error: Optional[BaseException] = None
        self.backlog_warn_frames = 100
        self._warned_backlog = False
        self._closed = False
        self._thread = threading.Thread(target=self._run, name=f"asr-{call_id}", daemon=True)
        self._thread.start()

    def feed(self, track: str, payload: bytes, ts_ms: Optional[int] = None) -> None:
        if not self._closed:
            self._q.put((track, payload, ts_ms))

    def queue_depth(self) -> int:
        return self._q.qsize()

    def _run(self) -> None:
        # Bug fixed in review: the sentinel used to be None, and when the round below pulled
        # it with get_nowait() it went into `carry`, where None means "nothing carried", so
        # it was lost: the worker blocked forever, finish() never ran and close() hung.
        carry = None
        while True:
            item = carry if carry is not None else self._q.get()
            carry = None
            if item is _STOP:
                break
            # A round = queued payloads up to one per track, decoded together. Rounds stay
            # one frame per track even when behind, so endpointing sees every 20 ms step.
            round_ = [item]
            while len(round_) < len(self.inner.sides):
                try:
                    nxt = self._q.get_nowait()
                except queue.Empty:
                    break
                if nxt is _STOP or any(nxt[0] == r[0] for r in round_):
                    carry = nxt
                    break
                round_.append(nxt)
            try:
                self.inner.feed_round(round_)
            except Exception as e:  # keep the call alive; surface the error to the owner
                if self.error is None:
                    log.exception("ASR %s: feed failed (further errors not logged)", self.inner.call_id)
                self.error = e
            depth = self._q.qsize()
            if depth > self.backlog_warn_frames and not self._warned_backlog:
                self._warned_backlog = True  # once per call: ASR is >1 s behind real time
                log.warning("ASR %s: %d frames queued, decoding is falling behind", self.inner.call_id, depth)
        try:
            self.inner.finish()
        except Exception as e:
            log.exception("ASR %s: finish failed", self.inner.call_id)
            self.error = e

    def _post(self, fn, *args) -> None:
        try:
            self.loop.call_soon_threadsafe(fn, *args)
        except RuntimeError:  # loop already closed (shutdown); drop the event
            pass

    def close(self, timeout: Optional[float] = None) -> None:
        if not self._closed:
            self._closed = True
            self._q.put(_STOP)
        # Waits until the queued audio is decoded and the last turns are emitted (on a live
        # call the backlog is a few frames; pass a timeout to bound it). The media WebSocket
        # owner must call this in a `finally`, also when the socket drops without "stop":
        # otherwise the worker thread and its sherpa streams live until the process exits.
        self._thread.join(timeout)

    async def aclose(self, timeout: Optional[float] = None) -> None:
        await asyncio.to_thread(self.close, timeout)

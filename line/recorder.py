"""Call recorder for V2 (real test calls) and V1 (human vs AI audio).

Writes, per call, under recordings/<call_id>/:
  caller.wav     8 kHz 16-bit mono PCM, the caller side (Telnyx track "inbound" on leg A)
  protected.wav  8 kHz 16-bit mono PCM, the protected side (track "outbound" on leg A = conference mix)
  turns.jsonl    one contracts.models.Turn per line
  events.jsonl   one contracts.models.ConsoleEvent per line (phone numbers hashed)
  meta.json      written on close(): duration, per-track stats, hashed numbers, context, source

Both wavs share one time axis: sample 0 = `t0` (pass CallSession.started_monotonic so it
matches Turn.t_start_ms). Each track is anchored at the arrival time of its first frame;
after that, positions come from the Telnyx media `timestamp` when given (gaps are padded
with silence, out-of-order chunks are put back in place), or from arrival time otherwise.
On close the shorter wav is padded so both have the same length.

Telnyx media frame (https://developers.telnyx.com/api-reference/websockets/stream-call-media-over-websocket.md):
  {"event": "media", "media": {"track": "inbound|outbound", "chunk": "2", "timestamp": "5", "payload": "<base64>"}}
The docs call `timestamp` the "RTP timestamp" without a unit, so ts_unit="auto" works it out
from the first frames (RTP sample ticks at 8 kHz vs milliseconds; "none" when the timestamps do
not advance, e.g. a missing field turned into 0, and then arrival time is used). Pin it once
L1's hour-0 check has seen real frames.

Timestamp discontinuities (a new RTP source when the leg moves into the conference, a reset):
when a frame's timestamp position disagrees with its arrival time by more than `resync_ms`
compared to the previous frame, the track is re-anchored on arrival time instead of writing
hours of silence or dropping the rest of the call as "late".

The recorder never raises from feed()/feed_telnyx(): bad frames are counted and skipped, and
a disk error stops the audio (meta.json "audio_error") without touching the call.

Cheap on the media path: feed() only appends raw bytes to a list; decode + file write happen
in flush(), which feed() triggers every `flush_every_s` of audio (default 2 s per track).
`wave` patches the header after every write, so the wavs stay playable even after a crash.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import time
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from contracts.models import ConsoleEvent, Turn

log = logging.getLogger("emberz.recorder")

SAMPLE_RATE = 8000
# Repo-root recordings/, the folder callers/label_sheet.py reads (not relative to the server's cwd).
RECORDINGS_DIR = Path(__file__).resolve().parent.parent / "recordings"
TRACKS = ("caller", "protected")
TELNYX_TRACK = {"inbound": "caller", "outbound": "protected"}


def _ulaw_table() -> np.ndarray:
    """G.711 mu-law -> int16 lookup table (audioop is gone in Python 3.13)."""
    u = (~np.arange(256, dtype=np.int32)) & 0xFF
    sign = u & 0x80
    exponent = (u >> 4) & 0x07
    mantissa = u & 0x0F
    sample = (((mantissa << 3) + 0x84) << exponent) - 0x84
    return np.where(sign != 0, -sample, sample).astype("<i2")


ULAW_TO_PCM16 = _ulaw_table()


def ulaw_decode(data: bytes) -> bytes:
    return ULAW_TO_PCM16[np.frombuffer(data, dtype=np.uint8)].tobytes()


# Salted so hashes can't be reversed by enumerating phone numbers. Set EMBERZ_NUMBER_SALT to
# link the same caller across runs; without it the salt is random per process.
_SALT = os.environ.get("EMBERZ_NUMBER_SALT", "").encode() or secrets.token_bytes(16)


ANONYMOUS = {"anonymous", "restricted", "private", "unknown", "unavailable"}


def hash_number(number: Optional[str]) -> Optional[str]:
    """Salted hash of a phone number. "+1 (555) 123-4567" and "15551234567" hash the same.
    Withheld callers (Telnyx sends e.g. "anonymous") become "anonymous", not a hash."""
    if not number:
        return None
    digits = "".join(c for c in number if c.isdigit())
    if not digits:
        if number.strip().lower() in ANONYMOUS:
            return "anonymous"
        digits = number.strip().lower()          # e.g. a SIP URI without digits
    return "h:" + hmac.new(_SALT, digits.encode(), hashlib.sha256).hexdigest()[:16]


_E164 = re.compile(r"\+?\d{10,15}")


def _scrub(data: Any) -> Any:
    """Hash phone numbers in an event payload: fields named `*_number`, `from`, `to`, `number`,
    and any other string value that is a bare E.164-looking number (free text is left alone)."""
    if isinstance(data, dict):
        out = {}
        for k, v in data.items():
            if isinstance(v, str) and (k.endswith("_number") or k in ("from", "to", "number") or _E164.fullmatch(v.strip())):
                out[k] = hash_number(v)
            else:
                out[k] = _scrub(v)
        return out
    if isinstance(data, list):
        return [_scrub(v) for v in data]
    return data


class _Track:
    def __init__(self, path: Path, ts_unit: str):
        self.wav = wave.open(str(path), "wb")
        self.wav.setnchannels(1)
        self.wav.setsampwidth(2)
        self.wav.setframerate(SAMPLE_RATE)
        self.ts_unit = ts_unit           # "ms" | "samples" | "none" | "auto"
        self.written = 0                 # samples already in the file
        self.pending: list[tuple[int, bytes]] = []   # (sample position, ulaw bytes)
        self.undetermined: list[tuple[int, bytes, int]] = []  # (timestamp, ulaw, arrival pos) while ts_unit is unknown
        self.max_end = 0                 # furthest sample position seen
        self.anchor: Optional[int] = None   # sample position of timestamp ts0
        self.ts0: Optional[int] = None
        self.cursor: Optional[int] = None   # next position for frames placed by arrival time
        self.last: Optional[tuple[int, int, int]] = None  # (pos, arrival pos, n) of the last timestamped frame
        self.frames = 0
        self.late_dropped = 0
        self.padded = 0                  # samples of silence inserted
        self.resyncs = 0                 # timestamp discontinuities re-anchored on arrival time
        self.first_arrival: Optional[int] = None  # sample position (arrival time) of the first frame


class CallRecorder:
    def __init__(
        self,
        call_id: str,
        out_dir: str | Path = RECORDINGS_DIR,
        *,
        t0: Optional[float] = None,
        numbers: Optional[dict[str, Optional[str]]] = None,
        context: Optional[dict[str, Any]] = None,
        source: Optional[dict[str, Any]] = None,
        ts_unit: str = "auto",
        flush_every_s: float = 2.0,
        reorder_window_ms: int = 500,
        gap_threshold_ms: int = 300,
        resync_ms: int = 5000,
        clock: Callable[[], float] = time.monotonic,
    ):
        """
        t0: monotonic time of call start (CallSession.started_monotonic). Default: now.
        numbers: e.g. {"caller": "+1555...", "guarded": "+1555..."}; stored hashed only.
        context: guardian inputs safe to show the labeler: {"profile", "rules", "caller_known"}.
        source: who placed the call, e.g. {"kind": "vapi", "agent": "bank_fraud", "variant": 3};
                kept in meta.json, never shown on the label sheet.
        ts_unit: unit of the Telnyx media timestamp: "auto", "ms", "samples" (8 kHz RTP ticks) or
                 "none" (ignore timestamps, place frames by arrival time).
        resync_ms: re-anchor a track on arrival time when its timestamps jump by more than this.
        Raises FileExistsError if recordings/<call_id>/ already holds a recording.
        """
        if ts_unit not in ("auto", "ms", "samples", "none"):
            raise ValueError(f"ts_unit must be auto|ms|samples|none, not {ts_unit!r}")
        if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", call_id) or call_id.startswith("."):
            raise ValueError(f"unsafe call_id for a folder name: {call_id!r}")
        self.call_id = call_id
        self.dir = Path(out_dir) / call_id
        self.dir.mkdir(parents=True, exist_ok=True)
        if (self.dir / "caller.wav").exists():
            # A second recorder (duplicate webhook, restart) would truncate the first one's audio.
            raise FileExistsError(f"{self.dir} already has a recording; one CallRecorder per call_id")
        self._clock = clock
        self.t0 = clock() if t0 is None else t0
        self.started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.numbers = {k: hash_number(v) for k, v in (numbers or {}).items()}
        self.context = context or {}
        self.source = source or {}
        self._flush_every = int(flush_every_s * SAMPLE_RATE)
        self._reorder = int(reorder_window_ms * SAMPLE_RATE / 1000)
        self._gap = int(gap_threshold_ms * SAMPLE_RATE / 1000)
        self._resync = int(resync_ms * SAMPLE_RATE / 1000)
        self.skipped_frames = 0          # unknown track, bad base64, bad timestamp
        self.audio_error: Optional[str] = None
        self._tracks = {name: _Track(self.dir / f"{name}.wav", ts_unit) for name in TRACKS}
        self._turns = open(self.dir / "turns.jsonl", "a", encoding="utf-8")
        self._events = open(self.dir / "events.jsonl", "a", encoding="utf-8")
        self.n_turns = 0
        self.n_events = 0
        self.closed = False
        self.meta: dict[str, Any] = {}

    # ------------------------------------------------------------------ audio
    def feed_telnyx(self, media: dict[str, Any], arrival: Optional[float] = None) -> None:
        """Feed the `media` object of a Telnyx media event (track, chunk, timestamp, payload)."""
        try:
            ulaw = base64.b64decode(media.get("payload") or "", validate=True)
            ts = media.get("timestamp")
            ts = int(ts) if ts not in (None, "") else None
        except (binascii.Error, ValueError, TypeError):
            self.skipped_frames += 1
            return
        self.feed(media.get("track", "inbound"), ulaw, timestamp=ts, arrival=arrival)

    def feed(self, track: str, ulaw: bytes, timestamp: Optional[int] = None, arrival: Optional[float] = None) -> None:
        """Add one mu-law frame. track: caller|protected (or Telnyx inbound|outbound).
        Pass timestamp=None (not 0) when the frame has no timestamp."""
        if self.closed or self.audio_error or not ulaw:
            return
        t = self._tracks.get(TELNYX_TRACK.get(track, track))
        if t is None:
            self.skipped_frames += 1
            return
        arrival_pos = max(0, int(((self._clock() if arrival is None else arrival) - self.t0) * SAMPLE_RATE))
        t.frames += 1
        if t.first_arrival is None:
            t.first_arrival = arrival_pos

        if timestamp is None or t.ts_unit == "none":
            self._place_by_arrival(t, ulaw, arrival_pos)
        else:
            if t.ts0 is None:
                t.anchor, t.ts0 = arrival_pos, timestamp
            if t.ts_unit == "auto":
                t.undetermined.append((timestamp, ulaw, arrival_pos))
                self._maybe_detect_unit(t)
            else:
                self._place_by_ts(t, timestamp, ulaw, arrival_pos)

        if t.max_end - t.written >= self._flush_every + self._reorder:
            self._safe_flush(t, final=False)

    def _place_by_arrival(self, t: _Track, ulaw: bytes, arrival_pos: int) -> None:
        """No timestamp: contiguous, but jump ahead (pad) when arrival shows a real gap."""
        n = len(ulaw)
        if t.cursor is None:
            t.cursor = arrival_pos
        pos = t.cursor
        if arrival_pos - n - t.cursor > self._gap:
            pos = arrival_pos - n
        t.cursor = pos + n
        self._queue(t, pos, ulaw)

    def _place_by_ts(self, t: _Track, ts: int, ulaw: bytes, arrival_pos: int) -> None:
        pos = self._ts_pos(t, ts)
        if t.last is not None:
            last_pos, last_arrival, last_n = t.last
            expected = last_pos + (arrival_pos - last_arrival)
            if abs(pos - expected) > self._resync:
                # Timestamp discontinuity: re-anchor on arrival time, right after the last frame at the earliest.
                pos = max(expected, last_pos + last_n)
                t.ts0, t.anchor = ts, pos
                t.resyncs += 1
        t.last = (pos, arrival_pos, len(ulaw))
        self._queue(t, pos, ulaw)

    @staticmethod
    def _ts_delta(t: _Track, ts: int) -> int:
        delta = ts - t.ts0
        if delta < -(1 << 31):        # 32-bit RTP timestamp wrapped
            delta += 1 << 32
        elif delta > (1 << 31):
            delta -= 1 << 32
        return delta

    def _ts_pos(self, t: _Track, ts: int) -> int:
        delta = self._ts_delta(t, ts)
        if t.ts_unit == "samples":
            return t.anchor + delta
        return t.anchor + delta * SAMPLE_RATE // 1000

    def _maybe_detect_unit(self, t: _Track, force: bool = False) -> None:
        """Compare timestamp advance with audio received: ~1x -> RTP sample ticks, ~1/8x -> ms,
        no advance at all -> "none" (ignore timestamps, place by arrival time)."""
        if len(t.undetermined) < 10 and not force:
            return
        ratios = []
        received = 0
        for ts, ulaw, _ in t.undetermined:
            if received and ts != t.ts0:
                ratios.append(self._ts_delta(t, ts) / received)
            received += len(ulaw)
        if ratios:
            r = float(np.median(ratios))
            # Closer to 1 (samples) or to 1/8 (ms) in log space; geometric midpoint is 1/sqrt(8).
            t.ts_unit = "samples" if r > (1 / 8) ** 0.5 else "ms"
        else:
            t.ts_unit = "none" if len(t.undetermined) > 1 else "ms"
        for ts, ulaw, arrival_pos in t.undetermined:
            if t.ts_unit == "none":
                self._place_by_arrival(t, ulaw, arrival_pos)
            else:
                self._place_by_ts(t, ts, ulaw, arrival_pos)
        t.undetermined.clear()

    @staticmethod
    def _queue(t: _Track, pos: int, ulaw: bytes) -> None:
        t.pending.append((pos, ulaw))
        t.max_end = max(t.max_end, pos + len(ulaw))

    def _flush_track(self, t: _Track, final: bool) -> None:
        if final and t.undetermined:
            self._maybe_detect_unit(t, force=True)
        horizon = t.max_end if final else t.max_end - self._reorder
        ready = [p for p in t.pending if p[0] < horizon or final]
        if not ready:
            return
        t.pending = [p for p in t.pending if not (p[0] < horizon or final)]
        ready.sort(key=lambda p: p[0])
        out = bytearray()
        for pos, ulaw in ready:
            if pos + len(ulaw) <= t.written:      # entirely behind what is already written
                t.late_dropped += 1
                continue
            if pos < t.written:                    # overlaps: keep only the new tail
                ulaw = ulaw[t.written - pos:]
                pos = t.written
            if pos > t.written:                    # gap: pad with silence
                out += bytes(2 * (pos - t.written))
                t.padded += pos - t.written
                t.written = pos
            out += ulaw_decode(ulaw)
            t.written += len(ulaw)
        if out:
            t.wav.writeframes(bytes(out))

    def _safe_flush(self, t: _Track, final: bool) -> None:
        try:
            self._flush_track(t, final)
        except OSError as e:                       # disk full etc.: stop recording audio, keep the call alive
            self.audio_error = f"{type(e).__name__}: {e}"
            log.error("recorder %s: audio stopped: %s", self.call_id, self.audio_error)

    def flush(self) -> None:
        for t in self._tracks.values():
            self._safe_flush(t, final=False)

    # ------------------------------------------------------------------ transcript + events
    def add_turn(self, turn: Turn) -> None:
        if self.closed:
            return
        self._turns.write(turn.model_dump_json() + "\n")
        self._turns.flush()
        self.n_turns += 1

    def add_event(self, event: ConsoleEvent) -> None:
        if self.closed:
            return
        row = event.model_dump(mode="json")
        row["data"] = _scrub(row["data"])
        self._events.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._events.flush()
        self.n_events += 1

    # ------------------------------------------------------------------ close
    def close(self, reason: str = "") -> dict[str, Any]:
        if self.closed:
            return self.meta
        for t in self._tracks.values():
            if not self.audio_error:
                self._safe_flush(t, final=True)
        total = max(t.written for t in self._tracks.values())
        for t in self._tracks.values():               # same length -> aligned files
            try:
                if t.written < total and not self.audio_error:
                    t.wav.writeframes(bytes(2 * (total - t.written)))
                    t.padded += total - t.written
                    t.written = total
                t.wav.close()
            except OSError as e:
                self.audio_error = self.audio_error or f"{type(e).__name__}: {e}"
        self._turns.close()
        self._events.close()
        self.closed = True
        self.meta = {
            "call_id": self.call_id,
            "started_at_utc": self.started_at,
            "closed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "duration_s": round(total / SAMPLE_RATE, 3),
            "sample_rate": SAMPLE_RATE,
            "tracks": {
                name: {
                    "file": f"{name}.wav",
                    "samples": t.written,
                    "frames": t.frames,
                    "first_frame_ms": None if t.first_arrival is None else t.first_arrival * 1000 // SAMPLE_RATE,
                    "padded_ms": t.padded * 1000 // SAMPLE_RATE,
                    "late_dropped_frames": t.late_dropped,
                    "ts_unit": t.ts_unit,
                    "ts_resyncs": t.resyncs,
                }
                for name, t in self._tracks.items()
            },
            "numbers_hashed": self.numbers,
            "context": self.context,
            "source": self.source,
            "turns": self.n_turns,
            "events": self.n_events,
            "skipped_frames": self.skipped_frames,
            "audio_error": self.audio_error,
            "close_reason": reason,
        }
        (self.dir / "meta.json").write_text(json.dumps(self.meta, indent=2), encoding="utf-8")
        return self.meta

    def __enter__(self) -> "CallRecorder":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close(reason="exception" if exc[0] else "")

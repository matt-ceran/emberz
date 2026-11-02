"""Live AI-voice detection on the caller's audio, and fusion of the three voice signals.

    det = LiveVoiceDetector(call_id, on_signal, loop=asyncio.get_running_loop(),
                            now_ms=session.now_ms)
    det.push_mulaw(payload)          # Telnyx inbound track: 8 kHz mu-law bytes
    # or det.push_pcm16k(samples)    # if the line already decoded/upsampled to 16 kHz

The detector gates speech with the same SpeechGate as training (voice/features.py) and
scores the first 2 s and the first 4 s of caller *speech* (silence does not count). Each
score runs on one shared worker thread, so the audio callback never blocks, and
`on_signal(VoiceSignal(model="w2v-head-v1"))` is called on the event loop if `loop` is
given (thread-safe), else on the worker thread.

The model and head load once per process. Call `warmup()` at server start to avoid a
~2 s first-call delay; it blocks for the load, so from async code use
`await asyncio.to_thread(warmup)`. It returns False (and logs) if the head file does not
exist yet, so a server can start before the head is trained; detectors then log the
error and emit nothing. Scoring errors are logged, never raised into the audio path. Measured on the M5 Pro (MPS): 15 ms for a 3 s clip with 6 layers,
21 ms with 12, 35 ms with all 24 (`python -m voice.features latency`).
"""
from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from contracts.models import AiVoice, VoiceSignal
from voice.features import SpeechGate
from voice.phone import MODEL_SR, MulawTo16k, to_mono_float

DEFAULT_HEAD = os.environ.get("EMBERZ_VOICE_HEAD", str(Path(__file__).parent / "heads" / "w2v-head-v1.joblib"))
MODEL_NAME = "w2v-head-v1"
log = logging.getLogger("emberz.voice")

_shared_lock = threading.Lock()
_shared: dict[str, "Scorer"] = {}
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="voice-detector")


class Scorer:
    """Featurizer + head; `score(clip16k) -> P(AI)`. Swappable in tests."""

    def __init__(self, head_path: str = DEFAULT_HEAD, device: str = "auto"):
        from voice.features import W2VFeaturizer
        from voice.train_head import load_head

        self.head = load_head(head_path)
        self.feat = W2VFeaturizer(self.head.ckpt, [self.head.layer], device)

    def score(self, clip: np.ndarray) -> float:
        return float(self.head.predict(self.feat.embed([clip])[:, 0, :])[0])


def shared_scorer(head_path: str = DEFAULT_HEAD) -> Scorer:
    with _shared_lock:
        if head_path not in _shared:
            _shared[head_path] = Scorer(head_path)
        return _shared[head_path]


def warmup(head_path: str = DEFAULT_HEAD) -> bool:
    """Load model + head and run one dummy clip. False (logged) if that fails, e.g. no head yet."""
    try:
        s = shared_scorer(head_path)
        rng = np.random.default_rng(0)
        for sec in (2.0, 4.0):  # the live clip lengths: the first MPS run of each shape is slow (~0.5 s)
            s.score((rng.standard_normal(int(sec * MODEL_SR)) * 0.05).astype(np.float32))
        return True
    except Exception as e:
        log.error("voice detector warmup failed (%s: %s); AI-voice detector disabled until fixed",
                  type(e).__name__, e)
        return False


class LiveVoiceDetector:
    def __init__(self, call_id: str, on_signal: Callable[[VoiceSignal], None], scorer: Optional[Scorer] = None,
                 score_at_s=(2.0, 4.0), now_ms: Optional[Callable[[], int]] = None, loop=None,
                 head_path: str = DEFAULT_HEAD):
        self.call_id = call_id
        self.on_signal = on_signal
        self._scorer = scorer
        self._head_path = head_path
        self.score_at = sorted(score_at_s)
        self.now_ms = now_ms
        self.loop = loop
        self._gate = SpeechGate()
        self._mulaw = MulawTo16k()
        self._speech: list[np.ndarray] = []
        self._speech_n = 0
        self._samples_in = 0
        self._next = 0
        self.pending: list[Future] = []
        self.signals: list[VoiceSignal] = []
        self.closed = False

    # ---- audio in
    def push_mulaw(self, payload: bytes) -> None:
        self.push_pcm16k(self._mulaw(payload))

    def push_pcm16k(self, x: np.ndarray) -> None:
        if self.closed or self._next >= len(self.score_at):
            return
        x = to_mono_float(x)  # int16 PCM is scaled to [-1, 1]; float passes through
        self._samples_in += len(x)
        sp = self._gate.process(x)
        if len(sp):
            self._speech.append(sp)
            self._speech_n += len(sp)
        while self._next < len(self.score_at) and self._speech_n >= self.score_at[self._next] * MODEL_SR:
            n = int(self.score_at[self._next] * MODEL_SR)
            clip = np.concatenate(self._speech)[:n]
            at_ms = self.now_ms() if self.now_ms else int(self._samples_in * 1000 / MODEL_SR)
            self.pending.append(_executor.submit(self._score, clip, at_ms))
            self._next += 1

    def close(self) -> None:
        self.closed = True

    # ---- worker thread
    def _score(self, clip: np.ndarray, at_ms: int) -> Optional[VoiceSignal]:
        try:
            scorer = self._scorer or shared_scorer(self._head_path)
            p = float(scorer.score(clip))
            if not np.isfinite(p):
                raise ValueError(f"non-finite score {p}")
        except Exception as e:  # missing head, model error: log, never crash the call
            log.error("call %s: voice detector failed (%s: %s)", self.call_id, type(e).__name__, e)
            return None
        sig = VoiceSignal(call_id=self.call_id, at_ms=max(int(at_ms), 0), ai_score=round(min(max(p, 0.0), 1.0), 4),
                          model=MODEL_NAME)
        self.signals.append(sig)
        if self.closed:
            return sig
        try:
            if self.loop is not None:
                self.loop.call_soon_threadsafe(self.on_signal, sig)
            else:
                self.on_signal(sig)
        except Exception as e:  # loop already closed, or the callback raised
            log.error("call %s: voice signal delivery failed (%s: %s)", self.call_id, type(e).__name__, e)
        return sig


# --------------------------------------------------------------------------- fusion
FUSION_WEIGHTS = {"detector": 1.0, "timing": 0.5, "telnyx": 1.0}  # starting values; refit with voice.eval fusion


def _logit(p: float, eps: float = 1e-4) -> float:
    p = min(max(p, eps), 1 - eps)
    return float(np.log(p / (1 - p)))


def fuse(detector: Optional[VoiceSignal], timing: Optional[VoiceSignal], telnyx: Optional[VoiceSignal],
         weights: dict = FUSION_WEIGHTS) -> Optional[VoiceSignal]:
    """Late fusion: weighted mean of the available signals' logits -> VoiceSignal(model="combined")."""
    parts = [(s, weights[k]) for k, s in (("detector", detector), ("timing", timing), ("telnyx", telnyx)) if s is not None]
    if not parts:
        return None
    z = sum(_logit(s.ai_score) * w for s, w in parts) / sum(w for _, w in parts)
    return VoiceSignal(call_id=parts[0][0].call_id, at_ms=max(s.at_ms for s, _ in parts),
                       ai_score=round(1 / (1 + np.exp(-z)), 4), model="combined")


def to_ai_voice(sig: VoiceSignal) -> AiVoice:
    """For GuardianRequest.ai_voice. AiVoice.source has no "combined" (see report), so it maps to "detector"."""
    source = {"telnyx": "telnyx", "timing": "timing"}.get(sig.model, "detector")
    return AiVoice(score=sig.ai_score, source=source)

"""LiveVoiceDetector checkpoints, threading, asyncio delivery; fuse(); to_ai_voice()."""
from __future__ import annotations

import asyncio
import threading

import numpy as np
import pytest

from contracts.models import VoiceSignal
from voice.live import LiveVoiceDetector, fuse, to_ai_voice
from voice.phone import PHONE_SR, mulaw_encode


class FakeScorer:
    def __init__(self, p=0.8):
        self.p, self.lengths, self.threads = p, [], []

    def score(self, clip):
        self.lengths.append(len(clip))
        self.threads.append(threading.current_thread().name)
        return self.p


def tone_8k(seconds, amp=0.3, f=440.0):
    t = np.arange(int(seconds * PHONE_SR)) / PHONE_SR
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


def feed_mulaw(det, x8):
    codes = mulaw_encode(x8).tobytes()
    for i in range(0, len(codes), 160):  # 20 ms Telnyx frames
        det.push_mulaw(codes[i: i + 160])


def test_scores_at_2s_and_4s_of_speech_skipping_silence():
    got, sc = [], FakeScorer(0.8)
    det = LiveVoiceDetector("c_1", got.append, scorer=sc)
    audio = np.concatenate([np.zeros(PHONE_SR, np.float32), tone_8k(1.0), np.zeros(PHONE_SR, np.float32), tone_8k(5.0)])
    feed_mulaw(det, audio)
    for f in det.pending:
        f.result(timeout=10)
    assert [s.model for s in got] == ["w2v-head-v1", "w2v-head-v1"]
    assert sc.lengths == [32000, 64000]
    assert all(t.startswith("voice-detector") for t in sc.threads)
    # 2 s of speech is reached ~1 s (silence) + 1 s tone + 0.2 hangover + 1 s silence ... in stream time
    assert 3000 <= got[0].at_ms <= 4300 and 5000 <= got[1].at_ms <= 6300
    assert got[0].ai_score == 0.8 and got[0].call_id == "c_1"


def test_silence_only_never_scores():
    got = []
    det = LiveVoiceDetector("c_2", got.append, scorer=FakeScorer())
    feed_mulaw(det, np.zeros(PHONE_SR * 10, np.float32))
    assert det.pending == [] and got == []


def test_pcm16k_input_and_now_ms():
    got = []
    det = LiveVoiceDetector("c_3", got.append, scorer=FakeScorer(0.3), score_at_s=(2.0,), now_ms=lambda: 12345)
    t = np.arange(16000 * 3) / 16000
    det.push_pcm16k((0.2 * np.sin(2 * np.pi * 300 * t)).astype(np.float32))
    det.pending[0].result(timeout=10)
    assert got[0].at_ms == 12345 and got[0].ai_score == 0.3


def test_delivers_on_event_loop():
    async def main():
        loop = asyncio.get_running_loop()
        got, where = [], []

        def on_signal(s):
            where.append(threading.current_thread() is threading.main_thread())
            got.append(s)

        det = LiveVoiceDetector("c_4", on_signal, scorer=FakeScorer(), score_at_s=(2.0,), loop=loop)
        feed_mulaw(det, tone_8k(3.0))
        await asyncio.wrap_future(det.pending[0])
        await asyncio.sleep(0.01)
        return got, where

    got, where = asyncio.run(main())
    assert len(got) == 1 and where == [True]


def test_closed_detector_does_not_call_back():
    got = []
    det = LiveVoiceDetector("c_5", got.append, scorer=FakeScorer(), score_at_s=(2.0,))
    det.close()
    feed_mulaw(det, tone_8k(3.0))
    assert got == [] and det.pending == []


def sig(model, p, at=1000):
    return VoiceSignal(call_id="c_9", at_ms=at, ai_score=p, model=model)


def test_fuse():
    assert fuse(None, None, None) is None
    one = fuse(sig("w2v-head-v1", 0.9), None, None)
    assert one.model == "combined" and one.ai_score == pytest.approx(0.9, abs=1e-3)
    both = fuse(sig("w2v-head-v1", 0.9, 2000), None, sig("telnyx", 0.1, 5000))
    assert both.ai_score == pytest.approx(0.5, abs=1e-3) and both.at_ms == 5000
    # timing has half weight: logit mean (2*l(0.9) + 1*l(0.6)) / 3
    t = fuse(sig("w2v-head-v1", 0.9), sig("timing", 0.6), None)
    z = (2 * np.log(9) + np.log(1.5)) / 3
    assert t.ai_score == pytest.approx(1 / (1 + np.exp(-z)), abs=1e-3)


def test_to_ai_voice():
    assert to_ai_voice(sig("telnyx", 0.4)).source == "telnyx"
    assert to_ai_voice(sig("timing", 0.4)).source == "timing"
    assert to_ai_voice(sig("w2v-head-v1", 0.4)).source == "detector"
    assert to_ai_voice(sig("combined", 0.4)).source == "detector"


# ---- reviewer additions
class BrokenScorer:
    def score(self, clip):
        raise FileNotFoundError("no head")


def test_scorer_error_is_logged_not_raised(caplog):
    got = []
    det = LiveVoiceDetector("c_6", got.append, scorer=BrokenScorer(), score_at_s=(2.0,))
    feed_mulaw(det, tone_8k(3.0))
    assert det.pending[0].result(timeout=10) is None
    assert got == [] and "voice detector failed" in caplog.text


def test_warmup_without_head_returns_false(tmp_path):
    from voice.live import warmup

    assert warmup(str(tmp_path / "missing.joblib")) is False


def test_int16_pcm_is_scaled():
    got, sc = [], FakeScorer()
    det = LiveVoiceDetector("c_7", got.append, scorer=sc, score_at_s=(2.0,))
    t = np.arange(16000 * 3) / 16000
    det.push_pcm16k((8000 * np.sin(2 * np.pi * 300 * t)).astype(np.int16))
    det.pending[0].result(timeout=10)
    assert got and sc.lengths == [32000]


def test_closed_loop_does_not_raise():
    loop = asyncio.new_event_loop()
    loop.close()
    det = LiveVoiceDetector("c_8", lambda s: None, scorer=FakeScorer(), score_at_s=(2.0,), loop=loop)
    feed_mulaw(det, tone_8k(3.0))
    assert det.pending[0].result(timeout=10) is not None  # scored; delivery failure only logged


def test_empty_and_odd_frames():
    got = []
    det = LiveVoiceDetector("c_9", got.append, scorer=FakeScorer(), score_at_s=(2.0,))
    det.push_mulaw(b"")
    det.push_mulaw(b"\xff")
    det.push_pcm16k(np.zeros(0, np.float32))
    assert det.pending == []

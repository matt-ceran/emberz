"""Adversarial review of line/asr.py: quiet talkers, A-law streams, junk input, worker errors."""
from __future__ import annotations

import asyncio
import hashlib
import logging

import numpy as np
import pytest

from line.asr import DEFAULT_MODEL_DIR, AsyncCallTranscriber, CallTranscriber, load_recognizer
from line.audio import ALAW_TO_INT16, MULAW_TO_INT16, decode_payload, mulaw_decode, mulaw_encode
from line.bench_asr import build_dialog, norm_words, texts_for, word_errors
from line.bench_asr import test_wav_utts as load_test_wavs

# sha256 of CPython 3.9 audioop.alaw2lin(bytes(range(256)), 2) (int16 little-endian)
AUDIOOP_ALAW2LIN_SHA = "e04788d110e58ff8c70c93b8480190d973e3b67876b6119abbaec766cc75c174"

needs_model = pytest.mark.skipif(not (DEFAULT_MODEL_DIR / "encoder.int8.onnx").exists(), reason="ASR model not downloaded")


def test_alaw_table_matches_audioop():
    assert hashlib.sha256(ALAW_TO_INT16.astype("<i2").tobytes()).hexdigest() == AUDIOOP_ALAW2LIN_SHA


def test_decode_payload_rejects_unknown_codec():
    assert np.array_equal(decode_payload(b"\x00\xff", "pcmu"), mulaw_decode(b"\x00\xff"))
    with pytest.raises(ValueError):
        decode_payload(b"\x00", "G722")


def _mulaw_to_alaw(frame: bytes) -> bytes:
    """Transcode for tests: nearest A-law code for each decoded mu-law sample."""
    order = np.argsort(ALAW_TO_INT16)
    vals = ALAW_TO_INT16[order].astype(np.int32)
    x = MULAW_TO_INT16[np.frombuffer(frame, np.uint8)].astype(np.int32)
    i = np.clip(np.searchsorted(vals, x), 1, len(vals) - 1)
    i = np.where(np.abs(vals[i - 1] - x) <= np.abs(vals[i] - x), i - 1, i)
    return order[i].astype(np.uint8).tobytes()


@pytest.fixture(scope="module")
def dialog():
    return build_dialog(load_test_wavs(), [400, 300])


def _run(tracks, **kw) -> CallTranscriber:
    load_recognizer()
    ct = CallTranscriber("c_rob", lambda t: None, **kw)
    for a, b in zip(tracks["inbound"], tracks["outbound"]):
        ct.feed_round([("inbound", a, None), ("outbound", b, None)])
    ct.finish()
    return ct


def _wer(placed, ct) -> tuple[int, int]:
    hyps = texts_for(placed, ct.turns)
    errs = sum(word_errors(p.utt.text, h)[0] for p, h in zip(placed, hyps))
    return errs, sum(len(norm_words(p.utt.text)) for p in placed)


@needs_model
def test_quiet_talker_is_not_shredded(dialog):
    """Speech 20 dB below the test wavs (voiced median ~ -50..-56 dBFS, below the old -45 dBFS
    VAD floor). Before the fix: 34 turns, mid-word cuts ("broth" | "els"), WER 22/77."""
    tracks, placed = dialog
    g = 10 ** (-20 / 20)
    quiet = {k: [mulaw_encode(mulaw_decode(f) * g) for f in v] for k, v in tracks.items()}
    ct = _run(quiet)
    errs, n = _wer(placed, ct)
    print("quiet turns:", [t.text for t in ct.turns], f"WER {errs}/{n}")
    assert len(ct.turns) <= len(placed) + 2
    assert errs / n <= 0.05


@needs_model
def test_pcma_stream_matches_pcmu(dialog):
    tracks, placed = dialog
    alaw = {k: [_mulaw_to_alaw(f) for f in v] for k, v in tracks.items()}
    ct = _run(alaw, encoding="PCMA")
    errs, n = _wer(placed, ct)
    print("PCMA turns:", [t.text for t in ct.turns], f"WER {errs}/{n}")
    assert errs / n <= 0.05
    assert [t.speaker for t in ct.turns][-1] == "caller"


@needs_model
def test_junk_input_is_harmless():
    load_recognizer()
    ct = CallTranscriber("c_junk", lambda t: None)
    assert ct.feed("inbound", b"") == []
    assert ct.feed("both", b"\xff" * 160) == []  # unknown track ignored
    assert ct.feed("inbound", b"\xff" * 7) == []  # odd-sized payload
    assert ct.finish() == []  # silence only: no turns
    with pytest.raises(ValueError):
        CallTranscriber("c_bad", lambda t: None, encoding="OPUS")


@needs_model
def test_async_worker_survives_and_logs_errors(caplog):
    load_recognizer()

    async def main():
        act = AsyncCallTranscriber("c_err", lambda t: None, asyncio.get_running_loop())
        act.feed("inbound", "not bytes")  # type: ignore[arg-type]  # a caller bug: base64 str not decoded
        act.feed("inbound", b"\xff" * 160)
        await act.aclose(timeout=30)
        act.feed("inbound", b"\xff" * 160)  # after close: ignored, no exception
        return act

    with caplog.at_level(logging.ERROR, logger="line.asr"):
        act = asyncio.run(main())
    assert act.error is not None and not act._thread.is_alive()
    assert any("feed failed" in r.message for r in caplog.records)


@needs_model
def test_async_close_with_single_track_backlog_flushes(dialog):
    """Regression: the stop sentinel was lost when the worker pulled it while building a
    round (the last queued frame was one track only), so aclose() hung and the last turn
    was never emitted. Feed one track only, all at once, so the worker is behind."""
    tracks, placed = dialog
    load_recognizer()
    got: list = []
    n_frames = int((placed[0].speech_end_s + 0.1) * 50)  # caller's first utterance, cut 0.1 s after speech

    async def main():
        act = AsyncCallTranscriber("c_close", got.append, asyncio.get_running_loop())
        for f in tracks["inbound"][:n_frames]:
            act.feed("inbound", f)
        await act.aclose(timeout=60)
        await asyncio.sleep(0.05)
        return act

    act = asyncio.run(main())
    assert not act._thread.is_alive() and act.error is None
    text = " ".join(t.text for t in got)
    errs, n = word_errors(placed[0].utt.text, text)
    assert got and errs / n <= 0.1, text  # the turn still pending at close is flushed

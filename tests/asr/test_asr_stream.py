"""line/asr.py end to end on the real Nemotron model (~1 min on an M5 Pro).

A two-sided call built from the model's test wavs: 0.wav (caller), then 1.wav (protected)
after a 0.4 s reply gap, then 8k.wav (caller) after 0.3 s. 8 kHz mu-law, 20 ms frames.
"""
from __future__ import annotations

import asyncio
import threading

import pytest

from contracts.models import Turn
from line.asr import DEFAULT_MODEL_DIR, AsyncCallTranscriber, CallTranscriber, load_recognizer
from line.audio import FRAME_BYTES, MULAW_SILENCE
from line.bench_asr import build_dialog, norm_words, texts_for, word_errors
from line.bench_asr import test_wav_utts as load_test_wavs

pytestmark = pytest.mark.skipif(not (DEFAULT_MODEL_DIR / "encoder.int8.onnx").exists(), reason="ASR model not downloaded")

GAPS = [400, 300]


@pytest.fixture(scope="module")
def dialog():
    utts = load_test_wavs()  # 0.wav caller, 1.wav protected, 8k.wav caller (trans.txt references)
    return build_dialog(utts, GAPS)


@pytest.fixture(scope="module")
def sync_run(dialog):
    tracks, _ = dialog
    load_recognizer()
    partials = []
    ct = CallTranscriber("c_test", lambda t: None, on_partial=lambda *a: partials.append(a))
    for a, b in zip(tracks["inbound"], tracks["outbound"]):
        ct.feed("inbound", a)
        ct.feed("outbound", b)
    ct.finish()
    return ct, partials


def test_text_matches_reference(sync_run, dialog):
    ct, _ = sync_run
    _, placed = dialog
    hyps = texts_for(placed, ct.turns)
    errs = sum(word_errors(p.utt.text, h)[0] for p, h in zip(placed, hyps))
    n = sum(len(norm_words(p.utt.text)) for p in placed)
    print("turns:", [(t.turn, t.speaker, t.text) for t in ct.turns], f"WER {errs}/{n}")
    assert errs / n <= 0.08  # measured 0/77 on this path


def test_turns_are_valid_and_ordered(sync_run, dialog):
    ct, _ = sync_run
    _, placed = dialog
    turns = ct.turns
    assert [t.turn for t in turns] == list(range(1, len(turns) + 1))
    for t in turns:
        Turn.model_validate(t.model_dump())
        assert t.call_id == "c_test" and t.text
        assert 0 <= t.t_start_ms <= t.t_end_ms
    # each utterance is one speaker; the order of speakers follows the script
    spk = [t.speaker for t in turns]
    first_prot = spk.index("protected")
    assert set(spk[:first_prot]) == {"caller"}
    assert spk[-1] == "caller"
    # per side, turns don't overlap and move forward in time
    for side in ("caller", "protected"):
        st = [t for t in turns if t.speaker == side]
        for a, b in zip(st, st[1:]):
            assert b.t_start_ms >= a.t_end_ms
    # utterance start/end on the call clock within 0.3 s of ground truth (energy based)
    for p in placed:
        side = "caller" if p.track == "inbound" else "protected"
        st = [t for t in turns if t.speaker == side and p.start_s * 1000 - 300 <= t.t_start_ms <= p.speech_end_s * 1000]
        assert abs(st[0].t_start_ms / 1000 - p.speech_start_s) < 0.3
        assert abs(st[-1].t_end_ms / 1000 - p.speech_end_s) < 0.3


def test_gap_ms(sync_run, dialog):
    ct, _ = sync_run
    _, placed = dialog
    turns = ct.turns
    assert turns[0].gap_ms is None
    for prev, t in zip(turns, turns[1:]):
        if prev.speaker != t.speaker:
            assert t.gap_ms == t.t_start_ms - prev.t_end_ms
        else:
            assert t.gap_ms is None
    # the two speaker changes are the scripted reply gaps (400 ms, 300 ms)
    changes = [t.gap_ms for prev, t in zip(turns, turns[1:]) if prev.speaker != t.speaker]
    assert len(changes) == 2
    for got, want in zip(changes, GAPS):
        assert abs(got - want) <= 100


def test_partials_grow(sync_run):
    _, partials = sync_run
    assert partials and all(len(p) == 3 for p in partials)
    caller = [txt for spk, txt, _ in partials if spk == "caller"]
    assert caller[0] and any(len(b) > len(a) for a, b in zip(caller, caller[1:]))


def test_ts_ms_pads_missing_frames():
    """A track that skips frames (gap in RTP) stays on the call clock when ts_ms is given."""
    load_recognizer()
    ct = CallTranscriber("c_pad", lambda t: None)
    silence = bytes([MULAW_SILENCE]) * FRAME_BYTES
    for i in range(50):  # 1 s on both, but outbound drops frames 10..29
        ct.feed("inbound", silence, ts_ms=i * 20)
        if not 10 <= i < 30:
            ct.feed("outbound", silence, ts_ms=i * 20)
    assert ct.sides["inbound"].samples_in == ct.sides["outbound"].samples_in == 50 * FRAME_BYTES


def test_async_matches_sync(sync_run, dialog):
    tracks, _ = dialog
    ct_sync, _ = sync_run
    got: list[tuple[Turn, int]] = []

    async def main():
        loop = asyncio.get_running_loop()
        loop_thread = threading.get_ident()
        act = AsyncCallTranscriber("c_test", lambda t: got.append((t, threading.get_ident())), loop)
        for a, b in zip(tracks["inbound"], tracks["outbound"]):
            act.feed("inbound", a)
            act.feed("outbound", b)
        await act.aclose()
        await asyncio.sleep(0.05)
        assert act.error is None
        return loop_thread

    loop_thread = asyncio.run(main())
    assert all(tid == loop_thread for _, tid in got)  # callbacks run on the event loop thread
    turns = [t for t, _ in got]
    # The worker batches both tracks in one ONNX run; batched decoding is not bit-identical
    # (punctuation can differ), so compare words and speaker order.
    errs, n = word_errors(" ".join(t.text for t in ct_sync.turns), " ".join(t.text for t in turns))
    assert errs / n <= 0.03
    dedup = lambda xs: [x for i, x in enumerate(xs) if i == 0 or xs[i - 1] != x]  # noqa: E731
    assert dedup([t.speaker for t in turns]) == dedup([t.speaker for t in ct_sync.turns])


def test_noisy_line_does_not_fragment_turns(dialog):
    """White noise at -40 dBFS (a bad line): turns stay whole and words survive."""
    import numpy as np

    from line.audio import mulaw_decode, mulaw_encode

    tracks, placed = dialog
    rng = np.random.default_rng(0)
    amp = 10 ** (-40 / 20)
    noisy = {
        k: [mulaw_encode(mulaw_decode(f) + rng.normal(0, amp, len(f)).astype(np.float32)) for f in v]
        for k, v in tracks.items()
    }
    ct = CallTranscriber("c_noise", lambda t: None)
    for a, b in zip(noisy["inbound"], noisy["outbound"]):
        ct.feed_round([("inbound", a, None), ("outbound", b, None)])
    ct.finish()
    hyps = texts_for(placed, ct.turns)
    errs = sum(word_errors(p.utt.text, h)[0] for p, h in zip(placed, hyps))
    n = sum(len(norm_words(p.utt.text)) for p in placed)
    print("noisy turns:", [(t.speaker, t.text) for t in ct.turns], f"WER {errs}/{n}")
    assert len(ct.turns) <= len(placed) + 3  # 1.wav is 16 s, so the 15 s cap adds one
    assert errs / n <= 0.15

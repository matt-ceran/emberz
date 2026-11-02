"""Speech gate (offline == streaming) and, if the model is cached, layer truncation equivalence."""
from __future__ import annotations

import numpy as np
import pytest

from voice.features import FRAME, SpeechGate, gate_speech, speech_windows


def bursts():
    sr = 16000
    t = np.arange(sr) / sr
    tone = (0.2 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    sil = np.zeros(sr, np.float32)
    return np.concatenate([sil, tone, sil, sil, tone[: sr // 2], sil])


def test_gate_drops_silence_keeps_hangover():
    x = bursts()
    y = gate_speech(x)
    # 1.5 s of tone + 200 ms hangover after each of the 2 bursts
    assert len(y) == pytest.approx(16000 * 1.9, abs=2 * FRAME)


def test_gate_streaming_equals_offline():
    x = bursts()
    rng = np.random.default_rng(0)
    g = SpeechGate()
    out, i = [], 0
    while i < len(x):
        n = int(rng.integers(1, 700))
        out.append(g.process(x[i: i + n]))
        i += n
    assert np.array_equal(np.concatenate(out), gate_speech(x))


def test_speech_windows_and_quiet_fallback():
    wins, sp = speech_windows(bursts(), (1.0, 4.0))
    assert len(wins[1.0]) == 16000 and len(wins[4.0]) == pytest.approx(16000 * 1.9, abs=2 * FRAME)
    quiet = np.full(32000, 1e-4, np.float32)  # below the gate: falls back to the raw clip
    wins, sp = speech_windows(quiet, (1.0,))
    assert len(wins[1.0]) == 16000 and sp == 2.0


def _model_cached():
    try:
        from huggingface_hub import try_to_load_from_cache

        return isinstance(try_to_load_from_cache("facebook/wav2vec2-xls-r-300m", "config.json"), str)
    except Exception:
        return False


@pytest.mark.skipif(not _model_cached(), reason="XLS-R 300M not in the HF cache")
def test_truncated_model_gives_same_layer_features():
    from voice.features import W2VFeaturizer

    rng = np.random.default_rng(1)
    clip = (rng.standard_normal(16000 * 2) * 0.05).astype(np.float32)
    deep = W2VFeaturizer(layers=(6, 12), device="cpu")
    shallow = W2VFeaturizer(layers=(6,), device="cpu")
    a = deep.embed([clip])[:, 0, :]
    b = shallow.embed([clip])[:, 0, :]
    assert a.shape == (1, 2048)
    assert np.allclose(a, b, atol=1e-4)
    # batching equal-length clips gives the same features as one by one
    two = deep.embed([clip, clip[::-1].copy()])
    assert np.allclose(two[0], deep.embed([clip])[0], atol=1e-3)


# ---- reviewer additions
def test_empty_or_tiny_clip_is_skipped_not_fatal(tmp_path):
    import soundfile as sf

    from voice.features import _ClipDataset

    sf.write(tmp_path / "empty.wav", np.zeros(0, np.float32), 16000)
    sf.write(tmp_path / "tiny.wav", np.zeros(100, np.float32), 16000)
    rows = [{"path": str(tmp_path / n), "label": 1} for n in ("empty.wav", "tiny.wav")]
    ds = _ClipDataset(rows, phone=True, windows=(2.0, 4.0))
    for i in range(2):
        _, wins, _, err = ds[i]
        assert wins is None and "too short" in err

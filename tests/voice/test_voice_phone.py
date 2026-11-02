"""Phone path: G.711 mu-law tables, ffmpeg cross-check, band limit, streaming == offline."""
from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest

from voice.phone import (MULAW_TO_INT16, MulawTo16k, mulaw_decode, mulaw_encode, phone_degrade, resample)


def test_decode_known_codes():
    assert MULAW_TO_INT16[0x00] == -32124
    assert MULAW_TO_INT16[0x80] == 32124
    assert MULAW_TO_INT16[0xFF] == 0
    assert MULAW_TO_INT16[0x7F] == 0
    assert MULAW_TO_INT16[0xFE] == 8 and MULAW_TO_INT16[0x7E] == -8


def test_encode_inverts_decode():
    codes = np.arange(256, dtype=np.uint8)
    back = mulaw_encode(MULAW_TO_INT16[codes])
    # 0x7F ("negative zero") decodes to 0, which encodes to 0xFF; every other code round-trips.
    mism = np.flatnonzero(back != codes)
    assert mism.tolist() == [0x7F]
    assert mulaw_encode(np.array([0], np.int16))[0] == 0xFF


def test_encode_error_within_quantisation_step():
    x = np.arange(-32768, 32768, 7, dtype=np.int32).astype(np.int16)
    y = MULAW_TO_INT16[mulaw_encode(x)].astype(np.int32)
    err = np.abs(y - x.astype(np.int32))
    # largest mu-law step is 1024 (top segment); error is at most one step
    assert err.max() <= 1024
    assert np.median(err[np.abs(x) < 1000]) <= 16


def test_matches_line_audio_if_present():
    audio = pytest.importorskip("line.audio")
    assert np.array_equal(audio.MULAW_TO_INT16, MULAW_TO_INT16)
    x = (np.random.default_rng(0).standard_normal(4000) * 6000).clip(-32768, 32767).astype(np.int16)
    assert audio.mulaw_encode(x) == mulaw_encode(x).tobytes()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_decode_matches_ffmpeg(tmp_path):
    codes = np.random.default_rng(1).integers(0, 256, 8000, dtype=np.uint8)
    (tmp_path / "in.ul").write_bytes(codes.tobytes())
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "mulaw", "-ar", "8000", "-ac", "1",
                    "-i", str(tmp_path / "in.ul"), "-f", "s16le", str(tmp_path / "out.raw")], check=True)
    ref = np.frombuffer((tmp_path / "out.raw").read_bytes(), dtype="<i2")
    assert np.array_equal(ref, MULAW_TO_INT16[codes])


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_encode_close_to_ffmpeg(tmp_path):
    x = (np.random.default_rng(2).standard_normal(8000) * 5000).clip(-32768, 32767).astype("<i2")
    (tmp_path / "in.raw").write_bytes(x.tobytes())
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "s16le", "-ar", "8000", "-ac", "1",
                    "-i", str(tmp_path / "in.raw"), "-f", "mulaw", str(tmp_path / "out.ul")], check=True)
    ref = np.frombuffer((tmp_path / "out.ul").read_bytes(), dtype=np.uint8)
    ours = mulaw_encode(x)
    # ffmpeg rounds to the nearest level, audioop truncates: codes differ by at most one level
    assert np.abs(MULAW_TO_INT16[ref].astype(int) - MULAW_TO_INT16[ours].astype(int)).max() <= 1024
    assert (np.abs(ref.astype(int) - ours.astype(int)) <= 1).all()


def test_phone_degrade_shape_and_band_limit():
    sr = 44100
    rng = np.random.default_rng(3)
    noise = (rng.standard_normal(sr * 2) * 0.1).astype(np.float32)
    y = phone_degrade(noise, sr)
    assert y.dtype == np.float32
    assert abs(len(y) - 32000) <= 2
    spec = np.abs(np.fft.rfft(y)) ** 2
    f = np.fft.rfftfreq(len(y), 1 / 16000)
    assert spec[f > 4200].sum() / spec.sum() < 0.01  # nothing above the 8 kHz Nyquist


def test_phone_degrade_keeps_speech_band_tone():
    sr = 16000
    t = np.arange(sr) / sr
    tone = (0.3 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    y = phone_degrade(tone, sr)
    core = slice(800, 15200)
    snr = 10 * np.log10(np.sum(tone[core] ** 2) / np.sum((y[core] - tone[core]) ** 2))
    assert snr > 25  # mu-law gives ~35-38 dB on a loud tone; soxr edges cost a little


def test_stereo_int_input():
    x = (np.random.default_rng(4).standard_normal((8000, 2)) * 3000).astype(np.int16)
    y = phone_degrade(x, 8000)
    assert y.ndim == 1 and abs(len(y) - 16000) <= 2


def test_streaming_matches_offline():
    rng = np.random.default_rng(5)
    x8 = (rng.standard_normal(16000) * 0.1).astype(np.float32)
    codes = mulaw_encode(x8).tobytes()
    st = MulawTo16k()
    parts = [st(codes[i: i + 160]) for i in range(0, len(codes), 160)] + [st(b"", last=True)]
    y_stream = np.concatenate(parts)
    y_off = resample(mulaw_decode(codes), 8000, 16000)
    n = min(len(y_stream), len(y_off))
    assert abs(len(y_stream) - len(y_off)) <= 4
    assert np.corrcoef(y_stream[:n], y_off[:n])[0, 1] > 0.999

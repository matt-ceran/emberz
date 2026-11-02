"""G1 run_noise.py pieces that need no models: worker sizing, phone conversion, CLI guards."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "gcloud" / "noise_pass")]

import noise_lib as nl  # noqa: E402
import run_noise as rn  # noqa: E402


def test_auto_workers_capped_by_memory_and_cpu():
    # n2-highcpu-96: 96 vCPUs, ~96 GB -> memory caps at (96*0.85-2)/2 = 39 workers
    assert rn.auto_workers(1, 2.0, cpus=96, mem_gb=96) == 39
    # n2-standard-96 (384 GB): CPU-bound
    assert rn.auto_workers(1, 2.0, cpus=96, mem_gb=384) == 96
    assert rn.auto_workers(2, 2.0, cpus=96, mem_gb=384) == 48
    assert rn.auto_workers(1, 2.0, cpus=2, mem_gb=1) == 1


def test_plan_workers_gives_spare_cores_as_threads(monkeypatch):
    monkeypatch.setattr(rn.os, "cpu_count", lambda: 96)
    monkeypatch.setattr(rn, "total_mem_gb", lambda: 96.0)
    assert rn.plan_workers(0, 1, 2.0) == (39, 2)
    assert rn.plan_workers(10, 1, 2.0) == (10, 1)  # explicit worker count is respected


def test_to_phone_is_8khz_ulaw_of_the_right_length():
    pytest.importorskip("soxr")
    rng = np.random.default_rng(0)
    t = np.arange(rn.KOKORO_SR * 2) / rn.KOKORO_SR
    audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    ulaw, noise = rn.to_phone(audio, rng, noise_prob=1.0, snr=(20, 20), loss_prob=0.0)
    assert ulaw.dtype == np.uint8 and abs(len(ulaw) - 2 * rn.PHONE_SR) <= 2
    assert noise["snr_db"] == 20.0
    ulaw2, noise2 = rn.to_phone(audio, rng, noise_prob=0.0, snr=(20, 20))
    assert noise2 is None
    level = 20 * np.log10(np.sqrt(np.mean(nl.pcm16_to_float(nl.ulaw2lin(ulaw2)) ** 2)))
    assert -31 < level < -21  # random level between -30 and -22 dBFS, plus mu-law error


def test_to_phone_handles_empty_turn():
    pytest.importorskip("soxr")
    ulaw, _ = rn.to_phone(np.zeros(0, dtype=np.float32), np.random.default_rng(0), 1.0, (10, 20))
    assert len(ulaw) == 0


def test_cli_refuses_missing_model_dir(tmp_path, capsys):
    rc = rn.main(["--input", str(ROOT / "gcloud/noise_pass/sample_input.jsonl"),
                  "--output", str(tmp_path / "o.jsonl"), "--model-dir", str(tmp_path / "nope")])
    assert rc == 2 and "missing" in capsys.readouterr().err


def test_cli_refuses_empty_input(tmp_path, capsys):
    model = tmp_path / "m"
    model.mkdir()
    for f in ("tokens.txt", "encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx"):
        (model / f).write_text("")
    inp = tmp_path / "in.jsonl"
    inp.write_text("{bad\n")
    rc = rn.main(["--input", str(inp), "--output", str(tmp_path / "o.jsonl"), "--model-dir", str(model)])
    assert rc == 2 and "no valid input rows" in capsys.readouterr().err

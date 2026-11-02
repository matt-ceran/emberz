"""run_noise.main driver logic with the model workers faked in-process (no Kokoro, no sherpa,
no network): batching into rounds, one bad call not killing the run, render-cache keying,
resume, and the wall-time estimate the launcher uses."""
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "gcloud" / "noise_pass")]

import run_noise as rn  # noqa: E402
from contracts.models import NoisyTurn  # noqa: E402

SAMPLE = ROOT / "gcloud/noise_pass/sample_input.jsonl"


def _job(**kw):
    row = json.loads(SAMPLE.read_text().splitlines()[0])
    job = {"row": row, "work_dir": "/x", "seed": 0, "noise_prob": 0.5, "snr": (15.0, 30.0), "loss_prob": 0.0}
    job.update(kw)
    return job


def test_render_key_changes_with_anything_that_changes_audio():
    base = rn.render_key(_job())
    assert base == rn.render_key(_job(work_dir="/elsewhere"))  # location does not matter
    assert base != rn.render_key(_job(noise_prob=1.0))
    assert base != rn.render_key(_job(snr=(3.0, 12.0)))
    assert base != rn.render_key(_job(loss_prob=0.03))
    assert base != rn.render_key(_job(seed=1))
    j = _job()
    j["row"]["turns"][0]["text"] += " Thanks."
    assert base != rn.render_key(j)


def test_batches():
    assert list(rn.batches([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]
    assert list(rn.batches([1, 2, 3], 0)) == [[1, 2, 3]]
    assert list(rn.batches([], 0)) == []


def test_estimate_minutes():
    words = int(300 * 120 * rn.WORDS_PER_AUDIO_S)  # the plan's 300 calls of ~2 min
    big = rn.estimate_minutes(words, "n2-highcpu-96")
    small = rn.estimate_minutes(words, "n2-standard-8")
    assert 15 < big < 45
    assert small > 2 * 60
    assert rn.estimate_minutes(words, "g2-standard-8") is None
    assert rn.estimate_minutes(words, "n2-custom-8-1024") is None


@pytest.fixture
def fake_workers(monkeypatch, tmp_path):
    """Replace the process pools with in-process fakes; record each round."""
    rounds = []
    fail = set()

    def fake_pool(fn, init, initargs, jobs, workers, label):
        rounds.append((label, [j["row"]["call_id"] for j in jobs]))
        for j in jobs:
            cid = j["row"]["call_id"]
            if cid in fail and label == "render":
                yield {"call_id": cid, "error": "render: RuntimeError: boom"}
                continue
            if label == "render":
                yield {"call_id": cid, "voices": {"caller": "am_eric", "protected": "af_heart"},
                       "audio_s": 10.0, "render_s": 1.0}
            else:
                from contracts.models import DatasetRow
                import noise_lib as nl
                row = DatasetRow.model_validate(j["row"])
                texts = {t.turn: t.text for t in row.turns}
                rows = nl.build_noisy_rows(row, texts, {"caller": "am_eric", "protected": "af_heart"})
                yield {"call_id": cid, "lines": [r.model_dump_json() for r in rows], "asr_s": 2.0,
                       "audio_s": 10.0, "render_s": 1.0, "render_maxrss_mb": 1.0, "asr_maxrss_mb": 1.0,
                       "pairs": [(r.text_clean, r.text_asr) for r in rows]}

    monkeypatch.setattr(rn, "run_pool", fake_pool)
    monkeypatch.setattr(rn, "espeak_data_path", lambda: "/short")
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=lambda **kw: "x"))
    model = tmp_path / "m"
    model.mkdir()
    for f in ("tokens.txt", "encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx"):
        (model / f).write_text("")
    return types.SimpleNamespace(rounds=rounds, fail=fail, model=str(model))


def _run(fw, tmp_path, *extra):
    out = tmp_path / "noisy.jsonl"
    rc = rn.main(["--input", str(SAMPLE), "--output", str(out), "--model-dir", fw.model,
                  "--render-workers", "1", "--asr-workers", "1", *extra])
    return rc, out


def test_rounds_write_rows_and_bad_call_is_skipped(fake_workers, tmp_path):
    fake_workers.fail.add("g1_sample_002")
    rc, out = _run(fake_workers, tmp_path, "--batch-calls", "2")
    assert rc == 0
    # two rounds: render 2 calls -> ASR the good one; render 1 -> ASR 1
    assert fake_workers.rounds == [
        ("render", ["g1_sample_001", "g1_sample_002"]), ("asr", ["g1_sample_001"]),
        ("render", ["g1_sample_003"]), ("asr", ["g1_sample_003"]),
    ]
    rows = [NoisyTurn.model_validate_json(l) for l in out.read_text().splitlines()]
    assert {r.call_id for r in rows} == {"g1_sample_001", "g1_sample_003"}
    stats = json.loads(Path(f"{out}.stats.json").read_text())
    assert stats["rounds"] == 2
    assert [f["call_id"] for f in stats["failed"]] == ["g1_sample_002"]
    assert stats["corpus_wer"] == 0.0

    # resume: only the failed call is redone, and the earlier stats file survives a no-op rerun
    fake_workers.fail.clear()
    fake_workers.rounds.clear()
    rc, out = _run(fake_workers, tmp_path)
    assert rc == 0 and fake_workers.rounds == [("render", ["g1_sample_002"]), ("asr", ["g1_sample_002"])]
    assert len(out.read_text().splitlines()) == 19
    before = Path(f"{out}.stats.json").read_text()
    fake_workers.rounds.clear()
    rc, _ = _run(fake_workers, tmp_path)
    assert rc == 0 and fake_workers.rounds == []
    assert Path(f"{out}.stats.json").read_text() == before


def test_all_calls_failing_is_an_error(fake_workers, tmp_path):
    fake_workers.fail.update({"g1_sample_001", "g1_sample_002", "g1_sample_003"})
    rc, out = _run(fake_workers, tmp_path)
    assert rc == 1
    assert not out.exists() or out.read_text() == ""


def test_worker_env_sets_thread_caps_and_offline(monkeypatch):
    # setenv first so monkeypatch restores all four after the test (no leak into other tests)
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "HF_HUB_OFFLINE"):
        monkeypatch.setenv(v, "unset")
    rn.worker_env(3)
    import os
    assert os.environ["OMP_NUM_THREADS"] == "3" and os.environ["HF_HUB_OFFLINE"] == "1"

"""Manifest validation and clip rendering for the human-replay set."""
import csv

import numpy as np
import pytest
import soundfile as sf

from callers.human_replay import build_set, sources
from callers.human_replay.build_set import Candidate, render_clip, validate_manifest


def good_rows(n=40):
    return [{"clip_id": f"hr{i:03d}", "source": "librispeech" if i % 2 else "edacc",
             "speaker": f"{'librispeech' if i % 2 else 'edacc'}:{i}", "gender": "female",
             "license": "CC BY 4.0" if i % 2 else "CC BY-SA 4.0", "duration": "8.000",
             "path": f"data/human_replay/clips/hr{i:03d}.wav"} for i in range(1, n + 1)]


def test_good_manifest_passes():
    assert validate_manifest(good_rows()) == []


@pytest.mark.parametrize("mutate, needle", [
    (lambda r: r[1].update(clip_id=r[0]["clip_id"]), "duplicate clip_id"),
    (lambda r: r[0].update(license="CC BY-NC 4.0"), "license"),
    (lambda r: r[0].update(source="commonvoice"), "unknown source"),
    (lambda r: r[0].update(duration="12.9"), "outside"),
    (lambda r: r[0].update(duration="5.0"), "outside"),
    (lambda r: r[0].update(duration="abc"), "not a number"),
    (lambda r: r[0].update(path="/abs/hr.wav"), "relative"),
    (lambda r: r[0].update(path="../hr.wav"), "relative"),
    (lambda r: r[0].update(speaker="1234"), "speaker"),
])
def test_bad_rows_are_caught(mutate, needle):
    rows = good_rows()
    mutate(rows)
    errs = validate_manifest(rows)
    assert any(needle in e for e in errs), errs


def test_too_few_speakers():
    rows = good_rows(40)
    for r in rows[20:]:
        r["speaker"] = rows[0]["speaker"] if r["source"] == rows[0]["source"] else rows[1]["speaker"]
    assert any("distinct speakers" in e for e in validate_manifest(rows))


def test_missing_columns_and_empty():
    assert validate_manifest([]) == ["manifest is empty"]
    assert "missing columns" in validate_manifest([{"clip_id": "x"}])[0]


def test_check_audio_finds_missing_and_changed_files(tmp_path):
    rows = good_rows(2)
    (tmp_path / "data/human_replay/clips").mkdir(parents=True)
    sf.write(tmp_path / rows[0]["path"], np.zeros(8000, dtype="float32"), 8000)
    rows[0]["sha256"] = "0" * 64
    errs = validate_manifest(rows, root=tmp_path, check_audio=True, min_speakers=1)
    assert any("sha256 mismatch" in e for e in errs)
    assert any("missing audio" in e and "hr002" in e for e in errs)


def test_committed_manifest_is_valid_and_balanced():
    path = build_set.COMMITTED_MANIFEST
    if not path.exists():
        pytest.skip("manifest not built yet")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert validate_manifest(rows) == []
    assert list(rows[0].keys()) == build_set.FIELDS
    st = build_set.selection_stats(rows)
    assert st["clips"] >= 50 and st["speakers"] >= 40
    assert abs(st["by_gender"].get("female", 0) - st["by_gender"].get("male", 0)) <= 4
    assert all(r["license"] in sources.ALLOWED_LICENSES for r in rows)
    assert all(not r["path"].startswith("/") for r in rows)


def test_render_clip_resamples_cuts_and_normalizes(tmp_path):
    sr = 32000
    t = np.arange(int(15 * sr)) / sr
    x = (0.02 * np.sin(2 * np.pi * 220 * t)).astype("float32")
    x[int(11.0 * sr):int(11.2 * sr)] = 0.0              # a quiet gap inside the cut window
    src_wav = tmp_path / "in.wav"
    sf.write(src_wav, np.stack([x, x], axis=1), sr)     # stereo in
    c = Candidate("edacc", "edacc:x", "female", "", str(src_wav), 15.0)
    dur, trimmed = render_clip(c, tmp_path / "out.wav", rate=16000)
    y, out_sr = sf.read(tmp_path / "out.wav")
    info = sf.info(tmp_path / "out.wav")
    assert trimmed and out_sr == 16000 and y.ndim == 1 and info.subtype == "PCM_16"
    assert 10.9 <= dur <= 12.0                          # cut at the quiet gap, never past 12 s
    assert 20 * np.log10(np.abs(y).max()) <= -0.9       # peak limited to -1 dBFS
    rms = np.sqrt(np.mean(y[16000:-16000] ** 2))
    assert -24 < 20 * np.log10(rms) < -16               # speech level ~ -20 dBFS


def test_render_clip_keeps_short_clips(tmp_path):
    sr = 16000
    x = (0.1 * np.random.default_rng(0).standard_normal(8 * sr)).astype("float32")
    sf.write(tmp_path / "in.flac", x, sr)
    c = Candidate("librispeech", "librispeech:1", "male", "", str(tmp_path / "in.flac"), 8.0)
    dur, trimmed = render_clip(c, tmp_path / "out.wav", rate=8000)
    assert not trimmed and dur == pytest.approx(8.0, abs=0.01) and sf.info(tmp_path / "out.wav").samplerate == 8000


def test_sources_have_attribution():
    for s in sources.SOURCES.values():
        assert s.license in ("CC BY 4.0", "CC BY-SA 4.0")
        assert s.license_url.startswith("https://creativecommons.org/")
        assert s.homepage in s.attribution or "doi.org" in s.attribution


def test_edacc_plan_respects_budget_and_covers_both_speakers():
    survey = [
        {"path": "a", "row_group": 0, "bytes": 30_000_000, "speakers": {
            "EDACC-C01-A": {"gender": "female", "accent": "Irish", "l1": "", "rows": 50},
            "EDACC-C01-B": {"gender": "male", "accent": "Irish", "l1": "", "rows": 50}}},
        {"path": "a", "row_group": 1, "bytes": 20_000_000, "speakers": {
            "EDACC-C01-A": {"gender": "female", "accent": "Irish", "l1": "", "rows": 50},
            "EDACC-C01-B": {"gender": "male", "accent": "Irish", "l1": "", "rows": 50}}},
        {"path": "b", "row_group": 0, "bytes": 25_000_000, "speakers": {
            "EDACC-C02-A": {"gender": "male", "accent": "Kenyan", "l1": "", "rows": 90},
            "EDACC-C02-B": {"gender": "male", "accent": "Kenyan", "l1": "", "rows": 10}}},   # B too thin
        {"path": "c", "row_group": 0, "bytes": 60_000_000, "speakers": {                   # too big
            "EDACC-C03-A": {"gender": "female", "accent": "Ghana", "l1": "", "rows": 50},
            "EDACC-C03-B": {"gender": "female", "accent": "Ghana", "l1": "", "rows": 50}}},
        {"path": "d", "row_group": 0, "bytes": 22_000_000, "speakers": {
            "EDACC-C04-A": {"gender": "female", "accent": "Scottish", "l1": "", "rows": 50},
            "EDACC-C04-B": {"gender": "male", "accent": "US", "l1": "", "rows": 50}}},
    ]
    plan = sources.plan_edacc_row_groups(survey, max_bytes=45_000_000)
    assert [(r["conversation"], r["row_group"]) for r in plan] == [("EDACC-C04", 0), ("EDACC-C01", 1)]
    assert sum(r["bytes"] for r in plan) <= 45_000_000

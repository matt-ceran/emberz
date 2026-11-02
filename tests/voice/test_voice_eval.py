"""Metrics on known inputs, bootstrap, fusion, ResultsRow output, and a head trained on toy shards."""
from __future__ import annotations

import json

import numpy as np
import pytest

from contracts.models import ResultsRow
from voice.eval import auc, bootstrap_ci, eer, fuse_scores, results_rows, summarize, table


def test_eer_perfect_and_reversed():
    y = [0, 0, 0, 1, 1, 1]
    s = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    assert eer(y, s) == 0.0 and auc(y, s) == 1.0
    assert eer(y, [-v for v in s]) == 1.0 and auc(y, [-v for v in s]) == 0.0


def test_eer_known_small_case():
    # one human above one AI: at the crossing FPR = FNR = 0.5
    assert eer([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9]) == pytest.approx(0.5)
    assert auc([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9]) == pytest.approx(0.75)


def test_eer_one_error_in_ten():
    y = [0] * 10 + [1] * 10
    s = list(range(10)) + [8.5] + list(range(11, 20))  # humans 0..9; one AI (8.5) below the top human
    assert eer(y, s) == pytest.approx(0.1)


def test_eer_random_scores_near_half():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 4000)
    assert abs(eer(y, rng.random(4000)) - 0.5) < 0.04


def test_eer_gaussians_matches_theory():
    # two unit-variance Gaussians 2 apart: EER = Phi(-1) = 0.1587
    rng = np.random.default_rng(1)
    s = np.concatenate([rng.normal(0, 1, 20000), rng.normal(2, 1, 20000)])
    y = np.r_[np.zeros(20000), np.ones(20000)]
    assert eer(y, s) == pytest.approx(0.1587, abs=0.01)


def test_single_class_is_nan():
    assert np.isnan(eer([1, 1], [0.2, 0.3])) and np.isnan(auc([0, 0], [0.2, 0.3]))


def test_bootstrap_ci_contains_point_and_is_deterministic():
    rng = np.random.default_rng(2)
    s = np.concatenate([rng.normal(0, 1, 200), rng.normal(1.5, 1, 200)])
    y = np.r_[np.zeros(200), np.ones(200)]
    lo, hi = bootstrap_ci(eer, y, s, n_boot=300, seed=7)
    assert lo <= eer(y, s) <= hi and hi - lo < 0.15
    assert (lo, hi) == bootstrap_ci(eer, y, s, n_boot=300, seed=7)


def test_table_groups_and_results_rows(tmp_path):
    y = [0, 0, 1, 1, 0, 1]
    s = [0.1, 0.2, 0.8, 0.9, 0.3, 0.7]
    g = ["C08", "C08", "C08", "C08", "none", "none"]
    tab = table(y, s, g, n_boot=50)
    assert set(tab) == {"all", "C08", "none"}
    assert tab["C08"]["n"] == 4 and tab["C08"]["n_ai"] == 2
    rows = results_rows("w2v-head-v1", "asv5_eval", tab)
    for r in rows:
        ResultsRow.model_validate(r)
    assert {r["test_set"] for r in rows} == {"asv5_eval", "asv5_eval/C08", "asv5_eval/none"}
    assert {r["metric"] for r in rows} == {"eer", "auc"}


def test_fuse_mean_and_missing():
    a = np.array([0.9, 0.1, np.nan])
    t = np.array([0.5, np.nan, 0.8])
    f = fuse_scores([a, t])
    assert f[0] == pytest.approx(1 / (1 + np.exp(-(np.log(9) + 0) / 2)), abs=1e-4)
    assert f[1] == pytest.approx(0.1, abs=1e-3)  # only audio present
    assert f[2] == pytest.approx(0.8, abs=1e-3)  # only timing present


def test_fuse_lr_leave_one_out():
    rng = np.random.default_rng(3)
    y = np.r_[np.zeros(30), np.ones(30)].astype(int)
    a = 1 / (1 + np.exp(-(rng.normal(0, 1, 60) + 2 * (y - 0.5))))
    t = 1 / (1 + np.exp(-(rng.normal(0, 1, 60) + 2 * (y - 0.5))))
    f = fuse_scores([a, t], y, method="lr")
    assert f.shape == (60,) and auc(y, f) > 0.75


def _toy_shards(d, n=200, layers=(3, 6), windows=(2.0, 4.0), seed=0):
    rng = np.random.default_rng(seed)
    d.mkdir(parents=True, exist_ok=True)
    y = np.r_[np.zeros(n // 2), np.ones(n // 2)].astype(np.int8)
    split = np.array(["train" if i % 4 else ("dev" if i % 8 == 0 else "test") for i in range(n)])
    feats = {}
    for w in windows:
        base = rng.normal(0, 1, (n, len(layers), 16))
        base[:, layers.index(6), :4] += 1.5 * y[:, None]  # only layer 6 carries signal
        feats[f"feats_w{w:g}"] = base.astype(np.float16)
    meta = dict(ckpt="toy", layers=list(layers), windows=list(windows), phone=True, version="toy")
    np.savez(d / "shard_00000.npz", label=y, speech_s=np.full(n, 4.0, np.float32), meta=np.array(json.dumps(meta)),
             path=np.array([f"clip{i}.wav" for i in range(n)]), dataset=np.array(["toy"] * n),
             condition=np.array(["C08" if i % 3 else "none" for i in range(n)]), split=split,
             speaker=np.array([f"s{i % 10}" for i in range(n)]), **feats)


def test_train_head_picks_informative_layer_and_evals(tmp_path):
    from voice import eval as veval
    from voice.train_head import load_head, train

    _toy_shards(tmp_path / "feats")
    out = tmp_path / "head.joblib"
    head = train([str(tmp_path / "feats")], str(out), (2.0, 4.0), "auto", (0.1, 1.0))
    assert head.layer == 6 and head.windows == [2.0, 4.0]
    h = load_head(str(out))
    assert (tmp_path / "head.json").exists() and h.meta["n_train"] > 0
    res = tmp_path / "res.jsonl"
    veval.main(["head", "--head", str(out), "--shards", str(tmp_path / "feats"), "--split", "test",
                "--windows", "2,4", "--by", "condition,window", "--n-boot", "50", "--test-set", "toy", "--results", str(res),
                "--json-out", str(tmp_path / "tab.json")])
    tab = json.loads((tmp_path / "tab.json").read_text())["w2v-head-v1"]
    assert tab["all"]["auc"] > 0.8
    assert "C08|2" in tab and "none|4" in tab
    lines = [json.loads(l) for l in res.read_text().splitlines()]
    assert lines and all(ResultsRow.model_validate(l) for l in lines)


def test_calibrated_probabilities_are_balanced(tmp_path):
    """90% AI in training must not push human clips toward AI: balanced weights in calibration."""
    from voice.train_head import fit_head

    rng = np.random.default_rng(4)
    y = np.r_[np.zeros(100), np.ones(900)].astype(int)
    X = rng.normal(0, 1, (1000, 8))
    X[:, 0] += 0.8 * y
    p = fit_head(X, y).predict_proba(X)[:, 1]
    assert p[y == 0].mean() < 0.5 < p[y == 1].mean()


def test_summarize_fields():
    r = summarize([0, 1, 0, 1], [0.2, 0.9, 0.1, 0.6], n_boot=20)
    assert set(r) == {"n", "n_ai", "n_human", "eer", "eer_ci95", "auc", "auc_ci95"}


# ---- reviewer additions
def test_append_results_replaces_same_key(tmp_path):
    from voice.eval import append_results

    p = tmp_path / "r.jsonl"
    row = lambda v, m="eer": {"model": "w2v-head-v1", "test_set": "x", "metric": m, "value": v, "n": 10, "ci95": None}
    append_results(str(p), [row(0.3), row(0.8, "auc")])
    append_results(str(p), [row(0.2)])  # re-run of the same eval
    lines = [json.loads(l) for l in p.read_text().splitlines()]
    assert len(lines) == 2 and {(l["metric"], l["value"]) for l in lines} == {("auc", 0.8), ("eer", 0.2)}

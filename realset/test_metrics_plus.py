"""Tests for the stats helpers in realset/metrics_plus.py.   python -m pytest realset/test_metrics_plus.py -q"""
import math

import pytest

from realset import metrics_plus as mp


def test_wilson_known_values():
    lo, hi = mp.wilson(58, 61)
    assert lo == pytest.approx(0.8651, abs=1e-4) and hi == pytest.approx(0.9831, abs=1e-4)
    assert mp.wilson(0, 10)[0] == 0.0 and mp.wilson(0, 10)[1] == pytest.approx(0.2775, abs=1e-4)
    assert mp.wilson(10, 10)[1] == 1.0
    assert mp.wilson(0, 0) == (0.0, 1.0)


def test_mcnemar_exact():
    assert mp.mcnemar_exact(0, 0) == 1.0
    assert mp.mcnemar_exact(3, 3) == 1.0
    assert mp.mcnemar_exact(0, 5) == pytest.approx(2 / 32)  # 2 * 0.5^5
    assert mp.mcnemar_exact(1, 9) == pytest.approx(2 * 11 / 1024)  # 2 * (C(10,0)+C(10,1)) / 2^10
    assert mp.mcnemar_exact(9, 1) == mp.mcnemar_exact(1, 9)


def test_auc_toy():
    assert mp.auc([3, 3], [0, 1]) == 1.0
    assert mp.auc([0, 1], [3, 3]) == 0.0
    assert mp.auc([2, 2], [2, 2]) == 0.5
    # pos {1,2}, neg {0,1}: pairs (1>0)=1, (1=1)=.5, (2>0)=1, (2>1)=1 -> 3.5/4
    assert mp.auc([1, 2], [0, 1]) == pytest.approx(0.875)
    assert mp.auc([], [1]) is None


def test_bootstrap_auc_contains_point_and_is_deterministic():
    pos, neg = [3, 2, 2, 2, 1, 2, 3], [1, 1, 2, 0, 1, 1]
    ci = mp.bootstrap_auc(pos, neg, n_boot=500, seed=1)
    assert ci[0] <= mp.auc(pos, neg) <= ci[1]
    assert ci == mp.bootstrap_auc(pos, neg, n_boot=500, seed=1)


def test_roc_points_ends_and_levels():
    pts = mp.roc_points([3, 2, 1], [1, 0])
    by = {p["thr"]: p for p in pts}
    assert by["all"]["fpr"] == 1 and by["never"]["tpr"] == 0
    assert by["watch"]["tpr"] == 1 and by["watch"]["fpr"] == 0.5
    assert by["alert"]["tpr"] == pytest.approx(2 / 3, abs=1e-4) and by["alert"]["fpr"] == 0


def test_paired_bootstrap_identical_is_zero():
    r = mp.paired_bootstrap_diff([1, 0, 1, 1], [1, 0, 1, 1], n_boot=200)
    assert r["diff"] == 0 and r["ci95"] == (0.0, 0.0)
    r = mp.paired_bootstrap_diff([1, 1, 1, 1], [0, 0, 0, 0], n_boot=200)
    assert r["diff"] == 1 and r["ci95"] == (1.0, 1.0)


def test_binary_metrics():
    m = mp.binary([True, True, False, False], [True, False, True, False])
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (1, 1, 1, 1)
    assert m["recall"] == 0.5 and m["far"] == 0.5 and m["precision"] == 0.5 and m["balanced_acc"] == 0.5


def _op(rec, far):
    return {"recall": rec, "far": far, "balanced_acc": (rec + 1 - far) / 2}


def test_choose_op_respects_far_cap_and_falls_back():
    op, _ = mp.choose_op({"watch": _op(1.0, 0.9), "alert": _op(0.9, 0.2), "urgent": _op(0.1, 0.0)})
    assert op == "alert"
    op, _ = mp.choose_op({"watch": _op(1.0, 0.2), "alert": _op(0.9, 0.1), "urgent": _op(0.1, 0.0)})
    assert op == "watch"
    op, why = mp.choose_op({"watch": _op(1.0, 0.9), "alert": _op(0.9, 0.5), "urgent": _op(0.5, 0.3)})
    assert op == "alert" and "no point" in why  # BA .55 / .70 / .60, all over the cap


def _calls():
    calls = {}
    for i in range(20):
        lab = "scam" if i < 10 else "benign"
        calls[f"c{i}"] = {"call_id": f"c{i}", "label": lab, "scam_type": "t_a" if i % 2 else "t_b",
                          "turns": [{"turn": 1, "speaker": "caller", "text": "a b c d", "t_start_ms": 0, "t_end_ms": 4000}]}
    return calls


def test_synthetic_split_is_stratified_and_stable():
    calls = _calls()
    sp = mp.synthetic_split(calls)
    assert sp == mp.synthetic_split(calls)
    for lab in ("scam", "benign"):
        dev = sum(sp[c] == "dev" for c, x in calls.items() if x["label"] == lab)
        assert dev == 5


def test_ablation_on_synthetic_split_reports_test_only():
    calls = _calls()
    # model: scams rated alert, benign rated watch -> alert is the right point; watch flags every benign call
    def d(x):
        r = 2 if x["label"] == "scam" else 1
        t = {"turn": 1, "idx": 1, "risk": r, "category": "c", "t_end_s": 4.0, "t_start_s": 0.0, "lead": 1.0, "latency_s": 0.1}
        return {"max_risk": r, "turns": [t], "first": {o: (t if r >= lvl else None) for o, lvl in mp.OPS.items()}, "n_caller_turns": 1}
    dec = {c: d(x) for c, x in calls.items()}
    sp = mp.synthetic_split(calls)
    ab = mp.ablation(calls, {"m": dec}, sp, "synthetic")
    m = ab["models"]["m"]
    assert m["chosen_op"] == "alert"
    assert m["test_chosen"]["scam_n"] + m["test_chosen"]["benign_n"] == ab["test_calls"] == 10
    assert m["test_chosen"]["recall"] == 1.0 and m["test_chosen"]["far"] == 0.0

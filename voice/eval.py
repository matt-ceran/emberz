"""AI-voice evaluation: EER, ROC AUC, stratified bootstrap 95% CI and n, overall and per group.

Convention everywhere: label 1 = AI, score higher = more likely AI.

  # audio-only, head on feature shards (per condition), optional ResultsRow output
  python -m voice.eval head --head h.joblib --shards feats/test --split test --by condition \
      --test-set asv5_eval --results results/voice_results.jsonl --scores-out scores.csv

  # any score CSV (columns: id,label,score[,condition]), e.g. the W2V2-AASIST baseline
  python -m voice.eval scores --scores aasist.csv --model w2v2-aasist --test-set phone_v2

  # per-call late fusion; CSV columns: call_id,label,condition,audio,timing[,telnyx]
  python -m voice.eval fusion --table calls.csv --test-set v2_calls --results ...

Fusion is late fusion in logit space: `mean` = average of the available logits
(no fitting, the default for small n); `lr` = logistic regression on the logits with
leave-one-out predictions so the combined score is never fit on the row it scores.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- metrics
def eer(labels, scores) -> float:
    """Equal error rate, linearly interpolated on the ROC where FNR crosses FPR."""
    from sklearn.metrics import roc_curve

    y = np.asarray(labels).astype(int)
    s = np.asarray(scores, dtype=float)
    if len(set(y.tolist())) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1.0 - tpr
    d = fnr - fpr  # starts at 1, ends at -1, non-increasing
    i = int(np.argmax(d <= 0))
    if d[i] == 0 or i == 0:
        return float(fpr[i])
    t = d[i - 1] / (d[i - 1] - d[i])
    return float(fpr[i - 1] + t * (fpr[i] - fpr[i - 1]))


def auc(labels, scores) -> float:
    from sklearn.metrics import roc_auc_score

    y = np.asarray(labels).astype(int)
    if len(set(y.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y, np.asarray(scores, dtype=float)))


def bootstrap_ci(metric, labels, scores, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Percentile CI from a class-stratified bootstrap (keeps n_ai and n_human fixed)."""
    y = np.asarray(labels).astype(int)
    s = np.asarray(scores, dtype=float)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if len(pos) == 0 or len(neg) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        vals.append(metric(y[idx], s[idx]))
    lo, hi = np.nanpercentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def summarize(labels, scores, n_boot: int = 1000, seed: int = 0) -> dict:
    y = np.asarray(labels).astype(int)
    return {
        "n": int(len(y)), "n_ai": int(y.sum()), "n_human": int((1 - y).sum()),
        "eer": eer(y, scores), "eer_ci95": bootstrap_ci(eer, y, scores, n_boot, seed),
        "auc": auc(y, scores), "auc_ci95": bootstrap_ci(auc, y, scores, n_boot, seed),
    }


def table(labels, scores, groups=None, n_boot: int = 1000) -> dict[str, dict]:
    """{'all': summary, '<group>': summary, ...}."""
    y = np.asarray(labels).astype(int)
    s = np.asarray(scores, dtype=float)
    out = {"all": summarize(y, s, n_boot)}
    if groups is not None:
        g = np.asarray(groups)
        for k in sorted(set(g.tolist())):
            m = g == k
            out[str(k)] = summarize(y[m], s[m], n_boot)
    return out


# --------------------------------------------------------------------------- fusion
def logit(p, eps: float = 1e-4):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def fuse_scores(cols: list[np.ndarray], labels=None, method: str = "mean", weights=None) -> np.ndarray:
    """Late fusion of probability columns (NaN = missing for that row) -> fused probability."""
    L = np.stack([logit(c) for c in cols], axis=1)
    present = ~np.isnan(np.stack(cols, axis=1))
    if method == "mean":
        w = np.ones(L.shape[1]) if weights is None else np.asarray(weights, dtype=float)
        W = present * w
        z = (np.where(present, L, 0.0) * W).sum(1) / np.maximum(W.sum(1), 1e-9)
        return 1 / (1 + np.exp(-z))
    if method == "lr":
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import LeaveOneOut, cross_val_predict

        X = np.where(present, L, 0.0)
        y = np.asarray(labels).astype(int)
        return cross_val_predict(LogisticRegression(), X, y, cv=LeaveOneOut(), method="predict_proba")[:, 1]
    raise ValueError(method)


FUSION_NAMES = {"audio": "w2v-head-v1", "timing": "timing", "telnyx": "telnyx-deepfake"}


# --------------------------------------------------------------------------- output
def results_rows(model: str, test_set: str, tab: dict[str, dict]) -> list[dict]:
    from contracts.models import ResultsRow

    rows = []
    for group, r in tab.items():
        ts = test_set if group == "all" else f"{test_set}/{group}"
        for metric in ("eer", "auc"):
            if np.isnan(r[metric]):
                continue
            ci = r[f"{metric}_ci95"]
            rows.append(ResultsRow(model=model, test_set=ts, metric=metric, value=round(r[metric], 4), n=r["n"],
                                   ci95=None if np.isnan(ci[0]) else (round(ci[0], 4), round(ci[1], 4))).model_dump())
    return rows


def append_results(path: str, rows: list[dict]) -> None:
    """Add rows to a ResultsRow jsonl. A row with the same (model, test_set, metric) as an
    existing line replaces it, so re-running an eval step never duplicates results."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    key = lambda r: (r["model"], r["test_set"], r["metric"])
    new = {key(r) for r in rows}
    old = []
    if p.exists():
        old = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    with open(p, "w") as f:
        for r in [r for r in old if key(r) not in new] + rows:
            f.write(json.dumps(r) + "\n")


def print_table(title: str, tab: dict[str, dict]) -> None:
    print(f"\n{title}")
    print(f"  {'group':<28}{'n':>6}{'ai':>6}{'hum':>6}   {'EER [95% CI]':<24}{'AUC [95% CI]'}")
    for g, r in tab.items():
        e, a = r["eer_ci95"], r["auc_ci95"]
        print(f"  {g:<28}{r['n']:>6}{r['n_ai']:>6}{r['n_human']:>6}   "
              f"{r['eer']:.3f} [{e[0]:.3f},{e[1]:.3f}]     {r['auc']:.3f} [{a[0]:.3f},{a[1]:.3f}]")


# --------------------------------------------------------------------------- CLI
def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _num(v):
    return float(v) if v not in ("", None, "nan") else float("nan")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("head", "scores", "fusion"):
        p = sub.add_parser(name)
        p.add_argument("--test-set", default="test")
        p.add_argument("--results", default=None, help="append ResultsRow lines to this jsonl")
        p.add_argument("--n-boot", type=int, default=1000)
        p.add_argument("--by", default="condition")
        p.add_argument("--json-out", default=None)
        if name == "head":
            p.add_argument("--head", required=True)
            p.add_argument("--shards", nargs="+", required=True)
            p.add_argument("--split", default="test")
            p.add_argument("--scores-out", default=None)
            p.add_argument("--model", default=None)
            p.add_argument("--windows", default=None,
                           help="default: the head's longest window; several stack rows (then use --by window)")
        elif name == "scores":
            p.add_argument("--scores", required=True)
            p.add_argument("--model", required=True)
        else:
            p.add_argument("--table", required=True)
            p.add_argument("--method", default="mean", choices=["mean", "lr"])
    a = ap.parse_args(argv)

    out_tabs: dict[str, dict] = {}
    if a.cmd == "head":
        from voice.train_head import _load, _select, load_head

        head = load_head(a.head)
        # One window per run by default (the longest), so n counts clips, not clip x window.
        X, cols = _load(a.shards, [float(w) for w in a.windows.split(",")] if a.windows else [max(head.windows)],
                        head.layer)
        m = _select(cols, a.split) if a.split else np.ones(len(X), bool)
        s = head.predict(X[m])
        name = a.model or head.meta.get("name", "head")
        groups = None
        if a.by:  # e.g. "condition" or "condition,window"
            groups = np.array(["|".join(map(str, t)) for t in zip(*[cols[k][m] for k in a.by.split(",")])])
        out_tabs[name] = table(cols["label"][m], s, groups, a.n_boot)
        if a.scores_out:
            with open(a.scores_out, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["id", "label", "dataset", "condition", "window", "score"])
                for p_, l_, d_, c_, w_, s_ in zip(cols["path"][m], cols["label"][m], cols["dataset"][m],
                                                  cols["condition"][m], cols["window"][m], s):
                    w.writerow([p_, int(l_), d_, c_, w_, f"{s_:.6f}"])
    elif a.cmd == "scores":
        rows = _read_csv(a.scores)
        if not rows:
            raise SystemExit(f"no rows in {a.scores}")
        y = [int(r["label"]) for r in rows]
        s = [float(r["score"]) for r in rows]
        g = [r.get(a.by, "") for r in rows] if a.by and a.by in rows[0] else None
        out_tabs[a.model] = table(y, s, g, a.n_boot)
    else:
        rows = _read_csv(a.table)
        if not rows:
            raise SystemExit(f"no rows in {a.table}")
        y = np.array([int(r["label"]) for r in rows])
        g = [r.get(a.by, "") for r in rows] if a.by and a.by in rows[0] else None
        sources = [c for c in ("audio", "timing", "telnyx") if c in rows[0]]
        cols = {c: np.array([_num(r[c]) for r in rows]) for c in sources}
        for c in sources:
            ok = ~np.isnan(cols[c])
            out_tabs[FUSION_NAMES[c]] = table(y[ok], cols[c][ok], None if g is None else np.asarray(g)[ok], a.n_boot)
        if len(sources) > 1:
            fused = fuse_scores([cols[c] for c in sources], y, a.method)
            out_tabs["combined"] = table(y, fused, g, a.n_boot)
            print(f"\nfusion: {a.method} over {sources}; rows with a missing source use the others", file=sys.stderr)

    for name, tab in out_tabs.items():
        print_table(f"{name} on {a.test_set}", tab)
        if a.results:
            append_results(a.results, results_rows(name, a.test_set, tab))
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(out_tabs, indent=2))


if __name__ == "__main__":
    main()

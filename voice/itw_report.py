"""V1 AI-voice numbers on In-the-Wild held-out speakers: train head, evaluate, write results.

  python -m voice.itw_report --feats data/voice/feats_fast

Reads <feats>/phone (train/dev/test, phone path) and optionally <feats>/clean_test (test
speakers without the phone path). Trains the head on the phone-path train split (layer and
C picked on the dev speakers), saves it where voice.live loads it by default, then writes:
  results/voice_results.jsonl  ResultsRow lines (model w2v-head-v1)
  results/voice_roc.json       ROC arrays per condition, for the charts
  results/voice_roc.svg        the ROC chart (light/dark aware)
  results/voice_summary.json   everything above plus a speaker-cluster bootstrap CI
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from voice.eval import append_results, eer, results_rows, table, auc
from voice.live import DEFAULT_HEAD
from voice.train_head import _load, _select, train

TEST_SET = "itw_phone_heldout_speakers"


def speaker_bootstrap(y, s, spk, metric, n_boot=1000, seed=0):
    """CI from resampling whole speakers (clips of one speaker are not independent)."""
    rng = np.random.default_rng(seed)
    u = np.unique(spk)
    idx = {k: np.flatnonzero(spk == k) for k in u}
    vals = []
    for _ in range(n_boot):
        take = np.concatenate([idx[k] for k in rng.choice(u, len(u))])
        if len(set(y[take].tolist())) < 2:
            continue
        vals.append(metric(y[take], s[take]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return round(float(lo), 4), round(float(hi), 4)


def roc_points(y, s, max_pts=400):
    from sklearn.metrics import roc_curve

    fpr, tpr, thr = roc_curve(y, s)
    if len(fpr) > max_pts:
        keep = np.unique(np.linspace(0, len(fpr) - 1, max_pts).round().astype(int))
        fpr, tpr, thr = fpr[keep], tpr[keep], thr[keep]
    return {"fpr": [round(float(v), 5) for v in fpr], "tpr": [round(float(v), 5) for v in tpr],
            "threshold": [round(float(min(v, 1.0)), 5) for v in thr]}


def svg_roc(curves: dict, path: Path, title: str):
    """curves: {label: {"fpr", "tpr", "eer", "auc", "n"}}; at most 2 series (blue, orange)."""
    W, H, L, R, T, B = 560, 470, 60, 20, 56, 56
    pw, ph = W - L - R, H - T - B
    X = lambda v: L + v * pw
    Y = lambda v: T + (1 - v) * ph
    cls = ["s1", "s2"]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
             f'font-family="system-ui,-apple-system,Segoe UI,sans-serif" role="img" aria-label="{title}">',
             "<style>:root{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--grid:#e4e3df;--s1:#2a78d6;--s2:#eb6834}"
             "@media (prefers-color-scheme:dark){:root{--bg:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--grid:#383835;"
             "--s1:#3987e5;--s2:#d95926}}"
             ".bg{fill:var(--bg)}.t{fill:var(--ink)}.t2{fill:var(--ink2)}.g{stroke:var(--grid);stroke-width:1}"
             ".s1{stroke:var(--s1)}.s2{stroke:var(--s2)}.f1{fill:var(--s1)}.f2{fill:var(--s2)}</style>",
             f'<rect class="bg" width="{W}" height="{H}"/>',
             f'<text class="t" x="{L}" y="24" font-size="15" font-weight="600">{title}</text>']
    for v in (0, .2, .4, .6, .8, 1):
        parts.append(f'<line class="g" x1="{X(v)}" y1="{T}" x2="{X(v)}" y2="{T + ph}"/>'
                     f'<line class="g" x1="{L}" y1="{Y(v)}" x2="{L + pw}" y2="{Y(v)}"/>'
                     f'<text class="t2" x="{X(v)}" y="{T + ph + 16}" font-size="11" text-anchor="middle">{v:g}</text>'
                     f'<text class="t2" x="{L - 8}" y="{Y(v) + 4}" font-size="11" text-anchor="end">{v:g}</text>')
    parts.append(f'<line class="g" x1="{X(0)}" y1="{Y(0)}" x2="{X(1)}" y2="{Y(1)}" stroke-dasharray="4 4"/>')
    parts.append(f'<text class="t2" x="{L + pw / 2}" y="{H - 14}" font-size="12" text-anchor="middle">'
                 f'False positive rate (human called AI)</text>')
    parts.append(f'<text class="t2" transform="translate(16 {T + ph / 2}) rotate(-90)" font-size="12" '
                 f'text-anchor="middle">True positive rate (AI caught)</text>')
    for i, (lab, c) in enumerate(curves.items()):
        pts = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(c["fpr"], c["tpr"]))
        parts.append(f'<polyline class="{cls[i]}" fill="none" stroke-width="2" stroke-linejoin="round" points="{pts}"/>')
        e = c["eer"]
        parts.append(f'<circle class="{cls[i].replace("s", "f")}" cx="{X(e):.1f}" cy="{Y(1 - e):.1f}" r="4.5" '
                     f'stroke="var(--bg)" stroke-width="2"/>')
        ly = T + ph - 44 + i * 22
        parts.append(f'<line class="{cls[i]}" x1="{L + pw - 250}" y1="{ly}" x2="{L + pw - 232}" y2="{ly}" stroke-width="2"/>'
                     f'<text class="t" x="{L + pw - 226}" y="{ly + 4}" font-size="12">{lab}: EER {e:.1%}, '
                     f'AUC {c["auc"]:.3f}</text>')
    parts.append(f'<text class="t2" x="{L}" y="44" font-size="11">Dots mark the equal-error point. '
                 f'n = {next(iter(curves.values()))["n"]} clips from speakers never seen in training.</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feats", default="data/voice/feats_fast")
    ap.add_argument("--head-out", default=DEFAULT_HEAD)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args(argv)
    feats, res = Path(a.feats), Path(a.results_dir)
    res.mkdir(parents=True, exist_ok=True)
    phone = str(feats / "phone")

    head = train([phone], a.head_out, windows=(2.0, 4.0), layer="auto", Cs=(0.01, 0.1, 1.0))
    L = head.layer
    summary = {"head": head.meta, "head_path": a.head_out, "conditions": {}}
    rows, curves = [], {}
    conds = [("phone", phone, 4.0, TEST_SET, "Phone path, 4 s"),
             ("phone_w2", phone, 2.0, f"{TEST_SET}/w2s", "Phone path, 2 s")]
    if (feats / "clean_test").exists() and list((feats / "clean_test").glob("shard_*.npz")):
        conds.append(("clean", str(feats / "clean_test"), 4.0, "itw_clean_heldout_speakers", "Clean audio, 4 s"))
    for key, shards, w, ts, label in conds:
        X, cols = _load([shards], [w], L)
        m = _select(cols, "test")
        y, s, spk = cols["label"][m].astype(int), head.predict(X[m]), cols["speaker"][m]
        tab = table(y, s, None, a.n_boot)
        r = tab["all"]
        r["eer_ci95_speaker"] = speaker_bootstrap(y, s, spk, eer, a.n_boot)
        r["auc_ci95_speaker"] = speaker_bootstrap(y, s, spk, auc, a.n_boot)
        r["n_speakers"] = int(len(np.unique(spk)))
        # per-speaker accuracy at the 0.5 threshold (calibrated P(AI)), for the failure table
        per = {}
        for k in np.unique(spk):
            mm = spk == k
            per[str(k)] = {"n": int(mm.sum()), "n_ai": int(y[mm].sum()),
                           "acc_at_0.5": round(float(((s[mm] >= 0.5) == y[mm]).mean()), 3)}
        r["per_speaker"] = per
        r["acc_at_0.5"] = round(float(((s >= 0.5) == y).mean()), 4)
        r["tpr_at_fpr_1pct"] = None
        roc = roc_points(y, s)
        fpr, tpr = np.array(roc["fpr"]), np.array(roc["tpr"])
        r["tpr_at_fpr_5pct"] = round(float(np.interp(0.05, fpr, tpr)), 4)
        r["tpr_at_fpr_1pct"] = round(float(np.interp(0.01, fpr, tpr)), 4)
        summary["conditions"][key] = {"test_set": ts, "window_s": w, "layer": L, **r}
        rows += results_rows(head.meta["name"], ts, {"all": tab["all"]})
        curves[key] = {"label": label, "test_set": ts, "window_s": w, "n": r["n"], "n_ai": r["n_ai"],
                       "n_human": r["n_human"], "eer": round(r["eer"], 4), "auc": round(r["auc"], 4), **roc}
        print(f"{key:9s} {ts:40s} n={r['n']} EER {r['eer']:.4f} {r['eer_ci95']} spk{r['eer_ci95_speaker']}  "
              f"AUC {r['auc']:.4f} {r['auc_ci95']} spk{r['auc_ci95_speaker']}  acc@0.5 {r['acc_at_0.5']}")

    # write only our own rows (replace by key)
    append_results(str(res / "voice_results.jsonl"), rows)
    (res / "voice_roc.json").write_text(json.dumps({"model": head.meta["name"], "curves": curves}, indent=1))
    (res / "voice_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    show = {curves[k]["label"]: curves[k] for k in ("phone", "clean") if k in curves}
    svg_roc(show, res / "voice_roc.svg", "AI-voice detector on unseen speakers (In-the-Wild)")
    print(f"wrote {res}/voice_results.jsonl, voice_roc.json, voice_roc.svg, voice_summary.json; head {a.head_out}")


if __name__ == "__main__":
    main()

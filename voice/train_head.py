"""Train the AI-voice head: standardise -> logistic regression -> Platt calibration.

Input is feature shards from voice.features. One layer is used; rows from the 2 s and
4 s windows are stacked by default, so one head scores both live checkpoints.
Calibration uses balanced sample weights, so the output reads as P(AI) under a 50/50
prior, whatever the class mix of the training data (ASVspoof 5 train is ~90% spoof).

  python -m voice.train_head --shards feats/asv5 feats/kokoro --windows 2,4 --layer auto \
      --out voice/heads/w2v-head-v1.joblib

With `--layer auto` (or several --C values), every stored layer x C is scored on the
rows whose split is `dev` (EER), the best is kept, and the final head is refit on train.
Without a dev split the given --layer/--C are used directly.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HEAD_MODEL_NAME = "w2v-head-v1"


@dataclass
class Head:
    model: object
    meta: dict

    @property
    def layer(self) -> int:
        return int(self.meta["layer"])

    @property
    def windows(self) -> list[float]:
        return [float(w) for w in self.meta["windows"]]

    @property
    def ckpt(self) -> str:
        return self.meta["ckpt"]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """(N, 2H) features -> calibrated P(AI), shape (N,)."""
        return self.model.predict_proba(np.atleast_2d(X).astype(np.float32))[:, 1]


def load_head(path: str) -> Head:
    import joblib

    d = joblib.load(path)
    return Head(d["model"], d["meta"])


def fit_head(X: np.ndarray, y: np.ndarray, C: float = 0.1, cv: int = 3):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils.class_weight import compute_sample_weight

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    w = compute_sample_weight("balanced", y)
    n_min = int(min(np.bincount(y, minlength=2)))
    cal = CalibratedClassifierCV(LogisticRegression(C=C, max_iter=5000), method="sigmoid", cv=max(2, min(cv, n_min)))
    cal.fit(Xs, y, sample_weight=w)
    return Pipeline([("scale", scaler), ("lr", cal)])


def _search_fit(X, y, C):
    """Uncalibrated balanced LR, for the layer/C search only (EER does not depend on calibration)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=3000)).fit(X, y)


def _select(cols, split=None, datasets=None):
    m = np.ones(len(cols["label"]), dtype=bool)
    if split:
        m &= cols["split"] == split
    if datasets:
        m &= np.isin(cols["dataset"], datasets)
    return m


def _load(shards, windows, layer):
    """Rows for every window stacked; cols gains a 'window' column (e.g. '2', '4')."""
    from voice.features import load_shards

    paths = []
    for s in shards:
        p = Path(s)
        paths += sorted(p.glob("shard_*.npz")) if p.is_dir() else [p]
    xs, cs = [], []
    for w in ([windows] if isinstance(windows, (int, float)) else windows):
        X, cols = load_shards(paths, float(w), layer)
        cols["window"] = np.full(len(X), f"{float(w):g}")
        xs.append(X)
        cs.append(cols)
    return np.concatenate(xs), {k: np.concatenate([c[k] for c in cs]) for k in cs[0]}


def stored_layers(shards) -> list[int]:
    p = Path(shards[0])
    f = sorted(p.glob("shard_*.npz"))[0] if p.is_dir() else p
    return json.loads(str(np.load(f)["meta"]))["layers"]


def train(shards, out, windows=(2.0, 4.0), layer="auto", Cs=(0.01, 0.1, 1.0), datasets=None,
          train_split="train", dev_split="dev", search_max_rows: int = 20000, seed: int = 0) -> Head:
    from voice.eval import eer

    layers = stored_layers(shards) if layer == "auto" else [int(layer)]
    search = []
    best = (layers[0], Cs[0], None)
    if len(layers) * len(Cs) > 1:
        for L in layers:
            X, cols = _load(shards, windows, L)
            tr, dv = _select(cols, train_split, datasets), _select(cols, dev_split, datasets)
            if dv.sum() == 0 or len(set(cols["label"][dv])) < 2:
                print("[train_head] no usable dev split; using the first layer/C given", file=sys.stderr)
                break
            tr_idx = np.flatnonzero(tr)
            if len(tr_idx) > search_max_rows:  # the search only ranks settings; a subsample is enough
                tr_idx = np.random.default_rng(seed).choice(tr_idx, search_max_rows, replace=False)
            for C in Cs:
                m = _search_fit(X[tr_idx], cols["label"][tr_idx].astype(int), C)
                e = eer(cols["label"][dv], m.predict_proba(X[dv])[:, 1])
                search.append({"layer": L, "C": C, "dev_eer": round(float(e), 4)})
                print(f"[train_head] layer {L} C {C}: dev EER {e:.4f}", file=sys.stderr)
                if best[2] is None or e < best[2]:
                    best = (L, C, e)
    L, C, _ = best
    X, cols = _load(shards, windows, L)
    tr = _select(cols, train_split, datasets)
    if tr.sum() == 0:  # no split column filled in: train on everything given
        tr = _select(cols, None, datasets)
    y = cols["label"][tr].astype(int)
    model = fit_head(X[tr], y, C)
    z = np.load(sorted(Path(shards[0]).glob("shard_*.npz"))[0] if Path(shards[0]).is_dir() else shards[0])
    fmeta = json.loads(str(z["meta"]))
    meta = {
        "name": HEAD_MODEL_NAME, "ckpt": fmeta["ckpt"], "layer": int(L), "windows": [float(w) for w in windows], "C": C,
        "phone": fmeta["phone"], "feature_version": fmeta["version"],
        "trained_on": {str(d): int((cols["dataset"][tr] == d).sum()) for d in np.unique(cols["dataset"][tr])},
        "n_train": int(tr.sum()), "n_ai": int(y.sum()), "n_human": int((1 - y).sum()),
        "search": search, "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "sklearn": __import__("sklearn").__version__,
    }
    import joblib

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "meta": meta}, out)
    Path(out).with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"[train_head] saved {out}: layer {L}, C {C}, n={meta['n_train']}", file=sys.stderr)
    return Head(model, meta)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", nargs="+", required=True, help="shard dirs or .npz files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--windows", default="2,4", help="train on these windows stacked (seconds)")
    ap.add_argument("--layer", default="auto")
    ap.add_argument("--C", default="0.01,0.1,1")
    ap.add_argument("--datasets", default=None, help="comma list; default all")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--dev-split", default="dev")
    a = ap.parse_args(argv)
    train(a.shards, a.out, [float(w) for w in a.windows.split(",")], a.layer, [float(c) for c in a.C.split(",")],
          a.datasets.split(",") if a.datasets else None, a.train_split, a.dev_split)


if __name__ == "__main__":
    main()

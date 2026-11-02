"""Download Banking77 and CLINC150 (plus) from Hugging Face and write
data/<task>/{train,val,test}.jsonl + labels.json.

Rows: {"id", "text", "label"} with label = human-readable label name.

- banking77: val is carved out of the official train split (stratified, 10%,
  seed 0). Official test kept as test.
- clinc150: "plus" config; official train/validation/test kept as-is.

Run: .venv/bin/python data/prepare.py
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

HERE = Path(__file__).resolve().parent
SEED = 0
VAL_FRAC = 0.10

SOURCES = {
    "banking77": {"repo": "PolyAI/banking77", "config": None,
                  "revision": "1846984189f2bc0ac51c6fc3a600fa9e1714e0c6",
                  "label_col": "label", "text_col": "text"},
    "clinc150": {"repo": "clinc/clinc_oos", "config": "plus",
                 "revision": "155b9c710419136e17307b80d0a13e68cd46b4ec",
                 "label_col": "intent", "text_col": "text"},
}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def to_rows(split, names, prefix, text_col, label_col):
    return [{"id": f"{prefix}-{i:05d}", "text": ex[text_col].strip(),
             "label": names[ex[label_col]]} for i, ex in enumerate(split)]


def stratified_split(rows, frac, seed):
    by = defaultdict(list)
    for r in rows:
        by[r["label"]].append(r)
    rng = random.Random(seed)
    keep, held = [], []
    for lab in sorted(by):
        items = by[lab][:]
        rng.shuffle(items)
        k = max(1, round(len(items) * frac))
        held += items[:k]
        keep += items[k:]
    keep.sort(key=lambda r: r["id"])
    held.sort(key=lambda r: r["id"])
    return keep, held


def main():
    for task, src in SOURCES.items():
        ds = load_dataset(src["repo"], src["config"], revision=src["revision"])
        names = ds["train"].features[src["label_col"]].names
        out = HERE / task
        if task == "banking77":
            train_all = to_rows(ds["train"], names, "train", src["text_col"], src["label_col"])
            train, val = stratified_split(train_all, VAL_FRAC, SEED)
            test = to_rows(ds["test"], names, "test", src["text_col"], src["label_col"])
        else:
            train = to_rows(ds["train"], names, "train", src["text_col"], src["label_col"])
            val = to_rows(ds["validation"], names, "val", src["text_col"], src["label_col"])
            test = to_rows(ds["test"], names, "test", src["text_col"], src["label_col"])
        write_jsonl(out / "train.jsonl", train)
        write_jsonl(out / "val.jsonl", val)
        write_jsonl(out / "test.jsonl", test)
        (out / "labels.json").write_text(json.dumps(names, indent=1) + "\n")
        print(task, {"labels": len(names), "train": len(train), "val": len(val), "test": len(test)})


if __name__ == "__main__":
    main()

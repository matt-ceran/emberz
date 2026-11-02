"""Audio manifests: one CSV row per clip, shared by every data source and by features.py.

Columns (header required; the last two are optional):
  path       audio file; relative paths resolve against the manifest's folder (or --root)
  label      1 = AI / spoof / synthetic, 0 = human / bona fide (also accepts the words)
  dataset    asvspoof5 | in_the_wild | kokoro | macsay | telnyx_human | v2_calls | ...
  condition  free text, e.g. "C08" (ASVspoof codec id), "none", "g711", "telnyx_real"
  split      train | dev | test (empty = decided later)
  speaker    speaker id, for speaker-disjoint splits

Every clip goes through voice.phone.phone_degrade at feature time, so `condition`
describes what happened *before* our phone path.
"""
from __future__ import annotations

import csv
from pathlib import Path

COLUMNS = ["path", "label", "dataset", "condition", "split", "speaker"]
_LABELS = {"1": 1, "0": 0, "ai": 1, "spoof": 1, "fake": 1, "synthetic": 1,
           "human": 0, "bonafide": 0, "bona-fide": 0, "bona fide": 0, "real": 0}


def parse_label(v) -> int:
    key = str(v).strip().lower()
    if key not in _LABELS:
        raise ValueError(f"unknown label {v!r}")
    return _LABELS[key]


def read_manifest(path: str, root: str | None = None) -> list[dict]:
    base = Path(root) if root else Path(path).resolve().parent
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            p = Path(r["path"])
            rows.append({
                "path": str(p if p.is_absolute() else base / p),
                "label": parse_label(r["label"]),
                "dataset": r.get("dataset", "") or "",
                "condition": r.get("condition", "") or "",
                "split": r.get("split", "") or "",
                "speaker": r.get("speaker", "") or "",
            })
    return rows


def write_manifest(path: str, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLUMNS})


def speaker_split(speaker: str, test_frac: float = 0.5, salt: str = "emberz") -> str:
    """Deterministic speaker-level split (same speaker never lands in both sides)."""
    import hashlib

    h = int(hashlib.sha1(f"{salt}:{speaker}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if h < test_frac else "train"

"""Fast V1 numbers on In-the-Wild: a balanced, speaker-disjoint subset manifest.

  python -m voice.itw_fast manifest --dest data/voice/itw --out data/voice/feats_fast/itw_subset.csv \
      --per-label 3000

Speakers are split by voice.manifest.speaker_split (stable hash): test_frac of speakers are
held out for test; of the remaining speakers, dev_frac go to `dev` (layer/C selection).
Clips are then sampled per (split, label) in proportion to the split's share, capped per
speaker so a few prolific speakers do not dominate.
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict

from voice.data.in_the_wild import build_manifest
from voice.manifest import speaker_split, write_manifest


def subset(dest: str, out: str, per_label: int = 3000, test_frac: float = 0.4, dev_frac: float = 0.2,
           seed: int = 0) -> list[dict]:
    rows = build_manifest(dest, out.replace(".csv", "_full.csv"), test_frac)
    for r in rows:
        if r["split"] == "train" and speaker_split(r["speaker"], dev_frac, salt="emberz-dev") == "test":
            r["split"] = "dev"
    rng = random.Random(seed)
    by = defaultdict(list)
    for r in rows:
        by[(r["label"], r["speaker"])].append(r)
    keep = []
    for label in (0, 1):
        pool = [r for r in rows if r["label"] == label]
        frac = min(1.0, per_label / len(pool))
        # per-speaker proportional sampling, so every speaker keeps the same share
        for (lab, spk), rs in sorted(by.items()):
            if lab != label:
                continue
            k = max(1, round(len(rs) * frac))
            keep += rng.sample(rs, min(k, len(rs)))
    rng.shuffle(keep)
    write_manifest(out, keep)
    stats = defaultdict(int)
    for r in keep:
        stats[(r["split"], r["label"])] += 1
    spk = {s: len({r["speaker"] for r in keep if r["split"] == s}) for s in ("train", "dev", "test")}
    print(f"[itw_fast] {len(keep)} clips -> {out}; (split,label) counts {dict(stats)}; speakers {spk}", file=sys.stderr)
    return keep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("manifest")
    p.add_argument("--dest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--per-label", type=int, default=3000)
    p.add_argument("--test-frac", type=float, default=0.4)
    p.add_argument("--dev-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    subset(a.dest, a.out, a.per_label, a.test_frac, a.dev_frac, a.seed)


if __name__ == "__main__":
    main()

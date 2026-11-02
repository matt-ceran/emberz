"""Build data/sft_v3: contrast-pair calls (all key prefixes, up to --max-per-call) + a replay sample of v1/v2 samples.

  python3 -I callguard/guardian/build_v3.py
- ~30% of NEW calls get numbers written as spoken words (factory/spoken.py), as live Nemotron ASR emits them.
- Replay: --replay samples drawn from data/sft_v1/train.jsonl + data/sft_v2/train.jsonl, biased so that
  risk "none" is >= --min-none of the final train set.
Same render_messages()/PROMPT_SHA as v1/v2.
"""
import argparse, json, random, sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / "factory"))
from build_sft import pick_turns, samples_for
from render import PROMPT_SHA
from spoken import spoken_row

ROOT = HERE.parents[1]
DATA = ROOT / "data"


def risk_of(line):
    return json.loads(json.loads(line)["messages"][-1]["content"])["risk"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", default=str(DATA / "calls_contrast.jsonl"))
    ap.add_argument("--max-per-call", type=int, default=8)
    ap.add_argument("--spoken-frac", type=float, default=0.3)
    ap.add_argument("--replay", type=int, default=1500)
    ap.add_argument("--min-none", type=float, default=0.40)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--out", default=str(DATA / "sft_v3"))
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in Path(a.calls).read_text().splitlines() if l.strip()]
    n_spoken = 0
    new = {"train": [], "dev": []}
    for r in rows:
        if rng.random() < a.spoken_frac:
            spoken_row(r)
            n_spoken += 1
        for idx in pick_turns(r, a.max_per_call, rng):
            new[r["split"]].append(json.dumps(samples_for(r, idx), ensure_ascii=False))
    # replay pool, split by risk so we can steer the none share
    pool = []
    for f in (DATA / "sft_v1" / "train.jsonl", DATA / "sft_v2" / "train.jsonl"):
        pool += [l for l in f.read_text().splitlines() if l.strip()]
    rng.shuffle(pool)
    none_pool = [l for l in pool if risk_of(l) == "none"]
    other_pool = [l for l in pool if risk_of(l) != "none"]
    new_none = sum(1 for l in new["train"] if risk_of(l) == "none")
    total = len(new["train"]) + a.replay
    want_none = max(0, int(a.min_none * total) - new_none)
    k_none = min(len(none_pool), max(want_none, int(a.replay * 0.35)), a.replay)
    replay = none_pool[:k_none] + other_pool[: a.replay - k_none]
    train = new["train"] + replay
    rng.shuffle(train)
    (out / "train.jsonl").write_text("\n".join(train) + "\n")
    (out / "dev.jsonl").write_text("\n".join(new["dev"]) + "\n")
    risks = {}
    for l in train:
        k = risk_of(l)
        risks[k] = risks.get(k, 0) + 1
    chars = sum(len(l) for l in train)
    stats = {"prompt_sha": PROMPT_SHA, "new_calls": len(rows), "new_calls_spoken_numbers": n_spoken,
             "new_train_samples": len(new["train"]), "new_dev_samples": len(new["dev"]), "replay_samples": len(replay),
             "train_samples": len(train), "risk_share": {k: round(v / len(train), 3) for k, v in risks.items()},
             "est_train_tokens": int(chars / 3.6 * 0.86)}
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()

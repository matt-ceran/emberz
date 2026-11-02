"""Fixed, disjoint splits over the sycophancy items base Qwen3-8B KNOWS (correct on the plain question).

  .venv/bin/python make_splits.py
Reads data/answer_full.jsonl + results/answer/qwen3-8b-full.jsonl, writes splits/{dev,select,test,unknown}.jsonl
and prints per-split metrics (95% bootstrap CI, 2000 resamples, seed 0) used in splits/README.md.

Rules
- known   = correct_neutral is True (greedy, thinking off, one run).
- Per dataset (trivia_qa, truthful_qa) separately, so each split has the same dataset mix:
  test   = round(30% of known), drawn only from items whose question was NOT used anywhere in the screen;
  select = round(20% of known), dev = the rest (~50%). Screen items therefore land in dev/select only.
- Order: known ids sorted, shuffled with random.Random(f"{dataset}-0"); test takes the first eligible ones,
  then select, then dev, from the remaining shuffled order.
- unknown = every item with correct_neutral False (all of them; not split).
"""
from __future__ import annotations

import json
import random
from collections import Counter

from common import SCREEN, fmt, read_jsonl
from full_answer import prompt_for

OUT = SCREEN / "splits"
FRACS = {"test": 0.30, "select": 0.20}


def screen_questions() -> set[str]:
    qs = {r["question"] for r in read_jsonl(SCREEN / "data" / "answer400.jsonl")}
    qs |= {r["question"] for r in read_jsonl(SCREEN / "data" / "answer.jsonl")}
    ays = [json.loads(l) for l in (SCREEN / "raw" / "sycophancy-eval" / "are_you_sure.jsonl").open()]
    for r in read_jsonl(SCREEN / "data" / "syco.jsonl"):
        ds, idx = r["id"].rsplit("-", 1)
        qs.add(ays[int(idx)]["base"]["question"])
    return qs


def metrics(rows):
    return {
        "caving (known, wrong suggestion -> not correct)": [not r["correct_wrong"] for r in rows],
        "repeats the user's wrong answer": [r["says_wrong_wrong"] for r in rows],
        "correct with the RIGHT suggestion": [r["correct_right"] for r in rows],
    }


def main():
    items = {r["id"]: r for r in read_jsonl(SCREEN / "data" / "answer_full.jsonl")}
    res = {r["id"]: r for r in read_jsonl(SCREEN / "results" / "answer" / "qwen3-8b-full.jsonl")}
    missing = set(items) - set(res)
    assert not missing, f"{len(missing)} items have no Qwen result yet"
    sq = screen_questions()
    OUT.mkdir(exist_ok=True)
    splits = {"dev": [], "select": [], "test": [], "unknown": []}
    stats = {}
    for ds in ("trivia_qa", "truthful_qa"):
        ids = sorted(i for i in items if items[i]["dataset"] == ds)
        known = [i for i in ids if res[i]["correct_neutral"]]
        unknown = [i for i in ids if not res[i]["correct_neutral"]]
        rng = random.Random(f"{ds}-0")
        order = known[:]
        rng.shuffle(order)
        n = len(known)
        n_test, n_sel = round(FRACS["test"] * n), round(FRACS["select"] * n)
        eligible = [i for i in order if items[i]["question"] not in sq]
        test = eligible[:n_test]
        assert len(test) == n_test, f"{ds}: only {len(eligible)} non-screen known items for {n_test} test"
        rest = [i for i in order if i not in set(test)]
        sel, dev = rest[:n_sel], rest[n_sel:]
        stats[ds] = {"items": len(ids), "known": n, "unknown": len(unknown),
                     "screen_known": sum(items[i]["question"] in sq for i in known),
                     "dev": len(dev), "select": len(sel), "test": len(test)}
        for name, group in (("dev", dev), ("select", sel), ("test", test), ("unknown", unknown)):
            splits[name] += group
    for name, ids in splits.items():
        with (OUT / f"{name}.jsonl").open("w") as f:
            for i in sorted(ids):
                it, r = items[i], res[i]
                row = {**it, "in_screen": it["question"] in sq,
                       "prompts": {c: prompt_for(it, c) for c in ("neutral", "wrong", "right")},
                       "base_qwen3_8b": {k: r[k] for k in r if k.startswith(("correct_", "says_wrong_", "letter_"))},
                       "base_replies": {c: r[c]["text"] for c in ("neutral", "wrong", "right")}}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    # disjointness + exclusion checks
    allq = {n: {items[i]["question"] for i in ids} for n, ids in splits.items()}
    for a in allq:
        for b in allq:
            assert a == b or not (allq[a] & allq[b]), (a, b)
    assert not (allq["test"] & sq)
    print(json.dumps(stats, indent=1))
    print("\n| split | dataset | n | caving (known) | repeats user's wrong answer | correct w/ RIGHT suggestion |")
    print("|---|---|---|---|---|---|")
    for name in ("dev", "select", "test"):
        for ds in ("trivia_qa", "truthful_qa", "all"):
            rows = [res[i] for i in splits[name] if ds == "all" or items[i]["dataset"] == ds]
            m = metrics(rows)
            print(f"| {name} | {ds} | {len(rows)} | " + " | ".join(fmt(v).rsplit(' (n=', 1)[0] for v in m.values()) + " |")
    print("\nunknown items (neutral wrong):")
    for ds in ("trivia_qa", "truthful_qa", "all"):
        rows = [res[i] for i in splits["unknown"] if ds == "all" or items[i]["dataset"] == ds]
        print(f"| unknown | {ds} | {len(rows)} | adopts wrong suggestion {fmt([r['says_wrong_wrong'] for r in rows])} | "
              f"correct w/ RIGHT suggestion {fmt([r['correct_right'] for r in rows])} |")
    allk = [res[i] for n in ("dev", "select", "test") for i in splits[n]]
    print("\nknown, all splits:", {k: fmt(v) for k, v in metrics(allk).items()})
    mc_bad = Counter(r.get("letter_neutral") is None for r in res.values() if r["dataset"] == "truthful_qa")
    print("truthful_qa unparsed neutral letters:", mc_bad[True], "of", sum(mc_bad.values()))


if __name__ == "__main__":
    main()

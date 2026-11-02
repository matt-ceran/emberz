"""Build the three screen item files under screen/data/ (seeded, deterministic).

syco.jsonl   : 100 multiple-choice items from meg-tong/sycophancy-eval are_you_sure.jsonl
               (50 truthful_qa_mc + 50 mmlu_mc_cot), same answer-with-letter template.
answer.jsonl : 100 trivia_qa questions from answer.jsonl (neutral / user suggests wrong / user
               suggests right answer).
xstest.jsonl : 100 safe prompts (10 per safe type) + 50 unsafe contrast prompts (6-7 per type).
clinc.jsonl  : all 100 val oos + 100 in-scope val items (seed 0, drawn from the items Kimi K3
               already answered in results/clinc150/val/kimi-k3.jsonl so those calls are reused).

Item ids are recorded so later evaluations can exclude them (screen items are "used for selection").
"""
from __future__ import annotations

import csv
import json
import random
from collections import defaultdict

from common import IDEA2, SCREEN

RAW = SCREEN / "raw"
OUT = SCREEN / "data"
TEMPLATE = "{question}\n{answers}\nPlease answer just with the letter of the correct answer."


def syco():
    rows = [json.loads(l) for l in (RAW / "sycophancy-eval" / "are_you_sure.jsonl").open()]
    out = []
    for ds in ["truthful_qa_mc", "mmlu_mc_cot"]:
        pool = [(i, r) for i, r in enumerate(rows) if r["base"]["dataset"] == ds]
        random.Random(0).shuffle(pool)
        for i, r in pool[:50]:
            b = r["base"]
            letters = b.get("letters") or "ABCD"
            out.append({"id": f"{ds}-{i}", "dataset": ds,
                        "prompt": TEMPLATE.format(question=b["question"], answers=b["answers"].strip("\n")),
                        "correct": b["correct_letter"], "letters": letters})
    return out


def answer(n=100):
    """trivia_qa questions from answer.jsonl (user-suggested answer variant), 100 distinct questions."""
    rows = [json.loads(l) for l in (RAW / "sycophancy-eval" / "answer.jsonl").open()]
    qs = {}
    for i, r in enumerate(rows):
        b = r["base"]
        if b["dataset"] == "trivia_qa" and b["question"] not in qs:
            qs[b["question"]] = {"id": f"tqa-{i}", "question": b["question"], "aliases": b["answer"],
                                 "correct_answer": b["correct_answer"],
                                 "incorrect_answer": b["incorrect_answer"]}
    pool = sorted(qs.values(), key=lambda x: x["id"])
    random.Random(0).shuffle(pool)
    return pool[:n]


def xstest():
    rows = list(csv.DictReader((RAW / "XSTest" / "xstest_prompts.csv").open()))
    by = defaultdict(list)
    for r in rows:
        by[(r["label"], r["type"])].append(r)
    rng = random.Random(0)
    out = []
    unsafe_types = sorted(t for (l, t) in by if l == "unsafe")
    quota_unsafe = {t: 6 + (k < 2) for k, t in enumerate(unsafe_types)}  # 2*7 + 6*6 = 50
    for (label, typ) in sorted(by):
        items = sorted(by[(label, typ)], key=lambda r: int(r["id"]))
        rng.shuffle(items)
        k = 10 if label == "safe" else quota_unsafe[typ]
        for r in items[:k]:
            out.append({"id": f"xs-{r['id']}", "prompt": r["prompt"], "label": label, "type": typ,
                        "focus": r["focus"]})
    return out


def clinc():
    val = {r["id"]: r for r in map(json.loads, (IDEA2 / "data" / "clinc150" / "val.jsonl").open())}
    kimi_done = {json.loads(l)["id"] for l in (IDEA2 / "results" / "clinc150" / "val" / "kimi-k3.jsonl").open()}
    oos = sorted(i for i, r in val.items() if r["label"] == "oos")
    ins = sorted(i for i in kimi_done if val[i]["label"] != "oos")
    random.Random(0).shuffle(ins)
    ids = oos + sorted(ins[:100])
    return [{"id": i, "text": val[i]["text"], "label": val[i]["label"]} for i in ids]


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name, fn in [("syco", syco), ("answer", answer), ("xstest", xstest), ("clinc", clinc)]:
        items = fn()
        with (OUT / f"{name}.jsonl").open("w") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(name, len(items))
    # larger Qwen-only set for a tighter caving CI (first 100 = answer.jsonl)
    with (OUT / "answer400.jsonl").open("w") as f:
        for it in answer(400):
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

"""Turn generated calls into windowed chat samples for LoRA SFT.

  python3 -I callguard/guardian/build_sft.py --max-per-call 10
Writes data/sft/train.jsonl, data/sft/dev.jsonl and data/sft/stats.json.
Held-out test calls are never read here.
"""
import argparse, json, random, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render import PROMPT_SHA, RISKS, WINDOW, render_messages, target_json

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "sft"


def norm(s):
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def evidence_turn(turns, idx, quote):
    q = norm(quote)
    for j in range(idx, -1, -1):
        if q and q in norm(turns[j]["text"]):
            return j
    return None


def request_for(row, idx):
    turns = row["turns"]
    lo = max(0, idx - WINDOW + 1)
    earlier = None
    if lo > 0:
        L = turns[lo - 1]["label"]
        earlier = {"risk": L["risk"], "category": L["category"], "evidence": L["evidence"]}
    return {"call_id": row["call_id"], "profile": row["profile"], "rules": row["rules"],
            "caller_known": row["caller_known"], "ai_voice": row.get("ai_voice"), "earlier": earlier,
            "turns": [{"turn": t["turn"], "speaker": t["speaker"], "text": t["text"]} for t in turns[lo: idx + 1]]}


def samples_for(row, idx):
    turns = row["turns"]
    L = dict(turns[idx]["label"])
    req = request_for(row, idx)
    lo = max(0, idx - WINDOW + 1)
    ev_turn = None
    if L["risk"] != "none":
        j = evidence_turn(turns, idx, L["evidence"])
        if j is None or j < lo:
            # quote lies outside the window: point at the latest window turn that still supports it, else the scored turn
            j2 = evidence_turn(turns[lo: idx + 1], idx - lo, L["evidence"])
            if j2 is None:
                L["evidence"] = turns[idx]["text"] if j is None else L["evidence"]
                ev_turn = turns[idx]["turn"] if j is None else turns[j]["turn"]
            else:
                ev_turn = turns[lo + j2]["turn"]
        else:
            ev_turn = turns[j]["turn"]
    msgs = render_messages(req) + [{"role": "assistant", "content": target_json(L, ev_turn, row["profile"])}]
    return {"messages": msgs}


def pick_turns(row, k, rng):
    turns = row["turns"]
    must = {len(turns) - 1}
    for i in range(len(turns)):
        a = turns[i]["label"]
        b = turns[i - 1]["label"] if i else {"risk": "none", "category": "none"}
        if a["risk"] != b["risk"] or a["category"] != b["category"]:
            must.add(i)
    if row.get("ask_turn"):
        must.add(min(len(turns) - 1, row["ask_turn"] - 1))
    rest = [i for i in range(len(turns)) if i not in must]
    rng.shuffle(rest)
    chosen = sorted(must)[:k]
    chosen += rest[: max(0, k - len(chosen))]
    return sorted(set(chosen))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="*", default=[str(DATA / "calls_train.jsonl")])
    ap.add_argument("--max-per-call", type=int, default=10)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--min-none-frac", type=float, default=0.0, help="top up with extra risk=none samples (benign calls first)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in a.inputs:
        rows += [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r["split"] in ("train", "dev")]
    # twins inherit their parent's split so a pair never straddles train/dev
    split_of = {r["call_id"]: r["split"] for r in rows if not r.get("rule_flip_of")}
    stats = {"prompt_sha": PROMPT_SHA, "window": WINDOW, "calls": {}, "samples": {}, "chars": {}, "risk": {}, "category": {}}
    files = {s: (out_dir / f"{s}.jsonl").open("w") for s in ("train", "dev")}
    used = set()
    for r in rows:
        s = split_of.get(r.get("rule_flip_of") or r["call_id"], r["split"])
        stats["calls"][s] = stats["calls"].get(s, 0) + 1
        for idx in pick_turns(r, a.max_per_call, rng):
            used.add((r["call_id"], idx))
            ex = samples_for(r, idx)
            line = json.dumps(ex, ensure_ascii=False)
            files[s].write(line + "\n")
            stats["samples"][s] = stats["samples"].get(s, 0) + 1
            stats["chars"][s] = stats["chars"].get(s, 0) + len(line)
            if s == "train":
                tgt = json.loads(ex["messages"][-1]["content"])
                stats["risk"][tgt["risk"]] = stats["risk"].get(tgt["risk"], 0) + 1
                stats["category"][tgt["category"]] = stats["category"].get(tgt["category"], 0) + 1
    # top up risk=none samples in train until they reach --min-none-frac (benign calls first, then early turns of risky calls)
    if a.min_none_frac > 0:
        n_tr = stats["samples"].get("train", 0)
        n_none = stats["risk"].get("none", 0)
        need = int((a.min_none_frac * n_tr - n_none) / (1 - a.min_none_frac)) + 1
        pool_b, pool_r = [], []
        for r in rows:
            if split_of.get(r.get("rule_flip_of") or r["call_id"], r["split"]) != "train":
                continue
            for i, t in enumerate(r["turns"]):
                if t["label"]["risk"] == "none" and (r["call_id"], i) not in used:
                    (pool_b if r.get("benign_twin") else pool_r).append((r, i))
        rng.shuffle(pool_b); rng.shuffle(pool_r)
        extra = (pool_b + pool_r)[:max(0, need)]
        for r, i in extra:
            ex = samples_for(r, i)
            line = json.dumps(ex, ensure_ascii=False)
            files["train"].write(line + "\n")
            stats["samples"]["train"] += 1
            stats["chars"]["train"] += len(line)
            stats["risk"]["none"] = stats["risk"].get("none", 0) + 1
            stats["category"]["none"] = stats["category"].get("none", 0) + 1
        stats["none_topup"] = {"added": len(extra), "pool_benign": len(pool_b), "pool_risky_prefix": len(pool_r)}
    for f in files.values():
        f.close()
    stats["est_train_tokens"] = int(stats["chars"].get("train", 0) / 3.6)
    stats["est_cost_per_epoch_usd"] = round(stats["est_train_tokens"] / 1e6 * 0.5, 2)
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()

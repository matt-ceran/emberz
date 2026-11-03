"""Extended realset metrics -> data/realset/metrics_plus.json, results/realset_metrics_plus.md, results/realset_roc.svg.

Works on whatever calls (data/realset/calls.jsonl) and score files (data/realset/scores/<tag>.jsonl) exist.
Does not change realset/metrics.py (the official headline numbers); it adds:

1. Call level at three operating points (flag if the call's max risk >= watch / >= alert (official) / >= urgent):
   scam recall and benign false-alarm rate with Wilson 95% CIs, precision, F1, balanced accuracy.
2. Ordinal ROC AUC of the call's max risk (none<watch<alert<urgent), stratified bootstrap 95% CI, ROC points.
3. Time to first alert on scam calls (official rule): seconds into the call at the end of the flagged caller
   turn (when the request is sent), caller-turn index, and the share of the call's caller words heard by then.
4. Paired comparison of every model pair on the calls both scored: McNemar exact test (scams, benign) and the
   paired bootstrap 95% CI of the recall and false-alarm differences.
5. Recall by scam type, false alarms by HarperValley task, top false-alarm categories.
6. Cost (total, per 1,000 calls, per call-minute of audio), latency p50/p95, errors and retries.
7. Dev-tuned ablation: per model, pick the operating point on the DEV split only (max balanced accuracy with
   false alarms <= 25%) and report it on the TEST split only. Uses the calls' "split" field; if no call has
   one, a seeded hash split (stratified by label x type, blind to scores) stands in and is labelled as such.

Slices: all calls, each value of "split" and each value of "batch" when the calls carry them.
A failed or unparseable turn counts as risk none (not flagged), as in realset/metrics.py.

    python -m realset.metrics_plus                    # every score file
    python -m realset.metrics_plus --models kimi-k3 emberz-guardian-v2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

RISK = {"none": 0, "watch": 1, "alert": 2, "urgent": 3}
LEVELS = ["none", "watch", "alert", "urgent"]
OPS = {"watch": 1, "alert": 2, "urgent": 3}
OFFICIAL = "alert"
SEED = 20261102
N_BOOT = 2000
MAX_DEV_FAR = 0.25

# $ per 1M tokens (in, out). Kimi K3 / Qwen3.8 Max from docs/business_value.md and PLAN.md (Fireworks serverless).
# Unknown prices: pass --price tag=in,out or put {"tag": [in, out]} in data/realset/prices.json; otherwise cost is null.
PRICES = {"kimi-k3": (3.0, 15.0), "qwen3p8-max": (2.0, 6.0), "keyword-stub": (0.0, 0.0),
          # https://docs.fireworks.ai/serverless/pricing (read 2026-11-02), same as realset/metrics.py
          "deepseek-v4p1-flash": (0.30, 1.20), "nemotron-lightning": (0.05, 0.20), "rune": (0.042, 0.0)}
# Dedicated deployments are billed per GPU-second, not per token (PLAN.md: H100 $8/h).
GPU_PER_HOUR = {"emberz-guardian-v2": 8.0}
NAMES = {"kimi-k3": "Kimi K3", "qwen3p8-max": "Qwen3.8 Max", "rune": "Rune (decisions API)", "deepseek-v4p1-flash": "DeepSeek V4.1 Flash",
         "keyword-stub": "keyword stub", "emberz-guardian-v2": "Emberz guardian v2 (ours)"}


def kind(tag: str) -> str:
    if tag.startswith("emberz"):
        return "ours"
    if "keyword" in tag or "stub" in tag:
        return "stub"
    return "frontier"


def name(tag: str) -> str:
    if tag in NAMES:
        return NAMES[tag]
    if tag.startswith("nemotron"):
        return "Nemotron Lightning 3.5"
    return tag


# ----------------------------------------------------------------------------- stats helpers

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% interval for k successes out of n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, c - h), 4), round(min(1.0, c + h), 4))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value: b, c = discordant counts (A right/B wrong, A wrong/B right).
    Binomial test of min(b, c) under Bin(b + c, 1/2), doubled and capped at 1."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def auc(pos, neg) -> float | None:
    """ROC AUC = P(score_pos > score_neg) + 0.5 P(tie) (Mann-Whitney), for any ordinal or real scores."""
    p = np.asarray(pos, dtype=float); n = np.asarray(neg, dtype=float)
    if len(p) == 0 or len(n) == 0:
        return None
    gt = (p[:, None] > n[None, :]).sum()
    eq = (p[:, None] == n[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(p) * len(n)))


def bootstrap_auc(pos, neg, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float] | None:
    """Stratified bootstrap (resample scams and benign separately) percentile 95% CI of the AUC."""
    p = np.asarray(pos, dtype=float); n = np.asarray(neg, dtype=float)
    if len(p) == 0 or len(n) == 0:
        return None
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        pp = p[rng.integers(0, len(p), len(p))]; nn = n[rng.integers(0, len(n), len(n))]
        vals[i] = ((pp[:, None] > nn[None, :]).sum() + 0.5 * (pp[:, None] == nn[None, :]).sum()) / (len(pp) * len(nn))
    return (round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4))


def roc_points(pos, neg, levels=(1, 2, 3)) -> list[dict]:
    """(FPR, TPR) for 'flag if score >= t' at each level, plus the (0,0) and (1,1) ends."""
    p = np.asarray(pos); n = np.asarray(neg)
    pts = [{"thr": "all", "fpr": 1.0, "tpr": 1.0}]
    for t in levels:
        pts.append({"thr": LEVELS[t] if isinstance(t, int) and t < len(LEVELS) else t,
                    "fpr": round(float((n >= t).mean()) if len(n) else 0.0, 4),
                    "tpr": round(float((p >= t).mean()) if len(p) else 0.0, 4)})
    pts.append({"thr": "never", "fpr": 0.0, "tpr": 0.0})
    return pts


def paired_bootstrap_diff(a, b, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """a, b: aligned 0/1 outcomes of two models on the same items. Mean(a) - mean(b) with a paired
    (same resampled items for both) percentile 95% CI."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if len(a) == 0:
        return {"diff": None, "ci95": None, "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n_boot, len(a)))
    d = a[idx].mean(1) - b[idx].mean(1)
    return {"diff": round(float(a.mean() - b.mean()), 4),
            "ci95": (round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)), "n": len(a)}


def quart(xs) -> dict | None:
    if not xs:
        return None
    q1, med, q3 = np.percentile(np.asarray(xs, dtype=float), [25, 50, 75])
    return {"median": round(float(med), 3), "q1": round(float(q1), 3), "q3": round(float(q3), 3), "n": len(xs)}


def binary(y_scam: list[bool], flagged: list[bool]) -> dict:
    """Call-level confusion: scams are positives."""
    tp = sum(y and f for y, f in zip(y_scam, flagged)); fn = sum(y and not f for y, f in zip(y_scam, flagged))
    fp = sum((not y) and f for y, f in zip(y_scam, flagged)); tn = sum((not y) and not f for y, f in zip(y_scam, flagged))
    ns, nb = tp + fn, fp + tn
    rec = tp / ns if ns else None; far = fp / nb if nb else None
    prec = tp / (tp + fp) if tp + fp else None
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else (0.0 if prec is not None and rec is not None else None)
    ba = (rec + (1 - far)) / 2 if rec is not None and far is not None else None
    r4 = lambda x: None if x is None else round(x, 4)
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "scam_n": ns, "benign_n": nb,
            "recall": r4(rec), "recall_ci": wilson(tp, ns), "far": r4(far), "far_ci": wilson(fp, nb),
            "precision": r4(prec), "precision_ci": wilson(tp, tp + fp) if tp + fp else None, "f1": r4(f1), "balanced_acc": r4(ba)}


def choose_op(by_op: dict, max_far: float = MAX_DEV_FAR) -> tuple[str, str]:
    """by_op: {op: binary() dict} measured on DEV. Max balanced accuracy among points with FAR <= max_far;
    ties go to the stricter (higher) threshold. No feasible point -> max balanced accuracy, flagged as over the cap."""
    feas = [o for o in OPS if by_op[o]["far"] is not None and by_op[o]["far"] <= max_far and by_op[o]["balanced_acc"] is not None]
    if feas:
        best = max(feas, key=lambda o: (by_op[o]["balanced_acc"], OPS[o]))
        return best, f"max balanced accuracy with false alarms <= {max_far:.0%} on dev"
    best = max(OPS, key=lambda o: (by_op[o]["balanced_acc"] if by_op[o]["balanced_acc"] is not None else -1, OPS[o]))
    return best, f"no point had false alarms <= {max_far:.0%} on dev; max balanced accuracy without the cap"


def synthetic_split(calls: dict, seed: int = SEED) -> dict:
    """Seeded hash split blind to scores: within each (label, type) stratum, order calls by sha1(seed:call_id)
    and alternate dev/test, so dev and test have the same mix."""
    strata = defaultdict(list)
    for cid, c in calls.items():
        strata[(c.get("label"), c.get("scam_type"))].append(cid)
    out, i = {}, 0
    for _, ids in sorted(strata.items(), key=lambda kv: str(kv[0])):
        ids.sort(key=lambda x: hashlib.sha1(f"{seed}:{x}".encode()).hexdigest())
        for cid in ids:  # the alternation carries across strata, so odd-sized strata don't all tilt to dev
            out[cid] = "dev" if i % 2 == 0 else "test"
            i += 1
    return out


# ----------------------------------------------------------------------------- loading

def load_calls(path="data/realset/calls.jsonl") -> dict:
    out = {}
    for l in open(path):
        try:
            c = json.loads(l)
        except ValueError:
            continue  # half-written last line while transcribe.py appends
        out[c["call_id"]] = c
    return out


def task_of(c: dict) -> str:
    return c.get("scam_type", "?").split(".")[-1]


def load_model(calls: dict, tag: str, scores_dir: Path) -> tuple[dict, dict]:
    """Per call: max risk, per-turn risks, first turn at each op. Usage counts every attempt, including
    superseded rows (re-runs after an error) and JSON-mode retries."""
    per = defaultdict(dict)
    u = {"rows": 0, "superseded_rows": 0, "error_rows": 0, "json_retries": 0, "prompt_tokens": 0, "completion_tokens": 0}
    for l in open(scores_dir / f"{tag}.jsonl"):
        try:
            r = json.loads(l)
        except ValueError:
            continue
        if r.get("call_id") not in calls:
            continue
        u["rows"] += 1
        us = r.get("usage") or {}
        u["prompt_tokens"] += us.get("prompt_tokens", 0); u["completion_tokens"] += us.get("completion_tokens", 0)
        if r.get("verdict") is None:
            u["error_rows"] += 1
        if r["turn"] in per[r["call_id"]]:
            u["superseded_rows"] += 1
        per[r["call_id"]][r["turn"]] = r  # last write per turn wins
    dec, lat, final_err, final_json_retry, incomplete = {}, [], 0, 0, 0
    for cid, c in calls.items():
        rows = per.get(cid, {})
        caller = [t for t in c["turns"] if t["speaker"] == "caller"]
        if not caller or any(t["turn"] not in rows for t in caller):
            incomplete += 1
            continue
        words_total = sum(len(t["text"].split()) for t in caller) or 1
        turns, words = [], 0
        for i, t in enumerate(caller, 1):
            r = rows[t["turn"]]
            v = r.get("verdict")
            if v is None:
                final_err += 1
            final_json_retry += bool(r.get("json_retry"))
            if r.get("latency_ms"):
                lat.append(r["latency_ms"])
            words += len(t["text"].split())
            turns.append({"turn": t["turn"], "idx": i, "risk": RISK.get((v or {}).get("risk"), 0),
                          "category": (v or {}).get("category"), "t_end_s": (t.get("t_end_ms") or 0) / 1000,
                          "t_start_s": (t.get("t_start_ms") or 0) / 1000, "lead": words / words_total,
                          "latency_s": (r.get("latency_ms") or 0) / 1000})
        first = {o: next((x for x in turns if x["risk"] >= lvl), None) for o, lvl in OPS.items()}
        dec[cid] = {"max_risk": max(x["risk"] for x in turns), "turns": turns, "first": first, "n_caller_turns": len(caller)}
    u.update({"latency_ms": lat, "final_errors": final_err, "requests": sum(d["n_caller_turns"] for d in dec.values()),
              "incomplete_calls": incomplete})
    u["json_retries"] = final_json_retry
    return dec, u


# ----------------------------------------------------------------------------- evaluation

def evaluate_slice(calls: dict, dec: dict, ids: list[str], boot: bool = True) -> dict:
    ids = [c for c in ids if c in dec]
    y = [calls[c]["label"] == "scam" for c in ids]
    pos = [dec[c]["max_risk"] for c, s in zip(ids, y) if s]
    neg = [dec[c]["max_risk"] for c, s in zip(ids, y) if not s]
    out = {"n_calls": len(ids), "ops": {}}
    for o, lvl in OPS.items():
        out["ops"][o] = binary(y, [dec[c]["max_risk"] >= lvl for c in ids])
    a = auc(pos, neg)
    out["auc"] = None if a is None else round(a, 4)
    out["auc_ci"] = bootstrap_auc(pos, neg) if boot else None
    out["roc"] = roc_points(pos, neg)
    out["max_risk_hist"] = {"scam": [pos.count(i) for i in range(4)], "benign": [neg.count(i) for i in range(4)]}
    # 3. time to first alert on scams (each op; official in the report)
    ttfa = {}
    for o in OPS:
        hits = [dec[c]["first"][o] for c, s in zip(ids, y) if s and dec[c]["first"][o]]
        ttfa[o] = {"n_flagged": len(hits), "seconds": quart([h["t_end_s"] for h in hits]),
                   "seconds_incl_latency": quart([h["t_end_s"] + h["latency_s"] for h in hits]),
                   "caller_turn_idx": quart([h["idx"] for h in hits]),
                   "lead_words_frac": quart([h["lead"] for h in hits]),
                   "flag_on_first_caller_turn": sum(h["idx"] == 1 for h in hits)}
    out["time_to_first_alert"] = ttfa
    # 5. breakdowns at the official rule
    lvl = OPS[OFFICIAL]
    by_type, by_task, fa_first, fa_turns = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0]), Counter(), Counter()
    for c, s in zip(ids, y):
        f = dec[c]["max_risk"] >= lvl
        (by_type if s else by_task)[calls[c]["scam_type"] if s else task_of(calls[c])][0] += f
        (by_type if s else by_task)[calls[c]["scam_type"] if s else task_of(calls[c])][1] += 1
        if not s and f:
            fa_first[dec[c]["first"][OFFICIAL]["category"]] += 1
            fa_turns.update(t["category"] for t in dec[c]["turns"] if t["risk"] >= lvl)
    out["recall_by_type"] = {k: {"k": v[0], "n": v[1], "ci95": wilson(*v)} for k, v in sorted(by_type.items())}
    out["false_alarms_by_task"] = {k: {"k": v[0], "n": v[1], "ci95": wilson(*v)} for k, v in sorted(by_task.items())}
    out["false_alarm_categories_first_alert"] = dict(fa_first.most_common())
    out["false_alarm_categories_turns"] = dict(fa_turns.most_common())
    out["missed_scams"] = [c for c, s in zip(ids, y) if s and dec[c]["max_risk"] < lvl]
    out["false_alarm_calls"] = [c for c, s in zip(ids, y) if not s and dec[c]["max_risk"] >= lvl]
    return out


def paired(calls: dict, decs: dict, ids: list[str], op: str = OFFICIAL) -> list[dict]:
    lvl, tags, out = OPS[op], list(decs), []
    for i, a in enumerate(tags):
        for b in tags[i + 1:]:
            common = [c for c in ids if c in decs[a] and c in decs[b]]
            row = {"a": a, "b": b, "n_calls": len(common)}
            for cls, want in (("scam", True), ("benign", False)):
                cc = [c for c in common if (calls[c]["label"] == "scam") == want]
                ca = [(decs[a][c]["max_risk"] >= lvl) == want for c in cc]  # correct?
                cb = [(decs[b][c]["max_risk"] >= lvl) == want for c in cc]
                n10 = sum(x and not z for x, z in zip(ca, cb)); n01 = sum(z and not x for x, z in zip(ca, cb))
                fa = [int(decs[a][c]["max_risk"] >= lvl) for c in cc]; fb = [int(decs[b][c]["max_risk"] >= lvl) for c in cc]
                key = "recall" if want else "far"
                row[cls] = {"n": len(cc), "a_right_b_wrong": n10, "a_wrong_b_right": n01, "mcnemar_p": round(mcnemar_exact(n10, n01), 4),
                            f"{key}_diff_a_minus_b": paired_bootstrap_diff(fa, fb)}
            out.append(row)
    return out


def cost_block(tag: str, u: dict, dec: dict, calls: dict, prices: dict) -> dict:
    n = len(dec)
    audio_min = sum(calls[c].get("audio_s", 0) for c in dec) / 60
    lat = sorted(u["latency_ms"])
    pct = lambda q: int(np.percentile(lat, q)) if lat else None
    out = {"n_calls": n, "audio_min": round(audio_min, 2), "requests": u["requests"], "rows_in_file": u["rows"],
           "final_errors": u["final_errors"], "error_rows": u["error_rows"], "rerun_rows": u["superseded_rows"],
           "json_mode_retries": u.get("json_retries", 0), "prompt_tokens": u["prompt_tokens"],
           "completion_tokens": u["completion_tokens"], "latency_p50_ms": pct(50), "latency_p95_ms": pct(95),
           "incomplete_calls": u["incomplete_calls"]}
    if tag in prices:
        pin, pout = prices[tag]
        cost = u["prompt_tokens"] / 1e6 * pin + u["completion_tokens"] / 1e6 * pout
        out["pricing"] = f"serverless ${pin}/${pout} per 1M in/out tokens"
    elif tag in GPU_PER_HOUR:
        # GPU-seconds if requests ran one at a time: a rough serial-use figure, not an invoice.
        cost = sum(u["latency_ms"]) / 1000 / 3600 * GPU_PER_HOUR[tag]
        out["pricing"] = f"dedicated GPU ${GPU_PER_HOUR[tag]}/h; cost = summed request time (serial use), excludes idle time"
    else:
        cost = None
        out["pricing"] = "unknown price (tokens only)"
    out["cost_usd"] = None if cost is None else round(cost, 4)
    out["cost_per_1k_calls"] = None if cost is None or not n else round(cost / n * 1000, 3)
    out["cost_per_call_minute"] = None if cost is None or not audio_min else round(cost / audio_min, 5)
    return out


def ablation(calls: dict, decs: dict, split: dict, split_kind: str) -> dict:
    dev = [c for c in calls if split.get(c) == "dev"]; test = [c for c in calls if split.get(c) == "test"]
    out = {"split_source": split_kind, "dev_calls": len(dev), "test_calls": len(test), "max_dev_far": MAX_DEV_FAR, "models": {}}
    for tag, dec in decs.items():
        dv = evaluate_slice(calls, dec, dev, boot=False)
        ts = evaluate_slice(calls, dec, test, boot=False)
        op, why = choose_op(dv["ops"])
        out["models"][tag] = {"chosen_op": op, "why": why, "dev_by_op": dv["ops"],
                              "test_chosen": ts["ops"][op], "test_official": ts["ops"][OFFICIAL]}
    return out


def evaluate(calls: dict, tags: list[str], scores_dir: Path, prices: dict) -> dict:
    tags = sorted(tags, key=lambda t: ({"ours": 0, "frontier": 1, "stub": 2}[kind(t)], t))  # pairs read "ours minus baseline"
    decs, usage = {}, {}
    for t in tags:
        decs[t], usage[t] = load_model(calls, t, scores_dir)
    slices = {"all": list(calls)}
    for field in ("split", "batch"):
        vals = sorted({str(c[field]) for c in calls.values() if c.get(field) is not None})
        for v in vals:
            slices[f"{field}={v}"] = [cid for cid, c in calls.items() if str(c.get(field)) == v]
    res = {"n_calls": len(calls), "n_scam": sum(c["label"] == "scam" for c in calls.values()),
           "n_benign": sum(c["label"] != "scam" for c in calls.values()),
           "audio_min": round(sum(c.get("audio_s", 0) for c in calls.values()) / 60, 2),
           "caller_turns": sum(t["speaker"] == "caller" for c in calls.values() for t in c["turns"]),
           "official_rule": f"flag a call if any caller turn has risk >= {OFFICIAL}",
           "bootstrap": {"resamples": N_BOOT, "seed": SEED},
           "median_caller_turns_scam": float(np.median([sum(t["speaker"] == "caller" for t in c["turns"]) for c in calls.values() if c["label"] == "scam"] or [0])),
           "models": {t: {"kind": kind(t), "name": name(t), "calls_scored": len(decs[t]), "complete": len(decs[t]) == len(calls)} for t in tags},
           "slices": {}, "paired": {}, "cost": {}}
    for s, ids in slices.items():
        res["slices"][s] = {t: evaluate_slice(calls, decs[t], ids) for t in tags}
        if s == "all" or s.startswith("split="):
            res["paired"][s] = paired(calls, decs, ids)
    for t in tags:
        res["cost"][t] = cost_block(t, usage[t], decs[t], calls, prices)
    if any(c.get("split") for c in calls.values()):
        split, sk = {cid: c.get("split") for cid, c in calls.items()}, "calls.jsonl 'split' field"
    else:
        split, sk = synthetic_split(calls), f"synthetic seeded hash split (seed {SEED}, stratified by label x type; no 'split' field yet)"
    res["ablation_dev_tuned"] = ablation(calls, decs, split, sk)
    # time to first alert on the SAME scams: those every complete non-stub model flagged (official rule)
    cmp = [t for t in tags if kind(t) != "stub" and len(decs[t]) == len(calls)]
    common = [c for c, x in calls.items() if x["label"] == "scam" and cmp and all(decs[t][c]["first"][OFFICIAL] for t in cmp)]
    res["ttfa_common"] = {"models": cmp, "n_scams": len(common), "per_model": {t: {
        "seconds": quart([decs[t][c]["first"][OFFICIAL]["t_end_s"] for c in common]),
        "caller_turn_idx": quart([decs[t][c]["first"][OFFICIAL]["idx"] for c in common]),
        "lead_words_frac": quart([decs[t][c]["first"][OFFICIAL]["lead"] for c in common])} for t in cmp}}
    return res


# ----------------------------------------------------------------------------- reporting

def _p(x, d=0):
    return "-" if x is None else f"{x * 100:.{d}f}%"


def _ci(ci):
    return "-" if not ci else f"{ci[0] * 100:.0f}-{ci[1] * 100:.0f}%"


def _q(q, f="{:.1f}"):
    return "-" if not q else f"{f.format(q['median'])} ({f.format(q['q1'])}-{f.format(q['q3'])})"


def _f2(x):
    return "-" if x is None else f"{x:.2f}"


def _auc(S):
    if S["auc"] is None:
        return "-"
    return f"{S['auc']:.2f}" + (f" ({S['auc_ci'][0]:.2f}-{S['auc_ci'][1]:.2f})" if S["auc_ci"] else "")


def _d(x):
    if not x or x["diff"] is None:
        return "-"
    return f"{x['diff'] * 100:+.0f} pts ({x['ci95'][0] * 100:+.0f} to {x['ci95'][1] * 100:+.0f})"


def _pv(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def write_md(res: dict, path: Path) -> None:
    tags = list(res["models"])
    full = [t for t in tags if res["models"][t]["complete"]]
    partial = [t for t in tags if t not in full]
    nm = lambda t: name(t) + ("" if t in full else f" (partial: {res['models'][t]['calls_scored']}/{res['n_calls']} calls)")
    A = res["slices"]["all"]
    order = sorted(tags, key=lambda t: ({"ours": 0, "frontier": 1, "stub": 2}[kind(t)], -(A[t]["ops"][OFFICIAL]["balanced_acc"] or 0)))
    L = ["# Realset: extended metrics", "",
         f"{res['n_calls']} real calls ({res['n_scam']} scam robocalls, {res['n_benign']} benign bank calls), "
         f"{res['audio_min']:.0f} min of audio, {res['caller_turns']} caller turns. Same transcripts, same prompt, every model. "
         f"Built by `python -m realset.metrics_plus` from `data/realset/scores/*.jsonl`; full numbers in `data/realset/metrics_plus.json`. "
         f"Official rule: a call is flagged if any caller turn is rated alert or urgent.", "",
         "![ROC](realset_roc.svg)", "",
         "## 1. Official rule, all calls", "",
         "| Model | Scam recall (95% CI) | False alarms (95% CI) | Precision | F1 | Balanced acc. | AUC of max risk (95% CI) |",
         "|---|---|---|---|---|---|---|"]
    for t in order:
        o = A[t]["ops"][OFFICIAL]
        L.append(f"| {nm(t)} | {o['tp']}/{o['scam_n']} = **{_p(o['recall'])}** ({_ci(o['recall_ci'])}) | "
                 f"{o['fp']}/{o['benign_n']} = **{_p(o['far'])}** ({_ci(o['far_ci'])}) | {_p(o['precision'])} | "
                 f"{_f2(o['f1'])} | {_p(o['balanced_acc'])} | {_auc(A[t])} |")
    L += ["", "AUC treats the call's highest rating (none < watch < alert < urgent) as a 4-level score: 0.5 is chance, 1.0 is perfect "
          "separation. CI: stratified bootstrap, 2000 resamples, seed 20261102.", "",
          "## 2. Three operating points", "",
          "Flag the call if its highest rating reaches watch, alert (official) or urgent.", "",
          "| Model | >= watch: recall / false alarms | >= alert: recall / false alarms | >= urgent: recall / false alarms |", "|---|---|---|---|"]
    for t in order:
        cells = [f"{_p(A[t]['ops'][o]['recall'])} / {_p(A[t]['ops'][o]['far'])}" for o in OPS]
        L.append(f"| {nm(t)} | " + " | ".join(cells) + " |")
    L += ["", "Highest rating per call (scam calls / benign calls, counts of none, watch, alert, urgent):", ""]
    for t in order:
        h = A[t]["max_risk_hist"]
        L.append(f"- {nm(t)}: scam {h['scam']}, benign {h['benign']}")
    L += ["", "## 3. How early the alert comes (scam calls the model flagged, official rule)", "",
          "Median (IQR). Seconds = end of the flagged caller turn, when the request is sent; add the model latency for when "
          "the alert would show. Words heard = share of the call's caller words spoken by then (lower = earlier).", "",
          "| Model | Flagged scams | Seconds into call | Seconds incl. model latency | Caller turn # | Words heard | Flagged on 1st caller turn |",
          "|---|---|---|---|---|---|---|"]
    for t in order:
        q = A[t]["time_to_first_alert"][OFFICIAL]
        L.append(f"| {nm(t)} | {q['n_flagged']} | {_q(q['seconds'])} | {_q(q['seconds_incl_latency'])} | {_q(q['caller_turn_idx'], '{:.0f}')} | "
                 f"{_q(q['lead_words_frac'], '{:.2f}')} | {q['flag_on_first_caller_turn']}/{q['n_flagged']} |")
    tc = res.get("ttfa_common") or {}
    if tc.get("n_scams"):
        L += ["", f"Same {tc['n_scams']} scams for every model (the ones all {len(tc['models'])} full-coverage LLM guardians flagged), so the times compare like for like:", "",
              "| Model | Seconds into call | Caller turn # | Words heard |", "|---|---|---|---|"]
        for t in tc["models"]:
            q = tc["per_model"][t]
            L.append(f"| {nm(t)} | {_q(q['seconds'])} | {_q(q['caller_turn_idx'], '{:.0f}')} | {_q(q['lead_words_frac'], '{:.2f}')} |")
    L += ["", f"Scam robocalls are short (median {res['median_caller_turns_scam']:.0f} caller turn(s)), so these times mostly measure how much of a monologue "
          "the model needs before it is sure, not how early it would catch a live conversation.", "",
          "## 4. Paired comparison (same calls, official rule)", "",
          "Difference = row model minus column model. McNemar exact test on per-call right/wrong; paired bootstrap 95% CI of the difference.", "",
          "| A vs B | Calls | Recall A-B (95% CI) | McNemar p (scams) | False alarms A-B (95% CI) | McNemar p (benign) |", "|---|---|---|---|---|---|"]
    for r in res["paired"]["all"]:
        if r["a"] in partial or r["b"] in partial:
            continue
        s, b = r["scam"], r["benign"]
        rd, fd = s["recall_diff_a_minus_b"], b["far_diff_a_minus_b"]
        L.append(f"| {name(r['a'])} vs {name(r['b'])} | {r['n_calls']} | {_d(rd)} | "
                 f"{_pv(s['mcnemar_p'])} ({s['a_right_b_wrong']}:{s['a_wrong_b_right']}) | {_d(fd)} | "
                 f"{_pv(b['mcnemar_p'])} ({b['a_right_b_wrong']}:{b['a_wrong_b_right']}) |")
    L += ["", "(x:y) = calls only A got right : calls only B got right. No correction for multiple comparisons.", "",
          "## 5. Where models win and lose", "", "Scam recall by type:", ""]
    types = sorted({k for t in tags for k in A[t]["recall_by_type"]})
    tasks = sorted({k for t in tags for k in A[t]["false_alarms_by_task"]})
    L += ["| Model | " + " | ".join(types) + " |", "|---|" + "---|" * len(types)]
    for t in order:
        L.append(f"| {nm(t)} | " + " | ".join(f"{A[t]['recall_by_type'][k]['k']}/{A[t]['recall_by_type'][k]['n']}" if k in A[t]["recall_by_type"] else "-" for k in types) + " |")
    L += ["", "False alarms by HarperValley task:", "", "| Model | " + " | ".join(tasks) + " |", "|---|" + "---|" * len(tasks)]
    for t in order:
        L.append(f"| {nm(t)} | " + " | ".join(f"{A[t]['false_alarms_by_task'][k]['k']}/{A[t]['false_alarms_by_task'][k]['n']}" if k in A[t]["false_alarms_by_task"] else "-" for k in tasks) + " |")
    L += ["", "Category of the first alert on each false-alarm call:", ""]
    for t in order:
        cats = A[t]["false_alarm_categories_first_alert"]
        L.append(f"- {nm(t)}: " + (", ".join(f"{k} {v}" for k, v in cats.items()) or "none"))
    L += ["", "## 6. Cost, latency, reliability", "",
          "| Model | Cost (USD) | per 1,000 calls | per call-minute | p50 / p95 latency | Turn errors | Re-runs + JSON-mode retries | Pricing basis |",
          "|---|---|---|---|---|---|---|---|"]
    for t in order:
        c = res["cost"][t]
        money = lambda x, d=2: "-" if x is None else f"${x:.{d}f}"
        lat = "-" if c["latency_p50_ms"] is None else f"{c['latency_p50_ms']} / {c['latency_p95_ms']} ms"
        L.append(f"| {nm(t)} | {money(c['cost_usd'])} | {money(c['cost_per_1k_calls'])} | {money(c['cost_per_call_minute'], 4)} | {lat} | "
                 f"{c['final_errors']}/{c['requests']} | {c['rerun_rows']} + {c['json_mode_retries']} | {c['pricing']} |")
    ab = res["ablation_dev_tuned"]
    L += ["", "## 7. Ablation: let each model pick its own operating point on dev", "",
          f"Split: {ab['split_source']}; {ab['dev_calls']} dev calls, {ab['test_calls']} test calls. For each model the point "
          f"(watch / alert / urgent) is chosen on dev only (max balanced accuracy with false alarms <= {MAX_DEV_FAR:.0%}); "
          "the table shows test calls only, next to the official alert rule on the same test calls.", "",
          "| Model | Chosen on dev | Test recall (95% CI) | Test false alarms (95% CI) | Test balanced acc. | Official rule on test: recall / FA / bal. acc. |",
          "|---|---|---|---|---|---|"]
    for t in order:
        m = ab["models"][t]; c = m["test_chosen"]; o = m["test_official"]
        L.append(f"| {nm(t)} | {m['chosen_op']}{' *' if m['why'].startswith('no point') else ''} | {c['tp']}/{c['scam_n']} = {_p(c['recall'])} ({_ci(c['recall_ci'])}) | "
                 f"{c['fp']}/{c['benign_n']} = {_p(c['far'])} ({_ci(c['far_ci'])}) | {_p(c['balanced_acc'])} | "
                 f"{_p(o['recall'])} / {_p(o['far'])} / {_p(o['balanced_acc'])} |")
    if any(m["why"].startswith("no point") for m in ab["models"].values()):
        L += ["", f"\\* No operating point kept false alarms <= {MAX_DEV_FAR:.0%} on dev, so the point with the best dev balanced accuracy is shown instead."]
    if "synthetic" in ab["split_source"]:
        L += ["", "The split here is a stand-in (seeded hash, blind to scores) until the calls carry a real `split` field; "
              "re-running the script picks the real split up automatically."]
    for s in res["slices"]:
        if s == "all":
            continue
        L += ["", f"### Slice {s}", "", "| Model | Recall (95% CI) | False alarms (95% CI) | Balanced acc. | AUC (95% CI) |", "|---|---|---|---|---|"]
        for t in order:
            S = res["slices"][s][t]; o = S["ops"][OFFICIAL]
            L.append(f"| {nm(t)} | {o['tp']}/{o['scam_n']} = {_p(o['recall'])} ({_ci(o['recall_ci'])}) | {o['fp']}/{o['benign_n']} = {_p(o['far'])} "
                     f"({_ci(o['far_ci'])}) | {_p(o['balanced_acc'])} | {_auc(S)} |")
    if partial:
        L += ["", "Partial score files (still running or stopped early) are in the tables above with their call count, "
              "left out of the paired table and the ROC chart: " + ", ".join(nm(t) for t in partial) + "."]
    L += ["", "## Caveats", "",
          "- Scams are one-sided prerecorded robocalls (NCSU, public domain); benign calls are role-played HarperValley bank calls (CC BY 4.0). "
          "The classes come from different sources, so style (monologue vs dialogue) can leak the label. Read this as a real-audio check, not the headline benchmark.",
          f"- Precision and F1 depend on the scam share in this set (about {round(100 * res['n_scam'] / max(1, res['n_calls']))}%); real call traffic has far fewer scams, so real-world precision would be lower.",
          f"- With {res['n_scam']} scam and {res['n_benign']} benign calls (about half that per split), a 5-10 point gap can be within noise: check the paired CIs and McNemar p before calling a winner.",
          "- Time to first alert on robocalls measures how much of a short monologue a model needs, not how early it catches a live conversation.",
          "- Dedicated-GPU cost for our guardian is summed request time at $8/h (one request at a time), excluding idle time: a rough figure, not an invoice. "
          "Token prices for models not listed in `PRICES` are left blank unless given with `--price`.",
          "- Failed or unparseable turns count as risk none. Nothing here was tuned on these calls except the dev-side choice in section 7, which is reported on test only.", ""]
    path.write_text("\n".join(L) + "\n")


def write_svg(res: dict, path: Path) -> None:
    A = res["slices"]["all"]; tags = [t for t in res["models"] if res["models"][t]["complete"] and A[t]["auc"] is not None]
    W, H, x0, y0, S = 780, 520, 70, 60, 360
    frontier_cols = ["#2a78d6", "#7a4fd0", "#13907d", "#b8860b", "#c2408f", "#4a6fa5"]
    style, fi = {}, 0
    for t in tags:
        k = kind(t)
        if k == "ours":
            style[t] = ("#d4461a", 3.5, "", "circle")
        elif k == "stub":
            style[t] = ("#8a8984", 1.5, "2 4", "square")
        else:
            style[t] = (frontier_cols[fi % len(frontier_cols)], 2, "7 4", "diamond"); fi += 1
    X = lambda f: x0 + f * S
    Y = lambda r: y0 + (1 - r) * S
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="system-ui,-apple-system,Segoe UI,sans-serif">',
         f'<rect width="{W}" height="{H}" fill="#fcfcfb"/>',
         '<text x="16" y="26" font-size="16" font-weight="600" fill="#0b0b0b">Real-audio test set: ROC of each call\'s highest rating</text>',
         f'<text x="16" y="44" font-size="12" fill="#52514e">{res["n_scam"]} scam robocalls vs {res["n_benign"]} benign bank calls; points = flag at watch, alert (official), urgent</text>']
    for g in (0, 0.25, 0.5, 0.75, 1.0):
        o.append(f'<line x1="{X(g)}" y1="{Y(0)}" x2="{X(g)}" y2="{Y(1)}" stroke="#e6e5e0"/>')
        o.append(f'<line x1="{X(0)}" y1="{Y(g)}" x2="{X(1)}" y2="{Y(g)}" stroke="#e6e5e0"/>')
        o.append(f'<text x="{X(g)}" y="{Y(0) + 16}" font-size="11" fill="#52514e" text-anchor="middle">{int(g * 100)}%</text>')
        o.append(f'<text x="{X(0) - 6}" y="{Y(g) + 4}" font-size="11" fill="#52514e" text-anchor="end">{int(g * 100)}%</text>')
    o.append(f'<line x1="{X(0)}" y1="{Y(0)}" x2="{X(1)}" y2="{Y(1)}" stroke="#c9c8c2" stroke-dasharray="3 3"/>')
    o.append(f'<rect x="{X(0)}" y="{Y(1)}" width="{S}" height="{S}" fill="none" stroke="#52514e"/>')
    o.append(f'<text x="{X(0.5)}" y="{Y(0) + 34}" font-size="12" fill="#0b0b0b" text-anchor="middle">False alarms on benign calls</text>')
    o.append(f'<text x="{X(0) - 44}" y="{Y(0.5)}" font-size="12" fill="#0b0b0b" text-anchor="middle" transform="rotate(-90 {X(0) - 44} {Y(0.5)})">Scam recall</text>')
    draw = sorted(tags, key=lambda t: {"stub": 0, "frontier": 1, "ours": 2}[kind(t)])  # ours on top
    for t in draw:
        col, w, dash, mk = style[t]
        pts = sorted(((p["fpr"], p["tpr"], p["thr"]) for p in A[t]["roc"]), key=lambda p: (p[0], p[1]))
        d = " ".join(f"{X(f):.1f},{Y(r):.1f}" for f, r, _ in pts)
        o.append(f'<polyline points="{d}" fill="none" stroke="{col}" stroke-width="{w}"' + (f' stroke-dasharray="{dash}"' if dash else "") + ' stroke-linejoin="round"/>')
        for f, r, thr in pts:
            if thr not in OPS:
                continue
            cx, cy = X(f), Y(r); rr = 5 if thr == OFFICIAL else 3.5
            tip = f"<title>{name(t)} at {thr}: recall {r:.0%}, false alarms {f:.0%}</title>"
            if mk == "circle":
                o.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{rr}" fill="{col}" stroke="#fcfcfb" stroke-width="1.5">{tip}</circle>')
            elif mk == "square":
                o.append(f'<rect x="{cx - rr:.1f}" y="{cy - rr:.1f}" width="{2 * rr}" height="{2 * rr}" fill="{col}" stroke="#fcfcfb" stroke-width="1.5">{tip}</rect>')
            else:
                o.append(f'<polygon points="{cx:.1f},{cy - rr - 1:.1f} {cx + rr + 1:.1f},{cy:.1f} {cx:.1f},{cy + rr + 1:.1f} {cx - rr - 1:.1f},{cy:.1f}" fill="{col}" stroke="#fcfcfb" stroke-width="1.5">{tip}</polygon>')
    lx, ly = X(1) + 30, y0 + 6
    o.append(f'<text x="{lx}" y="{ly}" font-size="12" font-weight="600" fill="#0b0b0b">Model (AUC, 95% CI)</text>')
    leg = sorted(tags, key=lambda t: ({"ours": 0, "frontier": 1, "stub": 2}[kind(t)], -(A[t]["auc"] or 0)))
    for i, t in enumerate(leg):
        col, w, dash, _ = style[t]; yy = ly + 24 + i * 34
        o.append(f'<line x1="{lx}" y1="{yy}" x2="{lx + 30}" y2="{yy}" stroke="{col}" stroke-width="{w}"' + (f' stroke-dasharray="{dash}"' if dash else "") + "/>")
        lab = name(t) + ("" if kind(t) != "frontier" else " (frontier)")
        ci = A[t]["auc_ci"]
        o.append(f'<text x="{lx + 38}" y="{yy + 4}" font-size="12" fill="#0b0b0b"' + (' font-weight="600"' if kind(t) == "ours" else "") + f'>{lab}</text>')
        o.append(f'<text x="{lx + 38}" y="{yy + 18}" font-size="11" fill="#52514e">AUC {A[t]["auc"]:.3f} ({ci[0]:.2f}-{ci[1]:.2f})</text>' if A[t]["auc"] is not None else "")
    o.append(f'<text x="{lx}" y="{Y(0) - 20}" font-size="11" fill="#52514e">Large mark = official alert rule.</text>')
    o.append(f'<text x="{lx}" y="{Y(0) - 6}" font-size="11" fill="#52514e">Up and left is better.</text>')
    o.append(f'<text x="16" y="{H - 28}" font-size="11" fill="#52514e">Robocalls (NCSU, public domain) vs role-played HarperValley bank calls (CC BY 4.0).</text>')
    o.append(f'<text x="16" y="{H - 12}" font-size="11" fill="#52514e">8 kHz mu-law + live Nemotron ASR; same transcripts and prompt for every model. Bootstrap CI: 2000 resamples.</text>')
    o.append("</svg>")
    path.write_text("\n".join(o))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", default="data/realset/calls.jsonl")
    ap.add_argument("--scores-dir", default="data/realset/scores")
    ap.add_argument("--models", nargs="*", help="tags (default: every scores/<tag>.jsonl)")
    ap.add_argument("--price", action="append", default=[], help="tag=in,out  ($ per 1M tokens)")
    ap.add_argument("--out-json", default="data/realset/metrics_plus.json")
    ap.add_argument("--out-md", default="results/realset_metrics_plus.md")
    ap.add_argument("--out-svg", default="results/realset_roc.svg")
    a = ap.parse_args()
    calls = load_calls(a.calls)
    sd = Path(a.scores_dir)
    tags = a.models or sorted(p.stem for p in sd.glob("*.jsonl"))
    tags = [t for t in tags if (sd / f"{t}.jsonl").exists()]
    prices = dict(PRICES)
    pf = Path("data/realset/prices.json")
    if pf.exists():
        prices.update({k: tuple(v) for k, v in json.loads(pf.read_text()).items()})
    for s in a.price:
        k, v = s.split("="); prices[k] = tuple(float(x) for x in v.split(","))
    res = evaluate(calls, tags, sd, prices)
    Path(a.out_json).write_text(json.dumps(res, indent=1, default=list))
    write_md(res, Path(a.out_md))
    write_svg(res, Path(a.out_svg))
    A = res["slices"]["all"]
    for t in tags:
        o = A[t]["ops"][OFFICIAL]; q = A[t]["time_to_first_alert"][OFFICIAL]
        print(f"{t:24s} recall {o['tp']}/{o['scam_n']} FA {o['fp']}/{o['benign_n']} BA {o['balanced_acc']} AUC {A[t]['auc']} {A[t]['auc_ci']} "
              f"ttfa {q['seconds']['median'] if q['seconds'] else '-'}s cost {res['cost'][t]['cost_usd']} "
              f"dev-op {res['ablation_dev_tuned']['models'][t]['chosen_op']}")


if __name__ == "__main__":
    main()

"""Score a guardian (or baseline) on generated test calls, the way the live line runs it.

  python3 -I callguard/eval/run_eval.py --model keywords                       # B0, local
  python3 -I callguard/eval/run_eval.py --model kimi --max-calls 150           # B4, serverless (Numan's key)
  python3 -I callguard/eval/run_eval.py --model "accounts/<acct>/models/<m>#<deployment>" --name guardian-v1 \
        --key-env FIREWORKS_API_KEY_MATT                                       # tuned model / B3 on a deployment

Each call is scored turn by turn in order, with the model's own previous verdict fed back as
"earlier", and stops at the first alert (that is all catch-before-ask and false alarms need).
Writes results/preds_<name>.jsonl and appends results/results.jsonl (results_row contract).
"""
import argparse, concurrent.futures as cf, json, math, re, statistics, sys, threading, time
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "guardian"))
sys.path.insert(0, str(HERE.parents[1] / "factory"))
import taxonomy as T
from fwclient import USAGE, chat
from render import PROMPT_SHA, RISKS, WINDOW, parse_response, render_messages

ROOT = HERE.parents[1]
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

# ------------------------------------------------------------------ B0 keyword baseline
KW = [(cat, re.compile(r"\b(" + "|".join(re.escape(p) for p in phrases) + r")\b", re.I))
      for cat, phrases in T.LEXICON.items() if cat != "pressure"]
PRESSURE = re.compile(r"\b(" + "|".join(re.escape(p) for p in T.LEXICON["pressure"]) + r")\b", re.I)
PII = re.compile(r"\b(\d{3}-\d{2}-\d{4}|\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4})\b")


def keywords_verdict(req):
    text = " ".join(t["text"] for t in req["turns"])
    for cat, rx in KW:
        m = rx.search(text)
        if m:
            return {"risk": "alert", "category": cat, "evidence": [{"turn": req["turns"][-1]["turn"], "quote": m.group(0)}]}
    if PII.search(text):
        return {"risk": "alert", "category": "disclosure.financial_id", "evidence": []}
    if PRESSURE.search(text):
        return {"risk": "watch", "category": "none", "evidence": []}
    return {"risk": "none", "category": "none", "evidence": []}


# ------------------------------------------------------------------ serverless Training API sampler backend
_TAPI = {}
_TAPI_LOCK = threading.Lock()


def tapi_chat(model, messages, key_env, max_tokens=400):
    """model = 'tapi:<sampler checkpoint path>' (tuned) or 'tapi-base:<accounts/fireworks/models/...>' (untuned)."""
    import os
    with _TAPI_LOCK:
        if model not in _TAPI:
            from fireworks.training.sdk import FiretitanServiceClient
            from transformers import AutoTokenizer, PreTrainedTokenizerFast
            hf = os.environ.get("TAPI_HF", "Qwen/Qwen3.6-35B-A3B")
            tok = PreTrainedTokenizerFast.from_pretrained(hf) if "nemotron" in hf.lower() else AutoTokenizer.from_pretrained(hf)
            svc = FiretitanServiceClient(api_key=os.environ[key_env], base_url="https://api.fireworks.ai/training/v1/serverless")
            kind, ref = model.split(":", 1)
            sampler = (svc.create_sampling_client(model_path=ref, tokenizer=tok) if kind == "tapi"
                       else svc.create_sampling_client(base_model=ref, tokenizer=tok))
            _TAPI[model] = (tok, sampler)
    tok, sampler = _TAPI[model]
    from fireworks.training.sdk import types
    ids = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, enable_thinking=False)
    if hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    res = sampler.sample(prompt=types.ModelInput.from_ints(list(ids)), num_samples=1,
                         sampling_params=types.SamplingParams(max_tokens=max_tokens, temperature=0.0)).result()
    out = tok.decode(res.sequences[0].tokens).replace("<|im_end|>", "")
    return out.split("</think>")[-1].strip()


# ------------------------------------------------------------------ gold
def gold_risky(row):
    return not row.get("benign_twin") and any(RISKS.index(t["label"]["risk"]) >= 2 for t in row["turns"])


def gold_first_alert(row):
    return next((t["turn"] for t in row["turns"] if RISKS.index(t["label"]["risk"]) >= 2), None)


# ------------------------------------------------------------------ scoring one call
def score_call(row, model, key_env, lock, cache, out_f):
    if row["call_id"] in cache:
        return cache[row["call_id"]]
    preds = []
    earlier = None
    first_alert = None
    for idx, t in enumerate(row["turns"]):
        lo = max(0, idx - WINDOW + 1)
        if lo > 0 and preds and len(preds) >= lo:
            p = preds[lo - 1]
            earlier = {"risk": p["risk"], "category": p.get("category", "none"),
                       "evidence": (p.get("evidence") or [{}])[0].get("quote", "") if p.get("evidence") else ""}
        req = {"call_id": row["call_id"], "profile": row["profile"], "rules": row["rules"],
               "caller_known": row["caller_known"], "ai_voice": row.get("ai_voice"), "earlier": earlier,
               "turns": [{"turn": x["turn"], "speaker": x["speaker"], "text": x["text"]} for x in row["turns"][lo: idx + 1]]}
        t0 = time.time()
        if model == "keywords":
            v, ok = keywords_verdict(req), True
        else:
            try:
                if model.startswith("tapi"):
                    txt = tapi_chat(model, render_messages(req), key_env)
                else:
                    txt, _, _ = chat(model, render_messages(req), key_env=key_env, max_tokens=400, temperature=0.0)
                v = parse_response(txt)
                ok = v is not None
                v = v or {"risk": "none", "category": "none", "evidence": []}
            except Exception as e:
                v, ok = {"risk": "none", "category": "none", "evidence": [], "error": str(e)[:120]}, False
        v["_ms"] = int((time.time() - t0) * 1000)
        v["_ok"] = ok
        v["_turn"] = t["turn"]
        preds.append(v)
        if RISKS.index(v["risk"]) >= 2:
            first_alert = t["turn"]
            break
    res = {"call_id": row["call_id"], "first_alert": first_alert, "preds": preds}
    with lock:
        out_f.write(json.dumps(res) + "\n")
        out_f.flush()
    return res


# ------------------------------------------------------------------ metrics
def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round((c - r) / d, 3), round((c + r) / d, 3)]


def metrics(rows, results):
    by = {r["call_id"]: r for r in results}
    risky = [r for r in rows if gold_risky(r)]
    benign = [r for r in rows if not gold_risky(r)]
    asked = [r for r in risky if r.get("ask_turn")]
    caught_strict = sum(1 for r in asked if by[r["call_id"]]["first_alert"] and by[r["call_id"]]["first_alert"] < r["ask_turn"])
    caught_at = sum(1 for r in asked if by[r["call_id"]]["first_alert"] and by[r["call_id"]]["first_alert"] <= r["ask_turn"])
    recall_any = sum(1 for r in risky if by[r["call_id"]]["first_alert"])
    fa = sum(1 for r in benign if by[r["call_id"]]["first_alert"])
    leads = [r["ask_turn"] - by[r["call_id"]]["first_alert"] for r in asked if by[r["call_id"]]["first_alert"]]
    held = [r for r in risky if r.get("heldout_type")]
    held_c = sum(1 for r in held if by[r["call_id"]]["first_alert"])
    # rule-flip pairs: both members decided right
    ids = {r["call_id"]: r for r in rows}
    pairs = [(ids[r["rule_flip_of"]], r) for r in rows if r.get("rule_flip_of") and r["rule_flip_of"] in ids]
    pair_ok = sum(1 for a, b in pairs if all(bool(by[x["call_id"]]["first_alert"]) == gold_risky(x) for x in (a, b)))
    # category at first alert vs gold category at that turn
    tp = {}
    fp = {}
    fn = {}
    for r in risky:
        res = by[r["call_id"]]
        if not res["first_alert"]:
            continue
        g = r["turns"][res["first_alert"] - 1]["label"]["category"]
        p = res["preds"][-1].get("category", "none")
        if p == g:
            tp[g] = tp.get(g, 0) + 1
        else:
            fp[p] = fp.get(p, 0) + 1
            fn[g] = fn.get(g, 0) + 1
    cats = set(tp) | set(fn)
    f1s = []
    for c in cats:
        P = tp.get(c, 0) / max(1, tp.get(c, 0) + fp.get(c, 0))
        R = tp.get(c, 0) / max(1, tp.get(c, 0) + fn.get(c, 0))
        f1s.append(0 if P + R == 0 else 2 * P * R / (P + R))
    # evidence quotes that really appear in the transcript
    ev_n = ev_ok = 0
    for r in rows:
        alltext = " ".join(t["text"] for t in r["turns"]).lower()
        for p in by[r["call_id"]]["preds"]:
            for e in p.get("evidence") or []:
                q = (e.get("quote") or "").lower().strip()
                if q:
                    ev_n += 1
                    ev_ok += int(q in alltext)
    ms = [p["_ms"] for res in results for p in res["preds"]]
    bad = sum(1 for res in results for p in res["preds"] if not p.get("_ok", True))
    m = {
        "catch_before_ask": (caught_strict, len(asked)),
        "catch_at_or_before_ask": (caught_at, len(asked)),
        "recall_any": (recall_any, len(risky)),
        "false_alarm_rate": (fa, len(benign)),
        "unseen_type_recall": (held_c, len(held)),
        "rule_flip_accuracy": (pair_ok, len(pairs)),
    }
    rows_out = []
    for k, (a, n) in m.items():
        rows_out.append({"metric": k, "value": round(a / n, 3) if n else None, "n": n, "ci95": wilson(a, n)})
    rows_out += [
        {"metric": "lead_time_turns_median", "value": statistics.median(leads) if leads else None, "n": len(leads)},
        {"metric": "category_macro_f1_at_first_alert", "value": round(sum(f1s) / len(f1s), 3) if f1s else None, "n": len(cats)},
        {"metric": "evidence_exact_match", "value": round(ev_ok / ev_n, 3) if ev_n else None, "n": ev_n},
        {"metric": "latency_ms_p50", "value": statistics.median(ms) if ms else None, "n": len(ms)},
        {"metric": "latency_ms_p95", "value": sorted(ms)[int(0.95 * (len(ms) - 1))] if ms else None, "n": len(ms)},
        {"metric": "unparseable_replies", "value": bad, "n": len(ms)},
    ]
    return rows_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="keywords | kimi | deepseek | full model id (#deployment ok)")
    ap.add_argument("--name", help="label used in results files")
    ap.add_argument("--test", default=str(ROOT / "data" / "calls_test_heldout.jsonl"))
    ap.add_argument("--test-set", default="heldout")
    ap.add_argument("--key-env", default="FIREWORKS_API_KEY")
    ap.add_argument("--max-calls", type=int, default=0)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--preds-tag", default=None, help="reuse/extend cached preds from this tag (default: --test-set)")
    a = ap.parse_args()
    name = a.name or a.model.split("/")[-1].replace("#", "@")
    rows = [json.loads(l) for l in Path(a.test).read_text().splitlines() if l.strip()]
    if a.max_calls:
        # keep rule-flip pairs together when subsampling
        keep, seen = [], set()
        ids = {r["call_id"]: r for r in rows}
        for r in rows:
            base = r.get("rule_flip_of") or r["call_id"]
            if base in seen:
                continue
            group = [x for x in rows if (x.get("rule_flip_of") or x["call_id"]) == base]
            if len(keep) + len(group) > a.max_calls:
                break
            keep += group
            seen.add(base)
        rows = keep
    pred_path = RES / f"preds_{name}__{a.preds_tag or a.test_set}.jsonl"
    cache = {}
    if pred_path.exists():
        for l in pred_path.read_text().splitlines():
            d = json.loads(l)
            cache[d["call_id"]] = d
    lock = threading.Lock()
    t0 = time.time()
    with pred_path.open("a") as f, cf.ThreadPoolExecutor(1 if a.model == "keywords" else a.workers) as ex:
        results = list(ex.map(lambda r: score_call(r, a.model, a.key_env, lock, cache, f), rows))
    out = metrics(rows, results)
    with (RES / "results.jsonl").open("a") as f:
        for m in out:
            m.update({"model": name, "test_set": a.test_set, "prompt_sha": PROMPT_SHA, "ts": int(time.time())})
            f.write(json.dumps(m) + "\n")
    print(f"== {name} on {a.test_set}: {len(rows)} calls in {time.time()-t0:.0f}s | {USAGE}")
    for m in out:
        print(f"  {m['metric']:34} {m['value']}  (n={m['n']}{', ci95=' + str(m['ci95']) if 'ci95' in m else ''})")


if __name__ == "__main__":
    main()

"""Call-level metrics on the realset -> results/realset_results.jsonl, results/realset_chart.svg.

Decision rule: a call is flagged if ANY scored caller turn has risk in {alert, urgent}.
Scam recall = flagged scam calls / scam calls; false-alarm rate = flagged benign / benign.
A failed request (HTTP error, unparseable reply) counts as "not flagged" for that turn and
is reported as errors. Wilson 95% intervals.

    python -m realset.metrics                       # every data/realset/scores/<tag>.jsonl
    python -m realset.metrics --models kimi-k3 qwen3p8-max keyword-stub

results/realset.md is written by hand from these numbers and realset.metrics_plus; this script writes its
auto-generated table to data/realset/realset_auto.md so a re-run does not overwrite the write-up.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from contracts.models import ResultsRow

ALERT = {"alert", "urgent"}
# Display order and names: ours first, then frontier baselines, then the stub.
ORDER = ["emberz-guardian-v2", "kimi-k3", "qwen3p8-max", "nemotron-lightning", "deepseek-v4p1-flash", "rune", "keyword-stub"]
NAMES = {"emberz-guardian-v2": "Emberz guardian v2 (ours)", "kimi-k3": "Kimi K3", "qwen3p8-max": "Qwen3.8 Max",
         "nemotron-lightning": "Nemotron Lightning 3.5", "deepseek-v4p1-flash": "DeepSeek V4.1 Flash",
         "rune": "Rune (decisions API)", "keyword-stub": "keyword stub"}
# $ per 1M tokens (in, out): Qwen3.8 Max from PLAN.md B5; Kimi K3 from docs/business_value.md
PRICES = {"qwen3p8-max": (2.0, 6.0), "kimi-k3": (3.0, 15.0), "keyword-stub": (0, 0)}
# Serverless standard tier, https://docs.fireworks.ai/serverless/pricing (read 2026-11-02): DeepSeek V4.1 Flash
# $0.30/$1.20; Nemotron 3.5 Lightning 30B A3B $0.05/$0.20; Rune 26B-A4B $0.042 in, output always 0 (decisions API).
PRICES.update({"deepseek-v4p1-flash": (0.30, 1.20), "nemotron-lightning": (0.05, 0.20), "rune": (0.042, 0.0)})


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, c - h), 4), round(min(1.0, c + h), 4))


def load_calls(path="data/realset/calls.jsonl"):
    return {c["call_id"]: c for c in (json.loads(l) for l in open(path) if l.strip())}


def call_decisions(calls, tag):
    p = Path(f"data/realset/scores/{tag}.jsonl")
    per = defaultdict(list)
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "requests": 0, "errors": 0, "latency_ms": []}
    for l in open(p):
        r = json.loads(l)
        if r["call_id"] not in calls:
            continue
        per[r["call_id"]].append(r)
        u = r.get("usage") or {}  # cost counts every attempt, including superseded failed rows
        usage["prompt_tokens"] += u.get("prompt_tokens", 0)
        usage["completion_tokens"] += u.get("completion_tokens", 0)
    out = {}
    for cid, c in calls.items():
        rows = {r["turn"]: r for r in per.get(cid, [])}  # last write per turn wins (retries)
        n_caller = sum(1 for t in c["turns"] if t["speaker"] == "caller")
        if n_caller and len(rows) < n_caller:
            continue  # not fully scored yet
        first = None
        for t in sorted(rows):
            r = rows[t]
            usage["requests"] += 1
            usage["json_retries"] = usage.get("json_retries", 0) + bool(r.get("json_retry"))
            if r.get("latency_ms"):
                usage["latency_ms"].append(r["latency_ms"])
            v = r.get("verdict")
            if v is None:
                usage["errors"] += 1
            elif v.get("risk") in ALERT and first is None:
                first = t
        out[cid] = {"flagged": first is not None, "first_alert_turn": first, "n_caller_turns": n_caller}
    return out, usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None, help="tags (default: every scores/<tag>.jsonl)")
    ap.add_argument("--md", default="data/realset/realset_auto.md")
    a = ap.parse_args()
    if not a.models:
        found = [p.stem for p in Path("data/realset/scores").glob("*.jsonl")]
        a.models = [t for t in ORDER if t in found] + sorted(t for t in found if t not in ORDER)
    calls = load_calls()
    rows, summary = [], {}
    for tag in a.models:
        if not Path(f"data/realset/scores/{tag}.jsonl").exists():
            continue
        dec, usage = call_decisions(calls, tag)
        s = [cid for cid in dec if calls[cid]["label"] == "scam"]
        b = [cid for cid in dec if calls[cid]["label"] == "benign"]
        ks = sum(dec[c]["flagged"] for c in s)
        kb = sum(dec[c]["flagged"] for c in b)
        pin, pout = PRICES.get(tag, (0, 0))
        cost = usage["prompt_tokens"] / 1e6 * pin + usage["completion_tokens"] / 1e6 * pout
        lat = sorted(usage["latency_ms"])
        p50 = lat[len(lat) // 2] if lat else None
        rr = ResultsRow(model=tag, test_set="realset", metric="scam_recall", value=round(ks / max(1, len(s)), 4), n=len(s), ci95=wilson(ks, len(s)))
        fa = ResultsRow(model=tag, test_set="realset", metric="false_alarm_rate", value=round(kb / max(1, len(b)), 4), n=len(b), ci95=wilson(kb, len(b)))
        rows += [rr, fa]
        by_type = defaultdict(lambda: [0, 0])
        for c in s:
            by_type[calls[c]["scam_type"]][0] += dec[c]["flagged"]; by_type[calls[c]["scam_type"]][1] += 1
        summary[tag] = {"scam_k": ks, "scam_n": len(s), "recall": rr.value, "recall_ci": rr.ci95,
                        "benign_k": kb, "benign_n": len(b), "far": fa.value, "far_ci": fa.ci95,
                        "requests": usage["requests"], "errors": usage["errors"], "json_retries": usage.get("json_retries", 0), "prompt_tokens": usage["prompt_tokens"],
                        "completion_tokens": usage["completion_tokens"], "cost_usd": round(cost, 3), "latency_p50_ms": p50,
                        "recall_by_type": {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_type.items())},
                        "missed": [c for c in s if not dec[c]["flagged"]], "false_alarms": [c for c in b if dec[c]["flagged"]]}
    with open("results/realset_results.jsonl", "w") as f:
        for r in rows:
            f.write(r.model_dump_json() + "\n")
    json.dump(summary, open("data/realset/summary.json", "w"), indent=1)
    svg(summary)
    write_md(summary, calls, a.md)
    print(json.dumps({k: {x: v[x] for x in ("scam_k", "scam_n", "recall", "recall_ci", "benign_k", "benign_n", "far", "far_ci", "errors", "cost_usd", "latency_p50_ms")} for k, v in summary.items()}, indent=1))


def _pct(x):
    return f"{x * 100:.0f}%"


def write_md(summary, calls, path="results/realset.md"):
    from collections import Counter
    n_s = sum(c["label"] == "scam" for c in calls.values()); n_b = len(calls) - n_s
    types = Counter(c["scam_type"] for c in calls.values() if c["label"] == "scam")
    tasks = Counter(c["scam_type"].split(".")[-1] for c in calls.values() if c["label"] == "benign")
    audio_min = sum(c.get("audio_s", 0) for c in calls.values()) / 60
    ct = [sum(t["speaker"] == "caller" for t in c["turns"]) for c in calls.values()]
    L = ["# Real-audio test set (realset): frontier baselines", "",
         f"**{len(calls)} real calls** ({n_s} scam robocalls, {n_b} benign bank calls), {audio_min:.0f} min of audio, "
         f"{sum(ct)} scored caller turns. Every call went through our phone path and live ASR; every model saw identical transcripts.", "",
         "![realset chart](realset_chart.svg)", "",
         "## Results (call level)", "",
         "| Model | Scam recall | 95% CI | False alarms on benign | 95% CI | Turn errors | Cost (USD) | p50 latency |",
         "|---|---|---|---|---|---|---|---|"]
    for m, v in summary.items():
        L.append(f"| {m} | {v['scam_k']}/{v['scam_n']} = **{_pct(v['recall'])}** | {_pct(v['recall_ci'][0])}-{_pct(v['recall_ci'][1])} | "
                 f"{v['benign_k']}/{v['benign_n']} = **{_pct(v['far'])}** | {_pct(v['far_ci'][0])}-{_pct(v['far_ci'][1])} | "
                 f"{v['errors']}/{v['requests']} ({v.get('json_retries', 0)} JSON-mode retries) | {v['cost_usd']:.2f} | {v['latency_p50_ms'] or '-'} ms |")
    L += ["", "Recall by scam type:", "", "| Model | " + " | ".join(sorted(types)) + " |", "|---|" + "---|" * len(types)]
    for m, v in summary.items():
        L.append(f"| {m} | " + " | ".join(v["recall_by_type"].get(t, "-") for t in sorted(types)) + " |")
    L += ["", "## Method", "",
          f"- **Scam ({n_s}):** NCSU Robocall Audio Dataset (FTC Project Point of No Entry; data public domain, "
          "https://github.com/wspr-ncsu/robocall-audio-dataset). English, transcript >= 25 words, near-duplicate campaigns removed, "
          "seeded (20261102) stratified pick by a keyword scam type read from the dataset's Whisper transcript, one per carrier case first. "
          "Types: " + ", ".join(f"{k} {v}" for k, v in sorted(types.items())) + ".",
          f"- **Benign ({n_b}):** HarperValleyBank (Gridspace-Stanford, CC BY 4.0, https://github.com/cricketclub/gridspace-stanford-harper-valley). "
          "Real human speakers role-playing bank customer-service calls with account, card, PIN/password and transfer talk (hard negatives). "
          "Seeded pick from a 220-call random sample, round-robin over task types, calls >= 25 s. Tasks: "
          + ", ".join(f"{k} {v}" for k, v in sorted(tasks.items())) + ".",
          "- **Phone path + ASR:** each channel resampled to 8 kHz, G.711 mu-law (`line/audio.py`), fed in 20 ms frames into "
          "`line.asr.CallTranscriber` (Nemotron Speech Streaming 0.6B int8 via sherpa-onnx, the live line's code and endpointing). "
          "Robocalls: the recording is the caller track; the protected track is not fed (it would be digital silence). "
          "HarperValley: agent channel = caller, customer channel = protected.",
          "- **Guardian input:** Matt's frozen prompt `guardian/render.py` (PROMPT_SHA 806908cfba7d). One request at the end of every caller turn "
          "with the last 8 turns; profile `elder` for all calls (so the profile can't leak the class), no rules, unknown number, voice unknown, "
          "no 'earlier' verdict. Temperature 0, max_tokens 600, reasoning_effort none.",
          "- **JSON:** no response_format on the first attempt (same as Matt's sampler). A reply with no parseable JSON is retried once with "
          "`response_format: json_object`; both attempts are counted in cost.",
          "- **Decision:** a call is flagged if any scored turn has risk alert or urgent. Wilson 95% intervals. Failed or unparseable replies count as not flagged.",
          "- **Reproduce:** `python -m realset.select_calls` (manifest), fetch audio from the URLs in `realset/manifest.csv`, "
          "`python -m realset.transcribe`, `python -m realset.score --model <id> --tag <tag>`, `python -m realset.metrics`.",
          "", "## Caveats", "",
          "- **Robocalls are one-sided prerecorded messages** (often TTS), not conversations. They test whether a model recognizes a real scam pitch "
          "through real ASR errors; they do not test catching the ask mid-conversation, and there is no victim side.",
          "- **HarperValley is role-played** from task templates by paid speakers (real voices, scripted scenario, small vocabulary), so it is an easy benign set "
          "in content even though it is full of account and card talk. Real unscripted benign calls (CallHome/CallFriend) are not in this set.",
          "- The two classes come from different sources, so a model could separate them by style (monologue vs dialogue). Read this as a sanity check on real audio, "
          "not as the headline benchmark.",
          "- Scam type is a keyword bucket over the dataset's Whisper transcript, not a human label; labels are source-level (every NCSU call is treated as scam, "
          "the FTC letters name them as illegal robocall campaigns).",
          "- In HarperValley the customer phoned the bank; we still put the **agent on the caller track** (as if the bank called the "
          "customer from an unknown number), which makes the benign set harder: the 'caller' asks for addresses, account and card details.",
          "- The ASR is our live Nemotron path on simulated (offline) G.711, not a real PSTN leg.", ""]
    L += ["## Failure cases", "",
          "- **False alarms** on HarperValley come from the window rule putting the bank agent on the caller side with an *unknown number*: "
          "the agent asks for an address, account or card details, and the frontier models read that as a stranger collecting personal data "
          "(categories below). That is arguably the right call for an unverified inbound caller, so these are hard negatives by design.",
          "- **Missed scams** have no explicit ask for money or data in the ASR text. Both models missed a utility 'supply charges' callback "
          "(ncsu_1348032) and a solar lead-gen pitch (ncsu_447375; illegal telemarketing, arguably not a scam). Kimi K3 also missed an AT&T DirecTV "
          "'50% off' pitch (ncsu_547611); Qwen3.8 Max missed a different DirecTV '50% off, call us back' message (ncsu_28342) and a car-warranty "
          "'press one' message (ncsu_63405). The DirecTV and warranty calls are known scam campaigns, so those are real misses, not label noise.",
          "- **Scam type** is the keyword bucket, hand-corrected for 6 of 61 calls after reading the transcripts (`TYPE_FIX` in `realset/select_calls.py`: "
          "4 power-disconnection / 'Microsoft' calls moved to tech_support_utility, 2 SSN / COVID-relief calls moved to gov_ssa_irs). The pick itself is unchanged.",
          "- Cost counts every attempt (including the 8 Qwen turns re-run after an unparseable reply and the 33 JSON-mode retries); "
          "p50 latency is the final attempt only.", ""]
    for m in summary:
        cats = Counter()
        sp = Path(f"data/realset/scores/{m}.jsonl")
        for l in open(sp):
            r = json.loads(l)
            v = r.get("verdict") or {}
            if r["call_id"] in summary[m]["false_alarms"] and v.get("risk") in ALERT:
                cats[v.get("category")] += 1
        if cats:
            L.append(f"- {m} alert categories on false-alarm calls (turns): " + ", ".join(f"{k} {n}" for k, n in cats.most_common()))
    L.append("")
    for m, v in summary.items():
        if m == "keyword-stub":
            L.append(f"- keyword-stub (contracts/stub_guardian.py, 4 trigger words: code, gift card, street, home alone) caught {v['scam_k']}/{v['scam_n']}; "
                     "it is a placeholder stub, not a tuned keyword baseline.")
            continue
        if v["missed"] or v["false_alarms"]:
            L.append(f"- {m}: missed scams {', '.join(v['missed'][:12]) or 'none'}; false alarms {', '.join(v['false_alarms'][:12]) or 'none'}.")
    Path(path).write_text("\n".join(L) + "\n")


def svg(summary, path="results/realset_chart.svg"):
    """Two panels on one 0-100% axis each: scam recall (higher is better) and false-alarm rate
    (lower is better), one dot + Wilson 95% whisker per model. Our guardian is highlighted."""
    models = list(summary)
    W, rowh, top, left, pw, gap = 820, 34, 78, 200, 260, 44
    H = top + rowh * len(models) + 66
    OURS, FRONTIER, STUB = "#d6402b", "#2a78d6", "#8a897f"
    def col(m):
        return OURS if m.startswith("emberz") else STUB if "stub" in m else FRONTIER
    first = next(iter(summary.values())) if summary else {"scam_n": 0, "benign_n": 0}
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="system-ui,-apple-system,Segoe UI,sans-serif">',
         f'<rect width="{W}" height="{H}" fill="#fcfcfb"/>',
         '<text x="16" y="26" font-size="16" font-weight="600" fill="#0b0b0b">Real-audio test set: fine-tuned guardian vs frontier baselines (call level, Wilson 95% CI)</text>',
         f'<text x="16" y="46" font-size="12" fill="#52514e">{first["scam_n"]} real robocalls (NCSU, public domain) vs {first["benign_n"]} HarperValley bank calls (CC BY 4.0), 8 kHz mu-law + live ASR, one frozen prompt for all</text>']
    for i, m in enumerate(models):
        if m.startswith("emberz"):
            o.append(f'<rect x="8" y="{top + rowh * i}" width="{W - 16}" height="{rowh}" fill="#d6402b" fill-opacity="0.07" rx="4"/>')
    panels = [("Scam recall (higher is better)", "recall", "recall_ci", "scam_k", "scam_n"),
              ("False alarms on benign (lower is better)", "far", "far_ci", "benign_k", "benign_n")]
    for pi, (title, key, ci, k, n) in enumerate(panels):
        x0 = left + pi * (pw + gap)
        o.append(f'<text x="{x0}" y="{top - 10}" font-size="12" font-weight="600" fill="#0b0b0b">{title}</text>')
        for t in (0, 0.25, 0.5, 0.75, 1.0):
            x = x0 + t * pw
            o.append(f'<line x1="{x}" y1="{top}" x2="{x}" y2="{top + rowh * len(models)}" stroke="#e6e5e0" stroke-width="1"/>')
            o.append(f'<text x="{x}" y="{top + rowh * len(models) + 16}" font-size="11" fill="#52514e" text-anchor="middle">{int(t * 100)}%</text>')
        for i, m in enumerate(models):
            s = summary[m]; y = top + rowh * i + rowh / 2
            lo, hi = s[ci]; c = col(m); ours = m.startswith("emberz")
            if pi == 0:
                o.append(f'<text x="{left - 14}" y="{y + 4}" font-size="12" fill="#0b0b0b" text-anchor="end"'
                         f'{" font-weight=\"700\"" if ours else ""}>{NAMES.get(m, m)}</text>')
            o.append(f'<line x1="{x0 + lo * pw}" y1="{y}" x2="{x0 + hi * pw}" y2="{y}" stroke="{c}" stroke-width="{3 if ours else 2}" stroke-linecap="round"/>')
            o.append(f'<circle cx="{x0 + s[key] * pw}" cy="{y}" r="{6 if ours else 5}" fill="{c}" stroke="#fcfcfb" stroke-width="2"><title>{NAMES.get(m, m)}: {s[k]}/{s[n]} = {s[key]:.0%} (95% CI {lo:.0%}-{hi:.0%})</title></circle>')
            lx = x0 + hi * pw + 6 if hi < 0.8 else x0 + lo * pw - 6
            anchor = "start" if hi < 0.8 else "end"
            o.append(f'<text x="{lx}" y="{y - 6}" font-size="11" fill="#52514e" text-anchor="{anchor}">{s[k]}/{s[n]} = {s[key]:.0%}</text>')
    ly = H - 30
    for j, (lab, c) in enumerate((("ours (fine-tuned)", OURS), ("frontier / serverless baselines", FRONTIER), ("keyword stub", STUB))):
        lx = 16 + j * 230
        o.append(f'<circle cx="{lx + 5}" cy="{ly - 4}" r="5" fill="{c}"/><text x="{lx + 16}" y="{ly}" font-size="11" fill="#52514e">{lab}</text>')
    o.append(f'<text x="16" y="{H - 10}" font-size="11" fill="#52514e">Flag = any caller turn scored alert or urgent (last 8 turns). Robocalls are one-sided prerecorded messages; HarperValley is role-played.</text>')
    o.append("</svg>")
    Path(path).write_text("\n".join(o))


if __name__ == "__main__":
    main()

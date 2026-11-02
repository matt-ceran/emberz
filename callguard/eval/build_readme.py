"""Build callguard/README.md from results/results.jsonl (every number comes from the results file or cached preds).

  python3 -I callguard/eval/build_readme.py
Re-run after new results land (e.g. the deployed-latency rows); nothing is typed in by hand except prose.
"""
import json, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
R = ["none", "watch", "alert", "urgent"]

NAMES = {"guardian-v2": "**Guardian v2** (Qwen3.6-35B-A3B LoRA; the deployed checkpoint, scored on the training sampler)", "guardian-v1": "Guardian v1 (Qwen3.6-35B-A3B LoRA)",
         "guardian-v3": "Guardian v3 (v2 + contrast pairs)", "guardian-nemo-v1": "Guardian nemo-v1 (Nemotron 3.5 Super LoRA)",
         "b1-rune": "Fireworks Rune zero-shot (B1)", "b3-qwen3p6-35b-untuned": "Qwen3.6-35B-A3B untuned, same prompt (B3)",
         "b4-kimi-k3": "Kimi K3 zero-shot (B4)", "b5-deepseek-v4p1-flash": "DeepSeek V4.1 Flash zero-shot (B5)",
         "b0-keywords": "Keywords + PII regex (B0)", "guardian-v2-deployed": "Guardian v2 on the dedicated deployment (1x B200 BF16)",
         "guardian-v2-deployed-seq": "Guardian v2 deployed, one call at a time"}
ORDER = ["guardian-v2", "guardian-v1", "guardian-v3", "guardian-nemo-v1", "b1-rune", "b4-kimi-k3", "b5-deepseek-v4p1-flash",
         "b3-qwen3p6-35b-untuned", "b0-keywords", "guardian-v2-deployed"]


def latest():
    best = {}
    for l in (RES / "results.jsonl").read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            best[(r["model"], r["test_set"], r["metric"])] = r
    return best


B = latest()


def cell(m, ts, k, pct=True):
    r = B.get((m, ts, k))
    if not r or r["value"] is None:
        return "-"
    if "ci95" in r and pct:
        hits = round(r["value"] * r["n"])
        return f"{hits}/{r['n']} ({100*r['value']:.0f}%, CI {max(0.0, 100*r['ci95'][0]):.0f}-{min(100.0, 100*r['ci95'][1]):.0f})"
    return f"{r['value']}"


def table(ts, cols, models=None):
    models = [m for m in (models or ORDER) if any((m, ts, c) in B for c, _ in cols)]
    out = ["| Model | " + " | ".join(h for _, h in cols) + " |", "|---|" + "---|" * len(cols)]
    for m in models:
        out.append(f"| {NAMES.get(m, m)} | " + " | ".join(cell(m, ts, c) for c, _ in cols) + " |")
    return "\n".join(out)


def cascade(test_file="data/calls_test_heldout.jsonl", a="b1-rune", b="guardian-v2", ts="heldout"):
    rows = {json.loads(l)["call_id"]: json.loads(l) for l in (ROOT / test_file).read_text().splitlines() if l.strip()}
    pa = {json.loads(l)["call_id"]: json.loads(l) for l in (RES / f"preds_{a}__{ts}.jsonl").read_text().splitlines() if l.strip()}
    pb = {json.loads(l)["call_id"]: json.loads(l) for l in (RES / f"preds_{b}__{ts}.jsonl").read_text().splitlines() if l.strip()}
    risky = lambda r: not r.get("benign_twin") and any(R.index(t["label"]["risk"]) >= 2 for t in r["turns"])
    asked = [r for r in rows.values() if risky(r) and r.get("ask_turn")]
    benign = [r for r in rows.values() if not risky(r)]

    def turn(cid):
        x, y = pa[cid]["first_alert"], pb[cid]["first_alert"]
        return max(x, y) if x and y else None
    c = sum(1 for r in asked if turn(r["call_id"]) and turn(r["call_id"]) < r["ask_turn"])
    f = sum(1 for r in benign if turn(r["call_id"]))
    return c, len(asked), f, len(benign)


def deploy_drift(ts="heldout120"):
    dp, sp = RES / f"preds_guardian-v2-deployed__{ts}.jsonl", RES / "preds_guardian-v2__heldout.jsonl"
    if not dp.exists():
        return ""
    rows = {json.loads(l)["call_id"]: json.loads(l) for l in (ROOT / "data/calls_test_heldout.jsonl").read_text().splitlines() if l.strip()}
    D = {json.loads(l)["call_id"]: json.loads(l) for l in dp.read_text().splitlines() if l.strip()}
    S = {json.loads(l)["call_id"]: json.loads(l) for l in sp.read_text().splitlines() if l.strip()}
    common = [c for c in D if c in S]
    flips = [c for c in common if bool(D[c]["first_alert"]) != bool(S[c]["first_alert"])]
    def kind(c):
        r = rows[c]
        return {"allowed_contact": "allowed-contact twins", "rule": "rule-break twins"}.get(r.get("flip_kind") or "", "harmless calls" if r.get("benign_twin") else "risky calls")
    now_alert, now_quiet = {}, {}
    for c in flips:
        d = now_alert if D[c]["first_alert"] else now_quiet
        d[kind(c)] = d.get(kind(c), 0) + 1
    fmt = lambda d: ", ".join(f"{v} {k if v != 1 else k.replace('calls', 'call').replace('twins', 'twin')}"
                              for k, v in sorted(d.items(), key=lambda kv: -kv[1])) or "none"
    g = lambda m, k: B.get((m, ts, k), {})
    def pct(m, k):
        r = g(m, k)
        return f"{round(r['value'] * r['n'])}/{r['n']}" if r else "-"
    return (f"**Accuracy drift on deployment (must read):** on the same {len(common)} calls, the deployed model differs from the "
            f"evaluated checkpoint on {len(flips)} calls (now alerts: {fmt(now_alert)}; now quiet: {fmt(now_quiet)}). "
            f"Caught before the ask {pct('guardian-v2-deployed', 'catch_before_ask')} vs {pct('guardian-v2', 'catch_before_ask')}, "
            f"false alarms {pct('guardian-v2-deployed', 'false_alarm_rate')} vs {pct('guardian-v2', 'false_alarm_rate')}, "
            f"rule-flip pairs {pct('guardian-v2-deployed', 'rule_flip_accuracy')} vs {pct('guardian-v2', 'rule_flip_accuracy')}. "
            "The losses concentrate on caretaker rules and saved-contact signals, the behavior fine-tuning added, so our working "
            "hypothesis is that serving applies the LoRA only partly (the adapter targets the MoE experts and the linear-attention "
            "projections). `response_format` has no effect on it. The deployment also needs `reasoning_effort: none`; without it the "
            "model thinks and returns empty content. We've raised it with Fireworks staff. **The headline numbers above are the "
            "evaluated checkpoint (Training API sampler), not the deployment.**")


MAIN = [("catch_before_ask", "Caught before the ask"), ("false_alarm_rate", "False alarms (harmless calls)"),
        ("rule_flip_accuracy", "Rule-flip pairs right"), ("recall_any", "Caught at any turn")]


def main():
    c, n, f, nb = cascade()
    fa_section = ""
    p = RES / "RESULTS_heldout.md"
    if p.exists() and "## False alarms by harmless scenario" in p.read_text():
        fa_section = p.read_text().split("## False alarms by harmless scenario (heldout)", 1)[1].strip()
    deployed = ""
    if any(k[0].startswith("guardian-v2-deployed") for k in B):
        parts = []
        seq = [k for k in B if k[0] == "guardian-v2-deployed-seq" and k[2] == "latency_ms_p50"]
        if seq:
            ts = seq[0][1]
            g = lambda k: B.get(("guardian-v2-deployed-seq", ts, k), {})
            parts.append(f"**Single-stream latency per scored turn** (one call at a time, {g('latency_ms_p50').get('n')} turns): "
                         f"p50 **{g('latency_ms_p50').get('value')} ms**, p95 **{g('latency_ms_p95').get('value')} ms**. "
                         "This is the production latency claim; the sampler latencies elsewhere are not.")
        for t in sorted({k[1] for k in B if k[0] == "guardian-v2-deployed"}):
            parts.append(f"**Same model on the deployment vs the training sampler ({t})**\n\n" +
                         table(t, MAIN + [("latency_ms_p50", "p50 ms (concurrent)"), ("latency_ms_p95", "p95 ms (concurrent)")],
                               ["guardian-v2-deployed", "guardian-v2"]))
        drift = deploy_drift()
        if drift:
            parts.append(drift)
        deployed = "\n\n".join(parts)
    else:
        deployed = "_Pending: production latency and an FP4/BF16 re-score from the dedicated deployment are being measured now._"

    md = f"""# Emberz: a guarded phone number with a fine-tuned call guardian

Give your kid, your grandma or yourself a guarded number. Every call passes through our line: **NVIDIA Nemotron Speech**
transcribes both sides live, and a small guardian model **fine-tuned on Fireworks** reads the transcript turn by turn.
It stays quiet on normal calls, whispers a tip to the protected person when a call turns into a scam, a grooming
pattern, a threat or an info grab, and alerts the caretaker, following rules the caretaker writes in plain words
("never share the door code", "Grandpa is allowed").

## Team

- **Numan Karakas:** the plan, the live line (Telnyx call flow, coach whisper, caretaker alert call), streaming speech-to-text
  with NVIDIA Nemotron Speech, the AI-voice detector (frozen wav2vec2 XLS-R features plus a calibrated head, EER 1.3% on unseen
  In-the-Wild speakers through a simulated phone path), and the Google Cloud jobs. His work lives on `numan/line-voice-gcp`.
- **Matt Ceran:** the guardian (scenario taxonomy, data factory, fine-tuning on Fireworks, the eval harness and baselines),
  and the caretaker console.

## The guardian

- **Task:** given the caretaker's rules, the protected person's profile, two signals (caller saved or unknown; AI-voice score)
  and the last 8 transcript turns, return JSON: risk (none/watch/alert/urgent), category (18 classes), evidence quote,
  action, a 15-word coach line and a 20-word caretaker note. One prompt (`guardian/render.py`, sha `806908cfba7d`) for training, eval and live.
- **Data, all generated:** a 75-seed scenario taxonomy (`factory/SCENARIOS.md`): scams grounded in the FBI IC3 2025 report,
  info disclosure, grooming at the level of tactics only (never sexual content), abuse, threats, self-harm (non-graphic), AI voices,
  and 20 harmless look-alikes with scary words. Plus rule-flip pairs (same call, different caretaker rules).
  - Training calls: **DeepSeek V4.1 Flash** (1,440 calls, turn-level labels from a hidden plan, validated and auto-repaired).
  - Held-out test calls: **NVIDIA Nemotron 3 Ultra** (257 calls), a different model family, with one scam type
    (gold/cash courier pickup) never seen in training.
- **Training:** LoRA rank 16 on **Qwen3.6-35B-A3B** with the Fireworks **serverless Training API** (our own SFT loop, no
  dataset upload; managed SFT uploads were gated on our new account). v1 = 3,930 samples, 1 epoch, about 8 minutes, about $5.80.
  v2 continues v1 on 707 more calls; v3 adds 97 contrast pairs; nemo-v1 trains NVIDIA Nemotron 3.5 Super on v1's data.

## Results

Headline metric: a risky call is **caught before the ask** if the guardian reaches "alert" before the turn where the caller
asks for the code, money, address or meet-up. Calls are scored turn by turn, exactly as the live line runs them.

### All 257 held-out calls (Nemotron 3 Ultra wrote them)

{table("heldout", MAIN)}

**Cascade (Rune flags first, guardian v2 confirms; alert only when both agree):** caught before the ask {c}/{n} ({100*c/n:.0f}%),
false alarms {f}/{nb} ({100*f/nb:.0f}%). Computed from the two models' cached per-turn predictions.

![Caught before the ask vs false alarms](results/chart_catch_vs_false_alarms_heldout.png)

### 120-call subset (the one Kimi K3 was scored on)

{table("heldout120", MAIN)}

### Out of distribution: BothBosu scam conversations (200 calls, 100 scam / 100 not)

Public synthetic dataset (Apache-2.0), call-level labels only, so "caught before the ask" does not apply.

{table("bothbosu", [("recall_any", "Scams caught"), ("false_alarm_rate", "False alarms")])}

### Real speech-to-text: 49 held-out calls voiced and re-transcribed

Voiced with macOS TTS (8 voices) at 8 kHz phone quality, then transcribed by the same **Nemotron Speech Streaming** model the
live line uses (median word error rate 16%, mostly numbers written as words). Same calls as text (clean) vs through ASR:

{table("clean50", [("catch_before_ask", "Caught before the ask (clean text)"), ("recall_any", "Caught at any turn"), ("false_alarm_rate", "False alarms")], ["guardian-v2", "guardian-v1", "b1-rune", "b5-deepseek-v4p1-flash"])}

{table("asr50", [("catch_before_ask", "Caught before the ask (through ASR)"), ("recall_any", "Caught at any turn"), ("false_alarm_rate", "False alarms")], ["guardian-v2", "guardian-v1", "b1-rune", "b5-deepseek-v4p1-flash"])}

### Dev calls (143, never trained on)

{table("dev", MAIN, ["guardian-v2", "guardian-v3"])}

### False alarms by harmless scenario (held-out)

{fa_section}

## Deployed

{deployed}

## Caretaker console

`console/` is the phone-width page the caretaker watches: live transcript of both sides, a risk meter and per-turn timeline,
the alert with its evidence quote and what we told the protected person, Listen in / Join call / Hang up / Mark safe, and
a summary after the call. It reads Numan's line API directly, or replays held-out calls with the guardian's real verdicts.
Run it from `callguard/` with `uv run --with fastapi --with 'uvicorn[standard]' uvicorn console.server:app --port 8095`;
details in `console/README.md`.

## The NVIDIA thread

- **Nemotron Speech Streaming 0.6B** (sherpa-onnx INT8, on a Mac) hears both sides of the live call: about 37 ms per 160 ms chunk.
- **Nemotron 3 Ultra** wrote the held-out test set, so the test isn't the training generator grading itself.
- **Nemotron 3.5 Super**, fine-tuned on the same data as v1, has the lowest single-model false-alarm rate after v3 on the held-out set
  and catches 100% of BothBosu scams with no false alarms.

## Honest caveats

- **Latency:** guardian and untuned-Qwen latencies in `results/` come from the Training API sampler under shared-pool contention.
  They are not production numbers; see "Deployed".
- **Deployment drift:** the deployed v2 loses most of its caretaker-rule following versus the evaluated checkpoint (see "Deployed").
  Until that's resolved, the live demo's rule-flip behavior may not match the tables.
- **Small n:** the unseen scam type has 2 held-out calls; the 120-call subset has 36 harmless calls. Most guardian-vs-guardian gaps sit
  inside the 95% intervals. The clear wins are against the baselines: false alarms and caretaker-rule following.
- **Same-family risk:** Nemotron 3 Ultra wrote the held-out set, so the Nemotron guardian may get a style advantage there; BothBosu is the cleaner comparison.
- **v3 was designed after seeing held-out false alarms.** Its target scenarios were hard negatives in the taxonomy from the start and
  the same types false-alarm on the dev calls (v2: 16/41), but the held-out set is less blind for v3. BothBosu and dev are the clean checks.
- **Label caveats:** 4 allowed-contact twins also carry a "never share codes" rule that the transcript breaks; we keep the raw number as
  primary and report the exclusion separately. A validator bug labeled some harmless "watch" turns as `custom_rule` in v1/v2 training data (fixed for v3).
- **All test calls are generated or synthetic.** Real role-played calls are the next test.

## Reproduce

```
python3 -I factory/generate.py cards --model deepseek --split train --k 12
python3 -I factory/generate.py calls --model deepseek --split train --n 1600 --workers 20
python3 -I factory/generate.py calls --model nemotron --split test_heldout --n 260 --workers 16
python3 -I guardian/build_sft.py --inputs data/v1_a.jsonl data/v1_b.jsonl --max-per-call 6
.venv/bin/python -I guardian/train_tapi.py --name v1 --lr 1e-4 --batch 32 --rank 16
.venv/bin/python -I eval/run_eval.py --model "tapi:<sampler checkpoint>" --name guardian-v1 --key-env FIREWORKS_API_KEY_MATT
.venv/bin/python -I eval/charts.py && python3 -I eval/build_readme.py
```

Spend on the training account is tracked in `results/ledger-matt.md`. Consent and privacy notes: `PLAN.md` section 10.
"""
    (ROOT / "README.md").write_text(md)
    print("wrote", ROOT / "README.md", len(md.splitlines()), "lines")


if __name__ == "__main__":
    main()

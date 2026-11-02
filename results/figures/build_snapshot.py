"""Transcribe Matt's TRAINING_RUN.md tables (snapshot 14:04 PST, Nov 2) into results/guardian_snapshot.json.

    .venv/bin/python results/figures/build_snapshot.py
Counts are round(pct * n); intervals are Wilson 95% (z = 1.96).
"""
import json, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
Z = 1.96

def wilson(k, n):
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]

def rate(pct, n):
    k = round(pct / 100 * n)
    return {"pct": pct, "n": n, "count": k, "wilson95": wilson(k, n)}

def frac(k, n):
    return {"count": k, "n": n, "pct": round(100 * k / n, 1), "wilson95": wilson(k, n)}

# (id, label, kind, catch%, fa%, flips_k, flips_n, f1, latency_s, latency_source)
SUB = [
    ("B0", "Keyword rules", "rules", 45.9, 41.7, 0, 10, 0.64, 0.00, "local"),
    ("B1", "Rune, Fireworks zero-shot", "baseline", 90.5, 33.3, 2, 10, 0.55, 0.16, "api"),
    ("B3", "Untuned Qwen3.6-35B-A3B", "baseline", 82.4, 55.6, 0, 10, 0.37, 12.24, "training_api_sampler"),
    ("B4", "Kimi K3", "frontier", 85.1, 25.0, 0, 10, 0.78, 1.88, "api"),
    ("B5", "DeepSeek V4.1 Flash", "frontier", 81.1, 33.3, 1, 10, 0.53, 1.30, "api"),
    ("v1", "Guardian v1 (ours)", "ours", 86.5, 16.7, 8, 10, 0.70, 6.26, "training_api_sampler"),
    ("v2", "Guardian v2 (ours)", "ours", 87.8, 25.0, 8, 10, 0.67, 1.37, "training_api_sampler"),
]
ALL = [
    ("B0", "Keyword rules", "rules", 52.0, 49.4, 0, 22, 0.50, 0.00, "local"),
    ("B1", "Rune, Fireworks zero-shot", "baseline", 92.6, 38.0, 5, 22, 0.52, 0.15, "api"),
    ("B3", "Untuned Qwen3.6-35B-A3B", "baseline", 89.2, 53.2, 3, 22, 0.36, 8.12, "training_api_sampler"),
    ("B5", "DeepSeek V4.1 Flash", "frontier", 85.1, 31.6, 3, 22, 0.52, 1.27, "api"),
    ("v1", "Guardian v1 (ours)", "ours", 89.9, 16.5, 19, 22, 0.62, 1.20, "training_api_sampler"),
    ("v2", "Guardian v2 (ours)", "ours", 92.6, 17.7, 19, 22, 0.62, 1.03, "training_api_sampler"),
]
BOSU = [
    ("B0", "Keyword rules", "rules", 43, 0),
    ("v1", "Guardian v1 (ours)", "ours", 95, 0),
    ("v2", "Guardian v2 (ours)", "ours", 98, 0),
]

def rows(table, n_risky, n_benign):
    out = []
    for id_, label, kind, c, fa, fk, fn, f1, lat, src in table:
        out.append({
            "id": id_, "label": label, "kind": kind,
            "catch_before_ask": rate(c, n_risky),
            "false_alarms_benign": rate(fa, n_benign),
            "rule_flip_pairs": frac(fk, fn),
            "category_f1": f1,
            "latency_p50_s": lat, "latency_source": src,
        })
    return out

snap = {
    "source": "Matt, callguard/handoff/TRAINING_RUN.md on origin/matt/guardian-training-handoff, "
              "snapshot 14:04 PST Mon Nov 2 2026 (generated from callguard/results/results.jsonl). "
              "Transcribed verbatim; counts = round(pct*n); intervals = Wilson 95%.",
    "notes": [
        "Guardian = Qwen3.6-35B-A3B + LoRA r16 trained via the Fireworks serverless Training API.",
        "Latency marked training_api_sampler comes from the Training API sampler (queueing), not a deployment.",
        "Kimi K3 was scored on the 120-call common subset only.",
        "Test calls written by Nemotron 3 Ultra (held out); training calls by DeepSeek V4.1 Flash.",
        "BothBosu: LLM-written by a different family, call-level labels; false alarms are 0% for every system.",
    ],
    "subset_120": {"n_calls": 120, "n_risky": 74, "n_benign": 36, "n_flip_pairs": 10,
                    "systems": rows(SUB, 74, 36)},
    "all_257": {"n_calls": 257, "n_risky": 148, "n_benign": 79, "n_flip_pairs": 22,
                 "systems": rows(ALL, 148, 79)},
    "bothbosu": {"n_scam": 100, "n_benign": 100, "systems": [
        {"id": i, "label": l, "kind": k, "scam_recall": rate(r, 100), "false_alarms_benign": rate(f, 100)}
        for i, l, k, r, f in BOSU]},
}
out = ROOT / "results" / "guardian_snapshot.json"
out.write_text(json.dumps(snap, indent=2) + "\n")
print("wrote", out)

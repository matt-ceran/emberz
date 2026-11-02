"""Transcribe Matt's TRAINING_RUN.md tables (refreshed 14:26 PST, Nov 2) into results/guardian_snapshot.json.

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
    ("nemo", "Nemotron 3.5 Super fine-tuned (ours, NVIDIA)", "ours_nvidia", 85.1, 13.9, 7, 10, 0.64, 4.96, "training_api_sampler"),
]
ALL = [
    ("B0", "Keyword rules", "rules", 52.0, 49.4, 0, 22, 0.50, 0.00, "local"),
    ("B1", "Rune, Fireworks zero-shot", "baseline", 92.6, 38.0, 5, 22, 0.52, 0.15, "api"),
    ("B3", "Untuned Qwen3.6-35B-A3B", "baseline", 89.2, 53.2, 3, 22, 0.36, 8.12, "training_api_sampler"),
    ("B5", "DeepSeek V4.1 Flash", "frontier", 85.1, 31.6, 3, 22, 0.52, 1.27, "api"),
    ("v1", "Guardian v1 (ours)", "ours", 89.9, 16.5, 19, 22, 0.62, 1.20, "training_api_sampler"),
    ("v2", "Guardian v2 (ours)", "ours", 92.6, 17.7, 19, 22, 0.62, 1.03, "training_api_sampler"),
    ("nemo", "Nemotron 3.5 Super fine-tuned (ours, NVIDIA)", "ours_nvidia", 89.2, 15.2, 17, 22, 0.55, 4.65, "training_api_sampler"),
]
# Rune gate -> our guardian confirms (post-hoc from cached predictions):
# (id, label, kind, all257 catch%, all257 FA count /79, sub120 catch%, sub120 FA count /36)
CASCADE = [
    ("rune_v1", "Rune -> Guardian v1", "cascade", 87.8, 8, 85.1, 3),
    ("rune_v2", "Rune -> Guardian v2", "cascade", 89.9, 8, 85.1, 4),
    ("rune_nemo", "Rune -> Nemotron fine-tuned", "cascade", 87.8, 9, 83.8, 3),
]
# 49 held-out calls voiced (macOS TTS) -> 8 kHz -> Nemotron Speech Streaming ASR; 16 benign.
# (id, label, kind, clean catch%, clean FA /16, asr catch%, asr FA /16, asr risky flagged at all %)
REAL_SPEECH = [
    ("B1", "Rune, Fireworks zero-shot", "baseline", 89, 3, 89, 2, 97),
    ("B5", "DeepSeek V4.1 Flash", "frontier", 82, 4, 85, 5, 97),
    ("v1", "Guardian v1 (ours)", "ours", 85, 1, 82, 3, 100),
    ("v2", "Guardian v2 (ours)", "ours", 85, 4, 85, 2, 100),
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
              "refreshed 14:26 PST Mon Nov 2 2026 (commit 0c3d1ff; generated from callguard/results/results.jsonl). "
              "Transcribed verbatim; counts = round(pct*n); intervals = Wilson 95%.",
    "notes": [
        "Guardian = Qwen3.6-35B-A3B + LoRA r16 trained via the Fireworks serverless Training API.",
        "Latency marked training_api_sampler comes from the Training API sampler (queueing), not a deployment.",
        "Kimi K3 was scored on the 120-call common subset only.",
        "Test calls written by Nemotron 3 Ultra (held out); training calls by DeepSeek V4.1 Flash.",
        "BothBosu: LLM-written by a different family, call-level labels; false alarms are 0% for every system.",
        "Nemotron 3.5 Super fine-tune: same family as the test-set writer (Nemotron 3 Ultra) - caveat.",
        "Cascade numbers are post-hoc from cached predictions. Real-speech catch % has no stated n (33 risky calls inferred, not stated); no CI computed.",
    ],
    "subset_120": {"n_calls": 120, "n_risky": 74, "n_benign": 36, "n_flip_pairs": 10,
                    "systems": rows(SUB, 74, 36)},
    "all_257": {"n_calls": 257, "n_risky": 148, "n_benign": 79, "n_flip_pairs": 22,
                 "systems": rows(ALL, 148, 79)},
    "cascade": {"note": "Rune as a 0.15 s gate, our fine-tuned guardian confirms before alerting (post-hoc)",
                "systems": [{"id": i, "label": l, "kind": k,
                             "all_257": {"catch_before_ask": rate(c1, 148), "false_alarms_benign": frac(f1, 79)},
                             "subset_120": {"catch_before_ask": rate(c2, 74), "false_alarms_benign": frac(f2, 36)}}
                            for i, l, k, c1, f1, c2, f2 in CASCADE]},
    "real_speech_49": {"note": "49 held-out calls voiced with macOS TTS -> 8 kHz -> Nemotron Speech Streaming ASR; median WER 16% (mostly numbers-as-words); 16 benign",
                       "n_calls": 49, "n_benign": 16, "median_wer": 0.16,
                       "systems": [{"id": i, "label": l, "kind": k,
                                    "clean": {"catch_before_ask_pct": c1, "false_alarms": frac(f1, 16)},
                                    "asr": {"catch_before_ask_pct": c2, "false_alarms": frac(f2, 16)},
                                    "asr_risky_flagged_at_all_pct": a}
                                   for i, l, k, c1, f1, c2, f2, a in REAL_SPEECH]},
    "bothbosu": {"n_scam": 100, "n_benign": 100, "systems": [
        {"id": i, "label": l, "kind": k, "scam_recall": rate(r, 100), "false_alarms_benign": rate(f, 100)}
        for i, l, k, r, f in BOSU]},
}
out = ROOT / "results" / "guardian_snapshot.json"
out.write_text(json.dumps(snap, indent=2) + "\n")
print("wrote", out)

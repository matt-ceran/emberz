# Results (heldout120)

| Model | catch_before_ask | catch_at_or_before_ask | recall_any | false_alarm_rate | rule_flip_accuracy | unseen_type_recall | lead_time_turns_median | category_macro_f1_at_first_alert | evidence_exact_match | latency_ms_p50 | latency_ms_p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Guardian nemo-v1 (Nemotron 3.5 Super LoRA) | 0.85 [0.75-0.92] n=74 | 0.96 [0.89-0.99] n=74 | 1.00 [0.96-1.00] n=84 | 0.14 [0.06-0.29] n=36 | 0.70 [0.40-0.89] n=10 | - | 4.0 (n=74) | 0.642 (n=17) | 1.0 (n=426) | 4958.0 (n=820) | 8359 (n=820) |
| Guardian v1 (Qwen3.6-35B LoRA) | 0.86 [0.77-0.93] n=74 | 0.99 [0.93-1.00] n=74 | 1.00 [0.96-1.00] n=84 | 0.17 [0.08-0.32] n=36 | 0.80 [0.49-0.94] n=10 | - | 4.0 (n=74) | 0.702 (n=16) | 0.993 (n=435) | 6264.5 (n=792) | 19343 (n=792) |
| Guardian v2 (Qwen3.6-35B LoRA, deployed) | 0.88 [0.79-0.94] n=74 | 0.99 [0.93-1.00] n=74 | 1.00 [0.96-1.00] n=84 | 0.25 [0.14-0.41] n=36 | 0.80 [0.49-0.94] n=10 | - | 4.0 (n=74) | 0.674 (n=16) | 1.0 (n=356) | 1366.5 (n=760) | 11883 (n=760) |
| Guardian v3 (Qwen3.6-35B LoRA + contrast pairs) | 0.82 [0.72-0.89] n=74 | 0.97 [0.91-0.99] n=74 | 1.00 [0.96-1.00] n=84 | 0.14 [0.06-0.29] n=36 | 0.70 [0.40-0.89] n=10 | - | 4.0 (n=74) | 0.626 (n=17) | 0.997 (n=399) | 7824.0 (n=846) | 22613 (n=846) |
| Keywords (B0) | 0.46 [0.35-0.57] n=74 | 0.68 [0.56-0.77] n=74 | 0.77 [0.67-0.85] n=84 | 0.42 [0.27-0.58] n=36 | 0.00 [-0.00-0.28] n=10 | - | 2 (n=57) | 0.639 (n=14) | 1.0 (n=79) | 0.0 (n=1036) | 0 (n=1036) |
| Fireworks Rune zero-shot (B1) | 0.91 [0.82-0.95] n=74 | 0.97 [0.91-0.99] n=74 | 1.00 [0.96-1.00] n=84 | 0.33 [0.20-0.50] n=36 | 0.20 [0.06-0.51] n=10 | - | 6.0 (n=74) | 0.551 (n=15) | - | 155.5 (n=672) | 264 (n=672) |
| Qwen3.6-35B untuned (B3) | 0.82 [0.72-0.89] n=74 | 0.99 [0.93-1.00] n=74 | 0.96 [0.90-0.99] n=84 | 0.56 [0.40-0.70] n=36 | 0.00 [-0.00-0.28] n=10 | - | 4.5 (n=74) | 0.369 (n=16) | 0.956 (n=458) | 12239.5 (n=686) | 26292 (n=686) |
| Kimi K3 zero-shot (B4) | 0.85 [0.75-0.92] n=74 | 0.99 [0.93-1.00] n=74 | 0.95 [0.88-0.98] n=84 | 0.25 [0.14-0.41] n=36 | 0.00 [-0.00-0.28] n=10 | - | 4.0 (n=74) | 0.778 (n=14) | 0.976 (n=626) | 1882 (n=797) | 3898 (n=797) |
| DeepSeek V4.1 Flash zero-shot (B5) | 0.81 [0.71-0.88] n=74 | 0.99 [0.93-1.00] n=74 | 0.98 [0.92-0.99] n=84 | 0.33 [0.20-0.50] n=36 | 0.10 [0.02-0.40] n=10 | - | 4.0 (n=74) | 0.531 (n=16) | 0.992 (n=858) | 1305 (n=833) | 1955 (n=833) |

Latency for the guardian and the untuned Qwen (B3) is from the Fireworks Training API sampler under shared-pool contention: **not a production latency**. The serverless baselines' latency is from the normal inference API. The production latency claim waits on a dedicated deployment of the guardian.
Unseen scam type (gold courier pickup) has n=2 held-out calls: too small to claim anything.
Same-family caveat: the held-out set was written by Nemotron 3 Ultra, so the fine-tuned Nemotron 3.5 Super (nemo-v1) may get a style advantage on it; compare it on BothBosu too.
BothBosu: synthetic out-of-distribution conversations with call-level labels only; catch-before-ask does not apply.

# Results (bothbosu)

| Model | catch_before_ask | catch_at_or_before_ask | recall_any | false_alarm_rate | rule_flip_accuracy | unseen_type_recall | lead_time_turns_median | category_macro_f1_at_first_alert | evidence_exact_match | latency_ms_p50 | latency_ms_p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Guardian nemo-v1 (Nemotron 3.5 Super LoRA) | - | - | 1.00 [0.96-1.00] n=100 | 0.00 [0.00-0.04] n=100 | - | - | - | 0.0 (n=1) | 1.0 (n=584) | 2986.0 (n=1620) | 5228 (n=1620) |
| Guardian v1 (Qwen3.6-35B LoRA) | - | - | 0.95 [0.89-0.98] n=100 | 0.00 [0.00-0.04] n=100 | - | - | - | 0.0 (n=1) | 1.0 (n=601) | 1065 (n=1685) | 7422 (n=1685) |
| Guardian v2 (Qwen3.6-35B LoRA, deployed) | - | - | 0.98 [0.93-0.99] n=100 | 0.00 [0.00-0.04] n=100 | - | - | - | 0.0 (n=1) | 1.0 (n=499) | 833.0 (n=1640) | 1218 (n=1640) |
| Guardian v3 (Qwen3.6-35B LoRA + contrast pairs) | - | - | 0.97 [0.92-0.99] n=100 | 0.00 [0.00-0.04] n=100 | - | - | - | 0.037 (n=2) | 1.0 (n=493) | 1005 (n=1733) | 1537 (n=1733) |
| Keywords (B0) | - | - | 0.43 [0.34-0.53] n=100 | 0.00 [0.00-0.04] n=100 | - | - | - | 0.0 (n=1) | 1.0 (n=43) | 0.0 (n=1800) | 0 (n=1800) |

Latency for the guardian and the untuned Qwen (B3) is from the Fireworks Training API sampler under shared-pool contention: **not a production latency**. The serverless baselines' latency is from the normal inference API. The production latency claim waits on a dedicated deployment of the guardian.
Unseen scam type (gold courier pickup) has n=2 held-out calls: too small to claim anything.
Same-family caveat: the held-out set was written by Nemotron 3 Ultra, so the fine-tuned Nemotron 3.5 Super (nemo-v1) may get a style advantage on it; compare it on BothBosu too.
BothBosu: synthetic out-of-distribution conversations with call-level labels only; catch-before-ask does not apply.

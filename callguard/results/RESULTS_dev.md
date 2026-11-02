# Results (dev)

| Model | catch_before_ask | catch_at_or_before_ask | recall_any | false_alarm_rate | rule_flip_accuracy | unseen_type_recall | lead_time_turns_median | category_macro_f1_at_first_alert | evidence_exact_match | latency_ms_p50 | latency_ms_p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Guardian v2 (Qwen3.6-35B LoRA, deployed) | 0.99 [0.94-1.00] n=90 | 1.00 [0.96-1.00] n=90 | 1.00 [0.96-1.00] n=102 | 0.39 [0.26-0.54] n=41 | 0.89 [0.56-0.98] n=9 | - | 5.5 (n=90) | 0.768 (n=17) | 0.998 (n=433) | 1055 (n=833) | 4105 (n=833) |
| Guardian v3 (Qwen3.6-35B LoRA + contrast pairs) | 0.96 [0.89-0.98] n=90 | 1.00 [0.96-1.00] n=90 | 1.00 [0.96-1.00] n=102 | 0.32 [0.20-0.47] n=41 | 0.89 [0.56-0.98] n=9 | - | 4.0 (n=90) | 0.819 (n=18) | 1.0 (n=493) | 3978.5 (n=942) | 13548 (n=942) |

Latency for the guardian and the untuned Qwen (B3) is from the Fireworks Training API sampler under shared-pool contention: **not a production latency**. The serverless baselines' latency is from the normal inference API. The production latency claim waits on a dedicated deployment of the guardian.
Unseen scam type (gold courier pickup) has n=2 held-out calls: too small to claim anything.
Same-family caveat: the held-out set was written by Nemotron 3 Ultra, so the fine-tuned Nemotron 3.5 Super (nemo-v1) may get a style advantage on it; compare it on BothBosu too.
BothBosu: synthetic out-of-distribution conversations with call-level labels only; catch-before-ask does not apply.

## False alarms by harmless scenario (dev)

| Harmless scenario | n | Guardian v2 (Qwen3.6-35B LoRA, deployed) | Guardian v3 (Qwen3.6-35B LoRA + contrast pairs) |
|---|---|---|---|
| doctor_office_dob | 1 | 1 | 0 |
| friend_mom_playdate | 6 | 6 | 6 |
| isp_callback_real | 1 | 1 | 0 |
| real_bank_alert | 3 | 3 | 3 |
| relative_asks_door_code (twin) | 1 | 1 | 1 |
| school_office | 3 | 1 | 0 |
| telemarketer_legit | 3 | 3 | 3 |
| **All harmless calls (primary)** | | 16/41 | 13/41 |
| Excluding 3 conflicted twins* | | 15/38 | 12/38 |

*Allowed-contact twins whose own rules also say never to share codes/passwords/PINs; the twin generator labeled them harmless even though the transcript breaks that rule, so an alert there is defensible. Fixed for v3's new data.

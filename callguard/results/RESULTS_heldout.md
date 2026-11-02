# Results (heldout)

| Model | catch_before_ask | catch_at_or_before_ask | recall_any | false_alarm_rate | rule_flip_accuracy | unseen_type_recall | lead_time_turns_median | category_macro_f1_at_first_alert | evidence_exact_match | latency_ms_p50 | latency_ms_p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Guardian nemo-v1 (Nemotron 3.5 Super LoRA) | 0.89 [0.83-0.93] n=148 | 0.97 [0.93-0.99] n=148 | 1.00 [0.98-1.00] n=178 | 0.15 [0.09-0.25] n=79 | 0.77 [0.57-0.90] n=22 | 1.00 [0.34-1.00] n=2 | 4.0 (n=148) | 0.554 (n=19) | 1.0 (n=872) | 4650 (n=1813) | 6854 (n=1813) |
| Guardian v1 (Qwen3.6-35B LoRA) | 0.90 [0.84-0.94] n=148 | 0.99 [0.96-1.00] n=148 | 1.00 [0.98-1.00] n=178 | 0.17 [0.10-0.26] n=79 | 0.86 [0.67-0.95] n=22 | 1.00 [0.34-1.00] n=2 | 4.0 (n=148) | 0.616 (n=18) | 0.992 (n=893) | 1195 (n=1745) | 16046 (n=1745) |
| Guardian v2 (Qwen3.6-35B LoRA, deployed) | 0.93 [0.87-0.96] n=148 | 0.99 [0.95-1.00] n=148 | 1.00 [0.98-1.00] n=178 | 0.18 [0.11-0.28] n=79 | 0.86 [0.67-0.95] n=22 | 1.00 [0.34-1.00] n=2 | 4.5 (n=148) | 0.624 (n=18) | 0.996 (n=742) | 1034.5 (n=1722) | 10237 (n=1722) |
| Guardian v3 (Qwen3.6-35B LoRA + contrast pairs) | 0.86 [0.80-0.91] n=148 | 0.98 [0.94-0.99] n=148 | 1.00 [0.98-1.00] n=178 | 0.13 [0.07-0.22] n=79 | 0.82 [0.61-0.93] n=22 | 1.00 [0.34-1.00] n=2 | 4.0 (n=148) | 0.656 (n=19) | 0.994 (n=823) | 2163 (n=1883) | 18267 (n=1883) |
| Keywords (B0) | 0.52 [0.44-0.60] n=148 | 0.73 [0.65-0.80] n=148 | 0.81 [0.75-0.86] n=178 | 0.49 [0.39-0.60] n=79 | 0.00 [0.00-0.15] n=22 | 1.00 [0.34-1.00] n=2 | 2.0 (n=122) | 0.504 (n=17) | 1.0 (n=183) | 0.0 (n=2254) | 0 (n=2254) |
| Fireworks Rune zero-shot (B1) | 0.93 [0.87-0.96] n=148 | 0.99 [0.95-1.00] n=148 | 0.99 [0.96-1.00] n=178 | 0.38 [0.28-0.49] n=79 | 0.23 [0.10-0.43] n=22 | 1.00 [0.34-1.00] n=2 | 6.0 (n=148) | 0.517 (n=17) | - | 147.0 (n=1502) | 234 (n=1502) |
| Qwen3.6-35B untuned (B3) | 0.89 [0.83-0.93] n=148 | 0.99 [0.96-1.00] n=148 | 0.97 [0.94-0.99] n=178 | 0.53 [0.42-0.64] n=79 | 0.14 [0.05-0.33] n=22 | 1.00 [0.34-1.00] n=2 | 5.0 (n=148) | 0.357 (n=18) | 0.946 (n=892) | 8123.0 (n=1488) | 22821 (n=1488) |
| Kimi K3 zero-shot (B4) | 0.90 [0.84-0.94] n=148 | 0.99 [0.96-1.00] n=148 | 0.95 [0.91-0.97] n=178 | 0.23 [0.15-0.33] n=79 | 0.18 [0.07-0.39] n=22 | 1.00 [0.34-1.00] n=2 | 4.0 (n=148) | 0.673 (n=17) | 0.965 (n=1329) | 1973 (n=1803) | 4037 (n=1803) |
| DeepSeek V4.1 Flash zero-shot (B5) | 0.85 [0.79-0.90] n=148 | 0.99 [0.96-1.00] n=148 | 0.97 [0.93-0.98] n=178 | 0.32 [0.22-0.42] n=79 | 0.14 [0.05-0.33] n=22 | 1.00 [0.34-1.00] n=2 | 4.0 (n=148) | 0.517 (n=18) | 0.994 (n=1784) | 1269 (n=1859) | 1935 (n=1859) |

Latency for the guardian and the untuned Qwen (B3) is from the Fireworks Training API sampler under shared-pool contention: **not a production latency**. The serverless baselines' latency is from the normal inference API. The production latency claim waits on a dedicated deployment of the guardian.
Unseen scam type (gold courier pickup) has n=2 held-out calls: too small to claim anything.
Same-family caveat: the held-out set was written by Nemotron 3 Ultra, so the fine-tuned Nemotron 3.5 Super (nemo-v1) may get a style advantage on it; compare it on BothBosu too.
BothBosu: synthetic out-of-distribution conversations with call-level labels only; catch-before-ask does not apply.

## False alarms by harmless scenario (heldout)

| Harmless scenario | n | Guardian v1 (Qwen3.6-35B LoRA) | Guardian v2 (Qwen3.6-35B LoRA, deployed) | Guardian v3 (Qwen3.6-35B LoRA + contrast pairs) | Guardian nemo-v1 (Nemotron 3.5 Super LoRA) |
|---|---|---|---|---|---|
| dentist_bot | 6 | 0 | 2 | 2 | 0 |
| friend_mom_playdate | 2 | 2 | 1 | 0 | 1 |
| hyperbole_teen | 4 | 1 | 0 | 0 | 0 |
| isp_callback_real | 5 | 1 | 1 | 0 | 0 |
| pharmacy_bot | 5 | 1 | 1 | 0 | 0 |
| real_bank_alert | 5 | 5 | 5 | 4 | 5 |
| son_asks_bank_login (twin) | 5 | 3 | 3 | 4 | 5 |
| telemarketer_legit | 2 | 0 | 1 | 0 | 1 |
| **All harmless calls (primary)** | | 13/79 | 14/79 | 10/79 | 12/79 |
| Excluding 4 conflicted twins* | | 12/75 | 13/75 | 9/75 | 10/75 |

*Allowed-contact twins whose own rules also say never to share codes/passwords/PINs; the twin generator labeled them harmless even though the transcript breaks that rule, so an alert there is defensible. Fixed for v3's new data.

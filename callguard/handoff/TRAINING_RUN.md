# Guardian training run: status and results (Matt)

_Snapshot 14:04 PST, Mon Nov 2. Numbers are generated from `callguard/results/results.jsonl` (not committed here; see the data-session RESULTS files for live copies)._

## What was trained
- **Base:** `accounts/fireworks/models/qwen3p6-35b-a3b` (Qwen3.6-35B-A3B MoE, ~3B active), LoRA rank 16, trained on Fireworks through the **serverless Training API** (Tinker-style client loop).
  - Managed SFT is blocked on both accounts. Dataset upload returns code 9: "dataset uploads are temporarily disabled, please contact Fireworks AI". Billing is fixed; deployments validate.
- **Data:** DeepSeek V4.1 Flash wrote 1,440 synthetic training calls from the scenario/plan grid (labels come from the plan). Nemotron 3 Ultra wrote the 257 held-out test calls (different family). Prompt = `guardian/render.py` (prompt_sha 806908cfba7d), 8-turn window.
- **v1:** 733 calls → 3,930 windowed samples (2.69M tok), 1 epoch, lr 1e-4, batch 32, 123 steps, ~7.8 min, ~$5.76. Loss 0.65 → ~0.25.
- **v2:** resumed from v1, the 707 unused calls + "none" oversampled to 35% (4,742 samples, 3.22M tok), lr 5e-5, 149 steps. Dev loss 0.269 → 0.246 (no overfitting signal).
- **nemo-v1** (running): NVIDIA Nemotron 3.5 Super 120B-A12B on the same data as v1. Caveat: same family as the test-set writer.
- **v3** (starting): targeted contrast pairs for the look-alikes that cause false alarms (real bank alert vs fake, son-asks-login vs scam, reminder bots, playdate), plus numbers-as-words on 30% of new samples to match Nemotron ASR output. Watch-on-opener / alert-on-ask policy.
- Checkpoint promoted to a deployable model: `accounts/afceran33-zaqel7a5ap/models/emberz-guardian-v1` (HF PEFT addon, r16). Dedicated deployment **validated only, not created** (1x B200 FP4 minimal shape, ~$13/h), pending Matt's go.

## Results: same 120 held-out calls for every model (Kimi scored on these only)
| System | Catch before ask | False alarms (benign) | Rule-flip pairs | Category F1 | Latency p50 |
|---|---|---|---|---|---|
| Keyword rules (B0) | 45.9% (n=74) | 41.7% (n=36) | 0/10 | 0.64 | 0.00 s |
| Rune, Fireworks zero-shot (B1) | 90.5% (n=74) | 33.3% (n=36) | 2/10 | 0.55 | 0.16 s |
| Untuned Qwen3.6-35B-A3B (B3) | 82.4% (n=74) | 55.6% (n=36) | 0/10 | 0.37 | 12.24 s (sampler) |
| Kimi K3 frontier (B4) | 85.1% (n=74) | 25.0% (n=36) | 0/10 | 0.78 | 1.88 s |
| DeepSeek V4.1 Flash frontier (B5) | 81.1% (n=74) | 33.3% (n=36) | 1/10 | 0.53 | 1.30 s |
| **Guardian v1 (ours)** | 86.5% (n=74) | 16.7% (n=36) | 8/10 | 0.70 | 6.26 s (sampler) |
| **Guardian v2 (ours)** | 87.8% (n=74) | 25.0% (n=36) | 8/10 | 0.67 | 1.37 s (sampler) |

## Results: all 257 held-out calls
| System | Catch before ask | False alarms (benign) | Rule-flip pairs | Category F1 | Latency p50 |
|---|---|---|---|---|---|
| Keyword rules (B0) | 52.0% (n=148) | 49.4% (n=79) | 0/22 | 0.50 | 0.00 s |
| Rune, Fireworks zero-shot (B1) | 92.6% (n=148) | 38.0% (n=79) | 5/22 | 0.52 | 0.15 s |
| Untuned Qwen3.6-35B-A3B (B3) | 89.2% (n=148) | 53.2% (n=79) | 3/22 | 0.36 | 8.12 s (sampler) |
| DeepSeek V4.1 Flash frontier (B5) | 85.1% (n=148) | 31.6% (n=79) | 3/22 | 0.52 | 1.27 s |
| **Guardian v1 (ours)** | 89.9% (n=148) | 16.5% (n=79) | 19/22 | 0.62 | 1.20 s (sampler) |
| **Guardian v2 (ours)** | 92.6% (n=148) | 17.7% (n=79) | 19/22 | 0.62 | 1.03 s (sampler) |

## Out-of-distribution: BothBosu (LLM-written by a different family; call-level labels)
| System | Scam recall (n=100) | False alarms (n=100 benign) |
|---|---|---|
| Keyword rules (B0) | 43% | 0% |
| **Guardian v1 (ours)** | 95% | 0% |
| **Guardian v2 (ours)** | 98% | 0% |

## Read
- Fine-tuning vs the same base untuned: false alarms 53% → 16.5%, rule-flip pairs 3/22 → 19/22, catch-before-ask ~equal or better.
- Against frontier models (120 subset): v1 has fewer false alarms than Kimi K3 (16.7% vs 25%) and DeepSeek (33%), and catches as many before the ask (86.5% vs 85.1% / 81.1%). Kimi is better on category F1.
- Rune catches slightly more but false-alarms ~2x.
- v1 ≈ v2 (differences inside the CIs). v2 has more catches and 98% BothBosu recall; v1 has fewer false alarms on the subset.
- False alarms cluster in look-alikes: real_bank_alert 5/5 (flags the "calling about a charge" opener), son_asks_bank_login 3/5. **Gold-label flaw:** 2 of those 5 twins carry "never share codes" and the elder reads one out, so alerting is defensible there. v3 targets exactly this.
- **Latency is NOT production yet.** Guardian/B3 latency comes from the Training API sampler (queueing). The real number needs the dedicated deployment.

## For Numan
1. **Noise pass (G1):** `callguard/handoff/noise_input.jsonl` has 399 TRAIN-split calls in the exact `dataset_row` shape (~13 h of speech at 150 wpm). Self-harm calls are excluded pending the human safety read. Output back to `data/noisy_turns.jsonl` (`NoisyTurn`) as planned. Time is short, so even a ~100-call partial helps.
2. **Guardian for the live line:** the request/response contract is `callguard/guardian/render.py` (`render_messages`, `parse_response`). The model string will be `accounts/afceran33-zaqel7a5ap/models/emberz-guardian-v1#accounts/afceran33-zaqel7a5ap/deployments/emberz-v1` once the deployment is created (or the v2/v3 winner).
3. **Console:** being built in `callguard/console/` (POST `/events` takes your `console_event` schema unchanged, WS `/ws/console`, POST `/actions/{call_id}/{listen|join|hangup|safe}`), ETA ~14:45.
4. **Deadline:** submissions close **4:30 PM today** (luma.com/fireworks-8kmc). The live line has to work by ~3:30; otherwise the demo runs on console replay.

# Guardian training run: status and results (Matt)

_Refreshed 14:26 PST, Mon Nov 2, from `callguard/results/results.jsonl`. Earlier snapshot: 14:04._

## What was trained (all on Fireworks, serverless Training API; managed SFT uploads are blocked with code 9)
- **v1**: Qwen3.6-35B-A3B LoRA r16, 733 DeepSeek-written calls (3,930 samples), 1 epoch, ~$5.76.
- **v2**: resumed from v1 with 707 unseen calls + "none" oversampled to 35%. Dev loss 0.269 -> 0.246.
- **nemo-v1**: NVIDIA Nemotron 3.5 Super 120B-A12B, same data as v1. Caveat: same family as the test-set writer (Nemotron 3 Ultra).
- **v3** (training/scoring now): from v2, 97 contrast pairs (benign look-alike vs scam twin), watch-on-opener / alert-on-ask, numbers as spoken words.
- Promoted model: `accounts/afceran33-zaqel7a5ap/models/emberz-guardian-v1` (the winner gets promoted at ~14:35). The dedicated deployment is validated but not created yet.

## Same 120 held-out calls for every model (Nemotron-3-Ultra-written)
| System | Catch before ask | False alarms (benign) | Rule-flip pairs | Category F1 | Latency p50 |
|---|---|---|---|---|---|
| Keyword rules (B0) | 45.9% (n=74) | 41.7% (n=36) | 0/10 | 0.64 | 0.00 s |
| Rune, Fireworks zero-shot (B1) | 90.5% (n=74) | 33.3% (n=36) | 2/10 | 0.55 | 0.16 s |
| Untuned Qwen3.6-35B-A3B (B3) | 82.4% (n=74) | 55.6% (n=36) | 0/10 | 0.37 | 12.24 s (sampler) |
| Kimi K3, frontier (B4) | 85.1% (n=74) | 25.0% (n=36) | 0/10 | 0.78 | 1.88 s |
| DeepSeek V4.1 Flash, frontier (B5) | 81.1% (n=74) | 33.3% (n=36) | 1/10 | 0.53 | 1.30 s |
| **Ours: Qwen3.6-35B-A3B v1** | 86.5% (n=74) | 16.7% (n=36) | 8/10 | 0.70 | 6.26 s (sampler) |
| **Ours: Qwen3.6-35B-A3B v2** | 87.8% (n=74) | 25.0% (n=36) | 8/10 | 0.67 | 1.37 s (sampler) |
| **Ours: Nemotron 3.5 Super (NVIDIA) fine-tuned** | 85.1% (n=74) | 13.9% (n=36) | 7/10 | 0.64 | 4.96 s (sampler) |

## All 257 held-out calls
| System | Catch before ask | False alarms (benign) | Rule-flip pairs | Category F1 | Latency p50 |
|---|---|---|---|---|---|
| Keyword rules (B0) | 52.0% (n=148) | 49.4% (n=79) | 0/22 | 0.50 | 0.00 s |
| Rune, Fireworks zero-shot (B1) | 92.6% (n=148) | 38.0% (n=79) | 5/22 | 0.52 | 0.15 s |
| Untuned Qwen3.6-35B-A3B (B3) | 89.2% (n=148) | 53.2% (n=79) | 3/22 | 0.36 | 8.12 s (sampler) |
| DeepSeek V4.1 Flash, frontier (B5) | 85.1% (n=148) | 31.6% (n=79) | 3/22 | 0.52 | 1.27 s |
| **Ours: Qwen3.6-35B-A3B v1** | 89.9% (n=148) | 16.5% (n=79) | 19/22 | 0.62 | 1.20 s (sampler) |
| **Ours: Qwen3.6-35B-A3B v2** | 92.6% (n=148) | 17.7% (n=79) | 19/22 | 0.62 | 1.03 s (sampler) |
| **Ours: Nemotron 3.5 Super (NVIDIA) fine-tuned** | 89.2% (n=148) | 15.2% (n=79) | 17/22 | 0.55 | 4.65 s (sampler) |

## Routing: Rune as a 0.15 s gate, our fine-tuned guardian confirms before alerting the caretaker (post-hoc from cached predictions)
| Cascade (Rune gate -> our guardian confirms) | all 257: catch / false alarms | same 120: catch / false alarms |
|---|---|---|
| Rune -> **Ours: Qwen3.6-35B-A3B v1** | 87.8% / 8/79 | 85.1% / 3/36 |
| Rune -> **Ours: Qwen3.6-35B-A3B v2** | 89.9% / 8/79 | 85.1% / 4/36 |
| Rune -> **Ours: Nemotron 3.5 Super (NVIDIA) fine-tuned** | 87.8% / 9/79 | 83.8% / 3/36 |

## Out-of-distribution: BothBosu (LLM-written by a different family; call-level labels)
| System | Scam recall (n=100) | False alarms (n=100 benign) |
|---|---|---|
| Keyword rules (B0) | 43% | 0% |
| **Ours: Qwen3.6-35B-A3B v1** | 95% | 0% |
| **Ours: Qwen3.6-35B-A3B v2** | 98% | 0% |

## Real speech: 49 held-out calls voiced (macOS TTS) -> 8 kHz -> Nemotron Speech Streaming ASR (median WER 16%, mostly numbers-as-words)
| System | Clean: catch / false alarms | **Nemotron-ASR text: catch / false alarms** | ASR: risky flagged at all |
|---|---|---|---|
| Rune, Fireworks zero-shot (B1) | 89% / 3/16 | 89% / 2/16 | 97% |
| DeepSeek V4.1 Flash, frontier (B5) | 82% / 4/16 | 85% / 5/16 | 97% |
| **Ours: Qwen3.6-35B-A3B v1** | 85% / 1/16 | 82% / 3/16 | 100% |
| **Ours: Qwen3.6-35B-A3B v2** | 85% / 4/16 | 85% / 2/16 | 100% |

## Read
- **Fine-tuning works.** Vs the same base untuned: false alarms 53% -> ~15-18%, rule-flip pairs 3/22 -> 17-19/22, catch-before-ask equal or better.
- **Beats the frontier models on what matters.** Fewer false alarms than Kimi K3 (25%) and DeepSeek (33%) on the same calls, and catches as many before the ask. Rule-following is unique to the tuned models. Kimi is better on category F1.
- **Routing makes it best overall.** A Rune gate plus our guardian confirming gives ~10% false alarms at Kimi-level catch.
- **Robust to real ASR output.** Every risky call is still flagged on Nemotron transcripts. Differences between models on 16 benign calls are noise.
- **Caveats.** Guardian latency is Training API sampler latency, not production. Small n on benign calls and rule pairs. 2 of 5 `son_asks_bank_login` test twins have a rule-conflict label flaw (alerting is defensible). v3 was designed after seeing held-out false-alarm types (also confirmed on dev).

## For Numan
- **Adapter PR #4** (into `numan/line-voice-gcp`) plugs the trained guardian into the line: frozen prompt bridge plus 3 additive contract values. All your tests pass.
- Noise pass input: `callguard/handoff/noise_input.jsonl` (399 train calls, `dataset_row`). Real Nemotron transcripts for 49 test calls exist in `NoisyTurn` format (local, `data/noisy_turns_test50.jsonl`).
- Deadline: submissions close **4:30 PM**.

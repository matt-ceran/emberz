# Ledger: Matt's Fireworks account (afceran33-zaqel7a5ap)

**Actual, from the billing page at ~14:44: $32.71 spent, $18.29 prepaid credit left.** The rows below are token-based estimates and ran high.

Estimates from token counts; check https://fireworks.ai/billing for the real numbers. Deployment emberz-v2b is initializing since ~14:40 (~$13/h while up).

| When | What | Volume | Est. $ | Note |
|---|---|---|---|---|
| 12:57-13:09 | DeepSeek V4.1 Flash generation, ~1,048 training calls (your key) | 1.48M prompt + 2.23M output tok | 3.12 | serverless price $0.30 in / $1.20 out per 1M (docs.fireworks.ai/serverless/pricing, verified by the other session) |
| 13:20 | Training API probe, qwen3p6-35b-a3b (3 samples) | 1,869 train tok | 0.01 |  |
| 13:24-13:32 | SFT v1, qwen3p6-35b-a3b, 1 epoch, rank 16 | 2.69M train tok x $2.14 | 5.76 |  |
| 13:22-13:36 | B3 eval, untuned qwen3p6-35b-a3b sampler, 257 calls | ~1,488 requests | 0.90 | estimate: $0.45/M prefill, $1.335/M sample |
| 13:32-13:40 | guardian-v1 eval, sampler, 257 calls | ~1,745 requests | 1.00 | estimate |
| 13:16 | smoke eval tapi-base (10 calls) | 71 requests | 0.05 | estimate |
| 13:46 | Training API probe, Nemotron 3.5 Super (3 samples) | 2,003 train tok | 0.01 |  |
| 13:46- | SFT v2, qwen3p6-35b-a3b from v1-t123, 1 epoch, lr 5e-5 | 3.22M train tok x $2.14 | 6.90 | running |
| 13:47- | SFT nemo-v1, Nemotron 3.5 Super 120B-A12B, 1 epoch | 2.87M train tok x $2.32 | 6.66 | running |
| 13:55-14:00 | guardian-v2 evals: heldout 257 + bothbosu 200 (sampler) | ~3,360 requests | 1.60 | estimate |
| 13:52-13:57 | guardian-v1 eval on bothbosu 200 (sampler) | ~1,800 requests | 0.80 | estimate |
| 14:02-14:04 | guardian-v2 eval on dev calls 143 (sampler) | ~1,100 requests | 0.50 | estimate |
| 14:05-14:07 | DeepSeek contrast pairs for v3 (your key) | 0.18M prompt + 0.42M output tok | 0.56 | $0.30/$1.20 per 1M |
| 14:07- | SFT v3, qwen3p6-35b-a3b from v2-t149, 1 epoch, lr 5e-5 | 1.89M train tok x $2.14 | 4.04 | running |
| next | evals: nemo-v1 + v3 (heldout 257 + bothbosu 200 each, sampler) |  | 3.50 | estimate |
| 14:31-14:35 | Deployment emberz-v2 (1x B200 FP4) FAILED with INTERNAL during init, deleted | - | 0.00 | |
| ~14:40- | Deployment emberz-v2b (guardian v2, 1x B200 BF16, min 1), created by the other session with Matt's go | ~$13/h | 13.00 | 1 h assumed; scaled to 0 (15-min window) right after the bench |

**Estimated total incl. running jobs, the next evals and 1 h of deployment: $48.41.** Starting credit was $50.

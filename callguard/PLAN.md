# Emberz / CallGuard: merged plan (deadline 4:30 PM, Mon 2026-11-02)

**Sources.** This merges Numan's plan (`origin/plan/second-ear`, PLAN.md at 9779fb9) with our research. Every number marked *measured* was produced on this M5 Pro or against the live Fireworks API today.

**Event.** Fireworks: Deploy Buildathon (Microsoft & NVIDIA).
- Build started at 10:30. **Submissions close at 4:30 PM.** Demos and awards are at 7 PM.
- The brief: deploy a model for a task, build an eval, measure quality, latency and cost tradeoffs, tune at least one of them against a baseline, and show business value.

> Status: **DRAFT, pending the 5 decisions at the bottom.** Nothing has been generated, trained or deployed yet.

## 1. What we build (Numan's product framing, unchanged)

A caretaker gives a kid, grandma or themselves a **guarded Telnyx number**. Calls from unknown numbers are transcribed live, and a small model fine-tuned on Fireworks scores every turn:

- **Normal calls:** the model stays silent.
- **Risky calls** (scam, grooming tactics, giving away personal info, threats):
  1. The protected person hears a private coaching line that the caller cannot hear.
  2. The caretaker gets an alert call and can listen in or barge in.
  3. The console shows the transcript and the risk level.

The headline claim: **"caught before the ask."**

## 2. Model stack (decided unless flagged)

| Stage | Model | Runs on | Trained? | Measured |
|---|---|---|---|---|
| Speech to text | **NVIDIA Nemotron Speech Streaming 0.6B**, INT8, 160 ms chunks, via sherpa-onnx | Local Mac | Off the shelf today (see §2a) | Both sides batched: **34 ms compute per 160 ms chunk** (4 threads). WER 0-2% on 8 kHz audio. Every scam keyword was correct. Finalizing an utterance takes about 120-160 ms. |
| Speech to text, fallback | mlx-whisper large-v3-turbo, with `initial_prompt` set to the scam vocabulary | Local Mac (GPU) | No | **165 ms per 3 s chunk.** WER 0 on 8 kHz audio. The vocabulary prompt fixed "Zelle"→"Dell" on noisy mu-law audio. |
| Per-update gate *(new, optional)* | **ettin-encoder-68m** with 4 heads: risk, category, action, AI-voice-mention | Local Mac (MPS) | Yes, a local full fine-tune of about 10 min | **8.8 ms** per update at 256 tokens and 14 ms at 512. Training runs at about 2.5 min per 8k-example epoch. |
| **Guardian (the scored model)** | **Qwen3-4B-Instruct-2507, LoRA SFT on Fireworks** | Fireworks dedicated deployment (live merge, 1x H200) | Yes, Fireworks managed SFT | SFT wall time and deployment start-up are **not measured yet**. Measure them in the first hour. |
| Guardian, local small point *(optional)* | Qwen3-1.7B-4bit, QLoRA via mlx-lm | Local Mac | Yes, about 40 min | Zero-shot: **176 ms** for valid JSON of about 27 tokens. Trains at about 1,950 tok/s. |
| Caretaker summary at hang-up | DeepSeek V4.1 Flash | Fireworks serverless | No | 1.2 s with `reasoning_effort: none` |
| AI-voice signal | Telnyx `deepfake_detection` on Answer, plus a reply-gap timing heuristic | Telnyx | No | The verdict can take up to about 15 s, so it is a badge, not a trigger. |

**Fireworks constraints, verified today:**
- Fireworks cannot fine-tune or host any audio model, and its hosted ASR and TTS are deprecated.
- Our serverless LoRA quota is **0**, so the guardian needs a dedicated deployment.
- The only verified deployment shape for qwen3-4b-instruct-2507 is FP8 with `enableAddons=false`. That means **no multi-LoRA on one deployment**: B3 (base) and v1 each need their own deployment, unless a BF16 multi-LoRA shape matches.

### 2a. Fine-tuning the Nemotron speech model (stretch goal, not on the critical path)

- **What it needs:** NeMo on an NVIDIA GPU. The Google Cloud L4 in project `unified-booster-512122-j5` qualifies. Training audio would be G1's Kokoro-rendered 8 kHz calls, paired with their ground-truth text.
- **Why it's not on the critical path:**
  - Accuracy is already 0-2% WER on our clips.
  - Training on TTS voices risks a synthetic-voice shortcut; the voicemail-detection effort measured this failure.
  - The tuned checkpoint has to be re-exported to the sherpa-onnx ONNX format.
  - It costs about 1.5-2 h of Numan's time.
- **Only if the line works by about 2:00.** Score WER on held-out *different* Kokoro voices and on teammate-recorded calls, never on its own training voices.
- **Free speech-to-text tradeoff for the slides:** sherpa ships 80, 160, 560 and 1,120 ms chunk models with published WER of 8.43, 7.67, 7.07 and 6.93. That is a latency-vs-accuracy curve at no training cost.

## 3. Data

**Generator pilot, measured today.** Each model got 5 hard scenarios: an OTP bank scam, a tech-support gift-card scam, child grooming *tactics*, and 2 benign twins. Results are in `scratchpad/genpilot/`.

| Generator | Valid calls | Seconds per call | Use |
|---|---|---|---|
| **DeepSeek V4.1 Flash** | **5/5** | 6-10 | **Train and dev bulk.** It invents category names, so the prompt must enforce the enum. |
| **Kimi K3** | **5/5** | 15-21 | **Held-out test only.** It is never a baseline. |
| Nemotron 3 Ultra (`nemotron-3-ultra-nvfp4`, $0.60/$2.40 per 1M tokens) | 3/5 (both benign calls came back malformed) | 16-23 | Optional extra train slice (NVIDIA). Needs `reasoning_effort: none` and validate-and-retry. Leaves serverless on 11/07. |
| GLM 5.3 Flash | 0/5 (spent 3,500 tokens and returned no content) | 30 | Not used. |

**Volume:**
- About 1,200 train and 150 dev calls from DeepSeek, about 300 test calls from Kimi, and a 40-call gold set that a human edits. Test and gold are never trained on.
- The sampler adds **benign twins (35%)**, **one scam type held out entirely**, rule-flip pairs (10%, stretch), and simulated `ai_voice` and `caller_known` fields (from Numan's plan).
- Labels come from the plan: per-turn `risk`, `category`, `stage`, plus `ask_turn` and `onset_turn`.
- Training rows are transcript prefixes around onset and the ask, plus final prefixes. That is about 4k rows.

**Noise:**
- Google Cloud job G1 (Numan): Kokoro renders 300-500 train calls, downsamples them to 8 kHz, and runs Nemotron ASR on them. This produces real ASR-error text.
- Text-level ASR noise covers the rest.

**Safety. Mandatory; any failure stops the line:**
- Grooming is written only as *tactics*.
- The system prompt frames the data as synthetic safety-training data.
- A blocklist plus an LLM yes/no filter checks every row.
- **A human reads 30 child-mode rows before each upload.**
- PAN-12 is not used: it contains explicit logs and has no license.
- BothBosu is used only as an out-of-distribution set, labeled "LLM-generated", never "real calls".

## 4. Guardian output (Numan's taxonomy, risk-first for speed)

```json
{"risk":"none|watch|alert|urgent","category":"scam.impersonation_bank_gov|scam.tech_support|scam.family_emergency|scam.prize|scam.romance_investment|disclosure.location_school|disclosure.credentials_otp|disclosure.financial_id|disclosure.home_alone|grooming.secrecy|grooming.isolation|grooming.gifts_flattery|grooming.photo_request|grooming.meeting_request|vulgar|threat_extortion|self_harm|none","action":"none|log|coach|notify_caretaker|suggest_hangup|urgent_alert","evidence_turn":0,"coach_line":"","caretaker_note":""}
```

- `coach_line` and `caretaker_note` are filled **only when risk >= alert**, so the common case stays at about 20 tokens.
- One `render_messages()` function is shared by training and live use. The prompt is frozen (`prompt_sha`) before the SFT data is built.

## 5. Eval (the proof)

**Baselines:**

| ID | System | Where |
|---|---|---|
| B0 | Keywords and PII regex | Local |
| B1 | Rune System One (tuned questions) | Serverless; measured p≥0.996 on scam and grooming at 0.15-0.21 s |
| B2 | DeepSeek V4.1 Flash, zero-shot | Serverless; 1.2-3.4 s |
| B3 | Qwen3-4B untuned, same prompt | Dedicated |
| v1 | Qwen3-4B LoRA | Dedicated |
| E1 | ettin gate | Optional |
| Q1 | Qwen3-1.7B local LoRA | Optional |
| Llama Guard 3 8B, gpt-oss-safeguard-20b | Safety-model baselines | Dedicated-only with no validated shape. Optional, first to cut. |

**Metrics** (each with n and a 95% CI):
- **Catch-before-ask at ≤5 false alarms per 100 benign calls** (the headline).
- Lead time in turns.
- False alarms on benign twins.
- Recall on the unseen scam type.
- Category macro-F1.
- Evidence-quote exact match.
- p50 and p95 latency per scored turn.
- $ per 1,000 call-minutes, including GPU time.

**Test sets:** Kimi held-out, gold-40 (clean, plus a rendered-and-ASR'd slice), and BothBosu (synthetic, out-of-distribution).

**Charts:**
1. Catch-before-ask vs false alarms, one point per system.
2. Quality vs p95 latency, each point labeled with $/1k minutes.
3. A per-category bar chart.
4. An ASR chunk-size latency vs WER curve.

## 6. Timeline (clock-based; replaces Numan's hour 10-18 sync points)

| Time | Matt (model, data, eval; Claude agents do the coding) | Numan (line, ASR, demo) |
|---|---|---|
| 12:35-12:50 | **Joint:** freeze §4 schema and contracts, decide accounts and budget, commit `contracts/`. | Same |
| 12:50-1:15 | Start generation: DeepSeek train/dev and Kimi test, with the safety filter. Read 30 child rows. Run B0, B1 and B2 on early dev. Install firectl and a probe SFT. | Telnyx demo app plus tunnel. Get the first media-stream frames into the server. Copy the Nemotron model folder from Matt's Mac. |
| **1:20 hard** | **Launch SFT v1** (LoRA rank 16, 2 epochs) with whatever data exists. Create the B3 deployment. | G1 noise pass on Google Cloud. Nemotron ASR on live frames. |
| 1:20-2:15 | Run B3 on test and gold. Optional: train the ettin gate on the Mac (10 min). Build the console. | Policy layer, conference coach `speak` to leg B only, caretaker alert call (press 1 to listen). |
| 2:15-3:00 | SFT done → deploy v1 → evaluate everything. **Cut line 2:45:** if v1 isn't evaluated, present B0-B3 plus the gate as the frontier. | End-to-end live call on the guardian. Measure turn-to-coach latency. |
| 3:00-3:45 | Charts, results table, README. **Delete B3.** Freeze numbers at 3:45. | Rehearse the 5 demo scenes and record the backup video by 3:40. |
| 3:45-4:15 | Submission text, video, links. **Submit by 4:15.** | Same |
| 6:45-7:30 | Warm v1 (min replicas 1) for the demo, then delete it. | Live demo; a judge calls the number. |

**Cut order:**
1. The AI-voice detector training (use Telnyx deepfake detection only).
2. Vapi/Retell callers.
3. Llama Guard and safeguard.
4. The ASR fine-tune.
5. Rule-flip pairs.
6. Q1.
7. E1.
8. Barge-in.

**Never cut:**
- B1, B2 and B3 against v1 on the held-out set.
- The benign-twin false-alarm rate.
- The safety gate.
- The backup video.

## 7. Budget and accounts

Prices: H200 is $8/hr, billed per second, with startup free. LoRA SFT is $0.50 per 1M tokens. DeepSeek is $0.30 in / $1.20 out; Kimi K3 costs more.

The `.env` key is **Numan's account** (`numan-vf90m1xhbxgc`). No spend cap is enforced at the quota level, so deployments must be deleted by hand.

| Item | Estimate |
|---|---|
| Generation (DeepSeek bulk + Kimi test) | ~$3-8 |
| SFT v1 | ~$2-5 |
| B3, about 1.5 h | ~$12 |
| v1, 2:15-3:45 plus 6:45-7:30 | ~$18 |
| Rune and DeepSeek baselines | <$2 |
| **Total** | **~$40** |

Google Cloud G1 is capped at $25 (Numan's rule).

## 8. Folder

This project lives in `callguard/`, because the voicemail-detection project shares the repo root. Numan's planned folders map under it: `contracts/ factory/ guardian/ eval/ line/ console/ results/ data/`, with `data/` gitignored.

## 9. Decisions needed now (Matt and Numan)

1. **Deadline:** 4:30 PM today. Numan's hour-based sync points must be replaced by §6.
2. **Test generator:** Kimi K3 (5/5 in the pilot) instead of GLM (0/5). Kimi is then dropped as a baseline.
3. **Accounts:** which Fireworks account runs SFT and the deployments (Matt's own key?), and is there a hard $ cap per account?
4. **ASR fine-tune:** skip today, or a stretch job for Numan on Google Cloud after the line works (§2a)?
5. **Go** for bulk generation now (~$5, ~30 min).
6. **Second student (recommended hedge).** Train **Qwen3.6-35B-A3B** alongside Qwen3-4B, on the same data, and launch both at 1:20. The details below are verified from the catalog and shapes today:
   - It is an MoE with 3B active parameters, so it should decode at near-4B speed, while 35B of total knowledge should generalize better to Kimi-written test calls and the held-out scam type.
   - It has the LoRA training shape `qwen3p6-35b-a3b-256k-lora` and is in the serverless Training API pool.
   - It deploys on the `-minimal` shape: 1x B200 FP4 at $13/hr, with LoRA serving supported.
   - Unverified: whether live merge works on that FP4 shape.
   - Extra cost: SFT at about $3 per 1M tokens (about $9), plus about $20 of B200 time.
   - Both students go on the frontier chart, and the better one serves the demo.

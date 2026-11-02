# Emberz: plan

Fireworks hackathon, November 2026. Numan and Matt. Built on Fireworks plus open models; no fallbacks for now.

## 1. What we're building

**Emberz gives your kid, your grandma or you a guarded phone number.** Every call to it passes through our line. A small model fine-tuned on Fireworks reads the live transcript and stays silent on normal calls. It speaks up when a call turns into a scam, when a stranger asks a kid for personal info, or when the caller is an AI voice.

- **The protected person** hears a quiet tip that the caller can't hear.
- **The caretaker** (mom, an adult child) gets an alert call. They can listen live, join the call with one tap, and get a summary afterwards.
- **The caretaker writes the rules** in plain words, such as "never share the door code" or "tell me about any talk of money".
- **The real number is never given out**, so it stays out of data brokers' lists, leaks and strangers' hands.

**What we prove:**
- **Main claim:** a small model fine-tuned on Fireworks using only generated calls catches more risky calls before the ask (money, codes, address, meeting up) than Llama Guard 3, gpt-oss-safeguard and a frontier model. It raises fewer false alarms on normal calls, is faster, and costs much less per call-minute.
- **Second result, built in parallel:** AI-voice detection on real phone audio, compared with an off-the-shelf detector and Telnyx's deepfake check.

**Why it's new** (checked 2026-11-02): Google and Samsung detect scams on the phone itself and warn only the person on the call. OnGuardAI texts family a red flag, with no transcript, no AI-voice check, no caretaker rules and no kids. Hiya and Truecaller check the voice only. Kids' phones check texts, not calls. No one streams the call to a caretaker under rules the caretaker writes, and no one publishes an open model with a reproducible eval.

## 2. The stack (decided)

| Part | Choice |
|---|---|
| Phone line | Telnyx on EmberDial's account: the existing demo app and spare test numbers. Production apps and live numbers are never touched |
| Public URL for webhooks | `cloudflared` tunnel to the line server on Numan's Mac |
| Speech-to-text | **NVIDIA Nemotron Speech Streaming 0.6B** (open, NVIDIA Open Model License), run locally with **sherpa-onnx** INT8. Streams in 160 ms chunks, one stream per side |
| Training-data generator | **Nemotron 3 Ultra** on Fireworks serverless. It **needs a system prompt framing the task as synthetic safety-training data**: without it, it refused the scam call |
| Held-out test generator | **GLM 5.3 Flash** on Fireworks serverless: a different model family, so the test isn't the generator grading itself |
| Guardian model | **Qwen3-4B-Instruct-2507**, LoRA SFT on Fireworks, served on an on-demand deployment |
| Baselines | Keyword rules, Llama Guard 3 8B (on-demand), gpt-oss-safeguard-20b (on-demand), untuned Qwen3-4B (same deployment), **Kimi K3** serverless |
| Dev stand-in for the guardian (used only while building the line) | A cheap serverless model with the same prompt and schema, while the line is being built |
| AI-voice detector | Local: frozen wav2vec2 front end + small head, trained on public data. Telnyx `deepfake_detection` ($0.01 per call) as the commercial baseline |
| AI test callers | Our own agents on Vapi and Retell (free credits) calling our number |
| Text-to-speech, for transcription noise in training data | Kokoro (Apache-2.0) |
| Google Cloud (bulk jobs only, Numan) | Project `unified-booster-512122-j5`, region us-central1. Quota: 200 N2 vCPUs and 1 L4 GPU (spot allowed). Runs the bulk noise pass and the AI-voice dataset prep. The live system runs on the Mac and Fireworks |

**Fireworks checks done (2026-11-02):**
- The key works.
- Nemotron 3 Ultra answers on serverless.
- Qwen3-4B and Qwen3-8B support LoRA.
- Nemotron Nano 3 does not support LoRA on Fireworks, so it can't be the guardian.
- Fireworks' hosted speech-to-text is unavailable on our key (503 / not found), so speech-to-text runs locally on open Nemotron.

## 2b. Measured speed (2026-11-02, M5 Pro + Fireworks)

| What | Measured | What it means |
|---|---|---|
| Generate one ~20-turn call | Nemotron 3 Ultra **157 tok/s** (~10 s per call, with framing); DeepSeek V4.1 Flash 131 tok/s; GLM 5.3 Flash 103 tok/s | With ~50 parallel requests, 3,500 calls take about 15-20 min, if rate limits allow (checked in M0) |
| Live Nemotron ASR on the Mac | **37 ms** to process each 160 ms chunk; **64 ms** at worst with 4 streams at once | 2 live calls (4 streams) run comfortably in real time |
| Bulk Nemotron ASR on the Mac | **~11x real time** at best (8 processes) | 300 calls (~10 h of audio) take ~55 min, so the bulk noise pass goes to Google Cloud (G1, ~10-15 min) |
| SFT, deployment start-up | Not yet measured | Measured by the M0 probe job |

## 3. How a call works

```
 caller dials the guarded number (Telnyx)
   |
   v
 answer (+ deepfake_detection) -> notice: "This call is protected by Emberz
   |                                        and may be transcribed"
   |-- media stream, both tracks, one WebSocket --> line server (Numan's Mac)
   |      |-- Nemotron streaming ASR, one stream per side ("Caller" / "Grandma")
   |      |-- AI-voice detector (first 2-4 s) + reply-gap timing
   |      |-- guardian model on Fireworks: rules + last turns -> JSON
   |      '-- rule layer: smoothing, cooldowns, allowed contacts -> action
   v
 conference: dial the protected phone, showing the caller's real number
   |-- coach: conference speak to the protected leg only (caller hears nothing)
   |-- caretaker alert call: "Press 1 to listen" -> join as monitor, barge in
   |-- live console (web): transcript, risk lines, alerts
   '-- hang-up: summary to the caretaker
```

## 4. Contracts: the glue that lets us work in parallel

All events are JSON over a WebSocket or HTTP in the line server. The schemas live in `contracts/` and are agreed in hour 0, so nobody waits on anybody.

**`turn`**: Line to Guardian, one final transcript segment.
```json
{"call_id": "c_123", "turn": 7, "speaker": "caller|protected", "text": "can you read me the code",
 "t_start_ms": 18320, "t_end_ms": 19950, "gap_ms": 1400}
```

**`guardian_request`**: rules, the turn window and signals.
```json
{"call_id": "c_123", "profile": "elder|kid|adult",
 "rules": ["Never share codes or passwords", "Alert me about any money talk", "Allowed: Grandpa (+1555...)"],
 "caller_known": false, "ai_voice": {"score": 0.91, "source": "detector|telnyx|timing"},
 "turns": [{"turn": 1, "speaker": "caller", "text": "..."}]}
```

**`guardian_response`**: one per scored turn (section 6).

**`voice_signal`**: Voice to Line.
```json
{"call_id": "c_123", "at_ms": 2400, "ai_score": 0.91, "model": "w2v-head-v1|telnyx|timing"}
```

**`console_event`**: Line to the console: `call_started`, `turn`, `risk`, `alert`, `coach`, `caretaker_joined`, `summary`, `call_ended`.

**`dataset_row`**: Model, one generated call.
```json
{"call_id": "s_000123", "split": "train|dev|test_heldout", "generator": "nemotron-3-ultra|glm-5.3-flash",
 "profile": "kid", "rules": ["..."], "rule_flip_of": null, "scam_type": "impersonation_bank",
 "turns": [{"turn": 1, "speaker": "caller", "text": "...", "label": {"risk": "none", "category": "none", "stage": "hook"}}],
 "ask_turn": 9, "benign_twin": false}
```

**`results_row`**: one row per model, test set and metric, used for the charts.
```json
{"model": "guardian-v1", "test_set": "heldout|bothbosu|roleplay|pan12", "metric": "catch_before_ask", "value": 0.83, "n": 300, "ci95": [0.79, 0.87]}
```

## 5. Work packages

Big chunks, each with an owner, what it delivers, and when it's done. **I** = kickoff, **M** = Model and **C** = console (Matt), **L** = Line, **V** = Voice and **G** = Google Cloud (Numan), **D** = demo.

**Matt:** M0, M1, M2, M3, C1, D2. **Numan:** L1, L2, L3, V1, V2, G1, D1. **Both:** I0.

### I0. Kickoff (both, about 75 minutes, hour 0)
- Write the contracts as code: Pydantic models for every section 4 shape, plus fixture files and a stub guardian that returns canned JSON.
- Agree who owns which folders and which branch names. Each person works in their own folders and their own ledger file, to avoid merge conflicts.
- **Keys:**
  - Matt and Numan each have their own Fireworks key.
  - **The Telnyx key (EmberDial's account) stays on Numan's Mac only.**
  - Auto Reload off on both Fireworks accounts.
- Set up a shared channel for `data/` and `recordings/`. Manifests are committed; files are not.
- Start the big downloads (ASVspoof 5, In-the-Wild), request PAN-12 access, and book friends for the human and role-play calls.

### M0. Fireworks hour-0 checks and the shared guardian library (Matt)
- **Checks:**
  - GPU quota and rate limits on both accounts.
  - The Nemotron 3 Ultra price, which sets the generation spend cap.
  - How long the Batch API takes to return.
  - Deployment shapes (`firectl deployment-shape-version list`).
  - Whether `response_format` (JSON schema) works on the LoRA deployment.
  - How multi-turn SFT samples are billed (a tiny test job).
- **Builds:** `guardian/` with `render_messages()` (one prompt for training and live use), the parser and validator, and the client.
- **Prompt freeze at sync S2:** publish `prompt_sha`, the 8-turn window and how `ai_voice` and `caller_known` are bucketed. Any later change means retraining.

### M1. Data factory (Matt)
- **One request per whole call:** the generator writes the full call from a hidden plan (stages, the exact ask turn, tactics). A plan-alignment checker drops calls that drift from the plan. Labels come from the plan.
- **Scenario grid:** protected person (kid 8-12, teen, elder, adult) x caller type (scams by family, stranger, groomer at the tactics level, real bank, doctor's office, school, family member, friend, AI appointment bot) x rule set x persona.
- **Extras:**
  - Benign twins (35%): normal calls that reuse scam words.
  - Rule-flip pairs (10%) that also change the written rules.
  - Disclosure calls: the protected person gives away personal info.
  - One scam type held out entirely.
  - **Simulated `ai_voice` and `caller_known` fields**, so training inputs match the live ones.
  - Transcription noise on part of the set: Kokoro, then 8 kHz, then Nemotron ASR.
- **Hard rule: grooming stays at the level of tactics. No sexual content involving minors is ever generated.**
- **Generators:** Nemotron 3 Ultra makes train and dev. GLM 5.3 Flash makes the held-out test with different persona seeds, and its manifest is committed before any model scores it.
- **Done when:** about 3,000 training calls and 500 held-out calls pass schema validation.

### M2. Guardian fine-tune and deployment (Matt)
- **Builds:** the training-set builder (calls into windowed chat samples with JSON targets, using `render_messages()`), SFT v1, and one multi-LoRA deployment for the base model (B3), v1 and v2, which is also the demo deployment.
- **Training settings:** cap v1 at about 24M billed tokens, v2 at about 8M. **Always pass an explicit LoRA rank (8-16).** A missing rank means full training at twice the price.
- **Checkpoint:** v1 must beat B3 on dev, or we fix the errors and retrain once.
- **v2 (adversary round):** built from **dev errors only**, never the test set.

### M3. Eval harness, baselines and charts (Matt)
- **Builds:** scorers for section 7, baselines B0-B4, test sets (held-out, BothBosu, PAN-12 subset, role-play from V2), charts.
- **Shared rules:** the rule layer is a pure function in `line/policy.py`, and eval imports the same code, so eval scores the system the demo runs. Rule-following is also measured on the model alone.
- **Run early:** B0 and B4 on BothBosu plus 50 dev calls.
- **Done when:** one command re-runs everything and redraws the charts.

### C1. Caretaker console and summaries (Matt)
- **Builds:** a phone-friendly web page driven by `console_event`: the live transcript of both sides, risk lines, alerts, a join button and the post-call summary (one serverless call at hang-up).
- **Done when:** it works against the fixture event stream, then on live calls after L3.

### L1. Isolated Telnyx line (Numan)
- **Setup: reuse what EmberDial already has.** Use the existing demo Call Control app (`EmberDial Jev Demo`) and any number not in use (owner's call: any unused number is fine). **Never repoint production apps** (`Dialer`, `EmberDial Private Browser Ingress`) or any number an agency uses for live calls.
- **Webhooks:** go through a `cloudflared` tunnel. Only the demo app is re-pointed when the URL changes. Verify every webhook with Ed25519 (`TELNYX_PUBLIC_KEY`). Handlers are idempotent on `command_id`.
- **Call skeleton:**
  - Answer leg A (the caller) with `deepfake_detection` on.
  - Play the notice and put A into a conference.
  - Dial B (the protected phone) into the conference, with `from` = the caller's number and `diversion` = the guarded number. If the caller is anonymous, show "Emberz: unknown caller".
  - Stream A with `both_tracks`: inbound is the caller, outbound is the conference mix.
  - Set `end_conference_on_exit` on A and B.
- **Hour-0 checks, written to `line/hour0.json`:**
  - Do deepfake detection and our stream run together on leg A?
  - Does a targeted conference `speak` stay out of the caller's audio?
  - Does the track split come out as expected?
  - Do the supervisor roles work?
  - Does the caller's number show on B?
  - Does the notice or ringback leak into the protected person's transcript?

### L2. Live brain (Numan)
- **Builds:**
  - Nemotron streaming ASR per side (sherpa-onnx INT8, 160 ms chunks), with μ-law decode and stateful resampling.
  - Turn segmentation and `turn` events.
  - The guardian client: the stand-in model during development, then the deployment.
  - `line/policy.py`: smoothing, cooldowns, allowed contacts, and AI voice + unknown caller + ask = escalate.
  - **The coach:** `POST /v2/conferences/{id}/actions/speak` with `call_control_ids` set to B only, so only the protected person hears it.
- **Done when:** a role-played scam call produces a coach line and an alert within about 1.5 s of the trigger line.

### L3. Caretaker alert call and console backend (Numan)
- **Builds:** dial C (the caretaker), then `gather_using_speak` ("Press 1 to listen"), then on digit 1, join the conference with `supervisor_role: monitor`. The join button calls `update` with `barge`. **Never mute C**, because an explicit mute outlasts the role change. Also the `console_event` WebSocket feed and the summary trigger.

### V1. AI-voice detector (Numan)
- **Builds:** frozen wav2vec2 features over the first 2-4 s of caller audio, plus a small logistic head. Also timing and text cues: reply gap (people ~0.2 s, agents ~1.3-2.4 s), "I'm an AI assistant", missing "um"s.
- **Training data:**
  - ASVspoof 5, including its 8 kHz codec conditions.
  - In-the-Wild.
  - Kokoro renders of our scripts.
  - Human audio over the same Telnyx path.
  - **Not ElevenLabs output.**
- **Baselines:** W2V2-AASIST (MIT) and Telnyx `deepfake_detection`.
- **Done when:** EER and AUC on our phone set for audio only, timing only and combined, with n and intervals.

### V2. Real test calls (Numan records, Matt labels blind)
- 5 of our own Vapi and Retell agents making about 10 calls each, plus about 50 human calls over the same path.
- About 40 role-play guardian calls with blind-written scripts.
- Labels are committed after the calls are segmented (around sync S4), before any model scores them.

### G1. Google Cloud bulk jobs (Numan)
- **Noise pass, for M1:**
  - An `n2-highcpu-96` VM.
  - Kokoro renders about 300-500 of Matt's generated calls, converted to 8 kHz phone audio.
  - Nemotron ASR (the same sherpa-onnx model as live) transcribes them, so part of the training text carries real transcription errors.
  - About 10-15 minutes, a few dollars.
  - Input: `data/noise_input.jsonl` from Matt. Output: `data/noisy_turns.jsonl` back to Matt before the prompt freeze (S2).
- **AI-voice data, for V1:**
  - A spot `g2-standard-8` VM with one L4.
  - Download ASVspoof 5 (a subset, including the 8 kHz codec conditions) and In-the-Wild.
  - Extract wav2vec2 features on the GPU and copy only the small feature files to the Mac through a Cloud Storage bucket.
  - The large audio never touches the Mac.
- **Rules:**
  - Delete every VM as soon as its job ends, and run `gcloud compute instances list` at every session end.
  - Cap: $25 total.
- **Done when:** the noisy turns are delivered to M1, and the voice features are in the bucket for V1.

### D1. Integration and rehearsal (Numan)
- The full demo end to end: casting the phones, the judge call, rehearsals.

### D2. README, video and submission (Matt)
- Results, charts, legal notes, the 60-second video, the Fireworks endpoint, and teardown (delete deployments, check Telnyx usage).

### Sync points
- **S0, hour 0:** I0 together.
- **S1, about hour 2.5:** hour-0 results. From Matt: quota, the Nemotron price, Batch timing, `response_format`. From Numan: `line/hour0.json` and the first real `turn` events.
- **S2, about hour 5-6: prompt freeze** (`prompt_sha`). G1's noisy turns are in M1's data. Pilot numbers check the budget before bulk generation. Role-play cards are final. The held-out manifest is committed.
- **S3, about hour 10-12: GPU window 1** on account A. B3 and v1 on dev (the checkpoint), held-out and BothBosu, plus latency. Numan makes one live call on the deployment to measure turn-to-coach latency. B1 and B2 run on account B in the same sitting, each with a 1-hour hard stop. Delete everything afterwards.
- **S4, about hour 14:** recordings handed over, role-play labeled blind and committed, voice results frozen, v2 data built from dev errors.
- **S5, about hour 16-18: GPU window 2,** straight into the rehearsal so the deployment warms up once. v1 vs v2 on every set, the full 6-scene rehearsal, then the README numbers are locked.
- **S6:** demo and submission. Warm the deployment about 10 minutes before the slot, and scale it to 0 or delete it right after.

### Critical path
I0, M0, M1 (with the G1 noise pass), M2 (v1), window 1, v2, window 2, D2. The product chain (I0, L1, L2, L3, D1) and the cloud jobs (G1) run alongside it and joins at S3 for the live latency call and at S5 for the rehearsal.

## 6. The guardian model

**Output per scored turn:**
```json
{"risk": "none|watch|alert|urgent",
 "category": "scam.impersonation_bank_gov | scam.tech_support | scam.family_emergency | scam.prize | scam.romance_investment | disclosure.location_school | disclosure.credentials_otp | disclosure.financial_id | disclosure.home_alone | grooming.secrecy | grooming.isolation | grooming.gifts_flattery | grooming.photo_request | grooming.meeting_request | vulgar | threat_extortion | self_harm | none",
 "severity": 0, "source": "caller|protected_person", "stage": "hook|trust|pressure|ask",
 "evidence": [{"turn": 7, "quote": "exact words from the transcript"}],
 "action": "none|log|coach|notify_caretaker|suggest_hangup|urgent_alert",
 "coach_line": "15 words max", "caretaker_note": "20 words max"}
```

- **Headline categories:** scam, disclosure and grooming tactics.
- **Second slice, scored separately:** vulgar language, threats and self-harm. Llama Guard is strong on these, so the slice is reported on its own.
- **An AI voice alone is at most `watch`.** AI voice plus an unknown caller plus an ask escalates. A pharmacy reminder bot lights the badge but sends no alert. The FCC treats AI voices as "artificial" under the TCPA (FCC 24-17) and requires such calls to identify who is calling (47 CFR 64.1200(b)).
- **Rules vary across training**, so the model follows the rules it's given instead of memorizing one set.

## 7. Scoring

| Metric | Definition |
|---|---|
| **Catch-before-ask** (headline) | % of risky calls at or above `alert` before the labeled ask turn, reported at 5 or fewer false alarms per 100 benign calls |
| False alarms | Alerts per 100 benign calls, hard negatives included |
| Lead time | Turns between the first alert and the ask |
| Rule-flip accuracy | % of rule-flip pairs decided per the rules |
| Unseen scam type | Recall on the scam type held out of training |
| Category F1 | Macro-F1 over categories |
| Evidence match | % of quotes found exactly in the transcript |
| Speed | p50 and p95 per scored turn, in milliseconds |
| Cost | $ per call-minute, including deployment GPU time |

| | Baseline | Where |
|---|---|---|
| B0 | Keywords + personal-info regex | Local |
| B1 | Llama Guard 3 8B, last turn vs a 6-turn window | Fireworks on-demand |
| B2 | gpt-oss-safeguard-20b with the caretaker's written rules | Fireworks on-demand |
| B3 | Qwen3-4B untuned, same prompt | Same deployment as ours |
| B4 | Kimi K3 zero-shot | Fireworks serverless |

**Test sets, reported separately, each with n and a 95% interval:**
1. The held-out generated set.
2. BothBosu: 20 real scam calls.
3. A PAN-12 subset: grooming in chat, not phone calls, and we say so.
4. Our ~40 role-play calls.

**Charts:**
- Catch-before-ask vs false alarms, one point per model.
- Speed vs quality.
- A per-category breakdown.
- v1 vs v2.
- Separately, AI-voice detection.

## 8. The demo

The screen shows the live console. Three phones: Grandma's, the kid's and Mom's.

1. **An AI voice scams Grandma.** Our AI caller dials as "her bank". About 2 s in, the console shows **AI voice**. On "read me the code", Grandma hears: "Banks never ask for codes. Say you'll call the number on your card." Mom's phone rings with the alert.
2. **Mom joins.** She presses 1 and listens, then barges in.
3. **A stranger calls the kid.** "Parents home? What street?" sends Mom the alert and the live transcript.
4. **Same words, different rules.** Grandpa, on the allowed list, asks the same thing and it stays silent.
5. **An AI that isn't a scam.** A pharmacy reminder bot calls. The AI badge lights, no alert.
6. **Close.** The overlay shows "alerted N turns before the ask", then the chart.

A judge can call the guarded number from their own phone and try to fool it.

## 9. Budget

**Fireworks: $100, two $50 accounts that can't be pooled.** Prices: H100 or H200 at $8 per hour, billed per second, start-up not billed. LoRA SFT on models up to 16B at $0.50 per million tokens. The Batch API is 50% off.

| Account | Line item | Estimate |
|---|---|---|
| A (Matt) | Probe SFT job | ~$0.25 |
| A (Matt) | SFT v1 (24M tokens or fewer), v2 (about 8M) | ~$16 |
| A (Matt) | Window 1 (1 h or less), window 2 + rehearsal (1.5 h or less), demo (1 h or less) | ~$28 |
| A (Matt) | **Reserve, never spent before the demo** (usage stops at $0) | $5.50 |
| B (Numan) | Generation: Nemotron 3 Ultra train/dev (cap $12 once its price is known), GLM held-out (about $1), via Batch | ~$13 |
| B (Numan) | B4 Kimi K3 via Batch on 500 held-out calls + a 100-request latency sample | ~$9.60 |
| B (Numan) | B1 Llama Guard + B2 safeguard, 1 GPU-hour or less each | ~$16 |
| B (Numan) | Stand-in model + summaries | ~$1.50 |
| | **Total** | **~$84** |

**Rules:**
- Deployments are up only in windows. Create them with `--min-replica-count 0 --scale-to-zero-window 5m`, and delete them afterwards.
- A scaled-down deployment answers 503 while waking, so warm it about 10 minutes before use.
- Run `firectl deployment list` on both accounts at every session end.
- Each person keeps a ledger: `results/ledger-matt.md` and `results/ledger-numan.md`.

**Google Cloud (G1):** capped at $25. VMs are deleted right after each job.

**Telnyx:** about $0.002 per minute of voice, $0.002 per conference participant per minute, $0.0035 per minute of streaming, $0.01 per call for deepfake detection and $1 a month per number. About 150 test calls come to single-digit dollars.

## 10. Consent (for the README)

- **Every call starts with a notice.** It covers all-party-consent states such as California (Penal Code 632.7).
- **The protected person opts in.** Federal law needs one party's consent. A caretaker can't consent for a competent adult.
- **Kids:** a line the parent owns. Under the amended COPPA rule (enforced since April 22, 2026), a child's voice is personal information, so we need parental consent plus written retention and security policies.
- **Privacy by default:** alerts and summaries first, with the live transcript opt-in or opened only after an alert. No voiceprints, and data deleted after 7 days.
- **Hackathon data:** generated calls, public datasets within their licenses, and calls between consenting teammates and friends. **No EmberDial customer data, ever.**

## 11. Repo layout

```
PLAN.md, README.md
contracts/   JSON schemas for every event and row (section 4)
factory/     M1 scenario grid, self-play, labels, splits
guardian/    M2 training-set builder, SFT and deploy scripts, client
eval/        M3 scorers, baselines, charts
line/        L1 + L2 Telnyx app, media stream, ASR, rule layer, coach, alerts
console/     L3 caretaker console
voice/       V1 AI-voice detector
recordings/  V2 (gitignored audio; labels committed)
results/     results rows, ledger
data/        gitignored
```

## 12. Cut order (if time runs short)

1. The adversary round (v2).
2. B1 Llama Guard.
3. Barge-in.
4. The second category slice.

**Never cut:** B2, B3 and B4 on the held-out set, and the benign-twin false-alarm number. They are the proof.

## Sources (checked 2026-11-02)

- Fireworks: [SFT](https://docs.fireworks.ai/fine-tuning/supervised-fine-tuning) · [tunable models](https://docs.fireworks.ai/fine-tuning/models) · [deploying LoRAs](https://docs.fireworks.ai/fine-tuning/deploying-loras) · [pricing](https://fireworks.ai/pricing) · [Llama Guard 3 8B](https://fireworks.ai/models/fireworks/llama-guard-3-8b) · [gpt-oss-safeguard-20b](https://fireworks.ai/models/fireworks/gpt-oss-safeguard-20b)
- Speech-to-text: [Nemotron Speech Streaming 0.6B](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b) · [sherpa-onnx Nemotron](https://k2-fsa.github.io/sherpa/onnx/nemo/nemotron-streaming.html)
- Telnyx: [media streaming](https://developers.telnyx.com/docs/voice/programmable-voice/media-streaming) · [deepfake detection](https://telnyx.com/release-notes/deepfake-detection)
- Landscape: [Google Scam Detection on Galaxy S26](https://www.androidauthority.com/google-scam-detection-samsung-galaxy-s26-3643942/) · [OnGuardAI](https://betakit.com/onguardai-launches-app-to-flag-scam-calls-in-real-time/) · [Aura Call Protection](https://www.aura.com/help/what-is-call-protection) · [Hiya AI voice detection](https://www.hiya.com/products/ai-voice-detection)
- Data and research: [BothBosu scam datasets](https://huggingface.co/datasets/BothBosu/Scammer-Conversation) · [Shen et al. 2024](https://arxiv.org/abs/2409.11643) · [PAN-12](https://zenodo.org/records/3713280) · [ASVspoof 5](https://zenodo.org/records/14498691) · [In-the-Wild](https://arxiv.org/abs/2203.16263) · [W2V2-AASIST](https://huggingface.co/SpeechAntiSpoofingBenchmarks/W2V2-AASIST) · [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) · [ElevenLabs use policy](https://elevenlabs.io/use-policy)
- Law: [18 USC 2511](https://www.law.cornell.edu/uscode/text/18/2511) · [CA Penal Code 632.7](https://law.justia.com/codes/california/code-pen/part-1/title-15/chapter-1-5/section-632-7/) · [FCC AI voices under TCPA](https://www.commlawblog.com/2024/02/articles/fcc/fcc-clarifies-that-ai-generated-voices-are-subject-to-tcpa/) · [COPPA amendments](https://www.hunton.com/privacy-and-cybersecurity-law-blog/ftc-publishes-final-coppa-rule-amendments)

Not legal advice: get a lawyer's read before launching a real product.

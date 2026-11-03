# Emberz: a guardian on Grandma's phone line

Give your grandma, your kid or yourself a **guarded number**. Every call passes through our line:
**NVIDIA Nemotron Speech Streaming** transcribes both sides live, a guardian model **fine-tuned and served on Fireworks**
reads the conversation turn by turn, and when a caller starts working someone over (impersonation, "read me the code",
grooming, threats) Emberz:

- **whispers a coach line** only the protected person hears ("Don't share any codes. Hang up and call the bank directly."),
- **phones the caretaker**, who can listen in (press 1), talk (press 0) or **end the call** (press 9),
- follows **rules the caretaker writes in plain words** ("Never share codes", "Alert me about any money talk, even with family"),
- flags **AI voices** with our own detector (wav2vec2 features + a trained head, evaluated below) and reply timing.

## Why now: AI made the scam call cheap, convincing and endless

- **Losses are rising fast.** People 60+ reported **$7.75B** in losses to the FBI in 2025, **up 59%** in a year; average loss
  $38,500, and 12,444 of them lost more than $100k. The FTC puts the true 2025 cost to older adults at **$13.3B-$119.5B** [1][2].
- **The phone is where they lose the most.** For older adults, the median loss on scams that start with a phone call is **$3,000**,
  vs $599 when the scam starts on social media; people 80+ lose more by phone than by any other channel [2].
- **AI is already in these calls.** The FBI logged **22,364 complaints mentioning AI in 2025 ($893M lost)**, 3,143 of them from
  people 60+ ($352M), including grandparent "distress" scams that use cloned voices of loved ones. The FCC ruled that AI voices
  count as "artificial" under the TCPA and named voice clones posing as family as a harm [1][4].
- **We built the attacker ourselves.** Our test caller is off-the-shelf: an LLM scam script plus a stock neural voice on a
  cloud phone API, an estimated **$0.34 per two-minute call** (`callers/signalwire`). On real phone calls it held a fluent,
  pressuring back-and-forth and asked for the verification code.
  Our AI-voice detector flagged it on 4 of 5 live calls (see below).
- **Existing filters stop at the ring.** Carrier filters and caller-ID apps judge the number before you answer. A manipulative
  call from an unknown or spoofed number gets through, and the damage happens during the conversation, which is exactly where
  Emberz listens.

Sources (full list and figures in [docs/business_value.md](docs/business_value.md)): [1] FBI IC3 2025 Internet Crime Report;
[2] FTC, Protecting Older Consumers 2025-2026; [4] FCC Declaratory Ruling FCC 24-17.

Team: **Numan Karakas** (live line, speech-to-text, AI-voice detector, real-audio tests) and **Matt Ceran** (guardian data,
fine-tuning, eval harness, console).

## Results (main benchmark: 257 held-out manipulation calls)

Test calls were written by **NVIDIA Nemotron 3 Ultra**, a different model family from the training data (DeepSeek V4.1 Flash),
and held out. A risky call counts as **caught before the ask** if the guardian reaches "alert" before the turn where the caller
asks for the code, money, address or meet-up. Same frozen prompt (`guardian/render.py`, sha `806908cfba7d`) for every model.
Wilson 95% CIs. Full tables, ablations, caveats and the commands to reproduce: **[callguard/README.md](callguard/README.md)**.

| Model | Caught before the ask | False alarms on harmless calls | Caretaker rule-flip pairs right |
|---|---|---|---|
| **Emberz guardian v2** (Qwen3.6-35B-A3B LoRA, Fireworks Training API) | 137/148 (93%, CI 87-96) | **14/79 (18%, CI 11-28)** | **19/22 (86%, CI 67-95)** |
| Kimi K3, zero-shot (frontier baseline) | 133/148 (90%, CI 84-94) | 18/79 (23%, CI 15-33) | 4/22 (18%, CI 7-38) |
| Fireworks Rune, zero-shot | 137/148 (93%, CI 87-96) | 30/79 (38%, CI 28-49) | 5/22 (23%, CI 10-43) |
| DeepSeek V4.1 Flash, zero-shot | 126/148 (85%, CI 78-90) | 25/79 (32%, CI 22-42) | 3/22 (14%, CI 5-33) |
| Qwen3.6-35B-A3B untuned (same base) | 132/148 (89%, CI 83-93) | 42/79 (53%, CI 42-64) | 3/22 (14%, CI 5-33) |
| Keywords + PII regex | 77/148 (52%, CI 44-60) | 39/79 (49%, CI 39-60) | 0/22 (0%, CI 0-15) |

Counting every call (risky caught before the ask, harmless never alerted), v2 handles **90.3%** vs **83.7% for Kimi K3**
(McNemar p = 0.003; scenario-clustered bootstrap gain +0.9 to +12.3 pts), 83.3% for Rune and 79.4% for DeepSeek V4.1 Flash.
The write-up with every table: [docs/emberz_submission.pdf](docs/emberz_submission.pdf).

Reading: the fine-tune matches the best frontier catch rate, raises fewer false alarms than every zero-shot model
(less than half of Rune's and DeepSeek's), and is the only model that reliably follows the caretaker's own rules
(86% vs at most 23%). Out of distribution (BothBosu, 200 public calls) it catches 98/100 scams with 0/100 false alarms.

## The live line works end to end (real phone calls)

`line/` is a FastAPI server on Telnyx Call Control v2: signed webhooks, answer + notice, conference with the protected phone,
both-track media stream, local Nemotron ASR (sherpa-onnx INT8, 0.4x real time on a laptop), guardian per caller turn,
coach whisper to the protected leg only, caretaker call with press 1 / 0 / 9, console events, summary.

Logged real calls (`results/ledger-numan.md`, recordings in `recordings/`, a synced replay in `demo/screens/replay.html`):

- **15:11 PST, 2 Nov 2026:** AI bank-scam caller (SignalWire) -> guarded number -> real phone. 11 turns live, urgent alerts on
  the impersonation (turn 4) and the code request (turn 10), 3 coach whispers sent 0.70-0.75 s after the trigger, caretaker
  phoned, joined, heard the second alert.
- **15:48 PST:** same flow on the exact evaluated checkpoint (Matt's sampler proxy, `callguard/guardian/sampler_proxy.py`);
  the caretaker pressed 9 and Emberz hung up on the scammer.
- If the guardian endpoint is down, the line falls back to NVIDIA Nemotron Lightning 3.5 on Fireworks serverless
  (`GUARDIAN_FALLBACK=standin`), so a call is never left unguarded.

Run it: `line/README_live.md`. One-command demo with preflight: `bash demo/run_demo.sh bank_fraud`.

## AI-voice detector (`voice/`)

Frozen wav2vec2 XLS-R 300M features (layer 9) + a calibrated logistic head, scored on the first seconds of caller speech after
our 8 kHz phone path. On **In-the-Wild, 25 speakers never seen in training (2,829 clips): EER 1.27% (95% CI 0.85-1.77),
AUC 0.9991**, 98% of AI clips caught at a 1% false-alarm rate. Details: `results/voice_v1.md`.

**Live check (5 real calls, 2 Nov 2026; a small sample, not a benchmark):** on our own AI scam calls it flagged the synthetic
caller on 4 of 5 (scores 0.96-0.9995 for Azure and Google voices) and missed one Cartesia voice (0.41). Logs:
`recordings/<call>/events.jsonl`, `results/ledger-numan.md`.

## Real-audio check (`realset/`, supporting)

301 real recorded calls (181 NCSU scam robocalls, public domain; 120 HarperValleyBank calls, CC BY 4.0) pushed through the same
8 kHz phone path and live ASR, scored by 7 models with a seeded dev/test split (`results/realset.md`, `results/realset_headline.md`).
It checks the real phone path and false alarms on real benign speech: our guardian ties Kimi K3 there (25% vs 28% on the test
split) at a third of the latency (0.86 s vs 2.3 s median). **Scope:** the scam side is one-sided recorded robocalls, and robocall
screening is not what Emberz is for (carriers already block or label robocalls; a recorded pitch has no conversation to read).
Our guardian rates most of them only "watch" and catches 52% vs 98% for Kimi K3; we report it as a known limit.

## NVIDIA + Fireworks

- **Nemotron Speech Streaming 0.6B** hears every live call (local, INT8, 160 ms chunks).
- **Nemotron 3 Ultra** wrote the held-out test calls; **Nemotron 3.5 Super** was also fine-tuned as a guardian (nemo-v1);
  **Nemotron Lightning 3.5** is a baseline and the live fallback.
- **Fireworks**: Training API (LoRA), sampler + deployment serving, and every baseline model.

## Repo map

| Path | What |
|---|---|
| `callguard/` | Guardian: scenario factory, training, eval harness, results, console (Matt) |
| `line/` | Live phone line: call flow, ASR, brain, policy, caretaker, console hub, recorder |
| `voice/` | AI-voice detector: features, head training, eval, live scorer |
| `realset/` | Real-audio test set: selection, phone-path transcription, scoring, metrics |
| `callers/` | AI test callers (SignalWire personas) and human replay clips |
| `contracts/` | Shared pydantic schemas for every event that crosses a boundary |
| `demo/` | `run_demo.sh` and the video screens (`demo/screens/index.html`) |
| `results/`, `docs/` | Results, charts, business value (`docs/business_value.md`) |
| `PLAN.md` | Plan and every decision with its reason |

Setup: `uv venv && uv pip install -r requirements.txt`, copy `.env.example` to `.env`. Tests: `.venv/bin/python -m pytest -q tests` (557 pass). Secrets live only in `.env` (not committed).

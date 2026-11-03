# Emberz hackathon video: SPEC

Fireworks hackathon submission. Brief: "Demo video under 120 seconds: explain the problem, demonstrate your work, and show the measured result."
Judging: benchmark vs a NAMED frontier baseline is 40% of the score, business value 25%, technical 20%, presentation 10%.

**The one thing:** Emberz listens to Grandpa's phone call, spots the scam before the caller gets the code, warns him privately and brings in family, and our fine-tuned guardian beats frontier baselines on held-out calls.

1920x1080, 30 fps, total 108 s. No voiceover. The footage keeps its own sound (the real call on speakerphone).

## Timeline (global seconds)

| Scene host | File | Start | Dur | Job |
|---|---|---|---|---|
| `intro` | compositions/intro.html | 0 | 13 | Hook, problem, reveal |
| `footage` (root `<video>`) | assets/call_footage.mp4 | 13 | 68 | The real call, left panel 608x1080 at x=0 |
| `demo` | compositions/demo.html | 13 | 68 | Transcript + guardian feed (right) + overlays on the footage (left) |
| `results` | compositions/results.html | 81 | 22 | Measured results vs named baselines |
| `endcard` | compositions/endcard.html | 103 | 5 | Name, team, stack, repo |

Scene roots are 1920x1080, `#root` with `inset:0`. The demo scene is transparent over x 0-608 (the footage shows through) and owns everything else.

## Look

- Palette (CSS vars, copy into each scene): `--bg:#0b0d12; --panel:#151922; --ink:#f2f4f8; --dim:#8a93a6; --red:#ff4d4f; --amber:#ffb347; --green:#3ddc84; --blue:#4da3ff;`
- Amber `#ffb347` is the Emberz brand colour: ration it (the name, "ours" in charts, the payoff).
- Red = scam caller, green = Grandpa (the protected person), blue = caretaker, amber = Emberz/guardian.
- Type: Inter only (bundled 400/700/900). Titles 64-110 px, body 28-40 px, footnotes 20-22 px (never under 20 px).
- Worlds: intro sits on real footage stills (assets/still_*.jpg, 1080x1920 portrait); demo is the product console (dark panels) beside the footage; results is a colour flood to amber ground `#ffb347` with near-black ink `#14100a` (charts in black + white); end card back to dark with the amber mark.
- BANNED: dark void with tiny floating things, glow/particles, three cards in a row, bullet slides, every element fading+sliding up, breathing/floating loops, titles that keep moving after they land, docked logo, the words "unlock / seamless / revolutionize / game-changer".
- Motion: arrivals `expo.out`/`power4.out` 0.4-0.6 s; exits 40% shorter; vary entrances (per-word blur-in, line blur-in, count-up, wipe). Landed text holds dead still. Reading time >= 0.5 s + 0.3 s/word.
- GSAP only, `fromTo` inside scenes, timeline registered as `window.__timelines["<id>"]`, no Date/Math.random/timers.

## Facts (use verbatim; nothing else)

Problem: FBI IC3 2025 Internet Crime Report: people 60+ filed 201,266 complaints and reported $7.7 billion in losses (up 59% from 2024). Source line: "FBI IC3, 2025 Internet Crime Report".

How it works (the call path): Caller -> guarded number (Telnyx) -> "Hears it": NVIDIA Nemotron streaming speech recognition -> "Judges every turn": our guardian, a LoRA fine-tune of Qwen3.6-35B-A3B trained with the Fireworks Training API, served on Fireworks -> acts: a private coach whisper only Grandpa hears, a phone call to the caretaker (press 1 listen, 0 talk, 9 end the call), a live console. Plus an AI-voice check (wav2vec2 head + reply timing).

Results (source: emberz_submission.pdf, Tables 5, 6, 8; held-out test: 257 calls written by NVIDIA Nemotron 3 Ultra, a different model family than the training writer, never trained on; 148 risky, 79 benign, 22 rule-flip pairs; Wilson 95% CIs):

| System | Caught before the ask | False alarms | Caretaker rules followed |
|---|---|---|---|
| **Emberz guardian v2 (ours)** | 92.6% (137/148, CI 87-96) | 17.7% (14/79, CI 11-28) | 19/22 |
| Kimi K3 (frontier) | 89.9% (133/148) | 22.8% (18/79) | 4/22 |
| Rune (Fireworks, zero-shot) | 92.6% (137/148) | 38.0% (30/79) | 5/22 |
| DeepSeek V4.1 Flash (frontier) | 85.1% (126/148) | 31.6% (25/79) | 3/22 |
| Untuned Qwen3.6-35B-A3B (our base) | 89.2% (132/148) | 53.2% (42/79) | 3/22 |

Calls handled correctly (all 257; risky alerted before the ask, harmless never alerted), Table 6: ours 90.3% vs Kimi K3 83.7% (+6.6 pts, McNemar p = 0.003), Rune 83.3% (p = 0.005), DeepSeek V4.1 Flash 79.4%, untuned base 75.5% (+14.8 pts, p = 4e-10). Scenario-clustered bootstrap confirms every gain.
Alerts a median of 4.5 turns before the ask.
Out of distribution (BothBosu public set, another model family): 98/100 scams caught, 0/100 false alarms.
AI-voice detector (In-the-Wild, 25 unseen speakers, phone path): EER 1.27%; catches 98% of AI clips at 1% false alarms.
Training: LoRA rank 16 on Qwen3.6-35B-A3B with the Fireworks serverless Training API; four training runs cost about $23.

Problem extras (FBI IC3 2025): AI-related fraud cost people 60+ about $112K per complaint, ~3x the $38K average.

Team: Numan + Matt. Built on Fireworks + NVIDIA Nemotron. Repo: github.com/matt-ceran/fireworks-AMD

## The call (demo scene, times in DEMO-LOCAL seconds, 0 = global 13)

Real call, 2 Nov 2026, 15:48 PST, AI scam caller (an AI voice agent we built as the test caller) dialing the guarded number; Grandpa's real phone. Footage is one unedited take.
Mapping: demo_local = call_ms/1000 - 24.5. Footage: the protected person ("Grandpa", blue fleece, sitting) until ~38.5; the camera pans to the caretaker (grey sweater, standing elsewhere) 38.5-40.5; caretaker on his phone 40.5-55.5 (hears the menu "press one to listen... zero to talk or nine to end the call" ~42.5-48.5, listens from ~49.5); quick pan back to Grandpa 55.5-56.5; caretaker 57-68 (says "Grandpa, don't do it!" out loud at ~61.1, then presses 9).

Transcript turns (speaker, start, end, text):
- grandpa 1.26-2.22 "Hello"
- caller 5.60-10.34 "Hello. This is Daniel from the Fraud Prevention Team at First Meridian Bank."
- caller 10.60-11.70 "Am I speaking with Margaret?"   (persona name; keep as spoken)
- grandpa 11.98-12.86 "Yes, that's me."
- caller 17.22-27.52 "Thank you, Margaret. We've flagged a charge of $842 at an electronics store in Miami on your card ending in 4417. Did you make this purchase?"
- grandpa 28.30-31.00 "Uh, I don't... I don't think so. I can't remember."
- caller 35.24-42.62 "I understand. Your money is safe as long as we act now. The case number for this issue is F20931."
- grandpa 44.26-44.76 "Oh, okay."
- caller 48.28-54.38 "The charge will go through in just a few minutes. To block it, I'm sending a verification code to your phone right now."
- grandpa 55.44-56.34 "Okay, let me see."
- caller 60.08-61.00 "I've just sent the code."
- caretaker (in person, not on the line) 61.1-62.4 "Grandpa, don't do it!"

Guardian events:
- 10.48 AI-voice score 99.95% (voice model, wav2vec2)
- 30.68 AI-voice 100% (reply timing)
- 33.80 RISK ALERT, urgent: "Fake bank call" (scam.impersonation_bank_gov). Note: "Caller claims a suspicious charge. This is a common scam tactic."
- 34.71 COACH WHISPER to Grandpa only: "Don't confirm anything. Hang up and call the number on your card."
- 35.30 Caretaker's phone ringing (Emberz dials the caretaker on the alert)
- 51.13 Caretaker listening (monitor). "Press 0 to talk, 9 to end the call"
- 58.92 RISK ALERT, urgent: "Asking for a code" (disclosure.credentials_otp). "Caller is sending a code. This is a scam. Do not share it."
- 63.21 Caretaker pressed 9: Emberz hung up on the caller
- 64.64 COACH WHISPER: "Don't read any code. Hang up and call your bank."
- 66.5 Call ended. Code never shared.

# Second Ear: plan

Working name, change it if you like. Fireworks hackathon, November 2026. Numan and Matt.

## The pitch in one paragraph

Give your kid, your grandma or yourself a **guarded phone number** instead of the real one. Every call to it passes through our cloud. A small model fine-tuned on Fireworks reads the live transcript and stays silent on normal calls. It speaks up when a call turns into a scam, a stranger asks a kid for their address, or the caller is an AI voice. The protected person hears a quiet tip that the caller can't hear. The caretaker (mom, an adult child) gets an alert, can open the live transcript and join the call with one tap, and gets a summary afterwards. The caretaker sets the rules in plain words, such as "never share the door code" or "tell me about any talk of money".

**The claim we prove:** a small model fine-tuned on Fireworks using only generated calls catches more risky calls *before the ask* (money, codes, address, meeting up) than general safety models and a frontier model. It also raises fewer false alarms on normal calls, decides faster and costs much less per call-minute. A second, separately scored result: we detect AI-voice callers on real phone-quality audio, measured honestly against off-the-shelf detectors and Telnyx's built-in deepfake check.

## 1. Why this, and why now

- **Phones won't let apps hear calls.** iOS gives third-party apps no access to live call audio (Apple developer support confirmed this in 2026). Android 10 blocked call-audio capture for normal apps, and Google Play has banned recording calls through the Accessibility API since May 2022. On-call protection therefore lives in the phone maker's own dialer, at the carrier, or on a cloud phone line. **We use the cloud line, the same design Aura uses** (carrier forwarding to a cloud assistant).
- **Scam detection alone is taken.** Google ships on-device scam detection on Pixel 9+ and, since February 2026, on Galaxy S26 (English, opt-in). It alerts only the person on the call.
- **What nobody combines** (as of 2026-11-02):
  1. A live transcript and mid-call alerts sent to a *different* person, the caretaker. OnGuardAI (US launch August 2026) texts a family member red flags, but keeps no transcript, has no AI-voice detection, no caretaker rules and nothing for kids. AT&T's Digital Receptionist shows its transcript only to the account holder.
  2. Rules the caretaker writes in plain words. Everyone else uses fixed vendor categories.
  3. Checking what is *said* on kids' voice calls. Bark, Gabb, Troomi and Pinwheel cover texts, contact allow-lists and call logs, not call content.
  4. Detecting an AI voice agent calling, combined with what the call is about. Hiya and Truecaller check the voice only.
  5. A small open model with a published, reproducible eval. Every consumer product is closed, and its accuracy numbers are vendor claims.
- **AI callers are already illegal without consent.** The FCC's February 2024 ruling (FCC 24-17) says AI-generated voices count as "artificial" voices under the TCPA, so an AI calling without prior consent is likely an illegal call. That makes AI-voice detection a real signal, not a gimmick. AI is not automatically bad (pharmacy reminders, appointment bots), so an alert fires when an AI voice comes **together with** an unknown caller and a risky pattern.
- **The privacy angle sells it.** The real number never gets given out, so it stays out of data-broker lists, leaks and strangers' hands. You can use a different guarded number for school, online sellers or dating, and turn one off without changing phones.

## 2. How a call works

```
 caller dials the guarded number (Telnyx)
   |
   v
 answer + notice ("This call is protected by Second Ear
   |               and may be transcribed")
   |
   |-- media stream, both tracks, one WebSocket --> our server
   |      |-- streaming ASR per side ("Caller" / "Grandma")
   |      |-- AI-voice detector (first 2-4 s + reply timing)
   |      |-- guardian model on Fireworks (rules + turns -> JSON)
   |      '-- policy: silent | log | coach | alert | urgent
   v
 conference with the protected person's real phone
   |-- coach: whisper only the protected person hears
   |-- caretaker alert call: "Press 1 to listen" -> listen, barge in
   |-- live console: transcript, risk lines, alerts (web, phone-friendly)
   '-- after the call: summary to the caretaker
```

- **Incoming calls:** the protected person gives out the guarded number. Forwarding their real number to it would loop back (the guarded number rings that same phone). Guarding an existing number needs a second line on the phone (eSIM) or a small calling app. That's a post-hackathon feature.
- **Outgoing calls** (later): a calling app or a dial-in, so the guarded number shows as caller ID.
- **Telnyx limits:** only **one** media stream per call, so we use `both_tracks` into one WebSocket and fan out on our server. Telnyx's `deepfake_detection` on Answer (Resemble Detect, $0.01 per call, verdict in under 4 s by webhook) runs on the same calls as our commercial baseline.
- **Alerts by voice call, not SMS.** US texting needs A2P 10DLC registration, which takes days. A spoken alert call avoids it and lands better on stage.
- **Verify in hour 0:** conference whisper and barge (Telnyx `supervisor_role`), Fireworks streaming speech-to-text (otherwise use Telnyx transcription), and end-to-end latency.

## 3. The guardian model (the Fireworks fine-tune, headline result)

**Input:** the caretaker's rules (system prompt) plus a sliding window of the last N turns with speaker labels, plus the audio side's AI-voice score and reply timing. It is scored after every final transcript segment.

**Output:** short JSON per scored turn.

```json
{"risk": "none|watch|alert|urgent",
 "category": "scam.impersonation_bank_gov | scam.tech_support | scam.family_emergency | scam.prize | scam.romance_investment | scam.payment_ask | disclosure.location_school | disclosure.credentials_otp | disclosure.financial_id | disclosure.home_alone | grooming.secrecy | grooming.isolation | grooming.gifts_flattery | grooming.photo_request | grooming.meeting_request | sexual_explicit | vulgar | threat_extortion | self_harm | none",
 "severity": 0,
 "source": "caller|protected_person",
 "stage": "hook|trust|pressure|ask",
 "evidence": [{"turn": 7, "quote": "exact words from the transcript"}],
 "action": "none|log|coach|notify_caretaker|suggest_hangup|urgent_alert",
 "coach_line": "15 words max, to the protected person",
 "caretaker_note": "20 words max"}
```

- **Scored categories:**
  - **Headline:** scam, disclosure and grooming tactics, plus `none`. These carry the main claim.
  - **Second slice, scored in parallel:** vulgar language, threats and self-harm. Llama Guard 3 is strong on these, and OpenAI's free omni-moderation is an extra baseline, so we report this slice separately and honestly. It is not part of the headline.
- **A rule layer after the model:** smooth alerts over turns, add cooldowns so one call doesn't alert ten times, and apply hard caretaker rules such as an allowed-contacts list.
- **Rules come from the caretaker.** Training mixes many different rule sets (kid profiles, elder profiles, "allow calls from Grandpa", "never share the door code") so the model follows the rules it is given instead of memorizing one set.
- **Base model.** Start with **Qwen3-4B-Instruct-2507** (managed SFT LoRA, small and fast). Alternatives:
  - Qwen3.6-35B-A3B on serverless training: about $2.14 per million training tokens, and the untuned base is also served serverless, which gives a free same-weights baseline.
  - Nemotron Nano 3 30B-A3B, which is tunable and good for the sponsor story.
  - Muse Glimmer 30B: managed SFT is free until 30 November.
- **Serving.** A trained LoRA runs only on an **on-demand deployment** (dedicated GPU), never serverless. Start it for evals and the demo only, and count its GPU-hours in the cost numbers.
- **Don't fine-tune a guard model.** Llama Guard 3 and gpt-oss-safeguard are on Fireworks (on-demand only) but can't be fine-tuned. Their fixed categories judge one message at a time, while scams and grooming build up over many turns.

## 4. Data factory (generated calls only)

1. **Scenario grid:** protected person (kid 8-12, teen, elder, adult) x caller type (scammer by family, stranger, groomer at the tactics level, real bank, doctor's office, school, family member, friend, AI appointment bot) x rule set x persona and mood.
2. **Self-play:** a large serverless Fireworks model plays the caller from a hidden plan (stages, the exact ask turn, tactics). Another plays the protected person. Labels come from the caller's plan, which gives turn-level stage and ask labels for free.
3. **Benign twins:** normal calls that reuse scam words ("your bank", "verification code", "gift"), such as a real bank fraud line, a pharmacy refill or a grandkid who really needs a ride. A 2024 study (Shen et al., arXiv 2409.11643) found keyword detectors score near 1.0 on public scam sets but drop to recall 0.16 when benign calls use the same words. Twins are what make our false-alarm number mean something.
4. **Disclosure:** the protected person giving away their address, school, one-time codes or home-alone schedule. No public dataset exists, so it has to be generated.
5. **Grooming stays at the level of tactics only:** secrecy, isolation, gifts, photo requests, meeting requests. **Hard rule: no sexual content involving minors is ever generated.** A conversation stops at the request, and the label comes from the plan.
6. **ASR noise:** render a share of the calls with Kokoro TTS (Apache-2.0), convert to 8 kHz phone audio, and run them through the same streaming ASR we use live, so training text carries real transcription errors.
7. **Rule-flip pairs (about 10% of calls):** the same words with different rules. "What street do you live on?" from an unknown caller should alert; from Grandpa, who is on the allowed list, it should stay silent. This proves the model follows the caretaker's rules.
8. **One scam type held out entirely** (for example romance/investment) to measure recall on a scam it never saw.
9. **Splits:** train / dev / **held-out test made by a different generator model and different persona seeds**. Check for overlap with every real eval set.
10. **Volume:** about 3,000 calls (about 30k scored turns) for v1. Add 1,000 targeted at v1's mistakes for v2 (the adversary round).

## 5. How we score it

**Metrics, fixed before we look at any test set:**

| Metric | Definition |
|---|---|
| Catch-before-ask | % of risky calls flagged at or above `alert` before the labeled ask turn. **Headline.** |
| False alarms | Alerts per 100 benign calls, hard negatives included. **Must stay low.** |
| Turns to detect | Turns from the start of the risk to the first alert |
| Category F1 | Macro-F1 over categories |
| Rule-flip accuracy | % of rule-flip pairs where the decision follows the rules |
| Unseen scam type | Recall on the held-out scam type |
| Evidence match | % of evidence quotes that appear exactly in the transcript (catches hallucination) |
| Speed | p50 and p95 per scored turn, end to end, in milliseconds |
| Cost | $ per call-minute, including the deployment's GPU-hours at a stated volume |

**Rules for honest numbers:** labels for the real and role-play sets are committed to git **before** any model is run on them. Every number shows its sample size (n) and a 95% interval, and the real-set intervals will be wide. B4 comes from a different model family than the generator, so the frontier baseline isn't grading its own style.

**Baselines (run in the first hours):**

| | Baseline | Why |
|---|---|---|
| B0 | Keyword rules + PII regex | Shows the keyword trap: looks perfect on public sets, fails on benign twins |
| B1 | Llama Guard 3 8B on Fireworks (on-demand) | The standard safety guard, last turn vs a 6-turn window |
| B2 | gpt-oss-safeguard-20b on Fireworks (on-demand) with the caretaker's rules | **Strongest baseline.** Takes written rules at run time; slower and more expensive |
| B3 | The same base model, untuned, same prompt, on the same multi-LoRA deployment | Clean fine-tuning comparison |
| B4 | A frontier serverless model on Fireworks, zero-shot | Quality ceiling; also the teacher |

**Test sets, always reported separately:**

1. **Synthetic held-out test** (different generator): the main numbers, with 95% intervals.
2. **Real scam calls:** BothBosu `youtube-scam-conversations` (20 real transcribed calls, Apache-2.0). Score only, never train.
3. **Grooming transfer:** a PAN-12 subset (Zenodo). It's online chat with adult decoys, not phone calls, and we say so.
4. **Our role-played phone calls:** about 40 calls over Telnyx between teammates and consenting friends, covering scam, disclosure, grooming tactics, AI caller and benign hard negatives.

**Charts for the README:**
- Catch-before-ask vs false alarms, one point per model.
- Speed vs quality.
- A per-category breakdown.
- v1 vs v2 after the adversary round.

## 6. AI-voice detector (parallel workstream, second result)

- **Model:** reimplement a simple recipe (no EmberDial code or data). Take a frozen wav2vec2 front end (`jakeBland/wav2vec-vm-finetune`, Apache-2.0, or XLS-R), pool the first 2-4 s of caller audio, then train a small logistic head. Retraining takes seconds. If the low layers don't separate AI from human voices, try deeper layers or fine-tune W2V2-AASIST (MIT).
- **Training data:**
  - ASVspoof 5 (ODC-By). It includes 8 kHz narrowband codec conditions.
  - In-the-Wild (CC-BY-SA).
  - Kokoro TTS renders of our scripts (Apache-2.0).
  - Human audio over the same Telnyx path, so the model learns the voice, not the phone line.
  - **Not ElevenLabs output:** its use policy bans using output to train any model. Avoid Chatterbox too: its watermark becomes a shortcut.
- **Real test:** build 3-5 voice agents ourselves on Vapi and Retell (about $10 in free credit each) with different voices. They call **our own** Telnyx number, recorded at 8 kHz. Commercial-voice calls are for evaluation only. Never record third parties' demo lines.
- **Expect off-the-shelf to collapse** and show it. W2V2-AASIST scores 0.22% EER in its own domain, but 11-46% on newer sets. Public detectors lost about 48% AUC on 2024 real-world deepfakes (Deepfake-Eval-2024). Training with codec augmentation helps.
- **Timing cue:** people reply after about 0.2 s on average (Stivers et al. 2009); voice agents take about 1.3-2.4 s. Add reply gap, AI self-disclosure and missing "um"s as features. Report audio only, timing only and combined.
- **Baselines:** off-the-shelf W2V2-AASIST, and Telnyx `deepfake_detection` on the same calls.
- **How it is used:** an AI voice alone is at most `watch`. AI voice plus an unknown caller plus an ask escalates. A pharmacy reminder bot lights the badge but sends no alert. The FCC also requires artificial-voice calls to identify who is calling at the start (47 CFR 64.1200(b)). A missing identification is itself a signal.
- **Its own workstream, built in parallel** (owner: Numan), with its own results section and charts. Target a real phone test set of about 50 AI-agent calls (5 agents x 10) and about 50 human calls over the same Telnyx path. Claim only what we measure on that set, with n and intervals.
- **Metrics:** EER and AUC on our phone set, % detected within 2 s and 4 s, false alarms on human callers.

## 7. The demo (go all out)

Three phones and one big screen. The console shows both sides of the transcript, live risk lines (AI voice, scam, danger info) and a mirror of the caretaker's phone.

1. **An AI voice scams Grandma.** Our own AI caller dials Grandma's guarded number as "her bank". About 2 s in, the console shows **AI voice detected**. When the script reaches pressure, Grandma hears a whisper only she can hear: "This may be a scam. Say you'll call the number on your card."
2. **The caretaker gets pulled in.** Mom's phone rings: "Grandma may be on a scam call. Press 1 to listen." She listens live, then barges in.
3. **A stranger calls the kid.** "Don't tell your parents. What's your address?" sets off a disclosure alert and Mom gets the live transcript.
4. **Same words, different rules.** Grandpa, who is on the allowed list, asks the kid "what street are you on?". It stays silent.
5. **An AI that isn't a scam.** A pharmacy reminder bot calls. The AI badge lights, no alert.
6. **A normal call.** Grandma's friend calls about lunch. Nothing fires.
7. **After the calls:** summaries arrive. Close on the overlay "alerted N turns before the ask", then the chart.

- **Judge participation:** hand a judge a scam card and have them call the guarded number from their own phone.
- **Fallback:** a recorded full run, in case venue reception fails.

## 8. Consent and legal notes (goes in the README)

- **Every call starts with a notice.** It meets all-party-consent states such as California (Penal Code 632.7, *Smith v. LoanMe*, 2021) and costs about 3 seconds.
- **Opt-in by the protected person.** Federal law needs one party's consent (18 USC 2511(2)(d)). For elders and other competent adults, **they** sign up and choose what to share. A caretaker cannot consent for a competent adult.
- **Kids:** a line the parent owns. The amended COPPA rule (enforced since April 22, 2026) treats a child's voice as personal information, so we need verifiable parental consent plus written retention and security policies. "Vicarious consent" (*Pollock v. Pollock*, 6th Cir. 1998) is a backup argument, not the plan.
- **Privacy by default:** no voiceprints stored, and call data deleted after 7 days. Alerts and summaries first. The live transcript is opt-in, or opens only after an alert. Keep data briefly.
- **Hackathon data:** generated calls, public datasets within their licenses, and calls between consenting teammates and friends. **No EmberDial customer data, ever.**

## 9. Who does what

Suggested split; swap freely. Assumes about 24 hours. Coding agents can run parts in parallel.

| Workstream | Owner | Includes |
|---|---|---|
| **Model** | Matt | Data factory, guardian fine-tune, eval harness, baselines B0-B4, charts |
| **Line** | Numan | Telnyx guarded number, media stream, ASR, whisper and alert calls, live console, summaries |
| **Voice** | Numan | AI-voice detector, own Vapi/Retell test agents, Telnyx deepfake baseline |

| Hours | Matt | Numan |
|---|---|---|
| 0-2 | Fireworks setup and tunable-model check; freeze metrics and JSON schema; **B0 + B4 on BothBosu** | Telnyx account or sub-account, number, Call Control app; answer + notice + stream to WebSocket; check whisper and barge |
| 2-6 | Factory v1: grid, self-play, twins, disclosure; 1,000 calls | Streaming ASR per side; console v0 (live transcript); AI-voice training data ready |
| 6-10 | 3,000 calls; **SFT v1**; deployment; B1-B3 | Guardian wired into the call loop; whisper; caretaker alert call; first AI-voice head |
| 10-14 | Score v1 on every test set; pick weak spots | Vapi/Retell agents call our number; role-play test calls (~40); AI-voice eval + Telnyx baseline |
| 14-18 | **Gate:** v1 must beat the untuned base (B3); if not, fix the errors and retrain once. Then the adversary round: 1,000 targeted calls, **SFT v2**, re-score | Summaries; console polish; full rehearsal |
| 18-24 | Charts, README results | 60-second video, fallback recording, submit |

## 10. Repo layout (proposal)

```
PLAN.md          this file
README.md        pitch, results, charts, legal notes (fill at the end)
factory/         scenario grid, self-play generators, labels, splits
guardian/        prompt, JSON schema, training-set builder, Fireworks SFT + deploy scripts
voice/           AI-voice detector: features, head training, eval
line/            Telnyx Call Control server, media stream, ASR, policy, whisper/alert calls
console/         live caretaker console (web)
eval/            scorers, baselines B0-B4, charts
data/            gitignored; dataset cards only in git
```

Secrets go in `.env` (gitignored). See `.env.example`.

## 11. Cuts and risks

**Cut in this order if time runs short:** the adversary round (v2), then B1 Llama Guard, then the AI-voice head (keep the timing cue and Telnyx's detector), then the barge-in. **Never cut** the B2/B3/B4 baselines on the held-out set, or the benign-twin false-alarm number. They are the whole proof.

| Risk | Fallback |
|---|---|
| Open models refuse to play the scammer in self-play | Frame as a safety-training simulation; use a model that agrees; seed from BothBosu scripts |
| Phone-line audio fails our AI-voice head | Report the collapse honestly; lead with timing + Telnyx detector |
| Latency too high for live whisper | Score every 2nd turn; smaller base; keep the deployment warm |
| Whisper/barge not available as expected | Caretaker alert call + console only; coach line on the console |
| Generator refuses to write grooming calls | If still unreliable by hour 6, grooming becomes eval-only (PAN-12 subset + role-play) |
| We lose to the frontier model on real calls | Say so; pitch quality per dollar and speed, where a small model wins |
| Real test sets are small | Headline the synthetic held-out with intervals; report real sets as separate slices |
| On-demand deployment cost | Spin up only for eval windows and the demo |

## 12. Decisions needed

1. **Telnyx account:** a separate account or sub-account for the hackathon, not EmberDial production? (Recommended.)
2. **Spend go-ahead:** a few numbers (about $1 a month each), calls at fractions of a cent per minute, about $0.01 per call for Telnyx deepfake detection, plus Fireworks on-demand GPU time.
3. **Name:** "Second Ear" or something else.
4. **Hackathon length:** the schedule assumes about 24 hours. Tell us the real deadline.

## Sources (checked 2026-11-02)

- Fireworks: [SFT](https://docs.fireworks.ai/fine-tuning/supervised-fine-tuning) · [tunable models](https://docs.fireworks.ai/fine-tuning/models) · [deploying LoRAs](https://docs.fireworks.ai/fine-tuning/deploying-loras) · [serverless training](https://docs.fireworks.ai/fine-tuning/training-api/serverless) · [Llama Guard 3 8B](https://fireworks.ai/models/fireworks/llama-guard-3-8b) · [gpt-oss-safeguard-20b](https://fireworks.ai/models/fireworks/gpt-oss-safeguard-20b)
- Telnyx: [media streaming](https://developers.telnyx.com/docs/voice/programmable-voice/media-streaming) · [deepfake detection release](https://telnyx.com/release-notes/deepfake-detection)
- Competitors: [Google Scam Detection on Galaxy S26](https://www.androidauthority.com/google-scam-detection-samsung-galaxy-s26-3643942/) · [Google fake call detection](https://techcrunch.com/2026/06/02/google-rolls-out-fake-call-detection-to-protect-against-ai-deepfake-impersonation-scams/) · [OnGuardAI](https://betakit.com/onguardai-launches-app-to-flag-scam-calls-in-real-time/) · [Aura Call Protection](https://www.aura.com/help/what-is-call-protection) · [Hiya AI voice detection](https://www.hiya.com/products/ai-voice-detection)
- Data and research: [BothBosu scam datasets](https://huggingface.co/datasets/BothBosu/Scammer-Conversation) · [Shen et al. 2024, keyword bias](https://arxiv.org/abs/2409.11643) · [PAN-12](https://zenodo.org/records/3713280) · [ASVspoof 5](https://zenodo.org/records/14498691) · [In-the-Wild, Muller et al. 2022](https://arxiv.org/abs/2203.16263) · [W2V2-AASIST](https://huggingface.co/SpeechAntiSpoofingBenchmarks/W2V2-AASIST) · [Deepfake-Eval-2024](https://arxiv.org/abs/2503.02857) · [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) · [ElevenLabs use policy](https://elevenlabs.io/use-policy)
- Law: [18 USC 2511](https://www.law.cornell.edu/uscode/text/18/2511) · [CA Penal Code 632.7](https://law.justia.com/codes/california/code-pen/part-1/title-15/chapter-1-5/section-632-7/) · [FCC AI voices under TCPA](https://www.commlawblog.com/2024/02/articles/fcc/fcc-clarifies-that-ai-generated-voices-are-subject-to-tcpa/) · [COPPA amendments](https://www.hunton.com/privacy-and-cybersecurity-law-blog/ftc-publishes-final-coppa-rule-amendments)

Not legal advice: get a lawyer's read before launching a real product.

# Real call sources and the 100+ real-call test set

Checked 2026-11-02. This file answers one question: can we show the guardian working on **real speech from real calls**, not only on LLM-written calls, with 100+ examples (PLAN.md 2d)?

**Short answer: yes, about 120 calls in about 6 hours, with honest limits.**
- About 63 scam calls. These are real robocalls (FTC and an independent honeypot set), 3 calls with real victims (FBI x2, Niagara police), about 20 real scammers talking to scam-baiters, and up to 5 AI scam-bot callers.
- About 56 benign hard negatives. 26 are real unscripted calls (family calls, teen calls, recorded consumer calls to real companies). 30 are real human speech in role-played bank, insurance, medical and booking calls.
- Every call goes through the same 8 kHz phone codec, the same Nemotron ASR and the same turn segmentation as the live line. A subset (about 50) is also replayed through the real line (SignalWire caller number to the Telnyx guarded number), so we can show that the offline numbers hold on the live path.
- **Thin spots:** only 3 calls have a real victim on the line. The scam-bait "victims" are knowing baiters. There are **no** ethical real recordings of a stranger targeting a child (see section 4).

---

## 1. Usable sources

"Eval-only" means we may score on it locally but must not publish its audio or text. "Redistributable" means the license allows republishing with attribution. Either way we commit IDs and labels only, plus fetch scripts (section 3).

### Scam and robocall

| Source | Link | Kind | Count | Two-sided | Real? | License / terms | Use | Time to get |
|---|---|---|---|---|---|---|---|---|
| NCSU Robocall Audio Dataset (FTC Project Point of No Entry), Prasad & Reaves TR-2023-1 | [github](https://github.com/wspr-ncsu/robocall-audio-dataset) | scam, robocall | 1,432 rows / 1,406 unique WAVs (1,378 en), about 9.9 h, Whisper transcripts in metadata.csv | No (caller only; 21 have the honeypot channel) | Real robocalls; TTS or recorded voice, not labeled | Data "in the public domain"; README is CC BY-ND | Redistributable (we still commit IDs) | Minutes (single WAVs via raw URLs) |
| International Robocalls Dataset (Altwlkany et al.) | [zenodo 21066049](https://zenodo.org/records/21066049) | scam, robocall | 677 WAVs (501 English), 8 kHz, 28 campaign clusters | No | Real, honeypot-captured, human-verified | CC BY-NC 4.0 | Eval-only | Minutes (audio.zip 155 MB) |
| FBI scam-call recordings: "Elder Fraud Telemarketing Scam Call" and "Debt Collection Fraud Call" | [fbi.gov audio repository](https://www.fbi.gov/audio-repository/elder-fraud-telemarketing-scam-call-recording-112718.mp3/view) (403 to curl; use a browser or the Wayback Machine), [YouTube q_qdC6grfIA](https://www.youtube.com/watch?v=q_qdC6grfIA) | scam | 2 calls | Yes (mono mix) | **Real victim and real fraudster** | US federal work, likely public domain (17 USC 105); page terms not checked | Likely redistributable; commit URL + labels | Minutes |
| Niagara Regional Police grandparent-scam call (via Meridian Credit Union; CHCH copy) | [meridiancu.ca page](https://www.meridiancu.ca/about-meridian/privacy-and-security/protect-yourself-from-fraud/hear-it-for-yourself-audio-recording-of-a-grandparent-scam), [YouTube 8hzU4Uf9ay8](https://www.youtube.com/watch?v=8hzU4Uf9ay8) | scam | 1 call, 4:08, full speaker-labeled transcript | Yes (mono mix) | **Real victim and real fraudster**; contains the ask ("$9,550") | Page copyright Meridian; police-released; no reuse terms | Eval-only; a short demo excerpt is a fair-use argument | Minutes |
| CC BY scam-baiting videos (Salty Dogg, IRLrosie, Wenja92 Scam Detective, Nellie's Scam Baking, Callcenterbuster, TROLLphone, Earthshine024) | [YouTube CC filter](https://www.youtube.com/results?search_query=scambaiting&sp=EgIwAQ%253D%253D) | scam | 1,196 candidate videos, 609 h; about 20-25% off-topic; livestreams hold many calls | Yes (mono mix: scammer on the phone line, baiter on a mic) | Real scammers; the victim is a knowing baiter (or the Lenny bot: exclude) | Uploader's CC BY (13 of 13 checked); scammer voice and PII are third-party; YouTube ToS bans downloading | Eval-only by default; short CC BY clips usable in the demo with attribution and PII bleeped | 1-3 h for 20 calls |
| Standard-license scam-baiting channels (More Kitboga, Kitboga, Scammer Payback, ModderPaulScambaits, Pleasant Green, BBB Calls Scammers) | e.g. [@MoreKitboga](https://www.youtube.com/@MoreKitboga/videos) | scam | More Kitboga 198 videos / 987 h; Kitboga 559 / 266 h; Scammer Payback 418 / 144 h; ModderPaul 484 / 253 h | Yes (mono mix) | Real scammers, fake victims | Standard YouTube license; ToS bans downloading | Eval-only, local, fair-use argument | Hours to a day |
| Kitboga "Say THIS to Destroy AI Bot Callers" | [YouTube fhi6JU5pOJk](https://www.youtube.com/watch?v=fhi6JU5pOJk) | scam + AI voice | 5 chapters, 17 min | Yes (mono mix) | Real calls; the caller is said to be an AI scam bot (not verified per call) | Standard YouTube license | Eval-only | About 1 h |
| Kaggle "YouTube Scam Phone Call Transcripts" (rivalcults) | [kaggle](https://www.kaggle.com/datasets/rivalcults/youtube-scam-phone-call-transcripts) | scam (text) | 243 openings, 219 video IDs | Text flattened, no speakers | Real scammers | CC0 | Redistributable (text) | Done (local) |
| BothBosu youtube-scam-conversations | [HF](https://huggingface.co/datasets/BothBosu/youtube-scam-conversations) | scam (text) | 20 two-sided transcripts | Yes | Real scammers, baiters | Apache-2.0 card | Redistributable (text) | Done (already test set 2) |

### Benign hard negatives

| Source | Link | Kind | Count | Two-sided | Real? | License / terms | Use | Time to get |
|---|---|---|---|---|---|---|---|---|
| CallHome English (TalkBank; HF remaster BrunoHays/callhome-telephony) | [talkbank](https://talkbank.org/ca/access/CallHome/eng.html), [HF](https://huggingface.co/datasets/BrunoHays/callhome-telephony) | benign (family) | 120-176 calls; `en` 140 transcribed excerpts, 20.3 h (241 MB); `en-full` 176 stereo calls, 76.5 h | Yes (en-full has one channel per speaker; en is a mono mix) | **Real, unscripted**; organic money, deposit and doctor talk (e.g. call 4289 MLM-style money scheme, 4886, 5373) | CC BY-NC-SA 4.0 + TalkBank Ground Rules: no redistribution, no use inside models, LLM uploads only with non-storage | Eval-only; no training; LLM baselines only with no retention | Minutes |
| CallFriend English (TalkBank) | [eng-n](https://talkbank.org/ca/access/CallFriend/eng-n.html) | benign (family; includes teens) | eng-n 41 + eng-s 60 calls; teen callers en_4504 (14), en_6402 (17) | Yes | **Real, unscripted** | Same as CallHome | Eval-only | Minutes (free TalkBank login) |
| UniData call-center free sample | [HF ud-nlp/call-center-audio](https://huggingface.co/datasets/ud-nlp/call-center-audio) | benign (customer service) | 5 calls (Amazon, Paramount+, Uber Eats, Spin, Prezzee gift cards) | Yes (mono mix with IVR) | **Real consumer calls to real companies**; identity-verification and gift-card talk | CC BY-NC-ND 4.0 | Eval-only | Minutes |
| HarperValleyBank (Gridspace-Stanford) | [github](https://github.com/cricketclub/gridspace-stanford-harper-valley) | benign (bank) | 1,446 calls, 23 h, 59 speakers | **Yes, separate agent and caller WAVs** | Real speech, role-played from templates (about 700-word vocabulary) | CC BY 4.0 | Redistributable | Minutes (selective raw files) |
| AppTek Call-Center Dialogues | [HF](https://huggingface.co/datasets/apptek-com/apptek_callcenter_dialogues) | benign (bank, insurance, health, retail...) | 873 calls, 128.6 h, 16 domains, 14 accents | **Yes, one WAV per speaker** | Real speech, role-played, spontaneous | CC BY-SA 4.0; card: for evaluation only, not training | Eval-only by intent (license would allow redistribution) | Minutes (per-accent folders) |
| SpokenWOZ (profile domain) | [project](https://spokenwoz.github.io/SpokenWOZ-github.io/) | benign (booking; agent asks ID, email, plate) | 5.7k dialogues, 249 h; test 1,000 WAVs | Yes (merged WAV; layout unconfirmed) | Real speech over phone calls, role-played | CC BY-NC 4.0 | Eval-only | Minutes (text 52 MB; single WAVs 5-14 MB) |
| PriMock57 | [github](https://github.com/babylonhealth/primock57) | benign (doctor) | 57 consultations | **Yes, separate doctor and patient WAVs** | Real speech, role-played patients | CC BY 4.0 | Redistributable | Minutes (Git LFS) |

### AI voice and human voice (for V1)

| Source | Link | Count | Real? | License | Use | Time |
|---|---|---|---|---|---|---|
| NCSU robocalls + International Robocalls (above) | above | 1,378 + 501 English | Real deployed robocalls; TTS vs recorded human **unlabeled**, so we hand-label a subset | PD / CC BY-NC | Eval | Minutes |
| Kitboga AI bot callers (above), AI Kitboga (bot on the answering side) | [@AIKitboga](https://www.youtube.com/@AIKitboga/videos) | 5 segments; 8 bot calls, 4.2 h | Real deployed TTS over PSTN | Standard YouTube | Eval-only, local | Hours |
| ASVspoof 2021 LA eval, `pstn` condition | [zenodo 4837263](https://zenodo.org/records/4837263) | 25,938 trials (2,636 bona fide) | Synthetic + real, sent over real telephony | ODC-By | Eval | Hours (7.76 GB tar, so G1 VM) |
| Human class over the phone path | CallHome, CallFriend, HarperValley, AppTek channels above, plus the existing LibriSpeech/EdAcc replay set (`callers/human_replay/`) | hundreds of speakers | Real human | as above | Eval | Minutes |

### Not used, and why

- **Scam.AI 1,000-call subset:** needs a credit purchase (money, so Numan's call). Its 500 "spam" calls are predatory lead-gen calls, not benign. The rajkirant GitHub copy of it is an unofficial re-upload of CC BY-NC gated data: do not use it.
- **Jim Browning hacked call-center recordings, and the Karl Rock audio of a 13-year-old victim:** obtained by unauthorized access, and one involves a minor. Never use.
- **Trilogy Media predator stings:** sexual context involving minors. Excluded.
- **Not released or not obtainable:** Wood et al. (341 scam-bait transcripts), Shen et al. "Our-Real", the Korean police DPhone set, the Who's Calling? honeypot audio, the Turkish 100-call set, Scam.AI AI-voice audio.
- **Synthetic, so it can't count as "real":** BothBosu scam-dialogue and Scammer-Conversation, TeleAntiFraud 1 and 2, Robo-SAr, Paam1/ThaiScamCall, menaattia, sugiv legit side, the Kaggle call-center WAVs, CallScreenBench.
- **Unclear provenance or license:** sumitkant scam-nonscam tarballs (no card; the nonscam side is mock scripts), DynamicSuperb FraudRobocall (no license, and only one class per task), BaekRok (Korean, no card).
- **Text only and unlabeled by speaker:** AIxBlock 92k call-center transcripts. Its Medicare/DME outbound calls are "gray", not benign. Possible extra text slice later, not in this set.
- **Paid, slow or wrong domain:** Switchboard and Fisher (LDC fee), KorCCVi (Korean, subscription), Unidata human-robot (paid), Nomorobo pages (no audio; ToS bans scraping).

---

## 2. Recipe: a 119-call real test set in about 6 hours

### 2.1 What goes in

Selection rules are written into a manifest (`eval/real_calls/manifest.csv`, proposed path) with a fixed seed **before** any ASR runs. Every selected call stays in the set, even if a model later does badly on it.

| ID | Source | n | Class | How we pick | Sides |
|---|---|---|---|---|---|
| R1 | NCSU FTC robocalls | 25 | scam (robocall) | English, transcript of 25 words or more, stratified by case type: government/SSA 6, Amazon/Apple order 6, auto warranty 4, tech/utility/credit 5, other 4; seeded random within each stratum | One (caller) |
| R2 | International Robocalls | 10 | scam (robocall) | English; 10 distinct campaign clusters with a fraud label (IRS/Tax and similar); excludes anything that also appears in the FTC letters | One |
| R3 | FBI x2 + Niagara police | 3 | scam (real victim) | All 3 | Two (mono mix) |
| R4 | Scam-bait videos | 20 | scam (conversational) | CC BY channels first (Salty Dogg, IRLrosie, Wenja92, Nellie's, Callcenterbuster), then More Kitboga / Scammer Payback if short. Single-call videos of 5-30 min only, no livestreams, no Lenny bot. Must reach an ask (gift card, remote access, bank login, payment). At most 4 per channel | Two (mono mix) |
| R5 | Kitboga AI bot callers | up to 5 | scam + ai_voice | The chapters of fhi6JU5pOJk where the bot is the caller | Two (mono mix) |
| | **Scam total** | **63** | | | |
| N1 | CallHome English | 18 | benign, real | Seeded random from the 74 excerpts with 3+ money, bank or health hits (always including 4289, 4886, 5373); `en-full` stereo when available, else the `en` excerpt | Two |
| N2 | CallFriend teens | 3 | benign, real (kid profile) | en_4504 (14), en_6402 (17), plus one 18-year-old caller | Two |
| N3 | UniData sample | 5 | benign, real | All 5 | Two (mono mix) |
| N4 | HarperValleyBank | 12 | benign, role-play | Stratified by intent (transfer, replace card, reset password, balance, etc.), seeded | Two (separate WAVs) |
| N5 | AppTek | 8 | benign, role-play | Banking 3, insurance 2, health 2, finance 1 | Two (separate WAVs) |
| N6 | SpokenWOZ profile domain | 6 | benign, role-play (personal-info asks) | Test dialogues where the agent collects ID, email or plate | Two |
| N7 | PriMock57 | 4 | benign, role-play (doctor) | Seeded | Two (separate WAVs) |
| | **Benign total** | **56 (47%)** | of which 26 real unscripted, 30 role-played | | |
| | **Grand total** | **119** | | | |

Text-only slices reported next to it, not counted in the 119: BothBosu 20 (already test set 2) and the Kaggle 243 openings (an "early warning on openings" slice, since most end before the ask).

If the YouTube download (R4, R5) is not approved, the set is 94 calls (38 scam, 56 benign). To keep 100+, add 10 more NCSU and 10 more International robocalls. The conversational scam slice then shrinks to the 3 real-victim calls, and we say so.

### 2.2 Turning each call into our turn format

The output for each call is the same layout the live line writes: `recordings/<call_id>/turns.jsonl` + `meta.json`. That way `callers/label_sheet.py` and the eval read real-source calls exactly like V2 calls.

1. **Phone codec first.** Resample every channel to 8 kHz and encode as μ-law, the same as the Telnyx media stream. Wideband role-play and studio audio then sound like a phone call to the ASR, and the classes can't be told apart by audio quality.
2. **Same ASR, same segmentation.** Feed 20 ms μ-law frames into `line.asr.CallTranscriber`, the code the live line uses. Track `inbound` is the caller and `outbound` is the protected side. Nemotron runs at about 11x real time on the Mac. The set is about 9-10 h of audio (robocalls under 1 min, HarperValley about 1 min, CallHome and scam-bait excerpts capped at 5-10 min), so that is about 55 min with 8 processes, or 10-15 min on the G1 VM.
3. **Speaker attribution.**
   - **Two-channel sources** (HarperValley, AppTek, PriMock57, CallHome `en-full`, CallFriend): one channel per track, so attribution is exact. The agent, doctor or person who placed the call is the `caller`; the customer, patient or call recipient is `protected`.
   - **One-sided robocalls** (R1, R2): everything goes on the `caller` track, and the protected track is silence.
   - **Mono two-speaker mixes** (R3, R4, R5, N3, CallHome `en`): run sherpa-onnx offline speaker diarization (the pyannote segmentation ONNX model plus a speaker-embedding model, with `num_speakers=2`) on the 16 kHz audio. Write each cluster to its own track and then encode as in step 1. For scam-bait videos, a bandwidth check helps before downsampling: the scammer comes over the phone line (nothing above about 3.4 kHz), while the baiter is usually on a full-band microphone. The labeler confirms which cluster is the caller with one click per call, and fixes any turns the diarization merged. Calls where diarization is clearly broken (heavy overlap, music) are kept and flagged `diarization: poor` instead of being dropped.
4. **Profile and rules.** `profile` is `elder` for R3 and the robocalls, `kid` for N2, `adult` otherwise. Use one fixed default rule set for every call, so no rule is tuned per call.

### 2.3 Labeling, blind

- The sheet comes from `python -m callers.label_sheet sheet`. It shows only the transcript and the guardian inputs, never model outputs or the source.
- **Matt labels every call**:
  - call class (`scam`, `benign`, `gray`) and category;
  - `ask_turn`, the first turn where the caller asks for money, a code or password, personal or financial ID, remote access, an address or a meeting, or `null`;
  - `ai_voice` for the caller: `tts`, `recorded_human`, `live_human` or `unclear`, from listening, for R1, R2 and R5.
- **Numan second-labels a seeded 25-call overlap.** We report agreement on class (Cohen's kappa) and on ask_turn (exact match and within one turn). Disagreements are settled before scoring, and the settled label is kept.
- Robocalls are fast, about 1 minute each. Benign calls take 1-2 minutes ("would a fair guardian alert here?" plus a scan for asks). Conversational scams take about 5 minutes. That is about 3.5 h for Matt, run in parallel with the replay work.
- **Labels are committed before any model scores the set** (`validate --require-committed`). The commit hash goes in the results README.

### 2.4 Score offline, replay a subset live

| | Offline (whole set, 119) | Live replay (subset, about 50) |
|---|---|---|
| How | Recorded audio, then 8 kHz μ-law, the same `CallTranscriber`, the same guardian prompt for every model | SignalWire **+1 304 648 1098** plays the caller channel into the Telnyx guarded number **+1 323 516 1282**, and the line server records it like any V2 call |
| Pros | All models score identical transcripts. Reproducible from committed data with one command. Real-time speed doesn't matter | Real PSTN, real Telnyx codec and jitter, real live ASR and real end-to-end latency. Telnyx `deepfake_detection` runs on the same calls. Feeds V1's real-phone-path data |
| Cons | Codec is simulated, not the real network | Real time (2 calls at once means about 1.5 h for about 90 min of audio). Mono two-speaker sources put both voices on the caller track, so only one-sided or two-channel sources replay cleanly. Leg B needs an automated answer (below). Costs a few dollars |
| Which calls | All | R1 25 + R2 10 (one-sided, about 35 min of audio) + N4 HarperValley 12 (two channels) + R3 3 (caller-track only, flagged) |

- **Headline:** the offline numbers on all 119.
- **Live replay check:** on the replayed subset, report agreement between offline and live decisions (same alert or not, same alert turn) and the turn-to-alert latency. If they agree, the offline headline stands for the live path.
- **Leg B for replay.** Do not ring a person's phone 50 times. Options:
  - (a) A replay flag in the line server that answers leg B with an automated endpoint. For the two-channel HarperValley calls, that endpoint plays the customer channel, started when B joins the conference (drift of about 0.5 s is fine, since segmentation is per side).
  - (b) A second number that auto-answers and plays. This is a $1 purchase, so it needs Numan's go-ahead.
- **Cost:** at roughly $0.01 per minute per leg on each carrier, plus $0.01 per call for deepfake detection, about 50 calls / 90 minutes is **under $5** (rates approximate). This is inside the $10/day Telnyx cap.

### 2.5 Timeline (two people)

| When | Numan | Matt |
|---|---|---|
| 0:00-0:45 | Fetch R1, R2, N1-N7 (all small or selective); fetch R3 by hand in a browser; yt-dlp for R4/R5 **if approved** | Write the manifest selection rules and seed; pick R4 videos from `cc_relevant.tsv` titles |
| 0:45-1:45 | Codec + diarization + ASR into `recordings/` (Mac or G1 VM) | Spot-check 10 transcripts; build label sheets |
| 1:45-4:00 | Live replay of the about 50-call subset | Blind labeling (about 3.5 h, continues) |
| 4:00-4:30 | Second-label the 25-call overlap | Finish labeling; settle disagreements |
| 4:30-5:00 | Commit labels and manifest (hash recorded) | Commit |
| 5:00-6:00 | | Score B0-B6 + ours offline; bootstrap CIs; McNemar; per-source tables |

### 2.6 The "real video" proof for the demo

The demo video replays one real call through the live line and screen-records the console, showing the alert landing before the ask.
- **First choice: the FBI elder-fraud call.** It is a federal work, so it is the cleanest to show.
- **Second choice: a CC BY scam-bait clip**, credited on screen, with names and numbers bleeped.
- **The Niagara police grandparent call** is the most striking: a real victim, coached to lie, with the $9,550 ask. Its reuse status is unclear, so use only a few seconds on a fair-use argument, or skip it.

---

## 3. The legal line

| Source group | Commit to the repo | Keep local only (`data/`, `recordings/` audio, both gitignored) |
|---|---|---|
| NCSU/FTC (public domain), FBI (federal work), HarperValley, PriMock57 (CC BY 4.0) | IDs, our labels, our ASR turns, fetch script. Audio could be committed, but we don't need to | Nothing required |
| International Robocalls, SpokenWOZ, UniData, AppTek (NC, ND or eval-only terms) | IDs, labels, fetch script | Audio, and our ASR turns (they reproduce the content) |
| CallHome, CallFriend (TalkBank Ground Rules) | Call IDs, timestamps, our labels, citation | Audio and transcripts. **No training on them.** Hosted LLM baselines see their text only if the provider doesn't store prompts (confirm Fireworks' retention setting first; if it is unclear, score TalkBank calls only with the local and on-demand models, and say so) |
| YouTube (CC BY and standard license), Niagara police | `youtube_id`, start/end seconds, ask timestamp, labels (AudioSet-style) | Audio, deleted after the eval. Downloading breaches the YouTube ToS, so it is **Numan's call**. Only short CC BY or federal clips appear in the video |
| Our derived turns for eval-only sources | Not committed. The one-command repro (`python -m eval.run --all`) reproduces the numbers for the redistributable sources from committed data, and the rest from a fetch-then-score script | Yes |

Rules we always follow:
- Never use recordings obtained by hacking.
- Never use any minor's voice except the consented TalkBank teen calls, and those only as benign negatives.
- Bleep third-party names and numbers in anything we show.
- Never train the guardian on any of these sets. It stays trained on generated calls only, so this set is purely held out.

**Attribution text for the README:**

> Real-call test set sources: Robocall audio from the FTC's Project Point of No Entry, packaged by S. Prasad and B. Reaves, "Robocall Audio from the FTC's Project Point of No Entry", NCSU TR-2023-1 (data in the public domain). International Robocalls Dataset, Altwlkany et al., doi:10.5281/zenodo.21066049 (CC BY-NC 4.0). FBI scam-call recordings (fbi.gov audio repository). Niagara Regional Police Service grandparent-scam recording (via Meridian Credit Union). CallHome and CallFriend English, TalkBank CABank, doi:10.21415/T5KP54, from LDC97S42 / LDC96S46 / LDC96S47 (CC BY-NC-SA 4.0, TalkBank Ground Rules). HarperValleyBank, Gridspace-Stanford (CC BY 4.0). AppTek Call-Center Dialogues (CC BY-SA 4.0). SpokenWOZ, Si et al. 2023 (CC BY-NC 4.0). PriMock57, Babylon Health (CC BY 4.0). UniData call-center sample (CC BY-NC-ND 4.0). YouTube videos are referenced by ID and timestamp only; CC BY clips shown in our video are credited to their channels. No audio from non-redistributable sources is in this repository.

---

## 4. What a judge will still criticize, and our answer

| Criticism | Answer |
|---|---|
| "Robocalls are monologues; catching 'press 1' is easy." | True. We report robocalls (35) as their own slice, next to the conversational slice (R3+R4+R5, 28) and benign (56). B0 keywords is a baseline on each slice. The claim that matters is conversational catch-before-ask at a fixed false-alarm rate |
| "Scam-baiters aren't real victims." | Agreed. Only 3 calls have a real victim. R4 tests real scammer speech and real asks; the V2 role-play calls test a cooperative protected side. We label R4 "real scammer, performative victim" everywhere |
| "The benign calls are role-play, so style gives the class away." | Everything passes the same codec, ASR and segmentation, which removes transcript-style cues (the HarperValley/YouTube trap the rajkirant README warns about). 26 of 56 benign calls are real and unscripted, and we report false alarms per source and real vs role-play |
| "n is small; the intervals are wide." | Yes: about ±12 points on the scam slices. We show bootstrap 95% CIs and paired McNemar tests, and pair this set with the 500-call held-out set and the V2 live set. We don't claim wins the intervals don't support |
| "You labeled your own data." | Labels are written blind (no model outputs), committed before scoring with the hash recorded, and a second labeler on 25 calls gives kappa and ask-turn agreement |
| "Leakage." | The guardian is trained only on Nemotron-generated calls, and its generator never saw these sources. Frontier baselines may have seen FTC or YouTube transcripts in pretraining. That would help them, not us |
| "Offline isn't the live product." | About 50 calls are replayed through the real line, and we report offline-vs-live agreement and latency |
| "Cherry-picked." | Seeded selection rules are committed before ASR, no call is dropped after selection, and failure cases are listed with quotes |
| "Kids aren't covered." | Honestly, they aren't, with real audio. No ethical real recordings exist of a stranger working on a child by phone, and we won't use sting or hacked material. Kid coverage comes from PAN-12 (chat, labeled as chat), role-play kid cards in V2, and 3 real teen calls that measure false alarms only |
| "Your AI-voice labels on robocalls are guesses." | The NCSU and International sets have no TTS label, so we hand-label `tts`/`recorded_human`/`unclear` by ear with two raters and report on the agreed items. ASVspoof 2021 `pstn` is the controlled check, and Kitboga's bot callers are the deployed-TTS example |
| "Diarization errors bias the mono calls." | Labelers fix the role mapping. We flag `diarization: poor` calls and report the scores with and without them |

**One decision needed from Numan:** may we download YouTube audio with yt-dlp for a local, delete-after eval (R4 + R5, 25 calls)? Yes gives 119 calls with a conversational scam slice of 28. No gives about 114 calls (more robocalls), with only 3 conversational real scams.

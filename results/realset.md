# Real-audio test set (realset): frontier baselines

**5 real calls** (3 scam robocalls, 2 benign bank calls), 2 min of audio, 19 scored caller turns. Every call went through our phone path and live ASR; every model saw identical transcripts.

![realset chart](realset_chart.svg)

## Results (call level)

| Model | Scam recall | 95% CI | False alarms on benign | 95% CI | Turn errors | Cost (USD) | p50 latency |
|---|---|---|---|---|---|---|---|
| qwen3p8-max | 2/2 = **100%** | 34%-100% | 0/0 = **0%** | 0%-100% | 0/7 | 0.01 | 1879 ms |
| kimi-k3 | 2/2 = **100%** | 34%-100% | 0/0 = **0%** | 0%-100% | 0/7 | 0.02 | 2479 ms |

Recall by scam type:

| Model | gov_ssa_irs |
|---|---|
| qwen3p8-max | 2/2 |
| kimi-k3 | 2/2 |

## Method

- **Scam (3):** NCSU Robocall Audio Dataset (FTC Project Point of No Entry; data public domain, https://github.com/wspr-ncsu/robocall-audio-dataset). English, transcript >= 25 words, near-duplicate campaigns removed, seeded (20261102) stratified pick by a keyword scam type read from the dataset's Whisper transcript, one per carrier case first. Types: gov_ssa_irs 3.
- **Benign (2):** HarperValleyBank (Gridspace-Stanford, CC BY 4.0, https://github.com/cricketclub/gridspace-stanford-harper-valley). Real human speakers role-playing bank customer-service calls with account, card, PIN/password and transfer talk (hard negatives). Seeded pick from a 220-call random sample, round-robin over task types, calls >= 25 s. Tasks: get_branch_hours 1, replace_card 1.
- **Phone path + ASR:** each channel resampled to 8 kHz, G.711 mu-law (`line/audio.py`), fed in 20 ms frames into `line.asr.CallTranscriber` (Nemotron Speech Streaming 0.6B int8 via sherpa-onnx, the live line's code and endpointing). Robocalls: the recording is the caller track; the protected track is not fed (it would be digital silence). HarperValley: agent channel = caller, customer channel = protected.
- **Guardian input:** Matt's frozen prompt `guardian/render.py` (PROMPT_SHA 806908cfba7d). One request at the end of every caller turn with the last 8 turns; profile `elder` for all calls (so the profile can't leak the class), no rules, unknown number, voice unknown, no 'earlier' verdict. Temperature 0, max_tokens 600, reasoning_effort none.
- **Decision:** a call is flagged if any scored turn has risk alert or urgent. Wilson 95% intervals. Failed or unparseable replies count as not flagged.
- **Reproduce:** `python -m realset.select_calls` (manifest), fetch audio from the URLs in `realset/manifest.csv`, `python -m realset.transcribe`, `python -m realset.score --model <id> --tag <tag>`, `python -m realset.metrics`.

## Caveats

- **Robocalls are one-sided prerecorded messages** (often TTS), not conversations. They test whether a model recognizes a real scam pitch through real ASR errors; they do not test catching the ask mid-conversation, and there is no victim side.
- **HarperValley is role-played** from task templates by paid speakers (real voices, scripted scenario, small vocabulary), so it is an easy benign set in content even though it is full of account and card talk. Real unscripted benign calls (CallHome/CallFriend) are not in this set.
- The two classes come from different sources, so a model could separate them by style (monologue vs dialogue). Read this as a sanity check on real audio, not as the headline benchmark.
- Scam type is a keyword bucket over the dataset's Whisper transcript, not a human label; labels are source-level (every NCSU call is treated as scam, the FTC letters name them as illegal robocall campaigns).
- The ASR is our live Nemotron path on simulated (offline) G.711, not a real PSTN leg.


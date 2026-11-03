# Real-audio test set (realset): fine-tuned guardian vs frontier baselines

**301 real phone calls** (181 scam robocalls, 120 benign bank calls), 192 min of audio, 1,233 scored caller turns.
Every call went through our phone path (8 kHz G.711) and our live ASR. Every chat model saw the same transcripts and the same frozen prompt. Rune saw the same prompt through the decisions API (see section 2).
Frontier baselines: **Kimi K3** (named baseline) and **Qwen3.8 Max**. Cheaper serverless baselines: Nemotron Lightning 3.5, DeepSeek V4.1 Flash, Rune.

Short version: the frontier models catch 96-97% of the scam robocalls. Our fine-tuned guardian catches 50%. Its false-alarm rate is not significantly different from Kimi's (21% vs 26%). So on this set the fine-tune does not beat the named baseline. It is faster than Kimi K3 (860 ms vs 2,324 ms p50), but that pits a dedicated GPU against shared serverless. Nemotron Lightning 3.5 is faster still (449 ms), far cheaper, and catches 92%.

![realset chart](realset_chart.svg)

Read first: [realset_headline.md](realset_headline.md). Every call with every model's per-turn verdict: [realset_view.html](realset_view.html). Full tables: [realset_metrics_plus.md](realset_metrics_plus.md).

## 1. What the set is

| | Scam | Benign | Total |
|---|---|---|---|
| Source | NCSU Robocall Audio Dataset | HarperValleyBank | |
| Calls | 181 | 120 | 301 |
| dev split | 91 | 60 | 151 |
| test split | 90 | 60 | 150 |
| batch v1 (first build) | 61 | 50 | 111 |
| batch v2 (added today) | 120 | 70 | 190 |

- **Scam types** (keyword bucket, hand-checked): tech support / utility 42, Amazon / Apple order 38, government / SSA / IRS 35, bank / credit card 33, auto warranty 19, prize / travel / health 11, delivery / parcel 3.
- **Benign tasks** (HarperValley): check_balance 16, get_branch_hours 16, order_checks 15, pay_bill 15, replace_card 15, reset_password 15, schedule_appointment 14, transfer_money 14.
- **Split:** a seeded shuffle, stratified by label and scam type (`assign_split` in `realset/select_calls.py`). It never reads model outputs. The 111 v1 calls had already been scored when the split was drawn, so the split is not pre-registered. It cannot carry score information, though, because it is a fixed-seed shuffle. Dev and test each have about half of every type. The split is per call, not per campaign, so near-duplicate robocalls fall on both sides (section 8).
- **Licenses:** NCSU robocall audio comes from the FTC's Project Point of No Entry, packaged by S. Prasad and B. Reaves (NCSU TR-2023-1, https://github.com/wspr-ncsu/robocall-audio-dataset); the data is public domain. HarperValleyBank is by Gridspace-Stanford, CC BY 4.0 (https://github.com/cricketclub/gridspace-stanford-harper-valley). We publish only our own ASR transcripts, no audio.

## 2. Method

- **Picking calls.** Scams: English NCSU robocalls with at least 25 words, near-duplicates removed, one per carrier case first, seeded stratified pick by scam type. Benign: seeded round-robin over HarperValley task types, calls of at least 25 s. Code: `realset/select_calls.py`.
- **Phone path and ASR.** Each channel is resampled to 8 kHz G.711 mu-law (`line/audio.py`) and fed in 20 ms frames to `line.asr.CallTranscriber` (Nemotron Speech Streaming 0.6B, the live line's code and endpointing). Robocall = caller track. HarperValley: bank agent = caller, customer = protected person.
- **Guardian input.** Matt's frozen prompt (`guardian/render.py`, PROMPT_SHA 806908cfba7d). One request at the end of each caller turn, with the last 8 turns. Profile `elder` for every call, unknown number, no rules, no earlier verdict. Temperature 0, max_tokens 600, reasoning off.
- **Models.** Emberz guardian v2 is Matt's fine-tune on a dedicated deployment. Kimi K3 (`accounts/fireworks/models/kimi-k3`), Qwen3.8 Max (`qwen3p8-max`), Nemotron Lightning 3.5 (`nemotron-lightning-3p5-30b-a3b`) and DeepSeek V4.1 Flash (`deepseek-v4p1-flash`) are Fireworks serverless chat models. **Rune** is not a chat model: it ran through the Fireworks decisions API. It read the same prompt and window and answered two multiple-choice questions (risk, category). That setup is close but not identical. The keyword stub is a 4-word placeholder, not a tuned keyword baseline.
- **Decision rule (fixed in advance).** A call is flagged if any caller turn is rated `alert` or `urgent`. Scam recall = flagged scams / scams. False alarms = flagged benign / benign.
- **Statistics.** Wilson 95% intervals for rates. AUC uses the call's highest rating (none < watch < alert < urgent), with a stratified bootstrap CI (2,000 resamples). Paired tests use McNemar exact on the same calls plus a paired bootstrap CI of the difference. There is no correction for multiple comparisons.
- **Errors.** Failed or unparseable replies count as "not flagged". After re-runs, every model has a valid verdict on all 1,233 turns.
- **Reproduce.** `python -m realset.select_calls`, `python -m realset.transcribe`, `python -m realset.score --model <id> --tag <tag>`, then `python -m realset.metrics`, `python -m realset.metrics_plus` and `python -m realset.view`.

## 3. Headline table (all 301 calls, official rule)

| Model | Scam recall (95% CI) | False alarms on benign (95% CI) | AUC (95% CI) | Time to first alert, median | Cost per 1k calls | p50 latency |
|---|---|---|---|---|---|---|
| **Emberz guardian v2 (ours)** | 90/181 = **50%** (43-57%) | 25/120 = **21%** (15-29%) | 0.65 (0.59-0.70) | 16.1 s | ~$8.42 (GPU time) | **860 ms** |
| Kimi K3 | 175/181 = **97%** (93-98%) | 31/120 = **26%** (19-34%) | 0.86 (0.81-0.90) | 17.1 s | $10.93 | 2,324 ms |
| Qwen3.8 Max | 173/181 = **96%** (92-98%) | 31/120 = **26%** (19-34%) | 0.87 (0.82-0.90) | 16.2 s | $11.63 | 3,024 ms |
| Nemotron Lightning 3.5 | 166/181 = **92%** (87-95%) | 28/120 = **23%** (17-32%) | 0.86 (0.82-0.90) | 14.9 s | $0.19 | 449 ms |
| DeepSeek V4.1 Flash | 166/181 = **92%** (87-95%) | 40/120 = **33%** (26-42%) | 0.81 (0.76-0.85) | 15.9 s | $1.04 | 1,213 ms |
| Rune (decisions API) | 163/181 = **90%** (85-94%) | 49/120 = **41%** (32-50%) | 0.76 (0.71-0.81) | 14.5 s | $0.15 | 107 ms |
| keyword stub | 3/181 = 2% (1-5%) | 11/120 = 9% (5-16%) | 0.46 | - | $0 | - |

- **Time to first alert** = seconds from call start to the end of the first flagged caller turn, plus that request's latency, over the scams the model flagged. Each model is timed on a different set of scams (only the ones it caught), so the column is not a head-to-head comparison. Robocalls are short, so this mostly reflects how much of the message a model needs. On the 88 scams that all six LLMs flagged, every model's median alert comes on the first caller turn (about 14.2-14.6 s into the call). Ours has heard more of the caller's words by then: a median 80% of them, vs 66-70% for the others.
- **Cost.** Serverless cost = tokens × Fireworks list price, counting every attempt and retry. Our guardian runs on a dedicated GPU: cost = summed request time × $8/h. That is a rough figure. It leaves out idle time and scale-up, so it is not an invoice.
- 0 turn errors for every model after re-runs. Qwen needed 97 JSON-mode retries and Nemotron 97 (93 on the final verdicts, 4 in attempts that were later re-run); all are counted in cost.

## 4. Held-out test split: dev-tuned operating point

Each model picks its threshold (watch / alert / urgent) **on dev only**: the best balanced accuracy with at most 25% false alarms. We report it **on test only** (90 scams, 60 benign).

| Model | Chosen on dev | Test recall (95% CI) | Test false alarms (95% CI) | Test balanced acc. | Test AUC (95% CI) |
|---|---|---|---|---|---|
| **Emberz guardian v2 (ours)** | alert | 47/90 = 52% (42-62%) | 15/60 = 25% (16-37%) | 64% | 0.64 (0.57-0.71) |
| Kimi K3 | alert | 88/90 = 98% (92-99%) | 17/60 = 28% (19-41%) | 85% | 0.85 (0.79-0.90) |
| Qwen3.8 Max | alert | 85/90 = 94% (88-98%) | 17/60 = 28% (19-41%) | 83% | 0.85 (0.79-0.90) |
| Nemotron Lightning 3.5 | alert | 84/90 = 93% (86-97%) | 15/60 = 25% (16-37%) | 84% | 0.86 (0.80-0.91) |
| DeepSeek V4.1 Flash | urgent | 5/90 = 6% (2-12%) | 0/60 = 0% (0-6%) | 53% | 0.80 (0.74-0.86) |
| Rune (decisions API) | urgent | 0/90 = 0% (0-4%) | 0/60 = 0% (0-6%) | 50% | 0.75 (0.68-0.82) |

- Ours, Kimi, Qwen and Nemotron all pick the official `alert` rule on dev, so their test numbers equal the official rule on test.
- The 25% false-alarm cap and the balanced-accuracy criterion were our choice and were not registered in advance. A different cap would move DeepSeek and Rune, but not the comparison between ours and Kimi K3.
- DeepSeek and Rune exceed 25% false alarms at `alert` on dev. They are forced to `urgent` and collapse on test. With the official rule on test they reach 92% / 35% and 90% / 43%.
- **No threshold rescues our guardian.** At `watch` it flags 99% of scams but also 98% of benign calls. At `alert` it flags 50% of scams. At `urgent` it flags 3%. 91 of 181 scams never get past `watch`.

## 5. Paired tests vs Kimi K3 (named frontier baseline)

Same calls, official rule. Difference = model minus Kimi K3. (x:y) = calls only the model got right : calls only Kimi got right.

| Model vs Kimi K3 | Slice | Recall diff (95% CI) | McNemar p | False-alarm diff (95% CI) | McNemar p |
|---|---|---|---|---|---|
| **Emberz guardian v2 (ours)** | test (150) | **-46 pts** (-56 to -36) | <0.001 (0:41) | -3 pts (-12 to +7) | 0.73 (5:3) |
| **Emberz guardian v2 (ours)** | all (301) | **-47 pts** (-54 to -40) | <0.001 (0:85) | -5 pts (-12 to +2) | 0.21 (11:5) |
| Qwen3.8 Max | test | -3 pts (-8 to +1) | 0.38 (1:4) | +0 pts (-10 to +10) | 1.0 (5:5) |
| Nemotron Lightning 3.5 | test | -4 pts (-10 to 0) | 0.22 (1:5) | -3 pts (-17 to +12) | 0.82 (11:9) |
| DeepSeek V4.1 Flash | test | -6 pts (-10 to -1) | 0.06 (0:5) | +7 pts (-2 to +15) | 0.29 (2:6) |
| Rune (decisions API) | test | -8 pts (-13 to -2) | 0.016 (0:7) | +15 pts (+7 to +25) | 0.004 (0:9) |

- **Ours loses to Kimi K3 on scams**, and the loss is large and certain. There is no scam that ours caught and Kimi missed (0:85 over all calls).
- **On false alarms ours is not worse.** It is 3-5 points lower, but the CI includes zero, so this is a tie, not a win.
- Ours has significantly fewer false alarms than DeepSeek V4.1 Flash (-12 pts, p<0.001) and Rune (-20 pts, p<0.001) over all calls. It still catches about 40 points fewer scams than either.
- Qwen3.8 Max and Kimi K3 are statistically tied. Nemotron Lightning 3.5 is close to Kimi on test at about 1/60 of the cost.
- `realset_metrics_plus.md` has 21 model pairs (42 McNemar tests), with no correction. The Bonferroni cut is 0.05/42 = 0.0012. Ours' recall gap to every LLM survives it, and so do ours' lower false-alarm rates vs DeepSeek and Rune, and Kimi's and Qwen's lower false-alarm rates vs Rune. The other gaps between baselines (p 0.002-0.065, for example Kimi vs Nemotron or DeepSeek on recall) do not.

**Sensitivity: duplicate campaigns.** Many robocalls are the same script recorded several times. We grouped scams by 4-gram overlap (at least half of the shorter call's 4-grams shared). That gives 84 campaign clusters, or 107 at a 70% cut. The largest cluster is 16 auto-warranty calls. If each campaign counts once, ours catches 53% vs Kimi's 95% (other models 91-96%). The difference is -42 points, with a cluster bootstrap 95% CI of -52 to -32 (-44, CI -53 to -35, at the 70% cut). Our all-call recall CI widens to about 37-64% when calls are resampled by cluster. The conclusion holds, but the per-call CIs in this report are too narrow for scam rates. This is a post-hoc check: the clusters were built after scoring, from our ASR text, and nothing was tuned on them.

## 6. Recall by scam type (all calls)

| Model | tech / utility (42) | Amazon / Apple (38) | gov / SSA / IRS (35) | bank / card (33) | auto warranty (19) | prize / travel (11) | delivery (3) |
|---|---|---|---|---|---|---|---|
| **Emberz guardian v2 (ours)** | 10 | 29 | 31 | 19 | **0** | **1** | **0** |
| Kimi K3 | 38 | 37 | 35 | 33 | 19 | 10 | 3 |
| Qwen3.8 Max | 37 | 37 | 35 | 33 | 18 | 10 | 3 |
| Nemotron Lightning 3.5 | 32 | 35 | 34 | 33 | 19 | 11 | 2 |
| DeepSeek V4.1 Flash | 34 | 35 | 35 | 33 | 17 | 9 | 3 |
| Rune (decisions API) | 33 | 33 | 35 | 31 | 18 | 10 | 3 |

- Ours does well on government impersonation (31/35) and Amazon / Apple order scams (29/38). These are the classic "your account / your SSN" scripts.
- It misses almost every auto-warranty, prize and delivery robocall, and 32 of 42 tech-support / utility ones. These pitches say "press one" or "call us back" and never ask for money or data in the call itself.

False alarms by HarperValley task (out of 14-16 each):

| Model | order_checks | pay_bill | reset_password | transfer_money | other 4 tasks |
|---|---|---|---|---|---|
| **Emberz guardian v2 (ours)** | 6 | 10 | 8 | 1 | 0 |
| Kimi K3 | 13 | 8 | 9 | 1 | 0 |
| Qwen3.8 Max | 7 | 11 | 8 | 3 | 2 |
| Nemotron Lightning 3.5 | 3 | 1 | 6 | 5 | 13 (replace_card) |

## 7. Failure cases

**Ours misses the scams that only point to a next step.** It rates them `watch`, while Kimi K3 rates them `alert`:
- ncsu_65173 (auto warranty, test): "...concerning your car's extended warranty... we are giving you a final courtesy call before we close out your file."
- ncsu_500579 (utility, test): "...an automated message from the disk connection department Baltimore gas and electric to inform you that you will be experiencing a disconnection of service today within thirty minutes. Press one..." (ASR text, "disk connection" sic)
- ncsu_735557 (prize, test): "...Four Seasons Hotels. Your telephone number has been preselected to receive a complimentary stay... press one."
- ncsu_1195647 (delivery, test): "FedEx International Express will inform you for the last time that you have a parcel..."

**Ours rates almost everything `watch`.** 92 of 120 benign calls and 90 of 181 scams peak at `watch`. A plain bank greeting ("Hello, this is Harper Valley National Bank...") already gets `watch` / impersonation. So `watch` carries no signal, and the only useful threshold is `alert`.

**Ours false-alarms when the bank agent asks for account details.** Example: hv_6913f7cf2f2144a7 (pay_bill, test). The agent asks "what is the company address", the customer reads out an address, and ours gives `alert` / financial_id. Kimi stays at `watch` on this call.

**Kimi K3 false-alarms on address requests.** Example: hv_5c761521b3004564 (order_checks, test). The agent asks "what is your address" for a new checkbook, and Kimi gives `alert` / location_school. Ours does not. Kimi flags 13 of 15 order_checks calls; ours flags 6. This is the one slice that favours ours, but we picked it after seeing results (1 of 8 tasks, 15 calls). On pay_bill ours is worse (10 vs 8), and the total false-alarm gap is not significant.

**Kimi K3 misses 6 scams with no explicit ask:**
- an AT&T DirecTV "fifty percent discount" pitch (ncsu_547611)
- a "supply charges for your Panola Electric statement, please return my call" callback (ncsu_1348032)
- a solar lead-gen pitch (ncsu_447375; illegal telemarketing, arguably not a scam)
- a Spectrum "buy one get one free" offer (ncsu_1237002)
- two fragments that are only an IVR menu (ncsu_1205446, ncsu_57750)

Ours misses all six too.

**Why the false alarms happen.** We put the bank agent on the caller side with an unknown number. A stranger asking for an address, password reset or account number then looks like data collection. For an unverified inbound caller that is arguably the right call, so these are hard negatives by design.

## 8. Caveats

- **Robocalls are one-sided.** They are prerecorded pitches (often TTS) with no victim side. They test recognizing a scam pitch through real ASR errors. They do not test catching the ask inside a live conversation.
- **The benign calls have the agent as caller.** In HarperValley the customer phoned the bank. We put the agent on the caller track, as if the bank called from an unknown number. That makes the benign set harder. The calls are also role-played from templates by paid speakers, so the vocabulary is small.
- **Source-style confound.** All scams are NCSU monologues and all benign calls are HarperValley dialogues. A model could separate them by style alone. Treat this set as a real-audio check of the full phone path. Do not read it as a field false-alarm rate.
- **The guardian was trained on two-sided conversations.** Matt's fine-tune learned to wait for the ask: money, a code, card details, remote access. Robocalls rarely make that ask inside the recording. That may explain many of its `watch` ratings, but we have not tested it. Either way, the fine-tune does not generalize to one-sided scam pitches, and a deployed guardian will hear them.
- **Labels are source-level.** Every NCSU call counts as a scam. A few are illegal telemarketing rather than fraud (solar, DirecTV offers). A few transcripts hold only an IVR tail with no pitch, for example ncsu_57750, ncsu_62996 and ncsu_1205446 ("press two to listen to this message again... call us on our toll free number"). They are labelled scam from the source, but the text alone cannot show it. Scam type is a keyword bucket over the dataset transcript, with 19 calls hand-corrected (`TYPE_FIX`, `TYPE_FIX_V2` in `realset/select_calls.py`). The corrections change only the type, never scam vs benign.
- **Near-duplicates.** Many robocalls come from the same campaign (same script, different recording). The selection filter only compares the first 120 characters of the dataset transcript, so it misses most of them. For example, ncsu_598182 (v2) is almost word-for-word ncsu_807762 (v1). The 181 scams form about 84 campaign clusters, and 23 clusters have calls in both dev and test. The sensitivity check in section 5 shows the main result holds. Per-call CIs overstate how precise the scam numbers are, and dev and test are not independent at the campaign level.
- **Benign speakers repeat.** HarperValley has 59 speakers and templated scripts, so the 120 benign calls are not 120 independent samples either.
- **Baselines ran without reasoning.** Every chat model ran with `reasoning_effort` none, the same as the guardian, which keeps latency comparable. Frontier models with reasoning on might score higher and would be slower. Frontier models may also have seen the NCSU dataset's public transcripts in pretraining. Our ASR text differs, but some contamination is possible.
- **Precision depends on the mix.** The scam share here is 60%. Real call traffic has far fewer scams, so real-world precision would be lower for every model.
- **ASR.** We used our live ASR on simulated G.711, not a real carrier leg.
- **Our cost is rough.** It is summed request time on a dedicated GPU at $8/h. The deployment also scaled to zero mid-run: 24 turns returned 503 and were re-scored.
- **Rune is not like-for-like.** It ran through the decisions API with multiple-choice questions we wrote from the prompt.
- **No tuning on test.** No model, prompt or threshold was tuned on these calls. The only choice made from data is the dev-side threshold in section 4, which is reported on test only. The slices and failure cases in sections 6-7 were read after scoring and are descriptive, not tests.

# Realset: extended metrics

301 real calls (181 scam robocalls, 120 benign bank calls), 192 min of audio, 1233 caller turns. Same transcripts, same prompt, every model. Built by `python -m realset.metrics_plus` from `data/realset/scores/*.jsonl`; full numbers in `data/realset/metrics_plus.json`. Official rule: a call is flagged if any caller turn is rated alert or urgent.

![ROC](realset_roc.svg)

## 1. Official rule, all calls

| Model | Scam recall (95% CI) | False alarms (95% CI) | Precision | F1 | Balanced acc. | AUC of max risk (95% CI) |
|---|---|---|---|---|---|---|
| Emberz guardian v2 (ours) | 90/181 = **50%** (43-57%) | 25/120 = **21%** (15-29%) | 78% | 0.61 | 64% | 0.65 (0.59-0.70) |
| Kimi K3 | 175/181 = **97%** (93-98%) | 31/120 = **26%** (19-34%) | 85% | 0.90 | 85% | 0.86 (0.81-0.90) |
| Qwen3.8 Max | 173/181 = **96%** (92-98%) | 31/120 = **26%** (19-34%) | 85% | 0.90 | 85% | 0.87 (0.82-0.90) |
| Nemotron Lightning 3.5 | 166/181 = **92%** (87-95%) | 28/120 = **23%** (17-32%) | 86% | 0.89 | 84% | 0.86 (0.82-0.90) |
| DeepSeek V4.1 Flash | 166/181 = **92%** (87-95%) | 40/120 = **33%** (26-42%) | 81% | 0.86 | 79% | 0.81 (0.76-0.85) |
| Rune (decisions API) | 163/181 = **90%** (85-94%) | 49/120 = **41%** (32-50%) | 77% | 0.83 | 75% | 0.76 (0.71-0.81) |
| keyword stub | 3/181 = **2%** (1-5%) | 11/120 = **9%** (5-16%) | 21% | 0.03 | 46% | 0.46 (0.43-0.49) |

AUC treats the call's highest rating (none < watch < alert < urgent) as a 4-level score: 0.5 is chance, 1.0 is perfect separation. CI: stratified bootstrap, 2000 resamples, seed 20261102.

## 2. Three operating points

Flag the call if its highest rating reaches watch, alert (official) or urgent.

| Model | >= watch: recall / false alarms | >= alert: recall / false alarms | >= urgent: recall / false alarms |
|---|---|---|---|
| Emberz guardian v2 (ours) | 99% / 98% | 50% / 21% | 3% / 2% |
| Kimi K3 | 100% / 97% | 97% / 26% | 3% / 0% |
| Qwen3.8 Max | 99% / 96% | 96% / 26% | 14% / 0% |
| Nemotron Lightning 3.5 | 97% / 61% | 92% / 23% | 12% / 0% |
| DeepSeek V4.1 Flash | 100% / 90% | 92% / 33% | 7% / 0% |
| Rune (decisions API) | 96% / 41% | 90% / 41% | 0% / 0% |
| keyword stub | 2% / 9% | 2% / 9% | 0% / 0% |

Highest rating per call (scam calls / benign calls, counts of none, watch, alert, urgent):

- Emberz guardian v2 (ours): scam [1, 90, 85, 5], benign [3, 92, 23, 2]
- Kimi K3: scam [0, 6, 170, 5], benign [4, 85, 31, 0]
- Qwen3.8 Max: scam [1, 7, 148, 25], benign [5, 84, 31, 0]
- Nemotron Lightning 3.5: scam [5, 10, 144, 22], benign [47, 45, 28, 0]
- DeepSeek V4.1 Flash: scam [0, 15, 154, 12], benign [12, 68, 40, 0]
- Rune (decisions API): scam [8, 10, 163, 0], benign [71, 0, 49, 0]
- keyword stub: scam [178, 0, 3, 0], benign [109, 0, 11, 0]

## 3. How early the alert comes (scam calls the model flagged, official rule)

Median (IQR). Seconds = end of the flagged caller turn, when the request is sent; add the model latency for when the alert would show. Words heard = share of the call's caller words spoken by then (lower = earlier).

| Model | Flagged scams | Seconds into call | Seconds incl. model latency | Caller turn # | Words heard | Flagged on 1st caller turn |
|---|---|---|---|---|---|---|
| Emberz guardian v2 (ours) | 90 | 14.6 (12.9-18.7) | 16.1 (14.2-20.2) | 1 (1-2) | 0.81 (0.61-1.00) | 55/90 |
| Kimi K3 | 175 | 14.3 (12.1-14.9) | 17.1 (14.8-18.9) | 1 (1-1) | 0.78 (0.56-0.99) | 142/175 |
| Qwen3.8 Max | 173 | 14.3 (11.4-14.9) | 16.2 (14.3-19.3) | 1 (1-1) | 0.73 (0.54-0.98) | 146/173 |
| Nemotron Lightning 3.5 | 166 | 14.3 (11.4-15.1) | 14.9 (11.9-16.2) | 1 (1-1) | 0.73 (0.52-0.98) | 135/166 |
| DeepSeek V4.1 Flash | 166 | 14.5 (12.9-17.4) | 15.9 (14.2-18.6) | 1 (1-2) | 0.81 (0.58-1.00) | 111/166 |
| Rune (decisions API) | 163 | 14.4 (10.8-15.3) | 14.5 (10.9-15.4) | 1 (1-1) | 0.72 (0.54-0.98) | 134/163 |
| keyword stub | 3 | 13.6 (11.1-14.9) | 13.6 (11.1-14.9) | 1 (1-2) | 1.00 (0.78-1.00) | 2/3 |

Same 88 scams for every model (the ones all 6 full-coverage LLM guardians flagged), so the times compare like for like:

| Model | Seconds into call | Caller turn # | Words heard |
|---|---|---|---|
| Emberz guardian v2 (ours) | 14.6 (12.9-18.7) | 1 (1-2) | 0.81 (0.61-1.00) |
| DeepSeek V4.1 Flash | 14.5 (12.4-15.7) | 1 (1-2) | 0.70 (0.56-0.95) |
| Kimi K3 | 14.3 (11.0-14.8) | 1 (1-1) | 0.67 (0.46-0.93) |
| Nemotron Lightning 3.5 | 14.2 (10.3-14.8) | 1 (1-1) | 0.66 (0.44-0.93) |
| Qwen3.8 Max | 14.2 (10.2-14.7) | 1 (1-1) | 0.66 (0.46-0.93) |
| Rune (decisions API) | 14.2 (10.2-14.8) | 1 (1-1) | 0.66 (0.46-0.93) |

Scam robocalls are short (median 2 caller turn(s)), so these times mostly measure how much of a monologue the model needs before it is sure, not how early it would catch a live conversation.

## 4. Paired comparison (same calls, official rule)

Difference = row model minus column model. McNemar exact test on per-call right/wrong; paired bootstrap 95% CI of the difference.

| A vs B | Calls | Recall A-B (95% CI) | McNemar p (scams) | False alarms A-B (95% CI) | McNemar p (benign) |
|---|---|---|---|---|---|
| Emberz guardian v2 (ours) vs DeepSeek V4.1 Flash | 301 | -42 pts (-49 to -35) | <0.001 (0:76) | -12 pts (-18 to -7) | <0.001 (15:0) |
| Emberz guardian v2 (ours) vs Kimi K3 | 301 | -47 pts (-54 to -40) | <0.001 (0:85) | -5 pts (-12 to +2) | 0.210 (11:5) |
| Emberz guardian v2 (ours) vs Nemotron Lightning 3.5 | 301 | -42 pts (-49 to -35) | <0.001 (0:76) | -2 pts (-12 to +8) | 0.736 (19:16) |
| Emberz guardian v2 (ours) vs Qwen3.8 Max | 301 | -46 pts (-53 to -39) | <0.001 (0:83) | -5 pts (-12 to +2) | 0.238 (12:6) |
| Emberz guardian v2 (ours) vs Rune (decisions API) | 301 | -40 pts (-48 to -33) | <0.001 (2:75) | -20 pts (-28 to -13) | <0.001 (24:0) |
| Emberz guardian v2 (ours) vs keyword stub | 301 | +48 pts (+41 to +55) | <0.001 (87:0) | +12 pts (+6 to +19) | 0.001 (2:16) |
| DeepSeek V4.1 Flash vs Kimi K3 | 301 | -5 pts (-8 to -2) | 0.004 (0:9) | +8 pts (+1 to +14) | 0.049 (4:13) |
| DeepSeek V4.1 Flash vs Nemotron Lightning 3.5 | 301 | +0 pts (-4 to +4) | 1.000 (6:6) | +10 pts (+0 to +21) | 0.096 (16:28) |
| DeepSeek V4.1 Flash vs Qwen3.8 Max | 301 | -4 pts (-7 to -1) | 0.016 (0:7) | +8 pts (+1 to +14) | 0.049 (4:13) |
| DeepSeek V4.1 Flash vs Rune (decisions API) | 301 | +2 pts (-3 to +6) | 0.629 (10:7) | -8 pts (-12 to -3) | 0.004 (9:0) |
| DeepSeek V4.1 Flash vs keyword stub | 301 | +90 pts (+86 to +94) | <0.001 (163:0) | +24 pts (+17 to +32) | <0.001 (1:30) |
| Kimi K3 vs Nemotron Lightning 3.5 | 301 | +5 pts (+2 to +9) | 0.012 (10:1) | +2 pts (-8 to +12) | 0.749 (18:21) |
| Kimi K3 vs Qwen3.8 Max | 301 | +1 pts (-2 to +4) | 0.688 (4:2) | +0 pts (-7 to +8) | 1.000 (9:9) |
| Kimi K3 vs Rune (decisions API) | 301 | +7 pts (+3 to +10) | 0.002 (13:1) | -15 pts (-22 to -9) | <0.001 (18:0) |
| Kimi K3 vs keyword stub | 301 | +95 pts (+92 to +98) | <0.001 (172:0) | +17 pts (+8 to +25) | <0.001 (5:25) |
| Nemotron Lightning 3.5 vs Qwen3.8 Max | 301 | -4 pts (-7 to -1) | 0.065 (2:9) | -2 pts (-12 to +7) | 0.736 (19:16) |
| Nemotron Lightning 3.5 vs Rune (decisions API) | 301 | +2 pts (-3 to +6) | 0.629 (10:7) | -18 pts (-28 to -8) | 0.002 (32:11) |
| Nemotron Lightning 3.5 vs keyword stub | 301 | +90 pts (+85 to +94) | <0.001 (163:0) | +14 pts (+5 to +23) | 0.006 (9:26) |
| Qwen3.8 Max vs Rune (decisions API) | 301 | +6 pts (+2 to +9) | 0.013 (12:2) | -15 pts (-22 to -8) | <0.001 (20:2) |
| Qwen3.8 Max vs keyword stub | 301 | +94 pts (+90 to +97) | <0.001 (170:0) | +17 pts (+8 to +25) | <0.001 (5:25) |
| Rune (decisions API) vs keyword stub | 301 | +88 pts (+83 to +93) | <0.001 (160:0) | +32 pts (+22 to +40) | <0.001 (1:39) |

(x:y) = calls only A got right : calls only B got right. No correction for multiple comparisons.

## 5. Where models win and lose

Scam recall by type:

| Model | amazon_apple_order | auto_warranty | bank_credit_card | delivery_parcel | gov_ssa_irs | prize_travel_health | tech_support_utility |
|---|---|---|---|---|---|---|---|
| Emberz guardian v2 (ours) | 29/38 | 0/19 | 19/33 | 0/3 | 31/35 | 1/11 | 10/42 |
| Kimi K3 | 37/38 | 19/19 | 33/33 | 3/3 | 35/35 | 10/11 | 38/42 |
| Qwen3.8 Max | 37/38 | 18/19 | 33/33 | 3/3 | 35/35 | 10/11 | 37/42 |
| Nemotron Lightning 3.5 | 35/38 | 19/19 | 33/33 | 2/3 | 34/35 | 11/11 | 32/42 |
| DeepSeek V4.1 Flash | 35/38 | 17/19 | 33/33 | 3/3 | 35/35 | 9/11 | 34/42 |
| Rune (decisions API) | 33/38 | 18/19 | 31/33 | 3/3 | 35/35 | 10/11 | 33/42 |
| keyword stub | 0/38 | 0/19 | 2/33 | 0/3 | 0/35 | 0/11 | 1/42 |

False alarms by HarperValley task:

| Model | check_balance | get_branch_hours | order_checks | pay_bill | replace_card | reset_password | schedule_appointment | transfer_money |
|---|---|---|---|---|---|---|---|---|
| Emberz guardian v2 (ours) | 0/16 | 0/16 | 6/15 | 10/15 | 0/15 | 8/15 | 0/14 | 1/14 |
| Kimi K3 | 0/16 | 0/16 | 13/15 | 8/15 | 0/15 | 9/15 | 0/14 | 1/14 |
| Qwen3.8 Max | 1/16 | 1/16 | 7/15 | 11/15 | 0/15 | 8/15 | 0/14 | 3/14 |
| Nemotron Lightning 3.5 | 0/16 | 0/16 | 3/15 | 1/15 | 13/15 | 6/15 | 0/14 | 5/14 |
| DeepSeek V4.1 Flash | 2/16 | 0/16 | 10/15 | 15/15 | 0/15 | 11/15 | 0/14 | 2/14 |
| Rune (decisions API) | 2/16 | 0/16 | 14/15 | 15/15 | 2/15 | 11/15 | 0/14 | 5/14 |
| keyword stub | 0/16 | 0/16 | 5/15 | 6/15 | 0/15 | 0/15 | 0/14 | 0/14 |

Category of the first alert on each false-alarm call:

- Emberz guardian v2 (ours): disclosure.location_school 8, disclosure.credentials_otp 8, disclosure.financial_id 6, scam.impersonation_bank_gov 3
- Kimi K3: disclosure.location_school 16, disclosure.credentials_otp 9, scam.impersonation_bank_gov 4, disclosure.financial_id 2
- Qwen3.8 Max: scam.impersonation_bank_gov 12, disclosure.location_school 10, disclosure.financial_id 4, scam.tech_support 3, disclosure.credentials_otp 2
- Nemotron Lightning 3.5: scam.impersonation_bank_gov 20, scam.tech_support 5, scam.family_emergency 3
- DeepSeek V4.1 Flash: disclosure.location_school 17, scam.impersonation_bank_gov 13, disclosure.credentials_otp 7, scam.tech_support 2, disclosure.financial_id 1
- Rune (decisions API): scam.impersonation_bank_gov 30, none 12, disclosure.financial_id 3, scam.tech_support 3, disclosure.credentials_otp 1
- keyword stub: scam.impersonation_bank_gov 9, disclosure.location_school 2

## 6. Cost, latency, reliability

| Model | Cost (USD) | per 1,000 calls | per call-minute | p50 / p95 latency | Turn errors | Re-runs + JSON-mode retries | Pricing basis |
|---|---|---|---|---|---|---|---|
| Emberz guardian v2 (ours) | $2.53 | $8.42 | $0.0132 | 860 / 1558 ms | 0/1233 | 24 + 0 | dedicated GPU $8.0/h; cost = summed request time (serial use), excludes idle time |
| Kimi K3 | $3.29 | $10.93 | $0.0171 | 2324 / 4547 ms | 0/1233 | 0 + 0 | serverless $3.0/$15.0 per 1M in/out tokens |
| Qwen3.8 Max | $3.50 | $11.63 | $0.0182 | 3024 / 6084 ms | 0/1233 | 8 + 97 | serverless $2.0/$6.0 per 1M in/out tokens |
| Nemotron Lightning 3.5 | $0.06 | $0.19 | $0.0003 | 449 / 1345 ms | 0/1233 | 4 + 93 | serverless $0.05/$0.2 per 1M in/out tokens |
| DeepSeek V4.1 Flash | $0.31 | $1.04 | $0.0016 | 1213 / 1816 ms | 0/1233 | 0 + 0 | serverless $0.3/$1.2 per 1M in/out tokens |
| Rune (decisions API) | $0.05 | $0.15 | $0.0002 | 107 / 188 ms | 0/1233 | 0 + 0 | serverless $0.042/$0.0 per 1M in/out tokens |
| keyword stub | $0.00 | $0.00 | $0.0000 | - | 0/1233 | 0 + 0 | serverless $0.0/$0.0 per 1M in/out tokens |

## 7. Ablation: let each model pick its own operating point on dev

Split: calls.jsonl 'split' field; 151 dev calls, 150 test calls. For each model the point (watch / alert / urgent) is chosen on dev only (max balanced accuracy with false alarms <= 25%); the table shows test calls only, next to the official alert rule on the same test calls.

| Model | Chosen on dev | Test recall (95% CI) | Test false alarms (95% CI) | Test balanced acc. | Official rule on test: recall / FA / bal. acc. |
|---|---|---|---|---|---|
| Emberz guardian v2 (ours) | alert | 47/90 = 52% (42-62%) | 15/60 = 25% (16-37%) | 64% | 52% / 25% / 64% |
| Kimi K3 | alert | 88/90 = 98% (92-99%) | 17/60 = 28% (19-41%) | 85% | 98% / 28% / 85% |
| Qwen3.8 Max | alert | 85/90 = 94% (88-98%) | 17/60 = 28% (19-41%) | 83% | 94% / 28% / 83% |
| Nemotron Lightning 3.5 | alert | 84/90 = 93% (86-97%) | 15/60 = 25% (16-37%) | 84% | 93% / 25% / 84% |
| DeepSeek V4.1 Flash | urgent | 5/90 = 6% (2-12%) | 0/60 = 0% (0-6%) | 53% | 92% / 35% / 79% |
| Rune (decisions API) | urgent | 0/90 = 0% (0-4%) | 0/60 = 0% (0-6%) | 50% | 90% / 43% / 73% |
| keyword stub | urgent | 0/90 = 0% (0-4%) | 0/60 = 0% (0-6%) | 50% | 1% / 8% / 46% |

### Slice split=dev

| Model | Recall (95% CI) | False alarms (95% CI) | Balanced acc. | AUC (95% CI) |
|---|---|---|---|---|
| Emberz guardian v2 (ours) | 43/91 = 47% (37-57%) | 10/60 = 17% (9-28%) | 65% | 0.65 (0.57-0.72) |
| Kimi K3 | 87/91 = 96% (89-98%) | 14/60 = 23% (14-35%) | 86% | 0.87 (0.81-0.92) |
| Qwen3.8 Max | 88/91 = 97% (91-99%) | 14/60 = 23% (14-35%) | 87% | 0.88 (0.83-0.93) |
| Nemotron Lightning 3.5 | 82/91 = 90% (82-95%) | 13/60 = 22% (13-34%) | 84% | 0.86 (0.80-0.92) |
| DeepSeek V4.1 Flash | 83/91 = 91% (84-95%) | 19/60 = 32% (21-44%) | 80% | 0.81 (0.75-0.88) |
| Rune (decisions API) | 82/91 = 90% (82-95%) | 23/60 = 38% (27-51%) | 76% | 0.77 (0.70-0.84) |
| keyword stub | 2/91 = 2% (1-8%) | 6/60 = 10% (5-20%) | 46% | 0.46 (0.42-0.50) |

### Slice split=test

| Model | Recall (95% CI) | False alarms (95% CI) | Balanced acc. | AUC (95% CI) |
|---|---|---|---|---|
| Emberz guardian v2 (ours) | 47/90 = 52% (42-62%) | 15/60 = 25% (16-37%) | 64% | 0.64 (0.57-0.71) |
| Kimi K3 | 88/90 = 98% (92-99%) | 17/60 = 28% (19-41%) | 85% | 0.85 (0.79-0.90) |
| Qwen3.8 Max | 85/90 = 94% (88-98%) | 17/60 = 28% (19-41%) | 83% | 0.85 (0.79-0.90) |
| Nemotron Lightning 3.5 | 84/90 = 93% (86-97%) | 15/60 = 25% (16-37%) | 84% | 0.86 (0.80-0.91) |
| DeepSeek V4.1 Flash | 83/90 = 92% (85-96%) | 21/60 = 35% (24-48%) | 79% | 0.80 (0.74-0.86) |
| Rune (decisions API) | 81/90 = 90% (82-95%) | 26/60 = 43% (32-56%) | 73% | 0.75 (0.68-0.82) |
| keyword stub | 1/90 = 1% (0-6%) | 5/60 = 8% (4-18%) | 46% | 0.46 (0.42-0.50) |

### Slice batch=v1

| Model | Recall (95% CI) | False alarms (95% CI) | Balanced acc. | AUC (95% CI) |
|---|---|---|---|---|
| Emberz guardian v2 (ours) | 31/61 = 51% (39-63%) | 9/50 = 18% (10-31%) | 66% | 0.67 (0.58-0.75) |
| Kimi K3 | 58/61 = 95% (87-98%) | 10/50 = 20% (11-33%) | 88% | 0.88 (0.82-0.94) |
| Qwen3.8 Max | 57/61 = 93% (84-97%) | 9/50 = 18% (10-31%) | 88% | 0.89 (0.83-0.94) |
| Nemotron Lightning 3.5 | 56/61 = 92% (82-96%) | 11/50 = 22% (13-35%) | 85% | 0.87 (0.80-0.92) |
| DeepSeek V4.1 Flash | 55/61 = 90% (80-95%) | 15/50 = 30% (19-44%) | 80% | 0.82 (0.75-0.89) |
| Rune (decisions API) | 58/61 = 95% (87-98%) | 20/50 = 40% (28-54%) | 78% | 0.79 (0.71-0.86) |
| keyword stub | 1/61 = 2% (0-9%) | 4/50 = 8% (3-19%) | 47% | 0.47 (0.43-0.51) |

### Slice batch=v2

| Model | Recall (95% CI) | False alarms (95% CI) | Balanced acc. | AUC (95% CI) |
|---|---|---|---|---|
| Emberz guardian v2 (ours) | 59/120 = 49% (40-58%) | 16/70 = 23% (15-34%) | 63% | 0.63 (0.56-0.70) |
| Kimi K3 | 117/120 = 98% (93-99%) | 21/70 = 30% (21-42%) | 84% | 0.84 (0.78-0.90) |
| Qwen3.8 Max | 116/120 = 97% (92-99%) | 22/70 = 31% (22-43%) | 83% | 0.85 (0.79-0.90) |
| Nemotron Lightning 3.5 | 110/120 = 92% (85-95%) | 17/70 = 24% (16-36%) | 84% | 0.86 (0.80-0.91) |
| DeepSeek V4.1 Flash | 111/120 = 92% (86-96%) | 25/70 = 36% (26-47%) | 78% | 0.80 (0.74-0.86) |
| Rune (decisions API) | 105/120 = 88% (80-92%) | 29/70 = 41% (31-53%) | 73% | 0.75 (0.68-0.81) |
| keyword stub | 2/120 = 2% (0-6%) | 7/70 = 10% (5-19%) | 46% | 0.46 (0.42-0.49) |

## Caveats

- Scams are one-sided prerecorded robocalls (NCSU, public domain); benign calls are role-played HarperValley bank calls (CC BY 4.0). The classes come from different sources, so style (monologue vs dialogue) can leak the label. Read this as a real-audio check, not the headline benchmark.
- Precision and F1 depend on the scam share in this set (about 60%); real call traffic has far fewer scams, so real-world precision would be lower.
- With 181 scam and 120 benign calls (about half that per split), a 5-10 point gap can be within noise: check the paired CIs and McNemar p before calling a winner.
- Time to first alert on robocalls measures how much of a short monologue a model needs, not how early it catches a live conversation.
- Dedicated-GPU cost for our guardian is summed request time at $8/h (one request at a time), excluding idle time: a rough figure, not an invoice. Token prices for models not listed in `PRICES` are left blank unless given with `--price`.
- Failed or unparseable turns count as risk none. Nothing here was tuned on these calls except the dev-side choice in section 7, which is reported on test only.


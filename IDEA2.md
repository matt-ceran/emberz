# Idea 2: make a cheap small model as good as a frontier model at one niche task, by editing its weights

Fireworks hackathon, November 2026. Numan and Matt. Written 2026-11-02. Status: idea, nothing built yet. The main plan (Emberz) lives on `numan/line-voice-gcp`.

## 1. The concept

**Pay a frontier model once to make a small model as good as itself at your task, instead of paying it on every call.**

1. You type: "Make Qwen3-8B better at routing bank support tickets."
2. Claude works in a Python session with the small model loaded. It reads the weights, runs small test batches and edits the weights directly. There is no training run and no gradient descent.
3. The edited model ships to Fireworks and serves traffic.
4. A chart shows base vs edited, with the frontier model's score as a reference line. Under it: cost per 1,000 calls, latency, and how long the edit took and what it cost.

One niche task first (see section 5). Each edit targets one task. We never claim the small model got better at everything, and we show it didn't get worse at other things.

## 2. Where it comes from

- **Weight-Space Programming** (Takehi, Teoh, Zhang, Khattab, MIT CSAIL, 2026-11-02): https://rikiyat.github.io/blog/weight-space-programming/ . A frontier model (GPT-6 Astra) reads and writes neural-network weights directly.
  - It named which game a policy network was trained on from the raw weights alone.
  - It wrote game-playing networks from scratch that beat PPO trained for 100M steps.
  - It edited 0.016% of Qwen3.5-0.8B's weights in a Python session, with 50 forward-pass batches and no training: MATH-500 7.9% → 48.1%, GSM8K 13.8% → 67.1%, IFEval 21.0% → 62.3%. The edit fixed the reasoning loops (93% of answers hit the token limit before, 33% after).
  - It could not spot a sycophantic model from the weights alone, only from attention patterns.
  - Full logs (prompts, code, networks): https://github.com/RikiyaT/lm_understand-logs
- **Neuron Persona Library** (Joshua Fonseca Rivera): https://neuronpersonas.com/ . Every MLP neuron of 9 open models (Qwen3-8B and 14B, Llama 3.1 8B, Gemma 4 12B, OLMo 3 7B, Phi-4-mini and others) pushed up and down and asked "Who are you?", with replies labelled by Claude Haiku. The data is public JSON (`data/<model>/library.json`, `combos.json`). A possible map of where behaviours live; one prompt only. No license seen, so cite it and ask before redistributing.

**What's different here:** the paper is a research demo measured against the model's own base. This is a production use, measured against a frontier model on price, latency and quality, served on Fireworks.

## 3. Roles

| Role | Who |
|---|---|
| Editor | Claude, run as a Claude Code session with Python and the model loaded (subscription; no API key needed). Every session transcript is saved to the repo, like the paper's logs |
| Model being edited | Qwen3-8B to start. Bigger (14B-32B) needs an 80 GB GPU we'd have to rent |
| Main baseline | Claude doing the task directly from a prompt (the "pay every call" side) |
| Second baseline | Kimi K3 or GLM 5.3 on Fireworks serverless |
| Retrain baseline | LoRA SFT of the same small model on Fireworks, with the same examples the editor saw |
| Serving | Fireworks dedicated deployment with the edited weights (LoRA adapter or full model upload) |

Open question: do the hackathon rules allow a non-Fireworks model as the core editor? To check.

## 4. The judging rubric and how this scores

| Criterion | Weight | How we score |
|---|---|---|
| Benchmark | 40% | Same held-out test set for every system. Base vs edited, frontier reference, confidence intervals, cost per 1,000 calls, latency. The rubric accepts efficiency gains that keep useful quality, so matching the frontier model within the error bars at a fraction of the price counts |
| Business value | 25% | "Pay once instead of every call." Break-even: the edit pays for itself after K calls. Who: any team running a repetitive task through a frontier API |
| Technical execution | 20% | Edit vs retrain on the same data, the per-call cost of patching the prompt, what else broke, and the failure cases |
| Presentation | 10% | The "Make X better at Y" box, then the bar chart. A notebook is allowed |
| Documentation | rest | One command reruns the eval; edit transcripts and edited weights published |

Originality is not scored. Fine-tuning and an app are optional.

## 5. The niche

Pick by measuring, not guessing: the task where the base small model is clearly weaker than the frontier model, so there is room to climb, and where outputs are short, so test batches are cheap.

| Candidate | Data | Why |
|---|---|---|
| Bank support ticket routing (default) | Banking77, 77 intents | Real, repetitive, public, short labels |
| Intent detection with out-of-scope | CLINC150 | Same, plus "none of these" handling |
| Field extraction from messages | To pick | A second task type, if time allows |

Doing it on 2-3 tasks shows it's a method, not a lucky one-off.

> **Update 1:45 PM (Matt + Claude): proposed niche change, see §11.** Banking77 is the weakest candidate for an edit. Screen sycophancy, over-refusal and CLINC150 out-of-scope first.

## 6. Benchmark table

Same held-out test cases (100+, more is better) for every row:

| System | Accuracy (95% CI) | Cost per 1k calls | Latency (p50) |
|---|---|---|---|
| Qwen3-8B base | | | |
| Qwen3-8B edited by Claude | | | |
| Qwen3-8B LoRA SFT on the same examples | | | |
| Claude, from a prompt | | | |
| Claude, with the editor's examples pasted into the prompt | | | |
| Kimi K3 on Fireworks | | | |

Plus: edit time, edit cost at API prices (from the session's token count), and a regression check on tasks the edit wasn't for.

Cost caveat: Fireworks bills Kimi K3 per token, but uploaded weights run on a dedicated deployment billed per GPU hour. Cost per 1k for our model is reported at a stated load.

## 7. Fireworks facts (checked 2026-11-02)

From https://docs.fireworks.ai/models/uploading-custom-models and https://docs.fireworks.ai/fine-tuning/deploying-loras:
- Uploaded LoRA adapters are accepted: `adapter_config.json` plus `adapter_model.safetensors`, rank 4-64, target modules `q/k/v/o_proj`, `up/down/gate_proj` (and `lm_head` for some families).
- Full custom models (Hugging Face format) can be uploaded too, so any edit can ship even if it doesn't fit a LoRA.
- Uploaded LoRAs deploy only on dedicated (on-demand) deployments, not serverless. Multi-LoRA needs a BF16 shape (`--enable-addons` doesn't work on FP8/FP4).
- Serverless models on our key include `kimi-k3`, `glm-5p3`, `qwen3p8-max`, `qwen3p8-2p4t-a95b`, `deepseek-v4p1-flash`, `minimax-m3`, `nemotron-3-ultra-nvfp4`, most with 1M context.

## 8. Risks

- **A healthy model has fewer easy fixes.** The paper's jump came from fixing a broken model (loops). On a normal task the edit may gain much less. Step 3 is the go/no-go.
- **Pasting examples into the prompt may work as well.** That baseline is in the table; our answer is that every prompt patch adds cost and latency to every call.
- **The edit may break other things.** Measured by the regression check.
- **Rules:** whether Claude can be the core editor.

## 9. The start

1. **Pick the niche:** base Qwen3-8B, Claude and Kimi K3 on Banking77 and CLINC150. Keep the task with the biggest gap. A few dollars.
2. **Prove the path to Fireworks:** upload a test adapter to a dedicated deployment and call it.
3. **First edit:** a Claude Code session with Qwen3-8B loaded (Mac if fast enough, otherwise the Google Cloud L4) tries to close the gap on one task without breaking anything. Go/no-go for the whole idea.
4. **Then:** the full table, more tasks, the demo box and chart.

Spending (L4 hours, Fireworks deployment hours, API calls) gets an OK before it starts.

## 10. Deep dive: how the paper's Qwen fix actually worked (read from their logs, 2026-11-02)

Source: `exp8_math/` in https://github.com/RikiyaT/lm_understand-logs (logs CC BY 4.0, scripts MIT, so we can reuse the task prompt and rules with attribution).

**Setup the editor had**
- A persistent Python kernel over the weights, **CPU only, no GPU, no internet**, at most 300 code runs. Helpers: `load(name)`, `save_edit(name, tensor)`, `drop_edit`, the tokenizer.
- It was **not allowed to run the model itself**. Only a forward-pass tool could, with a budget of 10, 25 or 50 passes. One pass was either:
  - a generation of up to 512 answers ("both" = paired original vs edited on the same problems and seeds, with the paired difference and its standard error), optionally saving activations; or
  - a teacher-forced read of up to 1,024 texts returning log-probs, hidden states, MLP activations and attention. Takes seconds to minutes.
- **Forbidden:** gradients, optimizers, any iterative fitting of weights to a loss; edits to the final norm, embeddings or output layer; multiplying a whole tensor by a constant.
- The prompt told it to diagnose first, localize causally (dose-response, not just correlation), measure with paired comparisons and keep an edit only if its gain beats its standard error on problems not used to choose it.

**What it found and changed**
- 85 of its first 96 practice answers ran to the 32,768-token limit while repeating. It contrasted matched productive vs repetitive contexts, found a "repetition direction", and checked it with dose-response tests, layer isolation and a norm-matched random control.
- 50-pass edit: subtract that direction from the 64 MLP output columns that write it most, in layers 15 and 19 (131,072 weights). 10-pass edit: a rank-one projection `W' = W - 1.5 u(u^T W)` on 30 matrices. 25-pass edit: zero one attention head.
- Its own held-out check: 10.6% → 60.4% on 1,152 practice problems, 579 rescues vs 5 regressions; 576 of the rescues were answers that used to be cut off.

**How long and how reliable**
- **Not minutes: hours.** The 50-pass run took 158 turns, 80 Python runs, 4 conversation restarts and 20.3 hours (a full generation pass took up to ~30 min). The 10-pass run took 11.6 h.
- **One run per setting, and fragile.** Runs not in the post: 0 and 5 passes made no edit; a shorter prompt (v4) with 10 passes made no edit. Only the single 50-pass run gave the big jump.
- **Gemma 3 270M (no loops): nothing.** MATH-500 6.8% → at best 7.7%. The errors were ordinary reasoning mistakes with no single component to switch off. **Editing removes an obstacle that hides an ability; it doesn't add an ability.**

**What this changes for us**
1. **Niche choice rule:** the small model must already *have* the ability, with a removable obstacle in the way (format failures, a bias toward some labels, loops, over-refusal, language mixing). Plain confusions between similar labels are Gemma-like and probably won't move. The first measurement must break base Qwen3-8B's errors down into "obstacle" vs "confusion" before we commit to a niche.
2. **Time claim:** report measured edit time honestly. Our passes are much cheaper (short label outputs, 8B on an L4: seconds per batch, not 30 min), so we expect far less than 20 h, but we measure it.
3. **Reliability:** run 3 independent edit sessions per setting and report all of them, not the best one.
4. **Credibility of "no training":** copy their rules (no gradients, a budget of forward passes through a tool, `save_edit` with refused edit types) and publish full transcripts, as they did.
5. **Harness:** their prompt (`runs/p50c/message.txt`) is a ready template: diagnose, localize causally, paired measurement, keep only edits that beat their standard error on unused data.

**Niches that fit "removable obstacle" (to test alongside Banking77/CLINC150)**
- Invalid or off-list outputs and label bias in classification (if base Qwen3-8B shows them).
- Over-refusal: refusing harmless requests. Refusal is known to run largely through a single direction. Must keep refusing truly harmful requests, measured on both sides (e.g. XSTest / OR-Bench style sets).
- Loops in thinking mode on long tasks (the paper's exact failure).
- Language mixing (Qwen drifting into Chinese mid-answer).

## 11. Proposed niche change (Matt + Claude, 2026-11-02 1:45 PM)

**Why change it.** The paper's big jump came from a model that was *broken in one systematic way*: Qwen3.5-0.8B looped, and 93% of its MATH-500 answers hit the token limit. The edit removed that obstacle; it did not add math skill. On Gemma 3 270M, whose errors were ordinary, the edit gained essentially nothing (MATH-500 6.8 to at most 7.7). Banking77's typical errors are confusions between near-identical intents (for example card arrival vs card delivery estimate), which is missing knowledge, the Gemma case. **So pick the niche by the *type* of error, not only by the size of the gap.**

**What makes a good niche:**
1. The base model fails in one systematic way, and there is prior evidence that a single edit fixes that kind of failure.
2. Grading is automatic and objective, with short outputs, so test batches are cheap.
3. It has a built-in counter-metric showing the edit didn't just flip the model into the opposite failure.
4. It has a business line and a gap to the frontier model.

**Candidates** (all datasets checked on Hugging Face):

| Niche | How the base model breaks | Prior evidence that one edit fixes it | Counter-metric ("did it break?") | Business line |
|---|---|---|---|---|
| **1. Caving under pushback** (`meg-tong/sycophancy-eval`, MIT) | Drops a correct answer when the user says "I think it's X" | Sycophancy was steered with a single direction (Panickssery/Rimsky et al. 2024). The paper's Exp 7 found one head (10.2) doing it | Still accepts the user when the user is right; plain accuracy unchanged | "Your support bot agrees with customers who are wrong" |
| **2. Over-refusal** (`Paul/XSTest`, CC-BY-4.0: 250 safe-looking + 200 unsafe contrast prompts) | Refuses "how do I kill a stuck process?" | Refusal is one direction (Arditi et al. 2024) | The 200 unsafe prompts must still be refused | "Bots that refuse paying customers" |
| 3. Looping on math (`openai/gsm8k`; Qwen3-1.7B or R1-Distill-Qwen-1.5B) | Loops until the token cap | The paper itself | Regression suite below | Cost per correct answer |
| 4. Out-of-scope (`clinc/clinc_oos`, CC-BY-3.0) | Forces a real intent instead of "none" | A plausible single bias, untested | In-scope accuracy | Routing that admits it doesn't know |
| 5. Banking77 (`PolyAI/banking77`, CC-BY-4.0) | Confuses near-identical intents | Weak: knowledge errors, the Gemma case | | Keep as the contrast where LoRA wins |

**Pick without measuring: #1, caving under pushback.**
- It has the strongest prior for a single-direction fix.
- Its outputs are short.
- Its counter-metric is honest: a model that never changes its mind is also broken.
- The business story is easy to grasp.

Caveats on the others:
- **#2:** Qwen models are fairly permissive, so it may have little room on Qwen3-8B. Llama 3.1 8B or Gemma would be stronger targets.
- **#2 safety:** the edit weakens refusal, so the weights stay private unless the unsafe-prompt refusal rate is unchanged.
- **#3:** generations are long, so test batches are slow.

**Regression suite** (the "we show it didn't get worse" part of §1), run before and after every edit:
- 200 questions from `cais/mmlu`
- 100 from `openai/gsm8k`
- 100 from `google/IFEval`
- Perplexity on plain text
- The niche's own counter-metric

**Proposed start (replaces §9 step 1):**
1. **About 35 min:** a screen of base Qwen3-8B and Kimi K3 on about 100 prompts each from #1, #2 and #4.
2. **Go/no-go:** pick the niche with the biggest *systematic* failure and the biggest gap to Kimi. A few dollars of Kimi calls; Qwen3-8B runs on the Mac at bf16 (16 GB of 24 GB; speed not yet measured).
3. **Then:** §9 steps 2-4 on that one niche.

**Rules to fix up front.** On a healthy model, the edits most likely to work compute directions from labelled examples' activations (difference of means). Decide and write down what counts as "no training". The paper forbade backward passes, optimizers and iterative fitting, and refused edits to the embedding and output layers. Disclose it in the write-up.

**Evidence for the open question in §3 (can the editor be a non-Fireworks model?).** If the rules require a Fireworks editor, `weightsmith/` shows Fireworks models can do weight reasoning to a limited degree. On the paper's Exp 5 (write a MinAtar Breakout network from the game's source code; 200 held-out episodes):
- DeepSeek V4.1 Flash with **one** feedback pass: **25.6 ± 3.8**. That ties PPO at 500M steps (25.3) for about $0.12.
- Qwen3.8 2.4T one-shot: **20.8 ± 2.6**.
- Astra: 517.4.

The harness already runs any Fireworks model as the editor. Details: `weightsmith/PLAN.md`.

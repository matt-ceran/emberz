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

> **Update 1:45 PM (Matt + Claude): proposed niche change, see §10.** Banking77 is probably the weakest candidate for an edit (not yet measured). Screen over-refusal, sycophancy and CLINC150 out-of-scope first.

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

## 10. Proposed niche change (Matt + Claude, 2026-11-02 1:45 PM)

**Why change it.** The paper's big jump came from a model that was *broken in one systematic way*: Qwen3.5-0.8B looped, and 93% of its MATH-500 answers hit the token limit. The edit removed that obstacle; it did not add math skill. On Gemma 3 270M, whose errors were ordinary, the edit gained essentially nothing (MATH-500 6.8 to at most 7.7). We expect Banking77's errors to be confusions between near-identical intents (e.g. `card_arrival` vs `card_delivery_estimate`). If so, they look more like Gemma's case (no single component to switch off) than Qwen's loops. Not yet measured; the screen checks it. **So pick the niche by the *type* of error, not only by the size of the gap.**

**What makes a good niche:**
1. The base model fails in one systematic way, and there is prior evidence that a single edit fixes that kind of failure.
2. Grading is automatic and objective, with short outputs, so test batches are cheap.
3. It has a built-in counter-metric showing the edit didn't just flip the model into the opposite failure.
4. It has a business line and a gap to the frontier model.

**Candidates** (all datasets checked on Hugging Face):

| Niche | How the base model breaks | Prior evidence that one edit fixes it | Counter-metric ("did it break?") | Business line |
|---|---|---|---|---|
| **1. Caving under pushback** (`meg-tong/sycophancy-eval`, MIT) | Drops a correct answer when the user says "I think it's X" | Opinion sycophancy was steered with one vector (CAA: Panickssery/Rimsky et al. 2024, on Llama 2). The paper's Exp 7 only *detected* a sycophantic model, from middle-layer attention heads (e.g. head 10.2); it did not edit one | Still accepts the user when the user is right; plain accuracy unchanged | "Your support bot agrees with customers who are wrong" |
| **2. Over-refusal** (`Paul/XSTest`, CC-BY-4.0: 250 safe-looking + 200 unsafe contrast prompts) | Refuses "how do I kill a stuck process?" | **Most direct precedent:** ablating a single "false-refusal" vector cut over-refusal while keeping safety, training-free (Wang, Hu, Röttger & Plank 2024, arXiv 2410.03415). Refusal itself is one direction (Arditi et al. 2024), but removing *that* removes all refusal | The 200 unsafe prompts must still be refused | "Bots that refuse paying customers" |
| 3. Looping on math (`openai/gsm8k`; Qwen3-1.7B or R1-Distill-Qwen-1.5B) | Likely loops until the token cap (both model cards warn of endless repetitions under non-recommended sampling; not measured on GSM8K) | The paper (Qwen3.5-0.8B on MATH-500) | Regression suite below | Cost per correct answer |
| 4. Out-of-scope (`clinc/clinc_oos`, CC-BY-3.0) | Forces a real intent instead of "none" | A plausible single bias, untested | In-scope accuracy | Routing that admits it doesn't know |
| 5. Banking77 (`PolyAI/banking77`, CC-BY-4.0) | Confuses near-identical intents | Weak: knowledge errors, the Gemma case | | Keep as the contrast where LoRA wins |

**Let the screen pick. On priors alone:**
- **#2, over-refusal, has the most direct precedent** for a single-vector fix that keeps safety (Wang et al. 2024). Its risk is room: Qwen may rarely over-refuse.
- **#1, caving under pushback, has the better business story** and an honest counter-metric (a model that never changes its mind is also broken). Its prior is weaker: CAA on Llama 2 opinion sycophancy.
- Both need thinking mode off to keep outputs short.

Caveats on the others:
- **#2:** Qwen models are fairly permissive, so it may have little room on Qwen3-8B. Llama 3.1 8B or Gemma may be stronger targets (to check in the screen).
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
2. **Go/no-go:** pick the niche with the biggest *systematic* failure and the biggest gap to Kimi. A few dollars of Kimi calls; Qwen3-8B should fit on the Mac at bf16 (8.2B params ≈ 16.4 GB of 24 GB; not yet loaded, speed not measured).
3. **Then:** §9 steps 2-4 on that one niche.

**Rules to fix up front.** We expect the edits most likely to work on a healthy model to compute directions from labelled examples' activations (difference of means, as in Arditi et al. and CAA). Decide and write down what counts as "no training". The paper forbade backward passes, optimizers and iterative fitting, and refused edits to the embedding and output layers. Disclose it in the write-up.

**Evidence for the open question in §3 (can the editor be a non-Fireworks model?).** If the rules require a Fireworks editor, `weightsmith/` shows Fireworks models can do weight reasoning to a limited degree. On a re-run of the paper's Exp 5 task (write a MinAtar Breakout network from the game's source code), scored on 200 held-out episodes in our harness. The paper's PPO and Astra numbers come from its own 1,000-episode, 600-input setup:
- DeepSeek V4.1 Flash with **one** feedback pass: **25.6 ± 3.8**. That ties PPO at 500M steps (25.3) for about $0.11.
- Qwen3.8 2.4T one-shot: **20.8 ± 2.6**.
- Astra: 517.4.

The harness already runs any Fireworks model as the editor. Details: `weightsmith/PLAN.md`.

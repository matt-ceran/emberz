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

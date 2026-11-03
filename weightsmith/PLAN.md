# WeightSmith: can open models on Fireworks do weight-space programming?

**Status (1:45 PM, 11/2):** Matt stopped every run at 1:02 PM. This folder is the feasibility evidence and harness behind the niche plan in `IDEA2.md` §10. It is not a separate project.

## The discovery (Takehi et al., 11/2; rikiyat.github.io/blog/weight-space-programming)

A frontier LLM ("GPT-6 Astra", run through Codex CLI; reasoning effort xhigh in Exp 5, max in Exp 8) reads and writes neural-network weights by reasoning about them, with no gradient descent.

- **Exp 5, write a network from scratch.** Astra got the MinAtar Breakout source code and wrote a 600→32→32→6 tanh policy in one reply. It scored **517.4**, against **25.3 for PPO at 500M steps**.
- **Exp 8, edit an LLM.** On Qwen3.5-0.8B, Astra had a Python kernel over the weights and a budget of forward passes. It found a "loop direction" and subtracted 6× its component from the MLP output matrices in layers 15 and 19: `W' = W − 6·u(uᵀW)` on 64 columns per layer, 131,072 weights in total.

| Exp 8 run | MATH-500 | GSM8K | Wall-clock |
|---|---|---|---|
| Unedited | 7.9 | 13.8 | — |
| 10 passes | 12.3 | 25.7 | 11.6 h |
| 25 passes | 14.1 | 25.1 | 8.5 h |
| 50 passes | **48.1** | **67.1** | 20.3 h |

  The gain came from fixing a loop: 93% of answers hit the token limit before the edit and 33% after. The edit did not add math skill.
- **Negative control.** Gemma 3 270M, which rarely loops (8.5% of its answers hit the token limit), gained essentially nothing: MATH-500 went from 6.8 to at most 7.7.
- **Logs:** github.com/RikiyaT/lm_understand-logs. `exp8_math/` includes the exact edit (`edit.npz`) and `apply_edit.py`.

## Our question

Astra is not available to us. Can the open models Fireworks serves act as the weight editor? We tested the cheapest version, Exp 5 on Breakout.

## Results (Breakout, all on CPU)

Setup:
- Network: 400→32→32→6 tanh. Breakout has 4 input channels, which is why the input is 400 rather than the post's 600.
- Play: actions are sampled, sticky actions 0.1, cap of 10,000 steps.
- Scoring: **200 held-out eval episodes** (seed 0). Feedback rounds use 32 practice episodes on different seeds.
- **Declared deviation:** the model returns numpy code that constructs the weights. That code may not simulate, search, fit, or import minatar; the restriction is enforced by an import/regex check plus a 60 s subprocess sandbox.

| Policy | Mean score ± 95% CI | Notes |
|---|---|---|
| All-zero weights (uniform random) | 0.5 ± 0.1 | floor |
| Hand-coded "move toward ball x" (not a network) | 10.1 ± 1.4 | reference |
| Hand-coded brick-aware landing predictor (not a network) | 117.5 ± 11.9 | reference; it simulates, so it breaks the rules above |
| PPO 100M / 500M steps (post, 1,000 episodes, 600-input net) | 15.8 / 25.3 | baseline |
| Astra one-shot (post) | 517.4 | ceiling |
| Kimi K3 one-shot, 2 samples | 7.4 ± 1.2 and 3.4 ± 0.2 | 103-230 s, 6-13k output tokens |
| DeepSeek V4.1 Flash one-shot, 2 samples | 7.3 ± 0.9; the other sample hit the 32k token cap with no code | 149-177 s |
| Nemotron 3 Ultra one-shot, 2 samples | 3.7 ± 0.6; the other sample hit the 32k cap | 168-230 s |
| MiniMax M3, GLM 5.3, Qwen3.8 Max, Qwen3.8 2.4T one-shot (32k cap) | no code: all four spent the whole 32k cap reasoning | |
| **Qwen3.8 2.4T (A95B) one-shot, 100k cap** | **20.8 ± 2.6** | 556 s, 71.6k output tokens |
| Qwen3.8 Max one-shot, 100k cap | HTTP 504 gateway timeout at 600 s | needs streaming |
| **DeepSeek V4.1 Flash + feedback** (`breakout_refine.py`, run 1) | round 0: **10.9 ± 1.7** → round 2: **25.6 ± 3.8** | ~93k output tokens (≈ $0.11 at $0.30/$1.20 per 1M tokens), about 9 min. Round 1 hit the cap, so round 2 was designed after **one** practice forward pass. The run crashed on a 429 in round 3. |
| Kimi K3 + feedback | round 0 practice mean 2.6; round 1 was an IndentationError | stopped |

**What it shows:**
- Open models on Fireworks write valid, correctly indexed networks.
- With **one** feedback pass, DeepSeek V4.1 Flash statistically **ties** PPO at 500M steps (25.6 ± 3.8 vs 25.3) for about 11 cents.
- Qwen3.8 2.4T one-shot (20.8) beats PPO at 100M steps.
- Both are far below Astra (517).
- **Their limit is strategy, not mechanics.** Kimi built a one-unit "stay under the ball" comparator and called it "essentially optimal". DeepSeek's round-2 fix aims one step ahead of the ball (`target = 2·ball_x − trail_x`, plus a wall correction).
- Caveat: the post's PPO numbers come from its own setup (600-input network, 1,000 episodes).

**Why it matters for Idea2:** IDEA2.md asks whether the rules allow Claude as the editor. If they don't, these numbers show Fireworks models can do weight reasoning to a limited degree. The harness below already runs any Fireworks model as the editor.

## Files

- `probe/breakout_oneshot.py`: one-shot write test; any model list; `WS_MAX_TOKENS` sets the token cap.
- `probe/breakout_refine.py`: feedback loop with a forward-pass budget, Breakout only. Has retries on 429/5xx.
- `probe/ws_minatar.py`: generic harness for any MinAtar game. Scores every round on held-out episodes and has retries. **Its streaming (SSE) change is untested.** Smoke-tested offline; never run end to end.
- `probe/ref_policies.py`: the two hand-coded reference policies (10.1 and 117.5; reproduced at 1:45 PM).
- `probe/summarize.py`: collects `runs/*/*/result.json` from `ws_minatar.py` into one table.
- `probe/runs/`: raw replies (`*.json`, `round*.md`), the generated weight code (`*.py`), and the weights (`*.npz`).
- Setup: `cd weightsmith && uv venv .venv && uv pip install --python .venv/bin/python minatar numpy`. The scripts read `FIREWORKS_API_KEY` from the repo-root `.env`.

## Known issues

- Reasoning-heavy models need far more than 32k tokens, and the long replies need streaming to avoid the ~600 s gateway timeout.
- The shared Fireworks key returned HTTP 429 under load. CallGuard's bulk generation uses the same DeepSeek model.
- Zero-weight floors for the other games, measured offline (200 episodes): Seaquest 0.1, Asterix 0.6. No model has been scored on them yet.

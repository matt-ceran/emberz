"""Weight-space programming harness for any MinAtar game (generalizes breakout_oneshot.py / breakout_refine.py).

An LLM on Fireworks writes numpy code that constructs the weights of a fixed (100*C)->32->32->6 tanh policy from the
game's source code alone. Round 0 is one-shot. Each later round costs one "forward pass": the network plays
PRACTICE_EPISODES practice episodes (seeds disjoint from eval) and the model sees the score plus ASCII frames of the
last steps of the worst episodes, then rewrites the weights. No training, fitting, search or simulation is allowed
inside the code. The final and best-on-practice networks are scored on held-out eval episodes.

Usage: .venv/bin/python probe/ws_minatar.py --game seaquest --model deepseek-v4p1-flash --rounds 3 [--tag r1]
"""
import argparse, json, os, re, subprocess, sys, tempfile, time, urllib.request
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
API = "https://api.fireworks.ai/inference/v1/chat/completions"
H1, H2, N_ACT = 32, 32, 6
MAX_TOKENS = int(os.environ.get("WS_MAX_TOKENS", "64000"))
PRACTICE_SEED, EVAL_SEED, PRACTICE_EPISODES = 1000, 0, 32
ACTION_NAMES = ['n', 'l', 'u', 'r', 'd', 'f']
LETTERS = "ABCDEFGHIJ"


def api_key():
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("FIREWORKS_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("FIREWORKS_API_KEY missing from .env")


def make_env(game, seed):
    from minatar import Environment
    env = Environment(game, sticky_action_prob=0.1)
    env.seed(seed)
    return env


def game_info(game):
    import minatar
    env = make_env(game, 0)
    chans = sorted(env.env.channels.items(), key=lambda kv: kv[1])
    src = (Path(minatar.__file__).parent / "environments" / f"{game}.py").read_text()
    minimal = [ACTION_NAMES[i] for i in env.minimal_action_set()]
    return env.n_channels, chans, src, minimal


def shapes(game):
    C = game_info(game)[0]
    return {"W1": (H1, 100 * C), "b1": (H1,), "W2": (H2, H1), "b2": (H2,), "W3": (N_ACT, H2), "b3": (N_ACT,)}


def prompt(game):
    C, chans, src, minimal = game_info(game)
    ch = ", ".join(f"{n}={i}" for n, i in chans)
    return f"""You will design the weights of a small neural network policy for the MinAtar game `{game}`, by reasoning alone.
There is no training data, no gradient descent, no simulator access and no feedback before scoring.

## Game source code (exactly what the policy will be scored on)
```python
{src}
```
The environment wrapper adds sticky actions: with probability 0.1 the previous action is repeated instead of the chosen one.
Difficulty ramping is ON (default MinAtar settings). Episodes are capped at 10,000 steps. Score = total reward per episode,
averaged over many episodes.

## Network (fixed architecture, you choose every weight)
- Input x: `state.astype(np.float32).reshape(-1)`, length {100*C}, where `state` has shape (10, 10, {C}) indexed [y, x, channel].
  Flat index = (y*10 + x)*{C} + c. Channels: {ch}. y=0 is the top row.
- h1 = tanh(W1 @ x + b1), W1 shape (32, {100*C}), b1 shape (32,)
- h2 = tanh(W2 @ h1 + b2), W2 shape (32, 32), b2 shape (32,)
- logits = W3 @ h2 + b3, W3 shape (6, 32), b3 shape (6,)
- The action is SAMPLED from softmax(logits). Action indices: 0='n', 1='l', 2='u', 3='r', 4='d', 5='f' (see action_map in the source).
  Actions that matter in this game: {minimal}. The network has no memory between steps.

## Your job
Reason about what the network must compute to play well, then assign weights that implement it.
Make the logits large enough that sampling is close to deterministic where you want it to be.

Return exactly one Python code block defining `build_weights()` that returns a dict with float32 numpy arrays
"W1","b1","W2","b2","W3","b3" of the shapes above. Rules for the code: import only numpy (and math); do not simulate the game,
do not search over or fit weights to any data, do not import minatar. Construct the weights analytically from your design.
Briefly explain the design (what each group of hidden units computes) before the code block."""


BANNED = re.compile(r"\b(minatar|subprocess|socket|open\(|__import__|eval\(|exec\(|os\.|sys\.|shutil|pathlib|requests|urllib)")


def extract_code(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    blocks = [b for b in re.findall(r"```(?:python)?\n(.*?)```", text, flags=re.S) if "def build_weights" in b]
    return blocks[-1] if blocks else None


def build(code, game):
    """Run the model's code in an isolated subprocess (numpy only, 60 s) and load the weights."""
    m = BANNED.search(code)
    if m:
        return None, f"banned token: {m.group(0)}"
    for imp in re.findall(r"^\s*(?:import|from)\s+([\w.]+)", code, flags=re.M):
        if imp.split(".")[0] not in ("numpy", "math"):
            return None, f"disallowed import {imp}"
    with tempfile.TemporaryDirectory() as td:
        Path(td, "wcode.py").write_text(code + "\n\nimport numpy as _np\n_w = build_weights()\n_np.savez('w.npz', **{k: _np.asarray(v, dtype=_np.float32) for k, v in _w.items()})\n")
        try:
            p = subprocess.run([sys.executable, "-I", "wcode.py"], cwd=td, capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            return None, "build timed out (60 s)"
        if p.returncode != 0:
            return None, "build error: " + p.stderr[-600:]
        w = dict(np.load(Path(td, "w.npz")))
    for k, s in shapes(game).items():
        if k not in w or w[k].shape != s:
            return None, f"bad shape {k}: {None if k not in w else w[k].shape}, want {s}"
        if not np.isfinite(w[k]).all():
            return None, f"non-finite values in {k}"
    return w, None


def policy_probs(w, st):
    z = w["W3"] @ np.tanh(w["W2"] @ np.tanh(w["W1"] @ st.astype(np.float32).reshape(-1) + w["b1"]) + w["b2"]) + w["b3"]
    p = np.exp(z - z.max()); return p / p.sum()


def rollout(w, game, episodes, seed, keep_last=0):
    env = make_env(game, seed); rng = np.random.default_rng(seed + 1)
    out = []
    for ep in range(episodes):
        env.reset(); total, t, done, hist = 0.0, 0, False, []
        while not done and t < 10_000:
            st = env.state(); p = policy_probs(w, st); a = int(rng.choice(N_ACT, p=p))
            if keep_last:
                hist.append((st, a, p)); hist = hist[-keep_last:]
            r, done = env.act(a); total += r; t += 1
        out.append({"ep": ep, "score": total, "steps": t, "done": done, "hist": hist})
    return out


def evaluate(w, game, episodes=200, seed=EVAL_SEED):
    s = np.array([e["score"] for e in rollout(w, game, episodes, seed)])
    return {"mean": float(s.mean()), "ci95": float(1.96 * s.std(ddof=1) / np.sqrt(len(s))), "median": float(np.median(s)),
            "max": float(s.max()), "n": len(s)}


def render(st, chans):
    rows = []
    for y in range(10):
        row = ""
        for x in range(10):
            on = [c for _, c in chans if st[y, x, c]]
            row += LETTERS[on[-1]] if on else " "
        rows.append(f"|{row}|")
    return rows


def feedback(w, game, rnd):
    C, chans, _, _ = game_info(game)
    eps = rollout(w, game, PRACTICE_EPISODES, PRACTICE_SEED, keep_last=6)
    s = np.array([e["score"] for e in eps])
    worst = sorted(eps, key=lambda e: e["score"])[:2]
    traces = []
    for e in worst:
        frames = []
        for i, (st, a, p) in enumerate(e["hist"]):
            probs = " ".join(f"{ACTION_NAMES[j]}={p[j]:.2f}" for j in range(N_ACT))
            frames.append(f"step -{len(e['hist'])-i}: chose '{ACTION_NAMES[a]}'  probs: {probs}\n" + "\n".join(render(st, chans)))
        end = "episode ended" if e["done"] else "hit the step cap"
        traces.append(f"Practice episode {e['ep']} (score {e['score']:.0f}, {end} at step {e['steps']}). Last {len(e['hist'])} frames:\n" + "\n\n".join(frames))
    legend = ", ".join(f"'{LETTERS[c]}'={n}" for n, c in chans) + " (if several channels are on in a cell, the higher channel index is shown)"
    text = (f"Forward pass {rnd+1}: your network played {PRACTICE_EPISODES} practice episodes. Mean score {s.mean():.2f} "
            f"(median {np.median(s):.0f}, min {s.min():.0f}, max {s.max():.0f}).\n\nThe two lowest-scoring episodes:\n\n"
            + "\n\n".join(traces) + f"\n\nLegend: {legend}. y=0 is the top row.\n\n"
            "Diagnose what went wrong and improve the design. Return the full revised build_weights() code block. "
            "Same rules: numpy/math only, no simulation, search or fitting inside the code.")
    return float(s.mean()), text


def _stream(req):
    """Read a Fireworks SSE stream (avoids the ~600 s gateway timeout on long reasoning replies)."""
    content, reasoning, usage, finish = [], [], {}, None
    with urllib.request.urlopen(req, timeout=300) as r:  # timeout applies per read, not to the whole stream
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                if d.get("content"): content.append(d["content"])
                if d.get("reasoning_content"): reasoning.append(d["reasoning_content"])
                finish = ch.get("finish_reason") or finish
    return "".join(content), "".join(reasoning), usage, finish


def chat(model, key, messages):
    body = {"model": f"accounts/fireworks/models/{model}", "messages": messages, "max_tokens": MAX_TOKENS,
            "temperature": 0.6, "reasoning_effort": "high", "stream": True, "stream_options": {"include_usage": True}}
    t0 = time.time()
    for attempt in range(8):
        req = urllib.request.Request(API, data=json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                                              "Accept": "text/event-stream"})
        try:
            content, reasoning, usage, _ = _stream(req); break
        except urllib.error.HTTPError as e:
            msg = e.read().decode()[:400]
            if "reasoning_effort" in msg and "reasoning_effort" in body: body.pop("reasoning_effort"); continue
            if "stream_options" in msg and "stream_options" in body: body.pop("stream_options"); continue
            if e.code in (429, 500, 502, 503, 504) and attempt < 7: time.sleep(min(120, 10 * 2 ** attempt)); continue
            raise RuntimeError(f"HTTP {e.code}: {msg}")
        except (TimeoutError, urllib.error.URLError, ConnectionError):
            if attempt < 7: time.sleep(15); continue
            raise
    return content, reasoning, usage, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="breakout")
    ap.add_argument("--model", default="deepseek-v4p1-flash")
    ap.add_argument("--rounds", type=int, default=3, help="forward passes after the one-shot round (0 = one-shot only)")
    ap.add_argument("--eval-episodes", type=int, default=200)
    ap.add_argument("--tag", default="r1")
    a = ap.parse_args()
    key = api_key(); out = HERE / "runs" / a.game / f"{a.model}__{a.tag}"; out.mkdir(parents=True, exist_ok=True)
    messages = [{"role": "user", "content": prompt(a.game)}]
    log, valid, tok_out, tok_in, wall = [], {}, 0, 0, 0.0
    for rnd in range(a.rounds + 1):
        content, reasoning, usage, secs = chat(a.model, key, messages)
        tok_out += usage.get("completion_tokens", 0); tok_in += usage.get("prompt_tokens", 0); wall += secs
        messages.append({"role": "assistant", "content": content})
        (out / f"round{rnd}.md").write_text(content)
        code = extract_code(content) or extract_code(reasoning)
        entry = {"round": rnd, "secs": round(secs, 1), "completion_tokens": usage.get("completion_tokens")}
        if not code:
            entry["error"] = "no build_weights code block"
            fb = "Your reply had no ```python block defining build_weights(). Keep the reasoning shorter and return one."
        else:
            (out / f"round{rnd}.py").write_text(code)
            w, err = build(code, a.game)
            if err:
                entry["error"] = err; fb = f"Your build_weights() could not be used: {err}\nFix it and return the full code block again."
            else:
                np.savez(out / f"round{rnd}.npz", **w); valid[rnd] = w
                if rnd < a.rounds:
                    entry["practice_mean"], fb = feedback(w, a.game, rnd)
        log.append(entry); print(json.dumps({"game": a.game, "model": a.model, "tag": a.tag, **entry}), flush=True)
        if rnd < a.rounds:
            messages.append({"role": "user", "content": fb})
    res = {"game": a.game, "model": a.model, "tag": a.tag, "rounds": a.rounds, "max_tokens": MAX_TOKENS,
           "prompt_tokens_total": tok_in, "completion_tokens_total": tok_out, "llm_secs_total": round(wall, 1), "log": log}
    if valid:
        res["eval"] = {f"round{r}": evaluate(w, a.game, a.eval_episodes) for r, w in valid.items()}
        res["final_round"] = max(valid)
    (out / "result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "log"}), flush=True)


if __name__ == "__main__":
    main()

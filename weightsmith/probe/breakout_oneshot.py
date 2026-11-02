"""Feasibility probe: can open models on Fireworks write a MinAtar Breakout policy network's
weights from the game rules alone (no data, no training, no rollouts)?

Mirrors Exp 5 of rikiyat.github.io/blog/weight-space-programming, with one declared deviation:
the model returns a numpy function that *constructs* the weights analytically, rather than
pasting ~14k numbers as text. The code may not import minatar, simulate, search, or fit.

Usage:  .venv/bin/python probe/breakout_oneshot.py [--episodes 200] [--samples 2] [--models kimi-k3,deepseek-v4p1-flash]
"""
import argparse, concurrent.futures as cf, json, os, re, subprocess, sys, tempfile, time, urllib.request
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "runs"
ROOT = HERE.parent.parent
API = "https://api.fireworks.ai/inference/v1/chat/completions"
D_IN, H1, H2, N_ACT = 400, 32, 32, 6
MAX_TOKENS = int(os.environ.get("WS_MAX_TOKENS", "32000"))


def api_key():
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("FIREWORKS_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("FIREWORKS_API_KEY missing from .env")


def prompt():
    import minatar
    src = (Path(minatar.__file__).parent / "environments" / "breakout.py").read_text()
    return f"""You will design the weights of a small neural network policy for the game MinAtar Breakout, by reasoning alone.
There is no training data, no gradient descent, no simulator access and no feedback before scoring.

## Game source code (exactly what the policy will be scored on)
```python
{src}
```
The environment wrapper adds sticky actions: with probability 0.1 the previous action is repeated instead of the chosen one.
Episodes are capped at 10,000 steps. Score = total reward (bricks broken) per episode, averaged over many episodes.

## Network (fixed architecture, you choose every weight)
- Input x: `state.astype(np.float32).reshape(-1)`, length 400, where `state` has shape (10, 10, 4) indexed [y, x, channel].
  So flat index = (y*10 + x)*4 + c, with channels paddle=0, ball=1, trail=2 (ball's previous cell), brick=3. y=0 is the top row, the paddle is on row y=9.
- h1 = tanh(W1 @ x + b1), W1 shape (32, 400), b1 shape (32,)
- h2 = tanh(W2 @ h1 + b2), W2 shape (32, 32), b2 shape (32,)
- logits = W3 @ h2 + b3, W3 shape (6, 32), b3 shape (6,)
- The action is SAMPLED from softmax(logits). Action indices: 0='n' (no-op), 1='l', 2='u', 3='r', 4='d', 5='f'. Only 'l' (1) and 'r' (3) move the paddle; 0, 2, 4, 5 do nothing.
- The network has no memory between steps.

## Your job
Reason about what the network must compute to play well (e.g. where the ball is, which way it is moving, where it will land, where the paddle is), then assign weights that implement it.
Make the logits large enough that sampling is close to deterministic.

Return exactly one Python code block defining `build_weights()` that returns a dict with float32 numpy arrays
"W1","b1","W2","b2","W3","b3" of the shapes above. Rules for the code: import only numpy (and math); do not simulate the game,
do not search over or fit weights to any data, do not import minatar. Construct the weights analytically from your design.
Briefly explain the design (what each group of hidden units computes) before the code block."""


def call_model(model, key, temperature=0.6):
    body = {"model": f"accounts/fireworks/models/{model}", "messages": [{"role": "user", "content": prompt()}],
            "max_tokens": MAX_TOKENS, "temperature": temperature, "reasoning_effort": "high"}
    t0 = time.time()
    for attempt in range(2):
        req = urllib.request.Request(API, data=json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                resp = json.load(r)
            break
        except urllib.error.HTTPError as e:
            msg = e.read().decode()[:400]
            if "reasoning_effort" in msg and "reasoning_effort" in body:
                body.pop("reasoning_effort"); continue
            return {"error": f"HTTP {e.code}: {msg}", "secs": time.time() - t0}
        except Exception as e:
            return {"error": repr(e)[:300], "secs": time.time() - t0}
    msg = resp["choices"][0]["message"]
    return {"content": msg.get("content") or "", "reasoning": msg.get("reasoning_content") or "",
            "usage": resp.get("usage", {}), "finish": resp["choices"][0].get("finish_reason"), "secs": time.time() - t0}


BANNED = re.compile(r"\b(minatar|subprocess|socket|open\(|__import__|eval\(|exec\(|os\.|sys\.|shutil|pathlib|requests|urllib)")


def extract_code(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    blocks = re.findall(r"```(?:python)?\n(.*?)```", text, flags=re.S)
    blocks = [b for b in blocks if "def build_weights" in b]
    return blocks[-1] if blocks else None


def build(code, workdir):
    """Run the model's code in an isolated subprocess (numpy only, 60 s) and load the weights."""
    if BANNED.search(code):
        return None, f"banned token: {BANNED.search(code).group(0)}"
    for imp in re.findall(r"^\s*(?:import|from)\s+([\w.]+)", code, flags=re.M):
        if imp.split(".")[0] not in ("numpy", "math"):
            return None, f"disallowed import {imp}"
    (workdir / "wcode.py").write_text(code + "\n\nimport numpy as _np\n_w = build_weights()\n_np.savez('w.npz', **{k: _np.asarray(v, dtype=_np.float32) for k, v in _w.items()})\n")
    p = subprocess.run([sys.executable, "-I", "wcode.py"], cwd=workdir, capture_output=True, text=True, timeout=60)
    if p.returncode != 0:
        return None, "build error: " + p.stderr[-600:]
    w = dict(np.load(workdir / "w.npz"))
    want = {"W1": (H1, D_IN), "b1": (H1,), "W2": (H2, H1), "b2": (H2,), "W3": (N_ACT, H2), "b3": (N_ACT,)}
    for k, s in want.items():
        if k not in w or w[k].shape != s:
            return None, f"bad shape {k}: {None if k not in w else w[k].shape}, want {s}"
        if not np.isfinite(w[k]).all():
            return None, f"non-finite values in {k}"
    return w, None


def evaluate(w, episodes, seed=0, max_steps=10_000):
    from minatar import Environment
    env = Environment("breakout", sticky_action_prob=0.1)
    env.seed(seed)
    rng = np.random.default_rng(seed + 1)
    scores, lengths = [], []
    for _ in range(episodes):
        env.reset(); total, t, done = 0.0, 0, False
        while not done and t < max_steps:
            x = env.state().astype(np.float32).reshape(-1)
            z = w["W3"] @ np.tanh(w["W2"] @ np.tanh(w["W1"] @ x + w["b1"]) + w["b2"]) + w["b3"]
            p = np.exp(z - z.max()); p /= p.sum()
            r, done = env.act(int(rng.choice(N_ACT, p=p)))
            total += r; t += 1
        scores.append(total); lengths.append(t)
    s = np.array(scores)
    return {"mean": float(s.mean()), "ci95": float(1.96 * s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 1 else 0.0,
            "median": float(np.median(s)), "max": float(s.max()), "mean_steps": float(np.mean(lengths)), "n": len(s)}


def run_one(model, i, key, episodes):
    tag = f"{model}__s{i}" + (f"__mt{MAX_TOKENS}" if MAX_TOKENS != 32000 else "")
    res = {"model": model, "sample": i}
    r = call_model(model, key)
    res.update({k: r.get(k) for k in ("secs", "usage", "finish", "error")})
    (OUT / f"{tag}.json").write_text(json.dumps(r, indent=1))
    if "error" in r and r["error"]:
        return res
    code = extract_code(r["content"]) or extract_code(r["reasoning"])
    if not code:
        res["error"] = "no build_weights code block"; return res
    (OUT / f"{tag}.py").write_text(code)
    with tempfile.TemporaryDirectory() as td:
        w, err = build(code, Path(td))
    if err:
        res["error"] = err; return res
    np.savez(OUT / f"{tag}.npz", **w)
    res["score"] = evaluate(w, episodes)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="kimi-k3,deepseek-v4p1-flash,nemotron-3-ultra-nvfp4")
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--episodes", type=int, default=200)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    key = api_key()
    zero = {k: np.zeros(s, np.float32) for k, s in {"W1": (H1, D_IN), "b1": (H1,), "W2": (H2, H1), "b2": (H2,), "W3": (N_ACT, H2), "b3": (N_ACT,)}.items()}
    results = [{"model": "baseline: all-zero weights (uniform random actions)", "score": evaluate(zero, a.episodes)}]
    print(json.dumps(results[0]), flush=True)
    jobs = [(m, i) for m in a.models.split(",") for i in range(a.samples)]
    with cf.ThreadPoolExecutor(len(jobs)) as ex:
        futs = {ex.submit(run_one, m, i, key, a.episodes): (m, i) for m, i in jobs}
        for f in cf.as_completed(futs):
            r = f.result(); results.append(r)
            print(json.dumps(r), flush=True)
    (OUT / f"summary_{int(time.time())}.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()

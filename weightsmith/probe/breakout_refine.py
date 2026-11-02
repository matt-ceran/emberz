"""Forward-pass-budget variant of the Breakout probe (cf. Exp 6/8 of the weight-space-programming post).

Round 0 is the one-shot prompt. After each round the model gets one "forward pass": its network is played for
32 practice episodes (seeds disjoint from eval) and it sees the mean score plus ASCII frames of the last steps
before it lost the ball. It then rewrites build_weights(). Still no training, fitting or search inside the code.
The final network (last valid round) and the best-on-practice network are both scored on 200 held-out eval episodes.

Usage: .venv/bin/python probe/breakout_refine.py --model kimi-k3 --rounds 5
"""
import argparse, json, tempfile, time, urllib.request
from pathlib import Path
import numpy as np
import breakout_oneshot as B

PRACTICE_SEED, EVAL_SEED = 1000, 0
SYM = {0: "=", 1: "o", 2: ".", 3: "#"}
ACT = {0: "n", 1: "l", 2: "n", 3: "r", 4: "n", 5: "n"}


def render(st):
    rows = []
    for y in range(10):
        row = ""
        for x in range(10):
            ch = " "
            for c in (3, 2, 0, 1):  # later channels drawn on top: ball over paddle over trail over brick
                if st[y, x, c]: ch = SYM[c]
            row += ch
        rows.append(f"|{row}|")
    return rows


def practice(w, episodes=32, seed=PRACTICE_SEED, show=2, last=6):
    from minatar import Environment
    env = Environment("breakout", sticky_action_prob=0.1); env.seed(seed)
    rng = np.random.default_rng(seed + 1)
    scores, traces = [], []
    for ep in range(episodes):
        env.reset(); total, t, done, hist = 0.0, 0, False, []
        while not done and t < 10_000:
            st = env.state(); x = st.astype(np.float32).reshape(-1)
            z = w["W3"] @ np.tanh(w["W2"] @ np.tanh(w["W1"] @ x + w["b1"]) + w["b2"]) + w["b3"]
            p = np.exp(z - z.max()); p /= p.sum()
            a = int(rng.choice(6, p=p))
            hist.append((st, a, p)); hist = hist[-last:]
            r, done = env.act(a); total += r; t += 1
        scores.append(total)
        if done and len(traces) < show:
            frames = []
            for i, (st, a, p) in enumerate(hist):
                pl, pr = p[1], p[3]; pn = 1 - pl - pr
                frames.append(f"step -{len(hist)-i}: chose '{ACT[a]}'  P(l)={pl:.2f} P(r)={pr:.2f} P(no-op)={pn:.2f}\n" + "\n".join(render(st)))
            traces.append(f"Practice episode {ep} (score {total:.0f}, lost the ball at step {t}). Last {len(hist)} frames before losing:\n" + "\n\n".join(frames))
    s = np.array(scores)
    return float(s.mean()), s, traces


def chat(model, key, messages):
    body = {"model": f"accounts/fireworks/models/{model}", "messages": messages, "max_tokens": B.MAX_TOKENS,
            "temperature": 0.6, "reasoning_effort": "high"}
    t0 = time.time()
    for attempt in range(8):
        req = urllib.request.Request(B.API, data=json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                resp = json.load(r); break
        except urllib.error.HTTPError as e:
            msg = e.read().decode()[:400]
            if "reasoning_effort" in msg and "reasoning_effort" in body: body.pop("reasoning_effort"); continue
            if e.code in (429, 500, 502, 503, 504) and attempt < 7:
                time.sleep(min(120, 10 * 2 ** attempt)); continue  # rate limit / transient: back off and retry
            raise RuntimeError(f"HTTP {e.code}: {msg}")
        except (TimeoutError, urllib.error.URLError) as e:
            if attempt < 7: time.sleep(15); continue
            raise
    m = resp["choices"][0]["message"]
    return m.get("content") or "", m.get("reasoning_content") or "", resp.get("usage", {}), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="kimi-k3")
    ap.add_argument("--rounds", type=int, default=5, help="forward passes after the one-shot round")
    ap.add_argument("--eval-episodes", type=int, default=200)
    a = ap.parse_args()
    key = B.api_key(); out = B.OUT / f"refine__{a.model}"; out.mkdir(parents=True, exist_ok=True)
    messages = [{"role": "user", "content": B.prompt()}]
    log, best, last_valid, tok = [], None, None, 0
    for rnd in range(a.rounds + 1):
        content, reasoning, usage, secs = chat(a.model, key, messages)
        tok += usage.get("completion_tokens", 0)
        messages.append({"role": "assistant", "content": content})
        (out / f"round{rnd}.md").write_text(content)
        code = B.extract_code(content) or B.extract_code(reasoning)
        entry = {"round": rnd, "secs": round(secs, 1), "completion_tokens": usage.get("completion_tokens")}
        if not code:
            entry["error"] = "no build_weights code block"; fb = "Your reply had no ```python block defining build_weights(). Please return one."
        else:
            with tempfile.TemporaryDirectory() as td:
                w, err = B.build(code, Path(td))
            if err:
                entry["error"] = err; fb = f"Your build_weights() could not be used: {err}\nFix it and return the full code block again."
            else:
                np.savez(out / f"round{rnd}.npz", **w); last_valid = (rnd, w)
                mean, s, traces = practice(w)
                entry["practice_mean"] = mean
                if best is None or mean > best[1]: best = (rnd, mean, w)
                fb = (f"Forward pass {rnd+1}: your network played 32 practice episodes. Mean score {mean:.2f} "
                      f"(median {np.median(s):.0f}, max {s.max():.0f}). For reference, a uniform-random policy scores about 0.5.\n\n"
                      + ("\n\n".join(traces) if traces else "It never lost the ball within 10,000 steps.")
                      + "\n\nLegend: '#' brick, 'o' ball, '.' trail (ball's previous cell), '=' paddle, y=0 is the top row."
                        "\n\nDiagnose why the ball was lost and improve the design. Return the full revised build_weights() code block. "
                        "Same rules: numpy/math only, no simulation, search or fitting inside the code.")
        log.append(entry); print(json.dumps(entry), flush=True)
        if rnd < a.rounds:
            messages.append({"role": "user", "content": fb})
    res = {"model": a.model, "rounds": a.rounds, "completion_tokens_total": tok, "log": log}
    if last_valid:
        res["final_round"] = last_valid[0]; res["final_eval"] = B.evaluate(last_valid[1], a.eval_episodes, seed=EVAL_SEED)
    if best:
        res["best_round"] = best[0]; res["best_eval"] = B.evaluate(best[2], a.eval_episodes, seed=EVAL_SEED)
    (out / "result.json").write_text(json.dumps(res, indent=1)); print(json.dumps({k: v for k, v in res.items() if k != "log"}), flush=True)


if __name__ == "__main__":
    main()

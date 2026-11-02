"""Hand-coded Breakout reference policies (not networks), for calibrating the LLM-written networks' scores.
Same scoring as the probe: sticky actions 0.1, 10,000-step cap, 200 episodes, seed 0.
  track: move toward the ball's current x.
  land3: roll the ball forward with the game's own rules (bricks included) and move toward the landing x.
         It simulates, so it is NOT allowed for the LLM-written networks; it is only a reference.
Usage: .venv/bin/python probe/ref_policies.py
"""
import numpy as np
from minatar import Environment


def run(policy, episodes=200, seed=0):
    env = Environment("breakout", sticky_action_prob=0.1); env.seed(seed); s = []
    for _ in range(episodes):
        env.reset(); tot = 0; t = 0; done = False
        while not done and t < 10000:
            r, done = env.act(policy(env.state())); tot += r; t += 1
        s.append(tot)
    s = np.array(s); return round(float(s.mean()), 1), round(float(1.96 * s.std(ddof=1) / np.sqrt(len(s))), 1)


def pos(st, c):
    ys, xs = np.nonzero(st[:, :, c]); return (ys[0], xs[0]) if len(ys) else (None, None)


def track(st):
    _, px = pos(st, 0); _, bx = pos(st, 1)
    return 3 if bx > px else 1 if bx < px else 0


def land3(st):
    _, px = pos(st, 0); by, bx = pos(st, 1); ty, tx = pos(st, 2); bricks = st[:, :, 3].copy()
    d = {(-1, -1): 0, (1, -1): 1, (1, 1): 2, (-1, 1): 3}.get(
        (int(np.sign(bx - tx)) or (1 if bx == 0 else -1), int(np.sign(by - ty)) or 1), 2)
    x, y, tgt = bx, by, bx
    for _ in range(60):  # brick-aware rollout with the env's rules (strike flag ignored)
        lx, ly = x, y
        nx, ny = x + (1 if d in (1, 2) else -1), y + (1 if d in (2, 3) else -1)
        if nx < 0 or nx > 9: nx = min(9, max(0, nx)); d = [1, 0, 3, 2][d]
        if ny < 0: ny = 0; d = [3, 2, 1, 0][d]
        elif bricks[ny, nx]: bricks[ny, nx] = 0; ny = ly; d = [3, 2, 1, 0][d]
        elif ny == 9: tgt = nx if abs(nx - px) <= abs(lx - px) else lx; break
        x, y = nx, ny
    return 3 if tgt > px else 1 if tgt < px else 0


if __name__ == "__main__":
    print("track ball x (mean, ci95):", run(track))
    print("brick-aware landing predictor (mean, ci95):", run(land3))

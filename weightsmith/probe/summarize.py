"""Collect every ws_minatar.py result.json into one markdown table (score per round on held-out eval episodes)."""
import json
from pathlib import Path

RUNS = Path(__file__).resolve().parent / "runs"
REF = {  # from the weight-space-programming post (1,000 episodes, their 600-input net)
    "breakout": {"PPO 100M": 15.8, "PPO 500M": 25.3, "Astra one-shot": 517.4},
    "seaquest": {"PPO 100M": 5.6, "PPO 500M": 88.9, "Astra one-shot": 53.1},
    "asterix": {"PPO 100M": 25.8, "PPO 500M": 55.2, "Astra one-shot": 53.2},
}
# $ per 1M tokens (in, out); fill in as prices are confirmed
PRICE = {"deepseek-v4p1-flash": (0.30, 1.20)}


def main():
    rows = []
    for f in sorted(RUNS.glob("*/*/result.json")):
        r = json.loads(f.read_text())
        ev = r.get("eval", {})
        curve = " → ".join(f"r{k[5:]}: {v['mean']:.1f}±{v['ci95']:.1f}" for k, v in sorted(ev.items(), key=lambda kv: int(kv[0][5:])))
        errs = sum(1 for e in r["log"] if e.get("error"))
        best = max(ev.values(), key=lambda v: v["mean"])["mean"] if ev else None
        p = PRICE.get(r["model"])
        cost = f"${(r['prompt_tokens_total'] * p[0] + r['completion_tokens_total'] * p[1]) / 1e6:.2f}" if p else "?"
        ref = REF.get(r["game"], {})
        rows.append(f"| {r['game']} | {r['model']} | {r['tag']} | {r['rounds']} | {curve or '-'} | {best if best is not None else '-'} | "
                    f"{ref.get('PPO 500M', '-')} | {ref.get('Astra one-shot', '-')} | {errs} | {r['completion_tokens_total']:,} | {cost} | {r['llm_secs_total']/60:.1f} |")
    print("| Game | Model | Tag | Feedback rounds | Held-out score per round | Best | PPO 500M | Astra | Failed rounds | Output tokens | $ | LLM minutes |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    print("\n".join(rows))


if __name__ == "__main__":
    main()

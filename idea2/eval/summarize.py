"""Print a markdown table of every summary in results/<task>/<split>/.

  .venv/bin/python eval/summarize.py --split val
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="val")
    p.add_argument("--include-smoke", action="store_true")
    a = p.parse_args()
    print("| task | system | n | accuracy (95% CI) | invalid | p50 / p95 latency s | "
          "tokens/call in / out (reasoning) | cost per 1k calls | run cost |")
    print("|---|---|---|---|---|---|---|---|---|")
    total = 0.0
    for f in sorted(ROOT.glob(f"results/*/{a.split}/*.summary.json")):
        s = json.loads(f.read_text())
        if s["system"].endswith("-smoke") and not a.include_smoke:
            continue
        tpc = s["tokens_per_call"]
        cost = s["cost_usd"]
        total += cost or 0
        c1k = s["cost_per_1k_calls_usd"]
        print(f"| {s['task']} | {s['system']} | {s['n_done']}{'' if s['complete'] else ' (incomplete)'} "
              f"| {s['accuracy']:.3f} ({s['ci95'][0]:.3f}-{s['ci95'][1]:.3f}) "
              f"| {s['invalid_rate']:.1%} | {s['latency_p50_s']:.2f} / {s['latency_p95_s']:.2f} "
              f"| {tpc['prompt']:.0f} / {tpc['completion']:.0f} ({tpc['reasoning'] or 0:.0f}) "
              f"| {'n/a' if c1k is None else f'${c1k:.2f}'} | {'n/a' if cost is None else f'${cost:.2f}'} |")
    print(f"\nEstimated spend in these runs: ${total:.2f}")


if __name__ == "__main__":
    main()

"""Evaluate any OpenAI-compatible chat endpoint on one task/split.

Writes per-item rows to results/<task>/<split>/<system>.jsonl and a summary to
results/<task>/<split>/<system>.summary.json. Resumable: ids already in the
jsonl are skipped. Retries with backoff on 429 / 5xx / connection errors.

Example:
  .venv/bin/python eval/run_eval.py --task banking77 --split val --system glm-5p3 \
      --base-url https://api.fireworks.ai/inference/v1 \
      --model accounts/fireworks/models/glm-5p3 --api-key-env FIREWORKS_API_KEY \
      --env-file ../../fireworks-AMD/.env --n 1000 --concurrency 16 --max-tokens 4096
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from openai import (APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI,
                    RateLimitError)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompt import build_messages, parse, prompt_hash, system_prompt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PRICES = json.loads((Path(__file__).resolve().parent / "prices.json").read_text())


# ---------------------------------------------------------------- data
def load_split(task: str, split: str) -> list[dict]:
    if split == "test" and os.environ.get("ALLOW_TEST") != "1":
        sys.exit("Refusing to run on the test split without ALLOW_TEST=1 "
                 "(test is for final reporting only, never for selection/tuning).")
    path = ROOT / "data" / task / f"{split}.jsonl"
    return [json.loads(l) for l in path.open()]


def stratified_sample(rows: list[dict], n: int | None, seed: int = 0) -> list[dict]:
    """Proportional allocation per label (largest remainder), seeded."""
    if not n or n >= len(rows):
        return rows
    by = defaultdict(list)
    for r in rows:
        by[r["label"]].append(r)
    total = len(rows)
    quotas = {l: n * len(v) / total for l, v in by.items()}
    alloc = {l: int(q) for l, q in quotas.items()}
    rest = n - sum(alloc.values())
    for l in sorted(quotas, key=lambda l: (-(quotas[l] - alloc[l]), l))[:rest]:
        alloc[l] += 1
    rng = random.Random(seed)
    out = []
    for l in sorted(by):
        items = sorted(by[l], key=lambda r: r["id"])
        rng.shuffle(items)
        out += items[: alloc[l]]
    return sorted(out, key=lambda r: r["id"])


# ---------------------------------------------------------------- stats
def bootstrap_ci(x: np.ndarray, b: int = 2000, seed: int = 0) -> tuple[float, float]:
    if len(x) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(b, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def call_cost(row: dict, price: dict | None) -> float | None:
    if not price or price.get("input") is None:
        return None
    cached = row.get("cached_tokens") or 0
    fresh = (row.get("prompt_tokens") or 0) - cached
    return (fresh * price["input"] + cached * price["cached_input"]
            + (row.get("completion_tokens") or 0) * price["output"]) / 1e6


def summarize(rows: list[dict], expected: int, args, price: dict | None) -> dict:
    correct = np.array([r["correct"] for r in rows], dtype=float)
    lat = np.array([r["latency_s"] for r in rows], dtype=float)
    lo, hi = bootstrap_ci(correct)
    tok = lambda k: int(sum((r.get(k) or 0) for r in rows))  # noqa: E731
    costs = [call_cost(r, price) for r in rows]
    total_cost = None if any(c is None for c in costs) else float(sum(costs))
    n = len(rows)
    methods = defaultdict(int)
    for r in rows:
        methods[r["parse"]] += 1
    return {
        "task": args.task, "split": args.split, "system": args.system, "model": args.model,
        "base_url": args.base_url, "extra_body": args.extra_body_obj,
        "max_tokens": args.max_tokens, "prompt_sha": prompt_hash(args.task),
        "n_expected": expected, "n_done": n, "complete": n == expected,
        "accuracy": float(correct.mean()) if n else None,
        "ci95": [lo, hi], "bootstrap_resamples": 2000,
        "invalid_rate": methods["invalid"] / n if n else None,
        "parse_methods": dict(methods),
        "finish_length_rate": sum(r.get("finish_reason") == "length" for r in rows) / n if n else None,
        "latency_p50_s": float(np.percentile(lat, 50)) if n else None,
        "latency_p95_s": float(np.percentile(lat, 95)) if n else None,
        "tokens": {"prompt": tok("prompt_tokens"), "cached": tok("cached_tokens"),
                   "completion": tok("completion_tokens"), "reasoning": tok("reasoning_tokens")},
        "tokens_per_call": {"prompt": tok("prompt_tokens") / n if n else None,
                            "completion": tok("completion_tokens") / n if n else None,
                            "reasoning": tok("reasoning_tokens") / n if n else None},
        "cost_usd": total_cost,
        "cost_per_1k_calls_usd": total_cost / n * 1000 if (total_cost is not None and n) else None,
        "price": price,
    }


# ---------------------------------------------------------------- calls
RETRYABLE = (RateLimitError, APIConnectionError, APITimeoutError)


async def one_call(client, args, item, sem, state):
    msgs = build_messages(args.task, item["text"])
    kwargs = dict(model=args.model, messages=msgs, temperature=0, max_tokens=args.max_tokens)
    if args.extra_body_obj:
        kwargs["extra_body"] = args.extra_body_obj
    async with sem:
        if state["stop"]:
            return None
        for attempt in range(args.retries):
            t0 = time.perf_counter()
            try:
                resp = await client.chat.completions.create(**kwargs)
                latency = time.perf_counter() - t0
                break
            except Exception as e:  # noqa: BLE001
                status = getattr(e, "status_code", None)
                retry = isinstance(e, RETRYABLE) or (isinstance(e, APIStatusError) and status
                                                     and (status == 429 or status >= 500))
                if not retry or attempt == args.retries - 1:
                    print(f"[fail] {item['id']}: {type(e).__name__} {status} {str(e)[:200]}",
                          file=sys.stderr)
                    state["failed"] += 1
                    return None
                wait = min(60.0, 2 ** attempt) * (0.5 + random.random())
                ra = getattr(getattr(e, "response", None), "headers", {}) or {}
                if ra.get("retry-after"):
                    try:
                        wait = max(wait, float(ra["retry-after"]))
                    except ValueError:
                        pass
                await asyncio.sleep(wait)
    ch = resp.choices[0]
    msg = ch.message
    content = msg.content or ""
    reasoning = getattr(msg, "reasoning_content", None) or (msg.model_extra or {}).get("reasoning_content")
    u = resp.usage
    ptd = getattr(u, "prompt_tokens_details", None)
    ctd = getattr(u, "completion_tokens_details", None)
    pred, method = parse(args.task, content)
    row = {
        "id": item["id"], "gold": item["label"], "raw": content, "pred": pred,
        "correct": pred == item["label"], "parse": method, "finish_reason": ch.finish_reason,
        "latency_s": round(latency, 4),
        "prompt_tokens": getattr(u, "prompt_tokens", None),
        "cached_tokens": getattr(ptd, "cached_tokens", None) if ptd else None,
        "completion_tokens": getattr(u, "completion_tokens", None),
        "reasoning_tokens": getattr(ctd, "reasoning_tokens", None) if ctd else None,
        "reasoning_chars": len(reasoning) if reasoning else 0,
    }
    if args.save_reasoning and reasoning:
        row["reasoning"] = reasoning
    c = call_cost(row, args.price)
    if c is not None:
        state["cost"] += c
        if args.max_cost and state["cost"] >= args.max_cost and not state["stop"]:
            print(f"[stop] cost cap ${args.max_cost} reached", file=sys.stderr)
            state["stop"] = True
    return row


async def run(args) -> None:
    rows = stratified_sample(load_split(args.task, args.split), args.n, seed=0)
    out_dir = ROOT / "results" / args.task / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.system}.jsonl"
    done = {}
    if out.exists():
        for l in out.open():
            r = json.loads(l)
            done[r["id"]] = r
    want = {r["id"] for r in rows}
    todo = [r for r in rows if r["id"] not in done]
    print(f"{args.task}/{args.split}/{args.system}: {len(rows)} items, {len(rows) - len(todo)} "
          f"done, {len(todo)} to run; prompt sha {prompt_hash(args.task)} "
          f"(~{len(system_prompt(args.task))} chars)", file=sys.stderr)

    client = AsyncOpenAI(base_url=args.base_url, api_key=os.environ.get(args.api_key_env, "EMPTY"),
                         max_retries=0, timeout=args.timeout)
    sem = asyncio.Semaphore(args.concurrency)
    state = {"cost": sum(call_cost(r, args.price) or 0 for r in done.values() if r["id"] in want),
             "failed": 0, "stop": False}
    tasks = [asyncio.create_task(one_call(client, args, it, sem, state)) for it in todo]
    n_new = 0
    t0 = time.time()
    with out.open("a") as f:
        for fut in asyncio.as_completed(tasks):
            row = await fut
            if row is None:
                continue
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            done[row["id"]] = row
            n_new += 1
            if n_new % 50 == 0:
                acc = np.mean([r["correct"] for r in done.values() if r["id"] in want])
                print(f"  {n_new}/{len(todo)}  acc {acc:.3f}  cost ${state['cost']:.3f}  "
                      f"{time.time() - t0:.0f}s", file=sys.stderr)
    await client.close()

    final = [done[i] for i in sorted(want) if i in done]
    summ = summarize(final, len(rows), args, args.price)
    summ["failed_this_run"] = state["failed"]
    (out_dir / f"{args.system}.summary.json").write_text(json.dumps(summ, indent=1) + "\n")
    ci = summ["ci95"]
    print(json.dumps({k: summ[k] for k in ("n_done", "accuracy", "invalid_rate",
                                           "latency_p50_s", "cost_usd")}),
          f"CI [{ci[0]:.3f}, {ci[1]:.3f}]")


def load_env_file(path: str) -> None:
    for line in Path(path).expanduser().read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.removeprefix("export ").strip()
        os.environ.setdefault(k, v.strip().strip('"').strip("'"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=["banking77", "clinc150"])
    p.add_argument("--split", required=True, choices=["train", "val", "test"])
    p.add_argument("--system", required=True, help="name used for the results file")
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--api-key-env", default="FIREWORKS_API_KEY")
    p.add_argument("--env-file", help="optional KEY=VALUE file to load the key from")
    p.add_argument("--n", type=int, default=None, help="stratified sample size (seed 0)")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--max-tokens", type=int, default=32,
                   help="32 is enough for a label; reasoning models need ~2048-8192")
    p.add_argument("--extra-body", default=None,
                   help='JSON, e.g. \'{"chat_template_kwargs": {"enable_thinking": false}}\'')
    p.add_argument("--price-key", default=None, help="key in prices.json (default: --model)")
    p.add_argument("--max-cost", type=float, default=None, help="stop launching calls past $X")
    p.add_argument("--retries", type=int, default=8)
    p.add_argument("--timeout", type=float, default=300)
    p.add_argument("--save-reasoning", action="store_true")
    args = p.parse_args()
    if args.env_file:
        load_env_file(args.env_file)
    args.extra_body_obj = json.loads(args.extra_body) if args.extra_body else None
    args.price = PRICES.get(args.price_key or args.model)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

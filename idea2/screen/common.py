"""Shared helpers for the niche screen: model backends, resumable jsonl, bootstrap CIs, cost.

Backends (pick with --backend):
  openai : any OpenAI-compatible endpoint (Kimi K3 / GLM on Fireworks, or a vLLM server for Qwen)
  mlx    : Qwen3-8B on this Mac via mlx-lm (greedy, thinking off via the chat template)

Every call goes through `Backend.chat(messages, max_tokens) -> dict` with keys
text, reasoning, finish_reason, prompt_tokens, cached_tokens, completion_tokens,
reasoning_tokens, latency_s, gen_tps (mlx only).
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

SCREEN = Path(__file__).resolve().parent
IDEA2 = SCREEN.parent
PRICES = json.loads((IDEA2 / "eval" / "prices.json").read_text())
DEFAULT_ENV = Path.home() / "Projects" / "fireworks-AMD" / ".env"
FW = "https://api.fireworks.ai/inference/v1"


# ---------------------------------------------------------------- env / io
def load_env_file(path: str | Path) -> None:
    p = Path(path).expanduser()
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.removeprefix("export ").strip(), v.strip().strip('"').strip("'"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open()] if path.exists() else []


def done_ids(path: Path) -> set:
    return {r["id"] for r in read_jsonl(path)}


# ---------------------------------------------------------------- stats
def bootstrap_ci(x, b: int = 2000, seed: int = 0) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(b, len(x)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def fmt(x) -> str:
    x = list(x)
    if not x:
        return "n/a (n=0)"
    lo, hi = bootstrap_ci(x)
    return f"{100*np.mean(x):.1f}% [{100*lo:.1f}, {100*hi:.1f}] (n={len(x)})"


def call_cost(row: dict, model: str) -> float:
    p = PRICES.get(model)
    if not p or p.get("input") is None:
        return 0.0
    cached = row.get("cached_tokens") or 0
    fresh = (row.get("prompt_tokens") or 0) - cached
    return (fresh * p["input"] + cached * p["cached_input"]
            + (row.get("completion_tokens") or 0) * p["output"]) / 1e6


# ---------------------------------------------------------------- backends
class OpenAIBackend:
    def __init__(self, base_url: str, model: str, api_key_env: str = "FIREWORKS_API_KEY",
                 concurrency: int = 8, extra_body: dict | None = None, timeout: float = 300,
                 retries: int = 8):
        from openai import AsyncOpenAI
        self.model = model
        self.client = AsyncOpenAI(base_url=base_url, api_key=os.environ.get(api_key_env, "EMPTY"),
                                  max_retries=0, timeout=timeout)
        self.sem = asyncio.Semaphore(concurrency)
        self.extra_body = extra_body
        self.retries = retries
        self.cost = 0.0

    async def chat(self, messages: list[dict], max_tokens: int) -> dict:
        from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError
        kw = dict(model=self.model, messages=messages, temperature=0, max_tokens=max_tokens)
        if self.extra_body:
            kw["extra_body"] = self.extra_body
        async with self.sem:
            for attempt in range(self.retries):
                t0 = time.perf_counter()
                try:
                    resp = await self.client.chat.completions.create(**kw)
                    break
                except Exception as e:  # noqa: BLE001
                    st = getattr(e, "status_code", None)
                    retry = isinstance(e, (RateLimitError, APIConnectionError, APITimeoutError)) or (
                        isinstance(e, APIStatusError) and st and (st == 429 or st >= 500))
                    if not retry or attempt == self.retries - 1:
                        raise
                    wait = min(60.0, 2 ** attempt) * (0.5 + random.random())
                    hdr = getattr(getattr(e, "response", None), "headers", {}) or {}
                    try:
                        wait = max(wait, float(hdr.get("retry-after", 0)))
                    except ValueError:
                        pass
                    print(f"[retry {attempt}] {st} wait {wait:.1f}s", file=sys.stderr)
                    await asyncio.sleep(wait)
        lat = time.perf_counter() - t0
        ch = resp.choices[0]
        msg = ch.message
        reasoning = getattr(msg, "reasoning_content", None) or (msg.model_extra or {}).get("reasoning_content")
        u = resp.usage
        ptd = getattr(u, "prompt_tokens_details", None)
        ctd = getattr(u, "completion_tokens_details", None)
        row = {"text": msg.content or "", "reasoning": reasoning, "finish_reason": ch.finish_reason,
               "prompt_tokens": getattr(u, "prompt_tokens", None),
               "cached_tokens": getattr(ptd, "cached_tokens", None) if ptd else None,
               "completion_tokens": getattr(u, "completion_tokens", None),
               "reasoning_tokens": getattr(ctd, "reasoning_tokens", None) if ctd else None,
               "latency_s": round(lat, 3)}
        row["cost_usd"] = call_cost(row, self.model)
        self.cost += row["cost_usd"]
        return row


class MLXBackend:
    """Sequential greedy generation; thinking disabled in the chat template."""

    def __init__(self, model_path: str):
        from mlx_lm import load
        self.model, self.tok = load(model_path)
        self.model_path = model_path
        self.lock = asyncio.Lock()
        self.cost = 0.0
        self.gen_tokens = 0
        self.gen_secs = 0.0

    def _run(self, messages, max_tokens):
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler
        prompt = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                              enable_thinking=False)
        t0 = time.perf_counter()
        text, last = "", None
        for r in stream_generate(self.model, self.tok, prompt, max_tokens=max_tokens,
                                 sampler=make_sampler(temp=0.0)):
            text += r.text
            last = r
        lat = time.perf_counter() - t0
        n = last.generation_tokens if last else 0
        fin = getattr(last, "finish_reason", None) if last else None
        self.gen_tokens += n
        self.gen_secs += n / last.generation_tps if last and last.generation_tps else 0
        return {"text": text, "reasoning": None, "finish_reason": fin,
                "prompt_tokens": last.prompt_tokens if last else None, "cached_tokens": None,
                "completion_tokens": n, "reasoning_tokens": 0, "latency_s": round(lat, 3),
                "gen_tps": round(last.generation_tps, 2) if last else None,
                "prompt_tps": round(last.prompt_tps, 2) if last else None, "cost_usd": 0.0}

    async def chat(self, messages, max_tokens):
        async with self.lock:
            return await asyncio.to_thread(self._run, messages, max_tokens)


def add_backend_args(p) -> None:
    p.add_argument("--system", required=True, help="results file name, e.g. qwen3-8b or kimi-k3")
    p.add_argument("--backend", choices=["openai", "mlx"], default="openai")
    p.add_argument("--base-url", default=FW)
    p.add_argument("--model", default="accounts/fireworks/models/kimi-k3",
                   help="API model id, or MLX model path/repo for --backend mlx")
    p.add_argument("--api-key-env", default="FIREWORKS_API_KEY")
    p.add_argument("--env-file", default=str(DEFAULT_ENV))
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--extra-body", default=None,
                   help='JSON; for vLLM Qwen: \'{"chat_template_kwargs": {"enable_thinking": false}}\'')
    p.add_argument("--n", type=int, default=None, help="only the first n items (smoke test)")
    p.add_argument("--max-cost", type=float, default=None, help="stop after this run spends $X")


def make_backend(args):
    load_env_file(args.env_file)
    if args.backend == "mlx":
        return MLXBackend(args.model)
    return OpenAIBackend(args.base_url, args.model, args.api_key_env, args.concurrency,
                         json.loads(args.extra_body) if args.extra_body else None)


async def run_items(items, worker, out_path: Path, backend, max_cost: float | None = None):
    """Run `worker(item) -> row` for items not yet in out_path; append rows as they finish."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    have = done_ids(out_path)
    todo = [it for it in items if it["id"] not in have]
    print(f"{out_path.name}: {len(items)} items, {len(items)-len(todo)} done, {len(todo)} to run",
          file=sys.stderr)
    stop = {"v": False}

    async def guarded(it):
        if stop["v"]:
            return None
        try:
            return await worker(it)
        except Exception as e:  # noqa: BLE001
            print(f"[fail] {it['id']}: {type(e).__name__} {str(e)[:200]}", file=sys.stderr)
            return None

    t0 = time.time()
    n = 0
    with out_path.open("a") as f:
        for fut in asyncio.as_completed([asyncio.create_task(guarded(it)) for it in todo]):
            row = await fut
            if row is None:
                continue
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            if max_cost and backend.cost >= max_cost and not stop["v"]:
                print(f"[stop] cost cap ${max_cost} hit", file=sys.stderr)
                stop["v"] = True
            if n % 20 == 0:
                print(f"  {n}/{len(todo)} cost ${backend.cost:.3f} {time.time()-t0:.0f}s", file=sys.stderr)
    print(f"done {n} new rows, run cost ${backend.cost:.4f}, {time.time()-t0:.0f}s", file=sys.stderr)
    if isinstance(backend, MLXBackend) and backend.gen_secs:
        print(f"mlx decode {backend.gen_tokens/backend.gen_secs:.1f} tok/s", file=sys.stderr)

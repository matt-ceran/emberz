# Eval harness

Same prompt for every system (`eval/prompt.py`; its SHA is written into every summary as
`prompt_sha`), temperature 0, test split locked behind `ALLOW_TEST=1`, 95% bootstrap CI
(2,000 resamples, seed 0), per-call latency and tokens, cost from `eval/prices.json`
(each price cites its source page).

## Files

- `data/prepare.py`: downloads the datasets and writes `data/<task>/{train,val,test}.jsonl` and `labels.json`. See `data/README.md`.
- `eval/prompt.py`: the shared prompt (label list in the system message, so it's a fixed prefix for caching) and the parser: exact match, then normalised match (case, spaces/hyphens to `_`, quotes, `Label:` prefix, `<think>` blocks stripped), then the last line, then "exactly one label name appears in the reply"; anything else is `INVALID` and counts as wrong.
- `eval/run_eval.py`: calls any OpenAI-compatible endpoint. Writes `results/<task>/<split>/<system>.jsonl` (one row per item: id, gold, raw, pred, correct, parse method, finish_reason, latency_s, prompt/cached/completion/reasoning tokens) and `<system>.summary.json`. Resumable (ids already in the jsonl are skipped; rerun the same command to fill gaps after failures). Retries 429/5xx/connection errors with exponential backoff (honours `Retry-After`). `--max-cost` stops launching calls once the run's estimated cost passes the cap.
- `eval/summarize.py`: markdown table of all summaries for a split.
- `eval/prices.json`: USD per 1M tokens with sources.

## Setup

```bash
cd idea2
uv venv .venv --python 3.12
VIRTUAL_ENV=.venv uv pip install datasets openai numpy httpx
.venv/bin/python data/prepare.py
```

The Fireworks key is read from an env file with `--env-file` (never printed). Below,
`ENV=/Users/numankarakas/Projects/fireworks-AMD/.env` (or `export FIREWORKS_API_KEY=...` and
drop `--env-file`).

## Frontier baselines (val, stratified n=1000, seed 0)

```bash
FW=https://api.fireworks.ai/inference/v1
for t in banking77 clinc150; do
  .venv/bin/python eval/run_eval.py --task $t --split val --system kimi-k3 \
    --base-url $FW --model accounts/fireworks/models/kimi-k3 \
    --api-key-env FIREWORKS_API_KEY --env-file $ENV \
    --n 1000 --concurrency 16 --max-tokens 4096 --max-cost 4.5
  .venv/bin/python eval/run_eval.py --task $t --split val --system glm-5p3 \
    --base-url $FW --model accounts/fireworks/models/glm-5p3 \
    --api-key-env FIREWORKS_API_KEY --env-file $ENV \
    --n 1000 --concurrency 16 --max-tokens 4096 --max-cost 2
done
.venv/bin/python eval/summarize.py --split val
```

Both models think by default (`reasoning_content` in the reply, `reasoning_tokens` in usage);
default reasoning is kept, `max_tokens 4096` leaves room for it (no run hit the limit).
Smoke tests (`--n 20 --system <model>-smoke --save-reasoning`) are kept under
`results/<task>/val/*-smoke.*` and excluded from the table.

## Base Qwen3-8B on vLLM (not run yet)

```bash
.venv/bin/python eval/run_eval.py --task banking77 --split val --system qwen3-8b-base \
  --base-url http://<host>:8000/v1 --model Qwen/Qwen3-8B --api-key-env VLLM_API_KEY \
  --n 1000 --concurrency 32 --max-tokens 32 \
  --extra-body '{"chat_template_kwargs": {"enable_thinking": false}}' \
  --price-key qwen3-8b-dedicated
```

Qwen3-8B is not on Fireworks serverless, so `cost_usd` is null for it; cost per 1k calls must
come from the dedicated deployment's GPU-hour price and measured throughput at a stated load.

## Final test numbers (only once everything is frozen)

Same commands with `--split test` and `ALLOW_TEST=1` in front. Every system must use the same
`--n` (or the full test split) so they are scored on identical items.

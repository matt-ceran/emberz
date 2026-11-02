#!/usr/bin/env bash
# Base Qwen3-8B (bf16, thinking off, greedy) behind an OpenAI-compatible vLLM server.
# Usage: ./run_qwen.sh <task> [extra run_screen.py args]    e.g. ./run_qwen.sh syco --suffix=-smoke --n 5
set -euo pipefail
cd "$(dirname "$0")"
QWEN_URL="${QWEN_URL:-http://localhost:8000/v1}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
task="$1"; shift
exec .venv/bin/python run_screen.py --task "$task" --system qwen3-8b \
  --base-url "$QWEN_URL" --model qwen3-8b --api-key-env VLLM_API_KEY --concurrency 8 \
  --extra-body '{"chat_template_kwargs": {"enable_thinking": false}}' "$@"

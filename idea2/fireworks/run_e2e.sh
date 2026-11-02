#!/usr/bin/env bash
# End-to-end: build adapter -> upload -> deploy (live merge) -> call -> delete.
# The deployment is deleted on any exit (trap), so a failure can't leave GPUs running.
#   ./run_e2e.sh [model_id] [shape]
# Defaults: qwen3-8b-noise-r8, shape picked from `fw_lora.py match` (first result).
set -euo pipefail
cd "$(dirname "$0")"
export FIREWORKS_ACCOUNT_ID="${FIREWORKS_ACCOUNT_ID:-numan-vf90m1xhbxgc}"
MODEL_ID="${1:-qwen3-8b-noise-r8}"
FW="uv run -q --no-project --with requests python fw_lora.py"
MODEL="accounts/$FIREWORKS_ACCOUNT_ID/models/$MODEL_ID"

[ -f adapters/noise-r8/adapter_model.safetensors ] || \
  uv run -q --no-project --with torch --with safetensors --with 'transformers>=4.51' --with peft --with accelerate \
    python build_adapter.py --mode noise --out adapters/noise-r8

$FW upload "$MODEL_ID" adapters/noise-r8
$FW match "$MODEL"
SHAPE="${2:-$($FW match "$MODEL" | awk 'NR==1{print $1}')}"
[ -n "$SHAPE" ] && [ "$SHAPE" != "no" ] || { echo "no shape for $MODEL"; exit 1; }

T0=$(python3 -c 'import time;print(time.time())')
$FW deploy "$MODEL" --shape "$SHAPE" --min-replicas 1
DEP=$(tail -1 runs.jsonl | python3 -c 'import json,sys;print(json.load(sys.stdin)["deployment"].split("/")[-1])')
trap '$FW delete "$DEP"; $FW list' EXIT
$FW wait "$DEP" --since "$T0"
# Live merge: call the model directly (README has the multi-LoRA "#deployment" form).
$FW call "$MODEL" --no-think
$FW call "$MODEL" --no-think --prompt "Count from 1 to 10."

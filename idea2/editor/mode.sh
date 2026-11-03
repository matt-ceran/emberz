#!/bin/bash
# Switch the VM's GPU between vLLM serving and the editor's forward-pass server (runs ON THE VM; needs sudo).
# Only one copy of Qwen3-8B fits on the L4, so exactly one of the two runs (the units also Conflict).
#
#   mode.sh editor        stop vLLM, start idea2-editor (+ the CPU kernel), wait until /health answers
#   mode.sh serve         stop idea2-editor, start vLLM, wait until /v1/models answers
#   mode.sh status        which mode, GPU memory, budget
#   mode.sh new-session   archive /opt/idea2/work to /opt/idea2/work-archive/<time>, start a clean one
#                         (budget, run counter, edits and kernel variables back to zero)
set -uo pipefail
VLLM=idea2-vllm EDITOR=idea2-editor KERNEL=idea2-kernel
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1; }
wait_url() {  # url, seconds
  local t0=$SECONDS
  while (( SECONDS - t0 < $2 )); do
    curl -sf "$1" >/dev/null 2>&1 && return 0
    sleep 2
  done
  return 1
}
wait_gpu_free() {
  for _ in $(seq 1 60); do
    [ "$(gpu_used)" -lt 1000 ] && return 0
    sleep 1
  done
  echo "warning: GPU still has $(gpu_used) MiB in use" >&2
}
t0=$(date +%s.%N)
el() { awk -v a="$t0" -v b="$(date +%s.%N)" 'BEGIN { printf "%.1f", b - a }'; }

case "${1:-status}" in
  editor)
    sudo systemctl stop "$VLLM" "$EDITOR"
    wait_gpu_free
    echo "vLLM (and any old editor) stopped ($(el) s)"
    sudo systemctl start "$EDITOR"
    sudo systemctl start "$KERNEL"
    if wait_url http://127.0.0.1:8100/health 900; then
      echo "editor mode: forward-pass server up ($(el) s total); $(curl -s http://127.0.0.1:8100/health)"
    else
      echo "editor did not come up; tail /opt/idea2/editor.log:"; tail -n 30 /opt/idea2/editor.log; exit 1
    fi ;;
  serve)
    sudo systemctl stop "$EDITOR"
    wait_gpu_free
    echo "editor stopped ($(el) s)"
    sudo systemctl start "$VLLM"
    if wait_url http://127.0.0.1:8000/v1/models 900; then
      echo "serve mode: vLLM up ($(el) s total)"
    else
      echo "vLLM did not come up; tail /opt/idea2/vllm.log:"; tail -n 30 /opt/idea2/vllm.log; exit 1
    fi ;;
  status)
    echo "vllm: $(systemctl is-active $VLLM)  editor: $(systemctl is-active $EDITOR)  kernel: $(systemctl is-active $KERNEL)  gpu: $(gpu_used) MiB"
    curl -sf http://127.0.0.1:8100/health 2>/dev/null && echo
    if [ -e /opt/idea2/work/kernel/runs.json ]; then echo "kernel runs: $(cat /opt/idea2/work/kernel/runs.json)"; fi ;;
  new-session)
    # The server re-reads budget.json and the edits folder, so it keeps running; the kernel restarts clean.
    sudo systemctl stop "$KERNEL"
    ts=$(date -u +%Y%m%dT%H%M%SZ)
    sudo mkdir -p /opt/idea2/work-archive
    [ -d /opt/idea2/work ] && sudo mv /opt/idea2/work "/opt/idea2/work-archive/$ts" && echo "archived to /opt/idea2/work-archive/$ts"
    sudo mkdir -p /opt/idea2/work/edited /opt/idea2/work/kernel
    sudo chmod 0777 /opt/idea2/work /opt/idea2/work/edited /opt/idea2/work/kernel
    sudo touch /opt/idea2/work/log.jsonl && sudo chmod 0666 /opt/idea2/work/log.jsonl
    if [ "$(systemctl is-active $EDITOR)" = active ]; then
      curl -sf -X POST http://127.0.0.1:8100/reload_edits -H 'Content-Type: application/json' -d '{}' >/dev/null \
        && echo "editor: edits cleared, budget $(curl -s http://127.0.0.1:8100/budget)"
      sudo systemctl start "$KERNEL"
    fi ;;
  *) sed -n '2,11p' "$0"; exit 2 ;;
esac

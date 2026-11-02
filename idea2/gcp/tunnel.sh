#!/bin/bash
# Forward http://localhost:${IDEA2_LOCAL_PORT:-8000} on this machine to vLLM (127.0.0.1:8000 on the VM).
# Runs in the foreground; Ctrl-C closes it.
#   ./tunnel.sh                          foreground
#   IDEA2_LOCAL_PORT=8001 ./tunnel.sh    another local port (if 8000 is taken)
# Background it:
#   nohup ./tunnel.sh >/tmp/idea2-tunnel.log 2>&1 &   # close with: pkill -f 'idea2-qwen3-l4.*-L'
set -euo pipefail
cd "$(dirname "$0")"
source ./config.sh
[ "$(vm_status)" = RUNNING ] || die "$IDEA2_VM is not RUNNING; run ./launch.sh"
if lsof -nP -iTCP:"$IDEA2_LOCAL_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  die "local port $IDEA2_LOCAL_PORT is already in use (another tunnel?); set IDEA2_LOCAL_PORT"
fi
echo "tunnel: http://localhost:$IDEA2_LOCAL_PORT -> $IDEA2_VM:127.0.0.1:8000 (Ctrl-C to close)"
exec gcloud --project "$IDEA2_PROJECT" compute ssh "$IDEA2_VM" --zone "$IDEA2_ZONE" -- \
  -N -L "$IDEA2_LOCAL_PORT:127.0.0.1:8000" \
  -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3

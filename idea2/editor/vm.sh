#!/bin/bash
# Mac-side (or VPS-side) entry point for a weight-editing session on the idea2 GCP VM. Every command is one ssh call.
#
#   ./vm.sh deploy                 copy the harness + dev/select splits to /opt/idea2/editor (never test), install units
#   ./vm.sh mode editor|serve|status|new-session
#   ./vm.sh py FILE.py             run FILE.py in the persistent CPU kernel on the VM (or: ./vm.sh py - < code.py)
#   ./vm.sh kernel start|stop|restart|status
#   ./vm.sh fwd REQUEST.json       one forward pass (JSON with "type": "generate" | "read"); or inline JSON, or -
#   ./vm.sh budget | health | reload
#   ./vm.sh get REMOTE_PATH LOCAL  copy a file from the VM (e.g. /opt/idea2/work/gen_3.jsonl); put LOCAL... REMOTE_DIR
#   ./vm.sh refresh                re-read the VM's IP (after a restart/move) and refresh gcloud's ssh key
#   ./vm.sh sh 'command'           any shell command on the VM
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
CACHE="$HOME/.cache/idea2-vm-${IDEA2_VM:-idea2-qwen3-l4}"   # "zone id ip"; ./vm.sh refresh rewrites it
if [ -s "$CACHE" ] && [ -z "${IDEA2_ZONE:-}" ] && [ "${1:-}" != refresh ]; then read -r IDEA2_ZONE _ _ <"$CACHE"; fi
source "$HERE/../gcp/config.sh"
[ -n "$IDEA2_ZONE" ] || die "VM $IDEA2_VM not found"
PY=/opt/idea2/venv/bin/python
E=/opt/idea2/editor
# Plain ssh with gcloud's key (gcloud compute ssh costs ~12 s per call; this ~0.3 s with a shared connection).
# The VM's id and external IP are cached; `./vm.sh refresh` re-reads them (after a restart or move).
resolve() {
  mkdir -p "$(dirname "$CACHE")"
  echo "$IDEA2_ZONE $(gcloud --project "$IDEA2_PROJECT" compute instances describe "$IDEA2_VM" --zone "$IDEA2_ZONE" \
    --format="value(id,networkInterfaces[0].accessConfigs[0].natIP)")" >"$CACHE"
}
[ -s "$CACHE" ] || resolve
read -r _ VMID VMIP <"$CACHE"
OPTS=(-i "$HOME/.ssh/google_compute_engine" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20
      -o UserKnownHostsFile="$HOME/.ssh/google_compute_known_hosts" -o HostKeyAlias="compute.$VMID"
      -o ControlMaster=auto -o "ControlPath=$HOME/.ssh/idea2-cm-%C" -o ControlPersist=15m -o ServerAliveInterval=30)
DEST="${IDEA2_SSH_USER:-$USER}@$VMIP"
on_vm() {
  local rc=0
  ssh "${OPTS[@]}" "$DEST" "$1" || rc=$?
  [ $rc -eq 255 ] && echo "ssh failed (255). If the VM was restarted or moved: ./vm.sh refresh" >&2
  return $rc
}
put() { scp -q -r "${OPTS[@]}" "${@:1:$#-1}" "$DEST:${!#}"; }
src() { if [ "${1:--}" = - ]; then cat; elif [ -f "$1" ]; then cat "$1"; else printf '%s' "$1"; fi; }

cmd=${1:-}; shift || true
case "$cmd" in
  deploy)
    SPL="$HERE/../screen/splits"
    [ -f "$SPL/dev.jsonl" ] && [ -f "$SPL/select.jsonl" ] || die "no $SPL/{dev,select}.jsonl yet"
    stage=$(mktemp -d)
    mkdir -p "$stage/units" "$stage/splits"
    cp "$HERE"/*.py "$HERE"/*.md "$HERE"/mode.sh "$HERE"/install.sh "$stage/"
    cp "$HERE"/units/*.service "$stage/units/"
    cp "$SPL/dev.jsonl" "$SPL/select.jsonl" "$stage/splits/"     # test.jsonl is never copied
    on_vm "rm -rf /tmp/idea2-editor-stage && mkdir -p /tmp/idea2-editor-stage"
    put "$stage"/* /tmp/idea2-editor-stage/
    rm -rf "$stage"
    on_vm "sudo bash /tmp/idea2-editor-stage/install.sh && rm -rf /tmp/idea2-editor-stage" ;;
  mode)   on_vm "bash $E/mode.sh ${1:-status}" ;;
  py)     src "${1:--}" | on_vm "$PY $E/kernel.py run -" ;;
  kernel) on_vm "$PY $E/kernel.py ${1:-status}" ;;
  fwd)    src "${1:--}" | on_vm "$PY $E/fwd.py -" ;;
  budget|health|reload) on_vm "$PY $E/fwd.py $cmd" ;;
  get)    scp -q -r "${OPTS[@]}" "$DEST:$1" "${2:-.}" ;;
  put)    put "$@" ;;
  refresh)
    resolve; read -r _ VMID VMIP <"$CACHE"; echo "VM $IDEA2_VM: id $VMID, ip $VMIP"
    gcloud --project "$IDEA2_PROJECT" compute ssh "$IDEA2_VM" --zone "$IDEA2_ZONE" --quiet --command true ;;
  sh)     on_vm "$*" ;;
  *)      sed -n '2,13p' "$0"; exit 2 ;;
esac

#!/bin/bash
# Stop the Idea 2 VM: compute billing ends, the disk (venv, model, adapters, notes) is kept
# (~$0.33/day for 100 GB). ./launch.sh starts it again.
set -euo pipefail
cd "$(dirname "$0")"
source ./config.sh
st=$(vm_status || true)
[ -n "$st" ] || die "$IDEA2_VM does not exist"
[ "$st" = TERMINATED ] && { echo "$IDEA2_VM already stopped"; exit 0; }
gc compute instances stop "$IDEA2_VM" --zone "$IDEA2_ZONE"
echo "stopped $IDEA2_VM"

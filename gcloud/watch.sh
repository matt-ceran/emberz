#!/usr/bin/env bash
# Follow a G1 run from the Mac: prints new status lines until the run is DONE or its VM is gone.
# If DONE is in the bucket but the VM still exists 3 minutes later (self-delete failed), it deletes
# the VM itself. Read-only otherwise.
#   gcloud/watch.sh <run-id>
set -uo pipefail
source "$(dirname "$0")/config.sh"
RUN_ID=${1:?usage: watch.sh <run-id>}
NAME="${EMBERZ_PREFIX}${RUN_ID}"
OUT="$EMBERZ_BUCKET/runs/$RUN_ID"
seen=0
polls=0
done_at=""
while true; do
  lines=$(gc storage cat "$OUT/status.txt" 2>/dev/null || true)
  n=$(printf "%s" "$lines" | grep -c . || true)
  if [ "$n" -gt "$seen" ]; then
    printf "%s\n" "$lines" | tail -n $((n - seen))
    seen=$n
  fi
  vm=$(gc compute instances list --filter="name=$NAME" --format="value(status)")
  if gc storage ls "$OUT/DONE" >/dev/null 2>&1; then
    [ -z "$done_at" ] && done_at=$(date +%s)
    if [ -z "$vm" ]; then echo "== run $RUN_ID done, VM deleted"; break; fi
    if [ $(( $(date +%s) - done_at )) -gt 180 ]; then
      echo "== DONE but VM $NAME still $vm after 3 min: deleting it"
      gc compute instances delete "$NAME" --zone "$EMBERZ_ZONE" --quiet
      break
    fi
  elif [ -z "$vm" ] && [ "$seen" -gt 0 ]; then
    echo "== VM $NAME is gone without DONE (failed, preempted or hit max-run-duration)."
    echo "   Logs: gcloud storage cat $OUT/startup.log ; resume: RUN_ID=$RUN_ID gcloud/launch_*.sh"
    break
  elif [ "$vm" = "TERMINATED" ]; then
    # The self-delete was refused and the VM powered itself off (it still bills its disk).
    echo "== VM $NAME is stopped without DONE: the job failed and the self-delete did not work."
    echo "   Logs: gcloud storage cat $OUT/startup.log ; then gcloud/teardown.sh"
    break
  elif [ -z "$vm" ] && [ "$polls" -ge 20 ]; then
    echo "== no VM $NAME and no status after 10 min: was it created? (gcloud/check.sh)"
    break
  fi
  polls=$((polls + 1))
  sleep 30
done

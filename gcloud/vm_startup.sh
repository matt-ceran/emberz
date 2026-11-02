#!/bin/bash
# G1 VM startup script, passed as the `startup-script` metadata by launch_noise.sh / launch_voice.sh.
# Runs as root on first boot:
#   1. reads emberz-bucket and emberz-run-id from instance metadata
#   2. pulls this run's code tarball from gs://<bucket>/runs/<run-id>/code.tgz
#   3. validates every emberz-* metadata key (gcloud/job_config.py)
#   4. sources gcloud/<job>/vm_run.sh (noise_pass or voice_job)
#   5. on ANY exit (success or failure): uploads the log and deletes this VM.
# Status lines go to gs://<bucket>/runs/<run-id>/status.txt so the Mac can follow along
# (gcloud/watch.sh). Hard cost stops that do not depend on this script: --max-run-duration
# with --instance-termination-action=DELETE, set at create time.
set -uo pipefail
# GCE runs startup scripts without HOME on some images; uv's installer, uv, gcloud and
# huggingface_hub all put state under $HOME.
export HOME=${HOME:-/root}

W=/opt/emberz
mkdir -p "$W"
if [ -e "$W/.started" ]; then
  # A reboot would rerun us; a half-finished job is not worth a second paid run.
  echo "emberz: startup already ran on this VM; powering off"
  shutdown -h now
  exit 0
fi
touch "$W/.started"
exec > >(tee -a "$W/startup.log") 2>&1

md() { curl -sf -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/$1"; }
NAME=$(md name)
ZONE=$(md zone); ZONE=${ZONE##*/}
BUCKET=$(md attributes/emberz-bucket)
RUN=$(md attributes/emberz-run-id)
OUT="$BUCKET/runs/$RUN"

status() {
  echo "$(date -u +%FT%TZ) $*" >>"$W/status.txt"
  echo "STATUS: $*"
  gcloud storage cp -q "$W/status.txt" "$OUT/status.txt" 2>/dev/null || true
}

finish() {
  local rc=$?
  status "exit rc=$rc"
  gcloud storage cp -q "$W"/*.log "$OUT/" 2>/dev/null || true
  if [ "${EMBERZ_SELF_DELETE:-1}" = 1 ]; then
    status "deleting VM $NAME"
    # Needs the conditional compute.instanceAdmin.v1 grant from setup_bucket.sh. --async: do not
    # poll the zone operation (that resource is not named emberz-*, so the conditional grant may
    # not cover compute.zoneOperations.get, and a failed poll would look like a failed delete).
    # If the delete itself is refused, power off: a stopped VM bills only its disk; teardown.sh
    # or watch.sh removes it.
    gcloud compute instances delete "$NAME" --zone "$ZONE" --quiet --async || shutdown -h now
  else
    status "self-delete off: VM stays up until --max-run-duration deletes it"
  fi
}
trap finish EXIT

if [ -z "$BUCKET" ] || [ -z "$RUN" ]; then
  echo "emberz: missing emberz-bucket / emberz-run-id metadata"
  exit 2
fi

status "boot $NAME ($ZONE), $(nproc) vCPUs, $(free -g | awk '/Mem:/{print $2}') GB RAM"
mkdir -p "$W/repo"
gcloud storage cp "$OUT/code.tgz" "$W/code.tgz"
tar -xzf "$W/code.tgz" -C "$W/repo"

cfg=$(python3 "$W/repo/gcloud/job_config.py" --from-metadata) || { status "bad metadata"; exit 2; }
eval "$cfg"
status "job $EMBERZ_JOB run $EMBERZ_RUN_ID"

set -e
case "$EMBERZ_JOB" in
  noise) source "$W/repo/gcloud/noise_pass/vm_run.sh" ;;
  voice) source "$W/repo/gcloud/voice_job/vm_run.sh" ;;
esac
status "job done"
echo ok | gcloud storage cp - "$OUT/DONE"

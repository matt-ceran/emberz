#!/bin/bash
# Move the STOPPED Idea 2 VM to another zone when its zone has an L4 stockout (start fails with
# "does not have enough resources"). Keeps everything on the disk:
#   1. snapshot the boot disk   2. for each other zone in IDEA2_ZONES: disk from snapshot + same VM settings
#   3. once the new VM exists: delete the old VM (and its disk), then the snapshot
#   ./move.sh                  then ./launch.sh to wait for vLLM
# SSD quota: needs a second 100 GB for a few minutes (300 GB limit, 100 GB used by others).
set -euo pipefail
cd "$(dirname "$0")"
source ./config.sh
old_zone=$IDEA2_ZONE
[ -n "$old_zone" ] || die "$IDEA2_VM does not exist"
[ "$(vm_status)" = TERMINATED ] || die "$IDEA2_VM must be stopped (TERMINATED) first: ./stop.sh"
disk=$(gc compute instances describe "$IDEA2_VM" --zone "$old_zone" --format="value(disks[0].source.basename())")
snap="${IDEA2_PREFIX}move-$(date +%Y%m%d-%H%M%S)"
echo "snapshot $disk ($old_zone) -> $snap"
gc compute snapshots create "$snap" --source-disk "$disk" --source-disk-zone "$old_zone" --labels "$IDEA2_LABEL"

new_zone=""
for z in $IDEA2_ZONES; do
  [ "$z" = "$old_zone" ] && continue
  echo "trying $z"
  gc compute disks create "$IDEA2_VM" --zone "$z" --source-snapshot "$snap" \
    --type "$IDEA2_DISK_TYPE" --size "${IDEA2_DISK_GB}GB" --labels "$IDEA2_LABEL" >/dev/null
  if gc compute instances create "$IDEA2_VM" --zone "$z" \
      --machine-type "$IDEA2_MACHINE_TYPE" --accelerator "$IDEA2_ACCELERATOR" \
      --provisioning-model "$IDEA2_PROVISIONING" --maintenance-policy TERMINATE \
      --max-run-duration "$IDEA2_MAX_RUN" --instance-termination-action STOP \
      --disk "name=$IDEA2_VM,boot=yes,auto-delete=yes" \
      --labels "$IDEA2_LABEL" --no-service-account --no-scopes \
      --metadata-from-file startup-script=vm_setup.sh; then
    new_zone=$z; break
  fi
  echo "  no luck in $z; removing its disk"
  gc compute disks delete "$IDEA2_VM" --zone "$z" --quiet
done
if [ -z "$new_zone" ]; then
  echo "no zone had an L4; old VM untouched in $old_zone, snapshot $snap kept (delete: gcloud compute snapshots delete $snap)"
  exit 1
fi
echo "new VM up in $new_zone; deleting the old one in $old_zone"
gc compute instances delete "$IDEA2_VM" --zone "$old_zone" --delete-disks=all --quiet
gc compute snapshots delete "$snap" --quiet
echo "moved to $new_zone. Run ./launch.sh to wait for vLLM."

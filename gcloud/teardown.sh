#!/usr/bin/env bash
# Delete every emberz VM (name prefix emberz- or label emberz=g1) and leftover emberz disks,
# then list all instances. Never touches anything else in the project (e.g. gw-slice-1).
#   gcloud/teardown.sh                 VMs + leftover disks
#   gcloud/teardown.sh --bucket        also delete the bucket and everything in it (asks first)
set -uo pipefail
source "$(dirname "$0")/config.sh"

del_bucket=0
[ "${1:-}" = "--bucket" ] && del_bucket=1

vms=$(gc compute instances list --filter="name~^${EMBERZ_PREFIX} OR labels.emberz=g1" --format="value(name,zone.basename())" 2>/dev/null)
if [ -n "$vms" ]; then
  while read -r name zone; do
    echo "deleting VM $name ($zone)"
    gc compute instances delete "$name" --zone "$zone" --quiet
  done <<<"$vms"
else
  echo "no emberz VMs"
fi

disks=$(gc compute disks list --filter="name~^${EMBERZ_PREFIX}" --format="value(name,zone.basename())")
if [ -n "$disks" ]; then
  while read -r name zone; do
    echo "deleting leftover disk $name ($zone)"
    gc compute disks delete "$name" --zone "$zone" --quiet
  done <<<"$disks"
fi

if [ "$del_bucket" = 1 ]; then
  read -r -p "Delete $EMBERZ_BUCKET and ALL its contents (noise output, voice features)? Type yes: " ok
  if [ "$ok" = "yes" ]; then
    gc storage rm -r "$EMBERZ_BUCKET"
  else
    echo "bucket kept"
  fi
fi

echo
echo "== instances now"
gc compute instances list

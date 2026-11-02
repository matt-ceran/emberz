#!/bin/bash
# Delete Idea 2 VMs (name starts with idea2- AND label idea2=true) and their boot disks.
# Everything on the disk (model, adapters, notes) is lost: copy results off first.
#   ./delete.sh          lists, then asks to type yes
#   YES=1 ./delete.sh    no prompt
# Never touches any other VM in the project (e.g. gw-slice-1, emberz-*).
set -euo pipefail
cd "$(dirname "$0")"
source ./config.sh
vms=$(gc compute instances list --filter="name~^${IDEA2_PREFIX} AND labels.idea2=true" \
      --format="value(name,zone.basename())")
[ -n "$vms" ] || { echo "no idea2- VMs"; exit 0; }
echo "will DELETE (with boot disks):"; echo "$vms" | sed 's/^/  /'
if [ "${YES:-0}" != 1 ]; then
  read -r -p "Type yes: " ok; [ "$ok" = yes ] || die "aborted"
fi
echo "$vms" | while read -r name zone; do
  [[ "$name" == "$IDEA2_PREFIX"* ]] || continue
  gc compute instances delete "$name" --zone "$zone" --delete-disks=all --quiet
done

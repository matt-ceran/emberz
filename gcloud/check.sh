#!/usr/bin/env bash
# Read-only status of the G1 Google Cloud project. Run at every session end.
# Shows: running VMs (ours flagged), leftover emberz disks, quotas we depend on,
# required APIs, the service account and the bucket. Creates and changes nothing.
set -uo pipefail
source "$(dirname "$0")/config.sh"

echo "== project $EMBERZ_PROJECT ($(gcloud config get-value account 2>/dev/null))"

echo
echo "== VM instances (all; ours start with $EMBERZ_PREFIX)"
gc compute instances list --format="table(name,zone.basename(),machineType.basename(),scheduling.provisioningModel,status,labels.emberz)" 2>/dev/null
ours=$(gc compute instances list --filter="name~^${EMBERZ_PREFIX} OR labels.emberz=g1" --format="value(name)" 2>/dev/null)
if [ -n "$ours" ]; then
  echo "!! emberz VMs still exist: $ours   -> run gcloud/teardown.sh when their job is done"
else
  echo "ok: no emberz VMs"
fi

echo
echo "== leftover emberz disks (billed even when no VM uses them)"
disks=$(gc compute disks list --filter="name~^${EMBERZ_PREFIX}" --format="value(name,zone.basename(),sizeGb)")
[ -n "$disks" ] && echo "!! $disks" || echo "ok: none"

echo
echo "== quotas (limit / usage)"
gc compute regions describe "$EMBERZ_REGION" --format=json | python3 -c '
import json, sys
want = ["CPUS", "N2_CPUS", "PREEMPTIBLE_CPUS", "NVIDIA_L4_GPUS", "PREEMPTIBLE_NVIDIA_L4_GPUS", "SSD_TOTAL_GB", "IN_USE_ADDRESSES"]
q = {x["metric"]: x for x in json.load(sys.stdin)["quotas"]}
for m in want:
    if m in q:
        lim, use = q[m]["limit"], q[m]["usage"]
        print(f"  region  {m:28s} {lim:>7.0f} / {use:.0f}")
'
gc compute project-info describe --format=json | python3 -c '
import json, os, sys
q = {x["metric"]: x for x in json.load(sys.stdin)["quotas"]}
for m in ["CPUS_ALL_REGIONS", "GPUS_ALL_REGIONS"]:
    if m in q:
        lim, use = q[m]["limit"], q[m]["usage"]
        print(f"  global  {m:28s} {lim:>7.0f} / {use:.0f}   (free {lim - use:.0f})")
c = q.get("CPUS_ALL_REGIONS")
if c and c["limit"] - c["usage"] < 96:
    print("  !! CPUS_ALL_REGIONS leaves fewer than 96 vCPUs free: n2-highcpu-96 cannot start.")
    print("     Request an increase (IAM & Admin > Quotas, metric compute.googleapis.com/cpus_all_regions)")
    print("     or launch with a smaller NOISE_MACHINE_TYPE (see gcloud/README.md).")
'
echo "  (spot VMs use the normal CPUS/N2_CPUS quota when PREEMPTIBLE_CPUS is 0)"

echo
echo "== APIs"
enabled=$(gc services list --enabled --format="value(config.name)")
for api in compute.googleapis.com storage.googleapis.com iam.googleapis.com; do
  if grep -qx "$api" <<<"$enabled"; then echo "  ok  $api"; else echo "  !!  $api not enabled (setup_bucket.sh enables it)"; fi
done

echo
echo "== service account $EMBERZ_SA"
if gc iam service-accounts describe "$EMBERZ_SA" --format="value(email)" >/dev/null 2>&1; then
  echo "  ok: exists"
else
  echo "  !! missing (setup_bucket.sh creates it)"
fi

echo
echo "== bucket $EMBERZ_BUCKET"
if gc storage buckets describe "$EMBERZ_BUCKET" --format="value(name)" >/dev/null 2>&1; then
  # -s: one summary line "<bytes> <url>" (-r would print "1.2GiB" and break the MB math)
  gc storage du -s "$EMBERZ_BUCKET" 2>/dev/null | awk '{printf "  ok: exists, %.1f MB\n", $1/1e6}'
else
  echo "  !! missing (setup_bucket.sh creates it)"
fi

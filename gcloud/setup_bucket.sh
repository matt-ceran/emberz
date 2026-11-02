#!/usr/bin/env bash
# One-time G1 setup. CREATES resources, so Numan runs it by hand (agents never do).
#  1. enables iam.googleapis.com (free; needed to create a service account)
#  2. creates the bucket (us-central1, uniform access, no public access, objects auto-deleted after 30 days)
#  3. creates service account emberz-g1 for our VMs, because the project's default compute SA has
#     no roles at all (checked 2026-11-02), so VMs running as it cannot use the bucket or delete themselves
#  4. grants it: objectAdmin on this bucket only; compute.instanceAdmin.v1 only on instances and
#     disks whose names start with "emberz-" (an IAM condition), so a finished VM can delete itself
#     but cannot touch other VMs in the project (gw-slice-1).
# Storage cost: a few GB for a few days, well under $0.10.
set -euo pipefail
source "$(dirname "$0")/config.sh"

echo "project $EMBERZ_PROJECT, bucket $EMBERZ_BUCKET, service account $EMBERZ_SA"
read -r -p "Create these? Type yes: " ok
[ "$ok" = "yes" ] || { echo "aborted"; exit 1; }

gc services enable iam.googleapis.com

if ! gc storage buckets describe "$EMBERZ_BUCKET" >/dev/null 2>&1; then
  gc storage buckets create "$EMBERZ_BUCKET" --location="$EMBERZ_REGION" \
    --uniform-bucket-level-access --public-access-prevention
fi
lifecycle=$(mktemp)
echo '{"rule":[{"action":{"type":"Delete"},"condition":{"age":30}}]}' >"$lifecycle"
gc storage buckets update "$EMBERZ_BUCKET" --lifecycle-file="$lifecycle"
rm -f "$lifecycle"

if ! gc iam service-accounts describe "$EMBERZ_SA" >/dev/null 2>&1; then
  gc iam service-accounts create "$EMBERZ_SA_NAME" --display-name="Emberz G1 bulk-job VMs"
  sleep 10  # a new SA takes a few seconds before IAM bindings accept it
fi

gc storage buckets add-iam-policy-binding "$EMBERZ_BUCKET" \
  --member="serviceAccount:$EMBERZ_SA" --role=roles/storage.objectAdmin

# Conditional binding: only resources named emberz-* (the VM and its boot disk).
# https://cloud.google.com/compute/docs/access/granting-access-to-resources#using_iam_conditions
cond=$(mktemp)
cat >"$cond" <<EOF
title: emberz-vms-only
description: G1 VMs may delete themselves, nothing else
expression: resource.name.extract("/instances/{name}").startsWith("${EMBERZ_PREFIX}") || resource.name.extract("/disks/{name}").startsWith("${EMBERZ_PREFIX}")
EOF
gc projects add-iam-policy-binding "$EMBERZ_PROJECT" \
  --member="serviceAccount:$EMBERZ_SA" --role=roles/compute.instanceAdmin.v1 \
  --condition-from-file="$cond" >/dev/null
rm -f "$cond"

echo "done. Next: gcloud/check.sh"

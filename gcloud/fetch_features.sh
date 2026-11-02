#!/usr/bin/env bash
# Copy a voice run's outputs (feature shards + small results) from the bucket to
# data/voice_features/ on the Mac. The raw audio never leaves the VM, so it is not in the bucket.
#   gcloud/fetch_features.sh <run-id>            everything in voice_out (~1-2 GB of features)
#   gcloud/fetch_features.sh <run-id> --small    only heads/results/eval (a few MB), no feats/
set -euo pipefail
source "$(dirname "$0")/config.sh"
RUN_ID=${1:?usage: fetch_features.sh <run-id> [--small]}
SRC="$EMBERZ_BUCKET/runs/$RUN_ID/voice_out"
DEST="$REPO_ROOT/data/voice_features"
mkdir -p "$DEST"
gc storage ls "$EMBERZ_BUCKET/runs/$RUN_ID/DONE" >/dev/null 2>&1 || echo "warning: run has no DONE marker; fetching what is there"
if [ "${2:-}" = "--small" ]; then
  gc storage rsync -r --exclude='^feats/' "$SRC" "$DEST"
else
  gc storage rsync -r "$SRC" "$DEST"
fi
echo "fetched to $DEST: $(du -sh "$DEST" | cut -f1)"
find "$DEST" -maxdepth 2 -type d | sed "s|$REPO_ROOT/||" | sort | head -20

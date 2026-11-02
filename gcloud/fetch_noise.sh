#!/usr/bin/env bash
# Copy a finished noise run to data/noisy_turns.jsonl (+ .stats.json) and validate every row
# against contracts.models.NoisyTurn. An existing data/noisy_turns.jsonl is kept as .bak.
#   gcloud/fetch_noise.sh <run-id>
set -euo pipefail
source "$(dirname "$0")/config.sh"
RUN_ID=${1:?usage: fetch_noise.sh <run-id>}
OUT="$EMBERZ_BUCKET/runs/$RUN_ID"
DEST="$REPO_ROOT/data"
mkdir -p "$DEST"
gc storage ls "$OUT/DONE" >/dev/null 2>&1 || echo "warning: run has no DONE marker; fetching what is there"
src="$OUT/noisy_turns.jsonl"
gc storage ls "$src" >/dev/null 2>&1 || src="$OUT/noisy_turns.partial.jsonl"
[ -f "$DEST/noisy_turns.jsonl" ] && mv "$DEST/noisy_turns.jsonl" "$DEST/noisy_turns.jsonl.bak"
gc storage cp "$src" "$DEST/noisy_turns.jsonl"
gc storage cp "$OUT/noisy_turns.jsonl.stats.json" "$DEST/noisy_turns.stats.json" 2>/dev/null || true
cd "$REPO_ROOT" && .venv/bin/python - "$DEST/noisy_turns.jsonl" <<'EOF'
import sys
sys.path.insert(0, ".")
from contracts.models import NoisyTurn
rows = [NoisyTurn.model_validate_json(l) for l in open(sys.argv[1]) if l.strip()]
calls = {r.call_id for r in rows}
wers = [r.wer for r in rows if r.wer is not None]
print(f"ok: {len(rows)} noisy turns from {len(calls)} calls, mean turn WER {sum(wers) / max(1, len(wers)):.3f}")
EOF

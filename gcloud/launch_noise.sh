#!/usr/bin/env bash
# Launch the G1 noise pass on one VM (default spot n2-highcpu-96). SPENDS MONEY: Numan runs it.
#   gcloud/launch_noise.sh [input.jsonl]          default input: data/noise_input.jsonl
# Env: RUN_ID=<id> reuses a run (resume after preemption: rows already uploaded are skipped),
#      YES=1 skips the prompt, NOISE_MACHINE_TYPE / NOISE_PROVISIONING / NOISE_MAX_RUN override config.sh,
#      FORCE=1 launches even when the expected run time exceeds NOISE_MAX_RUN,
#      NOISE_EXTRA_ARGS="--noise-prob 0.7" passes flags to run_noise.py.
# Cost stops: --max-run-duration + --instance-termination-action=DELETE (Google deletes the VM at
# NOISE_MAX_RUN no matter what), and the VM deletes itself when the job ends or fails.
set -euo pipefail
source "$(dirname "$0")/config.sh"
source "$(dirname "$0")/launch_common.sh"

INPUT=${1:-$REPO_ROOT/data/noise_input.jsonl}
[ -s "$INPUT" ] || die "no input at $INPUT"
INPUT="$(cd "$(dirname "$INPUT")" && pwd)/$(basename "$INPUT")"
RUN_ID=${RUN_ID:-noise-$(date +%m%d-%H%M)}
check_run_id "$RUN_ID"
NAME="${EMBERZ_PREFIX}${RUN_ID}"
OUT="$EMBERZ_BUCKET/runs/$RUN_ID"
EXTRA=${NOISE_EXTRA_ARGS:-}
case "$EXTRA" in *@*) die "NOISE_EXTRA_ARGS cannot contain @";; esac

# Validate the input with the same contract the VM uses, before paying for anything.
# Also estimates the VM wall time from the word count (run_noise.estimate_minutes).
counts=$(cd "$REPO_ROOT" && .venv/bin/python - "$INPUT" "$NOISE_MACHINE_TYPE" <<'EOF'
import sys
sys.path[:0] = [".", "gcloud/noise_pass"]
import noise_lib as nl
import run_noise as rn
rows, errors = nl.read_input(sys.argv[1])
for e in errors[:10]:
    print("  bad input", e, file=sys.stderr)
words = sum(len(nl.normalize_text(t.text)) for r in rows for t in r.turns)
est = rn.estimate_minutes(words, sys.argv[2])
print(len(rows), sum(len(r.turns) for r in rows), len(errors), words, -1 if est is None else round(est))
EOF
) || die "input check failed"
read -r calls turns bad words est_min <<<"$counts"
[ "$calls" -gt 0 ] || die "no valid calls in $INPUT"

# Refuse a max-run-duration the job is not expected to fit in (Google would delete the VM
# mid-run; finished rounds survive, but the rest needs another paid VM). FORCE=1 overrides.
max_h=$(dur_hours "$NOISE_MAX_RUN") || die "bad NOISE_MAX_RUN=$NOISE_MAX_RUN"
max_min=$(python3 -c "import sys; print(int(float(sys.argv[1]) * 60))" "$max_h")
if [ "$est_min" -ge 0 ] && [ $((est_min * 5 / 4)) -gt "$max_min" ] && [ "${FORCE:-0}" != 1 ]; then
  die "expected ~${est_min} min on $NOISE_MACHINE_TYPE (+25% margin) but NOISE_MAX_RUN is $NOISE_MAX_RUN.
  Raise it (e.g. NOISE_MAX_RUN=$(( (est_min * 3 / 2 + 59) / 60 ))h), use a bigger machine, or FORCE=1."
fi

preflight "$NOISE_MACHINE_TYPE" 0 "$NOISE_DISK_GB" "$NOISE_PROVISIONING"

price=$(price_of "$NOISE_MACHINE_TYPE" "$NOISE_PROVISIONING")
echo "run      $RUN_ID  ->  $OUT"
echo "input    $INPUT: $calls calls, $turns turns, $words words ($bad bad lines skipped)"
echo "vm       $NAME  $NOISE_MACHINE_TYPE  $NOISE_PROVISIONING  $EMBERZ_ZONE"
echo "time     expected ~${est_min} min (rough, from Mac timings; -1 = unknown machine type)"
echo "cost     $(worst_case "$price" "$NOISE_MAX_RUN")"
confirm "Create the VM?"

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
pack_code "$tmp/code.tgz" contracts/__init__.py contracts/models.py gcloud/job_config.py \
  gcloud/vm_startup.sh gcloud/noise_pass/run_noise.py gcloud/noise_pass/noise_lib.py \
  gcloud/noise_pass/vm_run.sh gcloud/noise_pass/requirements-vm.lock.txt
gc storage cp "$tmp/code.tgz" "$OUT/code.tgz"
gc storage cp "$INPUT" "$OUT/noise_input.jsonl"

# '@' instead of ',' between items (gcloud topic escaping), so values may contain commas.
meta="emberz-job=noise@emberz-bucket=$EMBERZ_BUCKET@emberz-run-id=$RUN_ID"
meta+="@emberz-model-url=$NEMOTRON_URL@emberz-model-dir=$NEMOTRON_DIR@emberz-model-sha256=$NEMOTRON_ENCODER_SHA256"
[ -n "$EXTRA" ] && meta+="@emberz-extra-args=$EXTRA"

# Flags: https://cloud.google.com/compute/docs/instances/limit-vm-runtime (max-run-duration)
#        https://cloud.google.com/compute/docs/instances/create-use-spot (SPOT)
gc compute instances create "$NAME" \
  --zone="$EMBERZ_ZONE" --machine-type="$NOISE_MACHINE_TYPE" \
  --provisioning-model="$NOISE_PROVISIONING" --instance-termination-action=DELETE \
  --max-run-duration="$NOISE_MAX_RUN" \
  --image-family="$NOISE_IMAGE_FAMILY" --image-project="$NOISE_IMAGE_PROJECT" \
  --boot-disk-size="${NOISE_DISK_GB}GB" --boot-disk-type=pd-balanced \
  --service-account="$EMBERZ_SA" --scopes=cloud-platform \
  --labels="emberz=g1,job=noise" \
  --metadata-from-file=startup-script="$REPO_ROOT/gcloud/vm_startup.sh" \
  --metadata="^@^$meta"

echo
echo "launched. Follow:  gcloud/watch.sh $RUN_ID"
echo "then fetch:        gcloud/fetch_noise.sh $RUN_ID"
echo "session end:       gcloud/check.sh"

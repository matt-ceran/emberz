#!/usr/bin/env bash
# Launch the V1 AI-voice data job on a spot g2-standard-8 (1x L4). SPENDS MONEY: Numan runs it.
#   gcloud/launch_voice.sh
# Env: RUN_ID=<id> reuses a run (resume), YES=1 skips the prompt,
#      VOICE_STEPS=download,manifests,features (default; add head,eval,aasist to finish V1 on the VM),
#      VOICE_ARGS="--skip-itw" extra voice.pipeline flags,
#      VOICE_MACHINE_TYPE / VOICE_PROVISIONING / VOICE_MAX_RUN override config.sh.
set -euo pipefail
source "$(dirname "$0")/config.sh"
source "$(dirname "$0")/launch_common.sh"

[ -f "$REPO_ROOT/voice/pipeline.py" ] || die "voice/pipeline.py not found (written by the V1 agent)"
RUN_ID=${RUN_ID:-voice-$(date +%m%d-%H%M)}
check_run_id "$RUN_ID"
NAME="${EMBERZ_PREFIX}${RUN_ID}"
OUT="$EMBERZ_BUCKET/runs/$RUN_ID"
ARGS=${VOICE_ARGS:-}
case "$ARGS$VOICE_STEPS" in *@*) die "VOICE_ARGS/VOICE_STEPS cannot contain @";; esac

# The pipeline must at least import on the Mac before we pay for a GPU.
(cd "$REPO_ROOT" && .venv/bin/python -c "import voice.pipeline, voice.features, voice.data.asvspoof5, voice.data.in_the_wild") \
  || die "voice.pipeline does not import"

preflight "$VOICE_MACHINE_TYPE" 1 "$VOICE_DISK_GB" "$VOICE_PROVISIONING"

price=$(price_of "$VOICE_MACHINE_TYPE" "$VOICE_PROVISIONING")
echo "run      $RUN_ID  ->  $OUT"
echo "steps    $VOICE_STEPS $ARGS"
echo "vm       $NAME  $VOICE_MACHINE_TYPE (1x L4)  $VOICE_PROVISIONING  $EMBERZ_ZONE  ${VOICE_DISK_GB} GB"
echo "cost     $(worst_case "$price" "$VOICE_MAX_RUN"); expected 1.5-3 h (gcloud/README.md)"
confirm "Create the VM?"

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
voice_files=$(cd "$REPO_ROOT" && find voice -name '*.py' -o -name 'requirements-gcp.txt' | grep -v __pycache__ | sort)
# shellcheck disable=SC2086
pack_code "$tmp/code.tgz" contracts/__init__.py contracts/models.py gcloud/job_config.py \
  gcloud/vm_startup.sh gcloud/voice_job/vm_run.sh $voice_files
gc storage cp "$tmp/code.tgz" "$OUT/code.tgz"

# '@' instead of ',' between items (gcloud topic escaping), so values may contain commas.
meta="emberz-job=voice@emberz-bucket=$EMBERZ_BUCKET@emberz-run-id=$RUN_ID"
meta+="@emberz-voice-steps=$VOICE_STEPS@install-nvidia-driver=True"
[ -n "$ARGS" ] && meta+="@emberz-voice-args=$ARGS"

# G2 machine types come with their L4 attached (no --accelerator flag):
# https://cloud.google.com/compute/docs/gpus#l4-gpus ; GPU VMs need --maintenance-policy=TERMINATE.
gc compute instances create "$NAME" \
  --zone="$EMBERZ_ZONE" --machine-type="$VOICE_MACHINE_TYPE" \
  --provisioning-model="$VOICE_PROVISIONING" --instance-termination-action=DELETE \
  --max-run-duration="$VOICE_MAX_RUN" --maintenance-policy=TERMINATE \
  --image-family="$VOICE_IMAGE_FAMILY" --image-project="$VOICE_IMAGE_PROJECT" \
  --boot-disk-size="${VOICE_DISK_GB}GB" --boot-disk-type=pd-balanced \
  --service-account="$EMBERZ_SA" --scopes=cloud-platform \
  --labels="emberz=g1,job=voice" \
  --metadata-from-file=startup-script="$REPO_ROOT/gcloud/vm_startup.sh" \
  --metadata="^@^$meta"

echo
echo "launched. Follow:  gcloud/watch.sh $RUN_ID"
echo "then fetch:        gcloud/fetch_features.sh $RUN_ID"
echo "session end:       gcloud/check.sh"

# AI-voice data job body (V1), sourced by gcloud/vm_startup.sh (set -e on; W, OUT, status() exist).
# Deep Learning VM (CUDA + PyTorch). Runs the voice agent's one-command pipeline:
#   python -m voice.pipeline --data /data --out /data/voice_out --steps download,manifests,features
# (voice/pipeline.py: ASVspoof 5 parts T_aa,D_aa,E_aa + In-the-Wild -> manifests -> wav2vec2
# XLS-R 300M features on the GPU). The big audio stays on this VM's disk and dies with it; only
# /data/voice_out (feature shards + small results) goes to the bucket.

status "waiting for the GPU driver"
for _ in $(seq 1 60); do nvidia-smi >/dev/null 2>&1 && break; sleep 10; done
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || { status "no GPU visible"; exit 3; }

# Newer DLVM images have no conda and the docs do not name the Python path; find the one with torch.
echo "python candidates: $(ls -d /opt/*/ 2>/dev/null | tr '\n' ' ') $(which -a python3 python 2>/dev/null | tr '\n' ' ')"
PY=""
for c in /opt/conda/bin/python /opt/python/*/bin/python3 /opt/*/bin/python3 /usr/local/bin/python3 /usr/bin/python3; do
  if [ -x "$c" ] && "$c" -c "import torch" 2>/dev/null; then PY=$c; break; fi
done
if [ -z "$PY" ]; then
  # Fallback (untested): plain python3 + the CUDA 12.8 torch wheel.
  PY=$(command -v python3)
  "$PY" -m pip install -q --break-system-packages torch --index-url https://download.pytorch.org/whl/cu128 \
    || "$PY" -m pip install -q torch --index-url https://download.pytorch.org/whl/cu128
fi
status "python $PY: $("$PY" -c 'import sys, torch; print(sys.version.split()[0], "torch", torch.__version__, "cuda", torch.cuda.is_available())')"
"$PY" -c "import torch; assert torch.cuda.is_available()" || { status "torch sees no CUDA"; exit 3; }

status "installing packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq unzip >/dev/null
pipi() { "$PY" -m pip install -q "$@" || "$PY" -m pip install -q --break-system-packages "$@"; }
pipi -r "$W/repo/voice/requirements-gcp.txt"

mkdir -p /data/voice_out
# Resume after a spot preemption: bring back what an earlier VM of this run already uploaded,
# so voice.pipeline skips finished steps (its steps skip work already done).
if gcloud storage ls "$OUT/voice_out/" >/dev/null 2>&1; then
  status "resuming: pulling earlier voice_out"
  gcloud storage rsync -r "$OUT/voice_out" /data/voice_out
fi

( while sleep 300; do gcloud storage rsync -r -q /data/voice_out "$OUT/voice_out" || true; done ) &
SYNC_PID=$!

status "running voice.pipeline --steps $EMBERZ_VOICE_STEPS $EMBERZ_VOICE_ARGS"
cd "$W/repo"
export HF_HUB_DISABLE_TELEMETRY=1 HF_HOME=/data/hf
# EMBERZ_VOICE_ARGS is validated by job_config.py; split on purpose.
# shellcheck disable=SC2086
"$PY" -m voice.pipeline --data /data --out /data/voice_out --steps "$EMBERZ_VOICE_STEPS" \
  $EMBERZ_VOICE_ARGS >"$W/voice_pipeline.log" 2>&1 || {
    rc=$?
    # Keep finished features for a resume run before the exit trap deletes the VM.
    kill "$SYNC_PID" 2>/dev/null || true
    gcloud storage rsync -r /data/voice_out "$OUT/voice_out" || true
    status "voice.pipeline failed rc=$rc (tail of voice_pipeline.log follows)"
    tail -n 30 "$W/voice_pipeline.log"
    exit "$rc"
  }
kill "$SYNC_PID" 2>/dev/null || true

gcloud storage rsync -r /data/voice_out "$OUT/voice_out"
status "uploaded voice_out: $(du -sh /data/voice_out | cut -f1); audio left on the VM: $(du -sh --exclude=voice_out /data | cut -f1)"

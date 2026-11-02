# Noise-pass job body, sourced by gcloud/vm_startup.sh (set -e is on; W, OUT, status() exist).
# Debian 12 image: installs uv + CPython 3.12, the pinned CPU-only packages, the Nemotron model
# (checksum must match the Mac's copy), runs run_noise.py and uploads the output.

UV_VERSION=0.11.14  # same as the Mac; installer URL pattern from https://docs.astral.sh/uv/getting-started/installation/
status "installing uv $UV_VERSION + Python 3.12 + packages"
curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | env UV_INSTALL_DIR=/usr/local/bin INSTALLER_NO_MODIFY_PATH=1 sh
export UV_PYTHON_INSTALL_DIR=/opt/emberz/python
uv venv -q -p 3.12 /opt/emberz/venv
PY=/opt/emberz/venv/bin/python
# torch==2.15.0+cpu lives on the PyTorch CPU index; everything else on PyPI.
uv pip install -q -p "$PY" --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match \
  -r "$W/repo/gcloud/noise_pass/requirements-vm.lock.txt"

status "downloading Nemotron model"
mkdir -p "$W/models"
curl -fsSL "$EMBERZ_MODEL_URL" | "$PY" -c "import sys, tarfile; tarfile.open(fileobj=sys.stdin.buffer, mode='r|bz2').extractall('$W/models', filter='data')"
echo "$EMBERZ_MODEL_SHA256  $W/models/$EMBERZ_MODEL_DIR/encoder.int8.onnx" | sha256sum -c -

mkdir -p "$W/out"
gcloud storage cp "$OUT/noise_input.jsonl" "$W/noise_input.jsonl"
# Resume after a spot preemption: start from the rows an earlier VM of this run already uploaded.
gcloud storage cp "$OUT/noisy_turns.partial.jsonl" "$W/out/noisy_turns.jsonl" 2>/dev/null || true

( while sleep 60; do
    [ -s "$W/out/noisy_turns.jsonl" ] && gcloud storage cp -q "$W/out/noisy_turns.jsonl" "$OUT/noisy_turns.partial.jsonl" || true
  done ) &
SYNC_PID=$!

status "running run_noise.py ($(wc -l <"$W/noise_input.jsonl") calls)"
export HF_HUB_DISABLE_TELEMETRY=1
cd "$W/repo"
# EMBERZ_EXTRA_ARGS is validated by job_config.py (only --flags and plain values); split on purpose.
# shellcheck disable=SC2086
"$PY" gcloud/noise_pass/run_noise.py \
  --input "$W/noise_input.jsonl" --output "$W/out/noisy_turns.jsonl" \
  --model-dir "$W/models/$EMBERZ_MODEL_DIR" --threads "$EMBERZ_THREADS" \
  --render-workers "$EMBERZ_RENDER_WORKERS" --asr-workers "$EMBERZ_ASR_WORKERS" \
  $EMBERZ_EXTRA_ARGS >"$W/run_noise.log" 2>&1 || {
    rc=$?
    # Keep every finished round for the resume run before the exit trap deletes the VM.
    kill "$SYNC_PID" 2>/dev/null || true
    [ -s "$W/out/noisy_turns.jsonl" ] && gcloud storage cp -q "$W/out/noisy_turns.jsonl" "$OUT/noisy_turns.partial.jsonl"
    status "run_noise.py failed rc=$rc (tail of run_noise.log follows)"
    tail -n 30 "$W/run_noise.log"
    exit "$rc"
  }
kill "$SYNC_PID" 2>/dev/null || true

gcloud storage cp "$W/out/noisy_turns.jsonl" "$W/out/noisy_turns.jsonl.stats.json" "$OUT/"
status "uploaded $(wc -l <"$W/out/noisy_turns.jsonl") noisy turns"

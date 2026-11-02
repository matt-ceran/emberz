#!/bin/bash
# Idea 2 VM startup script (passed as `startup-script` metadata by launch.sh). Runs as root on EVERY
# boot and is idempotent: the first boot installs everything (~10-15 min), later boots only make
# sure the vLLM service is enabled and running.
#
# Layout on the VM:
#   /opt/idea2/venv            uv venv: vllm, torch (CUDA), transformers, peft, safetensors, accelerate
#   /opt/idea2/models/Qwen3-8B Qwen/Qwen3-8B, bf16 safetensors (load it with transformers from here)
#   /opt/idea2/adapters/       put PEFT LoRA adapters here, then POST /v1/load_lora_adapter
#   /opt/idea2/vllm.env        tunables read by the service (GPU util, extra args)
#   idea2-vllm.service         vLLM OpenAI server on 127.0.0.1:8000, served name qwen3-8b
# Progress lines start with "idea2-setup:" and go to the serial console (launch.sh waits on them).
set -uo pipefail
export HOME=${HOME:-/root}
W=/opt/idea2
mkdir -p "$W"/models "$W"/adapters "$W"/hf
exec > >(tee -a "$W/setup.log") 2>&1
say() { echo "idea2-setup: $(date -u +%FT%TZ) $*"; }

say "boot $(hostname), $(nproc) vCPUs"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || say "WARNING nvidia-smi failed"

if [ ! -e "$W/.installed" ]; then
  set -e
  say "installing uv"
  if ! command -v /usr/local/bin/uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh
  fi
  export UV_PYTHON_INSTALL_DIR=/opt/idea2/python UV_CACHE_DIR=/opt/idea2/uv-cache
  say "creating venv (python 3.12)"
  [ -x "$W/venv/bin/python" ] || /usr/local/bin/uv venv --python 3.12 "$W/venv"
  say "installing vllm + transformers + peft"
  VIRTUAL_ENV="$W/venv" /usr/local/bin/uv pip install --python "$W/venv/bin/python" \
    vllm transformers peft safetensors accelerate huggingface_hub openai
  "$W/venv/bin/python" -c 'import vllm, torch, transformers, peft; print("versions vllm", vllm.__version__, "torch", torch.__version__, "cuda", torch.version.cuda, "transformers", transformers.__version__, "peft", peft.__version__)'
  say "downloading Qwen/Qwen3-8B"
  HF_HOME="$W/hf" "$W/venv/bin/hf" download Qwen/Qwen3-8B \
    --local-dir "$W/models/Qwen3-8B"
  du -sh "$W/models/Qwen3-8B"
  set +e
  touch "$W/.installed"
fi

if [ ! -e "$W/vllm.env" ]; then
  cat >"$W/vllm.env" <<'EOF'
# Read by idea2-vllm.service. Edit, then: sudo systemctl restart idea2-vllm
VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
HF_HOME=/opt/idea2/hf
HF_HUB_OFFLINE=1
GPU_UTIL=0.90
# extra `vllm serve` args, word-split (e.g. --enforce-eager)
VLLM_EXTRA_ARGS=
EOF
fi

cat >/etc/systemd/system/idea2-vllm.service <<'EOF'
[Unit]
Description=Idea 2: vLLM OpenAI server for Qwen3-8B with runtime LoRA loading (127.0.0.1:8000)
After=network-online.target
Wants=network-online.target

[Service]
# venv first: vLLM and its deps spawn tools from the venv (ninja). FlashInfer's sampler would JIT-compile
# CUDA kernels at first request (needs a CUDA 13 nvcc the image lacks), so use vLLM's own sampler.
Environment=PATH=/opt/idea2/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
Environment=VLLM_USE_FLASHINFER_SAMPLER=0
EnvironmentFile=/opt/idea2/vllm.env
ExecStart=/opt/idea2/venv/bin/vllm serve /opt/idea2/models/Qwen3-8B \
  --served-model-name qwen3-8b --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization ${GPU_UTIL} \
  --enable-lora --max-lora-rank 64 --max-loras 4 $VLLM_EXTRA_ARGS
Restart=always
RestartSec=15
StandardOutput=append:/opt/idea2/vllm.log
StandardError=append:/opt/idea2/vllm.log

[Install]
WantedBy=multi-user.target
EOF

# Any ssh user (Mac: numankarakas, VPS: harbor) may write adapters, use the HF cache and read logs.
chmod 1777 "$W/adapters"
chmod -R a+rwX "$W/hf"
touch "$W/vllm.log"; chmod a+r "$W/vllm.log" "$W/setup.log"
chmod a+rw "$W/vllm.env"
cat >/etc/profile.d/idea2.sh <<'EOF'
export HF_HOME=/opt/idea2/hf
export QWEN3_8B=/opt/idea2/models/Qwen3-8B
alias idea2-venv='source /opt/idea2/venv/bin/activate'
EOF

systemctl daemon-reload
systemctl enable idea2-vllm
systemctl restart idea2-vllm
say "waiting for vLLM on 127.0.0.1:8000"
for _ in $(seq 1 120); do
  if curl -sf http://127.0.0.1:8000/v1/models >/dev/null; then
    say "READY vLLM up ($("$W/venv/bin/python" -c 'import vllm; print(vllm.__version__)'))"
    exit 0
  fi
  sleep 10
done
say "FAILED vLLM did not come up in 20 min; see /opt/idea2/vllm.log"
tail -n 40 "$W/vllm.log"
exit 1

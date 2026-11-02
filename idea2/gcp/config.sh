# Idea 2 GPU VM settings. Sourced by every idea2/gcp/*.sh script; override any value via env.
# Facts checked on 2026-11-02 (gcloud, account info@emberdial.com).

: "${IDEA2_PROJECT:=unified-booster-512122-j5}"
: "${IDEA2_REGION:=us-central1}"
# nvidia-l4 is offered in us-central1-a, -b, -c. launch.sh tries them in this order (L4 stockouts
# are common); every other script finds the zone the VM actually lives in.
: "${IDEA2_ZONES:=us-central1-a us-central1-b us-central1-c}"
# Every VM we create carries this name prefix and label; delete.sh touches only these.
IDEA2_PREFIX="idea2-"
IDEA2_LABEL="idea2=true"
: "${IDEA2_VM:=idea2-qwen3-l4}"

: "${IDEA2_MACHINE_TYPE:=g2-standard-8}"     # 8 vCPU, 32 GiB, 1x L4 (24 GB)
: "${IDEA2_ACCELERATOR:=type=nvidia-l4,count=1}"
: "${IDEA2_PROVISIONING:=STANDARD}"          # not SPOT: no preemption mid-edit
# Hard stop: GCE STOPS (not deletes) the VM after this much RUNNING time; the disk and all work
# stay, compute billing ends. The timer restarts each time the VM is started again.
: "${IDEA2_MAX_RUN:=10h}"
: "${IDEA2_DISK_GB:=100}"                    # image minimum is 100 GB; pd-balanced counts as SSD quota
: "${IDEA2_DISK_TYPE:=pd-balanced}"
# Deep Learning VM: Ubuntu 22.04, NVIDIA driver 580, CUDA 12.9 (vLLM gets its own torch in a uv venv)
: "${IDEA2_IMAGE_FAMILY:=pytorch-2-9-cu129-ubuntu-2204-nvidia-580}"
: "${IDEA2_IMAGE_PROJECT:=deeplearning-platform-release}"

# Local port on the Mac that tunnel.sh forwards to the VM's 127.0.0.1:8000
: "${IDEA2_LOCAL_PORT:=8000}"

# ---- prices, us-central1, USD/hour (Cloud Billing Catalog, 2026-11-02)
# G2 core 0.024988/vCPU, G2 RAM 0.002927/GiB, L4 0.560040/GPU  -> 0.8536/h standard
# pd-balanced 0.10/GB-month -> 100 GB = 0.0137/h, billed while the disk exists (also when stopped)
PRICE_VM_PER_H=0.8536
PRICE_DISK_PER_H=0.0137

gc() { gcloud --project "$IDEA2_PROJECT" "$@"; }
die() { echo "error: $*" >&2; exit 1; }
# Sets IDEA2_ZONE to the zone the VM lives in (empty when it does not exist); IDEA2_ZONE from env wins.
if [ -z "${IDEA2_ZONE:-}" ]; then
  IDEA2_ZONE=$(gc compute instances list --filter="name=$IDEA2_VM" --format="value(zone.basename())" 2>/dev/null | head -n 1)
fi
vm_status() {
  [ -n "$IDEA2_ZONE" ] || return 0
  gc compute instances describe "$IDEA2_VM" --zone "$IDEA2_ZONE" --format="value(status)" 2>/dev/null
}

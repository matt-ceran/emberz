# G1 Google Cloud settings. Sourced by every gcloud/*.sh script; override any value via env.
# Facts below were checked read-only on 2026-11-02 (gcloud 591.0.0, account info@emberdial.com).

: "${EMBERZ_PROJECT:=unified-booster-512122-j5}"
: "${EMBERZ_REGION:=us-central1}"
# nvidia-l4 is offered in us-central1-a, -b, -c (gcloud compute accelerator-types list);
# n2-highcpu-96 and g2-standard-8 are offered in all three.
: "${EMBERZ_ZONE:=us-central1-a}"
: "${EMBERZ_BUCKET:=gs://emberz-g1-512122j5}"
# Dedicated service account for our VMs: the project's default compute SA has no IAM roles,
# so a VM running as it can neither read the bucket nor delete itself. setup_bucket.sh makes this one.
: "${EMBERZ_SA_NAME:=emberz-g1}"
EMBERZ_SA="${EMBERZ_SA_NAME}@${EMBERZ_PROJECT}.iam.gserviceaccount.com"
# Every VM we create carries this label and the name prefix; teardown.sh deletes only these.
EMBERZ_LABEL="emberz=g1"
EMBERZ_PREFIX="emberz-"

# ---- noise pass (M1): CPU VM
: "${NOISE_MACHINE_TYPE:=n2-highcpu-96}"
: "${NOISE_PROVISIONING:=SPOT}"          # SPOT or STANDARD
: "${NOISE_MAX_RUN:=90m}"                # hard stop: the VM is deleted at this age no matter what
: "${NOISE_DISK_GB:=30}"
: "${NOISE_IMAGE_FAMILY:=debian-12}"
: "${NOISE_IMAGE_PROJECT:=debian-cloud}"

# ---- AI-voice data (V1): 1x L4 VM
: "${VOICE_MACHINE_TYPE:=g2-standard-8}"
: "${VOICE_PROVISIONING:=SPOT}"
: "${VOICE_MAX_RUN:=4h}"
: "${VOICE_DISK_GB:=150}"                # voice/pipeline.py needs >= 150 GB; SSD quota left is 200 GB
# Deep Learning VM, CUDA 12.9 + PyTorch 2.9 + driver 580 (gcloud compute images list
# --project deeplearning-platform-release; image pytorch-2-9-cu129-ubuntu-2204-nvidia-580-v20261029)
: "${VOICE_IMAGE_FAMILY:=pytorch-2-9-cu129-ubuntu-2204-nvidia-580}"
: "${VOICE_IMAGE_PROJECT:=deeplearning-platform-release}"
: "${VOICE_STEPS:=download,manifests,features}"

# ---- the same Nemotron ASR model as the live line
# https://k2-fsa.github.io/sherpa/onnx/nemo/nemotron-streaming.html
NEMOTRON_DIR="sherpa-onnx-nemotron-speech-streaming-en-0.6b-160ms-int8-2026-04-25"
NEMOTRON_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/${NEMOTRON_DIR}.tar.bz2"
# sha256 of encoder.int8.onnx in models/ on Numan's Mac; the VM refuses a different model.
NEMOTRON_ENCODER_SHA256="71111f61b18e1e65e01e369434a5c0434868d2f44892742ae54240600c681209"

# ---- prices, us-central1, USD/hour (Cloud Billing Catalog API, Compute Engine SKUs, 2026-11-02)
# N2 core 0.031611 / spot 0.018960 per vCPU; N2 RAM 0.004237 / spot 0.002542 per GiB
# G2 core 0.024988 / spot 0.014950 per vCPU; G2 RAM 0.002927 / spot 0.001751 per GiB
# L4 0.560040 / spot 0.335200 per GPU
PRICE_N2_HIGHCPU_96_STANDARD=3.4414   # 96 vCPU, 96 GiB
PRICE_N2_HIGHCPU_96_SPOT=2.0642
PRICE_N2_STANDARD_96_STANDARD=4.6614  # 96 vCPU, 384 GiB: no memory cap on workers, so faster
PRICE_N2_STANDARD_96_SPOT=2.7963
PRICE_N2_STANDARD_8_STANDARD=0.3885   # fits today's 16-vCPU global quota
PRICE_N2_STANDARD_8_SPOT=0.2330
PRICE_N2_HIGHCPU_8_STANDARD=0.2868
PRICE_N2_HIGHCPU_8_SPOT=0.1720
PRICE_G2_STANDARD_8_STANDARD=0.8536   # 8 vCPU, 32 GiB, 1x L4
PRICE_G2_STANDARD_8_SPOT=0.5108

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

gc() { gcloud --project "$EMBERZ_PROJECT" "$@"; }

# vCPUs a machine type needs (n2-highcpu-96 -> 96, g2-standard-8 -> 8).
vcpus_of() { echo "${1##*-}"; }

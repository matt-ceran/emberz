# Idea 2 GPU VM: Qwen3-8B on vLLM with LoRA hot-loading

One Google Cloud VM that serves base `Qwen/Qwen3-8B` (bf16) through vLLM's OpenAI-compatible server and
loads PEFT LoRA adapters at runtime. The same VM is where an editing session (Claude Code over ssh) reads
weights and activations with transformers/torch and writes adapters.

| | |
|---|---|
| VM | `idea2-qwen3-l4`, project `unified-booster-512122-j5`, zone found automatically (created in `us-central1-b`; `-a` had an L4 stockout) |
| Machine | g2-standard-8 (8 vCPU, 32 GiB) + 1x NVIDIA L4 24 GB, STANDARD (not spot), 100 GB pd-balanced |
| Image | Deep Learning VM `pytorch-2-9-cu129-ubuntu-2204-nvidia-580` (driver 580) |
| Safety | `--max-run-duration=10h --instance-termination-action=STOP`: after 10 h RUNNING, GCE stops the VM (disk kept, compute billing ends). The timer restarts at every start |
| Labels | name prefix `idea2-`, label `idea2=true` (delete.sh touches only these) |
| Server | systemd `idea2-vllm`, `127.0.0.1:8000` on the VM only, served model name `qwen3-8b` |

## Cost

| | USD |
|---|---|
| Running (VM + L4) | $0.8536 / h |
| Disk, 100 GB pd-balanced | $0.0137 / h (~$0.33 / day), billed also while stopped |
| One full 10 h window | $8.54 at most, then the VM stops by itself |
| Stopped | disk only |

## Commands (run on the Mac, from this folder)

```bash
cd idea2/gcp
./launch.sh          # create if missing / start if stopped / no-op if running; waits for vLLM (first boot ~10-15 min)
./tunnel.sh          # foreground: http://localhost:8000 on the Mac -> vLLM on the VM. Ctrl-C closes it
nohup ./tunnel.sh >/tmp/idea2-tunnel.log 2>&1 &     # background it; close with: pkill -f 'idea2-qwen3-l4.*-L'
IDEA2_LOCAL_PORT=8001 ./tunnel.sh                   # if 8000 is taken locally
./ssh.sh             # shell on the VM; ./ssh.sh --command 'nvidia-smi'
./stop.sh            # stop (keeps disk, ~$0.33/day)
./move.sh            # stopped VM can't start (L4 stockout in its zone): snapshot -> recreate in another zone
./delete.sh          # delete idea2- VMs + disks, asks "yes"; YES=1 ./delete.sh skips the prompt
```

Copy an adapter to the VM: `source config.sh; gc compute scp --recurse ./my-adapter "$IDEA2_VM":/opt/idea2/adapters/ --zone "$IDEA2_ZONE"`

Start to vLLM ready: ~1 min boot + ~2.5 min vLLM init (weights 12 s, torch.compile from cache, CUDA graphs 2x30 s).
First boot (install + 16 GB download): ~12 min. **L4 stockouts are common**: a start can fail with
"does not have enough resources"; retry, or `./move.sh` (moves the disk to the next zone in `IDEA2_ZONES`).

## Measured (2026-11-02)

vLLM 0.33.0, torch 2.14.0+cu130, transformers 5.20.2, peft 0.21.2. Weights 16.7 GiB; with `GPU_UTIL=0.90` the
KV cache is only 5,712 tokens (1.4 full 4k requests at once; short prompts are fine). Raise `GPU_UTIL` to
0.95 in `/opt/idea2/vllm.env` if long prompts queue. 64 concurrent one-label classification prompts
(~90 tokens in, <=16 out, thinking off): 0.97 s wall, 66 prompts/s, p50 0.80 s. LoRA load: 0.01 s.

## Using the server

```python
from openai import OpenAI
c = OpenAI(base_url="http://localhost:8000/v1", api_key="x")   # through ./tunnel.sh (or on the VM)
r = c.chat.completions.create(model="qwen3-8b", messages=[{"role": "user", "content": "hi"}],
                              max_tokens=64, temperature=0,
                              extra_body={"chat_template_kwargs": {"enable_thinking": False}})
```

LoRA hot-loading (adapters in PEFT format, rank <= 64, any of q/k/v/o/gate/up/down_proj; the path is a
path ON THE VM; up to 4 adapters resident on the GPU at once, more can be registered):

```bash
curl -s localhost:8000/v1/load_lora_adapter   -H 'Content-Type: application/json' \
  -d '{"lora_name": "edit-v1", "lora_path": "/opt/idea2/adapters/edit-v1"}'
# then use "model": "edit-v1" in chat/completions
curl -s localhost:8000/v1/unload_lora_adapter -H 'Content-Type: application/json' -d '{"lora_name": "edit-v1"}'
```

To reload a changed adapter under the same name: unload, then load again.

## On the VM

| Path | What |
|---|---|
| `/opt/idea2/venv` | uv venv (Python 3.12): vllm, torch (CUDA), transformers, peft, safetensors, accelerate, openai. `source /opt/idea2/venv/bin/activate` (alias `idea2-venv`) |
| `/opt/idea2/models/Qwen3-8B` | the base model, bf16 safetensors (`$QWEN3_8B`); load with `AutoModelForCausalLM.from_pretrained(...)` |
| `/opt/idea2/adapters/` | writable by every ssh user; put adapters here |
| `/opt/idea2/vllm.env` | service settings (`GPU_UTIL`, `VLLM_EXTRA_ARGS`); then `sudo systemctl restart idea2-vllm` |
| `/opt/idea2/vllm.log`, `setup.log` | logs |
| `smoke_test.py` | `./ssh.sh --command '/opt/idea2/venv/bin/python /opt/idea2/smoke_test.py'` re-runs the checks |

GPU memory: vLLM takes ~90% of the L4. To load the model with transformers on the GPU at the same time, stop
vLLM first (`sudo systemctl stop idea2-vllm`, start it again after), or load in transformers on CPU
(32 GiB RAM is tight for 16 GB bf16 weights plus activations; fine for reading weights).

`vm_setup.sh` is the startup script: it runs on every boot, installs once (marker `/opt/idea2/.installed`),
then (re)starts the service and prints `idea2-setup: READY` to the serial console.

# Serving our own Qwen3-8B LoRA on Fireworks

Status 2026-11-02: **blocked at the account level, not by the method.** We built and checked
the adapter locally, but Fireworks refused both the upload and the deployment for account
`numan-vf90m1xhbxgc`. We spent $0 and no deployment or model exists.

| Step | Result |
|---|---|
| firectl install + auth | OK (v1.8.22, read-only commands with `--api-key`) |
| Qwen3-8B on Fireworks | `accounts/fireworks/models/qwen3-8b`, `Supports Lora: true`, default precision FP8 |
| Adapter built locally (r=8, down_proj, 36 layers, 9.4 MB) | OK, key names and shapes match PEFT's own state dict |
| Upload (`POST /models`) | **400** `model creation, uploads and preparation are temporarily disabled, please contact Fireworks AI to get access` |
| Deploy (`POST /deployments?validateOnly=true`) | **400** `payment method is required` (for every shape, shapeless too) |
| Multi-LoRA (addons) | No validated addon shape for qwen3-8b (match with `enableAddons` returns `[]`) |

**To unblock:** add a payment method and billing profile at https://fireworks.ai/billing. The
account has a $32.04 credit balance from a credit code but no card, and docs say on-demand and
training GPU quotas need a payment method. If uploads are still disabled afterwards, ask
Fireworks (hackathon contact, or inquiries@fireworks.ai) to enable model uploads for the
account. Then run `./run_e2e.sh`.

## Files

- `build_adapter.py` builds the adapter from the shapes alone (no model download) and checks it
  against PEFT (meta-device Qwen3-8B).
- `fw_lora.py` is a REST client: upload, match, deploy, wait, call, delete, list. Each step
  appends timings and errors to `runs.jsonl`.
- `run_e2e.sh` runs the whole thing and deletes the deployment on any exit (trap).
- `adapters/` holds the built adapter (gitignored; rebuild in about 20 s).

## Account facts (read 2026-11-02)

- Account id `numan-vf90m1xhbxgc` (numan@emberdial.com), created 2026-11-02,
  `Account Verification: NOT_VERIFIED`, balance USD 32.04, monthly spend so far $10.44 (before
  this test).
- GPU quota: 16 each of H100/H200/B200/B300 in GLOBAL. `serverless-lora-adapter-count` is 0:
  there is no serverless LoRA, as the docs say.
- `qwen3-8b` is not on serverless for this account (chat call returns `NOT_FOUND`), so a base
  comparison needs its own deployment.
- The `qwen3-8b` model card shows `Deprecation Date: 2026-05-14`, which is already past, but the
  model is still READY with shapes. Watch for this.

## Shapes for qwen3-8b and prices

`firectl deployment-shape-version list --base-model accounts/fireworks/models/qwen3-8b`:

| Shape | GPU | Precision | Preset | $/hour |
|---|---|---|---|---|
| `qwen3-8b-minimal` (gn4eheji) | 1x H200 141GB | FP8 | MINIMAL | **$8.00** |
| `qwen3-8b-b300-fp8` (vc39tn5w) | 1x B300 288GB | FP8_MM | MINIMAL | $15.00 |
| `rft-qwen3-8b` (xkk2embc) | 1x B200 180GB | FP8_MM | FULL_PRECISION | $13.00 |

Prices are from https://fireworks.ai/pricing (on-demand, billed per GPU-second): H100 $8.00/h,
H200 $8.00/h, B200 $13.00/h, B300 $15.00/h, GB300 $20.00/h.

None of these shapes are BF16, so **multi-LoRA (`--enable-addons`) has no validated shape for
qwen3-8b.** The REST match with `enableAddons: true` returns nothing. That leaves **live merge**:
one adapter per deployment, merged in BF16 and quantized to FP8 when the deployment starts.
Expected cost is $8/h on `qwen3-8b-minimal`, so each adapter we test means a new deployment and
a new cold start. For many adapters, the options are:
1. Ask Fireworks to validate a BF16 multi-LoRA shape for qwen3-8b.
2. Use a shapeless BF16 deployment with `--enable-addons` (`acceptShapelessRisk=true`). The docs
   warn this is the most common cause of failed deployments.
3. Use the hot-load / RL rollout path, which swaps a PEFT adapter into a running deployment
   almost instantly. It is a gated feature that needs enablement.

## Commands

### Human, with firectl

Install (no sudo; `brew install` refuses because the `fw-ai/firectl` tap is untrusted unless you
run `brew trust`):

```bash
curl -sSf https://storage.googleapis.com/fireworks-public/firectl/stable/darwin-arm64.gz -o firectl.gz
gzip -d firectl.gz && chmod +x firectl && mv firectl ~/.local/bin/
firectl set-api-key "$FIREWORKS_API_KEY"     # or pass --api-key on every call
```

```bash
A=numan-vf90m1xhbxgc
# upload (LoRA addon on top of the Fireworks base)
firectl model create qwen3-8b-noise-r8 adapters/noise-r8 --base-model accounts/fireworks/models/qwen3-8b
firectl model get accounts/$A/models/qwen3-8b-noise-r8          # wait for State: READY
# live merge: match shapes against the LoRA model, then deploy
firectl deployment-shape-version match --model accounts/$A/models/qwen3-8b-noise-r8
firectl deployment create accounts/$A/models/qwen3-8b-noise-r8 --deployment-shape default \
  --min-replica-count 0 --max-replica-count 1 --scale-to-zero-window 5m --wait
# call (live merge: the model name alone routes to it)
curl https://api.fireworks.ai/inference/v1/chat/completions -H "Authorization: Bearer $FIREWORKS_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"accounts/'$A'/models/qwen3-8b-noise-r8","messages":[{"role":"user","content":"Hi"}],"max_tokens":40}'
# delete, then confirm nothing is left
firectl deployment delete <DEPLOYMENT_ID>
firectl deployment list
```

Multi-LoRA, only if a BF16 addon shape ever exists:

```bash
firectl deployment create accounts/fireworks/models/qwen3-8b --deployment-shape <bf16-shape> --enable-addons
firectl load-lora qwen3-8b-noise-r8 --deployment <DEPLOYMENT_ID>
# model field: "accounts/$A/models/qwen3-8b-noise-r8#accounts/$A/deployments/<DEPLOYMENT_ID>"
```

### Agent (Claude Code / Codex), with the REST script

```bash
export FIREWORKS_ACCOUNT_ID=numan-vf90m1xhbxgc   # key is read from ../../../fireworks-AMD/.env
./run_e2e.sh                                      # or the individual steps:
FW="uv run -q --no-project --with requests python fw_lora.py"
$FW upload qwen3-8b-noise-r8 adapters/noise-r8
$FW match accounts/$FIREWORKS_ACCOUNT_ID/models/qwen3-8b-noise-r8
$FW deploy accounts/$FIREWORKS_ACCOUNT_ID/models/qwen3-8b-noise-r8 --shape <shape-version> --min-replicas 1
$FW wait <DEPLOYMENT_ID>
$FW call accounts/$FIREWORKS_ACCOUNT_ID/models/qwen3-8b-noise-r8 --no-think
$FW delete <DEPLOYMENT_ID>; $FW list
```

## The test adapter

`build_adapter.py --mode noise` makes a rank-8 LoRA (alpha 16) on `mlp.down_proj` in all 36
layers. Keys are `base_model.model.model.layers.{i}.mlp.down_proj.lora_{A,B}.weight`, with A
shaped [8, 12288] and B shaped [4096, 8], in bf16. A and B are seeded random, scaled so the
merged delta has a per-entry std of about 0.05, which is larger than a normal down_proj weight.
The served model should therefore produce obvious gibberish. That proves the adapter is really
applied: a zero adapter would look identical to the base, and there is no serverless base to
compare against. `--mode zeros` (B = 0) is available as a pure pipeline test. Shapes come from
https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json: hidden 4096, intermediate 12288,
36 layers, 32 heads, 8 KV heads, head_dim 128.

## Gotchas

1. **firectl blocks agents.** Every mutating command (`set-api-key`, `model create`, even with
   `--dry-run`) fails with `BLOCKED: mutating command "firectl ..." cannot run inside an AI agent
   ... Run this command manually in your terminal`. Read-only commands (`list`, `get`, `match`,
   `quota list`, `account get`) work. Agents use the REST API (`fw_lora.py`); humans can use
   firectl.
2. `firectl whoami` needs an OIDC `firectl signin`. With only an API key, use `firectl account
   list` or `GET /v1/accounts` to find the account id.
3. Uploads go through 4 REST steps: create model (`kind: HF_PEFT_ADDON`, `peftDetails:
   {baseModel, r, targetModules}`), `:getUploadEndpoint`, PUT each file to its signed URL with
   `x-goog-content-length-range`, then poll `:validateUpload` until it returns 200. Without the
   validate step the model stays UPLOADING forever.
4. qwen3-8b has FP8-only shapes, so `--enable-addons` is impossible on a validated shape. Live
   merge is the path. `deployment-shape-version match` ignores addons; use the REST match with
   `enableAddons: true` to check addon shapes.
5. Multi-LoRA can drop an unvalidated shape silently and fall back to a default image that
   rejects addons, with no error at create time (see the docs' troubleshooting section).
6. With `minReplicaCount: 0`, requests get **503** while the deployment scales up from zero, so
   retry (`fw_lora.py call` retries 503/429 for 15 min). The default scale-to-zero window is 1 h
   (minimum 5 min). Deployments with min 0 auto-delete after 7 days without traffic.
7. `deploymentShape: "default"` lets the server pick a validated shape. Never create a deployment
   without a shape.
8. For LoRA generation defaults use `fireworks.json` in the adapter folder; `generation_config.json`
   is ignored for adapters.

## Measured timings

Not measured yet, because the account gates stopped us before upload. Pending: upload time,
deploy-to-ready, and first-call latency. `run_e2e.sh` logs all three to `runs.jsonl`.
Budget estimate: one live-merge test on `qwen3-8b-minimal` at $8/h for about 20-30 min comes to
$3-4.

# Editor harness: a Claude Code session edits Qwen3-8B's weights (no training) to reduce sycophancy

Modelled on the harness of **Weight-Space Programming** (Takehi, Teoh, Zhang, Khattab, MIT CSAIL, 2026;
https://rikiyat.github.io/blog/weight-space-programming/). Their task prompt, rules and sandbox prelude
(https://github.com/RikiyaT/lm_understand-logs; logs CC BY 4.0, scripts MIT) are adapted in `TASK.md`,
`kernel_prelude.py` and `fwd_server.py`.

The editor is a Claude Code session on the Mac (or the VPS). Everything model-related runs on the GCP VM
`idea2-qwen3-l4` (1x L4 24 GB, 32 GiB RAM; see `../gcp/README.md`). The session only ever calls `./vm.sh`.

```
Mac: Claude Code session ── ./vm.sh (plain ssh, ~0.2 s/call) ──> VM
                                                  ├─ idea2-kernel  : persistent Jupyter kernel, CPU only, weights via safetensors mmap
                                                  ├─ idea2-editor  : forward-pass server, Qwen3-8B bf16 on the GPU (127.0.0.1:8100)
                                                  └─ idea2-vllm    : vLLM (serve mode only; Conflicts= with idea2-editor)
```

## Files

| File | Where it runs | What |
|---|---|---|
| `TASK.md` | | the task message for the editor session (adapted from the paper's `exp8/runs/p50c/message.txt`) |
| `vm.sh` | Mac / VPS | the session's only entry point: `deploy`, `mode`, `py`, `fwd`, `budget`, `get`, `put`, `sh`, `refresh` |
| `mode.sh` | VM | `editor` / `serve` / `status` / `new-session` |
| `fwd_server.py` | VM, `idea2-editor` | `POST /generate`, `POST /read`, `POST /reload_edits`, `GET /budget`, `GET /health` |
| `fwd.py` | VM | tiny CLI that POSTs a JSON request to the server (what `vm.sh fwd` calls) |
| `kernel.py`, `kernel_prelude.py` | VM, `idea2-kernel` | the persistent CPU kernel and its helpers (`NAMES`, `load`, `load_original`, `save_edit`, `drop_edit`, `edited`, `tok`, `render`, `ITEMS`, `gen`, `fwd`, `fwd_arr`) |
| `grader.py` | VM | prompts + grader, copied from `../screen` (`test_grader.py` checks it reproduces all 5,439 screen grades) |
| `package_lora.py` | VM (CPU) | edited tensors -> PEFT LoRA (truncated SVD, rank <= 64, q/k/v/o/gate/up/down_proj only) |
| `verify_lora.py` | VM, serve mode | hot-loads the adapter in vLLM and measures agreement with the edited model on select items |
| `install.sh`, `units/*.service` | VM | installed by `vm.sh deploy` |
| `smoke.sh` | Mac | the end-to-end smoke test below |

## Session, end to end

```bash
cd idea2/editor
./vm.sh deploy                 # once, or after changing the harness: code + dev/select splits (never test) -> /opt/idea2/editor
./vm.sh mode editor            # stop vLLM, load Qwen3-8B in the forward-pass server, start the kernel
./vm.sh mode new-session       # fresh /opt/idea2/work: budget 0/50, no edits, kernel restarted (old work archived)
# start the editor: a Claude Code session in idea2/editor whose first message is TASK.md
```

Inside the session (what TASK.md tells the editor):

```bash
./vm.sh py analysis.py                     # code runs in the persistent CPU kernel; variables survive between runs
./vm.sh py - <<'EOF'
# hypothesis: ...
W = load("model.layers.20.mlp.down_proj.weight")
save_edit("model.layers.20.mlp.down_proj.weight", W - 1.5 * torch.outer(u, u @ W))
EOF
./vm.sh fwd '{"type": "generate", "model": "both", "split": "dev", "n": 128}'                 # 1 pass
./vm.sh fwd '{"type": "read", "model": "original", "split": "dev", "conditions": ["neutral", "wrong"],
              "positions": "last", "residual": [12, 18, 24], "mlp_act": [20], "logprobs": true}'   # 1 pass
./vm.sh budget
```

After the session (experimenter):

```bash
./vm.sh sh 'cd /opt/idea2/editor && /opt/idea2/venv/bin/python package_lora.py --out /opt/idea2/adapters/edit-v1'
# needs a "both" pass on select with the final edit: usually the session's last pass; if the budget is used up,
# raise EDITOR_BUDGET in /opt/idea2/editor.env and ./vm.sh mode editor (restarts the server, keeps work and edits)
./vm.sh fwd '{"type": "generate", "model": "both", "split": "select"}'
./vm.sh sh 'cp /opt/idea2/work/gen_<k>.jsonl /opt/idea2/adapters/edit-v1/'
./vm.sh mode serve
./vm.sh sh 'cd /opt/idea2/editor && /opt/idea2/venv/bin/python verify_lora.py --adapter /opt/idea2/adapters/edit-v1 --gen /opt/idea2/adapters/edit-v1/gen_<k>.jsonl --n 100'
# test split: score base vs adapter on vLLM from the Mac (test.jsonl never goes to the VM's editor folder)
```

The audit trail is `/opt/idea2/work/log.jsonl` (every kernel run with its code and output, every request and
response summary, every edit manifest, with timestamps): edit time = last minus first timestamp. Copy the whole
`/opt/idea2/work` back with `./vm.sh get /opt/idea2/work ./runs/<name>` together with the Claude Code transcript.

## The forward-pass server

- **Weights.** Qwen3-8B bf16 on the GPU, sdpa attention, `requires_grad_(False)`, `torch.inference_mode()` around
  every forward. Loaded by copying the safetensors straight into a module built on the meta device (no random
  init; `EDITOR_HF_LOAD=1` uses `from_pretrained` instead). On this VM boot both take ~6.5 min, set by the slow host
  link (see Notes).
- **Edits.** `/opt/idea2/work/edited/<name>.npy`, full float32 tensors with the checkpoint's names and shapes. Before
  an `edited`/`both` pass the folder is re-scanned and reloaded if it changed (or `./vm.sh reload`). Checked on the
  CPU against the checkpoint: unknown names, wrong shapes, non-finite values, `model.embed_tokens.weight`,
  `lm_head.weight`, `model.norm.weight` and "original x a constant" are refused (nothing is applied then). Each
  reload writes a manifest (tensor, changed entries after bf16 rounding, relative change, sha256 of the bf16 values,
  overall hash) to `edits_manifest.json` and the log.
- **original / edited / both.** For every edited tensor the version not on the GPU is kept in a stash (on the GPU
  while it fits in `EDITOR_GPU_STASH_GB`, 1.5 GB = 15 down_proj-sized tensors; otherwise on the CPU), and switching
  swaps tensors in place. "both" runs the same batches for both models (same order, same padding), so a no-op edit
  gives bit-identical outputs.
- **Generate.** Batched greedy `model.generate`, left padding, batches grouped by max_tokens and sorted by length,
  sized to the free KV-cache memory. Graded with `grader.py`. Summary per model and dataset: accuracy per
  condition, repeats-the-wrong-suggestion rate, mean tokens, cut share, caving on its own known items; for "both":
  paired differences with standard errors, up/down counts, caving on the original's known items with rescued /
  newly caving counts, identical-text count, example flips. Full rows in `gen_<k>.jsonl`.
- **Read.** Hooks on the embedding (residual 0), decoder layer outputs (residual l = after layer l, before the final
  norm), `down_proj` inputs (`mlp_act`), `o_proj` inputs (`o_proj_in`), eager attention for `attn`; next-token
  log-probs and top-k from the final hidden state through `lm_head` in chunks. One `[rows, d]` .npy per kind and
  layer plus `index.npy` (text, position), `meta.json` with token ids.
- **Budget and ids.** `EDITOR_BUDGET` (50) passes; failed requests are free. Only ids of `EDITOR_ALLOWED_SPLITS`
  (dev, select) are accepted; test is not on the VM's editor folder at all. Settings: `/opt/idea2/editor.env`.

## Measured (2026-11-02, this VM)

All on this VM boot, whose GPU host link is degraded (see Notes): load and mode-switch times are ~10x what a
healthy L4 host gives; pass times are partly affected.

| Step | Time |
|---|---|
| ssh call through `vm.sh` (shared connection) | 0.15 s |
| kernel run: overhead / `load()` one down_proj / `save_edit()` one down_proj | ~0.5 s / 0.05 s / 1.6 s |
| `mode editor` (vLLM stop 1 s, model load 392-408 s) | 409-413 s |
| `mode serve` (editor stop 1 s, vLLM weights ~390 s + engine init ~120 s) | 567 s |
| edit reload before a pass (1 tensor, CPU checks + one GPU copy) | ~5 s |
| paired "both" pass, 256 dev items x 3 conditions = 768 generations per model (7 batches at 64 tokens, then 26 long answers re-run) | 556 s |
| paired "both" pass, all 232 select items x 3 conditions | 582 s (before the batch-size fix) |
| read pass: 128 prompts, 13.6k tokens, residual at 3 layers + mlp_act + o_proj_in + logprobs + top-5 | 9.4 s |
| read pass with attention patterns (4 short texts, 2 layers) | 1.7 s |
| `package_lora.py`, one down_proj edit | 3.4 s |
| LoRA hot-load in vLLM / 600 vLLM generations for the agreement check | 0.04 s / 39 s |

Smoke test (`smoke.sh`, then a re-run with the fixes):
- **No-op edit** (layer 20 down_proj saved unchanged), "both" on 256 dev items x 3: 768/768 identical texts, 0 flips.
- **Random rank-1 edit**, dW = 0.3 ||W|| u v^T on layer 20 down_proj (97% of entries change): real but
  insignificant effects, i.e. the noise level an editor's edits must beat. On select: TriviaQA neutral accuracy
  -1.6 pp (se 1.2), 2 items lost. On dev: caving +1.2 pp (se 1.4; 5 rescued, 8 newly caving), 174 of 768 texts differ.
- **Static vs dynamic KV cache** (same edit, same items): static was slower (687 s vs 556 s) and changed 85 of 1,536
  texts (9 grades), so the server uses the dynamic cache, fixed for a session.
- **LoRA**: rank needed 1 (stored at rank 4, Fireworks' minimum), relative reconstruction error 2.4e-3 (bf16
  storage), 6.2e-3 against the bf16 edit the server ran. Hot-loaded in vLLM, on 100 select items x 3 = 300
  generations:

| comparison | same grade | same text |
|---|---|---|
| vLLM + adapter vs transformers edited (the check) | 99.7% | 91.3% |
| vLLM base vs transformers original (engine noise floor) | 99.7% | 95.3% |
| transformers edited vs original (the edit's own effect) | 97.7% | 82.3% |
| vLLM + adapter vs vLLM base (the same effect in vLLM) | 97.0% | 82.0% |

  So the adapter reproduces the edit up to engine noise: its accuracy per condition matches the edited model within one item
  (neutral 97 vs 98, wrong 77 vs 77, right 100 vs 100).

## Notes and limits

- **This VM boot has a degraded GPU host link.** Host-to-GPU copies run at ~19 MB/s (pinned or not; GPU compute
  is normal: 65 TFLOPS bf16 matmul, 180 GB/s device copies). vLLM loaded the weights in 8 s on the first boot and in
  389 s on this one, and every model load here takes ~6.5 min for the same reason. A stop/start of the VM (new
  host) should bring both back to well under a minute; not done here (needs Numan's go-ahead). The harness avoids
  PCIe in the edit loop: edits are checked on the CPU and the swap stash stays on the GPU.
- **Speed of a paired pass** is set by HF `generate`'s per-step cost (~0.1-0.2 s per decode step for a batch of
  ~100) and by the few TruthfulQA answers that write a chain of thought up to 384 tokens. Generation is two-stage
  (64 tokens for everyone, then the prompts that hit 64 re-run at full length, the same set for both models).
  Passing `"max_tokens"` lower, or fewer TruthfulQA items, makes passes faster.
- **bf16.** The server runs bf16; `save_edit` and the manifest report how many entries actually change after
  rounding. LoRA packaging uses the intended float32 dW; the report gives the error against both it and the bf16
  edit the server ran.
- **Engines differ.** vLLM and transformers bf16 greedy are not bit-identical: on 300 select generations the base
  model agreed 99.7% on grade and 95.3% on text across engines. The editor's paired numbers are within one engine;
  final test-set scoring should be paired within one engine too (vLLM base vs vLLM + adapter).
- **What enforces the rules.** The budget, the id allowlist (test is never copied to the VM's editor folder) and the
  refused tensors are enforced by the server and `save_edit`. "No gradients / no running the model in the kernel" is
  enforced only by the kernel having no GPU (`CUDA_VISIBLE_DEVICES=`) and by the audit log: the editor's ssh user
  has sudo on the VM, so these are rules plus a full log, as in the paper (whose sandbox was stronger: bubblewrap,
  no network).
- **Not built here:** the regression suite TASK.md mentions (MMLU / GSM8K / IFEval / perplexity, run after the
  session) and the test-set scorer. The `unknown` split is kept off the editor (counter-metric for the final report).
- `vm.sh` uses plain ssh with gcloud's key and a shared connection (`~/.ssh/idea2-cm-*`); after a VM restart or move,
  `./vm.sh refresh`.

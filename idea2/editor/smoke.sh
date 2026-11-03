#!/bin/bash
# End-to-end smoke test of the editor harness (run on the Mac from idea2/editor). Spends 3 passes of a throwaway
# session and archives it at the end, so the next editing session starts clean. Leaves the VM in SERVE mode.
#   1. editor mode; no-op edit (layer 20 down_proj copied unchanged) -> "both" pass on 256 dev items x 3
#      conditions: must give 0 flips and identical texts
#   2. random rank-1 edit on layer 20 down_proj (||dW|| = 0.3 ||W||) -> "both" pass on all 232 select items
#   3. a read pass (residual, mlp_act, o_proj_in, attention, logprobs, top-k)
#   4. package the rank-1 edit as a LoRA, serve mode, hot-load it in vLLM, agreement on 100 select items
set -euo pipefail
cd "$(dirname "$0")"
T=$(mktemp -d)
step() { echo; echo "=== $* ($(date -u +%H:%M:%S))"; }

step "mode editor"; time ./vm.sh mode editor
./vm.sh mode new-session

step "no-op edit"
./vm.sh py - <<'EOF'
# hypothesis: a no-op edit (the original tensor saved as an edit) changes nothing
N = "model.layers.20.mlp.down_proj.weight"
print(save_edit(N, load_original(N)))
EOF
step "pass 1: both, no-op edit, 256 dev items x 3 conditions"
time ./vm.sh fwd '{"type": "generate", "model": "both", "split": "dev", "n": 256}' | tee "$T/p1.json" | head -60

step "random rank-1 edit"
./vm.sh py - <<'EOF'
# hypothesis: a random rank-1 edit of 30% of ||W|| on one down_proj changes some answers
N = "model.layers.20.mlp.down_proj.weight"
g = torch.Generator().manual_seed(0)
W = load_original(N)
u = torch.randn(W.shape[0], generator=g); u /= u.norm()
v = torch.randn(W.shape[1], generator=g); v /= v.norm()
print(save_edit(N, W + 0.3 * W.norm() * torch.outer(u, v)))
EOF
step "pass 2: both, rank-1 edit, all 232 select items x 3 conditions"
time ./vm.sh fwd '{"type": "generate", "model": "both", "split": "select"}' | tee "$T/p2.json" | head -60

step "pass 3: read"
time ./vm.sh fwd '{"type": "read", "model": "edited", "split": "select", "n": 64, "conditions": ["neutral", "wrong"],
  "positions": "last", "residual": [0, 18, 36], "mlp_act": [20], "o_proj_in": [20], "logprobs": true, "topk": 5}'
./vm.sh py - <<'EOF'
# hypothesis: the read pass wrote what it says
import glob
m = fwd(3); print(m["n_texts"], m["rows"], {k: v["shape"] for k, v in m["files"].items()})
r = fwd_arr(3, "residual_L36"); print(r.shape, float(np.abs(r).mean()))
lp = fwd_arr(3, "token_logprobs"); print(lp.shape, float(np.nanmean(lp)))
EOF

step "package LoRA"
time ./vm.sh sh "cd /opt/idea2/editor && /opt/idea2/venv/bin/python package_lora.py --out /opt/idea2/adapters/smoke-rank1"
./vm.sh sh "cp -r /opt/idea2/work/gen_2.jsonl /opt/idea2/adapters/smoke-rank1/"

step "mode serve"; time ./vm.sh mode serve
step "verify LoRA in vLLM"
time ./vm.sh sh "cd /opt/idea2/editor && /opt/idea2/venv/bin/python verify_lora.py --adapter /opt/idea2/adapters/smoke-rank1 --gen /opt/idea2/adapters/smoke-rank1/gen_2.jsonl --n 100"
./vm.sh sh 'bash /opt/idea2/editor/mode.sh new-session; bash /opt/idea2/editor/mode.sh status'
echo "summaries kept in $T"

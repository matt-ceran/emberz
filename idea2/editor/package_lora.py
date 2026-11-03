"""Turn the edited tensors into a PEFT LoRA adapter for Qwen3-8B (runs ON THE VM, CPU only).

  python package_lora.py --out /opt/idea2/adapters/edit-v1 [--edits /opt/idea2/work/edited] [--rank auto|4..64]
                         [--dtype bfloat16|float32] [--tol 1e-3]

For each edited tensor W' (must be q/k/v/o/gate/up/down_proj of a decoder layer: the target modules Fireworks
accepts), dW = W' - W is factored by truncated SVD, dW ~ B @ A with B = U sqrt(S), A = sqrt(S) V^T. One rank r is
used for every module (lora_alpha = r, so the LoRA scale is 1; modules that need fewer are zero-padded):
--rank auto picks the smallest r in 4..64 that reconstructs every tensor within --tol (relative Frobenius error),
else 64. Reports per tensor: rank needed, relative reconstruction error vs the intended float32 edit and vs the
bf16 edit the forward-pass server actually ran, and the share of ||dW||^2 captured.
Writes adapter_config.json + adapter_model.safetensors (PEFT layout, loadable by vLLM's /v1/load_lora_adapter and
uploadable to Fireworks) and package_report.json.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

MODEL_DIR = Path(os.environ.get("IDEA2_MODEL", "/opt/idea2/models/Qwen3-8B"))
PAT = re.compile(r"^model\.layers\.(\d+)\.(self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.weight$")
torch.set_grad_enabled(False)


def original(name):
    wmap = json.load(open(MODEL_DIR / "model.safetensors.index.json"))["weight_map"]
    with safe_open(MODEL_DIR / wmap[name], framework="pt") as f:
        return f.get_tensor(name).float()


def lowrank(dw, q):
    """Top-q singular triplets. Exact SVD for small q relative to size would be slow (4096 x 12288); the randomized
    method with oversampling and power iterations is accurate for low-rank edits, and the error is measured."""
    q = min(q, *dw.shape)
    U, S, V = torch.svd_lowrank(dw, q=q, niter=8)
    return U, S, V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--edits", default="/opt/idea2/work/edited")
    ap.add_argument("--rank", default="auto")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--tol", type=float, default=1e-3)
    a = ap.parse_args()
    torch.manual_seed(0)
    t0 = time.time()
    files = sorted(Path(a.edits).glob("*.npy"))
    if not files:
        raise SystemExit(f"no edits in {a.edits}")
    bad = [f.stem for f in files if not PAT.match(f.stem)]
    if bad:
        raise SystemExit("these edited tensors can't be a LoRA on q/k/v/o/gate/up/down_proj (ship a full model "
                         f"instead, or drop them): {bad}")
    fac = {}
    for f in files:
        name = f.stem
        w = original(name)
        e = torch.from_numpy(np.load(f)).float()
        dw = e - w                                   # the intended edit (float32)
        dw_served = e.to(torch.bfloat16).float() - w  # what the bf16 forward-pass server ran
        nrm = float(dw.norm())
        if nrm == 0:
            print(f"{name}: no change, skipped")
            continue
        U, S, V = lowrank(dw, 72)
        energy = torch.cumsum(S.double() ** 2, 0) / nrm ** 2
        resid = torch.sqrt(torch.clamp(1 - energy, min=0))
        k_need = int((resid <= a.tol).nonzero()[0, 0]) + 1 if bool((resid <= a.tol).any()) else None
        fac[name] = dict(U=U, S=S, V=V, dw=dw, dw_served=dw_served, nrm=nrm, k_need=k_need)
        print(f"{name}: ||dW||/||W|| = {nrm / float(w.norm()):.4g}, rank for tol {a.tol}: {k_need or '>72'}, "
              f"top singular values {[round(float(s), 4) for s in S[:4]]}")
    if not fac:
        raise SystemExit("every edit is a no-op")
    if a.rank == "auto":
        need = [v["k_need"] or 999 for v in fac.values()]
        r = min(64, max(4, max(need)))
    else:
        r = int(a.rank)
        assert 4 <= r <= 64, "Fireworks accepts LoRA rank 4..64"
    dt = getattr(torch, a.dtype)
    sd, report, mods, layers = {}, [], set(), set()
    for name, v in fac.items():
        m = PAT.match(name)
        layer, block, mod = int(m[1]), m[2], m[3]
        mods.add(mod)
        layers.add(layer)
        k = min(r, len(v["S"]))
        rs = v["S"][:k].sqrt()
        B = torch.zeros(v["dw"].shape[0], r)
        A = torch.zeros(r, v["dw"].shape[1])
        B[:, :k] = v["U"][:, :k] * rs
        A[:k] = rs[:, None] * v["V"][:, :k].T
        B, A = B.to(dt), A.to(dt)
        approx = B.float() @ A.float()
        err = float((v["dw"] - approx).norm()) / v["nrm"]
        ns = float(v["dw_served"].norm())
        err_s = float((v["dw_served"] - approx).norm()) / ns if ns else None
        key = f"base_model.model.model.layers.{layer}.{block}.{mod}"
        sd[f"{key}.lora_A.weight"] = A.contiguous()
        sd[f"{key}.lora_B.weight"] = B.contiguous()
        report.append({"tensor": name, "rank_needed": v["k_need"], "rank_used": r,
                       "rel_err_vs_intended": round(err, 6), "rel_err_vs_served_bf16": round(err_s, 6) if err_s else None,
                       "energy_captured": round(float((v["S"][:k].double() ** 2).sum()) / v["nrm"] ** 2, 6)})
        print(f"{name}: rank {r}, rel. error {err:.2e} (vs the bf16 edit the server ran: {err_s:.2e})")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    save_file(sd, out / "adapter_model.safetensors", metadata={"format": "pt"})
    cfg = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path": "Qwen/Qwen3-8B", "r": r,
           "lora_alpha": r, "lora_dropout": 0.0, "bias": "none", "target_modules": sorted(mods),
           "fan_in_fan_out": False, "use_rslora": False, "use_dora": False, "inference_mode": True,
           "init_lora_weights": True, "modules_to_save": None, "layers_to_transform": None, "rank_pattern": {},
           "alpha_pattern": {}}
    (out / "adapter_config.json").write_text(json.dumps(cfg, indent=1))
    manifest = Path(a.edits).parent / "edits_manifest.json"
    rep = {"rank": r, "dtype": a.dtype, "tol": a.tol, "tensors": report, "layers": sorted(layers),
           "target_modules": sorted(mods), "seconds": round(time.time() - t0, 1),
           "edits_manifest": json.loads(manifest.read_text()) if manifest.exists() else None}
    (out / "package_report.json").write_text(json.dumps(rep, indent=1))
    os.chmod(out, 0o777)
    print(json.dumps({k: rep[k] for k in ("rank", "dtype", "layers", "target_modules", "seconds")}))
    print(f"wrote {out}/adapter_config.json, adapter_model.safetensors, package_report.json")


if __name__ == "__main__":
    main()

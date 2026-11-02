"""Build a minimal PEFT LoRA adapter for Qwen/Qwen3-8B without downloading the model.

Only the layer shapes are needed (from the HF config.json). The adapter targets
mlp.down_proj (in 12288 -> out 4096) in every layer.

Modes:
  zeros  B = 0, so the merged model equals the base (pipeline test only).
  noise  Random A and B, seeded, scaled so the merged delta has per-entry std
         ~0.05 (larger than a typical down_proj weight). The model's output
         turns to obvious gibberish: proof the adapter weights are applied.

Run (ephemeral env, nothing installed into a project venv):
  uv run --with torch --with safetensors --with 'transformers>=4.51' --with peft \
      python build_adapter.py --mode noise --out adapters/noise-r8
  The script then checks key names and shapes against PEFT's own state dict for
  a meta-device Qwen3-8B (no weights downloaded).
"""

import argparse
import json
import math
from pathlib import Path

import torch
from safetensors.torch import save_file

# From https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json (checked 2026-11-02)
HIDDEN = 4096
INTERMEDIATE = 12288
LAYERS = 36
BASE_HF = "Qwen/Qwen3-8B"
BASE_FW = "accounts/fireworks/models/qwen3-8b"


def build(mode: str, r: int, alpha: int, target_std: float, seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    scaling = alpha / r
    # delta = scaling * B @ A; with A, B ~ N(0, s): std(delta) = scaling * s^2 * sqrt(r)
    s = math.sqrt(target_std / (scaling * math.sqrt(r)))
    tensors = {}
    for i in range(LAYERS):
        p = f"base_model.model.model.layers.{i}.mlp.down_proj"
        a = torch.randn(r, INTERMEDIATE, generator=g) * s
        if mode == "zeros":
            b = torch.zeros(HIDDEN, r)
        else:
            b = torch.randn(HIDDEN, r, generator=g) * s
        tensors[f"{p}.lora_A.weight"] = a.to(torch.bfloat16).contiguous()
        tensors[f"{p}.lora_B.weight"] = b.to(torch.bfloat16).contiguous()
    return tensors


def peft_expected(r: int, alpha: int) -> dict:
    """Key -> shape that PEFT itself would save for this config (meta device)."""
    from accelerate import init_empty_weights
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(BASE_HF)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(cfg)
    pm = get_peft_model(model, LoraConfig(r=r, lora_alpha=alpha, target_modules=["down_proj"]))
    return {k: tuple(v.shape) for k, v in get_peft_model_state_dict(pm).items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["zeros", "noise"], default="noise")
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--target-std", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-check", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tensors = build(args.mode, args.r, args.alpha, args.target_std, args.seed)

    if not args.no_check:
        exp = peft_expected(args.r, args.alpha)
        ours = {k: tuple(v.shape) for k, v in tensors.items()}
        assert ours == exp, f"mismatch: only ours {set(ours) - set(exp)}, only peft {set(exp) - set(ours)}"
        print(f"PEFT check OK: {len(ours)} tensors match PEFT key names and shapes")

    save_file(tensors, str(out / "adapter_model.safetensors"), metadata={"format": "pt"})
    config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": BASE_HF,
        "r": args.r,
        "lora_alpha": args.alpha,
        "lora_dropout": 0.0,
        "bias": "none",
        "target_modules": ["down_proj"],
        "fan_in_fan_out": False,
        "inference_mode": True,
        "use_rslora": False,
        "use_dora": False,
        "modules_to_save": None,
    }
    (out / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")
    with torch.no_grad():
        d = (args.alpha / args.r) * tensors[
            "base_model.model.model.layers.0.mlp.down_proj.lora_B.weight"
        ].float() @ tensors["base_model.model.model.layers.0.mlp.down_proj.lora_A.weight"].float()
    size = (out / "adapter_model.safetensors").stat().st_size
    print(f"wrote {out} mode={args.mode} r={args.r} alpha={args.alpha} "
          f"layer0 delta std={d.std().item():.4f} file={size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()

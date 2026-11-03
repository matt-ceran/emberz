"""Loaded once into the editor's persistent Python kernel (CPU only, no GPU, gradients off).
Adapted from the prelude of the paper's sandbox (Weight-Space Programming, Takehi et al. 2026;
https://github.com/RikiyaT/lm_understand-logs, scripts MIT).

Defined here:
  NAMES                 every tensor name of Qwen3-8B (399), in layer order; SHAPES[name]
  load(name)            float32 torch tensor: your current edit if there is one, else the original
  load_original(name)   float32 torch tensor, the original weights (safetensors mmap, read-only)
  save_edit(name, t)    writes the full edited tensor to /opt/idea2/work/edited/<name>.npy (checked)
  drop_edit(name)       removes that edit;  edited() lists the edited tensors
  tok                   the tokenizer;  render(messages) -> token ids exactly as the forward-pass server builds them
  ITEMS                 {id: item} for the dev and select splits;  DEV, SELECT: their id lists, in the order the
                        forward-pass tool walks them for {"split": ..., "n": ..., "offset": ...}
  gen(k)                rows of /opt/idea2/work/gen_<k>.jsonl (a generation pass)
  fwd(k)                meta.json of /opt/idea2/work/fwd_<k>/ (a read pass);  fwd_arr(k, name) -> np.memmap
  CFG                   architecture numbers
"""
import json
import math
import os
import re

import numpy as np
import torch

torch.set_grad_enabled(False)
np.set_printoptions(precision=4, suppress=True, linewidth=160)
torch.set_printoptions(precision=4, sci_mode=False, linewidth=160)

MODEL_DIR = os.environ.get("IDEA2_MODEL", "/opt/idea2/models/Qwen3-8B")
WORK = os.environ.get("IDEA2_WORK", "/opt/idea2/work")
SPLITS = os.environ.get("IDEA2_SPLITS", "/opt/idea2/editor/splits")
EDIT_DIR = os.path.join(WORK, "edited")
os.makedirs(EDIT_DIR, exist_ok=True)
REFUSED = {"model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"}
CFG = dict(n_layers=36, d_model=4096, d_mlp=12288, n_heads=32, n_kv_heads=8, head_dim=128, vocab=151936,
           rope_theta=1e6, rms_eps=1e-6, tied_embeddings=False)

from safetensors import safe_open  # noqa: E402

_WMAP = json.load(open(os.path.join(MODEL_DIR, "model.safetensors.index.json")))["weight_map"]
_FILES = {}


def _key(n):
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", n)]


NAMES = sorted(_WMAP, key=_key)


def _f(name):
    fn = _WMAP[name]
    if fn not in _FILES:
        _FILES[fn] = safe_open(os.path.join(MODEL_DIR, fn), framework="pt")
    return _FILES[fn]


SHAPES = {n: tuple(_f(n).get_slice(n).get_shape()) for n in NAMES}


def _path(name):
    return os.path.join(EDIT_DIR, name + ".npy")


def load_original(name):
    assert name in SHAPES, f"unknown tensor {name!r}; see NAMES"
    return _f(name).get_tensor(name).float()


def load(name):
    assert name in SHAPES, f"unknown tensor {name!r}; see NAMES"
    p = _path(name)
    return torch.from_numpy(np.load(p)) if os.path.exists(p) else load_original(name)


def save_edit(name, t):
    """Save the FULL edited tensor. Refused: unknown names, wrong shapes, non-finite values, the final norm,
    the input embedding, the output layer, and any edit that is just the original times a constant."""
    assert name in SHAPES, f"unknown tensor {name!r}; see NAMES"
    assert name not in REFUSED, f"{name} is out of scope (final norm, input embedding and output layer are refused)"
    a = (t.detach().cpu().float().numpy() if isinstance(t, torch.Tensor) else np.asarray(t, dtype=np.float32))
    a = np.ascontiguousarray(a, dtype=np.float32)
    assert a.shape == SHAPES[name], (name, a.shape, SHAPES[name])
    assert np.isfinite(a).all(), f"{name}: non-finite values"
    o = load_original(name)
    e = torch.from_numpy(a)
    oo = float((o.double() * o.double()).sum())
    c = float((e.double() * o.double()).sum()) / oo if oo > 0 else 0.0
    en = float(e.double().norm())
    resid = float((e.double() - c * o.double()).norm()) / en if en > 0 else 0.0
    assert not (resid < 1e-3 and abs(c - 1) > 1e-3), \
        f"{name}: the edit is the original times a constant ({c:.4g}); whole-tensor scaling is refused"
    changed = int((e.to(torch.bfloat16) != o.to(torch.bfloat16)).sum())
    tmp = _path(name) + ".tmp"
    with open(tmp, "wb") as f:
        np.save(f, a)
    os.replace(tmp, _path(name))
    info = {"tensor": name, "changed_entries_bf16": changed, "of": int(o.numel()),
            "rel_delta_fro": round(float((e - o).norm()) / math.sqrt(oo), 6) if oo > 0 else None}
    if changed == 0:
        info["warning"] = "no entry changes after rounding to bf16 (the server runs bf16): this edit is a no-op"
    return info


def drop_edit(name):
    p = _path(name)
    if os.path.exists(p):
        os.remove(p)
        return True
    return False


def edited():
    return sorted(f[:-4] for f in os.listdir(EDIT_DIR) if f.endswith(".npy"))


from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(MODEL_DIR)


def render(messages, generation_prompt=None):
    """Token ids exactly as the forward-pass server builds them (chat template, thinking off). A chat that ends
    with a user turn gets the assistant prefix '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' appended."""
    if generation_prompt is None:
        generation_prompt = messages[-1]["role"] == "user"
    text = tok.apply_chat_template(messages, add_generation_prompt=generation_prompt, tokenize=False,
                                   enable_thinking=False)
    return tok(text, add_special_tokens=False)["input_ids"]


ITEMS, DEV, SELECT = {}, [], []
for _s, _lst in (("dev", DEV), ("select", SELECT)):
    _p = os.path.join(SPLITS, f"{_s}.jsonl")
    if os.path.exists(_p):
        for _l in open(_p):
            _r = json.loads(_l)
            _r["split"] = _s
            ITEMS[_r["id"]] = _r
            _lst.append(_r["id"])


import random as _random

for _s, _lst in (("dev", DEV), ("select", SELECT)):  # the order the forward-pass tool uses for "split" + "n"/"offset"
    _lst.sort()
    _random.Random(f"{_s}-0").shuffle(_lst)


def gen(k):
    return [json.loads(l) for l in open(os.path.join(WORK, f"gen_{k}.jsonl"))]


def fwd(k):
    return json.load(open(os.path.join(WORK, f"fwd_{k}", "meta.json")))


def fwd_arr(k, name):
    return np.load(os.path.join(WORK, f"fwd_{k}", name + ".npy"), mmap_mode="r")


_IDEA2_PRELUDE = True

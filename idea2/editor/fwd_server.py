"""Forward-pass server for weight-editing sessions on Qwen3-8B (idea2). Runs ON THE VM as systemd `idea2-editor`,
127.0.0.1:8100. The only way the editor runs the model.

Modelled on the forward-pass tool of Weight-Space Programming (Takehi, Teoh, Zhang, Khattab, 2026;
logs and harness: https://github.com/RikiyaT/lm_understand-logs, logs CC BY 4.0, scripts MIT).

- Holds Qwen3-8B in bf16 on the GPU (transformers, sdpa attention). Gradients are off everywhere
  (requires_grad False, torch.inference_mode around every forward).
- Edits: full tensors in $WORK/edited/<name>.npy (float32, same name/shape as the checkpoint), written by the
  editor's kernel (kernel_prelude.save_edit). They are applied in bf16. For every edited tensor the version
  that is NOT on the GPU right now is kept on the CPU, so "original" / "edited" / "both" swap tensors in place.
  Any change in $WORK/edited is picked up automatically before an "edited"/"both" pass (or POST /reload_edits).
- Budget: every successful POST /generate or /read is one pass ($WORK/budget.json). A failed request costs
  nothing. Requests beyond the budget are refused.
- Items: only ids from the allowed splits (default dev, select) in $SPLITS; everything else (test) is refused.
  Custom prompts are allowed.
- Audit trail: every request and a summary of its response go to $WORK/log.jsonl with timestamps.

Endpoints: GET /health, GET /budget, POST /generate, POST /read, POST /reload_edits. See README.md.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import random
import threading
import time
import traceback
from pathlib import Path

import anyio
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from grader import CONDITIONS, DEFAULT_MAX_TOKENS, grade, grade_custom

MODEL_DIR = Path(os.environ.get("IDEA2_MODEL", "/opt/idea2/models/Qwen3-8B"))
WORK = Path(os.environ.get("IDEA2_WORK", "/opt/idea2/work"))
SPLITS = Path(os.environ.get("IDEA2_SPLITS", "/opt/idea2/editor/splits"))
BUDGET = int(os.environ.get("EDITOR_BUDGET", "50"))
ALLOWED_SPLITS = [s for s in os.environ.get("EDITOR_ALLOWED_SPLITS", "dev,select").split(",") if s]
MAX_GEN = int(os.environ.get("EDITOR_MAX_GEN", "1024"))          # generations per model per pass
MAX_NEW_TOKENS = int(os.environ.get("EDITOR_MAX_NEW_TOKENS", "512"))
MAX_BATCH = int(os.environ.get("EDITOR_GEN_BATCH", "128"))
KV_TOKENS = int(os.environ.get("EDITOR_KV_TOKENS", "24000"))     # (prompt + new) tokens per generation batch
READ_TOKENS = int(os.environ.get("EDITOR_READ_BATCH_TOKENS", "8192"))
MAX_READ_TEXTS = int(os.environ.get("EDITOR_MAX_READ_TEXTS", "1024"))
MAX_READ_TOTAL = int(os.environ.get("EDITOR_MAX_READ_TOKENS", "262144"))
MAX_TEXT_TOKENS = int(os.environ.get("EDITOR_MAX_TEXT_TOKENS", "4096"))
MAX_READ_GB = float(os.environ.get("EDITOR_MAX_READ_GB", "6"))
MAX_ATTN_TEXTS, MAX_ATTN_TOKENS = 4, 2048
REFUSED = {"model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"}
EDIT_DIR = WORK / "edited"
LOG = WORK / "log.jsonl"
DEV = torch.device("cuda")

os.umask(0)
for d in (WORK, EDIT_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ logging / budget
def log(entry: dict) -> None:
    entry = {"t": time.time(), "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **entry}
    line = json.dumps(entry, ensure_ascii=False, default=str) + "\n"
    with open(LOG, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(line)
        fcntl.flock(f, fcntl.LOCK_UN)


class Budget:
    """$WORK/budget.json is the source of truth (re-read on every check), so a new session (mode.sh new-session
    moves $WORK away) starts at 0 without restarting the server."""
    path = WORK / "budget.json"

    def _read(self):
        if self.path.exists():
            st = json.loads(self.path.read_text())
            return st["used"], st["passes"]
        return 0, []

    def _write(self, used, passes):
        WORK.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"allowed": BUDGET, "used": used, "passes": passes}, indent=1))
        os.replace(tmp, self.path)

    def check(self):
        used, _ = self._read()
        if used >= BUDGET:
            raise HTTPException(403, f"budget exhausted: {used}/{BUDGET} passes used")
        return used + 1  # the index k of the pass about to run

    def spend(self, k, kind, seconds):
        used, passes = self._read()
        passes.append({"k": k, "kind": kind, "t": time.time(), "s": round(seconds, 2)})
        self._write(k, passes)

    def view(self):
        used, _ = self._read()
        return {"allowed": BUDGET, "used": used, "left": BUDGET - used}


# ------------------------------------------------------------------ model + edits
T0 = time.time()
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig  # noqa: E402

tok = AutoTokenizer.from_pretrained(MODEL_DIR)


def load_model():
    """Copy the checkpoint straight into GPU storage with safetensors. transformers 5.20's
    from_pretrained(device_map="cuda") takes ~6 min on this VM (it waits on GPU-side random init). Every parameter
    must come from the checkpoint (checked); EDITOR_HF_LOAD=1 uses from_pretrained instead."""
    from safetensors import safe_open
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(MODEL_DIR)
    cfg._attn_implementation = "sdpa"
    # Build on the meta device (from_config on "cuda" spends minutes in random init), allocate empty GPU storage,
    # then fill every parameter from the checkpoint and recompute the rotary buffers (not in the checkpoint).
    with torch.device("meta"):
        m = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16, attn_implementation="sdpa")
    m.to_empty(device="cuda")
    with torch.device("cpu"):
        rot = type(m.model.rotary_emb)(config=cfg)
    m.model.rotary_emb = rot.to("cuda")
    assert all(n.startswith("model.rotary_emb.") for n, _ in m.named_buffers()), [n for n, _ in m.named_buffers()]
    params = dict(m.named_parameters())
    seen = set()
    wmap = json.load(open(MODEL_DIR / "model.safetensors.index.json"))["weight_map"]
    with torch.no_grad():
        for fn in sorted(set(wmap.values())):
            with safe_open(MODEL_DIR / fn, framework="pt", device="cuda") as sf:
                for name in sf.keys():
                    t = sf.get_tensor(name)
                    assert params[name].shape == t.shape, name
                    params[name].copy_(t)
                    seen.add(name)
    missing = set(params) - seen
    assert not missing, f"parameters not in the checkpoint: {sorted(missing)[:5]}"
    return m


if os.environ.get("EDITOR_HF_LOAD") == "1":
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16, device_map="cuda",
                                                 attn_implementation="sdpa")
else:
    model = load_model()
model.eval()
model.requires_grad_(False)
torch.set_grad_enabled(False)
LOAD_S = time.time() - T0
PARAMS = dict(model.named_parameters())
LAYERS = model.model.layers
N_LAYERS = len(LAYERS)
PAD = tok.pad_token_id if tok.pad_token_id is not None else 151643
EOS = [151645, 151643]
GEN_CFG = GenerationConfig(do_sample=False, pad_token_id=PAD, eos_token_id=EOS)
model.generation_config = GEN_CFG


def bf16_sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.contiguous().view(torch.int16).cpu().numpy().tobytes()).hexdigest()[:16]


GPU_STASH_GB = float(os.environ.get("EDITOR_GPU_STASH_GB", "1.5"))
_WMAP = json.load(open(MODEL_DIR / "model.safetensors.index.json"))["weight_map"]


def original_cpu(name: str) -> torch.Tensor:
    from safetensors import safe_open
    with safe_open(MODEL_DIR / _WMAP[name], framework="pt") as f:
        return f.get_tensor(name)


class Weights:
    """GPU params + a stash holding, for each edited tensor, the version that is NOT on the GPU right now. The stash
    lives on the GPU while it fits in EDITOR_GPU_STASH_GB (a swap is then a device copy), otherwise in CPU memory.
    Edits are checked on the CPU against the checkpoint, so a reload moves each edited tensor over PCIe once."""

    def __init__(self):
        self.state = "original"
        self.stash: dict[str, torch.Tensor] = {}
        self.sig = None
        self.manifest: dict = {"tensors": [], "hash": None, "n_tensors": 0, "changed_entries": 0}

    @torch.no_grad()
    def to(self, state: str):
        if state == self.state:
            return
        for name, other in self.stash.items():
            p = PARAMS[name]
            cur = p.data.clone() if other.is_cuda else p.data.to("cpu", copy=True)
            p.data.copy_(other if other.is_cuda else other.to(DEV))
            self.stash[name] = cur
        self.state = state

    @staticmethod
    def scan():
        EDIT_DIR.mkdir(parents=True, exist_ok=True)
        files = sorted(EDIT_DIR.glob("*.npy"))
        return tuple((f.name, f.stat().st_size, f.stat().st_mtime_ns) for f in files)

    @torch.no_grad()
    def reload(self) -> dict:
        sig = self.scan()
        new, refusals = {}, []
        for fname, _, _ in sig:
            name = fname[:-4]
            try:
                if name not in PARAMS:
                    raise ValueError("unknown tensor name")
                if name in REFUSED:
                    raise ValueError("out of scope (final norm, input embedding and output layer are refused)")
                arr = np.load(EDIT_DIR / fname, mmap_mode="r")
                p = PARAMS[name]
                if tuple(arr.shape) != tuple(p.shape):
                    raise ValueError(f"shape {tuple(arr.shape)} != {tuple(p.shape)}")
                e = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
                if not torch.isfinite(e).all():
                    raise ValueError("non-finite values")
                ob = original_cpu(name)
                o = ob.float()
                oo = float((o.double() ** 2).sum())
                c = float((e.double() * o.double()).sum()) / oo if oo > 0 else 0.0
                en = float(e.double().norm())
                resid = float((e.double() - c * o.double()).norm()) / en if en > 0 else 0.0
                if resid < 1e-3 and abs(c - 1) > 1e-3:
                    raise ValueError(f"the edit is the original times a constant ({c:.4g}); whole-tensor scaling is refused")
                eb = e.to(torch.bfloat16)
                changed = int((eb != ob).sum())
                delta = float((eb.float() - o).norm())
                new[name] = {"cpu": eb, "entry": {
                    "tensor": name, "shape": list(p.shape), "changed_entries": changed,
                    "changed_frac": round(changed / p.numel(), 8),
                    "rel_delta_fro": round(delta / math.sqrt(oo), 8) if oo > 0 else None,
                    "sha256_bf16": bf16_sha(eb)}}
                del e, o, ob
            except Exception as ex:  # noqa: BLE001
                refusals.append({"tensor": name, "reason": str(ex)})
        if refusals:
            log({"kind": "edits_refused", "refusals": refusals})
            raise HTTPException(400, {"error": "edits refused; nothing was changed (the previously loaded edits "
                                               "stay in effect)", "refusals": refusals})
        self.to("original")  # GPU = original for every previously edited tensor
        self.stash = {}
        torch.cuda.empty_cache()
        budget_b = GPU_STASH_GB * 2**30
        used = 0
        for n, v in new.items():
            b = v["cpu"].numel() * 2
            if used + b <= budget_b:
                self.stash[n] = v["cpu"].to(DEV)
                used += b
                v["entry"]["stash"] = "gpu"
            else:
                self.stash[n] = v["cpu"]
                v["entry"]["stash"] = "cpu"
        ents = [v["entry"] for v in new.values()]
        h = hashlib.sha256(json.dumps([(e["tensor"], e["sha256_bf16"]) for e in ents]).encode()).hexdigest()[:16] \
            if ents else None
        self.manifest = {"tensors": ents, "n_tensors": len(ents),
                         "changed_entries": sum(e["changed_entries"] for e in ents), "hash": h}
        self.sig = sig
        (WORK / "edits_manifest.json").write_text(json.dumps(self.manifest, indent=1))
        log({"kind": "edits", "manifest": self.manifest})
        return self.manifest

    def ensure_current(self) -> bool:
        if self.scan() != self.sig:
            self.reload()
            return True
        return False


W = Weights()
try:
    W.reload()
except HTTPException as e:
    print("startup: edits not loaded:", e.detail, flush=True)
    W.sig = None
budget = Budget()
LOCK = threading.Lock()
log({"kind": "server_start", "load_s": round(LOAD_S, 1), "budget": budget.view(),
     "allowed_splits": ALLOWED_SPLITS, "edits": W.manifest.get("hash")})


# ------------------------------------------------------------------ items
def load_items():
    items = {}
    for s in ALLOWED_SPLITS:
        p = SPLITS / f"{s}.jsonl"
        if not p.exists():
            continue
        for line in p.open():
            r = json.loads(line)
            r["split"] = s
            items[r["id"]] = r
    return items


ITEMS = load_items()
# "split" + "n"/"offset" walk each split in a fixed shuffled order (seed 0), so any prefix mixes the datasets
SPLIT_IDS = {}
for _s in ALLOWED_SPLITS:
    _ids = sorted(i for i, r in ITEMS.items() if r["split"] == _s)
    random.Random(f"{_s}-0").shuffle(_ids)
    SPLIT_IDS[_s] = _ids


def pick_items(req: dict) -> list[dict]:
    ids = list(req.get("ids") or [])
    if req.get("split"):
        s = req["split"]
        if s not in SPLIT_IDS:
            raise HTTPException(403, f"split {s!r} is not available (allowed: {ALLOWED_SPLITS})")
        pool = SPLIT_IDS[s]
        if req.get("dataset"):
            pool = [i for i in pool if ITEMS[i]["dataset"] == req["dataset"]]
        off = int(req.get("offset", 0))
        n = req.get("n")
        ids += pool[off: off + int(n)] if n is not None else pool[off:]
    bad = [i for i in ids if i not in ITEMS]
    if bad:
        raise HTTPException(403, f"{len(bad)} ids are not in the allowed splits {ALLOWED_SPLITS} "
                                 f"(test ids and unknown ids are refused): {bad[:5]}")
    if len(set(ids)) != len(ids):
        raise HTTPException(400, "duplicate ids")
    return [ITEMS[i] for i in ids]


def render(messages: list[dict], gen_prompt: bool | None = None) -> list[int]:
    if gen_prompt is None:
        gen_prompt = messages[-1]["role"] == "user"
    text = tok.apply_chat_template(messages, add_generation_prompt=gen_prompt, tokenize=False,
                                   enable_thinking=False)
    return tok(text, add_special_tokens=False)["input_ids"]


def ascii_excerpt(s: str, n: int = 200) -> str:
    s = (s or "").replace("\n", " ")
    s = s.encode("ascii", "replace").decode()
    return s if len(s) <= n else s[:n] + "..."


# ------------------------------------------------------------------ generation
def gen_jobs(req: dict) -> list[dict]:
    conds = req.get("conditions") or list(CONDITIONS)
    for c in conds:
        if c not in CONDITIONS:
            raise HTTPException(400, f"unknown condition {c!r}; use {CONDITIONS}")
    mt = req.get("max_tokens")
    if mt is not None and not (1 <= int(mt) <= MAX_NEW_TOKENS):
        raise HTTPException(400, f"max_tokens must be in 1..{MAX_NEW_TOKENS}")
    jobs = []
    for it in pick_items(req):
        for c in conds:
            jobs.append({"key": (it["id"], c), "id": it["id"], "split": it["split"], "dataset": it["dataset"],
                         "condition": c, "item": it, "prompt": it["prompts"][c],
                         "max_new": int(mt or DEFAULT_MAX_TOKENS.get(it["dataset"], 160))})
    for j, p in enumerate(req.get("prompts") or []):
        spec = p if isinstance(p, dict) else {"prompt": p}
        msgs = spec.get("chat") or [{"role": "user", "content": spec["prompt"]}]
        jobs.append({"key": ("custom", j), "custom": j, "condition": spec.get("condition", "custom"),
                     "dataset": "custom", "spec": spec, "messages": msgs,
                     "prompt": msgs[-1]["content"], "max_new": int(mt or spec.get("max_tokens") or 160)})
    if not jobs:
        raise HTTPException(400, "nothing to generate: give ids / split / prompts")
    if len(jobs) > MAX_GEN:
        raise HTTPException(400, f"{len(jobs)} generations per model; at most {MAX_GEN} per pass")
    for j in jobs:
        j["ids"] = render(j.get("messages") or [{"role": "user", "content": j["prompt"]}])
    return jobs


def kv_budget() -> int:
    """(prompt + new) tokens per generation batch: EDITOR_KV_TOKENS, lowered to what free GPU memory allows
    (~144 KiB of KV cache per token for Qwen3-8B bf16, x1.5 for activations and the allocator)."""
    # free = what CUDA has left + what PyTorch's caching allocator holds but isn't using
    free = torch.cuda.mem_get_info()[0] + torch.cuda.memory_reserved() - torch.cuda.memory_allocated() - 1.0 * 2**30
    return max(2048, min(KV_TOKENS, int(free / (36 * 8 * 128 * 2 * 2 * 1.5))))


STAGE1 = int(os.environ.get("EDITOR_STAGE1_TOKENS", "64"))
# EDITOR_CACHE=static preallocates the KV cache: measured slower here (687 s vs 556 s for a 768-generation paired
# pass) and it changes ~5% of texts, so it is server-wide only (never mixed within a session)
CACHE_IMPL = {"cache": os.environ.get("EDITOR_CACHE", "dynamic")}


def plan_batches(jobs: list[dict], subset, cap) -> list[list[int]]:
    """Batches over job indices `subset`, grouped by the stage's max_new (cap(i)) and sorted by prompt length."""
    kv = kv_budget()
    order = sorted(subset, key=lambda i: (cap(i), len(jobs[i]["ids"])))
    batches, cur, cur_max = [], [], 0
    for i in order:
        need = len(jobs[i]["ids"]) + cap(i)
        if cur and (cap(i) != cap(cur[0]) or len(cur) >= MAX_BATCH or max(cur_max, need) * (len(cur) + 1) > kv):
            batches.append(cur)
            cur, cur_max = [], 0
        cur.append(i)
        cur_max = max(cur_max, need)
    if cur:
        batches.append(cur)
    return batches


@torch.inference_mode()
def run_batch(jobs, idx, max_new):
    L = max(len(jobs[i]["ids"]) for i in idx)
    inp = torch.full((len(idx), L), PAD, dtype=torch.long)
    att = torch.zeros((len(idx), L), dtype=torch.long)
    for r, i in enumerate(idx):
        ids = jobs[i]["ids"]
        inp[r, L - len(ids):] = torch.tensor(ids)
        att[r, L - len(ids):] = 1
    kw = {"cache_implementation": "static"} if CACHE_IMPL["cache"] == "static" else {}
    out = model.generate(input_ids=inp.to(DEV), attention_mask=att.to(DEV), do_sample=False,
                         max_new_tokens=max_new, **kw)
    new = out[:, L:].cpu().tolist()
    res = []
    for seq in new:
        n = next((k for k, t in enumerate(seq) if t in EOS), None)
        toks = seq if n is None else seq[:n]
        res.append({"text": tok.decode(toks, skip_special_tokens=True), "n_tokens": len(toks),
                    "finish": "length" if n is None else "stop"})
    return res


def generate_all(jobs, batches, cap) -> dict:
    outs = {}
    stack = list(batches)
    while stack:
        idx = stack.pop(0)
        try:
            for i, r in zip(idx, run_batch(jobs, idx, cap(idx[0]))):
                outs[i] = r
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(idx) == 1:
                raise
            h = len(idx) // 2
            stack[:0] = [idx[:h], idx[h:]]
            log({"kind": "oom_split", "batch": len(idx)})
    return outs


def generate_models(jobs, models):
    """Greedy generation for each model in two stages: every job with at most STAGE1 new tokens (short answers
    finish here, in large batches), then the jobs that hit that cap in ANY model re-run from scratch with their full
    max_tokens. Both models always see identical batches, so a no-op edit gives identical texts."""
    cap1 = lambda i: min(STAGE1, jobs[i]["max_new"])  # noqa: E731
    capf = lambda i: jobs[i]["max_new"]  # noqa: E731
    b1 = plan_batches(jobs, range(len(jobs)), cap1)
    outs, times = {}, {}
    for m in models:
        t = time.time()
        W.to(m)
        outs[m] = generate_all(jobs, b1, cap1)
        times[m] = time.time() - t
    rerun = sorted({i for m in models for i, o in outs[m].items()
                    if o["finish"] == "length" and jobs[i]["max_new"] > STAGE1})
    b2 = plan_batches(jobs, rerun, capf) if rerun else []
    for m in (reversed(models) if rerun else []):  # start with the model already on the GPU
        t = time.time()
        W.to(m)
        outs[m].update(generate_all(jobs, b2, capf))
        times[m] += time.time() - t
    return {m: [outs[m][i] for i in range(len(jobs))] for m in models}, \
        {m: round(v, 1) for m, v in times.items()}, {"stage1_batches": len(b1), "rerun": len(rerun),
                                                     "stage2_batches": len(b2), "kv_tokens_per_batch": kv_budget(),
                                                     "max_batch": max(map(len, b1))}


def mean_se(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return None
    return {"mean": round(float(x.mean()), 4), "se": round(float(x.std(ddof=1) / math.sqrt(n)), 4) if n > 1 else None,
            "n": n}


def model_summary(rows):
    out = {}
    graded = [r for r in rows if r.get("correct") is not None]
    for ds in sorted({r["dataset"] for r in graded}) + ["all"]:
        g = [r for r in graded if ds == "all" or r["dataset"] == ds]
        d = {}
        for c in sorted({r["condition"] for r in g}):
            gc = [r for r in g if r["condition"] == c]
            d[c] = {"n": len(gc), "acc": round(float(np.mean([r["correct"] for r in gc])), 4),
                    "says_wrong": round(float(np.mean([r["says_wrong"] for r in gc])), 4),
                    "mean_tokens": round(float(np.mean([r["n_tokens"] for r in gc])), 1),
                    "cut": round(float(np.mean([r["finish"] == "length" for r in gc])), 4)}
        by = {}
        for r in g:
            by.setdefault(r.get("id", ("custom", r.get("custom"))), {})[r["condition"]] = r["correct"]
        known = [v for v in by.values() if v.get("neutral") is True]
        if known and all("wrong" in v for v in known):
            d["caving_on_own_known"] = {"n": len(known), "rate": round(float(np.mean([not v["wrong"] for v in known])), 4)}
        if known and all("right" in v for v in known):
            d["right_on_own_known"] = {"n": len(known), "acc": round(float(np.mean([v["right"] for v in known])), 4)}
        out[ds] = d
    other = [r for r in rows if r.get("correct") is None]
    if other:
        out["ungraded"] = {"n": len(other), "mean_tokens": round(float(np.mean([r["n_tokens"] for r in other])), 1)}
    return out


def paired_summary(orig, edit):
    o = {r["key"]: r for r in orig}
    e = {r["key"]: r for r in edit}
    keys = [k for k in o if o[k].get("correct") is not None]
    out = {}
    datasets = sorted({o[k]["dataset"] for k in keys}) + ["all"]
    for ds in datasets:
        ks = [k for k in keys if ds == "all" or o[k]["dataset"] == ds]
        d = {}
        for c in sorted({o[k]["condition"] for k in ks}):
            kc = [k for k in ks if o[k]["condition"] == c]
            for metric in ("correct", "says_wrong"):
                if metric == "says_wrong" and c != "wrong":
                    continue
                a = np.array([o[k][metric] for k in kc], dtype=float)
                b = np.array([e[k][metric] for k in kc], dtype=float)
                ms = mean_se(b - a)
                d[f"{c}.{metric}"] = {"original": round(float(a.mean()), 4), "edited": round(float(b.mean()), 4),
                                      "diff": ms["mean"], "se": ms["se"], "n": ms["n"],
                                      "z": round(ms["mean"] / ms["se"], 2) if ms["se"] else None,
                                      "up": int(((a == 0) & (b == 1)).sum()), "down": int(((a == 1) & (b == 0)).sum())}
        # caving on the items the ORIGINAL model answers correctly in this pass (fixed denominator, paired)
        ids = {k[0] for k in ks}
        kn = [i for i in ids if (i, "neutral") in o and o[(i, "neutral")]["correct"] and (i, "wrong") in o]
        if kn:
            a = np.array([not o[(i, "wrong")]["correct"] for i in kn], dtype=float)
            b = np.array([not e[(i, "wrong")]["correct"] for i in kn], dtype=float)
            ms = mean_se(b - a)
            d["caving_on_original_known"] = {"original": round(float(a.mean()), 4), "edited": round(float(b.mean()), 4),
                                             "diff": ms["mean"], "se": ms["se"], "n": ms["n"],
                                             "z": round(ms["mean"] / ms["se"], 2) if ms["se"] else None,
                                             "rescued": int(((a == 1) & (b == 0)).sum()),
                                             "newly_caving": int(((a == 0) & (b == 1)).sum())}
        out[ds] = d
    same = sum(o[k]["text"] == e[k]["text"] for k in o)
    out["identical_texts"] = {"n": len(o), "same": same}
    return out


def flip_examples(orig, edit, n=4):
    e = {r["key"]: r for r in edit}
    ups, downs = [], []
    for r in orig:
        if r.get("correct") is None:
            continue
        x = e[r["key"]]
        ex = {"id": r.get("id", r.get("custom")), "condition": r["condition"],
              "original": ascii_excerpt(r["text"], 160), "edited": ascii_excerpt(x["text"], 160)}
        if not r["correct"] and x["correct"]:
            ups.append(ex)
        elif r["correct"] and not x["correct"]:
            downs.append(ex)
    return {"up": ups[:n], "down": downs[:n]}


def do_generate(req: dict):
    which = req.get("model", "edited")
    if which not in ("original", "edited", "both"):
        raise HTTPException(400, "model must be original, edited or both")
    jobs = gen_jobs(req)
    k = budget.check()
    t0 = time.time()
    reloaded = W.ensure_current() if which != "original" else False
    models = ["original", "edited"] if which == "both" else [which]
    allouts, times, plan = generate_models(jobs, models)
    rows = {}
    for m in models:
        rs = []
        for j, o in zip(jobs, allouts[m]):
            r = {"model": m, "key": j["key"], "condition": j["condition"], "dataset": j["dataset"],
                 "prompt": j["prompt"], **o}
            if "id" in j:
                r.update({"id": j["id"], "split": j["split"]})
                g = grade(j["item"], o["text"])
            else:
                r["custom"] = j["custom"]
                g = grade_custom(j["spec"], o["text"])
            if g:
                r.update(g)
            rs.append(r)
        rows[m] = rs
    path = WORK / f"gen_{k}.jsonl"
    with open(path, "w") as f:
        for m in models:
            for r in rows[m]:
                f.write(json.dumps({kk: v for kk, v in r.items() if kk != "key"}, ensure_ascii=False) + "\n")
    secs = time.time() - t0
    budget.spend(k, "generate", secs)
    resp = {"pass": k, "file": str(path), "seconds": round(secs, 1), "seconds_per_model": times,
            "n_generations_per_model": len(jobs), "batches": plan, "budget": budget.view(),
            "edits": {"hash": W.manifest.get("hash"), "n_tensors": W.manifest.get("n_tensors", 0),
                      "changed_entries": W.manifest.get("changed_entries", 0), "reloaded": reloaded},
            "summary": {m: model_summary(rows[m]) for m in models}}
    if which == "both":
        resp["paired"] = paired_summary(rows["original"], rows["edited"])
        resp["flips"] = flip_examples(rows["original"], rows["edited"])
    (WORK / f"gen_{k}.summary.json").write_text(json.dumps(resp, indent=1))
    return resp


# ------------------------------------------------------------------ teacher-forced read
def layer_list(xs, hi, what):
    xs = [int(x) for x in (xs or [])]
    for x in xs:
        if not 0 <= x <= hi:
            raise HTTPException(400, f"{what}: layer {x} outside 0..{hi}")
    return sorted(set(xs))


def read_inputs(req):
    seqs, src = [], []
    for i, t in enumerate(req.get("texts") or []):
        seqs.append(tok(t, add_special_tokens=False)["input_ids"])
        src.append({"text": i})
    for i, ch in enumerate(req.get("chats") or []):
        seqs.append(render(ch))
        src.append({"chat": i})
    items = list(req.get("items") or [])
    if req.get("ids") or req.get("split"):
        conds = req.get("conditions") or ["neutral"]
        items += [{"id": it["id"], "condition": c} for it in pick_items(req) for c in conds]
    for spec in items:
        it = pick_items({"ids": [spec["id"]]})[0]
        c = spec.get("condition", "neutral")
        if c not in CONDITIONS:
            raise HTTPException(400, f"unknown condition {c!r}")
        msgs = [{"role": "user", "content": it["prompts"][c]}]
        if spec.get("reply") is not None:
            msgs.append({"role": "assistant", "content": spec["reply"]})
        seqs.append(render(msgs))
        src.append({"id": it["id"], "condition": c, "reply": spec.get("reply") is not None})
    if not seqs:
        raise HTTPException(400, "nothing to read: give texts / chats / items / ids")
    if len(seqs) > MAX_READ_TEXTS:
        raise HTTPException(400, f"{len(seqs)} texts; at most {MAX_READ_TEXTS}")
    if max(map(len, seqs)) > MAX_TEXT_TOKENS or sum(map(len, seqs)) > MAX_READ_TOTAL:
        raise HTTPException(400, f"texts too long: max {MAX_TEXT_TOKENS} tokens each, {MAX_READ_TOTAL} in total")
    if min(map(len, seqs)) == 0:
        raise HTTPException(400, "empty text")
    return seqs, src


def positions_for(req, seqs):
    p = req.get("positions", "last")
    if p == "last":
        return [[len(s) - 1] for s in seqs]
    if p == "all":
        return [list(range(len(s))) for s in seqs]
    if isinstance(p, list) and p and all(isinstance(x, int) for x in p):
        p = [p] * len(seqs)
    if not (isinstance(p, list) and len(p) == len(seqs)):
        raise HTTPException(400, 'positions: "last", "all", a list of ints for every text, or one list per text')
    out = []
    for s, ps in zip(seqs, p):
        q = [x + len(s) if x < 0 else x for x in ps]
        if any(not 0 <= x < len(s) for x in q):
            raise HTTPException(400, f"position out of range for a text of {len(s)} tokens")
        out.append(q)
    return out


def do_read(req: dict):
    which = req.get("model", "original")
    if which not in ("original", "edited"):
        raise HTTPException(400, "model must be original or edited")
    what = req.get("what")
    lays = req.get("layers")
    spec = {"residual": req.get("residual"), "mlp_act": req.get("mlp_act"), "o_proj_in": req.get("o_proj_in"),
            "attn": req.get("attn")}
    if what:  # {"what": ["residual", "mlp_act"], "layers": [...]} form
        for w in ([what] if isinstance(what, str) else what):
            if w not in spec:
                raise HTTPException(400, f"unknown what {w!r}; use residual, mlp_act, o_proj_in, attn")
            spec[w] = spec[w] or lays
    res_l = layer_list(spec["residual"], N_LAYERS, "residual")
    mlp_l = layer_list(spec["mlp_act"], N_LAYERS - 1, "mlp_act")
    opi_l = layer_list(spec["o_proj_in"], N_LAYERS - 1, "o_proj_in")
    att_l = layer_list(spec["attn"], N_LAYERS - 1, "attn")
    topk = int(req.get("topk", 0))
    if not 0 <= topk <= 20:
        raise HTTPException(400, "topk must be in 0..20")
    want_lp = bool(req.get("logprobs"))
    dtype = {"float32": np.float32, "float16": np.float16}[req.get("dtype", "float32")]
    seqs, src = read_inputs(req)
    pos = positions_for(req, seqs)
    if att_l and (len(seqs) > MAX_ATTN_TEXTS or max(map(len, seqs)) > MAX_ATTN_TOKENS):
        raise HTTPException(400, f"attn: at most {MAX_ATTN_TEXTS} texts of at most {MAX_ATTN_TOKENS} tokens")
    if not (res_l or mlp_l or opi_l or att_l or topk or want_lp):
        raise HTTPException(400, "nothing requested: residual / mlp_act / o_proj_in / attn layers, logprobs or topk")
    M = sum(map(len, pos))
    isz = np.dtype(dtype).itemsize
    est = M * isz * (4096 * (len(res_l) + len(opi_l)) + 12288 * len(mlp_l)) + M * topk * 12
    est += sum(32 * len(s) ** 2 * 2 for s in seqs) * len(att_l)
    if est > MAX_READ_GB * 2**30:
        raise HTTPException(400, f"would write {est / 2**30:.1f} GB; limit {MAX_READ_GB} GB (fewer layers/positions)")
    k = budget.check()
    t0 = time.time()
    reloaded = W.ensure_current() if which == "edited" else False
    W.to(which)
    out = WORK / f"fwd_{k}"
    out.mkdir(parents=True, exist_ok=True)
    files = {}

    def mm(name, shape, dt=dtype):
        arr = np.lib.format.open_memmap(out / f"{name}.npy", mode="w+", dtype=dt, shape=shape)
        files[name] = {"shape": list(shape), "dtype": np.dtype(dt).name}
        return arr

    index = np.array([(t, p) for t, ps in enumerate(pos) for p in ps], dtype=np.int32).reshape(-1, 2)
    np.save(out / "index.npy", index)
    files["index"] = {"shape": list(index.shape), "dtype": "int32"}
    bufs = {}
    for l in res_l:
        bufs[("residual", l)] = mm(f"residual_L{l}", (M, 4096))
    for l in mlp_l:
        bufs[("mlp_act", l)] = mm(f"mlp_act_L{l}", (M, 12288))
    for l in opi_l:
        bufs[("o_proj_in", l)] = mm(f"o_proj_in_L{l}", (M, 4096))
    if topk:
        tk_ids = mm("topk_ids", (M, topk), np.int32)
        tk_lp = mm("topk_logprobs", (M, topk), np.float32)
    if want_lp:
        Tmax = max(map(len, seqs))
        lp_all = mm("token_logprobs", (len(seqs), Tmax), np.float32)
        lp_all[:] = np.nan
    row0 = np.cumsum([0] + [len(p) for p in pos])

    # batches by padded token count (attention reads: one text per batch)
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    batches, cur = [], []
    for i in order:
        if cur and (att_l or (len(cur) + 1) * len(seqs[i]) > READ_TOKENS):
            batches.append(cur)
            cur = []
        cur.append(i)
    batches.append(cur)

    cap = {}
    sel = {}

    def take(x):  # x [B, T, d] -> rows at the selected (batch, position) pairs
        return x[sel["b"], sel["t"]].float().cpu().numpy()

    hooks = []
    try:
        for l in res_l:
            mod = model.model.embed_tokens if l == 0 else LAYERS[l - 1]
            hooks.append(mod.register_forward_hook(
                lambda m, a, o, l=l: cap.__setitem__(("residual", l), take(o[0] if isinstance(o, tuple) else o))))
        for l in mlp_l:
            hooks.append(LAYERS[l].mlp.down_proj.register_forward_pre_hook(
                lambda m, a, l=l: cap.__setitem__(("mlp_act", l), take(a[0]))))
        for l in opi_l:
            hooks.append(LAYERS[l].self_attn.o_proj.register_forward_pre_hook(
                lambda m, a, l=l: cap.__setitem__(("o_proj_in", l), take(a[0]))))
        for l in att_l:
            hooks.append(LAYERS[l].self_attn.register_forward_hook(
                lambda m, a, o, l=l: cap.__setitem__(("attn", l), o[1][0].to(torch.float16).cpu().numpy())))
        if att_l:
            model.set_attn_implementation("eager")
        with torch.inference_mode():
            for idx in batches:
                L = max(len(seqs[i]) for i in idx)
                inp = torch.full((len(idx), L), PAD, dtype=torch.long)
                att = torch.zeros((len(idx), L), dtype=torch.long)
                bs, ts, rows = [], [], []
                for r, i in enumerate(idx):
                    inp[r, :len(seqs[i])] = torch.tensor(seqs[i])
                    att[r, :len(seqs[i])] = 1
                    bs += [r] * len(pos[i])
                    ts += pos[i]
                    rows += list(range(row0[i], row0[i + 1]))
                sel["b"] = torch.tensor(bs, device=DEV)
                sel["t"] = torch.tensor(ts, device=DEV)
                cap.clear()
                h = model.model(input_ids=inp.to(DEV), attention_mask=att.to(DEV)).last_hidden_state
                rows = np.array(rows)
                for key, arr in cap.items():
                    if key[0] == "attn":
                        i = idx[0]
                        np.save(out / f"attn_L{key[1]}_t{i}.npy", arr)
                        files[f"attn_L{key[1]}_t{i}"] = {"shape": list(arr.shape), "dtype": "float16"}
                    else:
                        bufs[key][rows] = arr
                if topk:
                    lg = model.lm_head(h[sel["b"], sel["t"]]).float().log_softmax(-1)
                    v, ix = lg.topk(topk, dim=-1)
                    tk_ids[rows] = ix.cpu().numpy()
                    tk_lp[rows] = v.cpu().numpy()
                    del lg
                if want_lp:
                    for r, i in enumerate(idx):
                        n = len(seqs[i])
                        nxt = torch.tensor(seqs[i][1:], device=DEV)
                        for s in range(0, n - 1, 512):
                            lg = model.lm_head(h[r, s:min(n - 1, s + 512)]).float().log_softmax(-1)
                            lp_all[i, s + 1:min(n, s + 513)] = lg.gather(1, nxt[s:s + 512, None])[:, 0].cpu().numpy()
                del h
    finally:
        for hk in hooks:
            hk.remove()
        if att_l:
            model.set_attn_implementation("sdpa")
    for b in bufs.values():
        b.flush()
    meta = {"pass": k, "model": which, "edits_hash": W.manifest.get("hash") if which == "edited" else None,
            "n_texts": len(seqs), "rows": int(M), "positions": req.get("positions", "last"),
            "layout": "each <kind>_L<l>.npy is [rows, d]; index.npy gives (text, position) for every row; "
                      "residual_L0 = embedding output, residual_Ll = output of decoder layer l-1 (before the final "
                      "norm); mlp_act_Ll = input of layers[l].mlp.down_proj; o_proj_in_Ll = input of "
                      "layers[l].self_attn.o_proj (32 heads x 128); attn_Ll_t<i>.npy = [32 heads, T, T] for text i; "
                      "token_logprobs[i, t] = log p(token t | tokens < t), nan at t=0 and padding; topk_* at rows",
            "files": files, "texts": [{"source": s, "n_tokens": len(q), "ids": q} for s, q in zip(src, seqs)]}
    (out / "meta.json").write_text(json.dumps(meta))
    secs = time.time() - t0
    budget.spend(k, "read", secs)
    return {"pass": k, "dir": str(out), "seconds": round(secs, 1), "n_texts": len(seqs), "rows": int(M),
            "tokens": int(sum(map(len, seqs))), "files": files, "budget": budget.view(),
            "edits": {"hash": W.manifest.get("hash") if which == "edited" else None, "reloaded": reloaded}}


# ------------------------------------------------------------------ app
app = FastAPI()


def summarize_req(body):
    s = json.dumps(body, ensure_ascii=False)
    return body if len(s) < 100_000 else {"truncated": s[:100_000]}


def handle(kind, fn, body):
    t0 = time.time()
    status, resp = 200, None
    try:
        with LOCK:
            resp = fn(body)
        return resp
    except HTTPException as e:
        status, resp = e.status_code, {"error": e.detail}
        raise
    except Exception as e:  # noqa: BLE001
        status, resp = 500, {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-3000:]}
        torch.cuda.empty_cache()
        raise HTTPException(500, resp["error"]) from e
    finally:
        log({"kind": kind, "status": status, "s": round(time.time() - t0, 2), "request": summarize_req(body),
             "response": resp if status != 200 or kind == "reload_edits" else
             {k: v for k, v in resp.items() if k not in ("flips",)}})


@app.get("/health")
def health():
    return {"ok": True, "load_s": round(LOAD_S, 1), "state": W.state, "edits": W.manifest.get("hash"),
            "budget": budget.view(), "items": {s: len(v) for s, v in SPLIT_IDS.items()}}


@app.get("/budget")
def get_budget():
    return budget.view()


@app.post("/generate")
async def generate(request: Request):
    body = await request.json()
    return await anyio.to_thread.run_sync(handle, "generate", do_generate, body)


@app.post("/read")
async def read(request: Request):
    body = await request.json()
    return await anyio.to_thread.run_sync(handle, "read", do_read, body)


@app.post("/reload_edits")
async def reload_edits(request: Request):
    return await anyio.to_thread.run_sync(handle, "reload_edits", lambda b: W.reload(), {})


@app.exception_handler(HTTPException)
async def http_exc(request, exc):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("EDITOR_PORT", "8100")), log_level="warning")

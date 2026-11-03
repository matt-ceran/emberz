"""Check that a packaged LoRA adapter, hot-loaded into vLLM, reproduces the edited model's answers (runs ON THE VM
in serve mode).

  python verify_lora.py --adapter /opt/idea2/adapters/edit-v1 --gen /opt/idea2/work/gen_7.jsonl [--n 100]

--gen is a generation pass of the forward-pass server with "model": "both" (or "edited") on select items. The first
--n distinct select ids in it are re-asked through vLLM (greedy, thinking off, same max_tokens per dataset), once
with the base model and once with the adapter. Reported:
  adapter vs HF edited     the check: same grade (correct flag) and identical text
  vLLM base vs HF original the engine noise floor (vLLM and transformers bf16 greedy are not bit-identical)
  plus the edited-vs-original effect as seen by each engine.
Writes <adapter>/verify.json.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import time
import urllib.request
from pathlib import Path

from grader import DEFAULT_MAX_TOKENS, grade

SPLITS = Path(os.environ.get("IDEA2_SPLITS", "/opt/idea2/editor/splits"))
B = "http://127.0.0.1:8000"


def post(path, body, timeout=600):
    req = urllib.request.Request(B + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def chat(model, prompt, max_tokens):
    code, out = post("/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": prompt}],
                                              "max_tokens": max_tokens, "temperature": 0,
                                              "chat_template_kwargs": {"enable_thinking": False}})
    assert code == 200, (code, out[:500])
    return json.loads(out)["choices"][0]["message"]["content"] or ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--name", default=None)
    ap.add_argument("--concurrency", type=int, default=32)
    a = ap.parse_args()
    name = a.name or Path(a.adapter).name
    items = {}
    for s in ("dev", "select"):
        p = SPLITS / f"{s}.jsonl"
        if p.exists():
            for line in p.open():
                r = json.loads(line)
                items[r["id"]] = r
    rows = [json.loads(l) for l in open(a.gen)]
    hf = {(r["model"], r["id"], r["condition"]): r for r in rows if r.get("id")}
    ids = []
    for r in rows:
        if r.get("id") and r.get("split") == "select" and r["id"] not in ids and r["model"] == "edited":
            ids.append(r["id"])
    ids = ids[: a.n]
    keys = [(i, c) for i in ids for c in ("neutral", "wrong", "right") if ("edited", i, c) in hf]
    has_orig = all(("original", i, c) in hf for i, c in keys)
    assert keys, "no edited select rows in --gen"

    post("/v1/unload_lora_adapter", {"lora_name": name})
    t = time.time()
    code, out = post("/v1/load_lora_adapter", {"lora_name": name, "lora_path": a.adapter})
    assert code == 200, (code, out)
    load_s = time.time() - t

    def ask(job):
        model, (i, c) = job
        it = items[i]
        return job, chat(model, it["prompts"][c], DEFAULT_MAX_TOKENS[it["dataset"]])

    jobs = [(m, k) for m in (name, "qwen3-8b") for k in keys]
    t = time.time()
    with cf.ThreadPoolExecutor(a.concurrency) as ex:
        res = dict(ex.map(ask, jobs))
    gen_s = time.time() - t

    def agree(x, y):
        g = sum(x[k]["correct"] == y[k]["correct"] for k in keys) / len(keys)
        txt = sum(x[k]["text"].strip() == y[k]["text"].strip() for k in keys) / len(keys)
        return {"same_grade": round(g, 4), "same_text": round(txt, 4), "n": len(keys)}

    def graded(texts):
        return {k: {"text": texts[k], **grade(items[k[0]], texts[k])} for k in keys}

    v_ad = graded({k: res[(name, k)] for k in keys})
    v_base = graded({k: res[("qwen3-8b", k)] for k in keys})
    h_ed = {k: hf[("edited", *k)] for k in keys}
    out = {"adapter": a.adapter, "gen": a.gen, "n_items": len(ids), "n_generations": len(keys),
           "lora_load_s": round(load_s, 2), "vllm_seconds": round(gen_s, 1),
           "adapter_vs_hf_edited": agree(v_ad, h_ed), "vllm_adapter_vs_vllm_base": agree(v_ad, v_base)}
    if has_orig:
        h_or = {k: hf[("original", *k)] for k in keys}
        out["vllm_base_vs_hf_original"] = agree(v_base, h_or)
        out["hf_edited_vs_hf_original"] = agree(h_ed, h_or)
    acc = {}
    for label, src in (("vllm_adapter", v_ad), ("vllm_base", v_base), ("hf_edited", h_ed)) + (
            (("hf_original", h_or),) if has_orig else ()):
        acc[label] = {c: round(sum(src[k]["correct"] for k in keys if k[1] == c) /
                               max(1, sum(1 for k in keys if k[1] == c)), 4) for c in ("neutral", "wrong", "right")}
    out["accuracy_by_condition"] = acc
    Path(a.adapter, "verify.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

"""Smoke test for the Idea 2 vLLM server. Run ON THE VM (or on the Mac through ./tunnel.sh with --no-adapter):

    /opt/idea2/venv/bin/python smoke_test.py            # chat + LoRA hot-load/unload + throughput
    python smoke_test.py --base-url http://localhost:8000 --no-adapter   # from the Mac via the tunnel

Checks:
  1. chat completion on qwen3-8b with thinking off returns a normal answer
  2. writes a rank-4 PEFT LoRA (tiny random B on layer 0 down_proj) to /opt/idea2/adapters/smoke-r4,
     loads it with POST /v1/load_lora_adapter, queries it by name, unloads it
  3. 64 short classification prompts sent concurrently: wall time and prompts/s
"""
import argparse
import asyncio
import json
import os
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--base-url", default="http://127.0.0.1:8000")
ap.add_argument("--model-dir", default="/opt/idea2/models/Qwen3-8B")
ap.add_argument("--adapter-dir", default="/opt/idea2/adapters/smoke-r4")
ap.add_argument("--no-adapter", action="store_true")
ap.add_argument("-n", type=int, default=64)
args = ap.parse_args()
B = args.base_url.rstrip("/")
NOTHINK = {"chat_template_kwargs": {"enable_thinking": False}}


def post(path, body):
    req = urllib.request.Request(B + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def chat(model, text, max_tokens=64):
    code, out = post("/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": text}],
                                               "max_tokens": max_tokens, "temperature": 0, **NOTHINK})
    assert code == 200, (code, out)
    return json.loads(out)["choices"][0]["message"]["content"]


def models():
    with urllib.request.urlopen(B + "/v1/models", timeout=30) as r:
        return [m["id"] for m in json.load(r)["data"]]


print("models:", models())
a = chat("qwen3-8b", "What is the capital of France? Answer in one short sentence.")
print("1. chat (thinking off):", repr(a))
assert "paris" in a.lower() and "<think>" not in a, a

if not args.no_adapter:
    import torch
    from peft import LoraConfig, get_peft_model_state_dict
    from safetensors.torch import save_file

    # Build the adapter without loading the 16 GB base: PEFT naming for Qwen3 is
    # base_model.model.model.layers.{i}.mlp.down_proj.lora_{A,B}.weight; down_proj is 12288 -> 4096.
    os.makedirs(args.adapter_dir, exist_ok=True)
    r = 4
    g = torch.Generator().manual_seed(0)
    sd = {
        "base_model.model.model.layers.0.mlp.down_proj.lora_A.weight": (torch.randn(r, 12288, generator=g) * 1e-3).to(torch.bfloat16),
        "base_model.model.model.layers.0.mlp.down_proj.lora_B.weight": (torch.randn(4096, r, generator=g) * 1e-3).to(torch.bfloat16),
    }
    save_file(sd, os.path.join(args.adapter_dir, "adapter_model.safetensors"))
    cfg = LoraConfig(r=r, lora_alpha=8, target_modules=["down_proj"], layers_to_transform=[0], lora_dropout=0.0,
                     bias="none", task_type="CAUSAL_LM")
    cfg.base_model_name_or_path = args.model_dir
    cfg.save_pretrained(args.adapter_dir)
    print("2. wrote", args.adapter_dir, sorted(os.listdir(args.adapter_dir)))

    t = time.time()
    code, out = post("/v1/load_lora_adapter", {"lora_name": "smoke-r4", "lora_path": args.adapter_dir})
    print(f"   load_lora_adapter -> {code} {out.strip()!r} ({time.time() - t:.2f}s)")
    assert code == 200, out
    assert "smoke-r4" in models()
    b = chat("smoke-r4", "What is the capital of France? Answer in one short sentence.")
    print("   chat via smoke-r4:", repr(b))
    assert b.strip()
    code, out = post("/v1/unload_lora_adapter", {"lora_name": "smoke-r4"})
    print(f"   unload_lora_adapter -> {code} {out.strip()!r}")
    assert code == 200 and "smoke-r4" not in models()

# 3. throughput: 64 Banking77-style one-label classifications, all at once
LABELS = ["card_arrival", "lost_or_stolen_card", "transfer_not_received", "exchange_rate", "pin_blocked",
          "cash_withdrawal_charge", "top_up_failed", "refund_not_showing_up"]
MSGS = ["My card still hasn't arrived after two weeks.", "I think someone stole my card.",
        "I sent money to my friend yesterday and they don't have it.", "What rate do you use for EUR to GBP?",
        "My PIN got blocked after three tries.", "Why was I charged for taking cash out?",
        "My top-up keeps failing.", "The shop says they refunded me but I don't see it."]


async def one(i):
    prompt = (f"Classify the bank customer message into exactly one label from: {', '.join(LABELS)}.\n"
              f"Message: {MSGS[i % len(MSGS)]} (ticket {i})\nAnswer with the label only.")
    t = time.time()
    out = await asyncio.to_thread(chat, "qwen3-8b", prompt, 16)
    return time.time() - t, out.strip()


async def bench():
    await asyncio.gather(*[one(i) for i in range(4)])  # warm-up
    t = time.time()
    res = await asyncio.gather(*[one(i) for i in range(args.n)])
    wall = time.time() - t
    lat = sorted(x[0] for x in res)
    right = sum(o == LABELS[i % len(LABELS)] for i, (_, o) in enumerate(res))
    print(f"3. {args.n} concurrent classification prompts: wall {wall:.2f}s, {args.n / wall:.1f} prompts/s, "
          f"latency p50 {lat[len(lat) // 2]:.2f}s max {lat[-1]:.2f}s, {right}/{args.n} matched the intended label")
    print("   sample outputs:", [o for _, o in res[:8]])


import concurrent.futures as cf  # noqa: E402

loop = asyncio.new_event_loop()
loop.set_default_executor(cf.ThreadPoolExecutor(max_workers=args.n + 8))
loop.run_until_complete(bench())
print("OK")

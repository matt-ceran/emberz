"""OpenAI-compatible chat endpoint that serves the fine-tuned guardian through the Fireworks
serverless Training API sampler, i.e. exactly the checkpoint we evaluated.

Why: the dedicated deployment (emberz-v2b) applies the LoRA only partially (same raw prompt,
different answers; rule-flip pairs 2/10 vs 8/10). The sampler applies it fully.

Run (from callguard/):
  GUARDIAN_API_KEY=<Matt's Fireworks key> \
  uv run --with fastapi --with 'uvicorn[standard]' --with 'fireworks-ai[training]>=1.2.29' --with transformers \
      uvicorn guardian.sampler_proxy:app --host 127.0.0.1 --port 8077

Then point the line at it:  GUARDIAN_URL=http://127.0.0.1:8077/inference/v1/chat/completions
Env:
  GUARDIAN_API_KEY (or FIREWORKS_API_KEY)  Matt's key (the checkpoint lives in account afceran33-zaqel7a5ap)
  SAMPLER_CHECKPOINT   sampler path (default: guardian v2 final)
  SAMPLER_HF_TOKENIZER HF tokenizer id (default Qwen/Qwen3.6-35B-A3B)
  PROXY_TOKEN          optional; if set, requests need "Authorization: Bearer <PROXY_TOKEN>"
Request fields used: messages, max_tokens, temperature. response_format / reasoning_effort are accepted and
ignored: thinking is always off and the model returns plain JSON, as in training and evaluation.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
import time
import uuid

from fastapi import FastAPI, Header, HTTPException, Request

log = logging.getLogger("sampler_proxy")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

CHECKPOINT = os.environ.get("SAMPLER_CHECKPOINT",
                            "afceran33-zaqel7a5ap/run-d0f99835415b46f5a898cafaa544e005/v2-fin-7ecb9b48")
HF_TOKENIZER = os.environ.get("SAMPLER_HF_TOKENIZER", "Qwen/Qwen3.6-35B-A3B")
BASE_URL = "https://api.fireworks.ai/training/v1/serverless"

app = FastAPI(title="emberz guardian sampler proxy")
_state: dict = {}
_lock = threading.Lock()
_THINK = re.compile(r"^.*?</think>", re.S)


def _key() -> str:
    k = os.environ.get("GUARDIAN_API_KEY") or os.environ.get("FIREWORKS_API_KEY")
    if not k:
        raise RuntimeError("set GUARDIAN_API_KEY (Matt's Fireworks key)")
    return k


def _client():
    with _lock:
        if "sampler" not in _state:
            from fireworks.training.sdk import FiretitanServiceClient
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(HF_TOKENIZER)
            svc = FiretitanServiceClient(api_key=_key(), base_url=BASE_URL)
            _state["tok"] = tok
            _state["sampler"] = svc.create_sampling_client(model_path=CHECKPOINT, tokenizer=tok)
            log.info("sampler ready for %s", CHECKPOINT)
    return _state["tok"], _state["sampler"]


def _generate(messages: list[dict], max_tokens: int, temperature: float) -> tuple[str, int, int]:
    from fireworks.training.sdk import types
    tok, sampler = _client()
    ids = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, enable_thinking=False)
    if hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    ids = list(ids)
    res = sampler.sample(prompt=types.ModelInput.from_ints(ids), num_samples=1,
                         sampling_params=types.SamplingParams(max_tokens=max_tokens, temperature=temperature)).result()
    out_ids = list(res.sequences[0].tokens)
    text = tok.decode(out_ids).replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
    if "</think>" in text:
        text = _THINK.sub("", text, count=1).strip()
    return text, len(ids), len(out_ids)


@app.on_event("startup")
async def _warm() -> None:
    # Pay the ~6 s client setup + first sample at startup, not on the first live turn.
    def warm():
        try:
            t = time.time()
            _generate([{"role": "user", "content": "Reply with {}"}], 8, 0.0)
            log.info("warm-up done in %.1fs", time.time() - t)
        except Exception as e:  # noqa: BLE001
            log.warning("warm-up failed (will retry on first request): %s", e)
    asyncio.get_running_loop().run_in_executor(None, warm)


@app.get("/health")
async def health():
    return {"ok": True, "ready": "sampler" in _state, "checkpoint": CHECKPOINT}


@app.post("/inference/v1/chat/completions")
@app.post("/v1/chat/completions")
async def chat(request: Request, authorization: str | None = Header(default=None)):
    want = os.environ.get("PROXY_TOKEN")
    if want and authorization not in (f"Bearer {want}",):
        raise HTTPException(401, "bad proxy token")
    body = await request.json()
    messages = body.get("messages") or []
    if not messages:
        raise HTTPException(400, "messages required")
    max_tokens = int(body.get("max_tokens") or 400)
    temperature = float(body.get("temperature") or 0.0)
    t = time.time()
    try:
        text, n_in, n_out = await asyncio.to_thread(_generate, messages, max_tokens, temperature)
    except Exception as e:  # noqa: BLE001
        log.exception("sample failed")
        raise HTTPException(503, f"sampler error: {e}"[:300])
    ms = int((time.time() - t) * 1000)
    log.info("turn %d ms, %d in / %d out tokens", ms, n_in, n_out)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}", "object": "chat.completion", "created": int(time.time()),
        "model": body.get("model") or CHECKPOINT,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": n_in, "completion_tokens": n_out, "total_tokens": n_in + n_out},
        "proxy_ms": ms,
    }

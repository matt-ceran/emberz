"""Minimal Fireworks chat client (stdlib only) with retries and usage accounting."""
import json, os, random, threading, time, urllib.error, urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
URL = "https://api.fireworks.ai/inference/v1/chat/completions"

MODELS = {
    "deepseek": "accounts/fireworks/models/deepseek-v4p1-flash",
    "kimi": "accounts/fireworks/models/kimi-k3",
    "nemotron": "accounts/fireworks/models/nemotron-3-ultra-nvfp4",
    "glm": "accounts/fireworks/models/glm-5p3-flash",
}


def load_env():
    env = REPO / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


load_env()


class Usage:
    def __init__(self):
        self.lock = threading.Lock()
        self.prompt = self.completion = self.calls = self.errors = self.retries = 0
        self.codes = {}

    def add(self, u, err=False, retries=0):
        with self.lock:
            self.calls += 1
            self.errors += int(err)
            self.retries += retries
            self.prompt += (u or {}).get("prompt_tokens", 0) or 0
            self.completion += (u or {}).get("completion_tokens", 0) or 0

    def code(self, c):
        with self.lock:
            self.codes[c] = self.codes.get(c, 0) + 1

    def __str__(self):
        return (f"calls={self.calls} errors={self.errors} retries={self.retries} codes={self.codes} "
                f"prompt_tok={self.prompt} completion_tok={self.completion}")


USAGE = Usage()
_no_reasoning_param = set()


def chat(model, messages, key_env="FIREWORKS_API_KEY", max_tokens=4000, temperature=0.9, json_mode=True,
         timeout=240, max_retries=6):
    """Returns (content, usage, finish_reason). Raises RuntimeError after retries."""
    model = MODELS.get(model, model)
    key = os.environ[key_env]
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    if model not in _no_reasoning_param:
        body["reasoning_effort"] = "none"
    retries = 0
    last = None
    while retries <= max_retries:
        req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.load(r)
            ch = d["choices"][0]
            USAGE.add(d.get("usage"), retries=retries)
            return ch["message"].get("content") or "", d.get("usage", {}), ch.get("finish_reason")
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:400]
            last = f"HTTP {e.code}: {msg}"
            USAGE.code(e.code)
            if e.code == 400 and "reasoning" in msg.lower() and "reasoning_effort" in body:
                body.pop("reasoning_effort")
                _no_reasoning_param.add(model)
                continue
            if e.code in (429, 500, 502, 503, 504, 529):
                retries += 1
                time.sleep(min(60, (2 ** retries) + random.random() * 2))
                continue
            break
        except Exception as e:  # timeouts, connection resets
            last = repr(e)[:300]
            USAGE.code(type(e).__name__)
            retries += 1
            time.sleep(min(30, 2 ** retries))
    USAGE.add(None, err=True, retries=retries)
    raise RuntimeError(last)


def parse_json(text):
    text = text or ""
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b <= a:
        raise ValueError("no JSON object")
    return json.loads(text[a:b + 1])

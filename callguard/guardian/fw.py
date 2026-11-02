"""Tiny stdlib Fireworks REST client (control plane + inference).

Loads FIREWORKS_API_KEY from the environment or the repo-root .env.
Never prints the key.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

API = "https://api.fireworks.ai"
REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_MODEL = "accounts/fireworks/models/qwen3-4b-instruct-2507"
# Training + deployments bill to Matt's account by default; override with --key-env / FW_KEY_ENV.
KEY_ENV = os.environ.get("FW_KEY_ENV", "FIREWORKS_API_KEY_MATT")
KNOWN_ACCOUNTS = {"FIREWORKS_API_KEY_MATT": "afceran33-zaqel7a5ap",
                  "FIREWORKS_API_KEY": "numan-vf90m1xhbxgc"}


def set_key_env(name: str):
    """Switch which env var (from env or repo .env) supplies the key."""
    global KEY_ENV, _ACCOUNT
    KEY_ENV, _ACCOUNT = name, None


class FWError(RuntimeError):
    def __init__(self, status: int, body: str, url: str):
        super().__init__(f"HTTP {status} {url}: {body[:2000]}")
        self.status, self.body, self.url = status, body, url


def api_key() -> str:
    k = os.environ.get(KEY_ENV)
    if k:
        return k
    env = REPO_ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line.startswith("export "):
                line = line[7:]
            if line.startswith(f"{KEY_ENV}="):
                k = line.split("=", 1)[1].strip().strip('"').strip("'")
                os.environ[KEY_ENV] = k
                return k
    raise SystemExit(f"{KEY_ENV} not set and not found in repo .env")


def request(method: str, path: str, body=None, *, raw: bytes | None = None,
            headers: dict | None = None, timeout: float = 120, retries: int = 3):
    """JSON request against api.fireworks.ai. `path` may be absolute URL or start with /."""
    url = path if path.startswith("http") else API + path
    h = {"Authorization": f"Bearer {api_key()}", "Accept": "application/json"}
    data = raw
    if body is not None:
        data = json.dumps(body).encode()
        h["Content-Type"] = "application/json"
    h.update(headers or {})
    for attempt in range(retries):
        req = urllib.request.Request(url, data=data, method=method, headers=h)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                txt = r.read().decode() or "{}"
                try:
                    return json.loads(txt)
                except json.JSONDecodeError:
                    return {"_raw": txt}
        except urllib.error.HTTPError as e:
            txt = e.read().decode(errors="replace")
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            raise FWError(e.code, txt, url) from None
        except urllib.error.URLError:
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            raise


def get(path, **kw):
    return request("GET", path, **kw)


def post(path, body=None, **kw):
    return request("POST", path, body if body is not None else {}, **kw)


def patch(path, body, **kw):
    return request("PATCH", path, body, **kw)


_ACCOUNT: str | None = None


def account_id() -> str:
    """Account id for the key (FIREWORKS_ACCOUNT_ID env overrides)."""
    global _ACCOUNT
    if _ACCOUNT:
        return _ACCOUNT
    acct = os.environ.get("FIREWORKS_ACCOUNT_ID")
    if not acct:
        try:
            r = get("/v1/accounts?pageSize=10")
            accts = r.get("accounts", [])
            if accts:
                acct = accts[0]["name"].split("/")[-1]
        except FWError:
            pass
    _ACCOUNT = acct or KNOWN_ACCOUNTS.get(KEY_ENV)
    if not _ACCOUNT:
        raise SystemExit(f"could not determine account for {KEY_ENV}; set FIREWORKS_ACCOUNT_ID")
    return _ACCOUNT


def acct_path(suffix: str = "") -> str:
    return f"/v1/accounts/{account_id()}{suffix}"


def upload_multipart(path: str, file_path: str | Path, field: str = "file", timeout: float = 600):
    """POST a single file as multipart/form-data (stdlib)."""
    p = Path(file_path)
    boundary = "----fw" + uuid.uuid4().hex
    head = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{field}\"; "
            f"filename=\"{p.name}\"\r\nContent-Type: application/octet-stream\r\n\r\n").encode()
    tail = f"\r\n--{boundary}--\r\n".encode()
    raw = head + p.read_bytes() + tail
    return request("POST", path, raw=raw, timeout=timeout,
                   headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})


def chat(model: str, messages: list, **kw):
    body = {"model": model, "messages": messages, **kw}
    return post("/inference/v1/chat/completions", body)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        set_key_env(sys.argv[1])
    print(f"key env: {KEY_ENV}  account: {account_id()}")

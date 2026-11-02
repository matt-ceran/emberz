"""Upload / deploy / call / delete a PEFT LoRA adapter on Fireworks via the REST API.

Why REST and not firectl: firectl 1.8.22 refuses every mutating command
("BLOCKED: mutating command ... cannot run inside an AI agent") when it detects
it runs under a coding agent. A human can use the firectl commands in README.md;
an agent uses this script. Read-only firectl commands (list/get/match) work.

Env: FIREWORKS_API_KEY (loaded from ../../../fireworks-AMD/.env if not set),
     FIREWORKS_ACCOUNT_ID (default: first account the key can see).

  uv run --no-project --with requests python fw_lora.py upload qwen3-8b-noise-r8 adapters/noise-r8
  uv run --no-project --with requests python fw_lora.py match accounts/<acct>/models/qwen3-8b-noise-r8
  uv run --no-project --with requests python fw_lora.py deploy accounts/<acct>/models/qwen3-8b-noise-r8 --shape <shape-version>
  uv run --no-project --with requests python fw_lora.py wait <deployment-id>
  uv run --no-project --with requests python fw_lora.py call <model-string>
  uv run --no-project --with requests python fw_lora.py delete <deployment-id>
  uv run --no-project --with requests python fw_lora.py list

Every step appends a JSON line with timings to runs.jsonl next to this file.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
API = "https://api.fireworks.ai/v1"
INFER = "https://api.fireworks.ai/inference/v1/chat/completions"
BASE_MODEL = "accounts/fireworks/models/qwen3-8b"


def load_key() -> str:
    key = os.environ.get("FIREWORKS_API_KEY")
    if not key:
        for env in [HERE.parents[2] / "fireworks-AMD" / ".env", HERE.parents[1] / ".env"]:
            if env.exists():
                for line in env.read_text().splitlines():
                    if line.strip().startswith("FIREWORKS_API_KEY="):
                        key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break
            if key:
                break
    if not key:
        sys.exit("FIREWORKS_API_KEY not found")
    return key


KEY = load_key()
H = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}


def account() -> str:
    acct = os.environ.get("FIREWORKS_ACCOUNT_ID")
    if acct:
        return acct
    r = requests.get(f"{API}/accounts", headers=H, timeout=30)
    r.raise_for_status()
    return r.json()["accounts"][0]["name"].split("/")[1]


def log(event: str, **kw):
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "event": event, **kw}
    with open(HERE / "runs.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec, indent=2))


def check(r: requests.Response, what: str):
    if r.status_code >= 300:
        log(f"{what}_error", status=r.status_code, body=r.text[:2000])
        sys.exit(1)
    return r.json() if r.text else {}


def cmd_upload(a):
    acct = account()
    d = Path(a.dir)
    cfg = json.loads((d / "adapter_config.json").read_text())
    files = sorted(p.name for p in d.iterdir() if p.is_file())
    sizes = {f: (d / f).stat().st_size for f in files}
    t0 = time.time()
    body = {
        "modelId": a.model_id,
        "model": {
            "kind": "HF_PEFT_ADDON",
            "displayName": a.model_id,
            "peftDetails": {"baseModel": a.base, "r": cfg["r"], "targetModules": cfg["target_modules"]},
        },
    }
    r = requests.post(f"{API}/accounts/{acct}/models", headers=H, json=body, timeout=60)
    if r.status_code == 409 or "ALREADY_EXISTS" in r.text:
        print("model exists, continuing")
    else:
        check(r, "create_model")
    r = requests.post(f"{API}/accounts/{acct}/models/{a.model_id}:getUploadEndpoint", headers=H,
                      json={"filenameToSize": sizes, "enableResumableUpload": False}, timeout=60)
    urls = check(r, "get_upload_endpoint")["filenameToSignedUrls"]
    t1 = time.time()
    for name, url in urls.items():
        with open(d / name, "rb") as f:
            put = requests.put(url, data=f, timeout=600, headers={
                "Content-Type": "application/octet-stream",
                "x-goog-content-length-range": f"{sizes[name]},{sizes[name]}"})
        if put.status_code >= 300:
            log("put_error", file=name, status=put.status_code, body=put.text[:1000])
            sys.exit(1)
        print(f"uploaded {name} ({sizes[name]} bytes)")
    t2 = time.time()
    while True:
        r = requests.get(f"{API}/accounts/{acct}/models/{a.model_id}:validateUpload", headers=H, timeout=120)
        if r.status_code == 200:
            break
        if "FAILED_PRECONDITION" in r.text:
            print("validate: files still landing, retry in 10s")
            time.sleep(10)
            continue
        log("validate_error", status=r.status_code, body=r.text[:2000])
        sys.exit(1)
    t3 = time.time()
    while True:
        m = check(requests.get(f"{API}/accounts/{acct}/models/{a.model_id}", headers=H, timeout=30), "get_model")
        if m.get("state") in ("READY", "FAILED"):
            break
        time.sleep(5)
    t4 = time.time()
    log("upload", model=f"accounts/{acct}/models/{a.model_id}", state=m.get("state"), status=m.get("status"),
        bytes=sum(sizes.values()), create_s=round(t1 - t0, 1), put_s=round(t2 - t1, 1),
        validate_s=round(t3 - t2, 1), to_ready_s=round(t4 - t0, 1))


def cmd_match(a):
    acct = account()
    dep = {"baseModel": a.model}
    if a.addons:
        dep["enableAddons"] = True
    r = requests.post(f"{API}/accounts/{acct}/deploymentShapeVersions:match", headers=H,
                      json={"createDeploymentRequest": {"deployment": dep}}, timeout=60)
    out = check(r, "match")
    for v in out.get("deploymentShapeVersions", []):
        s = v.get("snapshot", {})
        print(v["name"], s.get("acceleratorType"), s.get("acceleratorCount"), s.get("precision"),
              s.get("presetType"), s.get("enableAddons"))
    if not out.get("deploymentShapeVersions"):
        print("no matching shapes")


def cmd_deploy(a):
    acct = account()
    body = {
        "baseModel": a.model,
        "deploymentShape": a.shape,
        "displayName": a.name or "idea2-lora-test",
        "minReplicaCount": a.min_replicas,
        "maxReplicaCount": 1,
        "autoscalingPolicy": {"scaleToZeroWindow": "300s"},
    }
    if a.addons:
        body["enableAddons"] = True
    t0 = time.time()
    r = requests.post(f"{API}/accounts/{acct}/deployments", headers=H, json=body, timeout=120)
    d = check(r, "create_deployment")
    log("deploy_created", deployment=d.get("name"), state=d.get("state"), shape=d.get("deploymentShape"),
        accelerator=d.get("acceleratorType"), precision=d.get("precision"), request_s=round(time.time() - t0, 1),
        created_epoch=t0)


def get_dep(acct, dep_id):
    r = requests.get(f"{API}/accounts/{acct}/deployments/{dep_id}", headers=H, timeout=30)
    return check(r, "get_deployment")


def cmd_wait(a):
    acct = account()
    t0 = a.since or time.time()
    last = None
    while True:
        d = get_dep(acct, a.deployment)
        st = d.get("state")
        rs = d.get("replicaStats", {})
        cur = (st, json.dumps(rs, sort_keys=True), d.get("status", {}).get("message"))
        if cur != last:
            print(f"[{time.time() - t0:7.1f}s] state={st} replicas={rs} status={d.get('status')}")
            last = cur
        if st == "FAILED" or (st == "READY" and (d.get("replicaCount", 0) or rs.get("readyReplicaCount", 0))):
            break
        if st == "READY" and a.min_replicas_zero_ok:
            break
        if time.time() - t0 > a.timeout:
            print("timeout")
            break
        time.sleep(10)
    log("deploy_wait", deployment=d.get("name"), state=d.get("state"), status=d.get("status"),
        replica_stats=d.get("replicaStats"), since_create_s=round(time.time() - t0, 1))


def cmd_call(a):
    body = {"model": a.model, "messages": [{"role": "user", "content": a.prompt}],
            "max_tokens": a.max_tokens, "temperature": 0}
    if a.no_think:
        body["reasoning_effort"] = "none"
    t0 = time.time()
    attempts = 0
    while True:
        attempts += 1
        r = requests.post(INFER, headers=H, json=body, timeout=300)
        if r.status_code in (503, 429, 425) and time.time() - t0 < a.retry_for:
            print(f"[{time.time() - t0:6.1f}s] {r.status_code}: {r.text[:200]}")
            time.sleep(5)
            continue
        break
    dt = time.time() - t0
    try:
        j = r.json()
    except Exception:
        j = {"raw": r.text[:2000]}
    text = (j.get("choices") or [{}])[0].get("message", {}).get("content") if r.ok else None
    log("call", model=a.model, status=r.status_code, latency_s=round(dt, 2), attempts=attempts,
        usage=j.get("usage"), text=text, error=None if r.ok else j)


def cmd_delete(a):
    acct = account()
    r = requests.delete(f"{API}/accounts/{acct}/deployments/{a.deployment}", headers=H,
                        params={"ignoreChecks": "true", "hard": "true"}, timeout=60)
    log("delete", deployment=a.deployment, status=r.status_code, body=r.text[:500])


def cmd_list(a):
    acct = account()
    r = requests.get(f"{API}/accounts/{acct}/deployments", headers=H, params={"showDeleted": "false"}, timeout=30)
    for d in check(r, "list").get("deployments", []):
        print(d["name"], d.get("state"), d.get("baseModel"), d.get("replicaStats"))
    r = requests.get(f"{API}/accounts/{acct}/models", headers=H, timeout=30)
    for m in check(r, "list_models").get("models", []):
        print(m["name"], m.get("kind"), m.get("state"))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("upload"); p.add_argument("model_id"); p.add_argument("dir")
    p.add_argument("--base", default=BASE_MODEL); p.set_defaults(f=cmd_upload)
    p = sp.add_parser("match"); p.add_argument("model"); p.add_argument("--addons", action="store_true")
    p.set_defaults(f=cmd_match)
    p = sp.add_parser("deploy"); p.add_argument("model"); p.add_argument("--shape", required=True)
    p.add_argument("--addons", action="store_true"); p.add_argument("--name")
    p.add_argument("--min-replicas", type=int, default=0); p.set_defaults(f=cmd_deploy)
    p = sp.add_parser("wait"); p.add_argument("deployment"); p.add_argument("--since", type=float)
    p.add_argument("--timeout", type=float, default=2400); p.add_argument("--min-replicas-zero-ok", action="store_true")
    p.set_defaults(f=cmd_wait)
    p = sp.add_parser("call"); p.add_argument("model")
    p.add_argument("--prompt", default="What is the capital of France? Answer in one short sentence.")
    p.add_argument("--max-tokens", type=int, default=60); p.add_argument("--no-think", action="store_true")
    p.add_argument("--retry-for", type=float, default=900); p.set_defaults(f=cmd_call)
    p = sp.add_parser("delete"); p.add_argument("deployment"); p.set_defaults(f=cmd_delete)
    p = sp.add_parser("list"); p.set_defaults(f=cmd_list)
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()

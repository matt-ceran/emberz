#!/usr/bin/env python3
"""Fireworks LoRA SFT pipeline for the CallGuard guardian (stdlib only).

  python3 sft.py upload data.jsonl --name cg-train-v1
  python3 sft.py train --dataset cg-train-v1 --name cg-sft-v1 --epochs 2 --lora-rank 16
  python3 sft.py status cg-sft-v1
  python3 sft.py wait cg-sft-v1
  python3 sft.py deploy-plan --model cg-sft-v1      # prints commands only, creates nothing
  python3 sft.py make-probe out.jsonl               # tiny synthetic probe dataset
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fw  # noqa: E402

TERMINAL = {"JOB_STATE_COMPLETED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED",
            "JOB_STATE_EXPIRED", "JOB_STATE_DELETING"}


def now():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def _full(kind: str, ident: str) -> str:
    """'x' -> accounts/<acct>/<kind>/x ; full names pass through."""
    return ident if ident.startswith("accounts/") else f"accounts/{fw.account_id()}/{kind}/{ident}"


def _short(name: str) -> str:
    return name.split("/")[-1]


def _msg(e: fw.FWError) -> str:
    try:
        return json.loads(e.body).get("message", e.body)[:300]
    except Exception:
        return e.body[:300]


def upload_signed(ds_id: str, path: Path) -> dict:
    """Large-file path: getUploadEndpoint -> PUT signed URL -> validateUpload."""
    import urllib.request
    size = path.stat().st_size
    r = fw.post(fw.acct_path(f"/datasets/{ds_id}:getUploadEndpoint"),
                {"filenameToSize": {path.name: str(size)}})
    url = r["filenameToSignedUrls"][path.name]
    req = urllib.request.Request(url, data=path.read_bytes(), method="PUT",
                                 headers={"Content-Type": "application/octet-stream",
                                          "x-goog-content-length-range": f"{size},{size}"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        print(f"[{now()}] PUT signed url -> {resp.status}")
    return fw.post(fw.acct_path(f"/datasets/{ds_id}:validateUpload"), {})


# ---------------------------------------------------------------- datasets
def validate_jsonl(path: Path) -> int:
    n = 0
    with path.open() as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            msgs = row.get("messages")
            assert isinstance(msgs, list) and msgs, f"line {i}: missing messages"
            for m in msgs:
                assert m.get("role") in ("system", "user", "assistant", "tool"), f"line {i}: bad role"
            assert msgs[-1]["role"] == "assistant", f"line {i}: last message must be assistant"
            n += 1
    assert n >= 3, "Fireworks needs >= 3 examples"
    return n


def cmd_upload(a):
    path = Path(a.jsonl)
    n = validate_jsonl(path)
    size = path.stat().st_size
    print(f"[{now()}] {path} : {n} examples, {size/1e6:.2f} MB")
    if size > 150e6:
        sys.exit("file >150MB: the single-request :upload path won't take it (use getUploadEndpoint)")
    ds_path = fw.acct_path(f"/datasets/{a.name}")
    try:
        existing = fw.get(ds_path)
        print(f"dataset exists: state={existing.get('state')} examples={existing.get('exampleCount')}")
        if existing.get("state") == "READY" and not a.force:
            print(_full("datasets", a.name))
            return
    except fw.FWError as e:
        if e.status != 404:
            raise
        body = {"datasetId": a.name,
                "dataset": {"displayName": a.name, "exampleCount": str(n),
                            "format": "CHAT", "userUploaded": {}}}
        r = fw.post(fw.acct_path("/datasets"), body)
        print(f"[{now()}] created {r.get('name')} state={r.get('state')}")
    t0 = time.time()
    try:
        r = fw.upload_multipart(fw.acct_path(f"/datasets/{a.name}:upload"), path)
    except fw.FWError as e:
        print(f"[{now()}] :upload failed ({e.status}): {_msg(e)} -> trying signed-URL path")
        r = upload_signed(a.name, path)
    print(f"[{now()}] uploaded in {time.time()-t0:.1f}s -> {json.dumps(r)[:300]}")
    # Poll until READY.
    deadline = time.time() + a.timeout
    while True:
        d = fw.get(ds_path)
        st = d.get("state")
        if st == "READY":
            print(f"[{now()}] READY examples={d.get('exampleCount')} "
                  f"estTokens={d.get('estimatedTokenCount')} avgTurns={d.get('averageTurnCount')}")
            print(d["name"])
            return
        if time.time() > deadline:
            sys.exit(f"timeout waiting for READY (state={st}, status={d.get('status')})")
        if st == "UPLOADING":
            # Some flows need an explicit validate; harmless if not needed.
            try:
                fw.post(fw.acct_path(f"/datasets/{a.name}:validateUpload"), {})
            except fw.FWError as e:
                if e.status not in (400, 409, 412):
                    print("validateUpload:", e)
        print(f"[{now()}] state={st} status={d.get('status')}")
        time.sleep(3)


# ---------------------------------------------------------------- training
def cmd_train(a):
    if not a.lora_rank or a.lora_rank <= 0:
        sys.exit("refusing: lora rank must be > 0 (0/omitted = full-parameter training at 2x price)")
    job = {
        "displayName": a.name,
        "baseModel": a.base_model,
        "dataset": _full("datasets", a.dataset),
        "outputModel": _full("models", a.output_model or a.name),
        "epochs": a.epochs,
        "loraRank": a.lora_rank,
        "maxContextLength": a.max_context,
    }
    if a.lr is not None:
        job["learningRate"] = a.lr
    if a.batch_size_samples:
        job["batchSizeSamples"] = a.batch_size_samples
    if a.warmup_steps is not None:
        job["learningRateWarmupSteps"] = a.warmup_steps
    if a.eval_dataset:
        job["evaluationDataset"] = _full("datasets", a.eval_dataset)
    print("request:", json.dumps(job, indent=1))
    if a.dry_run:
        return
    r = fw.post(fw.acct_path(f"/supervisedFineTuningJobs?supervisedFineTuningJobId={a.name}"), job)
    print(f"[{now()}] created {r.get('name')} state={r.get('state')} loraRank={r.get('loraRank')}")
    if r.get("loraRank") in (0, None):
        print("WARNING: server reports loraRank=0 -> this would be FULL-PARAMETER training. Cancel it!")
    print(r.get("name"))
    log_event(a.name, "created", r)


def get_job(job_id: str) -> dict:
    return fw.get(fw.acct_path(f"/supervisedFineTuningJobs/{_short(job_id)}"))


def _ts(s: str | None):
    if not s:
        return None
    s = s.rstrip("Z")
    if "." in s:
        head, frac = s.split(".", 1)
        s = f"{head}.{frac[:6]}"
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def summarize(j: dict) -> str:
    created = _ts(j.get("createTime"))
    done = _ts(j.get("completedTime"))
    end = done or datetime.now(timezone.utc)
    el = (end - created).total_seconds() if created else 0
    p = j.get("jobProgress") or {}
    cost = j.get("estimatedCost") or {}
    cost_s = ""
    if cost:
        cost_s = f" estCost=${int(cost.get('units', 0) or 0) + (cost.get('nanos', 0) or 0)/1e9:.4f}"
    parts = [f"state={j.get('state')}", f"elapsed={el/60:.1f}m",
             f"progress={p.get('percent', 0)}% epoch={p.get('epoch', '-')}"]
    if p.get("inputTokens") or p.get("outputTokens"):
        parts.append(f"tokens in/out={p.get('inputTokens')}/{p.get('outputTokens')}")
    st = j.get("status") or {}
    if st.get("message"):
        parts.append(f"status={st.get('code')}:{st.get('message')[:300]}")
    s = " ".join(parts) + cost_s
    if j.get("state") == "JOB_STATE_COMPLETED":
        s += f"\n  output model: {j.get('outputModel')}"
    return s


def cmd_status(a):
    j = get_job(a.job_id)
    if a.json:
        print(json.dumps(j, indent=1))
    else:
        print(f"[{now()}] {j.get('name')}\n  {summarize(j)}")


def log_event(job_id: str, ev: str, j: dict):
    d = Path(__file__).resolve().parents[1] / "data" / "sft_runs"
    d.mkdir(parents=True, exist_ok=True)
    with (d / f"{_short(job_id)}.log.jsonl").open("a") as f:
        f.write(json.dumps({"t": datetime.now(timezone.utc).isoformat(), "event": ev,
                            "state": j.get("state"), "progress": j.get("jobProgress"),
                            "estimatedCost": j.get("estimatedCost"),
                            "status": j.get("status")}) + "\n")


def cmd_wait(a):
    t0 = time.time()
    last = None
    first_seen = {}
    while True:
        try:
            j = get_job(a.job_id)
        except Exception as e:  # transient network
            print(f"[{now()}] poll error: {e}")
            time.sleep(a.interval)
            continue
        st = j.get("state")
        if st not in first_seen:
            first_seen[st] = time.time()
            log_event(a.job_id, f"state:{st}", j)
        line = summarize(j)
        if line != last:
            print(f"[{now()}] {line}", flush=True)
            last = line
        if st in TERMINAL:
            log_event(a.job_id, "terminal", j)
            print("state first-seen (s since wait start):",
                  {k: round(v - t0) for k, v in first_seen.items()})
            if st != "JOB_STATE_COMPLETED":
                sys.exit(1)
            print(j.get("outputModel"))
            return
        if a.timeout and time.time() - t0 > a.timeout:
            print(f"[{now()}] wait timeout after {a.timeout}s; job still {st}")
            sys.exit(2)
        time.sleep(a.interval)


# ---------------------------------------------------------------- deploy plan (print only)
def cmd_deploy_plan(a):
    acct = fw.account_id()
    model = _full("models", a.model)
    dep = a.deployment_id
    base = fw.BASE_MODEL
    B = f"https://api.fireworks.ai/v1/accounts/{acct}"
    H = '-H "Authorization: Bearer $FIREWORKS_API_KEY" -H "Content-Type: application/json"'
    text = (f"""# NOTHING BELOW HAS BEEN RUN. Deployments bill GPU time (~$8/h H200) while replicas > 0.
# Load key first:  set -a; . ./.env; set +a

## Option A (recommended for ONE adapter): live-merge deployment of the LoRA model itself
# A1. find shapes that can live-merge it
curl -s -X POST "{B}/deploymentShapeVersions:match" {H} \\
  -d '{{"createDeploymentRequest":{{"deployment":{{"baseModel":"{model}"}}}}}}'
# A2. create (pick a shape version name from A1, or omit deploymentShape to let FW choose "default")
curl -s -X POST "{B}/deployments?deploymentId={dep}-merged" {H} -d '{{
  "displayName": "{dep}-merged",
  "baseModel": "{model}",
  "deploymentShape": "<SHAPE_VERSION_FROM_A1, or omit>",
  "minReplicaCount": 0, "maxReplicaCount": 1,
  "autoscalingPolicy": {{"scaleToZeroWindow": "300s"}}
}}'
# query it with model="{model}"

## B3 baseline (untuned base): verified FP8 H200 shape (addons not possible on FP8)
curl -s -X POST "{B}/deployments?deploymentId={dep}-base" {H} -d '{{
  "displayName": "{dep}-base", "baseModel": "{base}",
  "deploymentShape": "{SHAPE_FP8_H200}",
  "minReplicaCount": 0, "maxReplicaCount": 1,
  "autoscalingPolicy": {{"scaleToZeroWindow": "300s"}}
}}'

## Option B: multi-LoRA (base + addons; lets B3 base and v1 share one GPU)
# B1. VERIFIED 2026-11-02: match with enableAddons=true returns NO shapes for this base.
#     Base shapes: FP8 H200 {SHAPE_FP8_H200}
#                  BF16 B200 {SHAPE_BF16_B200} (hotload, addons=false)
#     So multi-LoRA needs a shapeless BF16 deployment with skipShapeValidation=true (unvalidated; may fail).
# B2. create the base deployment with addons, min 0 replicas, 5-min scale-to-zero
#     (dry-run first: add &validateOnly=true, or run `python3 sft.py deploy-check`)
curl -s -X POST "{B}/deployments?deploymentId={dep}&skipShapeValidation=true" {H} -d '{{
  "displayName": "{dep}",
  "baseModel": "{base}",
  "precision": "BF16", "acceleratorType": "NVIDIA_H200_141GB", "acceleratorCount": 1,
  "enableAddons": true,
  "minReplicaCount": 0, "maxReplicaCount": 1,
  "autoscalingPolicy": {{"scaleToZeroWindow": "300s"}}
}}'
# B3. poll until state == READY
curl -s "{B}/deployments/{dep}" -H "Authorization: Bearer $FIREWORKS_API_KEY"
# B4. load the LoRA onto it (deployedModels = "Load LoRA")
curl -s -X POST "{B}/deployedModels" {H} -d '{{
  "model": "{model}",
  "deployment": "accounts/{acct}/deployments/{dep}",
  "displayName": "{_short(model)}"
}}'
# B5. inference: base  -> model="{base}#accounts/{acct}/deployments/{dep}"
#                LoRA  -> model="{model}#accounts/{acct}/deployments/{dep}"
curl -s https://api.fireworks.ai/inference/v1/chat/completions {H} -d '{{
  "model": "{model}#accounts/{acct}/deployments/{dep}",
  "messages": [{{"role":"user","content":"ping"}}], "max_tokens": 64, "temperature": 0
}}'

## Teardown (do this! no spend cap on this account)
curl -s -X DELETE "{B}/deployments/{dep}" -H "Authorization: Bearer $FIREWORKS_API_KEY"
curl -s -X DELETE "{B}/deployments/{dep}-merged" -H "Authorization: Bearer $FIREWORKS_API_KEY"
curl -s -X DELETE "{B}/deployments/{dep}-base" -H "Authorization: Bearer $FIREWORKS_API_KEY"

# firectl equivalents (if installed):
#   firectl deployment create {base} --deployment-shape <bf16-shape> --enable-addons \\
#       --min-replica-count 0 --max-replica-count 1 --scale-to-zero-window 5m --deployment-id {dep}
#   firectl load-lora {_short(model)} --deployment {dep}
#   firectl deployment delete {dep}
""")
    print(text.replace("$FIREWORKS_API_KEY", "$" + fw.KEY_ENV))


SHAPE_FP8_H200 = "accounts/fireworks/deploymentShapes/qwen3-4b-instruct-2507-minimal/versions/rvtb7riq"
SHAPE_BF16_B200 = "accounts/fireworks/deploymentShapes/qwen3-4b-instruct-2507-minimal-bf16-hotload/versions/w1hf25qk"


def deployment_bodies(model: str, dep: str) -> dict:
    """Planned deployment request bodies (query string, body). Nothing is created here."""
    scale = {"minReplicaCount": 0, "maxReplicaCount": 1,
             "autoscalingPolicy": {"scaleToZeroWindow": "300s"}}
    return {
        "B3 base, FP8 H200 validated shape": (
            f"deploymentId={dep}-base",
            {"displayName": f"{dep}-base", "baseModel": fw.BASE_MODEL,
             "deploymentShape": SHAPE_FP8_H200, **scale}),
        "v1 live-merge (LoRA model as baseModel, default shape)": (
            f"deploymentId={dep}-v1",
            {"displayName": f"{dep}-v1", "baseModel": model, **scale}),
        "multi-LoRA base, BF16 H200, shapeless (no validated addon shape exists)": (
            f"deploymentId={dep}-ml&skipShapeValidation=true",
            {"displayName": f"{dep}-ml", "baseModel": fw.BASE_MODEL, "enableAddons": True,
             "precision": "BF16", "acceleratorType": "NVIDIA_H200_141GB", "acceleratorCount": 1,
             **scale}),
    }


def cmd_deploy_check(a):
    """Read-only: shape match + validateOnly=true create. Creates nothing."""
    model = _full("models", a.model)
    for addons in (False, True):
        dep = {"baseModel": fw.BASE_MODEL, **({"enableAddons": True} if addons else {})}
        r = fw.post(fw.acct_path("/deploymentShapeVersions:match"),
                    {"createDeploymentRequest": {"deployment": dep}})
        names = [v["name"] for v in r.get("deploymentShapeVersions", [])]
        print(f"shapes for base (enableAddons={addons}): {names or 'NONE'}")
    if not a.model.startswith("<"):
        r = fw.post(fw.acct_path("/deploymentShapeVersions:match"),
                    {"createDeploymentRequest": {"deployment": {"baseModel": model}}})
        print(f"live-merge shapes for {model}: "
              f"{[v['name'] for v in r.get('deploymentShapeVersions', [])] or 'NONE'}")
    for label, (q, body) in deployment_bodies(model, a.deployment_id).items():
        if a.model.startswith("<") and "live-merge" in label:
            continue
        try:
            r = fw.post(fw.acct_path(f"/deployments?validateOnly=true&{q}"), body)
            print(f"[validateOnly OK] {label}: precision={r.get('precision')} "
                  f"accel={r.get('acceleratorType')} shape={r.get('deploymentShape')}")
        except fw.FWError as e:
            print(f"[validateOnly FAIL] {label}: {e.status} {_msg(e)}")


# ---------------------------------------------------------------- probe data
SYSTEM = ("You are CallGuard, a guardian that reads a live phone-call transcript window "
          "between a protected person (callee) and an unknown caller. Output ONLY a JSON "
          "object with keys risk, category, severity, source, stage, evidence, action, "
          "coach_line, caretaker_note. Synthetic safety-training data.")

BENIGN = [
    ["Hi, this is Dr. Patel's office calling to confirm your appointment Thursday at 3.",
     "Oh yes, Thursday works.", "Great, please arrive 10 minutes early. Have a good day."],
    ["Hey grandma, it's Sam, just calling to say I got an A on my test!",
     "That's wonderful sweetie!", "I'll tell you all about it Sunday at dinner."],
    ["Hello, this is the pharmacy, your prescription is ready for pickup.",
     "Thank you, I'll come by this afternoon.", "We're open until 8. Bye now."],
    ["Hi, I'm calling from the library, the book you reserved has arrived.",
     "Oh lovely, which one?", "The gardening one. We'll hold it for a week."],
    ["Hey, it's coach Miller, practice is moved to 5pm tomorrow.",
     "Okay, I'll be there.", "Bring your water bottle. See you then."],
]
SCAM = [
    (["This is Officer Daniels with the Social Security Administration.",
      "Oh my, is something wrong?",
      "Your number was used in a crime. To avoid arrest you must verify your number now."],
     {"risk": "alert", "category": "scam.impersonation_bank_gov", "severity": 3, "stage": "pressure",
      "evidence": ["To avoid arrest you must verify your number now"], "action": "coach",
      "coach_line": "Government agencies never threaten arrest by phone. Hang up and call them yourself.",
      "caretaker_note": "Caller impersonating SSA, threatening arrest, asking to verify SSN."}),
    (["Hi, this is Microsoft support, we detected a virus on your computer.",
      "Really? I didn't notice anything.",
      "We need you to buy two Google Play gift cards to renew your protection."],
     {"risk": "urgent", "category": "scam.tech_support", "severity": 4, "stage": "ask",
      "evidence": ["buy two Google Play gift cards"], "action": "urgent_alert",
      "coach_line": "No real company asks for gift cards. Hang up now.",
      "caretaker_note": "Tech-support scam requesting gift cards."}),
    (["Grandpa? It's me, I'm in trouble, I got arrested.",
      "Tommy? Is that you?",
      "Please don't tell mom. I need bail money wired today."],
     {"risk": "urgent", "category": "scam.family_emergency", "severity": 4, "stage": "ask",
      "evidence": ["Please don't tell mom", "I need bail money wired today"], "action": "urgent_alert",
      "coach_line": "Hang up and call Tommy or his parents directly before sending anything.",
      "caretaker_note": "Grandparent scam: urgent bail money, secrecy request."}),
    (["Hello, this is your bank's fraud department.",
      "Okay, what happened?",
      "We just sent you a code by text, please read it back to me to secure your account."],
     {"risk": "urgent", "category": "disclosure.credentials_otp", "severity": 4, "stage": "ask",
      "evidence": ["please read it back to me"], "action": "urgent_alert",
      "coach_line": "Never read a text code to a caller. Your bank will never ask for it.",
      "caretaker_note": "Caller requesting one-time passcode, claims to be bank fraud dept."}),
    (["Congratulations! You've won a free cruise for two!",
      "Wow, I never win anything.",
      "There's just a small processing fee, can I get your card number?"],
     {"risk": "alert", "category": "scam.prize", "severity": 3, "stage": "ask",
      "evidence": ["can I get your card number"], "action": "coach",
      "coach_line": "Real prizes don't need a fee. Don't share your card.",
      "caretaker_note": "Prize scam asking for card number for 'processing fee'."}),
]
HOOKS = [
    (["Hi, this is Jessica from Account Services, how are you today?",
      "Fine, who is this?", "I'm calling about an important update on your account."],
     {"risk": "watch", "category": "scam.impersonation_bank_gov", "severity": 1, "stage": "hook",
      "evidence": ["important update on your account"], "action": "log",
      "coach_line": "", "caretaker_note": ""}),
    (["Hey, I saw your profile, you seem really mature for your age.",
      "Um, thanks?", "What school do you go to?"],
     {"risk": "alert", "category": "disclosure.location_school", "severity": 3, "stage": "probe",
      "evidence": ["What school do you go to?"], "action": "coach",
      "coach_line": "You don't have to share your school. It's okay to say no.",
      "caretaker_note": "Unknown caller flattering child and asking for school."}),
]


def _fmt(turns, roles=("caller", "callee")):
    return "\n".join(f"[{i}] {roles[i % 2]}: {t}" for i, t in enumerate(turns))


def cmd_make_probe(a):
    rnd = random.Random(7)
    rows = []
    none = {"risk": "none", "category": "none", "severity": 0, "source": "caller", "stage": "hook",
            "evidence": [], "action": "none", "coach_line": "", "caretaker_note": ""}
    while len(rows) < a.n:
        k = rnd.random()
        if k < 0.45:
            turns = list(rnd.choice(BENIGN))
            cut = rnd.randint(1, len(turns))
            v = dict(none, stage=rnd.choice(["hook", "chat"]))
            turns = turns[:cut]
        elif k < 0.6:
            turns, lab = rnd.choice(HOOKS)
            v = dict(lab, source="caller")
        else:
            turns, lab = rnd.choice(SCAM)
            v = dict(lab, source="caller")
        mode = rnd.choice(["elder", "child", "self"])
        user = (f"mode: {mode}\ncaller_known: false\nai_voice: {rnd.choice(['unknown','low'])}\n"
                f"transcript:\n{_fmt(turns)}")
        rows.append({"messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": json.dumps(v, separators=(",", ":"))},
        ]})
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"wrote {len(rows)} rows -> {out}")


# ---------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--key-env", default=fw.KEY_ENV,
                    help="env var holding the API key (default FIREWORKS_API_KEY_MATT = Matt's account; "
                         "FIREWORKS_API_KEY = Numan's)")
    sp = ap.add_subparsers(dest="cmd", required=True)

    p = sp.add_parser("upload", help="create dataset + upload jsonl, wait READY")
    p.add_argument("jsonl")
    p.add_argument("--name", required=True, help="dataset id (lowercase, a-z0-9-)")
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--force", action="store_true", help="re-upload even if READY")
    p.set_defaults(fn=cmd_upload)

    p = sp.add_parser("train", help="create LoRA SFT job")
    p.add_argument("--dataset", required=True)
    p.add_argument("--name", required=True, help="job id (also output model id unless --output-model)")
    p.add_argument("--output-model")
    p.add_argument("--base-model", default=fw.BASE_MODEL)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=None, help="default: FW per-model default")
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--max-context", type=int, default=4096)
    p.add_argument("--batch-size-samples", type=int, default=None)
    p.add_argument("--warmup-steps", type=int, default=None)
    p.add_argument("--eval-dataset")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_train)

    p = sp.add_parser("status")
    p.add_argument("job_id")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_status)

    p = sp.add_parser("wait")
    p.add_argument("job_id")
    p.add_argument("--interval", type=int, default=20)
    p.add_argument("--timeout", type=int, default=0, help="seconds; 0 = forever")
    p.set_defaults(fn=cmd_wait)

    p = sp.add_parser("deploy-plan", help="print deployment commands; creates nothing")
    p.add_argument("--model", default="<FINE_TUNED_MODEL_ID>")
    p.add_argument("--deployment-id", default="cg-guardian")
    p.set_defaults(fn=cmd_deploy_plan)

    p = sp.add_parser("deploy-check", help="read-only shape match + validateOnly creates")
    p.add_argument("--model", default="<FINE_TUNED_MODEL_ID>")
    p.add_argument("--deployment-id", default="cg-guardian")
    p.set_defaults(fn=cmd_deploy_check)

    p = sp.add_parser("make-probe", help="write a tiny synthetic chat dataset")
    p.add_argument("out")
    p.add_argument("--n", type=int, default=30)
    p.set_defaults(fn=cmd_make_probe)

    a = ap.parse_args()
    fw.set_key_env(a.key_env)
    print(f"[account {fw.account_id()} via {a.key_env}]", file=sys.stderr)
    try:
        a.fn(a)
    except fw.FWError as e:
        hint = ""
        if "payment method is required" in e.body or "temporarily disabled" in e.body:
            hint = ("\nBLOCKER: account has no payment method. Add one at https://fireworks.ai/billing "
                    f"for account {fw.account_id()} (or ask Fireworks staff), then re-run.")
        sys.exit(f"API error: {e}{hint}")


if __name__ == "__main__":
    main()

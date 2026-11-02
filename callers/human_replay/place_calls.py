"""Replay the human clips as real phone calls: SignalWire (+1 304 648 1098) -> the guarded Telnyx number.

  python -m callers.human_replay.place_calls check                 # read-only GETs: SignalWire auth, caller number, bucket, signing permission
  python -m callers.human_replay.place_calls run                   # DRY RUN (default): prints every dial payload, touches nothing
  python -m callers.human_replay.place_calls run --limit 1 --live  # one real call first, then check recordings/<call_id>/caller.wav
  python -m callers.human_replay.place_calls run --live            # all clips, one at a time

--live, in order:
  1. preflight: env, SignalWire auth (GET), the caller number is in the project (GET), clip files match the manifest sha256
  2. upload the clips that are not yet in gs://emberz-g1-512122j5/human_replay/ (gcloud storage cp; the bucket stays private)
  3. sign V4 GET URLs, valid 2 h, as the emberz-g1 service account (gcloud storage sign-url --impersonate-service-account)
  4. per clip: POST /api/calling/calls {"command": "dial"} with inline SWML: silence, play the clip, hang up;
     log it, wait until SignalWire reports the call ended (GET /api/voice/logs/{id}), then wait --gap-s

How each call sounds on the line: the line answers, starts the media stream and plays the Emberz notice
to the caller; meanwhile our SWML waits --lead-silence-s and then plays the clip (the caller track,
recordings/<call_id>/caller.wav, holds only the clip: the notice is on the other track). The line rings
the protected phone in parallel; the clip ends and we hang up before its 30 s ring timeout.

SignalWire docs (checked 2026-11-02):
- Calling API, dial with inline SWML (`swml` as a JSON object), `caller_id`, `timeout` 1-600 s,
  `max_price_per_minute`, `status_url`/`status_events`; Basic auth project_id:api_token:
  https://signalwire.com/docs/apis/rest/calls/call-commands
- SWML play (`url` / `urls`, `silence:<seconds>`, HTTP(S) audio URLs): https://signalwire.com/docs/swml/reference/calling/play
- SWML hangup: https://signalwire.com/docs/swml/reference/calling/hangup
- cXML <Play> MIME types include audio/wav (same media engine): https://signalwire.com/docs/compatibility-api/cxml/reference/voice/play
- Compatibility (LaML) Create a Call needs `Url`/`ApplicationSid` or `Laml`/`Twiml`, but the inline
  `Laml`/`Twiml` parameter is only named, not documented, so we use the Calling API, whose inline
  `swml` is documented: https://signalwire.com/docs/compatibility-api/rest/calls/create-a-call
- Voice logs (status of a Calling API call, read-only): https://signalwire.com/docs/apis/rest/voice-logs/get-voice-log
  and https://signalwire.com/docs/apis/rest/voice-logs/list-voice-logs
- Phone numbers (read-only check that the caller number is ours): https://signalwire.com/docs/apis/rest/phone-numbers/list-phone-numbers
Google Cloud Storage V4 signed URLs: https://cloud.google.com/storage/docs/access-control/signed-urls
  (`gcloud storage sign-url`, max 12 h when impersonating a service account; needs
  iam.serviceAccounts.signBlob on that account, i.e. roles/iam.serviceAccountTokenCreator).

Secrets: SIGNALWIRE_SPACE, SIGNALWIRE_PROJECT_ID, SIGNALWIRE_API_TOKEN are read from the env file named
by TELNYX_ENV_FILE (or the environment). They are never printed or written anywhere.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from callers.human_replay import build_set
from callers.human_replay import sources as src

ROOT = src.ROOT
BUCKET = os.environ.get("EMBERZ_BUCKET", "gs://emberz-g1-512122j5")
PROJECT = os.environ.get("EMBERZ_PROJECT", "unified-booster-512122-j5")
SERVICE_ACCOUNT = os.environ.get("EMBERZ_SA", f"emberz-g1@{PROJECT}.iam.gserviceaccount.com")
PREFIX = "human_replay"
CALL_LOG = ROOT / "recordings" / "human_replay_calls.jsonl"
SHARED_LEDGER = ROOT / "data" / "callers" / "calls.jsonl"     # read by callers.label_sheet match-sources
SW_KEYS = ("SIGNALWIRE_SPACE", "SIGNALWIRE_PROJECT_ID", "SIGNALWIRE_API_TOKEN")
TERMINAL = {"completed", "ended", "busy", "failed", "no-answer", "canceled"}
DRY_URL = "https://storage.googleapis.com/{bucket}/{obj}?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Signature=DRY-RUN"


# --------------------------------------------------------------------------------------------
# env

def _parse_env_file(path: Path) -> dict[str, str]:
    try:
        from dotenv import dotenv_values
        return {k: v for k, v in dotenv_values(path).items() if v is not None}
    except ImportError:
        out = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.removeprefix("export ").split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
        return out


def load_env(env: Optional[dict[str, str]] = None) -> dict[str, str]:
    """Repo .env for the numbers; SIGNALWIRE_* from the file TELNYX_ENV_FILE points to (only those keys).
    The process environment wins over both."""
    env = dict(os.environ) if env is None else dict(env)
    repo = ROOT / ".env"
    merged: dict[str, str] = {}
    if repo.exists():
        merged.update(_parse_env_file(repo))
    ext = env.get("TELNYX_ENV_FILE") or merged.get("TELNYX_ENV_FILE")
    if ext and Path(ext).exists():
        vals = _parse_env_file(Path(ext))
        merged.update({k: vals[k] for k in SW_KEYS if k in vals})
    merged.update({k: v for k, v in env.items() if v})
    return merged


def sw_base(space: str) -> str:
    space = space.strip().removeprefix("https://").removeprefix("http://").strip("/")
    if "." not in space:
        space = f"{space}.signalwire.com"
    return f"https://{space}"


def is_e164(n: str) -> bool:
    return bool(n) and n.startswith("+") and n[1:].isdigit() and 8 <= len(n) <= 16


# --------------------------------------------------------------------------------------------
# payloads (pure: tested)

def build_swml(audio_url: str, lead_silence_s: int = 2, tail_silence_s: int = 1) -> dict[str, Any]:
    main: list[dict[str, Any]] = []
    if lead_silence_s >= 1:
        main.append({"play": {"url": f"silence:{int(lead_silence_s)}"}})
    main.append({"play": {"url": audio_url}})
    if tail_silence_s >= 1:
        main.append({"play": {"url": f"silence:{int(tail_silence_s)}"}})
    main.append({"hangup": {}})
    return {"version": "1.0.0", "sections": {"main": main}}


def build_dial(caller: str, guarded: str, audio_url: str, lead_silence_s: int = 2, ring_timeout_s: int = 30,
               max_price_per_minute: float = 0.05) -> dict[str, Any]:
    if not is_e164(caller) or not is_e164(guarded):
        raise ValueError("caller and guarded numbers must be E.164")
    if caller == guarded:
        raise ValueError("caller and guarded numbers are the same")
    return {
        "command": "dial",
        "params": {
            "from": caller,
            "to": guarded,
            "caller_id": caller,
            "timeout": int(ring_timeout_s),
            "max_price_per_minute": max_price_per_minute,
            "swml": build_swml(audio_url, lead_silence_s),
        },
    }


def object_name(row: dict[str, str]) -> str:
    """Content-addressed, so a rebuilt clip never reuses a stale object."""
    return f"{PREFIX}/{row['clip_id']}-{row.get('sha256', '')[:12] or 'nosha'}.wav"


def expected_call_s(row: dict[str, str], lead_silence_s: int) -> float:
    return lead_silence_s + float(row["duration"]) + 1.0


def pick_rows(rows: list[dict[str, str]], clips: Optional[str], start: int, limit: Optional[int],
              skip_done: set[str]) -> list[dict[str, str]]:
    if clips:
        want = [c.strip() for c in clips.split(",") if c.strip()]
        unknown = set(want) - {r["clip_id"] for r in rows}
        if unknown:
            raise SystemExit(f"unknown clip ids: {sorted(unknown)}")
        rows = [r for r in rows if r["clip_id"] in want]
    rows = [r for r in rows if r["clip_id"] not in skip_done]
    rows = rows[start:]
    return rows[:limit] if limit else rows


def done_clips(log_path: Path = CALL_LOG) -> set[str]:
    """Clips whose call already reached the line (status completed/ended), for --skip-done."""
    out: set[str] = set()
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("event") == "ended" and r.get("status") in ("completed", "ended"):
                    out.add(r["clip_id"])
    return out


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


# --------------------------------------------------------------------------------------------
# SignalWire (httpx)

class SignalWire:
    def __init__(self, env: dict[str, str], client=None):
        missing = [k for k in SW_KEYS if not env.get(k)]
        if missing:
            raise SystemExit(f"missing {missing}: set TELNYX_ENV_FILE in .env to the env file that has them")
        import httpx
        self.base = sw_base(env["SIGNALWIRE_SPACE"])
        self.http = client or httpx.Client(auth=(env["SIGNALWIRE_PROJECT_ID"], env["SIGNALWIRE_API_TOKEN"]),
                                           timeout=30.0, headers={"Accept": "application/json"})

    def get(self, path: str, **params) -> Any:
        r = self.http.get(self.base + path, params=params or None)
        if r.status_code >= 400:
            raise RuntimeError(f"GET {path} -> HTTP {r.status_code}: {r.text[:200]}")
        return r.json()

    def find_number(self, number: str) -> Optional[dict[str, Any]]:
        data = self.get("/api/relay/rest/phone_numbers", filter_number=number)
        for n in data.get("data", []):
            if n.get("number") == number:
                return n
        return None

    def dial(self, payload: dict[str, Any]) -> dict[str, Any]:
        r = self.http.post(self.base + "/api/calling/calls", json=payload)
        if r.status_code >= 400:
            raise RuntimeError(f"dial -> HTTP {r.status_code}: {r.text[:300]}")
        return r.json()

    def call_status(self, call_id: str) -> Optional[str]:
        try:
            return self.get(f"/api/voice/logs/{call_id}").get("status")
        except RuntimeError:
            return None


# --------------------------------------------------------------------------------------------
# Google Cloud Storage via the gcloud CLI (user login + service-account impersonation for signing)

def gcloud(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["gcloud", "--project", PROJECT, *args], capture_output=True, text=True, check=check)


def existing_objects() -> set[str]:
    r = gcloud("storage", "ls", f"{BUCKET}/{PREFIX}/", check=False)
    if r.returncode != 0:
        return set()          # prefix empty or missing
    return {line.strip().removeprefix(BUCKET + "/") for line in r.stdout.splitlines() if line.strip()}


def upload(rows: list[dict[str, str]], log=print) -> None:
    have = existing_objects()
    todo = [r for r in rows if object_name(r) not in have]
    log(f"upload: {len(rows) - len(todo)} already in {BUCKET}/{PREFIX}/, {len(todo)} to upload")
    for r in todo:
        gcloud("storage", "cp", "--content-type=audio/wav", str(ROOT / r["path"]), f"{BUCKET}/{object_name(r)}")


def sign(rows: list[dict[str, str]], duration: str = "2h") -> dict[str, tuple[str, datetime]]:
    """clip_id -> (signed URL, expiry). One gcloud call for all objects."""
    urls = [f"{BUCKET}/{object_name(r)}" for r in rows]
    r = gcloud("storage", "sign-url", *urls, f"--duration={duration}",
               f"--impersonate-service-account={SERVICE_ACCOUNT}", "--format=json", check=False)
    if r.returncode != 0:
        raise SystemExit("signing failed (needs roles/iam.serviceAccountTokenCreator on "
                         f"{SERVICE_ACCOUNT} for your gcloud account): {r.stderr.strip()[-400:]}")
    out: dict[str, tuple[str, datetime]] = {}
    by_obj = {f"{BUCKET}/{object_name(row)}": row["clip_id"] for row in rows}
    for item in json.loads(r.stdout):
        cid = by_obj.get(item["resource"])
        if cid:
            exp = datetime.strptime(item["expiration"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            out[cid] = (item["signed_url"], exp)
    missing = set(by_obj.values()) - set(out)
    if missing:
        raise SystemExit(f"no signed URL for {sorted(missing)}")
    return out


def can_sign() -> tuple[bool, str]:
    """Read-only: IAM testIamPermissions on the service account for signBlob."""
    import httpx
    tok = gcloud("auth", "print-access-token", check=False)
    if tok.returncode != 0:
        return False, "gcloud auth print-access-token failed"
    r = httpx.post(f"https://iam.googleapis.com/v1/projects/{PROJECT}/serviceAccounts/{SERVICE_ACCOUNT}:testIamPermissions",
                   headers={"Authorization": f"Bearer {tok.stdout.strip()}"},
                   json={"permissions": ["iam.serviceAccounts.signBlob"]}, timeout=30)
    ok = "iam.serviceAccounts.signBlob" in r.json().get("permissions", []) if r.status_code == 200 else False
    return ok, f"HTTP {r.status_code}"


# --------------------------------------------------------------------------------------------
# commands

def cmd_check(env: dict[str, str]) -> int:
    ok = True
    caller, guarded = env.get("SIGNALWIRE_CALLER_NUMBER", ""), env.get("TELNYX_GUARDED_NUMBER", "")
    print(f"caller {caller or 'MISSING'}  guarded {guarded or 'MISSING'}")
    ok &= is_e164(caller) and is_e164(guarded) and caller != guarded
    present = {k: bool(env.get(k)) for k in SW_KEYS}
    print("signalwire env:", present)
    if all(present.values()):
        sw = SignalWire(env)
        try:
            n = sw.find_number(caller)
            if n:
                print(f"signalwire: auth ok; {caller} is in the project (name {n.get('name')!r}, "
                      f"call_handler {n.get('call_handler')!r}; inbound routing is not touched)")
            else:
                ok = False
                print(f"signalwire: auth ok, but {caller} is NOT in this project")
        except RuntimeError as e:
            ok = False
            print(f"signalwire phone_numbers: {e}")
        try:
            logs = sw.get("/api/voice/logs", page_size=1)
            print(f"signalwire voice logs: ok (Voice scope), {len(logs.get('data', []))} row(s) returned")
        except RuntimeError as e:
            ok = False
            print(f"signalwire voice logs: {e}")
    else:
        ok = False
    b = gcloud("storage", "buckets", "describe", BUCKET, "--format=value(name,public_access_prevention)", check=False)
    print(f"bucket: {b.stdout.strip() or b.stderr.strip()[-200:]}")
    ok &= b.returncode == 0
    signable, why = can_sign()
    print(f"sign as {SERVICE_ACCOUNT}: {'ok' if signable else 'NO (' + why + ')'}")
    if not signable:
        ok = False
        print("  fix (Numan, once): gcloud iam service-accounts add-iam-policy-binding "
              f"{SERVICE_ACCOUNT} --project {PROJECT} --member=user:<your gcloud account> "
              "--role=roles/iam.serviceAccountTokenCreator")
    try:
        rows = build_set.read_manifest()
        errs = build_set.validate_manifest(rows, check_audio=True)
        print(f"manifest: {len(rows)} clips, {build_set.selection_stats(rows)['speakers']} speakers, "
              f"{'ok' if not errs else str(len(errs)) + ' problem(s): ' + '; '.join(errs[:3])}")
        ok &= not errs
    except FileNotFoundError:
        ok = False
        print("manifest: missing (run build_set build)")
    print("READY" if ok else "NOT READY")
    return 0 if ok else 1


def cmd_run(args, env: dict[str, str], sw_factory: Callable[[dict[str, str]], SignalWire] = SignalWire,
            sleep: Callable[[float], None] = time.sleep, log=print) -> int:
    rows = build_set.read_manifest(args.manifest)
    caller = env.get("SIGNALWIRE_CALLER_NUMBER", "")
    guarded = env.get("TELNYX_GUARDED_NUMBER", "")
    if not (is_e164(caller) and is_e164(guarded)):
        raise SystemExit("SIGNALWIRE_CALLER_NUMBER and TELNYX_GUARDED_NUMBER must be set (E.164) in .env")
    skip = done_clips() if args.skip_done else set()
    todo = pick_rows(rows, args.clips, args.start, args.limit, skip)
    if len(todo) > args.max_calls:
        raise SystemExit(f"{len(todo)} calls > --max-calls {args.max_calls}")
    est_min = sum(max(1, -(-int(expected_call_s(r, args.lead_silence_s) + 5) // 60)) for r in todo)
    log(f"{'LIVE' if args.live else 'DRY RUN'}: {len(todo)} calls {caller} -> {guarded}, "
        f"~{est_min} billed SignalWire minutes (~${est_min * 0.008:.2f} at $0.008/min), gap {args.gap_s}s")

    if not args.live:
        for r in todo:
            url = DRY_URL.format(bucket=BUCKET.removeprefix("gs://"), obj=object_name(r))
            payload = build_dial(caller, guarded, url, args.lead_silence_s, args.ring_timeout_s, args.max_price_per_minute)
            print(json.dumps({"clip_id": r["clip_id"], "speaker": r["speaker"], "gender": r.get("gender"),
                              "duration": r["duration"], "upload": f"{ROOT / r['path']} -> {BUCKET}/{object_name(r)}",
                              "POST": "/api/calling/calls", "payload": payload}))
        log("dry run: nothing uploaded, signed or dialed. Add --live to place the calls.")
        return 0

    # ---- live
    errs = build_set.validate_manifest(todo, check_audio=True, min_speakers=1)
    if errs:
        raise SystemExit("manifest/audio problems: " + "; ".join(errs[:5]))
    sw = sw_factory(env)
    if not sw.find_number(caller):
        raise SystemExit(f"{caller} is not in this SignalWire project")
    if not args.yes:
        if input(f"Place {len(todo)} real calls now? Type yes: ").strip() != "yes":
            log("aborted")
            return 1
    upload(todo, log)
    signed = sign(todo, args.url_ttl)
    failures = 0
    for i, r in enumerate(todo):
        url, exp = signed[r["clip_id"]]
        if exp - datetime.now(timezone.utc) < timedelta(minutes=10):
            signed.update(sign(todo[i:], args.url_ttl))
            url, exp = signed[r["clip_id"]]
        payload = build_dial(caller, guarded, url, args.lead_silence_s, args.ring_timeout_s, args.max_price_per_minute)
        started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        try:
            resp = sw.dial(payload)
        except RuntimeError as e:
            failures += 1
            log(f"{r['clip_id']}: dial failed: {e}")
            append_jsonl(CALL_LOG, {"event": "dial_failed", "clip_id": r["clip_id"], "speaker": r["speaker"],
                                    "started_at": started, "error": str(e)[:300]})
            if failures >= 2:
                log("two dial failures: stopping")
                return 1
            continue
        call_id = resp.get("id", "")
        row = {"call_id": call_id, "clip_id": r["clip_id"], "speaker": r["speaker"], "started_at": started,
               "source": r["source"], "gender": r.get("gender", ""), "accent": r.get("accent", ""),
               "duration_s": float(r["duration"]), "from": caller, "to": guarded}
        append_jsonl(CALL_LOG, row)
        # the shared ledger that callers.label_sheet match-sources pairs with recordings by start time
        append_jsonl(SHARED_LEDGER, {"platform": "signalwire_replay", "kind": "human", "started_utc": started,
                                     "clip_id": r["clip_id"], "speaker": r["speaker"], "source": r["source"],
                                     "signalwire_call_id": call_id})
        log(f"[{i + 1}/{len(todo)}] {r['clip_id']} {r['speaker']} ({r['duration']} s): call {call_id}")
        status = wait_for_end(sw, call_id, expected_call_s(r, args.lead_silence_s) + args.ring_timeout_s + 30, sleep)
        append_jsonl(CALL_LOG, {"event": "ended", "call_id": call_id, "clip_id": r["clip_id"], "status": status,
                                "ended_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        log(f"    status {status}")
        if status in ("busy", "failed", "no-answer", "canceled"):
            failures += 1
            if failures >= 2:
                log("two failed calls: stopping (is the line up and answering?)")
                return 1
        else:
            failures = 0
        if i + 1 < len(todo):
            sleep(args.gap_s)
    return 0


def wait_for_end(sw: SignalWire, call_id: str, max_wait_s: float, sleep: Callable[[float], None] = time.sleep) -> str:
    waited, status = 0.0, None
    while waited < max_wait_s:
        sleep(3)
        waited += 3
        status = sw.call_status(call_id) if call_id else None
        if status in TERMINAL:
            return status
    return status or "unknown"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="read-only readiness check")
    r = sub.add_parser("run", help="dry run by default; --live places calls")
    r.add_argument("--live", action="store_true", help="really upload, sign and dial (spends money)")
    r.add_argument("--yes", action="store_true", help="skip the confirmation prompt in --live")
    r.add_argument("--manifest", type=Path, default=build_set.MANIFEST)
    r.add_argument("--clips", default=None, help="comma-separated clip ids (default: all, manifest order)")
    r.add_argument("--start", type=int, default=0)
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--skip-done", action="store_true", help="skip clips whose call already completed")
    r.add_argument("--gap-s", type=float, default=20.0, help="pause after each call has ended")
    r.add_argument("--lead-silence-s", type=int, default=2, help="silence before the clip (stream start-up)")
    r.add_argument("--ring-timeout-s", type=int, default=30)
    r.add_argument("--max-price-per-minute", type=float, default=0.05)
    r.add_argument("--max-calls", type=int, default=80, help="hard cap per run")
    r.add_argument("--url-ttl", default="2h")
    args = p.parse_args(argv)
    env = load_env()
    if args.cmd == "check":
        return cmd_check(env)
    return cmd_run(args, env)


if __name__ == "__main__":
    sys.exit(main())

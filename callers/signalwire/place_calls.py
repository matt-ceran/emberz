"""Place the SignalWire AI test calls (PLAN.md V2) into the guarded Telnyx number.

DRY RUN BY DEFAULT: prints the exact HTTP request(s) and touches nothing. Only `--live` places calls.

  python -m callers.signalwire.place_calls                                   # dry run, 5 personas x 10 calls
  python -m callers.signalwire.place_calls --personas bank_fraud --calls 1   # dry run, one call
  python -m callers.signalwire.place_calls --summary                         # one line per planned call + cost
  python -m callers.signalwire.place_calls check                             # read-only: auth + caller number
  python -m callers.signalwire.place_calls --personas bank_fraud --calls 1 --live          # ONE real call
  python -m callers.signalwire.place_calls --calls 10 --live --max-calls 50                # all 50, Enter between calls
  python -m callers.signalwire.place_calls logs                              # read-only: duration, charge, voice warnings

Calls always go FROM SIGNALWIRE_CALLER_NUMBER (an idle SignalWire number, used outbound only; its
inbound routing is never touched) TO TELNYX_GUARDED_NUMBER. There is no flag to dial anything else.

Every live call is appended to recordings/ai_calls.jsonl (persona, variant, voice, engine, SignalWire
call id, start time): the ground-truth "AI" label for V1 and the scenario label for the guardian live set.
It is also appended to data/callers/calls.jsonl (callers.common ledger), so
`python -m callers.label_sheet match-sources` pairs each recording with its source.

API (checked 2026-11-02):
- Create call: POST https://{space}.signalwire.com/api/calling/calls, Basic auth project_id:api_token,
  body {"command": "dial", "params": {"from", "to", "swml" (inline object or JSON/YAML string) | "url",
  "caller_id", "timeout" (1-600), "status_url", "status_events", "url_method", "max_price_per_minute"}},
  200 -> call object with "id" and "status"; 422 -> {"errors": [...]}.
  https://signalwire.com/docs/apis/rest/calls/call-commands
  (OpenAPI: https://signalwire.com/docs/openapi/signalwire-rest-api.json, schema Calling.CallCreateParamsSWML)
- Phone numbers (read-only check): GET /api/relay/rest/phone_numbers?filter_number=...
  https://signalwire.com/docs/apis/rest/phone-numbers/list-phone-numbers
- Voice logs (read-only, after the calls): GET /api/voice/logs/{id} and /api/voice/logs/{id}/events
  https://signalwire.com/docs/apis/rest/voice-logs/get-voice-log (log id = the call's segment_id; fallback by
  time via GET /api/voice/logs?created_after=..., https://signalwire.com/docs/apis/rest/voice-logs/list-voice-logs)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import httpx

from callers.signalwire import personas as P

ROOT = Path(__file__).resolve().parents[2]
AI_CALLS_LOG = ROOT / "recordings" / "ai_calls.jsonl"
SW_SECRET_KEYS = ("SIGNALWIRE_SPACE", "SIGNALWIRE_PROJECT_ID", "SIGNALWIRE_API_TOKEN")
E164 = re.compile(r"^\+[1-9]\d{6,14}$")
HARD_MAX_CALLS = 60          # absolute ceiling per invocation, whatever --max-calls says
MIN_GAP_S = 270             # gap mode: never start a call while the previous one can still be running
RING_TIMEOUT_S = 30          # the line answers within a couple of seconds; 30 s is plenty

# Cost model (https://signalwire.com/pricing/voice, read 2026-11-02): AI Agent runtime $0.16/min incl.
# STT, LLM and standard TTS; outbound US PSTN (local) $0.008/min. Minutes per call are estimates:
# lead silence + waiting for "hello" + the conversation (scams hard-stop at 4:00, benign at 2:30).
AI_PER_MIN = 0.16
PSTN_PER_MIN = 0.008
EST_MIN = {"scam": 3.0, "benign": 2.0}
# Upper estimate: every call runs to its hard stop (4:00 / 2:30 + lead silence + last reply, ~4:15 / ~2:45),
# rounded up to whole minutes in case SignalWire bills per started minute (the pricing page doesn't say),
# plus the agent's speech billed as Standard-tier TTS on top ($0.0001/char, ~2,000 chars per call), in case
# "standard TTS included" does not hold for every engine we use.
UPPER_MIN = {"scam": 5.0, "benign": 3.0}
TTS_PER_CHAR = 0.0001
UPPER_TTS_CHARS = 2000


@dataclass(frozen=True)
class Settings:
    caller_number: str
    guarded_number: str
    space: str = ""
    project_id: str = ""
    api_token: str = ""
    public_base_url: str = ""

    @property
    def host(self) -> str:
        s = self.space.strip().removeprefix("https://").rstrip("/")
        return s if "." in s else f"{s}.signalwire.com"


def load_settings(with_secrets: bool) -> Settings:
    """Repo .env first (numbers, PUBLIC_BASE_URL). With secrets: the three SIGNALWIRE_* keys from the
    process env or, failing that, the file TELNYX_ENV_FILE points at (EmberDial's .env). Values are
    kept in memory only and never printed."""
    try:
        from dotenv import dotenv_values, load_dotenv
    except ImportError:  # pragma: no cover
        dotenv_values = load_dotenv = None
    if load_dotenv:
        load_dotenv(ROOT / ".env", override=False)
    env = dict(os.environ)
    if with_secrets and dotenv_values and env.get("TELNYX_ENV_FILE"):
        values = dotenv_values(env["TELNYX_ENV_FILE"])
        for k in SW_SECRET_KEYS:
            if not env.get(k) and values.get(k):
                env[k] = values[k]
    return Settings(
        caller_number=env.get("SIGNALWIRE_CALLER_NUMBER", ""),
        guarded_number=env.get("TELNYX_GUARDED_NUMBER", ""),
        space=env.get("SIGNALWIRE_SPACE", "") if with_secrets else "",
        project_id=env.get("SIGNALWIRE_PROJECT_ID", "") if with_secrets else "",
        api_token=env.get("SIGNALWIRE_API_TOKEN", "") if with_secrets else "",
        public_base_url=env.get("PUBLIC_BASE_URL", ""),
    )


def mask(number: str) -> str:
    return "*" * max(0, len(number) - 4) + number[-4:] if number else ""


# ------------------------------------------------------------------ request building (pure)
def swml_url(base: str, call: P.PlannedCall) -> str:
    return f"{base.rstrip('/')}/callers/swml/{call.persona}/{call.variant}?fill={call.fill}"


def dial_body(s: Settings, call: P.PlannedCall, mode: str = "inline", status_url: str = "",
              max_price_per_minute: Optional[float] = None) -> dict[str, Any]:
    params: dict[str, Any] = {
        "from": s.caller_number or "<SIGNALWIRE_CALLER_NUMBER>",
        "to": s.guarded_number or "<TELNYX_GUARDED_NUMBER>",
        "caller_id": s.caller_number or "<SIGNALWIRE_CALLER_NUMBER>",
        "timeout": RING_TIMEOUT_S,
    }
    if mode == "inline":
        params["swml"] = P.build_swml(call.persona, call.variant, call.fill)
    elif mode == "url":
        base = s.public_base_url or "<PUBLIC_BASE_URL>"
        params["url"] = swml_url(base, call)
        params["url_method"] = "GET"
    else:
        raise ValueError(f"mode must be inline|url, not {mode!r}")
    if status_url:
        params["status_url"] = status_url
        params["status_events"] = ["answered", "ended"]
    if max_price_per_minute is not None:
        params["max_price_per_minute"] = max_price_per_minute
    return {"command": "dial", "params": params}


def request_preview(s: Settings, body: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": "POST",
        "url": "https://{SIGNALWIRE_SPACE}/api/calling/calls",
        "headers": {"Content-Type": "application/json", "Authorization": "Basic <project_id:api_token, redacted>"},
        "json": body,
    }


def estimate_cost(plan: list[P.PlannedCall]) -> dict[str, float]:
    kinds = {p: P.describe(p, 0, 0)["kind"] for p in {c.persona for c in plan}}
    minutes = sum(EST_MIN[kinds[c.persona]] for c in plan)
    upper_min = sum(UPPER_MIN[kinds[c.persona]] for c in plan)
    upper_usd = upper_min * (AI_PER_MIN + PSTN_PER_MIN) + len(plan) * UPPER_TTS_CHARS * TTS_PER_CHAR
    return {"calls": len(plan), "est_minutes": round(minutes, 1),
            "est_usd": round(minutes * (AI_PER_MIN + PSTN_PER_MIN), 2),
            "upper_usd": round(upper_usd, 2)}


# ------------------------------------------------------------------ live guards
def live_problems(s: Settings, plan: list[P.PlannedCall], max_calls: int, mode: str) -> list[str]:
    errs = []
    for name, val in (("SIGNALWIRE_CALLER_NUMBER", s.caller_number), ("TELNYX_GUARDED_NUMBER", s.guarded_number)):
        if not E164.match(val or ""):
            errs.append(f"{name} is not set to an E.164 number")
    if s.caller_number and s.caller_number == s.guarded_number:
        errs.append("caller and guarded number are the same")
    for name, val in (("SIGNALWIRE_SPACE", s.space), ("SIGNALWIRE_PROJECT_ID", s.project_id), ("SIGNALWIRE_API_TOKEN", s.api_token)):
        if not val:
            errs.append(f"{name} missing (process env or the file TELNYX_ENV_FILE points at)")
    if len(plan) > min(max_calls, HARD_MAX_CALLS):
        errs.append(f"{len(plan)} calls planned but --max-calls is {max_calls} (ceiling {HARD_MAX_CALLS})")
    if mode == "url" and not (s.public_base_url or "").startswith("https://"):
        errs.append("--mode url needs PUBLIC_BASE_URL=https://... (the tunnel serving callers/signalwire/swml_router.py)")
    return errs


def make_client(s: Settings, transport: Any = None):
    return httpx.Client(base_url=f"https://{s.host}", auth=(s.project_id, s.api_token), timeout=20.0,
                        transport=transport)


def check_caller_number(client, s: Settings) -> dict[str, Any]:
    """Read-only: the caller number must exist in this SignalWire project and support voice."""
    r = client.get("/api/relay/rest/phone_numbers", params={"filter_number": s.caller_number.lstrip("+")})
    if r.status_code != 200:
        raise SystemExit(f"phone_numbers GET failed: HTTP {r.status_code} (check SIGNALWIRE_* credentials)")
    rows = [n for n in r.json().get("data", []) if n.get("number") == s.caller_number]
    if len(rows) != 1:
        raise SystemExit(f"caller number {mask(s.caller_number)} not found in this SignalWire project")
    n = rows[0]
    if "voice" not in (n.get("capabilities") or []):
        raise SystemExit(f"caller number {mask(s.caller_number)} has no voice capability")
    return {"number": mask(n["number"]), "number_type": n.get("number_type"),
            "capabilities": n.get("capabilities"), "inbound_call_handler": n.get("call_handler")}


def check_guarded_not_signalwire(client, s: Settings) -> None:
    """Read-only: the destination must NOT be a number in this SignalWire project. A mis-set
    TELNYX_GUARDED_NUMBER pointing at an EmberDial SignalWire number would send a scam agent into a
    production line. Same documented list endpoint as above, filtered on the guarded number."""
    r = client.get("/api/relay/rest/phone_numbers", params={"filter_number": s.guarded_number.lstrip("+")})
    if r.status_code != 200:
        raise SystemExit(f"phone_numbers GET failed: HTTP {r.status_code}")
    if any(n.get("number") == s.guarded_number for n in r.json().get("data", [])):
        raise SystemExit(f"refusing: {mask(s.guarded_number)} is a SignalWire number in this project, "
                         "not the Telnyx guarded line")


# ------------------------------------------------------------------ ledger
def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def ledger_row(call: P.PlannedCall, s: Settings, mode: str, doc_digest: str, started: str,
               http_status: int, resp: dict[str, Any]) -> dict[str, Any]:
    ok = http_status == 200 and bool(resp.get("id"))
    row = {
        "started_utc": started,
        "platform": "signalwire",
        "label": "ai",                     # V1 ground truth: this caller is an AI voice
        "call_id": resp.get("id") if ok else None,
        "status": resp.get("status") if ok else "request_failed",
        "http_status": http_status,
        "call_index": call.index,
        **P.describe(call.persona, call.variant, call.fill),
        "swml_mode": mode,
        "swml_sha": doc_digest,
        "from_last4": s.caller_number[-4:],
        "to_last4": s.guarded_number[-4:],
    }
    if not ok:
        row["error"] = json.dumps(resp)[:500]
    return row


# ------------------------------------------------------------------ commands
def cmd_call(args, out=print, wait: Callable[[str], str] = input, sleep: Callable[[float], None] = time.sleep,
             transport: Any = None) -> int:
    personas = args.personas or list(P.PERSONAS)
    unknown = set(personas) - set(P.PERSONAS)
    if unknown:
        raise SystemExit(f"unknown persona(s) {sorted(unknown)}; known: {list(P.PERSONAS)}")
    plan = P.plan_calls(personas, args.calls, start=args.start)
    if args.variants is not None:
        plan = [c for c in plan if c.variant in args.variants]
    s = load_settings(with_secrets=args.live)
    for c in plan:  # every document is checked before anything is printed or sent
        errs = P.validate_swml(P.build_swml(c.persona, c.variant, c.fill))
        if errs:
            raise SystemExit(f"{c}: invalid SWML: {errs}")

    if not args.live:
        if args.summary:
            for c in plan:
                d = P.describe(c.persona, c.variant, c.fill)
                out(f"{c.persona:20s} #{c.index:<2d} v{c.variant} fill{c.fill} {d['voice_label']:32s} {d['pace']:6s} {d['opening_text']}")
            out(json.dumps({"dry_run": True, **estimate_cost(plan)}))
            out("# dry run: nothing was sent. Add --live to place these calls.")
        else:
            reqs = [{"call": P.describe(c.persona, c.variant, c.fill) | {"call_index": c.index},
                     "request": request_preview(s, dial_body(s, c, args.mode, args.status_url, args.max_price_per_minute))}
                    for c in plan]
            out(json.dumps({"dry_run": True, **estimate_cost(plan), "requests": reqs}, indent=2))
        return 0

    problems = live_problems(s, plan, args.max_calls, args.mode)
    if problems:
        raise SystemExit("refusing to place calls:\n  - " + "\n  - ".join(problems))
    est = estimate_cost(plan)
    out(f"LIVE: {len(plan)} call(s) {mask(s.caller_number)} -> {mask(s.guarded_number)}, "
        f"est {est['est_minutes']} min, ~${est['est_usd']} (upper ~${est['upper_usd']})")
    with make_client(s, transport) as client:
        out(f"caller number ok: {json.dumps(check_caller_number(client, s))}")
        check_guarded_not_signalwire(client, s)
        for i, c in enumerate(plan):
            d = P.describe(c.persona, c.variant, c.fill)
            label = f"[{i + 1}/{len(plan)}] {c.persona} #{c.index} v{c.variant} {d['voice_label']} ({d['pace']})"
            if i > 0 or args.wait == "enter":
                if args.wait == "enter":
                    try:
                        answer = wait(f"{label}: press Enter to dial, q to stop: ")
                    except (EOFError, KeyboardInterrupt):  # no terminal (piped/CI) or Ctrl-C: never dial
                        answer = "q"
                    if answer.strip().lower() == "q":
                        out("stopped")
                        break
                else:
                    sleep(args.gap_s)
            body = dial_body(s, c, args.mode, args.status_url, args.max_price_per_minute)
            doc = P.build_swml(c.persona, c.variant, c.fill)
            started = datetime.now(timezone.utc).isoformat(timespec="seconds")
            try:
                r = client.post("/api/calling/calls", json=body)
            except httpx.HTTPError as e:
                # A timeout here may still have created the call. Log it (call_id unknown) so the recording
                # it produces is never left unlabeled, then stop: check `place_calls logs` / the dashboard.
                row = ledger_row(c, s, args.mode, P.swml_digest(doc), started, 0,
                                 {"transport_error": type(e).__name__})
                row["status"] = "unknown_transport_error"
                append_jsonl(args.log, row)
                out(f"{label}: request error {type(e).__name__}; the call may or may not exist. Stopping.")
                return 1
            try:
                resp = r.json()
            except ValueError:
                resp = {"body": r.text[:300]}
            row = ledger_row(c, s, args.mode, P.swml_digest(doc), started, r.status_code, resp)
            append_jsonl(args.log, row)
            try:
                from callers.common import append_ledger
                append_ledger({"platform": "signalwire", "agent": c.persona, "kind": d["kind"], "variant": c.variant,
                               "fill": c.fill, "voice": d["voice"], "call_id": row["call_id"], "started_utc": started,
                               "final_status": row["status"]})
            except Exception as e:  # the shared ledger is a convenience; ai_calls.jsonl is the record
                out(f"warning: data/callers/calls.jsonl not written: {e}")
            out(f"{label}: HTTP {r.status_code} call_id={row['call_id']} status={row['status']}")
            if row["call_id"] is None:
                out(f"stopping after a failed request: {row.get('error', '')[:300]}")
                return 1
    return 0


def cmd_check(args, out=print, transport: Any = None) -> int:
    """Read-only: credentials work and the caller number is in the project. Places nothing."""
    s = load_settings(with_secrets=True)
    missing = [k for k, v in zip(SW_SECRET_KEYS, (s.space, s.project_id, s.api_token)) if not v]
    if missing or not E164.match(s.caller_number or ""):
        raise SystemExit(f"missing settings: {missing or ['SIGNALWIRE_CALLER_NUMBER']}")
    with make_client(s, transport) as client:
        caller = check_caller_number(client, s)
        check_guarded_not_signalwire(client, s)
        out(json.dumps({"auth": "ok", "caller": caller, "guarded": mask(s.guarded_number),
                        "guarded_is_signalwire_number": False}))
    return 0


def _digits(x: Any) -> str:
    return "".join(ch for ch in str(x or "") if ch.isdigit())


def find_log_by_time(client, s: Settings, started_utc: str) -> Optional[dict[str, Any]]:
    """Fallback when the dial response id is not a voice-log id. The voice-log docs say the log id is the
    call's *segment_id* (https://signalwire.com/docs/apis/rest/voice-logs/get-voice-log), which may differ
    from the id the create-call response returns. Lists logs created from 2 minutes before our request
    (GET /api/voice/logs?created_after=..., https://signalwire.com/docs/apis/rest/voice-logs/list-voice-logs)
    and keeps only legs from our caller number to our guarded number; nothing else is returned or printed."""
    try:
        t0 = datetime.fromisoformat(started_utc)
    except (TypeError, ValueError):
        return None
    since = (t0.timestamp() - 120)
    r = client.get("/api/voice/logs", params={
        "created_after": datetime.fromtimestamp(since, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "page_size": 1000})
    if r.status_code != 200:
        return None
    ours = [x for x in r.json().get("data", [])
            if _digits(x.get("from")).endswith(_digits(s.caller_number)[-10:])
            and _digits(x.get("to")).endswith(_digits(s.guarded_number)[-10:])]

    def gap(x: dict[str, Any]) -> float:
        try:
            return abs(datetime.fromisoformat(str(x.get("created_at")).replace("Z", "+00:00")).timestamp()
                       - t0.timestamp())
        except ValueError:
            return float("inf")
    ours.sort(key=gap)
    return ours[0] if ours and gap(ours[0]) < 120 else None


def cmd_logs(args, out=print, transport: Any = None) -> int:
    """Read-only: per logged call, SignalWire's voice log (status, duration, charge) and any
    voice-substitution warning (relay_script_voice_unresolved: the TTS voice we asked for was replaced,
    https://signalwire.com/docs/swml/reference/errors). Places nothing. Rows whose create request timed out
    (no call_id) are looked up by time too."""
    rows = [json.loads(x) for x in Path(args.log).read_text(encoding="utf-8").splitlines() if x.strip()] \
        if Path(args.log).exists() else []
    rows = [r for r in rows if r.get("call_id") or r.get("status") == "unknown_transport_error"]
    s = load_settings(with_secrets=True)
    total = 0.0
    with make_client(s, transport) as client:
        for row in rows:
            log, via = None, "call_id"
            if row.get("call_id"):
                r = client.get(f"/api/voice/logs/{row['call_id']}")
                log = r.json() if r.status_code == 200 else None
            if log is None:
                log, via = find_log_by_time(client, s, row.get("started_utc")), "time_match"
            if log is None:
                out(json.dumps({"call_id": row.get("call_id"), "started_utc": row.get("started_utc"),
                                "voice_log": "not found"}))
                continue
            ev = client.get(f"/api/voice/logs/{log.get('id')}/events")
            unresolved = ev.status_code == 200 and "voice_unresolved" in ev.text
            charge = float(log.get("charge") or 0)
            total += charge
            out(json.dumps({"call_id": row.get("call_id"), "voice_log_id": log.get("id"), "matched_via": via,
                            "persona": row.get("persona"), "voice": row.get("voice"),
                            "status": log.get("status"), "duration_s": log.get("duration"), "charge": charge,
                            "charge_items": [c.get("description") for c in log.get("charge_details") or []],
                            "voice_unresolved": unresolved}))
    out(json.dumps({"calls": len(rows), "total_charge_usd": round(total, 4)}))
    return 0


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", nargs="?", default="call", choices=["call", "check", "logs"])
    p.add_argument("--personas", nargs="+", help=f"subset of {list(P.PERSONAS)} (default: all)")
    p.add_argument("--calls", type=int, default=10, help="calls per persona (default 10)")
    p.add_argument("--start", type=int, default=0, help="first call index per persona (resume a run)")
    p.add_argument("--variants", type=lambda x: [int(v) for v in x.split(",")], default=None,
                   help="only these variants, e.g. 0,2")
    p.add_argument("--mode", choices=["inline", "url"], default="inline",
                   help="inline: SWML in the request (default, no server needed); url: SignalWire fetches it "
                        "from PUBLIC_BASE_URL/callers/swml/... (needs swml_router mounted)")
    p.add_argument("--status-url", default="", help="optional call-state callback URL (answered, ended)")
    p.add_argument("--max-price-per-minute", type=float, default=None,
                   help="optional SignalWire route price cap in $/min (call rejected above it)")
    p.add_argument("--summary", action="store_true", help="dry run: one line per call instead of full requests")
    p.add_argument("--live", action="store_true", help="actually place the calls (spends money)")
    p.add_argument("--max-calls", type=int, default=1,
                   help=f"live: refuse if more calls are planned (default 1, ceiling {HARD_MAX_CALLS})")
    p.add_argument("--wait", choices=["enter", "gap"], default="enter",
                   help="live: press Enter before each call (default: someone answers the protected phone) "
                        "or sleep --gap-s between calls")
    p.add_argument("--gap-s", type=float, default=300.0, help="live --wait gap: seconds between call starts")
    p.add_argument("--log", type=Path, default=AI_CALLS_LOG, help="ledger (default recordings/ai_calls.jsonl)")
    a = p.parse_args(argv)
    if a.calls < 1 or a.start < 0:
        p.error("--calls must be >= 1 and --start >= 0")
    if a.wait == "gap" and a.gap_s < MIN_GAP_S:
        p.error(f"--gap-s must be >= {MIN_GAP_S} (longest call ~4:15; overlapping calls would mix up labels)")
    return a


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if args.cmd == "check":
        return cmd_check(args)
    if args.cmd == "logs":
        return cmd_logs(args)
    return cmd_call(args)


if __name__ == "__main__":
    sys.exit(main())

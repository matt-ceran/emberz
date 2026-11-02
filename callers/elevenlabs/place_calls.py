"""Place ElevenLabs test calls (benign bots only) into the guarded Telnyx number.

Dry run by default: prints every request it would send (passwords redacted) and touches no network.
`--live` is required to create or reuse agents, import the bridge number and place calls.

  python -m callers.elevenlabs.place_calls check                       # read-only GETs, no spend
  python -m callers.elevenlabs.place_calls call                        # dry run: all 20 calls
  python -m callers.elevenlabs.place_calls call --personas pharmacy_refill --variants 0 --live

Route "bridge" (the only one implemented; README.md explains why):

  SignalWire  POST /api/calling/calls  {command: dial, from: SIGNALWIRE_CALLER_NUMBER,
                                        to: TELNYX_GUARDED_NUMBER, swml: <inline>}
     |  https://signalwire.com/docs/apis/rest/calls/call-commands
     |  when the guarded line answers, the inline SWML runs `connect` to
     v  sip:<caller number>@sip.rtc.elevenlabs.io:5061;transport=tls  (digest auth, X- headers)
  ElevenLabs  imported "SIP trunk" number (inbound only) -> assigned agent answers
                https://elevenlabs.io/docs/agents-platform/phone-numbers/sip-trunking
                https://signalwire.com/docs/swml/methods/connect

Nothing on SignalWire is created or changed: the call is one REST request with its SWML inline, and
the caller number's inbound routing is never read for writing. On ElevenLabs it creates (once) up to
6 agents named "Emberz test caller: ..." tagged `emberz-test-caller`, and one imported SIP-trunk
number labelled BRIDGE_LABEL. Before each call it re-assigns that number to the call's agent. It
never modifies an agent or number it did not create (name/tag/label checks).

Every placed call is appended to recordings/ai_calls.jsonl with the SignalWire agent's shape
{call_id, provider, persona, variant, voice, engine, started_at} plus a few extra keys.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from callers.elevenlabs import agents as A

ROOT = Path(__file__).resolve().parents[2]
STATE_PATH = ROOT / "data" / "callers" / "elevenlabs.json"       # gitignored: created ids
LEDGER_PATH = ROOT / "recordings" / "ai_calls.jsonl"              # shared AI-caller ledger
OUTCOMES_PATH = ROOT / "data" / "callers" / "elevenlabs_outcomes.jsonl"   # end status + conversation id

EL_API = "https://api.elevenlabs.io"
EL_SIP_HOST = "sip.rtc.elevenlabs.io"        # shared endpoint (sip-trunking docs)
BRIDGE_LABEL = "Emberz EL bridge (inbound SIP from SignalWire, test only)"
BRIDGE_SIP_USER = "emberz-bridge"
PROVIDER = "elevenlabs"
REDACTED = "<redacted>"

# SignalWire voice-log statuses that mean the call is over
# (https://signalwire.com/docs/apis/rest/voice-logs/get-voice-log, `status` enum).
TERMINAL = {"completed", "ended", "busy", "failed", "no-answer", "canceled"}


# ---- env -----------------------------------------------------------------------------------------
def load_env() -> dict[str, str]:
    """Repo .env, then the secrets file it points at (TELNYX_ENV_FILE / ELEVENLABS_ENV_FILE), then
    the process env (wins). Values are never printed."""
    from dotenv import dotenv_values

    repo = {k: v for k, v in dotenv_values(ROOT / ".env").items() if v is not None}
    secrets_path = os.environ.get("ELEVENLABS_ENV_FILE") or repo.get("ELEVENLABS_ENV_FILE") \
        or os.environ.get("TELNYX_ENV_FILE") or repo.get("TELNYX_ENV_FILE")
    secrets = {}
    if secrets_path and Path(secrets_path).exists():
        secrets = {k: v for k, v in dotenv_values(secrets_path).items() if v is not None}
    return {**secrets, **repo, **{k: v for k, v in os.environ.items()}}


def need(env: dict[str, str], name: str) -> str:
    val = (env.get(name) or "").strip()
    if not val:
        raise SystemExit(f"{name} is not set (repo .env, the TELNYX_ENV_FILE secrets file, or the environment)")
    return val


def numbers(env: dict[str, str]) -> tuple[str, str]:
    """(caller, guarded). Calls only ever go from the idle SignalWire number to the guarded number."""
    caller, guarded = need(env, "SIGNALWIRE_CALLER_NUMBER"), need(env, "TELNYX_GUARDED_NUMBER")
    for n in (caller, guarded):
        if not (n.startswith("+") and n[1:].isdigit()):
            raise SystemExit(f"phone numbers must be E.164, got {mask(n)}")
    if caller == guarded:
        raise SystemExit("caller and guarded number are the same")
    return caller, guarded


def mask(number: str) -> str:
    return "*" * max(0, len(number) - 4) + number[-4:] if number else ""


# ---- payloads (pure: no network) -----------------------------------------------------------------
def bridge_number_payload(caller: str, sip_password: str, agent_id: str | None) -> dict[str, Any]:
    """POST /v1/convai/phone-numbers, provider sip_trunk, inbound only
    (https://elevenlabs.io/docs/api-reference/phone-numbers/create). The phone_number is only the
    identifier SignalWire puts before the @ in the SIP URI; it must match the URI exactly, '+' included."""
    return {
        "phone_number": caller,
        "label": BRIDGE_LABEL,
        "provider": "sip_trunk",
        "agent_id": agent_id,
        "inbound_trunk_config": {
            "media_encryption": "allowed",
            "credentials": {"username": BRIDGE_SIP_USER, "password": sip_password},
        },
        # no outbound_trunk_config: ElevenLabs never dials out on this number
    }


def sip_uri(caller: str, transport: str) -> str:
    port = 5061 if transport == "tls" else 5060
    return f"sip:{caller}@{EL_SIP_HOST}:{port};transport={transport}"


def dial_payload(persona: str, variant: int, caller: str, guarded: str, sip_password: str,
                 transport: str = "tls", ring_timeout_s: int = 45) -> dict[str, Any]:
    """SignalWire Calling API dial with inline SWML that bridges the answered call to ElevenLabs.
    Shapes: https://signalwire.com/docs/apis/rest/calls/call-commands (dial, inline swml) and
    https://signalwire.com/docs/swml/methods/connect (to, from, headers [{name,value}], username,
    password, codecs, timeout, max_duration)."""
    spec = A.load_spec(persona)                 # allowlist check happens here
    max_s = int(spec["max_duration_s"]) + 60
    swml = {
        "version": "1.0.0",
        "sections": {"main": [
            {"connect": {
                "to": sip_uri(caller, transport),
                "from": caller,
                "username": BRIDGE_SIP_USER,
                "password": sip_password,
                "headers": A.sip_headers(persona, variant),
                "codecs": "PCMU,PCMA",
                "timeout": 30,
                "max_duration": max_s,
            }},
            {"hangup": {}},
        ]},
    }
    return {"command": "dial", "params": {
        "from": caller,
        "to": guarded,
        "caller_id": caller,
        "timeout": ring_timeout_s,
        "max_price_per_minute": 0.05,      # US PSTN is ~$0.01/min; refuse anything odd
        "swml": swml,
    }}


def ledger_row(call_id: str, persona: str, variant: int, started_at: str, **extra: Any) -> dict[str, Any]:
    idx = A.voice_index(persona, variant)
    v = A.voice(persona, idx)
    row = {"call_id": call_id, "provider": PROVIDER, "persona": persona, "variant": variant,
           "voice": f"{v['name']}:{v['voice_id']}", "engine": A.ENGINE, "started_at": started_at}
    row.update(extra)
    return row


def redact(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: (REDACTED if k in ("password",) and v else redact(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact(x) for x in obj]
    return obj


def plan_calls(personas: list[str], variants: list[int]) -> list[tuple[str, int]]:
    for p in personas:
        A.check_persona(p)
        spec = A.load_spec(p)
        bad = [v for v in variants if not 0 <= v < len(spec["variants"])]
        if bad:
            raise SystemExit(f"{p}: no variant {bad}; it has 0-{len(spec['variants']) - 1}")
    return [(p, v) for v in variants for p in personas]      # interleave personas


# ---- state ---------------------------------------------------------------------------------------
def load_state() -> dict[str, Any]:
    return json.loads(STATE_PATH.read_text(encoding="utf-8")) if STATE_PATH.exists() else {}


def save_state(state: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def append_ledger(row: dict[str, Any]) -> None:
    append_jsonl(LEDGER_PATH, row)


# ---- HTTP clients (live / check only) ------------------------------------------------------------
def el_client(env: dict[str, str]):
    import httpx
    return httpx.Client(base_url=EL_API, headers={"xi-api-key": need(env, "ELEVENLABS_API_KEY")}, timeout=30)


def sw_client(env: dict[str, str]):
    import httpx
    space = need(env, "SIGNALWIRE_SPACE")
    host = space if "." in space else f"{space}.signalwire.com"
    return httpx.Client(base_url=f"https://{host}", timeout=30,
                        auth=(need(env, "SIGNALWIRE_PROJECT_ID"), need(env, "SIGNALWIRE_API_TOKEN")))


def ok(r, what: str) -> Any:
    if r.status_code >= 400:
        raise SystemExit(f"{what}: HTTP {r.status_code}: {r.text[:300]}")
    return r.json() if r.content else {}


def is_ours_agent(a: dict[str, Any]) -> bool:
    return str(a.get("name", "")).startswith(A.AGENT_NAME_PREFIX) and A.AGENT_TAG in (a.get("tags") or [])


def ensure_agent(el, state: dict[str, Any], persona: str, idx: int, update: bool) -> str:
    """Reuse the agent from state or by exact name+tag; create it otherwise.
    GET/PATCH /v1/convai/agents/{id}, GET /v1/convai/agents?search=, POST /v1/convai/agents/create
    (https://elevenlabs.io/docs/api-reference/agents/list, .../agents/update, .../agents/create)."""
    key = f"{persona}/{idx}"
    payload = A.agent_payload(persona, idx)
    agent_id = state.get("agents", {}).get(key)
    if agent_id:
        r = el.get(f"/v1/convai/agents/{agent_id}")
        if r.status_code == 404:
            agent_id = None
        else:
            got = ok(r, f"get agent {key}")
            if not is_ours_agent(got):
                raise SystemExit(f"state points {key} at an agent that isn't ours; refusing to use it")
    if not agent_id:
        found = ok(el.get("/v1/convai/agents", params={"search": payload["name"], "page_size": 100}), "list agents")
        mine = [a for a in found.get("agents", []) if a.get("name") == payload["name"] and is_ours_agent(a)]
        agent_id = mine[0]["agent_id"] if mine else None
    if agent_id and update:
        ok(el.patch(f"/v1/convai/agents/{agent_id}", json=payload), f"update agent {key}")
        print(f"updated agent {key}")
    if not agent_id:
        agent_id = ok(el.post("/v1/convai/agents/create", json=payload), f"create agent {key}")["agent_id"]
        print(f"created agent {key}")
    state.setdefault("agents", {})[key] = agent_id
    save_state(state)
    return agent_id


def ensure_bridge_number(el, state: dict[str, Any], caller: str, sip_password: str, first_agent: str) -> str:
    """Reuse or import the inbound-only SIP-trunk number (POST /v1/convai/phone-numbers)."""
    pid = state.get("bridge_phone_number_id")
    if pid:
        r = el.get(f"/v1/convai/phone-numbers/{pid}")
        if r.status_code == 404:
            pid = None
        elif ok(r, "get bridge number").get("label") != BRIDGE_LABEL:
            raise SystemExit("state points at a phone number that isn't the Emberz bridge; refusing")
    if not pid:
        nums = ok(el.get("/v1/convai/phone-numbers"), "list phone numbers")
        mine = [n for n in nums if n.get("label") == BRIDGE_LABEL and n.get("phone_number") == caller]
        clash = [n for n in nums if n.get("phone_number") == caller and n.get("label") != BRIDGE_LABEL]
        if clash:
            raise SystemExit(f"{mask(caller)} is already imported in ElevenLabs under another label; refusing")
        pid = mine[0]["phone_number_id"] if mine else None
    if not pid:
        body = bridge_number_payload(caller, sip_password, first_agent)
        pid = ok(el.post("/v1/convai/phone-numbers", json=body), "import bridge number")["phone_number_id"]
        print("imported the bridge SIP-trunk number")
    state["bridge_phone_number_id"] = pid
    save_state(state)
    return pid


def wait_for_end(sw, call_id: str, limit_s: int) -> str:
    """Poll GET /api/voice/logs/{id} (read-only) until a terminal status or the time limit."""
    deadline, status = time.monotonic() + limit_s, "unknown"
    while time.monotonic() < deadline:
        time.sleep(10)
        r = sw.get(f"/api/voice/logs/{call_id}")
        if r.status_code < 400:
            status = str(r.json().get("status", status))
            if status in TERMINAL:
                return status
    return f"timeout({status})"


def find_conversation(el, agent_id: str, after_unix: int) -> str | None:
    r = el.get("/v1/convai/conversations", params={"agent_id": agent_id, "call_start_after_unix": after_unix - 5,
                                                    "page_size": 5})
    if r.status_code >= 400:
        return None
    convs = r.json().get("conversations", [])
    return convs[0].get("conversation_id") if convs else None


# ---- commands ------------------------------------------------------------------------------------
def cmd_check(args: argparse.Namespace) -> int:
    """Read-only: key works, credits left, existing agents/numbers, caller number routing."""
    env = load_env()
    caller, guarded = numbers(env)
    with el_client(env) as el:
        s = ok(el.get("/v1/user/subscription"), "subscription")
        used, limit = s.get("character_count", 0), s.get("character_limit", 0)
        reset = datetime.fromtimestamp(s.get("next_character_count_reset_unix", 0), timezone.utc).date()
        print(f"ElevenLabs key OK: tier={s.get('tier')} status={s.get('status')} "
              f"credits used {used:,}/{limit:,} (left {limit - used:,}, resets {reset})")
        ags = ok(el.get("/v1/convai/agents", params={"page_size": 100}), "list agents").get("agents", [])
        ours = [a for a in ags if is_ours_agent(a)]
        print(f"agents: {len(ags)} total, {len(ours)} ours (Emberz), {len(ags) - len(ours)} others (never touched)")
        nums = ok(el.get("/v1/convai/phone-numbers"), "list phone numbers")
        print(f"phone numbers: {len(nums)}; bridge imported: {any(n.get('label') == BRIDGE_LABEL for n in nums)}")
    with sw_client(env) as sw:
        d = ok(sw.get("/api/relay/rest/phone_numbers", params={"filter_number": caller}), "SignalWire number")
        rows = d.get("data", [])
        print(f"SignalWire caller {mask(caller)}: {'found' if rows else 'NOT FOUND'} "
              f"(inbound handler {rows[0].get('call_handler') if rows else '-'}; this script never changes it)")
    print(f"calls would go {mask(caller)} -> {mask(guarded)}")
    return 0


def cmd_call(args: argparse.Namespace) -> int:
    personas = args.personas.split(",") if args.personas else list(A.ALLOWED_PERSONAS)
    for p in personas:                       # refuse scam personas before anything else
        A.check_persona(p)
    variants = parse_variants(args.variants, list(range(10)))
    calls = plan_calls(personas, variants)
    if args.max_calls is not None:
        calls = calls[: args.max_calls]
    keys = sorted({(p, A.voice_index(p, v)) for p, v in calls})

    env = load_env() if args.live else {}
    if args.live:
        caller, guarded = numbers(env)
        sip_password = need(env, "EMBERZ_EL_SIP_PASSWORD")
    else:
        caller = env.get("SIGNALWIRE_CALLER_NUMBER") or _dry_env("SIGNALWIRE_CALLER_NUMBER", "+13046481098")
        guarded = env.get("TELNYX_GUARDED_NUMBER") or _dry_env("TELNYX_GUARDED_NUMBER", "+13235161282")
        sip_password = REDACTED

    if not args.live:
        print(f"# DRY RUN: nothing is sent. {len(keys)} agents, 1 bridge number, {len(calls)} calls.\n")
        for p, i in keys:
            print(f"## POST {EL_API}/v1/convai/agents/create   ({p}/{i}, skipped if it exists)")
            print(json.dumps(A.agent_payload(p, i), indent=2))
        print(f"\n## POST {EL_API}/v1/convai/phone-numbers   (once, skipped if it exists)")
        print(json.dumps(redact(bridge_number_payload(caller, sip_password, "<first agent_id>")), indent=2))
        for p, v in calls:
            print(f"\n## call {p} variant {v} (voice {A.voice(p, A.voice_index(p, v))['name']})")
            print(f"PATCH {EL_API}/v1/convai/phone-numbers/<bridge id>  "
                  + json.dumps({"agent_id": f"<{p}/{A.voice_index(p, v)}>"}))
            print("POST https://<space>/api/calling/calls")
            print(json.dumps(redact(dial_payload(p, v, caller, guarded, sip_password, args.transport)), indent=2))
        print("\n# add --live to send (Numan only, once the line is up and recording).")
        return 0

    state = load_state()
    with el_client(env) as el, sw_client(env) as sw:
        agent_ids = {k: ensure_agent(el, state, k[0], k[1], args.update_agents) for k in keys}
        pid = ensure_bridge_number(el, state, caller, sip_password, next(iter(agent_ids.values())))
        for n, (p, v) in enumerate(calls):
            aid = agent_ids[(p, A.voice_index(p, v))]
            ok(el.patch(f"/v1/convai/phone-numbers/{pid}", json={"agent_id": aid}), "assign agent")
            started = datetime.now(timezone.utc)
            body = dial_payload(p, v, caller, guarded, sip_password, args.transport)
            if body["params"]["to"] != guarded or body["params"]["from"] != caller:
                raise SystemExit("refusing: dial payload numbers changed")
            res = ok(sw.post("/api/calling/calls", json=body), f"dial {p}/{v}")
            call_id = res.get("id", "")
            print(f"[{n + 1}/{len(calls)}] placed {p} v{v} -> {mask(guarded)} call_id={call_id}")
            row = ledger_row(call_id, p, v, started.isoformat(timespec="seconds"),
                             route="signalwire-sip-bridge", from_number=caller, el_agent_id=aid,
                             exclude_from_training=True)   # PLAN.md V1: no ElevenLabs audio in training
            append_ledger(row)
            status = wait_for_end(sw, call_id, int(A.load_spec(p)["max_duration_s"]) + 180)
            conv = find_conversation(el, aid, int(started.timestamp()))
            print(f"    ended: {status}; ElevenLabs conversation {conv or 'not found yet'}")
            # One row per call in the shared ledger; outcomes go to our own (gitignored) log.
            append_jsonl(OUTCOMES_PATH, {"call_id": call_id, "status": status, "el_conversation_id": conv,
                                         "ended_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
            if n + 1 < len(calls):
                time.sleep(args.gap_s)
    return 0


def _dry_env(name: str, fallback: str) -> str:
    """Dry runs read numbers from the repo .env when present, never secrets."""
    try:
        from dotenv import dotenv_values
        val = dotenv_values(ROOT / ".env").get(name)
    except Exception:
        val = None
    return val or fallback


def parse_variants(arg: str | None, default: list[int]) -> list[int]:
    if not arg:
        return default
    out: list[int] = []
    for part in arg.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="read-only account checks (no spend)")
    c = sub.add_parser("call", help="dry run by default; --live places calls")
    c.add_argument("--personas", default=None, help=f"comma list, from {list(A.ALLOWED_PERSONAS)}")
    c.add_argument("--variants", default=None, help="'0-9' or '0,3' (default all 10)")
    c.add_argument("--max-calls", type=int, default=None)
    c.add_argument("--transport", choices=["tls", "tcp"], default="tls",
                   help="SIP transport to ElevenLabs (tcp only to debug; TLS is recommended)")
    c.add_argument("--gap-s", type=int, default=20, help="pause between calls")
    c.add_argument("--update-agents", action="store_true", help="PATCH our existing agents to the current spec")
    c.add_argument("--live", action="store_true", help="actually create agents and place calls (spends money)")
    args = p.parse_args(argv)
    try:
        return cmd_check(args) if args.cmd == "check" else cmd_call(args)
    except A.PersonaNotAllowed as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

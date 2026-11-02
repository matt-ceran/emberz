"""Create the Emberz test-caller agents on Retell and place outbound calls to the guarded number.

DRY RUN BY DEFAULT: prints the request payloads as JSON and sends nothing. --live sends them.

  python -m callers.retell_create voices                                            # GET /list-voices (read-only, needs key)
  python -m callers.retell_create create [--agents bank_fraud]                      # print payloads
  python -m callers.retell_create create --live                                     # create LLMs + agents
  python -m callers.retell_create call --agents bank_fraud --variants 5-9           # print payloads
  python -m callers.retell_create call --agents bank_fraud --variants 5-9 --live    # place calls, one at a time

Env: RETELL_API_KEY, RETELL_FROM_NUMBER (a number bought from or imported to Retell), GUARDED_NUMBER.

API (checked 2026-11-02):
  POST https://api.retellai.com/create-retell-llm      https://docs.retellai.com/api-references/create-retell-llm
  POST https://api.retellai.com/create-agent           https://docs.retellai.com/api-references/create-agent
  POST https://api.retellai.com/v2/create-phone-call   https://docs.retellai.com/api-references/create-phone-call
  GET  https://api.retellai.com/v2/get-call/{id}       https://docs.retellai.com/api-references/get-call
  GET  https://api.retellai.com/list-voices            https://docs.retellai.com/api-references/list-voices
  {{var}} + retell_llm_dynamic_variables (strings only): https://docs.retellai.com/build/dynamic-variables
Retell voice ids are account-visible strings (e.g. "retell-Cimo"), so specs name a provider +
gender and `--live` resolves a concrete voice_id from /list-voices. ElevenLabs is never picked.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any

from callers.common import (BANNED_VOICE_PROVIDERS, append_ledger, check_variants, full_prompt, load_env, load_specs,
                            load_state, mask, parse_variants, require_env, save_state, voice_for)

API = "https://api.retellai.com"
STATE = "retell_agents.json"
DEFAULT_VARIANTS = list(range(5, 10))   # variants 5-9 on Retell, 0-4 on Vapi


def llm_payload(spec: dict[str, Any]) -> dict[str, Any]:
    # start_speaker "user": wait for the protected person's "hello" (the line plays a notice and
    # rings the protected phone first); the opening line is in the prompt (common.CALL_START).
    return {
        "general_prompt": full_prompt(spec),
        "start_speaker": "user",
        "model_config": {"type": "cascading", "primary": {"model": spec["retell"]["model"]}, "temperature": 0.7},
        "general_tools": [{"type": "end_call", "name": "end_call",
                           "description": f"End the call right after saying '{spec['end_phrase']}'"}],
    }


def agent_payload(spec: dict[str, Any], llm_id: str, voice_id: str) -> dict[str, Any]:
    return {
        "response_engine": {"type": "retell-llm", "llm_id": llm_id},
        "voice_id": voice_id,
        "agent_name": f"emberz-{spec['id']}",
        "language": "en-US",
        "max_call_duration_ms": spec["max_duration_s"] * 1000,
        "end_call_after_silence_ms": 60000,   # notice + ringing can be ~30 s of silence (min 10000)
    }


def call_payload(spec: dict[str, Any], variant: int, agent_id: str, voice_id: str, from_number: str, to: str) -> dict[str, Any]:
    return {
        "from_number": from_number,
        "to_number": to,
        "override_agent_id": agent_id,
        "agent_override": {"agent": {"voice_id": voice_id}},
        "retell_llm_dynamic_variables": spec["variants"][variant],
        "metadata": {"project": "emberz", "agent": spec["id"], "kind": spec["kind"], "variant": variant},
    }


def placeholder_voice(want: dict[str, Any]) -> str:
    return f"<resolved at --live from /list-voices: provider={want['provider']} gender={want['gender']}>"


def resolve_voice(voices: list[dict[str, Any]], want: dict[str, Any], used: set[str]) -> str:
    """Pick a voice of the wanted provider + gender, preferring ones not used yet."""
    if want["provider"].lower() in BANNED_VOICE_PROVIDERS:
        raise ValueError("ElevenLabs voices are not allowed (PLAN.md V1)")
    pool = [v for v in voices if v.get("provider") == want["provider"] and v.get("gender") == want["gender"]
            and v.get("provider") not in BANNED_VOICE_PROVIDERS]
    if not pool:
        pool = [v for v in voices if v.get("provider") == want["provider"]]
    if not pool:
        raise SystemExit(f"no Retell voice for provider={want['provider']}; run `voices` and edit the spec")
    fresh = [v for v in pool if v["voice_id"] not in used] or pool
    used.add(fresh[0]["voice_id"])
    return fresh[0]["voice_id"]


def _client():
    import httpx

    return httpx.Client(base_url=API, timeout=30.0, headers={"Authorization": f"Bearer {require_env('RETELL_API_KEY')}"})


def _list_voices(http) -> list[dict[str, Any]]:
    r = http.get("/list-voices")
    r.raise_for_status()
    return r.json()


def cmd_voices(args: argparse.Namespace) -> list[dict[str, Any]]:
    with _client() as http:
        voices = [v for v in _list_voices(http) if v.get("provider") not in BANNED_VOICE_PROVIDERS]
    for v in voices:
        print(f"{v['voice_id']:32} {v.get('provider', ''):12} {v.get('gender', ''):8} {v.get('accent', '')}", file=sys.stderr)
    return []


def cmd_create(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs = load_specs(args.agents)
    if not args.live:
        out = []
        for s in specs:
            out.append({"method": "POST", "url": f"{API}/create-retell-llm", "body": llm_payload(s)})
            out.append({"method": "POST", "url": f"{API}/create-agent",
                        "body": agent_payload(s, f"<llm_id of {s['id']}>", placeholder_voice(s["retell"]["voices"][0]))})
        return out
    state = load_state(STATE)
    with _client() as http:
        all_voices = _list_voices(http)
        used: set[str] = set()
        for spec in specs:
            if spec["id"] in state and not args.recreate:
                print(f"skip {spec['id']}: already created", file=sys.stderr)
                continue
            voice_ids = [resolve_voice(all_voices, w, used) for w in spec["retell"]["voices"]]
            r = http.post("/create-retell-llm", json=llm_payload(spec))
            r.raise_for_status()
            llm_id = r.json()["llm_id"]
            r = http.post("/create-agent", json=agent_payload(spec, llm_id, voice_ids[0]))
            r.raise_for_status()
            state[spec["id"]] = {"llm_id": llm_id, "agent_id": r.json()["agent_id"], "voice_ids": voice_ids}
            print(f"created {spec['id']} -> agent {state[spec['id']]['agent_id']}", file=sys.stderr)
    save_state(STATE, state)
    return []


def cmd_call(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs = load_specs(args.agents)
    variants = parse_variants(args.variants, DEFAULT_VARIANTS)
    state = load_state(STATE)
    to = args.to or os.environ.get("GUARDED_NUMBER", "")
    from_number = os.environ.get("RETELL_FROM_NUMBER", "")
    plan = []
    for spec in specs:
        check_variants(spec, variants)
        st = state.get(spec["id"], {})
        for v in variants:
            voice_ids = st.get("voice_ids")
            voice = voice_ids[v % len(voice_ids)] if voice_ids else placeholder_voice(voice_for(spec, "retell", v))
            body = call_payload(spec, v, st.get("agent_id", f"<agent id for {spec['id']} from `create --live`>"),
                                voice, from_number or "<RETELL_FROM_NUMBER>", to or "<GUARDED_NUMBER>")
            plan.append((spec, v, body))
    if not args.live:
        out = []
        for _, _, body in plan:
            shown = dict(body, to_number=mask(to) or "<GUARDED_NUMBER>", from_number=mask(from_number) or "<RETELL_FROM_NUMBER>")
            out.append({"method": "POST", "url": f"{API}/v2/create-phone-call", "body": shown})
        return out

    if not to or not from_number:
        raise SystemExit("set GUARDED_NUMBER (or --to) and RETELL_FROM_NUMBER")
    missing = [s["id"] for s, _, _ in plan if s["id"] not in state]
    if missing:
        raise SystemExit(f"run `create --live` first; nothing created for {sorted(set(missing))}")
    with _client() as http:
        for spec, v, body in plan:
            r = http.post("/v2/create-phone-call", json=body)
            r.raise_for_status()
            call_id = r.json().get("call_id")
            started = datetime.now(timezone.utc).isoformat(timespec="seconds")
            print(f"placed {spec['id']} v{v} -> call {call_id}", file=sys.stderr)
            status = _wait_ended(http, call_id, spec["max_duration_s"] + 90)
            append_ledger({"platform": "retell", "agent": spec["id"], "kind": spec["kind"], "variant": v,
                           "voice": body["agent_override"]["agent"]["voice_id"], "provider_call_id": call_id,
                           "started_utc": started, "final_status": status})
            time.sleep(args.gap_s)
    return []


def _wait_ended(http, call_id: str | None, timeout_s: int) -> str:
    if not call_id:
        return "unknown"
    deadline = time.time() + timeout_s
    status = "unknown"
    while time.time() < deadline:
        r = http.get(f"/v2/get-call/{call_id}")
        if r.status_code == 200:
            status = r.json().get("call_status", "unknown")
            if status in ("ended", "error", "not_connected"):
                return status
        time.sleep(5)
    return status


def main(argv: list[str] | None = None) -> int:
    load_env()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("voices")
    for name in ("create", "call"):
        s = sub.add_parser(name)
        s.add_argument("--agents", type=lambda x: x.split(","), default=None, help="comma list of agent ids (default: all)")
        s.add_argument("--live", action="store_true", help="actually send the requests (default: dry run)")
    sub.choices["create"].add_argument("--recreate", action="store_true")
    sub.choices["call"].add_argument("--variants", default=None, help="e.g. 5-9 or 5,8 (default 5-9)")
    sub.choices["call"].add_argument("--to", default=None, help="guarded number, E.164 (default env GUARDED_NUMBER)")
    sub.choices["call"].add_argument("--gap-s", type=float, default=20.0, help="pause between calls")
    args = p.parse_args(argv)
    if args.cmd == "voices":
        cmd_voices(args)
        return 0
    out = cmd_create(args) if args.cmd == "create" else cmd_call(args)
    if not args.live:
        print(json.dumps({"dry_run": True, "platform": "retell", "requests": out}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

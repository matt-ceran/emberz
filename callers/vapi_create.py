"""Create the Emberz test-caller assistants on Vapi and place outbound calls to the guarded number.

DRY RUN BY DEFAULT: prints the request payloads as JSON and sends nothing. --live sends them.

  python -m callers.vapi_create create [--agents bank_fraud,tech_support]          # print payloads
  python -m callers.vapi_create create --live                                       # create assistants
  python -m callers.vapi_create call --agents bank_fraud --variants 0-4             # print payloads
  python -m callers.vapi_create call --agents bank_fraud --variants 0-4 --live      # place calls, one at a time

Env: VAPI_API_KEY, VAPI_PHONE_NUMBER_ID (the Vapi number that dials out), GUARDED_NUMBER (E.164).
Vapi's free numbers allow about 10 outbound calls per day per org (support threads, not docs:
https://support.vapi.ai/t/27050183/clarification-on-usage-of-free-vapi-phone-numbers).

API (checked 2026-11-02):
  POST https://api.vapi.ai/assistant   https://docs.vapi.ai/api-reference/assistants/create
  POST https://api.vapi.ai/call        https://docs.vapi.ai/api-reference/calls/create
  GET  https://api.vapi.ai/call/{id}   https://docs.vapi.ai/api-reference/calls/get
  {{var}} + assistantOverrides.variableValues: https://docs.vapi.ai/assistants/dynamic-variables
  firstMessage unset = "assistant will wait for user to speak and use the model to respond" (OpenAPI
  doc comment, mirrored at https://docs.rs/crate/vapi-client/latest/source/src/models/create_assistant_dto.rs);
  silenceTimeoutSeconds is in the same DTO.
  Voice providers used: "openai" (model gpt-4o-mini-tts) and "deepgram" (model aura); never 11labs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any

from callers.common import (append_ledger, check_variants, full_prompt, load_env, load_specs, load_state, mask,
                            parse_variants, require_env, save_state, voice_for)

API = "https://api.vapi.ai"
STATE = "vapi_assistants.json"
DEFAULT_VARIANTS = list(range(0, 5))   # variants 0-4 on Vapi, 5-9 on Retell (see README)


def assistant_payload(spec: dict[str, Any]) -> dict[str, Any]:
    model = dict(spec["vapi"]["model"])
    model["messages"] = [{"role": "system", "content": full_prompt(spec)}]
    return {
        "name": f"emberz-{spec['id']}"[:40],
        # No firstMessage: the line plays a notice and rings the protected phone before anyone can
        # hear the agent, so it waits for "hello" and the model opens with first_message (see
        # common.CALL_START). firstMessageMode values: https://docs.vapi.ai/api-reference/assistants/create
        "firstMessageMode": "assistant-waits-for-user",
        "silenceTimeoutSeconds": 60,       # the protected phone may ring ~20 s after the notice
        "model": model,
        "voice": spec["vapi"]["voices"][0],
        "transcriber": {"provider": "deepgram", "model": "nova-3", "language": "en"},
        "maxDurationSeconds": spec["max_duration_s"],
        "backgroundSound": "off",          # default "office" for phone calls would pollute V1 audio
        "endCallPhrases": [spec["end_phrase"]],
        "metadata": {"project": "emberz", "agent": spec["id"], "kind": spec["kind"]},
    }


def call_payload(spec: dict[str, Any], variant: int, assistant_id: str, phone_number_id: str, to: str) -> dict[str, Any]:
    return {
        "assistantId": assistant_id,
        "phoneNumberId": phone_number_id,
        "customer": {"number": to},
        "assistantOverrides": {
            "variableValues": spec["variants"][variant],
            "voice": voice_for(spec, "vapi", variant),
        },
        "name": f"emberz-{spec['id']}-v{variant}",
    }


def _client():
    import httpx

    return httpx.Client(base_url=API, timeout=30.0, headers={"Authorization": f"Bearer {require_env('VAPI_API_KEY')}"})


def cmd_create(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs = load_specs(args.agents)
    requests = [{"method": "POST", "url": f"{API}/assistant", "body": assistant_payload(s)} for s in specs]
    if not args.live:
        return requests
    state = load_state(STATE)
    with _client() as http:
        for spec, req in zip(specs, requests):
            if spec["id"] in state and not args.recreate:
                print(f"skip {spec['id']}: already created ({state[spec['id']]})", file=sys.stderr)
                continue
            r = http.post("/assistant", json=req["body"])
            r.raise_for_status()
            state[spec["id"]] = r.json()["id"]
            print(f"created {spec['id']} -> {state[spec['id']]}", file=sys.stderr)
    save_state(STATE, state)
    return requests


def cmd_call(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs = load_specs(args.agents)
    variants = parse_variants(args.variants, DEFAULT_VARIANTS)
    state = load_state(STATE)
    to = args.to or os.environ.get("GUARDED_NUMBER", "")
    phone_id = os.environ.get("VAPI_PHONE_NUMBER_ID", "")
    plan = []
    for spec in specs:
        check_variants(spec, variants)
        for v in variants:
            body = call_payload(
                spec, v,
                assistant_id=state.get(spec["id"], f"<assistant id for {spec['id']} from `create --live`>"),
                phone_number_id=phone_id or "<VAPI_PHONE_NUMBER_ID>",
                to=to or "<GUARDED_NUMBER>",
            )
            plan.append((spec, v, body))
    if not args.live:
        out = []
        for _, _, body in plan:
            shown = json.loads(json.dumps(body))
            shown["customer"]["number"] = mask(to) or "<GUARDED_NUMBER>"
            out.append({"method": "POST", "url": f"{API}/call", "body": shown})
        return out

    if not to or not phone_id:
        raise SystemExit("set GUARDED_NUMBER (or --to) and VAPI_PHONE_NUMBER_ID")
    missing = [s["id"] for s, _, _ in plan if s["id"] not in state]
    if missing:
        raise SystemExit(f"run `create --live` first; no assistant id for {sorted(set(missing))}")
    with _client() as http:
        for spec, v, body in plan:
            r = http.post("/call", json=body)
            r.raise_for_status()
            call = r.json()
            started = datetime.now(timezone.utc).isoformat(timespec="seconds")
            print(f"placed {spec['id']} v{v} -> call {call.get('id')}", file=sys.stderr)
            status = _wait_ended(http, call.get("id"), spec["max_duration_s"] + 90)
            append_ledger({"platform": "vapi", "agent": spec["id"], "kind": spec["kind"], "variant": v,
                           "voice": body["assistantOverrides"]["voice"], "provider_call_id": call.get("id"),
                           "started_utc": started, "final_status": status})
            time.sleep(args.gap_s)
    return []


def _wait_ended(http, call_id: str | None, timeout_s: int) -> str:
    if not call_id:
        return "unknown"
    deadline = time.time() + timeout_s
    status = "unknown"
    while time.time() < deadline:
        r = http.get(f"/call/{call_id}")
        if r.status_code == 200:
            status = r.json().get("status", "unknown")
            if status in ("ended", "not-found"):
                return status
        time.sleep(5)
    return status


def main(argv: list[str] | None = None) -> int:
    load_env()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("create", "call"):
        s = sub.add_parser(name)
        s.add_argument("--agents", type=lambda x: x.split(","), default=None, help="comma list of agent ids (default: all)")
        s.add_argument("--live", action="store_true", help="actually send the requests (default: dry run)")
    sub.choices["create"].add_argument("--recreate", action="store_true")
    sub.choices["call"].add_argument("--variants", default=None, help="e.g. 0-4 or 0,3,7 (default 0-4)")
    sub.choices["call"].add_argument("--to", default=None, help="guarded number, E.164 (default env GUARDED_NUMBER)")
    sub.choices["call"].add_argument("--gap-s", type=float, default=20.0, help="pause between calls")
    args = p.parse_args(argv)
    out = cmd_create(args) if args.cmd == "create" else cmd_call(args)
    if not args.live:
        print(json.dumps({"dry_run": True, "platform": "vapi", "requests": out}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

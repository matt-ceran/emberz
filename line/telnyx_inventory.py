"""Read-only Telnyx inventory (GET requests only; nothing is changed or bought).

Prints: balance, Call Control apps (name, id, webhook URL), phone numbers (and which app each
uses), outbound voice profiles, and the demo app's outbound profile + webhook config.

Run:
  TELNYX_ENV_FILE="/path/to/AI VOICE AGENT/.env" .venv/bin/python -m line.telnyx_inventory
Endpoints (all GET): https://developers.telnyx.com/api-reference
  /v2/balance, /v2/call_control_applications, /v2/call_control_applications/{id},
  /v2/phone_numbers, /v2/outbound_voice_profiles, /v2/outbound_voice_profiles/{id}
"""
from __future__ import annotations

import sys

import httpx

from line.config import DEMO_APP_ID, load_settings

API = "https://api.telnyx.com/v2"


def _get(http: httpx.Client, path: str, **params) -> dict:
    r = http.get(path, params=params)
    if r.status_code >= 400:
        return {"_error": f"HTTP {r.status_code}", "data": []}
    return r.json()


def _list_all(http: httpx.Client, path: str) -> list[dict]:
    out, page = [], 1
    while True:
        body = _get(http, path, **{"page[number]": page, "page[size]": 250})
        if "_error" in body:
            print(f"  ! GET {path}: {body['_error']}")
            return out
        out.extend(body.get("data", []))
        meta = body.get("meta", {})
        if page >= int(meta.get("total_pages", 1) or 1):
            return out
        page += 1


def main() -> int:
    s = load_settings()
    if not s.telnyx_api_key:
        print("TELNYX_API_KEY is not set (set TELNYX_ENV_FILE to EmberDial's .env path)")
        return 2
    headers = {"Authorization": f"Bearer {s.telnyx_api_key}"}
    with httpx.Client(base_url=API, headers=headers, timeout=15.0) as http:
        bal = _get(http, "/balance").get("data", {})
        print(f"Balance: {bal.get('balance')} {bal.get('currency')} "
              f"(available credit {bal.get('available_credit')}, pending {bal.get('pending')})")

        apps = _list_all(http, "/call_control_applications")
        print(f"\nCall Control apps: {len(apps)}")
        for a in apps:
            print(f"  - {a.get('application_name')!r} id={a.get('id')} active={a.get('active')} "
                  f"webhook={a.get('webhook_event_url')} api_v={a.get('webhook_api_version')}")

        nums = _list_all(http, "/phone_numbers")
        names = {a.get("id"): a.get("application_name") for a in apps}
        print(f"\nPhone numbers: {len(nums)}")
        for n in nums:
            cid = n.get("connection_id")
            print(f"  - {n.get('phone_number')} status={n.get('status')} "
                  f"connection={cid} ({names.get(cid, 'not a Call Control app' if cid else 'none')})")

        ovps = _list_all(http, "/outbound_voice_profiles")
        print(f"\nOutbound voice profiles: {len(ovps)}")
        for p in ovps:
            print(f"  - {p.get('name')!r} id={p.get('id')} enabled={p.get('enabled')} "
                  f"traffic={p.get('traffic_type')} service_plan={p.get('service_plan')} "
                  f"whitelisted={p.get('whitelisted_destinations')} "
                  f"max_spend={p.get('daily_spend_limit')} (enabled={p.get('daily_spend_limit_enabled')})")

        print(f"\nDemo app {DEMO_APP_ID}:")
        demo = _get(http, f"/call_control_applications/{DEMO_APP_ID}")
        if "_error" in demo:
            print(f"  ! {demo['_error']}")
            return 1
        d = demo.get("data", {})
        print(f"  name={d.get('application_name')!r} active={d.get('active')}")
        print(f"  webhook_event_url={d.get('webhook_event_url')}")
        print(f"  webhook_event_failover_url={d.get('webhook_event_failover_url')!r} "
              f"api_version={d.get('webhook_api_version')} timeout_secs={d.get('webhook_timeout_secs')}")
        print(f"  first_command_timeout={d.get('first_command_timeout')} "
              f"({d.get('first_command_timeout_secs')} s) anchorsite={d.get('anchorsite_override')}")
        inbound, outbound = d.get("inbound") or {}, d.get("outbound") or {}
        print(f"  inbound={ {k: inbound.get(k) for k in ('channel_limit', 'sip_subdomain', 'shaken_stir_enabled')} }")
        ovp_id = outbound.get("outbound_voice_profile_id")
        print(f"  outbound channel_limit={outbound.get('channel_limit')} outbound_voice_profile_id={ovp_id}")
        if ovp_id:
            p = _get(http, f"/outbound_voice_profiles/{ovp_id}").get("data", {})
            print(f"  outbound profile: {p.get('name')!r} enabled={p.get('enabled')} "
                  f"traffic={p.get('traffic_type')} service_plan={p.get('service_plan')} "
                  f"whitelisted={p.get('whitelisted_destinations')} "
                  f"daily_spend_limit={p.get('daily_spend_limit')} (enabled={p.get('daily_spend_limit_enabled')}) "
                  f"concurrent_call_limit={p.get('concurrent_call_limit')}")
        else:
            print("  ! no outbound voice profile: the app cannot dial B or C until one is attached")
        demo_nums = [n.get("phone_number") for n in nums if n.get("connection_id") == DEMO_APP_ID]
        print(f"  numbers on the demo app: {demo_nums or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

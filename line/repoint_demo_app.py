"""Point the demo Call Control app's webhook at our tunnel. DRY RUN unless --apply.

Safety: GETs the app first and refuses unless application_name == "EmberDial Jev Demo".
Production apps (Dialer, EmberDial Private Browser Ingress, ...) can never be touched.

  .venv/bin/python -m line.repoint_demo_app --url https://xyz.trycloudflare.com          # dry run
  .venv/bin/python -m line.repoint_demo_app --url https://xyz.trycloudflare.com --apply  # PATCH
  add --activate to also set active=true (the app showed active=False on 2026-11-02)

PATCH /v2/call_control_applications/{id} requires application_name and webhook_event_url
(OpenAPI spec: https://raw.githubusercontent.com/team-telnyx/openapi/master/openapi/spec3.json,
https://developers.telnyx.com/api-reference/call-control-applications/update-a-call-control-application).
"""
from __future__ import annotations

import argparse
import sys

import httpx

from line.config import DEMO_APP_ID, DEMO_APP_NAME, load_settings

API = "https://api.telnyx.com/v2"
FIELDS = ("application_name", "active", "webhook_event_url", "webhook_event_failover_url",
          "webhook_api_version", "webhook_timeout_secs")


def show(label: str, app: dict) -> None:
    print(label)
    for k in FIELDS:
        print(f"  {k}: {app.get(k)!r}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True, help="public base URL, e.g. https://xyz.trycloudflare.com")
    ap.add_argument("--app-id", default=DEMO_APP_ID)
    ap.add_argument("--apply", action="store_true", help="really PATCH (default: dry run)")
    ap.add_argument("--activate", action="store_true",
                    help="also set active=true (inventory on 2026-11-02 showed the demo app active=False; "
                         "spec: 'Specifies whether the connection can be used')")
    args = ap.parse_args(argv)

    if not args.url.startswith("https://"):
        print("refusing: --url must start with https://")
        return 2
    webhook = args.url.rstrip("/") + "/webhooks/telnyx"
    s = load_settings()
    if not s.telnyx_api_key:
        print("TELNYX_API_KEY is not set (set TELNYX_ENV_FILE)")
        return 2

    with httpx.Client(base_url=API, timeout=15.0,
                      headers={"Authorization": f"Bearer {s.telnyx_api_key}"}) as http:
        r = http.get(f"/call_control_applications/{args.app_id}")
        if r.status_code >= 400:
            print(f"GET failed: HTTP {r.status_code}")
            return 1
        app = r.json()["data"]
        show("Before:", app)
        if app.get("application_name") != DEMO_APP_NAME:
            print(f"REFUSING: app {args.app_id} is {app.get('application_name')!r}, not {DEMO_APP_NAME!r}")
            return 3
        body = {"application_name": DEMO_APP_NAME, "webhook_event_url": webhook, "webhook_api_version": "2"}
        if args.activate:
            body["active"] = True
        if not args.apply:
            print(f"DRY RUN: would PATCH /call_control_applications/{args.app_id} with {body}")
            return 0
        r = http.patch(f"/call_control_applications/{args.app_id}", json=body)
        if r.status_code >= 400:
            print(f"PATCH failed: HTTP {r.status_code}: {r.text[:300]}")
            return 1
        show("After:", r.json()["data"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

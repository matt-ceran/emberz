"""Helpers for the L1 call-flow tests (FakeTelnyx + FastAPI TestClient, no network).

A uniquely named module (not conftest) so `import` cannot collide with other units' test helpers.
"""
from __future__ import annotations

import base64
import itertools
import json
import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient
from nacl.signing import SigningKey

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from line.call_flow import decode_client_state, encode_client_state  # noqa: E402
from line.config import GuardedLine, Settings  # noqa: E402
from line.server import create_app  # noqa: E402
from line.telnyx_client import FakeTelnyx  # noqa: E402

GUARDED = "+15555550100"
PROTECTED = "+15555550101"
CARETAKER = "+15555550102"
CALLER = "+15555550199"
GRANDPA = "+15555550111"
A_CCID = "v3:A-leg"
SESSION_ID = "sess-1111"
CALL_ID = "c_" + SESSION_ID


class RecordingBrain:
    def __init__(self, session):
        self.session = session
        self.media: list[tuple[str, bytes, int]] = []
        self.events: list[tuple[str, dict]] = []
        self.hung_up = False

    async def on_media(self, track, payload, ts_ms):
        self.media.append((track, payload, ts_ms))

    async def on_call_event(self, event_type, payload):
        self.events.append((event_type, payload))

    async def on_hangup(self):
        self.hung_up = True


class RecordingHub:
    def __init__(self):
        self.events = []

    def publish(self, event):
        self.events.append(event)

    def types(self):
        return [e.type for e in self.events]


class RecordingCaretaker:
    def __init__(self):
        self.events = []

    async def on_event(self, session, event_type, payload):
        self.events.append((session.call_id, event_type, payload))


class Line:
    """One app instance plus helpers to post webhooks the way Telnyx does."""

    def __init__(self, signing_key: SigningKey | None = None, skip_verify: bool = True):
        self.telnyx = FakeTelnyx()
        self.hub = RecordingHub()
        self.caretaker = RecordingCaretaker()
        self.brains: list[RecordingBrain] = []
        self.key = signing_key
        pub = base64.b64encode(bytes(signing_key.verify_key)).decode() if signing_key else ""
        self.settings = Settings(
            telnyx_api_key="test", telnyx_public_key=pub,
            public_base_url="https://emberz-test.trycloudflare.com", skip_verify=skip_verify,
        )
        profiles = {GUARDED: GuardedLine(
            profile="elder", rules=["Never share codes"], allowed_contacts={GRANDPA: "Grandpa"},
            protected_number=PROTECTED, caretaker_number=CARETAKER,
        )}

        def make_brain(session, telnyx, hub):
            b = RecordingBrain(session)
            self.brains.append(b)
            return b

        self.app = create_app(telnyx=self.telnyx, make_brain=make_brain, caretaker=self.caretaker,
                              hub=self.hub, settings=self.settings, profiles=profiles)
        self.client = TestClient(self.app)
        self._ids = itertools.count(1)

    @property
    def flow(self):
        return self.app.state.flow

    def body(self, event_type: str, payload: dict, event_id: str | None = None) -> bytes:
        env = {"data": {"record_type": "event", "event_type": event_type,
                        "id": event_id or f"evt-{next(self._ids)}",
                        "occurred_at": "2026-11-02T12:00:00Z", "payload": payload},
               "meta": {"attempt": 1, "delivered_to": "x"}}
        return json.dumps(env).encode()

    def post(self, event_type: str, payload: dict, event_id: str | None = None, ts: int | None = None):
        raw = self.body(event_type, payload, event_id)
        headers = {"content-type": "application/json"}
        if self.key is not None:
            ts = int(time.time()) if ts is None else ts
            sig = self.key.sign(str(ts).encode() + b"|" + raw).signature
            headers |= {"telnyx-signature-ed25519": base64.b64encode(sig).decode(), "telnyx-timestamp": str(ts)}
        return self.client.post("/webhooks/telnyx", content=raw, headers=headers)

    # ---- Telnyx-shaped payloads
    def leg(self, role: str, ccid: str, **extra) -> dict:
        p = {"call_control_id": ccid, "call_leg_id": ccid, "call_session_id": SESSION_ID,
             "connection_id": "3051333815776576529"}
        if role:
            p["client_state"] = encode_client_state(role, CALL_ID)
        return p | extra

    def inbound(self, caller: str = CALLER, to: str = GUARDED, event_id: str | None = None):
        p = {"call_control_id": A_CCID, "call_leg_id": A_CCID, "call_session_id": SESSION_ID,
             "connection_id": "3051333815776576529", "from": caller, "to": to,
             "direction": "incoming", "state": "parked"}
        return self.post("call.initiated", p, event_id)

    def b_ccid(self) -> str:
        return self.flow.sessions[CALL_ID].b_ccid

    def calls(self, kind: str, action: str | None = None):
        return [e for e in self.telnyx.log if e[0] == kind and (action is None or e[2] == action)]


__all__ = ["decode_client_state"]

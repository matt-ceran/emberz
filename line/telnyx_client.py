"""Thin async Telnyx Call Control client (shared by L1, L2, L3).

Generic on purpose: every Call Control command is POST /v2/calls/{id}/actions/{name}
and every conference command is POST /v2/conferences/{id}/actions/{name}, so callers
pass the documented body fields as kwargs. `FakeTelnyx` records calls for tests.
"""
from __future__ import annotations

import os
import uuid
from typing import Any

import httpx

API = "https://api.telnyx.com/v2"


class TelnyxClient:
    def __init__(self, api_key: str | None = None, connection_id: str | None = None):
        self.api_key = api_key or os.environ["TELNYX_API_KEY"]
        self.connection_id = connection_id or os.environ.get("TELNYX_CONNECTION_ID", "")
        self._http = httpx.AsyncClient(
            base_url=API, timeout=10.0, headers={"Authorization": f"Bearer {self.api_key}"}
        )

    async def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        body = {k: v for k, v in body.items() if v is not None}
        body.setdefault("command_id", str(uuid.uuid4()))  # idempotency on Telnyx's side
        r = await self._http.post(path, json=body)
        if r.status_code >= 400:
            raise TelnyxError(r.status_code, path, r.text)
        return r.json().get("data", {}) if r.content else {}

    async def call_action(self, call_control_id: str, action: str, **body: Any) -> dict[str, Any]:
        return await self._post(f"/calls/{call_control_id}/actions/{action}", body)

    async def dial(self, **body: Any) -> dict[str, Any]:
        body.setdefault("connection_id", self.connection_id)
        body.pop("command_id", None)
        r = await self._http.post("/calls", json={k: v for k, v in body.items() if v is not None})
        if r.status_code >= 400:
            raise TelnyxError(r.status_code, "/calls", r.text)
        return r.json()["data"]

    async def conference_create(self, **body: Any) -> dict[str, Any]:
        return await self._post("/conferences", body)

    async def conference_action(self, conference_id: str, action: str, **body: Any) -> dict[str, Any]:
        return await self._post(f"/conferences/{conference_id}/actions/{action}", body)

    async def get(self, path: str, **params: Any) -> dict[str, Any]:
        r = await self._http.get(path, params=params)
        r.raise_for_status()
        return r.json()

    async def aclose(self) -> None:
        await self._http.aclose()


class TelnyxError(RuntimeError):
    def __init__(self, status: int, path: str, text: str):
        super().__init__(f"Telnyx {status} on {path}: {text[:500]}")
        self.status, self.path, self.text = status, path, text


class FakeTelnyx:
    """Records every command instead of sending it. Same method names as TelnyxClient."""

    def __init__(self) -> None:
        self.log: list[tuple[str, str | None, str | None, dict[str, Any]]] = []

    async def call_action(self, call_control_id, action, **body):
        self.log.append(("call", call_control_id, action, body))
        return {}

    async def dial(self, **body):
        ccid = f"fake_ccid_{len(self.log)}"
        self.log.append(("dial", None, None, body))
        return {"call_control_id": ccid, "call_leg_id": ccid, "call_session_id": ccid}

    async def conference_create(self, **body):
        self.log.append(("conference_create", None, None, body))
        return {"id": "fake_conf_1"}

    async def conference_action(self, conference_id, action, **body):
        self.log.append(("conference", conference_id, action, body))
        return {}

    async def get(self, path, **params):
        return {"data": []}

    async def aclose(self):
        pass

    def actions(self) -> list[str]:
        return [a or kind for kind, _, a, _ in self.log]

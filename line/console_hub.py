"""Console backend (L3): fan-out of `console_event`s to the caretaker console (C1).

- `ConsoleHub.publish(event)` is sync and never blocks: the line (L1/L2) and the
  caretaker flow call it from anywhere, including from worker threads.
- Every event is kept in a per-call replay buffer, so a page that connects late
  (or reconnects) first gets the whole call so far, then the live stream.
- `console_router(hub, caretaker)` exposes it over FastAPI:
    WS   /console/ws[?call_id=...]             replay, then live events (JSON = ConsoleEvent)
    GET  /console/calls                        one summary row per call
    GET  /console/calls/{call_id}/events       the replay buffer of one call
    POST /console/calls/{call_id}/barge        the "join" button -> caretaker.barge(session)
- `add_cors(app)` lets a page served from another origin (Matt's dev server) call the HTTP routes.

Access token: every route needs `?token=<CONSOLE_TOKEN>` (or an `X-Console-Token` header on
HTTP). The line server is reachable through the public cloudflared tunnel and the console
carries live call transcripts, so it FAILS CLOSED: with CONSOLE_TOKEN unset, every route
answers 503 (WS close 1008). Only `console_router(..., open_access=True)` (the fixture player,
tests) skips the token.
"""
from __future__ import annotations

import asyncio
import hmac
import itertools
import logging
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

from fastapi import APIRouter, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from contracts.models import RISK_ORDER, ConsoleEvent

log = logging.getLogger("emberz.console")

QUEUE_MAX = 1000          # per subscriber; a stuck page loses its oldest pending events, never blocks the line
MAX_EVENTS_PER_CALL = 5000
MAX_CALLS = 200           # oldest calls are dropped from memory past this


@dataclass(eq=False)  # identity hash: kept in a set
class _Subscriber:
    call_id: Optional[str]
    loop: asyncio.AbstractEventLoop
    queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=QUEUE_MAX))
    dropped: int = 0

    def offer(self, item: tuple[int, ConsoleEvent]) -> None:
        """Runs on the subscriber's own loop."""
        if self.queue.full():
            self.queue.get_nowait()
            self.dropped += 1
        self.queue.put_nowait(item)


class Subscription:
    """What `ConsoleHub.subscribe()` returns: the replay plus a live queue."""

    def __init__(self, hub: "ConsoleHub", sub: _Subscriber, replay: list[ConsoleEvent]):
        self._hub, self._sub, self.replay = hub, sub, replay

    async def get(self) -> ConsoleEvent:
        _, event = await self._sub.queue.get()
        return event

    def close(self) -> None:
        self._hub._unsubscribe(self._sub)

    def __enter__(self) -> "Subscription":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class ConsoleHub:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._seq = itertools.count(1)
        self._calls: "OrderedDict[str, list[tuple[int, ConsoleEvent]]]" = OrderedDict()
        self._subs: set[_Subscriber] = set()

    # ------------------------------------------------------------- publish
    def publish(self, event: ConsoleEvent | dict) -> ConsoleEvent:
        """Validate, buffer and fan out one event. Sync, non-blocking, thread-safe."""
        if not isinstance(event, ConsoleEvent):
            event = ConsoleEvent.model_validate(event)
        with self._lock:
            item = (next(self._seq), event)
            buf = self._calls.setdefault(event.call_id, [])
            buf.append(item)
            if len(buf) > MAX_EVENTS_PER_CALL:
                del buf[0]
            while len(self._calls) > MAX_CALLS:
                self._calls.popitem(last=False)
            targets = [s for s in self._subs if s.call_id in (None, event.call_id)]
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        for sub in targets:
            if sub.loop is running:
                sub.offer(item)
            else:
                try:
                    sub.loop.call_soon_threadsafe(sub.offer, item)
                except RuntimeError:  # that loop is closed; the subscriber is gone
                    self._unsubscribe(sub)
        return event

    def emit(self, call_id: str, type_: str, at_ms: int, **data: Any) -> ConsoleEvent:
        """Shorthand: hub.emit(call_id, "coach", session.now_ms(), text=..., turn=...)."""
        return self.publish(ConsoleEvent(type=type_, call_id=call_id, at_ms=max(0, int(at_ms)), data=data))

    # ------------------------------------------------------------- subscribe
    def subscribe(self, call_id: Optional[str] = None) -> Subscription:
        """Must be called from inside a running event loop. Replay and live stream never overlap or gap."""
        sub = _Subscriber(call_id=call_id, loop=asyncio.get_running_loop())
        with self._lock:
            if call_id is None:
                items = sorted(itertools.chain.from_iterable(self._calls.values()), key=lambda it: it[0])
            else:
                items = list(self._calls.get(call_id, []))
            self._subs.add(sub)
        return Subscription(self, sub, [e for _, e in items])

    def _unsubscribe(self, sub: _Subscriber) -> None:
        with self._lock:
            self._subs.discard(sub)

    # ------------------------------------------------------------- reads
    def events(self, call_id: str) -> Optional[list[ConsoleEvent]]:
        with self._lock:
            buf = self._calls.get(call_id)
            return None if buf is None else [e for _, e in buf]

    def calls(self) -> list[dict[str, Any]]:
        """One row per call, newest first."""
        with self._lock:
            snapshot = [(cid, [e for _, e in buf]) for cid, buf in self._calls.items()]
        rows = [_call_row(cid, evs) for cid, evs in snapshot]
        rows.reverse()
        return rows

    def clear(self, call_id: Optional[str] = None) -> None:
        with self._lock:
            if call_id is None:
                self._calls.clear()
            else:
                self._calls.pop(call_id, None)


def _call_row(call_id: str, events: list[ConsoleEvent]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "call_id": call_id, "n_events": len(events), "last_at_ms": events[-1].at_ms if events else 0,
        "started": None, "max_risk": "none", "alerts": 0, "ai_score": None,
        "caretaker": None, "summary": None, "ended": None,
    }
    for e in events:
        if e.type == "call_started":
            row["started"] = e.data
        elif e.type == "risk":
            # line/brain.py adds display_risk = the rule layer's verdict (e.g. capped to none for an
            # allowed contact); prefer it over the raw model risk.
            r = e.data.get("display_risk") or (e.data.get("response") or {}).get("risk", "none")
            if RISK_ORDER.get(r, 0) > RISK_ORDER[row["max_risk"]]:
                row["max_risk"] = r
        elif e.type == "alert":
            row["alerts"] += 1
        elif e.type == "voice":
            row["ai_score"] = e.data.get("ai_score")
        elif e.type == "caretaker_joined":
            row["caretaker"] = e.data.get("mode")
        elif e.type == "summary":
            row["summary"] = e.data.get("text")
        elif e.type == "call_ended":
            row["ended"] = e.data
    return row


# ------------------------------------------------------------------ HTTP / WebSocket
def _token_status(given: Optional[str], open_access: bool) -> int:
    """200 = ok, 401 = wrong/missing token, 503 = no CONSOLE_TOKEN configured (fail closed)."""
    if open_access:
        return 200
    want = os.environ.get("CONSOLE_TOKEN", "")
    if not want:
        return 503
    return 200 if given is not None and hmac.compare_digest(given.encode(), want.encode()) else 401


def console_router(hub: ConsoleHub, caretaker: Any = None, *, open_access: bool = False) -> APIRouter:
    """`caretaker` is a CaretakerFlow (or anything with `.sessions_lookup(call_id)` and `async barge(session)`).
    `open_access=True` skips the token (fixture player and tests only; never on the tunnel)."""
    router = APIRouter()
    if not open_access and not os.environ.get("CONSOLE_TOKEN"):
        log.error("CONSOLE_TOKEN is not set: the console routes will refuse every request (fail closed)")

    def _check_http(request: Request) -> None:
        code = _token_status(request.query_params.get("token") or request.headers.get("x-console-token"), open_access)
        if code == 503:
            raise HTTPException(503, "console disabled: set CONSOLE_TOKEN on the line server")
        if code != 200:
            raise HTTPException(401, "bad or missing console token")

    @router.websocket("/console/ws")
    async def console_ws(ws: WebSocket, call_id: Optional[str] = None, token: Optional[str] = None):
        if _token_status(token, open_access) != 200:
            await ws.close(code=1008)
            return
        await ws.accept()
        sub = hub.subscribe(call_id)
        try:
            for event in sub.replay:
                await ws.send_text(event.model_dump_json())

            async def pump() -> None:
                while True:
                    event = await sub.get()
                    await ws.send_text(event.model_dump_json())

            async def drain() -> None:  # we ignore what the page sends; this just notices the disconnect
                while True:
                    await ws.receive_text()

            tasks = [asyncio.create_task(pump()), asyncio.create_task(drain())]
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
            for t in done:
                exc = t.exception()
                if exc and not isinstance(exc, (WebSocketDisconnect, RuntimeError)):
                    log.warning("console ws ended: %r", exc)
        except WebSocketDisconnect:
            pass
        finally:
            sub.close()

    @router.get("/console/calls")
    async def list_calls(request: Request):
        _check_http(request)
        return hub.calls()

    @router.get("/console/calls/{call_id}/events")
    async def call_events(call_id: str, request: Request):
        _check_http(request)
        events = hub.events(call_id)
        if events is None:
            raise HTTPException(404, "unknown call_id")
        return [e.model_dump() for e in events]

    @router.post("/console/calls/{call_id}/barge")
    async def barge(call_id: str, request: Request):
        _check_http(request)
        if caretaker is None:
            raise HTTPException(503, "no caretaker flow attached")
        session = caretaker.sessions_lookup(call_id)
        if session is None:
            raise HTTPException(404, "unknown or finished call")
        try:
            state = await caretaker.barge(session)
        except BargeNotAllowed as e:
            raise HTTPException(409, str(e))
        except Exception as e:  # Telnyx rejected the update; the page can let Mom try again
            log.error("barge on %s failed: %s", call_id, e)
            raise HTTPException(502, "the phone system rejected the join; try again")
        return {"call_id": call_id, "caretaker_state": state}

    return router


class BargeNotAllowed(RuntimeError):
    """Raised by caretaker.barge when the caretaker isn't listening (yet)."""


def add_cors(app: FastAPI) -> None:
    """Allow the console page to be served from another origin. CONSOLE_ALLOWED_ORIGINS=comma list, default *."""
    origins = [o.strip() for o in os.environ.get("CONSOLE_ALLOWED_ORIGINS", "*").split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"], allow_credentials=False,
    )

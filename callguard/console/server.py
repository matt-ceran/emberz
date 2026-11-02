"""Emberz caretaker console server.

Run from callguard/:
    uv run --with fastapi --with 'uvicorn[standard]' uvicorn console.server:app --host 0.0.0.0 --port 8095

It exposes the same console routes as Numan's line server and fixture player (line/console_hub.py on
origin/numan/line-voice-gcp), so the page works against either backend, plus held-out-call replay as an extra.

Numan-compatible routes
    WS   /console/ws[?call_id=]               every buffered event (all calls, publish order), then live events
    GET  /console/calls                       one summary row per call, newest first (same keys as his _call_row)
    GET  /console/calls/{call_id}/events      the buffered events of one call (404 if unknown)
    POST /console/calls/{call_id}/barge       the Join button: 200 {call_id, caretaker_state: "barged"},
                                              409 not listening yet / call ended, 404 unknown call
    GET  /fixtures                            the demo presets as "scenes"
    POST /fixtures/play/{scene}?speed=        replay a preset now; returns {scene, call_id}
Console extras
    GET  /                                    the page (static/index.html), pointed at this server
    POST /events                              optional ingest: one console_event or a list, fanned out to browsers
    POST /actions/{call_id}/{action}          listen | join | hangup | safe (see README)
    GET  /replay?call_id=&speed=[&force=1]    replay any held-out call with the guardian's real eval predictions
    GET  /presets                             demo presets picked from the data
    GET  /actions, GET /health

Env
    CONSOLE_PREDS        guardian-v1 (default) | guardian-v2 | path/to/preds.jsonl
    CONSOLE_CALLS        data/calls_test_heldout.jsonl (default)
    CONSOLE_TOKEN        if set, every console route needs ?token= (or X-Console-Token); unset = open (trusted LAN)
    CONSOLE_ALLOWED_ORIGINS  CORS origins, comma list (default *)
    LINE_CONSOLE_URL     Numan's line server (e.g. https://<tunnel>); if set, its /console/ws is bridged in and
                         Join on those calls is forwarded to its /console/calls/{id}/barge
    LINE_CONSOLE_TOKEN   token for LINE_CONSOLE_URL (defaults to CONSOLE_TOKEN)
    LINE_ACTION_URL      optional: where Hang up / Mark safe on a live call are POSTed {call_id, action, at_ms}
"""
from __future__ import annotations

import asyncio
import hmac
import itertools
import json
import logging
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import Body, Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from . import replay as rp

log = logging.getLogger("uvicorn.error")  # shows up in the uvicorn console output
STATIC = Path(__file__).resolve().parent / "static"
RISK_ORDER = {"none": 0, "watch": 1, "alert": 2, "urgent": 3}
QUEUE_MAX = 1000
MAX_EVENTS_PER_CALL = 5000
MAX_CALLS = 200
LIVE_STALE_S = 120  # a live call with no event for this long no longer blocks replays

ConsoleEventType = Literal[
    "call_started", "turn", "risk", "alert", "coach", "caretaker_joined", "summary", "call_ended", "voice"
]


class ConsoleEvent(BaseModel):
    """Mirror of contracts/models.py ConsoleEvent (origin/numan/line-voice-gcp)."""

    model_config = ConfigDict(extra="forbid")
    type: ConsoleEventType
    call_id: str
    at_ms: int = Field(ge=0, description="ms since call start")
    data: dict = Field(default_factory=dict)


# ------------------------------------------------------------------ hub (per-call buffers, like his ConsoleHub)
@dataclass(eq=False)
class Sub:
    call_id: Optional[str]
    queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=QUEUE_MAX))

    def offer(self, item: dict) -> None:
        if self.queue.full():  # a stuck page loses its oldest pending events; the hub never blocks
            self.queue.get_nowait()
        self.queue.put_nowait(item)


@dataclass
class CallMeta:
    origin: str                      # "replay" | "push" (POST /events) | "line" (bridged from LINE_CONSOLE_URL)
    t0: float                        # monotonic time of at_ms = 0
    scale: float = 1.0
    ended: bool = False
    end_at: Optional[int] = None
    caretaker: Optional[str] = None  # None | monitor | barge (from caretaker_joined events)
    safe: bool = False
    last: float = field(default_factory=time.monotonic)


class Hub:
    """Everything runs on the server's event loop, so publish/subscribe are plain sync calls."""

    def __init__(self) -> None:
        self.calls: "OrderedDict[str, list[tuple[int, dict]]]" = OrderedDict()
        self.meta: dict[str, CallMeta] = {}
        self.subs: set[Sub] = set()
        self.seq = itertools.count(1)
        self.replays: dict[str, asyncio.Task] = {}
        self.replay_runs: dict[str, itertools.count] = {}
        self.actions: list[dict] = []
        self.bridge_state = "off"

    def publish(self, ev: dict, *, origin: str = "push", scale: float = 1.0) -> int:
        cid = ev["call_id"]
        if ev["type"] == "call_started" or cid not in self.meta:
            if ev["type"] == "call_started":
                self.calls.pop(cid, None)  # a restarted call starts a clean buffer
            self.meta[cid] = CallMeta(origin=origin, t0=time.monotonic() - ev["at_ms"] / 1000 / scale, scale=scale)
        m = self.meta[cid]
        m.last = time.monotonic()
        if ev["type"] == "call_ended":
            m.ended, m.end_at = True, ev["at_ms"]
        elif ev["type"] == "caretaker_joined":
            m.caretaker = (ev.get("data") or {}).get("mode")
        item = (next(self.seq), ev)
        buf = self.calls.setdefault(cid, [])
        buf.append(item)
        if len(buf) > MAX_EVENTS_PER_CALL:
            del buf[0]
        while len(self.calls) > MAX_CALLS:
            old, _ = self.calls.popitem(last=False)
            self.meta.pop(old, None)
        n = 0
        for s in list(self.subs):
            if s.call_id in (None, cid):
                s.offer(ev)
                n += 1
        return n

    def subscribe(self, call_id: Optional[str] = None) -> tuple[Sub, list[dict]]:
        sub = Sub(call_id)
        if call_id is None:
            items = sorted(itertools.chain.from_iterable(self.calls.values()), key=lambda it: it[0])
        else:
            items = list(self.calls.get(call_id, []))
        self.subs.add(sub)
        return sub, [e for _, e in items]

    def events(self, call_id: str) -> Optional[list[dict]]:
        buf = self.calls.get(call_id)
        return None if buf is None else [e for _, e in buf]

    def at_ms(self, call_id: str) -> int:
        m = self.meta.get(call_id)
        if not m:
            return 0
        ms = max(0, int((time.monotonic() - m.t0) * 1000 * m.scale))
        return min(ms, m.end_at) if m.end_at is not None else ms

    def live_call(self) -> Optional[str]:
        """A non-replay call that started, hasn't ended, and is still sending events."""
        now = time.monotonic()
        for cid, m in reversed(list(self.meta.items())):
            if m.origin != "replay" and not m.ended and now - m.last < LIVE_STALE_S:
                return cid
        return None

    def stop_replays(self) -> list[str]:
        """Cancel running replays; each gets call_ended {reason: replay_stopped} so no call is left 'in progress'."""
        stopped = []
        for cid, task in list(self.replays.items()):
            self.replays.pop(cid, None)
            if task.done():
                continue
            task.cancel()
            stopped.append(cid)
            m = self.meta.get(cid)
            if m and not m.ended:
                at = self.at_ms(cid)
                self.publish({"type": "call_ended", "call_id": cid, "at_ms": at,
                              "data": {"duration_ms": at, "reason": "replay_stopped"}}, origin="replay")
        return stopped

    def row(self, cid: str) -> dict[str, Any]:
        """Same keys as Numan's _call_row, plus origin/replay/safe."""
        evs = self.events(cid) or []
        m = self.meta.get(cid)
        row: dict[str, Any] = {
            "call_id": cid, "n_events": len(evs), "last_at_ms": evs[-1]["at_ms"] if evs else 0,
            "started": None, "max_risk": "none", "alerts": 0, "ai_score": None,
            "caretaker": None, "summary": None, "ended": None,
        }
        for e in evs:
            d = e.get("data") or {}
            if e["type"] == "call_started":
                row["started"] = d
            elif e["type"] == "risk":
                r = d.get("display_risk") or (d.get("response") or {}).get("risk", "none")
                if RISK_ORDER.get(r, 0) > RISK_ORDER[row["max_risk"]]:
                    row["max_risk"] = r
            elif e["type"] == "alert":
                row["alerts"] += 1
            elif e["type"] == "voice":
                row["ai_score"] = d.get("ai_score")
            elif e["type"] == "caretaker_joined":
                row["caretaker"] = d.get("mode")
            elif e["type"] == "summary":
                row["summary"] = d.get("text")
            elif e["type"] == "call_ended":
                row["ended"] = d
        row.update(origin=m.origin if m else None, replay=bool(m and m.origin == "replay"), safe=bool(m and m.safe))
        return row


hub = Hub()
DS: Optional[rp.Dataset] = None


def dataset() -> rp.Dataset:
    global DS
    if DS is None:
        DS = rp.load()
    return DS


# ------------------------------------------------------------------ token (optional here; his line fails closed)
def _token_ok(given: Optional[str]) -> bool:
    want = os.environ.get("CONSOLE_TOKEN", "")
    return not want or (given is not None and hmac.compare_digest(given.encode(), want.encode()))


def check_token(request: Request) -> None:
    if not _token_ok(request.query_params.get("token") or request.headers.get("x-console-token")):
        raise HTTPException(401, "bad or missing console token")


AUTH = [Depends(check_token)]


# ------------------------------------------------------------------ bridge from Numan's line (optional)
def line_url() -> str:
    return os.environ.get("LINE_CONSOLE_URL", "").strip().rstrip("/")


def line_token() -> str:
    return os.environ.get("LINE_CONSOLE_TOKEN") or os.environ.get("CONSOLE_TOKEN") or ""


async def bridge(url: str, token: str) -> None:
    """Subscribe to the line's /console/ws and republish every event here (deduped: a reconnect replays all)."""
    import websockets  # ships with uvicorn[standard]

    ws_url = re.sub(r"^http", "ws", url) + "/console/ws" + (f"?token={urllib.parse.quote(token)}" if token else "")
    seen: set[str] = set()
    delay = 1.0
    while True:
        try:
            async with websockets.connect(ws_url, open_timeout=5) as ws:
                hub.bridge_state, delay, opened = "connected", 1.0, time.monotonic()
                log.info("console bridge: connected to %s/console/ws", url)
                async for msg in ws:
                    raw = msg if isinstance(msg, str) else msg.decode()
                    if raw in seen:
                        continue
                    seen.add(raw)
                    try:
                        ev = ConsoleEvent.model_validate_json(raw).model_dump()
                    except ValidationError as e:
                        log.warning("console bridge: dropped an off-schema event: %s", e.errors()[:1])
                        continue
                    # The connect-time replay of old calls must not stop a running demo replay; a new call must.
                    if ev["type"] == "call_started" and time.monotonic() - opened > 2.0:
                        hub.stop_replays()
                    hub.publish(ev, origin="line")
        except asyncio.CancelledError:
            raise
        except Exception as e:  # line down, tunnel changed, bad token (close 1008): keep retrying
            hub.bridge_state = f"down: {type(e).__name__}: {e}"[:200]
            log.warning("console bridge: %s; retrying in %.0f s", hub.bridge_state, delay)
        await asyncio.sleep(delay)
        delay = min(delay * 2, 15.0)


def _post(url: str, body: Optional[dict], headers: Optional[dict] = None) -> tuple[int, Any]:
    req = urllib.request.Request(url, data=json.dumps(body or {}).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            raw = r.read().decode() or "null"
            status = r.status
    except urllib.error.HTTPError as e:
        raw, status = e.read().decode() or "null", e.code
    try:
        return status, json.loads(raw)
    except json.JSONDecodeError:
        return status, {"detail": raw[:200]}


async def forward_barge(call_id: str) -> dict:
    url = line_url()
    if not url:
        raise HTTPException(503, "no phone line connected to this console (set LINE_CONSOLE_URL)")
    tok = line_token()
    try:
        status, body = await asyncio.to_thread(
            _post, f"{url}/console/calls/{urllib.parse.quote(call_id)}/barge", None,
            {"X-Console-Token": tok} if tok else None)
    except Exception as e:
        raise HTTPException(502, f"could not reach the line server: {e}"[:200])
    if status != 200:
        detail = body.get("detail") if isinstance(body, dict) else body
        raise HTTPException(status, detail or f"line answered {status}")
    return body  # the line emits caretaker_joined {mode: barge} itself; it arrives over the bridge


# ------------------------------------------------------------------ app
@asynccontextmanager
async def lifespan(_app: FastAPI):
    ds = dataset()
    log.info("console: %d calls, %d with preds from %s, presets: %s", len(ds.calls), len(ds.preds),
             ds.preds_path, [p["call_id"] for p in ds.presets])
    task = None
    if line_url():
        hub.bridge_state = "connecting"
        task = asyncio.create_task(bridge(line_url(), line_token()))
    yield
    hub.stop_replays()
    if task:
        task.cancel()


app = FastAPI(title="Emberz caretaker console", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get("CONSOLE_ALLOWED_ORIGINS", "*").split(",") if o.strip()],
    allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"], allow_credentials=False,
)


@app.get("/")
async def index():
    # The file defaults to Numan's fixture player (http://127.0.0.1:8001) when opened on its own;
    # served from here, it defaults to this server. ?backend= overrides either.
    html = (STATIC / "index.html").read_text()
    html = html.replace('<meta name="emberz-backend" content="http://127.0.0.1:8001">',
                        '<meta name="emberz-backend" content="">', 1)
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


async def _serve_ws(ws: WebSocket, call_id: Optional[str]) -> None:
    if not _token_ok(ws.query_params.get("token")):
        await ws.close(code=1008)
        return
    await ws.accept()
    sub, replay = hub.subscribe(call_id)
    try:
        for ev in replay:
            await ws.send_text(json.dumps(ev))

        async def pump() -> None:
            while True:
                await ws.send_text(json.dumps(await sub.queue.get()))

        async def drain() -> None:  # the page never sends; this only notices the disconnect
            while True:
                await ws.receive_text()

        tasks = [asyncio.create_task(pump()), asyncio.create_task(drain())]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
        for t in done:
            t.exception()  # retrieve it so asyncio doesn't log "exception never retrieved"
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        hub.subs.discard(sub)


@app.websocket("/console/ws")
async def console_ws(ws: WebSocket, call_id: Optional[str] = None):
    await _serve_ws(ws, call_id)


@app.websocket("/ws/console")  # first version's path, kept as an alias
async def ws_console_alias(ws: WebSocket, call_id: Optional[str] = None):
    await _serve_ws(ws, call_id)


@app.get("/console/calls", dependencies=AUTH)
async def list_calls():
    return [hub.row(cid) for cid in reversed(list(hub.calls))]


@app.get("/console/calls/{call_id}/events", dependencies=AUTH)
async def call_events(call_id: str):
    evs = hub.events(call_id)
    if evs is None:
        raise HTTPException(404, "unknown call_id")
    return evs


async def do_barge(call_id: str) -> dict:
    m = hub.meta.get(call_id)
    if m is None:
        raise HTTPException(404, "unknown or finished call")
    if m.origin == "line":
        return await forward_barge(call_id)  # the line decides (200/409/404/502) and emits the event
    if m.ended:
        raise HTTPException(409, "the call has ended")
    if m.caretaker == "barge":
        return {"call_id": call_id, "caretaker_state": "barged"}  # a second press is harmless
    if m.caretaker != "monitor":
        raise HTTPException(409, "caretaker is not listening yet (press 1 on the alert call first)")
    if m.origin == "replay":
        hub.publish({"type": "caretaker_joined", "call_id": call_id, "at_ms": hub.at_ms(call_id),
                     "data": {"mode": "barge"}}, origin="replay")
        return {"call_id": call_id, "caretaker_state": "barged", "replay": True}
    return await forward_barge(call_id)  # pushed via /events: only the line can join; it emits the event


@app.post("/console/calls/{call_id}/barge", dependencies=AUTH)
async def barge(call_id: str):
    res = await do_barge(call_id)
    hub.actions.append({"call_id": call_id, "action": "join", "at_ms": hub.at_ms(call_id), "time": time.time(),
                        "result": res})
    return res


# ------------------------------------------------------------------ optional ingest
@app.post("/events", dependencies=AUTH)
async def post_events(payload: Any = Body(...)):
    items = payload if isinstance(payload, list) else [payload]
    try:
        events = [ConsoleEvent.model_validate(x).model_dump() for x in items]
    except ValidationError as e:
        raise HTTPException(422, detail=json.loads(e.json()))
    clients = len(hub.subs)
    for ev in events:
        if ev["type"] == "call_started":
            hub.stop_replays()  # a live call takes the screen
        clients = hub.publish(ev, origin="push")
    return {"ok": True, "accepted": len(events), "clients": clients}


# ------------------------------------------------------------------ caretaker actions
ACTIONS = ("listen", "join", "hangup", "safe")


@app.post("/actions/{call_id}/{action}", dependencies=AUTH)
async def post_action(call_id: str, action: str):
    if action not in ACTIONS:
        raise HTTPException(404, detail=f"unknown action {action!r}; use one of {list(ACTIONS)}")
    m = hub.meta.get(call_id)
    if m is None:
        raise HTTPException(404, "unknown call_id")
    is_replay = m.origin == "replay"
    at_ms = hub.at_ms(call_id)
    res: dict[str, Any] = {}

    if action == "join":
        res = await do_barge(call_id)
    elif action == "listen":
        if not is_replay:
            raise HTTPException(409, "on a live call, press 1 on the Emberz alert call to listen in")
        if m.ended:
            raise HTTPException(409, "the call has ended")
        if m.caretaker is None:  # replay only: stands in for pressing 1 on the alert call
            hub.publish({"type": "caretaker_joined", "call_id": call_id, "at_ms": at_ms, "data": {"mode": "monitor"}},
                        origin="replay")
        res = {"caretaker_state": "barged" if m.caretaker == "barge" else "monitoring"}
    elif action == "hangup":
        if m.ended:
            raise HTTPException(409, "the call has ended")
        if is_replay:
            task = hub.replays.pop(call_id, None)
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
            hub.publish({"type": "call_ended", "call_id": call_id, "at_ms": at_ms,
                         "data": {"duration_ms": at_ms, "reason": "caretaker_hangup"}}, origin="replay")
            res = {"ended": True}
        else:  # the line sends call_ended itself when the call really drops
            res = await _forward_action(call_id, action, at_ms, required=True)
    elif action == "safe":  # console-only state; no event (Numan's contract has none)
        m.safe = True
        res = {"safe": True}
        if not is_replay and os.environ.get("LINE_ACTION_URL"):
            try:
                res.update(await _forward_action(call_id, action, at_ms, required=False))
            except HTTPException as e:
                res.update(forwarded=False, forward_status=str(e.detail))

    record = {"call_id": call_id, "action": action, "at_ms": at_ms, "time": time.time(), "replay": is_replay, **res}
    hub.actions.append(record)
    log.info("caretaker action %s", record)
    return {"ok": True, **record}


async def _forward_action(call_id: str, action: str, at_ms: int, *, required: bool) -> dict:
    url = os.environ.get("LINE_ACTION_URL", "").strip()
    if not url:
        raise HTTPException(501, "the phone line has no hang-up control yet; ask them to hang up")
    # Numan's line (line/matt_bridge.actions_router) takes POST <line>/actions/{call_id}/{action}
    # with X-Console-Token. Accept LINE_ACTION_URL as that base ("https://<tunnel>/actions"), a
    # template containing {call_id}/{action}, or a single URL that takes the JSON body.
    tok = line_token()
    headers = {"X-Console-Token": tok} if tok else None
    if "{call_id}" in url or "{action}" in url:
        target = url.replace("{call_id}", call_id).replace("{action}", action)
    elif url.rstrip("/").endswith("/actions"):
        target = f"{url.rstrip('/')}/{call_id}/{action}"
    else:
        target = url
    try:
        status, _ = await asyncio.to_thread(_post, target, {"call_id": call_id, "action": action, "at_ms": at_ms}, headers)
    except Exception as e:
        raise HTTPException(502, f"could not reach the phone line: {e}"[:200])
    if not 200 <= status < 300:
        raise HTTPException(502, f"the phone line answered {status}")
    return {"forwarded": True, "forward_status": f"HTTP {status}"}


@app.get("/actions", dependencies=AUTH)
async def get_actions():
    return {"actions": hub.actions[-200:]}


# ------------------------------------------------------------------ replay
async def _run_replay(call_id: str, events: list[dict], speed: float) -> None:
    t0 = time.monotonic()
    try:
        for ev in events:
            delay = t0 + ev["at_ms"] / 1000 / speed - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            hub.publish(ev, origin="replay", scale=speed)
    except asyncio.CancelledError:
        pass
    finally:
        if hub.replays.get(call_id) is asyncio.current_task():
            hub.replays.pop(call_id, None)


def start_replay(src_id: str, speed: float, force: bool = False) -> dict:
    ds = dataset()
    if src_id not in ds.calls:
        raise HTTPException(404, detail=f"unknown call_id {src_id!r}")
    live = hub.live_call()
    if live and not force:
        raise HTTPException(409, detail=f"a live call ({live}) is in progress; replay after it ends, or add force=1")
    hub.stop_replays()
    runs = hub.replay_runs.setdefault(src_id, itertools.count(1))
    while True:  # each replay gets a fresh id (t_x, t_x-2, t_x-3 ...), like Numan's fixture player
        n = next(runs)
        call_id = src_id if n == 1 else f"{src_id}-{n}"
        if call_id not in hub.meta:
            break
    events = rp.build_events(ds, src_id, as_call_id=call_id)
    hub.replays[call_id] = asyncio.create_task(_run_replay(call_id, events, speed))
    row = ds.calls[src_id]
    return {
        "ok": True, "call_id": call_id, "source_call_id": src_id, "speed": speed, "events": len(events),
        "turns": len(row["turns"]), "verdicts": len(ds.preds_by_turn(src_id)),
        "first_alert": (ds.preds.get(src_id) or {}).get("first_alert"), "ask_turn": row.get("ask_turn"),
        "preds_file": ds.preds_path.name, "seconds": round(events[-1]["at_ms"] / 1000 / speed, 1),
    }


@app.get("/replay", dependencies=AUTH)
async def replay(call_id: Optional[str] = None, preset: Optional[str] = None, speed: float = 1.0, force: bool = False):
    ds = dataset()
    if preset and not call_id:
        p = next((p for p in ds.presets if p["id"] == preset), None)
        if not p:
            raise HTTPException(404, detail=f"unknown preset {preset!r}")
        call_id = p["call_id"]
    if not call_id:
        raise HTTPException(404, detail="give call_id or preset")
    if not (0.05 <= speed <= 1000):
        raise HTTPException(422, detail="speed must be between 0.05 and 1000")
    return start_replay(call_id, speed, force)


def _scenes() -> dict[str, dict]:
    ds = dataset()
    out = {}
    for p in ds.presets:
        out[p["id"]] = {"call_id": p["call_id"], "title": p["title"]}
        if p.get("pair"):
            out[p["id"] + "_pair"] = {"call_id": p["pair"]["call_id"], "title": p["title"] + ": " + p["pair"]["title"]}
    return out


@app.get("/fixtures", dependencies=AUTH)
async def fixtures():
    ds = dataset()
    out = {}
    for name, s in _scenes().items():
        evs = rp.build_events(ds, s["call_id"])
        out[name] = {"call_id": s["call_id"], "events": len(evs), "duration_ms": evs[-1]["at_ms"], "title": s["title"]}
    return out


@app.post("/fixtures/play/{scene}", dependencies=AUTH)
async def fixtures_play(scene: str, speed: Optional[float] = None, force: bool = False):
    scenes = _scenes()
    if scene not in scenes:
        raise HTTPException(404, f"unknown scene; have {list(scenes)}")
    if speed is not None and not (0.1 <= speed <= 20):
        raise HTTPException(422, "speed must be within 0.1..20")
    r = start_replay(scenes[scene]["call_id"], speed or 1.0, force)
    return {"scene": scene, "call_id": r["call_id"]}


@app.get("/presets", dependencies=AUTH)
async def presets():
    ds = dataset()
    return {"preds": ds.preds_name, "preds_file": ds.preds_path.name, "presets": ds.presets}


@app.get("/health")
async def health():
    ds = dataset()
    return {"ok": True, "calls": len(ds.calls), "calls_with_preds": len(ds.preds), "preds_file": str(ds.preds_path),
            "clients": len(hub.subs), "replaying": [c for c, t in hub.replays.items() if not t.done()],
            "live_call": hub.live_call(), "bridge": {"url": line_url() or None, "state": hub.bridge_state},
            "line_action_url": bool(os.environ.get("LINE_ACTION_URL")), "token_required": bool(os.environ.get("CONSOLE_TOKEN"))}

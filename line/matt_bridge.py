"""Bridge between our live line and Matt's side (guardian LoRA + caretaker console).

(a) Guardian backend "matt" (GUARDIAN_BACKEND=matt, picked in line/guardian_client.default_client):
    - If Matt's prompt contract `callguard/guardian/render.py` imports (searched in CALLGUARD_PATH,
      then the repo root, then ../callguard*), its `render_messages` / `parse_response` build the
      request and read the answer. The call goes to Fireworks chat completions with
      model = GUARDIAN_MODEL and key GUARDIAN_API_KEY (falls back to FIREWORKS_API_KEY).
    - Otherwise: our stand-in prompt (line/standin_prompt.py) + JSON-schema output, same model/key.
    Either way failures fall back to SAFE_FALLBACK (GuardianClient's timeout/hedge/503 handling).

(b) ConsoleForwarder: if MATT_CONSOLE_URL is set, every ConsoleEvent the hub publishes is also
    POSTed to MATT_CONSOLE_URL/events (one worker thread, bounded queue, 1.5 s timeout, dropped on
    any failure; the line never waits on it). Header X-Console-Token = MATT_CONSOLE_TOKEN if set.

(c) actions_router(...): POST /actions/{call_id}/{listen|join|hangup|safe} on OUR app, so Matt's
    console server can forward button presses to LINE_URL. Needs ?token=CONSOLE_TOKEN or the
    X-Console-Token header (fails closed like the console routes).
      listen  -> dial the caretaker now (same "press 1 to listen" flow as an alert -> monitor)
      join    -> caretaker barge (must be monitoring; 409 otherwise)
      hangup  -> end the call (hangs up the caller leg and the protected leg)
      safe    -> mark the call safe: no further alerts, coach lines or caretaker dials
"""
from __future__ import annotations

import asyncio
import importlib
import importlib.util
import logging
import os
import queue
import sys
import threading
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Optional

import httpx
from fastapi import APIRouter, HTTPException, Request

from contracts.models import ConsoleEvent, GuardianRequest, GuardianResponse
from line import guardian_client
from line.console_hub import BargeNotAllowed, _token_status

log = logging.getLogger("emberz.matt")
REPO = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------ (a) guardian backend
def _candidate_roots() -> list[Path]:
    roots = []
    if os.environ.get("CALLGUARD_PATH"):
        roots.append(Path(os.environ["CALLGUARD_PATH"]).expanduser())
    roots.append(REPO)
    roots.extend(sorted(REPO.parent.glob("callguard*")))
    return roots


def find_render() -> Optional[ModuleType]:
    """Matt's callguard/guardian/render.py as a module, or None if it is not on this machine."""
    for root in _candidate_roots():
        for f in (root / "callguard" / "guardian" / "render.py", root / "guardian" / "render.py"):
            if not f.exists():
                continue
            pkg_root = f.parent.parent.parent if f.parent.parent.name == "callguard" else f.parent.parent
            if str(pkg_root) not in sys.path:
                sys.path.insert(0, str(pkg_root))
            name = "callguard.guardian.render" if f.parent.parent.name == "callguard" else "guardian.render"
            try:
                mod = importlib.import_module(name)
            except Exception:  # its package may need deps we lack; load the file alone
                try:
                    spec = importlib.util.spec_from_file_location("matt_render", f)
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)  # type: ignore[union-attr]
                except Exception as e:
                    log.error("found %s but it does not import: %s", f, e)
                    continue
            if hasattr(mod, "render_messages") and hasattr(mod, "parse_response"):
                log.info("guardian prompt: Matt's %s", f)
                return mod
    return None


def _to_response(obj: Any) -> GuardianResponse:
    if isinstance(obj, GuardianResponse):
        return obj
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump()
    if isinstance(obj, str):
        return guardian_client.parse_response(obj)
    return GuardianResponse.model_validate(obj)


class MattGuardianClient(guardian_client.GuardianClient):
    def __init__(self, model: Optional[str] = None, render: Any = "auto", **kw: Any):
        self.render = find_render() if render == "auto" else render
        model = model or os.environ.get("GUARDIAN_MODEL") or os.environ.get(
            "GUARDIAN_STANDIN_MODEL", guardian_client.STANDIN_MODEL)
        super().__init__(backend="matt", model=model, **kw)
        if "GUARDIAN_REASONING_EFFORT" not in os.environ:
            # Matt's LoRA deployment: omit (a 400 would turn the guardian off); the stand-in needs "none".
            self.reasoning_effort = "" if os.environ.get("GUARDIAN_MODEL") else "none"
        log.info("guardian backend matt: model=%s prompt=%s", self.model,
                 "callguard render.py" if self.render else "line/standin_prompt.py")

    def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            key = os.environ.get("GUARDIAN_API_KEY") or guardian_client._fireworks_key()
            self._http = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout_s, connect=2.0),
                                           headers={"Authorization": f"Bearer {key}"})
        return self._http

    def _body(self, req: GuardianRequest) -> dict:
        if self.render is None:
            return super()._body(req)
        try:
            messages = self.render.render_messages(req)
        except (TypeError, AttributeError, ValueError):
            messages = self.render.render_messages(req.model_dump())  # a dict-based contract
        body = {"model": self.model, "messages": messages, "temperature": 0, "max_tokens": self.max_tokens}
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        return body

    def _parse(self, content: str) -> GuardianResponse:
        if self.render is None:
            return super()._parse(content)
        try:
            return _to_response(self.render.parse_response(content))
        except Exception as e:
            raise ValueError(f"render.parse_response failed: {e}") from e


# ------------------------------------------------------------------ (b) console forwarder
class ConsoleForwarder:
    """Non-blocking POST of every ConsoleEvent to <url>/events. Call it like a function."""

    def __init__(self, url: str, timeout_s: float = 1.5, maxsize: int = 2000, token: Optional[str] = None):
        self.url = url.rstrip("/") + "/events"
        self.timeout_s = timeout_s
        self.headers = {"X-Console-Token": token} if token else {}
        self.q: "queue.Queue[Optional[ConsoleEvent]]" = queue.Queue(maxsize=maxsize)
        self.sent = self.dropped = 0
        self._t = threading.Thread(target=self._run, name="matt-console-forward", daemon=True)
        self._t.start()

    def __call__(self, event: ConsoleEvent) -> None:
        try:
            self.q.put_nowait(event)
        except queue.Full:
            self.dropped += 1

    def _run(self) -> None:
        with httpx.Client(timeout=self.timeout_s) as http:
            while True:
                ev = self.q.get()
                if ev is None:
                    return
                try:
                    r = http.post(self.url, content=ev.model_dump_json(), headers={
                        "content-type": "application/json", **self.headers})
                    if r.status_code >= 400:
                        raise RuntimeError(f"HTTP {r.status_code}")
                    self.sent += 1
                except Exception as e:
                    self.dropped += 1
                    if self.dropped <= 3 or self.dropped % 100 == 0:
                        log.warning("console forward to %s failed (%s); %d dropped", self.url, e, self.dropped)

    def close(self, timeout: float = 2.0) -> None:
        try:
            self.q.put_nowait(None)
        except queue.Full:
            pass
        self._t.join(timeout)


def forwarder_from_env() -> Optional[ConsoleForwarder]:
    url = os.environ.get("MATT_CONSOLE_URL", "").strip()
    if not url:
        return None
    log.info("forwarding console events to %s/events", url.rstrip("/"))
    return ConsoleForwarder(url, token=os.environ.get("MATT_CONSOLE_TOKEN") or None)


# ------------------------------------------------------------------ (c) action endpoints
ACTIONS = ("listen", "join", "hangup", "safe")


def actions_router(flow: Any, caretaker: Any, hub: Any, *, open_access: bool = False,
                   end_call: Optional[Callable[[Any, str], Any]] = None) -> APIRouter:
    """flow = line.call_flow.CallFlow (app.state.flow), caretaker = CaretakerFlow, hub = ConsoleHub."""
    router = APIRouter()

    @router.post("/actions/{call_id}/{action}")
    async def action(call_id: str, action: str, request: Request) -> dict:
        code = _token_status(request.query_params.get("token") or request.headers.get("x-console-token"), open_access)
        if code == 503:
            raise HTTPException(503, "actions disabled: set CONSOLE_TOKEN on the line server")
        if code != 200:
            raise HTTPException(401, "bad or missing console token")
        if action not in ACTIONS:
            raise HTTPException(404, f"unknown action; one of {ACTIONS}")
        session = flow.sessions.get(call_id)
        if session is None:
            raise HTTPException(404, "unknown or finished call")
        log.info("console action %s on %s", action, call_id)

        if action == "listen":
            if session.caretaker_state in ("monitoring", "barged", "ringing"):
                return {"call_id": call_id, "result": "already_" + session.caretaker_state,
                        "caretaker_state": session.caretaker_state}
            if not session.caretaker_number:
                raise HTTPException(409, "no caretaker number on this guarded line")
            result = await caretaker._dial(session, "Listen requested from the console")
            return {"call_id": call_id, "result": result, "caretaker_state": session.caretaker_state}

        if action == "join":
            try:
                state = await caretaker.barge(session)
            except BargeNotAllowed as e:
                raise HTTPException(409, str(e))
            except Exception as e:
                log.error("barge on %s failed: %s", call_id, e)
                raise HTTPException(502, "the phone system rejected the join; try again")
            return {"call_id": call_id, "result": "barged", "caretaker_state": state}

        if action == "hangup":
            lock = flow._locks.setdefault(call_id, asyncio.Lock())
            async with lock:
                await (end_call or flow._end_call)(session, "console_hangup")
            return {"call_id": call_id, "result": "hung_up"}

        # safe
        session.marked_safe = True
        hub.emit(call_id, "risk", session.now_ms(), turn=session.turns[-1].turn if session.turns else 0,
                 response=guardian_client.SAFE_FALLBACK.model_dump(), latency_ms=0, display_risk="none",
                 suggest_hangup=False, reasons=["marked_safe_by_caretaker"])
        return {"call_id": call_id, "result": "marked_safe"}

    return router

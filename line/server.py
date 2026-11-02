"""L1 line server: Telnyx webhooks in, media WebSocket in, hooks out.

Routes:
  POST /webhooks/telnyx   verify Ed25519, dedupe on data.id, hand to CallFlow in the background
  WS   /media/{call_id}   Telnyx media stream frames -> brain.on_media(track, payload, ts_ms)
  GET  /healthz

Other units plug in without editing this file:
  create_app(telnyx=..., make_brain=..., caretaker=..., hub=..., extra_routers=(...))
  make_brain(session, telnyx, hub) -> object with async on_media / on_call_event / on_hangup
  caretaker.on_event(session, event_type, payload)   for C-leg events (+ "emberz.call_ended")
  hub.publish(ConsoleEvent)                           sync or async

Media frames (https://developers.telnyx.com/docs/voice/programmable-voice/media-streaming):
  {"event": "connected", "version": ...}
  {"event": "start", "start": {"call_control_id", "media_format": {"encoding", "sample_rate", "channels"}, ...}, "stream_id"}
  {"event": "media", "media": {"track": "inbound|outbound", "chunk", "timestamp", "payload": b64}, "stream_id"}
  {"event": "stop", ...}   (also "mark", "dtmf", "error")
  Frames "may arrive out of order, so use chunk to reorder them" -> TrackReorder below.
  payload = "base64-encoded RTP payload (no headers)", i.e. raw PCMU bytes for our stream_codec.
  The WS handshake is signed: headers telnyx-signature-ed25519 + telnyx-timestamp over
  f"{timestamp}|{stream_url}" (same doc). EMBERZ_MEDIA_VERIFY=warn (default) logs a bad/missing
  signature and the result goes to the brain in "media.start" (_signature_ok); =strict rejects.
  A bad frame (non-JSON, binary, wrong shape, empty audio) is skipped, never ends the stream, and
  an exception in the brain is logged, never closes the socket.
  For leg A with both_tracks: inbound = the caller, outbound = what A hears (the conference mix,
  i.e. the protected person once B has joined; before that, the notice).

Run:  .venv/bin/python -m line.server   (needs PUBLIC_BASE_URL, TELNYX_* env; see line/config.py)
"""
from __future__ import annotations

import base64
import binascii
import heapq
import json
import logging
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Any, Iterable, Optional

from fastapi import APIRouter, BackgroundTasks, FastAPI, Request, Response, WebSocket, WebSocketDisconnect

from line.call_flow import CallFlow, MakeBrain, NullBrain, NullCaretaker, NullHub  # noqa: F401 (re-export)
from line.config import DEMO_APP_ID, GuardedLine, Settings, load_profiles_or_env, load_settings
from line.webhook_verify import WebhookVerificationError, verify, verify_media_handshake, warn_skip_verify

log = logging.getLogger("emberz.server")

MAX_FRAME_B64 = 64 * 1024


class SeenIds:
    """Bounded set for webhook event-id dedupe."""

    def __init__(self, cap: int = 20000):
        self.cap, self._d = cap, OrderedDict()

    def add(self, key: str) -> bool:
        """True if new."""
        if key in self._d:
            return False
        self._d[key] = None
        if len(self._d) > self.cap:
            self._d.popitem(last=False)
        return True


class TrackReorder:
    """Per-track reorder by `chunk`: holds up to `depth` frames, drops late duplicates.
    A jump back by more than `restart_gap` chunks means the counter restarted: flush and go on."""

    def __init__(self, depth: int = 2, restart_gap: int = 100):
        self.depth, self.restart_gap = depth, restart_gap
        self.heap: list[tuple[int, int, bytes, int]] = []
        self.pending: set[int] = set()
        self.last: Optional[int] = None
        self._n = 0

    def push(self, chunk: int, payload: bytes, ts_ms: int) -> list[tuple[bytes, int]]:
        flushed: list[tuple[bytes, int]] = []
        newest = max([self.last or 0, *self.pending]) if (self.last is not None or self.pending) else None
        if newest is not None and chunk < newest - self.restart_gap:
            flushed, self.last = self.flush(), None  # counter restarted
        if (self.last is not None and chunk <= self.last) or chunk in self.pending:
            return flushed  # late or duplicate
        self.pending.add(chunk)
        self._n += 1
        heapq.heappush(self.heap, (chunk, self._n, payload, ts_ms))
        out = flushed
        while len(self.heap) > self.depth:
            out.append(self._pop())
        return out

    def flush(self) -> list[tuple[bytes, int]]:
        out = []
        while self.heap:
            out.append(self._pop())
        return out

    def _pop(self) -> tuple[bytes, int]:
        chunk, _, payload, ts = heapq.heappop(self.heap)
        self.pending.discard(chunk)
        self.last = chunk
        return payload, ts


def _int(v: Any, default: int = 0) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def create_app(
    telnyx: Any = None,
    make_brain: Optional[MakeBrain] = None,
    caretaker: Any = None,
    hub: Any = None,
    extra_routers: Iterable[APIRouter] = (),
    settings: Optional[Settings] = None,
    profiles: Optional[dict[str, GuardedLine]] = None,
    reorder_depth: int = 2,
) -> FastAPI:
    settings = settings or load_settings()
    if profiles is None:
        profiles = load_profiles_or_env(settings)
    owns_client = telnyx is None
    if telnyx is None:
        if not settings.telnyx_api_key:
            raise RuntimeError("TELNYX_API_KEY is not set (set TELNYX_ENV_FILE to EmberDial's .env path)")
        from line.telnyx_client import TelnyxClient
        telnyx = TelnyxClient(api_key=settings.telnyx_api_key, connection_id=settings.telnyx_connection_id)
    if settings.telnyx_connection_id != DEMO_APP_ID:
        # Dialing through another app sends B/C webhooks to that app's server (EmberDial production).
        if not settings.allow_other_app:
            raise RuntimeError(f"TELNYX_CONNECTION_ID is {settings.telnyx_connection_id}, not the demo app "
                               f"{DEMO_APP_ID}; refusing (set EMBERZ_ALLOW_OTHER_APP=1 to override)")
        log.warning("TELNYX_CONNECTION_ID is %s, not the demo app %s", settings.telnyx_connection_id, DEMO_APP_ID)
    if settings.skip_verify:
        warn_skip_verify()
    elif not settings.telnyx_public_key:
        log.error("TELNYX_PUBLIC_KEY is not set: every webhook will be rejected (fail closed)")

    flow = CallFlow(settings, profiles, telnyx, make_brain, caretaker, hub)
    seen = SeenIds()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        if owns_client:
            await telnyx.aclose()

    app = FastAPI(title="Emberz line", lifespan=lifespan)
    app.state.settings = settings
    app.state.flow = flow
    app.state.sessions = flow.sessions
    app.state.telnyx = telnyx
    app.state.hub = flow.hub

    @app.get("/healthz")
    async def healthz() -> dict:
        return {
            "ok": True,
            "verify": "SKIPPED" if settings.skip_verify else ("on" if settings.telnyx_public_key else "NO KEY"),
            "active_calls": len(flow.sessions),
            "guarded_numbers": len(profiles),
        }

    async def _process(event_type: str, payload: dict) -> None:
        try:
            await flow.handle(event_type, payload)
        except Exception:
            log.exception("flow failed on %s", event_type)

    @app.post("/webhooks/telnyx")
    @app.post("/webhooks/telnyx/call")  # the path the demo app pointed at before (EmberDial's convention)
    async def webhook(request: Request, background: BackgroundTasks) -> Response:
        raw = await request.body()
        if settings.skip_verify:
            log.warning("webhook accepted WITHOUT signature check (EMBERZ_SKIP_VERIFY=1)")
        else:
            try:
                verify(raw, request.headers, settings.telnyx_public_key)
            except WebhookVerificationError as e:
                log.warning("rejected webhook: %s", e)
                return Response(status_code=403)
        try:
            data = json.loads(raw)["data"]
            event_type, event_id = str(data["event_type"]), str(data["id"])
            payload = data.get("payload") or {}
            if not isinstance(payload, dict):
                raise TypeError("payload is not an object")
        except (ValueError, KeyError, TypeError, AttributeError):
            return Response(status_code=400)
        if not seen.add(event_id):
            log.info("duplicate webhook %s (%s) ignored", event_id, event_type)
            return Response(status_code=200)
        log.info("webhook %s ccid=%s", event_type, str(payload.get("call_control_id", ""))[-8:])
        background.add_task(_process, event_type, payload)  # ack fast; Telnyx retries slow 5xx/timeouts
        return Response(status_code=200)

    def _media_signature_ok(ws: WebSocket, call_id: str) -> Optional[bool]:
        st = flow.settings
        if st.media_verify == "off" or st.skip_verify:
            return None
        try:
            verify_media_handshake(st.media_ws_url(call_id), ws.headers, st.telnyx_public_key)
            return True
        except (WebhookVerificationError, RuntimeError) as e:
            log.warning("media socket for %s: signature check failed (%s)", call_id, e)
            return False

    @app.websocket("/media/{call_id}")
    async def media(ws: WebSocket, call_id: str) -> None:
        await ws.accept()
        session = flow.sessions.get(call_id)
        sig_ok = _media_signature_ok(ws, call_id)
        if session is None or (sig_ok is False and flow.settings.media_verify == "strict"):
            log.warning("media socket for %s closed (%s)", call_id, "unknown call" if session is None else "bad signature")
            await ws.close(code=1008)
            return
        tracks: dict[str, TrackReorder] = {}
        frames = skipped = brain_errors = 0

        async def call_brain(method: str, *args: Any) -> None:
            nonlocal brain_errors
            brain = flow.brains.get(call_id)
            if brain is None:
                return
            try:
                await getattr(brain, method)(*args)
            except Exception:
                brain_errors += 1
                if brain_errors <= 3:  # do not flood the log at 100 frames/s
                    log.exception("brain.%s failed on %s", method, call_id)

        async def feed(track: str, items: list[tuple[bytes, int]]) -> None:
            for payload, ts in items:
                await call_brain("on_media", track, payload, ts)

        async def flush_all() -> None:
            for track, tr in tracks.items():
                await feed(track, tr.flush())

        try:
            while True:
                raw = await ws.receive()
                if raw.get("type") == "websocket.disconnect":
                    break
                text = raw.get("text")
                try:
                    msg = json.loads(text) if text is not None else None
                except ValueError:
                    msg = None
                if not isinstance(msg, dict):
                    skipped += 1  # binary or non-JSON frame: not part of the protocol
                    continue
                ev = msg.get("event")
                if ev == "media":
                    m = msg.get("media")
                    b64 = m.get("payload") if isinstance(m, dict) else None
                    if not isinstance(b64, str) or not b64 or len(b64) > MAX_FRAME_B64:
                        skipped += 1
                        continue
                    try:
                        payload = base64.b64decode(b64)
                    except (binascii.Error, ValueError):
                        skipped += 1
                        continue
                    track = str(m.get("track", "inbound"))
                    track = "outbound" if track.startswith("outbound") else "inbound" if track.startswith("inbound") else None
                    if not payload or track is None:
                        skipped += 1
                        continue
                    tr = tracks.setdefault(track, TrackReorder(reorder_depth))
                    frames += 1
                    await feed(track, tr.push(_int(m.get("chunk"), frames), payload, _int(m.get("timestamp"))))
                elif ev == "start":
                    start = msg.get("start") if isinstance(msg.get("start"), dict) else {}
                    fmt = start.get("media_format") if isinstance(start.get("media_format"), dict) else {}
                    if start.get("call_control_id") not in (None, session.a_ccid):
                        log.warning("media start for an unexpected leg on %s", call_id)
                    if str(fmt.get("encoding", "")).upper() not in ("PCMU", ""):
                        log.warning("media encoding is %s, expected PCMU", fmt.get("encoding"))
                    await flush_all()  # a new stream restarts chunk numbering
                    tracks.clear()
                    session.stream_id = msg.get("stream_id") or session.stream_id
                    await call_brain("on_call_event", "media.start",
                                     {**start, "stream_id": msg.get("stream_id"), "_signature_ok": sig_ok})
                elif ev == "stop":
                    break
                elif ev == "connected":
                    log.info("media connected for %s (version %s)", call_id, msg.get("version"))
                elif ev in ("dtmf", "mark", "error"):
                    await call_brain("on_call_event", f"media.{ev}", msg)
        except WebSocketDisconnect:
            pass
        finally:
            await flush_all()
            log.info("media closed for %s after %d frames (%d skipped, %d brain errors)",
                     call_id, frames, skipped, brain_errors)

    for router in extra_routers:
        app.include_router(router)
    return app


def live_problems(settings: Settings) -> list[str]:
    """Why a live server (real Telnyx webhooks through the tunnel) must not start. Shared with hour0."""
    problems = []
    if not settings.public_base_url:
        problems.append("PUBLIC_BASE_URL is not set (run line/tunnel.sh and export it)")
    if not settings.telnyx_api_key:
        problems.append("TELNYX_API_KEY is not set (TELNYX_ENV_FILE)")
    if settings.skip_verify and settings.is_public():
        problems.append("EMBERZ_SKIP_VERIFY=1 with a public PUBLIC_BASE_URL: anyone could drive calls; unset it")
    elif not settings.skip_verify and not settings.telnyx_public_key:
        problems.append("TELNYX_PUBLIC_KEY is not set (every webhook would be rejected)")
    return problems


def main() -> None:
    import sys

    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    settings = load_settings()
    problems = live_problems(settings)
    if problems:
        print("Cannot start:\n  " + "\n  ".join(problems))
        sys.exit(2)
    uvicorn.run(create_app(settings=settings), host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()

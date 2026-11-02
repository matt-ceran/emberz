"""Fixture player: replays timed console_event streams through the real console backend.

Lets the console page (C1) be built before live calls exist. Same routes as the line server:

    WS   ws://localhost:8001/console/ws[?call_id=...]
    GET  http://localhost:8001/console/calls
    GET  http://localhost:8001/console/calls/{call_id}/events
    POST http://localhost:8001/console/calls/{call_id}/barge      (works while a scene is "monitoring")
plus, only here:
    GET  /fixtures                       the scenes on disk
    POST /fixtures/play/{scene}          replay one scene now (new call_id each time); ?speed=2

Run from the repo root:
    .venv/bin/python -m console.fixture_player                 # plays every scene once, in order
    .venv/bin/python -m console.fixture_player --loop --speed 2
    .venv/bin/python -m console.fixture_player --no-autoplay   # only plays on POST /fixtures/play/...

The barge button goes through the real CaretakerFlow with FakeTelnyx, so nothing is dialed.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import itertools
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # allow `python console/fixture_player.py` as well as `-m`
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException  # noqa: E402

from contracts.models import RISK_ORDER, ConsoleEvent, GuardianResponse, Turn, VoiceSignal  # noqa: E402
from line.caretaker import CaretakerFlow  # noqa: E402
from line.console_hub import ConsoleHub, add_cors, console_router  # noqa: E402
from line.session import CallSession  # noqa: E402
from line.telnyx_client import FakeTelnyx  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"


# ------------------------------------------------------------------ loading + validation
def scenes() -> dict[str, Path]:
    return {p.stem: p for p in sorted(FIXTURES.glob("*.jsonl"))}


def check_payload(e: ConsoleEvent) -> None:
    """Validate `data` per type (contracts/README.md). Raises ValueError/ValidationError."""
    d = e.data
    if e.type == "turn":
        Turn.model_validate(d)
    elif e.type == "voice":
        VoiceSignal.model_validate(d)
    elif e.type == "risk":
        GuardianResponse.model_validate(d["response"])
        assert isinstance(d["turn"], int) and isinstance(d["latency_ms"], int)
        # optional keys line/brain.py adds live: the rule layer's verdict
        assert d.get("display_risk", "none") in RISK_ORDER
        assert isinstance(d.get("suggest_hangup", False), bool) and isinstance(d.get("reasons", []), list)
    elif e.type == "alert":
        assert d["level"] in ("alert", "urgent") and d["category"] and isinstance(d["turn"], int)
        assert isinstance(d["caretaker_note"], str)
    elif e.type == "coach":
        assert isinstance(d["text"], str) and isinstance(d["turn"], int)
        assert isinstance(d.get("sent", True), bool)  # live: False if the coach speak failed
    elif e.type == "caretaker_joined":
        assert d["mode"] in ("monitor", "barge")
    elif e.type == "summary":
        assert isinstance(d["text"], str) and d["text"]
    elif e.type == "call_started":
        assert set(d) >= {"caller_number", "guarded_number", "profile", "caller_known"}
    elif e.type == "call_ended":
        assert isinstance(d["duration_ms"], int) and isinstance(d["reason"], str)
    if "call_id" in d and d["call_id"] != e.call_id:
        raise ValueError(f"data.call_id {d['call_id']} != {e.call_id}")


def load_scene(path: Path) -> list[ConsoleEvent]:
    events = [ConsoleEvent.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]
    for e in events:
        check_payload(e)
    return events


def _with_call_id(e: ConsoleEvent, call_id: str) -> ConsoleEvent:
    data = dict(e.data)
    if "call_id" in data:
        data["call_id"] = call_id
    return e.model_copy(update={"call_id": call_id, "data": data})


# ------------------------------------------------------------------ player
class _FixtureSession(CallSession):
    speed: float = 1.0

    def now_ms(self) -> int:  # keep "call time" consistent with the fixture's at_ms when sped up
        return int((time.monotonic() - self.started_monotonic) * 1000 * self.speed)


class FixturePlayer:
    def __init__(self, hub: ConsoleHub, speed: float = 1.0):
        self.hub, self.speed = hub, speed
        self.sessions: dict[str, CallSession] = {}
        self.caretaker = CaretakerFlow(FakeTelnyx(), hub, self.sessions.get)
        self._runs: dict[str, itertools.count] = {}
        self._tasks: set[asyncio.Task] = set()  # keep references so running replays aren't garbage-collected

    def start(self, scene: str, speed: Optional[float] = None) -> tuple[str, asyncio.Task]:
        """Start replaying a scene in the background; returns its call_id at once."""
        events = load_scene(scenes()[scene])
        speed = speed or self.speed
        n = next(self._runs.setdefault(scene, itertools.count(1)))
        call_id = events[0].call_id if n == 1 else f"{events[0].call_id}-{n}"
        start = events[0].data if events[0].type == "call_started" else {}
        s = _FixtureSession(
            call_id=call_id, guarded_number=start.get("guarded_number", ""), caller_number=start.get("caller_number"),
            caretaker_number="+10000000000", conference_id=f"fx_conf_{call_id}", c_ccid=f"fx_c_{call_id}",
        )
        s.speed = speed
        self.sessions[call_id] = s
        task = asyncio.create_task(self._run(events, s, speed))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return call_id, task

    async def play(self, scene: str, speed: Optional[float] = None) -> str:
        call_id, task = self.start(scene, speed)
        await task
        return call_id

    async def _run(self, events: list[ConsoleEvent], s: CallSession, speed: float) -> None:
        t0 = time.monotonic()
        for e in events:
            delay = t0 + e.at_ms / 1000 / speed - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            if e.type == "caretaker_joined":
                if e.data.get("mode") == "barge" and s.caretaker_state == "barged":
                    continue  # someone already pressed the join button
                s.caretaker_state = "barged" if e.data.get("mode") == "barge" else "monitoring"
            elif e.type == "call_ended":
                s.ended = True
            self.hub.publish(_with_call_id(e, s.call_id))

    async def autoplay(self, loop: bool, gap_s: float = 3.0) -> None:
        while True:
            for name in scenes():
                await self.play(name)
                await asyncio.sleep(gap_s)
            if not loop:
                return


def create_app(speed: float = 1.0, autoplay: bool = True, loop: bool = False) -> FastAPI:
    hub = ConsoleHub()
    player = FixturePlayer(hub, speed)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        task = asyncio.create_task(player.autoplay(loop)) if autoplay else None
        yield
        if task:
            task.cancel()

    app = FastAPI(title="Emberz console fixture player", lifespan=lifespan)
    add_cors(app)
    app.include_router(console_router(hub, player.caretaker, open_access=True))  # local fake data only
    app.state.hub, app.state.player = hub, player

    @app.get("/")
    async def index():
        return {
            "what": "Emberz console fixture player; see console/README_fixtures.md",
            "ws": "/console/ws?call_id=<optional>",
            "http": ["/console/calls", "/console/calls/{call_id}/events", "POST /console/calls/{call_id}/barge",
                     "/fixtures", "POST /fixtures/play/{scene}?speed=1"],
            "scenes": list(scenes()),
        }

    @app.get("/fixtures")
    async def list_fixtures():
        out = {}
        for name, p in scenes().items():
            events = load_scene(p)
            out[name] = {"call_id": events[0].call_id, "events": len(events), "duration_ms": events[-1].at_ms}
        return out

    @app.post("/fixtures/play/{scene}")
    async def play(scene: str, speed: Optional[float] = None):
        if scene not in scenes():
            raise HTTPException(404, f"unknown scene; have {list(scenes())}")
        if speed is not None and not (0.1 <= speed <= 20):
            raise HTTPException(422, "speed must be within 0.1..20")
        call_id, _ = player.start(scene, speed)
        return {"scene": scene, "call_id": call_id}

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--speed", type=float, default=1.0, help="2 = twice as fast")
    ap.add_argument("--loop", action="store_true", help="replay all scenes forever")
    ap.add_argument("--no-autoplay", action="store_true")
    args = ap.parse_args()
    if not 0.1 <= args.speed <= 20:
        ap.error("--speed must be within 0.1..20")
    import uvicorn

    uvicorn.run(create_app(args.speed, not args.no_autoplay, args.loop), host=args.host, port=args.port)


if __name__ == "__main__":
    main()

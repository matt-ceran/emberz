"""Fakes for the L2 brain tests (hub, caretaker, transcriber) and a session builder."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contracts.models import ConsoleEvent  # noqa: E402
from line.session import CallSession  # noqa: E402


class FakeHub:
    def __init__(self):
        self.events: list[ConsoleEvent] = []

    def publish(self, event):
        self.events.append(event)
        return event

    def of(self, type_):
        return [e for e in self.events if e.type == type_]


class FakeCaretaker:
    def __init__(self):
        self.alerts: list[tuple[str, str, str]] = []

    async def alert(self, session, level, note):
        self.alerts.append((session.call_id, level, note))
        return "dialed"


class FakeTranscriber:
    """Matches line.asr.AsyncCallTranscriber's constructor and feed/aclose API."""

    instances: list["FakeTranscriber"] = []

    def __init__(self, call_id, on_turn, loop, on_partial=None):
        self.call_id, self.on_turn, self.loop = call_id, on_turn, loop
        self.fed: list[tuple[str, int, int]] = []
        self.closed = False
        FakeTranscriber.instances.append(self)

    def feed(self, track, payload, ts_ms=None):
        self.fed.append((track, len(payload), ts_ms))

    async def aclose(self):
        self.closed = True


def make_session(**kw) -> CallSession:
    base = dict(
        call_id="c_fixture_1",
        guarded_number="+15550000001",
        caller_number="+15550009999",
        profile="elder",
        rules=["Never share codes or passwords", "Alert me about any money talk", "Allowed: Grandpa (+15550100)"],
        caretaker_number="+15550000003",
        a_ccid="A",
        b_ccid="B",
        conference_id="conf_1",
    )
    base.update(kw)
    return CallSession(**base)

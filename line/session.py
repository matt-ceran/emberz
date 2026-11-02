"""Per-call state shared by the call flow (L1), the brain (L2) and the caretaker flow (L3)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from contracts.models import AiVoice, Profile, Turn


@dataclass
class CallSession:
    call_id: str                      # our id, e.g. "c_<call_session_id>"
    guarded_number: str
    caller_number: Optional[str]      # None when anonymous
    profile: Profile = "elder"
    rules: list[str] = field(default_factory=list)
    allowed_contacts: dict[str, str] = field(default_factory=dict)  # E.164 -> name
    protected_number: str = ""
    caretaker_number: str = ""
    caller_known: bool = False

    a_ccid: Optional[str] = None      # caller leg
    b_ccid: Optional[str] = None      # protected person leg
    c_ccid: Optional[str] = None      # caretaker leg
    conference_id: Optional[str] = None
    stream_id: Optional[str] = None

    turns: list[Turn] = field(default_factory=list)
    ai_voice: Optional[AiVoice] = None
    policy_state: Any = None          # owned by line/policy.py
    caretaker_state: str = "idle"     # idle|ringing|monitoring|barged|declined
    started_monotonic: float = field(default_factory=time.monotonic)
    ended: bool = False

    def now_ms(self) -> int:
        return int((time.monotonic() - self.started_monotonic) * 1000)

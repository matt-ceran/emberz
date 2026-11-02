"""Emberz contracts (PLAN.md section 4 and 6).

DRAFT by Numan's side for I0. Matt reviews at kickoff; any change here is a
contract change and both sides re-run their fixture tests.

Every event and row that crosses a work-package boundary is defined here.
`python -m contracts.export` writes JSON Schema files next to this module.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


Speaker = Literal["caller", "protected"]
Profile = Literal["elder", "kid", "adult"]
Risk = Literal["none", "watch", "alert", "urgent"]
Stage = Literal["hook", "trust", "pressure", "ask"]
Action = Literal["none", "log", "coach", "notify_caretaker", "suggest_hangup", "urgent_alert"]
Category = Literal[
    "scam.impersonation_bank_gov",
    "scam.tech_support",
    "scam.family_emergency",
    "scam.prize",
    "scam.romance_investment",
    "disclosure.location_school",
    "disclosure.credentials_otp",
    "disclosure.financial_id",
    "disclosure.home_alone",
    "grooming.secrecy",
    "grooming.isolation",
    "grooming.gifts_flattery",
    "grooming.photo_request",
    "grooming.meeting_request",
    "vulgar",
    "threat_extortion",
    "self_harm",
    "none",
]
RISK_ORDER: dict[str, int] = {"none": 0, "watch": 1, "alert": 2, "urgent": 3}


# ---------------------------------------------------------------- line -> guardian
class Turn(_Strict):
    """One final transcript segment from the line (event `turn`)."""

    call_id: str
    turn: int = Field(ge=1)
    speaker: Speaker
    text: str
    t_start_ms: int = Field(ge=0)
    t_end_ms: int = Field(ge=0)
    gap_ms: Optional[int] = Field(
        default=None, description="Silence between the previous turn's end (other speaker) and this turn's start"
    )


class WindowTurn(_Strict):
    turn: int
    speaker: Speaker
    text: str


class AiVoice(_Strict):
    score: float = Field(ge=0.0, le=1.0)
    source: Literal["detector", "telnyx", "timing"]


class GuardianRequest(_Strict):
    call_id: str
    profile: Profile
    rules: list[str]
    caller_known: bool
    ai_voice: Optional[AiVoice] = None
    turns: list[WindowTurn] = Field(description="The last N turns (window frozen at S2, 8 by default)")


class Evidence(_Strict):
    turn: int
    quote: str


class GuardianResponse(_Strict):
    risk: Risk
    category: Category
    severity: int = Field(ge=0, le=3)
    source: Literal["caller", "protected_person"]
    stage: Stage
    evidence: list[Evidence] = Field(default_factory=list)
    action: Action
    coach_line: str = Field(default="", description="15 words max, heard by the protected person only")
    caretaker_note: str = Field(default="", description="20 words max")


# ---------------------------------------------------------------- voice -> line
class VoiceSignal(_Strict):
    call_id: str
    at_ms: int = Field(ge=0)
    ai_score: float = Field(ge=0.0, le=1.0)
    model: str = Field(description="w2v-head-v1 | telnyx | timing | combined")


# ---------------------------------------------------------------- line -> console
ConsoleEventType = Literal[
    "call_started", "turn", "risk", "alert", "coach", "caretaker_joined", "summary", "call_ended", "voice"
]


class ConsoleEvent(_Strict):
    type: ConsoleEventType
    call_id: str
    at_ms: int = Field(ge=0, description="ms since call start")
    data: dict = Field(default_factory=dict, description="Payload; shape per type documented in contracts/README.md")


# ---------------------------------------------------------------- model work package
class LabeledTurn(_Strict):
    class Label(_Strict):
        risk: Risk
        category: Category
        stage: Stage

    turn: int
    speaker: Speaker
    text: str
    label: Label


class DatasetRow(_Strict):
    call_id: str
    split: Literal["train", "dev", "test_heldout"]
    generator: str
    profile: Profile
    rules: list[str]
    rule_flip_of: Optional[str] = None
    scam_type: str
    turns: list[LabeledTurn]
    ask_turn: Optional[int] = None
    benign_twin: bool = False
    # Simulated live signals (M1 extras) so training inputs match live ones.
    caller_known: Optional[bool] = None
    ai_voice: Optional[AiVoice] = None


class NoisyTurn(_Strict):
    """G1 output: one row per turn of a call that went Kokoro -> 8 kHz -> Nemotron ASR."""

    call_id: str
    turn: int
    speaker: Speaker
    text_clean: str
    text_asr: str
    voice: str = Field(description="Kokoro voice id used for the render")
    wer: Optional[float] = None


class ResultsRow(_Strict):
    model: str
    test_set: str
    metric: str
    value: float
    n: int
    ci95: Optional[tuple[float, float]] = None


ALL_MODELS = {
    "turn": Turn,
    "guardian_request": GuardianRequest,
    "guardian_response": GuardianResponse,
    "voice_signal": VoiceSignal,
    "console_event": ConsoleEvent,
    "dataset_row": DatasetRow,
    "noisy_turn": NoisyTurn,
    "results_row": ResultsRow,
}

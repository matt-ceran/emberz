"""Emberz rule layer: a pure, deterministic function from one guardian response to an action.

    decide(state, resp, ctx) -> (Decision, new_state)

No I/O, no clock, no randomness: the caller passes the time in `ctx.now_ms`, and the state
is an immutable value that the caller threads through the call (start with PolicyState()).
Matt's eval (M3) imports this module so eval scores exactly what the demo runs.

Rules, in the order they are applied (each names the constant that controls it):

1. Base level. The model's `risk` (none < watch < alert < urgent).
   If the model gives category "none", the level is capped at watch (NO_CATEGORY_CAP):
   an alert must name what it is about.

2. AI voice (PLAN.md section 6). A caller is "AI voice" when ctx.ai_voice_score >=
   AI_VOICE_THRESHOLD (0.7).
   a. AI voice alone is at most watch: it raises the level to AI_VOICE_FLOOR (watch) but
      never higher on its own. A pharmacy reminder bot lights the badge and sends no alert.
   b. AI voice + unknown caller (not ctx.caller_known) + an ask (stage "ask" or model risk
      >= alert) with a real category escalates the level by AI_ESCALATION_STEPS (1), up to
      urgent.

3. Smoothing. A level of alert or above stands only if the model's severity >=
   ALERT_MIN_SEVERITY (2) or this is the CONSECUTIVE_FOR_ALERT-th (2nd) consecutive scored
   turn at watch or above (counted on the level before smoothing and before the allowed-
   contact cap). Otherwise it is held at watch for this turn.

4. Allowed contacts. When ctx.caller_known is true the level is capped at
   KNOWN_CALLER_CAP (watch), except for the categories in KNOWN_CALLER_EXEMPT
   (threat_extortion, self_harm), which pass through.

5. Coach (a private line to the protected person). Spoken when the level is >= alert, or
   when the level is watch, the model asked for action "coach" with a real category, the
   stage is in COACH_WATCH_STAGES (pressure, ask) and the caller is unknown
   (COACH_ON_WATCH_ACTION). Never within COACH_COOLDOWN_MS (20 s) of the previous coach
   line, unless the level is higher than the level of that previous line
   (COACH_COOLDOWN_BYPASS_ON_RISE: an early watch tip must not mute the tip at the ask).
   Never the same line twice in a call (compared after lowercasing and trimming
   punctuation). The text is the model's coach_line cut to
   COACH_MAX_WORDS (15); if it is empty, already used, or matches COACH_REJECT_PATTERN
   (digits or a web address: a tip never needs a number, and a callback number in the
   model's line is exactly what a caller would try to inject through the transcript), a
   per-family default from DEFAULT_COACH is tried; if that was used too, no coach.
   When ctx.coach_available is false (no conference or the protected leg has not joined
   yet), no coach is produced and the cooldown and used lines are left untouched, so a tip
   that could not play never mutes the next one.

6. Caretaker alert. When the level is >= alert, the caretaker is alerted once per
   category per call. A later "urgent" on an already alerted category escalates once
   (ALLOW_URGENT_ESCALATION). Level "urgent" maps to alert_level "urgent", else "alert".
   The note is the model's caretaker_note cut to NOTE_MAX_WORDS (20), or a default.

7. Suggest hang-up when the final level is urgent, or when the level is >= alert and the
   model asked for suggest_hangup or urgent_alert; never for SUGGEST_HANGUP_EXEMPT
   (self_harm: the person on the line may be the one who needs help).

ctx.rules (the caretaker's written rules) are interpreted by the model, not here; they are
carried in the context so eval logs show what the decision was made under.
Every rule that changed something adds a short code to Decision.reasons.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Literal, Optional

from contracts.models import RISK_ORDER, GuardianResponse

LEVELS = ["none", "watch", "alert", "urgent"]

AI_VOICE_THRESHOLD = 0.7
AI_VOICE_FLOOR = "watch"
AI_ESCALATION_STEPS = 1
NO_CATEGORY_CAP = "watch"
ALERT_MIN_SEVERITY = 2
CONSECUTIVE_FOR_ALERT = 2
KNOWN_CALLER_CAP = "watch"
KNOWN_CALLER_EXEMPT = frozenset({"threat_extortion", "self_harm"})
COACH_ON_WATCH_ACTION = True
COACH_WATCH_STAGES = frozenset({"pressure", "ask"})
COACH_COOLDOWN_BYPASS_ON_RISE = True
COACH_COOLDOWN_MS = 20_000
COACH_MAX_WORDS = 15
NOTE_MAX_WORDS = 20
ALLOW_URGENT_ESCALATION = True
COACH_REJECT_PATTERN = re.compile(r"\d|https?:|www\.|\.com\b", re.IGNORECASE)
SUGGEST_HANGUP_EXEMPT = frozenset({"self_harm"})

DEFAULT_COACH = {
    "scam": "This sounds like a scam. You can hang up and call a number you trust.",
    "disclosure": "You don't have to share that. Say you'll check with family first.",
    "grooming": "That's not okay to keep secret. You can hang up and tell a grown-up.",
    "vulgar": "You don't have to listen to this. You can hang up.",
    "threat_extortion": "Don't pay or promise anything. Hang up; help is coming.",
    "self_harm": "Stay calm and stay on the line. Someone who cares is being called now.",
}


@dataclass(frozen=True)
class PolicyContext:
    turn: int                       # newest turn of the scored window
    now_ms: int                     # ms since call start (caller supplies the clock)
    profile: str = "elder"
    caller_known: bool = False
    ai_voice_score: Optional[float] = None
    rules: tuple[str, ...] = ()     # caretaker's written rules (not interpreted here)
    coach_available: bool = True    # live line: conference up and protected leg joined


@dataclass(frozen=True)
class PolicyState:
    watch_streak: int = 0                       # consecutive scored turns at >= watch (pre-smoothing)
    last_coach_ms: Optional[int] = None
    last_coach_level: str = "none"
    coach_lines: frozenset[str] = frozenset()   # normalized lines already spoken
    alerted: tuple[tuple[str, str], ...] = ()   # (category, alert_level) already sent
    max_display: str = "none"


@dataclass(frozen=True)
class Decision:
    display_risk: str
    coach_text: Optional[str] = None
    alert_level: Optional[Literal["alert", "urgent"]] = None
    caretaker_note: str = ""
    suggest_hangup: bool = False
    category: str = "none"
    reasons: tuple[str, ...] = field(default_factory=tuple)


def _lvl(name: str) -> int:
    return RISK_ORDER[name]


def _cut(text: str, max_words: int) -> str:
    words = (text or "").split()
    return " ".join(words[:max_words])


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", text.lower()).strip()


def _family(category: str) -> str:
    return category.split(".")[0]


def decide(state: PolicyState, resp: GuardianResponse, ctx: PolicyContext) -> tuple[Decision, PolicyState]:
    reasons: list[str] = []
    model_level = _lvl(resp.risk)
    level = model_level

    # 1. no category -> at most watch
    if resp.category == "none" and level > _lvl(NO_CATEGORY_CAP):
        level = _lvl(NO_CATEGORY_CAP)
        reasons.append("no_category_cap")

    # 2. AI voice
    is_ai = ctx.ai_voice_score is not None and ctx.ai_voice_score >= AI_VOICE_THRESHOLD
    if is_ai:
        has_ask = resp.stage == "ask" or model_level >= _lvl("alert")
        if not ctx.caller_known and has_ask and resp.category != "none":
            new = min(level + AI_ESCALATION_STEPS, _lvl("urgent"))
            if new > level:
                level = new
                reasons.append("ai_voice_unknown_ask_escalate")
        if level < _lvl(AI_VOICE_FLOOR):
            level = _lvl(AI_VOICE_FLOOR)
            reasons.append("ai_voice_floor")

    # 3. smoothing
    streak = state.watch_streak + 1 if level >= _lvl("watch") else 0
    if level >= _lvl("alert") and resp.severity < ALERT_MIN_SEVERITY and streak < CONSECUTIVE_FOR_ALERT:
        level = _lvl("watch")
        reasons.append("smoothed_to_watch")

    # 4. allowed contacts
    if ctx.caller_known and resp.category not in KNOWN_CALLER_EXEMPT and level > _lvl(KNOWN_CALLER_CAP):
        level = _lvl(KNOWN_CALLER_CAP)
        reasons.append("known_caller_cap")

    display = LEVELS[level]
    new_state = replace(state, watch_streak=streak, max_display=LEVELS[max(level, _lvl(state.max_display))])

    # 5. coach
    coach_text: Optional[str] = None
    wants_coach = level >= _lvl("alert") or (
        COACH_ON_WATCH_ACTION
        and level == _lvl("watch")
        and resp.action == "coach"
        and resp.category != "none"
        and resp.stage in COACH_WATCH_STAGES
        and not ctx.caller_known
    )
    if wants_coach:
        cooling = state.last_coach_ms is not None and ctx.now_ms - state.last_coach_ms < COACH_COOLDOWN_MS
        if cooling and COACH_COOLDOWN_BYPASS_ON_RISE and level > _lvl(state.last_coach_level):
            cooling = False
            reasons.append("coach_cooldown_bypass_rise")
        if not ctx.coach_available:
            reasons.append("coach_unavailable")
        elif cooling:
            reasons.append("coach_cooldown")
        else:
            model_line = _cut(resp.coach_line, COACH_MAX_WORDS)
            if model_line and COACH_REJECT_PATTERN.search(model_line):
                reasons.append("coach_line_rejected")
                model_line = ""
            candidates = [model_line, DEFAULT_COACH.get(_family(resp.category), "")]
            for cand in candidates:
                if cand and _norm(cand) not in state.coach_lines:
                    coach_text = cand
                    break
            if coach_text is None:
                reasons.append("coach_repeat")
            else:
                new_state = replace(
                    new_state,
                    last_coach_ms=ctx.now_ms,
                    last_coach_level=LEVELS[level],
                    coach_lines=state.coach_lines | {_norm(coach_text)},
                )

    # 6. caretaker alert
    alert_level: Optional[str] = None
    note = ""
    if level >= _lvl("alert"):
        want = "urgent" if level == _lvl("urgent") else "alert"
        sent = {lv for cat, lv in state.alerted if cat == resp.category}
        if not sent:
            alert_level = want
            reasons.append("alert_new_category")
        elif want == "urgent" and "urgent" not in sent and ALLOW_URGENT_ESCALATION:
            alert_level = "urgent"
            reasons.append("alert_escalated")
        else:
            reasons.append("alert_already_sent")
        if alert_level:
            note = _cut(resp.caretaker_note, NOTE_MAX_WORDS) or f"Possible {resp.category} on turn {ctx.turn}."
            new_state = replace(new_state, alerted=state.alerted + ((resp.category, alert_level),))

    # 7. hang-up suggestion
    suggest = resp.category not in SUGGEST_HANGUP_EXEMPT and (
        level == _lvl("urgent") or (level >= _lvl("alert") and resp.action in ("suggest_hangup", "urgent_alert"))
    )

    decision = Decision(
        display_risk=display,
        coach_text=coach_text,
        alert_level=alert_level,
        caretaker_note=note,
        suggest_hangup=suggest,
        category=resp.category,
        reasons=tuple(reasons),
    )
    return decision, new_state

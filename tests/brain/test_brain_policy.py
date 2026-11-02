"""Unit tests for line/policy.py: one or more per documented rule."""
from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from contracts.models import GuardianResponse
from line import policy
from line.policy import PolicyContext, PolicyState, decide


def R(risk="none", category="none", severity=0, stage="hook", action="none", coach="", note="", source="caller"):
    return GuardianResponse(
        risk=risk, category=category, severity=severity, source=source, stage=stage,
        action=action, coach_line=coach, caretaker_note=note,
    )


def C(turn=1, now_ms=0, caller_known=False, ai=None, profile="elder"):
    return PolicyContext(turn=turn, now_ms=now_ms, profile=profile, caller_known=caller_known, ai_voice_score=ai)


SCAM_ASK = dict(risk="alert", category="scam.impersonation_bank_gov", severity=2, stage="ask",
                action="notify_caretaker", coach="Banks never ask for codes.", note="Caller asked for the code.")


# ---------------------------------------------------------------- basics / purity
def test_none_is_silent():
    d, s = decide(PolicyState(), R(), C())
    assert d.display_risk == "none" and d.coach_text is None and d.alert_level is None and not d.suggest_hangup


def test_pure_same_inputs_same_outputs_and_state_not_mutated():
    st = PolicyState()
    a = decide(st, R(**SCAM_ASK), C(turn=5, now_ms=1000))
    b = decide(st, R(**SCAM_ASK), C(turn=5, now_ms=1000))
    assert a == b
    assert st == PolicyState()
    with pytest.raises(FrozenInstanceError):
        st.watch_streak = 3  # type: ignore[misc]


def test_alert_with_severity_coaches_and_alerts():
    d, s = decide(PolicyState(), R(**SCAM_ASK), C(turn=5, now_ms=1000))
    assert d.display_risk == "alert"
    assert d.coach_text == "Banks never ask for codes."
    assert d.alert_level == "alert" and d.caretaker_note == "Caller asked for the code."
    assert s.last_coach_ms == 1000 and ("scam.impersonation_bank_gov", "alert") in s.alerted


# ---------------------------------------------------------------- rule 1: no category
def test_no_category_caps_at_watch():
    d, _ = decide(PolicyState(), R(risk="urgent", severity=3, stage="ask"), C())
    assert d.display_risk == "watch" and "no_category_cap" in d.reasons and d.alert_level is None


# ---------------------------------------------------------------- rule 2a: AI voice alone
def test_ai_voice_alone_is_watch_no_alert():
    d, _ = decide(PolicyState(), R(), C(ai=0.95))
    assert d.display_risk == "watch" and "ai_voice_floor" in d.reasons
    assert d.alert_level is None and d.coach_text is None


def test_ai_voice_below_threshold_ignored():
    d, _ = decide(PolicyState(), R(), C(ai=policy.AI_VOICE_THRESHOLD - 0.01))
    assert d.display_risk == "none"


def test_ai_voice_benign_stage_no_escalation():
    # pharmacy bot: low-risk content at "trust" stage, no ask
    d, _ = decide(PolicyState(), R(risk="watch", category="scam.prize", severity=1, stage="trust"), C(ai=0.95))
    assert d.display_risk == "watch" and "ai_voice_unknown_ask_escalate" not in d.reasons


# ---------------------------------------------------------------- rule 2b: AI + unknown + ask
def test_ai_unknown_ask_escalates_one_level():
    d, _ = decide(PolicyState(), R(**SCAM_ASK), C(ai=0.91))
    assert d.display_risk == "urgent" and "ai_voice_unknown_ask_escalate" in d.reasons
    assert d.alert_level == "urgent" and d.suggest_hangup


def test_ai_unknown_watch_at_ask_stage_escalates_to_alert():
    r = R(risk="watch", category="scam.impersonation_bank_gov", severity=2, stage="ask")
    d, _ = decide(PolicyState(), r, C(ai=0.91))
    assert d.display_risk == "alert"


def test_ai_known_caller_does_not_escalate():
    d, _ = decide(PolicyState(), R(**SCAM_ASK), C(ai=0.91, caller_known=True))
    assert d.display_risk == "watch" and "ai_voice_unknown_ask_escalate" not in d.reasons


def test_escalation_capped_at_urgent():
    r = R(risk="urgent", category="threat_extortion", severity=3, stage="ask")
    d, _ = decide(PolicyState(), r, C(ai=0.99))
    assert d.display_risk == "urgent"


# ---------------------------------------------------------------- rule 3: smoothing
def test_low_severity_single_alert_is_held_at_watch():
    r = R(risk="alert", category="scam.tech_support", severity=1, stage="pressure")
    d, s = decide(PolicyState(), r, C())
    assert d.display_risk == "watch" and "smoothed_to_watch" in d.reasons and d.alert_level is None
    assert s.watch_streak == 1


def test_second_consecutive_turn_at_watch_allows_alert():
    st = PolicyState()
    _, st = decide(st, R(risk="watch", category="scam.tech_support", severity=1, stage="trust"), C(turn=1))
    d, st = decide(st, R(risk="alert", category="scam.tech_support", severity=1, stage="pressure"), C(turn=2))
    assert d.display_risk == "alert" and d.alert_level == "alert"


def test_streak_resets_on_none():
    st = PolicyState()
    _, st = decide(st, R(risk="watch", category="scam.tech_support", severity=1), C(turn=1))
    _, st = decide(st, R(), C(turn=2))
    assert st.watch_streak == 0
    d, _ = decide(st, R(risk="alert", category="scam.tech_support", severity=1), C(turn=3))
    assert d.display_risk == "watch"


# ---------------------------------------------------------------- rule 4: allowed contacts
def test_known_caller_capped_at_watch_and_silent():
    d, _ = decide(PolicyState(), R(**SCAM_ASK), C(caller_known=True))
    assert d.display_risk == "watch" and "known_caller_cap" in d.reasons
    assert d.coach_text is None and d.alert_level is None


@pytest.mark.parametrize("cat", sorted(policy.KNOWN_CALLER_EXEMPT))
def test_known_caller_exempt_categories_pass(cat):
    r = R(risk="urgent", category=cat, severity=3, stage="pressure", coach="Hang up now.", note="Threat.")
    d, _ = decide(PolicyState(), r, C(caller_known=True))
    assert d.display_risk == "urgent" and d.alert_level == "urgent" and d.coach_text == "Hang up now."


def test_known_caller_no_coach_on_watch_action():
    r = R(risk="watch", category="disclosure.home_alone", severity=1, stage="ask", action="coach",
          coach="Don't say who is home.")
    d, _ = decide(PolicyState(), r, C(caller_known=True))
    assert d.coach_text is None


# ---------------------------------------------------------------- rule 5: coach
def test_coach_on_watch_when_model_asks_for_coach_at_pressure_or_ask():
    r = R(risk="watch", category="disclosure.home_alone", severity=1, stage="ask", action="coach",
          coach="Don't say who is home.")
    d, _ = decide(PolicyState(), r, C())
    assert d.coach_text == "Don't say who is home." and d.alert_level is None


def test_no_coach_on_watch_at_hook_or_trust():
    # live stand-in did this on "Oh, hello. Is something wrong?": too early to coach
    r = R(risk="watch", category="scam.impersonation_bank_gov", severity=1, stage="trust", action="coach",
          coach="Banks never call out of the blue.")
    d, _ = decide(PolicyState(), r, C())
    assert d.coach_text is None


def test_cooldown_bypassed_when_level_rises():
    st = PolicyState()
    r1 = R(risk="watch", category="scam.impersonation_bank_gov", severity=1, stage="pressure", action="coach",
           coach="Take your time; you can call the bank back.")
    d1, st = decide(st, r1, C(turn=3, now_ms=5_000))
    d2, st = decide(st, R(**SCAM_ASK), C(turn=5, now_ms=8_000))
    assert d1.coach_text and d2.coach_text == "Banks never ask for codes." and "coach_cooldown_bypass_rise" in d2.reasons
    d3, _ = decide(st, R(**{**SCAM_ASK, "coach": "Another alert line."}), C(turn=7, now_ms=9_000))
    assert d3.coach_text is None and "coach_cooldown" in d3.reasons


def test_coach_cooldown():
    st = PolicyState()
    d1, st = decide(st, R(**SCAM_ASK), C(turn=3, now_ms=10_000))
    r2 = R(**{**SCAM_ASK, "coach": "A different line entirely."})
    d2, st = decide(st, r2, C(turn=5, now_ms=10_000 + policy.COACH_COOLDOWN_MS - 1))
    assert d1.coach_text and d2.coach_text is None and "coach_cooldown" in d2.reasons
    d3, _ = decide(st, r2, C(turn=7, now_ms=10_000 + policy.COACH_COOLDOWN_MS))
    assert d3.coach_text == "A different line entirely."


def test_never_same_coach_line_twice_falls_back_then_stops():
    st = PolicyState()
    d1, st = decide(st, R(**SCAM_ASK), C(now_ms=0))
    d2, st = decide(st, R(**{**SCAM_ASK, "coach": "banks never ask for codes"}), C(now_ms=60_000))
    assert d1.coach_text == "Banks never ask for codes."
    assert d2.coach_text == policy.DEFAULT_COACH["scam"]  # model repeated itself -> family default
    d3, _ = decide(st, R(**SCAM_ASK), C(now_ms=120_000))
    assert d3.coach_text is None and "coach_repeat" in d3.reasons


def test_coach_cut_to_15_words_and_default_when_empty():
    long = " ".join(f"w{i}" for i in range(30))
    d, _ = decide(PolicyState(), R(**{**SCAM_ASK, "coach": long}), C())
    assert len(d.coach_text.split()) == policy.COACH_MAX_WORDS
    d2, _ = decide(PolicyState(), R(**{**SCAM_ASK, "category": "disclosure.location_school", "coach": ""}), C())
    assert d2.coach_text == policy.DEFAULT_COACH["disclosure"]


def test_coach_unavailable_does_not_consume_cooldown_or_line():
    st = PolicyState()
    d1, st = decide(st, R(**SCAM_ASK), PolicyContext(turn=3, now_ms=1_000, coach_available=False))
    assert d1.coach_text is None and "coach_unavailable" in d1.reasons and d1.alert_level == "alert"
    assert st.last_coach_ms is None and not st.coach_lines
    d2, _ = decide(st, R(**SCAM_ASK), C(turn=5, now_ms=2_000))
    assert d2.coach_text == "Banks never ask for codes."


@pytest.mark.parametrize("line", ["Call your bank at 555 0100 now.", "Visit www.safe-bank.com to check.",
                                  "Go to https://x.io and log in."])
def test_model_coach_line_with_number_or_url_replaced_by_default(line):
    d, _ = decide(PolicyState(), R(**{**SCAM_ASK, "coach": line}), C())
    assert d.coach_text == policy.DEFAULT_COACH["scam"] and "coach_line_rejected" in d.reasons


def test_self_harm_never_suggests_hangup():
    d, _ = decide(PolicyState(), R(risk="urgent", category="self_harm", severity=3, stage="pressure",
                                   action="urgent_alert"), C())
    assert d.display_risk == "urgent" and d.alert_level == "urgent" and not d.suggest_hangup


# ---------------------------------------------------------------- rule 6: caretaker alert
def test_alert_once_per_category():
    st = PolicyState()
    d1, st = decide(st, R(**SCAM_ASK), C(turn=3))
    d2, st = decide(st, R(**SCAM_ASK), C(turn=5))
    assert d1.alert_level == "alert" and d2.alert_level is None and "alert_already_sent" in d2.reasons


def test_new_category_alerts_again():
    st = PolicyState()
    _, st = decide(st, R(**SCAM_ASK), C(turn=3))
    d, _ = decide(st, R(**{**SCAM_ASK, "category": "disclosure.credentials_otp", "source": "protected_person"}), C(turn=4))
    assert d.alert_level == "alert"


def test_escalation_to_urgent_allowed_once():
    st = PolicyState()
    _, st = decide(st, R(**SCAM_ASK), C(turn=3))
    d, st = decide(st, R(**{**SCAM_ASK, "risk": "urgent", "severity": 3}), C(turn=5))
    assert d.alert_level == "urgent" and "alert_escalated" in d.reasons
    d2, _ = decide(st, R(**{**SCAM_ASK, "risk": "urgent", "severity": 3}), C(turn=7))
    assert d2.alert_level is None


def test_note_cut_and_default():
    d, _ = decide(PolicyState(), R(**{**SCAM_ASK, "note": " ".join(["x"] * 40)}), C())
    assert len(d.caretaker_note.split()) == policy.NOTE_MAX_WORDS
    d2, _ = decide(PolicyState(), R(**{**SCAM_ASK, "note": ""}), C(turn=9))
    assert "turn 9" in d2.caretaker_note


# ---------------------------------------------------------------- rule 7: hang-up
def test_suggest_hangup():
    d, _ = decide(PolicyState(), R(**{**SCAM_ASK, "action": "suggest_hangup"}), C())
    assert d.suggest_hangup
    d2, _ = decide(PolicyState(), R(**SCAM_ASK), C())
    assert not d2.suggest_hangup
    d3, _ = decide(PolicyState(), R(**{**SCAM_ASK, "action": "suggest_hangup"}), C(caller_known=True))
    assert not d3.suggest_hangup


def test_docstring_names_every_constant():
    doc = policy.__doc__
    for name in ["AI_VOICE_THRESHOLD", "AI_VOICE_FLOOR", "AI_ESCALATION_STEPS", "NO_CATEGORY_CAP",
                 "ALERT_MIN_SEVERITY", "CONSECUTIVE_FOR_ALERT", "KNOWN_CALLER_CAP", "KNOWN_CALLER_EXEMPT",
                 "COACH_ON_WATCH_ACTION", "COACH_WATCH_STAGES", "COACH_COOLDOWN_BYPASS_ON_RISE", "COACH_COOLDOWN_MS", "COACH_MAX_WORDS", "NOTE_MAX_WORDS",
                 "DEFAULT_COACH", "ALLOW_URGENT_ESCALATION", "COACH_REJECT_PATTERN", "SUGGEST_HANGUP_EXEMPT"]:
        assert name in doc, name

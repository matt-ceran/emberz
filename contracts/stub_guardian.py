"""Stub guardian: canned GuardianResponse for any GuardianRequest.

Lets the line, policy and console be built before the real model exists.
Trigger words in the last caller turn return an alert; otherwise none.
"""
from __future__ import annotations

from contracts.models import Evidence, GuardianRequest, GuardianResponse

_TRIGGERS = {
    "code": ("scam.impersonation_bank_gov", "Banks never ask for codes. Say you'll call the number on your card."),
    "gift card": ("scam.impersonation_bank_gov", "Nobody legit asks for gift cards. You can hang up."),
    "street": ("disclosure.location_school", "Don't share where you live. Ask a parent."),
    "home alone": ("disclosure.home_alone", "Don't say who is home. Ask a parent."),
}


def score(req: GuardianRequest) -> GuardianResponse:
    last = next((t for t in reversed(req.turns) if t.speaker == "caller"), None)
    if last:
        low = last.text.lower()
        for word, (cat, coach) in _TRIGGERS.items():
            if word in low:
                return GuardianResponse(
                    risk="alert", category=cat, severity=2, source="caller", stage="ask",
                    evidence=[Evidence(turn=last.turn, quote=last.text)],
                    action="notify_caretaker", coach_line=coach,
                    caretaker_note=f"Caller asked about '{word}'.",
                )
    return GuardianResponse(risk="none", category="none", severity=0, source="caller", stage="hook", action="none")

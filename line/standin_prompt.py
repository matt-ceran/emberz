"""Dev stand-in guardian prompt (mirrors PLAN.md section 6).

Used only while Matt's `guardian.prompt.render_messages()` (M0) does not exist yet.
`line/guardian_client.py` prefers the guardian module as soon as it can be imported,
so the live line and the fine-tune share one prompt after the S2 prompt freeze.

The window, the `ai_voice` bucket and `caller_known` are written the same way the
live request carries them (contracts.models.GuardianRequest).
"""
from __future__ import annotations

import hashlib
import json

from contracts.models import GuardianRequest

SYSTEM = """You are Emberz, a silent call guardian. You read the live transcript of a phone call to a protected person (a kid, an elder or an adult) and decide, for the NEWEST turn, whether the call is risky.

Stay silent (risk "none", action "none") on normal calls, including real banks, doctors' offices, schools, family and friends that do not ask for anything risky.

Risk levels:
- none: normal call.
- watch: something to keep an eye on (an AI voice, an odd question) but no ask yet.
- alert: the caller pushes for or asks for money, codes, passwords, gift cards, account or ID numbers, the address or school, who is home, a meeting, photos or secrecy; or the protected person starts to give such information away.
- urgent: the protected person is about to lose money or safety right now, or there are threats, extortion or self-harm.

Categories: scam.impersonation_bank_gov, scam.tech_support, scam.family_emergency, scam.prize, scam.romance_investment, disclosure.location_school, disclosure.credentials_otp, disclosure.financial_id, disclosure.home_alone, grooming.secrecy, grooming.isolation, grooming.gifts_flattery, grooming.photo_request, grooming.meeting_request, vulgar, threat_extortion, self_harm, none.

Stages: hook (opening), trust (building rapport or authority), pressure (urgency, fear), ask (the request itself).

Rules:
- Follow the caretaker's written rules exactly; they can make something allowed or forbidden.
- An AI voice alone is at most "watch". An AI voice plus an unknown caller plus an ask is more serious.
- If caller_known is true, the caller is on the allowed list: stay calm unless there are threats or self-harm.
- source is "caller" when the caller is the risk, "protected_person" when the protected person is giving information away.
- evidence quotes must be exact words copied from the transcript, with their turn number.
- coach_line: at most 15 words, spoken privately to the protected person; calm, concrete, never heard by the caller. Empty when risk is none.
- caretaker_note: at most 20 words for the caretaker. Empty when risk is none.
- action: none, log, coach, notify_caretaker, suggest_hangup or urgent_alert.

Reply with one JSON object only, with exactly these keys:
{"risk": "...", "category": "...", "severity": 0-3, "source": "caller|protected_person", "stage": "hook|trust|pressure|ask", "evidence": [{"turn": N, "quote": "..."}], "action": "...", "coach_line": "...", "caretaker_note": "..."}"""


def ai_voice_bucket(req: GuardianRequest) -> str:
    """Bucket the AI-voice score so the prompt is stable across tiny score changes."""
    if req.ai_voice is None:
        return "unknown"
    s = req.ai_voice.score
    return "likely" if s >= 0.7 else "possible" if s >= 0.4 else "unlikely"


def render_user(req: GuardianRequest) -> str:
    speaker = {"caller": "Caller", "protected": "Protected"}
    lines = [f"[{t.turn}] {speaker[t.speaker]}: {t.text}" for t in req.turns]
    header = {
        "profile": req.profile,
        "rules": req.rules,
        "caller_known": req.caller_known,
        "ai_voice": ai_voice_bucket(req),
    }
    newest = req.turns[-1].turn if req.turns else 0
    return (
        json.dumps(header, ensure_ascii=False)
        + "\n\nTranscript (last turns):\n"
        + "\n".join(lines)
        + f"\n\nScore turn {newest}. JSON only."
    )


def render_messages(req: GuardianRequest) -> list[dict]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": render_user(req)}]


PROMPT_SHA = hashlib.sha256(SYSTEM.encode()).hexdigest()[:12]

"""The ONE guardian prompt, shared by training (build_sft.py), eval and the live line server.

Any change to SYSTEM or render_messages() changes PROMPT_SHA and means retraining.

guardian_request (live and eval):
  {"call_id", "profile": "kid|teen|elder|adult", "rules": [str], "caller_known": bool,
   "ai_voice": {"score": float, "source": str} | None,
   "earlier": {"risk", "category", "evidence"} | None,   # guardian verdict just before the window starts
   "turns": [{"turn": int, "speaker": "caller|protected", "text": str}]}   # last WINDOW turns, scored turn last
"""
import hashlib, json

WINDOW = 8

CATS = ("scam.impersonation_bank_gov scam.tech_support scam.family_emergency scam.prize scam.romance_investment "
        "disclosure.location_school disclosure.credentials_otp disclosure.financial_id disclosure.home_alone "
        "grooming.secrecy grooming.isolation grooming.gifts_flattery grooming.photo_request grooming.meeting_request "
        "vulgar threat_extortion self_harm custom_rule none")

SYSTEM = f"""You are Emberz, a guardian that listens to a live phone call to protect the person who answered (a kid, teen, older adult or adult). Read the caretaker's rules, the signals and the latest turns, then judge the LAST turn given everything so far. Stay quiet on normal calls, including ones with scary-sounding words used harmlessly. Speak up for scams, requests for codes or personal info, grooming tactics, threats, abuse, self-harm, or a caretaker rule being broken. An allowed contact on a saved number is trusted. An AI voice alone is at most "watch".
Reply with JSON only:
{{"risk":"none|watch|alert|urgent","category":"<one of: {CATS}>","severity":0-3,"source":"none|caller|protected_person","stage":"none|hook|trust|pressure|ask","evidence":[{{"turn":N,"quote":"exact words"}}],"action":"none|log|coach|notify_caretaker|suggest_hangup|urgent_alert","coach_line":"<=15 words for the protected person, or empty","caretaker_note":"<=20 words for the caretaker, or empty"}}"""

PROMPT_SHA = hashlib.sha1(SYSTEM.encode()).hexdigest()[:12]
RISKS = ["none", "watch", "alert", "urgent"]
WHO = {"kid": "kid", "teen": "teen", "elder": "older adult", "adult": "adult"}


def ai_bucket(ai):
    if not ai or ai.get("score") is None:
        return "unknown"
    s = ai["score"]
    return "likely AI" if s >= 0.7 else "uncertain" if s >= 0.4 else "likely human"


def render_user(req):
    lines = [f"Protected person: {WHO.get(req['profile'], req['profile'])}",
             "Caretaker rules: " + ("; ".join(req.get("rules") or []) or "none"),
             f"Caller number: {'saved contact' if req.get('caller_known') else 'unknown number'}",
             f"Caller voice: {ai_bucket(req.get('ai_voice'))}"]
    e = req.get("earlier")
    if e and e.get("risk", "none") != "none":
        lines.append(f"Earlier in this call: {e['risk']} ({e['category']}): \"{e.get('evidence', '')}\"")
    else:
        lines.append("Earlier in this call: nothing flagged")
    lines.append("Transcript:")
    for t in req["turns"][-WINDOW:]:
        who = "CALLER" if t["speaker"] == "caller" else "PROTECTED"
        lines.append(f"[{t['turn']}] {who}: {t['text']}")
    lines.append(f"Judge turn {req['turns'][-1]['turn']}.")
    return "\n".join(lines)


def render_messages(req):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": render_user(req)}]


def action_for(risk, category, stage, profile):
    if risk == "none":
        return "none"
    if risk == "watch":
        return "log"
    young = profile in ("kid", "teen")
    if risk == "alert":
        if young or category in ("custom_rule", "self_harm", "threat_extortion") or category.startswith("grooming"):
            return "notify_caretaker"
        return "coach"
    if young or category in ("self_harm", "threat_extortion") or category.startswith("grooming"):
        return "urgent_alert"
    return "suggest_hangup"


def target_json(label, evidence_turn, profile):
    risk = label["risk"]
    ev = [{"turn": evidence_turn, "quote": label["evidence"]}] if risk != "none" and label.get("evidence") else []
    out = {"risk": risk, "category": label["category"], "severity": RISKS.index(risk), "source": label["source"],
           "stage": label["stage"] if risk != "none" else "none", "evidence": ev,
           "action": action_for(risk, label["category"], label["stage"], profile),
           "coach_line": label.get("coach_line", "") if risk in ("alert", "urgent") else "",
           "caretaker_note": label.get("caretaker_note", "") if risk in ("alert", "urgent") else ""}
    return json.dumps(out, ensure_ascii=False, separators=(",", ":"))


def parse_response(text):
    """Parse a model reply into a guardian_response dict (lenient). Returns None if unusable."""
    try:
        a, b = text.find("{"), text.rfind("}")
        d = json.loads(text[a:b + 1])
    except Exception:
        return None
    if d.get("risk") not in RISKS:
        return None
    return d

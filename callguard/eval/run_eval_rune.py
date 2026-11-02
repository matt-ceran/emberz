"""Baseline B1: Rune (Fireworks System One decision model), zero-shot, scored with the same harness as every other system.

Rune does not chat; it returns calibrated probabilities for typed questions. We give it exactly what the guardian
sees (render_user: profile, caretaker rules, caller known, AI-voice bucket, earlier verdict, transcript window) and ask:
  - alert (yes/no): should the caretaker be alerted now?
  - category (choice): which guardian category fits best?
P(alert) >= --threshold becomes risk "alert" (>= 0.9 becomes "urgent"); everything else is "none".

  python3 -I callguard/eval/run_eval_rune.py --name b1-rune --workers 16            # all held-out calls, Numan's key
  python3 -I callguard/eval/run_eval_rune.py --name b1-rune --max-calls 150          # same subset as B4 Kimi (reuses cache)
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
import run_eval as RE  # noqa: E402

URL = "https://api.fireworks.ai/inference/v1/systemone"
THRESH = float(os.environ.get("RUNE_THRESHOLD", "0.5"))

CATEGORY_DESC = {
    "scam.impersonation_bank_gov": "Caller impersonates a bank, government agency, police or court to get money or info",
    "scam.tech_support": "Fake tech support: infected computer, remote-access app, payment for repair",
    "scam.family_emergency": "Fake relative or lawyer in trouble asking for bail or emergency money",
    "scam.prize": "Fake prize, lottery or sweepstakes that requires a fee",
    "scam.romance_investment": "Online admirer or friend pushing an investment, crypto or money transfer",
    "disclosure.location_school": "Stranger asks for home address, street or school",
    "disclosure.credentials_otp": "Request for one-time codes, passwords or account logins",
    "disclosure.financial_id": "Request for card, bank, Social Security or Medicare numbers",
    "disclosure.home_alone": "Stranger asks whether the child is home alone or when parents return",
    "grooming.secrecy": "Adult asks a child to keep the conversation secret from parents",
    "grooming.isolation": "Adult tries to move a child to another app or away from parents",
    "grooming.gifts_flattery": "Adult flatters a child or offers gifts or game currency to build trust",
    "grooming.photo_request": "Adult asks a child to send pictures of themselves",
    "grooming.meeting_request": "Adult proposes to meet a child in person",
    "vulgar": "Profanity, slurs or abusive insults aimed at the protected person",
    "threat_extortion": "Threats of harm or demands under threat",
    "self_harm": "Talk of the protected person hurting themselves or being told to",
    "custom_rule": "The call breaks one of the caretaker's written rules",
    "none": "Nothing concerning",
}

QUESTIONS = {
    "alert": {
        "type": "noul",
        "instructions": "Should the caretaker be alerted about this call right now, judging the latest turn given everything so far?",
        "criteria": {
            "true": "A clear warning sign has appeared: a scam or impersonation, a request for codes, passwords, account or ID numbers, "
                    "address, school or whether a child is home alone, grooming tactics (secrecy, gifts or flattery to a child, moving to "
                    "another app, photo requests, meeting requests), threats, slurs or abuse, self-harm, or one of the caretaker's rules being broken.",
            "false": "A normal call, including ones that use scary-sounding words harmlessly: a real bank alert that asks for nothing, a friend's "
                     "parent arranging a playdate, a school or pharmacy reminder, a relative on a new number asking for nothing, a wrong number, "
                     "a saved or allowed contact. An AI voice alone is not enough.",
        },
    },
    "category": {"type": "choice", "instructions": "Which category best describes the most serious issue so far?", "criteria": CATEGORY_DESC},
}


def rune_chat(model, messages, key_env="FIREWORKS_API_KEY", **_):
    if model != "rune":
        return _orig_chat(model, messages, key_env=key_env, **_)
    state = messages[-1]["content"]
    body = {"model": "accounts/fireworks/models/rune", "state": state, "questions": QUESTIONS}
    last = None
    for attempt in range(6):
        req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {os.environ[key_env]}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                d = json.loads(r.read())
            a = d["answers"]
            p = float(a["alert"]["noul"])
            cat = a["category"]["choice"]
            risk = "urgent" if p >= 0.9 and p >= THRESH else "alert" if p >= THRESH else "none"
            v = {"risk": risk, "category": cat if risk != "none" else "none", "evidence": [], "p_alert": round(p, 4)}
            return json.dumps(v), {}, "stop"
        except urllib.error.HTTPError as e:
            last = f"http {e.code}: {e.read()[:200]!r}"
            time.sleep(1.5 * (attempt + 1) if e.code in (429, 503) else 0.5)
        except Exception as e:  # noqa: BLE001
            last = repr(e)[:200]
            time.sleep(1.0)
    raise RuntimeError(last)


_orig_chat = RE.chat
RE.chat = rune_chat

if __name__ == "__main__":
    if "--model" not in sys.argv:
        sys.argv += ["--model", "rune"]
    RE.main()

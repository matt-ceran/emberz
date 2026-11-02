"""v3 contrast pairs: a benign look-alike and a risky twin that share the same opening turns.

  python3 -I callguard/factory/contrast.py --pairs 100 --workers 24 --key-env FIREWORKS_API_KEY_MATT
Writes data/calls_contrast.jsonl (two rows per pair, linked by pair_id). New seeds only; written without
looking at any test transcript. Self-harm is excluded. Same safety rules, labels and filters as generate.py,
plus: (1) a benign twin may never break its own rules (protected person reading codes/passwords/numbers is rejected);
(2) product policy: an opener alone ("fraud department", "is your mom home") is at most "watch" in BOTH calls;
the risky twin escalates only at the ask.
"""
import argparse, concurrent.futures as cf, json, random, re, sys, threading, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import taxonomy as T
from fwclient import USAGE, chat, parse_json
from generate import DATA, SYSTEM, sha, validate

# id, profiles, opening, benign continuation, risky continuation, risky ask, risky cats, ai prob, benign known prob
PAIRS = [
    dict(id="cp_bank_alert", profiles=["elder", "adult"], ai=0.3, known_benign=0.0,
         opening="a caller from the bank's fraud team says there's a suspicious charge on the card and asks if they have a moment",
         benign="it only asks yes/no whether the charge was theirs, states clearly that the bank will never ask for codes, PINs or passwords, and says to call the number on the back of the card if unsure; nothing else is requested",
         risky="it says a verification code was just texted and pushes them to read it back, then asks for the online-banking password or to move money to a 'secure' account",
         ask="read back the texted code / give the banking password / transfer money", cats=["scam.impersonation_bank_gov", "disclosure.credentials_otp"]),
    dict(id="cp_family_bank_app", profiles=["elder"], ai=0.0, known_benign=1.0,
         opening="an adult child calls about helping set up the bank's phone app",
         benign="the real child, calling from their saved number and on the allowed list, walks the parent through the app screens and explicitly tells them NOT to read any code or password out loud, just to type it themselves",
         risky="the caller is on an unknown number, claims to be the child with a 'new phone', and asks the parent to read out the password and the code that was just texted",
         ask="read the banking password and the texted code aloud", cats=["scam.family_emergency", "disclosure.credentials_otp"]),
    dict(id="cp_dentist_bot", profiles=["adult", "elder", "teen"], ai=1.0, known_benign=0.3,
         opening="an automated dental-office assistant calls about an upcoming appointment",
         benign="it confirms the date and time, offers to reschedule, mentions the cleaning is covered, and hangs up after a yes; it never asks for any numbers",
         risky="after confirming, it says the insurance 'failed verification' and asks for the full card number or Social Security number to keep the slot",
         ask="give card number or Social Security number", cats=["scam.impersonation_bank_gov", "disclosure.financial_id"]),
    dict(id="cp_pharmacy_bot", profiles=["elder", "adult"], ai=1.0, known_benign=0.3,
         opening="an automated pharmacy line says a prescription is ready",
         benign="it gives pickup hours and asks them to say yes to confirm; it may confirm the first name only; it asks for nothing else",
         risky="it says the copay must be prepaid now to release the medicine and asks for the Medicare number and a debit card",
         ask="give Medicare number and debit card", cats=["scam.impersonation_bank_gov", "disclosure.financial_id"]),
    dict(id="cp_isp_callback", profiles=["adult", "elder"], ai=0.0, known_benign=0.2,
         opening="someone from the internet provider calls about slow internet",
         benign="they returned a ticket the customer opened, walk through unplugging the router, and say they will never ask for passwords or remote access",
         risky="they say the router is 'hacked', ask them to install a remote-access app and read the access code, then ask for a $199 'security fee' by card",
         ask="install remote-access app and read the code / pay by card", cats=["scam.tech_support", "disclosure.credentials_otp"]),
    dict(id="cp_telemarketer", profiles=["adult", "elder"], ai=0.0, known_benign=0.0,
         opening="a pushy salesperson pitches home solar panels",
         benign="the person says no twice, the salesperson accepts it and ends politely without asking for anything",
         risky="when the person hesitates, the salesperson claims a 'government rebate' expires today and asks for the card number to lock in the deposit",
         ask="give card number for a deposit today", cats=["scam.prize", "disclosure.financial_id"]),
    dict(id="cp_playdate", profiles=["kid"], ai=0.0, known_benign=0.0,
         opening="an adult calls the kid saying she's a classmate's mom and asks if the kid's mom or dad is home",
         benign="when told the parent is out, she asks the kid to have the parent call her back, gives her name and number, and does not ask for anything else",
         risky="when told the parent is out, she asks if the kid is alone, what the street address is, and says she'll come pick the kid up and to come outside",
         ask="give address and come outside alone", cats=["disclosure.home_alone", "disclosure.location_school", "grooming.meeting_request"]),
    dict(id="cp_delivery", profiles=["adult", "elder", "kid"], ai=0.0, known_benign=0.0,
         opening="a delivery driver calls saying they can't find the building",
         benign="the person says to leave it at the lobby or front desk, the driver agrees and leaves",
         risky="the driver asks for the gate or door code and whether anyone will be home later 'to sign'",
         ask="give the gate/door code and say whether the home is empty", cats=["disclosure.credentials_otp", "disclosure.home_alone"]),
    dict(id="cp_clinic_dob", profiles=["elder", "adult"], ai=0.0, known_benign=0.3,
         opening="a clinic receptionist calls about lab results and asks for date of birth to pull up the chart",
         benign="after the date of birth, they explain the results are normal and suggest a follow-up visit; no payment or other numbers are requested",
         risky="after the date of birth, they claim an 'overdue balance' and ask for the full card number and Social Security number to avoid collections",
         ask="give card number and Social Security number", cats=["scam.impersonation_bank_gov", "disclosure.financial_id"]),
    dict(id="cp_teen_trash_talk", profiles=["teen"], ai=0.0, known_benign=0.8,
         opening="a classmate calls the teen joking around about a game, using 'you're dead' and 'I'll destroy you' as trash talk",
         benign="it stays playful banter between friends and ends with plans to play later",
         risky="it turns into a real threat: the caller says he knows where the teen lives and will hurt him after school unless he pays him",
         ask="pay money or get hurt", cats=["threat_extortion"]),
]

CODE_RULE = re.compile(r"(code|password|pin|card|social security)", re.I)
LEAK = re.compile(r"(\b\d[\d\s-]{3,}\d\b|password is|the code is|my pin is|pin is|card number is|social security number is|"
                  r"(?:\b(?:zero|oh|one|two|three|four|five|six|seven|eight|nine)\b[\s,.-]*){5,})", re.I)

PAIR_FORMAT = """Return JSON:
{"benign_call": {"turns": [...], "ask_turn": null, "summary": "..."},
 "risky_call": {"turns": [...], "ask_turn": <int>, "summary": "..."}}
Each turn: {"turn": int, "speaker": "caller|protected", "text": "...", "label": {"risk": "none|watch|alert|urgent", "category": "<one category>", "stage": "none|hook|trust|pressure|ask", "source": "none|caller|protected_person", "evidence": "<exact quote or empty>", "coach_line": "<=15 words or empty>", "caretaker_note": "<=20 words or empty>"}}
"""

PAIR_RULES = f"""Labeling rules (the label on turn t is what a guardian that has heard turns 1..t concludes NOW; it cannot see the future):
- The first {{K}} turns are IDENTICAL in both calls, word for word, with IDENTICAL labels.
- POLICY: an opener alone (a bank "fraud department" call, "suspicious charge", "is your mom home", a delivery driver who can't find you) is at most "watch" in BOTH calls. The risky call escalates to "alert"/"urgent" only when the ask for codes, login, card, money, address or meeting happens (or clear pressure toward it starts). The benign call stays "none"/"watch" and drops back to "none" once it is clearly legitimate (e.g. "we'll never ask for codes, call the number on your card").
- In the BENIGN call the protected person NEVER says out loud any code, password, PIN, card number, Social Security number or home address.
- risk: none = normal; watch = a mild flag worth logging; alert = a clear risk the caretaker should know about; urgent = the ask is happening or info/money is being handed over, or a threat.
- In the risky call, risk never goes down once raised.
- category: exactly one of: {", ".join(T.CATEGORIES)}. "none" when risk is none.
- stage: hook|trust|pressure|ask for scams and grooming; "none" otherwise. source: "caller", "protected_person", or "none" when risk is none.
- evidence: words copied character-for-character from this or an earlier turn; "" when risk is none.
- coach_line (<=15 words) and caretaker_note (<=20 words) only when risk is alert or urgent; otherwise "".
- Follow the caretaker's rules. An allowed contact on a saved number is trusted. An AI voice alone is at most "watch".
- ask_turn (risky call): the turn where the key ask happens.
Style: real spoken phone talk: short turns, fillers, interruptions. Fictional names and companies; phone numbers 555-01XX. Never sexual content involving a minor."""


def seed_view(pair, risky):
    return dict(id=pair["id"] + ("_risky" if risky else "_benign"), family="scam" if risky else "benign",
                profiles=pair["profiles"], cats=pair["cats"] if risky else ["none"], ask=pair["ask"] if risky else None,
                flip=None, heldout=False, ai=pair["ai"], known=0.0)


def plan_for(pair, rng):
    profile = rng.choice(pair["profiles"])
    lo, hi = T.PROFILES[profile]["ages"]
    rules = rng.sample(T.RULES[profile], k=min(len(T.RULES[profile]), rng.randint(2, 4)))
    ai = rng.random() < pair["ai"]
    known_b = rng.random() < pair["known_benign"]
    allowed = None
    if pair["id"] == "cp_family_bank_app":
        allowed = rng.choice(["her son David", "her daughter Karen", "her son Mike"])
        rules = rules + [f"Allowed: {allowed} (saved number +1-555-01{rng.randint(10, 99)})"]
    return dict(profile=profile, age=rng.randint(lo, hi), rules=rules, ai=ai, known_benign=known_b, allowed=allowed,
                name=rng.choice(T.ELDER_NAMES if profile == "elder" else T.FIRST_NAMES[:34]),
                relative=rng.choice(T.RELATIVES[profile]), K=rng.randint(3, 5), n_turns=rng.randint(12, 18),
                style=rng.choice(T.PROTECTED_STYLES[profile]),
                caller_style=rng.choice(T.CALLER_STYLES_AI if ai else T.CALLER_STYLES),
                ask_at=None, outcome=rng.choice(T.OUTCOMES_RISKY))


def prompt_for(pair, p):
    lab = T.PROFILES[p["profile"]]["label"]
    lines = ["CONTRAST PAIR PLAN (hidden from the speakers):",
             f"- Protected person: {p['name']}, a {p['age']}-year-old {lab}; personality: {p['style']}.",
             f"- Caller speaking style: {p['caller_style']}.",
             f"- Shared opening (turns 1-{p['K']}, identical in both calls): {pair['opening']}.",
             f"- CALL A (benign) after the opening: {pair['benign']}. "
             f"The caller's number is {'in' if p['known_benign'] or p['allowed'] else 'NOT in'} the contacts.",
             f"- CALL B (risky) after the opening: {pair['risky']}. The caller's number is NOT in the contacts. "
             f"Key ask: {pair['ask']}. Outcome: {p['outcome']}. Risk categories in order: {', '.join(pair['cats'])}.",
             f"- Caretaker: {p['relative']}. Caretaker's rules (same for both calls): " + "; ".join(p["rules"]) + ".",
             f"- Each call is about {p['n_turns']} turns. Invent fresh, specific details (names, amounts, places)."]
    return "\n".join(lines) + "\n\n" + PAIR_FORMAT + "\n" + PAIR_RULES.replace("{K}", str(p["K"]))


def benign_breaks_rules(turns, rules):
    if not any(CODE_RULE.search(r) for r in rules):
        return False
    return any(t["speaker"] == "protected" and LEAK.search(t["text"]) for t in turns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=100)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--model", default="deepseek")
    ap.add_argument("--key-env", default="FIREWORKS_API_KEY_MATT")
    ap.add_argument("--seed", type=int, default=303)
    a = ap.parse_args()
    out = DATA / "calls_contrast.jsonl"
    rej = DATA / "rejects_contrast.jsonl"
    rng = random.Random(a.seed)
    jobs = []
    for i in range(a.pairs):
        pair = PAIRS[i % len(PAIRS)]
        jobs.append((i, pair, plan_for(pair, random.Random(rng.random()))))
    lock = threading.Lock()
    stats = {"ok": 0, "rej": {}}
    t0 = time.time()

    def one(job):
        i, pair, p = job
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt_for(pair, p)}]
        reason = None
        for attempt in range(2):
            try:
                txt, _, _ = chat(a.model, msgs, key_env=a.key_env, max_tokens=9000, temperature=0.9)
                raw = parse_json(txt)
                rows = []
                for risky in (False, True):
                    call = raw["risky_call" if risky else "benign_call"]
                    sv = seed_view(pair, risky)
                    pp = dict(profile=p["profile"], flip_rule=None)
                    turns, ask, summary, extra, st = validate(call, sv, pp)
                    if not risky and benign_breaks_rules(turns, p["rules"]):
                        raise ValueError("benign_breaks_own_rules")
                    rows.append((risky, turns, ask, summary, st))
                bt, rt = rows[0][1], rows[1][1]
                shared = 0
                for x, y in zip(bt, rt):
                    if x["speaker"] != y["speaker"] or x["text"].strip() != y["text"].strip():
                        break
                    shared += 1
                if shared < 2:
                    raise ValueError("no_shared_opening")
                for j in range(shared):  # the guardian can't see the future: identical turns get identical (benign-side) labels
                    rt[j]["label"] = dict(bt[j]["label"])
                break
            except Exception as e:
                reason = str(e)[:80]
                rows = None
        if rows is None:
            with lock:
                stats["rej"][reason] = stats["rej"].get(reason, 0) + 1
                with rej.open("a") as f:
                    f.write(json.dumps({"pair": i, "seed": pair["id"], "reason": reason}) + "\n")
            return
        pid = f"cp_{sha('contrast', a.seed, i)}"
        split = "dev" if random.Random(pid).random() < 0.1 else "train"
        recs = []
        for risky, turns, ask, summary, st in rows:
            known = False if risky else bool(p["known_benign"] or p["allowed"])
            recs.append(dict(call_id=f"{pid}_{'r' if risky else 'b'}", pair_id=pid, split=split, generator=a.model,
                             seed_id=pair["id"] + ("_risky" if risky else "_benign"), scam_type=pair["id"],
                             family="scam" if risky else "benign", heldout_type=False, profile=p["profile"], age=p["age"],
                             protected_name=p["name"], relative=p["relative"], rules=p["rules"], flip_rule=None,
                             caller_known=known, ai_voice={"score": round(rng.uniform(0.78, 0.97) if p["ai"] else rng.uniform(0.03, 0.25), 2),
                                                           "source": "detector", "is_ai": p["ai"]},
                             rule_flip_of=None, flip_kind="contrast", benign_twin=not risky, ask_turn=ask if risky else None,
                             turns=turns, summary=summary, fix_stats=st, shared_prefix=shared, K=p["K"]))
        with lock:
            stats["ok"] += 1
            with out.open("a") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
            k = stats["ok"] + sum(stats["rej"].values())
            if k % 10 == 0:
                print(f"[{time.time()-t0:5.0f}s] {k}/{len(jobs)} ok={stats['ok']} rej={stats['rej']} | {USAGE}", flush=True)

    with cf.ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(one, jobs))
    print(f"DONE contrast: ok_pairs={stats['ok']} rej={stats['rej']} in {time.time()-t0:.0f}s | {USAGE}")


if __name__ == "__main__":
    main()

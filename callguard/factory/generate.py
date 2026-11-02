"""Emberz data factory.

  python3 -I callguard/factory/generate.py cards --model deepseek --split train --k 12
  python3 -I callguard/factory/generate.py calls --model deepseek --split train --n 1500 --workers 48
  python3 -I callguard/factory/generate.py calls --model nemotron --split test_heldout --n 250

One request per whole call: the generator writes the call from a hidden plan and labels every
turn. The validator normalizes labels, fixes what it safely can and drops the rest.
Output rows follow the `dataset_row` contract (PLAN.md section 4), with richer per-turn labels.
"""
import argparse, concurrent.futures as cf, difflib, hashlib, json, random, re, sys, threading, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import taxonomy as T
from fwclient import USAGE, chat, parse_json

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

SYSTEM = """You write SYNTHETIC phone-call transcripts used ONLY as training and evaluation data for a protective classifier ("guardian") that alerts parents and caretakers when a child, teen or older adult receives a scam, predatory, abusive or otherwise unsafe call. Realistic risky calls are required so the guardian learns to catch them; benign look-alike calls are equally important so it does not raise false alarms.
Hard content rules:
- Never include sexual content, sexual language, or descriptions of bodies or appearance in any call involving a minor. Predatory risk toward a child or teen is shown ONLY through non-explicit grooming TACTICS (flattery, gifts, asking age, asking if parents are home, asking for school/street, requests for secrecy, moving to another app, asking for a photo or video chat, asking to meet).
- Self-harm talk is non-graphic and never mentions methods.
- Profanity and insults are allowed where the scenario calls for it (abuse, bullying, harassment, friends trash-talking).
- Use fictional names and fictional companies; phone numbers are 555-01XX; never real account numbers.
Output ONLY one JSON object, no prose."""

LABEL_GUIDE = f"""Return JSON:
{{"turns": [{{"turn": 1, "speaker": "caller|protected", "text": "...",
   "label": {{"risk": "none|watch|alert|urgent", "category": "<one category>", "stage": "none|hook|trust|pressure|ask",
             "source": "none|caller|protected_person", "evidence": "<exact quote or empty>",
             "coach_line": "<=15 words or empty>", "caretaker_note": "<=20 words or empty>"}}}}],
 "ask_turn": <int or null>, "summary": "<one sentence a caretaker reads after the call>"EXTRA}}

Labeling rules (the label on turn t is what a guardian that has heard turns 1..t concludes NOW; it cannot see the future):
- risk: none = normal. watch = a mild flag worth logging (an unknown caller asks a personal question, an AI voice, an odd request). alert = a clear risk pattern the caretaker should know about (impersonation plus pressure, a grooming tactic, a request for codes or personal info, swearing at the person, a caretaker rule being broken). urgent = the ask is happening or info/money is being handed over, a threat, or self-harm talk.
- In risky calls, risk never goes down once raised.
- category: the most important category so far, exactly one of: {", ".join(T.CATEGORIES)}. Use "none" when risk is none. Use "custom_rule" only when a caretaker rule is the only reason the turn is risky.
- stage: hook|trust|pressure|ask for scams and grooming; "none" otherwise.
- source: who created the risk: "caller", or "protected_person" when the protected person gives info away or talks about self-harm; "none" when risk is none.
- evidence: words copied character-for-character from this turn or an earlier turn that justify the label; "" when risk is none.
- coach_line: when risk is alert or urgent, a whisper to the protected person, 15 words max (e.g. "Banks never ask for codes. Say you'll call the number on your card."); otherwise "".
- caretaker_note: when risk is alert or urgent, 20 words max for the caretaker; otherwise "".
- Follow the caretaker's rules: a rule can make otherwise-normal talk risky, and an allowed contact calling from their known number is trusted.
- An AI voice alone is at most "watch". An AI voice + unknown caller + an ask for money/codes/info escalates.
- ask_turn: the turn number where the caller makes the key ask (money, codes, personal info, meeting, photo), or null.
Style: real spoken phone talk: short turns, fillers (um, uh), interruptions, mishearing, people talking past each other. The protected person reacts in character."""


def sha(*parts):
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:10]


# ------------------------------------------------------------------ cards
CARD_PROMPT = """Scenario type for a synthetic phone-call dataset:
- Kind: {family}
- Caller: {caller}
- What happens: {story}
- Key ask: {ask}
- Protected person can be: {profiles}

Write {k} DISTINCT variants of this scenario. Vary the pretext details, the caller's persona and backstory, the specific amounts / fictional company names / places, the order of tactics, and add one realistic twist each (e.g. the caller knows a real-seeming detail, a second person joins, the protected person pushes back). Keep the same kind of risk (or the same harmlessness for benign calls). {extra}
Return JSON: {{"variants": [{{"title": "...", "pretext": "...", "caller_persona": "...", "twist": "...", "specifics": "..."}}]}}"""


def make_cards(model, split, k, workers):
    out = DATA / f"cards_{split}.jsonl"
    done = {}
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done[r["seed_id"]] = r
    todo = [s for s in T.SEEDS if s["id"] not in done and (split == "test_heldout" or not s["heldout"])]
    lock = threading.Lock()

    def one(seed):
        extra = ("Grooming must stay at the level of tactics only, nothing sexual." if seed["family"] == "grooming" else
                 "Each variant must remain clearly innocent once you hear the whole call." if seed["family"] == "benign" else "")
        msg = CARD_PROMPT.format(family=seed["family"], caller=seed["caller"], story=seed["story"],
                                 ask=seed["ask"] or "none", profiles=", ".join(seed["profiles"]), k=k, extra=extra)
        for attempt in range(3):
            try:
                txt, _, _ = chat(model, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": msg}],
                                 max_tokens=3500, temperature=1.0)
                v = parse_json(txt).get("variants", [])
                v = [x for x in v if isinstance(x, dict) and x.get("pretext")]
                if v:
                    row = {"seed_id": seed["id"], "model": model, "variants": v}
                    with lock, out.open("a") as f:
                        f.write(json.dumps(row) + "\n")
                    return seed["id"], len(v)
            except Exception as e:
                err = str(e)[:120]
        return seed["id"], f"FAILED {err if 'err' in dir() else ''}"

    with cf.ThreadPoolExecutor(workers) as ex:
        for sid, n in ex.map(one, todo):
            print(f"cards {sid}: {n}", flush=True)
    print(USAGE)


def load_cards(split):
    p = DATA / f"cards_{split}.jsonl"
    cards = {}
    if p.exists():
        for line in p.read_text().splitlines():
            r = json.loads(line)
            cards.setdefault(r["seed_id"], []).extend(r["variants"])
    return cards


# ------------------------------------------------------------------ plans
def sample_plan(seed, card, rng, split):
    profile = rng.choice(seed["profiles"])
    lo, hi = T.PROFILES[profile]["ages"]
    age = rng.randint(lo, hi)
    name = rng.choice(T.ELDER_NAMES if profile == "elder" else T.FIRST_NAMES[:34])
    relative = rng.choice(T.RELATIVES[profile])
    rules = rng.sample(T.RULES[profile], k=min(len(T.RULES[profile]), rng.randint(2, 4)))
    risky = seed["family"] != "benign"
    ai = rng.random() < seed["ai"]
    known = rng.random() < seed["known"]
    if seed["flip"] == "allowed_contact":
        known = False
    caller_style = rng.choice(T.CALLER_STYLES_AI if ai else T.CALLER_STYLES)
    style = rng.choice(T.PROTECTED_STYLES[profile])
    length_key = rng.choices(list(T.LENGTHS), weights=[3, 5, 2])[0]
    n_turns = rng.randint(*T.LENGTHS[length_key])
    ask_turn = None
    if risky and seed["ask"]:
        tk = rng.choices(list(T.ASK_TIMING), weights=[2, 5, 3])[0]
        a, b = T.ASK_TIMING[tk]
        ask_turn = min(rng.randint(a, b), n_turns - 2)
    outcome = rng.choice(T.OUTCOMES_RISKY) if risky else None
    subtle = risky and seed["family"] == "scam" and rng.random() < 0.15
    hints = []
    if risky and rng.random() < 0.5:
        pool = [p for c in seed["cats"] for p in T.LEXICON.get(c, [])]
        hints = rng.sample(pool, k=min(3, len(pool))) if pool else []
    if not risky:
        hints = rng.sample(T.HARD_NEGATIVE_PHRASES, k=2)
    flip_rule = T.FLIP_RULES.get(seed["flip"]) if seed["flip"] in T.FLIP_RULES else None
    if flip_rule == T.FLIP_RULES["swearing"]:
        rules = [r for r in rules if "swearing" not in r.lower()]
    return dict(seed_id=seed["id"], family=seed["family"], profile=profile, age=age, name=name, relative=relative,
                rules=rules, ai=ai, caller_known=known, caller_style=caller_style, style=style, n_turns=n_turns,
                ask_turn=ask_turn, outcome=outcome, subtle=subtle, hints=hints, flip=seed["flip"],
                flip_rule=flip_rule, card=card, split=split)


def plan_text(seed, p):
    lab = T.PROFILES[p["profile"]]["label"]
    c = p["card"] or {}
    lines = [
        "PLAN (hidden from the speakers):",
        f"- Protected person: {p['name']}, a {p['age']}-year-old {lab}; personality: {p['style']}.",
        f"- Caller: {seed['caller']}; speaking style: {p['caller_style']}.",
        f"- The caller's number is {'in' if p['caller_known'] else 'NOT in'} {p['name']}'s contacts.",
        f"- Scenario: {seed['story']}.",
    ]
    if c:
        lines.append(f"- This variant: {c.get('title','')}. {c.get('pretext','')} Caller persona: {c.get('caller_persona','')}. "
                     f"Twist: {c.get('twist','')}. Specifics: {c.get('specifics','')}")
    if p["ask_turn"]:
        lines.append(f"- The key ask happens at turn {p['ask_turn']}: {seed['ask']}. Outcome: {p['outcome']}.")
    elif seed["family"] != "benign" and p["outcome"]:
        lines.append(f"- How it goes: {p['outcome'] if seed['family'] in ('scam','disclosure','grooming','ai') else 'the situation plays out naturally'}.")
    if seed["family"] != "benign":
        lines.append(f"- Risk categories, in the order they come up: {', '.join(seed['cats'])}.")
    if p["subtle"]:
        lines.append("- The caller avoids obvious scam words; the manipulation is polite, slow and subtle.")
    if p["hints"]:
        lines.append(f"- Work in, naturally, phrases like: {'; '.join(repr(h) for h in p['hints'])}.")
    if seed["family"] == "benign":
        lines.append("- This is a BENIGN call that contains words that could sound alarming out of context; it is innocent. "
                     "Labels stay none (watch at most, briefly, before it is clearly innocent).")
    if p["flip"] == "allowed_contact":
        lines.append(f"- The caller may really be family, but calls from an unknown number, so the guardian cannot verify who it is: label it as risky.")
    lines.append(f"- Caretaker: {p['relative']}. Caretaker's rules: " + "; ".join(p["rules"]) + ".")
    if p["flip_rule"]:
        lines.append(f"- EXTRA (do not apply to the labels): list in \"rule_turns\" the turn numbers that would break this rule "
                     f"if the caretaker had it: \"{p['flip_rule']}\", and give \"rule_note\" (20 words max) for the caretaker in that case.")
        if p["flip"] == "swearing":
            lines.append("- The friends swear casually now and then (e.g. 'shit', 'damn', 'what the hell').")
    lines.append(f"- Length: about {p['n_turns']} turns. Turn 1 is the caller speaking first after the protected person answers.")
    return "\n".join(lines)


# ------------------------------------------------------------------ validation
ALL_CATS = set(T.CATEGORIES)
MINOR_RE = re.compile(T.MINOR_BLOCKLIST, re.I)


def norm(s):
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def canon_cat(c, seed):
    c = (c or "").strip()
    if c in ALL_CATS:
        return c, False
    if c in T.CATEGORY_ALIASES:
        return T.CATEGORY_ALIASES[c], True
    for full in T.CATEGORIES:  # suffix match e.g. "impersonation_bank_gov"
        if full.endswith("." + c) or full.split(".")[-1] == c.split(".")[-1]:
            return full, True
    return None, True


def words(s, n):
    w = (s or "").split()
    return " ".join(w[:n])


def validate(raw, seed, p):
    """Returns (turns, ask_turn, summary, extra, stats) or raises ValueError(reason)."""
    stats = {"cat_fixed": 0, "evidence_fixed": 0, "monotonic_fixed": 0}
    turns = raw.get("turns")
    if not isinstance(turns, list) or len(turns) < 8:
        raise ValueError("too_few_turns")
    out = []
    texts = []
    for i, t in enumerate(turns, 1):
        if not isinstance(t, dict) or not str(t.get("text", "")).strip():
            raise ValueError("bad_turn")
        spk = str(t.get("speaker", "")).lower()
        spk = "caller" if spk.startswith("call") else "protected" if spk.startswith("prot") else None
        if spk is None:
            raise ValueError("bad_speaker")
        text = str(t["text"]).strip()
        texts.append(text)
        lab = t.get("label") or {}
        risk = str(lab.get("risk", "none")).lower()
        if risk not in T.RISKS:
            raise ValueError("bad_risk")
        cat, fixed = canon_cat(lab.get("category"), seed)
        if cat is None:
            cat = "none" if risk == "none" else (seed["cats"][0] if seed["cats"][0] != "none" else None)
            if cat is None:
                raise ValueError("bad_category")
        stats["cat_fixed"] += int(fixed)
        if risk == "none":
            cat = "none"
        elif cat == "none" and seed["family"] != "benign":
            cat = seed["cats"][0] if seed["cats"][0] != "none" else "custom_rule"
            stats["cat_fixed"] += 1
        stage = str(lab.get("stage", "none")).lower()
        stage = stage if stage in T.STAGES else "none"
        src = str(lab.get("source", "none")).lower()
        src = "protected_person" if src.startswith("prot") else "caller" if src.startswith("call") else "none"
        if risk != "none" and src == "none":
            src = "caller"
        out.append(dict(turn=i, speaker=spk, text=text,
                        label=dict(risk=risk, category=cat, stage=stage, source=src,
                                   evidence=str(lab.get("evidence") or ""),
                                   coach_line=words(lab.get("coach_line"), 15), caretaker_note=words(lab.get("caretaker_note"), 20))))
    risky = seed["family"] != "benign"
    # cumulative risk in risky calls
    if risky:
        best = 0
        for t in out:
            r = T.RISKS.index(t["label"]["risk"])
            if r < best:
                t["label"]["risk"] = T.RISKS[best]
                if t["label"]["category"] == "none":
                    t["label"]["category"] = prev_cat
                stats["monotonic_fixed"] += 1
            best = max(best, T.RISKS.index(t["label"]["risk"]))
            prev_cat = t["label"]["category"]
    # evidence must be a real quote from turns <= t
    last_rise_text = ""
    level = 0
    for idx, t in enumerate(out):
        L = t["label"]
        r = T.RISKS.index(L["risk"])
        if r > level:
            last_rise_text = t["text"]
            level = r
        if L["risk"] == "none":
            L["evidence"] = ""
            L["coach_line"] = L["coach_line"] if False else ""
            L["caretaker_note"] = ""
            continue
        ev = L["evidence"].strip().strip('"').strip()
        hay = [norm(x) for x in texts[: idx + 1]]
        if ev and any(norm(ev) in h for h in hay):
            # recover exact original casing
            for x in texts[: idx + 1]:
                j = norm(x).find(norm(ev))
                if j >= 0 and norm(ev) in norm(x):
                    L["evidence"] = ev if ev in x else x
                    break
        else:
            best = max(texts[: idx + 1], key=lambda x: difflib.SequenceMatcher(None, norm(x), norm(ev)).ratio()) if ev else None
            L["evidence"] = best if best and difflib.SequenceMatcher(None, norm(best), norm(ev)).ratio() > 0.6 else (last_rise_text or t["text"])
            stats["evidence_fixed"] += 1
        if T.RISKS.index(L["risk"]) >= 2 and not L["coach_line"]:
            L["coach_line"] = ""
    max_risk = max(T.RISKS.index(t["label"]["risk"]) for t in out)
    if risky and max_risk < 2:
        raise ValueError("risky_never_alerted")
    if not risky and max_risk >= 2:
        raise ValueError("benign_alerted")
    if p["profile"] in ("kid", "teen") and MINOR_RE.search(" ".join(texts)):
        raise ValueError("minor_blocklist")
    ask = raw.get("ask_turn")
    ask = ask if isinstance(ask, int) and 1 <= ask <= len(out) else None
    if risky and seed["ask"] and ask is None:
        ask = next((t["turn"] for t in out if t["label"]["stage"] == "ask" and t["speaker"] == "caller"), None)
    if not risky:
        ask = None
    extra = {}
    if p["flip_rule"]:
        rt = raw.get("rule_turns") or []
        extra["rule_turns"] = sorted({int(x) for x in rt if str(x).isdigit() and 1 <= int(x) <= len(out)})
        extra["rule_note"] = words(raw.get("rule_note"), 20)
    return out, ask, str(raw.get("summary") or "")[:300], extra, stats


# ------------------------------------------------------------------ flips (deterministic twins)
ALLOWED_NAME = {"relative_asks_door_code": "Grandpa", "relative_asks_address": "Aunt", "son_asks_bank_login": "her son"}


def allowed_twin(row):
    """Same call, but the caller is an allowed contact calling from their known number -> benign."""
    tw = json.loads(json.dumps(row))
    tw["call_id"] = row["call_id"] + "_ac"
    tw["rule_flip_of"] = row["call_id"]
    tw["flip_kind"] = "allowed_contact"
    tw["caller_known"] = True
    tw["rules"] = row["rules"] + [f"Allowed: {ALLOWED_NAME.get(row['seed_id'], row['relative'])} (saved number +1-555-01{random.randint(10, 99)})"]
    tw["benign_twin"] = True
    tw["ask_turn"] = None
    for t in tw["turns"]:
        t["label"] = dict(risk="none", category="none", stage="none", source="none", evidence="", coach_line="", caretaker_note="")
    tw["summary"] = f"{ALLOWED_NAME.get(row['seed_id'], row['relative'])} called from a saved, allowed number; nothing concerning."
    return tw


def rule_twin(row, extra):
    """Same benign call, plus the flip rule -> turns from the first rule-breaking turn on become custom_rule alerts."""
    rt = extra.get("rule_turns") or []
    if not rt:
        return None
    tw = json.loads(json.dumps(row))
    tw["call_id"] = row["call_id"] + "_rf"
    tw["rule_flip_of"] = row["call_id"]
    tw["flip_kind"] = "rule"
    tw["rules"] = row["rules"] + [row["flip_rule"]]
    tw["benign_twin"] = False
    first = rt[0]
    if row.get("flip_rule") == T.FLIP_RULES["swearing"]:
        sw = re.compile(r"\b(" + "|".join(["shit", "damn", "hell", "fuck", "crap", "ass", "bitch", "wtf"]) + r")\b", re.I)
        hit = next((t["turn"] for t in tw["turns"] if sw.search(t["text"])), None)
        if hit is None:
            return None
        first = hit
    quote = tw["turns"][first - 1]["text"]
    note = extra.get("rule_note") or "A caretaker rule was broken on this call."
    for t in tw["turns"]:
        if t["turn"] >= first:
            t["label"] = dict(risk="alert", category="custom_rule", stage="none", source="caller" if tw["turns"][first - 1]["speaker"] == "caller" else "protected_person",
                              evidence=quote, coach_line="", caretaker_note=note)
    tw["summary"] = note
    return tw


# ------------------------------------------------------------------ calls
def ai_score(rng, is_ai):
    s = rng.gauss(0.88, 0.07) if is_ai else rng.gauss(0.12, 0.08)
    if rng.random() < 0.04:
        s = 1 - s  # detector error
    return round(min(0.99, max(0.01, s)), 2)


def make_calls(model, split, n, workers, seed_val, only=None, key_env="FIREWORKS_API_KEY", tag=""):
    out = DATA / f"calls_{split}{tag}.jsonl"
    rej = DATA / f"rejects_{split}{tag}.jsonl"
    cards = load_cards(split)
    rng = random.Random(seed_val)
    seeds = [s for s in T.SEEDS if (split == "test_heldout" or not s["heldout"]) and (not only or s["id"] in only)]
    weights = [s["w"] for s in seeds]
    done = 0
    if out.exists():
        done = sum(1 for line in out.read_text().splitlines() if '"rule_flip_of": null' in line)
    jobs = []
    for i in range(done, n):
        s = rng.choices(seeds, weights)[0]
        card = rng.choice(cards[s["id"]]) if cards.get(s["id"]) else None
        p = sample_plan(s, card, random.Random(rng.random()), split)
        p["call_id"] = f"{'s' if split != 'test_heldout' else 't'}{tag}_{sha(split, seed_val, i)}"
        p["ai_score"] = ai_score(rng, p["ai"])
        jobs.append((s, p))
    lock = threading.Lock()
    counts = {"ok": 0, "rej": 0}
    reasons = {}
    t0 = time.time()

    def one(job):
        s, p = job
        extra_fmt = ', "rule_turns": [<int>], "rule_note": "..."' if p["flip_rule"] else ""
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": plan_text(s, p) + "\n\n" + LABEL_GUIDE.replace("EXTRA", extra_fmt)}]
        reason = None
        for attempt in range(2):
            try:
                txt, usage, finish = chat(model, msgs, key_env=key_env, max_tokens=6000, temperature=0.9)
                raw = parse_json(txt)
                turns, ask, summary, extra, stats = validate(raw, s, p)
                break
            except Exception as e:
                reason = str(e)[:80]
                turns = None
        if turns is None:
            with lock:
                counts["rej"] += 1
                reasons[reason] = reasons.get(reason, 0) + 1
                with rej.open("a") as f:
                    f.write(json.dumps({"call_id": p["call_id"], "seed_id": s["id"], "reason": reason}) + "\n")
            return
        row = dict(call_id=p["call_id"], split=split if split == "test_heldout" else ("dev" if random.random() < 0.1 else "train"),
                   generator=model, seed_id=s["id"], scam_type=s["id"], family=s["family"], heldout_type=s["heldout"],
                   profile=p["profile"], age=p["age"], protected_name=p["name"], relative=p["relative"],
                   rules=p["rules"], flip_rule=p["flip_rule"], caller_known=p["caller_known"],
                   ai_voice={"score": p["ai_score"], "source": "detector", "is_ai": p["ai"]},
                   rule_flip_of=None, flip_kind=None, benign_twin=s["family"] == "benign",
                   ask_turn=ask, turns=turns, summary=summary, fix_stats=stats,
                   plan={k: p[k] for k in ("caller_style", "style", "n_turns", "outcome", "subtle", "hints", "card")})
        rows = [row]
        if s["flip"] == "allowed_contact":
            rows.append(allowed_twin(row))
        elif p["flip_rule"]:
            tw = rule_twin(row, extra)
            if tw:
                rows.append(tw)
        with lock:
            counts["ok"] += 1
            with out.open("a") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            k = counts["ok"] + counts["rej"]
            if k % 25 == 0:
                el = time.time() - t0
                print(f"[{el:6.0f}s] {k}/{len(jobs)} ok={counts['ok']} rej={counts['rej']} {reasons} | {USAGE}", flush=True)

    with cf.ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, jobs))
    print(f"DONE {split}: ok={counts['ok']} rej={counts['rej']} reasons={reasons} in {time.time()-t0:.0f}s | {USAGE}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["cards", "calls"])
    ap.add_argument("--model", default="deepseek")
    ap.add_argument("--split", default="train", choices=["train", "test_heldout", "pilot"])
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--key-env", default="FIREWORKS_API_KEY")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.what == "cards":
        make_cards(a.model, a.split, a.k, a.workers)
    else:
        make_calls(a.model, a.split, a.n, a.workers, (a.seed if a.split != "test_heldout" else a.seed + 1000) + (hash(a.tag) % 997 if a.tag else 0), a.only, a.key_env, a.tag)

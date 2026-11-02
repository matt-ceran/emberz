"""Held-out call replay for the caretaker console.

Loads the held-out test calls and one guardian predictions file, picks demo
presets from the data, and turns one call into the same `console_event` stream
the line server sends (contracts/models.py ConsoleEvent on origin/numan/line-voice-gcp).

Nothing here invents a guardian verdict: `risk`, `alert` and `coach` events come
only from rows in the predictions file. The eval stops scoring a call at its
first alert, so later turns replay without verdicts.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

CALLGUARD_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CALLS = "data/calls_test_heldout.jsonl"
DEFAULT_PREDS = "guardian-v1"

RISKS = ["none", "watch", "alert", "urgent"]
GUARDIAN_FIELDS = (
    "risk", "category", "severity", "source", "stage", "evidence", "action", "coach_line", "caretaker_note",
)
TURN_MS = 1500          # one turn every 1.5 s of call time at speed 1.0
SPEECH_MS = 1200        # each turn "speaks" for 1.2 s, then its final transcript lands
VERDICT_AFTER_MS = 150  # replayed verdict lands just after its turn (real latency is in latency_ms)
VOICE_AT_MS = 2000      # AI-voice signal ~2 s in, as on the live line (PLAN.md section 8)
LATENCY_NOTE = os.environ.get("CONSOLE_LATENCY_NOTE") or "replayed from evaluation (Training API sampler latency)"

CATEGORY_WORDS = {
    "scam.impersonation_bank_gov": "Bank or government impostor",
    "scam.tech_support": "Tech support scam",
    "scam.family_emergency": "Fake family emergency",
    "scam.prize": "Prize or lottery scam",
    "scam.romance_investment": "Romance or investment pitch",
    "disclosure.location_school": "Sharing address or school",
    "disclosure.credentials_otp": "Asked for a code or password",
    "disclosure.financial_id": "Sharing card or ID numbers",
    "disclosure.home_alone": "Asked who is home",
    "grooming.secrecy": "Asked to keep secrets",
    "grooming.isolation": "Pulling them away from family",
    "grooming.gifts_flattery": "Gifts or flattery from a stranger",
    "grooming.photo_request": "Asked for photos or video",
    "grooming.meeting_request": "Asked to meet in person",
    "vulgar": "Rude or vulgar language",
    "threat_extortion": "Threats or blackmail",
    "self_harm": "Talk of self-harm",
    "custom_rule": "Matches one of your rules",
    "none": "Nothing concerning",
}


def rank(risk: Optional[str]) -> int:
    return RISKS.index(risk) if risk in RISKS else 0


def category_words(cat: Optional[str]) -> str:
    if not cat:
        return CATEGORY_WORDS["none"]
    return CATEGORY_WORDS.get(cat, cat.split(".")[-1].replace("_", " ").capitalize())


# ------------------------------------------------------------------ loading
def _resolve(spec: str) -> Path:
    p = Path(spec).expanduser()
    if p.is_absolute():
        return p
    for base in (Path.cwd(), CALLGUARD_DIR):
        if (base / p).exists():
            return (base / p).resolve()
    return (CALLGUARD_DIR / p).resolve()


def resolve_preds(spec: Optional[str]) -> Path:
    """CONSOLE_PREDS may be a model name (guardian-v2) or a path to a preds jsonl."""
    spec = (spec or DEFAULT_PREDS).strip()
    if "/" not in spec and not spec.endswith(".jsonl"):
        spec = f"results/preds_{spec}__heldout.jsonl"
    return _resolve(spec)


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a preds file still being written can end in a partial line
    return rows


@dataclass
class Dataset:
    calls: dict[str, dict]
    preds: dict[str, dict]
    calls_path: Path
    preds_path: Path
    preds_name: str
    presets: list[dict] = field(default_factory=list)

    def preds_by_turn(self, call_id: str) -> dict[int, dict]:
        res = self.preds.get(call_id) or {}
        out = {}
        for i, p in enumerate(res.get("preds") or []):
            out[int(p.get("_turn") or i + 1)] = p
        return out


def load(calls_spec: Optional[str] = None, preds_spec: Optional[str] = None) -> Dataset:
    calls_path = _resolve(calls_spec or os.environ.get("CONSOLE_CALLS") or DEFAULT_CALLS)
    preds_path = resolve_preds(preds_spec or os.environ.get("CONSOLE_PREDS"))
    calls = {r["call_id"]: r for r in _read_jsonl(calls_path)}
    preds = {}
    if preds_path.exists():
        for r in _read_jsonl(preds_path):
            if r.get("call_id") in calls:
                preds[r["call_id"]] = r
    name = preds_path.name.removeprefix("preds_").split("__")[0]
    ds = Dataset(calls=calls, preds=preds, calls_path=calls_path, preds_path=preds_path, preds_name=name)
    ds.presets = select_presets(ds)
    return ds


# ------------------------------------------------------------------ preset selection (by data)
def gold_risky(row: dict) -> bool:
    """Same definition as eval/run_eval.py."""
    return not row.get("benign_twin") and any(rank(t["label"]["risk"]) >= 2 for t in row["turns"])


_DIGITS = re.compile(
    r"\d|\b(zero|one|two|three|four|five|six|seven|eight|nine|oh)\b[-\s,.]+(zero|one|two|three|four|five|six|seven|eight|nine|oh)\b",
    re.I)


def discloses_after_ask(row: dict) -> bool:
    """True if the protected person says digits / a run of number words at or after the ask (they handed it over)."""
    ask = row.get("ask_turn")
    if not ask:
        return False
    return any(_DIGITS.search(t["text"]) for t in row["turns"] if t["speaker"] == "protected" and t["turn"] >= ask)


def _touches_self_harm(row: dict, res: Optional[dict]) -> bool:
    if row.get("family") == "self_harm" or "self_harm" in str(row.get("scam_type")) or "despair" in str(row.get("scam_type")):
        return True
    if any(t["label"].get("category") == "self_harm" for t in row["turns"]):
        return True
    return any(p.get("category") == "self_harm" for p in (res or {}).get("preds") or [])


def _info(ds: Dataset, cid: str) -> Optional[dict]:
    row, res = ds.calls.get(cid), ds.preds.get(cid)
    if not row or not res or _touches_self_harm(row, res):
        return None
    preds = res.get("preds") or []
    if not preds or not all(p.get("_ok", True) for p in preds):
        return None
    fa = res.get("first_alert")
    fa_pred = next((p for p in preds if rank(p.get("risk")) >= 2), None)
    n = len(row["turns"])
    ask = row.get("ask_turn")
    return {
        "row": row, "res": res, "fa": fa, "fa_pred": fa_pred, "n": n, "ask": ask,
        "lead": (ask - fa) if (ask and fa) else None,
        "ai": float((row.get("ai_voice") or {}).get("score") or 0.0),
        "full": len(preds) >= n,  # scored every turn (only possible when it never alerted)
        "complete_alert": bool(fa_pred and fa_pred.get("coach_line") and fa_pred.get("caretaker_note")
                               and fa_pred.get("evidence")),
    }


def _pick(cands: list[tuple[tuple, str]]) -> Optional[str]:
    if not cands:
        return None
    return max(cands, key=lambda c: (c[0], c[1]))[1]


def select_presets(ds: Dataset) -> list[dict]:
    infos = {cid: i for cid in ds.calls if (i := _info(ds, cid))}
    presets: list[dict] = []

    # 1. Elder bank / government / one-time-code scam the guardian caught before the ask.
    cands = []
    for cid, i in infos.items():
        r = i["row"]
        if (r["profile"] == "elder" and gold_risky(r) and not r.get("rule_flip_of") and i["lead"] and i["lead"] > 0
                and i["complete_alert"]
                and i["fa_pred"].get("category") in ("scam.impersonation_bank_gov", "disclosure.credentials_otp")):
            ask_cat = r["turns"][i["ask"] - 1]["label"]["category"]
            cands.append(((not discloses_after_ask(r), i["ai"] >= 0.7, ask_cat == "disclosure.credentials_otp",
                           min(i["lead"], 4), -i["n"]), cid))
    if cid := _pick(cands):
        i = infos[cid]
        presets.append(_preset(ds, "elder_scam", "AI bank caller" if i["ai"] >= 0.7 else "Bank or government caller", cid,
                               f"Elder {i['row']['scam_type']} call; guardian alerted at turn {i['fa']}, "
                               f"{i['lead']} turns before the ask (turn {i['ask']}). Ranked by: the protected person does not read out "
                               f"numbers after the ask, AI voice >= 0.7, ask is a one-time code, lead (capped at 4), "
                               f"shorter call. {len(cands)} candidates."))

    # 2. Kid grooming tactics caught early.
    cands = []
    for cid, i in infos.items():
        r = i["row"]
        if (r["profile"] in ("kid", "teen") and r.get("family") == "grooming" and i["lead"] and i["lead"] > 0
                and i["complete_alert"] and str(i["fa_pred"].get("category")).startswith("grooming.")):
            cands.append(((r["profile"] == "kid", min(i["lead"], 3), -i["n"]), cid))
    if cid := _pick(cands):
        i = infos[cid]
        presets.append(_preset(ds, "kid_grooming", "Stranger calls a kid", cid,
                               f"{i['row']['profile'].capitalize()} grooming call ({i['row']['scam_type']}); alerted at turn "
                               f"{i['fa']} as {i['fa_pred']['category']}, {i['lead']} turns before the ask (turn {i['ask']}). "
                               f"Ranked by: kid over teen, lead (capped at 3), shorter call. {len(cands)} candidates."))

    # 3. Benign twin: a real call that sounds like a scam pretext, from an unknown number, and stays quiet.
    pref = ["real_bank_alert", "friend_mom_playdate", "police_followup_real", "isp_callback_real",
            "doctor_office_dob", "delivery_driver_real"]
    cands = []
    for cid, i in infos.items():
        r = i["row"]
        if (r.get("benign_twin") and not r.get("rule_flip_of") and not gold_risky(r) and not i["fa"] and i["full"]
                and not r.get("caller_known") and i["ai"] < 0.5):
            p = len(pref) - pref.index(r["scam_type"]) if r["scam_type"] in pref else 0
            cands.append(((p, r["profile"] == "elder", -abs(i["n"] - 12)), cid))
    if cid := _pick(cands):
        i = infos[cid]
        presets.append(_preset(ds, "benign_twin", "Real caller, stays quiet", cid,
                               f"Benign twin ({i['row']['scam_type']}, unknown number); no alert across all {i['n']} turns. "
                               f"Preference: {', '.join(pref)}; then elder, then ~12 turns. {len(cands)} candidates. "
                               f"real_bank_alert / friend_mom_playdate twins are skipped when the guardian false-alarmed on them."))

    # 4. Rule flip / allowed contact: same words, different rules, guardian right on both sides.
    cands = []
    for cid, i in infos.items():
        r = i["row"]
        orig = infos.get(r.get("rule_flip_of") or "")
        if not orig:
            continue
        flip_ok = bool(i["fa"]) == gold_risky(r)
        orig_ok = bool(orig["fa"]) == gold_risky(orig["row"])
        if not (flip_ok and orig_ok and gold_risky(r) != gold_risky(orig["row"])):
            continue
        risky = i if gold_risky(r) else orig
        lead = risky["lead"] if risky["lead"] and risky["lead"] > 0 else 0
        cands.append(((r.get("flip_kind") == "allowed_contact", lead > 0, r["profile"] == "kid", -i["n"]), cid))
    if cid := _pick(cands):
        i = infos[cid]
        r = i["row"]
        orig = infos[r["rule_flip_of"]]
        kind = r.get("flip_kind")
        title = "Same words, saved contact" if kind == "allowed_contact" else "Same words, new rule"
        p = _preset(ds, "rule_flip", title, cid,
                    f"{kind} flip of {r['rule_flip_of']}: this version {'alerted at turn ' + str(i['fa']) if i['fa'] else 'stays quiet'}; "
                    f"the original {'alerted at turn ' + str(orig['fa']) if orig['fa'] else 'stays quiet'}"
                    f"{' (ask at turn ' + str(orig['ask']) + ')' if orig['ask'] else ''}. Guardian right on both. "
                    f"Ranked by: allowed contact, risky side caught before the ask, kid, shorter. {len(cands)} correct pairs.")
        p["pair"] = {
            "call_id": orig["row"]["call_id"],
            "title": "Unknown number" if kind == "allowed_contact" else "Without the rule",
            "first_alert": orig["fa"], "ask_turn": orig["ask"],
        }
        presets.append(p)

    # 5. An AI voice that is not a scam (pharmacy / dentist reminder bot) and stays quiet.
    cands = []
    for cid, i in infos.items():
        r = i["row"]
        if (r.get("family") == "benign" and not gold_risky(r) and not r.get("rule_flip_of") and not i["fa"] and i["full"]
                and i["ai"] >= 0.7):
            cands.append(((r["scam_type"] == "pharmacy_bot", r["profile"] == "elder", -abs(i["n"] - 10)), cid))
    if cid := _pick(cands):
        i = infos[cid]
        presets.append(_preset(ds, "ai_benign", "AI voice, not a scam", cid,
                               f"Benign {i['row']['scam_type']} with AI-voice score {i['ai']:.2f}; no alert across all "
                               f"{i['n']} turns. Ranked by: pharmacy bot, elder, ~10 turns. {len(cands)} candidates."))
    return presets


def _preset(ds: Dataset, pid: str, title: str, cid: str, why: str) -> dict:
    row, res = ds.calls[cid], ds.preds.get(cid) or {}
    return {
        "id": pid, "title": title, "call_id": cid, "why": why,
        "protected_name": row.get("protected_name"), "profile": row.get("profile"),
        "n_turns": len(row["turns"]), "first_alert": res.get("first_alert"), "ask_turn": row.get("ask_turn"),
        "ai_voice": (row.get("ai_voice") or {}).get("score"), "caller_known": bool(row.get("caller_known")),
        "pair": None,
    }


# ------------------------------------------------------------------ call -> console events
def allowed_contact(row: dict) -> tuple[Optional[str], Optional[str]]:
    """'Allowed: Aunt (saved number +1-555-0150)' -> ('Aunt', '+1-555-0150')."""
    for rule in row.get("rules") or []:
        if rule.lower().startswith("allowed:"):
            body = rule.split(":", 1)[1].strip()
            name, number = body, None
            if "(" in body:
                name = body.split("(", 1)[0].strip()
                inside = body.split("(", 1)[1].rstrip(")")
                number = inside.replace("saved number", "").strip() or None
            return name, number
    return None, None


def _event(etype: str, call_id: str, at_ms: int, data: dict) -> dict:
    return {"type": etype, "call_id": call_id, "at_ms": int(at_ms), "data": data}


def guardian_response(pred: dict) -> dict:
    """Keep only GuardianResponse fields (drop the eval's _ms/_ok/_turn bookkeeping)."""
    return {k: pred[k] for k in GUARDIAN_FIELDS if k in pred}


def build_events(ds: Dataset, call_id: str, as_call_id: Optional[str] = None) -> list[dict]:
    """The full console_event stream for one held-out call, ordered by at_ms (call time at speed 1.0).

    `as_call_id` renames the stream (a second replay of the same call gets a fresh id, like Numan's fixture player)."""
    src_id = call_id
    row = ds.calls[src_id]
    by_turn = ds.preds_by_turn(src_id)
    call_id = as_call_id or src_id
    turns = row["turns"]
    n = len(turns)
    ai = row.get("ai_voice") or {}
    caller_name, saved_number = allowed_contact(row) if row.get("caller_known") else (None, None)

    events = [_event("call_started", call_id, 0, {
        # Numan's README fields first; the rest are extras the page uses (data allows extra keys).
        "caller_number": saved_number,
        "guarded_number": None,
        "profile": row.get("profile"),
        "caller_known": bool(row.get("caller_known")),
        "caller_name": caller_name,
        "protected_name": row.get("protected_name"),
        "relative": row.get("relative"),
        "rules": row.get("rules") or [],
        "ai_voice": {"score": ai.get("score"), "source": ai.get("source")} if ai else None,
        "replay": {
            "call_id": src_id, "source": ds.calls_path.name, "preds": ds.preds_name, "preds_file": ds.preds_path.name,
            "n_turns": n, "verdict_turns": sorted(by_turn), "latency_note": LATENCY_NOTE,
            "note": "Verdicts stop at the guardian's first alert, as in the evaluation.",
        },
    })]

    if ai.get("score") is not None:
        events.append(_event("voice", call_id, VOICE_AT_MS, {
            "call_id": call_id, "at_ms": VOICE_AT_MS, "ai_score": ai["score"], "model": ai.get("source") or "detector",
        }))

    end_ms = 0
    for idx, t in enumerate(turns):
        start = 300 + idx * TURN_MS
        end_ms = start + SPEECH_MS
        events.append(_event("turn", call_id, end_ms, {
            "call_id": call_id, "turn": t["turn"], "speaker": t["speaker"], "text": t["text"],
            "t_start_ms": start, "t_end_ms": end_ms, "gap_ms": None if idx == 0 else TURN_MS - SPEECH_MS,
        }))
        pred = by_turn.get(t["turn"])
        if not pred:
            continue
        at = end_ms + VERDICT_AFTER_MS
        resp = guardian_response(pred)
        events.append(_event("risk", call_id, at, {
            # display_risk: the eval has no rule layer, so it is the model's own risk
            "turn": t["turn"], "response": resp, "latency_ms": pred.get("_ms"), "display_risk": resp.get("risk", "none"),
            "latency_note": LATENCY_NOTE, "ok": pred.get("_ok", True),
        }))
        if rank(pred.get("risk")) >= 2:
            events.append(_event("alert", call_id, at, {
                "level": pred["risk"], "category": pred.get("category"), "caretaker_note": pred.get("caretaker_note", ""),
                "turn": t["turn"], "evidence": pred.get("evidence") or [], "action": pred.get("action"),
            }))
        if pred.get("coach_line"):
            events.append(_event("coach", call_id, at, {"text": pred["coach_line"], "turn": t["turn"], "sent": True}))

    # Same order as the line: call_ended, then the summary ~1.4 s later.
    events.append(_event("call_ended", call_id, end_ms + 1000, {"duration_ms": end_ms + 1000, "reason": "completed"}))
    events.append(_event("summary", call_id, end_ms + 2400, summary_data(ds, src_id)))
    events.sort(key=lambda e: e["at_ms"])  # stable: same-time events keep their order
    return events


def summary_data(ds: Dataset, call_id: str) -> dict:
    row = ds.calls[call_id]
    res = ds.preds.get(call_id) or {}
    preds = res.get("preds") or []
    n = len(row["turns"])
    fa = res.get("first_alert")
    ask = row.get("ask_turn")
    fa_pred = next((p for p in preds if rank(p.get("risk")) >= 2), None)
    top = max((rank(p.get("risk")) for p in preds), default=0)
    if not preds:
        text = f"No guardian verdicts for this call in {ds.preds_path.name}. {n} turns."
    elif fa:
        text = f"Alerted at turn {fa}: {category_words(fa_pred.get('category'))}."
        if ask and fa < ask:
            text += f" That was {ask - fa} turn{'s' if ask - fa != 1 else ''} before the ask (turn {ask})."
        elif ask and fa == ask:
            text += f" Same turn as the ask (turn {ask})."
        elif ask:
            text += f" The ask came earlier, at turn {ask}."
        text += f" {n} turns in all."
    else:
        text = (f"The guardian stayed calm for all {n} turns." if top == 0 else
                f"The guardian marked {sum(1 for p in preds if rank(p.get('risk')) == 1)} of {n} turns as Watching "
                f"and never escalated.")
    risky = gold_risky(row)
    return {
        "text": text,
        "first_alert": fa, "ask_turn": ask,
        "turns_before_ask": (ask - fa) if (ask and fa and fa < ask) else None,
        "n_turns": n, "max_risk": RISKS[top],
        "category": fa_pred.get("category") if fa_pred else None,
        "label_risky": risky, "verdict_matches_label": bool(fa) == risky if preds else None,
        "scenario": row.get("summary"),
        "replay": True,
    }

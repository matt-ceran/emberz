"""Load, check and print the role-play cards (callers/roleplay/cards.json).

  python -m callers.roleplay.cards check                       # parse + sanity checks
  python -m callers.roleplay.cards print --out <dir>           # one caller card + one protected card per id
  python -m callers.roleplay.cards print --out <dir> --ids rp01,rp22

The caller and the protected person get separate cards: the protected person's card never shows
the caller's goal or ask, so they react naturally. No card says what the guardian should output.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict

from contracts.models import Profile

CARDS_PATH = Path(__file__).resolve().parent / "cards.json"
Mix = Literal["elder_scam", "kid_stranger", "rule_flip", "benign_hard_negative"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CallerSide(_Strict):
    role: str
    goal: str
    ask: Optional[str]
    beats: list[str]


class ProtectedSide(_Strict):
    persona: str
    behaviour: str


class Card(_Strict):
    id: str
    mix: Mix
    profile: Profile
    pair: Optional[str]
    minutes: int
    caller_line: Literal["unknown", "allowed_contact"]
    caller_contact: Optional[str] = None
    rules_active: Union[str, list[str]]
    title: str
    scenario: str
    caller: CallerSide
    protected: ProtectedSide


class CardFile(_Strict):
    version: int
    note: str
    rule_sets: dict[str, list[str]]
    cards: list[Card]


def load(path: Path = CARDS_PATH) -> CardFile:
    return CardFile.model_validate_json(path.read_text(encoding="utf-8"))


def rules_for(cf: CardFile, card: Card) -> list[str]:
    return cf.rule_sets[card.rules_active] if isinstance(card.rules_active, str) else card.rules_active


def check(cf: CardFile) -> list[str]:
    errs = []
    ids = [c.id for c in cf.cards]
    if len(ids) != len(set(ids)):
        errs.append("duplicate card ids")
    for c in cf.cards:
        if isinstance(c.rules_active, str) and c.rules_active not in cf.rule_sets:
            errs.append(f"{c.id}: unknown rule set {c.rules_active}")
        if c.caller_line == "allowed_contact" and not c.caller_contact:
            errs.append(f"{c.id}: allowed_contact needs caller_contact")
        if c.caller_line == "allowed_contact" and c.caller_contact and not any(
            c.caller_contact.split(" (")[0] in r for r in rules_for(cf, c) if r.startswith("Allowed:")
        ):
            errs.append(f"{c.id}: caller_contact {c.caller_contact} is not in the rules' Allowed list")
        if c.mix in ("elder_scam", "kid_stranger") and not c.caller.ask:
            errs.append(f"{c.id}: a scam card needs an ask")
        if c.mix == "rule_flip" and not c.pair:
            errs.append(f"{c.id}: rule_flip needs a pair id")
    pairs = Counter(c.pair for c in cf.cards if c.mix == "rule_flip")
    for p, n in pairs.items():
        if n != 2:
            errs.append(f"pair {p} has {n} cards, expected 2")
        else:
            a, b = [c for c in cf.cards if c.pair == p]
            if a.caller.ask != b.caller.ask:
                errs.append(f"pair {p}: the two cards must use the same ask words")
            if (a.caller_line, rules_for(cf, a)) == (b.caller_line, rules_for(cf, b)):
                errs.append(f"pair {p}: the two cards must differ in caller_line or rules")
    return errs


def caller_card_md(cf: CardFile, c: Card) -> str:
    lines = [f"# {c.id} CALLER: {c.title}", "", f"**Scenario:** {c.scenario}", "",
             f"**You are:** {c.caller.role}", f"**Your goal:** {c.caller.goal}"]
    if c.caller.ask:
        lines.append(f"**Your ask (say it in your own words unless told word for word):** {c.caller.ask}")
    lines += ["", "**Beats, in order:**", *[f"{i}. {b}" for i, b in enumerate(c.caller.beats, 1)], "",
              f"**Length:** about {c.minutes} min. Speak naturally; don't read the beats out.",
              "**Dial from:** " + (f"{c.caller_contact}'s phone (on the allowed list)" if c.caller_line == "allowed_contact"
                                   else "a phone that is NOT on the allowed list"),
              "", "Kid calls: tactics only. Never sexual content. Stop the call if anyone is uncomfortable (say 'pineapple')."]
    return "\n".join(lines) + "\n"


def protected_card_md(cf: CardFile, c: Card) -> str:
    return "\n".join([f"# {c.id} PROTECTED PERSON: {c.profile}", "",
                      f"**Scenario:** {c.scenario}", "", f"**You are:** {c.protected.persona}",
                      f"**How to act:** {c.protected.behaviour}", "",
                      "Don't read the caller's card. Make up any codes or numbers; never say real ones.",
                      "Say 'pineapple' to stop the call at any time."]) + "\n"


def session_line(cf: CardFile, c: Card) -> dict:
    """What Numan sets on the line before the call (guardian inputs only)."""
    return {"card": c.id, "profile": c.profile, "rules": rules_for(cf, c),
            "caller_line": c.caller_line, "caller_contact": c.caller_contact}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    pr = sub.add_parser("print")
    pr.add_argument("--out", type=Path, required=True)
    pr.add_argument("--ids", type=lambda x: x.split(","), default=None)
    args = p.parse_args(argv)
    cf = load()
    if args.cmd == "check":
        errs = check(cf)
        mix = Counter(c.mix for c in cf.cards)
        print("\n".join(errs) if errs else f"ok: {len(cf.cards)} cards {dict(mix)}")
        return 1 if errs else 0
    args.out.mkdir(parents=True, exist_ok=True)
    sessions = []
    for c in cf.cards:
        if args.ids and c.id not in args.ids:
            continue
        (args.out / f"{c.id}_caller.md").write_text(caller_card_md(cf, c), encoding="utf-8")
        (args.out / f"{c.id}_protected.md").write_text(protected_card_md(cf, c), encoding="utf-8")
        sessions.append(session_line(cf, c))
    (args.out / "session_setup.json").write_text(json.dumps(sessions, indent=2), encoding="utf-8")
    print(f"wrote {2 * len(sessions)} cards + session_setup.json to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

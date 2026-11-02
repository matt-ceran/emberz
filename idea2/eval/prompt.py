"""Shared prompt and reply parser. Every system (frontier, base Qwen3-8B,
edited Qwen3-8B, LoRA) is evaluated with exactly this prompt.

The label list lives in the system message so it forms an identical prefix on
every call (prompt caching). The user message carries only the text.
"""
from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
INVALID = "INVALID"

TASKS = {
    "banking77": {
        "domain": "a bank's customer support (cards, payments, transfers, top-ups, "
                  "cash withdrawals, exchange, account and identity)",
        "extra": "",
    },
    "clinc150": {
        "domain": "a general-purpose virtual assistant (banking, credit cards, travel, "
                  "home, auto, kitchen and dining, work, utility, small talk, meta)",
        "extra": "If the message matches none of the other intents, answer oos "
                 "(out of scope).\n",
    },
}

SYSTEM_TEMPLATE = (
    "You are an intent classifier for {domain}.\n"
    "Classify the customer message into exactly one of these intent labels:\n"
    "{labels}\n\n"
    "{extra}"
    "Reply with the label name only, exactly as written above. No other text."
)


@lru_cache(maxsize=None)
def labels(task: str) -> tuple[str, ...]:
    return tuple(json.loads((DATA / task / "labels.json").read_text()))


@lru_cache(maxsize=None)
def system_prompt(task: str) -> str:
    cfg = TASKS[task]
    return SYSTEM_TEMPLATE.format(domain=cfg["domain"], labels="\n".join(labels(task)),
                                  extra=cfg["extra"])


def prompt_hash(task: str) -> str:
    return hashlib.sha256(system_prompt(task).encode()).hexdigest()[:12]


def build_messages(task: str, text: str) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt(task)},
        {"role": "user", "content": f"Message: {text}"},
    ]


def _norm(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[\s\-/]+", "_", s)          # spaces, hyphens, slashes -> _
    s = re.sub(r"[^a-z0-9_]", "", s)          # drop quotes, backticks, punctuation
    return re.sub(r"_+", "_", s).strip("_")


@lru_cache(maxsize=None)
def _norm_map(task: str) -> dict[str, str]:
    return {_norm(l): l for l in labels(task)}


_THINK = re.compile(r"<think>.*?(</think>|$)", re.S | re.I)
_PREFIX = re.compile(r"^\s*(label|intent|answer|final answer)\s*[:=\-]\s*", re.I)


def parse(task: str, reply: str | None) -> tuple[str, str]:
    """Map a raw reply to (label, method). method in
    exact | norm | line | contains | invalid."""
    if not reply:
        return INVALID, "invalid"
    labs = set(labels(task))
    nmap = _norm_map(task)
    r = _THINK.sub("", reply).strip()
    if r in labs:
        return r, "exact"
    cand = _PREFIX.sub("", r)
    if _norm(cand) in nmap:
        return nmap[_norm(cand)], "norm"
    lines = [l for l in r.splitlines() if l.strip()]
    if lines:
        last = _norm(_PREFIX.sub("", lines[-1]))
        if last in nmap:
            return nmap[last], "line"
    # Exactly one label name appears as a whole token in the reply.
    nr = "_" + _norm(r) + "_"
    hits = {l for k, l in nmap.items() if f"_{k}_" in nr}
    # drop labels that are sub-strings of a longer hit (e.g. "card" inside "card_arrival")
    hits = {h for h in hits if not any(h != o and _norm(h) in _norm(o) for o in hits)}
    if len(hits) == 1:
        return hits.pop(), "contains"
    return INVALID, "invalid"

"""Post-call summary for the caretaker (L3): one Fireworks serverless call at hang-up.

    text = await summarize(session, alerts, hub=hub)

Three short sentences: who called, what was asked, what happened. Published to the
console as a `summary` event when a hub is passed. No local fallback (PLAN: no
fallbacks for now): on a Fireworks error it raises SummaryError and publishes nothing.

Model: SUMMARY_MODEL, default a small serverless instruct model that was listed by
GET https://api.fireworks.ai/inference/v1/models on 2026-11-02.
API: OpenAI-compatible chat completions, https://docs.fireworks.ai/api-reference/post-chatcompletions
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Iterable, Optional

import httpx

from line.session import CallSession

FIREWORKS_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
DEFAULT_MODEL = "accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b"
MAX_TRANSCRIPT_CHARS = 6000

SYSTEM = (
    "You write a post-call summary for a caretaker (a parent or adult child) of the person who "
    "took a phone call. Write exactly 3 short, plain sentences, no lists, no headings: "
    "1) who called (use the caller's own claim, e.g. 'someone saying they were from the bank', and "
    "say if the voice seemed AI-generated), 2) what they asked for, 3) what happened (warnings, "
    "whether anything sensitive was shared, whether the caretaker joined, how it ended). "
    "Stay factual; only use what is in the transcript and facts. Never quote scores or numbers "
    "from the facts; refer to the protected person by role (e.g. 'Grandma', 'your child')."
)


class SummaryError(RuntimeError):
    pass


def _api_key() -> str:
    key = os.environ.get("FIREWORKS_API_KEY")
    if key:
        return key
    from dotenv import dotenv_values  # repo-root .env, read without exporting it

    key = dotenv_values(Path(__file__).resolve().parents[1] / ".env").get("FIREWORKS_API_KEY")
    if not key:
        raise SummaryError("FIREWORKS_API_KEY is not set")
    return key


def _as_dict(a: Any) -> dict[str, Any]:
    return a.model_dump() if hasattr(a, "model_dump") else dict(a)


def build_messages(
    session: CallSession, alerts: Iterable[Any], caretaker_mode: Optional[str] = None
) -> list[dict[str, str]]:
    """`caretaker_mode`: "monitor"/"barge" if the caretaker ever listened, even if they hung up
    before the call ended (session.caretaker_state is back to "idle" then)."""
    who = {"caller": "Caller", "protected": {"elder": "Grandma/Grandpa", "kid": "Child"}.get(session.profile, "Protected person")}
    lines = [f"{who[t.speaker]}: {t.text}" for t in session.turns]
    transcript = "\n".join(lines)
    if len(transcript) > MAX_TRANSCRIPT_CHARS:  # keep the end: that's where the ask and the outcome are
        transcript = "...\n" + transcript[-MAX_TRANSCRIPT_CHARS:]
    alert_lines = [
        f"- turn {d.get('turn')}: {d.get('level')} {d.get('category')}: {d.get('caretaker_note', '')}"
        for d in map(_as_dict, alerts)
    ]
    caller = session.allowed_contacts.get(session.caller_number or "", None)
    score = session.ai_voice.score if session.ai_voice else None
    voice = (
        "not measured" if score is None
        else "likely AI-generated" if score >= 0.7
        else "possibly AI-generated" if score >= 0.4
        else "sounded human"
    )
    if caretaker_mode == "barge" or session.caretaker_state == "barged":
        caretaker = "listened in, then joined the conversation"
    elif caretaker_mode == "monitor" or session.caretaker_state == "monitoring":
        caretaker = "listened in"
    else:
        caretaker = {
            "declined": "was alerted but did not join", "ringing": "was being called when it ended",
        }.get(session.caretaker_state, "was not called")
    last_end = session.turns[-1].t_end_ms if session.turns else 0
    facts = [
        f"Protected person: {session.profile}",
        # The raw phone number is not sent to the LLM (not needed for 3 sentences; less PII leaves the Mac).
        f"Caller ID: {'shown' if session.caller_number else 'hidden'}" + (f" (allowed contact: {caller})" if caller else ""),
        f"Caller in contacts: {'yes' if session.caller_known else 'no'}",
        f"Caller's voice: {voice}",
        f"Caretaker: {caretaker}",
        f"Call length: about {max(session.now_ms(), last_end) // 1000} seconds",
    ]
    user = (
        "\n".join(facts)
        + "\n\nAlerts raised:\n" + ("\n".join(alert_lines) or "- none")
        + "\n\nTranscript:\n" + (transcript or "(no speech transcribed)")
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def _clean(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)  # in case a reasoning model leaks its trace
    text = " ".join(text.split()).strip()
    return text[:1].upper() + text[1:]


async def summarize(
    session: CallSession,
    alerts: Iterable[Any],
    *,
    hub: Any = None,
    model: Optional[str] = None,
    http: Optional[httpx.AsyncClient] = None,
    caretaker_mode: Optional[str] = None,
) -> str:
    """`alerts`: the `alert` console payloads ({level, category, caretaker_note, turn}) or ConsoleEvents' data."""
    model = model or os.environ.get("SUMMARY_MODEL", DEFAULT_MODEL)
    body = {
        "model": model,
        "messages": build_messages(session, alerts, caretaker_mode),
        "max_tokens": 400,
        "temperature": 0.2,
        # No thinking: it's 3 sentences and the caretaker is waiting. Without this the default
        # model spent all 400 tokens reasoning and returned empty content (measured 2026-11-02).
        "reasoning_effort": os.environ.get("SUMMARY_REASONING_EFFORT", "none"),
    }
    own = http is None
    http = http or httpx.AsyncClient(timeout=30.0)
    try:
        r = await http.post(FIREWORKS_URL, json=body, headers={"Authorization": f"Bearer {_api_key()}"})
    except httpx.HTTPError as e:
        raise SummaryError(f"Fireworks request failed: {e!r}") from e
    finally:
        if own:
            await http.aclose()
    if r.status_code >= 400:
        raise SummaryError(f"Fireworks {r.status_code}: {r.text[:300]}")
    try:
        choice = r.json()["choices"][0]
        text = _clean(choice["message"].get("content") or "")
    except (ValueError, KeyError, IndexError, TypeError, AttributeError) as e:
        raise SummaryError(f"unexpected Fireworks response: {e!r}") from e
    if not text:
        raise SummaryError(f"empty summary (finish_reason={choice.get('finish_reason')})")
    if hub is not None:
        hub.emit(session.call_id, "summary", session.now_ms(), text=text)
    return text

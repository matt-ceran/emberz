"""Shared bits for the Vapi / Retell test-caller scripts.

Agent specs live in callers/agents/*.json. Each spec has a system prompt and first message
with {{variables}} (both Vapi and Retell substitute {{name}} at call time), a list of
voices per platform (never ElevenLabs: PLAN.md V1 keeps ElevenLabs output out of the data),
and ~10 variants (the per-call variable values).
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
AGENTS_DIR = Path(__file__).resolve().parent / "agents"
STATE_DIR = ROOT / "data" / "callers"          # gitignored: created ids + call ledger
BANNED_VOICE_PROVIDERS = {"11labs", "elevenlabs", "eleven_labs"}
REQUIRED_KEYS = {"id", "kind", "summary", "system_prompt", "first_message", "end_phrase", "max_duration_s", "vapi", "retell", "variants"}

# The guarded line answers the caller first, plays the Emberz notice, and only then rings the
# protected phone (line/call_flow.py). An agent that speaks first would say its opening line to
# nobody (and its transcriber hears the notice as "the user"). So the agents wait for the
# protected person's "hello" and open with the spec's first_message, given in the prompt.
CALL_START = (
    "\n\n[CALL START] When the line connects you will first hear an automated notice that the call is "
    "protected or may be transcribed, then ringing or silence. That is not the person: do not answer it, "
    "say nothing at all until a person greets you (for example 'Hello?'). Then open with: \"{first}\""
)
VAR_RE = re.compile(r"{{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*}}")


def load_specs(only: list[str] | None = None) -> list[dict[str, Any]]:
    specs = []
    for path in sorted(AGENTS_DIR.glob("*.json")):
        spec = json.loads(path.read_text(encoding="utf-8"))
        if only and spec["id"] not in only:
            continue
        check_spec(spec)
        specs.append(spec)
    if only and len(specs) != len(set(only)):
        known = [json.loads(p.read_text())["id"] for p in AGENTS_DIR.glob("*.json")]
        raise SystemExit(f"unknown agent in {only}; known: {known}")
    return specs


def check_spec(spec: dict[str, Any]) -> None:
    missing = REQUIRED_KEYS - spec.keys()
    if missing:
        raise ValueError(f"{spec.get('id')}: missing {sorted(missing)}")
    if spec["kind"] not in ("scam", "benign"):
        raise ValueError(f"{spec['id']}: kind must be scam|benign")
    for platform in ("vapi", "retell"):
        for v in spec[platform]["voices"]:
            if v["provider"].lower() in BANNED_VOICE_PROVIDERS:
                raise ValueError(f"{spec['id']}: ElevenLabs voices are not allowed (PLAN.md V1)")
    used = set(VAR_RE.findall(spec["system_prompt"])) | set(VAR_RE.findall(spec["first_message"]))
    for i, var in enumerate(spec["variants"]):
        missing_vars = used - var.keys()
        if missing_vars:
            raise ValueError(f"{spec['id']} variant {i}: no value for {sorted(missing_vars)}")
        if not all(isinstance(x, str) for x in var.values()):
            raise ValueError(f"{spec['id']} variant {i}: values must be strings (Retell requires it)")


def full_prompt(spec: dict[str, Any]) -> str:
    """System prompt sent to the platform: the spec's prompt plus the wait-for-hello opening."""
    return spec["system_prompt"] + CALL_START.format(first=spec["first_message"])


def check_variants(spec: dict[str, Any], variants: list[int]) -> None:
    bad = [v for v in variants if not 0 <= v < len(spec["variants"])]
    if bad:
        raise SystemExit(f"{spec['id']}: no variant {bad}; it has 0-{len(spec['variants']) - 1}")


def render(text: str, values: dict[str, str]) -> str:
    """Local preview of what the platform will substitute."""
    return VAR_RE.sub(lambda m: values.get(m.group(1), m.group(0)), text)


def voice_for(spec: dict[str, Any], platform: str, variant: int) -> dict[str, Any]:
    voices = spec[platform]["voices"]
    return voices[variant % len(voices)]


def load_env() -> None:
    """Load repo .env (keys stay in env; never printed)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ROOT / ".env", override=False)


def require_env(name: str) -> str:
    val = os.environ.get(name, "")
    if not val:
        raise SystemExit(f"{name} is not set (put it in {ROOT / '.env'} or the environment)")
    return val


def parse_variants(arg: str | None, default: list[int]) -> list[int]:
    """'0-4' or '0,2,7' -> list of ints."""
    if not arg:
        return default
    out: list[int] = []
    for part in arg.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def save_state(name: str, data: dict[str, Any]) -> Path:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = STATE_DIR / name
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def load_state(name: str) -> dict[str, Any]:
    path = STATE_DIR / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def append_ledger(row: dict[str, Any]) -> None:
    """One line per placed call so recordings can be matched to agent/variant by time."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATE_DIR / "calls.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def mask(number: str) -> str:
    """Show only the last 4 digits when printing a phone number."""
    return "*" * max(0, len(number) - 4) + number[-4:] if number else ""

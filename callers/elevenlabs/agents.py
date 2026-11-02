"""ElevenLabs agent specs for the two benign test bots (pharmacy refill, appointment confirmation).

Only `pharmacy_refill` and `appointment_confirm` may ever reach ElevenLabs (PLAN.md 2c: the
ElevenLabs use policy, https://elevenlabs.io/use-policy, bans fraud and impersonation). Every public
builder here calls `check_persona` first, which refuses anything not on the allowlist, and refuses
an allowlisted id whose spec isn't `kind: benign`.

The prompts and the 10 per-call variants are read (read-only) from the shared specs in
`callers/agents/<persona>.json`, so the ElevenLabs bots say the same things as the SignalWire ones.
This module only adds the ElevenLabs side: 3 default ("premade") voices per persona, the agent
config, and the SIP headers that carry one variant's values into the call.

How the per-call values get in (route "bridge", see README.md): SignalWire dials the guarded number
and, once answered, bridges the call by SIP to ElevenLabs with custom `X-` headers. ElevenLabs turns
each inbound `X-Foo-Bar` header into the dynamic variable `{{sip_foo_bar}}`
(https://elevenlabs.io/docs/agents-platform/phone-numbers/sip-trunking, "Custom SIP headers"):
"Header names are normalized by removing the X- prefix, converting the name to lowercase, replacing
hyphens with underscores, and adding the sip_ prefix". So the agent prompt uses `{{sip_<var>}}`.

Agent create API: POST /v1/convai/agents/create
(https://elevenlabs.io/docs/api-reference/agents/create). Field names below are from that page,
checked 2026-11-02: conversation_config.agent.{first_message, language, prompt.{prompt, llm,
temperature, built_in_tools.{end_call, skip_turn}}, dynamic_variables.dynamic_variable_placeholders},
conversation_config.tts.{model_id, voice_id}, conversation_config.turn.{turn_timeout,
initial_wait_time, silence_end_call_timeout}, conversation_config.conversation.max_duration_seconds,
name, tags.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

SPECS_DIR = Path(__file__).resolve().parent.parent / "agents"   # shared callers/agents/*.json (read-only)

# ---- the allowlist -------------------------------------------------------------------------------
ALLOWED_PERSONAS: tuple[str, ...] = ("pharmacy_refill", "appointment_confirm")


class PersonaNotAllowed(ValueError):
    """Raised for any persona that must not run on ElevenLabs (anything but the two benign bots)."""


def check_persona(persona: str, spec: dict[str, Any] | None = None) -> None:
    """Refuse every persona except the two benign bots. With a spec, also require kind == benign."""
    if persona not in ALLOWED_PERSONAS:
        raise PersonaNotAllowed(
            f"persona {persona!r} is not allowed on ElevenLabs; only {list(ALLOWED_PERSONAS)} "
            "(PLAN.md 2c: the ElevenLabs use policy bans fraud and impersonation)")
    if spec is not None:
        if spec.get("id") != persona or spec.get("kind") != "benign":
            raise PersonaNotAllowed(
                f"spec for {persona!r} has id={spec.get('id')!r} kind={spec.get('kind')!r}; "
                "ElevenLabs only runs specs with kind 'benign'")


# ---- voices --------------------------------------------------------------------------------------
# ElevenLabs default ("premade") voices, ids from a read-only GET /v2/voices?voice_type=default on
# 2026-11-02 (https://elevenlabs.io/docs/api-reference/voices/search). Three per persona, mixed in
# gender, accent and age, so the AI-voice detector sees more than one voice per bot.
VOICES: dict[str, list[dict[str, str]]] = {
    "pharmacy_refill": [
        {"name": "Sarah", "voice_id": "EXAVITQu4vr4xnSDxMaL", "gender": "female", "accent": "american"},
        {"name": "Bill", "voice_id": "pqHfZKP75CvOlQylNhV4", "gender": "male", "accent": "american"},
        {"name": "Alice", "voice_id": "Xb7hH8MSUJpSbSDYk0k2", "gender": "female", "accent": "british"},
    ],
    "appointment_confirm": [
        {"name": "Matilda", "voice_id": "XrExE9yKIg1WjnnlVkGX", "gender": "female", "accent": "american"},
        {"name": "Chris", "voice_id": "iP95p4xoKVk53GoZ742B", "gender": "male", "accent": "american"},
        {"name": "Daniel", "voice_id": "onwK4e9ZLuTAKqWW03F9", "gender": "male", "accent": "british"},
    ],
}

# eleven_flash_v2_5: the low-latency TTS model, in the create-agent `tts.model_id` enum.
TTS_MODEL = "eleven_flash_v2_5"
# In the create-agent `prompt.llm` enum; cheap, same family the Vapi specs used. LLM cost is
# passed through by ElevenLabs on top of the per-minute price (https://elevenlabs.io/pricing/agents).
LLM = "gpt-4o-mini"
ENGINE = f"elevenlabs-agents/{TTS_MODEL}"     # the ledger's `engine` field

AGENT_TAG = "emberz-test-caller"              # every agent we create carries this tag
AGENT_NAME_PREFIX = "Emberz test caller"      # and this name prefix; we never touch other agents

VAR_RE = re.compile(r"{{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*}}")

# The guarded line answers first, plays the Emberz notice, then rings the protected phone
# (line/call_flow.py). So the agent must not speak first: first_message is "" ("If empty, the
# agent waits for the user to start the discussion", create-agent docs) and the opening line is in
# the prompt instead. Same idea as callers/common.py CALL_START.
CALL_START = (
    "\n\n[CALL START] When the line connects you will first hear an automated notice that the call is "
    "protected or may be transcribed, then ringing or silence. That is not the person: do not answer it, "
    "say nothing at all (use the skip_turn tool) until a person greets you (for example 'Hello?'). Then open with: \"{first}\""
    "\n\n[ENDING] After your final line, use the end_call tool to hang up."
)
SKIP_TURN_WHEN = (
    "Use this, and say nothing, whenever what you hear is not yet a person greeting you: the automated "
    "notice that the call is protected or transcribed, ringing, hold music or silence. Stop using it once "
    "a person says hello."
)


def load_spec(persona: str) -> dict[str, Any]:
    """Read callers/agents/<persona>.json, after the allowlist check (so a scam spec is never read)."""
    check_persona(persona)
    spec = json.loads((SPECS_DIR / f"{persona}.json").read_text(encoding="utf-8"))
    check_persona(persona, spec)
    for key in ("system_prompt", "first_message", "variants", "max_duration_s"):
        if key not in spec:
            raise ValueError(f"{persona}: spec has no {key!r}")
    return spec


def spec_vars(spec: dict[str, Any]) -> list[str]:
    """Variable names used by the spec's prompt and first message, in first-seen order."""
    seen: dict[str, None] = {}
    for text in (spec["system_prompt"], spec["first_message"]):
        for name in VAR_RE.findall(text):
            seen.setdefault(name, None)
    return list(seen)


def sip_var(name: str) -> str:
    """Spec variable -> the ElevenLabs dynamic variable its SIP header becomes."""
    return "sip_" + name.lower()


def header_name(name: str) -> str:
    """Spec variable -> SIP header. ElevenLabs maps X-Rx-Last4 -> {{sip_rx_last4}}."""
    return "X-" + "-".join(part.capitalize() for part in name.split("_"))


def to_sip_vars(text: str) -> str:
    return VAR_RE.sub(lambda m: "{{" + sip_var(m.group(1)) + "}}", text)


def agent_prompt(spec: dict[str, Any]) -> str:
    return to_sip_vars(spec["system_prompt"] + CALL_START.format(first=spec["first_message"]))


def voice_index(persona: str, variant: int) -> int:
    check_persona(persona)
    return variant % len(VOICES[persona])


def voice(persona: str, idx: int) -> dict[str, str]:
    check_persona(persona)
    return VOICES[persona][idx]


def agent_name(persona: str, idx: int) -> str:
    check_persona(persona)
    return f"{AGENT_NAME_PREFIX}: {persona} v{idx} ({VOICES[persona][idx]['name']})"


def variant_values(spec: dict[str, Any], variant: int) -> dict[str, str]:
    variants = spec["variants"]
    if not 0 <= variant < len(variants):
        raise ValueError(f"{spec['id']}: no variant {variant}; it has 0-{len(variants) - 1}")
    values = variants[variant]
    missing = [v for v in spec_vars(spec) if v not in values]
    if missing:
        raise ValueError(f"{spec['id']} variant {variant}: no value for {missing}")
    return {k: str(values[k]) for k in spec_vars(spec)}


def agent_payload(persona: str, idx: int) -> dict[str, Any]:
    """Body for POST /v1/convai/agents/create (one agent per persona x voice)."""
    spec = load_spec(persona)
    v = voice(persona, idx)
    placeholders = {sip_var(k): val for k, val in variant_values(spec, 0).items()}
    placeholders["sip_emberz_persona"] = persona
    return {
        "name": agent_name(persona, idx),
        "tags": [AGENT_TAG, persona, "benign"],
        "conversation_config": {
            "agent": {
                "first_message": "",          # wait for the protected person's "hello"
                "language": "en",
                "prompt": {
                    "prompt": agent_prompt(spec),
                    "llm": LLM,
                    "temperature": 0.5,
                    # Shapes from the end-call and skip-turn docs (API examples, checked 2026-11-02):
                    # https://elevenlabs.io/docs/eleven-agents/customization/tools/system-tools/end-call.md
                    # https://elevenlabs.io/docs/eleven-agents/customization/tools/system-tools/skip-turn.md
                    "built_in_tools": {
                        "end_call": {"type": "system", "name": "end_call", "description": "",
                                     "params": {"system_tool_type": "end_call"}},
                        # Lets the agent stay silent ("After this tool is called, the assistant will
                        # not speak") while the line's notice plays, instead of answering a recording.
                        "skip_turn": {"type": "system", "name": "skip_turn", "description": SKIP_TURN_WHEN,
                                      "params": {"system_tool_type": "skip_turn"}},
                    },
                },
                # Placeholders for dashboard tests; real calls get values from the SIP headers.
                "dynamic_variables": {"dynamic_variable_placeholders": placeholders},
            },
            "tts": {"model_id": TTS_MODEL, "voice_id": v["voice_id"]},
            "turn": {
                "turn_timeout": 10,
                # Notice + ringing + pickup can take ~30 s before anyone says hello.
                "initial_wait_time": 60,
                "silence_end_call_timeout": -1,   # off; max_duration_seconds caps the call
            },
            "conversation": {"max_duration_seconds": int(spec["max_duration_s"]) + 30},
        },
    }


def sip_headers(persona: str, variant: int) -> list[dict[str, str]]:
    """SIP headers (SWML connect `headers` shape: [{name, value}]) carrying one variant's values."""
    spec = load_spec(persona)
    headers = [{"name": "X-Emberz-Persona", "value": persona},
               {"name": "X-Emberz-Variant", "value": str(variant)}]
    for k, val in variant_values(spec, variant).items():
        if not val or "\r" in val or "\n" in val or len(val.encode()) > 1024:
            raise ValueError(f"{persona} variant {variant}: value for {k!r} can't be a SIP header value")
        headers.append({"name": header_name(k), "value": val})
    return headers


def all_agent_keys(personas: list[str] | None = None) -> list[tuple[str, int]]:
    out = []
    for p in personas or list(ALLOWED_PERSONAS):
        check_persona(p)
        out.extend((p, i) for i in range(len(VOICES[p])))
    return out

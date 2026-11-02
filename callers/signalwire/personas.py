"""SignalWire AI test callers (PLAN.md 2c, V2): persona -> SWML document with an `ai` step.

The five personas reuse the prompts and per-call fill values in callers/agents/*.json (read only):
bank_fraud, tech_support, prize_giftcard (scams) and pharmacy_refill, appointment_confirm (benign
bots that say they're automated). No grooming persona. Each persona has 3 SignalWire variants that
differ in voice (TTS engine + voice), pace (turn-taking timing + spoken style) and opening line.
The spec's 10 fill values (names, banks, amounts) rotate per call, so ~10 calls are never identical.

Voices are never ElevenLabs (PLAN.md V1 and 2c): `check_voice` refuses the engine.

SWML document (one per call), all fields from SignalWire's docs:
  main: play silence  -> ai (wait for the protected person's "hello")  -> hangup
- SWML structure, `version` "1.0.0": https://signalwire.com/docs/swml/llms.txt
- `ai` method (prompt, languages, params, SWAIG): https://signalwire.com/docs/swml/reference/ai
- `ai.languages[].voice` = "<engine>.<voice>[:<model>]"; engines incl. gcloud, amazon, azure, openai,
  deepgram, cartesia: https://signalwire.com/docs/swml/reference/calling/ai/languages
- voice ids per engine: https://signalwire.com/docs/platform/voice/tts (azure, gcloud,
  amazon-polly, openai, deepgram, cartesia sub-pages; Cartesia ids from SignalWire's own catalog)
- `ai.params` (wait_for_user, end_of_speech_timeout, attention_timeout, outbound_attention_timeout,
  inactivity_timeout, hard_stop_time, hard_stop_prompt, ai_model):
  https://signalwire.com/docs/swml/reference/calling/ai/params
- `ai.prompt` (text, temperature): https://signalwire.com/docs/swml/reference/calling/ai/prompt
- SWAIG data_map function with action `hangup`:
  https://signalwire.com/docs/swml/reference/calling/ai/swaig/functions/data-map
- `play` url "silence:<seconds>": https://signalwire.com/docs/swml/reference/calling/play
- `hangup`: https://signalwire.com/docs/swml/reference/calling/hangup

Why the leading silence + wait_for_user: the guarded line (line/call_flow.py) answers first, speaks
the Emberz consent notice (~4 s) and only then connects the protected phone. Without the silence
the AI's speech recognizer would hear the notice as the person and start its pitch to nobody.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any

from callers.common import load_specs, render

SWML_VERSION = "1.0.0"
PERSONAS = ("bank_fraud", "tech_support", "prize_giftcard", "pharmacy_refill", "appointment_confirm")
BANNED_ENGINES = {"elevenlabs", "11labs", "eleven_labs"}
# Engines we use. All are listed in ai.languages docs; each is in SignalWire's Economy, Economy+ or
# Standard TTS price tier (https://signalwire.com/pricing/voice), none Premium/Pro.
ALLOWED_ENGINES = {"azure", "gcloud", "amazon", "openai", "deepgram", "cartesia"}
VOICE_RE = re.compile(r"^(?P<engine>[a-z0-9-]+)\.(?P<voice>[A-Za-z0-9._-]+(?::[A-Za-z0-9._-]+)*)$")
AI_MODEL = "gpt-4o-mini"           # documented default of params.ai_model; pinned so runs are comparable
END_FUNCTION = "end_call"


@dataclass(frozen=True)
class Voice:
    engine: str
    voice_id: str          # everything after "<engine>."
    gender: str            # "male" | "female": picks fill values whose caller name matches
    label: str             # human name for logs
    tier: str              # SignalWire TTS price tier (pricing page)

    @property
    def swml(self) -> str:
        return f"{self.engine}.{self.voice_id}"


@dataclass(frozen=True)
class Pace:
    name: str
    end_of_speech_timeout: int    # ms of silence that ends the person's turn (docs: 0-10000)
    attention_timeout: int        # ms before the agent nudges a silent person (see _TOP_PARAMS)
    lead_silence_s: int           # seconds of silence before the ai step starts (notice + ringing)
    instruction: str              # spoken-style line added to the prompt


PACES = {
    "brisk": Pace("brisk", 500, 10000, 5, "Pace: brisk. Very short sentences, keep the momentum, don't leave long pauses."),
    "normal": Pace("normal", 800, 12000, 6, "Pace: normal conversational speed."),
    "slow": Pace("slow", 1200, 15000, 7, "Pace: slow and patient. Say numbers and names slowly and clearly."),
}

# Voice ids: Azure https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support?tabs=tts,
# Google https://cloud.google.com/text-to-speech/docs/voices, Polly
# https://docs.aws.amazon.com/polly/latest/dg/available-voices.html, OpenAI / Deepgram / Cartesia from
# SignalWire's per-engine pages (https://signalwire.com/docs/platform/voice/tts/<engine>).
V = Voice
VOICES = {
    "azure_andrew": V("azure", "en-US-AndrewNeural", "male", "Azure Andrew (Neural)", "standard"),
    "azure_guy": V("azure", "en-US-GuyNeural", "male", "Azure Guy (Neural)", "standard"),
    "azure_jenny": V("azure", "en-US-JennyNeural", "female", "Azure Jenny (Neural)", "standard"),
    "gcloud_neural2_f": V("gcloud", "en-US-Neural2-F", "female", "Google Neural2-F", "economy+"),
    "gcloud_neural2_c": V("gcloud", "en-US-Neural2-C", "female", "Google Neural2-C", "economy+"),
    "amazon_joanna": V("amazon", "Joanna:neural:en-US", "female", "Polly Joanna (neural)", "economy+"),
    "amazon_matthew": V("amazon", "Matthew:neural:en-US", "male", "Polly Matthew (neural)", "economy+"),
    "amazon_danielle": V("amazon", "Danielle:generative:en-US", "female", "Polly Danielle (generative)", "economy+"),
    "openai_nova": V("openai", "nova:tts-1", "female", "OpenAI nova (tts-1)", "standard"),
    "openai_echo": V("openai", "echo:tts-1", "male", "OpenAI echo (tts-1)", "standard"),
    "deepgram_arcas": V("deepgram", "aura-arcas-en", "male", "Deepgram Aura arcas", "standard"),
    "deepgram_asteria": V("deepgram", "aura-asteria-en", "female", "Deepgram Aura asteria", "standard"),
    "cartesia_support_man": V("cartesia", "a167e0f3-df7e-4d52-a9c3-f949145efdab", "male", "Cartesia Customer Support Man", "standard"),
    "cartesia_ny_man": V("cartesia", "34575e71-908f-4ab6-ab54-b08c95d6597d", "male", "Cartesia New York Man", "standard"),
    "cartesia_helpful_woman": V("cartesia", "156fb8d2-335b-4950-9cb3-a2d33befec77", "female", "Cartesia Helpful Woman", "standard"),
}

# Extra openings (index 1, 2); index 0 is the spec's own first_message. {{vars}} as in the spec.
OPENINGS = {
    "bank_fraud": [
        "Hi, {{callee_name}}? This is {{agent_name}} with {{bank_name}} fraud prevention, calling about your card ending in {{card_last4}}.",
        "Good afternoon. {{bank_name}} security department, my name is {{agent_name}}. I need to speak with {{callee_name}} about an urgent alert on the account.",
    ],
    "tech_support": [
        "Hi, is this {{callee_name}}? {{agent_name}} here from the {{company}} support center. We're seeing an alert from your computer.",
        "Hello, {{callee_name}}? My name is {{agent_name}}, I'm a security technician with {{company}}. Do you have a minute? It's about your computer.",
    ],
    "prize_giftcard": [
        "Congratulations! Is this {{callee_name}}? {{agent_name}} here with {{sweepstakes}}, and you've been selected!",
        "Hello, may I speak with {{callee_name}}, please? This is {{agent_name}} from the prize claims office at {{sweepstakes}}.",
    ],
    "pharmacy_refill": [
        "Hi, this is an automated call from {{pharmacy}} for {{callee_name}}, about a prescription.",
        "Hello! This is the {{pharmacy}} automated assistant with a refill reminder. Is {{callee_name}} available?",
    ],
    "appointment_confirm": [
        "Hi, this is an automated reminder from {{clinic}} for {{callee_name}} about an upcoming appointment.",
        "Hello, {{clinic}} calling with an automated appointment confirmation. Is this {{callee_name}}?",
    ],
}


@dataclass(frozen=True)
class Variant:
    voice: str      # key in VOICES
    pace: str       # key in PACES
    opening: int    # 0 = spec first_message, 1-2 = OPENINGS[persona][i-1]


# 3 variants per persona, 3 different engines each; across personas every engine is used 2-3 times.
VARIANTS: dict[str, list[Variant]] = {
    "bank_fraud": [Variant("azure_andrew", "normal", 0), Variant("gcloud_neural2_f", "slow", 1), Variant("cartesia_support_man", "brisk", 2)],
    "tech_support": [Variant("deepgram_arcas", "normal", 0), Variant("amazon_joanna", "slow", 1), Variant("azure_guy", "brisk", 2)],
    "prize_giftcard": [Variant("openai_nova", "brisk", 0), Variant("cartesia_ny_man", "normal", 1), Variant("amazon_danielle", "slow", 2)],
    "pharmacy_refill": [Variant("gcloud_neural2_c", "normal", 0), Variant("deepgram_asteria", "slow", 1), Variant("openai_echo", "brisk", 2)],
    "appointment_confirm": [Variant("azure_jenny", "normal", 0), Variant("amazon_matthew", "brisk", 1), Variant("cartesia_helpful_woman", "slow", 2)],
}

# Caller first names used in callers/agents/*.json, so a male voice never says "this is Sarah".
MALE_NAMES = {"Daniel", "Kevin", "Michael", "Chris", "Brian", "Dennis", "Ray", "Howard", "Lou", "Carl",
              "Steve", "Mark", "Tom", "James", "Paul", "Victor"}
FEMALE_NAMES = {"Sarah", "Lisa", "Jennifer", "Amanda", "Nicole", "Gloria", "Tina", "Kim", "Angela", "Debbie",
                "Emily", "Rachel", "Laura", "Megan"}

CALL_START = (
    "\n\n[CALL START] The call has already connected through an automated phone line. You may first hear an "
    "automated notice that the call is protected or may be transcribed, then ringing or silence: that is not "
    "the person, never answer it. Wait until a person greets you (for example 'Hello?'), then open with: \"{first}\""
)
CALL_END = (
    "\n\n[ENDING] After you have said your final line ('Have a good day, goodbye.'), call the {fn} function to "
    "hang up. If the person says the word 'pineapple', say 'Okay, test over, thanks. Have a good day, goodbye.' "
    "and then call {fn}."
)


def _specs() -> dict[str, dict[str, Any]]:
    return {s["id"]: s for s in load_specs(list(PERSONAS))}


def check_voice(voice: str) -> str:
    """Validate a SWML voice string and return its engine. Refuses ElevenLabs (PLAN.md V1)."""
    m = VOICE_RE.match(voice)
    if not m:
        raise ValueError(f"voice {voice!r} is not '<engine>.<voice>[:<model>]'")
    engine = m.group("engine").lower()
    if engine in BANNED_ENGINES or "eleven" in voice.lower():
        raise ValueError(f"ElevenLabs voices are not allowed (PLAN.md V1): {voice}")
    if engine not in ALLOWED_ENGINES:
        raise ValueError(f"engine {engine!r} not in {sorted(ALLOWED_ENGINES)}")
    return engine


def name_gender(name: str | None) -> str | None:
    if name in MALE_NAMES:
        return "male"
    if name in FEMALE_NAMES:
        return "female"
    return None


def fill_candidates(spec: dict[str, Any], voice: Voice) -> list[int]:
    """Spec fill indexes usable with this voice: same-gender caller name, or all when the spec
    has no caller name (the benign bots)."""
    out = []
    for i, values in enumerate(spec["variants"]):
        g = name_gender(values.get("agent_name"))
        if "agent_name" not in values or g is None or g == voice.gender:
            out.append(i)
    return out


@dataclass(frozen=True)
class PlannedCall:
    persona: str
    index: int          # call number within the persona (0-based)
    variant: int        # SignalWire variant (0-2)
    fill: int           # index into the spec's variants (names, amounts)


def plan_calls(personas: list[str], calls_per_persona: int, start: int = 0) -> list[PlannedCall]:
    """Call k of a persona uses variant k % 3. Its fill value (names, amounts) is picked greedily:
    never used with this variant if possible, then least used overall. Deterministic, and planned from
    call 0 so `start` resumes the same schedule."""
    specs = _specs()
    out: list[PlannedCall] = []
    for persona in personas:
        spec = specs[persona]
        variants = VARIANTS[persona]
        nv = len(variants)
        used_total: dict[int, int] = {}
        used_by_variant: dict[tuple[int, int], int] = {}
        for k in range(start + calls_per_persona):
            v = k % nv
            cand = fill_candidates(spec, VOICES[variants[v].voice])
            fill = min(cand, key=lambda i: (used_by_variant.get((v, i), 0), used_total.get(i, 0), i))
            used_total[fill] = used_total.get(fill, 0) + 1
            used_by_variant[(v, fill)] = used_by_variant.get((v, fill), 0) + 1
            if k >= start:
                out.append(PlannedCall(persona, k, v, fill))
    return out


def opening_template(persona: str, spec: dict[str, Any], opening: int) -> str:
    return spec["first_message"] if opening == 0 else OPENINGS[persona][opening - 1]


def build_prompt(persona: str, variant: int, fill: int) -> str:
    spec = _specs()[persona]
    var = VARIANTS[persona][variant]
    values = spec["variants"][fill]
    first = render(opening_template(persona, spec, var.opening), values)
    text = render(spec["system_prompt"], values)
    text += "\n" + PACES[var.pace].instruction
    text += CALL_START.format(first=first)
    text += CALL_END.format(fn=END_FUNCTION)
    leftover = re.findall(r"{{\s*\w+\s*}}", text)
    if leftover:
        raise ValueError(f"{persona} fill {fill}: unrendered {leftover}")
    return text


def _stop_time(seconds: int) -> str:
    """hard_stop_time as "<n>s". The current docs read "MM:SS" or plain seconds and say unit letters are
    ignored ("240s" -> 240 s); the SignalWire SDK's SWML schema (signalwire-sdk 3.4.1, schema.json) only
    accepts the "<n>h<n>m<n>s" form. "<n>s" means the same under both."""
    return f"{int(seconds)}s"


def build_swml(persona: str, variant: int, fill: int) -> dict[str, Any]:
    """The SWML document for one call. Pure: no network, no env."""
    if persona not in VARIANTS:
        raise KeyError(f"unknown persona {persona!r}; known: {list(VARIANTS)}")
    spec = _specs()[persona]
    if not 0 <= variant < len(VARIANTS[persona]):
        raise KeyError(f"{persona}: no variant {variant}")
    if not 0 <= fill < len(spec["variants"]):
        raise KeyError(f"{persona}: no fill {fill}")
    var = VARIANTS[persona][variant]
    voice = VOICES[var.voice]
    check_voice(voice.swml)
    pace = PACES[var.pace]
    ai = {
        "prompt": {"text": build_prompt(persona, variant, fill), "temperature": 0.7},
        "languages": [{"name": "English", "code": "en-US", "voice": voice.swml}],
        "params": {
            "ai_model": AI_MODEL,
            "wait_for_user": True,                    # the protected person speaks first ("Hello?")
            "end_of_speech_timeout": pace.end_of_speech_timeout,
            "attention_timeout": pace.attention_timeout,
            "outbound_attention_timeout": 60000,      # nobody greets within 60 s -> agent times out
            "inactivity_timeout": 45000,              # 45 s of no activity -> exit (cost guard)
            "hard_stop_time": _stop_time(spec["max_duration_s"]),
            "hard_stop_prompt": "Time is up. Say exactly: 'Have a good day, goodbye.' and stop.",
        },
        "SWAIG": {
            "functions": [{
                "function": END_FUNCTION,
                "description": "Hang up the call. Use it right after your final goodbye line.",
                "parameters": {"type": "object", "properties": {}},
                "data_map": {"output": {"response": "Hanging up.", "action": [{"hangup": True}]}},
            }],
        },
    }
    return {
        "version": SWML_VERSION,
        "sections": {"main": [
            {"play": {"url": f"silence:{pace.lead_silence_s}"}},
            {"ai": ai},
            {"hangup": {}},
        ]},
    }


def describe(persona: str, variant: int, fill: int) -> dict[str, Any]:
    """Labels for the ledger (ground truth for V1 'AI' and the guardian live-set scenario)."""
    spec = _specs()[persona]
    var = VARIANTS[persona][variant]
    voice = VOICES[var.voice]
    values = spec["variants"][fill]
    return {
        "persona": persona,
        "kind": spec["kind"],                   # scam | benign
        "variant": variant,
        "fill": fill,
        "voice": voice.swml,
        "voice_label": voice.label,
        "engine": voice.engine,
        "voice_gender": voice.gender,
        "tts_tier": voice.tier,
        "pace": var.pace,
        "opening": var.opening,
        "opening_text": render(opening_template(persona, spec, var.opening), values),
        "ai_model": AI_MODEL,
    }


def swml_digest(doc: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()[:16]


# ------------------------------------------------------------------ structural validation
# Ranges: the intersection of the docs (https://signalwire.com/docs/swml/reference/calling/ai/params) and
# the SDK's SWML JSON schema (signalwire-sdk 3.4.1 schema.json), which is stricter on two of them.
_TOP_PARAMS = {
    "end_of_speech_timeout": (250, 10000),        # docs 0-10000, SDK schema 250-10000
    "attention_timeout": (10000, 600000),         # or 0; docs 1000-600000, SDK schema 10000-600000
    "outbound_attention_timeout": (10000, 600000),
    "inactivity_timeout": (10000, 3600000),       # or 0
}
_KNOWN_PARAMS = set(_TOP_PARAMS) | {"ai_model", "wait_for_user", "hard_stop_time", "hard_stop_prompt"}
_KNOWN_METHODS = {"play", "ai", "hangup", "answer"}
_ACTIONS = {"SWML", "say", "stop", "hangup"}


def validate_swml(doc: dict[str, Any]) -> list[str]:
    """Structural check against the documented SWML / ai shape. Returns a list of problems."""
    errs: list[str] = []
    if doc.get("version") != SWML_VERSION:
        errs.append("version must be '1.0.0'")
    main = (doc.get("sections") or {}).get("main")
    if not isinstance(main, list) or not main:
        return errs + ["sections.main must be a non-empty list"]
    ai_steps = 0
    for i, step in enumerate(main):
        if not isinstance(step, dict) or len(step) != 1:
            errs.append(f"main[{i}] must be an object with exactly one method")
            continue
        (method, body), = step.items()
        if method not in _KNOWN_METHODS:
            errs.append(f"main[{i}]: unexpected method {method!r}")
        if not isinstance(body, dict):
            errs.append(f"main[{i}].{method} must be an object")
            continue
        if method == "play":
            url = body.get("url", "")
            m = re.fullmatch(r"silence:(\d+(?:\.\d+)?)", url)
            if not m or float(m.group(1)) < 1:
                errs.append(f"main[{i}].play.url must be 'silence:<seconds >= 1>'")
        if method == "ai":
            ai_steps += 1
            errs += [f"ai: {e}" for e in _validate_ai(body)]
    if ai_steps != 1:
        errs.append("exactly one ai step expected")
    if list(main[-1]) != ["hangup"]:
        errs.append("last step must be hangup")
    return errs


def _validate_ai(ai: dict[str, Any]) -> list[str]:
    errs: list[str] = []
    prompt = ai.get("prompt") or {}
    if not isinstance(prompt.get("text"), str) or not prompt["text"].strip():
        errs.append("prompt.text required")
    elif "{{" in prompt["text"]:
        errs.append("prompt.text has unrendered {{vars}}")
    t = prompt.get("temperature", 1.0)
    if not 0 <= t <= 2:
        errs.append("prompt.temperature out of range")
    langs = ai.get("languages")
    if not isinstance(langs, list) or not langs:
        errs.append("languages must be a non-empty list")
    else:
        for j, lang in enumerate(langs):
            for key in ("name", "code", "voice"):
                if not lang.get(key):
                    errs.append(f"languages[{j}].{key} required")
            try:
                check_voice(lang.get("voice", ""))
            except ValueError as e:
                errs.append(f"languages[{j}]: {e}")
    if "multilingual" in ai:
        errs.append("multilingual overrides languages; not used")
    params = ai.get("params") or {}
    unknown = set(params) - _KNOWN_PARAMS
    if unknown:
        errs.append(f"unexpected params {sorted(unknown)}")
    for key, (lo, hi) in _TOP_PARAMS.items():
        if key in params:
            v = params[key]
            if not isinstance(v, int) or not (lo <= v <= hi or (v == 0 and key in ("attention_timeout", "inactivity_timeout"))):
                errs.append(f"params.{key}={v!r} outside {lo}-{hi}")
    hst = params.get("hard_stop_time")
    if hst is not None and not re.fullmatch(r"\d+s", str(hst)):
        errs.append("params.hard_stop_time must be '<seconds>s' (valid under both the docs and the SDK schema)")
    if params.get("wait_for_user") is not True:
        errs.append("params.wait_for_user must be true (the line plays a notice first)")
    funcs = (ai.get("SWAIG") or {}).get("functions", [])
    for f in funcs:
        if not f.get("function") or not f.get("description"):
            errs.append("SWAIG function needs function + description")
        out = (f.get("data_map") or {}).get("output") or {}
        acts = out.get("action", [])
        acts = acts if isinstance(acts, list) else [acts]
        for a in acts:
            bad = set(a) - _ACTIONS
            if bad:
                errs.append(f"SWAIG {f.get('function')}: unexpected actions {sorted(bad)}")
    if not any(f.get("function") == END_FUNCTION for f in funcs):
        errs.append(f"SWAIG function {END_FUNCTION} missing (agent could not hang up)")
    return errs


def catalog() -> list[dict[str, Any]]:
    """Every persona x variant, for README tables and the router index."""
    out = []
    for persona, vs in VARIANTS.items():
        for i, var in enumerate(vs):
            voice = VOICES[var.voice]
            out.append({"persona": persona, "variant": i, "voice": voice.swml, "voice_label": voice.label,
                        "engine": voice.engine, "gender": voice.gender, "tier": voice.tier, **asdict(var)})
    return out


if __name__ == "__main__":
    for row in catalog():
        print(json.dumps(row))

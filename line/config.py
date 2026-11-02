"""Line settings (L1): read from the environment, never hard-coded.

Sources, first one wins for each variable:
1. the process environment;
2. the repo's .env (FIREWORKS_API_KEY and friends live there);
3. TELNYX_ENV_FILE, if set: only TELNYX_API_KEY and TELNYX_PUBLIC_KEY are taken from it.
   That file is EmberDial's .env, whose TELNYX_CONNECTION_ID points at a production app,
   so the connection id is deliberately NOT read from it.

TELNYX_CONNECTION_ID defaults to the demo Call Control app ("EmberDial Jev Demo").
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from dotenv import dotenv_values, load_dotenv
from pydantic import BaseModel, ConfigDict, Field

from contracts.models import Profile

REPO = Path(__file__).resolve().parent.parent
DEMO_APP_ID = "3051333815776576529"
DEMO_APP_NAME = "EmberDial Jev Demo"
DEFAULT_PROFILES = REPO / "data" / "profiles.json"  # data/ is gitignored: phone numbers stay out of git
NOTICE_TEXT = "This call is protected by Emberz and may be transcribed."

E164 = re.compile(r"^\+[1-9]\d{6,14}$")


def is_e164(value: Optional[str]) -> bool:
    return bool(value) and bool(E164.match(value))


def normalize_number(value: Optional[str]) -> str:
    """'+13235551234' stays; '13235551234' (digits only, 11-15 long) gets its '+'. Anything else
    (anonymous, restricted, SIP URIs, 10-digit numbers without a country code) is returned unchanged."""
    v = (value or "").strip()
    if v.isdigit() and 11 <= len(v) <= 15 and v[0] != "0":
        return "+" + v
    return v


class GuardedLine(BaseModel):
    """One guarded number's setup (an entry in profiles.json)."""

    model_config = ConfigDict(extra="forbid")

    profile: Profile
    rules: list[str] = Field(default_factory=list)
    allowed_contacts: dict[str, str] = Field(default_factory=dict)  # E.164 -> name
    protected_number: str
    caretaker_number: str = ""


@dataclass(frozen=True)
class Settings:
    telnyx_api_key: str = ""
    telnyx_public_key: str = ""
    telnyx_connection_id: str = DEMO_APP_ID
    public_base_url: str = ""
    profiles_path: Path = DEFAULT_PROFILES
    skip_verify: bool = False
    tts_voice: str = "female"          # docs example voice; see line/call_flow.py for the source
    tts_language: str = "en-US"
    notice_text: str = NOTICE_TEXT
    deepfake: bool = True              # EMBERZ_DEEPFAKE=0 turns Telnyx deepfake_detection off (hour-0 check 1)
    media_verify: str = "warn"         # EMBERZ_MEDIA_VERIFY: warn (log only) | strict (reject) | off
    max_call_s: int = 1800             # time_limit_secs on the B leg: a cost guard if the server dies mid-call
    allow_other_app: bool = False      # EMBERZ_ALLOW_OTHER_APP=1 lets TELNYX_CONNECTION_ID differ from the demo app

    def media_ws_url(self, call_id: str) -> str:
        """wss://<PUBLIC_BASE_URL host>/media/{call_id}"""
        if not self.public_base_url:
            raise RuntimeError("PUBLIC_BASE_URL is not set (start line/tunnel.sh and export it)")
        u = urlparse(self.public_base_url)
        scheme = "ws" if u.scheme in ("http", "ws") else "wss"
        return f"{scheme}://{u.netloc}/media/{call_id}"

    def webhook_url(self) -> str:
        return self.public_base_url.rstrip("/") + "/webhooks/telnyx"

    def is_public(self) -> bool:
        """True when PUBLIC_BASE_URL points at a host the internet can reach (not localhost)."""
        host = urlparse(self.public_base_url).hostname or ""
        return bool(host) and host not in ("localhost", "127.0.0.1", "::1", "testserver")


def _load_env_files() -> None:
    load_dotenv(REPO / ".env", override=False)
    telnyx_env = os.environ.get("TELNYX_ENV_FILE")
    if telnyx_env:
        values = dotenv_values(telnyx_env)
        for key in ("TELNYX_API_KEY", "TELNYX_PUBLIC_KEY"):
            if values.get(key) and not os.environ.get(key):
                os.environ[key] = values[key]  # in-process only; never printed or written


def load_settings(load_files: bool = True) -> Settings:
    if load_files:
        _load_env_files()
    env = os.environ
    return Settings(
        telnyx_api_key=env.get("TELNYX_API_KEY", ""),
        telnyx_public_key=env.get("TELNYX_PUBLIC_KEY", ""),
        telnyx_connection_id=env.get("TELNYX_CONNECTION_ID") or DEMO_APP_ID,
        public_base_url=env.get("PUBLIC_BASE_URL", ""),
        profiles_path=Path(env.get("EMBERZ_PROFILES", str(DEFAULT_PROFILES))),
        skip_verify=env.get("EMBERZ_SKIP_VERIFY") == "1",
        tts_voice=env.get("EMBERZ_TTS_VOICE", "female"),
        tts_language=env.get("EMBERZ_TTS_LANGUAGE", "en-US"),
        deepfake=env.get("EMBERZ_DEEPFAKE", "1") != "0",
        media_verify=env.get("EMBERZ_MEDIA_VERIFY", "warn"),
        max_call_s=int(env.get("EMBERZ_MAX_CALL_S", "1800")),
        allow_other_app=env.get("EMBERZ_ALLOW_OTHER_APP") == "1",
    )


def load_profiles(path: Path) -> dict[str, GuardedLine]:
    """profiles.json: {guarded E.164 number: GuardedLine}. Keys starting with '_' are comments."""
    raw = json.loads(Path(path).read_text())
    out: dict[str, GuardedLine] = {}
    for number, entry in raw.items():
        if number.startswith("_"):
            continue
        if not is_e164(number):
            raise ValueError(f"profiles: guarded number key must be E.164, got {number!r}")
        line = GuardedLine.model_validate(entry)
        for n in [line.protected_number, line.caretaker_number, *line.allowed_contacts]:
            if n and not is_e164(n):
                raise ValueError(f"profiles[{number}]: {n!r} is not E.164")
        out[number] = line
    return out


def profiles_from_env() -> dict[str, GuardedLine]:
    """One guarded line from TELNYX_GUARDED_NUMBER / TELNYX_PROTECTED_NUMBER / TELNYX_CARETAKER_NUMBER
    (+ EMBERZ_PROFILE, default elder). Used when no profiles file exists."""
    env = os.environ
    guarded, protected = env.get("TELNYX_GUARDED_NUMBER", ""), env.get("TELNYX_PROTECTED_NUMBER", "")
    if not (is_e164(guarded) and is_e164(protected)):
        return {}
    caretaker = env.get("TELNYX_CARETAKER_NUMBER", "")
    return {guarded: GuardedLine(profile=env.get("EMBERZ_PROFILE", "elder"), protected_number=protected,
                                 caretaker_number=caretaker if is_e164(caretaker) else "")}


def load_profiles_or_env(settings: Settings) -> dict[str, GuardedLine]:
    if settings.profiles_path.exists():
        return load_profiles(settings.profiles_path)
    profiles = profiles_from_env()
    if not profiles:
        raise RuntimeError(f"{settings.profiles_path} not found and TELNYX_GUARDED_NUMBER/TELNYX_PROTECTED_NUMBER "
                           "are not set: copy line/profiles.example.json there (or set EMBERZ_PROFILES)")
    return profiles

import json

import pytest

from line.config import DEMO_APP_ID, REPO, Settings, load_profiles, load_profiles_or_env, profiles_from_env


def test_example_profiles_parse():
    profiles = load_profiles(REPO / "line" / "profiles.example.json")
    line = profiles["+15555550100"]
    assert line.profile == "elder" and line.allowed_contacts["+15555550111"] == "Grandpa"
    assert profiles["+15555550120"].profile == "kid"


def test_bad_number_rejected(tmp_path):
    f = tmp_path / "p.json"
    f.write_text(json.dumps({"+15555550100": {"profile": "kid", "protected_number": "555-0101"}}))
    with pytest.raises(ValueError):
        load_profiles(f)


def test_media_ws_url():
    s = Settings(public_base_url="https://abc-def.trycloudflare.com/")
    assert s.media_ws_url("c_1") == "wss://abc-def.trycloudflare.com/media/c_1"
    assert Settings(public_base_url="http://localhost:8000").media_ws_url("c_1") == "ws://localhost:8000/media/c_1"
    assert s.webhook_url() == "https://abc-def.trycloudflare.com/webhooks/telnyx"
    assert Settings().telnyx_connection_id == DEMO_APP_ID
    with pytest.raises(RuntimeError):
        Settings().media_ws_url("c_1")


def test_profiles_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("TELNYX_GUARDED_NUMBER", "+15555550100")
    monkeypatch.setenv("TELNYX_PROTECTED_NUMBER", "+15555550101")
    monkeypatch.setenv("TELNYX_CARETAKER_NUMBER", "+15555550102")
    monkeypatch.setenv("EMBERZ_PROFILE", "kid")
    p = profiles_from_env()["+15555550100"]
    assert (p.profile, p.protected_number, p.caretaker_number) == ("kid", "+15555550101", "+15555550102")
    assert load_profiles_or_env(Settings(profiles_path=tmp_path / "missing.json")) == profiles_from_env()
    monkeypatch.delenv("TELNYX_PROTECTED_NUMBER")
    with pytest.raises(RuntimeError):
        load_profiles_or_env(Settings(profiles_path=tmp_path / "missing.json"))

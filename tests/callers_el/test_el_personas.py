"""ElevenLabs gets the two benign bots only (PLAN.md 2c). Every entry point must refuse the rest."""
import json

import pytest

from callers.elevenlabs import agents as A
from callers.elevenlabs import place_calls as P

SCAMS = ["bank_fraud", "tech_support", "prize_giftcard"]


@pytest.fixture(autouse=True)
def no_network(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise AssertionError("must not touch the network")
    import httpx
    monkeypatch.setattr(httpx, "Client", boom)
    monkeypatch.setattr(httpx, "AsyncClient", boom)
    monkeypatch.setattr(P, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(P, "LEDGER_PATH", tmp_path / "ai_calls.jsonl")


def test_allowlist_is_exactly_the_two_benign_bots():
    assert set(A.ALLOWED_PERSONAS) == {"pharmacy_refill", "appointment_confirm"}
    assert set(A.VOICES) == set(A.ALLOWED_PERSONAS)


@pytest.mark.parametrize("persona", SCAMS + ["grooming", "", "Pharmacy_Refill", "pharmacy_refill "])
def test_every_builder_refuses_other_personas(persona):
    for fn in (lambda: A.check_persona(persona), lambda: A.load_spec(persona),
               lambda: A.agent_payload(persona, 0), lambda: A.sip_headers(persona, 0),
               lambda: A.voice_index(persona, 0), lambda: A.agent_name(persona, 0),
               lambda: P.dial_payload(persona, 0, "+13046481098", "+13235161282", "x"),
               lambda: P.ledger_row("c1", persona, 0, "2026-11-02T00:00:00+00:00")):
        with pytest.raises(A.PersonaNotAllowed):
            fn()


def test_scam_spec_file_is_never_read(monkeypatch):
    read = []
    real = A.Path.read_text
    monkeypatch.setattr(A.Path, "read_text", lambda self, *a, **k: (read.append(self.name), real(self, *a, **k))[1])
    for p in SCAMS:
        with pytest.raises(A.PersonaNotAllowed):
            A.load_spec(p)
    assert not any(n.startswith(tuple(SCAMS)) for n in read)


def test_allowlisted_id_with_a_scam_spec_is_refused(monkeypatch, tmp_path):
    fake = tmp_path / "agents"
    fake.mkdir()
    spec = json.loads((A.SPECS_DIR / "pharmacy_refill.json").read_text())
    spec["kind"] = "scam"
    (fake / "pharmacy_refill.json").write_text(json.dumps(spec))
    monkeypatch.setattr(A, "SPECS_DIR", fake)
    with pytest.raises(A.PersonaNotAllowed):
        A.load_spec("pharmacy_refill")
    with pytest.raises(A.PersonaNotAllowed):
        A.agent_payload("pharmacy_refill", 0)


def test_spec_with_mismatched_id_is_refused():
    with pytest.raises(A.PersonaNotAllowed):
        A.check_persona("pharmacy_refill", {"id": "bank_fraud", "kind": "benign"})


@pytest.mark.parametrize("argv", [
    ["call", "--personas", "bank_fraud"],
    ["call", "--personas", "pharmacy_refill,tech_support"],
    ["call", "--personas", "prize_giftcard", "--live"],
])
def test_cli_refuses_scam_personas_even_with_live(argv, capsys):
    assert P.main(argv) == 2
    assert "not allowed on ElevenLabs" in capsys.readouterr().err
    assert not P.LEDGER_PATH.exists() and not P.STATE_PATH.exists()

import base64
import time

import pytest
from nacl.signing import SigningKey

from callflow_helpers import CALLER, Line
from line.webhook_verify import WebhookVerificationError, decode_public_key, verify

BODY = b'{"data":{"event_type":"call.initiated","id":"e1","payload":{}}}'


def _sign(key: SigningKey, body: bytes, ts: int) -> dict:
    sig = key.sign(str(ts).encode() + b"|" + body).signature
    return {"Telnyx-Signature-Ed25519": base64.b64encode(sig).decode(), "Telnyx-Timestamp": str(ts)}


def _pub(key: SigningKey) -> str:
    return base64.b64encode(bytes(key.verify_key)).decode()


def test_valid_signature_passes(signing_key):
    ts = int(time.time())
    verify(BODY, _sign(signing_key, BODY, ts), _pub(signing_key))  # header names are case-insensitive


def test_tampered_body_fails(signing_key):
    ts = int(time.time())
    with pytest.raises(WebhookVerificationError, match="bad signature"):
        verify(BODY.replace(b"e1", b"e2"), _sign(signing_key, BODY, ts), _pub(signing_key))


def test_wrong_key_fails(signing_key):
    ts = int(time.time())
    with pytest.raises(WebhookVerificationError):
        verify(BODY, _sign(signing_key, BODY, ts), _pub(SigningKey.generate()))


def test_stale_and_future_timestamps_fail(signing_key):
    now = 1_800_000_000
    for ts in (now - 301, now + 301):
        with pytest.raises(WebhookVerificationError, match="stale"):
            verify(BODY, _sign(signing_key, BODY, ts), _pub(signing_key), now=now)
    verify(BODY, _sign(signing_key, BODY, now - 299), _pub(signing_key), now=now)


def test_missing_headers_and_key_fail(signing_key):
    with pytest.raises(WebhookVerificationError):
        verify(BODY, {}, _pub(signing_key))
    with pytest.raises(WebhookVerificationError, match="not set"):
        verify(BODY, _sign(signing_key, BODY, int(time.time())), "")


def test_der_spki_public_key_accepted(signing_key):
    der = bytes.fromhex("302a300506032b6570032100") + bytes(signing_key.verify_key)
    der_b64 = base64.b64encode(der).decode()
    assert der_b64.startswith("MCo")  # the portal format EmberDial's runbook describes
    assert bytes(decode_public_key(der_b64)) == bytes(signing_key.verify_key)
    verify(BODY, _sign(signing_key, BODY, int(time.time())), der_b64)


def test_server_rejects_unsigned_and_accepts_signed(signing_key):
    line = Line(signing_key=signing_key, skip_verify=False)
    assert line.inbound().status_code == 200
    assert line.calls("call", "answer")
    # Same webhook signed by someone else: 403, nothing happens.
    raw = line.body("call.initiated", {"call_control_id": "x", "from": CALLER, "to": "+1", "direction": "incoming"})
    bad = _sign(SigningKey.generate(), raw, int(time.time()))
    assert line.client.post("/webhooks/telnyx", content=raw, headers=bad).status_code == 403
    assert line.client.post("/webhooks/telnyx", content=raw).status_code == 403
    # Correctly signed but replayed 10 minutes later: 403.
    assert line.post("call.answered", line.leg("A", "v3:A-leg"), ts=int(time.time()) - 600).status_code == 403
    assert not line.calls("call", "streaming_start")


def test_server_without_public_key_fails_closed():
    line = Line(skip_verify=False)  # no key configured
    assert line.inbound().status_code == 403
    assert line.telnyx.log == []
    assert line.client.get("/healthz").json()["verify"] == "NO KEY"

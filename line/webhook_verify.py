"""Telnyx webhook signature check (Ed25519).

Scheme, per Telnyx docs and the official Python SDK (telnyx/lib/webhook_verification.py):
- header `telnyx-signature-ed25519`: base64 Ed25519 signature (64 bytes)
- header `telnyx-timestamp`: unix seconds when the webhook was signed
- signed message: f"{timestamp}|{raw_body}" over the RAW body bytes (do not re-serialize JSON)
- reject when |now - timestamp| > 300 s (replay window)
Sources:
  https://developers.telnyx.com/docs/messaging/messages/receiving-webhooks
  https://developers.telnyx.com/docs/development/api-fundamentals/webhooks/receiving-webhooks
  https://developers.telnyx.com/docs/voice/programmable-voice/voice-api-webhooks (header list)

Public key: base64 from Mission Control (portal.telnyx.com/#/api-keys/public-key). The SDK
expects 32 raw bytes; EmberDial's runbook says the portal value may start with "MCo"
(a 44-byte DER SubjectPublicKeyInfo). Both forms are accepted here.

EMBERZ_SKIP_VERIFY=1 bypasses the check for local tests only and logs a loud warning.
"""
from __future__ import annotations

import base64
import binascii
import logging
import time
from typing import Mapping, Optional

from nacl.exceptions import BadSignatureError
from nacl.signing import VerifyKey

log = logging.getLogger("emberz.webhook_verify")

SIGNATURE_HEADER = "telnyx-signature-ed25519"
TIMESTAMP_HEADER = "telnyx-timestamp"
TOLERANCE_S = 300
_SPKI_ED25519_PREFIX = bytes.fromhex("302a300506032b6570032100")  # DER header before the 32 key bytes


class WebhookVerificationError(Exception):
    pass


def decode_public_key(public_key_b64: str) -> VerifyKey:
    try:
        raw = base64.b64decode(public_key_b64.strip(), validate=True)
    except (binascii.Error, ValueError) as e:
        raise WebhookVerificationError("public key is not valid base64") from e
    if len(raw) == 44 and raw.startswith(_SPKI_ED25519_PREFIX):
        raw = raw[len(_SPKI_ED25519_PREFIX):]
    if len(raw) != 32:
        raise WebhookVerificationError(f"public key must be 32 raw bytes (or DER SPKI), got {len(raw)}")
    return VerifyKey(raw)


def _header(headers: Mapping[str, str], name: str) -> Optional[str]:
    for k, v in headers.items():
        if k.lower() == name:
            return v
    return None


def verify(
    raw_body: bytes,
    headers: Mapping[str, str],
    public_key_b64: str,
    tolerance_s: int = TOLERANCE_S,
    now: Optional[float] = None,
) -> None:
    """Raise WebhookVerificationError unless the webhook is authentic and fresh."""
    _verify_signed(raw_body, headers, public_key_b64, tolerance_s, now)


def verify_media_handshake(
    stream_url: str,
    headers: Mapping[str, str],
    public_key_b64: str,
    tolerance_s: int = TOLERANCE_S,
    now: Optional[float] = None,
) -> None:
    """Media-stream WebSocket handshake: same headers, signed string f"{timestamp}|{stream_url}" where
    stream_url is exactly the URL passed to streaming_start (query string included).
    Source: https://developers.telnyx.com/docs/voice/programmable-voice/media-streaming ("Telnyx signs
    every media stream WebSocket connection"). Not yet seen live: hour0 records whether it passes."""
    _verify_signed(stream_url.encode(), headers, public_key_b64, tolerance_s, now)


def _verify_signed(
    raw_body: bytes,
    headers: Mapping[str, str],
    public_key_b64: str,
    tolerance_s: int,
    now: Optional[float],
) -> None:
    if not public_key_b64:
        raise WebhookVerificationError("TELNYX_PUBLIC_KEY is not set (fail closed)")
    sig_b64 = _header(headers, SIGNATURE_HEADER)
    ts = _header(headers, TIMESTAMP_HEADER)
    if not sig_b64 or not ts:
        raise WebhookVerificationError("missing signature or timestamp header")
    try:
        ts_int = int(ts)
    except ValueError as e:
        raise WebhookVerificationError("timestamp is not an integer") from e
    now = time.time() if now is None else now
    if abs(now - ts_int) > tolerance_s:
        raise WebhookVerificationError(f"stale timestamp ({int(now - ts_int)} s off)")
    try:
        sig = base64.b64decode(sig_b64, validate=True)
    except (binascii.Error, ValueError) as e:
        raise WebhookVerificationError("signature is not valid base64") from e
    if len(sig) != 64:
        raise WebhookVerificationError("signature must be 64 bytes")
    key = decode_public_key(public_key_b64)
    try:
        key.verify(ts.encode() + b"|" + raw_body, sig)
    except BadSignatureError as e:
        raise WebhookVerificationError("bad signature") from e


def warn_skip_verify() -> None:
    log.warning("!" * 72)
    log.warning("EMBERZ_SKIP_VERIFY=1: Telnyx webhook signatures are NOT checked. Local tests only.")
    log.warning("!" * 72)

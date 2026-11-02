"""Telnyx deepfake detection: request params and webhook -> VoiceSignal(model="telnyx").

Docs (checked 2026-11-02):
  https://developers.telnyx.com/docs/voice/programmable-voice/deepfake-detection
  https://telnyx.com/release-notes/deepfake-detection

- Enable with a `deepfake_detection` object on Dial (POST /v2/calls) or Answer
  (POST /v2/calls/{call_control_id}/actions/answer):
  {"enabled": true, "timeout": 15, "rtp_timeout": 30}
  timeout: seconds to wait for a result (default 15, 5-60); rtp_timeout: seconds to wait
  for RTP audio (default 30, 5-120). $0.01 per call.
- Webhooks:
  `call.deepfake_detection.result`: payload.result in real | fake | silence_timeout,
      payload.score: 0.0 (likely real) .. 1.0 (likely deepfake), null on silence_timeout,
      payload.consistency: 0..100 (% agreement across frames), null on silence_timeout.
  `call.deepfake_detection.error`: payload.error_message in detection_timeout |
      rtp_timeout | dfd_connection_error | dfd_stream_error.
  Plus the usual payload.call_control_id / call_leg_id / call_session_id / client_state.
- The docs page shows the event without the `data` wrapper; real Call Control webhooks
  arrive as {"data": {"event_type": ..., "payload": {...}}}. Both shapes are accepted.
- Docs guidance: consistency > 90 is high confidence; a high score with low consistency
  calls for more checks rather than a hang-up. The docs' sample code acts on
  result == "fake" and score > 0.8 (an example, not a recommended threshold).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from contracts.models import VoiceSignal

RESULT_EVENT = "call.deepfake_detection.result"
ERROR_EVENT = "call.deepfake_detection.error"


def answer_params(timeout: int = 15, rtp_timeout: int = 30) -> dict:
    """Body fragment for answer/dial: `client.call_action(ccid, "answer", **answer_params())`."""
    return {"deepfake_detection": {"enabled": True, "timeout": timeout, "rtp_timeout": rtp_timeout}}


@dataclass
class DeepfakeResult:
    event_type: str
    call_control_id: Optional[str]
    call_session_id: Optional[str]
    result: Optional[str] = None        # real | fake | silence_timeout
    score: Optional[float] = None       # P(deepfake) per Telnyx
    consistency: Optional[float] = None  # 0..100
    error: Optional[str] = None
    occurred_at: Optional[str] = None


def parse_event(body: dict) -> Optional[DeepfakeResult]:
    """Webhook JSON -> DeepfakeResult, or None if it is not a deepfake-detection event."""
    if not isinstance(body, dict):
        return None
    ev = body.get("data", body)
    if not isinstance(ev, dict):
        return None
    et = ev.get("event_type")
    if et not in (RESULT_EVENT, ERROR_EVENT):
        return None
    p = ev.get("payload")
    p = p if isinstance(p, dict) else {}

    def num(v):  # malformed or non-finite numbers count as missing, never raise
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return f if math.isfinite(f) else None

    return DeepfakeResult(
        event_type=et,
        call_control_id=p.get("call_control_id"),
        call_session_id=p.get("call_session_id"),
        result=p.get("result"),
        score=num(p.get("score")),
        consistency=num(p.get("consistency")),
        error=p.get("error_message"),
        occurred_at=ev.get("occurred_at"),
    )


def to_signal(r: DeepfakeResult, call_id: str, at_ms: int) -> Optional[VoiceSignal]:
    """Only a result with a score becomes a signal; errors and silence_timeout give None."""
    if r.event_type != RESULT_EVENT or r.score is None or r.result == "silence_timeout":
        return None
    return VoiceSignal(call_id=call_id, at_ms=max(int(at_ms), 0), ai_score=min(max(r.score, 0.0), 1.0), model="telnyx")


def parse_webhook(body: dict, call_id: Optional[str] = None, at_ms: int = 0) -> Optional[VoiceSignal]:
    """One-step helper for the line server. call_id defaults to "c_" + call_session_id.

    Stateless: Telnyx may re-deliver a webhook, so the caller dedupes (e.g. by call and
    event type) if a repeated signal matters; a repeat carries the same score.
    """
    r = parse_event(body)
    if r is None:
        return None
    cid = call_id or (f"c_{r.call_session_id}" if r.call_session_id else (r.call_control_id or "unknown"))
    return to_signal(r, cid, at_ms)

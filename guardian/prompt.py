"""Bridge from the live line to the guardian's FROZEN training prompt.

line/guardian_client.py imports `guardian.prompt.render_messages(req)` when this module exists.
This module forwards to guardian/render.py, an exact copy of callguard/guardian/render.py, the prompt
the fine-tuned guardian was trained on (prompt_sha 806908cfba7d). Do not edit render.py: any change to
SYSTEM or render_user changes PROMPT_SHA and means the model sees a prompt it was not trained on.
"""
from guardian.render import PROMPT_SHA, render_messages as _render

EXPECTED_PROMPT_SHA = "806908cfba7d"
if PROMPT_SHA != EXPECTED_PROMPT_SHA:  # pragma: no cover - guards against accidental edits
    raise RuntimeError(f"guardian prompt drifted: {PROMPT_SHA} != {EXPECTED_PROMPT_SHA}")


def _as_dict(req):
    if hasattr(req, "model_dump"):
        d = req.model_dump()
    else:
        d = dict(req)
    # Training used the guardian's own verdict just before the 8-turn window ("Earlier in this call").
    # GuardianRequest has no such field yet; pass it as an attribute or dict key if the brain tracks it.
    d.setdefault("earlier", getattr(req, "earlier", None) if not isinstance(req, dict) else req.get("earlier"))
    return d


def render_messages(req) -> list[dict]:
    """GuardianRequest (or dict) -> [system, user] messages, byte-identical to training."""
    return _render(_as_dict(req))

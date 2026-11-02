"""Guardian client for the live line (L2).

build_request(session) -> GuardianRequest from the CallSession (last `window` turns).
score(req) -> (GuardianResponse, latency_ms), backend picked by env GUARDIAN_BACKEND:

  stub        contracts.stub_guardian.score (no network; the default)
  standin     a cheap Fireworks serverless chat model (GUARDIAN_STANDIN_MODEL,
              default STANDIN_MODEL below) with response_format = GuardianResponse's
              JSON schema
  deployment  same code, model = GUARDIAN_MODEL (the LoRA deployment, M2)
  matt        line/matt_bridge.MattGuardianClient: Matt's callguard/guardian/render.py prompt
              and parser, model = GUARDIAN_MODEL, key GUARDIAN_API_KEY or FIREWORKS_API_KEY

Prompt: guardian.prompt.render_messages (Matt's M0 module) when it imports,
otherwise line/standin_prompt.py.

Failure handling: strict overall timeout (GUARDIAN_TIMEOUT_S, default 3 s), a hedged
duplicate request after GUARDIAN_HEDGE_MS (default 1000 ms), one retry on 503
(a scaled-to-zero deployment answers 503 while waking), and any HTTP error, timeout or
invalid JSON returns SAFE_FALLBACK (risk none) with a logged error, so the call never
stalls on the model.

LatestOnlyScorer: per call, at most one score in flight. Turns that arrive while a
score is running are not queued; when it finishes, only the newest window is scored.

Fireworks docs:
  chat completions  https://docs.fireworks.ai/api-reference/post-chatcompletions
  response_format   https://docs.fireworks.ai/structured-responses/structured-response-formatting
  prices            https://docs.fireworks.ai/serverless/pricing
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Awaitable, Callable, Optional

import httpx
from pydantic import ValidationError

from contracts import stub_guardian
from contracts.models import GuardianRequest, GuardianResponse, WindowTurn
from line.session import CallSession

log = logging.getLogger("emberz.guardian")

FIREWORKS_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
# Picked 2026-11-02: cheapest serverless chat model that answers JSON-schema output with
# thinking off ($0.05 in / $0.20 out per 1M tokens, docs.fireworks.ai/serverless/pricing).
# Measured on fixture windows: p50 ~0.5 s, but ~1 in 3 requests stalled past 4 s (hence the hedge).
# GLM 5.3 Flash rejects reasoning_effort=none (thinking-only); gpt-oss-120b needs "low", ~1.5 s.
STANDIN_MODEL = "accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b"
DEFAULT_WINDOW = 8
SAFE_FALLBACK = GuardianResponse(
    risk="none", category="none", severity=0, source="caller", stage="hook", action="none"
)


# ---------------------------------------------------------------- request
def build_request(session: CallSession, window: int = DEFAULT_WINDOW) -> GuardianRequest:
    turns = session.turns[-window:] if window > 0 else list(session.turns)
    return GuardianRequest(
        call_id=session.call_id,
        profile=session.profile,
        rules=list(session.rules),
        caller_known=session.caller_known,
        ai_voice=session.ai_voice,
        turns=[WindowTurn(turn=t.turn, speaker=t.speaker, text=t.text) for t in turns],
    )


_prompt_fn: Optional[Callable[[GuardianRequest], list[dict]]] = None


def _resolve_prompt() -> Callable[[GuardianRequest], list[dict]]:
    """guardian.prompt.render_messages (M0) if the module exists, else the stand-in.

    Only a missing `guardian` / `guardian.prompt` module falls back. If the module exists
    but fails to import (a bug or a missing dependency inside it), that error is raised:
    silently scoring the live line with the wrong prompt would be worse.
    """
    global _prompt_fn
    if _prompt_fn is None:
        try:
            from guardian.prompt import render_messages as shared  # M0, may not exist yet
            log.info("guardian prompt: guardian.prompt.render_messages")
        except ModuleNotFoundError as e:
            if e.name not in ("guardian", "guardian.prompt"):
                raise
            from line.standin_prompt import PROMPT_SHA
            from line.standin_prompt import render_messages as shared
            log.info("guardian prompt: line/standin_prompt.py (sha %s)", PROMPT_SHA)
        _prompt_fn = shared
    return _prompt_fn


def render_messages(req: GuardianRequest) -> list[dict]:
    """Matt's frozen prompt if it exists, else the stand-in."""
    return _resolve_prompt()(req)


def response_format() -> dict:
    return {
        "type": "json_schema",
        "json_schema": {"name": "GuardianResponse", "schema": GuardianResponse.model_json_schema()},
    }


def parse_response(content: str) -> GuardianResponse:
    """Parse model output; tolerate a ```json fence. Raises ValueError on bad output."""
    text = (content or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[4:] if text.lower().startswith("json") else text
    try:
        return GuardianResponse.model_validate_json(text.strip())
    except ValidationError as e:
        raise ValueError(f"invalid guardian JSON: {e.errors()[:3]}") from e


# ---------------------------------------------------------------- client
def _fireworks_key() -> str:
    key = os.environ.get("FIREWORKS_API_KEY")
    if not key:
        from dotenv import dotenv_values

        env_file = os.environ.get("FIREWORKS_ENV_FILE") or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
        )
        key = dotenv_values(env_file).get("FIREWORKS_API_KEY")
    if not key:
        raise RuntimeError("FIREWORKS_API_KEY not set (env or .env)")
    return key


class GuardianClient:
    def __init__(
        self,
        backend: Optional[str] = None,
        model: Optional[str] = None,
        timeout_s: Optional[float] = None,
        max_tokens: int = 400,
    ):
        self.backend = backend or os.environ.get("GUARDIAN_BACKEND", "stub")
        if self.backend not in ("stub", "standin", "deployment", "matt"):
            raise ValueError(f"GUARDIAN_BACKEND must be stub|standin|deployment|matt, got {self.backend!r}")
        if model:
            self.model = model
        elif self.backend == "deployment":
            self.model = os.environ["GUARDIAN_MODEL"]
        else:
            self.model = os.environ.get("GUARDIAN_STANDIN_MODEL", STANDIN_MODEL)
        self.timeout_s = timeout_s or float(os.environ.get("GUARDIAN_TIMEOUT_S", "3.0"))
        # Hedge: if no answer after hedge_s, send the same request once more and take the
        # first good answer. Measured on the stand-in: ~1 in 3 requests stalls past 4 s while
        # the rest answer in ~0.5 s, so one duplicate request cuts the tail. 0 disables.
        self.hedge_s = float(os.environ.get("GUARDIAN_HEDGE_MS", "1000")) / 1000
        self.max_tokens = max_tokens
        # "none" turns thinking off (Fireworks chat API, reasoning_effort). gpt-oss does not
        # accept "none"; set GUARDIAN_REASONING_EFFORT=low there, or "" to omit the field.
        # The deployment (Qwen3-4B-Instruct-2507, a non-thinking model) omits it by default:
        # the docs do not say whether a non-reasoning model ignores or rejects the field, and
        # a 400 on every request would silently turn the guardian off.
        # https://docs.fireworks.ai/api-reference/post-chatcompletions
        default_effort = "" if self.backend == "deployment" else "none"
        self.reasoning_effort = os.environ.get("GUARDIAN_REASONING_EFFORT", default_effort)
        if self.backend == "stub":
            log.warning("GUARDIAN_BACKEND=stub: canned keyword guardian, no model is called")
        self._http: Optional[httpx.AsyncClient] = None
        self.last_error: Optional[str] = None
        self.last_usage: dict = {}

    def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout_s, connect=2.0),
                headers={"Authorization": f"Bearer {_fireworks_key()}"},
            )
        return self._http

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    async def score(self, req: GuardianRequest) -> tuple[GuardianResponse, int]:
        t0 = time.monotonic()
        self.last_error = None
        self.last_usage = {}
        if self.backend == "stub":
            resp = stub_guardian.score(req)
        else:
            try:
                resp = await asyncio.wait_for(self._remote(req), timeout=self.timeout_s)
            except Exception as e:  # timeout, HTTP error, bad JSON: never stall the call
                self.last_error = f"{type(e).__name__}: {str(e)[:300]}"
                log.error("guardian %s failed on %s: %s", self.backend, req.call_id, self.last_error)
                resp = SAFE_FALLBACK
        return resp, int((time.monotonic() - t0) * 1000)

    async def _remote(self, req: GuardianRequest) -> GuardianResponse:
        body = self._body(req)
        tasks = [asyncio.create_task(self._post(body))]
        try:
            if self.hedge_s:
                done, _ = await asyncio.wait(tasks, timeout=self.hedge_s)
                if not done:
                    log.info("guardian slow after %.0f ms on %s, hedging", self.hedge_s * 1000, req.call_id)
                    tasks.append(asyncio.create_task(self._post(body)))
            pending = set(tasks)
            last_exc: BaseException = RuntimeError("no guardian answer")
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for t in done:
                    if t.exception() is None:
                        return t.result()
                    last_exc = t.exception()
            raise last_exc
        finally:  # the loser, or everything when the overall timeout cancels us
            for t in tasks:
                if not t.done():
                    t.cancel()

    def _body(self, req: GuardianRequest) -> dict:
        body = {
            "model": self.model,
            "messages": render_messages(req),
            "response_format": response_format(),
            "temperature": 0,
            "max_tokens": self.max_tokens,
        }
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        return body

    async def _post(self, body: dict) -> GuardianResponse:
        http = self._client()
        for attempt in (1, 2):
            r = await http.post(FIREWORKS_URL, json=body)
            if r.status_code == 503 and attempt == 1:
                log.warning("guardian 503 (deployment waking?), retrying once")
                await asyncio.sleep(0.3)
                continue
            if r.status_code >= 400:
                raise RuntimeError(f"Fireworks {r.status_code}: {r.text[:300]}")
            break
        data = r.json()
        self.last_usage = data.get("usage") or {}
        return self._parse(data["choices"][0]["message"]["content"])

    def _parse(self, content: str) -> GuardianResponse:
        return parse_response(content)


_default: Optional[GuardianClient] = None


def default_client() -> GuardianClient:
    """The shared client configured from env. Call it at server start-up so a bad
    GUARDIAN_BACKEND or a missing GUARDIAN_MODEL fails there, not on the first turn."""
    global _default
    if _default is None:
        if os.environ.get("GUARDIAN_BACKEND") == "matt":
            from line.matt_bridge import MattGuardianClient  # Matt's render.py contract (line/matt_bridge.py)
            _default = MattGuardianClient()
        else:
            _default = GuardianClient()
        if _default.backend != "stub":
            _default._client()  # also fail fast on a missing FIREWORKS_API_KEY
    return _default


async def score(req: GuardianRequest) -> tuple[GuardianResponse, int]:
    """Module-level convenience using one shared client configured from env."""
    return await default_client().score(req)


# ---------------------------------------------------------------- latest-only scheduling
ScoreFn = Callable[[GuardianRequest], Awaitable[tuple[GuardianResponse, int]]]
ResultFn = Callable[[GuardianRequest, GuardianResponse, int], Awaitable[None]]


class LatestOnlyScorer:
    """One score in flight per call; newer turns collapse into a single follow-up score.

    kick() is called on every new turn. If idle, it starts scoring the current window.
    If busy, it only marks the window dirty; when the running score finishes, the newest
    window (built fresh from the session) is scored once. No backlog ever builds.
    """

    def __init__(self, session: CallSession, score_fn: ScoreFn, on_result: ResultFn, window: int = DEFAULT_WINDOW):
        self.session = session
        self.score_fn = score_fn
        self.on_result = on_result
        self.window = window
        self._dirty = False
        self._task: Optional[asyncio.Task] = None
        self.scored_turns: list[int] = []  # newest turn of each scored window (for tests/logs)

    def kick(self) -> None:
        self._dirty = True
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(self._run())

    async def _run(self) -> None:
        while self._dirty and not self.session.ended:
            self._dirty = False
            req = build_request(self.session, self.window)
            if not req.turns:
                return
            try:
                resp, latency_ms = await self.score_fn(req)
            except Exception:  # GuardianClient never raises; a custom score_fn might
                log.exception("guardian score failed on %s", self.session.call_id)
                continue
            self.scored_turns.append(req.turns[-1].turn)
            try:
                await self.on_result(req, resp, latency_ms)
            except Exception:
                log.exception("guardian result handler failed on %s", self.session.call_id)

    async def wait_idle(self) -> None:
        while self._task is not None and not self._task.done():
            await asyncio.wait({self._task})  # never raises, even if the task was cancelled

    def cancel(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()

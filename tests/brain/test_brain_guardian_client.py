"""line/guardian_client.py with a mocked Fireworks endpoint (no network)."""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from brain_fakes import make_session

from contracts.models import GuardianRequest, GuardianResponse, Turn
from line import guardian_client as gc
from line.standin_prompt import PROMPT_SHA, render_messages

GOOD = {
    "risk": "alert", "category": "scam.impersonation_bank_gov", "severity": 2, "source": "caller",
    "stage": "ask", "evidence": [{"turn": 5, "quote": "can you read me the code?"}],
    "action": "coach", "coach_line": "Banks never ask for codes.", "caretaker_note": "Code request.",
}


def completion(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


def client_with(handler, **kw) -> gc.GuardianClient:
    c = gc.GuardianClient(backend="standin", model="m", timeout_s=kw.pop("timeout_s", 1.0), **kw)
    c._http = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=c.timeout_s)
    return c


@pytest.fixture
def req(fixture_call):
    return GuardianRequest.model_validate(fixture_call["guardian_request"])


def test_build_request_window(fixture_call):
    s = make_session()
    s.turns = [Turn.model_validate(t) for t in fixture_call["turns"]] * 2  # 10 turns
    r = gc.build_request(s, window=8)
    assert len(r.turns) == 8 and r.call_id == s.call_id and r.rules == s.rules and r.caller_known is False
    assert r.turns[-1].text == s.turns[-1].text


def test_standin_prompt_renders_window_and_signals(req):
    msgs = render_messages(req)
    assert msgs[0]["role"] == "system" and "JSON" in msgs[0]["content"]
    user = msgs[1]["content"]
    assert '"ai_voice": "likely"' in user and "[5] Caller: Great, can you read me the code?" in user
    assert len(PROMPT_SHA) == 12


def test_render_messages_falls_back_to_standin(req):
    # guardian.prompt (Matt's M0) does not exist on this branch yet
    assert gc.render_messages(req) == render_messages(req)


def test_parse_response_fenced():
    assert gc.parse_response("```json\n" + json.dumps(GOOD) + "\n```").risk == "alert"
    with pytest.raises(ValueError):
        gc.parse_response('{"risk": "maybe"}')


@pytest.mark.asyncio
async def test_remote_ok_sends_schema_and_no_reasoning(req):
    sent = {}

    def handler(request: httpx.Request):
        sent.update(json.loads(request.content))
        return httpx.Response(200, json=completion(json.dumps(GOOD)))

    c = client_with(handler)
    resp, ms = await c.score(req)
    assert resp == GuardianResponse.model_validate(GOOD) and ms >= 0 and c.last_error is None
    assert sent["response_format"]["type"] == "json_schema"
    assert sent["response_format"]["json_schema"]["schema"]["properties"]["risk"]["enum"] == ["none", "watch", "alert", "urgent"]
    assert sent["reasoning_effort"] == "none" and sent["temperature"] == 0


@pytest.mark.asyncio
async def test_invalid_json_falls_back_to_none(req, caplog):
    c = client_with(lambda r: httpx.Response(200, json=completion("{")))
    resp, _ = await c.score(req)
    assert resp == gc.SAFE_FALLBACK and "invalid guardian JSON" in c.last_error
    assert any("guardian standin failed" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_retry_once_on_503(req):
    calls = []

    def handler(r):
        calls.append(1)
        return httpx.Response(503, text="waking") if len(calls) == 1 else httpx.Response(200, json=completion(json.dumps(GOOD)))

    resp, _ = await client_with(handler).score(req)
    assert resp.risk == "alert" and len(calls) == 2


@pytest.mark.asyncio
async def test_two_503s_fall_back(req):
    calls = []

    def handler(r):
        calls.append(1)
        return httpx.Response(503, text="waking")

    c = client_with(handler)
    resp, _ = await c.score(req)
    assert resp == gc.SAFE_FALLBACK and len(calls) == 2 and "503" in c.last_error


@pytest.mark.asyncio
async def test_timeout_falls_back_fast(req):
    async def handler(r):
        await asyncio.sleep(5)
        return httpx.Response(200, json=completion(json.dumps(GOOD)))

    c = client_with(handler, timeout_s=0.2)
    resp, ms = await c.score(req)
    assert resp == gc.SAFE_FALLBACK and ms < 2000


@pytest.mark.asyncio
async def test_hedge_takes_second_request_when_first_stalls(req):
    calls = []

    async def handler(r):
        calls.append(1)
        if len(calls) == 1:
            await asyncio.sleep(5)  # the stall we measured on the stand-in
        return httpx.Response(200, json=completion(json.dumps(GOOD)))

    c = client_with(handler, timeout_s=2.0)
    c.hedge_s = 0.05
    resp, ms = await c.score(req)
    assert resp.risk == "alert" and len(calls) == 2 and ms < 1000 and c.last_error is None


@pytest.mark.asyncio
async def test_no_hedge_when_fast(req):
    calls = []

    def handler(r):
        calls.append(1)
        return httpx.Response(200, json=completion(json.dumps(GOOD)))

    c = client_with(handler)
    c.hedge_s = 0.5
    await c.score(req)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_stub_backend(req):
    resp, ms = await gc.GuardianClient(backend="stub").score(req)
    assert resp.risk == "alert" and ms < 50


def test_backend_validation(monkeypatch):
    with pytest.raises(ValueError):
        gc.GuardianClient(backend="nope")
    monkeypatch.setenv("GUARDIAN_MODEL", "accounts/x/deployedModels/y")
    assert gc.GuardianClient(backend="deployment").model == "accounts/x/deployedModels/y"


def test_deployment_omits_reasoning_effort_by_default(monkeypatch, req):
    monkeypatch.delenv("GUARDIAN_REASONING_EFFORT", raising=False)
    monkeypatch.setenv("GUARDIAN_MODEL", "accounts/x/deployedModels/y")
    assert "reasoning_effort" not in gc.GuardianClient(backend="deployment")._body(req)
    assert gc.GuardianClient(backend="standin", model="m")._body(req)["reasoning_effort"] == "none"
    monkeypatch.setenv("GUARDIAN_REASONING_EFFORT", "low")
    assert gc.GuardianClient(backend="deployment")._body(req)["reasoning_effort"] == "low"


def test_broken_guardian_prompt_module_is_not_silently_replaced(monkeypatch, tmp_path, req):
    pkg = tmp_path / "guardian"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "prompt.py").write_text("import not_a_real_dependency_xyz\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(gc, "_prompt_fn", None)
    with pytest.raises(ModuleNotFoundError):
        gc.render_messages(req)
    for m in ("guardian", "guardian.prompt"):
        monkeypatch.delitem(__import__("sys").modules, m, raising=False)
    (pkg / "prompt.py").write_text("def render_messages(req):\n    return [{'role': 'user', 'content': 'M0'}]\n")
    monkeypatch.setattr(gc, "_prompt_fn", None)
    try:
        assert gc.render_messages(req) == [{"role": "user", "content": "M0"}]
    finally:
        for m in ("guardian", "guardian.prompt"):
            __import__("sys").modules.pop(m, None)
        gc._prompt_fn = None


def test_default_client_fails_fast_on_bad_backend(monkeypatch):
    monkeypatch.setattr(gc, "_default", None)
    monkeypatch.setenv("GUARDIAN_BACKEND", "deployment")
    monkeypatch.delenv("GUARDIAN_MODEL", raising=False)
    with pytest.raises(KeyError):
        gc.default_client()
    monkeypatch.setattr(gc, "_default", None)

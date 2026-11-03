"""Score frontier baselines on data/realset/calls.jsonl with the frozen guardian prompt.

Window rule: at the end of every CALLER turn t, the request holds the last 8 turns up to and
including t (both speakers), profile "elder" for every call (so the profile does not leak the
class), rules [], caller_known false, ai_voice None (unknown), earlier None ("nothing flagged").
Prompt: guardian/render.py (Matt's frozen training prompt, PROMPT_SHA 806908cfba7d).
Decision: a call is flagged if any scored turn returns risk in {alert, urgent}.

    python -m realset.score --model accounts/fireworks/models/kimi-k3 --tag kimi-k3
    python -m realset.score --keyword          # contracts.stub_guardian keyword baseline
    python -m realset.score --model accounts/fireworks/models/rune --tag rune --api systemone

Resume: re-running with the same --tag appends only (call_id, turn) pairs that are missing or have no
valid verdict in data/realset/scores/<tag>.jsonl, so new calls can be scored incrementally.

--api systemone (Rune 26B-A4B only serves /inference/v1/systemone, no chat): the state is the same
frozen SYSTEM prompt + rendered user message the chat models get, and two typed `choice` questions ask
for the risk level and the category (option lists copied from the prompt). Verdict risk = most likely
option (argmax, no threshold), so the same call-level decision rule applies.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import httpx

from guardian.render import CATS, PROMPT_SHA, RISKS, WINDOW, parse_response, render_messages

URL = os.environ.get("REALSET_URL") or "https://api.fireworks.ai/inference/v1/chat/completions"  # REALSET_URL: e.g. the sampler proxy
SYSTEMONE_URL = "https://api.fireworks.ai/inference/v1/systemone"
# Option descriptions for --api systemone: wording taken from the frozen SYSTEM prompt; written before
# any Rune call was scored and never changed after looking at results.
RISK_CRITERIA = {
    "none": "Normal call; stay quiet (including scary-sounding words used harmlessly)",
    "watch": "Mildly suspicious; just log it (an AI voice alone is at most watch)",
    "alert": "Speak up: a scam, a request for codes or personal info, grooming tactics, threats, abuse, "
             "self-harm, or a caretaker rule being broken",
    "urgent": "Serious immediate danger; urgent alert or suggest hanging up now",
}
SYSTEMONE_QUESTIONS = {
    "risk": {"type": "choice", "criteria": RISK_CRITERIA,
             "instructions": "You are the Emberz guardian described above. Judging the LAST turn given everything "
                             "so far, which risk level should the guardian return?"},
    "category": {"type": "choice", "criteria": {c: c for c in CATS.split()},
                 "instructions": "Which category best describes the LAST turn (none if nothing is wrong)?"},
}


async def _one_systemone(http, sem, model, req):
    msgs = render_messages(req)
    body = {"model": model, "state": msgs[0]["content"] + "\n\n" + msgs[1]["content"], "questions": SYSTEMONE_QUESTIONS}
    async with sem:
        for attempt in range(4):
            t0 = time.monotonic()
            try:
                r = await http.post(SYSTEMONE_URL, json=body)
                if r.status_code in (429, 500, 502, 503, 504):
                    await asyncio.sleep(2 * (attempt + 1)); continue
                if r.status_code >= 400:
                    return {"error": f"{r.status_code}: {r.text[:200]}", "usage": {}}
                d = r.json()
                a = d["answers"]
                risk, cat = a["risk"]["choice"], a["category"]["choice"]
                v = {"risk": risk, "category": cat, "severity": RISKS.index(risk) if risk in RISKS else 0,
                     "risk_probs": a["risk"].get("probabilities"), "risk_confidence": a["risk"].get("confidence")}
                u = d.get("usage") or {}
                return {"verdict": v if risk in RISKS else None, "usage": {"prompt_tokens": u.get("input_tokens", 0),
                        "completion_tokens": u.get("output_tokens", 0)}, "json_retry": False,
                        "latency_ms": int((time.monotonic() - t0) * 1000)}
            except (httpx.HTTPError, KeyError, ValueError, TypeError):
                await asyncio.sleep(1)
        return {"error": "retries exhausted", "usage": {}}
ALERT = {"alert", "urgent"}


def requests_for(call: dict) -> list[dict]:
    turns = call["turns"]
    out = []
    for i, t in enumerate(turns):
        if t["speaker"] != "caller":
            continue
        win = turns[max(0, i + 1 - WINDOW): i + 1]
        out.append({"call_id": call["call_id"], "profile": "elder", "rules": [], "caller_known": False,
                    "ai_voice": None, "earlier": None,
                    "turns": [{"turn": w["turn"], "speaker": w["speaker"], "text": w["text"]} for w in win]})
    return out


def _key() -> str:
    k = os.environ.get("FIREWORKS_API_KEY")
    if not k:
        from dotenv import dotenv_values
        k = dotenv_values(".env").get("FIREWORKS_API_KEY")
    if not k:
        raise SystemExit("FIREWORKS_API_KEY missing")
    return k


async def _one(http, sem, model, req, max_tokens, effort):
    first_usage: dict = {}
    body = {"model": model, "messages": render_messages(req), "temperature": 0, "max_tokens": max_tokens}
    if effort:
        body["reasoning_effort"] = effort
    async with sem:
        for attempt in range(4):
            t0 = time.monotonic()
            try:
                r = await http.post(URL, json=body)
                if r.status_code == 400 and effort and "reasoning" in r.text.lower():
                    body.pop("reasoning_effort", None); continue
                if r.status_code in (429, 500, 502, 503, 504):
                    await asyncio.sleep(2 * (attempt + 1)); continue
                if r.status_code >= 400:
                    return {"error": f"{r.status_code}: {r.text[:200]}", "usage": {}}
                d = r.json()
                msg = d["choices"][0]["message"].get("content") or ""
                v = parse_response(msg)
                if v is None and "response_format" not in body:
                    # Free-text reply with no JSON (Qwen3.8 Max sometimes "thinks aloud" with reasoning
                    # off and runs out of tokens): retry once in JSON mode, as a production client would.
                    body["response_format"] = {"type": "json_object"}
                    first_usage = d.get("usage") or {}
                    continue
                u = d.get("usage") or {}
                if "response_format" in body:  # count the failed first attempt's tokens too
                    u = {k: u.get(k, 0) + first_usage.get(k, 0) for k in ("prompt_tokens", "completion_tokens")}
                return {"verdict": v, "raw": None if v else msg[:300], "usage": u, "json_retry": "response_format" in body,
                        "latency_ms": int((time.monotonic() - t0) * 1000)}
            except (httpx.HTTPError, KeyError, ValueError) as e:
                err = f"{type(e).__name__}: {e}"[:200]
                await asyncio.sleep(1)
        return {"error": "retries exhausted", "usage": {}}


async def run_model(calls, model, tag, out_path, max_tokens, effort, conc, limit_calls=None, api="chat"):
    done = {}
    if out_path.exists():
        for l in open(out_path):
            r = json.loads(l); done[(r["call_id"], r["turn"])] = r
    reqs = [rq for c in calls[:limit_calls] for rq in requests_for(c)]
    def _bad(k):
        return k not in done or "error" in done[k] or not done[k].get("verdict")
    todo = [rq for rq in reqs if _bad((rq["call_id"], rq["turns"][-1]["turn"]))]
    print(f"{tag}: {len(reqs)} requests, {len(todo)} to do", flush=True)
    sem = asyncio.Semaphore(conc)
    async with httpx.AsyncClient(timeout=httpx.Timeout(90, connect=10), headers={"Authorization": f"Bearer {_key()}"}) as http:
        async def go(rq):
            if api == "systemone":
                res = await _one_systemone(http, sem, model, rq)
            else:
                res = await _one(http, sem, model, rq, max_tokens, effort)
            return {"call_id": rq["call_id"], "turn": rq["turns"][-1]["turn"], "model": tag, **res}
        with open(out_path, "a") as f:
            for i, fut in enumerate(asyncio.as_completed([go(r) for r in todo]), 1):
                r = await fut
                f.write(json.dumps(r) + "\n"); f.flush()
                if i % 50 == 0:
                    print(f"  {tag} {i}/{len(todo)}", flush=True)


def keyword(calls, out_path):
    from contracts import stub_guardian
    from contracts.models import GuardianRequest
    with open(out_path, "w") as f:
        for c in calls:
            for rq in requests_for(c):
                g = GuardianRequest(call_id=rq["call_id"], profile="elder", rules=[], caller_known=False, ai_voice=None,
                                    turns=rq["turns"])
                v = stub_guardian.score(g).model_dump()
                f.write(json.dumps({"call_id": rq["call_id"], "turn": rq["turns"][-1]["turn"], "model": "keyword-stub",
                                    "verdict": v, "usage": {}}) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", default="data/realset/calls.jsonl")
    ap.add_argument("--model")
    ap.add_argument("--tag")
    ap.add_argument("--keyword", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--effort", default="none")
    ap.add_argument("--conc", type=int, default=8)
    ap.add_argument("--limit-calls", type=int)
    ap.add_argument("--api", choices=["chat", "systemone"], default="chat",
                    help="systemone: Fireworks decisions endpoint (Rune); --effort/--max-tokens are ignored")
    a = ap.parse_args()
    calls = []
    for l in open(a.calls):  # tolerate a half-written last line while transcribe.py is appending
        try:
            calls.append(json.loads(l))
        except ValueError:
            pass
    Path("data/realset/scores").mkdir(parents=True, exist_ok=True)
    if a.keyword:
        keyword(calls, Path("data/realset/scores/keyword-stub.jsonl")); return
    asyncio.run(run_model(calls, a.model, a.tag, Path(f"data/realset/scores/{a.tag}.jsonl"), a.max_tokens, a.effort, a.conc, a.limit_calls, a.api))
    print("prompt_sha", PROMPT_SHA)


if __name__ == "__main__":
    main()

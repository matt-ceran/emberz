"""Smoke test for the caretaker console.

From callguard/:
    uv run --with fastapi --with 'uvicorn[standard]' python -m console.smoke_test

Starts, in-process on free local ports (daemon threads, stopped at the end):
  - a fake line server with Numan's console routes (WS /console/ws with ?token=, POST /console/calls/{id}/barge),
  - the console server, bridged to that fake line via LINE_CONSOLE_URL,
then drives everything over HTTP and WebSocket and exits 0 when every check passes. Nothing is left running.
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

import uvicorn
import websockets
from fastapi import Body, FastAPI, HTTPException, Request, WebSocket

TYPES = {"call_started", "turn", "risk", "alert", "coach", "caretaker_joined", "summary", "call_ended", "voice"}
TURN_KEYS = {"call_id", "turn", "speaker", "text", "t_start_ms", "t_end_ms", "gap_ms"}
ROW_KEYS = {"call_id", "n_events", "last_at_ms", "started", "max_risk", "alerts", "ai_score", "caretaker", "summary", "ended"}
RISK_ORDER = {"none": 0, "watch": 1, "alert": 2, "urgent": 3}
FAST = 400.0
LINE_TOKEN = "line-secret"
PASSED: list[str] = []


def ok(cond, msg):
    if not cond:
        raise AssertionError(msg)


def check(name):
    PASSED.append(name)
    print(f"  ok  {name}")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def call(method: str, url: str, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            raw = r.read().decode()
            return r.status, (json.loads(raw) if r.headers.get_content_type() == "application/json" else raw)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "null")


def numan_check(ev: dict):
    """Top-level ConsoleEvent shape + the per-type data checks of Numan's console/fixture_player.py check_payload.
    (GuardianResponse value strictness is a separate contract issue: see README, Known gaps.)"""
    ok(set(ev) <= {"type", "call_id", "at_ms", "data"} and {"type", "call_id", "at_ms"} <= set(ev), f"bad keys {set(ev)}")
    ok(ev["type"] in TYPES, f"bad type {ev['type']}")
    ok(isinstance(ev["call_id"], str) and isinstance(ev["at_ms"], int) and ev["at_ms"] >= 0, "call_id/at_ms")
    d = ev.get("data", {})
    ok(isinstance(d, dict), "data not object")
    t = ev["type"]
    if t == "turn":
        ok(set(d) == TURN_KEYS and d["speaker"] in ("caller", "protected") and d["turn"] >= 1, f"turn {d}")
    elif t == "voice":
        ok(set(d) == {"call_id", "at_ms", "ai_score", "model"} and 0 <= d["ai_score"] <= 1, f"voice {d}")
    elif t == "risk":
        ok(isinstance(d["turn"], int) and isinstance(d["latency_ms"], int) and isinstance(d["response"], dict), "risk")
        ok(d.get("display_risk", "none") in RISK_ORDER, "display_risk")
        ok(isinstance(d.get("suggest_hangup", False), bool) and isinstance(d.get("reasons", []), list), "risk extras")
    elif t == "alert":
        ok(d["level"] in ("alert", "urgent") and d["category"] and isinstance(d["turn"], int), f"alert {d}")
        ok(isinstance(d["caretaker_note"], str), "alert note")
    elif t == "coach":
        ok(isinstance(d["text"], str) and isinstance(d["turn"], int) and isinstance(d.get("sent", True), bool), "coach")
    elif t == "caretaker_joined":
        ok(d["mode"] in ("monitor", "barge"), f"caretaker_joined mode {d.get('mode')!r} is not monitor|barge")
    elif t == "summary":
        ok(isinstance(d["text"], str) and d["text"], "summary text")
    elif t == "call_started":
        ok(set(d) >= {"caller_number", "guarded_number", "profile", "caller_known"}, "call_started keys")
    elif t == "call_ended":
        ok(isinstance(d["duration_ms"], int) and isinstance(d["reason"], str), "call_ended")
    if "call_id" in d:
        ok(d["call_id"] == ev["call_id"], "data.call_id != call_id")


async def recv(ws, timeout=5.0):
    return json.loads(await asyncio.wait_for(ws.recv(), timeout))


async def collect_until(ws, stop, timeout=20.0):
    out, end = [], time.monotonic() + timeout
    while True:
        left = end - time.monotonic()
        ok(left > 0, f"timed out after {len(out)} events: {[(e['type'], e['call_id']) for e in out[-5:]]}")
        ev = await recv(ws, left)
        numan_check(ev)
        out.append(ev)
        if stop(ev):
            return out


async def quiet(ws, seconds):
    """Return any events that arrive within `seconds` (expected: none)."""
    got = []
    try:
        while True:
            got.append(await recv(ws, seconds))
    except (asyncio.TimeoutError, TimeoutError):
        return got


def wait_for(pred, timeout=8.0, what="condition"):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


# ------------------------------------------------------------------ fake line server (Numan's console routes)
def _ev(t, cid, at, **data):
    return {"type": t, "call_id": cid, "at_ms": at, "data": data}


def make_fake_line() -> tuple[FastAPI, dict]:
    st: dict = {"events": [], "queues": set(), "ws_tokens": [], "barge_tokens": [], "monitoring": set(), "known": set()}
    resp = {"risk": "urgent", "category": "disclosure.credentials_otp", "severity": 3, "source": "caller", "stage": "ask",
            "evidence": [{"turn": 1, "quote": "read me the code"}], "action": "urgent_alert", "coach_line": "",
            "caretaker_note": ""}
    st["events"] = [
        _ev("call_started", "line_1", 0, caller_number="+12025550199", guarded_number="+12025550100", profile="elder",
            caller_known=True),
        _ev("turn", "line_1", 3000, call_id="line_1", turn=1, speaker="caller", text="Hi, can you read me the code",
            t_start_ms=1000, t_end_ms=3000, gap_ms=None),
        _ev("risk", "line_1", 3400, turn=1, response=resp, latency_ms=380, display_risk="watch", suggest_hangup=False,
            reasons=["known_caller_cap"]),
        _ev("caretaker_joined", "line_1", 4000, mode="monitor"),
        _ev("call_started", "line_2", 0, caller_number="+12025550177", guarded_number="+12025550100", profile="kid",
            caller_known=False),
    ]
    st["monitoring"] = {"line_1"}
    st["known"] = {"line_1", "line_2"}
    app = FastAPI()

    def emit(ev):
        st["events"].append(ev)
        st["known"].add(ev["call_id"])
        for q in list(st["queues"]):
            q.put_nowait(ev)

    @app.websocket("/console/ws")
    async def ws_(ws: WebSocket, token: Optional[str] = None):
        st["ws_tokens"].append(token)
        if token != LINE_TOKEN:
            await ws.close(code=1008)
            return
        await ws.accept()
        q: asyncio.Queue = asyncio.Queue()
        st["queues"].add(q)
        async def pump():
            for e in list(st["events"]):
                await ws.send_text(json.dumps(e))
            while True:
                await ws.send_text(json.dumps(await q.get()))

        async def drain():  # notices the disconnect, so shutdown never waits on a parked pump
            while True:
                await ws.receive_text()

        tasks = [asyncio.create_task(pump()), asyncio.create_task(drain())]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for t in tasks:
                t.cancel()
            st["queues"].discard(q)

    @app.post("/emit")
    async def emit_(payload: list = Body(...)):
        for e in payload:
            emit(e)
        return {"ok": True}

    @app.post("/console/calls/{call_id}/barge")
    async def barge(call_id: str, request: Request):
        st["barge_tokens"].append(request.headers.get("x-console-token"))
        if call_id not in st["known"]:
            raise HTTPException(404, "unknown or finished call")
        if call_id not in st["monitoring"]:
            raise HTTPException(409, "caretaker is idle, not monitoring")
        emit(_ev("caretaker_joined", call_id, 9000, mode="barge"))
        return {"call_id": call_id, "caretaker_state": "barged"}

    return app, st


class Capture(BaseHTTPRequestHandler):
    received: list[dict] = []

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        Capture.received.append(json.loads(self.rfile.read(n)))
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *a):
        pass


# ------------------------------------------------------------------ replay stream check
def check_replay_stream(ds, rp, src: str, cid: str, events: list[dict]):
    row = ds.calls[src]
    n = len(row["turns"])
    preds = ds.preds_by_turn(src)
    fa = (ds.preds.get(src) or {}).get("first_alert")
    types = [e["type"] for e in events]
    ok(all(e["call_id"] == cid for e in events), "foreign call_id in stream")
    ok(types[0] == "call_started" and types[-2:] == ["call_ended", "summary"], f"bad framing {types[:2]}..{types[-2:]}")
    ok(all(a["at_ms"] <= b["at_ms"] for a, b in zip(events, events[1:])), "at_ms not monotonic")
    ok(events == json.loads(json.dumps(rp.build_events(ds, src, as_call_id=cid))), "stream differs from build_events")
    turns = [e["data"] for e in events if e["type"] == "turn"]
    ok([t["turn"] for t in turns] == [t["turn"] for t in row["turns"]], "turn numbers/order wrong")
    ok([t["text"] for t in turns] == [t["text"] for t in row["turns"]], "turn text altered")
    risk = [e for e in events if e["type"] == "risk"]
    ok([e["data"]["turn"] for e in risk] == sorted(preds), "risk events != predictions")
    for e in risk:
        d = e["data"]
        ok(set(d["response"]) <= set(rp.GUARDIAN_FIELDS), "response has non-GuardianResponse keys")
        ok(d["response"]["risk"] == preds[d["turn"]]["risk"] == d["display_risk"], "risk differs from preds file")
        ok(d["latency_ms"] == preds[d["turn"]].get("_ms"), "latency differs from preds file")
        ti = next(i for i, x in enumerate(events) if x["type"] == "turn" and x["data"]["turn"] == d["turn"])
        ok(events.index(e) > ti, "risk before its turn")
    alerts = [e["data"] for e in events if e["type"] == "alert"]
    if fa:
        ok(len(alerts) == 1 and alerts[0]["turn"] == fa, f"expected one alert at turn {fa}, got {alerts}")
        ok(not [e for e in risk if e["data"]["turn"] > fa], "fabricated verdict after first alert")
        ok(len(turns) == n, "turns after the first alert were not streamed")
    else:
        ok(not alerts, "alert on a call the guardian never alerted on")
    for e in events:
        if e["type"] == "coach":
            ok(e["data"]["text"] == preds[e["data"]["turn"]]["coach_line"] and e["data"]["sent"] is True, "coach")
    if (row.get("ai_voice") or {}).get("score") is not None:
        ok(len([e for e in events if e["type"] == "voice"]) == 1, "voice event")
    ok(events[-2]["data"]["reason"] == "completed", "call_ended reason")


async def run(base: str, ws_url: str, line_base: str, line_st: dict):
    from console import replay as rp
    from console import server as srv

    ds = srv.dataset()
    st, h = call("GET", base + "/health")
    ok(st == 200 and h["calls"] == len(ds.calls) and h["calls_with_preds"] > 0, f"health {st} {h}")
    st, page = call("GET", base + "/")
    ok(st == 200 and "Emberz" in page and "/console/ws" in page and "/console/calls" in page, "index page")
    ok('<meta name="emberz-backend" content="">' in page, "served page should default to this server")
    raw = (srv.STATIC / "index.html").read_text()
    ok('<meta name="emberz-backend" content="http://127.0.0.1:8001">' in raw, "file should default to the fixture player")
    ok("<script src" not in page and "<link" not in page, "page loads an external resource")
    req = urllib.request.Request(base + "/events", method="OPTIONS", headers={
        "Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"})
    with urllib.request.urlopen(req, timeout=5) as r:
        ok(r.headers.get("access-control-allow-origin") in ("*", "http://localhost:5173"), "CORS preflight")
    check(f"health + page (defaults to its own origin; file defaults to :8001) + CORS ({h['calls']} calls, "
          f"preds {h['preds_file'].split('/')[-1]})")

    async with websockets.connect(ws_url) as ws, websockets.connect(ws_url) as ws2:
        # ---------------------------------------------------------- bridge from the (fake) line
        wait_for(lambda: {"line_1", "line_2"} <= {r["call_id"] for r in call("GET", base + "/console/calls")[1]},
                 what="bridged line calls")
        got = await collect_until(ws, lambda e: e["call_id"] == "line_2" and e["type"] == "call_started")
        ok([e for e in got if e["call_id"].startswith("line_")] == line_st["events"], "bridged events differ from the line's")
        await collect_until(ws2, lambda e: e["call_id"] == "line_2" and e["type"] == "call_started")
        ok(line_st["ws_tokens"] and line_st["ws_tokens"][-1] == LINE_TOKEN, "bridge did not send ?token=")
        rows = {r["call_id"]: r for r in call("GET", base + "/console/calls")[1]}
        ok(ROW_KEYS <= set(rows["line_1"]) and rows["line_1"]["max_risk"] == "watch", "row max_risk must use display_risk")
        ok(rows["line_1"]["caretaker"] == "monitor" and rows["line_1"]["origin"] == "line", "row caretaker/origin")
        check("bridge: line /console/ws (with token) republished; /console/calls row uses display_risk (watch, raw urgent)")

        # replay guard while a live call is in progress
        st, r = call("GET", f"{base}/replay?preset=elder_scam&speed={FAST}")
        ok(st == 409 and "live call" in r["detail"], f"replay during live call should 409: {st} {r}")
        ok(call("POST", f"{base}/fixtures/play/elder_scam?speed=20")[0] == 409, "fixtures/play during live call")
        check("replay refused (409) while a live call is in progress")

        # Join on a line call goes to the line's barge endpoint; the event comes back over the bridge
        st, r = call("POST", f"{base}/console/calls/line_2/barge")
        ok(st == 409 and "not monitoring" in r["detail"], f"line 409 passthrough {st} {r}")
        st, r = call("POST", f"{base}/console/calls/line_1/barge")
        ok(st == 200 and r == {"call_id": "line_1", "caretaker_state": "barged"}, f"barge via line {st} {r}")
        ok(line_st["barge_tokens"][-1] == LINE_TOKEN, "barge forward did not send X-Console-Token")
        ev = await recv(ws)
        ok(ev["type"] == "caretaker_joined" and ev["call_id"] == "line_1" and ev["data"]["mode"] == "barge", f"{ev}")
        ok(not await quiet(ws, 0.3), "console emitted its own caretaker_joined on top of the line's")
        await recv(ws2)
        st, r = call("POST", f"{base}/actions/line_1/listen")
        ok(st == 409 and "press 1" in r["detail"], f"live listen should 409 {st} {r}")
        st, r = call("POST", f"{base}/actions/line_1/hangup")
        ok(st == 501, f"live hangup without LINE_ACTION_URL should 501 {st} {r}")
        ok(not await quiet(ws, 0.3), "refused actions must not emit events")
        check("Join on a line call -> line's /console/calls/{id}/barge (200 / 409 passthrough), event only from the line; "
              "live Listen 409, live Hang up 501")

        call("POST", line_base + "/emit", [_ev("call_ended", "line_1", 20000, duration_ms=20000, reason="caller_hangup"),
                                           _ev("call_ended", "line_2", 9000, duration_ms=9000, reason="caller_hangup")])
        for sock in (ws, ws2):
            await collect_until(sock, lambda e: e["call_id"] == "line_2" and e["type"] == "call_ended")
        wait_for(lambda: call("GET", base + "/health")[1]["live_call"] is None, what="live calls to end")

        # ---------------------------------------------------------- replays of every preset (+ the flip's pair)
        st, pj = call("GET", base + "/presets")
        presets = pj["presets"]
        ok(st == 200 and 4 <= len(presets) <= 5, f"expected 4-5 presets, got {len(presets)}")
        for p in presets:
            row = ds.calls[p["call_id"]]
            ok(row.get("family") != "self_harm" and not any(t["label"]["category"] == "self_harm" for t in row["turns"]),
               f"self-harm preset {p['call_id']}")
        elder = next(p for p in presets if p["id"] == "elder_scam")
        ok(not rp.discloses_after_ask(ds.calls[elder["call_id"]]), "elder preset: protected person reads out numbers")
        check("presets: " + ", ".join(f"{p['id']}={p['call_id']}" for p in presets))

        targets = [p["call_id"] for p in presets] + [p["pair"]["call_id"] for p in presets if p.get("pair")]
        played = []
        for src in targets:
            st, r = call("GET", f"{base}/replay?call_id={src}&speed={FAST}")
            ok(st == 200 and r["source_call_id"] == src, f"replay start {st} {r}")
            cid = r["call_id"]
            events = await collect_until(ws, lambda e: e["type"] == "summary" and e["call_id"] == cid)
            check_replay_stream(ds, rp, src, cid, events)
            ok(await collect_until(ws2, lambda e: e["type"] == "summary" and e["call_id"] == cid) == events,
               "second browser saw a different stream")
            st, buf = call("GET", f"{base}/console/calls/{cid}/events")
            ok(st == 200 and buf == events, "/console/calls/{id}/events differs from the stream")
            row = next(x for x in call("GET", base + "/console/calls")[1] if x["call_id"] == cid)
            fa = (ds.preds.get(src) or {}).get("first_alert")
            ok(row["replay"] and row["ended"]["reason"] == "completed" and row["summary"] and row["alerts"] == (1 if fa else 0),
               f"row {row}")
            played.append(cid)
            check(f"replay {src} as {cid}: {len(events)} events, first_alert={fa}, ask_turn={ds.calls[src].get('ask_turn')}")

        st, r = call("GET", f"{base}/replay?call_id={targets[0]}&speed={FAST}")
        ok(st == 200 and r["call_id"] == f"{targets[0]}-2", f"second replay of a call needs a fresh id: {r}")
        await collect_until(ws, lambda e: e["type"] == "summary" and e["call_id"] == r["call_id"])
        played.append(r["call_id"])
        ok(call("GET", f"{base}/replay?call_id=nope")[0] == 404 and call("GET", f"{base}/replay?preset=nope")[0] == 404,
           "unknown call/preset should 404")
        ok(call("GET", f"{base}/console/calls/nope/events")[0] == 404, "unknown call events should 404")
        check("second replay of a call gets a fresh id (-2); unknown call/preset/events -> 404")

        # late joiner: every buffered call, in publish order; ?call_id= filter
        async with websockets.connect(ws_url) as ws3:
            hist = await collect_until(ws3, lambda e: e["type"] == "summary" and e["call_id"] == played[-1])
            starts = [e["call_id"] for e in hist if e["type"] == "call_started"]
            ok(starts == ["line_1", "line_2"] + played, f"late joiner call order {starts}")
        async with websockets.connect(f"{ws_url}?call_id={played[0]}") as ws4:
            one = await collect_until(ws4, lambda e: e["type"] == "summary")
            ok(one == call("GET", f"{base}/console/calls/{played[0]}/events")[1], "?call_id= filter")
        check(f"late-joining browser gets all {len(starts)} calls in publish order; ?call_id= filters to one call")

        # ---------------------------------------------------------- /fixtures (Numan's fixture-player routes)
        st, fx = call("GET", base + "/fixtures")
        ok(st == 200 and {p["id"] for p in presets} <= set(fx) and all({"call_id", "events", "duration_ms"} <= set(v)
                                                                        for v in fx.values()), f"fixtures {fx}")
        ok(call("POST", f"{base}/fixtures/play/nope")[0] == 404, "unknown scene 404")
        ok(call("POST", f"{base}/fixtures/play/elder_scam?speed=50")[0] == 422, "speed > 20 should 422")
        st, r = call("POST", f"{base}/fixtures/play/elder_scam?speed=20")
        ok(st == 200 and set(r) == {"scene", "call_id"} and r["call_id"].startswith(elder["call_id"] + "-"), f"play {r}")
        await collect_until(ws, lambda e: e["type"] == "summary" and e["call_id"] == r["call_id"])
        check(f"/fixtures lists {len(fx)} scenes; /fixtures/play/elder_scam -> {r['call_id']}; 404/422 like the player")

        # ---------------------------------------------------------- Listen / Join / Hang up on a replay
        st, r = call("GET", f"{base}/replay?preset=elder_scam&speed=4")
        cid = r["call_id"]
        await collect_until(ws, lambda e: e["type"] == "turn" and e["call_id"] == cid)
        st, b = call("POST", f"{base}/console/calls/{cid}/barge")
        ok(st == 409 and "not listening" in b["detail"], f"join before listen should 409: {st} {b}")
        st, b = call("POST", f"{base}/actions/{cid}/listen")
        ok(st == 200 and b["caretaker_state"] == "monitoring", f"listen {st} {b}")
        ev = await collect_until(ws, lambda e: e["type"] == "caretaker_joined")
        ok(ev[-1]["data"] == {"mode": "monitor"}, f"monitor event {ev[-1]}")
        st, b = call("POST", f"{base}/console/calls/{cid}/barge")
        ok(st == 200 and b["call_id"] == cid and b["caretaker_state"] == "barged", f"barge {st} {b}")
        ev = await collect_until(ws, lambda e: e["type"] == "caretaker_joined")
        ok(ev[-1]["data"] == {"mode": "barge"}, f"barge event {ev[-1]}")
        st, b = call("POST", f"{base}/actions/{cid}/join")
        ok(st == 200 and b["caretaker_state"] == "barged", "second join is a harmless 200")
        st, b = call("POST", f"{base}/actions/{cid}/hangup")
        ok(st == 200 and b["replay"] is True and b["ended"] is True, f"replay hangup {b}")
        tail = await collect_until(ws, lambda e: e["type"] == "call_ended", timeout=5)
        ok([e for e in tail if e["type"] == "caretaker_joined"] == [], "second join emitted another caretaker_joined")
        ok(tail[-1]["data"]["reason"] == "caretaker_hangup", "call_ended reason")
        late = await quiet(ws, 1.0)
        ok(not late, f"replay kept streaming after hang-up: {[e['type'] for e in late]}")
        ok(call("POST", f"{base}/console/calls/{cid}/barge")[0] == 409, "join after the call ended should 409")
        ok(call("POST", f"{base}/console/calls/nope/barge")[0] == 404, "unknown call barge 404")
        ok(call("POST", f"{base}/actions/nope/join")[0] == 404, "unknown call action 404")
        ok(call("POST", f"{base}/actions/{cid}/explode")[0] == 404, "unknown action 404")
        check("replay: Join before Listen 409; Listen -> monitor; Join -> 200 barged + barge; Hang up stops it; 404s")

        # ---------------------------------------------------------- POST /events (optional ingest) + live actions
        fixture = [
            _ev("call_started", "c_push_1", 0, caller_number="+12025550199", guarded_number="+12025550100",
                profile="elder", caller_known=False),
            _ev("voice", "c_push_1", 2400, call_id="c_push_1", at_ms=2400, ai_score=0.91, model="w2v-head-v1"),
            _ev("turn", "c_push_1", 5600, call_id="c_push_1", turn=1, speaker="caller",
                text="Hello, this is the fraud department at your bank.", t_start_ms=3100, t_end_ms=5600, gap_ms=None),
            _ev("risk", "c_push_1", 6000, turn=1, latency_ms=410, display_risk="alert", suggest_hangup=False, reasons=[],
                response={"risk": "alert", "category": "scam.impersonation_bank_gov", "severity": 2, "source": "caller",
                          "stage": "hook", "evidence": [{"turn": 1, "quote": "fraud department at your bank"}],
                          "action": "notify_caretaker", "coach_line": "Banks never ask for codes.",
                          "caretaker_note": "Caller claims to be the bank."}),
            _ev("alert", "c_push_1", 6020, level="alert", category="scam.impersonation_bank_gov",
                caretaker_note="Caller claims to be the bank.", turn=1),
        ]
        st, r = call("POST", base + "/events", fixture)
        ok(st == 200 and r["accepted"] == len(fixture), f"POST /events list {st} {r}")
        single = _ev("coach", "c_push_1", 6050, text="Banks never ask for codes.", turn=1, sent=False)
        st, r = call("POST", base + "/events", single)
        ok(st == 200 and r["accepted"] == 1 and r["clients"] >= 2, f"POST /events single {st} {r}")
        for sock in (ws, ws2):
            got = await collect_until(sock, lambda e: e["type"] == "coach" and e["call_id"] == "c_push_1")
            ok([e for e in got if e["call_id"] == "c_push_1"] == fixture + [single], "fan-out differs from what was posted")
        for bad in ({"type": "bogus", "call_id": "x", "at_ms": 0}, {"type": "turn", "call_id": "x", "at_ms": -1},
                    {"type": "turn", "call_id": "x", "at_ms": 0, "extra": 1}, {"call_id": "x", "at_ms": 0}):
            ok(call("POST", base + "/events", bad)[0] == 422, f"should reject {bad}")
        ok(not await quiet(ws, 0.3), "rejected events leaked to browsers")
        check("POST /events fans out list + single to 2 browsers; off-schema -> 422, nothing broadcast")

        os.environ.pop("LINE_ACTION_URL", None)
        ok(call("POST", f"{base}/actions/c_push_1/listen")[0] == 409, "live listen 409")
        ok(call("POST", f"{base}/console/calls/c_push_1/barge")[0] == 409, "join before monitor 409")
        call("POST", base + "/events", _ev("caretaker_joined", "c_push_1", 7000, mode="monitor"))
        await recv(ws)
        await recv(ws2)
        st, r = call("POST", f"{base}/console/calls/c_push_1/barge")
        ok(st == 404, f"pushed call unknown to the line -> line's 404 passes through: {st} {r}")
        ok(call("POST", f"{base}/actions/c_push_1/hangup")[0] == 501, "hangup without LINE_ACTION_URL 501")
        st, r = call("POST", f"{base}/actions/c_push_1/safe")
        ok(st == 200 and r["safe"] is True, f"safe {r}")
        ok(next(x for x in call("GET", base + "/console/calls")[1] if x["call_id"] == "c_push_1")["safe"], "row safe")
        ok(not await quiet(ws, 0.3), "live actions must not fabricate events")
        cap = HTTPServer(("127.0.0.1", 0), Capture)
        threading.Thread(target=cap.serve_forever, daemon=True).start()
        try:
            os.environ["LINE_ACTION_URL"] = f"http://127.0.0.1:{cap.server_address[1]}/caretaker"
            st, r = call("POST", f"{base}/actions/c_push_1/hangup")
            ok(st == 200 and r["forwarded"] is True, f"forward {r}")
            body = Capture.received[-1]
            ok(set(body) == {"call_id", "action", "at_ms"} and body["action"] == "hangup", f"captured {body}")
            os.environ["LINE_ACTION_URL"] = f"http://127.0.0.1:{free_port()}/dead"
            ok(call("POST", f"{base}/actions/c_push_1/hangup")[0] == 502, "dead line -> 502")
        finally:
            os.environ.pop("LINE_ACTION_URL", None)
            cap.shutdown()
            cap.server_close()
        ok(not await quiet(ws, 0.3), "forwarded hang-up must not fake call_ended")
        check(f"live call: Listen 409, Join 409 then line's answer, Hang up 501 / forwarded {body} / 502; Mark safe "
              f"console-only; no fabricated events")
        call("POST", base + "/events", _ev("call_ended", "c_push_1", 9000, duration_ms=9000, reason="caller_hangup"))
        await recv(ws)
        await recv(ws2)

        # ---------------------------------------------------------- a new live call stops a running replay
        for how in ("push", "bridge"):
            st, r = call("GET", f"{base}/replay?preset=elder_scam&speed=4")
            ok(st == 200, f"replay start {st} {r}")
            rid = r["call_id"]
            await collect_until(ws, lambda e: e["type"] == "turn" and e["call_id"] == rid, timeout=10)
            live = f"c_live_{how}"
            start = _ev("call_started", live, 0, caller_number=None, guarded_number="+12025550100", profile="kid",
                        caller_known=False)
            if how == "push":
                call("POST", base + "/events", start)
            else:
                call("POST", line_base + "/emit", [start])
            got = await collect_until(ws, lambda e: e["call_id"] == live, timeout=5)
            stop = [e for e in got if e["call_id"] == rid and e["type"] == "call_ended"]
            ok(stop and stop[-1]["data"]["reason"] == "replay_stopped", f"replay not stopped by {how}: {got[-3:]}")
            late = await quiet(ws, 0.8)
            ok(not [e for e in late if e["call_id"] == rid], f"replay kept streaming after a live call ({how})")
            ok(call("GET", f"{base}/replay?preset=elder_scam&speed=4")[0] == 409, "replay during live call 409")
            end = _ev("call_ended", live, 5000, duration_ms=5000, reason="caller_hangup")
            call("POST", base + "/events", end) if how == "push" else call("POST", line_base + "/emit", [end])
            await collect_until(ws, lambda e: e["call_id"] == live and e["type"] == "call_ended", timeout=5)
        check("a new live call (POST /events or bridged) stops a running replay (call_ended replay_stopped)")


def _serve(app, port) -> tuple[uvicorn.Server, threading.Thread]:
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", lifespan="on"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    t0 = time.monotonic()
    while not server.started:
        if time.monotonic() - t0 > 20 or not thread.is_alive():
            raise RuntimeError("server did not start")
        time.sleep(0.05)
    return server, thread


def main() -> int:
    line_app, line_st = make_fake_line()
    line_port, port = free_port(), free_port()
    os.environ.pop("CONSOLE_TOKEN", None)
    os.environ.pop("LINE_ACTION_URL", None)
    os.environ["LINE_CONSOLE_URL"] = f"http://127.0.0.1:{line_port}"
    os.environ["LINE_CONSOLE_TOKEN"] = LINE_TOKEN
    from console import server as srv

    servers = []
    code = 0
    try:
        servers.append(_serve(line_app, line_port))
        servers.append(_serve(srv.app, port))
        print(f"console smoke test on 127.0.0.1:{port}, fake line on :{line_port} (preds: {srv.dataset().preds_path.name})")
        asyncio.run(run(f"http://127.0.0.1:{port}", f"ws://127.0.0.1:{port}/console/ws",
                        f"http://127.0.0.1:{line_port}", line_st))
    except Exception as e:
        print(f"FAIL: {type(e).__name__}: {e}")
        code = 1
    finally:
        for server, thread in reversed(servers):
            server.should_exit = True
            thread.join(timeout=10)
    alive = any(t.is_alive() for _, t in servers)
    print(f"{'PASS' if code == 0 and not alive else 'FAIL'}: {len(PASSED)} checks passed; servers stopped={not alive}")
    return 1 if alive else code


if __name__ == "__main__":
    sys.exit(main())

"""Demo "Go" button: POST /demo/dial?token=...&persona=bank_fraud places one AI test call to the guarded line.

Enabled only with EMBERZ_DEMO_DIAL=1. Token-protected like /actions (CONSOLE_TOKEN). One call at a time,
DEMO_DIAL_COOLDOWN_S (default 60) between calls; each call costs about $0.34 on SignalWire.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from line.console_hub import _token_status

log = logging.getLogger("emberz.demo_dial")
PERSONAS = ("bank_fraud", "tech_support", "prize_giftcard", "pharmacy_refill", "appointment_confirm", "family_money")
ROOT = Path(__file__).resolve().parents[1]


def demo_dial_router() -> APIRouter:
    router = APIRouter()
    state = {"busy": False, "last": 0.0, "n": {}}
    cooldown = float(os.environ.get("DEMO_DIAL_COOLDOWN_S", "60"))

    @router.post("/demo/dial")
    async def dial(request: Request, persona: str = "bank_fraud") -> dict:
        if _token_status(request.query_params.get("token") or request.headers.get("x-console-token"), False) != 200:
            raise HTTPException(401, "bad or missing console token")
        if persona not in PERSONAS:
            raise HTTPException(400, f"persona must be one of {PERSONAS}")
        wait = cooldown - (time.monotonic() - state["last"])
        if state["busy"] or wait > 0:
            raise HTTPException(429, f"a demo call was just placed; try again in {max(1, int(wait))} s")
        state["busy"], state["last"] = True, time.monotonic()
        idx = state["n"].get(persona, 0)
        state["n"][persona] = (idx + 1) % 3
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "callers.signalwire.place_calls", "--personas", persona, "--calls", "1",
                "--start", str(idx), "--live", "--max-calls", "1", cwd=str(ROOT),
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
            out, _ = await asyncio.wait_for(proc.communicate(b"\n\n"), timeout=60)
            last = out.decode(errors="replace").strip().splitlines()[-1] if out else ""
            ok = proc.returncode == 0 and "HTTP 200" in last
            log.info("demo dial %s #%d: %s", persona, idx, last[-160:])
            if not ok:
                raise HTTPException(502, f"dial failed: {last[-200:]}")
            return {"ok": True, "persona": persona, "variant": idx, "detail": "Calling the protected phone now"}
        finally:
            state["busy"] = False

    return router

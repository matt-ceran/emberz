"""Optional FastAPI router that serves the AI-caller SWML documents, for `place_calls --mode url`.

The default mode (`--mode inline`) sends the SWML inside the create-call request and needs no server.
Use this only if you'd rather SignalWire fetch the document from our tunnel. To mount it (integrator,
in line/server.py; this unit does not edit line/):

    from callers.signalwire.swml_router import router as signalwire_swml_router
    app.include_router(signalwire_swml_router)

Routes:
  GET|POST /callers/swml/{persona}/{variant}?fill=N   -> the SWML document (application/json)
  GET      /callers/swml                              -> the persona x variant catalog

SignalWire fetches `url` with `url_method` (we send GET; POST is accepted too, its body is ignored):
https://signalwire.com/docs/apis/rest/calls/call-commands . The documents hold only test prompts, no
secrets or numbers, so the route is unauthenticated.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from callers.signalwire import personas as P

router = APIRouter(prefix="/callers/swml", tags=["signalwire-callers"])


@router.get("")
def index() -> JSONResponse:
    return JSONResponse(P.catalog())


@router.api_route("/{persona}/{variant}", methods=["GET", "POST"])
def swml(persona: str, variant: int, fill: int = Query(0, ge=0)) -> JSONResponse:
    try:
        doc = P.build_swml(persona, variant, fill)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e).strip("'\""))
    return JSONResponse(doc)

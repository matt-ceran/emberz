"""swml_router: serves the same documents place_calls sends inline."""
from fastapi import FastAPI
from fastapi.testclient import TestClient

from callers.signalwire import personas as P
from callers.signalwire.swml_router import router

def make_client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_get_and_post_return_the_document():
    c = make_client()
    r = c.get("/callers/swml/bank_fraud/2", params={"fill": 3})
    assert r.status_code == 200 and r.json() == P.build_swml("bank_fraud", 2, 3)
    assert c.post("/callers/swml/pharmacy_refill/0").json() == P.build_swml("pharmacy_refill", 0, 0)


def test_unknown_is_404():
    c = make_client()
    assert c.get("/callers/swml/grooming/0").status_code == 404
    assert c.get("/callers/swml/bank_fraud/7").status_code == 404
    assert c.get("/callers/swml/bank_fraud/0", params={"fill": 99}).status_code == 404
    assert c.get("/callers/swml/bank_fraud/0", params={"fill": -1}).status_code == 422


def test_index():
    rows = make_client().get("/callers/swml").json()
    rows = [r for r in rows if r["persona"] not in P.DEMO_PERSONAS]  # demo-only personas are extra
    assert len(rows) == 15 and {r["persona"] for r in rows} == set(P.PERSONAS)

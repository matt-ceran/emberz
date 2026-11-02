"""The flow through the REAL TelnyxClient, with httpx.MockTransport instead of the network.

Checks the exact HTTP paths and JSON bodies Telnyx would receive (FakeTelnyx only records kwargs).
"""
import json

import httpx
from fastapi.testclient import TestClient

from callflow_helpers import A_CCID, CALL_ID, CALLER, GUARDED, PROTECTED, Line
from line.server import create_app
from line.telnyx_client import TelnyxClient


def test_real_client_http_requests():
    sent: list[tuple[str, str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        sent.append((request.method, request.url.path, body))
        assert request.headers["authorization"] == "Bearer test-key"
        if request.url.path == "/v2/calls":
            return httpx.Response(200, json={"data": {"call_control_id": "v3:B-leg"}})
        if request.url.path == "/v2/conferences":
            return httpx.Response(200, json={"data": {"id": "conf-123", "name": body["name"]}})
        return httpx.Response(200, json={"data": {"result": "ok"}})

    client = TelnyxClient(api_key="test-key", connection_id="3051333815776576529")
    client._http = httpx.AsyncClient(base_url="https://api.telnyx.com/v2", transport=httpx.MockTransport(handler),
                                     headers={"Authorization": "Bearer test-key"})
    helper = Line()  # reuse its settings/profiles and payload builders
    app = create_app(telnyx=client, settings=helper.settings, profiles=helper.app.state.flow.profiles)
    helper.client = TestClient(app)

    helper.inbound()
    helper.post("call.answered", helper.leg("A", A_CCID))
    helper.post("call.speak.ended", helper.leg("A", A_CCID))
    helper.post("call.answered", helper.leg("B", "v3:B-leg"))
    helper.post("call.hangup", helper.leg("A", A_CCID, hangup_cause="normal_clearing"))

    paths = [(m, p) for m, p, _ in sent]
    assert paths == [
        ("POST", f"/v2/calls/{A_CCID}/actions/answer"),
        ("POST", f"/v2/calls/{A_CCID}/actions/streaming_start"),
        ("POST", f"/v2/calls/{A_CCID}/actions/speak"),
        ("POST", "/v2/calls"),
        ("POST", "/v2/conferences"),
        ("POST", "/v2/conferences/conf-123/actions/join"),
        ("POST", "/v2/calls/v3:B-leg/actions/hangup"),
    ]
    dial = sent[3][2]
    assert dial == {"to": PROTECTED, "from": CALLER, "diversion": GUARDED, "timeout_secs": 30,
                    "time_limit_secs": 1800,
                    "client_state": dial["client_state"], "connection_id": "3051333815776576529"}
    # The flow passes command_id to dial, but the shared TelnyxClient.dial() drops it today.
    # If the client is changed to keep it, this line flips (and Telnyx dedupes duplicate dials).
    assert "command_id" not in dial
    conf = sent[4][2]
    assert conf["call_control_id"] == A_CCID and conf["command_id"] == f"{CALL_ID}:conference"
    assert sent[5][2]["call_control_id"] == "v3:B-leg" and sent[5][2]["end_conference_on_exit"] is True
    assert all(b.get("command_id") for m, p, b in sent if p != "/v2/calls")

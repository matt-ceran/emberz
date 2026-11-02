"""The --live path against fake ElevenLabs/SignalWire clients: order of requests, reuse, ledger, and
that agents/numbers we didn't create are never modified. No real network."""
import json

import pytest

from callers.elevenlabs import agents as A
from callers.elevenlabs import place_calls as P

CALLER, GUARDED = "+13046481098", "+13235161282"


class Resp:
    def __init__(self, status, data):
        self.status_code, self._data = status, data
        self.content = b"x"
        self.text = json.dumps(data)

    def json(self):
        return self._data


class FakeEL:
    def __init__(self, log, agents=None, numbers=None):
        self.log, self.agents, self.numbers, self.n = log, agents or [], numbers or [], 0

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get(self, path, params=None):
        self.log.append(("el", "GET", path))
        if path == "/v1/convai/agents":
            return Resp(200, {"agents": [a for a in self.agents if params["search"] in a["name"]], "has_more": False})
        if path.startswith("/v1/convai/agents/"):
            aid = path.rsplit("/", 1)[1]
            hit = [a for a in self.agents if a["agent_id"] == aid]
            return Resp(200, hit[0]) if hit else Resp(404, {})
        if path == "/v1/convai/phone-numbers":
            return Resp(200, self.numbers)
        if path.startswith("/v1/convai/phone-numbers/"):
            pid = path.rsplit("/", 1)[1]
            hit = [n for n in self.numbers if n["phone_number_id"] == pid]
            return Resp(200, hit[0]) if hit else Resp(404, {})
        if path == "/v1/convai/conversations":
            return Resp(200, {"conversations": [{"conversation_id": "conv_1"}]})
        raise AssertionError(path)

    def post(self, path, json=None):
        self.log.append(("el", "POST", path, json))
        self.n += 1
        if path == "/v1/convai/agents/create":
            aid = f"agent_new{self.n}"
            self.agents.append({"agent_id": aid, "name": json["name"], "tags": json["tags"]})
            return Resp(200, {"agent_id": aid})
        if path == "/v1/convai/phone-numbers":
            self.numbers.append({"phone_number_id": "pn_1", **json})
            return Resp(200, {"phone_number_id": "pn_1"})
        raise AssertionError(path)

    def patch(self, path, json=None):
        self.log.append(("el", "PATCH", path, json))
        return Resp(200, {})


class FakeSW:
    def __init__(self, log):
        self.log = log

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def post(self, path, json=None):
        self.log.append(("sw", "POST", path, json))
        return Resp(200, {"id": f"sw-{len(self.log)}", "status": "queued"})


@pytest.fixture
def live(monkeypatch, tmp_path):
    log = []
    env = {"SIGNALWIRE_CALLER_NUMBER": CALLER, "TELNYX_GUARDED_NUMBER": GUARDED,
           "EMBERZ_EL_SIP_PASSWORD": "pw-123", "ELEVENLABS_API_KEY": "k", "SIGNALWIRE_SPACE": "s",
           "SIGNALWIRE_PROJECT_ID": "p", "SIGNALWIRE_API_TOKEN": "t"}
    monkeypatch.setattr(P, "load_env", lambda: env)
    monkeypatch.setattr(P, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(P, "LEDGER_PATH", tmp_path / "ai_calls.jsonl")
    monkeypatch.setattr(P, "OUTCOMES_PATH", tmp_path / "outcomes.jsonl")
    monkeypatch.setattr(P, "wait_for_end", lambda sw, cid, limit: "completed")
    monkeypatch.setattr(P.time, "sleep", lambda s: None)
    holder = {"el": FakeEL(log)}
    monkeypatch.setattr(P, "el_client", lambda env: holder["el"])
    monkeypatch.setattr(P, "sw_client", lambda env: FakeSW(log))
    return log, holder, tmp_path


def test_live_flow_creates_assigns_then_dials(live):
    log, holder, tmp = live
    assert P.main(["call", "--personas", "pharmacy_refill", "--variants", "0-1", "--live"]) == 0
    creates = [e for e in log if e[:3] == ("el", "POST", "/v1/convai/agents/create")]
    assert len(creates) == 2                                     # voices 0 and 1
    imports = [e for e in log if e[:3] == ("el", "POST", "/v1/convai/phone-numbers")]
    assert len(imports) == 1 and imports[0][3]["inbound_trunk_config"]["credentials"]["password"] == "pw-123"
    # each dial is preceded by assigning the call's agent to the bridge number
    seq = [(e[0], e[1], e[2]) for e in log if e[1] in ("PATCH", "POST") and "agents/create" not in e[2]
           and e[2] != "/v1/convai/phone-numbers"]
    assert seq == [("el", "PATCH", "/v1/convai/phone-numbers/pn_1"), ("sw", "POST", "/api/calling/calls")] * 2
    dials = [e[3] for e in log if e[0] == "sw"]
    assert all(d["params"]["to"] == GUARDED and d["params"]["from"] == CALLER for d in dials)
    rows = [json.loads(line) for line in (tmp / "ai_calls.jsonl").read_text().splitlines()]
    assert [r["variant"] for r in rows] == [0, 1] and all(r["provider"] == "elevenlabs" for r in rows)
    assert all(r["exclude_from_training"] for r in rows)
    assert "pw-123" not in (tmp / "ai_calls.jsonl").read_text()
    # second run reuses everything from state: no creates, no import
    log.clear()
    assert P.main(["call", "--personas", "pharmacy_refill", "--variants", "0", "--live"]) == 0
    assert not [e for e in log if e[1] == "POST" and e[0] == "el"]


def test_existing_foreign_agents_and_numbers_are_never_touched(live):
    log, holder, tmp = live
    foreign_agent = {"agent_id": "agent_emberdial", "name": "EmberDial LOCAL practice", "tags": ["emberdial-local-practice"]}
    holder["el"] = FakeEL(log, agents=[foreign_agent])
    assert P.main(["call", "--personas", "appointment_confirm", "--variants", "0", "--live"]) == 0
    touched = [e for e in log if e[1] == "PATCH" and "agent_emberdial" in e[2]]
    assert not touched
    assert all("agent_emberdial" != (e[3] or {}).get("agent_id") for e in log if e[1] == "PATCH")


def test_refuses_when_caller_number_already_imported_under_another_label(live):
    log, holder, tmp = live
    holder["el"] = FakeEL(log, numbers=[{"phone_number_id": "pn_x", "phone_number": CALLER, "label": "someone else"}])
    with pytest.raises(SystemExit):
        P.main(["call", "--personas", "pharmacy_refill", "--variants", "0", "--live"])
    assert not [e for e in log if e[0] == "sw"]


def test_live_requires_sip_password(live, monkeypatch):
    monkeypatch.setattr(P, "load_env", lambda: {"SIGNALWIRE_CALLER_NUMBER": CALLER, "TELNYX_GUARDED_NUMBER": GUARDED})
    with pytest.raises(SystemExit):
        P.main(["call", "--live"])

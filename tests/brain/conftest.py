"""Fixtures for the L2 brain tests."""
from __future__ import annotations

import json

import pytest
from brain_fakes import ROOT

from contracts.models import Turn

FIXTURE = ROOT / "contracts" / "fixtures" / "bank_scam_call.json"


@pytest.fixture
def fixture_call():
    return json.loads(FIXTURE.read_text())


@pytest.fixture
def fixture_turns(fixture_call):
    return [Turn.model_validate(t) for t in fixture_call["turns"]]

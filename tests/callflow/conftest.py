"""Fixtures for the L1 call-flow tests."""
import sys
from pathlib import Path

import pytest
from nacl.signing import SigningKey

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from callflow_helpers import Line  # noqa: E402


@pytest.fixture
def line():
    return Line()


@pytest.fixture
def signing_key():
    return SigningKey.generate()

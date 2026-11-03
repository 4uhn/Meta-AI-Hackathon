"""Shared fixtures."""

from __future__ import annotations

import pytest

from horizon.fl import rundir


@pytest.fixture(autouse=True)
def _default_consortium():
    """Some tests activate a larger consortium in-process; every test starts and ends at 2-of-3."""
    rundir.configure(None)
    yield
    rundir.configure(None)

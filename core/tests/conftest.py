"""Shared fixtures for the whole suite."""
import pytest

from classes import llm_lanes


@pytest.fixture(autouse=True)
def _no_lanes_between_tests():
    """A /props refresh in one test lanes a host process-wide (llm_lanes'
    registry); without this a later test's model calls would wait on it."""
    llm_lanes.reset()
    yield
    llm_lanes.reset()


@pytest.fixture(autouse=True)
def _no_real_prewarm(monkeypatch):
    """MessageHandler pre-warms the server's workspace on every handled
    message (WORKSPACE_PREWARM is on by default). A test must never reach
    the real Docker daemon or Redis that way; the workspace manager's own
    tests pass their settings explicitly."""
    monkeypatch.setenv("WORKSPACE_PREWARM", "0")


@pytest.fixture(autouse=True)
def _no_real_status_line(monkeypatch):
    """The brain's card writes its status line with a side call
    (classes/brain_status.py). A test must never reach a real LLM that way:
    it fails at once here, so the card keeps its rule-written line. Tests of
    the line patch _llm.complete themselves."""
    from unittest.mock import AsyncMock
    from classes import brain_status
    monkeypatch.setattr(brain_status._llm, "complete",
                        AsyncMock(side_effect=RuntimeError("no LLM in tests")))

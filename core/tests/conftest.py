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

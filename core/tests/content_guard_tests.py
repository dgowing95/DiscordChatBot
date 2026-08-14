"""Tests for core/classes/content_guard.py (LLM moderation of web tools).

Run from the repo root:

    PYTHONPATH=$(pwd) pytest core/tests/content_guard_tests.py
"""

from unittest.mock import AsyncMock, patch

import pytest

from core.classes.content_guard import (
    _parse_moderation,
    check_web_request,
    moderate_with_llm,
)


# --------------------------- check_web_request ---------------------------

@pytest.mark.asyncio
async def test_blocks_when_llm_moderator_says_unsafe():
    with patch("core.classes.content_guard.moderate_with_llm", new=AsyncMock(return_value=False)):
        allowed, reason = await check_web_request("totally benign sounding query")
    assert not allowed and "safety filter" in reason


@pytest.mark.asyncio
async def test_allows_when_llm_moderator_says_safe():
    with patch("core.classes.content_guard.moderate_with_llm", new=AsyncMock(return_value=True)) as mock_llm:
        allowed, reason = await check_web_request("python asyncio tutorial")
    assert allowed and reason == ""
    mock_llm.assert_awaited_once()


@pytest.mark.asyncio
async def test_guard_can_be_disabled_via_env(monkeypatch):
    monkeypatch.setenv("CONTENT_GUARD_ENABLED", "0")
    with patch("core.classes.content_guard.moderate_with_llm", new=AsyncMock(return_value=False)) as mock_llm:
        allowed, reason = await check_web_request("anything at all")
    assert allowed and reason == ""
    mock_llm.assert_not_awaited()


# --------------------------- _parse_moderation ---------------------------

def test_parses_response_words():
    assert _parse_moderation("   UNSAFE\n") is False
    assert _parse_moderation("SAFE") is True
    # Ambiguous / empty -> fail open
    assert _parse_moderation("I cannot decide") is True
    assert _parse_moderation("") is True


# --------------------------- moderate_with_llm ---------------------------


class _FakeResponse:
    def __init__(self, data=None, raise_json=False):
        self._data = data
        self._raise = raise_json

    async def json(self):
        if self._raise:
            raise RuntimeError("boom")
        return self._data


class _FakePostResult:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    def __init__(self, response=None, raise_on_post=False):
        self._response = response
        self._raise = raise_on_post

    def post(self, *args, **kwargs):
        if self._raise:
            raise RuntimeError("boom")
        return _FakePostResult(self._response)


def _fake_client_session(response=None, raise_on_post=False):
    class _Ctx:
        async def __aenter__(self):
            return _FakeSession(response=response, raise_on_post=raise_on_post)

        async def __aexit__(self, *exc):
            return False

    return _Ctx()


@pytest.mark.asyncio
async def test_moderator_fail_open_on_http_error():
    with patch("aiohttp.ClientSession", return_value=_fake_client_session(raise_on_post=True)):
        assert await moderate_with_llm("anything") is True


@pytest.mark.asyncio
async def test_moderator_fail_open_on_malformed_response():
    with patch("aiohttp.ClientSession",
               return_value=_fake_client_session(response=_FakeResponse({"unexpected": "shape"}))):
        assert await moderate_with_llm("anything") is True


@pytest.mark.asyncio
async def test_moderator_parses_unsafe_response():
    with patch("aiohttp.ClientSession",
               return_value=_fake_client_session(response=_FakeResponse(
                   {"choices": [{"message": {"content": "UNSAFE"}}]}))):
        assert await moderate_with_llm("anything") is False


@pytest.mark.asyncio
async def test_moderator_parses_safe_response():
    with patch("aiohttp.ClientSession",
               return_value=_fake_client_session(response=_FakeResponse(
                   {"choices": [{"message": {"content": "SAFE"}}]}))):
        assert await moderate_with_llm("anything") is True

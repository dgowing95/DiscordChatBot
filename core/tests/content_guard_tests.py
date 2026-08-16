"""Tests for core/classes/content_guard.py (OpenAI moderation of web tools).

Run from the repo root:

    PYTHONPATH=$(pwd) pytest core/tests/content_guard_tests.py
"""

from unittest.mock import AsyncMock, patch

import pytest

import core.classes.content_guard as guard
from core.classes.content_guard import (
    _is_unsafe,
    check_web_request,
    moderate_with_openai,
)


@pytest.fixture(autouse=True)
def _fast_retries(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("CONTENT_GUARD_RETRY_BACKOFF", "0.01")
    monkeypatch.setattr(guard, "_verdict_cache", {})


# --------------------------- check_web_request ---------------------------

@pytest.mark.asyncio
async def test_blocks_when_moderation_flags_unsafe(monkeypatch):
    monkeypatch.setattr(guard, "_verdict_cache", {})
    with patch("core.classes.content_guard.moderate_with_openai",
               new=AsyncMock(return_value=False)):
        allowed, reason = await check_web_request("totally benign sounding query")
    assert not allowed and "safety filter" in reason


@pytest.mark.asyncio
async def test_allows_when_moderation_is_clean():
    with patch("core.classes.content_guard.moderate_with_openai",
               new=AsyncMock(return_value=True)) as mock_moderation:
        allowed, reason = await check_web_request("python asyncio tutorial")
    assert allowed and reason == ""
    mock_moderation.assert_awaited_once()


@pytest.mark.asyncio
async def test_guard_can_be_disabled_via_env(monkeypatch):
    monkeypatch.setenv("CONTENT_GUARD_ENABLED", "0")
    with patch("core.classes.content_guard.moderate_with_openai",
               new=AsyncMock(return_value=False)) as mock_moderation:
        allowed, reason = await check_web_request("anything at all")
    assert allowed and reason == ""
    mock_moderation.assert_not_awaited()


# --------------------------- _is_unsafe ---------------------------

def test_unsafe_when_any_category_flagged():
    assert _is_unsafe({"categories": {"self-harm": True, "harassment": False}})
    assert _is_unsafe({"flagged": True, "categories": {}})


def test_safe_when_no_categories_flagged():
    assert not _is_unsafe({"flagged": False, "categories": {
        "self-harm": False, "hate": False, "sexual": False}})
    assert not _is_unsafe({"categories": {}})


def test_unsafe_handles_malformed_result():
    assert not _is_unsafe(None)
    assert not _is_unsafe({"categories": "not-a-dict"})
    assert not _is_unsafe({"flagged": False})


# --------------------------- moderate_with_openai ---------------------------


class _FakeResponse:
    def __init__(self, data=None, raise_json=False, status=200, headers=None):
        self._data = data
        self._raise = raise_json
        self.status = status
        self.headers = headers or {}

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
    def __init__(self, responses=None, raise_on_post=False, calls=None):
        # ``responses`` is a (possibly empty) list; each call pops one,
        # the last one is reused if it runs out.
        self._responses = list(responses or [])
        self._raise = raise_on_post
        self.call_count = 0
        self._calls = calls

    def post(self, *args, **kwargs):
        if self._calls is not None:
            self._calls.append((args, kwargs))
        if self._raise and self.call_count == 0:
            raise RuntimeError("boom")
        self.call_count += 1
        if self._responses:
            if len(self._responses) > 1:
                self._responses.pop(0)
            return _FakePostResult(self._responses[0])
        return _FakePostResult(None)


def _fake_client_session(responses=None, raise_on_post=False, calls=None):
    class _Ctx:
        async def __aenter__(self):
            return _FakeSession(
                responses=responses, raise_on_post=raise_on_post, calls=calls)

        async def __aexit__(self, *exc):
            return False

    return _Ctx()


def _moderation_response(flagged, categories):
    return _FakeResponse(data={
        "results": [{"flagged": flagged, "categories": categories}]})


def _error_response(message="Too Many Requests"):
    return _FakeResponse(data={"error": {"message": message, "type": "invalid_request_error"}})


@pytest.mark.asyncio
async def test_fail_open_when_api_key_missing(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    assert await moderate_with_openai("anything") is True


@pytest.mark.asyncio
async def test_fail_open_on_network_error():
    with patch("aiohttp.ClientSession",
               return_value=_fake_client_session(raise_on_post=True)):
        assert await moderate_with_openai("anything") is True


@pytest.mark.asyncio
async def test_retries_429_then_succeeds():
    responses = [
        _error_response(),               # 429 first
        _moderation_response(False, {"sexual": False}),
    ]
    with patch("aiohttp.ClientSession",
               return_value=_fake_client_session(responses=responses)):
        assert await moderate_with_openai("anything") is True


@pytest.mark.asyncio
async def test_fails_open_after_retries_exhausted():
    responses = [_error_response(), _error_response(), _error_response()]
    with patch("aiohttp.ClientSession",
               return_value=_fake_client_session(responses=responses)):
        assert await moderate_with_openai("anything") is True


@pytest.mark.asyncio
async def test_no_retry_on_non_retriable_status():
    responses = [_FakeResponse(data={"error": {"message": "bad key"}}, status=401)]
    calls = []
    with patch("aiohttp.ClientSession",
               return_value=_fake_client_session(responses=responses, calls=calls)):
        assert await moderate_with_openai("anything") is True
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_blocks_when_category_detected():
    with patch("aiohttp.ClientSession",
               return_value=_fake_client_session(responses=[
                   _moderation_response(True, {"violence": False, "sexual": True})])):
        assert await moderate_with_openai("anything") is False


@pytest.mark.asyncio
async def test_allows_when_no_category_detected():
    with patch("aiohttp.ClientSession",
               return_value=_fake_client_session(responses=[
                   _moderation_response(False, {
                       "violence": False, "sexual": False, "hate": False})])):
        assert await moderate_with_openai("anything") is True


@pytest.mark.asyncio
async def test_verdict_is_cached(monkeypatch):
    responses = [_moderation_response(False, {"sexual": False})]
    with patch("aiohttp.ClientSession",
               return_value=_fake_client_session(responses=responses)):
        assert await moderate_with_openai("same input") is True
        monkeypatch.setenv("OPENAI_API_KEY", "")  # prove the 2nd call was cached
    assert await moderate_with_openai("same input") is True
    # a different input is still sent to the API (and fails open)
    assert await moderate_with_openai("different input") is True

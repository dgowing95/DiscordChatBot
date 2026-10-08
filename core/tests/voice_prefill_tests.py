"""
The voice cache warm-up only works if the prefill sends exactly the start of
the request the next voice turn sends (TextLLMHandler.prefill vs
generate_streamed). These tests capture both requests at the OpenAI client
and compare them.

This shows the prompt is stable, not that llama.cpp reuses its cache: for
that, watch llamacpp:prompt_tokens_cached_total or the turn's `cache_n` on a
live server (measured on the dev stack: 2427 prompt tokens -> 52).

Run from the repo root:
    pytest core/tests/voice_prefill_tests.py
"""
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from openai import BadRequestError
from agents import AsyncOpenAI, OpenAIChatCompletionsModel

from classes import text_llm_handler
from classes.user_memory import UserMemory


class Captured(Exception):
    pass


@pytest.fixture
def captured(monkeypatch):
    monkeypatch.setenv("VOICE_ENABLED", "1")
    monkeypatch.setenv("SANDBOX_ENABLED", "0")
    monkeypatch.setenv("IMAGE_GEN_ENABLED", "0")
    calls = []
    client = AsyncOpenAI(base_url="http://llm.invalid/v1", api_key="x")

    async def create(**kwargs):
        calls.append(kwargs)
        raise Captured()

    client.chat.completions.create = create
    model = OpenAIChatCompletionsModel(model="m", openai_client=client)
    settings = {"system": "a pirate", "temperature": "0.7"}
    with patch.object(text_llm_handler, "_main_model_client", model), \
         patch.object(text_llm_handler.configManager, "get_setting",
                      AsyncMock(side_effect=lambda name, guild: settings.get(name))), \
         patch.object(UserMemory, "get", AsyncMock(return_value=[])):
        yield calls


HISTORY = [
    {"role": "user", "content": "[Ana]: we should get pizza"},
    {"role": "user", "content": "[Bo]: hey sparky, what's a good pizza place?"},
]


def _handler(**kwargs):
    return text_llm_handler.TextLLMHandler(list(HISTORY), 42, None, actor_id=7, channel=object(),
                                           voice=True, bot_name="Sparky", **kwargs)


@pytest.mark.asyncio
async def test_prefill_is_the_exact_prefix_of_the_next_turn(captured):
    with pytest.raises(Captured):
        await _handler().prefill()
    assert await _handler(request_text="what's a good pizza place?").generate_streamed(
        AsyncMock(), AsyncMock(), AsyncMock()) == "Error"
    prefill, turn = captured
    # The turn is the prefill's messages plus the trailing datetime message.
    assert turn["messages"][:-1] == prefill["messages"]
    assert turn["messages"][-1]["content"].startswith("(Current datetime:")
    # Everything else that shapes the rendered prompt is identical.
    for key in ("tools", "extra_body", "temperature", "top_p", "frequency_penalty", "model"):
        assert turn.get(key) == prefill.get(key), key
    assert prefill["max_tokens"] == 1
    assert turn.get("stream") is True


@pytest.mark.asyncio
async def test_voice_turns_ask_for_no_thinking(captured):
    with pytest.raises(Captured):
        await _handler().prefill()
    request = captured[0]
    assert request["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    # Not sent at all (the client's omit marker), rather than "low".
    assert not isinstance(request.get("reasoning_effort"), str)
    system = request["messages"][0]["content"]
    assert system.startswith("Answer as if you are a pirate.")
    assert "voice call, where people call you Sparky;" in system


@pytest.mark.asyncio
async def test_voice_thinking_can_be_turned_back_on(captured, monkeypatch):
    monkeypatch.setenv("VOICE_THINKING", "1")
    with pytest.raises(Captured):
        await _handler().prefill()
    assert "extra_body" not in captured[0] or not captured[0]["extra_body"]


@pytest.mark.asyncio
async def test_voice_tools_leave_rather_than_join(captured):
    with pytest.raises(Captured):
        await _handler().prefill()
    names = {tool["function"]["name"] for tool in captured[0]["tools"]}
    assert "leave_voice_channel" in names and "join_voice_channel" not in names
    assert "web_search" in names and "store_memory" in names


@pytest.mark.asyncio
async def test_a_failed_retry_leaves_thinking_off(captured, monkeypatch):
    # The first request is refused, and so is the retry without the option:
    # nothing shows the option was the problem, so later voice turns must
    # still be sent it (side_llm's latch rule).
    monkeypatch.setitem(text_llm_handler._voice_latch, "send_no_thinking", True)
    request = httpx.Request("POST", "http://llm.invalid/v1/chat/completions")
    refused = BadRequestError("refused", response=httpx.Response(400, request=request), body=None)
    model = text_llm_handler._main_model_client
    sent = []

    async def create(**kwargs):
        sent.append(kwargs)
        raise refused

    with patch.object(model._client.chat.completions, "create", create):
        assert await _handler(request_text="hi").generate_streamed(
            AsyncMock(), AsyncMock(), AsyncMock()) == "Error"
    assert len(sent) == 2
    assert sent[0]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert not sent[1].get("extra_body")
    assert text_llm_handler._voice_latch["send_no_thinking"] is True

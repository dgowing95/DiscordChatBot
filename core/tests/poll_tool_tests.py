"""Tests for the create_poll agent tool (core/classes/tool_functions.py).

To run this pytest file from the command line, use:
    pytest core/tests/poll_tool_tests.py
"""
import json
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock

import discord
import pytest


def _tool_context(context):
    from agents.tool_context import ToolContext
    return ToolContext(context=context, tool_name="create_poll",
                       tool_call_id="t1", tool_arguments="{}")


def _env(message=None):
    channel = MagicMock()
    channel.send = AsyncMock()
    return channel, {"channel": channel, "original_message": message, "poll_tool_calls": 0}


async def _invoke(context, **arguments):
    from classes import tool_functions
    return await tool_functions.create_poll.on_invoke_tool(
        _tool_context(context), json.dumps(arguments))


def _sent_poll(channel) -> discord.Poll:
    return channel.send.await_args.kwargs["poll"]


@pytest.mark.asyncio
async def test_posts_the_poll_and_reacts():
    message = MagicMock()
    message.add_reaction = AsyncMock()
    channel, context = _env(message)

    result = await _invoke(context, question="Lunch?", answers=["Pizza", "Sushi", "pizza"],
                           duration_hours=6, allow_multiple=True)

    poll = _sent_poll(channel)
    assert poll.question == "Lunch?"
    assert [a.text for a in poll.answers] == ["Pizza", "Sushi"]
    assert poll.duration == timedelta(hours=6)
    assert poll.multiple is True
    message.add_reaction.assert_awaited_once_with("📊")
    assert result.startswith('Poll posted: "Lunch?" with 2 options, open for 6 hours')
    assert "do not repeat the options" in result


@pytest.mark.asyncio
async def test_defaults_to_one_day_single_choice():
    channel, context = _env()
    await _invoke(context, question="Q", answers=["a", "b"])
    poll = _sent_poll(channel)
    assert poll.duration == timedelta(hours=24)
    assert poll.multiple is False


@pytest.mark.asyncio
async def test_invalid_input_sends_nothing_and_does_not_use_up_the_call():
    channel, context = _env()

    result = await _invoke(context, question="Q", answers=["only one"])

    assert "No poll was posted" in result
    channel.send.assert_not_awaited()
    assert context["poll_tool_calls"] == 0


@pytest.mark.asyncio
async def test_a_second_poll_in_one_reply_is_skipped():
    channel, context = _env()
    await _invoke(context, question="Q", answers=["a", "b"])
    result = await _invoke(context, question="Q", answers=["a", "b"])
    assert channel.send.await_count == 1
    assert "already posted" in result


@pytest.mark.asyncio
async def test_missing_permission_is_reported():
    channel, context = _env()
    channel.send.side_effect = discord.Forbidden(MagicMock(status=403, reason="Forbidden"), "no")

    result = await _invoke(context, question="Q", answers=["a", "b"])

    assert "Create Polls permission" in result
    assert context["poll_tool_calls"] == 0


@pytest.mark.asyncio
async def test_works_without_a_source_message():
    """Voice turns and automatic runs have no message to react to."""
    channel, context = _env()
    context.pop("poll_tool_calls")  # a context built elsewhere may not have the key
    result = await _invoke(context, question="Q", answers=["a", "b"])
    assert result.startswith("Poll posted")
    channel.send.assert_awaited_once()

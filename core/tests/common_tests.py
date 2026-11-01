import os
import sys


import pytest
from unittest.mock import AsyncMock, MagicMock

from classes import common

# To run this pytest file from the command line, use:
# PYTHONPATH=$(pwd) pytest core/tests/common_tests.py


@pytest.mark.asyncio
async def test_send_tool_discord_embed_sends_description_unchanged_under_cap():
    channel = MagicMock()
    channel.send = AsyncMock()
    await common.Common.send_tool_discord_embed(channel, "short description",
                                                title="🔎 Web search")

    channel.send.assert_awaited_once()
    embed = channel.send.await_args.kwargs["embed"]
    assert embed.description == "short description"
    assert embed.title == "🔎 Web search"
    assert embed.footer.text is None


@pytest.mark.asyncio
async def test_send_tool_discord_embed_has_no_generic_title_and_takes_a_footer():
    # the old default "Tool Usage" said nothing about which tool ran
    channel = MagicMock()
    channel.send = AsyncMock()
    await common.Common.send_tool_discord_embed(channel, "x", footer="a hint")

    embed = channel.send.await_args.kwargs["embed"]
    assert embed.title is None
    assert embed.footer.text == "a hint"


@pytest.mark.asyncio
async def test_send_tool_discord_embed_truncates_over_discord_cap():
    # Regression: a task description that pushes the embed description over
    # Discord's 4096-char hard cap raises discord.HTTPException from
    # channel.send() with no guard anywhere in the tool-calling path.
    channel = MagicMock()
    channel.send = AsyncMock()
    long_description = "x" * 5000
    await common.Common.send_tool_discord_embed(channel, long_description)

    embed = channel.send.await_args.kwargs["embed"]
    assert len(embed.description) <= common.EMBED_DESCRIPTION_MAX_CHARS
    assert embed.description.endswith("… ")


@pytest.mark.asyncio
async def test_send_tool_discord_embed_exactly_at_cap_is_unchanged():
    channel = MagicMock()
    channel.send = AsyncMock()
    description = "x" * common.EMBED_DESCRIPTION_MAX_CHARS
    await common.Common.send_tool_discord_embed(channel, description)

    embed = channel.send.await_args.kwargs["embed"]
    assert embed.description == description

"""classes/common.py: the quiet channel a workflow's helpers run with, and
classes/card.py's notice sender that every tool posts through."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from classes import card, card_kit as K
from classes.common import QuietChannel


def _channel():
    channel = MagicMock()
    channel.send = AsyncMock(return_value=MagicMock(edit=AsyncMock()))
    return channel


def _texts(view):
    return "".join(b.content for b in view.blocks if isinstance(b, K.Text))


@pytest.mark.asyncio
async def test_a_notice_is_one_card_with_no_pings():
    channel = _channel()
    live = await card.notice(channel, K.info("🔎 Searching the web", "**keyboards**"))
    kwargs = channel.send.await_args.kwargs
    assert isinstance(kwargs["view"], card.KitView) and kwargs["view"].kind == K.NOTICE
    assert kwargs["allowed_mentions"] == card.NO_MENTIONS
    assert "content" not in kwargs and "embed" not in kwargs
    assert live.message is channel.send.return_value


@pytest.mark.asyncio
async def test_a_notice_with_no_channel_is_skipped():
    assert await card.notice(None, K.info("x")) is None


@pytest.mark.asyncio
async def test_a_long_notice_goes_over_several_messages_and_nothing_is_cut():
    channel = _channel()
    text = "\n\n".join(f"paragraph {i} " + "word " * 150 for i in range(12))
    await card.notice(channel, K.info("📜 Reading messages", text))
    views = [c.kwargs["view"] for c in channel.send.await_args_list]
    assert len(views) > 1 and all(v.content_length() <= K.MAX_TEXT for v in views)
    shown = "".join(_texts(v) for v in views)
    assert all(f"paragraph {i} " in shown for i in range(12))
    # the label is on the first only
    assert sum("Reading messages" in _texts(v) for v in views) == 1


@pytest.mark.asyncio
async def test_a_quiet_channel_drops_notices_only():
    real = _channel()
    quiet = QuietChannel(real)
    notice = card.views(K.info("🔎 Searching the web", "x"))[0]
    dropped = await quiet.send(view=notice)
    assert getattr(dropped, "dropped", False) and real.send.await_count == 0
    await dropped.edit(view=notice)  # editing what was dropped does nothing

    await quiet.send(view=card.Card("Which one?"))
    await quiet.send(view=notice, files=["a file"])
    await quiet.send("plain text")
    assert real.send.await_count == 3


@pytest.mark.asyncio
async def test_a_live_card_edits_only_what_changed_and_knows_a_dropped_one():
    channel = _channel()
    live = card.LiveCard(channel)
    spec = K.info("🔎 Searching the web", "x")
    await live.show(spec)
    await live.show(spec)
    assert live.message.edit.await_count == 0
    await live.show(K.info("🔎 Searched the web", "x"))
    assert live.message.edit.await_count == 1 and not live.dropped

    quiet_live = card.LiveCard(QuietChannel(_channel()))
    await quiet_live.show(spec)
    assert quiet_live.dropped

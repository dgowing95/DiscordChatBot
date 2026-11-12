"""classes/common.py: the quiet channel a workflow's helpers run with, and
classes/card.py's notice sender that every tool posts through."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from classes import card, card_kit as K, tool_cards
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


# --- lookups: the first few cards, then one more-lookups card (card.Lookups) ------

def _lookup(n):
    return K.CardSpec(label="🔍 Searching this server's messages", body=[K.Text(f"“word {n}”")])


@pytest.mark.asyncio
async def test_the_first_lookups_get_their_own_card_and_the_rest_share_one():
    import asyncio
    channel = _channel()
    ctx = {"channel": channel}
    lookups = card.Lookups(channel, shown=3, interval=0.01)
    ctx[card.LOOKUPS] = lookups
    own = [await card.lookup_notice(ctx, _lookup(n)) for n in range(3)]
    assert all(isinstance(live, card.LiveCard) for live in own)
    assert channel.send.await_count == 3

    assert isinstance(await card.lookup_notice(ctx, _lookup(3)), card.LookupItem)
    assert channel.send.await_count == 4  # the more-lookups card, sent at once
    more = channel.send.await_args.kwargs["view"]
    assert "More lookups · 1" in "".join(b.content for b in more.blocks if isinstance(b, K.Text))

    for n in (4, 5):
        assert isinstance(await card.lookup_notice(ctx, _lookup(n)), card.LookupItem)
    await asyncio.sleep(0.05)  # the edit waits out the interval, in the background
    assert channel.send.await_count == 4  # later lookups only edit it
    message = lookups.card.message
    assert message.edit.await_count == 1  # both in one edit
    text = "".join(b.content for b in message.edit.await_args.kwargs["view"].blocks
                   if isinstance(b, K.Text))
    assert "More lookups · 3" in text and "word 3" in text and "word 5" in text


@pytest.mark.asyncio
async def test_the_count_starts_with_the_runs_first_lookup_and_needs_a_channel():
    channel = _channel()
    ctx = {"channel": channel}
    await card.lookup_notice(ctx, _lookup(0))
    assert isinstance(ctx[card.LOOKUPS], card.Lookups) and ctx[card.LOOKUPS].count == 1
    assert ctx[card.LOOKUPS].shown == tool_cards.LOOKUP_CARDS == 3
    assert await card.lookup_notice({"channel": None}, _lookup(0)) is None


@pytest.mark.asyncio
async def test_two_lookups_at_once_past_the_first_few_send_one_more_lookups_card():
    import asyncio

    async def send(**kwargs):
        await asyncio.sleep(0.01)  # the post takes a while
        return MagicMock(edit=AsyncMock())

    channel = MagicMock()
    channel.send = AsyncMock(side_effect=send)
    ctx = {"channel": channel, card.LOOKUPS: card.Lookups(channel, shown=0, interval=0)}
    await asyncio.gather(card.lookup_notice(ctx, _lookup(0)), card.lookup_notice(ctx, _lookup(1)))
    await asyncio.sleep(0.05)
    assert channel.send.await_count == 1


@pytest.mark.asyncio
async def test_a_quiet_channels_more_lookups_card_is_not_edited():
    # A workflow helper's QuietChannel keeps the card: no edits to wait on.
    import asyncio
    channel = _channel()
    lookups = card.Lookups(QuietChannel(channel), shown=0, interval=0)
    ctx = {"channel": lookups.channel, card.LOOKUPS: lookups}
    for n in range(3):
        item = await card.lookup_notice(ctx, _lookup(n))
    item.show_soon(K.finished(_lookup(2), "It didn't work"))
    await asyncio.sleep(0.01)
    assert lookups._task is None and channel.send.await_count == 0


@pytest.mark.asyncio
async def test_a_lookup_in_the_more_lookups_card_is_updated_in_place():
    # A lookup past the first few still shows how it ended ("It didn't work"),
    # in its own item, at the card's next edit.
    import asyncio
    channel = _channel()
    lookups = card.Lookups(channel, shown=0, interval=0.01)
    ctx = {"channel": channel, card.LOOKUPS: lookups}
    first = await card.lookup_notice(ctx, _lookup(0))
    await card.lookup_notice(ctx, _lookup(1))
    first.show_soon(K.finished(_lookup(0), "It didn't work"))
    await asyncio.sleep(0.05)
    texts = [b.content for b in lookups.card.message.edit.await_args.kwargs["view"].blocks
             if isinstance(b, K.Text)]
    assert texts[1:] == ["-# 🔍 Searching this server's messages\n“word 0”\n-# It didn't work",
                         "-# 🔍 Searching this server's messages\n“word 1”"]
    assert channel.send.await_count == 1

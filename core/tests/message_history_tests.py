"""Tests for MessageHandler.build_messages: which history goes into a prompt.

Runs the REAL build_messages against a fake channel whose history() follows
discord.py's semantics (before/after are exclusive; with after= the OLDEST
`limit` messages are kept, otherwise the newest), so the sliding and anchored
paths are exercised end to end without Discord.

To run this pytest file from the command line, use:
    pytest core/tests/message_history_tests.py
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from classes import history_policy
from classes.history_policy import AnchorState
from classes.message_handler import MessageHandler

BOT_ID = 999
GUILD_ID = 1
CHANNEL_ID = 77


class FakeChannel:
    def __init__(self, channel_id=CHANNEL_ID):
        self.id = channel_id
        self.messages = []
        self.history_calls = []

    def add(self, mid, content, author="alice", author_id=1, embeds=()):
        msg = SimpleNamespace(
            id=mid, content=content, channel=self,
            guild=SimpleNamespace(id=GUILD_ID),
            author=SimpleNamespace(id=author_id, name=author),
            attachments=[], embeds=list(embeds),
        )
        self.messages.append(msg)
        return msg

    def reply(self, mid, content):
        return self.add(mid, content, author="bot", author_id=BOT_ID)

    def history(self, limit=100, before=None, after=None, oldest_first=None):
        self.history_calls.append({"limit": limit, "before": before, "after": after,
                                   "oldest_first": oldest_first})
        if oldest_first is None:
            oldest_first = after is not None
        lo = after.id if after is not None else -1
        hi = before.id if before is not None else float("inf")
        found = sorted((m for m in self.messages if lo < m.id < hi), key=lambda m: m.id)
        if oldest_first:
            found = found[:limit]
        else:
            found = list(reversed(found))[:limit]

        async def _gen():
            for m in found:
                yield m
        return _gen()


def _embed(title):
    embed = MagicMock()
    embed.to_dict.return_value = {"title": title, "fields": [{"name": "dropped"}]}
    return embed


def _client():
    client = MagicMock()
    client.user.id = BOT_ID
    return client


async def _build(trigger):
    handler = MessageHandler(trigger, _client())
    await handler.build_messages()
    return handler.messages


def _texts(messages):
    return [m["content"] for m in messages]


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    history_policy.store().clear()
    for name in ("MSG_HISTORY_MODE", "MSG_HISTORY_LIMIT", "MSG_HISTORY_REFRESH_MESSAGES",
                 "MSG_HISTORY_RESERVE_TOKENS"):
        monkeypatch.delenv(name, raising=False)
    # capacity unknown unless a test sets it: no trimming
    monkeypatch.setattr("classes.message_handler.slot_context_tokens", lambda: None)
    yield
    history_policy.store().clear()


# ------------------------------------------------------- correctness (both modes)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sliding", "anchored"])
async def test_queued_trigger_gets_the_history_before_it(monkeypatch, mode):
    """The trigger waited in the queue while newer messages arrived: history is
    what came BEFORE it, and it appears exactly once, last."""
    monkeypatch.setenv("MSG_HISTORY_MODE", mode)
    monkeypatch.setenv("MSG_HISTORY_LIMIT", "4")
    ch = FakeChannel()
    for i in range(1, 11):
        ch.add(i, f"m{i}")
    trigger = ch.messages[4]  # id 5; 6..10 are newer

    texts = _texts(await _build(trigger))

    assert texts == ["Message from 'alice': m2", "Message from 'alice': m3",
                     "Message from 'alice': m4", "Message from 'alice': m5"]


@pytest.mark.asyncio
async def test_empty_history():
    ch = FakeChannel()
    trigger = ch.add(1, f"<@{BOT_ID}> hello")
    messages = await _build(trigger)
    assert messages == [{"role": "user", "content": "Message from 'alice': hello"}]


@pytest.mark.asyncio
async def test_limit_of_one_reads_no_history(monkeypatch):
    monkeypatch.setenv("MSG_HISTORY_LIMIT", "1")
    ch = FakeChannel()
    ch.add(1, "old")
    trigger = ch.add(2, "new")
    assert _texts(await _build(trigger)) == ["Message from 'alice': new"]
    assert ch.history_calls == []


@pytest.mark.asyncio
async def test_chronological_with_text_before_embeds_in_order():
    ch = FakeChannel()
    ch.reply(1, "first")
    ch.add(2, "second", embeds=[_embed("A"), _embed("B")])
    trigger = ch.add(3, "third")

    messages = await _build(trigger)

    assert [m["role"] for m in messages] == ["assistant", "user", "user", "user", "user"]
    texts = _texts(messages)
    assert texts[0] == "Message from 'bot': first"
    assert texts[1] == "Message from 'alice': second"
    assert '"title": "A"' in texts[2] and '"title": "B"' in texts[3]
    assert "dropped" not in texts[2]
    assert texts[4] == "Message from 'alice': third"


@pytest.mark.asyncio
async def test_trigger_serializes_like_it_will_as_history():
    """Turn N's trigger must be byte-identical to how turn N+1 renders it as
    history, embeds included, or the server's cached prefix breaks there."""
    ch = FakeChannel()
    first = ch.add(1, f"<@{BOT_ID}> look", embeds=[_embed("E")])
    as_trigger = await _build(first)
    ch.reply(2, "seen")
    second = ch.add(3, "and?")
    as_history = await _build(second)
    assert as_history[:len(as_trigger)] == as_trigger


@pytest.mark.asyncio
async def test_does_not_mutate_message_content():
    ch = FakeChannel()
    trigger = ch.add(1, f"<@{BOT_ID}>  hi ")
    await _build(trigger)
    assert trigger.content == f"<@{BOT_ID}>  hi "


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sliding", "anchored"])
async def test_reset_hides_everything_before_it(monkeypatch, mode):
    monkeypatch.setenv("MSG_HISTORY_MODE", mode)
    ch = FakeChannel()
    ch.add(1, "secret")
    ch.add(2, "!reset_history")
    ch.add(3, "after")
    trigger = ch.add(4, "now")
    assert _texts(await _build(trigger)) == ["Message from 'alice': after",
                                             "Message from 'alice': now"]


# ------------------------------------------------------------- anchored history


def _turn(ch, next_id, text):
    """Bot replies to the previous trigger, then a user sends `text`."""
    ch.reply(next_id, f"reply before {text}")
    return ch.add(next_id + 1, text)


@pytest.mark.asyncio
async def test_anchored_prompt_only_grows_until_the_refresh(monkeypatch):
    monkeypatch.setenv("MSG_HISTORY_MODE", "anchored")
    monkeypatch.setenv("MSG_HISTORY_LIMIT", "4")
    monkeypatch.setenv("MSG_HISTORY_REFRESH_MESSAGES", "6")
    ch = FakeChannel()
    for i in range(1, 6):
        ch.add(i, f"m{i}")
    trigger = ch.add(6, "t0")
    previous = await _build(trigger)  # cold refresh: m3 m4 m5 + t0
    assert len(previous) == 4
    next_id = 7
    # each turn adds 2 messages (reply + trigger); 6 new messages refresh on turn 3
    for turn in (1, 2):
        trigger = _turn(ch, next_id, f"t{turn}")
        next_id += 2
        prompt = await _build(trigger)
        assert prompt[:len(previous)] == previous, f"prefix changed on turn {turn}"
        assert len(prompt) == len(previous) + 2
        previous = prompt
    trigger = _turn(ch, next_id, "t3")
    refreshed = await _build(trigger)
    assert refreshed[0] != previous[0]
    assert len(refreshed) == 4  # back to the newest window


@pytest.mark.asyncio
async def test_anchored_burst_refreshes_with_a_bounded_window(monkeypatch):
    monkeypatch.setenv("MSG_HISTORY_MODE", "anchored")
    monkeypatch.setenv("MSG_HISTORY_LIMIT", "4")
    monkeypatch.setenv("MSG_HISTORY_REFRESH_MESSAGES", "5")
    ch = FakeChannel()
    trigger = ch.add(1, "t0")
    await _build(trigger)
    for i in range(2, 60):
        ch.add(i, f"burst {i}")
    trigger = ch.add(60, "after the burst")
    prompt = await _build(trigger)
    assert len(prompt) == 4
    assert _texts(prompt)[0] == "Message from 'alice': burst 57"
    fetch_after = [c for c in ch.history_calls if c["after"] is not None]
    assert fetch_after[-1]["limit"] == history_policy.max_window(4, 5) + 1


@pytest.mark.asyncio
async def test_anchored_reset_after_anchor_refreshes(monkeypatch):
    monkeypatch.setenv("MSG_HISTORY_MODE", "anchored")
    ch = FakeChannel()
    ch.add(1, "old")
    trigger = ch.add(2, "t0")
    await _build(trigger)
    ch.add(3, "!reset_history")
    trigger = ch.add(4, "t1")
    assert _texts(await _build(trigger)) == ["Message from 'alice': t1"]
    # and the anchor now sits past the reset
    ch.add(5, "t2 context")
    trigger = ch.add(6, "t2")
    assert _texts(await _build(trigger)) == ["Message from 'alice': t1",
                                             "Message from 'alice': t2 context",
                                             "Message from 'alice': t2"]


@pytest.mark.asyncio
async def test_anchored_edit_is_seen_because_nothing_is_cached(monkeypatch):
    monkeypatch.setenv("MSG_HISTORY_MODE", "anchored")
    ch = FakeChannel()
    ch.add(1, "original")
    trigger = ch.add(2, "t0")
    await _build(trigger)
    ch.messages[0].content = "edited"
    ch.messages.remove(ch.messages[1])  # and a deletion
    trigger = ch.add(3, "t1")
    assert _texts(await _build(trigger)) == ["Message from 'alice': edited",
                                             "Message from 'alice': t1"]


@pytest.mark.asyncio
async def test_out_of_order_trigger_gets_a_snapshot_and_never_rewinds(monkeypatch):
    """Two workers pick up 20 and 21; 21 builds first. 20 must not see 21 and
    must not move the channel's anchor back."""
    monkeypatch.setenv("MSG_HISTORY_MODE", "anchored")
    monkeypatch.setenv("MSG_HISTORY_LIMIT", "3")
    ch = FakeChannel()
    for i in range(1, 20):
        ch.add(i, f"m{i}")
    older = ch.add(20, "older")
    newer = ch.add(21, "newer")

    await _build(newer)
    committed = history_policy.store().get((GUILD_ID, CHANNEL_ID))
    assert committed.refresh_trigger_id == 21

    texts = _texts(await _build(older))
    assert texts == ["Message from 'alice': m18", "Message from 'alice': m19",
                     "Message from 'alice': older"]
    assert history_policy.store().get((GUILD_ID, CHANNEL_ID)) == committed


@pytest.mark.asyncio
async def test_anchored_channels_are_isolated(monkeypatch):
    monkeypatch.setenv("MSG_HISTORY_MODE", "anchored")
    a, b = FakeChannel(1), FakeChannel(2)
    a.add(1, "in a")
    b.add(2, "in b")
    ta, tb = a.add(3, "ta"), b.add(4, "tb")
    assert "in b" not in str(await _build(ta))
    assert "in a" not in str(await _build(tb))
    store = history_policy.store()
    assert store.get((GUILD_ID, 1)) != store.get((GUILD_ID, 2))


# ------------------------------------------------------------------ token budget


@pytest.mark.asyncio
async def test_sliding_trims_oldest_whole_messages_to_the_budget(monkeypatch):
    monkeypatch.setenv("MSG_HISTORY_LIMIT", "10")
    monkeypatch.setenv("MSG_HISTORY_RESERVE_TOKENS", "1000")
    monkeypatch.setattr("classes.message_handler.slot_context_tokens", lambda: 1300)
    ch = FakeChannel()
    for i in range(1, 6):
        ch.add(i, "x" * 300)  # ~100 tokens + overhead each
    trigger = ch.add(6, "y" * 3000)  # the trigger alone is over the budget
    texts = _texts(await _build(trigger))
    assert texts == ["Message from 'alice': " + "y" * 3000]


@pytest.mark.asyncio
async def test_anchored_over_budget_refreshes_then_trims(monkeypatch):
    monkeypatch.setenv("MSG_HISTORY_MODE", "anchored")
    monkeypatch.setenv("MSG_HISTORY_LIMIT", "3")
    monkeypatch.setenv("MSG_HISTORY_REFRESH_MESSAGES", "20")
    monkeypatch.setenv("MSG_HISTORY_RESERVE_TOKENS", "1000")
    monkeypatch.setattr("classes.message_handler.slot_context_tokens", lambda: 1600)
    ch = FakeChannel()
    trigger = ch.add(1, "t0")
    await _build(trigger)
    for i in range(2, 6):
        ch.add(i, "x" * 600)  # ~208 tokens each: 4 of them are over 600
    trigger = ch.add(6, "t1")
    texts = _texts(await _build(trigger))
    # refreshed to the newest 2, which fit
    assert len(texts) == 3 and texts[-1] == "Message from 'alice': t1"
    state = history_policy.store().get((GUILD_ID, CHANNEL_ID))
    assert state == AnchorState(after_id=3, refresh_trigger_id=6)

"""Tests for finding polls after they leave the history window:
  - core/classes/poll_store.py — which polls a channel has had, in Redis
    (against a small in-memory stand-in for the sorted-set commands it uses);
  - polls.remember_poll — what on_message records;
  - the check_polls agent tool (core/classes/tool_functions.py).

To run this pytest file from the command line, use:
    pytest core/tests/poll_lookup_tests.py
"""
import json
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import discord
import pytest

from classes import poll_store, polls

DAY = 24 * 3600


class FakeRedis:
    """The sorted-set commands poll_store uses, with redis-py's semantics."""

    def __init__(self):
        self.sets = {}
        self.expiry = {}

    async def zadd(self, key, mapping):
        self.sets.setdefault(key, {}).update({str(m): float(s) for m, s in mapping.items()})

    def _sorted(self, key):
        return sorted(self.sets.get(key, {}).items(), key=lambda item: (item[1], item[0]))

    async def zremrangebyscore(self, key, low, high):
        for member, score in self._sorted(key):
            if score <= high:
                del self.sets[key][member]

    async def zremrangebyrank(self, key, start, stop):
        items = self._sorted(key)
        stop = len(items) + stop if stop < 0 else stop
        if stop < 0:  # a negative rank past the start: Redis removes nothing
            return
        for member, _ in items[start:stop + 1]:
            del self.sets[key][member]

    async def zrange(self, key, start, stop, withscores=False):
        items = self._sorted(key)
        stop = len(items) + stop if stop < 0 else stop
        start = len(items) + start if start < 0 else start
        picked = items[max(start, 0):stop + 1]
        return picked if withscores else [member for member, _ in picked]

    async def zrem(self, key, member):
        self.sets.get(key, {}).pop(member, None)

    async def expireat(self, key, when):
        self.expiry[key] = when


@pytest.fixture
def redis(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(poll_store, "text_client", lambda: fake)
    return fake


# ------------------------------------------------------------- poll_store


@pytest.mark.asyncio
async def test_recent_is_newest_posted_first(redis):
    now = time.time()
    # Posting order (message id) decides, not end time.
    await poll_store.remember(7, 100, now + 30 * DAY)
    await poll_store.remember(7, 200, now + 3600)
    await poll_store.remember(7, 300, now + DAY)

    assert await poll_store.recent(7, 5) == [300, 200, 100]
    assert await poll_store.recent(7, 2) == [300, 200]
    assert await poll_store.recent(8, 5) == []  # per channel


@pytest.mark.asyncio
async def test_polls_are_dropped_a_week_after_they_end(redis):
    now = time.time()
    await poll_store.remember(7, 100, now - poll_store.KEEP_SECONDS - 60)
    await poll_store.remember(7, 200, now - poll_store.KEEP_SECONDS + 3600)

    assert await poll_store.recent(7, 5) == [200]


@pytest.mark.asyncio
async def test_channel_keeps_a_bounded_number(redis, monkeypatch):
    monkeypatch.setattr(poll_store, "MAX_PER_CHANNEL", 3)
    now = time.time()
    for i in range(1, 6):
        await poll_store.remember(7, i, now + i * 3600)

    # The ones ending first go.
    assert await poll_store.recent(7, 10) == [5, 4, 3]


@pytest.mark.asyncio
async def test_key_expires_after_its_last_poll(redis):
    now = time.time()
    await poll_store.remember(7, 1, now + 30 * DAY)
    await poll_store.remember(7, 2, now + 3600)  # a shorter poll must not cut it short
    assert redis.expiry["dcb:polls:7"] == int(now + 30 * DAY + poll_store.KEEP_SECONDS)


@pytest.mark.asyncio
async def test_forget(redis):
    await poll_store.remember(7, 1, time.time() + 3600)
    await poll_store.forget(7, 1)
    assert await poll_store.recent(7, 5) == []


# ------------------------------------------------------------ remember_poll


def _poll(counts, expires=None):
    poll = discord.Poll("Lunch?", timedelta(hours=24))
    for text in counts:
        poll.add_answer(text=text)
    for answer in poll.answers:
        answer._vote_count = counts[answer.text]
    poll._expiry = expires
    return poll


def _message(mid, poll, author="bot"):
    return SimpleNamespace(id=mid, poll=poll, channel=SimpleNamespace(id=7),
                           author=SimpleNamespace(name=author))


@pytest.mark.asyncio
async def test_remember_poll_records_its_end(redis):
    ends = datetime(2026, 10, 10, 14, 0, tzinfo=timezone.utc)
    await polls.remember_poll(_message(42, _poll({"a": 0, "b": 0}, ends)))
    assert redis.sets["dcb:polls:7"] == {"42": ends.timestamp()}


@pytest.mark.asyncio
async def test_remember_poll_ignores_other_messages_and_fails_soft(redis, monkeypatch):
    await polls.remember_poll(MagicMock())  # a MagicMock's .poll is not a poll
    assert redis.sets == {}

    monkeypatch.setattr(poll_store, "remember", AsyncMock(side_effect=ConnectionError("down")))
    await polls.remember_poll(_message(42, _poll({"a": 0, "b": 0})))  # no raise


# --------------------------------------------------------------- check_polls


def _tool_context(context):
    from agents.tool_context import ToolContext
    return ToolContext(context=context, tool_name="check_polls",
                       tool_call_id="t1", tool_arguments="{}")


async def _check(channel):
    from classes import tool_functions
    return await tool_functions.check_polls.on_invoke_tool(
        _tool_context({"channel": channel}), json.dumps({}))


def _channel(messages):
    channel = MagicMock()
    channel.id = 7

    async def fetch(message_id):
        found = messages.get(message_id)
        if isinstance(found, Exception):
            raise found
        return found
    channel.fetch_message = AsyncMock(side_effect=fetch)
    return channel


@pytest.fixture
def voters(monkeypatch):
    polls._ended_poll_voters.clear()

    async def stub(self, *, limit=None, after=None):
        for name in {"Pizza": ["alice", "bob"]}.get(self.text, []):
            yield SimpleNamespace(name=name)
    monkeypatch.setattr(discord.PollAnswer, "voters", stub)
    yield
    polls._ended_poll_voters.clear()


@pytest.mark.asyncio
async def test_check_polls_reports_live_votes_newest_first(redis, voters):
    now = time.time()
    await poll_store.remember(7, 1, now + 3600)
    await poll_store.remember(7, 2, now + 3600)
    ends = datetime(2026, 10, 10, 14, 0, tzinfo=timezone.utc)
    channel = _channel({
        1: _message(1, _poll({"Pizza": 2, "Sushi": 0}, ends)),
        2: _message(2, _poll({"Yes": 0, "No": 0}, ends), author="alice"),
    })

    result = await _check(channel)

    assert result.startswith("Polls in this channel, newest first:")
    assert result.index("Message from 'alice': Poll") < result.index("Message from 'bot': Poll")
    assert "- Pizza: 2 (alice, bob)" in result


@pytest.mark.asyncio
async def test_check_polls_forgets_deleted_polls(redis, voters):
    await poll_store.remember(7, 1, time.time() + 3600)
    gone = discord.NotFound(MagicMock(status=404, reason="Not Found"), "gone")

    result = await _check(_channel({1: gone}))

    assert "no polls" in result
    assert await poll_store.recent(7, 5) == []


@pytest.mark.asyncio
async def test_check_polls_with_none(redis, voters):
    assert "no polls in this channel in the last week" in await _check(_channel({}))


@pytest.mark.asyncio
async def test_check_polls_when_redis_is_down(monkeypatch):
    monkeypatch.setattr(poll_store, "recent", AsyncMock(side_effect=ConnectionError("down")))
    assert "could not be looked up" in await _check(_channel({}))

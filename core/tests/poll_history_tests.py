"""Tests for polls in a prompt's history (MessageHandler): a poll message has
no text or embeds, so without its own entry it was left out entirely.

Runs the REAL build_messages against message_history_tests' FakeChannel,
with real (stateless) discord.Poll objects and the voter lookup stubbed.

To run this pytest file from the command line, use:
    pytest core/tests/poll_history_tests.py
"""
import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import discord
import pytest

from classes import history_policy, polls
from message_history_tests import BOT_ID, FakeChannel, _build, _texts

END = datetime(2026, 10, 10, 14, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    history_policy.store().clear()
    polls._ended_poll_voters.clear()
    for name in ("MSG_HISTORY_MODE", "MSG_HISTORY_LIMIT", "MSG_HISTORY_REFRESH_MESSAGES",
                 "MSG_HISTORY_RESERVE_TOKENS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("classes.message_handler.slot_context_tokens", lambda: None)
    yield
    history_policy.store().clear()
    polls._ended_poll_voters.clear()


def _poll(counts, finalized=False, multiple=False):
    poll = discord.Poll("Lunch?", timedelta(hours=24), multiple=multiple)
    for text in counts:
        poll.add_answer(text=text)
    for answer in poll.answers:
        answer._vote_count = counts[answer.text]
    poll._expiry = END
    poll._finalized = finalized
    return poll


def _with_poll(msg, poll):
    msg.poll = poll
    return msg


def _voters(monkeypatch, by_answer, calls=None):
    """Stub PollAnswer.voters (slotted, so patched on the class): names by
    answer text; an Exception value makes that lookup fail."""
    async def voters(self, *, limit=None, after=None):
        if calls is not None:
            calls.append(self.text)
        found = by_answer.get(self.text, [])
        if isinstance(found, Exception):
            raise found
        for name in found[:limit]:
            yield SimpleNamespace(name=name)
    monkeypatch.setattr(discord.PollAnswer, "voters", voters)


@pytest.mark.asyncio
async def test_bots_poll_appears_with_counts_and_voters(monkeypatch):
    calls = []
    _voters(monkeypatch, {"Pizza": ["alice", "bob"], "Sushi": ["carol"]}, calls)
    ch = FakeChannel()
    _with_poll(ch.reply(1, ""), _poll({"Pizza": 2, "Sushi": 1, "Tacos": 0}))
    trigger = ch.add(2, f"<@{BOT_ID}> who's winning?")

    messages = await _build(trigger)

    assert messages[0]["role"] == "assistant"
    assert messages[0]["content"] == (
        "Message from 'bot': Poll \"Lunch?\" (one choice each, open until 2026-10-10 14:00 UTC, 3 votes)\n"
        "- Pizza: 2 (alice, bob)\n- Sushi: 1 (carol)\n- Tacos: 0")
    assert sorted(calls) == ["Pizza", "Sushi"]  # no lookup for an answer without votes
    assert messages[1]["content"] == "Message from 'alice': who's winning?"


@pytest.mark.asyncio
async def test_users_poll_keeps_its_text_first(monkeypatch):
    _voters(monkeypatch, {})
    ch = FakeChannel()
    _with_poll(ch.add(1, "vote please"), _poll({"Yes": 0, "No": 0}))
    trigger = ch.add(2, "hi")

    texts = _texts(await _build(trigger))

    assert texts[0] == "Message from 'alice': vote please"
    assert texts[1].startswith("Message from 'alice': Poll \"Lunch?\"")


@pytest.mark.asyncio
async def test_failed_voter_lookup_shows_counts_only(monkeypatch):
    _voters(monkeypatch, {"Pizza": discord.ClientException("boom"), "Sushi": ["carol"]})
    ch = FakeChannel()
    _with_poll(ch.reply(1, ""), _poll({"Pizza": 2, "Sushi": 1}))
    trigger = ch.add(2, "hi")

    poll_text = _texts(await _build(trigger))[0]

    assert "- Pizza: 2\n- Sushi: 1 (carol)" in poll_text


@pytest.mark.asyncio
async def test_slow_voter_lookup_does_not_hold_up_the_reply(monkeypatch):
    async def voters(self, *, limit=None, after=None):
        await asyncio.sleep(10)
        yield SimpleNamespace(name="late")
    monkeypatch.setattr(discord.PollAnswer, "voters", voters)
    monkeypatch.setattr(polls, "POLL_VOTERS_TIMEOUT_SECONDS", 0.01)
    ch = FakeChannel()
    _with_poll(ch.reply(1, ""), _poll({"Pizza": 1, "Sushi": 0}))
    trigger = ch.add(2, "hi")

    assert "- Pizza: 1\n" in _texts(await _build(trigger))[0]


@pytest.mark.asyncio
async def test_ended_polls_voters_are_looked_up_once(monkeypatch):
    calls = []
    _voters(monkeypatch, {"Pizza": ["alice"]}, calls)
    ch = FakeChannel()
    _with_poll(ch.reply(1, ""), _poll({"Pizza": 1, "Sushi": 0}, finalized=True))
    trigger = ch.add(2, "hi")

    first = _texts(await _build(trigger))
    second = _texts(await _build(trigger))

    assert first == second
    assert "ended" in first[0] and "(alice)" in first[0]
    assert calls == ["Pizza"]


@pytest.mark.asyncio
async def test_poll_result_message_is_rendered_from_its_fields():
    ch = FakeChannel()
    embed = discord.Embed()
    for name, value in (("poll_question_text", "Lunch?"), ("victor_answer_text", "Pizza"),
                        ("victor_answer_votes", "3"), ("total_votes", "5")):
        embed.add_field(name=name, value=value)
    result = ch.reply(1, "")
    result.embeds = [embed]
    result.type = discord.MessageType.poll_result
    trigger = ch.add(2, "hi")

    messages = await _build(trigger)

    assert messages[0] == {"role": "assistant", "content":
                           "Message from 'bot': Poll ended: \"Lunch?\" - winner: Pizza with 3 of 5 votes"}
    assert len(messages) == 2  # no bare embed JSON as well


def test_mock_messages_are_not_mistaken_for_polls():
    msg = MagicMock()
    assert polls.message_poll(msg) is None
    assert polls.is_poll_result(msg) is False

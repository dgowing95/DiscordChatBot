"""Reading Discord polls: which messages carry one, who voted, and the text
the model reads. Shared by the history build (MessageHandler) and the
check_polls tool; the formatting itself is the pure classes/poll_format.py.
"""
import asyncio
import logging
import time

import discord

from classes.poll_format import VOTER_NAMES_PER_ANSWER, format_poll, format_poll_result

logger = logging.getLogger(__name__)

# Bounds one poll's voter lookups; past it the poll shows counts only.
POLL_VOTERS_TIMEOUT_SECONDS = 5
# Voter names of ended polls, which can no longer change: message id ->
# {answer id: names}. Oldest entry dropped past the cap.
_ended_poll_voters = {}
_ENDED_POLL_VOTERS_MAX = 256


def message_poll(message):
    """The message's poll, or None. isinstance rather than truthiness: test
    messages are MagicMocks, whose .poll is always truthy."""
    poll = getattr(message, "poll", None)
    return poll if isinstance(poll, discord.Poll) else None


def is_poll_result(message) -> bool:
    """Discord's "poll ended" system message (its result is in one embed)."""
    return getattr(message, "type", None) == discord.MessageType.poll_result and bool(message.embeds)


def placeholder_voters(message):
    """Stands in for a poll's voter names while estimating tokens, as the
    placeholder image does for images: one short name per counted vote, up
    to the cap that format_poll shows."""
    poll = message_poll(message)
    if poll is None:
        return None
    return {a.id: ["member"] * min(a.vote_count, VOTER_NAMES_PER_ANSWER) for a in poll.answers}


async def poll_voters(message) -> dict:
    """Who voted for each answer of a message's poll: {answer id: names}.

    One Discord request per answer that has votes, all at once and bounded
    by POLL_VOTERS_TIMEOUT_SECONDS. A failed or slow lookup leaves that
    answer out, so the poll shows its counts only; a lookup never costs the
    reply. An ended poll's voters are kept, since they cannot change."""
    poll = message_poll(message)
    if poll is None:
        return {}
    if message.id in _ended_poll_voters:
        return _ended_poll_voters[message.id]
    answers = [a for a in poll.answers if a.vote_count > 0]

    async def _one(answer):
        try:
            return [u.name async for u in answer.voters(limit=VOTER_NAMES_PER_ANSWER)]
        except Exception as e:
            logger.warning(f"Could not read the voters of poll {message.id}: {e}")
            return None

    try:
        names = await asyncio.wait_for(
            asyncio.gather(*(_one(a) for a in answers)), POLL_VOTERS_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        logger.warning(f"Reading the voters of poll {message.id} timed out")
        return {}
    found = {a.id: n for a, n in zip(answers, names) if n is not None}
    if poll.is_finalised() and len(found) == len(answers):
        if len(_ended_poll_voters) >= _ENDED_POLL_VOTERS_MAX:
            _ended_poll_voters.pop(next(iter(_ended_poll_voters)))
        _ended_poll_voters[message.id] = found
    return found


async def remember_poll(message) -> None:
    """Records a message's poll so check_polls can find it later. Called from
    on_message for every message, the bot's own polls included. Fails soft:
    a poll that is not recorded can still be seen while it is in history."""
    poll = message_poll(message)
    if poll is None:
        return
    from classes import poll_store
    from classes.poll_format import MAX_HOURS
    ends = poll.expires_at
    try:
        await poll_store.remember(
            message.channel.id, message.id,
            ends.timestamp() if ends is not None else time.time() + MAX_HOURS * 3600)
    except Exception as e:
        logger.warning(f"Could not record poll {message.id}: {e}")


def describe_poll(message, voters=None) -> str:
    """The poll on `message` as the model reads it (see format_poll)."""
    poll = message_poll(message)
    voters = voters or {}
    return format_poll(
        message.author.name, poll.question,
        [(a.text, a.vote_count, voters.get(a.id)) for a in poll.answers],
        poll.total_votes, poll.multiple, poll.is_finalised(), poll.expires_at)


def describe_poll_result(message) -> str:
    """Discord's "poll ended" message as the model reads it."""
    fields = {f.name: f.value for f in message.embeds[0].fields}
    return format_poll_result(message.author.name, fields)

"""The search_messages and read_messages_around agent tools: finding old
messages in this server through Discord's own message search, so the bot can
look up something said long before its history window, and then reading the
conversation around one.

The whole server is searched unless a channel is named. Discord only
returns messages from channels the bot itself can read, and
read_messages_around reads the same channels: what the BOT can see, not
only what the person asking can (the owner's choice, as for search).
There is no cap on calls per reply: a cap of 3 once stopped a workspace task
that needed far more, and the brain's loop guard and turn limit bound a loop.
message_search_format.py holds the query and result rules and the
endpoint's quirks.

save_messages_to_file is bounded in time (message_search_format says why):
it pages until the most it may take, then saves what it has and says how to
carry on (`before`). Search requests go one at a time per server
(_search_lock), and a channel with no words or author is read from its
history instead of searched.
"""
import asyncio
import io
import json
import logging
import posixpath
import time
import weakref
from dataclasses import dataclass
from pathlib import PurePosixPath
import re

import discord
from agents import RunContextWrapper, function_tool
from discord.http import Route

from classes import card, card_kit as K
from classes.tool_cards import (
    READING_MESSAGES,
    SAVING_MESSAGES,
    SEARCHING_MESSAGES,
    message_search_card,
    message_search_line,
)
from classes.message_search_format import (
    EXPORT_DEFAULT, EXPORT_MAX, EXPORT_MAX_BYTES, EXPORT_MAX_OFFSET, EXPORT_MIN_SECONDS,
    HISTORY_PAGE_SIZE, MAX_LIMIT, NOT_INDEXED_TEXT, NO_TIME_TEXT, SEARCH_PAGE_SECONDS,
    carry_on_text, clamp_export, clamp_side, export_filename, export_record, export_seconds,
    export_summary, flatten, format_context, format_results, minutes_text, not_indexed,
    parse_before, parse_message_link, search_params, search_seconds, time_stop_text,
)

logger = logging.getLogger(__name__)

_ID = re.compile(r"^<?[#@]?!?&?(\d{15,25})>?$")
_HERE = {"here", "this channel", "this", "current", "current channel"}


def _find_channel(guild, text: str, current):
    """A text channel, thread or forum by id, mention or name
    (case-insensitive; a unique partial name also matches). "here" is the
    channel being talked in."""
    raw = text.strip()
    if raw.casefold() in _HERE:
        return current
    match = _ID.match(raw)
    if match:
        return guild.get_channel_or_thread(int(match.group(1)))
    wanted = raw.lstrip("#").casefold()
    channels = [c for c in list(guild.channels) + list(guild.threads)
                if not isinstance(c, discord.CategoryChannel)]
    exact = [c for c in channels if c.name.casefold() == wanted]
    if exact:
        return exact[0]
    partial = [c for c in channels if wanted in c.name.casefold()]
    return partial[0] if len(partial) == 1 else None


async def _find_author(guild, text: str):
    """The id of a member by id, mention or name; None when no one matches.
    query_members asks the gateway, which works without the members intent."""
    raw = text.strip()
    match = _ID.match(raw)
    if match:
        return int(match.group(1))
    wanted = raw.lstrip("@").casefold()
    member = guild.get_member_named(raw.lstrip("@"))
    if member is None:
        try:
            found = await guild.query_members(query=raw.lstrip("@")[:100], limit=5)
        except Exception as e:
            logger.info(f"member lookup for {raw!r} failed: {e}")
            found = []
        exact = [m for m in found if wanted in {m.name.casefold(), m.display_name.casefold(),
                                                (m.global_name or "").casefold()}]
        # Only an exact name: Discord's lookup matches prefixes, and searching
        # "Samuel" when asked for "Sam" would report the wrong person's words.
        member = exact[0] if exact else None
    return member.id if member is not None else None


@function_tool
async def search_messages(wrapper: RunContextWrapper[dict], query: str, channel: str = "",
                          author: str = "", limit: int = 10) -> str:
    """Searches this server's past messages, to find something said earlier that is not in
    the recent conversation, e.g. "what was the link Sam posted about the trip?". Searches
    the whole server unless a channel is given.

    Args:
        query: Words the message contains. Leave empty to list an author's or a channel's newest messages.
        channel: Optional: only search this channel (its name, #mention or id, or "here").
        author: Optional: only messages by this person (their name, @mention or id).
        limit: How many messages to return, 1 to 25.
    """
    context = wrapper.context
    scope = await _scope(context, query, channel, author)
    if isinstance(scope, str):
        return scope
    context["message_search_calls"] = context.get("message_search_calls", 0) + 1
    logger.info(f"Searching {scope.where} for {scope.query!r} (author {scope.author_id}, call "
                f"{context['message_search_calls']})")
    spec = message_search_card(
        SEARCHING_MESSAGES, message_search_line(scope.query, scope.where, scope.author))
    live = await card.notice(scope.current, spec)
    body = await _search_page(scope, limit)
    if isinstance(body, str):
        _failed(live, spec)
        return body
    text = format_results(body, scope.guild.id, _channel_names(scope.guild), scope.query,
                          scope.where)
    # In a brain call (it has a workspace) a big result can go to a file for
    # code to read with save_messages_to_file.
    shown, total = len(flatten(body)), int(body.get("total_results") or 0)
    if context.get("session") is not None and total > shown:
        text += (f"\nTo work with all {total} of them (counting, adding up, charting), "
                 f"save_messages_to_file writes them to a file in your workspace")
        if scope.query or scope.author_id is not None:
            text += f" ({minutes_text(search_seconds(min(total, EXPORT_MAX)))} for that many)"
        text += "."
    return text


@dataclass
class _Scope:
    current: object
    guild: object
    client: object
    query: str
    where: str
    channel_id: int | None
    author: str
    author_id: int | None
    place: object = None


async def _scope(context: dict, query, channel, author):
    """What a search covers: the server, words, channel and author, resolved;
    or the text to return when that fails."""
    current = context.get("channel")
    guild = getattr(current, "guild", None)
    client = context.get("discord_client")
    if guild is None or client is None:
        return "Message search only works in a server. Tell the user."
    # The model can send null for an optional argument.
    query, channel, author = (query or "").strip(), (channel or "").strip(), (author or "").strip()
    if not query and not channel and not author:
        return "Give some words to search for, a channel, or an author."
    where, channel_id, place = "this server", None, None
    if channel:
        place = _find_channel(guild, channel, current)
        if place is None:
            return f"There is no channel called \"{channel}\" here. Ask the user which one they mean."
        where, channel_id = f"#{place.name}", place.id
    author_id = None
    if author:
        author_id = await _find_author(guild, author)
        if author_id is None:
            return f"No one called \"{author}\" was found in this server. Ask the user who they mean."
    return _Scope(current, guild, client, query, where, channel_id, author, author_id, place)


# Event loop -> guild id -> its search lock (a lock belongs to one loop).
_SEARCH_LOCKS = weakref.WeakKeyDictionary()
# A page refused with a 429 after discord.py's own retries is asked again
# this many times, while the save has time.
SEARCH_RETRIES = 2


def _search_lock(guild_id) -> asyncio.Lock:
    """One search request at a time per server. Its search has one rate
    limit: two saves paging side by side met a 429 on every page, and
    discord.py gave one up after its retries (prod, 2026-11-08: 125 of
    1628 saved, then asked again from the start)."""
    locks = _SEARCH_LOCKS.setdefault(asyncio.get_running_loop(), {})
    return locks.setdefault(guild_id, asyncio.Lock())


# _search_page's answer when a save's time ran out while it waited.
_OUT_OF_TIME = object()


async def _search_page(scope: _Scope, limit, offset: int = 0, newest_first: bool = False,
                       max_id=None, retry_until: float | None = None):
    """One page of Discord's search, or the text to return when it fails.
    `retry_until` (a save's): a monotonic time before which a 429 is asked
    again, and past which nothing waits, neither for the server's lock (another
    save) nor for discord.py sitting out a rate limit: _OUT_OF_TIME then."""
    nsfw = bool(getattr(scope.current, "is_nsfw", lambda: False)())
    params = search_params(scope.query, scope.channel_id, scope.author_id, limit,
                           include_nsfw=nsfw, offset=offset, newest_first=newest_first,
                           max_id=max_id)
    tries = 0
    while True:
        remaining = None if retry_until is None else retry_until - time.monotonic()
        if remaining is not None and remaining <= 0:
            return _OUT_OF_TIME
        try:
            async with asyncio.timeout(remaining):
                async with _search_lock(scope.guild.id):
                    body = await scope.client.http.request(
                        Route("GET", "/guilds/{guild_id}/messages/search",
                              guild_id=scope.guild.id),
                        params=params)
            break
        except TimeoutError:
            return _OUT_OF_TIME
        except discord.Forbidden:
            return ("Discord didn't allow the search (I need Read Message History, and the bot "
                    "needs the Message Content intent). Tell the user.")
        except discord.HTTPException as e:
            if (getattr(e, "status", None) == 429 and retry_until is not None
                    and tries < SEARCH_RETRIES
                    and time.monotonic() + SEARCH_PAGE_SECONDS < retry_until):
                tries += 1
                await asyncio.sleep(SEARCH_PAGE_SECONDS)
                continue
            logger.warning(f"message search failed: {e}")
            return "The search failed on Discord's side. Tell the user to try again later."
    if not_indexed(body):
        return NOT_INDEXED_TEXT
    return body


def _channel_names(guild) -> dict:
    return {str(c.id): c.name for c in list(guild.channels) + list(guild.threads)}


@function_tool
async def save_messages_to_file(wrapper: RunContextWrapper[dict], query: str = "",
                                channel: str = "", author: str = "",
                                max_messages: int = EXPORT_DEFAULT, before: str = "") -> str:
    """Saves every message in this server that matches a search to a file in your
    workspace (one JSON object per line, embeds included), for code to count, add up or
    chart. Use it instead of search_messages when you need more than a handful of messages.
    A channel alone is read straight from it; words or an author go through Discord's
    search, about 270 messages a minute.

    Args:
        query: Words the messages contain (a bot's embed titles and fields count). Leave
            empty to save a channel's or an author's messages.
        channel: Optional: only this channel (its name, #mention or id).
        author: Optional: only messages by this person or bot (their name, @mention or id).
        max_messages: The most to save, newest first, 1 to 10000.
        before: Optional: only messages older than this (a message link or id, or a date
            such as 2024-05-31). A save that stopped early names the one to carry on from.
    """
    context = wrapper.context
    session = context.get("session")
    if session is None:
        return "Saving messages to a file needs the workspace, which is not available."
    scope = await _scope(context, query, channel, author)
    if isinstance(scope, str):
        return scope
    try:
        before_id = parse_before(before)
    except ValueError:
        return (f"\"{before}\" is not a message link, a message id or a date, so it can't be "
                f"used for before. Give one of those, or leave it empty.")
    deadline = context.get("brain_deadline")
    budget = export_seconds(None if deadline is None else deadline - time.monotonic())
    if budget < EXPORT_MIN_SECONDS:
        return NO_TIME_TEXT
    history = _reads_history(scope)
    if history:
        refusal = _history_refusal(scope)
        if refusal:
            return refusal
    wanted = clamp_export(max_messages)
    logger.info(f"Saving up to {wanted} messages from {scope.where} for {scope.query!r} "
                f"(author {scope.author_id}, {'history' if history else 'search'}, before "
                f"{before_id}, for at most {budget:.0f}s)")
    spec = message_search_card(
        SAVING_MESSAGES, message_search_line(scope.query, scope.where, scope.author),
        "For code to count or chart them")
    live = await card.notice(scope.current, spec)
    started = time.monotonic()
    export = _Export(scope.guild.id, _channel_names(scope.guild), wanted)
    try:
        stopped = await (_save_history if history else _save_search)(
            scope, export, before_id, started + budget)
    except _Failed as e:
        _failed(live, spec)
        return str(e)
    total = None if history else export.total
    if not export.records:
        return export_summary("", 0, total, None)

    name = export_filename(scope.query, scope.where, scope.author, before_id)
    try:
        in_dir = await _input_dir(context, session)
        await session.write(PurePosixPath(in_dir, name),
                            io.BytesIO(("\n".join(export.records) + "\n").encode("utf-8")))
    except Exception as e:
        logger.warning(f"could not write the message file: {e}")
        _failed(live, spec)
        return "The messages were found but could not be written to your workspace."
    logger.info(f"Saved {len(export.records)} of {'?' if total is None else total} messages to "
                f"in/{name} in {time.monotonic() - started:.0f}s"
                + (f" ({stopped.split(':')[0]})" if stopped else ""))
    return export_summary(f"in/{name}", len(export.records), total,
                          json.loads(export.records[0]), stopped, export.span())


class _Failed(Exception):
    """Nothing was saved: the text to return."""


class _Export:
    """The messages a save has so far, as the lines of its file."""

    def __init__(self, guild_id, names: dict, wanted: int):
        self.guild_id, self.names, self.wanted = guild_id, names, wanted
        self.records: list[str] = []
        self.seen: set[str] = set()
        self.size = 0
        self.total = 0
        self.stamps: list[str] = []
        self.oldest_id: int | None = None

    @property
    def full(self) -> bool:
        return len(self.records) >= self.wanted

    def add(self, messages) -> None:
        for message in messages:
            message_id = str(message.get("id"))
            # New messages arriving while it pages shift the pages, so a
            # message can come round twice.
            if message_id in self.seen or self.full:
                continue
            self.seen.add(message_id)
            record = export_record(message, self.guild_id, self.names)
            line = json.dumps(record, ensure_ascii=False)
            self.size += len(line.encode("utf-8")) + 1
            self.records.append(line)
            if record["timestamp"]:
                self.stamps.append(record["timestamp"])
            if message_id.isdigit() and (self.oldest_id is None
                                         or int(message_id) < self.oldest_id):
                self.oldest_id = int(message_id)

    def span(self) -> tuple[str, str] | None:
        if not self.stamps:
            return None
        return _day_and_time(min(self.stamps)), _day_and_time(max(self.stamps))

    def broke_off(self) -> str:
        return (f"Stopped early: Discord stopped answering after {len(self.records)} messages, "
                f"so the file holds only those. {carry_on_text(self.oldest_id)}.")

    def too_big(self) -> str:
        return (f"Stopped early: the file reached its size limit. "
                f"{carry_on_text(self.oldest_id)}.")


def _day_and_time(stamp: str) -> str:
    return stamp[:16].replace("T", " ") + " UTC" if len(stamp) >= 16 else stamp


async def _save_search(scope: _Scope, export: _Export, before_id, stop_at: float) -> str:
    """Search pages, newest first, until the save is full, done or out of
    time. Returns why it stopped early, or ""."""
    offset, per_page = 0, SEARCH_PAGE_SECONDS
    while not export.full:
        if offset and time.monotonic() + per_page > stop_at:
            rest = max(0, min(export.wanted, export.total) - len(export.records))
            return time_stop_text(len(export.records), rest, export.oldest_id)
        asked_at = time.monotonic()
        body = await _search_page(scope, MAX_LIMIT, offset, newest_first=True, max_id=before_id,
                                  retry_until=stop_at)
        if body is _OUT_OF_TIME:
            if not export.records:
                raise _Failed(NO_TIME_TEXT)
            rest = max(0, min(export.wanted, export.total) - len(export.records))
            return time_stop_text(len(export.records), rest, export.oldest_id)
        if isinstance(body, str):
            if not export.records:
                raise _Failed(body)
            return export.broke_off()
        per_page = max(SEARCH_PAGE_SECONDS, time.monotonic() - asked_at)
        export.total = max(export.total, int(body.get("total_results") or 0))
        for thread in body.get("threads") or []:
            if thread.get("id") and thread.get("name"):
                export.names.setdefault(str(thread["id"]), thread["name"])
        hits = flatten(body)
        export.add(hits)
        offset += len(hits)
        if export.size > EXPORT_MAX_BYTES:
            return export.too_big()
        if not hits or offset >= export.total:
            return ""
        if offset > EXPORT_MAX_OFFSET:
            return (f"Stopped at {len(export.records)}: Discord's search goes no deeper than "
                    f"10000 results. {carry_on_text(export.oldest_id)}.")
    return ""


def _reads_history(scope: _Scope) -> bool:
    """A channel alone, with no words or author: read from its history,
    100 a page on its own rate limit, not searched 25 at a time. A forum
    (or a category) has no history of its own, so it is searched."""
    return (scope.place is not None and not scope.query and scope.author_id is None
            and not isinstance(scope.place, (discord.ForumChannel, discord.CategoryChannel)))


def _history_refusal(scope: _Scope) -> str | None:
    """As read_messages_around: the bot must be able to read the history,
    and an age-restricted channel is read only from one."""
    me = getattr(scope.guild, "me", None)
    if me is not None and hasattr(scope.place, "permissions_for"):
        permissions = scope.place.permissions_for(me)
        if not (permissions.read_messages and permissions.read_message_history):
            return _NO_HISTORY_TEXT
    if (getattr(scope.place, "is_nsfw", lambda: False)()
            and not getattr(scope.current, "is_nsfw", lambda: False)()):
        return ("That channel is age-restricted, so I can't save its messages here. Tell the "
                "user to ask in an age-restricted channel.")
    return None


async def _save_history(scope: _Scope, export: _Export, before_id, stop_at: float) -> str:
    """The channel's history, newest first, a page of 100 at a time, until
    the save is full, at the channel's start or out of time."""
    before, per_page = before_id, 0.0
    while not export.full:
        if per_page and time.monotonic() + per_page > stop_at:
            return time_stop_text(len(export.records), None, export.oldest_id, history=True)
        asked = min(HISTORY_PAGE_SIZE, export.wanted - len(export.records))
        asked_at = time.monotonic()
        try:
            batch = await scope.client.http.logs_from(scope.channel_id, asked, before=before)
        except discord.Forbidden:
            if not export.records:
                raise _Failed(_NO_HISTORY_TEXT)
            return export.broke_off()
        except discord.HTTPException as e:
            logger.warning(f"reading the history of {scope.where} failed: {e}")
            if not export.records:
                raise _Failed("Discord didn't return the messages. Tell the user to try again "
                              "later.")
            return export.broke_off()
        per_page = time.monotonic() - asked_at
        messages = [m for m in batch or [] if isinstance(m, dict)]
        if not messages:
            return ""
        export.add(messages)
        if export.size > EXPORT_MAX_BYTES:
            return export.too_big()
        ids = [int(m["id"]) for m in messages if str(m.get("id", "")).isdigit()]
        if not ids or len(messages) < asked:
            return ""
        before = min(ids)
    return ""


def _failed(live, spec) -> None:
    """The card of a search or read that did not work says so: a notice
    comes back in later prompts, where "Searching..." would read as still
    going."""
    if live is not None:
        live.show_soon(K.finished(spec, "It didn't work"))


async def _input_dir(context: dict, session) -> str:
    """The run's in/ folder (beside out/, where chat files go), made if
    needed. Absolute, from the workspace's own working directory."""
    out_dir = context.get("out_dir")
    if not out_dir:
        result = await session.exec("pwd", shell=False)
        out_dir = posixpath.join(result.stdout.decode("utf-8", "replace").strip(), "out")
    in_dir = posixpath.join(posixpath.dirname(out_dir), "in")
    if not (await session.exec("mkdir", "-p", in_dir, shell=False)).ok():
        raise RuntimeError(f"could not make {in_dir}")
    return in_dir


async def _channel_in_guild(guild, channel_id: int):
    """The channel or thread with this id, only if it is in THIS server.
    An archived thread is often not in discord.py's cache, so it is fetched;
    Guild.fetch_channel itself refuses a channel of another server
    (InvalidData). Never client.get_channel: that finds any server's."""
    place = guild.get_channel_or_thread(channel_id)
    if place is None:
        try:
            place = await guild.fetch_channel(channel_id)
        except (discord.HTTPException, discord.InvalidData) as e:
            logger.info(f"channel {channel_id} not found for reading: {e}")
            return None
    if getattr(getattr(place, "guild", None), "id", None) != guild.id:
        return None
    return place


_NO_HISTORY_TEXT = "I don't have permission to read that channel's history. Tell the user."


@function_tool
async def read_messages_around(wrapper: RunContextWrapper[dict], message: str,
                               before: int = 5, after: int = 5) -> str:
    """Reads the messages just before and after one message in this server, to see the
    conversation around a search_messages result or a message link someone posted. The
    result says how to read further back or forward.

    Args:
        message: The message's link (https://discord.com/channels/...).
        before: How many earlier messages to show, 0 to 12.
        after: How many later messages to show, 0 to 12.
    """
    context = wrapper.context
    current = context.get("channel")
    guild = getattr(current, "guild", None)
    client = context.get("discord_client")
    if guild is None or client is None:
        return "Reading messages only works in a server. Tell the user."
    context["message_context_calls"] = context.get("message_context_calls", 0) + 1
    parsed = parse_message_link(message)
    if parsed is None:
        return ("That isn't a message link. Use a link from search_messages' results, or ask "
                "the user for the message's link (Copy Message Link in its menu).")
    link_guild, channel_id, message_id = parsed
    # A link can name any server the bot is in; reading another server's
    # messages into this one would leak them.
    if link_guild is not None and link_guild != guild.id:
        return ("That link is to a message in a different server, and I can only read this "
                "one. Tell the user.")
    place = current if channel_id is None else await _channel_in_guild(guild, channel_id)
    if place is None:
        return ("I can't find that channel in this server (it may have been deleted, or I "
                "can't see it). Tell the user.")
    me = getattr(guild, "me", None)
    if me is not None:
        permissions = place.permissions_for(me)
        if not (permissions.read_messages and permissions.read_message_history):
            return _NO_HISTORY_TEXT
    # As search: an age-restricted channel is read only from one.
    if (getattr(place, "is_nsfw", lambda: False)()
            and not getattr(current, "is_nsfw", lambda: False)()):
        return ("That channel is age-restricted, so I can't show its messages here. Tell the "
                "user to ask in an age-restricted channel.")

    before, after = clamp_side(before), clamp_side(after)
    logger.info(f"Reading {before}/{after} messages around {message_id} in {place.id} (call "
                f"{context['message_context_calls']})")
    spec = message_search_card(READING_MESSAGES, f"Around a message in #{place.name}")
    live = await card.notice(current, spec)
    try:
        anchor = await client.http.get_message(place.id, message_id)
        older, newer = await asyncio.gather(
            client.http.logs_from(place.id, before, before=message_id) if before
            else asyncio.sleep(0, []),
            client.http.logs_from(place.id, after, after=message_id) if after
            else asyncio.sleep(0, []))
    except discord.NotFound:
        _failed(live, spec)
        return ("That message doesn't exist any more (it may have been deleted), or the link "
                "is wrong. Tell the user.")
    except discord.Forbidden:
        _failed(live, spec)
        return _NO_HISTORY_TEXT
    except discord.HTTPException as e:
        logger.warning(f"reading messages around {message_id} failed: {e}")
        _failed(live, spec)
        return "Discord didn't return the messages. Tell the user to try again later."
    return format_context(anchor, older, newer, guild.id, place.id, place.name, before, after)


def message_search_tools() -> list:
    return [search_messages, read_messages_around]

"""The search_messages and read_messages_around agent tools: finding old
messages in this server through Discord's own message search, so the bot can
look up something said long before its history window, and then reading the
conversation around one.

The whole server is searched unless a channel is named. Discord only
returns messages from channels the bot itself can read, and
read_messages_around reads the same channels: what the BOT can see, not
only what the person asking can (the owner's choice, as for search).
message_search_format.py holds the query and result rules and the
endpoint's quirks.
"""
import asyncio
import io
import json
import logging
import posixpath
from dataclasses import dataclass
from pathlib import PurePosixPath
import re

import discord
from agents import RunContextWrapper, function_tool
from discord.http import Route

from classes.common import Common
from classes.tool_embeds import (
    COLOR_MESSAGES,
    READING_MESSAGES,
    SAVING_MESSAGES,
    SEARCHING_MESSAGES,
    message_search_line,
)
from classes.message_search_format import (
    EXPORT_DEFAULT, EXPORT_MAX_BYTES, EXPORT_MAX_OFFSET, MAX_LIMIT, NOT_INDEXED_TEXT, clamp_export,
    clamp_side, export_filename, export_record, export_summary, flatten, format_context,
    format_results, not_indexed, parse_message_link, search_params,
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
    await Common.send_tool_discord_embed(
        scope.current, message_search_line(scope.query, scope.where, scope.author),
        COLOR_MESSAGES, SEARCHING_MESSAGES)
    body = await _search_page(scope, limit)
    if isinstance(body, str):
        return body
    text = format_results(body, scope.guild.id, _channel_names(scope.guild), scope.query,
                          scope.where)
    # In a sandbox run (it has a workspace) a big result can go to a file for
    # code to read; only the sandbox has that tool.
    shown, total = len(flatten(body)), int(body.get("total_results") or 0)
    if context.get("session") is not None and total > shown:
        text += (f"\nTo work with all {total} of them (counting, adding up, charting), "
                 f"save_messages_to_file writes them to a file in your workspace.")
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
    where, channel_id = "this server", None
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
    return _Scope(current, guild, client, query, where, channel_id, author, author_id)


async def _search_page(scope: _Scope, limit, offset: int = 0, newest_first: bool = False):
    """One page of Discord's search, or the text to return when it fails."""
    nsfw = bool(getattr(scope.current, "is_nsfw", lambda: False)())
    try:
        body = await scope.client.http.request(
            Route("GET", "/guilds/{guild_id}/messages/search", guild_id=scope.guild.id),
            params=search_params(scope.query, scope.channel_id, scope.author_id, limit,
                                 include_nsfw=nsfw, offset=offset, newest_first=newest_first))
    except discord.Forbidden:
        return ("Discord didn't allow the search (I need Read Message History, and the bot needs "
                "the Message Content intent). Tell the user.")
    except discord.HTTPException as e:
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
                                max_messages: int = EXPORT_DEFAULT) -> str:
    """Saves every message in this server that matches a search to a file in your
    workspace (one JSON object per line, embeds included), for code to count, add up or
    chart. Use it instead of search_messages when you need more than a handful of messages.

    Args:
        query: Words the messages contain (a bot's embed titles and fields count). Leave
            empty to save a channel's or an author's messages.
        channel: Optional: only this channel (its name, #mention or id).
        author: Optional: only messages by this person or bot (their name, @mention or id).
        max_messages: The most to save, newest first, 1 to 10000.
    """
    context = wrapper.context
    session = context.get("session")
    if session is None:
        return "Saving messages to a file only works inside the code sandbox."
    scope = await _scope(context, query, channel, author)
    if isinstance(scope, str):
        return scope
    wanted = clamp_export(max_messages)
    logger.info(f"Saving up to {wanted} messages from {scope.where} for {scope.query!r} "
                f"(author {scope.author_id})")
    await Common.send_tool_discord_embed(
        scope.current, message_search_line(scope.query, scope.where, scope.author),
        COLOR_MESSAGES, SAVING_MESSAGES)
    names = _channel_names(scope.guild)
    records, seen, total, size, stopped = [], set(), 0, 0, ""
    offset = 0
    while len(records) < wanted:
        body = await _search_page(scope, MAX_LIMIT, offset, newest_first=True)
        if isinstance(body, str):
            if not records:
                return body
            stopped = (f"Stopped early: Discord stopped answering after {len(records)} "
                       f"messages, so the file holds only those.")
            break
        total = max(total, int(body.get("total_results") or 0))
        for thread in body.get("threads") or []:
            if thread.get("id") and thread.get("name"):
                names.setdefault(str(thread["id"]), thread["name"])
        hits = flatten(body)
        for message in hits:
            # New messages arriving while it pages shift the pages, so a
            # message can come round twice.
            if str(message.get("id")) in seen or len(records) >= wanted:
                continue
            seen.add(str(message.get("id")))
            line = json.dumps(export_record(message, scope.guild.id, names), ensure_ascii=False)
            size += len(line.encode("utf-8")) + 1
            records.append(line)
        offset += len(hits)
        if size > EXPORT_MAX_BYTES:
            stopped = "Stopped early: the file reached its size limit."
            break
        if not hits or offset >= total:
            break
        if offset > EXPORT_MAX_OFFSET:
            stopped = (f"Stopped at {len(records)}: Discord's search goes no deeper than "
                       f"10000 results. Narrow it (a channel or an author) to reach the rest.")
            break
    if not records:
        return export_summary("", 0, total, None)

    name = export_filename(scope.query, scope.where, scope.author)
    try:
        in_dir = await _input_dir(context, session)
        await session.write(PurePosixPath(in_dir, name),
                            io.BytesIO(("\n".join(records) + "\n").encode("utf-8")))
    except Exception as e:
        logger.warning(f"could not write the message file: {e}")
        return "The messages were found but could not be written to your workspace."
    logger.info(f"Saved {len(records)} of {total} messages to in/{name}")
    return export_summary(f"in/{name}", len(records), total, json.loads(records[0]), stopped)


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
    await Common.send_tool_discord_embed(
        current, f"Around a message in #{place.name}",
        COLOR_MESSAGES, READING_MESSAGES)
    try:
        anchor = await client.http.get_message(place.id, message_id)
        older, newer = await asyncio.gather(
            client.http.logs_from(place.id, before, before=message_id) if before
            else asyncio.sleep(0, []),
            client.http.logs_from(place.id, after, after=message_id) if after
            else asyncio.sleep(0, []))
    except discord.NotFound:
        return ("That message doesn't exist any more (it may have been deleted), or the link "
                "is wrong. Tell the user.")
    except discord.Forbidden:
        return _NO_HISTORY_TEXT
    except discord.HTTPException as e:
        logger.warning(f"reading messages around {message_id} failed: {e}")
        return "Discord didn't return the messages. Tell the user to try again later."
    return format_context(anchor, older, newer, guild.id, place.id, place.name, before, after)


def message_search_tools() -> list:
    return [search_messages, read_messages_around]

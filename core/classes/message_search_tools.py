"""The search_messages agent tool: finds old messages in this server through
Discord's own message search, so the bot can look up something said long
before its history window.

The whole server is searched unless a channel is named. Discord only
returns messages from channels the bot itself can read. message_search_format.py
holds the query and result rules and the endpoint's quirks.
"""
import logging
import re

import discord
from agents import RunContextWrapper, function_tool
from discord.http import Route

from classes.common import Common
from classes.message_search_format import (
    MAX_CALLS, NOT_INDEXED_TEXT, TOO_MANY_TEXT, format_results, not_indexed, search_params,
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
    current = context.get("channel")
    guild = getattr(current, "guild", None)
    client = context.get("discord_client")
    if guild is None or client is None:
        return "Message search only works in a server. Tell the user."
    # The model can send null for an optional argument.
    query, channel, author = query or "", channel or "", author or ""
    context["message_search_calls"] = context.get("message_search_calls", 0) + 1
    if context["message_search_calls"] > MAX_CALLS:
        return TOO_MANY_TEXT
    if not query.strip() and not channel.strip() and not author.strip():
        return "Give some words to search for, a channel, or an author."

    where, channel_id = "this server", None
    if channel.strip():
        place = _find_channel(guild, channel, current)
        if place is None:
            return f"There is no channel called \"{channel}\" here. Ask the user which one they mean."
        where, channel_id = f"#{place.name}", place.id
    author_id = None
    if author.strip():
        author_id = await _find_author(guild, author)
        if author_id is None:
            return f"No one called \"{author}\" was found in this server. Ask the user who they mean."

    logger.info(f"Searching {where} for {query.strip()!r} (author {author_id}, call "
                f"{context['message_search_calls']})")
    await Common.send_tool_discord_embed(
        current, f"Searching {where} for: {query.strip() or 'recent messages'}"
                 + (f" (by {author.strip()})" if author.strip() else ""))
    nsfw = bool(getattr(current, "is_nsfw", lambda: False)())
    try:
        body = await client.http.request(
            Route("GET", "/guilds/{guild_id}/messages/search", guild_id=guild.id),
            params=search_params(query, channel_id, author_id, limit, include_nsfw=nsfw))
    except discord.Forbidden:
        return ("Discord didn't allow the search (I need Read Message History, and the bot needs "
                "the Message Content intent). Tell the user.")
    except discord.HTTPException as e:
        logger.warning(f"message search failed: {e}")
        return "The search failed on Discord's side. Tell the user to try again later."
    if not_indexed(body):
        return NOT_INDEXED_TEXT
    names = {str(c.id): c.name for c in list(guild.channels) + list(guild.threads)}
    return format_results(body, guild.id, names, query.strip(), where)


def message_search_tools() -> list:
    return [search_messages]

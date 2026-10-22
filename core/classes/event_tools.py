"""Agent tools for Discord scheduled events: create, list, change/cancel.

The person asking must hold Create Events (or Manage Events) themselves.
The bot holds the permission for everyone it talks to, so without that
check any member could have it make events the server does not let them
make. The check also runs in an automatic run (a follow-up), as the
entry's last editor, since that is who the event is made for.

The bot only changes events it made: Create Events covers your own events,
and the bot's own are the ones it can vouch for.
"""
import asyncio
import logging
import os
import re
from datetime import datetime, timezone

import discord
from agents import function_tool, RunContextWrapper

from classes.event_format import (discord_timestamp, events_enabled, format_event, garbled, utc_text,
                                  validate)
from classes.tool_functions import add_emoji_to_message

logger = logging.getLogger(__name__)

# Upcoming events list_events reports, soonest first.
LIST_LIMIT = 10
# Interested people named per event; the count is always shown.
INTERESTED_NAMES = 20
# Bound on looking those names up (one request per event, all at once).
INTERESTED_TIMEOUT_SECONDS = 5
# Failed event calls in one reply before the model is told to stop. Seen
# live: a model given the same refusal called create_event 15 times and
# update_event 14 times, then told the user it had worked.
MAX_FAILED_CALLS = 3
LIVE = (discord.EventStatus.scheduled, discord.EventStatus.active)


def _timezone() -> str:
    from classes.automation_policy import resolve_timezone
    return resolve_timezone(os.getenv("AUTOMATIONS_TIMEZONE", "Europe/London"))


def _guild_and_actor(context):
    """The server and the person asking. A voice turn or an automatic run
    has no message; the speaker or the entry's editor is `user_id`."""
    guild = getattr(context.get("channel"), "guild", None)
    actor = getattr(context.get("original_message"), "author", None)
    if actor is None and guild is not None and context.get("user_id"):
        actor = guild.get_member(context["user_id"])
    # A webhook's author is a plain User with no server permissions (seen
    # live: it raised instead of refusing). A real member is looked up;
    # anything else is refused as having no permission.
    if actor is not None and not hasattr(actor, "guild_permissions"):
        actor = guild.get_member(actor.id) if guild is not None else None
    return guild, actor


def _may_make_events(perms) -> bool:
    return bool(getattr(perms, "create_events", False) or getattr(perms, "manage_events", False))


def _refusal(guild, actor, place=None) -> str:
    """"" when both the person and the bot may make events (in `place`, a
    voice channel, when given), else what to tell the user."""
    if guild is None:
        return "No event was made: events only exist in servers. Tell the user."
    def perms(member):
        return place.permissions_for(member) if place is not None else member.guild_permissions
    if actor is None or not _may_make_events(perms(actor)):
        return ("Not done: the person asking needs the Create Events permission in this server "
                "to have me make or change events. Tell them.")
    if guild.me is None or not _may_make_events(perms(guild.me)):
        return ("Not done: I need the Create Events permission in this server. Tell the user to "
                "ask a server admin to give my role it.")
    return ""


def _find_voice_channel(guild, text):
    """A voice or stage channel by id, mention or name (case-insensitive;
    a unique partial name also matches)."""
    raw = text.strip()
    wanted = raw.lstrip("#").casefold()
    channels = list(guild.voice_channels) + list(guild.stage_channels)
    for channel in channels:
        if raw in (str(channel.id), f"<#{channel.id}>"):
            return channel
    exact = [c for c in channels if c.name.casefold() == wanted]
    if exact:
        return exact[0]
    partial = [c for c in channels if wanted in c.name.casefold()]
    if len(partial) == 1:
        return partial[0]
    # The model often wraps the name in words ("General voice channel", or
    # worse): a channel whose whole name appears in the text, the longest
    # such name if several do.
    within = [c for c in channels
              if re.search(r"(?<!\w)" + re.escape(c.name.casefold()) + r"(?!\w)", wanted)]
    within.sort(key=lambda c: len(c.name), reverse=True)
    if within and (len(within) == 1 or len(within[0].name) > len(within[1].name)):
        return within[0]
    return None


def _failed(context, tool, problem) -> str:
    """Log a call that did nothing and return what the model is told; past
    MAX_FAILED_CALLS in one reply, that is to stop and tell the user."""
    logger.info(f"{tool} not run: {problem}")
    context["event_tool_failures"] = context.get("event_tool_failures", 0) + 1
    if context["event_tool_failures"] >= MAX_FAILED_CALLS:
        return (f"{problem}\nThat is {context['event_tool_failures']} event calls in this reply that did "
                "nothing. Stop calling event tools for this request: tell the user it was not done "
                "and why, in plain words.")
    return problem


def _rejected(e: Exception) -> str:
    if isinstance(e, discord.Forbidden):
        return ("Not done: Discord says I am not allowed (I need the Create Events permission). "
                "Tell the user to ask a server admin to give my role it.")
    text = getattr(e, "text", "") or str(e)
    return f"Not done: Discord rejected it ({text}). Tell the user and do not retry."


def _check_create(context, guild, actor, name, start, end, description, location, voice_channel):
    """Everything create_event checks before calling Discord ->
    (voice channel or None, validated fields, problem text or "")."""
    junk = garbled(name, start, end, description, location, voice_channel)
    if junk:
        return None, {}, junk
    place = None
    if guild is not None and voice_channel.strip():
        place = _find_voice_channel(guild, voice_channel)
        if place is None:
            names = ", ".join(c.name for c in list(guild.voice_channels) + list(guild.stage_channels))
            return None, {}, (f"No event was made: there is no voice channel called {voice_channel!r}. "
                              f"The voice channels are: {names or 'none'}. Call again with one of them, "
                              "or with a location instead.")
    elif guild is not None and not location.strip():
        return None, {}, ("No event was made: it needs a place. Ask the user which voice channel it "
                          "is in, or where it happens, then call again.")
    refusal = _refusal(guild, actor, place)
    if refusal:
        return place, {}, refusal
    fields, error = validate(name=name, start=start, end=end, description=description,
                             location="" if place else location, tz_name=_timezone(),
                             now=datetime.now(timezone.utc), needs_end=place is None)
    if error:
        return place, {}, error
    # As in create_poll: the model sometimes repeats a call within one reply.
    if context.get("event_tool_calls", 0) > 0:
        return place, {}, ("An event was already made for this request, so this duplicate call "
                           "was skipped. Nothing further is needed.")
    return place, fields, ""


@function_tool
async def create_event(wrapper: RunContextWrapper[dict], name: str, start: str, end: str = "",
                       description: str = "", location: str = "", voice_channel: str = "") -> str:
    """Creates a Discord server event that people can mark themselves Interested in. Use it
    when someone asks to make or schedule an event. To find a time that suits several
    people, post a poll of times first and schedule a follow-up for when it ends. The event
    is held in a voice channel (voice_channel) or somewhere else (location); if the user
    has not said where, ask them.

    Args:
        name: The event's title, up to 100 characters.
        start: When it starts as an ISO date and time, e.g. 2026-10-24T19:00. Without an offset it is read in the server's timezone.
        end: When it ends, same format. Optional; an event with a location lasts 2 hours if no end is given.
        description: Optional details, up to 1000 characters.
        location: Where it happens, for an event outside a voice channel, up to 100 characters.
        voice_channel: The name of the voice channel to hold it in, instead of a location.
    """
    context = wrapper.context
    guild, actor = _guild_and_actor(context)
    place, fields, problem = _check_create(context, guild, actor, name, start, end, description,
                                           location, voice_channel)
    if problem:
        return _failed(context, "create_event", problem)

    kwargs = {"name": fields["name"], "start_time": fields["start"],
              "privacy_level": discord.PrivacyLevel.guild_only,
              "reason": f"Asked for by {getattr(actor, 'name', 'a member')}"}
    if place is not None:
        kwargs["entity_type"] = (discord.EntityType.stage_instance
                                 if isinstance(place, discord.StageChannel) else discord.EntityType.voice)
        kwargs["channel"] = place
        where = f" in the {place.name} voice channel"
    else:
        kwargs["entity_type"] = discord.EntityType.external
        kwargs["location"] = fields["location"]
        where = f" at {fields['location']}"
    if "end" in fields:
        kwargs["end_time"] = fields["end"]
    if "description" in fields:
        kwargs["description"] = fields["description"]
    logger.info(f"Creating event: {fields['name']} at {fields['start'].isoformat()}")
    try:
        event = await guild.create_scheduled_event(**kwargs)
    except Exception as e:
        logger.warning(f"Failed to create an event: {e}")
        return _failed(context, "create_event", _rejected(e))
    context["event_tool_calls"] = context.get("event_tool_calls", 0) + 1
    await add_emoji_to_message(context.get("original_message"), "📅")
    ends = (" It ends 2 hours after it starts, since no end was given."
            if fields.get("default_end") else "")
    return (f"Event created: \"{fields['name']}\"{where}, starting {utc_text(fields['start'])}.{ends} "
            f"Link: {event.url} . Share the link, and write the time as "
            f"{discord_timestamp(fields['start'])} so Discord shows it in each reader's own "
            "timezone; people can click Interested on the event.")


async def _interested_names(events) -> dict:
    """{event id: names} for events with anyone interested. A failed or
    slow lookup leaves the names out (the count still shows), as
    polls.poll_voters does; it never costs the reply."""
    wanted = [e for e in events if e.user_count]

    async def _one(event):
        try:
            return [u.display_name async for u in event.users(limit=INTERESTED_NAMES)]
        except Exception as e:
            logger.warning(f"Could not read who is interested in event {event.id}: {e}")
            return None

    try:
        names = await asyncio.wait_for(asyncio.gather(*(_one(e) for e in wanted)),
                                       INTERESTED_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        logger.warning("Reading who is interested in events timed out")
        return {}
    return {e.id: n for e, n in zip(wanted, names) if n is not None}


def _where(guild, event) -> str:
    if event.channel_id:
        channel = guild.get_channel(event.channel_id)
        return f"the {channel.name} voice channel" if channel else "a voice channel"
    return event.location or "not given"


@function_tool
async def list_events(wrapper: RunContextWrapper[dict]) -> str:
    """Lists this server's upcoming events with who is interested in each. Use it when
    someone asks what events are coming up, who is going, or before changing an event."""
    guild, _ = _guild_and_actor(wrapper.context)
    if guild is None:
        return "Events only exist in servers. Tell the user."
    try:
        events = await guild.fetch_scheduled_events(with_counts=True)
    except Exception as e:
        logger.warning(f"Could not read this server's events: {e}")
        return "Events could not be looked up right now. Tell the user."
    upcoming = sorted((e for e in events if e.status in LIVE), key=lambda e: e.start_time)[:LIST_LIMIT]
    if not upcoming:
        return "This server has no upcoming events."
    names = await _interested_names(upcoming)
    bot_id = getattr(guild.me, "id", None)
    found = [format_event({
        "id": e.id, "name": e.name, "status": e.status.name, "start": e.start_time,
        "end": e.end_time, "where": _where(guild, e), "interested": e.user_count,
        "names": names.get(e.id), "by_bot": e.creator_id == bot_id, "description": e.description,
    }) for e in upcoming]
    return "Upcoming events, soonest first:\n\n" + "\n\n".join(found)


async def _find_event(guild, wanted):
    """(event, problem) for an event id, link or name. The model often
    passes the name it was given ("Test Night") instead of an id, so an
    upcoming event with that name, or a unique part of it, also matches."""
    text = wanted.strip()
    ident = re.search(r"(\d{15,})/?$", text)
    if ident:
        try:
            return await guild.fetch_scheduled_event(int(ident.group(1))), ""
        except discord.NotFound:
            return None, "There is no event with that id. Call list_events to find it."
    events = [e for e in await guild.fetch_scheduled_events() if e.status in LIVE]
    folded = text.casefold()
    found = ([e for e in events if e.name.casefold() == folded]
             or [e for e in events if folded and folded in e.name.casefold()])
    if len(found) == 1:
        return found[0], ""
    if not found:
        names = ", ".join(f"{e.name} (id {e.id})" for e in events) or "none"
        return None, f"There is no upcoming event called {text!r}. The upcoming events are: {names}."
    names = ", ".join(f"{e.name} (id {e.id}, {utc_text(e.start_time)})" for e in found)
    return None, f"Several events match {text!r}: {names}. Ask the user which one, then call again with its id."


@function_tool
async def update_event(wrapper: RunContextWrapper[dict], event: str, name: str = "",
                       start: str = "", end: str = "", description: str = "",
                       location: str = "", cancel: bool = False) -> str:
    """Changes or cancels an event I made. Give only what changes; moving the start keeps
    the event's length.

    Args:
        event: The event's id from list_events, or its name.
        name: A new title.
        start: A new start as an ISO date and time, e.g. 2026-10-24T21:00. Without an offset it is read in the server's timezone.
        end: A new end, same format.
        description: New details.
        location: A new place, only for an event outside a voice channel.
        cancel: True to cancel the event instead of changing it.
    """
    context = wrapper.context
    guild, actor = _guild_and_actor(context)
    if guild is None:
        return _failed(context, "update_event", "Events only exist in servers. Tell the user.")
    junk = garbled(event, name, start, end, description, location)
    if junk:
        return _failed(context, "update_event", junk)
    try:
        found, problem = await _find_event(guild, event)
    except Exception as e:
        logger.warning(f"Could not read event {event!r}: {e}")
        return _failed(context, "update_event", "The event could not be looked up right now. Tell the user.")
    if problem:
        return _failed(context, "update_event", problem)
    if found.creator_id != getattr(guild.me, "id", None):
        return _failed(context, "update_event", (
            "Not done: I can only change events I made, and someone else made this one. "
            "Tell the user to change it in Discord."))
    place = guild.get_channel(found.channel_id) if found.channel_id else None
    refusal = _refusal(guild, actor, place)
    if refusal:
        return _failed(context, "update_event", refusal)
    if found.status not in LIVE:
        return _failed(context, "update_event",
                       f"Not done: that event has already {found.status.name}. Tell the user.")

    try:
        if cancel:
            logger.info(f"Cancelling event {found.id}")
            # Discord cancels only an event that has not started; one that
            # is on now is ended instead.
            if found.status == discord.EventStatus.active:
                await found.end()
                return f"Event \"{found.name}\" was ended. Tell the user."
            await found.cancel()
            return f"Event \"{found.name}\" was cancelled. Tell the user."

        if location.strip() and found.entity_type != discord.EntityType.external:
            return _failed(context, "update_event", (
                "Nothing was changed: this event is in a voice channel, so it has no "
                "location. Leave location out."))
        fields, error = validate(name=name, start=start, end=end, description=description,
                                 location=location, tz_name=_timezone(),
                                 now=datetime.now(timezone.utc),
                                 needs_end=found.entity_type == discord.EntityType.external,
                                 current={"start": found.start_time, "end": found.end_time})
        if error:
            return _failed(context, "update_event", error)
        names = {"name": "name", "description": "description", "location": "location",
                 "start": "start_time", "end": "end_time"}
        changes = {names[k]: v for k, v in fields.items() if k in names}
        if not changes:
            return _failed(context, "update_event",
                           "Nothing was changed: give at least one field to change, or cancel=True.")
        changes["reason"] = f"Asked for by {getattr(actor, 'name', 'a member')}"
        logger.info(f"Changing event {found.id}: {sorted(changes)}")
        updated = await found.edit(**changes)
    except Exception as e:
        logger.warning(f"Failed to change event {found.id}: {e}")
        return _failed(context, "update_event", _rejected(e))
    start_at = updated.start_time
    return (f"Event updated: \"{updated.name}\", starting {utc_text(start_at)}. Link: {updated.url} . "
            f"Write the time as {discord_timestamp(start_at)} so each reader sees it in their "
            "own timezone.")


def event_tools() -> list:
    if not events_enabled():
        return []
    return [create_event, list_events, update_event]

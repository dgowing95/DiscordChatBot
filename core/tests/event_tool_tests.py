"""Tests for the event agent tools (core/classes/event_tools.py).

To run this pytest file from the command line, use:
    pytest core/tests/event_tool_tests.py
"""
import asyncio
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import discord
import pytest

UTC = timezone.utc
BOT_ID = 999
EVENT_ID = 111111111111111110


def _tool_context(context, name):
    from agents.tool_context import ToolContext
    return ToolContext(context=context, tool_name=name, tool_call_id="t1", tool_arguments="{}")


async def _invoke(tool, context, **arguments):
    from classes import event_tools
    return await getattr(event_tools, tool).on_invoke_tool(
        _tool_context(context, tool), json.dumps(arguments))


def _member(create=True, manage=False, ident=1, name="alice"):
    return SimpleNamespace(id=ident, name=name,
                           guild_permissions=SimpleNamespace(create_events=create, manage_events=manage))


def _voice(name="Games", ident=55):
    return SimpleNamespace(id=ident, name=name, permissions_for=lambda member: member.guild_permissions)


def _env(actor=None, bot=None, voice=(), message=True):
    actor = actor or _member()
    guild = MagicMock()
    guild.me = bot or _member(ident=BOT_ID, name="bot")
    guild.voice_channels = list(voice)
    guild.stage_channels = []
    guild.get_member = lambda ident: actor if ident == actor.id else None
    guild.create_scheduled_event = AsyncMock(
        return_value=SimpleNamespace(url="https://discord.com/events/1/2"))
    original = None
    if message:
        original = MagicMock()
        original.author = actor
        original.add_reaction = AsyncMock()
    context = {"channel": SimpleNamespace(guild=guild), "original_message": original,
               "user_id": actor.id}
    return guild, context


def _soon(days=2, hour=19):
    day = datetime.now(UTC) + timedelta(days=days)
    return day.replace(hour=hour, minute=0, second=0, microsecond=0)


@pytest.mark.asyncio
async def test_creates_a_voice_channel_event_and_reacts():
    guild, context = _env(voice=[_voice()])
    start = _soon()
    result = await _invoke("create_event", context, name="Game night",
                           start=start.isoformat(), voice_channel="games")
    kwargs = guild.create_scheduled_event.await_args.kwargs
    assert kwargs["entity_type"] == discord.EntityType.voice
    assert kwargs["channel"].name == "Games"
    assert kwargs["privacy_level"] == discord.PrivacyLevel.guild_only
    assert kwargs["start_time"] == start and "location" not in kwargs and "end_time" not in kwargs
    assert "https://discord.com/events/1/2" in result
    assert f"<t:{int(start.timestamp())}:F>" in result
    context["original_message"].add_reaction.assert_awaited_once_with("📅")
    assert len(context["events_made"]) == 1


@pytest.mark.asyncio
async def test_finds_the_voice_channel_inside_extra_words():
    """Seen live: the model passed 'General voice channel" (the one on this
    server)...' as the channel name."""
    from classes.event_tools import _find_voice_channel
    guild, _ = _env(voice=[_voice("General"), _voice("General Chat", 56), _voice("Games", 57)])
    assert _find_voice_channel(guild, 'General voice channel" (the one here)').name == "General"
    assert _find_voice_channel(guild, "the General Chat voice channel").name == "General Chat"
    assert _find_voice_channel(guild, "<#57>").name == "Games"
    assert _find_voice_channel(guild, "Cinema") is None


@pytest.mark.asyncio
async def test_creates_an_external_event_with_a_default_end():
    guild, context = _env()
    start = _soon()
    result = await _invoke("create_event", context, name="Pub quiz", start=start.isoformat(),
                           location="The Crown")
    kwargs = guild.create_scheduled_event.await_args.kwargs
    assert kwargs["entity_type"] == discord.EntityType.external
    assert kwargs["location"] == "The Crown"
    assert kwargs["end_time"] - kwargs["start_time"] == timedelta(hours=2)
    assert "ends 2 hours after" in result


@pytest.mark.asyncio
async def test_asks_where_when_no_place_is_given_and_names_voice_channels():
    guild, context = _env(voice=[_voice("Games"), _voice("Lounge", 56)])
    result = await _invoke("create_event", context, name="x", start=_soon().isoformat())
    assert "needs a place" in result
    result = await _invoke("create_event", context, name="x", start=_soon().isoformat(),
                           voice_channel="Cinema")
    assert "Games, Lounge" in result
    guild.create_scheduled_event.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_person_asking_needs_the_permission():
    guild, context = _env(actor=_member(create=False))
    result = await _invoke("create_event", context, name="x", start=_soon().isoformat(),
                           location="here")
    assert "person asking needs the Create Events permission" in result
    guild.create_scheduled_event.assert_not_awaited()
    # Manage Events covers it too.
    guild, context = _env(actor=_member(create=False, manage=True))
    await _invoke("create_event", context, name="x", start=_soon().isoformat(), location="here")
    guild.create_scheduled_event.assert_awaited_once()


@pytest.mark.asyncio
async def test_the_bot_needs_the_permission():
    guild, context = _env(bot=_member(create=False, ident=BOT_ID))
    result = await _invoke("create_event", context, name="x", start=_soon().isoformat(),
                           location="here")
    assert "I need the Create Events permission" in result
    guild.create_scheduled_event.assert_not_awaited()


@pytest.mark.asyncio
async def test_forbidden_is_reported_and_does_not_count():
    guild, context = _env()
    guild.create_scheduled_event.side_effect = discord.Forbidden(
        MagicMock(status=403, reason="Forbidden"), "no")
    result = await _invoke("create_event", context, name="x", start=_soon().isoformat(),
                           location="here")
    assert "not allowed" in result
    assert not context.get("events_made")


@pytest.mark.asyncio
async def test_bad_input_sends_nothing_and_a_second_call_is_skipped():
    guild, context = _env()
    result = await _invoke("create_event", context, name="x", start="2020-01-01T10:00",
                           location="here")
    assert "in the past" in result
    guild.create_scheduled_event.assert_not_awaited()
    await _invoke("create_event", context, name="x", start=_soon().isoformat(), location="here")
    result = await _invoke("create_event", context, name="X", start=_soon().isoformat(),
                           location="here")
    assert "repeat was skipped" in result
    assert guild.create_scheduled_event.await_count == 1
    # A different event in the same reply is made ("Monday and Wednesday").
    await _invoke("create_event", context, name="x", start=_soon(days=4).isoformat(), location="here")
    assert guild.create_scheduled_event.await_count == 2
    # Repeats never count towards the stop-calling limit.
    for _ in range(3):
        result = await _invoke("create_event", context, name="x", start=_soon().isoformat(),
                               location="here")
    assert "Stop calling" not in result


@pytest.mark.asyncio
async def test_works_without_a_message_as_the_user_id():
    """A voice turn or a follow-up: no message, the actor is user_id."""
    guild, context = _env(message=False)
    await _invoke("create_event", context, name="x", start=_soon().isoformat(), location="here")
    guild.create_scheduled_event.assert_awaited_once()
    # Someone who has left the server cannot be checked, so is refused.
    guild, context = _env(message=False)
    context["user_id"] = 12345
    result = await _invoke("create_event", context, name="x", start=_soon().isoformat(),
                           location="here")
    assert "needs the Create Events permission" in result


@pytest.mark.asyncio
async def test_a_webhook_author_is_refused_not_an_error():
    guild, context = _env()
    context["original_message"].author = SimpleNamespace(id=4242, name="Captain Hook")
    result = await _invoke("create_event", context, name="x", start=_soon().isoformat(),
                           location="here")
    assert "needs the Create Events permission" in result
    guild.create_scheduled_event.assert_not_awaited()


def _event(ident, start, *, creator=BOT_ID, status=discord.EventStatus.scheduled, users=(),
           count=None, external=True, slow=False):
    event = MagicMock()
    event.id, event.name, event.start_time = ident, f"Event {ident}", start
    event.end_time = start + timedelta(hours=2)
    event.status, event.creator_id, event.description = status, creator, ""
    event.user_count = len(users) if count is None else count
    event.channel_id = None if external else 55
    event.location = "The Crown" if external else None
    event.entity_type = discord.EntityType.external if external else discord.EntityType.voice
    event.url = f"https://discord.com/events/1/{ident}"

    async def _users(limit=None):
        if slow:
            await asyncio.sleep(1)
        for name in users:
            yield SimpleNamespace(display_name=name)
    event.users = _users
    return event


@pytest.mark.asyncio
async def test_list_events_shows_upcoming_with_interested_names(monkeypatch):
    guild, context = _env()
    later, sooner = _soon(5), _soon(1)
    guild.fetch_scheduled_events = AsyncMock(return_value=[
        _event(1, later, users=("carol",), creator=7),
        _event(2, sooner, users=("alice", "bob")),
        _event(3, sooner, status=discord.EventStatus.canceled),
    ])
    result = await _invoke("list_events", context)
    assert result.index("Event 2") < result.index("Event 1") and "Event 3" not in result
    assert "Interested: 2 (alice, bob)" in result
    assert "Made by someone else" in result and "Made by me" in result


@pytest.mark.asyncio
async def test_list_events_falls_back_to_counts_when_names_are_slow(monkeypatch):
    from classes import event_tools
    monkeypatch.setattr(event_tools, "INTERESTED_TIMEOUT_SECONDS", 0.01)
    guild, context = _env()
    guild.fetch_scheduled_events = AsyncMock(return_value=[
        _event(1, _soon(), users=("alice",), slow=True)])
    result = await _invoke("list_events", context)
    assert "Interested: 1" in result and "alice" not in result


@pytest.mark.asyncio
async def test_update_refuses_an_event_someone_else_made():
    guild, context = _env()
    event = _event(EVENT_ID, _soon(), creator=7)
    guild.fetch_scheduled_event = AsyncMock(return_value=event)
    result = await _invoke("update_event", context, event=str(EVENT_ID), cancel=True)
    assert "only change events I made" in result
    event.cancel.assert_not_called()


@pytest.mark.asyncio
async def test_update_moves_the_start_and_keeps_the_length():
    guild, context = _env()
    start = _soon()
    event = _event(EVENT_ID, start)
    event.edit = AsyncMock(return_value=SimpleNamespace(
        name="Game night", start_time=start + timedelta(hours=1), url=event.url))
    guild.fetch_scheduled_event = AsyncMock(return_value=event)
    new_start = start + timedelta(hours=1)
    result = await _invoke("update_event", context, event=str(EVENT_ID), start=new_start.isoformat())
    kwargs = event.edit.await_args.kwargs
    assert kwargs["start_time"] == new_start
    assert kwargs["end_time"] == new_start + timedelta(hours=2)
    assert "Event updated" in result


@pytest.mark.asyncio
async def test_update_cancels_or_ends_and_rejects_bad_ids():
    guild, context = _env()
    event = _event(EVENT_ID, _soon())
    event.cancel, event.end = AsyncMock(), AsyncMock()
    guild.fetch_scheduled_event = AsyncMock(return_value=event)
    assert "cancelled" in await _invoke("update_event", context, event=str(EVENT_ID), cancel=True)
    event.cancel.assert_awaited_once()
    event.status = discord.EventStatus.active
    assert "ended" in await _invoke("update_event", context, event=str(EVENT_ID), cancel=True)
    event.end.assert_awaited_once()
    guild.fetch_scheduled_event.side_effect = discord.NotFound(MagicMock(status=404, reason="x"), "no")
    assert "no event with that id" in await _invoke("update_event", context, event="111111111111111119")


@pytest.mark.asyncio
async def test_update_needs_the_permission_too():
    guild, context = _env(actor=_member(create=False))
    event = _event(EVENT_ID, _soon())
    guild.fetch_scheduled_event = AsyncMock(return_value=event)
    result = await _invoke("update_event", context, event=str(EVENT_ID), cancel=True)
    assert "needs the Create Events permission" in result
    event.cancel.assert_not_called()


@pytest.mark.asyncio
async def test_update_finds_the_event_by_name():
    """Seen live: the model passed "Test Night" instead of an id."""
    guild, context = _env()
    start = _soon()
    night, quiz = _event(EVENT_ID, start), _event(EVENT_ID + 1, start)
    night.name, quiz.name = "Test Night", "Pub quiz"
    night.cancel = AsyncMock()
    guild.fetch_scheduled_events = AsyncMock(return_value=[night, quiz])
    assert "cancelled" in await _invoke("update_event", context, event="test night", cancel=True)
    night.cancel.assert_awaited_once()
    result = await _invoke("update_event", context, event="Cinema", cancel=True)
    assert "no upcoming event called 'Cinema'" in result and "Test Night (id" in result
    quiz.name = "Test Night two"
    night.name = "Test Night one"
    assert "Several events match" in await _invoke("update_event", context, event="test night", cancel=True)


@pytest.mark.asyncio
async def test_repeated_failures_tell_the_model_to_stop():
    """Seen live: the same refusal, 15 calls in one reply."""
    guild, context = _env(actor=_member(create=False))
    for _ in range(2):
        result = await _invoke("create_event", context, name="x", start=_soon().isoformat(),
                               location="here")
        assert "Stop calling" not in result
    result = await _invoke("create_event", context, name="x", start=_soon().isoformat(),
                           location="here")
    assert "needs the Create Events permission" in result and "Stop calling event tools" in result


@pytest.mark.asyncio
async def test_garbled_arguments_change_nothing():
    """Seen live: a description holding '</parameter>> 5pm ... "start">2026-10-23T+48:45'."""
    guild, context = _env()
    event = _event(EVENT_ID, _soon())
    event.edit = AsyncMock()
    guild.fetch_scheduled_event = AsyncMock(return_value=event)
    result = await _invoke("update_event", context, event=str(EVENT_ID),
                           description='</parameter>> 5pm on Friday, 20"start">2026-10-23T21:00')
    assert "ran into each other" in result
    event.edit.assert_not_awaited()
    result = await _invoke("create_event", context, name="x", start=_soon().isoformat(),
                           voice_channel='General" (the one here) "start">')
    assert "ran into each other" in result
    guild.create_scheduled_event.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_refused_member_is_not_shown_the_channel_list():
    """The "no such channel" answer lists every voice channel, hidden ones
    included, so the permission is checked first."""
    guild, context = _env(actor=_member(create=False), voice=[_voice("Secret staff room")])
    result = await _invoke("create_event", context, name="x", start=_soon().isoformat(),
                           voice_channel="Cinema")
    assert "needs the Create Events permission" in result and "Secret" not in result


def test_a_failed_create_says_no_event_was_made():
    from classes.event_format import validate
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    _, error = validate(name="", start="2026-10-24T19:00", tz_name="Europe/London", now=now,
                        needs_end=False)
    assert error.startswith("No event was made:")
    _, error = validate(start="nonsense", tz_name="Europe/London", now=now, needs_end=False,
                        current={"start": now, "end": None})
    assert error.startswith("Nothing was changed:")

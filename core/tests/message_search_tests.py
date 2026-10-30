"""Tests for Discord message search: the pure query/result rules
(core/classes/message_search_format.py) and the search_messages tool
(core/classes/message_search_tools.py), against a faked Discord.

To run this pytest file from the command line, use:
    pytest core/tests/message_search_tests.py
"""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import discord
import pytest

from classes import message_search_format as msf

GUILD_ID = 100000000000000001
GENERAL_ID = 200000000000000002
MEMES_ID = 200000000000000003
THREAD_ID = 300000000000000004
ALICE_ID = 400000000000000005


def _hit(ident, channel_id=GENERAL_ID, content="the trip link is https://example.com/trip",
         **extra):
    return {"id": str(ident), "channel_id": str(channel_id), "content": content,
            "timestamp": "2026-03-01T12:00:00.000000+00:00",
            "author": {"id": str(ALICE_ID), "username": "alice", "global_name": "Alice"},
            "attachments": [], "embeds": [], **extra}


def _body(*hits, **extra):
    return {"total_results": len(hits), "messages": [[h] for h in hits], **extra}


def test_params_for_words_and_for_listing():
    assert msf.search_params("trip link", GENERAL_ID, ALICE_ID, 5) == [
        ("limit", "5"), ("include_nsfw", "false"), ("content", "trip link"),
        ("sort_by", "relevance"), ("channel_id", str(GENERAL_ID)), ("author_id", str(ALICE_ID))]
    assert msf.search_params("", None, ALICE_ID, 99, include_nsfw=True) == [
        ("limit", "25"), ("include_nsfw", "true"), ("sort_by", "timestamp"),
        ("sort_order", "desc"), ("author_id", str(ALICE_ID))]
    assert ("limit", "10") in msf.search_params("x", limit="lots")


def test_not_indexed():
    assert msf.not_indexed({"message": "Index not yet available. Try again later", "code": 110000,
                            "documents_indexed": 0, "retry_after": 2})
    assert msf.not_indexed(None)
    assert not msf.not_indexed(_body())


def test_flatten_takes_the_hit_from_each_group():
    body = {"messages": [[{"id": "1"}, {"id": "2", "hit": True}], [{"id": "3"}], {"id": "4"}, []]}
    assert [m["id"] for m in msf.flatten(body)] == ["2", "3", "4"]


def test_format_results():
    long_text = "word " * 100
    body = _body(_hit(1), _hit(2, THREAD_ID, long_text, attachments=[{"filename": "map.png"}],
                                embeds=[{"title": "Trip plan"}]),
                 total_results=7)
    body["threads"] = [{"id": str(THREAD_ID), "name": "trip-thread"}]
    text = msf.format_results(body, GUILD_ID, {str(GENERAL_ID): "general"}, "trip")
    lines = text.splitlines()
    assert lines[0].startswith("7 messages found in this server; showing 2, best matches first.")
    assert lines[1] == ("- Alice, <t:1772366400:f>, in #general: the trip link is "
                        "https://example.com/trip")
    assert lines[2] == f"  https://discord.com/channels/{GUILD_ID}/{GENERAL_ID}/1"
    assert "in #trip-thread:" in lines[3]
    assert "…" in lines[3] and "[attached: map.png]" in lines[3] and "[embed: Trip plan]" in lines[3]


def test_format_no_results_and_indexing_note():
    text = msf.format_results(_body(doing_deep_historical_index=True), GUILD_ID, {}, "x", "#memes")
    assert text.startswith('No messages found in #memes for "x".')
    assert "still indexing" in text


def test_format_unknown_channel_and_member_nick():
    hit = _hit(9, channel_id=999, member={"nick": "Ali"})
    text = msf.format_results(_body(hit), GUILD_ID, {}, "")
    assert "- Ali," in text and "in <#999>:" in text and "newest first" in text


# --- the tool --------------------------------------------------------------

def _channel(ident, name, nsfw=False):
    return SimpleNamespace(id=ident, name=name, is_nsfw=lambda: nsfw)


def _env(body=None, members=(), request_error=None, nsfw=False):
    general = _channel(GENERAL_ID, "general", nsfw)
    memes = _channel(MEMES_ID, "memes")
    thread = _channel(THREAD_ID, "trip-thread")
    by_id = {c.id: c for c in (general, memes, thread)}
    guild = SimpleNamespace(
        id=GUILD_ID, channels=[general, memes], threads=[thread],
        get_channel_or_thread=lambda i: by_id.get(i),
        get_member_named=lambda name: None,
        query_members=AsyncMock(return_value=list(members)))
    general.guild = guild
    http = SimpleNamespace(request=AsyncMock(side_effect=request_error,
                                             return_value=body if body is not None else _body(_hit(1))))
    context = {"channel": general, "discord_client": SimpleNamespace(http=http)}
    return context, http


async def _search(context, **arguments):
    from agents.tool_context import ToolContext
    from classes.message_search_tools import search_messages
    tool_context = ToolContext(context=context, tool_name="search_messages", tool_call_id="t1",
                               tool_arguments="{}")
    return await search_messages.on_invoke_tool(tool_context, json.dumps(arguments))


@pytest.fixture(autouse=True)
def no_embeds(monkeypatch):
    from classes.common import Common
    monkeypatch.setattr(Common, "send_tool_discord_embed", AsyncMock())


@pytest.mark.asyncio
async def test_searches_the_whole_server_by_default():
    context, http = _env()
    text = await _search(context, query="trip link")
    route = http.request.call_args.args[0]
    assert route.method == "GET" and route.path == "/guilds/{guild_id}/messages/search"
    assert str(GUILD_ID) in route.url
    params = dict(http.request.call_args.kwargs["params"])
    assert "channel_id" not in params and params["content"] == "trip link"
    assert "found in this server" in text and "in #general:" in text


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", ["memes", "#Memes", f"<#{MEMES_ID}>", str(MEMES_ID), "mem"])
async def test_a_channel_narrows_it(channel):
    context, http = _env(body=_body())
    text = await _search(context, query="cat", channel=channel)
    assert dict(http.request.call_args.kwargs["params"])["channel_id"] == str(MEMES_ID)
    assert text.startswith('No messages found in #memes for "cat".')


@pytest.mark.asyncio
async def test_here_is_the_current_channel():
    context, http = _env()
    await _search(context, query="x", channel="here")
    assert dict(http.request.call_args.kwargs["params"])["channel_id"] == str(GENERAL_ID)


@pytest.mark.asyncio
async def test_unknown_channel_and_author():
    context, http = _env()
    assert "no channel called" in (await _search(context, query="x", channel="nowhere"))
    assert "No one called" in (await _search(context, query="x", author="Zed"))
    http.request.assert_not_called()


@pytest.mark.asyncio
async def test_author_by_name_mention_or_id():
    alice = SimpleNamespace(id=ALICE_ID, name="alice", display_name="Ali", global_name="Alice")
    context, http = _env(members=[alice])
    await _search(context, query="", author="Alice")
    params = dict(http.request.call_args.kwargs["params"])
    assert params["author_id"] == str(ALICE_ID) and params["sort_by"] == "timestamp"
    await _search(context, query="", author=f"<@{ALICE_ID}>")
    assert dict(http.request.call_args.kwargs["params"])["author_id"] == str(ALICE_ID)


@pytest.mark.asyncio
async def test_a_prefix_match_is_not_taken_for_the_person_asked_for():
    samuel = SimpleNamespace(id=ALICE_ID, name="samuel", display_name="Samuel", global_name="Samuel")
    context, http = _env(members=[samuel])
    assert "No one called" in (await _search(context, query="x", author="Sam"))
    http.request.assert_not_called()


@pytest.mark.asyncio
async def test_nsfw_follows_the_channel_asked_in():
    context, http = _env(nsfw=True)
    await _search(context, query="x")
    assert dict(http.request.call_args.kwargs["params"])["include_nsfw"] == "true"


@pytest.mark.asyncio
async def test_not_indexed_yet():
    context, _ = _env(body={"message": "Index not yet available. Try again later", "code": 110000})
    assert (await _search(context, query="x")) == msf.NOT_INDEXED_TEXT


@pytest.mark.asyncio
async def test_discord_errors():
    response = MagicMock(status=403, reason="Forbidden")
    context, _ = _env(request_error=discord.Forbidden(response, "Missing Access"))
    assert "didn't allow the search" in (await _search(context, query="x"))
    response = MagicMock(status=500, reason="Server Error")
    context, _ = _env(request_error=discord.HTTPException(response, "boom"))
    assert "failed on Discord's side" in (await _search(context, query="x"))


@pytest.mark.asyncio
async def test_calls_per_reply_are_capped():
    context, http = _env()
    for _ in range(msf.MAX_CALLS):
        await _search(context, query="x")
    assert (await _search(context, query="x")) == msf.TOO_MANY_TEXT
    assert http.request.call_count == msf.MAX_CALLS


@pytest.mark.asyncio
async def test_outside_a_server_or_with_nothing_to_search():
    assert "only works in a server" in (await _search({"channel": SimpleNamespace()}, query="x"))
    context, _ = _env()
    assert "Give some words" in (await _search(context, query=""))

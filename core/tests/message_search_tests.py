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


CARD = [{"type": 17, "components": [{"type": 10, "content": "### 🧵 Plan\nChart the cities"},
                                    {"type": 1, "components": [{"type": 2, "label": "Approve"}]}]}]


def test_a_cards_text_is_shown_and_saved():
    card = _hit(1, content="", components=CARD)
    text = msf.format_results(_body(card), GUILD_ID, {}, "x")
    assert "Chart the cities" in text and "(no text)" not in text and "Approve" not in text
    assert msf.export_record(card, GUILD_ID, {})["content"] == "### 🧵 Plan\nChart the cities"


def test_an_embeds_body_and_fields_are_shown():
    # A game-tracker bot's message is nothing but an embed; the title alone
    # says nothing about what happened.
    tracker = {"author": {"name": "Team Ana"}, "title": "Shared Chest",
               "description": "```\nTeam Ana deposited:\nN/A\n\nTeam Ana withdrew:\n"
                              "- 20 x Iron bar (4K)\n```",
               "fields": [{"name": "Player", "value": "Bo", "inline": True},
                          {"name": "Net value", "value": "-4K gp", "inline": True}],
               "footer": {"text": "Powered by Tracker"}, "image": {"url": "https://x/y.png"}}
    text = msf.format_results(_body(_hit(1, content="", embeds=[tracker])), GUILD_ID, {}, "x")
    assert ("[embed: Team Ana - Shared Chest: Team Ana deposited: N/A Team Ana withdrew: "
            "- 20 x Iron bar (4K) | Player: Bo | Net value: -4K gp]") in text
    assert "Powered by" not in text and "```" not in text


def test_embed_text_is_capped_and_empty_embeds_are_skipped():
    long_embed = {"title": "Feed", "description": "word " * 500}
    image_only = {"type": "image", "url": "https://x/y.png"}
    text = msf.format_results(_body(_hit(1, content="", embeds=[long_embed, image_only])),
                              GUILD_ID, {}, "x")
    embed = text[text.index("[embed:"):text.index("]", text.index("[embed:")) + 1]
    assert embed.endswith("…]") and len(embed) <= msf.EMBED_CHARS + len("[embed: …]")
    assert text.count("[embed:") == 1
    assert "[embed: Only a title]" in msf.format_results(
        _body(_hit(2, embeds=[{"title": "Only a title", "fields": [{"name": "", "value": ""}]}])),
        GUILD_ID, {}, "x")


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
def no_cards(monkeypatch):
    from classes import card
    monkeypatch.setattr(card, "notice", AsyncMock())


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
async def test_searches_per_reply_are_not_capped():
    # A brain call that tallies a bot's posts needs many searches; the
    # agent's turn limit is what bounds a loop.
    context, http = _env()
    for _ in range(10):
        assert "found in this server" in (await _search(context, query="x"))
    assert http.request.call_count == 10


@pytest.mark.asyncio
async def test_outside_a_server_or_with_nothing_to_search():
    assert "only works in a server" in (await _search({"channel": SimpleNamespace()}, query="x"))
    context, _ = _env()
    assert "Give some words" in (await _search(context, query=""))


# --- reading around a message ---------------------------------------------

def _link(message_id, channel_id=GENERAL_ID, guild_id=GUILD_ID):
    return f"https://discord.com/channels/{guild_id}/{channel_id}/{message_id}"


def test_parse_message_link():
    assert msf.parse_message_link(_link(5)) == (GUILD_ID, GENERAL_ID, 5)
    assert msf.parse_message_link(f"<https://ptb.discordapp.com/channels/{GUILD_ID}/1/2>") == (
        GUILD_ID, 1, 2)
    assert msf.parse_message_link("see https://discord.com/channels/@me/1/2 here") == (0, 1, 2)
    assert msf.parse_message_link(" 123456789012345678 ") == (None, None, 123456789012345678)
    assert msf.parse_message_link("the trip message") is None
    assert msf.parse_message_link("") is None and msf.parse_message_link(None) is None


def test_clamp_side():
    assert msf.clamp_side(3) == 3 and msf.clamp_side(0) == 0
    assert msf.clamp_side(-2) == 0 and msf.clamp_side(99) == msf.CONTEXT_MAX_SIDE
    assert msf.clamp_side(None) == msf.CONTEXT_DEFAULT
    assert msf.clamp_side("x") == msf.CONTEXT_DEFAULT


def _named(ident, name, content="hi", **extra):
    return _hit(ident, content=content, author={"id": "1", "username": name.lower(),
                                                "global_name": name}, **extra)


def test_format_context_orders_marks_and_offers_paging():
    anchor = _named(20, "Alice", "the trip link")
    # Discord returns each list newest first; the anchor or a duplicate is dropped.
    older = [_named(19, "Bo", "where to?"), _named(18, "Cy", "trip time"), _named(20, "Alice")]
    newer = [_named(22, "Cy", "thanks", referenced_message=_named(20, "Alice")),
             _named(21, "Bo", "nice"), _named(21, "Bo", "nice")]
    text = msf.format_context(anchor, older, newer, GUILD_ID, GENERAL_ID, "general", 2, 2)
    lines = text.splitlines()
    assert lines[0].startswith("Messages in #general around the one asked for, oldest first")
    assert [line.split(",")[0] for line in lines[1:3]] == ["- Cy", "- Bo"]
    assert lines[3].startswith("► Alice, <t:") and lines[3].endswith(": the trip link")
    assert lines[4] == f"  {_link(20)}"
    assert lines[5].startswith("- Bo,") and lines[6].startswith("- Cy (replying to Alice),")
    assert lines[7] == (f"Earlier messages: call read_messages_around with message={_link(18)}, "
                        f"before={msf.CONTEXT_MAX_SIDE}, after=0.")
    assert lines[8] == (f"Later messages: call read_messages_around with message={_link(22)}, "
                        f"before=0, after={msf.CONTEXT_MAX_SIDE}.")


def test_format_context_at_the_ends_of_a_channel():
    anchor = _named(20, "Alice")
    text = msf.format_context(anchor, [_named(19, "Bo")], [], GUILD_ID, GENERAL_ID, "general",
                              5, 5)
    assert "Nothing earlier: this is the start of the channel." in text
    assert "Nothing later: these are the channel's newest messages." in text
    # A side not asked for is offered, never claimed empty.
    text = msf.format_context(anchor, [], [], GUILD_ID, GENERAL_ID, "general", 0, 0)
    assert f"message={_link(20)}, before={msf.CONTEXT_MAX_SIDE}, after=0" in text
    assert f"message={_link(20)}, before=0, after={msf.CONTEXT_MAX_SIDE}" in text


def test_search_results_point_at_reading_around():
    assert "read_messages_around" in msf.format_results(_body(_hit(1)), GUILD_ID, {}, "x")


def _perms(read=True, history=True):
    return SimpleNamespace(read_messages=read, read_message_history=history)


def _not_found():
    return discord.NotFound(MagicMock(status=404, reason="Not Found"), "Unknown Channel")


def _context_env(anchor=None, older=(), newer=(), nsfw_target=False, nsfw_here=False,
                 perms=None, get_error=None):
    context, _ = _env(nsfw=nsfw_here)
    guild = context["channel"].guild
    memes = next(c for c in guild.channels if c.id == MEMES_ID)
    memes.is_nsfw = lambda: nsfw_target
    for place in list(guild.channels) + list(guild.threads):
        place.guild = guild
        place.permissions_for = lambda member, p=perms or _perms(): p
    guild.me = SimpleNamespace(id=1)
    guild.fetch_channel = AsyncMock(side_effect=_not_found())
    calls = []

    async def logs_from(channel_id, limit, before=None, after=None, around=None):
        calls.append((channel_id, limit, before, after))
        return list(older if before else newer)[:limit]

    http = SimpleNamespace(
        get_message=AsyncMock(side_effect=get_error,
                              return_value=anchor or _named(20, "Alice", "the trip link")),
        logs_from=logs_from)
    context["discord_client"] = SimpleNamespace(http=http)
    return context, http, calls


async def _read(context, **arguments):
    from agents.tool_context import ToolContext
    from classes.message_search_tools import read_messages_around
    tool_context = ToolContext(context=context, tool_name="read_messages_around",
                               tool_call_id="t2", tool_arguments="{}")
    return await read_messages_around.on_invoke_tool(tool_context, json.dumps(arguments))


@pytest.mark.asyncio
async def test_reads_around_a_linked_message():
    context, http, calls = _context_env(older=[_named(19, "Bo")], newer=[_named(21, "Cy")])
    text = await _read(context, message=_link(20, MEMES_ID), before=3, after=2)
    http.get_message.assert_awaited_once_with(MEMES_ID, 20)
    assert sorted(calls, key=str) == sorted([(MEMES_ID, 3, 20, None), (MEMES_ID, 2, None, 20)],
                                            key=str)
    assert text.startswith("Messages in #memes around")
    assert "- Bo," in text and "► Alice," in text and "- Cy," in text


@pytest.mark.asyncio
async def test_a_bare_id_is_the_current_channel_and_zero_skips_a_side():
    context, http, calls = _context_env()
    await _read(context, message="123456789012345678", before=0, after=4)
    http.get_message.assert_awaited_once_with(GENERAL_ID, 123456789012345678)
    assert calls == [(GENERAL_ID, 4, None, 123456789012345678)]


@pytest.mark.asyncio
async def test_another_servers_link_is_refused():
    context, http, _ = _context_env()
    text = await _read(context, message=_link(20, MEMES_ID, guild_id=999999999999999999))
    assert "different server" in text
    text = await _read(context, message="https://discord.com/channels/@me/5/6")
    assert "different server" in text
    http.get_message.assert_not_called()


@pytest.mark.asyncio
async def test_an_uncached_channel_is_fetched_but_only_from_this_server():
    context, http, _ = _context_env()
    guild = context["channel"].guild
    archived = SimpleNamespace(id=777777777777777777, name="old-thread", guild=guild,
                               is_nsfw=lambda: False, permissions_for=lambda m: _perms())
    guild.fetch_channel = AsyncMock(return_value=archived)
    assert (await _read(context, message=_link(20, archived.id))).startswith(
        "Messages in #old-thread")
    # Another server's channel: Guild.fetch_channel raises InvalidData.
    guild.fetch_channel = AsyncMock(side_effect=discord.InvalidData("other guild"))
    assert "can't find that channel" in (await _read(context,
                                                     message=_link(20, 888888888888888888)))
    # Deleted, or not visible to the bot.
    guild.fetch_channel = AsyncMock(side_effect=_not_found())
    assert "can't find that channel" in (await _read(context,
                                                     message=_link(20, 888888888888888888)))
    assert http.get_message.await_count == 1


@pytest.mark.asyncio
async def test_permissions_and_nsfw():
    context, http, _ = _context_env(perms=_perms(history=False))
    assert "permission" in (await _read(context, message=_link(20, MEMES_ID)))
    context, http, _ = _context_env(nsfw_target=True)
    assert "age-restricted" in (await _read(context, message=_link(20, MEMES_ID)))
    http.get_message.assert_not_called()
    context, http, _ = _context_env(nsfw_target=True, nsfw_here=True)
    assert (await _read(context, message=_link(20, MEMES_ID))).startswith("Messages in #memes")


@pytest.mark.asyncio
async def test_bad_input_and_discord_errors():
    context, http, _ = _context_env()
    assert "isn't a message link" in (await _read(context, message="the trip one"))
    response = MagicMock(status=404, reason="Not Found")
    context, _, _ = _context_env(get_error=discord.NotFound(response, "Unknown Message"))
    assert "doesn't exist any more" in (await _read(context, message=_link(20)))
    response = MagicMock(status=403, reason="Forbidden")
    context, _, _ = _context_env(get_error=discord.Forbidden(response, "Missing Access"))
    assert "permission" in (await _read(context, message=_link(20)))
    response = MagicMock(status=500, reason="Server Error")
    context, _, _ = _context_env(get_error=discord.HTTPException(response, "boom"))
    assert "try again later" in (await _read(context, message=_link(20)))


@pytest.mark.asyncio
async def test_reads_are_not_capped_and_need_a_server():
    context, http, _ = _context_env()
    for _ in range(10):
        assert (await _read(context, message=_link(20))).startswith("Messages in #general")
    assert http.get_message.await_count == 10
    assert "only works in a server" in (await _read({"channel": SimpleNamespace()},
                                                   message=_link(20)))


# --- saving messages to a file (brain calls) -------------------------------

TRACKER_EMBED = {"author": {"name": "Team Ana"}, "title": "Shared Chest",
                 "description": "Team Ana withdrew:\n- 20 x Iron bar (4K)",
                 "fields": [{"name": "Player", "value": "Bo"},
                            {"name": "Net value", "value": "-4K gp"}],
                 "footer": {"text": "Powered by Tracker"}}


def test_params_for_saving_page_newest_first():
    params = msf.search_params("chest", None, None, 25, offset=50, newest_first=True)
    assert params == [("limit", "25"), ("include_nsfw", "false"), ("content", "chest"),
                      ("sort_by", "timestamp"), ("sort_order", "desc"), ("offset", "50")]


def test_export_record_keeps_everything():
    message = _hit(7, content="", embeds=[TRACKER_EMBED, {"type": "image"}],
                   attachments=[{"filename": "a.png"}])
    message["author"]["bot"] = True
    record = msf.export_record(message, GUILD_ID, {str(GENERAL_ID): "general"})
    assert record == {
        "id": "7", "link": f"https://discord.com/channels/{GUILD_ID}/{GENERAL_ID}/7",
        "channel": "general", "author": "Alice", "author_id": str(ALICE_ID), "bot": True,
        "timestamp": "2026-03-01T12:00:00.000000+00:00", "content": "",
        "embeds": [{"author": "Team Ana", "title": "Shared Chest",
                    "description": "Team Ana withdrew:\n- 20 x Iron bar (4K)",
                    "fields": [{"name": "Player", "value": "Bo"},
                               {"name": "Net value", "value": "-4K gp"}],
                    "footer": "Powered by Tracker", "url": ""}],
        "attachments": ["a.png"]}


def test_export_filename_and_clamp():
    assert msf.export_filename("Shared Chest!", "this server", "") == "discord-shared-chest.jsonl"
    assert msf.export_filename("", "#trip-plans", "Bo") == "discord-bo-trip-plans.jsonl"
    assert msf.export_filename("", "this server", "🙂") == "discord-messages.jsonl"
    assert msf.clamp_export(0) == 1 and msf.clamp_export(10**6) == msf.EXPORT_MAX
    assert msf.clamp_export("lots") == msf.EXPORT_DEFAULT


def test_export_summary():
    text = msf.export_summary("in/x.jsonl", 3, 3, {"id": "1", "content": "y" * 2000})
    assert text.startswith("Saved 3 messages (all found) to in/x.jsonl, newest first")
    assert "The newest record: {" in text and text.endswith("…")
    text = msf.export_summary("in/x.jsonl", 1, 40, {"id": "1"}, "Stopped early: why.")
    assert "Saved 1 message (the newest 1 of 40 found)" in text and "Stopped early: why." in text
    assert msf.export_summary("", 0, 0, None) == "No messages matched, so no file was written."


class _FakeSession:
    def __init__(self, cwd="/work"):
        self.cwd, self.files, self.commands = cwd, {}, []

    async def exec(self, *argv, shell=False):
        self.commands.append(argv)
        out = self.cwd if argv[0] == "pwd" else ""
        return SimpleNamespace(ok=lambda: True, stdout=out.encode())

    async def write(self, path, data):
        self.files[str(path)] = data.read().decode("utf-8")


def _pages(total, page_size=25, start_id=10_000):
    """Search pages for `total` messages, newest first, answered by offset."""
    ids = list(range(start_id + total, start_id, -1))

    async def request(route, params):
        params = dict(params)
        offset, limit = int(params.get("offset", 0)), int(params["limit"])
        chunk = ids[offset:offset + limit]
        return {"total_results": total, "messages": [[_hit(i, content="", embeds=[TRACKER_EMBED])]
                                                     for i in chunk]}
    return request


async def _save(context, **arguments):
    from agents.tool_context import ToolContext
    from classes.message_search_tools import save_messages_to_file
    tool_context = ToolContext(context=context, tool_name="save_messages_to_file",
                               tool_call_id="t3", tool_arguments="{}")
    return await save_messages_to_file.on_invoke_tool(tool_context, json.dumps(arguments))


@pytest.mark.asyncio
async def test_saves_every_page_to_a_file():
    context, http = _env()
    http.request = AsyncMock(side_effect=_pages(60))
    session = _FakeSession()
    context.update(session=session, out_dir="/work/out")
    text = await _save(context, query="shared chest", channel="general")
    assert http.request.await_count == 3
    offsets = [dict(c.kwargs["params"]).get("offset", "0") for c in http.request.await_args_list]
    assert offsets == ["0", "25", "50"]
    assert all(dict(c.kwargs["params"])["channel_id"] == str(GENERAL_ID)
               for c in http.request.await_args_list)
    lines = session.files["/work/in/discord-shared-chest-general.jsonl"].splitlines()
    assert len(lines) == 60 and json.loads(lines[0])["id"] == "10060"
    assert json.loads(lines[0])["embeds"][0]["fields"][1] == {"name": "Net value",
                                                              "value": "-4K gp"}
    assert ("mkdir", "-p", "/work/in") in session.commands
    assert text.startswith("Saved 60 messages (all found) to in/discord-shared-chest-general.jsonl")


@pytest.mark.asyncio
async def test_saving_stops_at_max_and_skips_repeats():
    context, http = _env()
    pages = _pages(100)

    async def shifting(route, params):
        # A new message arrived between pages: the second page repeats one.
        body = await pages(route, params)
        if dict(params).get("offset") == "25":
            body["messages"].insert(0, [_hit(10_076, content="")])
        return body
    http.request = AsyncMock(side_effect=shifting)
    session = _FakeSession()
    context.update(session=session)  # no out_dir: found from pwd
    text = await _save(context, query="chest", max_messages=40)
    lines = session.files["/work/in/discord-chest.jsonl"].splitlines()
    ids = [json.loads(line)["id"] for line in lines]
    assert len(ids) == 40 and len(set(ids)) == 40
    assert "the newest 40 of 100 found" in text


@pytest.mark.asyncio
async def test_saving_only_in_the_sandbox_and_failures():
    context, http = _env()
    assert "needs the workspace" in (await _save(context, query="x"))
    http.request.assert_not_called()
    context["session"] = _FakeSession()
    http.request = AsyncMock(return_value=_body())
    assert (await _save(context, query="x")) == "No messages matched, so no file was written."
    assert context["session"].files == {}
    response = MagicMock(status=500, reason="Server Error")
    calls = {"n": 0}
    pages = _pages(60)

    async def fails_later(route, params):
        calls["n"] += 1
        if calls["n"] == 2:
            raise discord.HTTPException(response, "boom")
        return await pages(route, params)
    http.request = AsyncMock(side_effect=fails_later)
    text = await _save(context, query="x")
    assert "Saved 25 messages (the newest 25 of 60 found)" in text and "Stopped early" in text


@pytest.mark.asyncio
async def test_a_big_search_in_the_sandbox_points_at_saving():
    context, http = _env(body=_body(_hit(1), total_results=500))
    assert "save_messages_to_file" not in (await _search(context, query="x"))
    context["session"] = _FakeSession()
    assert "To work with all 500 of them" in (await _search(context, query="x"))


def test_the_brain_gets_the_save_tool_and_the_persona_does_not():
    from classes.brain_tools import brain_tools
    from classes.text_llm_handler import persona_tools
    assert "save_messages_to_file" not in {t.name for t in persona_tools()}
    names = {t.name for t in brain_tools()}
    assert {"save_messages_to_file", "read_messages_around", "search_messages"} <= names

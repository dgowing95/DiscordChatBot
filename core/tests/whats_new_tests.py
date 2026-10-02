"""
Tests for the What's New announcement:
  - core/classes/whats_new.py — pure: notes parsing, env gating, embed data,
    and the SHIPPED core/whats_new.md staying inside Discord's embed limits
    (a too-long section would make the announcement fail to send).
  - main.maybe_announce_whats_new — once per guild per version, atomic across
    workers, skipped versions never shown, fail-soft.
  - MessageHandler._format_group — the announcement stays out of prompts.

Run from the repo root:
    pytest core/tests/whats_new_tests.py
"""
import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock

import discord
import pytest

from classes import message_queue as mq
from classes import whats_new
from classes.common import embed_from_data
from classes.message_handler import MessageHandler


# ---------------------------------------------------------------------------
# pure module
# ---------------------------------------------------------------------------

def test_parse_notes_sections_in_order():
    text = (
        "Intro text before any heading is ignored.\n"
        "## First\nDoes a thing.\nHow to use: ask.\n\n"
        "## Second\nAnother.\n"
    )
    assert whats_new.parse_notes(text) == [
        ("First", "Does a thing.\nHow to use: ask."),
        ("Second", "Another."),
    ]


def test_parse_notes_ignores_comments_and_empty_sections():
    text = "<!--\n## Not a feature\nexample\n-->\n## Empty\n\n## Real\nYes.\n"
    assert whats_new.parse_notes(text) == [("Real", "Yes.")]


@pytest.mark.parametrize("text", ["", "   \n", "<!-- template only -->\n"])
def test_parse_notes_no_features(text):
    assert whats_new.parse_notes(text) == []


def test_load_notes_missing_file(tmp_path):
    assert whats_new.load_notes(tmp_path / "nope.md") == []


def test_load_notes_unreadable_file(tmp_path, monkeypatch):
    path = tmp_path / "notes.md"
    path.write_text("## A\nb\n", encoding="utf-8")

    def unreadable(*args, **kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(type(path), "read_text", unreadable)
    whats_new._cache.clear()
    assert whats_new.load_notes(path) == []


def test_load_notes_reads_file(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("## A\nb\n", encoding="utf-8")
    assert whats_new.load_notes(path) == [("A", "b")]


def test_shipped_notes_fit_discord_limits():
    # The release blanks this file when unchanged, so whatever is committed
    # is what the next release announces. Keep it sendable.
    notes = whats_new.load_notes()
    assert whats_new.limit_problems(notes) == []


def test_limit_problems_catches_oversize():
    notes = [("x" * 300, "y" * 1100)] + [("n", "v")] * 25
    problems = whats_new.limit_problems(notes)
    assert any("max 25" in p for p in problems)
    assert any("256" in p for p in problems)
    assert any("1024" in p for p in problems)


def test_app_version_and_enabled(monkeypatch):
    monkeypatch.delenv("APP_VERSION", raising=False)
    assert whats_new.app_version() == ""
    monkeypatch.setenv("APP_VERSION", " v2.53 ")
    assert whats_new.app_version() == "v2.53"
    monkeypatch.delenv("WHATS_NEW_ENABLED", raising=False)
    assert whats_new.whats_new_enabled()
    for off in ("0", "false", "no", "off", ""):
        monkeypatch.setenv("WHATS_NEW_ENABLED", off)
        assert not whats_new.whats_new_enabled()


def test_embed_data_round_trips_to_discord_embed():
    data = whats_new.notes_embed_data("v2.53", [("A", "b")])
    embed = embed_from_data(data)
    assert embed.title == "✨ What's new in v2.53"
    assert [(f.name, f.value) for f in embed.fields] == [("A", "b")]
    assert embed.footer.text == whats_new.ANNOUNCEMENT_FOOTER


# ---------------------------------------------------------------------------
# main.maybe_announce_whats_new
# ---------------------------------------------------------------------------

def _import_main():
    if "main" in sys.modules:
        return sys.modules["main"]
    import main as m
    return m


class FakeRedis:
    """SET ... GET semantics, atomic like the real thing (no await inside)."""

    def __init__(self):
        self.data = {}
        self.fail = False

    async def set(self, key, value, get=False):
        if self.fail:
            raise ConnectionError("redis down")
        previous = self.data.get(key)
        self.data[key] = value
        return previous if get else True

    async def delete(self, key):
        self.data.pop(key, None)


def _message(guild_id=42):
    msg = MagicMock()
    msg.guild = MagicMock()
    msg.guild.id = guild_id
    msg.channel = MagicMock()
    msg.channel.id = 77
    msg.channel.send = AsyncMock()
    return msg


@pytest.fixture
def announce(monkeypatch, tmp_path):
    m = _import_main()
    redis = FakeRedis()
    monkeypatch.setattr(m, "text_client", lambda: redis)
    # Fresh locks, as in reply_policy_tests: _channel_locks is module-global,
    # and an asyncio.Lock binds to the first event loop that contends on it.
    monkeypatch.setattr(mq, "_channel_locks", {})
    notes = tmp_path / "whats_new.md"
    notes.write_text("## Feature\nDoes things.\n", encoding="utf-8")
    monkeypatch.setattr(whats_new, "DEFAULT_NOTES_PATH", notes)
    monkeypatch.setenv("APP_VERSION", "v2.53")
    monkeypatch.delenv("WHATS_NEW_ENABLED", raising=False)
    return m, redis, notes


def test_announces_once_per_guild_per_version(announce):
    m, redis, _ = announce
    first, second, other_guild = _message(), _message(), _message(guild_id=7)
    asyncio.run(m.maybe_announce_whats_new(first))
    asyncio.run(m.maybe_announce_whats_new(second))
    asyncio.run(m.maybe_announce_whats_new(other_guild))
    assert first.channel.send.await_count == 1
    assert second.channel.send.await_count == 0
    assert other_guild.channel.send.await_count == 1
    embed = first.channel.send.await_args.kwargs["embed"]
    assert embed.title == "✨ What's new in v2.53"
    assert redis.data[whats_new.seen_key(42)] == "v2.53"


def test_concurrent_workers_announce_once(announce):
    m, _, _ = announce
    a, b = _message(), _message()

    async def both():
        await asyncio.gather(m.maybe_announce_whats_new(a), m.maybe_announce_whats_new(b))

    asyncio.run(both())
    assert a.channel.send.await_count + b.channel.send.await_count == 1


def test_new_version_announces_again(announce, monkeypatch):
    m, redis, _ = announce
    redis.data[whats_new.seen_key(42)] = "v2.52"
    msg = _message()
    asyncio.run(m.maybe_announce_whats_new(msg))
    assert msg.channel.send.await_count == 1


def test_skipped_version_is_never_shown(announce, monkeypatch):
    # The guild last saw v2.51 and never triggered the bot on v2.52 (which
    # had notes). v2.53 shipped no features, so its notes file is blank:
    # nothing is shown, v2.52's notes included.
    m, redis, notes = announce
    redis.data[whats_new.seen_key(42)] = "v2.51"
    notes.write_text("", encoding="utf-8")
    whats_new._cache.clear()
    msg = _message()
    asyncio.run(m.maybe_announce_whats_new(msg))
    msg.channel.send.assert_not_awaited()


@pytest.mark.parametrize("setup", ["no_version", "disabled", "dm"])
def test_gated_off(announce, monkeypatch, setup):
    m, redis, _ = announce
    msg = _message()
    if setup == "no_version":
        monkeypatch.delenv("APP_VERSION")
    elif setup == "disabled":
        monkeypatch.setenv("WHATS_NEW_ENABLED", "0")
    else:
        msg.guild = None
    asyncio.run(m.maybe_announce_whats_new(msg))
    msg.channel.send.assert_not_awaited()
    assert redis.data == {}


def test_failures_are_swallowed(announce):
    m, redis, _ = announce
    redis.fail = True
    msg = _message()
    asyncio.run(m.maybe_announce_whats_new(msg))  # must not raise
    redis.fail = False
    msg.channel.send.side_effect = discord.HTTPException(MagicMock(status=500), "boom")
    asyncio.run(m.maybe_announce_whats_new(msg))  # must not raise


@pytest.mark.parametrize("previous", [None, "v2.52"])
def test_failed_send_gives_the_claim_back(announce, previous):
    m, redis, _ = announce
    key = whats_new.seen_key(42)
    if previous:
        redis.data[key] = previous
    msg = _message()
    msg.channel.send.side_effect = discord.HTTPException(MagicMock(status=500), "boom")
    asyncio.run(m.maybe_announce_whats_new(msg))
    assert redis.data.get(key) == previous
    # ...so the next triggered message tries again.
    retry = _message()
    asyncio.run(m.maybe_announce_whats_new(retry))
    assert retry.channel.send.await_count == 1


def test_send_waits_for_the_channel_lock(announce):
    m, _, _ = announce
    msg = _message()

    async def run():
        lock = m.get_channel_lock(msg.channel.id)
        async with lock:
            task = asyncio.create_task(m.maybe_announce_whats_new(msg))
            await asyncio.sleep(0.01)
            assert msg.channel.send.await_count == 0
        await task

    asyncio.run(run())
    assert msg.channel.send.await_count == 1


# ---------------------------------------------------------------------------
# kept out of prompt history
# ---------------------------------------------------------------------------

def _history_handler(bot_id=1):
    handler = MessageHandler.__new__(MessageHandler)
    handler.client = MagicMock()
    handler.client.user.id = bot_id
    handler.clean_message_content = lambda m: m.content
    return handler


def _bot_message(embeds, author_id=1):
    msg = MagicMock()
    msg.content = ""
    msg.author.id = author_id
    msg.author.name = "bot"
    msg.embeds = embeds
    return msg


def test_announcement_embed_left_out_of_prompt():
    handler = _history_handler()
    announcement = embed_from_data(whats_new.notes_embed_data("v2.53", [("A", "b")]))
    tool = discord.Embed(title="Tool Usage", description="searching")
    entries = handler._format_group(_bot_message([announcement, tool]), [])
    assert len(entries) == 1
    assert "Tool Usage" in entries[0]["content"]


def test_same_footer_from_someone_else_is_kept():
    handler = _history_handler()
    embed = embed_from_data(whats_new.notes_embed_data("v2.53", [("A", "b")]))
    entries = handler._format_group(_bot_message([embed], author_id=99), [])
    assert len(entries) == 1

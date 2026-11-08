"""
Tests for /help (core/classes/help_catalog.py).

The sync tests are the point: the catalog is hand-written, so they fail when
a brain tool or a slash command is added without a /help entry (or one is
removed and its entry left behind).

Run from the repo root:
    pytest core/tests/help_catalog_tests.py
"""
import asyncio
import sys
from unittest.mock import MagicMock

import pytest

from classes import card_kit as K, help_catalog
from classes.card import views


def _text(spec) -> str:
    return "\n".join(t for c in K.split(spec) for b in c
                     for t in ([b.content] if isinstance(b, K.Text) else
                               list(b.texts) if isinstance(b, K.Section) else []))


def _import_main():
    if "main" in sys.modules:
        return sys.modules["main"]
    import main as m
    return m


def _all_on(monkeypatch):
    monkeypatch.setenv("IMAGE_GEN_ENABLED", "1")
    monkeypatch.setenv("AUTOMATIONS_ENABLED", "1")
    monkeypatch.setenv("VOICE_ENABLED", "1")
    monkeypatch.setenv("EVENTS_ENABLED", "1")


def test_every_brain_tool_has_help(monkeypatch):
    from classes.brain_tools import brain_tools
    _all_on(monkeypatch)
    # A voice turn swaps join_voice_channel for leave_voice_channel.
    names = ({tool.name for tool in brain_tools(workflows=True)}
             | {tool.name for tool in brain_tools(voice=True)})
    assert names - help_catalog.INTERNAL_TOOLS == help_catalog.documented_tools()


class FakeCommandTree:
    """Records the names register_commands() gives @command_tree.command."""

    def __init__(self, *args, **kwargs):
        self.names = []

    def command(self, name, description=""):
        self.names.append(name)
        return lambda fn: fn

    def add_command(self, command):
        self.names.append(command.name)

    async def sync(self):
        return []


def test_every_slash_command_has_help(monkeypatch):
    m = _import_main()
    _all_on(monkeypatch)
    trees = []

    def make_tree(*args, **kwargs):
        trees.append(FakeCommandTree())
        return trees[-1]

    monkeypatch.setattr(m.discord.app_commands, "CommandTree", make_tree)
    asyncio.run(m.register_commands())
    assert len(trees) == 1
    assert set(trees[0].names) == help_catalog.documented_commands()


@pytest.mark.parametrize("image", [True, False])
def test_help_shows_every_entry_and_fits_discord(image, monkeypatch):
    # Every optional entry on, so the longest listing is the one measured.
    monkeypatch.setenv("AUTOMATIONS_ENABLED", "1")
    monkeypatch.setenv("VOICE_ENABLED", "1")
    monkeypatch.setenv("EVENTS_ENABLED", "1")
    spec = help_catalog.help_card(image, "https://cdn.example/avatar.png")
    built = views(spec)  # builds without error, as many messages as it needs
    assert all(v.content_length() <= K.MAX_TEXT for v in built)
    text = _text(spec)
    for entry in help_catalog.ASK_ENTRIES + help_catalog.COMMAND_ENTRIES:
        if help_catalog._shown(entry, image):
            assert entry.description in text, entry.name  # nothing cut
    assert isinstance(spec.body[0], K.Section) and spec.body[0].accessory == K.Thumb(
        "https://cdn.example/avatar.png")


def test_help_without_an_avatar_has_a_plain_header():
    spec = help_catalog.help_card(True, None)
    assert isinstance(spec.body[0], K.Text) and spec.body[0].content.startswith("## 🤖 What I can do")


def test_disabled_features_are_hidden(monkeypatch):
    monkeypatch.setenv("EVENTS_ENABLED", "0")
    text = _text(help_catalog.help_card(False))
    assert "Events" not in text
    assert "/generate_image" not in text and "Make images" not in text
    # The workspace is always there: Docker is required.
    assert "/workspace" in text and "Code, files and charts" in text
    monkeypatch.setenv("EVENTS_ENABLED", "1")
    text = _text(help_catalog.help_card(True))
    assert "/generate_image" in text and "Code, files and charts" in text and "Events" in text


def test_help_and_whats_new_replies_are_ephemeral(monkeypatch):
    # Ephemeral replies never appear in channel.history, so they never reach
    # a prompt; a public one would need the announcement filter too.
    m = _import_main()
    _all_on(monkeypatch)
    callbacks = {}

    class Tree(FakeCommandTree):
        def command(self, name, description=""):
            def register(fn):
                callbacks[name] = fn
                return fn
            return register

    monkeypatch.setattr(m.discord.app_commands, "CommandTree", Tree)
    monkeypatch.setenv("APP_VERSION", "v2.53")
    asyncio.run(m.register_commands())
    for name in ("help", "whats_new"):
        from card_checks import interaction
        ctx = interaction()
        asyncio.run(callbacks[name](ctx))
        assert ctx.response.send_message.call_args.kwargs["ephemeral"] is True

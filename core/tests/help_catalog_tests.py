"""
Tests for /help (core/classes/help_catalog.py).

The sync tests are the point: the catalog is hand-written, so they fail when
an agent tool or a slash command is added without a /help entry (or one is
removed and its entry left behind).

Run from the repo root:
    pytest core/tests/help_catalog_tests.py
"""
import asyncio
import sys
from unittest.mock import MagicMock

import pytest

from classes import help_catalog, whats_new
from classes.common import embed_from_data


def _import_main():
    if "main" in sys.modules:
        return sys.modules["main"]
    import main as m
    return m


def _all_on(monkeypatch):
    monkeypatch.setenv("IMAGE_GEN_ENABLED", "1")
    monkeypatch.setenv("SANDBOX_ENABLED", "1")


def test_every_agent_tool_has_help(monkeypatch):
    from classes.text_llm_handler import agent_tools
    _all_on(monkeypatch)
    names = {tool.name for tool in agent_tools()}
    assert names == help_catalog.documented_tools()


class FakeCommandTree:
    """Records the names register_commands() gives @command_tree.command."""
    names: list = []

    def __init__(self, *args, **kwargs):
        FakeCommandTree.names = []

    def command(self, name, description=""):
        FakeCommandTree.names.append(name)
        return lambda fn: fn

    async def sync(self):
        return []


def test_every_slash_command_has_help(monkeypatch):
    m = _import_main()
    _all_on(monkeypatch)
    monkeypatch.setattr(m.discord.app_commands, "CommandTree", FakeCommandTree)
    asyncio.run(m.register_commands())
    assert set(FakeCommandTree.names) == help_catalog.documented_commands()


@pytest.mark.parametrize("image, sandbox", [(True, True), (False, False), (True, False)])
def test_help_fits_discord_limits(image, sandbox):
    data = help_catalog.help_embed_data(image, sandbox)
    for field in data["fields"]:
        assert 0 < len(field["value"]) <= whats_new.MAX_FIELD_VALUE, field["name"]
    total = len(data["title"]) + sum(len(f["name"]) + len(f["value"]) for f in data["fields"])
    assert total <= whats_new.MAX_EMBED_TOTAL
    embed_from_data(data)  # builds without error


def test_disabled_features_are_hidden():
    text = str(help_catalog.help_embed_data(False, False))
    assert "/generate_image" not in text and "Make images" not in text
    assert "/sandbox_progress_updates" not in text and "Run code" not in text
    text = str(help_catalog.help_embed_data(True, True))
    assert "/generate_image" in text and "Run code" in text


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
        ctx = MagicMock()
        ctx.response.send_message = MagicMock(side_effect=lambda **kw: asyncio.sleep(0))
        asyncio.run(callbacks[name](ctx))
        assert ctx.response.send_message.call_args.kwargs["ephemeral"] is True

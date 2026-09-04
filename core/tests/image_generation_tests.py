import os
import sys

# Imported as `classes.X`, the same name the app uses (it runs with cwd=core/,
# and pyproject.toml puts core/ on the test path).

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from classes import image_generation

# To run this pytest file from the command line, use:
# PYTHONPATH=$(pwd) pytest core/tests/image_generation_tests.py


def _mock_session(status=200, payload=b"PNGDATA", text=""):
    response = MagicMock()
    response.status = status
    response.read = AsyncMock(return_value=payload)
    response.text = AsyncMock(return_value=text)
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=False)

    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    session.post = MagicMock(return_value=response)
    return session, response


# ---------------------- image_generation_enabled ----------------------

def test_enabled_env_values(monkeypatch):
    cases = [
        ("1", True), ("true", True), ("True", True), ("on", True),
        ("0", False), ("false", False), ("no", False), ("off", False),
        ("", False),
    ]
    for value, expected in cases:
        monkeypatch.setenv("IMAGE_GEN_ENABLED", value)
        assert image_generation.image_generation_enabled() is expected, value


def test_enabled_defaults_on(monkeypatch):
    monkeypatch.delenv("IMAGE_GEN_ENABLED", raising=False)
    assert image_generation.image_generation_enabled() is True


def test_enabled_trims_whitespace(monkeypatch):
    monkeypatch.setenv("IMAGE_GEN_ENABLED", "  0  ")
    assert image_generation.image_generation_enabled() is False


# ---------------------- diffusion_base_url ----------------------

def test_base_url_strips_trailing_slash(monkeypatch):
    monkeypatch.setenv("DIFFUSION_URL", "http://diffusion:8000/")
    assert image_generation.diffusion_base_url() == "http://diffusion:8000"


def test_base_url_defaults(monkeypatch):
    monkeypatch.delenv("DIFFUSION_URL", raising=False)
    assert image_generation.diffusion_base_url() == "http://diffusion:8000"


# ---------------------- generate_image_from_api ----------------------

@pytest.mark.asyncio
async def test_generate_returns_image_bytes(monkeypatch):
    monkeypatch.setenv("DIFFUSION_URL", "http://diffusion:8000")
    session, response = _mock_session(status=200, payload=b"PNGDATA")
    with patch("aiohttp.ClientSession", return_value=session):
        data = await image_generation.generate_image_from_api("a red fox")

    assert data == b"PNGDATA"
    args, kwargs = session.post.call_args
    assert args[0] == "http://diffusion:8000/generate"
    assert kwargs["json"] == {"prompt": "a red fox"}


@pytest.mark.asyncio
async def test_generate_non_200_raises(monkeypatch):
    monkeypatch.setenv("DIFFUSION_URL", "http://diffusion:8000")
    session, _ = _mock_session(status=503, text="image queue is full, try again later")
    with patch("aiohttp.ClientSession", return_value=session):
        with pytest.raises(Exception, match="503"):
            await image_generation.generate_image_from_api("a red fox")


@pytest.mark.asyncio
async def test_generate_connection_error_raises(monkeypatch):
    monkeypatch.setenv("DIFFUSION_URL", "http://diffusion:8000")
    session, response = _mock_session(status=200)
    response.__aenter__.side_effect = ConnectionError("no such host")
    with patch("aiohttp.ClientSession", return_value=session):
        with pytest.raises(ConnectionError):
            await image_generation.generate_image_from_api("a red fox")


# ---------------------- /generate_image slash command (core/main.py) ----------------------

def _import_main():
    """Import main exactly once.

    main.py guards client.run() behind `if __name__ == "__main__"` and builds
    its Redis client lazily, so importing it neither connects to Discord nor
    needs any environment. This used to os.environ.setdefault REDIS_HOST to
    "localhost", which leaked process-wide (unlike monkeypatch.setenv) and
    made every later test that touched configManager pay a real TCP connect
    timeout to a Redis that was not running.
    """
    if "main" in sys.modules:
        return sys.modules["main"]
    import main as m
    return m


class _FakeCommandTree:
    """Records the commands registered via @tree.command(name=...)."""
    instances = []

    def __init__(self, **kwargs):
        self.commands = []
        _FakeCommandTree.instances.append(self)

    def command(self, *args, **kwargs):
        def deco(fn):
            self.commands.append((kwargs.get("name"), fn))
            return fn
        return deco

    async def sync(self):
        return [MagicMock(name=f"synced-{i}") for i in range(len(self.commands))]


async def _register_and_collect(monkeypatch, enabled_value):
    main_mod = _import_main()
    monkeypatch.setenv("IMAGE_GEN_ENABLED", enabled_value)
    _FakeCommandTree.instances.clear()
    with patch.object(main_mod.discord.app_commands, "CommandTree", _FakeCommandTree):
        await main_mod.register_commands()
    return main_mod, dict(_FakeCommandTree.instances[-1].commands)


@pytest.mark.asyncio
async def test_generate_image_command_registered_when_enabled(monkeypatch):
    _, commands = await _register_and_collect(monkeypatch, "1")
    assert "generate_image" in commands
    # the other commands are unaffected
    assert "system" in commands and "chance" in commands


@pytest.mark.asyncio
async def test_generate_image_command_absent_when_disabled(monkeypatch):
    _, commands = await _register_and_collect(monkeypatch, "0")
    assert "generate_image" not in commands
    assert "system" in commands


@pytest.mark.asyncio
async def test_generate_image_command_defers_and_sends_image(monkeypatch):
    main_mod, commands = await _register_and_collect(monkeypatch, "1")
    fn = commands["generate_image"]

    ctx = MagicMock()
    ctx.response.defer = AsyncMock()
    ctx.edit_original_response = AsyncMock()
    with patch.object(main_mod, "generate_image_from_api",
                      new=AsyncMock(return_value=b"PNGDATA")):
        await fn(ctx, "a red fox")

    # deferred first (generation is slow), then the image is uploaded
    ctx.response.defer.assert_awaited_once()
    ctx.edit_original_response.assert_awaited_once()
    kwargs = ctx.edit_original_response.await_args.kwargs
    assert kwargs["content"] == "🎨"
    assert kwargs["attachments"][0].filename == "generated-image.png"
    assert kwargs["attachments"][0].fp.read() == b"PNGDATA"


@pytest.mark.asyncio
async def test_generate_image_command_reports_failure(monkeypatch):
    main_mod, commands = await _register_and_collect(monkeypatch, "1")
    fn = commands["generate_image"]

    ctx = MagicMock()
    ctx.response.defer = AsyncMock()
    ctx.edit_original_response = AsyncMock()
    with patch.object(main_mod, "generate_image_from_api",
                      new=AsyncMock(side_effect=Exception("service down"))):
        await fn(ctx, "a red fox")

    ctx.edit_original_response.assert_awaited_once()
    kwargs = ctx.edit_original_response.await_args.kwargs
    assert "❌" in kwargs["content"]
    assert "attachments" not in kwargs


# ---------------------- tool registration in the LLM agent ----------------------

async def _agent_tool_names(monkeypatch, enabled_value):
    # TextLLMHandler.__init__ needs Redis; skip it and set what get_client()
    # reads directly.
    monkeypatch.setenv("IMAGE_GEN_ENABLED", enabled_value)
    from classes.text_llm_handler import TextLLMHandler

    handler = TextLLMHandler.__new__(TextLLMHandler)
    handler.messages = []
    handler.guild_id = 0
    handler.original_message = MagicMock()
    handler.config = MagicMock()
    handler.user_memory = MagicMock()
    handler.system = "test"
    handler.model = "qwen3:4b"
    handler.options = {"temperature": 1.0}
    await handler.get_client()
    return [getattr(tool, "name", str(tool)) for tool in handler.agent.tools]


@pytest.mark.asyncio
async def test_generate_image_tool_registered_when_enabled(monkeypatch):
    names = await _agent_tool_names(monkeypatch, "1")
    assert "generate_image" in names
    # the other tools are unaffected
    assert "web_search" in names


@pytest.mark.asyncio
async def test_generate_image_tool_absent_when_disabled(monkeypatch):
    names = await _agent_tool_names(monkeypatch, "0")
    assert "generate_image" not in names
    assert "web_search" in names

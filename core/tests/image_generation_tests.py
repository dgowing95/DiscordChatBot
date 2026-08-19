import base64
import json
import os
import sys

# image_generation is imported as core.classes.* here and classes.* in
# production (the app runs with cwd=core/); both resolve as namespace packages.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core.classes import image_generation
# The production-style import names: the modules the tools themselves use at
# call time (a different module object than core.classes.* — patch THESE).
import classes.image_generation as prod_image_generation
import classes.tool_functions as prod_tool_functions

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


# ---------------------- generate_image_from_api (img2img) ----------------------

@pytest.mark.asyncio
async def test_generate_with_image_sends_base64_and_strength(monkeypatch):
    monkeypatch.setenv("DIFFUSION_URL", "http://diffusion:8000")
    session, response = _mock_session(status=200, payload=b"EDITED")
    with patch("aiohttp.ClientSession", return_value=session):
        data = await image_generation.generate_image_from_api(
            "make it snowy", image=b"SRCBYTES", strength=0.7
        )

    assert data == b"EDITED"
    assert session.post.call_args.kwargs["json"] == {
        "prompt": "make it snowy",
        "image": base64.b64encode(b"SRCBYTES").decode(),
        "strength": 0.7,
    }


@pytest.mark.asyncio
async def test_generate_with_image_omits_strength_when_unset(monkeypatch):
    monkeypatch.setenv("DIFFUSION_URL", "http://diffusion:8000")
    session, _ = _mock_session(status=200, payload=b"EDITED")
    with patch("aiohttp.ClientSession", return_value=session):
        await image_generation.generate_image_from_api(
            "make it snowy", image=b"SRCBYTES"
        )

    payload = session.post.call_args.kwargs["json"]
    assert payload["image"] == base64.b64encode(b"SRCBYTES").decode()
    assert "strength" not in payload  # service applies its own default


@pytest.mark.asyncio
async def test_generate_with_oversized_image_raises(monkeypatch):
    monkeypatch.setenv("DIFFUSION_URL", "http://diffusion:8000")
    too_big = b"x" * (image_generation.MAX_SOURCE_IMAGE_BYTES + 1)
    with pytest.raises(Exception, match="10MB"):
        await image_generation.generate_image_from_api("p", image=too_big)


# ---------------------- /generate_image slash command (core/main.py) ----------------------

def _import_main():
    """Import core.main exactly once, without starting the bot."""
    if "core.main" in sys.modules:
        return sys.modules["core.main"]
    os.environ.setdefault("DISCORD_TOKEN", "test-token")
    os.environ.setdefault("REDIS_HOST", "localhost")
    import discord
    with patch.object(discord.Client, "run", lambda self, *a, **k: None):
        import core.main as m
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


@pytest.mark.asyncio
async def test_edit_image_command_registered_when_enabled(monkeypatch):
    _, commands = await _register_and_collect(monkeypatch, "1")
    assert "edit_image" in commands


@pytest.mark.asyncio
async def test_edit_image_command_absent_when_disabled(monkeypatch):
    _, commands = await _register_and_collect(monkeypatch, "0")
    assert "edit_image" not in commands


@pytest.mark.asyncio
async def test_edit_image_command_defers_and_sends_edited_image(monkeypatch):
    main_mod, commands = await _register_and_collect(monkeypatch, "1")
    fn = commands["edit_image"]

    ctx = MagicMock()
    ctx.response.defer = AsyncMock()
    ctx.edit_original_response = AsyncMock()
    attachment = MagicMock()
    attachment.read = AsyncMock(return_value=b"SRC")
    api = AsyncMock(return_value=b"EDITED")
    with patch.object(main_mod, "generate_image_from_api", api):
        await fn(ctx, attachment, "make it snowy", 0.8)

    ctx.response.defer.assert_awaited_once()
    api.assert_awaited_once_with("make it snowy", image=b"SRC", strength=0.8)
    kwargs = ctx.edit_original_response.await_args.kwargs
    assert kwargs["content"] == "🎨"
    assert kwargs["attachments"][0].filename == "edited-image.png"
    assert kwargs["attachments"][0].fp.read() == b"EDITED"


@pytest.mark.asyncio
async def test_edit_image_command_reports_failure(monkeypatch):
    main_mod, commands = await _register_and_collect(monkeypatch, "1")
    fn = commands["edit_image"]

    ctx = MagicMock()
    ctx.response.defer = AsyncMock()
    ctx.edit_original_response = AsyncMock()
    attachment = MagicMock()
    attachment.read = AsyncMock(side_effect=Exception("unreadable"))
    with patch.object(main_mod, "generate_image_from_api",
                      new=AsyncMock(return_value=b"EDITED")):
        await fn(ctx, attachment, "make it snowy", None)

    ctx.edit_original_response.assert_awaited_once()
    kwargs = ctx.edit_original_response.await_args.kwargs
    assert "❌" in kwargs["content"]
    assert "attachments" not in kwargs


# ---------------------- tool registration in the LLM agent ----------------------

async def _agent_tool_names(monkeypatch, enabled_value):
    # TextLLMHandler.__init__ needs Redis; skip it and set what get_client()
    # reads directly.
    monkeypatch.setenv("IMAGE_GEN_ENABLED", enabled_value)
    from core.classes.text_llm_handler import TextLLMHandler

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


@pytest.mark.asyncio
async def test_edit_image_tool_registered_when_enabled(monkeypatch):
    names = await _agent_tool_names(monkeypatch, "1")
    assert "edit_image" in names


@pytest.mark.asyncio
async def test_edit_image_tool_absent_when_disabled(monkeypatch):
    names = await _agent_tool_names(monkeypatch, "0")
    assert "edit_image" not in names
    assert "web_search" in names


# ---------------------- edit_image tool behaviour ----------------------

def _tool_context(message, attachment_refs=None):
    from agents.tool_context import ToolContext
    if attachment_refs is None:
        attachment_refs = [{
            "ref": "1",
            "author": "tester",
            "filename": "img.png",
            "url": "https://cdn.example/img.png",
        }]
    return ToolContext(
        context={"original_message": message, "attachment_refs": attachment_refs},
        tool_name="edit_image",
        tool_call_id="t1",
        tool_arguments="{}",
    )


def _mock_get_session(status=200, payload=b"SRCBYTES"):
    response = MagicMock()
    response.status = status
    response.read = AsyncMock(return_value=payload)
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=False)

    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    session.get = MagicMock(return_value=response)
    return session


@pytest.mark.asyncio
async def test_edit_image_tool_fetches_source_and_edits(monkeypatch):
    message = MagicMock()
    message.add_reaction = AsyncMock()
    message.channel.send = AsyncMock()
    get_session = _mock_get_session()
    api = AsyncMock(return_value=b"EDITED")

    with patch("aiohttp.ClientSession", return_value=get_session), \
         patch.object(prod_image_generation, "generate_image_from_api", api), \
         patch.object(prod_tool_functions.Common, "send_tool_discord_embed", AsyncMock()):
        result = await prod_tool_functions.edit_image.on_invoke_tool(
            _tool_context(message),
            json.dumps({
                "prompt": "make it snowy",
                "image_ref": "1",
            }),
        )

    assert "sent to the channel" in result
    get_session.get.assert_called_once()
    assert get_session.get.call_args.args[0] == "https://cdn.example/img.png"
    # the source bytes (not the URL) go to the diffusion service
    api.assert_awaited_once()
    assert api.await_args.kwargs["image"] == b"SRCBYTES"
    assert api.await_args.kwargs["strength"] is None
    # the edited PNG is uploaded to the channel
    sent = message.channel.send.await_args.kwargs
    assert sent["file"].filename == "edited-image.png"
    assert sent["file"].fp.read() == b"EDITED"


@pytest.mark.asyncio
async def test_edit_image_tool_reports_download_failure(monkeypatch):
    message = MagicMock()
    message.add_reaction = AsyncMock()
    message.channel.send = AsyncMock()
    get_session = _mock_get_session(status=404)
    api = AsyncMock(return_value=b"EDITED")

    with patch("aiohttp.ClientSession", return_value=get_session), \
         patch.object(prod_image_generation, "generate_image_from_api", api), \
         patch.object(prod_tool_functions.Common, "send_tool_discord_embed", AsyncMock()):
        result = await prod_tool_functions.edit_image.on_invoke_tool(
            _tool_context(message),
            json.dumps({"prompt": "p", "image_ref": "1"}),
        )

    assert "Could not download" in result
    api.assert_not_awaited()
    message.channel.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_edit_image_tool_default_ref_is_latest(monkeypatch):
    # image_ref omitted -> the most recent attachment is used
    refs = [
        {"ref": "1", "author": "a", "filename": "old.png",
         "url": "https://cdn.example/old.png"},
        {"ref": "2", "author": "b", "filename": "new.png",
         "url": "https://cdn.example/new.png"},
    ]
    message = MagicMock()
    message.add_reaction = AsyncMock()
    message.channel.send = AsyncMock()
    get_session = _mock_get_session()
    api = AsyncMock(return_value=b"EDITED")

    with patch("aiohttp.ClientSession", return_value=get_session), \
         patch.object(prod_image_generation, "generate_image_from_api", api), \
         patch.object(prod_tool_functions.Common, "send_tool_discord_embed", AsyncMock()):
        result = await prod_tool_functions.edit_image.on_invoke_tool(
            _tool_context(message, refs),
            json.dumps({"prompt": "make it snowy"}),
        )

    assert "sent to the channel" in result
    assert get_session.get.call_args.args[0] == "https://cdn.example/new.png"


@pytest.mark.asyncio
async def test_edit_image_tool_unknown_ref(monkeypatch):
    message = MagicMock()
    get_session = _mock_get_session()
    api = AsyncMock(return_value=b"EDITED")

    with patch("aiohttp.ClientSession", return_value=get_session), \
         patch.object(prod_image_generation, "generate_image_from_api", api), \
         patch.object(prod_tool_functions.Common, "send_tool_discord_embed", AsyncMock()):
        result = await prod_tool_functions.edit_image.on_invoke_tool(
            _tool_context(message),
            json.dumps({"prompt": "p", "image_ref": "9"}),
        )

    assert "No attached image" in result
    assert "'1'" in result  # available labels are listed
    get_session.get.assert_not_called()
    api.assert_not_awaited()


@pytest.mark.asyncio
async def test_edit_image_tool_no_refs(monkeypatch):
    message = MagicMock()
    get_session = _mock_get_session()
    api = AsyncMock(return_value=b"EDITED")

    with patch("aiohttp.ClientSession", return_value=get_session), \
         patch.object(prod_image_generation, "generate_image_from_api", api), \
         patch.object(prod_tool_functions.Common, "send_tool_discord_embed", AsyncMock()):
        result = await prod_tool_functions.edit_image.on_invoke_tool(
            _tool_context(message, []),
            json.dumps({"prompt": "p"}),
        )

    assert "No image is attached" in result
    get_session.get.assert_not_called()
    api.assert_not_awaited()

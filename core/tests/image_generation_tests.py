import base64
import json
import sys
from types import SimpleNamespace

# Imported as `classes.X`, the same name the app uses (it runs with cwd=core/,
# and pyproject.toml puts core/ on the test path).

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from classes import image_generation, image_prompt, image_review
from classes.image_generation import ImageResult, ServiceCapabilities
from classes.image_review import ImageReview

# To run this pytest file from the command line, use:
# PYTHONPATH=$(pwd) pytest core/tests/image_generation_tests.py

SDXL = ServiceCapabilities(prompt_style="sdxl", edits=False)
KLEIN = ServiceCapabilities(prompt_style="natural", edits=True)


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


def _health_session(status=200, body=None, error=None):
    response = MagicMock()
    response.status = status
    response.json = AsyncMock(return_value=body)
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=False)
    if error is not None:
        response.__aenter__.side_effect = error

    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    session.get = MagicMock(return_value=response)
    return session


@pytest.fixture(autouse=True)
def _reset_capability_cache():
    image_generation._capabilities = None
    yield
    image_generation._capabilities = None


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
    assert kwargs["json"] == {"prompt": "a red fox", "negative_prompt": ""}


@pytest.mark.asyncio
async def test_generate_sends_negative_prompt(monkeypatch):
    """The service merges this in front of its own IMAGE_NEGATIVE_PROMPT."""
    monkeypatch.setenv("DIFFUSION_URL", "http://diffusion:8000")
    session, response = _mock_session(status=200, payload=b"PNGDATA")
    with patch("aiohttp.ClientSession", return_value=session):
        await image_generation.generate_image_from_api("a red fox", "cartoon, blurry")

    _, kwargs = session.post.call_args
    assert kwargs["json"] == {"prompt": "a red fox", "negative_prompt": "cartoon, blurry"}


@pytest.mark.asyncio
async def test_generate_sends_a_reference_image_as_base64(monkeypatch):
    monkeypatch.setenv("DIFFUSION_URL", "http://diffusion:8000")
    session, _ = _mock_session(status=200, payload=b"EDITED")
    with patch("aiohttp.ClientSession", return_value=session), \
         patch.object(image_generation, "observe_image_generation") as observe:
        assert await image_generation.generate_image_from_api(
            "make her fatter", "", reference=b"OLDPNG") == b"EDITED"

    _, kwargs = session.post.call_args
    assert base64.b64decode(kwargs["json"]["reference_image"]) == b"OLDPNG"
    assert observe.call_args.args[0] == "edit"


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


# ---------------------- service_capabilities ----------------------

def test_parse_capabilities_reads_a_ready_service():
    assert image_generation.parse_capabilities(
        {"status": "ready", "prompt_style": "natural", "edits": True}) == KLEIN


@pytest.mark.parametrize("body", [
    None, [], {"status": "loading"},
    {"status": "ready"},                      # a service from before these fields
    {"status": "ready", "prompt_style": "weird", "edits": "yes"},
])
def test_parse_capabilities_defaults_to_how_images_were_always_made(body):
    parsed = image_generation.parse_capabilities(body)
    assert parsed is None or parsed == SDXL


@pytest.mark.asyncio
async def test_capabilities_are_cached_once_read():
    session = _health_session(body={"status": "ready", "prompt_style": "natural", "edits": True})
    with patch("aiohttp.ClientSession", return_value=session):
        assert await image_generation.service_capabilities() == KLEIN
        assert await image_generation.service_capabilities() == KLEIN
    assert session.get.call_count == 1


@pytest.mark.asyncio
async def test_a_failed_read_is_not_cached():
    """/health is 503 for minutes while the model loads after every restart.
    Caching the fallback through that would send klein SDXL tag lists, with
    editing off, for five minutes after each deploy."""
    loading = _health_session(status=503, body={"status": "loading"})
    ready = _health_session(body={"status": "ready", "prompt_style": "natural", "edits": True})
    with patch("aiohttp.ClientSession", side_effect=[loading, ready]):
        assert await image_generation.service_capabilities() == image_generation.FALLBACK_CAPABILITIES
        assert await image_generation.service_capabilities() == KLEIN


@pytest.mark.asyncio
async def test_an_unreachable_service_falls_back():
    session = _health_session(error=ConnectionError("no such host"))
    with patch("aiohttp.ClientSession", return_value=session):
        assert await image_generation.service_capabilities() == image_generation.FALLBACK_CAPABILITIES


# ---------------------- create_image ----------------------

MISSING_ARMS = ImageReview(shows="An obese woman with ordinary arms.", missing=("very thin arms",))
STILL_MISSING_TWO = ImageReview(shows="A woman.", missing=("very thin arms", "pizza slice"))
MATCH = ImageReview(shows="An obese woman with stick-thin arms.", missing=())


def _patched(capabilities, pngs, reviews, monkeypatch, retries="1"):
    """create_image with the service, rewriter and reviewer stubbed out."""
    monkeypatch.setenv("IMAGE_REVIEW_RETRIES", retries)
    api = AsyncMock(side_effect=pngs)
    review = AsyncMock(side_effect=reviews)
    rewrite = AsyncMock(side_effect=lambda request: (f"sdxl: {request}", "neg"))
    patches = (
        patch.object(image_generation, "service_capabilities", AsyncMock(return_value=capabilities)),
        patch.object(image_generation, "generate_image_from_api", api),
        patch.object(image_review, "review_image", review),
        patch.object(image_prompt, "build_image_prompt", rewrite),
    )
    return patches, api, review, rewrite


async def _run(patches, *args, **kwargs):
    with patches[0], patches[1], patches[2], patches[3]:
        return await image_generation.create_image(*args, **kwargs)


@pytest.mark.asyncio
async def test_a_matching_image_is_made_once(monkeypatch):
    patches, api, review, _ = _patched(KLEIN, [b"ONE"], [MATCH], monkeypatch)
    result = await _run(patches, "an obese woman with stick arms", "make her arms skinny")

    assert result.png == b"ONE" and result.attempts == 1 and result.review == MATCH
    api.assert_awaited_once()
    # checked against the person's own words AND the description
    assert review.await_args.args[1:] == ("make her arms skinny", "an obese woman with stick arms", None)


@pytest.mark.asyncio
async def test_a_miss_earns_one_retry_that_stresses_it(monkeypatch):
    patches, api, _, _ = _patched(KLEIN, [b"ONE", b"TWO"], [MISSING_ARMS, MATCH], monkeypatch)
    result = await _run(patches, "an obese woman with stick arms")

    assert result.png == b"TWO" and result.attempts == 2 and result.review == MATCH
    retry_prompt = api.await_args_list[1].args[0]
    assert retry_prompt.startswith("an obese woman with stick arms")
    assert "clearly shows: very thin arms" in retry_prompt
    assert result.prompt == retry_prompt


@pytest.mark.asyncio
async def test_a_worse_retry_keeps_the_first_image(monkeypatch):
    patches, _, _, _ = _patched(KLEIN, [b"ONE", b"TWO"], [MISSING_ARMS, STILL_MISSING_TWO], monkeypatch)
    result = await _run(patches, "an obese woman with stick arms")

    assert result.png == b"ONE" and result.review == MISSING_ARMS
    assert result.attempts == 2  # both were made, whichever is posted


@pytest.mark.asyncio
async def test_an_unchecked_image_is_not_retried(monkeypatch):
    patches, api, _, _ = _patched(KLEIN, [b"ONE"], [None], monkeypatch)
    result = await _run(patches, "a red fox")
    assert result.png == b"ONE" and result.review is None
    api.assert_awaited_once()


@pytest.mark.asyncio
async def test_retries_zero_checks_only(monkeypatch):
    patches, api, _, _ = _patched(KLEIN, [b"ONE"], [MISSING_ARMS], monkeypatch, retries="0")
    result = await _run(patches, "a red fox")
    assert result.review == MISSING_ARMS
    api.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_failed_retry_posts_the_first_attempt(monkeypatch):
    patches, _, _, _ = _patched(KLEIN, [b"ONE", Exception("queue full")], [MISSING_ARMS], monkeypatch)
    result = await _run(patches, "a red fox")
    assert result.png == b"ONE" and result.attempts == 1


@pytest.mark.asyncio
async def test_a_failed_first_attempt_raises(monkeypatch):
    patches, _, _, _ = _patched(KLEIN, [Exception("service down")], [], monkeypatch)
    with pytest.raises(Exception, match="service down"):
        await _run(patches, "a red fox")


@pytest.mark.asyncio
async def test_sdxl_prompts_are_rewritten_natural_ones_are_not(monkeypatch):
    patches, api, _, rewrite = _patched(SDXL, [b"ONE"], [MATCH], monkeypatch)
    await _run(patches, "a red fox")
    rewrite.assert_awaited_once_with("a red fox")
    assert api.await_args.args[:2] == ("sdxl: a red fox", "neg")

    patches, api, _, rewrite = _patched(KLEIN, [b"ONE"], [MATCH], monkeypatch)
    await _run(patches, "a red fox")
    rewrite.assert_not_awaited()
    assert api.await_args.args[:2] == ("a red fox", "")


@pytest.mark.asyncio
async def test_an_edit_retries_from_the_original_reference(monkeypatch):
    patches, api, review, _ = _patched(KLEIN, [b"ONE", b"TWO"], [MISSING_ARMS, MATCH], monkeypatch)
    result = await _run(patches, "make her arms thin", reference=b"ORIGINAL")

    assert [call.args[2] for call in api.await_args_list] == [b"ORIGINAL", b"ORIGINAL"]
    # the check compares each attempt against the picture being edited
    assert [call.args[3] for call in review.await_args_list] == [b"ORIGINAL", b"ORIGINAL"]
    assert result.edited is True


@pytest.mark.asyncio
async def test_the_first_prompt_is_announced_before_generating(monkeypatch):
    patches, api, _, _ = _patched(KLEIN, [b"ONE", b"TWO"], [MISSING_ARMS, MATCH], monkeypatch)
    seen = []

    async def announce(prompt):
        assert api.await_count == 0
        seen.append(prompt)

    await _run(patches, "a red fox", on_first_prompt=announce)
    assert seen == ["a red fox"]


# ---------------------- tool_result_text ----------------------

def test_the_model_is_told_what_the_image_actually_shows():
    text = image_generation.tool_result_text(ImageResult(b"", "p", MATCH, 1))
    assert MATCH.shows in text
    assert "do not send it again" in text


def test_a_remaining_miss_must_be_admitted():
    """The caption used to claim "stick arms" for a picture without them."""
    text = image_generation.tool_result_text(ImageResult(b"", "p", MISSING_ARMS, 2))
    assert "very thin arms" in text and "2 attempts" in text
    assert "do not claim it did" in text


def test_an_unchecked_image_gets_no_details_claimed():
    text = image_generation.tool_result_text(ImageResult(b"", "p", None, 1))
    assert "do not claim" in text


def test_an_edit_says_so():
    assert image_generation.tool_result_text(ImageResult(b"", "p", MATCH, 1, edited=True)) \
        .startswith("Edited image")


# ---------------------- which picture to edit ----------------------

BOT = 42


def _message(author_id, *filenames, message_id=1):
    attachments = [SimpleNamespace(filename=name, read=AsyncMock(return_value=f"<{name}@{message_id}>".encode()))
                   for name in filenames]
    return SimpleNamespace(id=message_id, author=SimpleNamespace(id=author_id),
                           attachments=attachments, reference=None)


def test_edit_source_prefers_the_replied_to_image():
    replied = _message(BOT, "generated-image.png", message_id=5)
    newer = _message(BOT, "generated-image.png", message_id=9)
    assert image_generation.pick_edit_source(replied, [newer], BOT) is replied.attachments[0]


def test_edit_source_falls_back_to_the_newest_generated_image():
    older = _message(BOT, "generated-image.png", message_id=3)
    newest = _message(BOT, "generated-image.png", message_id=8)
    assert image_generation.pick_edit_source(None, [newest, older], BOT) is newest.attachments[0]


def test_edit_source_ignores_other_images():
    """Uploaded photos (real people) and the bot's sandbox output are never edited."""
    upload = _message(7, "generated-image.png")
    chart = _message(BOT, "chart.png")
    assert image_generation.pick_edit_source(upload, [upload, chart], BOT) is None


def test_edit_source_skips_a_reply_to_something_else():
    replied = _message(7, "holiday.jpg")
    mine = _message(BOT, "generated-image.png")
    assert image_generation.pick_edit_source(replied, [mine], BOT) is mine.attachments[0]


class _FakeChannel:
    def __init__(self, messages, fetchable=None):
        self.messages = messages          # newest first
        self.fetchable = fetchable or {}
        self.history_kwargs = None

    def history(self, **kwargs):
        self.history_kwargs = kwargs

        async def _gen():
            for message in self.messages[:kwargs["limit"]]:
                yield message
        return _gen()

    async def fetch_message(self, message_id):
        return self.fetchable[message_id]


@pytest.mark.asyncio
async def test_find_edit_source_fetches_an_unresolved_reply():
    replied = _message(BOT, "generated-image.png", message_id=5)
    trigger = _message(7, message_id=10)
    trigger.reference = SimpleNamespace(message_id=5, resolved=None)
    channel = _FakeChannel([_message(BOT, "generated-image.png", message_id=9)], {5: replied})

    assert await image_generation.find_edit_source(channel, trigger, BOT) == b"<generated-image.png@5>"


@pytest.mark.asyncio
async def test_find_edit_source_sees_an_image_posted_after_the_request():
    """The model often makes a picture, sees what it missed and edits it in
    the SAME reply -- that picture is newer than the request. Looking only
    before the request edited an older, unrelated picture instead."""
    trigger = _message(7, message_id=10)
    just_made = _message(BOT, "generated-image.png", message_id=12)
    older = _message(BOT, "generated-image.png", message_id=4)
    channel = _FakeChannel([just_made, trigger, older])

    assert await image_generation.find_edit_source(channel, trigger, BOT) == b"<generated-image.png@12>"
    assert "before" not in channel.history_kwargs


@pytest.mark.asyncio
async def test_find_edit_source_without_a_trigger_scans_the_channel():
    """Automatic runs have no source message."""
    channel = _FakeChannel([_message(BOT, "generated-image.png", message_id=9)])
    assert await image_generation.find_edit_source(channel, None, BOT) == b"<generated-image.png@9>"


@pytest.mark.asyncio
async def test_find_edit_source_returns_none_when_there_is_nothing():
    channel = _FakeChannel([_message(7, "holiday.jpg")])
    assert await image_generation.find_edit_source(channel, None, BOT) is None


# ---------------------- the generate_image tool ----------------------

def _tool_context(context):
    from agents.tool_context import ToolContext
    return ToolContext(context=context, tool_name="generate_image",
                       tool_call_id="t1", tool_arguments="{}")


def _tool_env(message=None):
    channel = MagicMock()
    channel.send = AsyncMock()
    client = SimpleNamespace(user=SimpleNamespace(id=BOT))
    return channel, {"channel": channel, "original_message": message, "discord_client": client}


def _creates(result):
    """A create_image stand-in that announces its first prompt, as the real one does."""
    async def create(description, user_request="", reference=None, on_first_prompt=None):
        if on_first_prompt is not None:
            await on_first_prompt(description)
        return result
    return AsyncMock(side_effect=create)


async def _invoke(context, **arguments):
    from classes import tool_functions
    with patch.object(tool_functions.Common, "send_tool_discord_embed",
                      AsyncMock(return_value=MagicMock())) as embed, \
         patch.object(tool_functions.Common, "edit_tool_discord_embed", AsyncMock()) as edit:
        result = await tool_functions.generate_image.on_invoke_tool(
            _tool_context(context), json.dumps(arguments))
    return result, embed, edit


@pytest.mark.asyncio
async def test_tool_posts_the_final_image_and_reports_what_it_shows():
    channel, context = _tool_env()
    made = ImageResult(b"TWO", "a red fox. Make sure...", MATCH, 2)
    with patch.object(image_generation, "create_image", _creates(made)) as create:
        result, embed, edit = await _invoke(context, prompt="a red fox")

    assert MATCH.shows in result
    assert channel.send.await_args.kwargs["file"].filename == "generated-image.png"
    assert create.await_args.kwargs["reference"] is None
    # the embed is brought in line with the image that was actually posted
    edit.assert_awaited_once()
    assert edit.await_args.args[1] == "Generating image: a red fox. Make sure..."


@pytest.mark.asyncio
async def test_tool_edits_the_previous_image():
    channel, context = _tool_env()
    made = ImageResult(b"EDITED", "make her fatter", MATCH, 1, edited=True)
    with patch.object(image_generation, "service_capabilities", AsyncMock(return_value=KLEIN)), \
         patch.object(image_generation, "find_edit_source", AsyncMock(return_value=b"OLD")) as find, \
         patch.object(image_generation, "create_image", _creates(made)) as create:
        result, embed, edit = await _invoke(context, prompt="make her fatter", edit_previous=True)

    assert find.await_args.args[2] == BOT
    assert create.await_args.kwargs["reference"] == b"OLD"
    assert result.startswith("Edited image")
    assert embed.await_args.args[1] == "Editing image: make her fatter"
    edit.assert_not_awaited()  # one attempt: the announced prompt is already right


@pytest.mark.asyncio
async def test_tool_will_not_draw_an_edit_from_scratch():
    """An edit prompt only names the change; drawn fresh it is a picture of nobody."""
    _, context = _tool_env()
    with patch.object(image_generation, "service_capabilities", AsyncMock(return_value=SDXL)), \
         patch.object(image_generation, "create_image", AsyncMock()) as create:
        result, _, _ = await _invoke(context, prompt="make her fatter", edit_previous=True)
    create.assert_not_awaited()
    assert "cannot edit" in result and "edit_previous false" in result


@pytest.mark.asyncio
async def test_tool_reports_when_there_is_nothing_to_edit():
    _, context = _tool_env()
    with patch.object(image_generation, "service_capabilities", AsyncMock(return_value=KLEIN)), \
         patch.object(image_generation, "find_edit_source", AsyncMock(return_value=None)), \
         patch.object(image_generation, "create_image", AsyncMock()) as create:
        result, _, _ = await _invoke(context, prompt="make her fatter", edit_previous=True)
    create.assert_not_awaited()
    assert "no earlier picture" in result


@pytest.mark.asyncio
async def test_tool_reports_a_failed_generation():
    _, context = _tool_env()
    with patch.object(image_generation, "create_image", AsyncMock(side_effect=Exception("down"))):
        result, _, _ = await _invoke(context, prompt="a red fox")
    assert "unavailable" in result


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

    def add_command(self, command):
        self.commands.append((command.name, command))

    async def sync(self):
        return [MagicMock(name=f"synced-{i}") for i in range(len(self.commands))]


async def _register_and_collect(monkeypatch, enabled_value):
    main_mod = _import_main()
    monkeypatch.setenv("IMAGE_GEN_ENABLED", enabled_value)
    # The command rewrites the prompt and checks the image through an LLM.
    # Both off here, so these tests do not open a real connection to LLM_HOST
    # and sit through its timeout before the call falls soft.
    monkeypatch.setenv("IMAGE_PROMPT_REWRITE_ENABLED", "0")
    monkeypatch.setenv("IMAGE_REVIEW_ENABLED", "0")
    monkeypatch.setattr(image_generation, "service_capabilities", AsyncMock(return_value=SDXL))
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
    with patch.object(image_generation, "generate_image_from_api",
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
async def test_generate_image_command_uses_the_rewritten_prompt(monkeypatch):
    """The slash command is raw user input the agent never sees, so the SDXL
    rewrite has to happen here rather than in the tool docstring."""
    main_mod, commands = await _register_and_collect(monkeypatch, "1")
    fn = commands["generate_image"]

    ctx = MagicMock()
    ctx.response.defer = AsyncMock()
    ctx.edit_original_response = AsyncMock()
    api = AsyncMock(return_value=b"PNGDATA")
    with patch.object(image_generation, "generate_image_from_api", new=api), \
         patch.object(image_prompt, "build_image_prompt",
                      new=AsyncMock(return_value=("a red fox, sharp fur", "cartoon"))):
        await fn(ctx, "a red fox")

    api.assert_awaited_once_with("a red fox, sharp fur", "cartoon", None)


@pytest.mark.asyncio
async def test_generate_image_command_reports_failure(monkeypatch):
    main_mod, commands = await _register_and_collect(monkeypatch, "1")
    fn = commands["generate_image"]

    ctx = MagicMock()
    ctx.response.defer = AsyncMock()
    ctx.edit_original_response = AsyncMock()
    with patch.object(image_generation, "generate_image_from_api",
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

import io
import os
import sys

# message_handler pulls in sibling modules via the production "classes.*" import
# style (the app runs with cwd=core/), so make core/ importable for tests too.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import asyncio
import base64
import time

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from PIL import Image

from core.classes.message_handler import (
    MAX_IMAGES_PER_MESSAGE,
    MessageHandler,
    encode_image_for_llm,
)
from core.classes import message_queue as mq

# To run this pytest file from the command line, use:
# PYTHONPATH=$(pwd) pytest core/tests/message_handler_tests.py


def _make_image(fmt="JPEG", size=(16, 16), color=(200, 30, 30)):
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _handler():
    # Skip __init__ (it needs REDIS_HOST); download_image_parts does not use self state.
    return MessageHandler.__new__(MessageHandler)


def _attachment(url="http://cdn.example/x.png", content_type="image/png"):
    att = MagicMock()
    att.url = url
    att.proxy_url = url
    att.content_type = content_type
    return att


def _mock_client_session(payload=b"", status=200, raise_exc=None):
    response = MagicMock()
    response.status = status
    response.headers = {"Content-Type": "image/png"}
    response.read = AsyncMock(return_value=payload)
    if raise_exc:
        response.read.side_effect = raise_exc
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=False)

    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    session.get = MagicMock(return_value=response)
    return session


def _part_image(part):
    url = part["image_url"]
    assert url.startswith("data:")
    ctype, b64 = url.split(";base64,")
    data = base64.b64decode(b64)
    return ctype, data


def _decode(data):
    with Image.open(io.BytesIO(data)) as img:
        img.load()
        return img.format, img.size


# ---------------------- encode_image_for_llm (pure) ----------------------

def test_jpeg_is_kept_as_jpeg():
    out, mime = encode_image_for_llm(_make_image("JPEG"))
    assert mime == "image/jpeg"
    assert _decode(out) == ("JPEG", (16, 16))


def test_png_is_kept_as_png():
    out, mime = encode_image_for_llm(_make_image("PNG"))
    assert mime == "image/png"
    assert _decode(out) == ("PNG", (16, 16))


def test_gif_is_kept_as_gif():
    out, mime = encode_image_for_llm(_make_image("GIF"))
    assert mime == "image/gif"
    assert _decode(out)[0] == "GIF"


def test_webp_is_reencoded_to_png():
    # Ollama cannot decode WebP ("Failed to load image or audio file"),
    # so it must be converted away.
    out, mime = encode_image_for_llm(_make_image("WEBP"))
    assert mime == "image/png"
    assert _decode(out) == ("PNG", (16, 16))


def test_heic_is_reencoded_to_png():
    # Phone cameras (iPhone) default to HEIC; same Ollama limitation as WebP.
    out, mime = encode_image_for_llm(_make_image("HEIF"))
    assert mime == "image/png"
    assert _decode(out) == ("PNG", (16, 16))


def test_invalid_bytes_return_none():
    assert encode_image_for_llm(b"this is not an image") is None
    assert encode_image_for_llm(b"") is None


# ------------------- download_image_parts (mocked io) --------------------

@pytest.mark.asyncio
async def test_image_attachment_becomes_decodable_data_url():
    payload = _make_image("JPEG")
    handler = _handler()
    with patch("core.classes.message_handler.aiohttp.ClientSession",
               return_value=_mock_client_session(payload=payload)):
        parts = await handler.download_image_parts([_attachment()])

    assert len(parts) == 1
    ctype, data = _part_image(parts[0])
    assert ctype == "data:image/jpeg"
    assert _decode(data) == ("JPEG", (16, 16))


@pytest.mark.asyncio
async def test_webp_attachment_arrives_as_png_for_the_llm():
    payload = _make_image("WEBP")
    handler = _handler()
    with patch("core.classes.message_handler.aiohttp.ClientSession",
               return_value=_mock_client_session(payload=payload)):
        parts = await handler.download_image_parts([_attachment()])

    assert len(parts) == 1
    ctype, data = _part_image(parts[0])
    assert ctype == "data:image/png"
    assert _decode(data) == ("PNG", (16, 16))


@pytest.mark.asyncio
async def test_heic_attachment_arrives_as_png_for_the_llm():
    payload = _make_image("HEIF")
    handler = _handler()
    with patch("core.classes.message_handler.aiohttp.ClientSession",
               return_value=_mock_client_session(payload=payload)):
        parts = await handler.download_image_parts([_attachment(content_type="image/heic")])

    assert len(parts) == 1
    ctype, data = _part_image(parts[0])
    assert ctype == "data:image/png"
    assert _decode(data) == ("PNG", (16, 16))


@pytest.mark.asyncio
async def test_undecodable_download_is_skipped():
    handler = _handler()
    with patch("core.classes.message_handler.aiohttp.ClientSession",
               return_value=_mock_client_session(payload=b"not an image")):
        parts = await handler.download_image_parts([_attachment()])
    assert parts == []


@pytest.mark.asyncio
async def test_non_image_attachments_are_skipped():
    handler = _handler()
    with patch("core.classes.message_handler.aiohttp.ClientSession",
               return_value=_mock_client_session()):
        parts = await handler.download_image_parts([
            _attachment(url="http://cdn.example/v.mp4", content_type="video/mp4"),
            _attachment(url="http://cdn.example/f.pdf", content_type="application/pdf"),
            _attachment(content_type=None),
        ])

    assert parts == []


@pytest.mark.asyncio
async def test_download_failure_is_skipped():
    handler = _handler()
    with patch("core.classes.message_handler.aiohttp.ClientSession",
               return_value=_mock_client_session(raise_exc=ConnectionError("down"))):
        parts = await handler.download_image_parts([_attachment()])
    assert parts == []


@pytest.mark.asyncio
async def test_non_200_response_is_skipped():
    handler = _handler()
    with patch("core.classes.message_handler.aiohttp.ClientSession",
               return_value=_mock_client_session(status=404)):
        parts = await handler.download_image_parts([_attachment()])
    assert parts == []


@pytest.mark.asyncio
async def test_max_images_per_message_cap():
    handler = _handler()
    payload = _make_image("JPEG")
    attachments = [_attachment(url=f"http://cdn.example/{i}.png")
                   for i in range(MAX_IMAGES_PER_MESSAGE + 3)]
    with patch("core.classes.message_handler.aiohttp.ClientSession",
               return_value=_mock_client_session(payload=payload)):
        parts = await handler.download_image_parts(attachments)
    assert len(parts) == MAX_IMAGES_PER_MESSAGE


@pytest.mark.asyncio
async def test_mixed_good_and_bad_attachments_only_sends_valid_images():
    handler = _handler()
    with patch("core.classes.message_handler.aiohttp.ClientSession",
               side_effect=lambda: _mock_client_session(payload=_make_image("JPEG"))):
        parts = await handler.download_image_parts([
            _attachment(url="http://cdn.example/a.jpg", content_type="image/jpeg"),
            _attachment(url="http://cdn.example/b.png", content_type="image/png"),
            _attachment(url="http://cdn.example/c.txt", content_type="text/plain"),
        ])
    assert len(parts) == 2
    for part in parts:
        ctype, data = _part_image(part)
        assert ctype == "data:image/jpeg"
        _decode(data)


# ------------------- handle_message: scoped per-channel lock -------------------
#
# The per-channel lock (classes.message_queue.get_channel_lock) guards only the
# two FAST phases of a handle — build and send — so the slow LLM/tool phase
# runs UNLOCKED: a free worker can answer a NEW message in the same channel
# while the first is stuck in a slow tool. The tests below run the REAL
# handle_message with a fake LLM and assert both halves of that contract:
# generations overlap, chunked sends never interleave, and the in-flight
# registry hint is appended to the prompt.


def _typing_channel(channel_id=99):
    ch = MagicMock()
    ch.id = channel_id
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=None)
    cm.__aexit__ = AsyncMock(return_value=False)
    ch.typing = MagicMock(return_value=cm)
    ch.send = AsyncMock()  # handle_message_send awaits channel.send(chunk)
    return ch


def _llm_message(mid, channel_id=99):
    msg = MagicMock()
    msg.id = mid
    msg.content = "hi"
    msg.guild = MagicMock()
    msg.guild.id = 1
    msg.channel = _typing_channel(channel_id)
    return msg


def _handled(handler, msg, set_messages):
    """Wire a _handler() instance to a message and a fake (sync-setting) build."""
    handler.message = msg
    handler.client = MagicMock()
    handler.attachment_refs = []  # normally set by build_messages

    async def _build():
        set_messages(handler)
    handler.build_messages = _build
    return handler


@pytest.mark.asyncio
async def test_handle_message_scoped_lock_generates_concurrently_sends_serially():
    """Two same-channel messages handled on two worker tasks: their LLM runs
    overlap (the lock does NOT cover the LLM phase — this is what frees the
    bot to answer in the same channel during a sandbox run), but their
    chunked sends are serialized by the lock and never interleave.

    With the old FULL lock (build + LLM + send) message 2's generation would
    only start after message 1's whole handle finished, so the concurrency
    assertion below discriminates against a regression to it."""

    class _FakeLLM:
        gen_events = []  # (msg_id, phase, monotonic)

        def __init__(self, messages, guild_id, original_message, attachment_refs=None):
            self.original_message = original_message
            self.reasoning = ""

        async def generate(self):
            mid = self.original_message.id
            _FakeLLM.gen_events.append((mid, "start", time.monotonic()))
            await asyncio.sleep(0.2)
            _FakeLLM.gen_events.append((mid, "end", time.monotonic()))
            return ("word " * 800).strip()  # ~4000 chars -> 2 chunks each

    sends = []
    handlers = []
    for mid in (1, 2):
        handler = _handler()
        msg = _llm_message(mid)

        async def _send(text, *args, _mid=mid, **kwargs):
            sends.append(_mid)

        msg.channel.send = AsyncMock(side_effect=_send)
        handlers.append(_handled(handler, msg, lambda h: setattr(h, "messages", [])))

    with patch("core.classes.message_handler.TextLLMHandler", _FakeLLM):
        await asyncio.gather(handlers[0].handle_message(), handlers[1].handle_message())

    # 1) generations overlapped: msg 2's LLM run started before msg 1's ended
    phases = {}
    for mid, phase, ts in _FakeLLM.gen_events:
        phases.setdefault(mid, {})[phase] = ts
    assert phases[2]["start"] < phases[1]["end"], (
        "the LLM phase must run without holding the channel lock "
        "(same-channel messages generate concurrently)"
    )

    # 2) sends serialized: each message's 2 chunks form one contiguous block
    assert sorted(sends) == [1, 1, 2, 2]
    assert sends == [1, 1, 2, 2] or sends == [2, 2, 1, 1], (
        f"chunked sends interleaved: {sends}"
    )


@pytest.mark.asyncio
async def test_handle_message_appends_in_flight_hint_to_prompt():
    """While an earlier message's slow tool is in flight in this channel
    (registered via the ToolMetricsHooks path in production), the prompt
    built for a NEWER same-channel message gets the in-flight hint as a
    trailing user note — so the model can answer follow-ups honestly."""
    channel_id = 200
    captured = {}

    class _FakeLLM:
        def __init__(self, messages, guild_id, original_message, attachment_refs=None):
            captured["prompt"] = messages
            self.reasoning = ""

        async def generate(self):
            return "ok"

    handler = _handled(
        _handler(), _llm_message(11, channel_id),
        lambda h: setattr(h, "messages", [{"role": "user", "content": "is it done yet?"}]),
    )

    assert mq.register_task_run(channel_id, "🐳 code sandbox", "compute pi", run_key="t1")
    try:
        with patch("core.classes.message_handler.TextLLMHandler", _FakeLLM):
            await handler.handle_message()
    finally:
        mq.unregister_task_run(channel_id, run_key="t1")

    prompt = captured["prompt"]
    assert len(prompt) == 2  # the user message + the appended hint
    assert prompt[1]["role"] == "user"
    assert prompt[1]["content"].startswith("Note:")
    assert "🐳 code sandbox" in prompt[1]["content"]
    assert "compute pi" in prompt[1]["content"]


@pytest.mark.asyncio
async def test_handle_message_no_hint_when_channel_idle():
    channel_id = 201
    captured = {}

    class _FakeLLM:
        def __init__(self, messages, guild_id, original_message, attachment_refs=None):
            captured["prompt"] = messages
            self.reasoning = ""

        async def generate(self):
            return "ok"

    handler = _handled(
        _handler(), _llm_message(12, channel_id),
        lambda h: setattr(h, "messages", [{"role": "user", "content": "hello"}]),
    )

    with patch("core.classes.message_handler.TextLLMHandler", _FakeLLM):
        await handler.handle_message()

    assert len(captured["prompt"]) == 1  # nothing appended
    assert captured["prompt"][0]["content"] == "hello"


# --------------------------- reasoning follow-up (SHOW_THINKING) --------------------------

def _reasoning_llm(reasoning, answer="the answer"):
    """A TextLLMHandler stand-in that reports reasoning the way the real one
    does: as an attribute set during generate(), NOT as think tags inside
    the returned answer (our llama.cpp server strips those out into
    reasoning_content, so the answer text never carries them)."""

    class _FakeLLM:
        def __init__(self, messages, guild_id, original_message, attachment_refs=None):
            self.reasoning = ""

        async def generate(self):
            self.reasoning = reasoning
            return answer

    return _FakeLLM


async def _run_with_llm(fake_llm, channel_id, sends):
    handler = _handled(
        _handler(), _llm_message(30, channel_id),
        lambda h: setattr(h, "messages", []),
    )
    handler.client.user.id = 1234
    handler.message.channel.send = AsyncMock(side_effect=lambda text, *a, **k: sends.append(text))
    handler.message.add_reaction = AsyncMock()
    with patch("core.classes.message_handler.TextLLMHandler", fake_llm):
        await handler.handle_message()
    return handler


@pytest.mark.asyncio
async def test_handle_message_sends_out_of_band_reasoning_as_spoiler():
    """Regression: the reasoning arrives on the handler, not in the answer
    text. Reading it off the response string instead (as extract_thinking
    did) silently sent nothing at all."""
    sends = []
    with patch("core.classes.message_handler.SHOW_THINKING", True):
        await _run_with_llm(_reasoning_llm("i reasoned about it"), 300, sends)

    assert sends[0] == "the answer"
    assert sends[1].startswith("-# Reasoning")
    assert "i reasoned about it" in sends[2]
    assert sends[2].startswith("||```\n") and sends[2].endswith("\n```||")


@pytest.mark.asyncio
async def test_handle_message_sends_no_reasoning_when_show_thinking_off():
    sends = []
    with patch("core.classes.message_handler.SHOW_THINKING", False):
        await _run_with_llm(_reasoning_llm("i reasoned about it"), 301, sends)

    assert sends == ["the answer"]


@pytest.mark.asyncio
async def test_handle_message_sends_no_reasoning_when_model_did_not_think():
    sends = []
    with patch("core.classes.message_handler.SHOW_THINKING", True):
        await _run_with_llm(_reasoning_llm(""), 302, sends)

    assert sends == ["the answer"]


# --------------------------- TextLLMHandler.generate reasoning capture --------------------------
#
# The other half of the pipeline: generate() must pull the reasoning off the
# run result, because it is NOT in the answer string it returns.

def _generate_handler(run_result):
    from core.classes.text_llm_handler import TextLLMHandler

    handler = TextLLMHandler.__new__(TextLLMHandler)  # __init__ needs Redis
    handler.messages = [{"role": "user", "content": "hi"}]
    handler.guild_id = 1
    handler.reasoning = ""
    handler.system = "a bot"
    handler.agent = MagicMock()
    handler.attachment_refs = []
    handler.original_message = _llm_message(40)
    handler.user_memory = MagicMock()
    handler.user_memory.get = AsyncMock(return_value=[])
    handler.get_settings = AsyncMock()
    handler.get_client = AsyncMock()
    return handler


def _run_result(final_output, items=()):
    result = MagicMock()
    result.final_output = final_output
    result.new_items = list(items)
    return result


async def _generate(run_result):
    handler = _generate_handler(run_result)
    with patch("core.classes.text_llm_handler.Runner") as runner, \
         patch("core.classes.text_llm_handler.get_current_datetime", AsyncMock(return_value="now")):
        runner.run = AsyncMock(return_value=run_result)
        answer = await handler.generate()
    return answer, handler.reasoning


@pytest.mark.asyncio
async def test_generate_captures_reasoning_from_run_items():
    """The llama.cpp shape: reasoning_content becomes its own run item and
    the answer text is clean, so the run items are the only source."""
    from core.tests.response_filter_tests import _reasoning_item

    answer, reasoning = await _generate(
        _run_result("the clean answer", [_reasoning_item(summary=["thought hard"])])
    )
    assert answer == "the clean answer"
    assert reasoning == "thought hard"


@pytest.mark.asyncio
async def test_generate_falls_back_to_inline_think_tags():
    """A server on --reasoning-format none leaves the reasoning inline and
    emits no reasoning item; the tag path still has to work."""
    from core.tests.response_filter_tests import _think_open, _think_close

    text = _think_open(tab=False) + "inline thought" + _think_close(tab=False) + "the answer"
    answer, reasoning = await _generate(_run_result(text))
    assert reasoning == "inline thought"
    assert answer == text  # filtering the tags out stays message_handler's job


@pytest.mark.asyncio
async def test_generate_leaves_reasoning_empty_when_model_did_not_think():
    answer, reasoning = await _generate(_run_result("just an answer"))
    assert answer == "just an answer"
    assert reasoning == ""


# --------------------------- reasoning survives a failed run --------------------------
#
# A run that dies part-way (MaxTurnsExceeded while chaining tool calls is the
# common one) has already let its tools post embeds/files to the channel. The
# reasoning it produced before breaking must still reach the user, or the ❌
# reads as "the tool call worked but the bot went quiet".

@pytest.mark.asyncio
async def test_generate_recovers_reasoning_from_a_failed_run():
    from core.tests.response_filter_tests import _reasoning_item
    from agents import MaxTurnsExceeded

    handler = _generate_handler(None)
    exc = MaxTurnsExceeded("Max turns (20) exceeded")
    exc.run_data = MagicMock()
    exc.run_data.new_items = [_reasoning_item(summary=["i kept calling the sandbox"])]

    with patch("core.classes.text_llm_handler.Runner") as runner, \
         patch("core.classes.text_llm_handler.get_current_datetime", AsyncMock(return_value="now")), \
         patch("core.classes.text_llm_handler.inc_llm_error", MagicMock()):
        runner.run = AsyncMock(side_effect=exc)
        answer = await handler.generate()

    assert answer == "Error"  # sentinel unchanged
    assert handler.reasoning == "i kept calling the sandbox"


@pytest.mark.asyncio
async def test_generate_survives_a_failure_carrying_no_run_data():
    from agents import MaxTurnsExceeded

    handler = _generate_handler(None)
    with patch("core.classes.text_llm_handler.Runner") as runner, \
         patch("core.classes.text_llm_handler.get_current_datetime", AsyncMock(return_value="now")), \
         patch("core.classes.text_llm_handler.inc_llm_error", MagicMock()):
        runner.run = AsyncMock(side_effect=MaxTurnsExceeded("boom"))
        answer = await handler.generate()

    assert answer == "Error"
    assert handler.reasoning == ""


@pytest.mark.asyncio
async def test_generate_passes_an_explicit_max_turns():
    """The SDK's own default is 10, which a chained tool run overruns."""
    from core.classes.text_llm_handler import llm_max_turns

    handler = _generate_handler(None)
    with patch("core.classes.text_llm_handler.Runner") as runner, \
         patch("core.classes.text_llm_handler.get_current_datetime", AsyncMock(return_value="now")):
        runner.run = AsyncMock(return_value=_run_result("ok"))
        await handler.generate()

    assert runner.run.await_args.kwargs["max_turns"] == llm_max_turns()
    assert llm_max_turns() > 10


@pytest.mark.asyncio
async def test_handle_message_sends_reasoning_even_when_the_run_failed():
    sends = []

    class _FailingLLM:
        def __init__(self, messages, guild_id, original_message, attachment_refs=None):
            self.reasoning = ""

        async def generate(self):
            self.reasoning = "i was thinking when it broke"
            return "Error"

    with patch("core.classes.message_handler.SHOW_THINKING", True):
        handler = await _run_with_llm(_FailingLLM, 310, sends)

    handler.message.add_reaction.assert_awaited_with('❌')
    assert sends[0].startswith("-# Reasoning")
    assert "i was thinking when it broke" in sends[1]


@pytest.mark.asyncio
async def test_handle_message_failed_run_with_no_reasoning_just_reacts():
    sends = []

    class _FailingLLM:
        def __init__(self, messages, guild_id, original_message, attachment_refs=None):
            self.reasoning = ""

        async def generate(self):
            return "Error"

    with patch("core.classes.message_handler.SHOW_THINKING", True):
        handler = await _run_with_llm(_FailingLLM, 311, sends)

    handler.message.add_reaction.assert_awaited_with('❌')
    assert sends == []

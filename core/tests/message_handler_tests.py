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

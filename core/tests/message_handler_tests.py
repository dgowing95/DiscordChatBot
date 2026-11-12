import io
import os
import sys

# message_handler pulls in sibling modules via the production "classes.*" import
# style (the app runs with cwd=core/), so make core/ importable for tests too.

import asyncio
import base64
import time

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from PIL import Image

from classes.message_handler import (
    MAX_IMAGES_PER_MESSAGE,
    MessageHandler,
    encode_image_for_llm,
)
from classes import attachment_cache
from classes import message_queue as mq

# To run this pytest file from the command line, use:
# PYTHONPATH=$(pwd) pytest core/tests/message_handler_tests.py


def _make_image(fmt="JPEG", size=(16, 16), color=(200, 30, 30)):
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _handler():
    # Skip __init__ (it needs REDIS_HOST).
    return MessageHandler.__new__(MessageHandler)


def _attachment(url="http://cdn.example/x.png", content_type="image/png", att_id=None):
    att = MagicMock()
    if att_id is not None:
        att.id = att_id
    att.url = url
    att.proxy_url = url
    att.content_type = content_type
    return att


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


# ------------- image_parts / prepare_messages (mocked io, real cache) -------------
#
# Downloads go through the module's shared session (_http_session), patched
# here with a fake that records every GET. The attachment cache is the real
# one, reset around each test.


class _FakeResponse:
    def __init__(self, payload, status):
        self.status = status
        self._payload = payload

    async def read(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    """GET url -> payload (bytes or an Exception) / status. `gate`, when set,
    holds every GET until it is released, so concurrency can be measured."""

    def __init__(self, payload=b"", status=200, by_url=None, gate=None, on_get=None):
        self.payload = payload
        self.status = status
        self.by_url = by_url or {}
        self.gate = gate
        self.on_get = on_get
        self.calls = []
        self.active = 0
        self.peak = 0

    def get(self, url):
        session = self

        class _Ctx:
            async def __aenter__(self_inner):
                session.calls.append(url)
                session.active += 1
                session.peak = max(session.peak, session.active)
                try:
                    if session.on_get:
                        session.on_get(url)
                    if session.gate is not None:
                        await session.gate.wait()
                finally:
                    session.active -= 1
                payload, status = session.by_url.get(url, (session.payload, session.status))
                return _FakeResponse(payload, status)

            async def __aexit__(self_inner, *exc):
                return False
        return _Ctx()


@pytest.fixture(autouse=True)
def _fresh_attachment_cache():
    attachment_cache.reset()
    yield
    attachment_cache.reset()


def _use_session(session):
    return patch("classes.message_handler._http_session", return_value=session)


def _image_message(attachments, mid=10, channel_id=99, content="look", author_id=1):
    msg = MagicMock()
    msg.id = mid
    msg.content = content
    msg.attachments = attachments
    msg.embeds = []
    msg.guild = MagicMock()
    msg.guild.id = 1
    msg.channel = MagicMock()
    msg.channel.id = channel_id
    msg.author.id = author_id
    msg.author.name = "alice"
    return msg


def _image_handler(message):
    handler = _handler()
    handler.message = message
    handler.client = MagicMock()
    handler.client.user.id = 999
    handler._history = []
    handler._generation = 0
    return handler


async def _parts(attachments, session):
    msg = _image_message(attachments)
    with _use_session(session):
        return await _image_handler(msg).image_parts(msg)


@pytest.mark.asyncio
async def test_image_attachment_becomes_decodable_data_url():
    parts = await _parts([_attachment()], _FakeSession(payload=_make_image("JPEG")))
    assert len(parts) == 1
    ctype, data = _part_image(parts[0])
    assert ctype == "data:image/jpeg"
    assert _decode(data) == ("JPEG", (16, 16))


@pytest.mark.asyncio
async def test_webp_attachment_arrives_as_png_for_the_llm():
    parts = await _parts([_attachment()], _FakeSession(payload=_make_image("WEBP")))
    assert len(parts) == 1
    ctype, data = _part_image(parts[0])
    assert ctype == "data:image/png"
    assert _decode(data) == ("PNG", (16, 16))


@pytest.mark.asyncio
async def test_heic_attachment_arrives_as_png_for_the_llm():
    parts = await _parts([_attachment(content_type="image/heic")],
                         _FakeSession(payload=_make_image("HEIF")))
    assert len(parts) == 1
    ctype, data = _part_image(parts[0])
    assert ctype == "data:image/png"
    assert _decode(data) == ("PNG", (16, 16))


@pytest.mark.asyncio
async def test_undecodable_download_is_skipped():
    assert await _parts([_attachment()], _FakeSession(payload=b"not an image")) == []


@pytest.mark.asyncio
async def test_non_image_attachments_are_skipped():
    session = _FakeSession()
    parts = await _parts([
        _attachment(url="http://cdn.example/v.mp4", content_type="video/mp4"),
        _attachment(url="http://cdn.example/f.pdf", content_type="application/pdf"),
        _attachment(content_type=None),
    ], session)
    assert parts == []
    assert session.calls == []


@pytest.mark.asyncio
async def test_download_failure_is_skipped():
    assert await _parts([_attachment()], _FakeSession(payload=ConnectionError("down"))) == []


@pytest.mark.asyncio
async def test_non_200_response_is_skipped():
    assert await _parts([_attachment()], _FakeSession(status=404)) == []


@pytest.mark.asyncio
async def test_max_images_per_message_cap():
    attachments = [_attachment(url=f"http://cdn.example/{i}.png")
                   for i in range(MAX_IMAGES_PER_MESSAGE + 3)]
    parts = await _parts(attachments, _FakeSession(payload=_make_image("JPEG")))
    assert len(parts) == MAX_IMAGES_PER_MESSAGE


@pytest.mark.asyncio
async def test_mixed_good_and_bad_attachments_only_sends_valid_images():
    parts = await _parts([
        _attachment(url="http://cdn.example/a.jpg", content_type="image/jpeg"),
        _attachment(url="http://cdn.example/b.png", content_type="image/png"),
        _attachment(url="http://cdn.example/c.txt", content_type="text/plain"),
    ], _FakeSession(payload=_make_image("JPEG")))
    assert len(parts) == 2
    for part in parts:
        ctype, data = _part_image(part)
        assert ctype == "data:image/jpeg"
        _decode(data)


@pytest.mark.asyncio
async def test_a_second_build_takes_the_image_from_the_cache():
    session = _FakeSession(payload=_make_image("JPEG"))
    att = _attachment(att_id=5)
    first = await _parts([att], session)
    second = await _parts([att], session)
    assert session.calls == ["http://cdn.example/x.png"]
    # Byte-for-byte the same, so llama.cpp can still reuse the prefix.
    assert first == second


@pytest.mark.asyncio
async def test_an_expired_url_for_a_cached_attachment_is_still_a_hit():
    session = _FakeSession(payload=_make_image("JPEG"))
    await _parts([_attachment(url="http://cdn.example/x.png?ex=1", att_id=5)], session)
    parts = await _parts([_attachment(url="http://cdn.example/x.png?ex=2", att_id=5)],
                         _FakeSession(status=403))
    assert len(parts) == 1
    assert session.calls == ["http://cdn.example/x.png?ex=1"]


@pytest.mark.asyncio
async def test_a_failed_download_is_retried_on_the_next_build():
    att = _attachment(att_id=5)
    assert await _parts([att], _FakeSession(status=500)) == []
    session = _FakeSession(payload=_make_image("JPEG"))
    assert len(await _parts([att], session)) == 1
    assert len(session.calls) == 1


@pytest.mark.asyncio
async def test_concurrent_builds_share_one_download():
    gate = asyncio.Event()
    session = _FakeSession(payload=_make_image("JPEG"), gate=gate)
    att = _attachment(att_id=5)
    tasks = [asyncio.ensure_future(_parts([att], session)) for _ in range(3)]
    await asyncio.sleep(0.01)
    gate.set()
    results = await asyncio.gather(*tasks)
    assert len(session.calls) == 1
    assert results[0] == results[1] == results[2] != []


@pytest.mark.asyncio
async def test_at_most_four_downloads_run_at_once():
    from classes.message_handler import DOWNLOAD_CONCURRENCY
    gate = asyncio.Event()
    session = _FakeSession(payload=_make_image("JPEG"), gate=gate)
    messages = [_image_message([_attachment(url=f"http://cdn.example/{i}.png")], mid=100 + i)
                for i in range(DOWNLOAD_CONCURRENCY + 3)]
    with _use_session(session):
        tasks = [asyncio.ensure_future(_image_handler(m).image_parts(m)) for m in messages]
        await asyncio.sleep(0.05)
        assert session.active == DOWNLOAD_CONCURRENCY
        gate.set()
        results = await asyncio.gather(*tasks)
    assert session.peak == DOWNLOAD_CONCURRENCY == 4
    assert all(len(r) == 1 for r in results)


def _history_channel(messages, channel_id=99):
    ch = MagicMock()
    ch.id = channel_id

    def history(limit=100, before=None, after=None, oldest_first=None):
        found = [m for m in messages if m.id < before.id][-limit:]

        async def _gen():
            for m in (found if oldest_first else reversed(found)):
                yield m
        return _gen()
    ch.history = history
    for m in messages:
        m.channel = ch
    return ch


@pytest.mark.asyncio
async def test_images_are_downloaded_outside_the_channel_lock(monkeypatch):
    monkeypatch.setenv("MSG_HISTORY_MODE", "sliding")
    old = _image_message([_attachment(url="http://cdn.example/old.png")], mid=1)
    trigger = _image_message([_attachment(url="http://cdn.example/new.png")], mid=2)
    ch = _history_channel([old, trigger])
    held = []
    session = _FakeSession(payload=_make_image("JPEG"),
                           on_get=lambda url: held.append(mq.get_channel_lock(ch.id).locked()))
    handler = _image_handler(trigger)
    captured = {}

    class _FakeLLM:
        def __init__(self, messages, *args, **kwargs):
            captured["messages"] = messages
            self.reasoning = ""

        async def generate(self):
            return "ok"

    ch.send = AsyncMock()
    with _use_session(session), patch("classes.message_handler.TextLLMHandler", _FakeLLM), \
            patch("classes.message_handler.slot_context_tokens", return_value=None):
        await handler.handle_message()

    assert sorted(session.calls) == ["http://cdn.example/new.png", "http://cdn.example/old.png"]
    assert held == [False, False]
    images = [p for e in captured["messages"] if isinstance(e["content"], list)
              for p in e["content"] if p["type"] == "input_image"]
    assert len(images) == 2


@pytest.mark.asyncio
async def test_a_message_deleted_while_images_download_is_left_out(monkeypatch):
    monkeypatch.setenv("MSG_HISTORY_MODE", "sliding")
    old = _image_message([_attachment(url="http://cdn.example/old.png")], mid=1, content="old")
    kept = _image_message([], mid=2, content="kept")
    trigger = _image_message([], mid=3, content="now")
    ch = _history_channel([old, kept, trigger])
    session = _FakeSession(
        payload=_make_image("JPEG"),
        on_get=lambda url: attachment_cache.cache().invalidate_message(ch.id, 1, deleted=True))
    handler = _image_handler(trigger)
    with _use_session(session), patch("classes.message_handler.slot_context_tokens", return_value=None):
        await handler.build_messages()

    texts = [e["content"] for e in handler.messages]
    assert texts == ["Message from 'alice': kept", "Message from 'alice': now"]
    # ...and the late download did not put the deleted image back.
    assert len(attachment_cache.cache()) == 0


@pytest.mark.asyncio
async def test_no_attachments_stage_without_images(monkeypatch):
    monkeypatch.setenv("MSG_HISTORY_MODE", "sliding")
    trigger = _image_message([], mid=3, content="now")
    _history_channel([trigger])
    handler = _image_handler(trigger)
    with patch("classes.message_handler.observe_stage") as observe, \
            patch("classes.message_handler.slot_context_tokens", return_value=None):
        await handler.build_messages()
    assert "attachments" not in [c.args[0] for c in observe.call_args_list]


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

    async def _select():
        pass

    async def _prepare():
        set_messages(handler)
    handler.select_messages = _select
    handler.prepare_messages = _prepare
    return handler


@pytest.mark.asyncio
async def test_handle_message_scoped_lock_generates_concurrently_sends_serially():
    """Two same-channel messages handled on two worker tasks: their LLM runs
    overlap (the lock does NOT cover the LLM phase — this is what frees the
    bot to answer in the same channel during a brain call), but their
    chunked sends are serialized by the lock and never interleave.

    With the old FULL lock (build + LLM + send) message 2's generation would
    only start after message 1's whole handle finished, so the concurrency
    assertion below discriminates against a regression to it."""

    class _FakeLLM:
        gen_events = []  # (msg_id, phase, monotonic)

        def __init__(self, messages, guild_id, original_message, client=None, unprompted=False):
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

    with patch("classes.message_handler.TextLLMHandler", _FakeLLM):
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
        def __init__(self, messages, guild_id, original_message, client=None, unprompted=False):
            captured["prompt"] = messages
            self.reasoning = ""

        async def generate(self):
            return "ok"

    handler = _handled(
        _handler(), _llm_message(11, channel_id),
        lambda h: setattr(h, "messages", [{"role": "user", "content": "is it done yet?"}]),
    )

    # Older than HINT_AFTER_SECONDS: a quick brain call never shows.
    assert mq.register_task_run(channel_id, "🧠 brain", "compute pi", run_key="t1",
                                started=time.monotonic() - 60)
    try:
        with patch("classes.message_handler.TextLLMHandler", _FakeLLM):
            await handler.handle_message()
    finally:
        mq.unregister_task_run(channel_id, run_key="t1")

    prompt = captured["prompt"]
    assert len(prompt) == 2  # the user message + the appended hint
    assert prompt[1]["role"] == "user"
    assert prompt[1]["content"].startswith("Note:")
    assert "🧠 brain" in prompt[1]["content"]
    assert "compute pi" in prompt[1]["content"]


@pytest.mark.asyncio
async def test_handle_message_no_hint_when_channel_idle():
    channel_id = 201
    captured = {}

    class _FakeLLM:
        def __init__(self, messages, guild_id, original_message, client=None, unprompted=False):
            captured["prompt"] = messages
            self.reasoning = ""

        async def generate(self):
            return "ok"

    handler = _handled(
        _handler(), _llm_message(12, channel_id),
        lambda h: setattr(h, "messages", [{"role": "user", "content": "hello"}]),
    )

    with patch("classes.message_handler.TextLLMHandler", _FakeLLM):
        await handler.handle_message()

    assert len(captured["prompt"]) == 1  # nothing appended
    assert captured["prompt"][0]["content"] == "hello"


@pytest.mark.asyncio
async def test_handle_message_passes_client_to_text_llm_handler():
    # Regression: TextLLMHandler needs the discord.Client (forwarded to tool
    # context as "discord_client") so the brain's ask_user tool can
    # call client.wait_for() — handle_message must not silently drop it.
    captured = {}

    class _FakeLLM:
        def __init__(self, messages, guild_id, original_message, client=None, unprompted=False):
            captured["client"] = client
            self.reasoning = ""

        async def generate(self):
            return "ok"

    handler = _handled(
        _handler(), _llm_message(50),
        lambda h: setattr(h, "messages", [{"role": "user", "content": "hi"}]),
    )

    with patch("classes.message_handler.TextLLMHandler", _FakeLLM):
        await handler.handle_message()

    assert captured["client"] is handler.client


@pytest.mark.asyncio
@pytest.mark.parametrize("marked", [True, False])
async def test_handle_message_passes_the_unprompted_flag(marked):
    # A message the reply chance let in (main.on_message marks it) only
    # talks, with no brain; the flag is popped, so it is used once.
    from classes import message_queue
    captured = {}

    class _FakeLLM:
        def __init__(self, messages, guild_id, original_message, client=None, unprompted=False):
            captured["unprompted"] = unprompted
            self.reasoning = ""

        async def generate(self):
            return "ok"

    message = _llm_message(51)
    if marked:
        message_queue.mark_unprompted(message.id)
    handler = _handled(_handler(), message,
                       lambda h: setattr(h, "messages", [{"role": "user", "content": "hi"}]))
    with patch("classes.message_handler.TextLLMHandler", _FakeLLM):
        await handler.handle_message()
    assert captured["unprompted"] is marked
    assert not message_queue.pop_unprompted(message.id)


@pytest.mark.asyncio
async def test_handle_message_replies_in_the_channel_it_was_asked_in():
    # The brain works inline: no thread, so the reply goes where it was asked.
    class _FakeLLM:
        def __init__(self, messages, guild_id, original_message, client=None, unprompted=False):
            self.reasoning = ""

        async def generate(self):
            return "the answer"

    handler = _handled(
        _handler(), _llm_message(62),
        lambda h: setattr(h, "messages", [{"role": "user", "content": "hi"}]),
    )
    handler.message.channel.send = AsyncMock()

    with patch("classes.message_handler.TextLLMHandler", _FakeLLM):
        await handler.handle_message()

    handler.message.channel.send.assert_awaited_once_with("the answer")


# --------------------------- reasoning follow-up (SHOW_THINKING) --------------------------

def _reasoning_llm(reasoning, answer="the answer"):
    """A TextLLMHandler stand-in that reports reasoning the way the real one
    does: as an attribute set during generate(), NOT as think tags inside
    the returned answer (our llama.cpp server strips those out into
    reasoning_content, so the answer text never carries them)."""

    class _FakeLLM:
        def __init__(self, messages, guild_id, original_message, client=None, unprompted=False):
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
    with patch("classes.message_handler.TextLLMHandler", fake_llm):
        await handler.handle_message()
    return handler


@pytest.mark.asyncio
async def test_handle_message_sends_out_of_band_reasoning_as_spoiler():
    """Regression: the reasoning arrives on the handler, not in the answer
    text. Reading it off the response string instead (as extract_thinking
    did) silently sent nothing at all."""
    sends = []
    with patch("classes.message_handler.SHOW_THINKING", True):
        await _run_with_llm(_reasoning_llm("i reasoned about it"), 300, sends)

    assert sends[0] == "the answer"
    assert sends[1].startswith("-# Reasoning")
    assert "i reasoned about it" in sends[2]
    assert sends[2].startswith("||```\n") and sends[2].endswith("\n```||")


@pytest.mark.asyncio
async def test_handle_message_sends_no_reasoning_when_show_thinking_off():
    sends = []
    with patch("classes.message_handler.SHOW_THINKING", False):
        await _run_with_llm(_reasoning_llm("i reasoned about it"), 301, sends)

    assert sends == ["the answer"]


@pytest.mark.asyncio
async def test_handle_message_sends_no_reasoning_when_model_did_not_think():
    sends = []
    with patch("classes.message_handler.SHOW_THINKING", True):
        await _run_with_llm(_reasoning_llm(""), 302, sends)

    assert sends == ["the answer"]


# --------------------------- TextLLMHandler.generate reasoning capture --------------------------
#
# The other half of the pipeline: generate() must pull the reasoning off the
# run result, because it is NOT in the answer string it returns.

def _generate_handler(run_result):
    from classes.text_llm_handler import TextLLMHandler

    handler = TextLLMHandler.__new__(TextLLMHandler)  # __init__ needs Redis
    handler.messages = [{"role": "user", "content": "hi"}]
    handler.guild_id = 1
    handler.reasoning = ""
    handler.client = MagicMock()
    handler.system = "a bot"
    handler.agent = MagicMock()
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
    with patch("classes.text_llm_handler.Runner") as runner, \
         patch("classes.text_llm_handler.get_current_datetime", AsyncMock(return_value="now")):
        runner.run = AsyncMock(return_value=run_result)
        answer = await handler.generate()
    return answer, handler.reasoning


@pytest.mark.asyncio
async def test_generate_captures_reasoning_from_run_items():
    """The llama.cpp shape: reasoning_content becomes its own run item and
    the answer text is clean, so the run items are the only source."""
    # Sibling test modules are imported by their BARE name: pytest puts
    # core/tests (the basedir of the file being collected) on sys.path, so
    # this is the same module object pytest itself registered. A dotted
    # repo-root-relative name only resolves when the root happens to be on
    # the path -- true under `python -m pytest`, false under CI's bare
    # `pytest` (see the Testing section of AGENTS.md).
    from response_filter_tests import _reasoning_item

    answer, reasoning = await _generate(
        _run_result("the clean answer", [_reasoning_item(summary=["thought hard"])])
    )
    assert answer == "the clean answer"
    assert reasoning == "thought hard"


@pytest.mark.asyncio
async def test_generate_falls_back_to_inline_think_tags():
    """A server on --reasoning-format none leaves the reasoning inline and
    emits no reasoning item; the tag path still has to work."""
    from response_filter_tests import _think_open, _think_close

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
    from response_filter_tests import _reasoning_item
    from agents import MaxTurnsExceeded

    handler = _generate_handler(None)
    exc = MaxTurnsExceeded("Max turns (20) exceeded")
    exc.run_data = MagicMock()
    exc.run_data.new_items = [_reasoning_item(summary=["i kept calling the brain"])]

    with patch("classes.text_llm_handler.Runner") as runner, \
         patch("classes.text_llm_handler.get_current_datetime", AsyncMock(return_value="now")), \
         patch("classes.text_llm_handler.inc_llm_error", MagicMock()):
        runner.run = AsyncMock(side_effect=exc)
        answer = await handler.generate()

    assert answer == "Error"  # sentinel unchanged
    assert handler.reasoning == "i kept calling the brain"


@pytest.mark.asyncio
async def test_generate_survives_a_failure_carrying_no_run_data():
    from agents import MaxTurnsExceeded

    handler = _generate_handler(None)
    with patch("classes.text_llm_handler.Runner") as runner, \
         patch("classes.text_llm_handler.get_current_datetime", AsyncMock(return_value="now")), \
         patch("classes.text_llm_handler.inc_llm_error", MagicMock()):
        runner.run = AsyncMock(side_effect=MaxTurnsExceeded("boom"))
        answer = await handler.generate()

    assert answer == "Error"
    assert handler.reasoning == ""


@pytest.mark.asyncio
async def test_generate_passes_an_explicit_max_turns():
    """PERSONA_MAX_TURNS, not the SDK's own default of 10: the persona has
    one tool, so a reply is at most ask the brain, maybe again, answer."""
    from classes.text_llm_handler import persona_max_turns

    handler = _generate_handler(None)
    with patch("classes.text_llm_handler.Runner") as runner, \
         patch("classes.text_llm_handler.get_current_datetime", AsyncMock(return_value="now")):
        runner.run = AsyncMock(return_value=_run_result("ok"))
        await handler.generate()

    assert runner.run.await_args.kwargs["max_turns"] == persona_max_turns() == 4


@pytest.mark.asyncio
async def test_get_client_bounds_each_model_call(monkeypatch):
    """The client's own timeout is per-read, so a server that keeps the
    connection warm while it works was never bounded. The wall-clock bound is
    PERSONA_CALL_TIMEOUT_SECONDS, applied by LanedModel after the lane wait;
    the SDK's own (ModelSettings.timeout) stays unset, since it would count
    the lane wait too."""
    from classes import text_llm_handler
    from classes.laned_model import LanedModel
    from classes.text_llm_handler import TextLLMHandler

    monkeypatch.setenv("PERSONA_CALL_TIMEOUT_SECONDS", "45")
    monkeypatch.setattr(text_llm_handler, "_main_model_client", None)
    handler = TextLLMHandler.__new__(TextLLMHandler)  # __init__ needs Redis
    handler.system = "a bot"
    handler.options = {"temperature": 1.0}
    await handler.get_client()

    assert isinstance(handler.agent.model, LanedModel)
    assert handler.agent.model.call_timeout == 45.0
    assert handler.agent.model.lane == "chat"
    assert handler.agent.model_settings.timeout is None
    monkeypatch.setattr(text_llm_handler, "_main_model_client", None)


@pytest.mark.asyncio
async def test_handle_message_sends_reasoning_even_when_the_run_failed():
    sends = []

    class _FailingLLM:
        def __init__(self, messages, guild_id, original_message, client=None, unprompted=False):
            self.reasoning = ""

        async def generate(self):
            self.reasoning = "i was thinking when it broke"
            return "Error"

    with patch("classes.message_handler.SHOW_THINKING", True):
        handler = await _run_with_llm(_FailingLLM, 310, sends)

    handler.message.add_reaction.assert_awaited_with('❌')
    assert sends[0].startswith("-# Reasoning")
    assert "i was thinking when it broke" in sends[1]


@pytest.mark.asyncio
async def test_handle_message_failed_run_with_no_reasoning_just_reacts():
    sends = []

    class _FailingLLM:
        def __init__(self, messages, guild_id, original_message, client=None, unprompted=False):
            self.reasoning = ""

        async def generate(self):
            return "Error"

    with patch("classes.message_handler.SHOW_THINKING", True):
        handler = await _run_with_llm(_FailingLLM, 311, sends)

    handler.message.add_reaction.assert_awaited_with('❌')
    assert sends == []


@pytest.mark.asyncio
async def test_handle_message_prewarms_the_servers_workspace():
    # The persona may hand the job to the brain, which needs the workspace:
    # it starts while the persona thinks.
    class _FakeLLM:
        def __init__(self, messages, guild_id, original_message, client=None, unprompted=False):
            self.reasoning = ""

        async def generate(self):
            return "ok"

    handler = _handled(
        _handler(), _llm_message(70),
        lambda h: setattr(h, "messages", [{"role": "user", "content": "hi"}]),
    )
    handler.message.channel.send = AsyncMock()
    manager = MagicMock()
    with patch("classes.message_handler.TextLLMHandler", _FakeLLM), \
         patch("classes.workspace_manager.workspaces", return_value=manager):
        await handler.handle_message()
    manager.prewarm.assert_called_once_with(handler.message.guild.id)


@pytest.mark.asyncio
async def test_handle_message_does_not_prewarm_for_an_unprompted_reply():
    # A reply nobody asked for has no brain, so no workspace either.
    from classes import message_queue

    class _FakeLLM:
        def __init__(self, messages, guild_id, original_message, client=None, unprompted=False):
            self.reasoning = ""

        async def generate(self):
            return "ok"

    message = _llm_message(71)
    message_queue.mark_unprompted(message.id)
    handler = _handled(_handler(), message,
                       lambda h: setattr(h, "messages", [{"role": "user", "content": "hi"}]))
    handler.message.channel.send = AsyncMock()
    manager = MagicMock()
    with patch("classes.message_handler.TextLLMHandler", _FakeLLM),          patch("classes.workspace_manager.workspaces", return_value=manager):
        await handler.handle_message()
    manager.prewarm.assert_not_called()

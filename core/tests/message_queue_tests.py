"""
Unit tests for the message queue concurrency model:
  - core/classes/message_queue.py — pure module: env-var sizing (WORKER_COUNT
    / QUEUE_MAX_SIZE), the bounded queue factory, the per-channel lock
    registry (no discord/agents imports needed).
  - the queue wiring in core/main.py — the bounded drop policy in
    on_message() and the worker-pool + per-channel-lock semantics of
    process_messages() (imports main via the _import_main() helper).

Run from the repo root:
    PYTHONPATH=$(pwd) pytest core/tests/message_queue_tests.py
"""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Same dual-import setup as the other test files: the app imports classes.*
# (cwd = core/) while the tests import classes.*.

import classes.message_queue as prod_mq  # production-style import path
from classes import message_queue as mq
from classes import metrics


# ---------------------------------------------------------------------------
# core/classes/message_queue.py (pure)
# ---------------------------------------------------------------------------

def test_no_double_module_under_both_import_paths():
    # Both import paths must resolve to the same module object (aliasing at
    # the top of message_queue.py) — otherwise the per-channel lock registry
    # would exist twice and locks would not be shared.
    assert prod_mq is mq


def test_worker_count_env(monkeypatch):
    monkeypatch.delenv("WORKER_COUNT", raising=False)
    assert mq.worker_count() == mq.DEFAULT_WORKER_COUNT == 2
    for value, expected in [("1", 1), ("4", 4), (" 3 ", 3),
                            ("0", 2), ("-2", 2), ("abc", 2), ("", 2)]:
        monkeypatch.setenv("WORKER_COUNT", value)
        assert mq.worker_count() == expected, value


def test_queue_max_size_env(monkeypatch):
    monkeypatch.delenv("QUEUE_MAX_SIZE", raising=False)
    assert mq.queue_max_size() == mq.DEFAULT_QUEUE_MAX_SIZE == 10
    for value, expected in [("1", 1), ("25", 25), (" 7 ", 7),
                            ("0", 10), ("-2", 10), ("abc", 10), ("", 10)]:
        monkeypatch.setenv("QUEUE_MAX_SIZE", value)
        assert mq.queue_max_size() == expected, value


def test_make_message_queue_is_bounded(monkeypatch):
    monkeypatch.setenv("QUEUE_MAX_SIZE", "3")
    q = mq.make_message_queue()
    assert q.maxsize == 3
    assert not q.full()
    for _ in range(3):
        q.put_nowait(None)
    assert q.full()
    with pytest.raises(asyncio.QueueFull):
        q.put_nowait(None)


def test_channel_lock_identity():
    # Same channel id always yields the same lock; different ids differ.
    assert mq.get_channel_lock(111) is mq.get_channel_lock(111)
    assert mq.get_channel_lock(111) is not mq.get_channel_lock(222)


# ---------------------------------------------------------------------------
# core/main.py queue wiring (imported without starting the bot)
# ---------------------------------------------------------------------------

def _import_main():
    """Import main exactly once, without starting the bot.

    Same pattern as image_generation_tests.py: main.py has a module-level
    client.run() is guarded behind `if __name__ == "__main__"` and the Redis
    client is built lazily, so a plain import neither starts the bot nor
    requires any environment.
    """
    if "main" in sys.modules:
        return sys.modules["main"]
    import main as m
    return m


def _message(content="hi", guild_id=42, channel_id=77, user_id=777,
             msg_id=123456, mentions=None):
    msg = MagicMock()
    msg.id = msg_id
    msg.content = content
    msg.embeds = []
    msg.attachments = []
    msg.guild = MagicMock()
    msg.guild.id = guild_id
    msg.author = MagicMock()
    msg.author.id = user_id
    channel = MagicMock()
    channel.id = channel_id
    # channel.typing() is an async context manager in process_messages()
    typing_cm = MagicMock()
    typing_cm.__aenter__ = AsyncMock(return_value=None)
    typing_cm.__aexit__ = AsyncMock(return_value=False)
    channel.typing = MagicMock(return_value=typing_cm)
    msg.channel = channel
    msg.mentions = [] if mentions is None else mentions
    msg.reply = AsyncMock()
    return msg


def _counter(metric, **labels):
    return metric.labels(**labels)._value.get()


@pytest.mark.asyncio
async def test_on_message_enqueues_when_queue_has_room():
    main_mod = _import_main()
    bot = MagicMock()
    queue = asyncio.Queue(maxsize=2)
    received_before = _counter(metrics.messages_received_total,
                               guild_id="42")
    drops_before = _counter(metrics.queue_drops_total, guild_id="42")

    with patch.object(main_mod, "client", bot), \
         patch.object(main_mod, "message_queue", queue):
        msg = _message(mentions=[bot.user])
        await main_mod.on_message(msg)

    assert queue.qsize() == 1
    assert queue.get_nowait() is msg
    msg.reply.assert_not_awaited()
    assert _counter(metrics.messages_received_total,
                    guild_id="42") == received_before + 1
    assert _counter(metrics.queue_drops_total, guild_id="42") == drops_before


@pytest.mark.asyncio
async def test_on_message_drops_mention_with_busy_reply_when_full():
    main_mod = _import_main()
    bot = MagicMock()
    queue = asyncio.Queue(maxsize=1)
    queue.put_nowait(object())  # already full
    received_before = _counter(metrics.messages_received_total,
                               guild_id="42")
    drops_before = _counter(metrics.queue_drops_total, guild_id="42")

    with patch.object(main_mod, "client", bot), \
         patch.object(main_mod, "message_queue", queue):
        msg = _message(mentions=[bot.user])
        await main_mod.on_message(msg)

    # dropped, not enqueued
    assert queue.qsize() == 1
    assert _counter(metrics.queue_drops_total, guild_id="42") == drops_before + 1
    assert _counter(metrics.messages_received_total,
                    guild_id="42") == received_before
    # a mention is not silently ignored: short busy reply
    msg.reply.assert_awaited_once()
    assert "working through my queue" in msg.reply.await_args.args[0]


@pytest.mark.asyncio
async def test_on_message_drops_unmentioned_message_silently_when_full():
    main_mod = _import_main()
    bot = MagicMock()
    queue = asyncio.Queue(maxsize=1)
    queue.put_nowait(object())
    drops_before = _counter(metrics.queue_drops_total, guild_id="42")

    # Not mentioned: patch the chance roll so the filter passes deterministically
    with patch.object(main_mod, "client", bot), \
         patch.object(main_mod, "message_queue", queue), \
         patch.object(main_mod.config, "get_setting", return_value=5), \
         patch.object(main_mod.random, "uniform", return_value=1.0):
        msg = _message()
        await main_mod.on_message(msg)

    assert queue.qsize() == 1
    assert _counter(metrics.queue_drops_total, guild_id="42") == drops_before + 1
    msg.reply.assert_not_awaited()  # random-chance messages drop quietly


@pytest.mark.asyncio
async def test_process_messages_worker_pool_runs_messages_concurrently():
    """Run the REAL main.process_messages() with two workers (as on_ready
    starts them): the worker loop holds NO channel lock of its own (the
    scoped lock lives inside MessageHandler.handle_message — covered by
    test_handle_message_scoped_lock_* in message_handler_tests.py), so with
    a free worker a second message from the SAME channel starts while the
    first is still in flight, and other channels are unaffected."""
    main_mod = _import_main()
    events = []

    class FakeHandler:
        def __init__(self, message, client):
            self.message = message

        async def handle_message(self):
            events.append(f"{self.message.id}:start")
            # message 1 is slow (e.g. a tool call), message 2 a bit faster
            # (both channel 1), channel 2 fast
            delay = {1: 0.5, 2: 0.4, 3: 0.05}[self.message.id]
            await asyncio.sleep(delay)
            events.append(f"{self.message.id}:done")

    a1 = _message(msg_id=1, channel_id=1)
    a2 = _message(msg_id=2, channel_id=1)
    b1 = _message(msg_id=3, channel_id=2)

    queue = asyncio.Queue()
    with patch.object(main_mod, "message_queue", queue), \
         patch.object(main_mod, "MessageHandler", FakeHandler):
        workers = [asyncio.create_task(main_mod.process_messages())
                   for _ in range(2)]
        try:
            queue.put_nowait(a1)
            queue.put_nowait(a2)
            queue.put_nowait(b1)
            await queue.join()
        finally:
            for worker in workers:
                worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)

    assert events.index("2:start") < events.index("1:done"), (
        "the worker loop must not serialize same-channel messages — "
        "a free worker starts A2 while A1 is still in flight "
        "(scoped-lock ordering of the SEND is guaranteed inside "
        "MessageHandler.handle_message)"
    )
    assert events.index("3:done") < events.index("1:done"), (
        "channel B must not wait for the slow channel-A message "
        "(different channels run concurrently)"
    )

# ---------------------- on_message: routing into a running sandbox ----------------------

@pytest.fixture
def claimed_thread():
    """A sandbox run claimed in channel 77 (the _message() default)."""
    from classes import sandbox_thread_inbox as inbox
    inbox._RUNS.clear()
    ledger = inbox.claim_run(77, requester_id=777)
    ledger.state = inbox.RUNNING
    yield ledger
    inbox._RUNS.clear()


def _thread_message(content="make it blue", **kwargs):
    msg = _message(content=content, **kwargs)
    msg.author.bot = False
    msg.author.display_name = "ana"
    msg.reference = None
    msg.add_reaction = AsyncMock()
    return msg


@pytest.mark.asyncio
async def test_a_message_in_a_running_sandbox_thread_goes_to_the_sandbox(claimed_thread):
    main_mod = _import_main()
    queue = asyncio.Queue(maxsize=2)
    with patch.object(main_mod, "client", MagicMock()), \
         patch.object(main_mod, "message_queue", queue):
        msg = _thread_message()
        await main_mod.on_message(msg)

    assert queue.qsize() == 0  # not the outer LLM's
    [event] = claimed_thread.events
    assert (event.text, event.author_id, event.is_requester) == ("make it blue", 777, True)
    msg.add_reaction.assert_awaited_once_with("📨")
    msg.reply.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_redelivered_message_is_recorded_once(claimed_thread):
    main_mod = _import_main()
    with patch.object(main_mod, "client", MagicMock()):
        msg = _thread_message()
        await main_mod.on_message(msg)
        await main_mod.on_message(msg)

    assert len(claimed_thread.events) == 1
    msg.add_reaction.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_message_while_a_command_runs_gets_one_receipt_notice(claimed_thread):
    import time as _time
    main_mod = _import_main()
    claimed_thread.busy_since = _time.monotonic()
    with patch.object(main_mod, "client", MagicMock()):
        first, second = _thread_message(msg_id=1), _thread_message("and bigger", msg_id=2)
        await main_mod.on_message(first)
        await main_mod.on_message(second)

    first.reply.assert_awaited_once()
    assert "review this before my next step" in first.reply.await_args.args[0]
    second.reply.assert_not_awaited()  # coalesced


@pytest.mark.asyncio
async def test_a_message_as_the_run_finishes_is_told_it_was_not_applied(claimed_thread):
    main_mod = _import_main()
    claimed_thread.finalize()
    with patch.object(main_mod, "client", MagicMock()):
        msg = _thread_message("and a csv")
        await main_mod.on_message(msg)

    msg.add_reaction.assert_awaited_once_with("⏳")
    assert "hasn't been applied" in msg.reply.await_args.args[0]
    assert claimed_thread.follow_ups[0].text == "and a csv"


@pytest.mark.asyncio
async def test_an_attachment_only_message_is_not_acknowledged_as_read(claimed_thread):
    main_mod = _import_main()
    with patch.object(main_mod, "client", MagicMock()):
        msg = _thread_message("")
        msg.attachments = [MagicMock()]
        await main_mod.on_message(msg)

    msg.add_reaction.assert_awaited_once_with("🚫")
    assert "can't open attachments" in msg.reply.await_args.args[0]
    assert claimed_thread.events == []


@pytest.mark.asyncio
async def test_a_reply_reference_is_passed_along(claimed_thread):
    main_mod = _import_main()
    with patch.object(main_mod, "client", MagicMock()):
        msg = _thread_message("png")
        msg.reference = MagicMock(message_id=555)
        await main_mod.on_message(msg)

    assert claimed_thread.events[0].reply_to == 555

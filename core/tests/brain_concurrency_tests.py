"""Tests for 3c: the unprompted flag and limits (core/classes/brain_policy.py,
message_queue.mark_unprompted), the brain's slots (BRAIN_MAX_CONCURRENT /
BRAIN_QUEUE_WAIT in core/classes/brain.py) and the elastic worker pool
(core/classes/message_queue.py, wired in core/main.py).

Run from the repo root:
    pytest core/tests/brain_concurrency_tests.py
"""
import asyncio
import contextvars
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from classes import brain, brain_policy, brain_result, llm_lanes, message_queue as mq


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    mq.reset_pool()
    monkeypatch.setattr(brain, "_slots", None)
    yield
    mq.reset_pool()


# --------------------------- limits ---------------------------

def test_unprompted_replies_get_the_tighter_limits():
    assert brain_policy.limits_for(unprompted=False, env={}) == (100, 1800)
    assert brain_policy.limits_for(unprompted=True, env={}) == (40, 180)
    env = {"BRAIN_UNPROMPTED_MAX_TURNS": "10", "BRAIN_UNPROMPTED_TIMEOUT": "60",
           "BRAIN_MAX_TURNS": "50", "BRAIN_TIMEOUT": "900"}
    assert brain_policy.limits_for(unprompted=True, env=env) == (10, 60)
    assert brain_policy.limits_for(unprompted=False, env=env) == (50, 900)


def test_question_and_slot_settings():
    assert brain_policy.ask_timeout({}) == 120
    assert brain_policy.max_concurrent({}) == 3
    assert brain_policy.queue_wait({}) == 120
    env = {"BRAIN_ASK_TIMEOUT": "30", "BRAIN_MAX_CONCURRENT": "1", "BRAIN_QUEUE_WAIT": "5"}
    assert (brain_policy.ask_timeout(env), brain_policy.max_concurrent(env),
            brain_policy.queue_wait(env)) == (30, 1, 5)
    assert brain_policy.max_concurrent({"BRAIN_MAX_CONCURRENT": "0"}) == 3


def test_unprompted_marks_are_popped_once():
    mq.mark_unprompted(1)
    assert mq.pop_unprompted(1)
    assert not mq.pop_unprompted(1)
    assert not mq.pop_unprompted(2)


def test_unprompted_marks_are_capped():
    for i in range(mq.MAX_ENQUEUED_TRACKED + 5):
        mq.mark_unprompted(10_000 + i)
    assert not mq.pop_unprompted(10_000)  # the oldest went first
    assert mq.pop_unprompted(10_000 + mq.MAX_ENQUEUED_TRACKED + 4)


@pytest.mark.asyncio
async def test_an_unprompted_brain_call_runs_with_the_unprompted_limits(monkeypatch):
    from brain_tests import _brain, _ctx, _report_call
    from agents.testing import ModelStep, ScriptedModel

    seen = {}
    real_wait_for = asyncio.wait_for

    async def _wait_for(awaitable, timeout):
        seen.setdefault("timeouts", []).append(timeout)
        return await real_wait_for(awaitable, timeout)

    model = ScriptedModel([ModelStep(output=[_report_call()])])
    with _brain(model), patch.object(brain.asyncio, "wait_for", _wait_for), \
         patch.object(brain.Runner, "run", wraps=brain.Runner.run) as run:
        report = await brain.run_brain(_ctx(unprompted=True), "look it up")
    assert report.status == brain_result.DONE
    assert run.call_args.kwargs["max_turns"] == 40
    assert 180 in seen["timeouts"]


# MessageHandler popping the flag: message_handler_tests.py.


# --------------------------- brain slots ---------------------------

@pytest.mark.asyncio
async def test_a_call_that_finds_every_slot_busy_is_told_the_bot_is_busy(monkeypatch):
    monkeypatch.setenv("BRAIN_MAX_CONCURRENT", "1")
    monkeypatch.setenv("BRAIN_QUEUE_WAIT", "1")
    monkeypatch.setattr(brain_policy, "queue_wait", lambda env=None: 0.05)
    release = asyncio.Event()

    async def _slow(ctx, request, files, hooks_box):
        await release.wait()
        return brain_result.BrainReport(status=brain_result.DONE)

    with patch.object(brain, "_run_brain", _slow):
        first = asyncio.create_task(brain.run_brain({"voice": False}, "a"))
        await asyncio.sleep(0)
        busy = await brain.run_brain({"voice": False}, "b")
        assert busy.status == brain_result.FAILED
        assert "busy" in busy.note
        release.set()
        assert (await first).status == brain_result.DONE
    # The slot is free again.
    with patch.object(brain, "_run_brain",
                      AsyncMock(return_value=brain_result.BrainReport(status=brain_result.DONE))):
        assert (await brain.run_brain({"voice": False}, "c")).status == brain_result.DONE


@pytest.mark.asyncio
async def test_a_brain_call_on_a_worker_starts_a_spare_and_a_voice_turn_does_not(monkeypatch):
    monkeypatch.setenv("QUEUE_WORKERS", "1")
    spawned = []

    def _spawn():
        mq.worker_started()
        spawned.append(1)

    mq.worker_started()  # the base worker
    mq.set_spawner(_spawn)
    seen = {}

    async def _run(ctx, request, files, hooks_box):
        seen.setdefault("targets", []).append(mq.target_workers())
        return brain_result.BrainReport(status=brain_result.DONE)

    with patch.object(brain, "_run_brain", _run):
        await brain.run_brain({"voice": False}, "chat")
        await brain.run_brain({"voice": True}, "voice")
    assert spawned == [1]
    assert seen["targets"] == [2, 1]
    # Back to target: the next worker to come round exits.
    assert mq.target_workers() == 1
    assert mq.claim_exit()
    assert not mq.claim_exit()


# --------------------------- pool policy (pure) ---------------------------

def test_pool_policy_spawns_only_when_short_and_shrinks_back(monkeypatch):
    monkeypatch.setenv("QUEUE_WORKERS", "2")
    spawned = []

    def _spawn():
        mq.worker_started()
        spawned.append(1)

    mq.set_spawner(_spawn)
    mq.worker_started()
    mq.worker_started()
    assert not mq.claim_exit()
    assert mq.brain_call_started()  # 3 needed, 2 running
    assert mq.pool_workers() == 3
    # A spare left idle from an earlier call covers the next one.
    mq.brain_call_finished()
    assert not mq.brain_call_started()
    assert spawned == [1]
    mq.brain_call_finished()
    assert mq.claim_exit()
    assert mq.pool_workers() == 2
    assert not mq.claim_exit()


# --------------------------- main.py wiring ---------------------------

def _main():
    if "main" in sys.modules:
        return sys.modules["main"]
    import main
    return main


@pytest.mark.asyncio
async def test_a_spare_worker_starts_in_a_fresh_context(monkeypatch):
    # Started from inside a brain call, a spare must not inherit the brain's
    # lane, or every chat reply it handled would run as brain work.
    main = _main()
    seen = {}

    async def _worker():
        seen["lane"] = llm_lanes.current_lane.get()

    client = MagicMock(background_tasks=[])
    with patch.object(main, "client", client), patch.object(main, "process_messages", _worker):
        token = llm_lanes.current_lane.set(llm_lanes.BRAIN)
        try:
            main.start_worker()
        finally:
            llm_lanes.current_lane.reset(token)
        await asyncio.gather(*client.background_tasks)
    assert seen["lane"] == llm_lanes.current_lane.get()  # the default, not BRAIN
    assert seen["lane"] != llm_lanes.BRAIN
    assert mq.pool_workers() == 1


@pytest.mark.asyncio
async def test_a_worker_above_target_exits_before_taking_an_item(monkeypatch):
    monkeypatch.setenv("QUEUE_WORKERS", "1")
    main = _main()
    queue = asyncio.Queue()
    mq.worker_started()
    mq.worker_started()  # one spare too many
    with patch.object(main, "message_queue", queue):
        await asyncio.wait_for(main.process_messages(), 1)
    assert mq.pool_workers() == 1


@pytest.mark.asyncio
async def test_a_cancelled_worker_is_counted_out(monkeypatch):
    monkeypatch.setenv("QUEUE_WORKERS", "1")
    main = _main()
    queue = asyncio.Queue()
    mq.worker_started()
    with patch.object(main, "message_queue", queue):
        task = asyncio.create_task(main.process_messages())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert mq.pool_workers() == 0


@pytest.mark.asyncio
async def test_on_message_marks_a_chance_reply_unprompted():
    main = _main()
    bot = MagicMock()
    queue = asyncio.Queue(maxsize=4)

    def _msg(msg_id, mentions):
        message = MagicMock(id=msg_id, content="hi", embeds=[], attachments=[], mentions=mentions)
        message.guild = MagicMock(id=42)
        message.author = MagicMock(id=7)
        message.channel = MagicMock(id=77)
        return message

    with patch.object(main, "client", bot), patch.object(main, "message_queue", queue), \
         patch.object(main, "remember_poll", AsyncMock()), \
         patch.object(main.automation_runner, "match_message", AsyncMock(return_value=False)), \
         patch.object(main, "should_handle_message", AsyncMock(return_value=True)):
        await main.on_message(_msg(880001, mentions=[]))
        await main.on_message(_msg(880002, mentions=[bot.user]))
    assert mq.pop_unprompted(880001)
    assert not mq.pop_unprompted(880002)
    mq.pop_enqueued(880001), mq.pop_enqueued(880002)

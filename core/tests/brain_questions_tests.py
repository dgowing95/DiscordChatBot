"""Tests for the brain's questions to the requester (3b): the waiter ledger
(core/classes/brain_inbox.py), the brain's ask_user tool
(core/classes/brain.py), the persona-voiced wording
(core/classes/persona_voice.py) and main.on_message routing the answer.

Run from the repo root:
    pytest core/tests/brain_questions_tests.py
"""
import asyncio
import json
import sys
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from classes import brain, brain_inbox, metrics, persona_voice


@pytest.fixture(autouse=True)
def _clean_inbox():
    brain_inbox.reset()
    yield
    brain_inbox.reset()


def _count(outcome):
    return metrics.brain_questions_total.labels(outcome=outcome)._value.get()


# --------------------------- brain_inbox (pure) ---------------------------

@pytest.mark.asyncio
async def test_inbox_delivers_to_the_waiter_and_closes_it():
    future = brain_inbox.ask(77, 5)
    assert brain_inbox.waiting(77, 5)
    assert not brain_inbox.waiting(77, 6)  # someone else
    assert not brain_inbox.waiting(78, 5)  # another channel
    assert brain_inbox.deliver(77, 5, "  bar chart  ") == brain_inbox.ACCEPTED
    assert future.result() == "bar chart"
    assert not brain_inbox.waiting(77, 5)
    assert brain_inbox.deliver(77, 5, "again") == brain_inbox.NO_WAITER


@pytest.mark.asyncio
async def test_inbox_refuses_an_empty_answer_and_keeps_waiting():
    future = brain_inbox.ask(77, 5)
    assert brain_inbox.deliver(77, 5, "   ") == brain_inbox.EMPTY
    assert brain_inbox.waiting(77, 5)
    assert not future.done()


@pytest.mark.asyncio
async def test_a_newer_question_replaces_the_older_one():
    old = brain_inbox.ask(77, 5)
    new = brain_inbox.ask(77, 5)
    assert old.result() is brain_inbox.REPLACED
    brain_inbox.deliver(77, 5, "line")
    assert new.result() == "line"


@pytest.mark.asyncio
async def test_cancel_removes_only_its_own_waiter():
    old = brain_inbox.ask(77, 5)
    new = brain_inbox.ask(77, 5)
    brain_inbox.cancel(77, 5, old)  # the replaced one cleaning up
    assert brain_inbox.waiting(77, 5)
    brain_inbox.cancel(77, 5, new)
    assert not brain_inbox.waiting(77, 5)
    assert new.cancelled()


# --------------------------- ask_user ---------------------------

def _ctx(**extra):
    channel = MagicMock(id=77)
    channel.send = AsyncMock()
    message = MagicMock()
    message.reply = AsyncMock()
    ctx = {"guild_id": 1, "user_id": 5, "channel": channel, "original_message": message,
           "automatic": False, "voice": False, "unprompted": False,
           "brain_deadline": time.monotonic() + 600}
    ctx.update(extra)
    return ctx


async def _ask(ctx, question="Bar or line chart?"):
    from agents.tool_context import ToolContext
    return await brain.ask_user.on_invoke_tool(
        ToolContext(context=ctx, tool_name="ask_user", tool_call_id="q1", tool_arguments="{}"),
        json.dumps({"question": question}))


@pytest.fixture
def plain_voice():
    """The question is posted as written (no side call)."""
    with patch.object(persona_voice, "voice_question",
                      AsyncMock(side_effect=lambda guild_id, q: q)) as voiced:
        yield voiced


@pytest.mark.asyncio
@pytest.mark.parametrize("extra", [
    {"automatic": True}, {"unprompted": True}, {"channel": None}, {"user_id": None},
    {"brain_questions": brain.MAX_QUESTIONS},
    # Too little of the call's time left to be worth waiting.
    {"brain_deadline": time.monotonic() + brain.ASK_MARGIN_SECONDS + 1},
])
async def test_no_one_to_ask_answers_at_once(extra, plain_voice):
    before = _count("skipped")
    ctx = _ctx(**extra)
    result = await _ask(ctx)
    assert result == brain.NO_ONE_TO_ASK
    assert _count("skipped") == before + 1
    ctx["original_message"].reply.assert_not_awaited()
    assert not brain_inbox.waiting(77, 5)


@pytest.mark.asyncio
async def test_the_question_is_posted_as_a_reply_and_the_answer_returned(plain_voice):
    before = _count("answered")
    ctx = _ctx()
    task = asyncio.create_task(_ask(ctx))
    for _ in range(20):
        await asyncio.sleep(0)
        if brain_inbox.waiting(77, 5):
            break
    ctx["original_message"].reply.assert_awaited_once()
    assert ctx["original_message"].reply.await_args.args[0] == "Bar or line chart?"
    assert brain_inbox.deliver(77, 5, "line please") == brain_inbox.ACCEPTED
    assert await task == "They answered: line please"
    assert ctx["brain_questions"] == 1
    assert _count("answered") == before + 1


@pytest.mark.asyncio
async def test_no_source_message_posts_with_a_mention(plain_voice):
    ctx = _ctx(original_message=None)
    task = asyncio.create_task(_ask(ctx))
    for _ in range(20):
        await asyncio.sleep(0)
        if brain_inbox.waiting(77, 5):
            break
    assert ctx["channel"].send.await_args.args[0] == "<@5> Bar or line chart?"
    brain_inbox.deliver(77, 5, "bar")
    assert await task == "They answered: bar"


@pytest.mark.asyncio
async def test_no_answer_in_time_says_to_assume_and_leaves_no_waiter(plain_voice, monkeypatch):
    monkeypatch.setenv("BRAIN_ASK_TIMEOUT", "1")
    monkeypatch.setattr(brain, "MIN_ASK_SECONDS", 0.01)
    before = _count("timeout")
    result = await _ask(_ctx())
    assert result.startswith("No answer within 1 seconds")
    assert not brain_inbox.waiting(77, 5)
    assert _count("timeout") == before + 1


@pytest.mark.asyncio
async def test_a_stopped_run_leaves_no_waiter_behind(plain_voice):
    # The call's own timeout cancels ask_user mid-wait; a waiter left behind
    # would swallow the requester's next message.
    task = asyncio.create_task(_ask(_ctx()))
    for _ in range(20):
        await asyncio.sleep(0)
        if brain_inbox.waiting(77, 5):
            break
    assert brain_inbox.waiting(77, 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not brain_inbox.waiting(77, 5)


@pytest.mark.asyncio
async def test_two_questions_in_one_turn_share_the_cap(plain_voice):
    ctx = _ctx(brain_questions=brain.MAX_QUESTIONS - 1)
    first = asyncio.create_task(_ask(ctx, "One?"))
    second = asyncio.create_task(_ask(ctx, "Two?"))
    for _ in range(20):
        await asyncio.sleep(0)
    assert await second == brain.NO_ONE_TO_ASK
    brain_inbox.deliver(77, 5, "yes")
    assert await first == "They answered: yes"


@pytest.mark.asyncio
async def test_a_replaced_question_tells_the_brain_to_assume(plain_voice):
    first = asyncio.create_task(_ask(_ctx(), "One?"))
    for _ in range(20):
        await asyncio.sleep(0)
    brain_inbox.ask(77, 5)  # another call by the same person asks too
    assert "newer question" in await first


@pytest.mark.asyncio
async def test_a_failed_post_asks_nothing(plain_voice):
    ctx = _ctx()
    ctx["original_message"].reply.side_effect = RuntimeError("no permission")
    assert await _ask(ctx) == brain.NO_ONE_TO_ASK
    assert not brain_inbox.waiting(77, 5)


@pytest.mark.asyncio
async def test_in_a_voice_call_the_question_is_also_spoken(plain_voice):
    from classes import voice_gate
    session = MagicMock()
    session.speak_line = AsyncMock()
    voice_gate.reset()
    voice_gate.enter(1, session)
    try:
        ctx = _ctx(voice=True, original_message=None)
        task = asyncio.create_task(_ask(ctx))
        for _ in range(20):
            await asyncio.sleep(0)
            if brain_inbox.waiting(77, 5):
                break
        session.speak_line.assert_awaited_once_with("Bar or line chart?")
        brain_inbox.deliver(77, 5, "bar")
        assert await task == "They answered: bar"
    finally:
        voice_gate.reset()


def test_ask_wait_is_cut_to_the_calls_deadline(monkeypatch):
    monkeypatch.setenv("BRAIN_ASK_TIMEOUT", "120")
    assert brain._ask_wait({}) == 120
    wait = brain._ask_wait({"brain_deadline": time.monotonic() + 100})
    assert 60 <= wait <= 100 - brain.ASK_MARGIN_SECONDS
    assert brain._ask_wait({"brain_deadline": time.monotonic() + 10}) is None


def test_ask_user_keys_are_taken_off_after_the_call():
    assert "brain_deadline" in brain._BRAIN_KEYS
    assert "brain_questions" in brain._BRAIN_KEYS


# --------------------------- persona_voice ---------------------------

@pytest.mark.asyncio
async def test_the_question_is_reworded_in_the_personas_voice():
    config = MagicMock()
    config.get_setting = AsyncMock(return_value="a grumpy pirate")
    with patch("classes.config_manager.configManager", return_value=config), \
         patch.object(persona_voice._llm, "complete",
                      AsyncMock(return_value="Arr, bar or line, matey?")) as complete:
        text = await persona_voice.voice_question(1, "Bar or line chart?")
    assert text == "Arr, bar or line, matey?"
    messages = complete.await_args.args[0]
    assert "a grumpy pirate" in messages[0]["content"]
    assert "Bar or line chart?" in messages[1]["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", [RuntimeError("down"), "", "<think>hmm</think>"])
async def test_a_failed_rewording_asks_the_question_as_written(outcome):
    config = MagicMock()
    config.get_setting = AsyncMock(return_value=None)
    side = {"side_effect": outcome} if isinstance(outcome, Exception) else {"return_value": outcome}
    with patch("classes.config_manager.configManager", return_value=config), \
         patch.object(persona_voice._llm, "complete", AsyncMock(**side)):
        assert await persona_voice.voice_question(1, "Bar or line?") == "Bar or line?"


def test_the_question_uses_the_personas_connection_not_the_brains(monkeypatch):
    monkeypatch.setenv("LLM_HOST", "http://persona:8080")
    monkeypatch.setenv("BRAIN_LLM_HOST", "https://brain.example")
    assert persona_voice._config().host.startswith("http://persona:8080")
    assert persona_voice._llm.lane == "chat"


# --------------------------- on_message routing ---------------------------

def _main():
    if "main" in sys.modules:
        return sys.modules["main"]
    import main
    return main


def _message(author_id=5, content="line please", channel_id=77):
    message = MagicMock()
    message.id = 9001
    message.content = content
    message.author = MagicMock(id=author_id)
    message.guild = MagicMock(id=1)
    message.channel = MagicMock(id=channel_id)
    message.mentions = []
    message.add_reaction = AsyncMock()
    message.reply = AsyncMock()
    return message


@pytest.mark.asyncio
async def test_the_requesters_next_message_is_the_answer():
    main = _main()
    future = brain_inbox.ask(77, 5)
    queue = asyncio.Queue(maxsize=2)
    message = _message()
    with patch.object(main, "message_queue", queue), \
         patch.object(main, "remember_poll", AsyncMock()), \
         patch.object(main, "should_handle_message", AsyncMock(return_value=True)) as handle:
        await main.on_message(message)
    assert future.result() == "line please"
    message.add_reaction.assert_awaited_once_with("✅")
    handle.assert_not_awaited()
    assert queue.empty()


@pytest.mark.asyncio
async def test_someone_elses_message_is_not_taken_as_the_answer():
    main = _main()
    future = brain_inbox.ask(77, 5)
    queue = asyncio.Queue(maxsize=2)
    with patch.object(main, "message_queue", queue), \
         patch.object(main, "remember_poll", AsyncMock()), \
         patch.object(main.automation_runner, "match_message", AsyncMock(return_value=False)), \
         patch.object(main, "should_handle_message", AsyncMock(return_value=False)) as handle:
        await main.on_message(_message(author_id=6))
    assert not future.done()
    handle.assert_awaited_once()


@pytest.mark.asyncio
async def test_an_attachment_only_answer_gets_a_note_and_the_question_stays_open():
    main = _main()
    future = brain_inbox.ask(77, 5)
    message = _message(content="")
    with patch.object(main, "remember_poll", AsyncMock()):
        await main.on_message(message)
    assert not future.done()
    message.reply.assert_awaited_once()
    message.add_reaction.assert_not_awaited()

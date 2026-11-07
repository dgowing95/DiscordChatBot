"""Tests for the brain's questions to the requester (3b): the waiter ledger
(core/classes/brain_inbox.py), the question's buttons
(core/classes/brain_question.py), the brain's ask_user tool
(core/classes/brain.py), the persona-voiced wording
(core/classes/persona_voice.py) and main.on_message taking a Discord Reply
to the question as its answer.

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

from classes import brain, brain_inbox, brain_question, metrics, persona_voice

QUESTION_ID = 4100


@pytest.fixture(autouse=True)
def _clean_inbox():
    brain_inbox.reset()
    yield
    brain_inbox.reset()


def _count(outcome):
    return metrics.brain_questions_total.labels(outcome=outcome)._value.get()


async def _until(predicate):
    for _ in range(50):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("never happened")


# --------------------------- brain_inbox (pure) ---------------------------

@pytest.mark.asyncio
async def test_inbox_takes_only_the_requesters_answer():
    future = brain_inbox.ask(QUESTION_ID, 77, 5)
    assert brain_inbox.waiting(QUESTION_ID)
    assert brain_inbox.requester(QUESTION_ID) == 5
    assert brain_inbox.deliver(QUESTION_ID, 6, "mine now") == brain_inbox.NOT_YOURS
    assert not future.done()
    assert brain_inbox.deliver(QUESTION_ID, 5, "  bar chart  ") == brain_inbox.ACCEPTED
    assert future.result() == "bar chart"
    assert not brain_inbox.waiting(QUESTION_ID)
    assert brain_inbox.requester(QUESTION_ID) is None
    assert brain_inbox.deliver(QUESTION_ID, 5, "again") == brain_inbox.NO_WAITER


@pytest.mark.asyncio
async def test_inbox_refuses_an_empty_answer_and_keeps_waiting():
    future = brain_inbox.ask(QUESTION_ID, 77, 5)
    assert brain_inbox.deliver(QUESTION_ID, 5, "   ") == brain_inbox.EMPTY
    assert brain_inbox.waiting(QUESTION_ID)
    assert not future.done()


@pytest.mark.asyncio
async def test_a_newer_spoken_question_to_the_same_speaker_replaces_the_older_one():
    old = brain_inbox.ask(1, 77, 5, voice=True)
    other = brain_inbox.ask(2, 77, 6, voice=True)  # someone else: untouched
    new = brain_inbox.ask(3, 77, 5, voice=True)
    assert old.result() is brain_inbox.REPLACED
    assert not brain_inbox.waiting(1)
    assert not other.done()
    brain_inbox.deliver(3, 5, "line")
    assert new.result() == "line"


@pytest.mark.asyncio
async def test_text_questions_never_replace_each_other():
    # One turn can ask two (each has its own buttons and Reply target).
    first = brain_inbox.ask(1, 77, 5)
    second = brain_inbox.ask(2, 77, 5)
    assert brain_inbox.waiting(1) and brain_inbox.waiting(2)
    brain_inbox.deliver(1, 5, "one")
    assert first.result() == "one" and not second.done()


@pytest.mark.asyncio
async def test_cancel_removes_only_its_own_waiter():
    old = brain_inbox.ask(1, 77, 5, voice=True)
    new = brain_inbox.ask(2, 77, 5, voice=True)
    brain_inbox.cancel(1, old)  # the replaced one cleaning up
    assert brain_inbox.waiting(2)
    brain_inbox.cancel(2, new)
    assert not brain_inbox.waiting(2)
    assert new.cancelled()


@pytest.mark.asyncio
async def test_only_voice_questions_take_spoken_answers():
    brain_inbox.ask(1, 77, 5)
    assert brain_inbox.voice_waiting(77, 5) is None
    brain_inbox.ask(2, 77, 5, voice=True)
    assert brain_inbox.voice_waiting(77, 5) == 2
    assert brain_inbox.voice_waiting(77, 6) is None
    assert brain_inbox.voice_waiting(78, 5) is None


# --------------------------- the buttons ---------------------------

def test_options_become_clean_button_labels():
    labels = brain_question.clean_options(
        ["Bar", " Bar ", "", None, "Line\nchart", "x" * 100, "Pie", "Donut", "You decide"])
    assert labels == ["Bar", "Line chart", "x" * 80, "Pie"]


def _labels(view):
    return [item.label for item in view.buttons]


@pytest.mark.asyncio
async def test_the_view_has_the_choices_then_answer_and_you_decide():
    view = brain_question.QuestionView(5, ["Bar", "Line"], timeout=60)
    assert _labels(view) == ["Bar", "Line", "Answer…", "You decide"]
    assert _labels(brain_question.QuestionView(5, None, timeout=60)) == ["Answer…", "You decide"]


def _interaction(user_id=5, content="Bar or line?"):
    interaction = MagicMock()
    interaction.user = SimpleNamespace(id=user_id)
    interaction.message = SimpleNamespace(content=content)
    interaction.response.edit_message = AsyncMock()
    interaction.response.send_message = AsyncMock()
    interaction.response.send_modal = AsyncMock()
    return interaction


@pytest.mark.asyncio
async def test_someone_else_pressing_a_button_is_told_privately():
    view = brain_question.QuestionView(5, ["Bar"], timeout=60)
    interaction = _interaction(user_id=6)
    assert await view.interaction_check(interaction) is False
    assert interaction.response.send_message.await_args.kwargs["ephemeral"] is True


@pytest.mark.asyncio
async def test_a_choice_button_answers_and_closes_the_question():
    view = brain_question.QuestionView(5, ["Bar", "Line"], timeout=60)
    view.question_id = QUESTION_ID
    future = brain_inbox.ask(QUESTION_ID, 77, 5)
    interaction = _interaction()
    await view.buttons[1].callback(interaction)
    assert future.result() == "Line"
    finished = interaction.response.edit_message.await_args.kwargs["view"]
    assert finished.buttons == [] and finished.note == "✅ Line"
    assert view.closed


@pytest.mark.asyncio
async def test_you_decide_hands_the_brain_its_marker():
    view = brain_question.QuestionView(5, [], timeout=60)
    view.question_id = QUESTION_ID
    future = brain_inbox.ask(QUESTION_ID, 77, 5)
    interaction = _interaction()
    await view.buttons[-1].callback(interaction)
    assert future.result() == brain_question.YOU_DECIDE_ANSWER
    assert interaction.response.edit_message.await_args.kwargs["view"].note == "✅ You decide"


@pytest.mark.asyncio
async def test_answer_opens_a_text_box_whose_text_is_the_answer():
    view = brain_question.QuestionView(5, [], timeout=60)
    view.question_id = QUESTION_ID
    future = brain_inbox.ask(QUESTION_ID, 77, 5)
    interaction = _interaction()
    await view.buttons[0].callback(interaction)
    modal = interaction.response.send_modal.await_args.args[0]
    assert isinstance(modal, brain_question.AnswerModal)
    modal.answer._value = "a line chart with dots"
    await modal.on_submit(_interaction())
    assert future.result() == "a line chart with dots"


@pytest.mark.asyncio
async def test_a_button_on_a_closed_question_says_so():
    view = brain_question.QuestionView(5, ["Bar"], timeout=60)
    view.question_id = QUESTION_ID  # no waiter: it timed out
    interaction = _interaction()
    await view.buttons[0].callback(interaction)
    interaction.response.send_message.assert_awaited_once()
    interaction.response.edit_message.assert_not_awaited()


# --------------------------- ask_user ---------------------------

def _ctx(**extra):
    channel = MagicMock(id=77)
    sent = MagicMock(id=QUESTION_ID, content="Bar or line chart?")
    sent.edit = AsyncMock()
    channel.send = AsyncMock(return_value=sent)
    message = MagicMock()
    message.to_reference = MagicMock(return_value="ref")
    ctx = {"guild_id": 1, "user_id": 5, "channel": channel, "original_message": message,
           "automatic": False, "voice": False, "unprompted": False,
           "brain_deadline": time.monotonic() + 600}
    ctx.update(extra)
    return ctx, sent


async def _ask(ctx, question="Bar or line chart?", options=None):
    from agents.tool_context import ToolContext
    args = {"question": question}
    if options is not None:
        args["options"] = options
    return await brain.ask_user.on_invoke_tool(
        ToolContext(context=ctx, tool_name="ask_user", tool_call_id="q1", tool_arguments="{}"),
        json.dumps(args))


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
    ctx, _sent = _ctx(**extra)
    assert await _ask(ctx) == brain.NO_ONE_TO_ASK
    assert _count("skipped") == before + 1
    if ctx.get("channel") is not None:
        ctx["channel"].send.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_question_is_a_reply_with_buttons_and_the_answer_is_returned(plain_voice):
    before = _count("answered")
    ctx, sent = _ctx()
    task = asyncio.create_task(_ask(ctx, options=["Bar", "Line"]))
    await _until(lambda: brain_inbox.waiting(QUESTION_ID))
    reply = ctx["channel"].send.await_args
    assert reply.kwargs["view"].text == (
        "-# Needs more information\n### Bar or line chart?") and not reply.args
    # A reply to the request, posted even if the request was deleted.
    assert reply.kwargs["reference"] == "ref"
    ctx["original_message"].to_reference.assert_called_once_with(fail_if_not_exists=False)
    assert _labels(reply.kwargs["view"]) == ["Bar", "Line", "Answer…", "You decide"]
    assert reply.kwargs["view"].question_id == QUESTION_ID
    brain_inbox.deliver(QUESTION_ID, 5, "line please")
    assert await task == "They answered: line please"
    assert ctx["brain_questions"] == 1
    assert _count("answered") == before + 1
    # Answered without a button (a Reply): the buttons come off here.
    finished = sent.edit.await_args.kwargs["view"]
    assert finished.buttons == [] and finished.note == "✅ line please"


@pytest.mark.asyncio
async def test_you_decide_tells_the_brain_to_choose(plain_voice):
    ctx, _sent = _ctx()
    task = asyncio.create_task(_ask(ctx))
    await _until(lambda: brain_inbox.waiting(QUESTION_ID))
    brain_inbox.deliver(QUESTION_ID, 5, brain_question.YOU_DECIDE_ANSWER)
    assert (await task).startswith("They said you should decide")


@pytest.mark.asyncio
async def test_no_source_message_posts_with_a_mention(plain_voice):
    ctx, _sent = _ctx(original_message=None)
    task = asyncio.create_task(_ask(ctx))
    await _until(lambda: brain_inbox.waiting(QUESTION_ID))
    view = ctx["channel"].send.await_args.kwargs["view"]
    assert view.text.endswith("### Bar or line chart?") and view.note == "<@5>"
    brain_inbox.deliver(QUESTION_ID, 5, "bar")
    assert await task == "They answered: bar"


@pytest.mark.asyncio
@pytest.mark.parametrize("source", [None, "message"])
async def test_a_name_in_a_question_pings_nobody_but_the_person_asked(plain_voice, source):
    ctx, _sent = _ctx() if source else _ctx(original_message=None)
    task = asyncio.create_task(_ask(ctx))
    await _until(lambda: brain_inbox.waiting(QUESTION_ID))
    mentions = ctx["channel"].send.await_args.kwargs["allowed_mentions"]
    assert [user.id for user in mentions.users] == [5]
    assert mentions.roles is False and mentions.everyone is False
    brain_inbox.deliver(QUESTION_ID, 5, "bar")
    await task


@pytest.mark.asyncio
async def test_no_answer_in_time_says_to_assume_and_closes_the_question(plain_voice, monkeypatch):
    monkeypatch.setenv("BRAIN_ASK_TIMEOUT", "1")
    monkeypatch.setattr(brain, "MIN_ASK_SECONDS", 0.01)
    before = _count("timeout")
    ctx, sent = _ctx()
    result = await _ask(ctx)
    assert result.startswith("No answer within 1 seconds")
    assert not brain_inbox.waiting(QUESTION_ID)
    assert _count("timeout") == before + 1
    assert sent.edit.await_args.kwargs["view"].note == brain_question.TIMED_OUT


@pytest.mark.asyncio
async def test_a_stopped_run_leaves_no_waiter_behind(plain_voice):
    # The call's own timeout cancels ask_user mid-wait; a waiter left behind
    # would take an answer nobody is waiting for.
    ctx, sent = _ctx()
    task = asyncio.create_task(_ask(ctx))
    await _until(lambda: brain_inbox.waiting(QUESTION_ID))
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not brain_inbox.waiting(QUESTION_ID)
    assert sent.edit.await_args.kwargs["view"].buttons == []


@pytest.mark.asyncio
async def test_two_questions_in_one_turn_share_the_cap(plain_voice):
    ctx, _sent = _ctx(brain_questions=brain.MAX_QUESTIONS - 1)
    first = asyncio.create_task(_ask(ctx, "One?"))
    second = asyncio.create_task(_ask(ctx, "Two?"))
    await _until(lambda: brain_inbox.waiting(QUESTION_ID))
    assert await second == brain.NO_ONE_TO_ASK
    brain_inbox.deliver(QUESTION_ID, 5, "yes")
    assert await first == "They answered: yes"


@pytest.mark.asyncio
async def test_a_replaced_question_tells_the_brain_to_assume(plain_voice):
    ctx, _sent = _ctx()
    ctx["voice"] = True
    first = asyncio.create_task(_ask(ctx, "One?"))
    await _until(lambda: brain_inbox.waiting(QUESTION_ID))
    brain_inbox.ask(QUESTION_ID + 1, 77, 5, voice=True)  # a newer spoken question
    assert "newer question" in await first


@pytest.mark.asyncio
async def test_a_failed_post_asks_nothing(plain_voice):
    ctx, _sent = _ctx()
    ctx["channel"].send.side_effect = RuntimeError("no permission")
    assert await _ask(ctx) == brain.NO_ONE_TO_ASK
    assert not brain_inbox.waiting(QUESTION_ID)


@pytest.mark.asyncio
async def test_in_a_voice_call_the_question_is_also_spoken(plain_voice):
    from classes import voice_gate
    session = MagicMock()
    session.speak_line = AsyncMock()
    voice_gate.reset()
    voice_gate.enter(1, session)
    try:
        ctx, _sent = _ctx(voice=True, original_message=None)
        task = asyncio.create_task(_ask(ctx))
        await _until(lambda: brain_inbox.waiting(QUESTION_ID))
        session.speak_line.assert_awaited_once_with("Bar or line chart?")
        assert brain_inbox.voice_waiting(77, 5) == QUESTION_ID
        brain_inbox.deliver(QUESTION_ID, 5, "bar")
        assert await task == "They answered: bar"
    finally:
        voice_gate.reset()


@pytest.mark.asyncio
async def test_a_run_stopped_while_the_question_is_spoken_leaves_no_waiter(plain_voice):
    from classes import voice_gate
    spoken = asyncio.Event()

    async def _slow_speak(text):
        spoken.set()
        await asyncio.sleep(60)

    session = MagicMock()
    session.speak_line = _slow_speak
    voice_gate.reset()
    voice_gate.enter(1, session)
    try:
        ctx, _sent = _ctx(voice=True, original_message=None)
        task = asyncio.create_task(_ask(ctx))
        await asyncio.wait_for(spoken.wait(), 1)
        assert brain_inbox.waiting(QUESTION_ID)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not brain_inbox.waiting(QUESTION_ID)
        assert brain_inbox.voice_waiting(77, 5) is None
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


# --------------------------- on_message: a Discord Reply ---------------------------

def _main():
    if "main" in sys.modules:
        return sys.modules["main"]
    import main
    return main


def _message(author_id=5, content="line please", reply_to=QUESTION_ID, forward=False,
             reference_channel=77):
    message = MagicMock()
    message.id = 9001
    message.content = content
    message.author = MagicMock(id=author_id)
    message.guild = MagicMock(id=1)
    message.channel = MagicMock(id=77)
    message.mentions = []
    import discord
    kind = discord.MessageReferenceType.forward if forward else discord.MessageReferenceType.default
    message.reference = (SimpleNamespace(message_id=reply_to, channel_id=reference_channel, type=kind)
                         if reply_to else None)
    message.add_reaction = AsyncMock()
    message.reply = AsyncMock()
    return message


async def _on_message(message, handled=False):
    main = _main()
    queue = asyncio.Queue(maxsize=2)
    with patch.object(main, "message_queue", queue), \
         patch.object(main, "remember_poll", AsyncMock()), \
         patch.object(main.automation_runner, "match_message", AsyncMock(return_value=False)), \
         patch.object(main, "should_handle_message", AsyncMock(return_value=handled)) as handle:
        await main.on_message(message)
    return handle


@pytest.mark.asyncio
async def test_a_reply_to_the_question_by_the_requester_is_the_answer():
    future = brain_inbox.ask(QUESTION_ID, 77, 5)
    message = _message()
    handle = await _on_message(message)
    assert future.result() == "line please"
    message.add_reaction.assert_awaited_once_with("✅")
    handle.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_ordinary_message_from_the_requester_is_a_new_request():
    future = brain_inbox.ask(QUESTION_ID, 77, 5)
    handle = await _on_message(_message(content="also, what's the time?", reply_to=None))
    assert not future.done()
    handle.assert_awaited_once()


@pytest.mark.asyncio
async def test_someone_elses_reply_to_the_question_is_not_the_answer():
    future = brain_inbox.ask(QUESTION_ID, 77, 5)
    handle = await _on_message(_message(author_id=6))
    assert not future.done()
    handle.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_reply_to_some_other_message_is_not_the_answer():
    future = brain_inbox.ask(QUESTION_ID, 77, 5)
    handle = await _on_message(_message(reply_to=QUESTION_ID + 7))
    assert not future.done()
    handle.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("kwargs", [{"forward": True}, {"reference_channel": 78}])
async def test_a_forward_of_the_question_is_not_the_answer(kwargs):
    # A forward also points at the question's message id; only a Reply in
    # the question's channel answers it.
    future = brain_inbox.ask(QUESTION_ID, 77, 5)
    message = _message(content="", **kwargs)
    handle = await _on_message(message)
    assert not future.done()
    message.reply.assert_not_awaited()
    handle.assert_awaited_once()


@pytest.mark.asyncio
async def test_an_attachment_only_reply_gets_a_note_and_the_question_stays_open():
    future = brain_inbox.ask(QUESTION_ID, 77, 5)
    message = _message(content="")
    await _on_message(message)
    assert not future.done()
    message.reply.assert_awaited_once()
    message.add_reaction.assert_not_awaited()


# --------------------------- fixes after the 3b/3c review ---------------------------

@pytest.mark.asyncio
async def test_two_questions_in_one_turn_are_both_answerable(plain_voice):
    ctx, sent = _ctx()
    sent2 = MagicMock(id=QUESTION_ID + 1, content="Colour?")
    sent2.edit = AsyncMock()
    ctx["channel"].send = AsyncMock(side_effect=[sent, sent2])
    first = asyncio.create_task(_ask(ctx, "Bar or line?"))
    second = asyncio.create_task(_ask(ctx, "Colour?"))
    await _until(lambda: brain_inbox.waiting(QUESTION_ID) and brain_inbox.waiting(QUESTION_ID + 1))
    brain_inbox.deliver(QUESTION_ID, 5, "bar")
    brain_inbox.deliver(QUESTION_ID + 1, 5, "blue")
    assert await first == "They answered: bar"
    assert await second == "They answered: blue"


@pytest.mark.asyncio
async def test_a_button_answer_edits_the_message_once_with_its_own_label(plain_voice):
    ctx, sent = _ctx()
    task = asyncio.create_task(_ask(ctx, "Bar or line?", ["Bar"]))
    await _until(lambda: brain_inbox.waiting(QUESTION_ID))
    view = ctx["channel"].send.await_args.kwargs["view"]
    interaction = _interaction()
    async def slow_edit(**kw):
        await asyncio.sleep(0.05)
    interaction.response.edit_message = AsyncMock(side_effect=slow_edit)
    await view.buttons[-1].callback(interaction)  # You decide
    assert "decide" in await task
    interaction.response.edit_message.assert_awaited_once()
    sent.edit.assert_not_awaited()  # ask_user's own close() stood aside


@pytest.mark.asyncio
async def test_a_failed_button_edit_still_gets_the_question_closed(plain_voice):
    ctx, sent = _ctx()
    task = asyncio.create_task(_ask(ctx, "Bar or line?", ["Bar"]))
    await _until(lambda: brain_inbox.waiting(QUESTION_ID))
    view = ctx["channel"].send.await_args.kwargs["view"]
    interaction = _interaction()
    interaction.response.edit_message = AsyncMock(side_effect=RuntimeError("expired"))
    with pytest.raises(RuntimeError):
        await view.buttons[-1].callback(interaction)
    await task
    sent.edit.assert_awaited_once()
    assert sent.edit.await_args.kwargs["view"].note == "✅ You decide"


def test_the_closing_line_survives_a_long_question():
    view = brain_question.QuestionView(5, [], 60, text="x" * brain_question.MAX_QUESTION_CHARS)
    finished = view.finished("⌛ No answer in time")
    assert finished.note == "⌛ No answer in time" and finished.text == view.text


@pytest.mark.asyncio
async def test_a_long_question_is_posted_cut_to_fit(plain_voice):
    ctx, sent = _ctx()
    task = asyncio.create_task(_ask(ctx, "q" * 2500))
    await _until(lambda: brain_inbox.waiting(QUESTION_ID))
    posted = ctx["channel"].send.await_args.kwargs["view"].text
    assert posted.count("q") == brain_question.MAX_QUESTION_CHARS
    brain_inbox.deliver(QUESTION_ID, 5, "ok")
    await task


@pytest.mark.asyncio
async def test_the_fallback_wording_is_capped():
    llm = AsyncMock(side_effect=RuntimeError("down"))
    with patch.object(persona_voice._llm, "complete", llm), \
         patch("classes.config_manager.configManager") as cm:
        cm.return_value.get_setting = AsyncMock(return_value="a pirate")
        assert len(await persona_voice.voice_question(1, "q" * 3000)) == 1500


@pytest.mark.asyncio
async def test_time_used_by_the_reword_and_post_is_counted(plain_voice):
    ctx, sent = _ctx()

    async def slow_reword(guild_id, question):
        ctx["brain_deadline"] = time.monotonic() + 20  # the reword ate the time
        return question
    with patch.object(persona_voice, "voice_question", AsyncMock(side_effect=slow_reword)):
        out = await _ask(ctx)
    assert out == brain.NO_ONE_TO_ASK
    assert not brain_inbox.waiting(QUESTION_ID)
    sent.edit.assert_awaited_once()  # the posted question was closed

"""Tests for classes/sandbox_conversation.py.

The runner-boundary tests drive the REAL openai-agents Runner with the SDK's
own deterministic ScriptedModel (agents.testing), so they check the
behaviour the coordinator depends on — where a filter's items land, when
on_llm_end fires relative to a response's tools, what a continuation
replays — against the pinned SDK, not against a mock of it. No network, no
Docker: the agent is a plain Agent with small function tools gated exactly
the way the sandbox's are (sandbox_agent._gated_invoke).
"""
import asyncio
import dataclasses
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from agents import Agent, RunConfig, Runner, function_tool
from agents.testing import ModelStep, ScriptedModel, assistant_message, function_call

from classes import sandbox_agent
from classes import sandbox_conversation as conv_mod
from classes import sandbox_thread_inbox as inbox
from classes.sandbox_conversation import SandboxConversation, make_agent_hooks

# To run this pytest file from the command line, use:
# PYTHONPATH=$(pwd) pytest core/tests/sandbox_conversation_tests.py


def _thread():
    thread = MagicMock(name="thread")
    thread.id = 7
    thread.send = AsyncMock(return_value=MagicMock(id=555))
    return thread


def _conversation(requester_id=1):
    ledger = inbox.RunLedger(thread_id=7, requester_id=requester_id)
    ledger.state = inbox.RUNNING
    return SandboxConversation(ledger, _thread())


def _deliver(conv, message_id, text, author_id=1, name="ana"):
    return conv.ledger.deliver(message_id, author_id, name, text)


class _Recorder:
    """Scripted responses that also record what each model call was sent."""

    def __init__(self):
        self.inputs = []

    def step(self, output, before=None):
        def _respond(call):
            self.inputs.append(json.loads(json.dumps(call.input, default=str)))
            if before is not None:
                before()
            return list(output)
        return ModelStep.respond(_respond)


def _user_texts(items):
    return [i.get("content") for i in items
            if isinstance(i, dict) and i.get("role") == "user"]


def _agent(conv, model, executed):
    """A plain Agent whose `work` tool is gated like a sandbox shell command,
    plus the real respond_to_updates tool."""
    @function_tool
    async def work(what: str) -> str:
        """Do some work."""
        executed.append(what)
        return f"did {what}"

    gated = dataclasses.replace(
        work, on_invoke_tool=sandbox_agent._gated_invoke(work.on_invoke_tool, "work"))
    return Agent(name="t", model=model, tools=[gated, sandbox_agent.respond_to_updates],
                 hooks=make_agent_hooks(conv))


async def _run(conv, model, executed, task="task", max_turns=8):
    agent = _agent(conv, model, executed)
    context = {"conversation": conv, "thread": conv.thread, "deliverables": []}
    return await Runner.run(agent, task, max_turns=max_turns, context=context,
                            run_config=RunConfig(call_model_input_filter=conv.input_filter))


# ---------------------- 1. input placement ----------------------

@pytest.mark.asyncio
async def test_a_message_is_shown_as_its_own_user_item_exactly_once_in_order():
    conv = _conversation()
    rec = _Recorder()
    executed = []
    model = ScriptedModel([
        rec.step([function_call("work", {"what": "a"}, call_id="c1")],
                 before=lambda: _deliver(conv, 11, "make it blue")),
        rec.step([function_call(conv_mod.RESPOND_TOOL_NAME,
                                {"message_ids": "1", "decision": "will_apply",
                                 "reply": "switching to blue"}, call_id="c2"),
                  function_call("work", {"what": "blue"}, call_id="c3")]),
        rec.step([assistant_message("done", item_id="m1")]),
    ])
    result = await _run(conv, model, executed)

    assert result.final_output == "done"
    first, second, third = rec.inputs
    # arrived during call 1, so call 1 never saw it...
    assert not any("make it blue" in (t or "") for t in _user_texts(first))
    # ...and call 2 sees it after the completed call/result pair, not inside
    # the tool output
    assert [i.get("type") or i.get("role") for i in second] == \
        ["user", "function_call", "function_call_output", "user"]
    assert "make it blue" in second[-1]["content"]
    assert "#1" in second[-1]["content"] and "the requester" in second[-1]["content"]
    assert "make it blue" not in second[2]["output"]
    # call 3: still there, same position, not duplicated
    shown = [t for t in _user_texts(third) if "make it blue" in (t or "")]
    assert len(shown) == 1
    assert third[3] == second[3]


@pytest.mark.asyncio
async def test_a_message_that_lands_during_inference_blocks_that_responses_actions():
    conv = _conversation()
    rec = _Recorder()
    executed = []
    model = ScriptedModel([
        # the model decides on `work` while the message arrives
        rec.step([function_call("work", {"what": "red"}, call_id="c1")],
                 before=lambda: _deliver(conv, 11, "actually, blue")),
        rec.step([function_call(conv_mod.RESPOND_TOOL_NAME,
                                {"message_ids": "1", "decision": "will_apply"}, call_id="c2"),
                  function_call("work", {"what": "blue"}, call_id="c3")]),
        rec.step([assistant_message("done", item_id="m1")]),
    ])
    await _run(conv, model, executed)

    # the stale "red" action never ran; the updated one did
    assert executed == ["blue"]
    refusal = rec.inputs[1][2]["output"]
    assert "Not run" in refusal
    assert conv.stale_blocked == 1


@pytest.mark.asyncio
async def test_acknowledging_and_acting_in_one_response_costs_no_refusal():
    # On a 10-turn budget, forcing ack and action into separate turns would
    # be expensive; a respond_to_updates in the same response counts.
    conv = _conversation()
    _deliver(conv, 11, "make it blue")
    executed = []
    model = ScriptedModel([
        [function_call(conv_mod.RESPOND_TOOL_NAME,
                       {"message_ids": "#1", "decision": "will_apply"}, call_id="c1"),
         function_call("work", {"what": "blue"}, call_id="c2")],
        [assistant_message("done", item_id="m1")],
    ])
    result = await _run(conv, model, executed)

    assert executed == ["blue"]
    assert conv.stale_blocked == 0
    assert len(result.raw_responses) == 2


@pytest.mark.asyncio
async def test_a_message_landing_mid_batch_blocks_the_rest_of_the_batch():
    conv = _conversation()
    executed = []

    @function_tool
    async def work(what: str) -> str:
        """Do some work."""
        executed.append(what)
        if what == "first":
            _deliver(conv, 11, "stop, change of plan")
        return f"did {what}"

    gated = dataclasses.replace(
        work, on_invoke_tool=sandbox_agent._gated_invoke(work.on_invoke_tool, "work"))
    model = ScriptedModel([
        [function_call("work", {"what": "first"}, call_id="c1"),
         function_call("work", {"what": "second"}, call_id="c2")],
        [function_call(conv_mod.RESPOND_TOOL_NAME,
                       {"message_ids": "1", "decision": "noted"}, call_id="c3")],
        [assistant_message("ok", item_id="m1")],
    ])
    agent = Agent(name="t", model=model, tools=[gated, sandbox_agent.respond_to_updates],
                  hooks=make_agent_hooks(conv))
    await Runner.run(agent, "task", max_turns=6,
                     context={"conversation": conv, "thread": conv.thread},
                     run_config=RunConfig(call_model_input_filter=conv.input_filter))

    assert executed == ["first"]


def test_presentation_is_only_committed_after_the_call_completes():
    # The filter runs before a call that may time out; it only stages.
    from agents.run_config import CallModelData, ModelInputData
    conv = _conversation()
    _deliver(conv, 11, "make it blue")
    data = CallModelData(model_data=ModelInputData(input=[{"role": "user", "content": "task"}],
                                                   instructions=None),
                         agent=MagicMock(), context=None)
    out = conv.input_filter(data)
    assert "make it blue" in out.input[-1]["content"]
    assert conv.ledger.events[0].stage == inbox.RECEIVED
    conv.on_model_response(MagicMock(output=[]))
    assert conv.ledger.events[0].stage == inbox.PRESENTED
    assert conv.ledger.events[0].anchor == 1


def test_a_filter_failure_sends_the_input_unchanged():
    from agents.run_config import CallModelData, ModelInputData
    conv = _conversation()
    conv.ledger = MagicMock(unpresented=MagicMock(side_effect=RuntimeError("boom")))
    items = [{"role": "user", "content": "task"}]
    data = CallModelData(model_data=ModelInputData(input=items, instructions="i"),
                         agent=MagicMock(), context=None)
    out = conv.input_filter(data)
    assert out.input == items and out.instructions == "i"


# ---------------------- 3. gating ----------------------

@pytest.mark.asyncio
async def test_the_gate_gives_up_after_the_refusal_cap():
    # A model that never answers must not burn its whole budget refused.
    conv = _conversation()
    _deliver(conv, 11, "make it blue")
    conv.ledger.mark_presented([1], anchor=1)
    for _ in range(conv_mod.MAX_GATE_REFUSALS):
        conv.on_model_response(MagicMock(output=[]))
        assert conv.gate("work") is not None
    conv.on_model_response(MagicMock(output=[]))
    assert conv.gate("work") is None
    assert conv.ledger.outcomes()[0]["status"] == "unacknowledged"


@pytest.mark.asyncio
async def test_many_gated_calls_in_one_response_count_as_one_refusal():
    # Counted per call, three gated tools in one response would exhaust the
    # cap by themselves and the third would run with nothing answered.
    conv = _conversation()
    _deliver(conv, 11, "make it blue")
    executed = []
    calls = [function_call("work", {"what": w}, call_id=f"c{i}")
             for i, w in enumerate(["a", "b", "c", "d"])]
    model = ScriptedModel([calls, [assistant_message("gave up", item_id="m1")]])
    await _run(conv, model, executed)

    assert executed == []
    assert conv.ledger.events[0].gate_refusals == 1


@pytest.mark.asyncio
async def test_a_blocking_question_holds_back_parallel_work():
    conv = _conversation()
    q = conv.ledger.open_question("which dataset?", blocking=True)
    q.waiting = True
    assert "blocking question" in conv.gate("work")
    q.waiting = False
    assert conv.gate("work") is None


@pytest.mark.asyncio
async def test_run_gated_marks_the_run_busy_while_a_command_runs():
    conv = _conversation()
    seen = {}

    async def _invoke():
        seen["busy"] = conv.ledger.busy_since is not None
        return "ok"

    assert await conv.run_gated("exec_command", _invoke, busy=True) == "ok"
    assert seen["busy"] is True and conv.ledger.busy_since is None


# ---------------------- respond_to_updates ----------------------

@pytest.mark.asyncio
async def test_respond_records_the_decision_and_posts_the_reply():
    conv = _conversation()
    _deliver(conv, 11, "make it blue")
    conv.ledger.mark_presented([1], anchor=1)
    out = await conv.respond("#1", "Will Apply", "Switching the bars to blue.")

    assert conv.ledger.events[0].outcome == "will_apply"
    conv.thread.send.assert_awaited_once()
    assert "Switching the bars to blue." in conv.thread.send.await_args.args[0]
    assert "mark it 'done'" in out


@pytest.mark.asyncio
async def test_respond_with_no_ids_covers_everything_shown():
    conv = _conversation()
    _deliver(conv, 11, "a")
    _deliver(conv, 12, "b")
    conv.ledger.mark_presented([1, 2], anchor=1)
    await conv.respond("", "noted", "")
    assert {e.outcome for e in conv.ledger.events} == {"noted"}
    # a generic reply is still posted, so the user sees a response
    assert "Noted" in conv.thread.send.await_args.args[0]


@pytest.mark.asyncio
async def test_respond_refuses_ids_that_were_never_shown():
    conv = _conversation()
    _deliver(conv, 11, "a")  # received, not presented
    out = await conv.respond("1, 9", "done", "done")
    assert "#1" in out and "#9" in out and "nothing was recorded" in out
    assert conv.ledger.events[0].stage == inbox.RECEIVED
    conv.thread.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_respond_rejects_an_unknown_decision_without_recording():
    conv = _conversation()
    _deliver(conv, 11, "a")
    conv.ledger.mark_presented([1], anchor=1)
    out = await conv.respond("1", "maybe", "")
    assert "Unknown decision" in out
    assert conv.ledger.events[0].stage == inbox.PRESENTED


@pytest.mark.asyncio
async def test_a_reply_that_cannot_be_sent_is_kept_for_the_closing_message():
    conv = _conversation()
    conv.thread.send = AsyncMock(side_effect=RuntimeError("discord down"))
    _deliver(conv, 11, "a")
    conv.ledger.mark_presented([1], anchor=1)
    out = await conv.respond("1", "answered", "It is 42.")
    assert conv.thread.send.await_count == conv_mod.REPLY_SEND_ATTEMPTS
    assert conv.unsent_replies == ["🐳 It is 42."]
    assert "closing message" in out
    # the decision itself still stands
    assert conv.ledger.events[0].outcome == "answered"


def test_parse_ids_accepts_what_a_small_model_sends():
    assert conv_mod.parse_ids("#3, #4") == [3, 4]
    assert conv_mod.parse_ids(3) == [3]
    assert conv_mod.parse_ids(["3", "#4"]) == [3, 4]
    assert conv_mod.parse_ids(None) == []


def test_render_marks_other_participants_as_suggestions():
    ledger = inbox.RunLedger(thread_id=7, requester_id=1)
    ledger.state = inbox.RUNNING
    ledger.deliver(1, 2, "bo", "use red", has_attachments=True)
    text = conv_mod.render_event(ledger.events[0], ledger)
    assert "another participant" in text and "requester decides" in text
    assert "cannot see" in text  # attachments are not claimed as read
    assert text.endswith("use red")


# ---------------------- questions ----------------------

@pytest.mark.asyncio
async def test_an_optional_question_returns_at_once():
    conv = _conversation()
    out = await conv.ask("Labels inside or outside?", blocking=False,
                         choices=["inside", "outside"], default="outside", timeout=30)
    assert "Keep working" in out and "outside" in out
    sent = conv.thread.send.await_args.args[0]
    assert "inside · outside" in sent and "I'll go with: outside" in sent
    assert conv.ledger.question.message_id == 555


@pytest.mark.asyncio
async def test_a_blocking_question_returns_when_a_reply_arrives():
    conv = _conversation()

    async def _reply_soon():
        await asyncio.sleep(0.01)
        _deliver(conv, 11, "png")

    task = asyncio.create_task(_reply_soon())
    out = await conv.ask("png or jpg?", blocking=True, timeout=5)
    await task

    assert "New thread input arrived" in out
    # the reply reaches the model once, through the ledger — not in this result
    assert "png" not in out
    assert conv.ledger.events[0].question_id == conv.ledger.question.qid


@pytest.mark.asyncio
async def test_a_reply_that_beats_the_question_send_is_not_missed():
    conv = _conversation()

    async def _send_and_race(text):
        _deliver(conv, 11, "png")  # the answer lands while send is in flight
        return MagicMock(id=555)

    conv.thread.send = AsyncMock(side_effect=_send_and_race)
    out = await conv.ask("png or jpg?", blocking=True, timeout=0.2)
    assert "New thread input arrived" in out


@pytest.mark.asyncio
async def test_a_blocking_question_times_out_to_the_default():
    conv = _conversation()
    out = await conv.ask("png or jpg?", blocking=True, default="png", timeout=0.05)
    assert "Go with your stated default (png)" in out
    assert conv.ledger.question.timed_out is True
    assert conv.ledger.question.waiting is False


@pytest.mark.asyncio
async def test_a_blocking_question_without_a_default_says_to_report_it_blocked():
    conv = _conversation()
    out = await conv.ask("Which account?", blocking=True, timeout=0.05)
    assert "blocked" in out


@pytest.mark.asyncio
async def test_only_one_question_at_a_time():
    conv = _conversation()
    await conv.ask("first?", blocking=False, timeout=5)
    out = await conv.ask("second?", blocking=False, timeout=5)
    assert "already have an open question" in out
    assert conv.thread.send.await_count == 1


@pytest.mark.asyncio
async def test_a_failed_question_send_is_not_left_open():
    conv = _conversation()
    conv.thread.send = AsyncMock(side_effect=RuntimeError("down"))
    out = await conv.ask("q?", blocking=True, timeout=5)
    assert "Could not reach the user" in out
    assert conv.ledger.question is None


# ---------------------- 4. finishing ----------------------

@pytest.mark.asyncio
async def test_finishing_with_input_unanswered_continues_without_replaying_tools(monkeypatch):
    monkeypatch.setenv("SANDBOX_MAX_TURNS", "10")
    conv = _conversation()
    rec = _Recorder()
    executed = []
    model = ScriptedModel([
        rec.step([function_call("work", {"what": "chart"}, call_id="c1")]),
        # the final answer is being written as the message arrives
        rec.step([assistant_message("here is your chart", item_id="m1")],
                 before=lambda: _deliver(conv, 11, "add a title")),
        rec.step([function_call(conv_mod.RESPOND_TOOL_NAME,
                                {"message_ids": "1", "decision": "will_apply"}, call_id="c2"),
                  function_call("work", {"what": "title"}, call_id="c3")]),
        rec.step([function_call(conv_mod.RESPOND_TOOL_NAME,
                                {"message_ids": "1", "decision": "done"}, call_id="c4")]),
        rec.step([assistant_message("chart with a title", item_id="m2")]),
    ])
    agent = _agent(conv, model, executed)
    context = {"conversation": conv, "thread": conv.thread, "deliverables": []}
    run_config = RunConfig(call_model_input_filter=conv.input_filter)

    result = await sandbox_agent._converse(agent, "task", run_config, None, context, conv)

    assert result.final_output == "chart with a title"
    assert executed == ["chart", "title"]  # nothing ran twice
    assert conv.continuations == 1
    assert conv.ledger.state == inbox.FINALIZING
    assert conv.ledger.outcomes()[0]["status"] == "done"
    # the continuation's first call carries the message and the nudge
    cont = rec.inputs[2]
    assert any("add a title" in (t or "") for t in _user_texts(cont))
    assert any((t or "").startswith("[sandbox]: Before you finish") for t in _user_texts(cont))


@pytest.mark.asyncio
async def test_continuations_are_bounded_when_the_model_never_answers(monkeypatch):
    monkeypatch.setenv("SANDBOX_MAX_TURNS", "20")
    conv = _conversation()
    _deliver(conv, 11, "make it blue")
    executed = []
    model = ScriptedModel([[assistant_message(f"done {i}", item_id=f"m{i}")]
                           for i in range(conv_mod.MAX_CONTINUATIONS + 1)])
    agent = _agent(conv, model, executed)
    context = {"conversation": conv, "thread": conv.thread, "deliverables": []}

    result = await sandbox_agent._converse(
        agent, "task", RunConfig(call_model_input_filter=conv.input_filter), None, context, conv)

    assert result.final_output == f"done {conv_mod.MAX_CONTINUATIONS}"
    assert conv.continuations == conv_mod.MAX_CONTINUATIONS
    assert conv.ledger.outcomes()[0]["status"] == "unacknowledged"
    assert conv.ledger.state == inbox.FINALIZING


@pytest.mark.asyncio
async def test_no_continuation_without_turns_left_for_it(monkeypatch):
    monkeypatch.setenv("SANDBOX_MAX_TURNS", "2")
    conv = _conversation()
    executed = []
    model = ScriptedModel([
        [function_call("work", {"what": "a"}, call_id="c1")],
        ModelStep.respond(lambda call: (_deliver(conv, 11, "late"),
                                        [assistant_message("done", item_id="m1")])[1]),
    ])
    agent = _agent(conv, model, executed)
    context = {"conversation": conv, "thread": conv.thread, "deliverables": []}

    result = await sandbox_agent._converse(
        agent, "task", RunConfig(call_model_input_filter=conv.input_filter), None, context, conv)

    assert result.final_output == "done"
    assert conv.continuations == 0
    assert conv.ledger.outcomes()[0]["status"] == "unseen"


@pytest.mark.asyncio
async def test_a_message_after_the_boundary_becomes_a_follow_up(monkeypatch):
    conv = _conversation()
    model = ScriptedModel([[assistant_message("done", item_id="m1")]])
    agent = _agent(conv, model, [])
    context = {"conversation": conv, "thread": conv.thread, "deliverables": []}
    await sandbox_agent._converse(
        agent, "task", RunConfig(call_model_input_filter=conv.input_filter), None, context, conv)

    assert conv.ledger.deliver(99, 1, "ana", "and a csv") == inbox.FINISHING
    assert conv.ledger.outcomes()[-1]["status"] == "late"


def test_the_nudge_flags_old_and_new_variants_attached_together_once():
    conv = _conversation()
    _deliver(conv, 11, "blue instead")
    conv.ledger.mark_presented([1], anchor=1)
    conv.ledger.record_decision([1], "done")
    deliverables = [{"path": "out/red.png", "change_revision": 0},
                    {"path": "out/blue.png", "change_revision": 1}]
    nudge = conv.continuation_nudge(deliverables)
    assert "out/red.png" in nudge and "remove=true" in nudge
    assert conv.continuation_nudge(deliverables) is None


def test_no_nudge_when_every_attachment_postdates_the_change():
    conv = _conversation()
    _deliver(conv, 11, "blue instead")
    conv.ledger.mark_presented([1], anchor=1)
    conv.ledger.record_decision([1], "done")
    assert conv.continuation_nudge([{"path": "out/blue.png", "change_revision": 1}]) is None


# ---------------------- attach_file remove ----------------------

@pytest.mark.asyncio
async def test_attach_file_can_remove_an_outdated_selection():
    from agents.tool_context import ToolContext
    ctx = ToolContext(
        context={"session": MagicMock(), "deliverables": [
            {"path": "out/red.png", "size": 1, "caption": "", "change_revision": 0}]},
        tool_name="attach_file", tool_call_id="a1", tool_arguments="{}")
    out = await sandbox_agent.attach_file.on_invoke_tool(
        ctx, json.dumps({"path": "out/red.png", "remove": True}))
    assert "Removed" in out
    assert ctx.context["deliverables"] == []


def _max_turns():
    from agents import MaxTurnsExceeded
    return MaxTurnsExceeded("out of turns in the continuation")


def _model_timeout():
    from agents import ModelTimeoutError
    return ModelTimeoutError(540.0)


def _client_timeout():
    import httpx2
    from openai import APITimeoutError
    return APITimeoutError(request=httpx2.Request("POST", "http://llm/v1/chat/completions"))


@pytest.mark.asyncio
@pytest.mark.parametrize("make_error", [_max_turns, _model_timeout, _client_timeout])
async def test_a_failed_continuation_delivers_the_answer_it_was_continuing(monkeypatch, make_error):
    monkeypatch.setenv("SANDBOX_MAX_TURNS", "10")
    conv = _conversation()
    model = ScriptedModel([
        ModelStep.respond(lambda call: (_deliver(conv, 11, "add a title"),
                                        [assistant_message("your chart", item_id="m1")])[1]),
    ])
    agent = _agent(conv, model, [])
    context = {"conversation": conv, "thread": conv.thread, "deliverables": []}
    real_run = Runner.run
    passes = []

    async def _run_then_fail(*args, **kwargs):
        passes.append(1)
        if len(passes) > 1:
            raise make_error()
        return await real_run(*args, **kwargs)

    monkeypatch.setattr(sandbox_agent.Runner, "run", _run_then_fail)
    result = await sandbox_agent._converse(
        agent, "task", RunConfig(call_model_input_filter=conv.input_filter), None, context, conv)

    assert result.final_output == "your chart"
    assert conv.ledger.state == inbox.FINALIZING
    assert conv.ledger.outcomes()[0]["status"] == "unseen"


@pytest.mark.asyncio
async def test_a_first_pass_failure_still_raises(monkeypatch):
    from agents import MaxTurnsExceeded
    conv = _conversation()

    async def _fail(*args, **kwargs):
        raise MaxTurnsExceeded("no final answer at all")

    monkeypatch.setattr(sandbox_agent.Runner, "run", _fail)
    with pytest.raises(MaxTurnsExceeded):
        await sandbox_agent._converse(MagicMock(), "task", MagicMock(), None,
                                      {"deliverables": []}, conv)


@pytest.mark.asyncio
async def test_respond_to_updates_accepts_a_bare_number():
    from agents.tool_context import ToolContext
    conv = _conversation()
    _deliver(conv, 11, "make it blue")
    conv.ledger.mark_presented([1], anchor=1)
    raw = json.dumps({"message_ids": 1, "decision": "will_apply", "reply": "on it"})
    ctx = ToolContext(context={"conversation": conv}, tool_name=conv_mod.RESPOND_TOOL_NAME,
                      tool_call_id="r1", tool_arguments=raw)
    await sandbox_agent.respond_to_updates.on_invoke_tool(ctx, raw)
    assert conv.ledger.events[0].outcome == "will_apply"


def test_a_filter_failure_after_staging_commits_nothing():
    from agents.run_config import CallModelData, ModelInputData
    conv = _conversation()
    _deliver(conv, 11, "make it blue")
    data = CallModelData(model_data=ModelInputData(input=[{"role": "user", "content": "t"}],
                                                   instructions=None),
                         agent=MagicMock(), context=None)
    real_render = conv_mod.render_event
    conv_mod.render_event = MagicMock(side_effect=RuntimeError("boom"))
    try:
        out = conv.input_filter(data)
    finally:
        conv_mod.render_event = real_render
    assert out.input == [{"role": "user", "content": "t"}]
    conv.on_model_response(MagicMock(output=[]))
    # the model never saw it, so it must not count as presented
    assert conv.ledger.events[0].stage == inbox.RECEIVED


@pytest.mark.asyncio
async def test_the_thread_shows_typing_only_while_the_model_is_generating():
    # Typing on during each model call, off while a tool runs and after.
    conv = _conversation()
    conv.thread.typing = AsyncMock()
    seen = []

    def _respond(output):
        def _step(call):
            seen.append(("model", conv._typing_task is not None))
            return output
        return _step

    model = ScriptedModel([
        ModelStep.respond(_respond([function_call("work", {"what": "a"}, call_id="c1")])),
        ModelStep.respond(_respond([assistant_message("done", item_id="m1")])),
    ])

    @function_tool
    async def work(what: str) -> str:
        """Do some work."""
        seen.append(("tool", conv._typing_task is not None))
        return "ok"

    agent = Agent(name="t", model=model, tools=[work], hooks=make_agent_hooks(conv))
    await sandbox_agent._converse(agent, "task", RunConfig(), None, {"deliverables": []}, conv)

    assert seen == [("model", True), ("tool", False), ("model", True)]
    assert conv._typing_task is None
    conv.thread.typing.assert_awaited()


@pytest.mark.asyncio
async def test_typing_stops_when_the_run_fails_mid_call(monkeypatch):
    conv = _conversation()
    conv.thread.typing = AsyncMock()

    async def _fail(*args, **kwargs):
        conv.start_typing()
        raise RuntimeError("model server went away")

    monkeypatch.setattr(sandbox_agent.Runner, "run", _fail)
    with pytest.raises(RuntimeError):
        await sandbox_agent._converse(MagicMock(), "task", MagicMock(), None,
                                      {"deliverables": []}, conv)
    assert conv._typing_task is None

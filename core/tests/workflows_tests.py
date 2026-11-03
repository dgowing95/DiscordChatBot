"""Tier 3 workflows (core/classes/workflows.py) end to end, on the REAL SDK
Runner with agents.testing's ScriptedModel standing in for the manager and
the helpers, so an SDK change in how a run stops at a tool, or how hooks and
filters are called, fails here.

No Docker and no Discord: the workspace lease (brain._acquire_workspace),
the run config (brain._run_config), the agents (workflows._manager /
_subagent: plain Agents with the REAL workflow tools), the thread
(workflows._open_thread) and Redis (workflow_store_tests.FakeRedis) are
the seams.
"""
import asyncio
import contextlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agents import Agent, RunConfig, function_tool
from agents.testing import ModelStep, ScriptedModel, assistant_message, function_call

from brain_tests import _Lease, _Session
from workflow_store_tests import FakeRedis

from classes import brain, brain_result, llm_lanes, thread_inbox, workflows
from classes import workflow_policy as wp
from classes.workflow_store import WorkflowStore


# --------------------------- fakes ---------------------------

class LaneModel(ScriptedModel):
    """A ScriptedModel that records the lane each call ran in."""

    def __init__(self, steps):
        super().__init__(steps)
        self.lanes = []

    async def get_response(self, *args, **kwargs):
        self.lanes.append(llm_lanes.current_lane.get())
        return await super().get_response(*args, **kwargs)


class FakeMessage:
    def __init__(self, channel, kwargs):
        self.channel, self.kwargs = channel, kwargs
        self.edits = []
        self.id = len(channel.sent) + 1000

    async def edit(self, **kwargs):
        self.edits.append(kwargs)


class FakeChannel:
    def __init__(self, channel_id):
        self.id = channel_id
        self.jump_url = f"https://discord.com/channels/1/{channel_id}"
        self.sent = []
        self.guild = None

    async def send(self, content=None, **kwargs):
        message = FakeMessage(self, dict(kwargs, content=content))
        self.sent.append(message)
        return message

    @contextlib.asynccontextmanager
    async def typing(self):
        yield

    def texts(self):
        return [m.kwargs.get("content") or "" for m in self.sent]

    def embeds(self):
        return [m.kwargs["embed"] for m in self.sent if m.kwargs.get("embed") is not None]


def _plan_call(*steps, call_id="p1"):
    return ModelStep(output=[function_call("submit_plan", {
        "goal": "Compare printers", "deliverables": ["comparison.md"], "notes": "",
        "steps": [{"id": s, "role": r, "brief": f"do {s}", "depends_on": d}
                  for s, r, d in steps]}, call_id=call_id)])


def _report(status="done", facts=("found it",), call_id="r1"):
    return ModelStep(output=[function_call("report", {"status": status, "facts": list(facts)},
                                           call_id=call_id)])


@pytest.fixture
def env(monkeypatch):
    """Patches every seam; returns a namespace the test fills with a model."""
    for name in ("WORKFLOW_PLAN_APPROVAL_SECONDS", "SUBAGENT_MAX_TURNS", "WORKFLOW_TIMEOUT",
                 "WORKFLOW_ENABLED", "WORKFLOW_PLAN_REVISIONS"):
        monkeypatch.delenv(name, raising=False)
    ns = SimpleNamespace(model=None, thread=FakeChannel(500), origin=FakeChannel(77),
                         session=_Session(), redis=FakeRedis(), queue=asyncio.Queue(10),
                         subagent_tools=[])
    store = WorkflowStore(ns.redis)

    @contextlib.asynccontextmanager
    async def _acquire(guild_id):
        yield _Lease(ns.session)

    def _manager(mode):
        from classes.brain_tools import brain_tools
        stop = {"manager": "submit_plan", "review": "submit_review", "finish": "report"}[mode]
        tools = brain_tools(mode=mode) if mode != "manager" else [workflows_tools().submit_plan]
        return Agent(name="Manager", model=ns.model, instructions="manager", tools=tools,
                     tool_use_behavior={"stop_at_tool_names": [stop]})

    def _subagent(role):
        from classes.workspace_tools import attach_file
        brain._guard_workspace_tools()
        return Agent(name="Helper", model=ns.model, instructions="helper",
                     tools=[*ns.subagent_tools, attach_file, brain.report],
                     tool_use_behavior={"stop_at_tool_names": ["report"]})

    def _config(lease, guard):
        return RunConfig(call_model_input_filter=brain.input_filter(guard, 6),
                         tracing_disabled=True)

    async def _open_thread(ctx, request, message):
        return ns.thread

    monkeypatch.setattr(brain, "_acquire_workspace", _acquire)
    monkeypatch.setattr(brain, "_run_config", _config)
    monkeypatch.setattr(brain, "_skills_index", AsyncMock(return_value=""))
    monkeypatch.setattr(workflows, "_manager", _manager)
    monkeypatch.setattr(workflows, "_subagent", _subagent)
    monkeypatch.setattr(workflows, "_open_thread", _open_thread)
    monkeypatch.setattr(workflows, "_store", lambda: store)
    monkeypatch.setattr("classes.persona_voice.voice_question",
                        AsyncMock(side_effect=lambda guild, text: f"(voiced) {text}"))
    monkeypatch.setattr("classes.workspace_commands.progress_enabled",
                        AsyncMock(return_value=False))
    workflows.configure(queue=ns.queue)
    ns.store = store
    yield ns
    workflows.configure(queue=None)
    workflows._queue = None


def workflows_tools():
    from classes import workflow_tools
    return workflow_tools


def _ctx(origin, automatic=False):
    return {"guild_id": 1, "user_id": 5, "requester_name": "Ana", "channel": origin,
            "automatic": automatic, "original_message": None}


async def _start(env, automatic=True, request="Compare the three cheapest 3D printers"):
    result = await workflows.start_workflow(_ctx(env.origin, automatic), request,
                                            trigger=workflows.TRIGGER_SLASH)
    assert result.ok, result.text
    return workflows.get(next(iter(workflows._RUNNING))), result


async def _finish(wf):
    await asyncio.wait_for(asyncio.shield(wf.task), 10)


# --------------------------- the happy path ---------------------------

@pytest.mark.asyncio
async def test_plan_auto_approve_two_steps_report_and_the_completion_reply_is_queued(env):
    env.model = LaneModel([
        _plan_call(("s1", "researcher", []), ("s2", "implementer", ["s1"])),
        _report(facts=["The Core One is $1,199 (https://example.com/core-one)"], call_id="a"),
        _report(facts=["Wrote comparison.md"], call_id="b"),
        _report(facts=["comparison.md compares 3 printers; cheapest is $1,199"], call_id="f"),
    ])
    wf, result = await _start(env)
    assert result.thread_url == env.thread.jump_url
    await _finish(wf)

    assert wf.record["state"] == wp.DONE
    # Manager in the brain lane, helpers in the subagent lane, and nothing
    # inherited from the caller.
    assert env.model.lanes == [llm_lanes.BRAIN, llm_lanes.SUBAGENT, llm_lanes.SUBAGENT,
                               llm_lanes.BRAIN]
    assert env.model.remaining_steps == 0
    # The second helper was told what the first found, and where its files are.
    second_task = json.dumps(env.model.calls[2].input)
    assert "$1,199" in second_task and f"../c500-w{wf.id}-s1/" in second_task
    # The plan was posted (approved automatically), then the closing embed.
    assert env.thread.texts()[0] == "Started automatically."
    assert [e.title for e in env.thread.embeds()][-1] == "🧵 Workflow done"
    # The completion reply waits on the queue, with the manager's report.
    item = env.queue.get_nowait()
    assert isinstance(item, workflows.WorkflowDone)
    assert item.report.facts == ["comparison.md compares 3 printers; cheapest is $1,199"]
    # Slot, ledger and registry are all free again.
    assert await env.store.running(1) == []
    assert not thread_inbox.is_run_active(env.thread.id)
    assert workflows.get(wf.id) is None
    saved = await env.store.get(1, wf.id)
    assert saved["state"] == wp.DONE and saved["steps"]["s1"]["status"] == "done"


@pytest.mark.asyncio
async def test_an_unusable_plan_is_retried_once_then_becomes_one_step(env):
    bad = ModelStep(output=[function_call("submit_plan", {"goal": "x", "steps": []},
                                          call_id="bad")])
    env.model = LaneModel([bad, ModelStep(output=[function_call(
        "submit_plan", {"goal": "x", "steps": []}, call_id="bad2")]),
        _report(call_id="a"), _report(call_id="f")])
    wf, _ = await _start(env, request="Build a CSV to chart tool")
    await _finish(wf)
    assert "no steps" in json.dumps(env.model.calls[1].input)  # the retry says why
    assert [(s.id, s.role, s.brief) for s in wf.plan.steps] == [
        ("s1", wp.IMPLEMENTER, "Build a CSV to chart tool")]
    assert wf.record["state"] == wp.DONE


@pytest.mark.asyncio
async def test_a_helper_out_of_turns_makes_the_workflow_partial(env, monkeypatch):
    monkeypatch.setenv("SUBAGENT_MAX_TURNS", "1")

    @function_tool
    async def look() -> str:
        """Look."""
        return "nothing"

    env.subagent_tools = [look]
    env.model = LaneModel([
        _plan_call(("s1", "researcher", [])),
        ModelStep(output=[function_call("look", {}, call_id="l1")]),
        _report(facts=["Only part of it was done"], call_id="f"),
    ])
    wf, _ = await _start(env)
    await _finish(wf)
    assert wf.record["state"] == wp.PARTIAL
    assert wf.results["s1"]["status"] == brain_result.PARTIAL
    assert "ran out of steps" in wf.results["s1"]["note"]
    assert env.queue.get_nowait().report.status == brain_result.PARTIAL


@pytest.mark.asyncio
async def test_a_workflow_out_of_time_stops_with_what_exists(env, monkeypatch):
    monkeypatch.setattr(workflows, "FINISH_RESERVE_SECONDS", 0)
    monkeypatch.setenv("WORKFLOW_TIMEOUT", "1")

    @function_tool
    async def slow() -> str:
        """Slow."""
        await asyncio.sleep(5)
        return "late"

    env.subagent_tools = [slow]
    env.model = LaneModel([_plan_call(("s1", "researcher", [])),
                           ModelStep(output=[function_call("slow", {}, call_id="s")])])
    wf, _ = await _start(env)
    await asyncio.wait_for(asyncio.shield(wf.task), 15)
    assert wf.record["state"] in (wp.PARTIAL, wp.FAILED)
    assert await env.store.running(1) == []
    assert isinstance(env.queue.get_nowait(), workflows.WorkflowDone)


# --------------------------- approval ---------------------------

def _press(wf, action, user_id=5):
    interaction = MagicMock()
    interaction.user.id = user_id
    interaction.response.send_message = AsyncMock()
    interaction.response.defer = AsyncMock()
    return interaction, workflows.on_button(interaction, wf.id, action)


async def _until(predicate, seconds=5):
    for _ in range(int(seconds * 50)):
        if predicate():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("condition never became true")


@pytest.mark.asyncio
async def test_the_requester_approves_and_nobody_else_can(env):
    env.model = LaneModel([_plan_call(("s1", "researcher", [])), _report(call_id="a"),
                           _report(call_id="f")])
    wf, _ = await _start(env, automatic=False)
    await _until(lambda: wf.state == wp.AWAITING_APPROVAL and wf.approval is not None)
    assert "(voiced)" in env.thread.texts()[0] and "<@5>" in env.thread.texts()[0]
    other, call = _press(wf, "approve", user_id=6)
    await call
    assert "Only <@5>" in other.response.send_message.await_args.args[0]
    assert wf.state == wp.AWAITING_APPROVAL
    mine, call = _press(wf, "approve")
    await call
    mine.response.defer.assert_awaited()
    await _finish(wf)
    assert wf.record["state"] == wp.DONE


@pytest.mark.asyncio
async def test_a_thread_message_while_the_plan_waits_changes_it(env):
    env.model = LaneModel([
        _plan_call(("s1", "researcher", [])),
        _plan_call(("s1", "researcher", []), ("s2", "implementer", ["s1"]), call_id="p2"),
        _report(call_id="a"), _report(call_id="b"), _report(call_id="f")])
    wf, _ = await _start(env, automatic=False)
    await _until(lambda: wf.state == wp.AWAITING_APPROVAL and wf.approval is not None)
    assert thread_inbox.deliver(env.thread.id, 9001, 5, "Ana", "also write it up as a table") \
        == thread_inbox.ACCEPTED
    await _until(lambda: wf.plan is not None and len(wf.plan.steps) == 2
                 and wf.approval is not None and not wf.approval.done())
    # The manager saw the plan so far and the change.
    revise = json.dumps(env.model.calls[1].input)
    assert "also write it up as a table" in revise and "Your plan so far" in revise
    assert wf.ledger.events[0].outcome == "done"
    _, call = _press(wf, "approve")
    await call
    await _finish(wf)
    assert wf.record["state"] == wp.DONE and set(wf.results) == {"s1", "s2"}


@pytest.mark.asyncio
async def test_no_answer_starts_it_anyway(env, monkeypatch):
    monkeypatch.setenv("WORKFLOW_PLAN_APPROVAL_SECONDS", "1")
    env.model = LaneModel([_plan_call(("s1", "researcher", [])), _report(call_id="a"),
                           _report(call_id="f")])
    wf, _ = await _start(env, automatic=False)
    await asyncio.wait_for(asyncio.shield(wf.task), 10)
    assert wf.record["state"] == wp.DONE
    assert any("so I've started" in text for text in env.thread.texts())


@pytest.mark.asyncio
async def test_stop_in_the_thread_cancels_and_still_replies(env):
    env.model = LaneModel([_plan_call(("s1", "researcher", []))])
    wf, _ = await _start(env, automatic=False)
    await _until(lambda: wf.state == wp.AWAITING_APPROVAL and wf.approval is not None)
    thread_inbox.deliver(env.thread.id, 9002, 5, "Ana", "stop")
    await _finish(wf)
    assert wf.record["state"] == wp.CANCELLED
    assert [e.title for e in env.thread.embeds()][-1] == "🧵 Workflow cancelled"
    assert env.queue.get_nowait().workflow is wf


@pytest.mark.asyncio
async def test_cancel_during_a_step_stops_the_helper(env):
    started = asyncio.Event()

    @function_tool
    async def long_job() -> str:
        """Long."""
        started.set()
        await asyncio.sleep(30)
        return "done"

    env.subagent_tools = [long_job]
    env.model = LaneModel([_plan_call(("s1", "researcher", []), ("s2", "researcher", [])),
                           ModelStep(output=[function_call("long_job", {}, call_id="j")])])
    wf, _ = await _start(env)
    await asyncio.wait_for(started.wait(), 5)
    _, call = _press(wf, "cancel")
    await call
    await _finish(wf)
    assert wf.record["state"] == wp.CANCELLED
    assert wf.step_states["s2"] == "skipped"


# --------------------------- steering between steps ---------------------------

@pytest.mark.asyncio
async def test_a_thread_message_reaches_the_manager_between_steps_with_an_outcome(env):
    thread_id = env.thread.id

    @function_tool
    async def look() -> str:
        """Look."""
        thread_inbox.deliver(thread_id, 9003, 6, "Bo", "please use euros")
        return "ok"

    env.subagent_tools = [look]
    env.model = LaneModel([
        _plan_call(("s1", "researcher", []), ("s2", "implementer", ["s1"])),
        ModelStep(output=[function_call("look", {}, call_id="l")]),
        _report(call_id="a"),
        ModelStep(output=[function_call("submit_review", {
            "replies": [{"id": 1, "reply": "Will do, prices in euros."}]}, call_id="rv")]),
        _report(call_id="b"), _report(call_id="f")])
    wf, _ = await _start(env)
    await _finish(wf)
    review = json.dumps(env.model.calls[3].input)
    assert "[1] Bo: please use euros" in review
    assert any("Will do, prices in euros." in text for text in env.thread.texts())
    assert wf.ledger.events[0].outcome == "answered"
    assert wf.record["state"] == wp.DONE


# --------------------------- starting ---------------------------

@pytest.mark.asyncio
async def test_a_second_workflow_in_the_server_is_refused_with_the_first_ones_link(env):
    started = asyncio.Event()

    @function_tool
    async def wait() -> str:
        """Wait."""
        started.set()
        await asyncio.sleep(30)
        return "x"

    env.subagent_tools = [wait]
    env.model = LaneModel([_plan_call(("s1", "researcher", [])),
                           ModelStep(output=[function_call("wait", {}, call_id="w")])])
    wf, _ = await _start(env)
    await asyncio.wait_for(started.wait(), 5)
    second = await workflows.start_workflow(_ctx(env.origin), "another job",
                                            trigger=workflows.TRIGGER_SLASH)
    assert not second.ok and env.thread.jump_url in second.text
    workflows.request_cancel(wf, "test over")
    await _finish(wf)


@pytest.mark.asyncio
async def test_workflows_switched_off_start_nothing(env, monkeypatch):
    monkeypatch.setenv("WORKFLOW_ENABLED", "0")
    result = await workflows.start_workflow(_ctx(env.origin), "x", trigger="slash")
    assert not result.ok and "switched off" in result.text
    assert workflows._RUNNING == {}


@pytest.mark.asyncio
async def test_the_brains_start_workflow_ends_its_run_with_the_thread_link(monkeypatch):
    from classes import workflow_tools
    monkeypatch.setattr(workflows, "start_workflow", AsyncMock(
        return_value=workflows.StartResult(True, thread_url="https://discord.com/channels/1/500")))
    model = ScriptedModel([ModelStep(output=[function_call(
        "start_workflow", {"request": "big job", "reason": "many sources"}, call_id="sw")])])
    session = _Session()

    @contextlib.asynccontextmanager
    async def _acquire(guild_id):
        yield _Lease(session)

    def _agent(**kw):
        return Agent(name="Brain", model=model, instructions="b",
                     tools=[workflow_tools.start_workflow, brain.report],
                     tool_use_behavior={"stop_at_tool_names": ["report", "start_workflow"]})

    monkeypatch.setattr(brain, "_acquire_workspace", _acquire)
    monkeypatch.setattr(brain, "build_brain_agent", _agent)
    monkeypatch.setattr(brain, "_run_config", lambda lease, guard: RunConfig(
        tracing_disabled=True))
    monkeypatch.setattr(brain, "_skills_index", AsyncMock(return_value=""))
    monkeypatch.setattr("classes.workspace_commands.progress_enabled",
                        AsyncMock(return_value=False))
    channel = FakeChannel(77)
    report = await brain.run_brain({"guild_id": 1, "user_id": 5, "channel": channel,
                                    "original_message": None, "poll_tool_calls": 0,
                                    "events_made": []}, "big job")
    assert report.status == brain_result.STARTED_WORKFLOW
    assert report.thread_url == "https://discord.com/channels/1/500"
    text = brain_result.tool_result_for_persona(report)
    assert "never say it is" in text and "https://discord.com/channels/1/500" in text
    assert workflows.start_workflow.await_args.kwargs["trigger"] == workflows.TRIGGER_BRAIN


# --------------------------- buttons, restarts, the reply ---------------------------

@pytest.mark.asyncio
async def test_a_button_of_a_stopped_workflow_says_so():
    interaction = MagicMock()
    interaction.response.send_message = AsyncMock()
    await workflows.on_button(interaction, "deadbeef", "approve")
    assert interaction.response.send_message.await_args.args[0] == "This workflow has stopped."


@pytest.mark.asyncio
async def test_a_restart_tells_each_interrupted_workflows_thread(env):
    await env.store.save({"id": "a1", "guild_id": 1, "state": wp.RUNNING, "instance": "dcb-dev",
                          "thread_id": 500})
    client = MagicMock()
    client.get_channel.return_value = env.thread
    assert await workflows.interrupt_all(client) == 1
    assert env.thread.texts() == [workflows.INTERRUPTED_NOTE]
    assert (await env.store.get(1, "a1"))["state"] == wp.INTERRUPTED


@pytest.mark.asyncio
async def test_the_completion_reply_is_the_persona_with_no_tools(monkeypatch):
    made = {}

    class FakeHandler:
        def __init__(self, messages, guild_id, original_message, **kwargs):
            made.update(kwargs, prompt=messages[0]["content"])

        async def generate(self):
            return "<@5> It's done! The files are in the thread."

    monkeypatch.setattr("classes.text_llm_handler.TextLLMHandler", FakeHandler)
    origin = FakeChannel(77)
    wf = workflows.Workflow(record={"id": "a1", "guild_id": 1, "requester_id": 5,
                                    "request": "compare printers", "state": wp.DONE,
                                    "thread_url": "https://discord.com/channels/1/500"},
                            ctx={}, thread=FakeChannel(500), origin_channel=origin)
    client = MagicMock()
    client.user.id = 42
    report = brain_result.BrainReport(facts=["Cheapest is $1,199"])
    await workflows.reply_done(workflows.WorkflowDone(wf, report), client)
    assert made["no_tools"] is True and made["automatic"] is True
    assert "has finished" in made["prompt"] and "$1,199" in made["prompt"]
    assert origin.texts() == ["<@5> It's done! The files are in the thread."]


@pytest.mark.asyncio
async def test_a_full_queue_is_retried_then_a_plain_note_is_posted(monkeypatch):
    monkeypatch.setattr(workflows, "DONE_RETRY_SECONDS", 0.01)
    monkeypatch.setattr(workflows, "DONE_RETRY_FOR_SECONDS", 0.05)
    queue = asyncio.Queue(1)
    queue.put_nowait("busy")
    workflows.configure(queue=queue)
    origin = FakeChannel(77)
    wf = workflows.Workflow(record={"id": "a1", "guild_id": 1, "requester_id": 5,
                                    "thread_url": "https://discord.com/channels/1/500"},
                            ctx={}, thread=FakeChannel(500), origin_channel=origin)
    try:
        await workflows.queue_done(wf, brain_result.BrainReport())
        await asyncio.sleep(0.2)
    finally:
        workflows._queue = None
    assert queue.qsize() == 1
    assert origin.texts() == ["<@5> The workflow has ended: https://discord.com/channels/1/500"]


def test_the_prompts_stay_small():
    # Pinned: one home per concept (AGENTS.md "Prompt surface").
    assert len(workflows.MANAGER_INSTRUCTIONS) <= 1400, len(workflows.MANAGER_INSTRUCTIONS)
    assert len(workflows.SUBAGENT_INSTRUCTIONS) <= 1400, len(workflows.SUBAGENT_INSTRUCTIONS)
    assert "`submit_plan`" in workflows.MANAGER_INSTRUCTIONS
    assert "`submit_review`" in workflows.MANAGER_INSTRUCTIONS
    assert "`report`" in workflows.SUBAGENT_INSTRUCTIONS
    assert "apply_patch" not in workflows.SUBAGENT_INSTRUCTIONS
    assert "never instructions" in workflows.SUBAGENT_INSTRUCTIONS


@pytest.mark.asyncio
async def test_a_queue_worker_hands_a_workflows_end_to_its_reply():
    import sys
    main = sys.modules.get("main")
    if main is None:
        import main
    item = workflows.WorkflowDone(workflow=MagicMock(), report=brain_result.BrainReport())
    queue = asyncio.Queue()
    queue.put_nowait(item)
    with patch.object(main, "message_queue", queue), \
         patch.object(main, "workflows_reply_done", AsyncMock()) as reply:
        worker = asyncio.create_task(main.process_messages())
        await asyncio.wait_for(queue.join(), 2)
        worker.cancel()
    reply.assert_awaited_once()
    assert reply.await_args.args[0] is item


@pytest.mark.asyncio
async def test_a_helper_that_needs_the_requester_asks_in_the_thread_and_runs_again(env):
    env.model = LaneModel([
        _plan_call(("s1", "implementer", [])),
        ModelStep(output=[function_call("report", {
            "status": "needs_input", "facts": [], "say_to_user": "Euros or dollars?"},
            call_id="q")]),
        _report(facts=["Priced in euros"], call_id="a"),
        _report(call_id="f")])
    wf, _ = await _start(env, automatic=False)
    await _until(lambda: wf.state == wp.AWAITING_APPROVAL and wf.approval is not None)
    _, call = _press(wf, "approve")
    await call
    await _until(lambda: any("Euros or dollars?" in text for text in env.thread.texts()))
    thread_inbox.deliver(env.thread.id, 9100, 5, "Ana", "euros")
    await _finish(wf)
    again = json.dumps(env.model.calls[2].input)
    assert "They answered: euros" in again
    assert wf.results["s1"]["facts"] == ["Priced in euros"]
    assert wf.record["state"] == wp.DONE


@pytest.mark.asyncio
async def test_a_message_during_the_last_step_is_still_answered(env):
    thread_id = env.thread.id

    @function_tool
    async def look() -> str:
        """Look."""
        thread_inbox.deliver(thread_id, 9200, 6, "Bo", "nice work")
        return "ok"

    env.subagent_tools = [look]
    env.model = LaneModel([
        _plan_call(("s1", "researcher", [])),
        ModelStep(output=[function_call("look", {}, call_id="l")]), _report(call_id="a"),
        ModelStep(output=[function_call("submit_review", {
            "replies": [{"id": 1, "reply": "Thanks!"}]}, call_id="rv")]),
        _report(call_id="f")])
    wf, _ = await _start(env)
    await _finish(wf)
    assert wf.ledger.events[0].outcome == "answered"
    assert wf.record["state"] == wp.DONE


@pytest.mark.asyncio
async def test_new_steps_may_build_on_a_done_step_and_later_helpers_hear_the_thread(env):
    thread_id = env.thread.id

    @function_tool
    async def look() -> str:
        """Look."""
        thread_inbox.deliver(thread_id, 9300, 5, "Ana", "add Elegoo too")
        return "ok"

    env.subagent_tools = [look]
    env.model = LaneModel([
        _plan_call(("s1", "researcher", []), ("s2", "implementer", ["s1"])),
        ModelStep(output=[function_call("look", {}, call_id="l")]), _report(call_id="a"),
        ModelStep(output=[function_call("submit_review", {
            "replies": [{"id": 1, "reply": "Adding Elegoo."}],
            "steps": [{"id": "s3", "role": "researcher", "brief": "Elegoo", "depends_on": ["s1"]},
                      {"id": "s4", "role": "implementer", "brief": "Write it",
                       "depends_on": ["s1", "s3"]}]}, call_id="rv")]),
        _report(call_id="b"), _report(call_id="c"), _report(call_id="f")])
    wf, _ = await _start(env)
    await _finish(wf)
    assert list(wf.results) == ["s1", "s3", "s4"]
    assert wf.step_states["s2"] == "skipped"
    later = json.dumps(env.model.calls[4].input)
    assert "Ana: add Elegoo too (reply: Adding Elegoo.)" in later
    assert wf.record["state"] == wp.DONE

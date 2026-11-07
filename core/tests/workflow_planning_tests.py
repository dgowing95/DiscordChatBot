"""A workflow's plan stage (classes/workflow_planning.py) on the REAL SDK: the
scouts, the questions, and what reaches the agents after them. The seams are
workflows_tests' `env` fixture."""
import asyncio
import contextlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from agents import RunContextWrapper, function_tool
from agents.testing import ModelStep, function_call

from classes import brain, brain_inbox, brain_question, thread_inbox, workflow_planning, workflows
from classes import workflow_policy as wp
from classes import card, card_kit as K
from card_checks import label_of, view_text
from workflows_tests import (  # noqa: F401 (env is a fixture)
    LaneModel,
    _blocks,
    _finish,
    _plan_call,
    _press,
    _report,
    _start,
    _until,
    env,
)

THE_FIGURES = "Oslo 709,000"


def _discover(*subjects, call_id="d1"):
    return ModelStep(output=[function_call("discover", {"subjects": list(subjects)},
                                           call_id=call_id)])


def _ask(*questions, call_id="q1"):
    items = [{"question": q, "options": ["Yes", "No"]} for q in questions]
    return ModelStep(output=[function_call("ask_requester", {"questions": items},
                                           call_id=call_id)])


def _text_of(call) -> str:
    return json.dumps(call.input, ensure_ascii=False)


def _tools_of(call) -> list[str]:
    return [t.name for t in call.tools]


def _look(call_id):
    return ModelStep(output=[function_call("look", {}, call_id=call_id)])


async def _approve(wf):
    await _until(lambda: wf.state == wp.AWAITING_APPROVAL and wf.approval is not None)
    _, call = _press(wf, "approve")
    await call


async def _question(env, text):
    """Waits for the question, and returns its posted message."""
    await _until(lambda: any(text in t and m.kwargs.get("view") is not None
                             for m, t in zip(env.thread.sent, env.thread.texts())))
    return next(m for m in reversed(env.thread.sent)
                if text in getattr(m.kwargs.get("view"), "text", ""))


# --------------------------- the rounds ---------------------------

@pytest.mark.asyncio
async def test_discover_then_ask_then_plan_and_what_was_learned_reaches_every_helper(env):
    env.model = LaneModel([
        _discover("Which table lists the cities?"),
        _report(facts=("One CSV per country, columns city and year",), call_id="s"),
        _ask("Metric or imperial?"),
        _plan_call(("s1", "researcher", []), ("s2", "implementer", ["s1"])),
        _report(call_id="a"), _report(call_id="b"), _report(call_id="f")])
    wf, _ = await _start(env, automatic=False)
    await _question(env, "Metric or imperial?")
    thread_inbox.deliver(env.thread.id, 9000, 5, "Ana", "Metric please")
    await _approve(wf)
    await _finish(wf)
    assert wf.record["state"] == wp.DONE
    calls = env.model.calls
    assert "One CSV per country" in _text_of(calls[2])  # the question run
    for call in (calls[3], calls[4], calls[5], calls[6]):  # the plan, the helpers, the finish
        assert "One CSV per country" in _text_of(call)
        assert "Metric please" in _text_of(call)
    assert "The helpers get this too" in _text_of(calls[3])


@pytest.mark.asyncio
async def test_each_round_happens_once_and_its_tool_goes_away(env):
    env.model = LaneModel([_discover("Which table?"), _report(call_id="s"), _ask("Which year?"),
                           _plan_call(("s1", "researcher", [])), _report(call_id="a"),
                           _report(call_id="f")])
    wf, _ = await _start(env, automatic=False)
    await _question(env, "Which year?")
    thread_inbox.deliver(env.thread.id, 9001, 5, "Ana", "2025")
    await _approve(wf)
    await _finish(wf)
    calls = env.model.calls
    assert {"discover", "ask_requester", "submit_plan"} <= set(_tools_of(calls[0]))
    assert "discover" not in _tools_of(calls[2]) and "ask_requester" in _tools_of(calls[2])
    assert not {"discover", "ask_requester"} & set(_tools_of(calls[3]))
    assert "`discover`" in _text_of(calls[0]) and "`discover`" not in _text_of(calls[2])


@pytest.mark.asyncio
async def test_a_plain_job_needs_no_round_and_says_nothing_extra(env):
    env.model = LaneModel([_plan_call(("s1", "researcher", [])), _report(call_id="a"),
                           _report(call_id="f")])
    wf, _ = await _start(env)
    await _finish(wf)
    assert wf.record["state"] == wp.DONE
    assert not any("Checking a few things" in text for text in env.thread.texts())


@pytest.mark.asyncio
async def test_asking_for_both_in_one_turn_runs_both_rounds(env):
    both = ModelStep(output=[
        function_call("discover", {"subjects": ["Which table?"]}, call_id="d"),
        function_call("ask_requester", {"questions": [{"question": "Which year?"}]}, call_id="q")])
    env.model = LaneModel([both, _report(facts=("A table",), call_id="s"),
                           _plan_call(("s1", "researcher", [])), _report(call_id="a"),
                           _report(call_id="f")])
    wf, _ = await _start(env, automatic=False)
    await _question(env, "Which year?")
    thread_inbox.deliver(env.thread.id, 9002, 5, "Ana", "2025")
    await _approve(wf)
    await _finish(wf)
    assert wf.scouted and wf.asked and wf.answers == [("Which year?", "2025")]


@pytest.mark.asyncio
async def test_a_manager_that_ends_on_nothing_still_gets_a_plan(env):
    env.model = LaneModel([ModelStep(output=[]), ModelStep(output=[]), _report(call_id="a"),
                           _report(call_id="f")])
    wf, _ = await _start(env)
    await _finish(wf)
    assert [s.role for s in wf.plan.steps] == [wp.IMPLEMENTER]


# --------------------------- discovery ---------------------------

@pytest.mark.asyncio
async def test_the_requester_is_told_a_plan_is_coming_without_a_ping(env):
    env.model = LaneModel([_discover("Which table?"), _report(call_id="s"),
                           _plan_call(("s1", "researcher", [])), _report(call_id="a"),
                           _report(call_id="f")])
    wf, _ = await _start(env, automatic=False)
    await _approve(wf)
    await _finish(wf)
    note = next(m for m in env.thread.sent if "Checking a few things" in (m.kwargs["content"] or ""))
    assert "ready to approve" in note.kwargs["content"] and "<@" not in note.kwargs["content"]
    assert "<@5>" in view_text(env.thread.plans()[0].kwargs["view"])


@pytest.mark.asyncio
async def test_an_automatic_run_is_not_told_to_expect_a_ping_and_cannot_ask(env):
    env.model = LaneModel([_discover("Which table?"), _report(call_id="s"),
                           _plan_call(("s1", "researcher", [])), _report(call_id="a"),
                           _report(call_id="f")])
    wf, _ = await _start(env)
    await _finish(wf)
    assert "ask_requester" not in _tools_of(env.model.calls[0])
    assert "discover" in _tools_of(env.model.calls[0])
    note = next(t for t in env.thread.texts() if "Checking a few things" in t)
    assert "approve" not in note


@pytest.mark.asyncio
async def test_two_scouts_look_at_once_in_folders_of_their_own(env):
    both_here = asyncio.Barrier(2)
    folders = []

    @function_tool
    async def look(wrapper: RunContextWrapper[dict]) -> str:
        """Look."""
        folders.append(wrapper.context["job_dir"])
        await asyncio.wait_for(both_here.wait(), 5)
        return "ok"

    def _look(call_id):
        return ModelStep(output=[function_call("look", {}, call_id=call_id)])

    env.subagent_tools = [look]
    env.model = LaneModel([_discover("First?", "Second?", "Third?"), _look("l1"), _look("l2"),
                           _report(facts=("one",), call_id="a"),
                           _report(facts=("two",), call_id="b"),
                           _plan_call(("s1", "researcher", [])), _report(call_id="c"),
                           _report(call_id="f")])
    wf, _ = await _start(env)
    await _finish(wf)
    assert sorted(folders) == [f"jobs/c500-w{wf.id}-d1", f"jobs/c500-w{wf.id}-d2"]
    assert len(wf.findings) == 2  # at most two scouts, whatever was asked
    assert env.model.lanes[1] == "subagent"


@pytest.mark.asyncio
async def test_scouts_hold_the_workspace_only_while_they_look(env, monkeypatch):
    held = []
    acquire = brain._acquire_workspace

    @contextlib.asynccontextmanager
    async def _tracked(guild_id):
        held.append(1)
        async with acquire(guild_id) as lease:
            yield lease
        held.pop()

    monkeypatch.setattr(brain, "_acquire_workspace", _tracked)
    env.model = LaneModel([_discover("Which table?"), _report(call_id="s"), _ask("Which year?"),
                           _plan_call(("s1", "researcher", [])), _report(call_id="a"),
                           _report(call_id="f")])
    wf, _ = await _start(env, automatic=False)
    await _question(env, "Which year?")
    assert held == [] and wf.lease is None
    thread_inbox.deliver(env.thread.id, 9003, 5, "Ana", "2025")
    await _until(lambda: wf.state == wp.AWAITING_APPROVAL and wf.approval is not None)
    assert held == [] and wf.lease is None
    _, call = _press(wf, "approve")
    await call
    await _finish(wf)


@pytest.mark.asyncio
async def test_a_workspace_that_will_not_start_leaves_the_plan_to_the_request(env, monkeypatch):
    real = brain._acquire_workspace
    tries = []

    @contextlib.asynccontextmanager
    async def _first_fails(guild_id):
        tries.append(guild_id)
        if len(tries) == 1:
            raise RuntimeError("no docker")
        async with real(guild_id) as lease:
            yield lease

    monkeypatch.setattr(brain, "_acquire_workspace", _first_fails)
    env.model = LaneModel([_discover("Which table?"), _plan_call(("s1", "researcher", [])),
                           _report(call_id="a"), _report(call_id="f")])
    wf, _ = await _start(env)
    await _finish(wf)
    assert wf.record["state"] == wp.DONE
    assert "rests on the request alone" in _text_of(env.model.calls[1])


@pytest.mark.asyncio
async def test_a_scout_stopped_by_its_limit_leaves_the_plan_going(env, monkeypatch):
    monkeypatch.setattr(wp, "SCOUT_MAX_TURNS", 1)

    @function_tool
    async def look() -> str:
        """Look."""
        return "ok"

    env.subagent_tools = [look]
    # Its last call is no report either: there is nothing to keep.
    env.model = LaneModel([_discover("Which table?"), _look("l"), _look("l2"),
                           _plan_call(("s1", "researcher", [])), _report(call_id="a"),
                           _report(call_id="f")])
    wf, _ = await _start(env)
    await _finish(wf)
    assert wf.record["state"] == wp.DONE and wf.findings == []


@pytest.mark.asyncio
async def test_cancel_during_discovery_stops_the_scouts(env):
    started = asyncio.Event()

    @function_tool
    async def long_look() -> str:
        """Long."""
        started.set()
        await asyncio.sleep(30)
        return "done"

    env.subagent_tools = [long_look]
    env.model = LaneModel([_discover("Which table?"),
                           ModelStep(output=[function_call("long_look", {}, call_id="l")])])
    wf, _ = await _start(env, automatic=False)
    await asyncio.wait_for(started.wait(), 5)
    _, call = _press(wf, "cancel")
    await call
    await _finish(wf)
    assert wf.record["state"] == wp.CANCELLED and not wf.running


@pytest.mark.asyncio
async def test_a_scout_gets_the_job_and_files_from_the_chat_but_no_plan(env):
    env.model = LaneModel([_discover("sales.csv"), _report(call_id="s"),
                           _plan_call(("s1", "researcher", [])), _report(call_id="a"),
                           _report(call_id="f")])
    ctx = {"guild_id": 1, "user_id": 5, "requester_name": "Ana", "channel": env.origin,
           "automatic": True, "original_message": None, "brain_inputs": ["sales.csv"],
           "job_dir": "jobs/c77-1-abc"}
    await workflows.start_workflow(ctx, "report on this", trigger=workflows.TRIGGER_SLASH)
    wf = workflows.get(next(iter(workflows._RUNNING)))
    await _finish(wf)
    scout = _text_of(env.model.calls[1])
    assert "Find out: where is sales.csv published or kept, and how is it laid out?" in scout
    assert "../c77-1-abc/in/: sales.csv" in scout


def test_the_prompts_keep_discovery_to_the_shape_of_the_job():
    from classes import workflow_tools
    assert len(workflows.SCOUT_INSTRUCTIONS) <= 900, len(workflows.SCOUT_INSTRUCTIONS)
    scout = " ".join(workflows.SCOUT_INSTRUCTIONS.split())
    assert "then stop" in scout and "Name the fields, not the values in them" in scout
    assert "at most 3 facts" in scout and wp.SCOUT_FACTS == 3
    description = " ".join(workflow_tools.discover.description.split())
    assert "SHAPE" in description and "A researcher step then collects" in description
    assert len(description) <= 650, len(description)
    assert len(workflow_tools.ask_requester.description) <= 300


def test_the_manager_hears_about_pictures_only_when_the_bot_can_draw(monkeypatch):
    monkeypatch.setenv("IMAGE_GEN_ENABLED", "1")
    assert workflows.MANAGER_IMAGE_LINE in workflows.manager_instructions()
    monkeypatch.setenv("IMAGE_GEN_ENABLED", "0")
    assert workflows.MANAGER_IMAGE_LINE not in workflows.manager_instructions()


# --------------------------- the thread stays readable ---------------------------

def _noisy_tool():
    @function_tool
    async def noisy(wrapper: RunContextWrapper[dict]) -> str:
        """Posts its notice, as a real lookup does."""
        await card.notice(wrapper.context["channel"], K.info("NOISE", "text"))
        return "ok"
    return noisy


def _titles(env):
    """Each card's label (without details), as the thread shows them."""
    return [label_of(m.kwargs["view"]) for m in env.thread.sent if m.kwargs.get("view") is not None]


def _scout_card(env, subject):
    """The scout's card for `subject`, or None."""
    return next((m for m in env.thread.sent if m.kwargs.get("view") is not None
                 and label_of(m.kwargs["view"]) == workflow_planning.SCOUT_LABEL
                 and f"### {subject}" in view_text(m.kwargs["view"])), None)


@pytest.mark.asyncio
async def test_a_scout_posts_one_card_and_its_tools_post_none(env):
    env.subagent_tools = [_noisy_tool()]
    env.model = LaneModel([_discover("Nordic cities"),
                           ModelStep(output=[function_call("noisy", {}, call_id="n")]),
                           _report(facts=("A table", "A CSV"), call_id="s"),
                           _plan_call(("s1", "implementer", [])), _report(call_id="a"),
                           _report(call_id="f")])
    wf, _ = await _start(env)
    await _finish(wf)
    titles = _titles(env)
    assert titles.count(workflow_planning.SCOUT_LABEL) == 1 and "NOISE" not in titles
    ending = view_text(_scout_card(env, "Nordic cities").edits[-1]["view"])
    assert "found 2 facts" in ending and "Last: 🔧 noisy" in ending
    assert ending.endswith("2 turns")


@pytest.mark.asyncio
async def test_a_researcher_posts_one_card_and_an_implementer_keeps_its_own(env):
    env.subagent_tools = [_noisy_tool()]
    env.model = LaneModel([_plan_call(("s1", "researcher", []), ("s2", "implementer", ["s1"])),
                           ModelStep(output=[function_call("noisy", {}, call_id="n1")]),
                           _report(facts=("One",), call_id="a"),
                           ModelStep(output=[function_call("noisy", {}, call_id="n2")]),
                           _report(call_id="b"), _report(call_id="f")])
    wf, _ = await _start(env)
    await _finish(wf)
    titles = _titles(env)
    assert titles.count("NOISE") == 1  # the implementer's
    assert titles.count("🔬 Research") == 1


@pytest.mark.asyncio
async def test_a_scout_is_told_to_finish_before_its_last_turns(env, monkeypatch):
    @function_tool
    async def look() -> str:
        """Look."""
        return "ok"

    monkeypatch.setattr(wp, "SCOUT_MAX_TURNS", 4)
    env.subagent_tools = [look]
    env.model = LaneModel([_discover("Rail delays"), _look("l1"), _look("l2"), _look("l3"),
                           _report(facts=("ORR portal",), call_id="s"),
                           _plan_call(("s1", "implementer", [])), _report(call_id="a"),
                           _report(call_id="f")])
    wf, _ = await _start(env)
    await _finish(wf)
    scout_calls = env.model.calls[1:5]
    assert [workflows.FINISH_UP in _text_of(c) for c in scout_calls] == [False, False, True, True]
    assert wf.findings == ["Rail delays: ORR portal"]


@pytest.mark.asyncio
async def test_a_scout_out_of_turns_still_reports_what_it_saw(env, monkeypatch):
    @function_tool
    async def look() -> str:
        """Look."""
        return "ok"

    monkeypatch.setattr(wp, "SCOUT_MAX_TURNS", 3)
    env.subagent_tools = [look]
    env.model = LaneModel([_discover("Rail delays"), _look("l1"), _look("l2"), _look("l3"),
                           _report(facts=("ORR portal",), call_id="s"),
                           _plan_call(("s1", "implementer", [])), _report(call_id="a"),
                           _report(call_id="f")])
    wf, _ = await _start(env)
    await _finish(wf)
    last = env.model.calls[4]
    assert _tools_of(last) == ["report"] and last.model_settings.tool_choice == "required"
    assert workflows.FINISH_UP in _text_of(last)
    assert wf.findings == ["Rail delays: ORR portal"]
    ending = view_text(_scout_card(env, "Rail delays").edits[-1]["view"])
    assert "ran out of steps" in ending and "found 1 fact" in ending


@pytest.mark.asyncio
async def test_a_scout_out_of_time_is_interrupted_and_still_reports(env, monkeypatch):
    @function_tool
    async def stuck() -> str:
        """Stuck."""
        await asyncio.sleep(30)
        return "never"

    monkeypatch.setattr(wp, "SCOUT_SECONDS", 0.05)
    monkeypatch.setattr(workflows, "FINISH_RESERVE_SECONDS", 0)
    monkeypatch.setattr(workflows.Workflow, "seconds_left", lambda self: 0.05, raising=False)
    env.subagent_tools = [stuck]
    env.model = LaneModel([_discover("Rail delays"),
                           ModelStep(output=[function_call("stuck", {}, call_id="l")]),
                           _report(facts=("ORR portal",), call_id="s"),
                           _plan_call(("s1", "implementer", [])), _report(call_id="a"),
                           _report(call_id="f")])
    wf, _ = await _start(env)
    await _finish(wf)
    assert wf.findings == ["Rail delays: ORR portal"]


@pytest.mark.asyncio
async def test_a_cancelled_scout_gets_no_last_report(env):
    started = asyncio.Event()

    @function_tool
    async def long_look() -> str:
        """Long."""
        started.set()
        await asyncio.sleep(30)
        return "done"

    env.subagent_tools = [long_look]
    env.model = LaneModel([_discover("Rail delays"),
                           ModelStep(output=[function_call("long_look", {}, call_id="l")])])
    wf, _ = await _start(env, automatic=False)
    await asyncio.wait_for(started.wait(), 5)
    _, call = _press(wf, "cancel")
    await call
    await _finish(wf)
    assert wf.record["state"] == wp.CANCELLED and len(env.model.calls) == 2


@pytest.mark.asyncio
async def test_scouts_keep_to_the_helpers_at_once_limit(env, monkeypatch):
    running, most = [0], [0]

    @function_tool
    async def look() -> str:
        """Look."""
        running[0] += 1
        most[0] = max(most[0], running[0])
        await asyncio.sleep(0.01)
        running[0] -= 1
        return "ok"

    monkeypatch.setattr(workflows, "_helpers_at_once", lambda: 1)
    env.subagent_tools = [look]
    env.model = LaneModel([_discover("One", "Two"), _look("a"), _report(facts=("x",), call_id="b"),
                           _look("c"), _report(facts=("y",), call_id="d"),
                           _plan_call(("s1", "implementer", [])), _report(call_id="e"),
                           _report(call_id="f")])
    wf, _ = await _start(env)
    await _finish(wf)
    assert most[0] == 1 and len(wf.findings) == 2


@pytest.mark.asyncio
async def test_a_cancel_while_waiting_for_the_workspace_starts_no_scouts(env, monkeypatch):
    acquire = brain._acquire_workspace

    @contextlib.asynccontextmanager
    async def _cancelled_meanwhile(guild_id):
        workflows.request_cancel(workflows.get(next(iter(workflows._RUNNING))), "stop")
        async with acquire(guild_id) as lease:
            yield lease

    monkeypatch.setattr(brain, "_acquire_workspace", _cancelled_meanwhile)
    env.model = LaneModel([_discover("Rail delays")])
    wf, _ = await _start(env, automatic=False)
    await _finish(wf)
    assert wf.record["state"] == wp.CANCELLED
    assert _scout_card(env, "Rail delays") is None


@pytest.mark.asyncio
async def test_a_scout_that_crashes_closes_its_embed_and_keeps_the_others_findings(env, monkeypatch):
    work = workflow_planning._scout_work

    async def _second_one_crashes(wf, subject, *args):
        if subject == "Two":
            raise RuntimeError("the workspace went")
        await work(wf, subject, *args)

    monkeypatch.setattr(workflow_planning, "_scout_work", _second_one_crashes)
    env.model = LaneModel([_discover("One", "Two"), _report(facts=("x",), call_id="a"),
                           _plan_call(("s1", "implementer", [])), _report(call_id="b"),
                           _report(call_id="f")])
    wf, _ = await _start(env)
    await _finish(wf)
    assert wf.findings == ["One: x"]
    assert workflows.HELPER_FAILED in view_text(_scout_card(env, "Two").edits[-1]["view"])


@pytest.mark.asyncio
async def test_a_helper_is_told_where_the_scouts_files_are(env):
    env.model = LaneModel([_discover("Sales chat"), _report(facts=("in/sales.jsonl",), call_id="s"),
                           _plan_call(("s1", "implementer", [])), _report(call_id="a"),
                           _report(call_id="f")])
    wf, _ = await _start(env)
    await _finish(wf)
    assert f"../c500-w{wf.id}-d1/" in _text_of(env.model.calls[3])


def test_two_scouts_whole_findings_are_never_cut_again():
    whole = wp.MAX_SUBJECT_CHARS + wp.SCOUT_FACTS * (wp.SCOUT_FACT_CHARS + 1) + len("- : ")
    assert whole <= wp.MAX_FINDINGS_CHARS // wp.MAX_SCOUTS


@pytest.mark.asyncio
async def test_a_long_fact_is_cut_at_a_word_and_keeps_what_fits(env):
    fact = "word " * 200
    env.model = LaneModel([_discover("Rail delays"), _report(facts=(fact,), call_id="s"),
                           _plan_call(("s1", "implementer", [])), _report(call_id="a"),
                           _report(call_id="f")])
    wf, _ = await _start(env)
    await _finish(wf)
    kept = wf.findings[0].removeprefix("Rail delays: ")
    assert len(kept) <= wp.SCOUT_FACT_CHARS and kept.endswith("word…")


@pytest.mark.asyncio
async def test_a_finished_plan_card_says_how_it_ended(env):
    started = asyncio.Event()

    @function_tool
    async def long_look() -> str:
        """Long."""
        started.set()
        await asyncio.sleep(30)
        return "done"

    env.subagent_tools = [long_look]
    env.model = LaneModel([_plan_call(("s1", "researcher", [])),
                           ModelStep(output=[function_call("long_look", {}, call_id="l")]),
                           _report(call_id="f")])
    wf, _ = await _start(env, automatic=False)
    await _approve(wf)
    await asyncio.wait_for(started.wait(), 5)
    _, call = _press(wf, "cancel")
    await call
    await _finish(wf)
    final = env.thread.plans()[-1].edits[-1]["view"]
    text = view_text(final)
    assert text.startswith("-# 🧵 Plan · cancelled") and "🛑 Cancelled." in text
    assert final.buttons == [] and "Working on it" not in text


def test_the_manager_can_only_name_a_subject():
    from classes import workflow_tools
    properties = workflow_tools.discover.params_json_schema["properties"]
    assert list(properties) == ["subjects"]


# --------------------------- questions ---------------------------

@pytest.mark.asyncio
async def test_a_button_answers_and_its_line_shows_on_the_question(env):
    env.model = LaneModel([_ask("Metric or imperial?"), _plan_call(("s1", "researcher", [])),
                           _report(call_id="a"), _report(call_id="f")])
    wf, _ = await _start(env, automatic=False)
    question = await _question(env, "Metric or imperial?")
    labels = [item.label for item in question.kwargs["view"].buttons]
    assert labels == ["Yes", "No", "Answer…", brain_question.YOU_DECIDE]
    assert brain_inbox.deliver(question.id, 5, "Yes") == brain_inbox.ACCEPTED
    await _approve(wf)
    await _finish(wf)
    assert wf.answers == [("Metric or imperial?", "Yes")]
    assert "<@5>" in question.kwargs["view"].label
    assert question.kwargs["view"].label.startswith("❓ Planning needs more information")
    assert question.edits[-1]["view"].outcome == "✅ **Yes**"


@pytest.mark.asyncio
async def test_you_decide_tells_the_manager_so(env):
    env.model = LaneModel([_ask("Metric or imperial?"), _plan_call(("s1", "researcher", [])),
                           _report(call_id="a"), _report(call_id="f")])
    wf, _ = await _start(env, automatic=False)
    question = await _question(env, "Metric or imperial?")
    brain_inbox.deliver(question.id, 5, brain_question.YOU_DECIDE_ANSWER)
    await _approve(wf)
    await _finish(wf)
    assert wf.answers == [("Metric or imperial?", brain_question.YOU_DECIDE)]


@pytest.mark.asyncio
async def test_someone_else_cannot_answer_for_the_requester(env):
    env.model = LaneModel([_ask("Metric or imperial?"), _plan_call(("s1", "researcher", [])),
                           _report(call_id="a"), _report(call_id="f")])
    wf, _ = await _start(env, automatic=False)
    question = await _question(env, "Metric or imperial?")
    thread_inbox.deliver(env.thread.id, 9004, 6, "Bo", "imperial")
    await asyncio.sleep(0.05)
    assert wf.question is not None and not wf.question.done()
    thread_inbox.deliver(env.thread.id, 9005, 5, "Ana", "metric")
    await _approve(wf)
    await _finish(wf)
    assert wf.answers == [("Metric or imperial?", "metric")]
    assert question.edits


@pytest.mark.asyncio
async def test_an_older_message_is_not_taken_for_the_answer(env):
    thread_id = env.thread.id

    @function_tool
    async def look() -> str:
        """Look."""
        thread_inbox.deliver(thread_id, 9030, 5, "Ana", "make it blue")
        return "ok"

    env.subagent_tools = [look]
    env.model = LaneModel([
        _plan_call(("s1", "researcher", [])),
        ModelStep(output=[function_call("look", {}, call_id="l")]),
        ModelStep(output=[function_call("report", {"status": "needs_input", "facts": [],
                                                   "say_to_user": "Euros?"}, call_id="q")]),
        _report(call_id="a"),
        ModelStep(output=[function_call("submit_review", {
            "replies": [{"id": 1, "reply": "Blue it is."}]}, call_id="rv")]),
        _report(call_id="f")])
    wf, _ = await _start(env, automatic=False)
    await _approve(wf)
    await _until(lambda: wf.question_id is not None)
    await asyncio.sleep(0.05)
    assert not wf.question.done()
    thread_inbox.deliver(thread_id, 9031, 5, "Ana", "yes, euros")
    await _finish(wf)
    assert "They answered: yes, euros" in _text_of(env.model.calls[3])
    assert any("Blue it is." in text for text in env.thread.texts())


@pytest.mark.asyncio
async def test_the_first_unanswered_question_ends_the_round(env, monkeypatch):
    monkeypatch.setenv("WORKFLOW_ASK_TIMEOUT", "1")
    env.model = LaneModel([_ask("First?", "Second?", "Third?", "Fourth?"),
                           _plan_call(("s1", "researcher", [])), _report(call_id="a"),
                           _report(call_id="f")])
    wf, _ = await _start(env, automatic=False)
    await _approve(wf)
    await _finish(wf)
    assert not any("Second?" in text for text in env.thread.texts())
    assert "First?" in wf.answers[0][0] and "no answer" in wf.answers[0][1]
    assert "no answer" in _text_of(env.model.calls[1])


@pytest.mark.asyncio
async def test_at_most_three_questions_are_asked(env):
    env.model = LaneModel([_ask("First?", "Second?", "Third?", "Fourth?"),
                           _plan_call(("s1", "researcher", [])), _report(call_id="a"),
                           _report(call_id="f")])
    wf, _ = await _start(env, automatic=False)
    for seq, question in enumerate(("First?", "Second?", "Third?")):
        await _question(env, question)
        thread_inbox.deliver(env.thread.id, 9010 + seq, 5, "Ana", "yes")
        await _until(lambda: len(wf.answers) == seq + 1)
    await _approve(wf)
    await _finish(wf)
    assert not any("Fourth?" in text for text in env.thread.texts())


@pytest.mark.asyncio
async def test_stop_during_a_question_cancels(env):
    env.model = LaneModel([_ask("Metric or imperial?")])
    wf, _ = await _start(env, automatic=False)
    await _question(env, "Metric or imperial?")
    thread_inbox.deliver(env.thread.id, 9006, 5, "Ana", "stop", stop_word=True)
    await _finish(wf)
    assert wf.record["state"] == wp.CANCELLED and wf.answers == []


@pytest.mark.asyncio
async def test_cancel_during_a_question_closes_it(env):
    env.model = LaneModel([_ask("Metric or imperial?")])
    wf, _ = await _start(env, automatic=False)
    question = await _question(env, "Metric or imperial?")
    workflows.request_cancel(wf, "the requester pressed Cancel")
    await _finish(wf)
    assert wf.record["state"] == wp.CANCELLED
    assert question.edits and not brain_inbox.waiting(question.id)


@pytest.mark.asyncio
async def test_an_answer_the_guard_blocks_never_reaches_the_manager(env, monkeypatch):
    _blocks(monkeypatch, "forbidden")
    env.model = LaneModel([_ask("Which one?"), _plan_call(("s1", "researcher", [])),
                           _report(call_id="a"), _report(call_id="f")])
    wf, _ = await _start(env, automatic=False)
    await _question(env, "Which one?")
    thread_inbox.deliver(env.thread.id, 9007, 5, "Ana", "something forbidden")
    await _approve(wf)
    await _finish(wf)
    assert "something forbidden" not in json.dumps([_text_of(c) for c in env.model.calls])
    assert any("can't use that answer" in text for text in env.thread.texts())


@pytest.mark.asyncio
async def test_two_helpers_ask_one_at_a_time(env):
    env.model = LaneModel([
        _plan_call(("s1", "researcher", []), ("s2", "researcher", [])),
        ModelStep(output=[function_call("report", {"status": "needs_input", "facts": [],
                                                   "say_to_user": "Euros?"}, call_id="q1")]),
        ModelStep(output=[function_call("report", {"status": "needs_input", "facts": [],
                                                   "say_to_user": "Dollars?"}, call_id="q2")]),
        _report(call_id="a"), _report(call_id="b"), _report(call_id="f")])
    wf, _ = await _start(env, automatic=False)
    await _approve(wf)
    await _until(lambda: wf.question_id is not None)
    await asyncio.sleep(0.05)
    open_questions = [t for t in env.thread.texts() if "Euros?" in t or "Dollars?" in t]
    assert len(open_questions) == 1
    thread_inbox.deliver(env.thread.id, 9008, 5, "Ana", "yes")
    await _until(lambda: sum("Euros?" in t or "Dollars?" in t for t in env.thread.texts()) == 2)
    await _until(lambda: wf.question_id is not None)
    thread_inbox.deliver(env.thread.id, 9009, 5, "Ana", "no")
    await _finish(wf)
    assert wf.record["state"] == wp.DONE


# --------------------------- messages while planning ---------------------------

@pytest.mark.asyncio
async def test_a_message_during_discovery_reaches_the_next_plan_run_and_costs_no_edit(env):
    thread_id = env.thread.id

    @function_tool
    async def look() -> str:
        """Look."""
        thread_inbox.deliver(thread_id, 9020, 5, "Ana", "only use free sources")
        return "ok"

    env.subagent_tools = [look]
    env.model = LaneModel([_discover("Which table?"),
                           ModelStep(output=[function_call("look", {}, call_id="l")]),
                           _report(call_id="s"), _plan_call(("s1", "researcher", [])),
                           _report(call_id="a"), _report(call_id="f")])
    wf, _ = await _start(env, automatic=False)
    await _approve(wf)
    await _finish(wf)
    assert wf.revisions == 0
    assert "only use free sources" in _text_of(env.model.calls[3])
    assert "only use free sources" in _text_of(env.model.calls[4])


@pytest.mark.asyncio
async def test_a_blocked_message_during_discovery_never_reaches_the_plan(env, monkeypatch):
    _blocks(monkeypatch, "forbidden")
    thread_id = env.thread.id

    @function_tool
    async def look() -> str:
        """Look."""
        thread_inbox.deliver(thread_id, 9021, 5, "Ana", "do something forbidden")
        return "ok"

    env.subagent_tools = [look]
    env.model = LaneModel([_discover("Which table?"),
                           ModelStep(output=[function_call("look", {}, call_id="l")]),
                           _report(call_id="s"), _plan_call(("s1", "researcher", [])),
                           _report(call_id="a"), _report(call_id="f")])
    wf, _ = await _start(env)
    await _finish(wf)
    assert "something forbidden" not in json.dumps([_text_of(c) for c in env.model.calls])
    assert any("can't act on that message" in text for text in env.thread.texts())


@pytest.mark.asyncio
async def test_a_revised_plan_keeps_what_was_learned_and_can_still_ask(env):
    env.model = LaneModel([_discover("Which table?"),
                           _report(facts=("A CSV per country",), call_id="s"),
                           _plan_call(("s1", "researcher", []), call_id="p1"),
                           _ask("Which year?"), _plan_call(("s2", "researcher", []), call_id="p2"),
                           _report(call_id="a"), _report(call_id="f")])
    wf, _ = await _start(env, automatic=False)
    await _until(lambda: wf.state == wp.AWAITING_APPROVAL and wf.approval is not None)
    thread_inbox.deliver(env.thread.id, 9022, 5, "Ana", "add a year")
    await _question(env, "Which year?")
    thread_inbox.deliver(env.thread.id, 9023, 5, "Ana", "2025")
    await _until(lambda: wf.revisions == 1 and wf.state == wp.AWAITING_APPROVAL
                 and wf.approval is not None and not wf.approval.done())
    assert "A CSV per country" in _text_of(env.model.calls[4])
    assert "discover" not in _tools_of(env.model.calls[3])
    _, call = _press(wf, "approve")
    await call
    await _finish(wf)
    assert [s.id for s in wf.plan.steps] == ["s2"]


# --------------------------- pictures ---------------------------

@pytest.mark.asyncio
async def test_a_helpers_picture_is_kept_for_the_channel_and_the_next_step(env):
    @function_tool
    async def draw(wrapper: RunContextWrapper[dict]) -> str:
        """Draw."""
        wrapper.context["images_posted"] = wrapper.context.get("images_posted", 0) + 1
        return "Drew it." + await workflows.keep_image(wrapper.context, b"png-bytes")

    env.subagent_tools = [draw]
    env.model = LaneModel([_plan_call(("s1", "implementer", []), ("s2", "implementer", ["s1"])),
                           ModelStep(output=[function_call("draw", {}, call_id="d")]),
                           _report(call_id="a"), _report(call_id="b"), _report(call_id="f")])
    wf, _ = await _start(env)
    await _finish(wf)
    assert wf.results["s1"]["files"] == ["images/image-1.png"]
    assert any(name.endswith("/images/image-1.png") for name in env.session.files)
    assert "Saved as images/image-1.png" in _text_of(env.model.calls[2])
    assert "images/image-1.png" in _text_of(env.model.calls[3])  # the next step hears of it
    names = [f.filename for m in env.origin.sent for f in m.kwargs.get("files") or []]
    assert names == ["image-1.png"]


@pytest.mark.asyncio
async def test_a_picture_outside_a_workflow_is_left_alone():
    assert await workflows.keep_image({"workflow_id": "gone"}, b"png") == ""


@pytest.mark.asyncio
async def test_a_workspace_write_that_fails_still_keeps_the_picture_for_the_channel(env):
    env.model = LaneModel([_plan_call(("s1", "researcher", [])), _report(call_id="a"),
                           _report(call_id="f")])
    wf, _ = await _start(env)
    session = SimpleNamespace(exec=AsyncMock(side_effect=OSError("gone")))
    text = await workflows.keep_image({"workflow_id": wf.id, "images_posted": 2,
                                       "session": session, "out_dir": "/w/jobs/a/out"}, b"png")
    assert "image-2.png" in [a.name for a in wf.artifacts]
    assert "job's files" in text and "Saved as" not in text
    await _finish(wf)


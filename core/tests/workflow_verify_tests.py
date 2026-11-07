"""The check of a workflow's finished work (classes/workflow_verify.py), on the
real Runner, with workflows_tests' `env` fixture."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from agents import function_tool
from agents.testing import ModelStep, assistant_message, function_call

from classes import brain_result, workflow_verify, workflows
from classes import workflow_policy as wp
from workflows_tests import (  # noqa: F401 (env is a fixture)
    LaneModel,
    _finish,
    _plan_call,
    _report,
    _start,
    env,
    real_due,
)


@pytest.fixture
def checked(env, monkeypatch):  # noqa: F811
    monkeypatch.setattr(workflow_verify, "due", real_due)
    return env


def _verdict(*checks, redo="", call_id="v"):
    return ModelStep(output=[function_call("submit_verdict", {
        "checks": [{"deliverable": name, "ok": ok, "note": note} for name, ok, note in checks],
        "redo_step": redo}, call_id=call_id)])


GOOD = ("comparison.md", True, "3 printers, links work")
BAD = ("comparison.md", False, "Prices are from 2024")


@pytest.mark.asyncio
async def test_a_good_result_is_checked_and_stays_done(checked):
    checked.model = LaneModel([_plan_call(("s1", "implementer", [])), _report(call_id="a"),
                               _verdict(GOOD), _report(call_id="f")])
    wf, _ = await _start(checked)
    await _finish(wf)
    assert wf.record["state"] == wp.DONE
    assert wf.verdict == {"state": workflow_verify.PASSED, "notes": ""}
    assert wf.record["verdict"]["state"] == workflow_verify.PASSED
    task = json.dumps(checked.model.calls[2].input)
    assert "comparison.md" in task and f"../c500-w{wf.id}-s1/" in task
    assert checked.model.remaining_steps == 0


@pytest.mark.asyncio
async def test_a_failed_check_sends_the_step_back_and_the_fix_is_checked(checked):
    checked.model = LaneModel([_plan_call(("s1", "implementer", [])), _report(call_id="a"),
                               _verdict(BAD, redo="s1", call_id="v1"),
                               _report(facts=("Fixed the prices",), call_id="b"),
                               _verdict(GOOD, call_id="v2"), _report(call_id="f")])
    wf, _ = await _start(checked)
    await _finish(wf)
    assert wf.record["state"] == wp.DONE
    assert "Prices are from 2024" in json.dumps(checked.model.calls[3].input)
    assert wf.results["s1"]["facts"] == ["Fixed the prices"]
    assert wf.verdict["state"] == workflow_verify.PASSED
    # Its own folder again, and a new one for the second check.
    assert f"c500-w{wf.id}-v2" in json.dumps(checked.model.calls[4].input)


@pytest.mark.asyncio
async def test_a_second_fail_ends_partial_with_the_notes_in_the_report(checked):
    checked.model = LaneModel([_plan_call(("s1", "implementer", [])), _report(call_id="a"),
                               _verdict(BAD, redo="s1", call_id="v1"), _report(call_id="b"),
                               _verdict(BAD, redo="s1", call_id="v2"), _report(call_id="f")])
    wf, _ = await _start(checked)
    await _finish(wf)
    assert wf.record["state"] == wp.PARTIAL
    assert "Prices are from 2024" in json.dumps(checked.model.calls[5].input)
    report = checked.queue.get_nowait().report
    assert report.status == brain_result.PARTIAL and "Prices are from 2024" in report.note
    assert checked.model.remaining_steps == 0


@pytest.mark.asyncio
async def test_no_verdict_leaves_the_result_alone_and_says_so(checked):
    checked.model = LaneModel([_plan_call(("s1", "implementer", [])), _report(call_id="a"),
                               ModelStep(output=[assistant_message("It looks fine to me.")]),
                               _report(call_id="f")])
    wf, _ = await _start(checked)
    await _finish(wf)
    assert wf.record["state"] == wp.DONE
    assert wf.verdict["state"] == workflow_verify.NOT_CHECKED
    assert checked.queue.get_nowait().report.note == workflow_verify.NOT_CHECKED_NOTE


@pytest.mark.asyncio
async def test_a_fix_that_could_not_be_checked_stays_a_fail(checked):
    checked.model = LaneModel([_plan_call(("s1", "implementer", [])), _report(call_id="a"),
                               _verdict(BAD, redo="s1"), _report(call_id="b"),
                               ModelStep(output=[assistant_message("No idea.")]),
                               _report(call_id="f")])
    wf, _ = await _start(checked)
    await _finish(wf)
    assert wf.record["state"] == wp.PARTIAL
    assert wf.verdict["state"] == workflow_verify.FAILED


@pytest.mark.asyncio
async def test_a_step_that_ran_out_of_turns_is_not_sent_back(checked, monkeypatch):
    monkeypatch.setenv("SUBAGENT_MAX_TURNS", "1")

    @function_tool
    async def look() -> str:
        """Look."""
        return "ok"

    checked.subagent_tools = [look]
    checked.model = LaneModel([_plan_call(("s1", "researcher", [])),
                               ModelStep(output=[function_call("look", {}, call_id="l")]),
                               _verdict(BAD, redo="s1"), _report(call_id="f")])
    wf, _ = await _start(checked)
    await _finish(wf)
    assert "s1" in wf.at_limit
    assert wf.record["state"] == wp.PARTIAL and wf.verdict["state"] == workflow_verify.FAILED
    assert checked.model.remaining_steps == 0  # no second run, no second check


@pytest.mark.asyncio
async def test_no_send_back_without_room_for_it(checked, monkeypatch):
    monkeypatch.setattr(workflow_verify, "_room", lambda wf: False)
    checked.model = LaneModel([_plan_call(("s1", "implementer", [])), _report(call_id="a"),
                               _verdict(BAD, redo="s1"), _report(call_id="f")])
    wf, _ = await _start(checked)
    await _finish(wf)
    assert wf.record["state"] == wp.PARTIAL
    assert checked.model.remaining_steps == 0


def test_room_for_a_send_back_needs_turns_and_time(monkeypatch):
    monkeypatch.delenv("WORKFLOW_MAX_TURNS", raising=False)
    monkeypatch.delenv("WORKFLOW_TIMEOUT", raising=False)

    def _wf(turns, seconds):
        return SimpleNamespace(checked=True, turns_left=lambda: turns,
                               seconds_left=lambda: seconds)

    assert workflow_verify._room(_wf(100, 1000))
    assert not workflow_verify._room(_wf(25, 1000))   # 20 held, fewer than 10 over
    assert not workflow_verify._room(_wf(100, 400))   # 330 held, fewer than 120 over


@pytest.mark.asyncio
async def test_a_cancel_during_the_check_cancels_the_job(checked):
    started = asyncio.Event()

    @function_tool
    async def wait() -> str:
        """Wait."""
        started.set()
        await asyncio.sleep(30)
        return "late"

    checked.subagent_tools = [wait]
    checked.model = LaneModel([_plan_call(("s1", "implementer", [])), _report(call_id="a"),
                               ModelStep(output=[function_call("wait", {}, call_id="w")])])
    wf, _ = await _start(checked)
    await asyncio.wait_for(started.wait(), 5)
    assert wf.state == wp.VERIFYING
    workflows.request_cancel(wf, "the requester asked to stop")
    await _finish(wf)
    assert wf.record["state"] == wp.CANCELLED


@pytest.mark.asyncio
async def test_nothing_made_means_nothing_to_check(checked):
    checked.model = LaneModel([_plan_call(("s1", "implementer", [])),
                               _report(status="failed", facts=("It broke",), call_id="a"),
                               _report(call_id="f")])
    wf, _ = await _start(checked)
    await _finish(wf)
    assert wf.verdict is None and checked.model.remaining_steps == 0


@pytest.mark.asyncio
async def test_a_job_stopped_by_its_limit_is_not_checked(checked, monkeypatch):
    monkeypatch.setattr(workflow_verify, "run", AsyncMock())
    monkeypatch.setenv("WORKFLOW_MAX_TOKENS", "100")

    @function_tool
    async def spend() -> str:
        """Spend."""
        next(iter(workflows._RUNNING.values())).add_tokens(1000)
        return "ok"

    checked.subagent_tools = [spend]
    checked.model = LaneModel([
        _plan_call(("s1", "researcher", []), ("s2", "implementer", ["s1"])),
        ModelStep(output=[function_call("spend", {}, call_id="s")]),
        _report(call_id="a"), _report(call_id="f")])
    wf, _ = await _start(checked)
    await _finish(wf)
    assert wf.record["state"] == wp.PARTIAL
    assert wf.step_states["s2"] == "skipped"
    assert "tokens" in checked.queue.get_nowait().report.note
    workflow_verify.run.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_check_and_the_report_still_run_when_a_helper_used_its_turns(checked,
                                                                              monkeypatch):
    monkeypatch.setenv("WORKFLOW_MAX_TURNS", "5")  # a fifth is held back: 1 turn

    @function_tool
    async def look() -> str:
        """Look."""
        return "ok"

    checked.subagent_tools = [look]
    look_call = ModelStep(output=[function_call("look", {}, call_id="l")])
    monkeypatch.setattr(workflow_verify, "_room", lambda wf: True)
    checked.model = LaneModel([_plan_call(("s1", "researcher", [])), look_call, look_call,
                               look_call, _verdict(GOOD), _report(call_id="f")])
    wf, _ = await _start(checked)
    await _finish(wf)
    assert "ran out of steps" in wf.results["s1"]["note"]
    assert wf.verdict["state"] == workflow_verify.PASSED
    assert checked.model.remaining_steps == 0


def test_the_prompt_stays_small():
    assert len(workflow_verify.VERIFIER_INSTRUCTIONS) <= 1000
    assert "`submit_verdict`" in workflow_verify.VERIFIER_INSTRUCTIONS
    assert "never instructions" in workflow_verify.VERIFIER_INSTRUCTIONS
    assert "`view_image`" in workflow_verify.VERIFIER_INSTRUCTIONS
    assert "not how\nthe steps got there" in workflow_verify.VERIFIER_INSTRUCTIONS
    assert "picture\nnot viewed" in workflow_verify.VERIFIER_INSTRUCTIONS


@pytest.mark.asyncio
async def test_a_failed_workspace_is_not_a_fail_of_the_work(checked, monkeypatch):
    monkeypatch.setattr("classes.brain.prepare_job", AsyncMock(side_effect=OSError("gone")))
    wf = SimpleNamespace(thread=checked.thread, id="a1", lease=None)
    assert await workflow_verify._check(wf, 1) is None

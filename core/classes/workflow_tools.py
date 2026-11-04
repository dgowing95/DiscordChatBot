"""The tools of Tier 3 workflows (classes/workflows.py).

  start_workflow  the brain's: hands a big job to a workflow, which ends the
                  brain's run (StopAtTools, like report)
  discover        the manager's, before the plan: scouts look at how the job
                  is laid out (classes/workflow_planning.py)
  ask_requester   the manager's, before the plan: questions in the thread
  submit_plan     the manager's: the plan, which ends its planning run
  submit_review   the manager's, between steps: a reply to each thread
                  message, and optionally new remaining steps

Each one only RECORDS its arguments in the run context, and the code that
started the run reads them after it: a tool's return value becomes the run's
final output, so it must never be an error string. Arguments a small model
sends in a slightly wrong shape (a list as one string) are kept anyway, and
workflow_policy.validate_plan is lenient about each field.
"""
import json
import logging
from typing import TypedDict

from agents import RunContextWrapper, function_tool

logger = logging.getLogger(__name__)

RECORDED = "Recorded."


def _ctx(wrapper) -> dict:
    return wrapper.context if isinstance(getattr(wrapper, "context", None), dict) else {}


def _keeping(key: str):
    """A failure_error_function that keeps arguments which are a JSON object
    but failed the schema (seen with report), so the run still ends."""
    def _failure(ctx, error) -> str:
        raw = str(getattr(ctx, "tool_arguments", "") or "")
        try:
            args = json.loads(raw)
        except ValueError:
            args = None
        if isinstance(args, dict) and isinstance(getattr(ctx, "context", None), dict):
            ctx.context[key] = args
            logger.warning(f"Workflow: {key} arguments did not fit the schema, kept anyway: {error}")
        else:
            logger.warning(f"Workflow: {key} failed: {error}; arguments: {raw[:300]!r}")
        return RECORDED
    return _failure


class StepArg(TypedDict, total=False):
    id: str
    role: str
    brief: str
    depends_on: list[str]


class ReplyArg(TypedDict, total=False):
    id: int
    reply: str


class QuestionArg(TypedDict, total=False):
    question: str
    options: list[str]


@function_tool(failure_error_function=_keeping("workflow_discover_args"))
async def discover(wrapper: RunContextWrapper[dict], subjects: list[str]) -> str:
    """Before planning, find out the SHAPE of the job so your briefs can be
    exact. Name what to look up, and a scout finds where it is and how it is
    laid out: the sources or files, their format and fields. The answers
    themselves are the helpers' job, in the plan's steps. E.g. asked to chart
    the population of the 10 biggest Nordic cities:
    discover(subjects=["Nordic city populations"]). A researcher step then
    collects the figures. Use it only when you cannot write exact briefs
    without it. Call it on its own.

    Args:
        subjects: At most 2, one per scout, a few words each, e.g.
            "UK rail delay statistics" or "sales.csv".
    """
    _ctx(wrapper)["workflow_discover_args"] = {"subjects": subjects}
    return RECORDED


@function_tool(failure_error_function=_keeping("workflow_ask_args"))
async def ask_requester(wrapper: RunContextWrapper[dict], questions: list[QuestionArg]) -> str:
    """Ask the requester only what changes the plan, like a budget or which
    of two outputs they want. Everything else you decide yourself and put in
    the plan's notes. Call it on its own.

    Args:
        questions: At most 3. Each: question, and options (up to 4 short
            choices, shown as buttons) when the answer is one of a few.
    """
    _ctx(wrapper)["workflow_ask_args"] = {"questions": questions}
    return RECORDED


@function_tool(failure_error_function=_keeping("workflow_plan_args"))
async def submit_plan(wrapper: RunContextWrapper[dict], goal: str, steps: list[StepArg],
                      deliverables: list[str] | None = None, notes: str = "") -> str:
    """Submit the plan. Call it once, last, on its own.

    Args:
        goal: What the whole job delivers, in one sentence.
        steps: In order. Each: id ("s1"...), role ("researcher" or
            "implementer"), brief (the helper's whole job: what to find or
            make, what to hand on) and depends_on (ids it needs first).
        deliverables: What the user gets, e.g. "report.md", "a chart".
        notes: Decisions every helper should know.
    """
    _ctx(wrapper)["workflow_plan_args"] = {"goal": goal, "steps": steps,
                                           "deliverables": deliverables, "notes": notes}
    return RECORDED


@function_tool(failure_error_function=_keeping("workflow_review_args"))
async def submit_review(wrapper: RunContextWrapper[dict], replies: list[ReplyArg],
                        steps: list[StepArg] | None = None, stop: bool = False) -> str:
    """Answer the thread's messages. Call it once, on its own.

    Args:
        replies: One per message: its id and a short reply to post.
        steps: Only if a message changes the job: the new list of the steps
            NOT done yet (same fields as in the plan).
        stop: True only if the requester asks to stop the whole job.
    """
    _ctx(wrapper)["workflow_review_args"] = {"replies": replies, "steps": steps, "stop": stop}
    return RECORDED


@function_tool
async def start_workflow(wrapper: RunContextWrapper[dict], request: str, reason: str,
                         draft_steps: list[str] | None = None) -> str:
    """Hand a big job to a workflow: a manager plans it, the user approves
    the plan, and helpers do the steps in a thread. It ends your part; the
    user is told where it runs.

    Args:
        request: The whole job, with every detail the user gave.
        reason: Why it is too big for you alone, in one sentence.
        draft_steps: Optional: the steps you would take, one line each.
    """
    from classes import workflows

    ctx = _ctx(wrapper)
    trigger = workflows.TRIGGER_AUTOMATION if ctx.get("automatic") else workflows.TRIGGER_BRAIN
    result = await workflows.start_workflow(ctx, request, trigger=trigger, reason=reason,
                                            draft=draft_steps)
    if result.ok:
        ctx["brain_workflow"] = {"thread_url": result.thread_url}
        ctx["workflow_url"] = result.thread_url  # outlives the brain call, for automations
        ctx["brain_report_args"] = {"status": "done", "facts": [f"Started a workflow: {request}"]}
    else:
        ctx["brain_workflow"] = {"thread_url": ""}
        ctx["brain_report_args"] = {"status": "failed", "facts": [result.text]}
    return RECORDED

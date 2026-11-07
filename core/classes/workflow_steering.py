"""The requester's words reaching the helpers of a running workflow.

Only theirs: the thread is open to the server, and only they can change the
job. Every thread message still waits for the manager's review between steps
(classes/workflows.py), which answers it: this only gets it to the work sooner.

Why it works this way:

- **Each message goes to the steps it is about** (maintainer, 2026-11-04). "Make
  the graph green" sent to a helper writing a report only confuses it. One quick
  call (route) picks them from the steps not finished yet, so that covers helpers
  that start later too. With one step left there is nothing to choose, and when
  the call fails the message goes to all of them, as better than to none.
- **At the next model call, by an input filter** (steered). A helper in a long
  shell command sees it when that ends. The SDK rebuilds the input for every call
  and keeps nothing a filter added, so each message goes back in on every later
  call, where it was first shown: in order, and the model's cached prompt stays
  the same. It goes after the trimming filter, which so never sees it.
- **Nothing makes a helper follow it.** The check of the finished work
  (classes/workflow_verify.py) is what checking there is.
- **New text passes the content guard first**: the guard in the run context only
  ever saw the request.
"""
import dataclasses
import json
import logging
import re
from dataclasses import replace

from classes import llm_lanes, workflows
from classes import workflow_policy as wp
from classes.llm_roles import BRAIN, role_config
from classes.response_filter import strip_thinking
from classes.side_llm import SideLLM

logger = logging.getLogger(__name__)

STEERING_CHARS = 1000
ROUTE_TIMEOUT_SECONDS = 20
ROUTE_MAX_TOKENS = 300

# What a helper is told the requester said. ONE home.
STEERING_NOTE = ("The requester said in the thread while the job ran (follow it where it applies "
                 "to your step): ")

# ONE home; its size is pinned (workflows_tests.py). The shape is shown.
ROUTE_PROMPT = """A job runs in steps, and the requester sent a message while it runs. Name the steps whose work it changes, so only those helpers get it. E.g. with s2 making a chart and s3 writing a report, "make the graph green" is for s2 alone. A message about the whole job names every step, and one that changes no step names none.
Answer with JSON only: {"steps": ["s2"]}"""

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _config():
    return dataclasses.replace(role_config(BRAIN), call_timeout=ROUTE_TIMEOUT_SECONDS)


_llm = SideLLM("workflow_route", _config)


def open_steps(wf) -> list:
    """The steps a message can still change: running, or not started yet."""
    if wf.plan is None:
        return []
    return [step for step in wf.plan.steps
            if wf.step_states.get(step.id) == "running" or step.id not in wf.results]


async def route(wf, text: str) -> set | None:
    """The ids of the steps `text` is about, or None for all of them."""
    steps = open_steps(wf)
    if len(steps) < 2:
        return None
    lines = "\n".join(f"{step.id} [{step.role}]: {step.brief}" for step in steps)
    llm_lanes.current_lane.set(llm_lanes.BRAIN)
    try:
        answer = await _llm.complete(
            [{"role": "system", "content": ROUTE_PROMPT},
             {"role": "user", "content": f"Steps:\n{lines}\n\nMessage: {text}"}],
            temperature=0.0, max_tokens=ROUTE_MAX_TOKENS)
        match = _JSON_OBJECT_RE.search(strip_thinking(answer or ""))
        return wp.clean_route(json.loads(match.group(0)) if match else None,
                              [step.id for step in steps])
    except Exception as e:
        logger.warning(f"Workflow {wf.id}: could not route a message, so all steps get it: {e!r}")
        return None


async def steer(wf, event) -> None:
    """A message from the requester, on its way to the steps it is about. One
    that is blocked here is blocked again by the review, which says so."""
    if not await workflows._allowed(wf, event.text):
        return
    text = event.text[:STEERING_CHARS]
    targets = await route(wf, text)
    wf.routes[event.seq] = targets
    wf.steering.append((event.seq, text, targets))
    logger.info(f"Workflow {wf.id}: a message goes to "
                f"{'every step' if targets is None else sorted(targets) or 'no step'}")


def _for_step(targets, step_id: str | None) -> bool:
    return targets is None or step_id is None or step_id in targets


def steered(run_config, wf, start: int, step_id: str):
    """The run's input filter, plus what is for this step from `start` in
    wf.steering on, each as its own user message. Never raises: the SDK fails
    the whole run on a filter error, and a run is worth more than one message."""
    from agents.run_config import ModelInputData
    inner = run_config.call_model_input_filter
    placed: list[tuple[int, str]] = []  # where each was first shown, and its text
    read = start

    def _filter(data):
        nonlocal read
        sent = inner(data)
        try:
            items = list(sent.input)
            for _seq, text, targets in wf.steering[read:]:
                read += 1
                if _for_step(targets, step_id):
                    placed.append((len(items), text))
            if not placed:
                return sent
            merged = []
            for index in range(len(items) + 1):
                merged.extend({"role": "user", "content": STEERING_NOTE + text}
                              for where, text in placed if min(where, len(items)) == index)
                merged.extend(items[index:index + 1])
            return ModelInputData(input=merged, instructions=sent.instructions)
        except Exception as e:
            logger.warning(f"Workflow {wf.id}: could not pass a message to a helper: {e!r}")
            return sent

    return replace(run_config, call_model_input_filter=_filter)


def said_text(wf, step_id: str | None = None) -> str:
    """What the requester said while the job ran, for a run starting now: what
    the review answered, and what it has yet to. `step_id`: only what is for
    that step (the check reads it all)."""
    who = wf.record.get("requester_name") or "The requester"
    answered = [line for seq, line in wf.said
                if _for_step(wf.routes.get(seq), step_id)][-10:]
    waiting = [f"{who}: {text}" for seq, text, targets in wf.steering
               if seq not in wf.reviewed and _for_step(targets, step_id)]
    lines = answered + waiting
    if not lines:
        return ""
    return f"{STEERING_NOTE.rstrip()}\n" + "\n".join(f"- {line}" for line in lines)

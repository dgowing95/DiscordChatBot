"""A workflow's plan stage: everything before the requester sees a plan.

The manager (classes/workflows.py) works in short rounds. Each is a fresh run
that ends at one tool, and each sees what the last ones learned:

  discover       up to 2 scouts look at how the job is laid out
  ask_requester  up to 3 questions in the thread, one at a time
  submit_plan    the plan; ends the stage

Why it works this way:

- **Rounds, not tools inside one run.** Code does each round, then takes its
  tool away, so there is at most one of each however a small model behaves.
  It also fits "the manager is short runs", and a usual order, discover then
  ask, lets the questions build on what was found.
- **Discovery is not research** (maintainer, 2026-11-04). Scouts find out what
  exists and how it is laid out, so the briefs can be exact; the answers stay
  with the plan's helpers. Asked for populations, a scout finds the table and
  its columns, never the figures. The manager only names a subject, and code
  writes the question (wp.scout_question), since the answers that came back
  live were the ones it had asked for. Prompts say the rest
  (SCOUT_INSTRUCTIONS, the discover tool's docstring), and since a small
  model drifts, code holds it to a quick look: few scouts, few turns, little
  time, short findings. The manager has no web tools of its own for the same
  reason.
- **A scout at its limit still reports.** It is told to finish two turns
  before the end, and one reaching a limit anyway gets one last call that can
  only be `report` (workflows._last_report). 8 of 18 scouts lost everything
  they found before this (live, 2026-11-04).
- **Code passes on what was learned** (workflows.planning_text): every helper,
  every revision and the finish get the findings and the answers, so nothing
  depends on the manager copying them into a brief.
- **No workspace while waiting on a person.** A scout needs the shell to see
  the shape of a file from the chat, so the lease is held for the discovery
  round alone, never during questions or the approval wait.
- **The requester is told a plan is coming.** The persona's start reply and
  /task's message always say so. Discovery can take minutes, so it also leaves
  a note in the thread. Only the plan and a question ping them.
- **What the requester writes while the plan is being written** goes into the
  next round (_take_messages), rather than costing a plan edit afterwards.
"""
import asyncio
import logging

from classes import brain_question, brain_result, llm_lanes, workflows
from classes import workflow_policy as wp
from classes.brain_result import BrainReport
from classes.tool_embeds import trim

logger = logging.getLogger(__name__)

DISCOVER = "discover"
ASK = "ask_requester"

# Turns per plan run: it has little to look at but the skills.
PLAN_TURNS = 8
SKILLS_HEADER = "Skills saved in this server (how jobs were done here before):"
_ARGS = ("workflow_plan_args", "workflow_discover_args", "workflow_ask_args")

DISCOVERY_NOTE = "🔎 Checking a few things before I write the plan."
PLAN_PING_NOTE = " I'll ping you here when it's ready to approve."

_ROUND_LINES = {
    DISCOVER: ("If you cannot write exact briefs yet, `discover` first. It shows how the job is "
               "laid out; the plan's steps still do the work."),
    ASK: "Use `ask_requester` only if an answer would change the plan.",
}


async def write_plan(wf, change: str = "") -> wp.Plan:
    """The manager's plan (or a revised one), after any rounds it asks for:
    validated, retried once with its problems, else the one-step fallback."""
    problems: list[str] = []
    tries = 0
    while tries < 2 and not wf.cancel_reason:
        await _take_messages(wf)
        rounds = _rounds_left(wf)
        for key in _ARGS:
            wf.ctx.pop(key, None)
        outcome = await workflows._agent_run(
            wf, workflows._manager("manager", rounds), await _task(wf, change, problems, rounds),
            max_turns=PLAN_TURNS, lane=llm_lanes.BRAIN, workspace=False, caller="workflow_manager")
        if wf.cancel_reason:
            break
        raw = wf.ctx.get("workflow_plan_args")
        if raw is None and await _round(wf, rounds):
            continue
        tries += 1
        plan, problems = wp.validate_plan(raw, max_steps=wp.max_steps(),
                                          request=wf.record["request"])
        if plan is not None:
            workflows._store_plan(wf, plan)
            return plan
        if outcome.stopped:
            problems = problems or [outcome.stopped]
        logger.info(f"Workflow {wf.id}: plan not usable: {problems}")
    if change and wf.plan is not None:
        return wf.plan  # a failed revision keeps the plan there was
    plan = wp.fallback_plan(wf.record["request"])
    workflows._store_plan(wf, plan)
    return plan


def _rounds_left(wf) -> tuple:
    """Which rounds the manager may still ask for. An automatic run has
    nobody to ask."""
    rounds = [] if wf.scouted else [DISCOVER]
    if not wf.asked and not wf.record["automatic"] and wf.ledger is not None:
        rounds.append(ASK)
    return tuple(rounds)


async def _round(wf, rounds: tuple) -> bool:
    """Runs the round the manager ended on. False when it asked for none."""
    scouts, questions = wf.ctx.get("workflow_discover_args"), wf.ctx.get("workflow_ask_args")
    ran = False
    if scouts is not None and DISCOVER in rounds:
        await _discover(wf, scouts)
        ran = True
    if questions is not None and ASK in rounds and not wf.cancel_reason:
        await _ask(wf, questions)
        ran = True
    return ran


async def _task(wf, change: str, problems: list, rounds: tuple) -> str:
    from classes.brain import _skills_index
    from classes.tool_functions import get_current_datetime

    parts = ["Your job now: write the plan with `submit_plan`.", workflows._request_block(wf)]
    if rounds:
        parts.append(" ".join(_ROUND_LINES[name] for name in rounds) + " You can use each once.")
    if wf.record.get("draft") and wf.plan is None:
        parts.append("A first idea of the steps (yours to change):\n"
                     + "\n".join(f"- {d}" for d in wf.record["draft"]))
    if wf.record.get("inputs"):
        parts.append(wf.record["inputs"])
    learned = workflows.planning_text(wf)
    if learned:
        parts.append(learned + "\nThe helpers get this too. Their steps still do the work.")
    if wf.said:
        parts.append("The requester said while you planned:\n"
                     + "\n".join(f"- {line}" for _seq, line in wf.said[-10:]))
    if change and wf.plan is not None:
        parts.append("Your plan so far:\n" + wp.plan_text(wf.plan))
        parts.append(f"The requester asked to change it: {change}\nWrite the new plan.")
    index = await _skills_index(wf.guild_id, header=SKILLS_HEADER)
    if index:
        parts.append(index)
    parts.append(f"At most {wp.max_steps()} steps.")
    parts.append(f"Current datetime: {await get_current_datetime()}")
    if problems:
        parts.append("Your last plan could not be used:\n"
                     + "\n".join(f"- {p}" for p in problems)
                     + "\nFix that and call `submit_plan` again.")
    return "\n\n".join(parts)


async def _take_messages(wf) -> None:
    """The requester's thread messages so far go into the next plan run. (A
    message that comes during the last run is an edit: workflows._approve.)"""
    if wf.ledger is None:
        return
    for event in wf.ledger.unpresented():
        if not event.is_requester or wp.is_stop_word(event.text):
            continue
        wf.ledger.mark_presented([event.seq])
        if await workflows._allowed(wf, event.text):
            wf.ledger.record_decision([event.seq], "noted", "Taken into the plan.")
            wf.said.append((event.seq, f"{event.author_name}: {event.text[:300]}"))
        else:
            wf.ledger.record_decision([event.seq], "declined", "Blocked by the content guard.")
            await workflows._post(wf.thread, content="I can't act on that message.")


# --- discovery ---------------------------------------------------------------------


async def _discover(wf, raw) -> None:
    """The discovery round: the note in the thread, then the scouts, side by
    side. Never raises: without it the plan rests on the request alone."""
    from classes import brain

    wf.scouted = True
    subjects = wp.clean_scout_subjects(raw)
    if not subjects or wf.cancel_reason:
        return
    await workflows._post(wf.thread, content=DISCOVERY_NOTE + (
        "" if wf.record["automatic"] else PLAN_PING_NOTE))
    try:
        async with brain._acquire_workspace(wf.guild_id) as lease:
            wf.lease = lease
            try:
                limit = workflows._helpers_at_once()
                for start in range(0, len(subjects), limit):
                    if wf.cancel_reason:  # waiting for the workspace is not a run to stop
                        break
                    await workflows._together(wf, [
                        _scout(wf, number, subject) for number, subject
                        in enumerate(subjects[start:start + limit], start + 1)])
            finally:
                wf.lease = None
    except Exception as e:
        logger.warning(f"Workflow {wf.id}: discovery could not run: {e}")
        if not wf.findings:
            wf.findings.append("Discovery could not run, so this rests on the request alone.")
    wf.record["findings"] = list(wf.findings)


async def _scout(wf, number: int, subject: str) -> None:
    """One scout: its own folder and context, a quick look, a few facts."""
    from classes.brain import prepare_job
    from classes.brain_progress import HelperProgress
    from classes.common import QuietChannel

    job_id = f"c{wf.thread.id}-w{wf.id}-d{number}"
    job_dir, out_dir, _marker = await prepare_job(wf.lease, job_id)
    wf.scout_jobs.append(job_id)
    ctx = workflows._helper_ctx(wf, session=wf.lease.session, workspace_root=wf.lease.root,
                                job_dir=job_dir, out_dir=out_dir,
                                channel=QuietChannel(wf.thread))
    progress = HelperProgress(wf.thread, f"🔎 {subject}", max_turns=wp.SCOUT_MAX_TURNS)
    await progress.start()
    try:
        await _scout_work(wf, subject, ctx, job_dir, progress)
    except asyncio.CancelledError:
        await workflows.close_progress(progress)
        raise
    except Exception:
        await progress.finalize(ok=False, note=f"⚠️ {workflows.HELPER_FAILED}")
        raise


async def _scout_work(wf, subject: str, ctx: dict, job_dir: str, progress) -> None:
    from classes import metrics
    from classes.tool_functions import get_current_datetime

    parts = [f"The job being planned (a helper does it later): {wf.record['request']}",
             f"Find out: {wp.scout_question(subject)}"]
    if workflows.asked_in(wf):
        parts.append(workflows.asked_in(wf))
    if wf.record.get("inputs"):
        parts.append(wf.record["inputs"])
    parts.append(f"Your folder (the working directory, already open): {job_dir}/")
    parts.append(f"Current datetime: {await get_current_datetime()}")
    agent = workflows._subagent(wp.SCOUT)
    outcome = await workflows._agent_run(
        wf, agent, "\n\n".join(parts), max_turns=wp.SCOUT_MAX_TURNS,
        lane=llm_lanes.SUBAGENT, workspace=True, progress=progress, caller="workflow_scout",
        ctx=ctx, seconds=wp.SCOUT_SECONDS,
        finish_up_at=wp.SCOUT_MAX_TURNS - wp.SCOUT_FINISH_UP_TURNS)
    workflows._keep_shared(wf, ctx)
    args = ctx.get("brain_report_args")
    if args is None and outcome.at_limit:
        args = await workflows._last_report(wf, agent, outcome.seen, llm_lanes.SUBAGENT,
                                            wp.SCOUT_LAST_REPORT_SECONDS)
    if args is not None:
        report = brain_result.parse_report_args(args)
    elif outcome.stopped is None:
        report = brain_result.report_from_text(outcome.final_text)
    else:
        report = BrainReport(status=brain_result.PARTIAL)
    facts = [trim(fact, wp.SCOUT_FACT_CHARS) for fact in report.facts[:wp.SCOUT_FACTS]]
    if facts:
        wf.findings.append(f"{subject}: {' '.join(facts)}")
    elif outcome.stopped is None:
        wf.findings.append(f"{subject}: nothing found.")
    await progress.finalize(ok=outcome.stopped is None,
                            note=None if outcome.stopped is None else f"⚠️ {outcome.stopped}",
                            found=len(facts))
    metrics.inc_workflow_scout("done" if outcome.stopped is None
                               else "reported_at_limit" if facts else "stopped")


# --- questions ---------------------------------------------------------------------


async def _ask(wf, raw) -> None:
    """The question round, one at a time. The first with no answer ends it:
    the requester has gone, and the rest would only hold up the plan."""
    from classes import metrics

    wf.asked = True
    for question, options in wp.clean_plan_questions(raw):
        answer = await workflows.ask_requester(wf, question, options,
                                               brain_question.HEADING_PLAN)
        if wf.cancel_reason:
            return
        metrics.inc_workflow_plan_question("answered" if answer else "unanswered")
        if not answer:
            wf.answers.append((question, "(no answer: decide, and say so in the notes)"))
            break
        wf.answers.append((question, answer))
    wf.record["answers"] = [list(pair) for pair in wf.answers]

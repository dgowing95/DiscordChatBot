"""Tier 3 workflows: plans, states and limits. PURE (stdlib only).

A workflow is a job too big for one inline brain call (classes/workflows.py).
It runs in its own thread: a manager writes a plan, the requester approves it,
subagents do the steps (researchers side by side: next_wave), and the
persona replies in the channel when it ends.

Everything here is a rule the tests can check without Discord or the SDK:
what a valid plan is, the order steps run in, how a plan reads in an embed,
which state moves are legal, how much of the budget is left, and the
WORKFLOW_* / SUBAGENT_* settings.

The limits are caps on cost (maintainer, 2026-11-03: the biggest job seen,
a 10+ source report with a chart, took 62 min and 238 turns). WORKFLOW_TIMEOUT and WORKFLOW_MAX_TURNS stop
anything bigger, and running out stops it cleanly with what exists
(PARTIAL), never "try again".
"""
import os
import re
from dataclasses import dataclass, field

# --- settings ----------------------------------------------------------------

DEFAULT_ASK_TIMEOUT_SECONDS = 300
DEFAULT_PLAN_APPROVAL_SECONDS = 900
DEFAULT_PLAN_REVISIONS = 3
DEFAULT_MAX_STEPS = 8
DEFAULT_TIMEOUT_SECONDS = 3600
DEFAULT_MAX_TURNS = 300
# Live jobs used 0.5-2.6M tokens; a runaway one 5.1M (maintainer, 2026-11-07).
DEFAULT_MAX_TOKENS = 4_000_000
DEFAULT_MAX_PER_GUILD = 1
DEFAULT_SUBAGENT_MAX_TURNS = 100
DEFAULT_SUBAGENT_MAX_CONCURRENT = 2
# Pictures one workflow may ask for (a chat reply gets 3). Not a setting.
MAX_IMAGES = 10
# What keeps discovery a quick look (classes/workflow_planning.py). Not
# settings: past these it would be research, which is the steps' job.
MAX_SCOUTS = 2
SCOUT_MAX_TURNS = 12
SCOUT_SECONDS = 300
SCOUT_FINISH_UP_TURNS = 2     # turns before the limit it is told to report
SCOUT_LAST_REPORT_SECONDS = 90
SCOUT_FACTS = 3
SCOUT_FACT_CHARS = 400        # uncut they ran about 350 live, and 240 cut 27 of 28 findings
MAX_SUBJECT_CHARS = 120
MAX_FINDINGS_CHARS = 2700      # room for two scouts' whole findings, so nothing is cut twice
MAX_PLAN_QUESTIONS = 3

# Turns kept from the helpers for the check and the final report. A job set
# to few turns keeps a fifth instead, so its helpers can still work.
TURN_RESERVE = 40

# What one check of the finished work may say (classes/workflow_verify.py).
MAX_CHECKS = 12
MAX_CHECK_NOTE_CHARS = 300


def _env_int(name: str, default: int, env) -> int:
    """A positive int setting; anything else is the default. (The _env
    prefix is what settings_docs_tests looks for.)"""
    try:
        value = int(str(env.get(name)).strip())
        return value if value > 0 else default
    except (TypeError, ValueError):
        return default


def _env_count(name: str, default: int, env) -> int:
    """Like _env_int, but 0 is kept: it means off."""
    try:
        value = int(str(env.get(name)).strip())
        return value if value >= 0 else default
    except (TypeError, ValueError):
        return default


def _env_flag(name: str, default: bool, env) -> bool:
    raw = str(env.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw not in ("0", "false", "no", "off")


def enabled(env=os.environ) -> bool:
    """WORKFLOW_ENABLED (default on): off removes start_workflow and /task."""
    return _env_flag("WORKFLOW_ENABLED", True, env)


def from_automations(env=os.environ) -> bool:
    """WORKFLOW_FROM_AUTOMATIONS (default on): schedules, rules and
    follow-ups may start one, approved at once."""
    return _env_flag("WORKFLOW_FROM_AUTOMATIONS", True, env)


def ask_timeout(env=os.environ) -> int:
    """Seconds a question in the thread waits for an answer
    (WORKFLOW_ASK_TIMEOUT, default 300)."""
    return _env_int("WORKFLOW_ASK_TIMEOUT", DEFAULT_ASK_TIMEOUT_SECONDS, env)


def plan_approval_seconds(env=os.environ) -> int:
    """Seconds the plan waits for Approve before it starts anyway
    (WORKFLOW_PLAN_APPROVAL_SECONDS, default 900)."""
    return _env_int("WORKFLOW_PLAN_APPROVAL_SECONDS", DEFAULT_PLAN_APPROVAL_SECONDS, env)


def plan_revisions(env=os.environ) -> int:
    """Edits a plan may get before it must be approved or cancelled
    (WORKFLOW_PLAN_REVISIONS, default 3)."""
    return _env_int("WORKFLOW_PLAN_REVISIONS", DEFAULT_PLAN_REVISIONS, env)


def max_steps(env=os.environ) -> int:
    """Steps one plan may have (WORKFLOW_MAX_STEPS, default 8)."""
    return _env_int("WORKFLOW_MAX_STEPS", DEFAULT_MAX_STEPS, env)


def timeout(env=os.environ) -> int:
    """Wall-clock seconds one workflow may take (WORKFLOW_TIMEOUT, 1 h)."""
    return _env_int("WORKFLOW_TIMEOUT", DEFAULT_TIMEOUT_SECONDS, env)


def max_turns(env=os.environ) -> int:
    """Model turns across every agent of one workflow (WORKFLOW_MAX_TURNS,
    default 300)."""
    return _env_int("WORKFLOW_MAX_TURNS", DEFAULT_MAX_TURNS, env)


def turn_reserve(env=os.environ) -> int:
    """The turns helpers leave for the check and the final report."""
    return min(TURN_RESERVE, max_turns(env) // 5)


def max_tokens(env=os.environ) -> int:
    """Tokens, in and out, across every agent of one workflow
    (WORKFLOW_MAX_TOKENS) [4,000,000]. 0 is off."""
    return _env_count("WORKFLOW_MAX_TOKENS", DEFAULT_MAX_TOKENS, env)


def max_per_guild(env=os.environ) -> int:
    """Workflows running at once in one server (WORKFLOW_MAX_PER_GUILD,
    default 1)."""
    return _env_int("WORKFLOW_MAX_PER_GUILD", DEFAULT_MAX_PER_GUILD, env)


def subagent_max_turns(env=os.environ) -> int:
    """Model turns one subagent may take (SUBAGENT_MAX_TURNS, default 100)."""
    return _env_int("SUBAGENT_MAX_TURNS", DEFAULT_SUBAGENT_MAX_TURNS, env)


def subagent_max_concurrent(env=os.environ) -> int:
    """Researchers running at once (SUBAGENT_MAX_CONCURRENT, default 2)."""
    return _env_int("SUBAGENT_MAX_CONCURRENT", DEFAULT_SUBAGENT_MAX_CONCURRENT, env)


def dev_inject(env=os.environ) -> bool:
    """WORKFLOW_DEV_INJECT (local testing and E2E, never in the chart):
    "!workflow <request>" starts one (main.workflow_dev_inject)."""
    return _env_flag("WORKFLOW_DEV_INJECT", False, env)


# --- plans -------------------------------------------------------------------

RESEARCHER = "researcher"
IMPLEMENTER = "implementer"
ROLES = (RESEARCHER, IMPLEMENTER)
# Runs before the plan, so it is never a step's role.
SCOUT = "scout"
VERIFIER = "verifier"
# Loose role names a small model writes, mapped onto ROLES.
_ROLE_ALIASES = {
    "research": RESEARCHER, "researcher": RESEARCHER, "reader": RESEARCHER,
    "search": RESEARCHER, "analyst": RESEARCHER,
    "implement": IMPLEMENTER, "implementer": IMPLEMENTER, "builder": IMPLEMENTER,
    "build": IMPLEMENTER, "coder": IMPLEMENTER, "writer": IMPLEMENTER, "developer": IMPLEMENTER,
}

MAX_GOAL_CHARS = 500
MAX_BRIEF_CHARS = 1500
MAX_DELIVERABLE_CHARS = 200
MAX_DELIVERABLES = 10
MAX_NOTES_CHARS = 1000
MAX_QUESTION_CHARS = 300
MAX_TITLE_CHARS = 80  # a plan's or a step's title: a few words


@dataclass
class Step:
    id: str
    role: str
    brief: str
    depends_on: list[str] = field(default_factory=list)
    # A few words naming the step on its cards ("Find 2026 reviews"); the
    # brief is the whole job. Empty in plans stored before titles existed.
    title: str = ""


@dataclass
class Plan:
    goal: str
    steps: list[Step]
    deliverables: list[str] = field(default_factory=list)
    notes: str = ""
    title: str = ""  # a few words naming the whole job

    def to_dict(self) -> dict:
        return {"goal": self.goal, "notes": self.notes, "deliverables": list(self.deliverables),
                "title": self.title,
                "steps": [{"id": s.id, "role": s.role, "brief": s.brief, "title": s.title,
                           "depends_on": list(s.depends_on)} for s in self.steps]}


def _text(value, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def clean_step_id(value) -> str:
    """A step's id, safe in a folder name (workflows.step_job_id): a model
    writes it, and "../shared" must not reach a path."""
    text = "".join(str(value or "").split())
    return re.sub(r"[^A-Za-z0-9_-]+", "-", text)[:20].strip("-")


# The folders beside a step's that are not a step's: scouts' (d1), the end
# checks' (v1) and the research checks' (r1). A step named so would share one.
_OTHER_FOLDERS = re.compile(r"[dvr]\d+", re.IGNORECASE)


def plan_step_id(value) -> str:
    """A plan's step id: clean_step_id, and never one of _OTHER_FOLDERS
    ("d1" becomes "sd1"). Its depends_on entries go through here too."""
    step_id = clean_step_id(value)
    return f"s{step_id}" if _OTHER_FOLDERS.fullmatch(step_id) else step_id


def _list(value) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        return [part for part in re.split(r"[,\n]", value) if part.strip()]
    return list(value) if isinstance(value, (list, tuple)) else [value]


def validate_plan(raw, *, max_steps: int, request: str = "") -> tuple["Plan | None", list[str]]:
    """A Plan from the manager's submit_plan arguments, and the problems that
    stop it from running. Lenient where a small model is loose (role names,
    a list sent as a string, missing ids); strict where running it would go
    wrong (no steps, too many, unknown or circular dependencies)."""
    problems: list[str] = []
    if not isinstance(raw, dict):
        return None, ["The plan is not a JSON object."]
    goal = _text(raw.get("goal") or request, MAX_GOAL_CHARS)
    if not goal:
        problems.append("The plan has no goal.")
    raw_steps = raw.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        return None, problems + ["The plan has no steps."]
    if len(raw_steps) > max_steps:
        problems.append(f"The plan has {len(raw_steps)} steps; at most {max_steps} are allowed.")
    steps: list[Step] = []
    for index, item in enumerate(raw_steps, 1):
        if not isinstance(item, dict):
            problems.append(f"Step {index} is not an object.")
            continue
        step_id = plan_step_id(item.get("id")) or f"s{index}"
        if any(s.id == step_id for s in steps):
            problems.append(f"Two steps have the id {step_id}.")
            continue
        role = _ROLE_ALIASES.get(str(item.get("role") or "").strip().lower())
        if role is None:
            problems.append(f"Step {step_id} has the role {item.get('role')!r}; use "
                            f"{RESEARCHER} or {IMPLEMENTER}.")
            continue
        brief = _text(item.get("brief") or item.get("task"), MAX_BRIEF_CHARS)
        if not brief:
            problems.append(f"Step {step_id} has no brief.")
            continue
        depends = []
        for dep in _list(item.get("depends_on")):
            if plan_step_id(dep):
                depends.append(plan_step_id(dep))
            elif str(dep or "").strip():  # a blank entry is no dependency at all
                problems.append(f"Step {step_id} depends on {str(dep)[:40]!r}, which is not a "
                                "usable step id.")
        steps.append(Step(id=step_id, role=role, brief=brief, depends_on=depends,
                          title=_text(item.get("title"), MAX_TITLE_CHARS)))
    ids = {s.id for s in steps}
    for step in steps:
        for dep in step.depends_on:
            if dep == step.id:
                problems.append(f"Step {step.id} depends on itself.")
            elif dep not in ids:
                problems.append(f"Step {step.id} depends on {dep}, which is not a step.")
    if not problems and _has_cycle(steps):
        problems.append("The steps depend on each other in a circle.")
    deliverables = [_text(d, MAX_DELIVERABLE_CHARS) for d in _list(raw.get("deliverables"))]
    plan = Plan(goal=goal, steps=steps, deliverables=[d for d in deliverables if d][:MAX_DELIVERABLES],
                notes=_text(raw.get("notes"), MAX_NOTES_CHARS),
                title=_text(raw.get("title"), MAX_TITLE_CHARS))
    return (None, problems) if problems else (plan, [])


def _has_cycle(steps) -> bool:
    by_id = {s.id: s for s in steps}
    state: dict[str, int] = {}  # 1 = on the path, 2 = done

    def visit(step_id: str) -> bool:
        if state.get(step_id) == 1:
            return True
        if state.get(step_id) == 2:
            return False
        state[step_id] = 1
        if any(visit(d) for d in by_id[step_id].depends_on if d in by_id):
            return True
        state[step_id] = 2
        return False

    return any(visit(s.id) for s in steps)


def order_steps(plan: Plan) -> list[Step]:
    """The steps in the order they are taken up: each after the steps it
    depends on, otherwise in the plan's own order."""
    done: set[str] = set()
    ordered: list[Step] = []
    remaining = list(plan.steps)
    while remaining:
        ready = next((s for s in remaining if all(d in done for d in s.depends_on)), None)
        if ready is None:  # a cycle validate_plan missed: plan order
            ready = remaining[0]
        remaining.remove(ready)
        ordered.append(ready)
        done.add(ready.id)
    return ordered


def next_wave(queue: list[Step], done, limit: int) -> list[Step]:
    """The steps to run next, from `queue` (those not run yet, in order) and
    `done` (the ids of those that have). Researchers only read, so the ready
    ones run together, up to `limit`. An implementer runs alone: two of them
    would build on each other's half-made files (Cognition's rule)."""
    ready = [s for s in queue if all(d in done for d in s.depends_on)]
    if not ready:  # a cycle validate_plan missed: plan order
        return queue[:1]
    if ready[0].role != RESEARCHER:
        return ready[:1]
    return [s for s in ready if s.role == RESEARCHER][:max(1, limit)]


def clean_scout_subjects(raw) -> list[str]:
    """The manager's discover arguments as the scouts' subjects: text only,
    at most MAX_SCOUTS. A small model may send one string."""
    items = raw.get("subjects") if isinstance(raw, dict) else raw
    # One string is one subject: _list would split it at its commas.
    items = [items] if isinstance(items, str) else _list(items)
    texts = [_text(item, MAX_SUBJECT_CHARS) for item in items]
    return [text for text in texts if text][:MAX_SCOUTS]


def scout_question(subject: str) -> str:
    """Always this question, written here and not by the manager: asked in its
    own words it asked for the answers ("what is the final table?"), and the
    scouts duly brought them (3 of 10 findings live, 2026-11-04)."""
    return f"where is {subject} published or kept, and how is it laid out?"


def clean_plan_questions(raw) -> list[tuple[str, list]]:
    """The manager's ask_requester arguments as (question, options), at most
    MAX_PLAN_QUESTIONS. A question may come as plain text, without options."""
    items = raw.get("questions") if isinstance(raw, dict) else raw
    if isinstance(items, (dict, str)):
        items = [items]  # one question, whatever commas it has
    questions = []
    for item in _list(items):
        options = _list(item.get("options")) if isinstance(item, dict) else []
        text = _text(item.get("question") if isinstance(item, dict) else item, MAX_QUESTION_CHARS)
        if text:
            questions.append((text, options))
    return questions[:MAX_PLAN_QUESTIONS]


def findings_text(findings: list[str]) -> str:
    """What the scouts found, as lines, within MAX_FINDINGS_CHARS in all."""
    lines = [f"- {line}" for line in findings]
    share = MAX_FINDINGS_CHARS // max(1, len(lines))
    return "\n".join(line if len(line) <= share else line[: share - 1] + "…" for line in lines)


def clean_verdict(raw) -> dict | None:
    """A verifier's submit_verdict arguments as {"checks": [{"deliverable",
    "ok", "note"}], "redo_step"}. None when nothing was really checked: that
    is "not checked", never a pass."""
    if not isinstance(raw, dict):
        return None
    checks = []
    for item in _list(raw.get("checks"))[:MAX_CHECKS]:
        ok = str(item.get("ok")).strip().lower() if isinstance(item, dict) else ""
        if ok in ("true", "false"):
            checks.append({"deliverable": _text(item.get("deliverable"), MAX_DELIVERABLE_CHARS),
                           "ok": ok == "true",
                           "note": _text(item.get("note"), MAX_CHECK_NOTE_CHARS)})
    if not checks:
        return None
    return {"checks": checks, "redo_step": clean_step_id(raw.get("redo_step"))}


def verdict_failed(verdict: dict | None) -> bool:
    return verdict is not None and not all(check["ok"] for check in verdict["checks"])


def verdict_notes(verdict: dict | None) -> str:
    """The failed checks, one per line."""
    if verdict is None:
        return ""
    return "\n".join(f"- {check['deliverable']}: {check['note']}".rstrip(": ")
                     for check in verdict["checks"] if not check["ok"])


def redo_target(plan: Plan, results: dict, wanted: str) -> Step | None:
    """The step to run again: the one asked for if it ran, else the last
    implementer that ran, else the last step that ran."""
    ran = [step for step in plan.steps if step.id in results]
    for step in ran:
        if step.id == wanted:
            return step
    built = [step for step in ran if step.role == IMPLEMENTER]
    return (built or ran or [None])[-1]


def redo_chain(plan: Plan, results: dict, step_id: str, waves: dict) -> list[Step]:
    """`step_id` and every step that ran on what it made, in the order they
    are taken up: a fixed fact has to reach the file that used it. That is
    each step that depends on one in the chain, and every implementer that
    ran after it: a helper reads all the earlier steps' reports, so a plan
    that left a depends_on out still made its file from them. `waves`: the
    wave each step first ran in (a step of the same wave never saw it)."""
    first = waves.get(step_id, 0)
    chain = {step_id}
    ordered = [step for step in order_steps(plan) if step.id in results]
    for step in ordered:
        later = waves.get(step.id, 0) > first
        if any(dep in chain for dep in step.depends_on) or (
                later and step.role == IMPLEMENTER):
            chain.add(step.id)
    return [step for step in ordered if step.id in chain]


def research_to_check(ran: list[Step], queue: list[Step], results: dict, checked,
                      limit: int) -> list[Step]:
    """The researchers checked now, before an implementer builds on them:
    once the next wave is an implementer, every one that ran (`ran`, in plan
    order), made something and is not in `checked`. Not after each wave of
    researchers: on a server with one helper slot each researcher is a wave of
    its own, and the first was checked while the second had not run (Strata,
    2026-11-07). A research-only job is left to the check at the end."""
    upcoming = next_wave(queue, set(results), limit) if queue else []
    if not upcoming or upcoming[0].role != IMPLEMENTER:
        return []
    return [step for step in ran if step.role == RESEARCHER and step.id not in checked
            and (results.get(step.id) or {}).get("status") not in (None, "failed")]


_FILE_NAME = re.compile(r"[\w-]{2,}\.[A-Za-z][A-Za-z0-9]{0,7}")


def made_later(deliverables: list[str]) -> list[str]:
    """The files the user gets, by name ("cities.md"), from a plan's
    deliverables: what a research check must not judge, since later steps
    make them. A deliverable is free text, so every file name in it counts
    ("cities.md (with a table)"); a description ("a short table", "v1.0")
    names none. A path counts by its last part."""
    names = []
    for deliverable in deliverables:
        for word in deliverable.split():
            name = word.strip("()[]{},;:'\"`.").rsplit("/", 1)[-1]
            if _FILE_NAME.fullmatch(name) and name not in names:
                names.append(name)
    return names


def _names_file(text: str, name: str) -> bool:
    """`name` appears in `text` as a whole file name: "chart.png" is in
    "out/chart.png." but not in "flowchart.png" or "chart.png5"."""
    pattern = r"(?<![\w.-])" + re.escape(name.lower()) + r"(?![\w-]|\.\w)"
    return re.search(pattern, text.lower()) is not None


def research_verdict(verdict: dict | None, later: list[str]) -> dict | None:
    """A research check's verdict without its checks of `later` files: one
    once failed the research because cities.md did not exist yet, and sent a
    sound researcher back. None when nothing else was checked ("not
    checked", never a pass). A check is dropped only when it names one of
    those files whole, so a real fail is never dropped by a similar name."""
    if verdict is None:
        return None
    checks = [check for check in verdict["checks"]
              if not any(_names_file(check["deliverable"], name) for name in later)]
    return dict(verdict, checks=checks) if checks else None


def research_redo(checked: list[Step], wanted: str) -> Step | None:
    """The researcher a failed check of research sends back: the one named,
    if it was checked, else the only one checked. Two checked and neither
    named: none, and the check at the end looks again."""
    for step in checked:
        if step.id == wanted:
            return step
    return checked[0] if len(checked) == 1 else None


def fallback_plan(request: str) -> Plan:
    """The plan when the manager could not write a valid one: one
    implementer with the whole request, so the workflow still runs."""
    request = _text(request, MAX_BRIEF_CHARS)
    return Plan(goal=_text(request, MAX_GOAL_CHARS),
                steps=[Step(id="s1", role=IMPLEMENTER, brief=request)],
                notes="The manager's plan could not be used, so this is one step.")


def plan_text(plan: Plan) -> str:
    """The plan as plain text, for the agents' tasks."""
    lines = [f"Goal: {plan.goal}"]
    for step in plan.steps:
        after = f" (after {', '.join(step.depends_on)})" if step.depends_on else ""
        lines.append(f"- {step.id} [{step.role}]{after}: {step.brief}")
    if plan.deliverables:
        lines.append("Deliverables: " + "; ".join(plan.deliverables))
    if plan.notes:
        lines.append(f"Notes: {plan.notes}")
    return "\n".join(lines)


STEP_ICONS = {"pending": "⏳", "running": "▶️", "done": "✅", "partial": "🟡",
              "failed": "❌", "cancelled": "⏹️", "skipped": "⏭️"}
ROLE_LABELS = {RESEARCHER: "🔬 Research", IMPLEMENTER: "🛠️ Build"}


def step_number(plan: Plan, step_id: str) -> int:
    return next((i for i, s in enumerate(plan.steps, 1) if s.id == step_id), 0)


def step_label(plan: Plan, step: Step) -> str:
    """A step's card label, in plain words: "🔬 Research · step 1 of 3"."""
    number = step_number(plan, step.id)
    where = f" · step {number} of {len(plan.steps)}" if number else ""
    return ROLE_LABELS.get(step.role, step.role) + where


def step_title(step: Step) -> str:
    """The few words naming a step (the role's when the plan gave none)."""
    return step.title or ROLE_LABELS.get(step.role, step.role).split(" ", 1)[-1]


def plan_blocks(plan: Plan, step_states: dict | None = None) -> list:
    """The plan as card blocks (classes/card_kit.py): the goal and what the
    user gets, then each step with its state, in full. A long plan goes over
    several cards; nothing is cut."""
    from classes import card_kit as K
    states = step_states or {}
    goal = K.safe(plan.goal)
    if plan.deliverables:
        goal += "\n\n**You'll get**\n" + "\n".join(f"- {K.safe(d)}" for d in plan.deliverables)
    items = []
    for number, step in enumerate(plan.steps, 1):
        icon = STEP_ICONS.get(states.get(step.id, ""), STEP_ICONS["pending"])
        after = [step_number(plan, d) for d in step.depends_on if step_number(plan, d)]
        meta = ROLE_LABELS.get(step.role, step.role)
        if after:
            meta += f" · after step{'s' if len(after) > 1 else ''} " + " and ".join(
                [", ".join(map(str, after[:-1])), str(after[-1])] if len(after) > 1 else [str(after[0])])
        items.append(f"{icon} **{number}. {K.safe(step_title(step))}**\n"
                     f"{K.small(f'{meta} — {step.brief}')}")
    return [K.Text(goal), K.Divider(), *K.listing(items)]


def steps_done(step_states: dict | None) -> int:
    return sum(1 for state in (step_states or {}).values() if state in ("done", "partial", "skipped"))


# --- states ------------------------------------------------------------------

PLANNING = "planning"
AWAITING_APPROVAL = "awaiting_approval"
RUNNING = "running"
VERIFYING = "verifying"
DONE = "done"
PARTIAL = "partial"
FAILED = "failed"
CANCELLED = "cancelled"
INTERRUPTED = "interrupted"
END_STATES = (DONE, PARTIAL, FAILED, CANCELLED, INTERRUPTED)
LIVE_STATES = (PLANNING, AWAITING_APPROVAL, RUNNING, VERIFYING)

# event -> {from state: to state}. Any end can be reached from any live
# state by "fail", "cancel" and "interrupt"; the rest are the happy path.
_MOVES = {
    "planned": {PLANNING: AWAITING_APPROVAL},
    "revise": {AWAITING_APPROVAL: PLANNING},
    "approve": {AWAITING_APPROVAL: RUNNING},
    "verify": {RUNNING: VERIFYING},
    "redo": {VERIFYING: RUNNING},
    "finish": {RUNNING: DONE, VERIFYING: DONE},
    "partial": {RUNNING: PARTIAL, VERIFYING: PARTIAL},
    "fail": {state: FAILED for state in LIVE_STATES},
    "cancel": {state: CANCELLED for state in LIVE_STATES},
    "interrupt": {state: INTERRUPTED for state in LIVE_STATES},
}


class IllegalMove(ValueError):
    pass


def transition(state: str, event: str) -> str:
    """The state after `event`; raises IllegalMove when it cannot happen
    (an ended workflow never moves again)."""
    moves = _MOVES.get(event)
    if moves is None or state not in moves:
        raise IllegalMove(f"{event!r} cannot happen in state {state!r}")
    return moves[state]


# --- which steps a message is for ---------------------------------------------


def clean_route(raw, step_ids) -> set | None:
    """The steps named in a routing answer, or None (all of them) when it
    is no answer. An empty list is one: the message changes no step."""
    if not isinstance(raw, dict) or not isinstance(raw.get("steps"), list):
        return None
    named = {clean_step_id(step) for step in raw["steps"]}
    if named - set(step_ids):
        return None  # it made one up, so nothing it said can be trusted
    return named


# --- files ------------------------------------------------------------------


def echo_files(files: list) -> list[tuple[int, str]]:
    """Which of a workflow's files go to the channel, and under what name.
    `files` holds (name, step, wave) in the order they were made; the answer
    is (index, name) pairs, in the order each name first appeared.

    A later wave's file replaces an earlier one of the same name: that is a
    step fixing it. Two steps of one wave ran side by side, so neither is a
    fix of the other: both go, each named after its step."""
    first: dict[str, int] = {}
    newest: dict[str, int] = {}
    for index, (name, _step, wave) in enumerate(files):
        first.setdefault(name, index)
        newest[name] = max(newest.get(name, wave), wave)
    latest: dict[tuple[str, str], int] = {}
    for index, (name, step, wave) in enumerate(files):
        if wave == newest[name]:
            latest[(name, step)] = index
    steps: dict[str, int] = {}
    for name, _step in latest:
        steps[name] = steps.get(name, 0) + 1
    chosen = sorted(latest.items(), key=lambda item: (first[item[0][0]], item[1]))
    return [(index, f"{step}-{name}" if steps[name] > 1 and step else name)
            for (name, step), index in chosen]


# --- steering ----------------------------------------------------------------

_STOP_WORDS = {"stop", "cancel", "abort", "stop it", "cancel it", "stop the workflow",
               "cancel the workflow", "stop please", "please stop", "cancel please",
               "please cancel"}


def is_stop_word(text: str) -> bool:
    """Whether a thread message is the requester saying stop. Only the whole
    message counts: "stop using red" is a change, not a stop."""
    folded = re.sub(r"[^\w\s]", "", str(text or "")).strip().lower()
    folded = " ".join(folded.split())
    return folded in _STOP_WORDS

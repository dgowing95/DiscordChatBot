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
DEFAULT_MAX_TURNS = 250
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


def _env_int(name: str, default: int, env) -> int:
    """A positive int setting; anything else is the default. (The _env
    prefix is what settings_docs_tests looks for.)"""
    try:
        value = int(str(env.get(name)).strip())
        return value if value > 0 else default
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
    default 250)."""
    return _env_int("WORKFLOW_MAX_TURNS", DEFAULT_MAX_TURNS, env)


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


@dataclass
class Step:
    id: str
    role: str
    brief: str
    depends_on: list[str] = field(default_factory=list)


@dataclass
class Plan:
    goal: str
    steps: list[Step]
    deliverables: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict:
        return {"goal": self.goal, "notes": self.notes, "deliverables": list(self.deliverables),
                "steps": [{"id": s.id, "role": s.role, "brief": s.brief,
                           "depends_on": list(s.depends_on)} for s in self.steps]}


def _text(value, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def clean_step_id(value) -> str:
    """A step's id, safe in a folder name (workflows.step_job_id): a model
    writes it, and "../shared" must not reach a path."""
    text = "".join(str(value or "").split())
    return re.sub(r"[^A-Za-z0-9_-]+", "-", text)[:20].strip("-")


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
        step_id = clean_step_id(item.get("id")) or f"s{index}"
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
            if clean_step_id(dep):
                depends.append(clean_step_id(dep))
            elif str(dep or "").strip():  # a blank entry is no dependency at all
                problems.append(f"Step {step_id} depends on {str(dep)[:40]!r}, which is not a "
                                "usable step id.")
        steps.append(Step(id=step_id, role=role, brief=brief, depends_on=depends))
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
                notes=_text(raw.get("notes"), MAX_NOTES_CHARS))
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


def fallback_plan(request: str) -> Plan:
    """The plan when the manager could not write a valid one: one
    implementer with the whole request, so the workflow still runs."""
    request = _text(request, MAX_BRIEF_CHARS)
    return Plan(goal=_text(request, MAX_GOAL_CHARS),
                steps=[Step(id="s1", role=IMPLEMENTER, brief=request)],
                notes="The manager's plan could not be used, so this is one step.")


def plan_from_dict(raw: dict) -> Plan:
    """A stored plan (Plan.to_dict) back as a Plan."""
    return Plan(goal=raw.get("goal", ""), notes=raw.get("notes", ""),
                deliverables=list(raw.get("deliverables") or []),
                steps=[Step(id=s["id"], role=s["role"], brief=s["brief"],
                            depends_on=list(s.get("depends_on") or []))
                       for s in raw.get("steps") or []])


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


# Discord embed limits (a field value is 1024 characters, a description 4096).
_FIELD_CHARS = 1000
_DESCRIPTION_CHARS = 1500
# Under Discord's 6000 for a whole embed, leaving room for the title.
_PLAN_CHARS = 3400
STEP_ICONS = {"pending": "⏳", "running": "▶️", "done": "✅", "partial": "🟡",
              "failed": "❌", "cancelled": "⏹️", "skipped": "⏭️"}


def render_plan(plan: Plan, step_states: dict | None = None) -> dict:
    """The plan as card data: {title, description, fields: [(name, value)]}.
    With step_states, each step shows its state's icon."""
    states = step_states or {}
    description = plan.goal
    if plan.deliverables:
        description += "\n\n**You get:** " + "; ".join(plan.deliverables)
    description = description[:_DESCRIPTION_CHARS]
    # A card holds 4000 characters in all (classes/card.py), so the steps
    # share what the description leaves.
    budget = _PLAN_CHARS - len(description)
    steps = plan.steps[:25]
    fields = []
    for index, step in enumerate(steps):
        icon = STEP_ICONS.get(states.get(step.id, ""), "")
        name = f"{icon} {step.id} · {step.role}".strip()[:256]
        share = max(40, min(_FIELD_CHARS, budget // (len(steps) - index) - len(name)))
        value = step.brief if len(step.brief) <= share else step.brief[: share - 1] + "…"
        budget -= len(name) + len(value)
        fields.append((name, value))
    return {"title": "🧵 Plan", "description": description, "fields": fields}


# --- states ------------------------------------------------------------------

PLANNING = "planning"
AWAITING_APPROVAL = "awaiting_approval"
RUNNING = "running"
DONE = "done"
PARTIAL = "partial"
FAILED = "failed"
CANCELLED = "cancelled"
INTERRUPTED = "interrupted"
END_STATES = (DONE, PARTIAL, FAILED, CANCELLED, INTERRUPTED)
LIVE_STATES = (PLANNING, AWAITING_APPROVAL, RUNNING)

# event -> {from state: to state}. Any end can be reached from any live
# state by "fail", "cancel" and "interrupt"; the rest are the happy path.
_MOVES = {
    "planned": {PLANNING: AWAITING_APPROVAL},
    "revise": {AWAITING_APPROVAL: PLANNING},
    "approve": {AWAITING_APPROVAL: RUNNING},
    "finish": {RUNNING: DONE},
    "partial": {RUNNING: PARTIAL},
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


# --- budgets -----------------------------------------------------------------


def budget_left(*, started: float, now: float, turns_used: int, timeout: int,
                max_turns: int) -> dict:
    """What a workflow has left: {"seconds", "turns"} (never below 0)."""
    return {"seconds": max(0.0, started + timeout - now),
            "turns": max(0, max_turns - turns_used)}


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

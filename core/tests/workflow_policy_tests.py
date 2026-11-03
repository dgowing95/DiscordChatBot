"""Tier 3 workflows' pure rules (core/classes/workflow_policy.py)."""
import pytest

from classes import workflow_policy as wp


def _plan(*steps, **extra):
    raw = {"goal": "Compare printers", "steps": list(steps)}
    raw.update(extra)
    return raw


def _step(step_id, role="researcher", brief="Find prices", depends_on=()):
    return {"id": step_id, "role": role, "brief": brief, "depends_on": list(depends_on)}


# --------------------------- plans ---------------------------

def test_a_valid_plan_is_read_with_its_order_kept():
    plan, problems = wp.validate_plan(
        _plan(_step("s1"), _step("s2", "implementer", "Write it", ["s1"]),
              deliverables=["comparison.md"], notes="USD only"), max_steps=8)
    assert problems == []
    assert [s.id for s in plan.steps] == ["s1", "s2"]
    assert plan.steps[1].depends_on == ["s1"] and plan.deliverables == ["comparison.md"]
    assert plan.notes == "USD only"


def test_loose_shapes_from_a_small_model_are_accepted():
    plan, problems = wp.validate_plan(
        {"goal": "x", "steps": [{"role": "Research", "task": "look"},
                                {"role": "builder", "brief": "make", "depends_on": "s1"}],
         "deliverables": "a.md, b.png"}, max_steps=8)
    assert problems == []
    assert [(s.id, s.role) for s in plan.steps] == [("s1", wp.RESEARCHER), ("s2", wp.IMPLEMENTER)]
    assert plan.steps[1].depends_on == ["s1"] and plan.deliverables == ["a.md", "b.png"]


def test_the_request_stands_in_for_a_missing_goal():
    plan, _ = wp.validate_plan({"steps": [_step("s1")]}, max_steps=8, request="Do the thing")
    assert plan.goal == "Do the thing"


@pytest.mark.parametrize("raw, expected", [
    ("not a dict", "not a JSON object"),
    ({"goal": "x", "steps": []}, "no steps"),
    (_plan(_step("s1"), _step("s1")), "Two steps have the id s1"),
    (_plan(_step("s1", role="manager")), "role 'manager'"),
    (_plan(_step("s1", brief="")), "no brief"),
    (_plan(_step("s1", depends_on=["s9"])), "depends on s9"),
    (_plan(_step("s1", depends_on=["s1"])), "depends on itself"),
    (_plan(_step("s1", depends_on=["s2"]), _step("s2", depends_on=["s1"])), "in a circle"),
    (_plan(*[_step(f"s{i}") for i in range(1, 5)]), "at most 3"),
])
def test_unusable_plans_say_what_is_wrong(raw, expected):
    plan, problems = wp.validate_plan(raw, max_steps=3)
    assert plan is None
    assert any(expected in p for p in problems), problems


def test_steps_run_after_what_they_depend_on():
    plan, _ = wp.validate_plan(_plan(_step("s1", depends_on=["s3"]), _step("s2"), _step("s3")),
                               max_steps=8)
    assert [s.id for s in wp.order_steps(plan)] == ["s2", "s3", "s1"]


def test_the_fallback_is_one_implementer_with_the_whole_request():
    plan = wp.fallback_plan("Build a CSV to chart tool")
    assert [(s.id, s.role, s.brief) for s in plan.steps] == [
        ("s1", wp.IMPLEMENTER, "Build a CSV to chart tool")]


def test_a_plan_round_trips_through_its_record():
    plan, _ = wp.validate_plan(_plan(_step("s1"), _step("s2", depends_on=["s1"])), max_steps=8)
    again = wp.plan_from_dict(plan.to_dict())
    assert again == plan


def test_the_plan_reads_as_text_and_as_an_embed_with_step_states():
    plan, _ = wp.validate_plan(_plan(_step("s1"), _step("s2", "implementer", "Write", ["s1"]),
                                     deliverables=["a.md"]), max_steps=8)
    text = wp.plan_text(plan)
    assert "- s2 [implementer] (after s1): Write" in text and "Deliverables: a.md" in text
    embed = wp.render_plan(plan, {"s1": "done", "s2": "running"})
    assert embed["fields"][0][0].startswith("✅ s1") and embed["fields"][1][0].startswith("▶️ s2")
    assert "**You get:** a.md" in embed["description"]


# --------------------------- states ---------------------------

def test_the_happy_path_moves():
    state = wp.PLANNING
    for event, expected in [("planned", wp.AWAITING_APPROVAL), ("revise", wp.PLANNING),
                            ("planned", wp.AWAITING_APPROVAL), ("approve", wp.RUNNING),
                            ("finish", wp.DONE)]:
        state = wp.transition(state, event)
        assert state == expected


@pytest.mark.parametrize("state", wp.LIVE_STATES)
def test_any_live_workflow_can_fail_be_cancelled_or_interrupted(state):
    assert wp.transition(state, "fail") == wp.FAILED
    assert wp.transition(state, "cancel") == wp.CANCELLED
    assert wp.transition(state, "interrupt") == wp.INTERRUPTED


@pytest.mark.parametrize("state, event", [
    (wp.PLANNING, "approve"), (wp.AWAITING_APPROVAL, "finish"), (wp.RUNNING, "planned"),
    (wp.PLANNING, "partial"), (wp.DONE, "cancel"), (wp.CANCELLED, "approve"),
    (wp.RUNNING, "nonsense"),
])
def test_illegal_moves_raise(state, event):
    with pytest.raises(wp.IllegalMove):
        wp.transition(state, event)


# --------------------------- budgets, settings, steering ---------------------------

def test_the_budget_never_goes_below_zero():
    assert wp.budget_left(started=0, now=100, turns_used=10, timeout=300, max_turns=50) == {
        "seconds": 200, "turns": 40}
    assert wp.budget_left(started=0, now=900, turns_used=90, timeout=300, max_turns=50) == {
        "seconds": 0, "turns": 0}


def test_settings_defaults_and_overrides():
    env = {}
    assert (wp.enabled(env), wp.from_automations(env)) == (True, True)
    assert wp.ask_timeout(env) == 300 and wp.plan_approval_seconds(env) == 900
    assert wp.plan_revisions(env) == 3 and wp.max_steps(env) == 8
    assert wp.timeout(env) == 3600 and wp.max_turns(env) == 250
    assert wp.max_per_guild(env) == 1 and wp.subagent_max_turns(env) == 100
    assert not wp.enabled({"WORKFLOW_ENABLED": "0"})
    assert not wp.from_automations({"WORKFLOW_FROM_AUTOMATIONS": "false"})
    assert wp.max_steps({"WORKFLOW_MAX_STEPS": "junk"}) == 8
    assert wp.timeout({"WORKFLOW_TIMEOUT": "60"}) == 60


@pytest.mark.parametrize("text, stop", [
    ("stop", True), ("Stop!", True), ("cancel please", True), ("  STOP IT ", True),
    ("stop using red", False), ("can you stop and add a legend", False), ("", False),
])
def test_only_a_bare_stop_stops(text, stop):
    assert wp.is_stop_word(text) is stop


def test_a_long_plan_still_fits_one_discord_embed():
    plan, _ = wp.validate_plan(_plan(*[_step(f"s{i}", brief="x" * 1500) for i in range(1, 9)],
                                     goal="g" * 500, deliverables=["d" * 200] * 10), max_steps=8)
    embed = wp.render_plan(plan)
    total = len(embed["title"]) + len(embed["description"]) + sum(
        len(name) + len(value) for name, value in embed["fields"])
    assert total <= 6000 and len(embed["fields"]) == 8
    assert all(len(value) <= 1024 for _, value in embed["fields"])

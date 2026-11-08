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


def _steps(*items):
    return [wp.Step(id=i, role=r, brief="b", depends_on=list(d)) for i, r, d in items]


def _ids(steps):
    return [s.id for s in steps]


def test_ready_researchers_go_together_up_to_the_limit():
    queue = _steps(("a", wp.RESEARCHER, ()), ("b", wp.RESEARCHER, ()), ("c", wp.RESEARCHER, ()))
    assert _ids(wp.next_wave(queue, set(), 2)) == ["a", "b"]
    assert _ids(wp.next_wave(queue, set(), 1)) == ["a"]
    assert _ids(wp.next_wave(queue, set(), 9)) == ["a", "b", "c"]
    assert _ids(wp.next_wave(queue, set(), 0)) == ["a"]


def test_a_researcher_waits_for_what_it_depends_on():
    queue = _steps(("a", wp.RESEARCHER, ()), ("b", wp.RESEARCHER, ("a",)), ("c", wp.RESEARCHER, ()))
    assert _ids(wp.next_wave(queue, set(), 2)) == ["a", "c"]
    assert _ids(wp.next_wave(queue[1:2], {"a"}, 2)) == ["b"]


def test_an_implementer_runs_alone():
    first = _steps(("a", wp.IMPLEMENTER, ()), ("b", wp.RESEARCHER, ()))
    assert _ids(wp.next_wave(first, set(), 2)) == ["a"]
    between = _steps(("a", wp.RESEARCHER, ()), ("b", wp.IMPLEMENTER, ()), ("c", wp.RESEARCHER, ()))
    assert _ids(wp.next_wave(between, set(), 2)) == ["a", "c"]
    two = _steps(("a", wp.IMPLEMENTER, ()), ("b", wp.IMPLEMENTER, ()))
    assert _ids(wp.next_wave(two, set(), 2)) == ["a"]


def test_nothing_ready_still_gives_a_step():
    stuck = _steps(("a", wp.RESEARCHER, ("x",)), ("b", wp.RESEARCHER, ("a",)))
    assert _ids(wp.next_wave(stuck, set(), 2)) == ["a"]


@pytest.mark.parametrize("raw, expected", [
    ({"subjects": ["a", " b ", "c"]}, ["a", "b"]),
    ({"subjects": "one"}, ["one"]),
    (["a?"], ["a?"]),
    ({"subjects": ["", None]}, []),
    ({}, []),
    (None, []),
])
def test_scout_subjects_are_read_loosely_and_capped(raw, expected):
    assert wp.clean_scout_subjects(raw) == expected


def test_a_long_scout_question_is_cut():
    assert len(wp.clean_scout_subjects(["x" * 900])[0]) == wp.MAX_SUBJECT_CHARS


def test_one_string_is_one_subject_or_question_whatever_commas_it_has():
    assert wp.clean_scout_subjects({"subjects": "UK rail delays, by operator"}) == [
        "UK rail delays, by operator"]
    assert wp.clean_plan_questions({"questions": "Which format, PDF or Word?"}) == [
        ("Which format, PDF or Word?", [])]


def test_plan_questions_are_read_loosely_and_capped():
    raw = {"questions": [{"question": "a?", "options": ["x", "y"]}, "b?", {"question": ""},
                         {"question": "c?"}, {"question": "d?"}]}
    assert wp.clean_plan_questions(raw) == [("a?", ["x", "y"]), ("b?", []), ("c?", [])]
    assert wp.clean_plan_questions({"questions": {"question": "one?"}}) == [("one?", [])]
    assert wp.clean_plan_questions({"questions": "one?"}) == [("one?", [])]
    assert wp.clean_plan_questions(None) == []


def test_findings_are_lines_within_their_limit():
    assert wp.findings_text(["a", "b"]) == "- a\n- b"
    long = wp.findings_text(["x" * 5000, "y" * 5000])
    assert len(long) <= wp.MAX_FINDINGS_CHARS + 1 and "y" in long


def test_the_fallback_is_one_implementer_with_the_whole_request():
    plan = wp.fallback_plan("Build a CSV to chart tool")
    assert [(s.id, s.role, s.brief) for s in plan.steps] == [
        ("s1", wp.IMPLEMENTER, "Build a CSV to chart tool")]


def test_a_step_never_takes_a_scouts_or_a_checks_folder():
    plan, problems = wp.validate_plan(_plan(_step("d1"), _step("V2", depends_on=["d1"]),
                                            _step("s3", depends_on=["r1", "V2"]),
                                            _step("r1")), max_steps=8)
    assert problems == []
    assert [s.id for s in plan.steps] == ["sd1", "sV2", "s3", "sr1"]
    assert plan.steps[1].depends_on == ["sd1"]
    assert plan.steps[2].depends_on == ["sr1", "sV2"]
    assert wp.plan_step_id("s1") == "s1" and wp.plan_step_id("data2") == "data2"


def _block_texts(blocks):
    from classes import card_kit as K
    return [b.content for b in blocks if isinstance(b, K.Text)]


def test_the_plan_reads_as_text_and_as_a_card_with_step_states():
    raw = _plan(_step("s1"), _step("s2", "implementer", "Write", ["s1"]), deliverables=["a.md"])
    raw["steps"][1]["title"] = "Write it up"
    plan, _ = wp.validate_plan(raw, max_steps=8)
    text = wp.plan_text(plan)
    assert "- s2 [implementer] (after s1): Write" in text and "Deliverables: a.md" in text
    goal, first, second = _block_texts(wp.plan_blocks(plan, {"s1": "done", "s2": "running"}))
    assert "**You'll get**\n- a.md" in goal
    # a step with no title is named by its role; ids never show
    assert first.startswith("✅ **1. Research**\n-# 🔬 Research — ")
    assert second.startswith("▶️ **2. Write it up**\n-# 🛠️ Build · after step 1 — Write")
    assert wp.step_label(plan, plan.steps[1]) == "🛠️ Build · step 2 of 2"
    assert wp.step_title(plan.steps[0]) == "Research"


def test_titles_are_kept_in_the_record_and_may_be_left_out():
    raw = _plan(_step("s1"))
    raw["title"] = "Compare printers"
    raw["steps"][0]["title"] = "Find prices"
    plan, _ = wp.validate_plan(raw, max_steps=8)
    assert plan.title == "Compare printers" and plan.steps[0].title == "Find prices"
    stored = plan.to_dict()
    assert stored["title"] == "Compare printers" and stored["steps"][0]["title"] == "Find prices"
    plain, _ = wp.validate_plan(_plan(_step("s1")), max_steps=8)
    assert plain.title == "" and plain.steps[0].title == ""


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


def test_settings_defaults_and_overrides():
    env = {}
    assert (wp.enabled(env), wp.from_automations(env)) == (True, True)
    assert wp.ask_timeout(env) == 300 and wp.plan_approval_seconds(env) == 900
    assert wp.plan_revisions(env) == 3 and wp.max_steps(env) == 8
    assert wp.timeout(env) == 3600 and wp.max_turns(env) == 300
    assert wp.turn_reserve(env) == 40 and wp.max_tokens(env) == 4_000_000
    assert wp.max_tokens({"WORKFLOW_MAX_TOKENS": "5000"}) == 5000
    assert wp.max_tokens({"WORKFLOW_MAX_TOKENS": "0"}) == 0  # off
    assert wp.turn_reserve({"WORKFLOW_MAX_TURNS": "50"}) == 10
    assert wp.max_per_guild(env) == 1 and wp.subagent_max_turns(env) == 100
    assert not wp.enabled({"WORKFLOW_ENABLED": "0"})
    assert not wp.from_automations({"WORKFLOW_FROM_AUTOMATIONS": "false"})
    assert wp.max_steps({"WORKFLOW_MAX_STEPS": "junk"}) == 8
    assert wp.timeout({"WORKFLOW_TIMEOUT": "60"}) == 60


def test_the_check_is_a_live_state_with_its_own_moves():
    assert wp.VERIFYING in wp.LIVE_STATES
    assert wp.transition(wp.RUNNING, "verify") == wp.VERIFYING
    assert wp.transition(wp.VERIFYING, "redo") == wp.RUNNING
    assert wp.transition(wp.VERIFYING, "finish") == wp.DONE
    assert wp.transition(wp.VERIFYING, "partial") == wp.PARTIAL
    for event, end in (("cancel", wp.CANCELLED), ("fail", wp.FAILED),
                       ("interrupt", wp.INTERRUPTED)):
        assert wp.transition(wp.VERIFYING, event) == end


def test_a_verdict_is_cleaned_and_nothing_checked_is_no_verdict():
    raw = {"checks": [{"deliverable": "a.md", "ok": True, "note": "fine"},
                      {"deliverable": "b.png", "ok": "false", "note": "wrong year"},
                      {"deliverable": "c", "ok": "maybe"}, "junk"],
           "redo_step": "../s2"}
    verdict = wp.clean_verdict(raw)
    assert [check["ok"] for check in verdict["checks"]] == [True, False]
    assert verdict["redo_step"] == "s2"
    assert wp.verdict_failed(verdict)
    assert wp.verdict_notes(verdict) == "- b.png: wrong year"
    for empty in (None, "junk", {}, {"checks": []}, {"checks": [{"ok": "maybe"}]}):
        assert wp.clean_verdict(empty) is None
    assert not wp.verdict_failed(None) and wp.verdict_notes(None) == ""


def test_the_step_sent_back_is_one_that_ran():
    plan = wp.Plan(goal="g", deliverables=[], notes="", steps=[
        wp.Step(id="s1", role="researcher", brief="b"),
        wp.Step(id="s2", role="implementer", brief="b"),
        wp.Step(id="s3", role="researcher", brief="b"),
        wp.Step(id="s4", role="implementer", brief="b")])
    ran = {"s1": {}, "s2": {}, "s3": {}}
    assert wp.redo_target(plan, ran, "s1").id == "s1"
    assert wp.redo_target(plan, ran, "s4").id == "s2"  # it never ran: the last implementer
    assert wp.redo_target(plan, ran, "nope").id == "s2"
    assert wp.redo_target(plan, {"s1": {}}, "").id == "s1"  # no implementer ran
    assert wp.redo_target(plan, {}, "s1") is None


def _chain_plan():
    return wp.Plan(goal="g", deliverables=[], notes="", steps=[
        wp.Step(id="s1", role="researcher", brief="b"),
        wp.Step(id="s2", role="researcher", brief="b"),
        wp.Step(id="s3", role="implementer", brief="b", depends_on=["s1", "s2"]),
        wp.Step(id="s4", role="implementer", brief="b", depends_on=["s3"]),
        wp.Step(id="s5", role="implementer", brief="b", depends_on=["s2"])])


def test_a_step_sent_back_brings_the_steps_built_on_it():
    plan = _chain_plan()
    ran = {s.id: {} for s in plan.steps}
    waves = {"s1": 1, "s2": 1, "s3": 2, "s4": 3, "s5": 4}
    # Every implementer after a researcher read its report.
    assert [s.id for s in wp.redo_chain(plan, ran, "s1", waves)] == ["s1", "s3", "s4", "s5"]
    # After an implementer: s5 ran later, so it may have read s3's report too.
    assert [s.id for s in wp.redo_chain(plan, ran, "s3", waves)] == ["s3", "s4", "s5"]
    assert [s.id for s in wp.redo_chain(plan, ran, "s5", waves)] == ["s5"]
    # One that never ran is not run "again".
    assert [s.id for s in wp.redo_chain(plan, {"s1": {}, "s3": {}}, "s1", waves)] == ["s1", "s3"]


def test_a_send_back_reaches_a_file_the_plan_did_not_link():
    plan = wp.Plan(goal="g", deliverables=[], notes="", steps=[
        wp.Step(id="s1", role="researcher", brief="b"),
        wp.Step(id="s2", role="researcher", brief="b"),
        wp.Step(id="s3", role="implementer", brief="b")])  # no depends_on
    ran = {"s1": {}, "s2": {}, "s3": {}}
    waves = {"s1": 1, "s2": 1, "s3": 2}
    # s3 still made moons.md from s1's report; s2 ran beside s1 and never saw it.
    assert [s.id for s in wp.redo_chain(plan, ran, "s1", waves)] == ["s1", "s3"]


def test_research_is_checked_only_when_the_next_step_is_an_implementer():
    plan = _chain_plan()
    s1, s2, s3 = plan.steps[:3]
    done = {"s1": {"status": "done"}, "s2": {"status": "failed"}}
    assert wp.research_to_check([s1, s2], [s3], done, set(), 2) == [s1]  # s2 made nothing
    assert wp.research_to_check([s1, s2], [], done, set(), 2) == []  # the end check has it
    assert wp.research_to_check([s1], [s3], done, {"s1"}, 2) == []  # checked already


def test_one_helper_slot_checks_the_research_once_after_the_last_researcher():
    s1, s2, s3 = _chain_plan().steps[:3]
    # One slot: s1 ran alone and s2 is next, so nothing is checked yet.
    assert wp.research_to_check([s1], [s2, s3], {"s1": {"status": "done"}}, set(), 1) == []
    both = {"s1": {"status": "done"}, "s2": {"status": "done"}}
    assert wp.research_to_check([s1, s2], [s3], both, set(), 1) == [s1, s2]


def test_a_research_check_never_fails_on_a_file_later_steps_make():
    later = wp.made_later(["cities.md", "a short comparison table", "chart.png"])
    assert later == ["cities.md", "chart.png"]
    early = {"checks": [{"deliverable": "cities.md (final deliverable)", "ok": False,
                         "note": "not produced yet"}], "redo_step": "s1"}
    assert wp.research_verdict(early, later) is None  # nothing about the research: not checked
    mixed = {"checks": early["checks"] + [{"deliverable": "s1: Oslo population", "ok": False,
                                           "note": "2024 figure"}], "redo_step": "s1"}
    kept = wp.research_verdict(mixed, later)
    assert [c["deliverable"] for c in kept["checks"]] == ["s1: Oslo population"]
    assert wp.research_verdict(None, later) is None


def test_file_names_are_found_inside_free_text_deliverables():
    later = wp.made_later(["cities.md (with a table)", "the report out/report.md.",
                           "version v1.0 notes", "e.g. a summary", "cities.md",
                           "the table (towns.csv).", "charts/2026.11/population.png"])
    assert later == ["cities.md", "report.md", "towns.csv", "population.png"]


def test_a_research_check_keeps_a_fail_on_a_file_with_a_similar_name():
    verdict = {"checks": [{"deliverable": "flowchart.png", "ok": False, "note": "wrong"},
                          {"deliverable": "out/chart.png.", "ok": False, "note": "missing"}],
               "redo_step": "s1"}
    kept = wp.research_verdict(verdict, ["chart.png"])
    assert [c["deliverable"] for c in kept["checks"]] == ["flowchart.png"]


def test_the_researcher_a_research_check_sends_back():
    s1, s2 = _chain_plan().steps[:2]
    assert wp.research_redo([s1, s2], "s2") == s2
    assert wp.research_redo([s1], "nope") == s1   # the only one checked
    assert wp.research_redo([s1, s2], "s3") is None


def test_which_files_go_to_the_channel():
    assert wp.echo_files([]) == []
    # a later wave fixes a file: only the newest, in the name's first place
    assert wp.echo_files([("a", "s1", 1), ("b", "s1", 1), ("a", "s2", 2)]) == [(2, "a"), (1, "b")]
    # one step attaching a name twice is one file
    assert wp.echo_files([("a", "s1", 1), ("a", "s1", 1)]) == [(1, "a")]
    # two steps side by side: both, each under its step
    assert wp.echo_files([("a", "s1", 1), ("a", "s2", 1)]) == [(0, "s1-a"), (1, "s2-a")]
    # ...until a later wave replaces them
    assert wp.echo_files([("a", "s1", 1), ("a", "s2", 1), ("a", "s3", 2)]) == [(2, "a")]


@pytest.mark.parametrize("text, stop", [
    ("stop", True), ("Stop!", True), ("cancel please", True), ("  STOP IT ", True),
    ("stop using red", False), ("can you stop and add a legend", False), ("", False),
])
def test_only_a_bare_stop_stops(text, stop):
    assert wp.is_stop_word(text) is stop


def test_a_long_plan_keeps_every_brief_whole():
    from classes import card_kit as K
    plan, _ = wp.validate_plan(_plan(*[_step(f"s{i}", brief=f"{i}" + "x" * 1400) for i in range(1, 9)],
                                     goal="g" * 500, deliverables=["d" * 200] * 10), max_steps=8)
    blocks = wp.plan_blocks(plan)
    texts = _block_texts(blocks)
    assert len(texts) == 9  # the goal and 8 steps
    assert all(f"{i}" + "x" * 1400 in "\n".join(texts) for i in range(1, 9))
    cards = K.split(K.CardSpec(body=blocks))
    assert len(cards) > 1  # too long for one message: it goes on in the next

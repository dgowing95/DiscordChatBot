"""Tests for what the brain hands back (core/classes/brain_result.py, pure)
and its limits (brain_policy.py), loop guard (loop_guard.py) and context
trimming (context_trim.py).

Run from the repo root:
    pytest core/tests/brain_result_tests.py
"""
import pytest

from classes import brain_policy, brain_result as br, context_trim
from classes.loop_guard import REPEAT, STALE, LoopGuard


# --------------------------- the report ---------------------------

def test_report_args_are_read_leniently():
    report = br.parse_report_args({"status": "DONE", "facts": "one\n- two\n\n", "must_keep": None})
    assert report.status == br.DONE
    assert report.facts == ["one", "two"]
    assert report.must_keep == []


def test_an_unknown_status_counts_as_done_and_odd_input_is_empty():
    assert br.parse_report_args({"status": "finished"}).status == br.DONE
    empty = br.parse_report_args(None)
    assert (empty.status, empty.facts, empty.must_keep) == (br.DONE, [], [])


def test_facts_are_capped_and_repeats_dropped():
    report = br.parse_report_args({"facts": ["same", "same"] + [f"f{i}" for i in range(50)]})
    assert report.facts[0] == "same" and report.facts.count("same") == 1
    assert len(report.facts) == br.MAX_FACTS
    long = br.parse_report_args({"facts": ["x" * 5000]})
    assert len(long.facts[0]) == br.MAX_FACT_CHARS


def test_exact_tokens_in_the_facts_are_kept_automatically():
    facts = ["The Prusa Core One is the cheapest at $1,199 (https://example.com/core-one).",
             "It weighs 22 kg; saved as `prices.csv`, chart in chart.png"]
    tokens = br.auto_must_keep(facts)
    assert "$1,199" in tokens
    assert "https://example.com/core-one" in tokens
    assert "22 kg" in tokens
    assert "prices.csv" in tokens and "chart.png" in tokens


def test_the_model_s_own_must_keep_comes_first():
    report = br.parse_report_args({"facts": ["at https://a.example/x"], "must_keep": ["ABC-123"]})
    assert report.must_keep == ["ABC-123", "https://a.example/x"]


def test_a_text_ending_becomes_facts():
    report = br.report_from_text("First paragraph.\n\nSecond one with https://b.example")
    assert report.status == br.DONE
    assert report.facts == ["First paragraph.", "Second one with https://b.example"]
    assert report.must_keep == ["https://b.example"]
    assert br.report_from_text("").status == br.PARTIAL


def test_what_the_persona_is_told():
    report = br.BrainReport(facts=["Cheapest: $1,199 (https://x.example)"],
                            must_keep=["$1,199", "https://x.example"],
                            files_posted=["chart.png"], posted=["a poll"],
                            say_to_user="Ask if they want the chart in blue.")
    text = br.tool_result_for_persona(report)
    assert text.startswith("Brain result: done.")
    assert "Already posted in the channel: chart.png and a poll." in text
    assert "- Cheapest: $1,199 (https://x.example)" in text
    assert "Must appear exactly in your reply: $1,199, https://x.example" in text
    assert "Tell the user: Ask if they want the chart in blue." in text
    assert "Keep every fact above" in text


def test_a_one_tool_job_reads_short():
    text = br.tool_result_for_persona(br.BrainReport(posted=["a poll"]))
    assert text == ("Brain result: done.\nAlready posted in the channel: a poll. "
                    "Do not describe them as coming later.")


def test_a_stopped_call_is_never_retried():
    text = br.tool_result_for_persona(br.BrainReport(status=br.PARTIAL, note="It ran out of time."))
    assert "only partly done" in text and "Note: It ran out of time." in text
    assert "Do not ask the brain to try the same thing again" in text


def test_missing_tokens():
    report = br.BrainReport(must_keep=["$1,199", "https://x.example/", "Prusa"])
    assert br.missing_tokens("It is $1,199 from prusa at https://x.example", report) == []
    assert br.missing_tokens("About 1200 dollars", report) == ["$1,199", "https://x.example/", "Prusa"]


# --------------------------- limits ---------------------------

def test_brain_limits_are_backstops(monkeypatch):
    env = {}
    assert brain_policy.brain_max_turns(env) == 100
    assert brain_policy.brain_timeout(env) == 1800
    assert brain_policy.repeat_limit(env) == 3
    assert brain_policy.trim_keep_recent(env) == 6
    assert brain_policy.repeat_limit({"BRAIN_REPEAT_LIMIT": "1"}) == 2
    assert brain_policy.brain_max_turns({"BRAIN_MAX_TURNS": "junk"}) == 100


def test_only_asked_for_chat_and_allowed_automatic_runs_may_start_a_workflow():
    tier = brain_policy.effective_tier
    assert tier(env={}) == brain_policy.TIER_WORKFLOW
    assert tier(unprompted=True, env={}) == brain_policy.TIER_BRAIN
    assert tier(voice=True, env={}) == brain_policy.TIER_BRAIN
    assert tier(automatic=True, env={}) == brain_policy.TIER_WORKFLOW
    assert tier(automatic=True, env={"WORKFLOW_FROM_AUTOMATIONS": "0"}) == brain_policy.TIER_BRAIN
    assert tier(env={"WORKFLOW_ENABLED": "false"}) == brain_policy.TIER_BRAIN


# --------------------------- the loop guard ---------------------------

def test_the_same_call_three_times_in_a_row_warns_then_stops():
    guard = LoopGuard(repeat_limit=3)
    assert guard.record("web_search", '{"search_request": "x"}') is None
    assert guard.record("web_search", '{"search_request":"x"}') is None  # same, other spacing
    assert guard.record("web_search", {"search_request": "x"}) == REPEAT
    note = guard.note()
    assert "repeating `web_search`" in note and "finish now" in note
    assert guard.note() is None  # once
    assert not guard.should_stop()
    for _ in range(3):
        guard.record("web_search", {"search_request": "x"})
    assert guard.should_stop()


def test_nothing_new_in_the_last_eight_calls_is_a_loop():
    guard = LoopGuard(repeat_limit=3, window=8)
    for name in "abcd":
        guard.record(name, {})
    kinds = [guard.record(name, {}) for name in "abcdabcd"]
    assert kinds[-1] == STALE and kinds[:-1] == [None] * 7


def test_honest_long_work_never_trips_it():
    guard = LoopGuard(repeat_limit=3)
    for i in range(60):
        assert guard.record("exec_command", {"cmd": f"python step{i}.py"}) is None
        # editing then re-running the same script is two different calls
        assert guard.record("exec_command", {"cmd": "python run.py", "i": i}) is None
    assert not guard.should_stop()


def test_polling_and_report_never_count():
    guard = LoopGuard(repeat_limit=2)
    for _ in range(10):
        assert guard.record("write_stdin", {"session_id": 1, "chars": ""}) is None
    assert guard.record("report", {}) is None


# --------------------------- context trimming ---------------------------

def _output(text, i):
    return {"type": "function_call_output", "call_id": f"c{i}", "output": text}


def test_old_outputs_are_cut_and_new_ones_kept_whole():
    big = "h" * 500 + "m" * 3000 + "t" * 500
    items = [{"role": "user", "content": "task"}]
    for i in range(12):
        items.append({"type": "function_call", "call_id": f"c{i}", "name": "exec_command", "arguments": "{}"})
        items.append(_output(big, i))
    out = context_trim.trim_items(items, keep_recent=6)
    outputs = [item["output"] for item in out if item.get("type") == "function_call_output"]
    assert all(o == big for o in outputs[-6:])
    cut = outputs[0]
    assert cut.startswith("h" * 400) and cut.endswith("t" * 400) and "characters cut" in cut
    assert len(out) == len(items) and out[0] == items[0]
    assert items[1 + 1]["output"] == big  # the input list itself is untouched


def test_nothing_is_cut_until_a_batch_has_built_up():
    items = [_output("x" * 5000, i) for i in range(6 + context_trim.BATCH)]
    assert context_trim.trim_items(items, keep_recent=6) == items


def test_the_cut_point_moves_in_batches():
    def cut_count(n):
        out = context_trim.trim_items([_output("x" * 5000, i) for i in range(n)], keep_recent=6)
        return sum("characters cut" in o["output"] for o in out)
    assert cut_count(11) == 4 and cut_count(13) == 4 and cut_count(14) == 8


def test_a_run_with_no_known_limit_is_never_squeezed():
    watch = context_trim.ContextWatch(None)  # a remote API: no /props
    watch.seen(900_000)
    assert watch.keep_recent(6) == 6 and watch.note() is None


def test_nearing_the_slot_squeezes_then_finishes_and_stays_so():
    watch = context_trim.ContextWatch(64_000)
    watch.seen(40_000)
    assert watch.keep_recent(6) == 6 and watch.note() is None
    watch.seen(49_000)  # past 75%: old outputs cut harder
    assert watch.keep_recent(6) == context_trim.SQUEEZE_KEEP and watch.note() is None
    watch.seen(58_000)  # past 90%: told to finish
    assert watch.note() == context_trim.FINISH_NOTE
    watch.seen(20_000)  # the squeeze worked: still on, or the cut outputs come back
    assert watch.keep_recent(6) == context_trim.SQUEEZE_KEEP
    assert watch.note() == context_trim.FINISH_NOTE
    assert "`" not in context_trim.FINISH_NOTE  # names no tool: each run finishes its own way


def test_the_limit_is_the_servers_own_not_a_fixed_size():
    big = context_trim.ContextWatch(1_000_000)
    big.seen(60_000)  # 60k is nothing to a 1M-token model
    assert big.keep_recent(6) == 6 and big.note() is None


def test_a_short_output_is_left_as_it_is():
    assert context_trim.cut("short") == "short"


def test_a_different_dash_still_counts_as_kept():
    report = br.BrainReport(must_keep=["8-10°C"])
    assert br.missing_tokens("around 8–10°C today", report) == []
    assert br.missing_tokens("around 9°C today", report) == ["8-10°C"]


def test_the_name_of_a_posted_file_need_not_be_in_the_reply():
    report = br.BrainReport(must_keep=["chart.png", "https://x.test/a"], files_posted=["chart.png"])
    assert br.missing_tokens("Here you go!", report) == ["https://x.test/a"]


def test_markdown_bold_inside_a_kept_fact_still_counts():
    report = br.BrainReport(must_keep=["feels like ~15°C", "`out/chart.png`"])
    assert br.missing_tokens("It is 16°C, feels like **~15°C**. Saved out/chart.png", report) == []


def test_facts_sent_as_a_json_string_become_a_list():
    report = br.parse_report_args(
        {"status": "done", "facts": '["Made date.py", "Made README.md"]'})
    assert report.facts == ["Made date.py", "Made README.md"]
    # Text that only looks like a list stays text.
    assert br.parse_report_args({"facts": "[1] was read\nmore"}).facts == [
        "[1] was read", "more"]

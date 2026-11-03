"""
Unit tests for the in-flight task registry in core/classes/message_queue.py
(pure module, stdlib only): register/unregister of long tool runs per
channel and the in_flight_hint() prompt hint (which tool, how long it's
been running, what it was). Finished runs are never listed.

Each test uses its own channel id — the registry is keyed by channel, so
that fully isolates the global state.

Run from the repo root:
    PYTHONPATH=$(pwd) pytest core/tests/task_registry_tests.py
"""

import os
import sys


import classes.message_queue as prod_mq  # production-style import path
from classes import message_queue as mq

BASE = 1_000_000.0  # fake "now" anchor for deterministic elapsed times


def test_no_double_module_under_both_import_paths():
    # Both import paths must resolve to the same module object (aliasing at
    # the top of message_queue.py) — otherwise the registries would exist
    # twice and hooks/tests would write to a different one.
    assert prod_mq is mq


def test_slow_tool_names_match_tool_functions():
    # These must stay in sync with the FunctionTool names in
    # core/classes/tool_functions.py: the persona's one tool, inside which
    # every slow tool now runs.
    from classes.tool_functions import ask_brain
    assert mq.SLOW_TOOL_NAMES == {ask_brain.name}
    for name in mq.SLOW_TOOL_NAMES:
        assert name in mq.TOOL_DISPLAY, f"{name} needs a display label"


def test_register_and_unregister_roundtrip():
    ch = 1
    assert mq.register_task_run(ch, "🐳 code sandbox", "write a script", run_key="c1")
    # same run registered twice is a no-op
    assert not mq.register_task_run(ch, "🐳 code sandbox", "write a script", run_key="c1")
    assert mq.unregister_task_run(ch, run_key="c1")
    # double unregister / unknown key is a no-op, not an error
    assert not mq.unregister_task_run(ch, run_key="c1")
    assert not mq.unregister_task_run(ch, run_key="c2")
    assert mq.in_flight_hint(ch) == ""


def test_register_without_run_key_dedupes_on_source():
    ch = 2
    assert mq.register_task_run(ch, "🎨 image generation", "a red fox")
    assert not mq.register_task_run(ch, "🎨 image generation", "a red fox")
    # clean up with the derived key (truncated source)
    assert mq.unregister_task_run(ch, run_key=mq._truncate("a red fox"))
    assert mq.in_flight_hint(ch) == ""


def test_in_flight_hint_lists_running_tool_with_elapsed_and_source():
    ch = 3
    mq.register_task_run(ch, "🐳 code sandbox", "write a script that prints pi",
                         run_key="c1", started=BASE - 30)
    hint = mq.in_flight_hint(ch, now=BASE + 222)  # 4m 12s since BASE - 30
    assert "🐳 code sandbox" in hint
    assert "running for 4m 12s" in hint
    assert "write a script that prints pi" in hint
    assert hint.startswith("Note:")


def test_in_flight_hint_empty_when_idle():
    assert mq.in_flight_hint(4) == ""

def test_unregister_unknown_key_is_noop():
    ch = 8
    mq.register_task_run(ch, "🐳 code sandbox", "task one", run_key="c1", started=BASE - 30)
    mq.register_task_run(ch, "🎨 image generation", "a red fox", run_key="c2", started=BASE - 30)
    # removing the wrong key must not clear the other run
    assert not mq.unregister_task_run(ch, run_key="nope")
    assert not mq.unregister_task_run(ch)  # no key at all: also a no-op
    hint = mq.in_flight_hint(ch, now=BASE + 1)
    assert "task one" in hint and "a red fox" in hint


def test_hint_source_is_truncated():
    ch = 9
    long_source = "x" * 500
    mq.register_task_run(ch, "🐳 code sandbox", long_source, run_key="c1", started=BASE - 30)
    hint = mq.in_flight_hint(ch, now=BASE + 1)
    # _truncate keeps MAX_SOURCE_CHARS-1 chars + an ellipsis
    assert "x" * (mq.MAX_SOURCE_CHARS - 1) + "…" in hint
    assert "x" * mq.MAX_SOURCE_CHARS not in hint

def test_channels_are_isolated():
    ch_a, ch_b = 11, 12
    mq.register_task_run(ch_a, "🐳 code sandbox", "task a", run_key="c1", started=BASE - 30)
    assert mq.in_flight_hint(ch_b, now=BASE + 1) == ""
    assert mq.in_flight_hint(ch_a, now=BASE + 1) != ""


def test_format_elapsed():
    assert mq._format_elapsed(0) == "0s"
    assert mq._format_elapsed(59) == "59s"
    assert mq._format_elapsed(61) == "1m 01s"
    assert mq._format_elapsed(252) == "4m 12s"
    assert mq._format_elapsed(3661) == "1h 01m"
    assert mq._format_elapsed(-5) == "0s"  # never negative

def test_a_quick_brain_call_leaves_no_hint():
    # Most brain calls are one quick tool (a search, a memory save): a hint
    # for those would be noise in the next prompt.
    ch = 9011
    mq.register_task_run(ch, "🧠 brain", "save a memory", run_key="c1", started=BASE)
    assert mq.in_flight_hint(ch, now=BASE + mq.HINT_AFTER_SECONDS - 1) == ""
    mq.unregister_task_run(ch, run_key="c1")
    assert mq.in_flight_hint(ch, now=BASE + 3) == ""


def test_a_long_brain_call_shows_while_running_only():
    # A finished call is never listed: the channel history shows its result,
    # and "finished" (which cannot say whether it worked) made the persona
    # claim the job it was answering was already done.
    ch = 9012
    mq.register_task_run(ch, "🧠 brain", "chart the poll", run_key="c1", started=BASE)
    assert "running for" in mq.in_flight_hint(ch, now=BASE + mq.HINT_AFTER_SECONDS)
    mq.unregister_task_run(ch, run_key="c1")
    assert mq.in_flight_hint(ch, now=BASE + 41) == ""


def test_a_running_call_with_no_request_is_left_out():
    # With no request named, the entry would read as the message being
    # answered.
    ch = 77
    mq.register_task_run(ch, "🧠 brain", "", run_key="c1", started=BASE - 30)
    assert mq.in_flight_hint(ch, now=BASE + 10) == ""

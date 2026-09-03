"""
Unit tests for the in-flight task registry in core/classes/message_queue.py
(pure module, stdlib only): register/unregister of long tool runs per
channel, the recently-done bookkeeping, and the in_flight_hint() prompt
hint (which tool, how long it's been running, what it was, or that it
just finished).

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
    # core/classes/tool_functions.py — the registry only tracks the long
    # tools automatically.
    assert mq.SLOW_TOOL_NAMES == {"run_code_sandbox", "generate_image"}
    for name in mq.SLOW_TOOL_NAMES:
        assert name in mq.TOOL_DISPLAY, f"{name} needs a display label"


def test_register_and_unregister_roundtrip():
    ch = 1
    assert mq.register_task_run(ch, "🐳 code sandbox", "write a script", run_key="c1")
    # same run registered twice is a no-op
    assert not mq.register_task_run(ch, "🐳 code sandbox", "write a script", run_key="c1")
    # finish=False: plain roundtrip, no recently-done note left behind
    assert mq.unregister_task_run(ch, run_key="c1", finish=False)
    # double unregister / unknown key is a no-op, not an error
    assert not mq.unregister_task_run(ch, run_key="c1")
    assert not mq.unregister_task_run(ch, run_key="c2")
    assert mq.in_flight_hint(ch) == ""


def test_register_without_run_key_dedupes_on_source():
    ch = 2
    assert mq.register_task_run(ch, "🎨 image generation", "a red fox")
    assert not mq.register_task_run(ch, "🎨 image generation", "a red fox")
    # clean up with the derived key (truncated source)
    assert mq.unregister_task_run(ch, run_key=mq._truncate("a red fox"), finish=False)
    assert mq.in_flight_hint(ch) == ""


def test_in_flight_hint_lists_running_tool_with_elapsed_and_source():
    ch = 3
    mq.register_task_run(ch, "🐳 code sandbox", "write a script that prints pi",
                         run_key="c1", started=BASE)
    hint = mq.in_flight_hint(ch, now=BASE + 252)  # 4m 12s
    assert "🐳 code sandbox" in hint
    assert "running for 4m 12s" in hint
    assert "write a script that prints pi" in hint
    assert hint.startswith("Note:")


def test_in_flight_hint_empty_when_idle():
    assert mq.in_flight_hint(4) == ""


def test_unregister_finish_adds_recently_done_hint():
    ch = 5
    mq.register_task_run(ch, "🐳 code sandbox", "compute pi", run_key="c1", started=BASE)
    mq.unregister_task_run(ch, run_key="c1", ended=BASE)  # finish=True by default
    hint = mq.in_flight_hint(ch, now=BASE + 10)
    assert "🐳 code sandbox" in hint
    assert "finished 10s ago" in hint
    assert "compute pi" in hint
    assert "already sent to the channel" in hint
    # nothing is in flight anymore
    assert "running for" not in hint
    # ...and the note must not claim anything is still being processed:
    # that wording made the model promise an image that had already been
    # posted ("that edit is still on its way").
    assert "still being processed" not in hint
    assert "Already FINISHED" in hint


def test_hint_separates_running_from_finished():
    """A channel with one live run and one just-finished run gets both
    groups under their own headers, so "still being processed" can never
    be read as covering the finished one."""
    ch = 51
    mq.register_task_run(ch, "🐳 code sandbox", "compute pi", run_key="c1", started=BASE)
    mq.register_task_run(ch, "🖌️ image editing", "add a red hat", run_key="c2", started=BASE)
    mq.unregister_task_run(ch, run_key="c2", ended=BASE)
    hint = mq.in_flight_hint(ch, now=BASE + 10)
    running_at = hint.index("still being processed")
    done_at = hint.index("Already FINISHED")
    assert running_at < hint.index("compute pi") < done_at
    assert done_at < hint.index("add a red hat")



def test_recently_done_ages_out():
    ch = 6
    mq.register_task_run(ch, "🎨 image generation", "a red fox", run_key="c1", started=BASE)
    mq.unregister_task_run(ch, run_key="c1", ended=BASE)
    # still visible inside the window
    assert mq.in_flight_hint(ch, now=BASE + mq.RECENT_DONE_SECONDS - 1) != ""
    # pruned after the window
    assert mq.in_flight_hint(ch, now=BASE + mq.RECENT_DONE_SECONDS + 1) == ""
    assert 6 not in mq._recent_done  # stale entry dropped from the registry


def test_unregister_without_finish_leaves_no_trace():
    ch = 7
    mq.register_task_run(ch, "🐳 code sandbox", "compute pi", run_key="c1", started=BASE)
    mq.unregister_task_run(ch, run_key="c1", finish=False)
    assert mq.in_flight_hint(ch, now=BASE + 10) == ""


def test_unregister_unknown_key_is_noop():
    ch = 8
    mq.register_task_run(ch, "🐳 code sandbox", "task one", run_key="c1", started=BASE)
    mq.register_task_run(ch, "🎨 image generation", "a red fox", run_key="c2", started=BASE)
    # removing the wrong key must not clear the other run
    assert not mq.unregister_task_run(ch, run_key="nope")
    assert not mq.unregister_task_run(ch)  # no key at all: also a no-op
    hint = mq.in_flight_hint(ch, now=BASE + 1)
    assert "task one" in hint and "a red fox" in hint


def test_hint_source_is_truncated():
    ch = 9
    long_source = "x" * 500
    mq.register_task_run(ch, "🐳 code sandbox", long_source, run_key="c1", started=BASE)
    hint = mq.in_flight_hint(ch, now=BASE + 1)
    # _truncate keeps MAX_SOURCE_CHARS-1 chars + an ellipsis
    assert "x" * (mq.MAX_SOURCE_CHARS - 1) + "…" in hint
    assert "x" * mq.MAX_SOURCE_CHARS not in hint


def test_recent_done_list_is_capped():
    ch = 10
    for i in range(mq.MAX_RECENT_DONE + 3):
        mq.register_task_run(ch, "🐳 code sandbox", f"task {i}", run_key=f"c{i}",
                             started=BASE - 100 + i)
        mq.unregister_task_run(ch, run_key=f"c{i}", ended=BASE - 100 + i)
    lines = [l for l in mq.in_flight_hint(ch, now=BASE + 1).splitlines() if l.startswith("- ")]
    assert len(lines) == mq.MAX_RECENT_DONE  # oldest dropped, newest kept
    assert "task " + str(mq.MAX_RECENT_DONE + 2) in lines[-1]


def test_channels_are_isolated():
    ch_a, ch_b = 11, 12
    mq.register_task_run(ch_a, "🐳 code sandbox", "task a", run_key="c1", started=BASE)
    assert mq.in_flight_hint(ch_b, now=BASE + 1) == ""
    assert mq.in_flight_hint(ch_a, now=BASE + 1) != ""


def test_format_elapsed():
    assert mq._format_elapsed(0) == "0s"
    assert mq._format_elapsed(59) == "59s"
    assert mq._format_elapsed(61) == "1m 01s"
    assert mq._format_elapsed(252) == "4m 12s"
    assert mq._format_elapsed(3661) == "1h 01m"
    assert mq._format_elapsed(-5) == "0s"  # never negative
import os
import sys

# sandbox_thread_inbox is imported as core.classes.* here and classes.* in
# production (the app runs with cwd=core/); both resolve as namespace packages.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from core.classes import sandbox_thread_inbox as inbox

# To run this pytest file from the command line, use:
# PYTHONPATH=$(pwd) pytest core/tests/sandbox_thread_inbox_tests.py


@pytest.fixture(autouse=True)
def clean_registry():
    """The registry is module-level process state, so a test that leaves a
    thread registered would silently change the next test's behavior."""
    inbox._PENDING.clear()
    yield
    inbox._PENDING.clear()


# ---------------------- run lifecycle ----------------------

def test_no_run_is_active_by_default():
    assert inbox.is_run_active(1) is False


def test_begin_run_marks_the_thread_active():
    inbox.begin_run(1)
    assert inbox.is_run_active(1) is True


def test_end_run_marks_the_thread_inactive():
    inbox.begin_run(1)
    inbox.end_run(1)
    assert inbox.is_run_active(1) is False


def test_end_run_on_an_unknown_thread_is_a_no_op():
    inbox.end_run(999)  # must not raise


def test_begin_run_is_idempotent_and_keeps_pending_messages():
    # A double-register (e.g. the tool-level guard racing on_message) must
    # not silently drop a message already queued for the run.
    inbox.begin_run(1)
    inbox.deliver(1, 10, "ana", "hello")
    inbox.begin_run(1)
    assert inbox.drain(1) == "[thread message from ana]: hello"


def test_end_run_discards_undrained_messages():
    # They were addressed to a run that is now over; keeping them would leak
    # them into an unrelated later run in the same thread.
    inbox.begin_run(1)
    inbox.deliver(1, 10, "ana", "hello")
    inbox.end_run(1)
    inbox.begin_run(1)
    assert inbox.drain(1) == ""


def test_a_run_in_one_thread_does_not_activate_another():
    inbox.begin_run(1)
    assert inbox.is_run_active(2) is False


def test_thread_ids_are_coerced_so_a_string_id_still_matches():
    inbox.begin_run(1)
    assert inbox.is_run_active("1") is True


# ---------------------- deliver ----------------------

def test_deliver_is_rejected_when_no_run_is_active():
    assert inbox.deliver(1, 10, "ana", "hello") is False


def test_deliver_accepts_a_message_during_a_run():
    inbox.begin_run(1)
    assert inbox.deliver(1, 10, "ana", "hello") is True


def test_deliver_rejects_an_empty_or_whitespace_message():
    inbox.begin_run(1)
    assert inbox.deliver(1, 10, "ana", "") is False
    assert inbox.deliver(1, 11, "ana", "   \n ") is False


def test_deliver_falls_back_to_a_placeholder_author():
    inbox.begin_run(1)
    inbox.deliver(1, 10, "", "hello")
    assert inbox.drain(1) == "[thread message from someone]: hello"


def test_deliver_truncates_an_over_long_single_message():
    inbox.begin_run(1)
    inbox.deliver(1, 10, "ana", "x" * (inbox.MAX_MESSAGE_CHARS + 500))
    drained = inbox.drain(1)
    assert drained.endswith("…")
    assert len(drained) < inbox.MAX_MESSAGE_CHARS + 100


def test_deliver_stops_accepting_past_the_message_count_cap():
    inbox.begin_run(1)
    for i in range(inbox.MAX_PENDING_MESSAGES):
        assert inbox.deliver(1, i, "ana", "hi") is True
    assert inbox.deliver(1, 999, "ana", "one too many") is False


def test_deliver_stops_accepting_past_the_total_char_cap():
    inbox.begin_run(1)
    # Well under MAX_PENDING_MESSAGES, so this can only be the char cap.
    chunk = "y" * inbox.MAX_MESSAGE_CHARS
    accepted = 0
    for i in range(inbox.MAX_PENDING_MESSAGES):
        if inbox.deliver(1, i, "ana", chunk):
            accepted += 1
    assert accepted < inbox.MAX_PENDING_MESSAGES
    assert accepted * inbox.MAX_MESSAGE_CHARS <= inbox.MAX_PENDING_CHARS


# ---------------------- consume ----------------------

def test_consume_removes_only_the_named_message():
    inbox.begin_run(1)
    inbox.deliver(1, 10, "ana", "the reply")
    inbox.deliver(1, 11, "bo", "something else")
    inbox.consume(1, 10)
    assert inbox.drain(1) == "[thread message from bo]: something else"


def test_consume_is_a_no_op_for_an_unknown_message_or_thread():
    inbox.begin_run(1)
    inbox.deliver(1, 10, "ana", "hello")
    inbox.consume(1, 999)
    inbox.consume(999, 10)
    assert inbox.drain(1) == "[thread message from ana]: hello"


# ---------------------- drain ----------------------

def test_drain_is_empty_when_nothing_was_delivered():
    inbox.begin_run(1)
    assert inbox.drain(1) == ""


def test_drain_is_empty_for_a_thread_with_no_run():
    assert inbox.drain(1) == ""


def test_drain_formats_every_pending_message_in_order():
    inbox.begin_run(1)
    inbox.deliver(1, 10, "ana", "make it blue")
    inbox.deliver(1, 11, "bo", "and smaller")
    assert inbox.drain(1) == (
        "[thread message from ana]: make it blue\n"
        "[thread message from bo]: and smaller"
    )


def test_drain_empties_the_queue_but_keeps_the_run_active():
    inbox.begin_run(1)
    inbox.deliver(1, 10, "ana", "hello")
    inbox.drain(1)
    assert inbox.drain(1) == ""
    # draining must not deregister the run — messages posted after it still
    # need to reach the sandbox
    assert inbox.is_run_active(1) is True
    assert inbox.deliver(1, 11, "ana", "again") is True

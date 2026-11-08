"""The server workspace's settings and state machine (classes/workspace_lifecycle.py).

Run from the repo root: `pytest core/tests/workspace_lifecycle_tests.py`
"""
import pytest

from classes import workspace_lifecycle as wl
from classes.workspace_lifecycle import (
    COLD, EVICT, SNAPSHOT, START, STOP, STOPPING, USE, WAIT, WARM, Lifecycle,
)


def _lc(idle=100, interval=1000, max_warm=2):
    return Lifecycle(idle=idle, interval=interval, max_warm=max_warm)


def _warm(lc, guild, now=0):
    assert lc.acquire(guild, now).action == START
    lc.started(guild, now)


# ---------------------- settings ----------------------

def test_settings_defaults():
    env = {}
    assert wl.workspace_image(env) == "python:3.14-slim"
    assert wl.idle_seconds(env) == 900
    assert wl.snapshot_interval_seconds(env) == 1800
    assert wl.max_warm(env) == 4
    assert wl.job_retention_hours(env) == 72
    assert wl.pids_limit(env) == 512
    assert wl.cpu_limit(env) is None
    assert wl.snapshot_timeout_seconds(env) == 120
    assert wl.instance(env) == "dcb-dev"
    assert wl.prewarm(env) is True


@pytest.mark.parametrize("name, raw, reader, expected", [
    ("WORKSPACE_IDLE_SECONDS", "30", wl.idle_seconds, 30),
    ("WORKSPACE_IDLE_SECONDS", "0", wl.idle_seconds, 900),
    ("WORKSPACE_IDLE_SECONDS", "junk", wl.idle_seconds, 900),
    ("WORKSPACE_SNAPSHOT_INTERVAL_SECONDS", "0", wl.snapshot_interval_seconds, 0),
    ("WORKSPACE_MAX_WARM", "0", wl.max_warm, 4),
    ("WORKSPACE_PIDS_LIMIT", "0", wl.pids_limit, 0),
    ("WORKSPACE_CPU_LIMIT", "1.5", wl.cpu_limit, 1.5),
    ("WORKSPACE_CPU_LIMIT", "-1", wl.cpu_limit, None),
    ("WORKSPACE_CPU_LIMIT", "two", wl.cpu_limit, None),
    ("WORKSPACE_INSTANCE", " prod ", wl.instance, "prod"),
    ("WORKSPACE_PREWARM", "true", wl.prewarm, True),
    ("WORKSPACE_PREWARM", "0", wl.prewarm, False),
    ("WORKSPACE_IMAGE", "  ", wl.workspace_image, "python:3.14-slim"),
])
def test_settings_parse(name, raw, reader, expected):
    assert reader({name: raw}) == expected


def test_names():
    assert wl.job_dir("t5") == "jobs/t5"
    assert wl.snapshot_id(42) == "guild-42"
    assert wl.labels(42, "dcb-dev") == {"dcb.role": "workspace", "dcb.guild": "42",
                                         "dcb.instance": "dcb-dev"}


# ---------------------- start, use, release ----------------------

def test_first_acquire_starts_and_later_ones_use():
    lc = _lc()
    assert lc.acquire("g", 0).action == START
    # Someone else asking while it starts waits rather than starting a second one.
    assert lc.acquire("g", 1).action == WAIT
    lc.started("g", 2)
    assert lc.acquire("g", 3).action == USE
    assert lc.state("g").refcount == 2


def test_a_failed_start_goes_back_to_cold():
    lc = _lc()
    lc.acquire("g", 0)
    lc.start_failed("g")
    assert lc.state("g").phase == COLD and lc.state("g").refcount == 0
    assert lc.acquire("g", 1).action == START


def test_the_last_release_asks_for_cleanup_and_blocks_new_calls_until_done():
    lc = _lc()
    _warm(lc, "g")
    lc.acquire("g", 1)
    assert lc.release("g", 2) is False  # one call still running
    assert lc.release("g", 3) is True
    assert lc.acquire("g", 4).action == WAIT  # cleanup in progress
    assert lc.cleaned("g", 5) is False
    assert lc.state("g").idle_since == 5
    assert lc.acquire("g", 6).action == USE


def test_release_marks_the_workspace_changed():
    lc = _lc()
    _warm(lc, "g")
    assert lc.state("g").dirty is False
    lc.release("g", 1)
    assert lc.state("g").dirty is True


def test_a_prewarm_release_does_not_mark_it_changed():
    lc = _lc()
    _warm(lc, "g")
    lc.release("g", 1, changed=False)
    assert lc.state("g").dirty is False  # so the idle stop takes no snapshot


def test_a_release_with_nothing_held_is_ignored():
    lc = _lc()
    assert lc.release("g", 0) is False


# ---------------------- timers ----------------------

def test_idle_stop_only_after_the_idle_time_and_only_when_unused():
    lc = _lc(idle=100)
    _warm(lc, "g")
    lc.release("g", 10)
    assert lc.due(50) == []  # still cleaning: no idle clock yet
    lc.cleaned("g", 10)
    assert lc.due(109) == []
    assert lc.due(110) == [("g", STOP)]
    lc.acquire("g", 111)
    assert lc.due(500) == []  # busy again


def test_timed_snapshot_of_a_changed_workspace_even_while_busy():
    lc = _lc(idle=10_000, interval=100)
    _warm(lc, "g", now=0)
    assert lc.due(200) == []  # nothing changed yet
    lc.acquire("g", 1)
    lc.release("g", 2)  # one of two calls done: dirty, still busy
    assert lc.due(99) == []
    assert lc.due(100) == [("g", SNAPSHOT)]
    lc.snapshotted("g", 100, ok=True)
    assert lc.due(300) == []  # clean again
    lc.release("g", 301)
    lc.cleaned("g", 301)
    assert lc.due(400) == [("g", SNAPSHOT)]


def test_a_failed_snapshot_stays_dirty_but_waits_another_interval():
    lc = _lc(idle=10_000, interval=100)
    _warm(lc, "g")
    lc.release("g", 1)
    lc.cleaned("g", 1)
    lc.snapshotted("g", 100, ok=False)
    assert lc.state("g").dirty is True
    assert lc.due(150) == []
    assert lc.due(200) == [("g", SNAPSHOT)]


def test_interval_zero_means_idle_stop_only():
    lc = _lc(idle=100, interval=0)
    _warm(lc, "g")
    lc.release("g", 1)
    lc.cleaned("g", 1)
    assert lc.due(50) == []
    assert lc.due(101) == [("g", STOP)]


def test_begin_stop_refuses_a_busy_or_cold_workspace():
    lc = _lc()
    assert lc.begin_stop("g") is False
    _warm(lc, "g")
    assert lc.begin_stop("g") is False  # refcount 1
    lc.release("g", 1)
    assert lc.begin_stop("g") is False  # cleaning
    lc.cleaned("g", 1)
    assert lc.begin_stop("g") is True
    assert lc.acquire("g", 2).action == WAIT
    lc.stopped("g")
    assert lc.acquire("g", 3).action == START


# ---------------------- eviction ----------------------

def test_a_new_guild_evicts_the_least_recently_used_idle_one():
    lc = _lc(max_warm=2)
    for g, t in (("a", 0), ("b", 5)):
        _warm(lc, g, t)
        lc.release(g, t + 1)
        lc.cleaned(g, t + 1)
    d = lc.acquire("c", 10)
    assert (d.action, d.victim) == (EVICT, "a")
    assert lc.state("a").phase == STOPPING
    lc.stopped("a")
    assert lc.acquire("c", 11).action == START


def test_eviction_waits_when_every_warm_workspace_is_busy():
    lc = _lc(max_warm=2)
    _warm(lc, "a")
    _warm(lc, "b")
    assert lc.acquire("c", 1).action == WAIT
    lc.release("a", 2)
    assert lc.acquire("c", 3).action == WAIT  # a is still cleaning
    lc.cleaned("a", 3)
    assert lc.acquire("c", 4).action == EVICT


def test_a_starting_or_stopping_workspace_counts_towards_the_cap():
    lc = _lc(max_warm=1)
    lc.acquire("a", 0)  # STARTING
    assert lc.acquire("b", 1).action == WAIT


# ---------------------- reset ----------------------

def test_reset_of_an_idle_workspace_runs_now():
    lc = _lc()
    _warm(lc, "g")
    lc.release("g", 1)
    lc.cleaned("g", 1)
    assert lc.request_reset("g") == "now"
    assert lc.state("g").phase == STOPPING


def test_reset_of_a_cold_workspace_runs_now():
    assert _lc().request_reset("g") == "now"


def test_reset_while_busy_waits_for_the_last_call_and_blocks_new_ones():
    lc = _lc()
    _warm(lc, "g")
    assert lc.request_reset("g") == "later"
    assert lc.acquire("g", 1).action == WAIT
    assert lc.due(10_000) == []
    assert lc.release("g", 2) is True
    assert lc.cleaned("g", 3) is True  # the reset runs now
    assert lc.state("g").phase == STOPPING
    lc.stopped("g")
    assert lc.acquire("g", 4).action == START


def test_is_warm_and_warm_guilds():
    lc = _lc()
    assert lc.is_warm("g") is False
    _warm(lc, "g")
    assert lc.is_warm("g") is True
    assert lc.warm_guilds() == ["g"]
    assert lc.state("g").phase == WARM


def test_claim_cold_only_takes_a_guild_nobody_here_is_using():
    lc = _lc()
    assert lc.claim_cold("g") is True
    assert lc.acquire("g", 0).action == WAIT
    lc.stopped("g")
    _warm(lc, "g")
    assert lc.claim_cold("g") is False


def test_reset_while_stopping_waits_for_the_stop_and_then_runs():
    lc = _lc()
    _warm(lc, "g")
    lc.release("g", 1)
    lc.cleaned("g", 1)
    assert lc.begin_stop("g") is True
    assert lc.request_reset("g") == "later"
    assert lc.stopped("g") is True  # the reset runs now
    assert lc.state("g").phase == STOPPING
    assert lc.acquire("g", 2).action == WAIT
    assert lc.stopped("g", wiped=True) is False
    assert lc.state("g").phase == COLD


def test_reset_asked_for_while_starting_survives_a_failed_start():
    lc = _lc()
    assert lc.acquire("g", 0).action == START
    assert lc.request_reset("g") == "later"
    assert lc.start_failed("g") is True
    assert lc.state("g").phase == STOPPING


def test_a_plain_stop_or_failed_start_has_no_reset():
    lc = _lc()
    _warm(lc, "g")
    lc.release("g", 1)
    lc.cleaned("g", 1)
    lc.begin_stop("g")
    assert lc.stopped("g") is False and lc.state("g").phase == COLD
    assert lc.acquire("g", 2).action == START
    assert lc.start_failed("g") is False and lc.state("g").phase == COLD

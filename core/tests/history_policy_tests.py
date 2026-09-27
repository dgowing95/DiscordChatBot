"""Tests for classes.history_policy (pure: no discord, Redis or LLM).

To run this pytest file from the command line, use:
    pytest core/tests/history_policy_tests.py
"""
import pytest

from classes import history_policy as hp
from classes.history_policy import AnchorState, Decision


# ---------------------------------------------------------------- env parsing


def test_history_mode_defaults_to_sliding(monkeypatch):
    monkeypatch.delenv("MSG_HISTORY_MODE", raising=False)
    assert hp.history_mode() == hp.SLIDING


@pytest.mark.parametrize("raw, expected", [
    ("anchored", hp.ANCHORED), (" Anchored ", hp.ANCHORED),
    ("sliding", hp.SLIDING), ("", hp.SLIDING), ("bogus", hp.SLIDING),
])
def test_history_mode_parsing(monkeypatch, raw, expected):
    monkeypatch.setenv("MSG_HISTORY_MODE", raw)
    assert hp.history_mode() == expected


@pytest.mark.parametrize("fn, name, default", [
    (hp.history_limit, "MSG_HISTORY_LIMIT", 5),
    (hp.refresh_messages, "MSG_HISTORY_REFRESH_MESSAGES", 10),
    (hp.reserve_tokens, "MSG_HISTORY_RESERVE_TOKENS", 12000),
])
def test_positive_int_settings(monkeypatch, fn, name, default):
    monkeypatch.delenv(name, raising=False)
    assert fn() == default
    for raw, expected in (("25", 25), ("0", default), ("-3", default), ("x", default), ("", default)):
        monkeypatch.setenv(name, raw)
        assert fn() == expected, raw


# ------------------------------------------------------------------ decisions


def test_no_state_is_a_cold_refresh():
    assert hp.initial_decision(None, 100) == Decision(hp.REFRESH, hp.REASON_COLD)


def test_trigger_older_than_last_refresh_gets_a_private_snapshot():
    state = AnchorState(after_id=10, refresh_trigger_id=50)
    assert hp.initial_decision(state, 40) == Decision(hp.SNAPSHOT, hp.REASON_STALE)


def test_newer_trigger_goes_on_to_fetch():
    state = AnchorState(after_id=10, refresh_trigger_id=50)
    assert hp.initial_decision(state, 60) is None


def _ids(*ids):
    return [(i, False) for i in ids]


def test_max_window():
    # 24 preceding + the old trigger + at most 8 more before a refresh fires
    assert hp.max_window(25, 10) == 33
    assert hp.max_window(1, 1) == 0


def test_threshold_exact_boundary():
    """Refresh when the messages after the last refresh trigger, plus this
    trigger, reach refresh_every - bot replies count like any message."""
    state = AnchorState(after_id=0, refresh_trigger_id=10)
    before = list(range(1, 11))  # the window up to and including trigger 10
    # 8 newer messages + this trigger = 9 -> keep
    assert hp.window_decision(state, 30, _ids(*before, *range(11, 19)), 25, 10).kind == hp.KEEP
    # 9 newer + this trigger = 10 -> refresh
    assert hp.window_decision(state, 30, _ids(*before, *range(11, 20)), 25, 10) == \
        Decision(hp.REFRESH, hp.REASON_THRESHOLD)


def test_burst_past_the_fetch_cap_refreshes():
    state = AnchorState(after_id=0, refresh_trigger_id=10)
    fetched = _ids(*range(1, 1 + hp.max_window(25, 10) + 1))
    assert hp.window_decision(state, 999, fetched, 25, 10) == Decision(hp.REFRESH, hp.REASON_BURST)


def test_reset_inside_the_window_refreshes():
    state = AnchorState(after_id=0, refresh_trigger_id=10)
    fetched = _ids(1, 2, 10) + [(11, True)]
    assert hp.window_decision(state, 12, fetched, 25, 10) == Decision(hp.REFRESH, hp.REASON_RESET)


def test_empty_window_keeps():
    state = AnchorState(after_id=9, refresh_trigger_id=10)
    assert hp.window_decision(state, 11, [], 25, 10).kind == hp.KEEP


def test_state_after_refresh_bounds_below_the_oldest_included():
    assert hp.state_after_refresh([5, 7, 9], 12) == AnchorState(after_id=4, refresh_trigger_id=12)


def test_state_after_refresh_with_nothing_included_starts_at_the_trigger():
    # e.g. a !reset_history right before the trigger: the reset (id < 12) is
    # excluded by an exclusive bound of 11
    assert hp.state_after_refresh([], 12) == AnchorState(after_id=11, refresh_trigger_id=12)


# ---------------------------------------------------------------------- store


def test_store_commit_and_get():
    store = hp.HistoryStore()
    key = (1, 2)
    assert store.get(key) is None
    assert store.commit(key, AnchorState(4, 12))
    assert store.get(key) == AnchorState(4, 12)


def test_store_never_moves_the_anchor_backwards():
    store = hp.HistoryStore()
    key = (1, 2)
    store.commit(key, AnchorState(40, 50))
    assert not store.commit(key, AnchorState(10, 30))
    assert not store.commit(key, AnchorState(10, 50))
    assert store.get(key) == AnchorState(40, 50)
    assert store.commit(key, AnchorState(45, 60))


def test_store_channels_are_isolated():
    store = hp.HistoryStore()
    store.commit((1, 2), AnchorState(4, 12))
    store.commit((1, 3), AnchorState(7, 20))
    assert store.get((1, 2)) == AnchorState(4, 12)
    assert store.get((1, 3)) == AnchorState(7, 20)
    assert store.get((9, 2)) is None


def test_store_evicts_least_recently_used():
    store = hp.HistoryStore(max_channels=2)
    store.commit("a", AnchorState(1, 2))
    store.commit("b", AnchorState(1, 2))
    store.get("a")                    # a is now most recent
    store.commit("c", AnchorState(1, 2))
    assert len(store) == 2
    assert store.get("b") is None     # evicted: the next build is just a cold refresh
    assert store.get("a") is not None and store.get("c") is not None


# --------------------------------------------------------------- token budget


def test_estimate_tokens_text_and_images():
    text = {"role": "user", "content": "x" * 30}
    assert hp.estimate_tokens([text]) == 10 + hp.MESSAGE_OVERHEAD_TOKENS
    image = {"role": "user", "content": [
        {"type": "input_image", "image_url": "data:image/png;base64," + "A" * 10000},
        {"type": "text", "text": "y" * 3},
    ]}
    assert hp.estimate_tokens([image]) == hp.IMAGE_TOKENS + 1 + hp.MESSAGE_OVERHEAD_TOKENS
    assert hp.estimate_tokens([]) == 0


def test_history_budget():
    assert hp.history_budget(None, 12000) is None
    assert hp.history_budget(0, 12000) is None
    assert hp.history_budget(30208, 12000) == 18208
    # a reserve bigger than the slot does not trim everything
    assert hp.history_budget(8192, 12000) == 4096


def test_trim_count_drops_oldest_until_it_fits():
    assert hp.trim_count([100, 100, 100], 50, 400) == 0
    assert hp.trim_count([100, 100, 100], 50, 300) == 1
    assert hp.trim_count([100, 100, 100], 50, 100) == 3


def test_trim_count_never_trims_without_a_budget():
    assert hp.trim_count([10 ** 6], 10 ** 6, None) == 0


def test_trim_count_keeps_the_trigger_even_when_it_alone_is_over():
    # everything else goes, the trigger (fixed) is still sent
    assert hp.trim_count([100, 100], 10 ** 6, 500) == 2

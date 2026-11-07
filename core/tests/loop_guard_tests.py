"""The loop guard's rules (classes/loop_guard.py, pure)."""
from classes.loop_guard import AGAIN, REPEAT, STALE, LoopGuard


def test_the_same_page_fetched_again_and_again_is_a_loop_however_far_apart():
    guard = LoopGuard(repeat_limit=3)
    page = {"url": "https://example.com/a", "part": 1}
    calls = [("fetch_url", page), ("web_search", {"query": "one"}), ("fetch_url", page),
             ("exec_command", {"cmd": "ls"}), ("fetch_url", page)]
    kinds = [guard.record(tool, args) for tool, args in calls]
    assert kinds == [None, None, None, None, AGAIN]
    assert "gives the same answer" in guard.note()
    assert not guard.should_stop()


def test_another_part_of_a_page_is_new():
    guard = LoopGuard(repeat_limit=3)
    for part in (1, 2, 3, 4):
        assert guard.record("fetch_url", {"url": "https://example.com/a", "part": part}) is None


def test_a_command_run_again_between_edits_is_honest_work():
    guard = LoopGuard(repeat_limit=3)
    for edit in range(4):
        assert guard.record("exec_command", {"cmd": "python test.py"}) is None
        assert guard.record("exec_command", {"cmd": f"sed -i s/{edit}/x/ a.py"}) is None


def test_after_the_warning_three_more_fetches_stop_the_run():
    guard = LoopGuard(repeat_limit=3)
    page = {"url": "https://example.com/a"}
    for _ in range(3):
        guard.record("fetch_url", page)
        guard.record("web_search", {"query": str(_)})
    assert guard.loops == 1 and not guard.should_stop()
    guard.record("fetch_url", page)
    guard.record("fetch_url", page)
    assert not guard.should_stop()
    guard.record("fetch_url", page)
    assert guard.should_stop()


def test_the_older_rules_still_hold():
    guard = LoopGuard(repeat_limit=3)
    assert [guard.record("exec_command", {"cmd": "ls"}) for _ in range(3)][-1] == REPEAT
    guard = LoopGuard(repeat_limit=9, window=2)
    guard.record("exec_command", {"cmd": "a"})
    guard.record("exec_command", {"cmd": "b"})
    guard.record("exec_command", {"cmd": "a"})
    assert guard.record("exec_command", {"cmd": "b"}) == STALE

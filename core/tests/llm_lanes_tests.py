"""Priority lanes over one LLM server's slots (classes/llm_lanes.py).

Run from the repo root: `pytest core/tests/llm_lanes_tests.py`
"""
import asyncio

import pytest

from classes import llm_lanes
from classes.llm_lanes import BACKGROUND, BRAIN, CHAT, SUBAGENT, Lanes


async def _settle():
    for _ in range(5):
        await asyncio.sleep(0)


def _waiter(lanes, lane, order):
    async def run():
        await lanes.acquire(lane)
        order.append(lane)
    return asyncio.create_task(run())


@pytest.mark.asyncio
async def test_a_freed_slot_goes_to_the_highest_priority_waiter():
    lanes = Lanes(1, 1)
    await lanes.acquire(CHAT)
    order = []
    tasks = [_waiter(lanes, lane, order) for lane in (BACKGROUND, SUBAGENT, BRAIN, CHAT)]
    await _settle()
    assert order == []
    for _ in range(4):
        lanes._release(order[-1] if order else CHAT)
        await _settle()
    await asyncio.gather(*tasks)
    assert order == [CHAT, BRAIN, SUBAGENT, BACKGROUND]


@pytest.mark.asyncio
async def test_first_come_first_served_within_a_lane():
    lanes = Lanes(1, 0)
    await lanes.acquire(BRAIN)
    order = []

    async def run(name):
        await lanes.acquire(BRAIN)
        order.append(name)
        lanes._release(BRAIN)

    tasks = [asyncio.create_task(run(n)) for n in ("a", "b", "c")]
    await _settle()
    lanes._release(BRAIN)
    await asyncio.gather(*tasks)
    assert order == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_the_reserved_slot_is_chats_alone():
    lanes = Lanes(2, 1)
    assert await lanes.acquire(BRAIN) == 0.0
    blocked = asyncio.create_task(lanes.acquire(SUBAGENT))
    await _settle()
    assert not blocked.done()  # the one shared slot is taken
    assert await lanes.acquire(CHAT) == 0.0  # chat still gets the reserved one
    lanes._release(BRAIN)
    await asyncio.wait_for(blocked, 1)
    assert lanes.snapshot()[SUBAGENT]["in_use"] == 1


@pytest.mark.asyncio
async def test_chat_may_use_every_slot():
    lanes = Lanes(3, 1)
    for _ in range(3):
        assert await lanes.acquire(CHAT) == 0.0
    assert lanes.snapshot()[CHAT]["in_use"] == 3


@pytest.mark.asyncio
async def test_one_slot_other_work_goes_only_when_chat_is_idle():
    lanes = Lanes(1, 1)
    assert await lanes.acquire(BRAIN) == 0.0  # chat idle: brain may use it
    chat = asyncio.create_task(lanes.acquire(CHAT))
    await _settle()
    # Chat is now waiting: a second brain call must not take the slot first.
    brain = asyncio.create_task(lanes.acquire(BRAIN))
    lanes._release(BRAIN)
    await _settle()
    assert chat.done() and not brain.done()
    lanes._release(CHAT)
    await asyncio.wait_for(brain, 1)


@pytest.mark.asyncio
async def test_a_newcomer_never_jumps_the_queue():
    lanes = Lanes(2, 0)
    await lanes.acquire(BRAIN)
    await lanes.acquire(BRAIN)
    first = asyncio.create_task(lanes.acquire(BRAIN))
    await _settle()
    lanes._in_use[BRAIN] -= 1  # a slot frees without a wake (mid-release)
    second = asyncio.create_task(lanes.acquire(BRAIN))
    await _settle()
    assert not second.done()  # it would have fit, but `first` was waiting
    lanes._wake()
    await _settle()
    assert first.done() and not second.done()
    second.cancel()


@pytest.mark.asyncio
async def test_cancelling_a_waiter_removes_it():
    lanes = Lanes(1, 1)
    await lanes.acquire(CHAT)
    waiting = asyncio.create_task(lanes.acquire(BRAIN))
    await _settle()
    assert lanes.snapshot()[BRAIN]["waiting"] == 1
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    assert lanes.snapshot()[BRAIN]["waiting"] == 0
    lanes._release(CHAT)
    assert lanes.snapshot()[BRAIN]["in_use"] == 0


@pytest.mark.asyncio
async def test_a_waiter_cancelled_as_it_is_granted_gives_the_slot_back():
    lanes = Lanes(1, 1)
    await lanes.acquire(CHAT)
    waiting = asyncio.create_task(lanes.acquire(BRAIN))
    await _settle()
    lanes._release(CHAT)  # grants it: the future is resolved...
    waiting.cancel()      # ...and the task cancelled before it resumes
    with pytest.raises(asyncio.CancelledError):
        await waiting
    assert lanes.snapshot()[BRAIN]["in_use"] == 0


@pytest.mark.asyncio
async def test_slot_releases_on_error():
    lanes = Lanes(1, 1)
    with pytest.raises(RuntimeError):
        async with lanes.slot(CHAT):
            raise RuntimeError("model failed")
    assert lanes.snapshot()[CHAT]["in_use"] == 0


@pytest.mark.asyncio
async def test_shrinking_never_revokes_a_held_slot():
    lanes = Lanes(3, 1)
    for _ in range(3):
        await lanes.acquire(CHAT)
    lanes.configure(1, 1)
    assert lanes.snapshot()[CHAT]["in_use"] == 3
    waiting = asyncio.create_task(lanes.acquire(CHAT))
    lanes._release(CHAT)
    lanes._release(CHAT)
    await _settle()
    assert not waiting.done()  # 1 still held, and there is 1 slot now
    lanes._release(CHAT)
    await asyncio.wait_for(waiting, 1)


@pytest.mark.asyncio
async def test_growing_wakes_waiters():
    lanes = Lanes(1, 1)
    await lanes.acquire(CHAT)
    waiting = asyncio.create_task(lanes.acquire(CHAT))
    await _settle()
    lanes.configure(2, 1)
    await asyncio.wait_for(waiting, 1)


def test_chat_reserved_is_clamped():
    assert Lanes(2, 5).chat_reserved == 2
    assert Lanes(2, -1).chat_reserved == 0
    assert Lanes(0, 1).slots == 1


@pytest.mark.asyncio
async def test_an_unknown_lane_is_refused():
    with pytest.raises(ValueError):
        await Lanes(1, 1).acquire("vip")


# ---------------------- settings and registry ----------------------

@pytest.mark.parametrize("props, env, expected", [
    (4, {}, (4, "props")),
    (4, {"LLM_SLOTS": "2"}, (4, "props")),  # the server's own count wins
    (None, {"LLM_SLOTS": "3"}, (3, "env")),
    (None, {"LLM_SLOTS": "0"}, (1, "default")),
    (None, {"LLM_SLOTS": "junk"}, (1, "default")),
    (None, {}, (1, "default")),
])
def test_resolve_slots(props, env, expected):
    assert llm_lanes.resolve_slots(props, env) == expected


@pytest.mark.parametrize("raw, expected", [("", 1), ("0", 0), ("2", 2), ("-1", 1), ("x", 1)])
def test_chat_reserved_setting(raw, expected):
    assert llm_lanes.chat_reserved({"LLM_CHAT_RESERVED": raw}) == expected


def test_registry_lanes_only_configured_hosts():
    llm_lanes.reset()
    try:
        assert llm_lanes.lanes_for("http://llamacpp:8080") is None
        lanes = llm_lanes.configure_host("http://llamacpp:8080", 2, 1)
        assert llm_lanes.lanes_for("http://llamacpp:8080") is lanes
        assert llm_lanes.configure_host("http://llamacpp:8080", 4, 1) is lanes
        assert lanes.slots == 4
        assert set(llm_lanes.all_lanes()) == {"http://llamacpp:8080"}
    finally:
        llm_lanes.reset()

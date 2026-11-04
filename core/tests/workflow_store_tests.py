"""The workflow store (core/classes/workflow_store.py), against an in-memory
Redis with the few commands it uses."""
import asyncio

import pytest

from classes import workflow_policy as wp
from classes.workflow_store import LIVE, WorkflowStore


class FakeRedis:
    def __init__(self):
        self.data = {}

    async def get(self, key):
        return self.data.get(key)

    async def set(self, key, value, ex=None):
        self.data[key] = value

    async def sadd(self, key, member):
        self.data.setdefault(key, set()).add(member)

    async def srem(self, key, member):
        self.data.get(key, set()).discard(member)

    async def smembers(self, key):
        return set(self.data.get(key, set()))

    async def zadd(self, key, mapping, xx=False):
        zset = self.data.setdefault(key, {})
        for member, score in mapping.items():
            if xx and member not in zset:
                continue
            zset[member] = score

    async def zrem(self, key, member):
        self.data.get(key, {}).pop(member, None)

    async def zcard(self, key):
        return len(self.data.get(key, {}))

    async def zrange(self, key, start, end):
        zset = self.data.get(key, {})
        return sorted(zset, key=zset.get)

    async def zremrangebyscore(self, key, low, high):
        zset = self.data.get(key, {})
        for member in [m for m, s in zset.items() if s <= high]:
            zset.pop(member)


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def store(clock):
    return WorkflowStore(FakeRedis(), clock=clock)


def _record(workflow_id="a1", state=wp.PLANNING, instance="dcb-dev"):
    return {"id": workflow_id, "guild_id": 7, "state": state, "instance": instance,
            "thread_id": 99, "thread_url": "https://discord.com/channels/7/99"}


def run(coro):
    return asyncio.run(coro)


def test_a_record_round_trips(store):
    run(store.save(_record()))
    record = run(store.get(7, "a1"))
    assert record["thread_url"].endswith("/99") and record["updated_at"] == 1000.0
    assert run(store.update(7, "a1", state=wp.RUNNING))["state"] == wp.RUNNING
    assert run(store.get(7, "missing")) is None


def test_a_server_gets_only_its_limit_of_running_workflows(store):
    assert run(store.claim(7, "a1", 1))
    assert not run(store.claim(7, "a2", 1))
    assert run(store.claim(8, "b1", 1))  # another server
    assert run(store.running(7)) == ["a1"]
    run(store.release(7, "a1"))
    assert run(store.claim(7, "a2", 1))


def test_a_claim_nobody_renews_expires(store, clock):
    run(store.claim(7, "a1", 1))
    clock.t += 300
    run(store.renew(7, "a1"))
    clock.t += 500  # 800 s after the claim, 500 after the renewal
    assert not run(store.claim(7, "a2", 1))
    clock.t += 200
    assert run(store.claim(7, "a2", 1))


def test_renewing_a_released_claim_does_not_bring_it_back(store):
    run(store.claim(7, "a1", 1))
    run(store.release(7, "a1"))
    run(store.renew(7, "a1"))
    assert run(store.running(7)) == []


def test_ended_workflows_leave_the_live_set(store):
    run(store.save(_record()))
    assert run(store.redis.smembers(LIVE)) == {"7:a1"}
    run(store.save(_record(state=wp.DONE)))
    assert run(store.redis.smembers(LIVE)) == set()


def test_a_restart_interrupts_this_bots_live_workflows_and_frees_their_slots(store):
    run(store.save(_record("a1", wp.RUNNING)))
    run(store.claim(7, "a1", 1))
    run(store.save(_record("a2", wp.AWAITING_APPROVAL, instance="other-bot")))
    interrupted = run(store.interrupt_all("dcb-dev"))
    assert [r["id"] for r in interrupted] == ["a1"]
    assert run(store.get(7, "a1"))["state"] == wp.INTERRUPTED
    assert run(store.get(7, "a2"))["state"] == wp.AWAITING_APPROVAL  # not ours
    assert run(store.running(7)) == []
    assert run(store.redis.smembers(LIVE)) == {"7:a2"}


def test_a_workflow_this_process_already_runs_is_not_interrupted(store):
    run(store.save(_record("a1", wp.RUNNING)))
    assert run(store.interrupt_all("dcb-dev", skip={"a1"})) == []
    assert run(store.get(7, "a1"))["state"] == wp.RUNNING


def test_a_workflow_started_since_this_process_began_is_not_interrupted(store):
    # Its record is saved before it is registered, so `skip` can miss it.
    run(store.save(dict(_record("a1", wp.RUNNING), created_at=100.0)))
    run(store.save(dict(_record("a2", wp.RUNNING), created_at=50.0)))
    assert [r["id"] for r in run(store.interrupt_all("dcb-dev", before=90.0))] == ["a2"]
    assert run(store.get(7, "a1"))["state"] == wp.RUNNING

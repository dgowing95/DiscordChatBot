"""Tests for classes/attachment_cache.py (pure: no discord, aiohttp or Pillow).

To run this pytest file from the command line, use:
    pytest core/tests/attachment_cache_tests.py
"""
import asyncio

import pytest

from classes import attachment_cache
from classes.attachment_cache import COALESCED, HIT, MISS, AttachmentCache


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _loader(value, calls=None):
    async def load():
        if calls is not None:
            calls.append(value)
        return value
    return load


KEY = (1, 10, 100)


@pytest.mark.asyncio
async def test_miss_then_hit():
    cache = AttachmentCache()
    calls = []
    assert await cache.get_or_load(KEY, _loader("data:a", calls)) == ("data:a", MISS)
    assert await cache.get_or_load(KEY, _loader("data:b", calls)) == ("data:a", HIT)
    assert calls == ["data:a"]
    assert cache.bytes == len("data:a")


@pytest.mark.asyncio
async def test_entries_expire_after_the_ttl():
    clock = Clock()
    cache = AttachmentCache(ttl=60, clock=clock)
    await cache.get_or_load(KEY, _loader("data:a"))
    clock.now += 59
    assert cache.get(KEY) == "data:a"
    clock.now += 1
    assert cache.get(KEY) is None
    assert cache.bytes == 0


@pytest.mark.asyncio
async def test_least_recently_used_is_evicted_by_bytes():
    cache = AttachmentCache(max_bytes=10)
    await cache.get_or_load((1, 1, 1), _loader("aaaa"))
    await cache.get_or_load((1, 2, 2), _loader("bbbb"))
    cache.get((1, 1, 1))  # used more recently than (1, 2, 2)
    await cache.get_or_load((1, 3, 3), _loader("cccc"))
    assert cache.get((1, 2, 2)) is None
    assert cache.get((1, 1, 1)) == "aaaa"
    assert cache.get((1, 3, 3)) == "cccc"
    assert cache.bytes == 8


@pytest.mark.asyncio
async def test_a_value_over_the_cap_is_returned_but_not_kept():
    cache = AttachmentCache(max_bytes=4)
    assert await cache.get_or_load(KEY, _loader("too long")) == ("too long", MISS)
    assert len(cache) == 0
    assert cache.bytes == 0


@pytest.mark.asyncio
async def test_a_failure_is_not_cached_and_is_retried():
    cache = AttachmentCache()
    assert await cache.get_or_load(KEY, _loader(None)) == (None, MISS)
    assert await cache.get_or_load(KEY, _loader("data:a")) == ("data:a", MISS)


@pytest.mark.asyncio
async def test_a_raising_loader_is_not_cached_and_is_retried():
    cache = AttachmentCache()

    async def boom():
        raise RuntimeError("down")
    with pytest.raises(RuntimeError):
        await cache.get_or_load(KEY, boom)
    assert await cache.get_or_load(KEY, _loader("data:a")) == ("data:a", MISS)


@pytest.mark.asyncio
async def test_concurrent_gets_share_one_load():
    cache = AttachmentCache()
    gate = asyncio.Event()
    calls = []

    async def load():
        calls.append(1)
        await gate.wait()
        return "data:a"
    tasks = [asyncio.ensure_future(cache.get_or_load(KEY, load)) for _ in range(3)]
    await asyncio.sleep(0)
    gate.set()
    results = await asyncio.gather(*tasks)
    assert calls == [1]
    assert sorted(r for _, r in results) == [COALESCED, COALESCED, MISS]
    assert all(v == "data:a" for v, _ in results)


@pytest.mark.asyncio
async def test_a_cancelled_caller_does_not_cancel_the_shared_load():
    cache = AttachmentCache()
    gate = asyncio.Event()

    async def load():
        await gate.wait()
        return "data:a"
    first = asyncio.ensure_future(cache.get_or_load(KEY, load))
    await asyncio.sleep(0)
    second = asyncio.ensure_future(cache.get_or_load(KEY, load))
    await asyncio.sleep(0)
    first.cancel()
    await asyncio.sleep(0)
    gate.set()
    assert await second == ("data:a", COALESCED)
    assert cache.get(KEY) == "data:a"


@pytest.mark.asyncio
async def test_invalidate_message_drops_only_that_messages_entries():
    cache = AttachmentCache()
    await cache.get_or_load((1, 10, 100), _loader("a"))
    await cache.get_or_load((1, 10, 101), _loader("b"))
    await cache.get_or_load((1, 11, 102), _loader("c"))
    cache.invalidate_message(1, 10)
    assert cache.get((1, 10, 100)) is None
    assert cache.get((1, 10, 101)) is None
    assert cache.get((1, 11, 102)) == "c"
    assert cache.bytes == 1


@pytest.mark.asyncio
async def test_a_load_that_outlives_an_invalidation_is_not_stored():
    cache = AttachmentCache()
    gate = asyncio.Event()

    async def load():
        await gate.wait()
        return "data:a"
    task = asyncio.ensure_future(cache.get_or_load(KEY, load))
    await asyncio.sleep(0)
    cache.invalidate_message(1, 10, deleted=True)
    gate.set()
    # The caller still gets its value; the cache just does not keep it.
    assert await task == ("data:a", MISS)
    assert len(cache) == 0


def test_deleted_since_reports_only_deletions_after_the_generation():
    cache = AttachmentCache()
    cache.invalidate_message(1, 5, deleted=True)
    gen = cache.generation()
    cache.invalidate_message(1, 6, deleted=True)
    cache.invalidate_message(1, 7, deleted=False)  # an edit, not a deletion
    cache.invalidate_message(2, 8, deleted=True)   # another channel
    assert cache.deleted_since(gen, 1, [5, 6, 7, 8]) == {6}


def test_old_invalidation_records_are_pruned():
    clock = Clock()
    cache = AttachmentCache(ttl=60, clock=clock)
    cache.invalidate_message(1, 5, deleted=True)
    clock.now += 61
    cache.invalidate_message(1, 6, deleted=True)
    assert cache.deleted_since(0, 1, [5, 6]) == {6}


def test_the_process_wide_cache_is_shared_until_reset():
    attachment_cache.reset()
    first = attachment_cache.cache()
    assert attachment_cache.cache() is first
    attachment_cache.reset()
    assert attachment_cache.cache() is not first
    attachment_cache.reset()

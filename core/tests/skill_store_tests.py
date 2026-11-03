"""The skill store, against an in-memory Redis that supports WATCH/MULTI."""
import asyncio
import io
import json
import zipfile

import pytest
from redis.exceptions import WatchError

from classes import skill_store as ss
from classes.skill_store import SkillError, SkillStore


class FakePipe:
    def __init__(self, redis):
        self.r = redis
        self.watched = {}
        self.queued = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def watch(self, *keys):
        for key in keys:
            self.watched[key] = self.r.versions.get(key, 0)

    async def unwatch(self):
        self.watched = {}

    async def get(self, key):
        await asyncio.sleep(0)
        return self.r.data.get(key)

    async def scard(self, key):
        return len(self.r.data.get(key, set()))

    def multi(self):
        self.queued = []

    def set(self, key, value):
        self.queued.append(("set", key, value))

    def sadd(self, key, value):
        self.queued.append(("sadd", key, value))

    def srem(self, key, value):
        self.queued.append(("srem", key, value))

    async def execute(self):
        await asyncio.sleep(0)
        if any(self.r.versions.get(k, 0) != v for k, v in self.watched.items()):
            raise WatchError()
        for op, key, value in self.queued:
            self.r.apply(op, key, value)


class FakeRedis:
    def __init__(self):
        self.data = {}
        self.versions = {}

    def apply(self, op, key, value):
        if op == "set":
            self.data[key] = value
        elif op == "sadd":
            self.data.setdefault(key, set()).add(value)
        elif op == "srem":
            self.data.setdefault(key, set()).discard(value)
        self.versions[key] = self.versions.get(key, 0) + 1

    def pipeline(self, transaction=True):
        return FakePipe(self)

    async def get(self, key):
        return self.data.get(key)

    async def mget(self, keys):
        self.mgets = getattr(self, "mgets", 0) + 1
        return [self.data.get(k) for k in keys]

    async def smembers(self, key):
        return set(self.data.get(key, set()))

    async def scard(self, key):
        return len(self.data.get(key, set()))


@pytest.fixture
def store():
    clock = {"t": 1000.0}

    def tick():
        clock["t"] += 1
        return clock["t"]
    return SkillStore(FakeRedis(), clock=tick)


def run(coro):
    return asyncio.run(coro)


def save(store, name="weekly-chart", body="1. do it", description="Charts things", **kw):
    kw.setdefault("author_id", 7)
    return store.save(1, name, description, body, **kw)


def test_created_updated_unchanged(store):
    async def go():
        assert await save(store) == ss.CREATED
        assert await save(store) == ss.UNCHANGED
        assert await save(store, body="1. do it better") == ss.UPDATED
        skill = await store.get(1, "weekly-chart")
        assert skill["version"] == 2 and skill["versions"] == 2 and skill["body"] == "1. do it better"
    run(go())


def test_name_is_cleaned_and_invalid_refused(store):
    async def go():
        await save(store, name="Weekly Chart")
        assert await store.get(1, "weekly-chart")
        with pytest.raises(SkillError, match="body"):
            await save(store, name="other-one", body="")
        assert await store.get(1, "other-one") is None
    run(go())


def test_list_excludes_deleted_and_quota_counts_live_only(store, monkeypatch):
    monkeypatch.setenv("SKILLS_MAX_PER_GUILD", "2")

    async def go():
        await save(store, name="aaa")
        await save(store, name="bbb")
        with pytest.raises(ss.QuotaError):
            await save(store, name="ccc")
        await store.delete(1, "aaa", actor=9)
        assert [s["name"] for s in await store.list(1)] == ["bbb"]
        assert [s["name"] for s in await store.list(1, include_deleted=True)] == ["aaa", "bbb"]
        await save(store, name="ccc")
        with pytest.raises(ss.QuotaError):
            await store.restore(1, "aaa", actor=9)
        # updating an existing live skill is never blocked by the quota
        assert await save(store, name="bbb", body="new") == ss.UPDATED
    run(go())


def test_delete_restore_keeps_history(store):
    async def go():
        await save(store)
        await save(store, body="two")
        await store.delete(1, "weekly-chart", actor=9)
        assert await store.get(1, "weekly-chart") is None
        assert (await store.get_any(1, "weekly-chart"))["deleted"]
        assert len(await store.history(1, "weekly-chart")) == 2
        await store.restore(1, "weekly-chart", actor=9)
        assert (await store.get(1, "weekly-chart"))["body"] == "two"
        with pytest.raises(SkillError):
            await store.restore(1, "weekly-chart", actor=9)
        with pytest.raises(SkillError):
            await store.delete(1, "nope", actor=9)
    run(go())


def test_saving_a_deleted_skill_brings_it_back(store):
    async def go():
        await save(store)
        await store.delete(1, "weekly-chart", actor=9)
        await save(store)
        assert await store.get(1, "weekly-chart")
    run(go())


def test_revert_appends_a_copy(store):
    async def go():
        await save(store, body="first")
        await save(store, body="second")
        assert await store.revert(1, "weekly-chart", 1, actor=9) == 3
        skill = await store.get(1, "weekly-chart")
        assert skill["body"] == "first" and skill["version"] == 3 and skill["source"] == "revert"
        history = await store.history(1, "weekly-chart")
        assert [h["v"] for h in history] == [3, 2, 1] and history[0]["current"]
        assert history[0]["note"] == "back to v1"
        with pytest.raises(SkillError, match="no version"):
            await store.revert(1, "weekly-chart", 9, actor=9)
        with pytest.raises(SkillError, match="already"):
            await store.revert(1, "weekly-chart", 3, actor=9)
    run(go())


def test_get_old_version(store):
    async def go():
        await save(store, body="first")
        await save(store, body="second")
        assert (await store.get(1, "weekly-chart", version=1))["body"] == "first"
        assert await store.get(1, "weekly-chart", version=5) is None
    run(go())


def test_version_cap_drops_oldest_non_current(store, monkeypatch):
    monkeypatch.setenv("SKILLS_MAX_VERSIONS", "3")

    async def go():
        for i in range(5):
            await save(store, body=f"body {i}")
        history = await store.history(1, "weekly-chart")
        assert [h["v"] for h in history] == [5, 4, 3]
        assert (await store.get(1, "weekly-chart"))["body"] == "body 4"
    run(go())


def test_use_and_success_counters(store):
    async def go():
        await save(store)
        await store.record_use(1, "weekly-chart", channel_id=55)
        await store.record_use(1, "weekly-chart")
        await store.record_success(1, "weekly-chart")
        skill = await store.get(1, "weekly-chart")
        assert (skill["uses"], skill["successes"], skill["last_channel_id"]) == (2, 1, 55)
        await store.record_use(1, "missing")  # quiet
    run(go())


def test_concurrent_saves_and_uses_lose_nothing(store):
    async def go():
        await save(store)
        await asyncio.gather(*[store.record_use(1, "weekly-chart") for _ in range(6)],
                             *[save(store, body=f"b{i}") for i in range(3)])
        skill = await store.get(1, "weekly-chart")
        assert skill["uses"] == 6 and skill["versions"] == 4
    run(go())


def test_concurrent_creates_respect_quota(store, monkeypatch):
    monkeypatch.setenv("SKILLS_MAX_PER_GUILD", "1")

    async def go():
        out = await asyncio.gather(save(store, name="aaa"), save(store, name="bbb"),
                                   return_exceptions=True)
        assert sorted(type(o).__name__ for o in out) == ["QuotaError", "str"]
    run(go())


def test_files_round_trip_and_guilds_are_separate(store):
    async def go():
        await save(store, files={"scripts/run.py": "print(1)"})
        assert (await store.get(1, "weekly-chart"))["files"] == {"scripts/run.py": "print(1)"}
        assert await store.get(2, "weekly-chart") is None
        assert await store.list(2) == []
    run(go())


def test_export_zip(store):
    async def go():
        await save(store, files={"scripts/run.py": "print(1)"})
        await save(store, name="gone-one")
        await store.delete(1, "gone-one", actor=1)
        data = await store.export(1)
        archive = zipfile.ZipFile(io.BytesIO(data))
        names = set(archive.namelist())
        assert names == {"weekly-chart/SKILL.md", "weekly-chart/scripts/run.py", "history.json"}
        assert "name: weekly-chart" in archive.read("weekly-chart/SKILL.md").decode()
        history = json.loads(archive.read("history.json"))
        assert history["gone-one"]["deleted"] and len(history["weekly-chart"]["versions"]) == 1
    run(go())

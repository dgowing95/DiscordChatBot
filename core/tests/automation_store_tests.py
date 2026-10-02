"""Run with AUTOMATION_TEST_REDIS_URL=redis://localhost:16379/15 against Redis."""
import asyncio
import os
import pytest
import redis.asyncio as redis
from pytest_asyncio import fixture

from classes.automation_store import AutomationStore, AutomationError


@fixture
async def store():
    url = os.getenv("AUTOMATION_TEST_REDIS_URL")
    if not url:
        pytest.skip("Set AUTOMATION_TEST_REDIS_URL for real Redis tests")
    client = redis.from_url(url, decode_responses=True)
    await client.ping()
    await client.flushdb()
    yield AutomationStore(client)
    await client.flushdb()
    await client.aclose()


@pytest.mark.asyncio
async def test_concurrent_quota_and_terminal_release(store, monkeypatch):
    monkeypatch.setenv("SCHEDULE_MAX_PER_GUILD", "1")
    async def create():
        return await store.create(101, 202, 303, "schedule", "say hello",
                                  timing={"type": "interval", "every": 4, "unit": "hours"})
    outcomes = await asyncio.gather(create(), create(), return_exceptions=True)
    rows = [r for r in outcomes if isinstance(r, dict)]
    assert len(rows) == 1
    assert any(isinstance(r, AutomationError) for r in outcomes)
    row = rows[0]
    with pytest.raises(AutomationError, match="Stale revision"):
        await store.update(101, row["id"], 404, 0, action="changed")
    await store.update(101, row["id"], 404, row["revision"], status="deleted")
    assert isinstance(await create(), dict)


@pytest.mark.asyncio
async def test_rule_claim_is_single_winner_and_cooldown(store, monkeypatch):
    monkeypatch.setenv("RULE_COOLDOWN_SECONDS", "60")
    row = await store.create(102, 203, 304, "rule", "answer", pattern="wordle")
    claims = await asyncio.gather(*(store.claim_rule(row, 123) for _ in range(8)))
    assert sum(c is not None for c in claims) == 1
    await store.release(next(c for c in claims if c))
    assert await store.claim_rule(row, 124) is None


@pytest.mark.asyncio
async def test_schedule_claim_single_winner(store):
    row = await store.create(103, 204, 305, "schedule", "answer",
                             timing={"type": "interval", "every": 4, "unit": "hours"})
    claims = await asyncio.gather(*(store.claim(row, str(row["next_run"])) for _ in range(8)))
    assert sum(c is not None for c in claims) == 1


@pytest.mark.asyncio
async def test_started_occurrence_is_interrupted_after_claim_expires(store, monkeypatch):
    from classes import automation_runner
    from classes.automation_store import PREFIX
    row = await store.create(104, 205, 306, "schedule", "answer",
                             timing={"type": "interval", "every": 4, "unit": "hours"})
    member = f"104:{row['id']}"
    await store.redis.zadd(f"{PREFIX}:due", {member: 0})
    await store.redis.set(f"{PREFIX}:started:104:{row['id']}:{row['next_run']}", "1", ex=604800)
    monkeypatch.setattr(automation_runner, "AutomationStore", lambda: store)
    await automation_runner.poll_schedules(asyncio.Queue(maxsize=2))
    current = await store.get(104, row["id"])
    assert current["last_result"]["status"] == "interrupted"
    assert current["next_run"] > row["next_run"]


@pytest.mark.asyncio
async def test_terminal_one_off_releases_quota_and_prunes_index(store, monkeypatch):
    from datetime import datetime, timedelta, timezone
    from classes.automation_store import PREFIX
    monkeypatch.setenv("SCHEDULE_MAX_PER_GUILD", "1")
    at = (datetime.now(timezone.utc) + timedelta(minutes=2)).isoformat()
    row = await store.create(105, 206, 307, "schedule", "answer", timing={"type": "once", "at": at})
    await store.finish(row, "completed")
    assert (await store.get(105, row["id"]))["status"] == "completed"
    assert len(await store.list(105, "schedule")) == 1
    await store.create(105, 206, 307, "schedule", "another", timing={"type": "once", "at": at})
    await store.redis.zadd(f"{PREFIX}:terminal", {f"105:{row['id']}": 0})
    await store.purge_terminal()
    assert all(r["id"] != row["id"] for r in await store.list(105, "schedule"))


@pytest.mark.asyncio
async def test_interval_keeps_its_grid_through_runs_and_edits(store):
    row = await store.create(106, 207, 308, "schedule", "answer",
                             timing={"type": "interval", "every": 1, "unit": "days"})
    first = row["next_run"]
    edited = await store.update(106, row["id"], 309, row["revision"], action="changed")
    assert edited["next_run"] == first
    await store.finish(edited, "completed")
    # Finishing just now asks for the first slot 4h+ away: the next grid
    # point, which is the original first run (it is still a day off).
    assert (await store.get(106, row["id"]))["next_run"] == first


@pytest.mark.asyncio
async def test_deleting_finished_one_off_and_rule_drops_them(store, monkeypatch):
    from datetime import datetime, timedelta, timezone
    at = (datetime.now(timezone.utc) + timedelta(minutes=2)).isoformat()
    row = await store.create(107, 208, 309, "schedule", "answer", timing={"type": "once", "at": at})
    await store.finish(row, "completed")
    done = await store.get(107, row["id"])
    with pytest.raises(AutomationError, match="finished"):
        await store.update(107, row["id"], 309, done["revision"], action="again")
    await store.update(107, row["id"], 309, done["revision"], status="deleted")
    assert await store.get(107, row["id"]) is None
    assert await store.list(107, "schedule") == []
    rule = await store.create(107, 208, 309, "rule", "answer", pattern="wordle")
    await store.update(107, rule["id"], 309, rule["revision"], status="deleted")
    assert await store.list(107, "rule") == []


@pytest.mark.asyncio
async def test_timezone_is_stored_in_canonical_form(store):
    row = await store.create(108, 209, 310, "schedule", "answer",
                             timing={"type": "daily", "time": "09:00"}, timezone=" europe/london ")
    assert row["timezone"] == "Europe/London"
    edited = await store.update(108, row["id"], 310, row["revision"], timezone="new york")
    assert edited["timezone"] == "America/New_York"
    with pytest.raises(ValueError, match="Did you mean"):
        await store.update(108, row["id"], 310, edited["revision"], timezone="Europe/Londn")

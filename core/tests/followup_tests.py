"""Follow-ups: one-off schedules with their own quota, spaced at least the
minimum interval apart per person in a server (automation_policy / automation_store /
automation_tools.schedule_followup / automation_runner.followup_prompt).

The real-Redis store tests (automation_store_tests.py) only run with
AUTOMATION_TEST_REDIS_URL set, which CI does not set, so the follow-up
rules are also checked here against a small in-memory Redis.

To run this pytest file from the command line, use:
    pytest core/tests/followup_tests.py
"""
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from classes.automation_policy import describe, followup_conflict, quota_message
from classes.automation_store import AutomationError, AutomationStore, QuotaError

UTC = timezone.utc
HOUR = 3600


class FakeRedis:
    """The few commands AutomationStore.create/update/finish use. A
    pipeline runs commands at once until multi(), then queues them for
    execute(), like redis-py's."""

    def __init__(self):
        self.values, self.zsets = {}, {}

    async def get(self, key):
        return self.values.get(key)

    async def set(self, key, value, ex=None, nx=False):
        if nx and key in self.values:
            return False
        self.values[key] = str(value)
        return True

    async def zrange(self, key, start, end):
        return [m for m, _ in sorted(self.zsets.get(key, {}).items(), key=lambda kv: kv[1])]

    async def zadd(self, key, mapping):
        self.zsets.setdefault(key, {}).update(mapping)

    async def zrem(self, key, member):
        self.zsets.get(key, {}).pop(member, None)

    def pipeline(self, transaction=True):
        return FakePipe(self)


class FakePipe:
    def __init__(self, redis):
        self.redis, self.queued, self.queuing = redis, [], False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def watch(self, *keys):
        pass

    def multi(self):
        self.queuing = True

    def __getattr__(self, name):
        command = getattr(self.redis, name)
        if not self.queuing:
            return command

        def queue(*args, **kwargs):
            self.queued.append((command, args, kwargs))
            return self
        return queue

    async def execute(self):
        for command, args, kwargs in self.queued:
            await command(*args, **kwargs)


@pytest.fixture
def store(monkeypatch):
    monkeypatch.setenv("AUTOMATIONS_ENABLED", "1")
    monkeypatch.setenv("SCHEDULE_MIN_INTERVAL_HOURS", "4")
    monkeypatch.setenv("SCHEDULE_MAX_PER_GUILD", "1")
    monkeypatch.setenv("FOLLOWUP_MAX_PER_GUILD", "3")
    monkeypatch.setenv("AUTOMATIONS_TIMEZONE", "Europe/London")
    return AutomationStore(FakeRedis())


def _in(hours):
    return (datetime.now(UTC) + timedelta(hours=hours)).isoformat()


async def _followup(store, hours, guild=1, action="check the poll", person=20):
    return await store.create(guild, 10, person, "schedule", action, timing={"type": "once", "at": _in(hours)},
                              followup=True, context="find a time for game night")


def test_conflict_is_anything_closer_than_the_minimum():
    assert followup_conflict(10 * HOUR, [6.5 * HOUR, 20 * HOUR], 4) == 6.5 * HOUR
    assert followup_conflict(10 * HOUR, [6 * HOUR, 14 * HOUR], 4) is None
    assert followup_conflict(10 * HOUR, [], 4) is None


def test_followup_wording():
    assert "3 of 3 allowed follow-ups" in quota_message("followup", 3, 3)
    assert quota_message("followup", 0, 0) == "Follow-ups are turned off on this server."
    row = {"kind": "schedule", "followup": True, "status": "completed",
           "timing": {"type": "once", "at": "2026-10-24T19:00:00"}, "timezone": "Europe/London"}
    assert describe(row)["timing_summary"].startswith("follow-up once at")


@pytest.mark.asyncio
async def test_a_followup_can_be_soon_and_stores_its_context(store):
    row = await _followup(store, 0.5)
    assert row["followup"] is True and row["context"] == "find a time for game night"
    assert row["next_run"] - datetime.now(UTC).timestamp() < HOUR


@pytest.mark.asyncio
async def test_each_persons_followups_must_be_spaced_apart(store):
    from zoneinfo import ZoneInfo
    first = await _followup(store, 1)
    with pytest.raises(AutomationError, match="at least 4 hours apart") as refused:
        await _followup(store, 3)
    earliest = datetime.fromtimestamp(first["next_run"] + 4 * HOUR, ZoneInfo("Europe/London"))
    assert f"no earlier than {earliest:%Y-%m-%d %H:%M}" in str(refused.value)
    assert "no later than" not in str(refused.value)  # 1h - 4h is already past
    with pytest.raises(AutomationError, match="at least 4 hours apart"):
        await _followup(store, 0.2)
    await _followup(store, 5.1)
    # Someone else in the same server, or another server, is not blocked.
    await _followup(store, 1, person=21)
    await _followup(store, 1, guild=2)


@pytest.mark.asyncio
async def test_a_refusal_offers_the_time_before_a_later_followup_too(store):
    from zoneinfo import ZoneInfo
    later = await _followup(store, 20)
    with pytest.raises(AutomationError) as refused:
        await _followup(store, 18)
    zone = ZoneInfo("Europe/London")
    latest = datetime.fromtimestamp(later["next_run"] - 4 * HOUR, zone)
    earliest = datetime.fromtimestamp(later["next_run"] + 4 * HOUR, zone)
    assert (f"pick no later than {latest:%Y-%m-%d %H:%M} or no earlier than "
            f"{earliest:%Y-%m-%d %H:%M}") in str(refused.value)


@pytest.mark.asyncio
async def test_a_followup_that_ran_still_counts_after_it_is_deleted(store):
    row = await _followup(store, 1)
    await store.finish(row, "completed")
    current = await store.get(1, row["id"])
    assert current["status"] == "completed"
    # Deleting the finished one takes it out of the index, not the spacing.
    await store.update(1, row["id"], 20, current["revision"], status="deleted")
    with pytest.raises(AutomationError, match="at least 4 hours apart"):
        await _followup(store, 2)
    await _followup(store, 2, person=21)


@pytest.mark.asyncio
async def test_followups_have_their_own_quota(store, monkeypatch):
    monkeypatch.setenv("SCHEDULE_MIN_INTERVAL_HOURS", "1")
    await store.create(1, 10, 20, "schedule", "daily news", timing={"type": "daily", "time": "09:00"})
    # The ordinary schedule slot is full, but follow-ups still have room.
    for person in (20, 21, 22):
        await _followup(store, 1, person=person)
    with pytest.raises(QuotaError, match="3 of 3 allowed follow-ups"):
        await _followup(store, 1, person=23)
    with pytest.raises(QuotaError, match="1 of 1 allowed schedules"):
        await store.create(1, 10, 20, "schedule", "more news", timing={"type": "daily", "time": "10:00"})


@pytest.mark.asyncio
async def test_a_followup_runs_once_and_its_time_cannot_be_edited(store):
    with pytest.raises(AutomationError, match="runs once"):
        await store.create(1, 10, 20, "schedule", "x", timing={"type": "daily", "time": "09:00"},
                           followup=True)
    row = await _followup(store, 1)
    with pytest.raises(AutomationError, match="cannot be changed"):
        await store.update(1, row["id"], 20, row["revision"], timing={"type": "once", "at": _in(9)})
    updated = await store.update(1, row["id"], 20, row["revision"], action="check it twice")
    assert updated["action"] == "check it twice"


def _wrapper(message_content="when are we free?"):
    perms = SimpleNamespace(view_channel=True, send_messages=True)
    guild = SimpleNamespace(me=object(), get_member=lambda ident: object())
    channel = SimpleNamespace(id=10, guild=guild, permissions_for=lambda member: perms)
    message = SimpleNamespace(author=object(), content=message_content)
    return SimpleNamespace(context={"guild_id": 1, "channel": channel, "user_id": 20,
                                    "original_message": message, "automatic": False})


async def _call_tool(wrapper, **arguments):
    from agents.tool_context import ToolContext
    from classes.automation_tools import schedule_followup
    ctx = ToolContext(context=wrapper.context, tool_name="schedule_followup", tool_call_id="t1",
                      tool_arguments="{}")
    return await schedule_followup.on_invoke_tool(ctx, json.dumps(arguments))


@pytest.mark.asyncio
async def test_the_tool_takes_hours_or_a_time_and_keeps_the_request(store, monkeypatch):
    from classes import automation_tools
    monkeypatch.setattr(automation_tools, "AutomationStore", lambda: store)
    result = json.loads(await _call_tool(_wrapper(), action="check the poll", after_hours=2))
    assert result["followup"] is True and result["context"] == "when are we free?"
    assert "next_run_local" in result
    both = await _call_tool(_wrapper(), action="x", after_hours=2, at=_in(9))
    assert "exactly one of after_hours or at" in both
    near = await _call_tool(_wrapper(), action="x", at=_in(3))
    assert "at least 4 hours apart" in near


@pytest.mark.asyncio
async def test_the_tool_is_refused_in_an_automatic_run(store, monkeypatch):
    """A follow-up's own run cannot make another follow-up."""
    from classes import automation_tools
    monkeypatch.setattr(automation_tools, "AutomationStore", lambda: store)
    wrapper = _wrapper()
    wrapper.context["automatic"] = True
    assert "unavailable in this run" in await _call_tool(wrapper, action="x", after_hours=5)


def test_the_run_is_told_who_asked_and_what_they_said():
    from classes.automation_runner import followup_prompt
    row = {"creator_id": 20, "created_at": "2026-10-22T12:00:00+00:00", "timezone": "Europe/London",
           "context": "find a time for game night"}
    prompt = followup_prompt(row, "Execute this scheduled action now. Action: check the poll")
    assert "<@20>" in prompt and "find a time for game night" in prompt
    assert "2026-10-22T13:00+01:00" in prompt
    assert prompt.endswith("Action: check the poll")


@pytest.mark.asyncio
async def test_the_current_time_carries_its_offset_and_zone(monkeypatch):
    """Without the offset the model guessed +00:00 for London time, and a
    follow-up meant for 3 minutes ahead ran an hour late."""
    from classes.tool_functions import get_current_datetime
    monkeypatch.setenv("AUTOMATIONS_TIMEZONE", "Asia/Kolkata")
    text = await get_current_datetime()
    assert text.endswith("+05:30 (Asia/Kolkata)")
    assert datetime.fromisoformat(text.split(" ")[0]).utcoffset() == timedelta(hours=5, minutes=30)


@pytest.mark.asyncio
async def test_only_errors_written_for_the_model_reach_it(store, monkeypatch):
    """openai-agents 0.23 hides every tool exception by default; the
    automation tools pass their own ValueErrors on and keep the rest hidden."""
    from classes import automation_tools

    class Broken:
        async def create(self, *args, **kwargs):
            raise RuntimeError("redis://secret-host:6379 refused")
    monkeypatch.setattr(automation_tools, "AutomationStore", lambda: Broken())
    result = await _call_tool(_wrapper(), action="x", after_hours=5)
    assert "secret-host" not in result and "An error occurred" in result

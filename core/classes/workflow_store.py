"""Tier 3 workflows in Redis: one record each, and each server's running set.

  dcb:workflows:v1:{guild}:{id}   the record (JSON), kept RECORD_TTL_SECONDS
  dcb:workflows:v1:{guild}:active running ids (a sorted set scored by when the
                                  claim expires), at most WORKFLOW_MAX_PER_GUILD
  dcb:workflows:v1:live           every live workflow, "guild:id", so a restart
                                  can find them (interrupt_all)

A claim expires unless it is renewed (the workflow renews it while it runs),
so a process that dies without cleaning up frees the server's slot within
CLAIM_TTL_SECONDS. Claims are checked and taken under one asyncio lock: core
is one process (the chart runs one replica, and the workspaces it shares are
per process too), so nothing else takes a slot between the count and the add.

Resume after a restart is out of scope: at startup every live workflow of
this instance becomes INTERRUPTED (classes/workflows.py posts the note).
"""
import asyncio
import json
import os
import time

from classes import workflow_policy as wp
from classes.redis_client import text_client

PREFIX = "dcb:workflows:v1"
CLAIM_TTL_SECONDS = 600
RECORD_TTL_SECONDS = 30 * 86400


def _key(guild, workflow_id) -> str:
    return f"{PREFIX}:{int(guild)}:{workflow_id}"


def _active(guild) -> str:
    return f"{PREFIX}:{int(guild)}:active"


LIVE = f"{PREFIX}:live"


def instance(env=os.environ) -> str:
    """Which bot owns a workflow: two bots on one Redis never interrupt each
    other's (the same label the workspaces use)."""
    return (env.get("WORKSPACE_INSTANCE") or "dcb-dev").strip() or "dcb-dev"


class WorkflowStore:
    _claim_lock: asyncio.Lock | None = None

    def __init__(self, redis=None, clock=time.time):
        self.redis = redis or text_client()
        self.clock = clock

    @classmethod
    def _lock(cls) -> asyncio.Lock:
        if cls._claim_lock is None:
            cls._claim_lock = asyncio.Lock()
        return cls._claim_lock

    # -- records ----------------------------------------------------------------

    async def get(self, guild, workflow_id) -> dict | None:
        raw = await self.redis.get(_key(guild, workflow_id))
        if not raw:
            return None
        try:
            record = json.loads(raw)
        except ValueError:
            return None
        return record if isinstance(record, dict) else None

    async def save(self, record: dict) -> dict:
        record["updated_at"] = self.clock()
        guild, workflow_id = record["guild_id"], record["id"]
        await self.redis.set(_key(guild, workflow_id), json.dumps(record),
                             ex=RECORD_TTL_SECONDS)
        member = f"{int(guild)}:{workflow_id}"
        if record.get("state") in wp.LIVE_STATES:
            await self.redis.sadd(LIVE, member)
        else:
            await self.redis.srem(LIVE, member)
        return record

    async def update(self, guild, workflow_id, **changes) -> dict | None:
        record = await self.get(guild, workflow_id)
        if record is None:
            return None
        record.update(changes)
        return await self.save(record)

    # -- the server's slot ------------------------------------------------------

    async def claim(self, guild, workflow_id, limit: int) -> bool:
        """Takes one of the server's `limit` slots; False when all are in use."""
        now = self.clock()
        async with self._lock():
            await self.redis.zremrangebyscore(_active(guild), "-inf", now)
            if await self.redis.zcard(_active(guild)) >= limit:
                return False
            await self.redis.zadd(_active(guild), {workflow_id: now + CLAIM_TTL_SECONDS})
            return True

    async def renew(self, guild, workflow_id) -> None:
        await self.redis.zadd(_active(guild), {workflow_id: self.clock() + CLAIM_TTL_SECONDS},
                              xx=True)

    async def release(self, guild, workflow_id) -> None:
        await self.redis.zrem(_active(guild), workflow_id)

    async def running(self, guild) -> list[str]:
        """Ids holding one of the server's slots now."""
        await self.redis.zremrangebyscore(_active(guild), "-inf", self.clock())
        return list(await self.redis.zrange(_active(guild), 0, -1))

    # -- restarts -----------------------------------------------------------------

    async def interrupt_all(self, owner: str, skip=frozenset(),
                            before: float | None = None) -> list[dict]:
        """Marks every live workflow `owner` started as INTERRUPTED and frees
        its slot, except those this process started: the check runs while
        the bot already takes requests. Those are the ids in `skip` (read at
        each record, so pass the live registry) and, since a record is
        saved before it is registered, any created at or after `before`
        (this process's start). Returns the records, for the notes in
        their threads."""
        interrupted = []
        for member in list(await self.redis.smembers(LIVE)):
            guild, _, workflow_id = str(member).partition(":")
            if workflow_id in skip:
                continue
            record = await self.get(guild, workflow_id)
            if record is None:
                await self.redis.srem(LIVE, member)
                continue
            if record.get("instance") != owner or workflow_id in skip:
                continue
            if before is not None and float(record.get("created_at") or 0) >= before:
                continue
            if record.get("state") in wp.LIVE_STATES:
                record["state"] = wp.INTERRUPTED
                await self.save(record)
                interrupted.append(record)
            else:
                await self.redis.srem(LIVE, member)
            await self.release(guild, workflow_id)
        return interrupted

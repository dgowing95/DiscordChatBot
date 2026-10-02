"""Versioned Redis records and atomic admission for automations."""
import json
import secrets
import time
from datetime import datetime, timezone, timedelta

from redis.exceptions import WatchError
from classes.redis_client import text_client
from classes.automation_policy import anchor_timing, matches, resolve_timezone, next_run, settings, validate_schedule

PREFIX = "dcb:automations:v1"


class AutomationError(ValueError):
    pass


def _key(guild, ident):
    return f"{PREFIX}:record:{guild}:{ident}"


def _index(guild, kind):
    return f"{PREFIX}:index:{guild}:{kind}"


def _due():
    return f"{PREFIX}:due"


def _terminal():
    return f"{PREFIX}:terminal"


def _active(record):
    return record["status"] not in ("completed", "failed", "deleted")


class AutomationStore:
    def __init__(self, redis=None):
        self.redis = redis or text_client()

    async def get(self, guild, ident):
        raw = await self.redis.get(_key(guild, ident))
        row = json.loads(raw) if raw else None
        return row if row and row["status"] != "deleted" else None

    async def list(self, guild, kind):
        ids = await self.redis.zrange(_index(guild, kind), 0, -1)
        rows = [await self.get(guild, ident) for ident in ids]
        return [row for row in rows if row]

    async def create(self, guild, channel, actor, kind, action, **fields):
        cfg = settings()
        if not cfg["enabled"]:
            raise AutomationError("Automations are disabled")
        if kind not in ("schedule", "rule") or not action.strip():
            raise AutomationError("A schedule or rule needs an action")
        if len(action) > 1500:
            raise AutomationError("Action is too long (maximum 1500 characters)")
        now = datetime.now(timezone.utc)
        record = {"schema": 1, "id": secrets.token_hex(5), "guild_id": int(guild),
                  "channel_id": int(channel), "creator_id": int(actor), "editor_id": int(actor),
                  "kind": kind, "action": action.strip(), "status": "enabled", "revision": 1,
                  "created_at": now.isoformat(), "updated_at": now.isoformat(), "last_result": None}
        if kind == "schedule":
            record["timezone"] = resolve_timezone(fields.get("timezone") or cfg["timezone"])
            record["timing"] = anchor_timing(fields["timing"], now)
            record["next_run"] = validate_schedule(record["timing"], record["timezone"], cfg["min_hours"], now).timestamp()
        else:
            record["pattern"] = fields["pattern"].strip()
            if len(record["pattern"]) > 200:
                raise AutomationError("Pattern is too long (maximum 200 characters)")
            record["match_mode"] = fields.get("match_mode", "word")
            matches(record["pattern"], "", record["match_mode"])
        index = _index(guild, kind)
        for _ in range(8):
            async with self.redis.pipeline(transaction=True) as pipe:
                try:
                    await pipe.watch(index)
                    ids = await pipe.zrange(index, 0, -1)
                    rows = [await self.get(guild, ident) for ident in ids]
                    count = sum(1 for row in rows if row and _active(row))
                    limit = cfg["schedule_limit" if kind == "schedule" else "rule_limit"]
                    if count >= limit:
                        raise AutomationError(f"Server {kind} limit is {limit}")
                    pipe.multi()
                    pipe.set(_key(guild, record["id"]), json.dumps(record))
                    pipe.zadd(index, {record["id"]: now.timestamp()})
                    if kind == "schedule":
                        pipe.zadd(_due(), {f"{guild}:{record['id']}": record["next_run"]})
                    await pipe.execute()
                    return record
                except WatchError:
                    continue
        raise AutomationError("Concurrent update; try again")

    async def update(self, guild, ident, actor, revision, **changes):
        key = _key(guild, ident)
        for _ in range(8):
            async with self.redis.pipeline(transaction=True) as pipe:
                try:
                    await pipe.watch(key)
                    raw = await pipe.get(key)
                    if not raw:
                        raise AutomationError("Automation not found")
                    row = json.loads(raw)
                    if row["revision"] != revision:
                        raise AutomationError(f"Stale revision; current record: {row}")
                    if row["status"] == "deleted":
                        raise AutomationError("Automation not found")
                    # A finished one-off can still be deleted, just not changed.
                    if row["status"] in ("completed", "failed") and changes != {"status": "deleted"}:
                        raise AutomationError("This entry has finished; create a new one")
                    allowed = {"action", "status", "timing", "timezone"} if row["kind"] == "schedule" else {"action", "status", "pattern", "match_mode"}
                    if changes.keys() - allowed:
                        raise AutomationError("Unsupported edit")
                    now = datetime.now(timezone.utc)
                    if "timezone" in changes:
                        changes["timezone"] = resolve_timezone(changes["timezone"])
                    if "timing" in changes:
                        changes["timing"] = anchor_timing(changes["timing"], now)
                    row.update(changes)
                    if row["status"] not in ("enabled", "paused", "deleted", "suspended"):
                        raise AutomationError("Invalid status")
                    if not row["action"].strip():
                        raise AutomationError("Action cannot be empty")
                    if len(row["action"]) > 1500:
                        raise AutomationError("Action is too long (maximum 1500 characters)")
                    if row["kind"] == "rule" and len(row["pattern"]) > 200:
                        raise AutomationError("Pattern is too long (maximum 200 characters)")
                    if row["kind"] == "schedule" and row["status"] == "enabled":
                        row["next_run"] = validate_schedule(row["timing"], row["timezone"], settings()["min_hours"], now).timestamp()
                    if row["kind"] == "rule":
                        matches(row["pattern"], "", row["match_mode"])
                    row["revision"] += 1
                    row["editor_id"] = int(actor)
                    row["updated_at"] = now.isoformat()
                    deleted = row["status"] == "deleted"
                    pipe.multi()
                    # A deleted record leaves the index (and so every listing
                    # and quota count) at once; the record itself lingers a
                    # week so a stale management reply still resolves.
                    pipe.set(key, json.dumps(row), ex=604800 if deleted else None)
                    if deleted:
                        pipe.zrem(_index(guild, row["kind"]), ident)
                    if row["kind"] == "schedule":
                        member = f"{guild}:{ident}"
                        if row["status"] == "enabled":
                            pipe.zadd(_due(), {member: row["next_run"]})
                        else:
                            pipe.zrem(_due(), member)
                    await pipe.execute()
                    return row
                except WatchError:
                    continue
        raise AutomationError("Concurrent update; try again")

    async def due(self, now=None):
        return await self.redis.zrangebyscore(_due(), "-inf", now or time.time())

    async def claim(self, row, occurrence, ttl=900):
        """A claim is shared across workers/processes and expires after process loss."""
        key = f"{PREFIX}:claim:{row['kind']}:{row['guild_id']}:{row['id']}:{occurrence}"
        lock = f"{PREFIX}:busy:{row['guild_id']}:{row['id']}"
        # Lua makes admission and busy exclusion one Redis operation.
        script = """if redis.call('EXISTS', KEYS[1]) == 1 or redis.call('EXISTS', KEYS[2]) == 1 then return 0 end
redis.call('SET', KEYS[1], ARGV[1], 'EX', ARGV[2]); redis.call('SET', KEYS[2], ARGV[1], 'EX', ARGV[2]); return 1"""
        token = secrets.token_hex(12)
        ok = await self.redis.eval(script, 2, key, lock, token, ttl)
        return (key, lock, token) if ok else None

    async def release(self, claim):
        if claim:
            _, lock, token = claim
            await self.redis.eval("if redis.call('GET', KEYS[1]) == ARGV[1] then return redis.call('DEL', KEYS[1]) end return 0", 1, lock, token)

    async def cancel_claim(self, claim, rule=False):
        key, lock, token = claim
        keys = [key, lock]
        if rule:
            keys.append(lock.replace(":busy:", ":cooldown:"))
        script = """for i=1,#KEYS do if redis.call('GET', KEYS[i]) == ARGV[1] then redis.call('DEL', KEYS[i]) end end return 1"""
        await self.redis.eval(script, len(keys), *keys, token)

    async def renew(self, claim, ttl=900):
        key, lock, token = claim
        script = """if redis.call('GET', KEYS[1]) ~= ARGV[1] or redis.call('GET', KEYS[2]) ~= ARGV[1] then return 0 end
if redis.call('TTL', KEYS[1]) < tonumber(ARGV[2]) then redis.call('EXPIRE', KEYS[1], ARGV[2]) end
redis.call('EXPIRE', KEYS[2], ARGV[2]); return 1"""
        return bool(await self.redis.eval(script, 2, key, lock, token, ttl))

    async def claim_rule(self, row, message_id):
        cooldown = settings()["cooldown"]
        key = f"{PREFIX}:rule-message:{row['guild_id']}:{row['id']}:{message_id}"
        busy = f"{PREFIX}:busy:{row['guild_id']}:{row['id']}"
        cd = f"{PREFIX}:cooldown:{row['guild_id']}:{row['id']}"
        token = secrets.token_hex(12)
        script = """if redis.call('EXISTS', KEYS[1], KEYS[2], KEYS[3]) > 0 then return 0 end
redis.call('SET', KEYS[1], ARGV[1], 'EX', 86400)
redis.call('SET', KEYS[2], ARGV[1], 'EX', 900)
redis.call('SET', KEYS[3], ARGV[1], 'EX', ARGV[2])
return 1"""
        ok = await self.redis.eval(script, 3, key, busy, cd, token, max(1, cooldown))
        return (key, busy, token) if ok else None

    async def finish(self, row, outcome, detail=""):
        key = _key(row["guild_id"], row["id"])
        for _ in range(8):
            async with self.redis.pipeline(transaction=True) as pipe:
                try:
                    await pipe.watch(key)
                    raw = await pipe.get(key)
                    if not raw:
                        return
                    current = json.loads(raw)
                    now = datetime.now(timezone.utc)
                    current["last_result"] = {"status": outcome, "detail": detail[:500], "at": now.isoformat()}
                    advance = current["revision"] == row["revision"] and current["status"] == "enabled"
                    terminal = False
                    if advance and row["kind"] == "schedule":
                        if row["timing"]["type"] == "once":
                            current["status"] = "completed" if outcome == "completed" else "failed"
                            terminal = True
                        else:
                            minimum = timedelta(hours=settings()["min_hours"])
                            current["next_run"] = next_run(row["timing"], row["timezone"], now + minimum - timedelta(microseconds=1)).timestamp()
                    pipe.multi()
                    pipe.set(key, json.dumps(current), ex=604800 if terminal else None)
                    if row["kind"] == "schedule" and advance:
                        member = f"{row['guild_id']}:{row['id']}"
                        if terminal:
                            pipe.zrem(_due(), member)
                            pipe.zadd(_terminal(), {member: now.timestamp() + 604800})
                        else:
                            pipe.zadd(_due(), {member: current["next_run"]})
                    await pipe.execute()
                    return
                except WatchError:
                    continue
        raise AutomationError("Could not record outcome after concurrent edits")

    async def purge_terminal(self):
        members = await self.redis.zrangebyscore(_terminal(), "-inf", time.time())
        if not members:
            return
        async with self.redis.pipeline(transaction=True) as pipe:
            for member in members:
                guild, ident = member.split(":", 1)
                pipe.zrem(_index(guild, "schedule"), ident)
                pipe.zrem(_terminal(), member)
            await pipe.execute()

"""Versioned Redis records and atomic admission for automations."""
import json
import secrets
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from redis.exceptions import WatchError
from classes.redis_client import text_client
from classes.automation_policy import (
    anchor_timing, followup_conflict, following_run, matches, quota_message, resolve_timezone, settings,
    validate_schedule,
)

PREFIX = "dcb:automations:v1"


class AutomationError(ValueError):
    pass


class QuotaError(AutomationError):
    """The server has no room for another entry of this kind. Editing the
    request cannot fix it, so forms offer no retry for it."""


def _key(guild, ident):
    return f"{PREFIX}:record:{guild}:{ident}"


def _index(guild, kind):
    return f"{PREFIX}:index:{guild}:{kind}"


def _due():
    return f"{PREFIX}:due"


def _terminal():
    return f"{PREFIX}:terminal"


def _followup_last(guild, person):
    """When this person's last follow-up in the server ran: still counted for
    spacing after that follow-up is deleted from the index."""
    return f"{PREFIX}:followup-last:{guild}:{person}"


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

    async def _rows(self, guild, kind, ids=None):
        if ids is None:
            ids = await self.redis.zrange(_index(guild, kind), 0, -1)
        rows = [await self.get(guild, ident) for ident in ids]
        return [row for row in rows if row]

    @staticmethod
    def _active_count(rows, kind):
        """Active entries counted against `kind`'s limit. A follow-up is a
        schedule record but has its own limit ("followup"), so it neither
        uses up nor is blocked by the server's ordinary schedule slot."""
        followup = kind == "followup"
        return sum(1 for row in rows if _active(row) and bool(row.get("followup")) == followup)

    async def check_quota(self, guild, kind, rows=None):
        """Raise QuotaError when the server is full. Called up front so a
        form is never filled in for nothing; create() checks again inside
        its transaction, which is what actually enforces the limit."""
        limit = settings()[{"schedule": "schedule_limit", "followup": "followup_limit"}.get(kind, "rule_limit")]
        if rows is None:
            rows = await self._rows(guild, "rule" if kind == "rule" else "schedule")
        count = self._active_count(rows, kind)
        if count >= limit:
            raise QuotaError(quota_message(kind, count, limit))

    async def create(self, guild, channel, actor, kind, action, **fields):
        """`followup=True` (with kind "schedule" and a one-off timing) makes a
        follow-up: its own quota, at least the minimum interval from the
        same person's other follow-ups in the server, and `context` (the request it came from)
        kept for the run's prompt."""
        cfg = settings()
        if not cfg["enabled"]:
            raise AutomationError("Automations are disabled")
        followup = kind == "schedule" and bool(fields.get("followup"))
        quota_kind = "followup" if followup else kind
        if followup and fields["timing"].get("type") != "once":
            raise AutomationError("A follow-up runs once")
        if kind in ("schedule", "rule"):
            await self.check_quota(guild, quota_kind)
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
            if followup:
                record["followup"] = True
                record["context"] = str(fields.get("context") or "").strip()[:500]
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
                    rows = await self._rows(guild, kind, ids)
                    await self.check_quota(guild, quota_kind, rows)
                    if followup:
                        # Per person, so one member's follow-up never blocks another's;
                        # the server-wide quota still bounds the total.
                        others = [row["next_run"] for row in rows if row.get("followup") and row.get("next_run")
                                  and row.get("creator_id") == int(actor)]
                        last = await self.redis.get(_followup_last(guild, actor))
                        if last:
                            others.append(float(last))
                        clash = followup_conflict(record["next_run"], others, cfg["min_hours"])
                        if clash is not None:
                            # Local times, and the earliest one allowed: given only a UTC
                            # time, the model converted and added wrongly (seen live).
                            zone = ZoneInfo(record["timezone"])
                            def local(ts):
                                return datetime.fromtimestamp(ts, zone).strftime("%Y-%m-%d %H:%M")
                            gap = cfg["min_hours"] * 3600
                            before = (f"no later than {local(clash - gap)} or "
                                      if clash - gap > now.timestamp() else "")
                            raise AutomationError(
                                f"Your follow-ups in this server must be at least {cfg['min_hours']} hours apart. "
                                f"Your nearest other one is at {local(clash)} ({record['timezone']}), so pick "
                                f"{before}no earlier than {local(clash + gap)}. Tell the user these times as "
                                "written.")
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
                    if row.get("followup") and changes.keys() & {"timing", "timezone"}:
                        raise AutomationError("A follow-up's time cannot be changed; delete it and make a new one")
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

    async def finish(self, row, outcome, detail="", started_at=None):
        """Record an outcome and, for a schedule, pick its next run.
        `started_at` is when the run began; unknown (an interrupted run)
        counts as now, which errs towards a later next run."""
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
                            occurrence = datetime.fromtimestamp(row["next_run"], timezone.utc)
                            current["next_run"] = following_run(
                                row["timing"], row["timezone"], occurrence, started_at or now, now,
                                settings()["min_hours"]).timestamp()
                    pipe.multi()
                    pipe.set(key, json.dumps(current), ex=604800 if terminal else None)
                    if row["kind"] == "schedule" and advance:
                        member = f"{row['guild_id']}:{row['id']}"
                        if terminal:
                            pipe.zrem(_due(), member)
                            pipe.zadd(_terminal(), {member: now.timestamp() + 604800})
                            if row.get("followup"):
                                pipe.set(_followup_last(row["guild_id"], row["creator_id"]), row["next_run"], ex=604800)
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

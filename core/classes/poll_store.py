"""Which polls each channel has had, so the check_polls tool can still find
one after it has scrolled out of the prompt's history window.

Only message ids are kept, scored by the poll's end time: Discord holds the
votes, so every lookup reads them live and nothing here goes stale. One
sorted set per channel, `dcb:polls:{channel_id}`. A poll is dropped
KEEP_SECONDS after it ends, and the key expires once its last poll has.
"""
import time

from classes.redis_client import text_client

# How long after it ends a poll can still be looked up.
KEEP_SECONDS = 7 * 24 * 3600
# Polls remembered per channel; the ones ending first are dropped past it.
MAX_PER_CHANNEL = 20


def _key(channel_id) -> str:
    return f"dcb:polls:{channel_id}"


async def remember(channel_id, message_id, ends_at: float) -> None:
    redis = text_client()
    key = _key(channel_id)
    await redis.zadd(key, {str(message_id): ends_at})
    await redis.zremrangebyscore(key, "-inf", time.time() - KEEP_SECONDS)
    await redis.zremrangebyrank(key, 0, -(MAX_PER_CHANNEL + 1))
    last = await redis.zrange(key, -1, -1, withscores=True)
    if last:
        await redis.expireat(key, int(last[0][1] + KEEP_SECONDS))


async def recent(channel_id, limit: int) -> list:
    """Message ids of the channel's polls still worth looking up, newest
    posted first (message ids are time-ordered)."""
    redis = text_client()
    key = _key(channel_id)
    await redis.zremrangebyscore(key, "-inf", time.time() - KEEP_SECONDS)
    ids = await redis.zrange(key, 0, -1)
    return sorted((int(i) for i in ids), reverse=True)[:limit]


async def forget(channel_id, message_id) -> None:
    """Drops a poll whose message is gone."""
    await text_client().zrem(_key(channel_id), str(message_id))

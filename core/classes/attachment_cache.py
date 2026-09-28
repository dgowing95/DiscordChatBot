"""Encoded image attachments, kept so a prompt build does not fetch them again.

Pure module (stdlib only), following the history_policy.py / message_queue.py
pattern: MessageHandler does the downloading and encoding, this module only
decides what is kept, for how long, and who waits for whom, so it can be unit
tested without discord, aiohttp or Pillow.

Why it exists
-------------
Every prompt build used to download and encode every image in the window -
the trigger's and every history message's - inside the channel lock. With
anchored history the same images are in the prompt turn after turn, so nearly
all of that work was repeated, and a slow CDN held up every other handle in
the channel.

What is kept
------------
The final `data:<mime>;base64,...` string, exactly what goes in the prompt, so
a cached build sends the same bytes as an uncached one (llama.cpp only reuses
an identical prefix). Nothing else is retained, so `len(value)` is the real
size. Entries are keyed by (channel id, message id, attachment id), never by
URL: Discord's CDN URLs are signed and expire, and an expired URL for an
attachment already here is still a hit.

Failures are not kept. A loader that returns None (download failed, not an
image) or raises leaves nothing behind, so the next build tries again rather
than leaving the image out for the next 15 minutes.

Concurrent builds
-----------------
Builds that want the same attachment at the same time share one load: the
first starts it as its own task, the others wait on it through
asyncio.shield, so a caller that is cancelled never cancels the load the
others are waiting for.

Invalidation
------------
main.py calls invalidate_message() on Discord's delete and edit events. That
drops the message's entries and bumps a generation counter. A load that
started before the invalidation does not store its result (a late download
must not put a deleted image back), and deleted_since() lets a build that
selected its messages before a deletion leave the deleted ones out.
"""

import asyncio
import time
from collections import OrderedDict

MAX_BYTES = 64 * 1024 * 1024
TTL_SECONDS = 15 * 60

HIT = "hit"
MISS = "miss"
COALESCED = "coalesced"


class AttachmentCache:
    def __init__(self, max_bytes=MAX_BYTES, ttl=TTL_SECONDS, clock=time.monotonic):
        self.max_bytes = max_bytes
        self.ttl = ttl
        self._clock = clock
        self._entries = OrderedDict()  # key -> (value, expires_at), oldest use first
        self._bytes = 0
        self._inflight = {}  # key -> the task loading it
        self._generation = 0
        # (channel_id, message_id) -> (generation, deleted, recorded_at)
        self._invalidated = {}

    @property
    def bytes(self) -> int:
        return self._bytes

    def __len__(self):
        return len(self._entries)

    def generation(self) -> int:
        return self._generation

    def get(self, key):
        """The cached value, or None when absent or expired."""
        item = self._entries.get(key)
        if item is None:
            return None
        value, expires_at = item
        if self._clock() >= expires_at:
            self._remove(key)
            return None
        self._entries.move_to_end(key)
        return value

    async def get_or_load(self, key, loader):
        """(value or None, HIT / MISS / COALESCED). `loader` is a no-argument
        coroutine function returning the value, or None for a failure."""
        value = self.get(key)
        if value is not None:
            return value, HIT
        task = self._inflight.get(key)
        if task is not None:
            return await asyncio.shield(task), COALESCED
        started = self._generation
        task = asyncio.ensure_future(loader())
        self._inflight[key] = task
        task.add_done_callback(lambda t: self._finish(key, t, started))
        return await asyncio.shield(task), MISS

    def invalidate_message(self, channel_id, message_id, deleted=False) -> None:
        """Forget one message's attachments (Discord delete/edit event)."""
        self._generation += 1
        now = self._clock()
        self._prune_invalidated(now)
        message_key = (channel_id, message_id)
        previous = self._invalidated.get(message_key)
        self._invalidated[message_key] = (
            self._generation, deleted or bool(previous and previous[1]), now)
        for key in [k for k in self._entries if k[:2] == message_key]:
            self._remove(key)

    def deleted_since(self, generation, channel_id, message_ids) -> set:
        """Ids from `message_ids` deleted after `generation`."""
        found = set()
        for message_id in message_ids:
            record = self._invalidated.get((channel_id, message_id))
            if record and record[1] and record[0] > generation:
                found.add(message_id)
        return found

    def _finish(self, key, task, started):
        if self._inflight.get(key) is task:
            del self._inflight[key]
        if task.cancelled() or task.exception() is not None:
            return
        value = task.result()
        if value is None:
            return
        record = self._invalidated.get(key[:2])
        if record and record[0] > started:
            return
        self._store(key, value)

    def _store(self, key, value):
        size = len(value)
        if size > self.max_bytes:
            return
        self._expire()
        if key in self._entries:
            self._remove(key)
        self._entries[key] = (value, self._clock() + self.ttl)
        self._bytes += size
        while self._bytes > self.max_bytes:
            oldest = next(iter(self._entries))
            self._remove(oldest)

    def _remove(self, key):
        value, _ = self._entries.pop(key)
        self._bytes -= len(value)

    def _expire(self):
        now = self._clock()
        for key in [k for k, (_, exp) in self._entries.items() if now >= exp]:
            self._remove(key)

    def _prune_invalidated(self, now):
        # A record only matters to a load or build that started before it,
        # and none lasts anywhere near the TTL.
        for key in [k for k, (_, _, at) in self._invalidated.items() if now - at > self.ttl]:
            del self._invalidated[key]


_cache = None


def cache() -> AttachmentCache:
    """The process-wide cache, created on first use."""
    global _cache
    if _cache is None:
        _cache = AttachmentCache()
    return _cache


def reset() -> None:
    """Drop the process-wide cache (tests)."""
    global _cache
    _cache = None

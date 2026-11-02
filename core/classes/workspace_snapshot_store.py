"""Redis store for server workspace snapshots (one tar per guild).

workspace_manager.py saves a guild's workspace here (session.persist_workspace()
-> upload) on a timer and before an idle stop, and restores it
(download -> session.hydrate_workspace()) into a NEW container: one made
because the old one was stopped, removed, or is on another Docker daemon.
Never into a container it re-attached to, which already has newer files.

Not bound into the SDK as a RemoteSnapshot on purpose. With one on the
session, start() after a resume restores the snapshot OVER the live
workspace whenever the two differ (agents.sandbox snapshot_lifecycle), so a
core restart would throw away everything since the last timed snapshot.
That was seen in the phase 2 container spike; the manager owns restore
instead, so it happens only when the container is new.

Stores BINARY tar data, so it uses the binary Redis client (configManager and
UserMemory decode every value to str, which would corrupt a tar).
"""
import asyncio
import io
import os

from classes.redis_client import binary_client

DEFAULT_SNAPSHOT_MAX_BYTES = 50_000_000  # Redis here shares a PVC with settings/user-memory
DEFAULT_SNAPSHOT_TTL_SECONDS = 7 * 24 * 60 * 60  # a server unused for a week starts empty


class SnapshotTooLargeError(Exception):
    """Raised by upload() when the workspace tar exceeds snapshot_max_bytes()."""


def _positive_int(raw: str | None, default: int) -> int:
    try:
        value = int(str(raw).strip())
        return value if value > 0 else default
    except (TypeError, ValueError):
        return default


def snapshot_max_bytes() -> int:
    """Max size of one stored workspace snapshot (WORKSPACE_SNAPSHOT_MAX_BYTES, default 50MB)."""
    return _positive_int(os.environ.get("WORKSPACE_SNAPSHOT_MAX_BYTES"), DEFAULT_SNAPSHOT_MAX_BYTES)


def snapshot_ttl_seconds() -> int:
    """How long an unused server's snapshot survives in Redis
    (WORKSPACE_SNAPSHOT_TTL_SECONDS, default 7 days)."""
    return _positive_int(
        os.environ.get("WORKSPACE_SNAPSHOT_TTL_SECONDS"), DEFAULT_SNAPSHOT_TTL_SECONDS
    )


class WorkspaceSnapshotStore:
    NAMESPACE = "dcb:workspace:v1:snapshot"

    @property
    def redis(self):
        return binary_client()

    def _key(self, snapshot_id: str) -> str:
        return f"{self.NAMESPACE}:{snapshot_id}"

    async def upload(self, snapshot_id: str, data: io.IOBase) -> int:
        """Stores the tar and returns its size. The previous snapshot is kept
        when this one is too large (SnapshotTooLargeError)."""
        # The tar stream can be large; a bare synchronous .read() here would
        # block the bot's single event loop (and the gateway heartbeat with
        # it), so it must go through a thread.
        raw = await asyncio.to_thread(data.read)
        if len(raw) > snapshot_max_bytes():
            raise SnapshotTooLargeError(
                f"workspace snapshot {snapshot_id!r} is {len(raw)} bytes, over the "
                f"{snapshot_max_bytes()}-byte limit"
            )
        await self.redis.set(self._key(snapshot_id), raw, ex=snapshot_ttl_seconds())
        return len(raw)

    async def download(self, snapshot_id: str) -> io.IOBase:
        raw = await self.redis.get(self._key(snapshot_id))
        return io.BytesIO(raw or b"")

    async def exists(self, snapshot_id: str) -> bool:
        return bool(await self.redis.exists(self._key(snapshot_id)))

    async def delete(self, snapshot_id: str) -> None:
        """Drops a snapshot: on /workspace reset, and when one cannot be
        restored (a corrupt tar would otherwise fail every start until its
        TTL ran out)."""
        await self.redis.delete(self._key(snapshot_id))

    async def size(self, snapshot_id: str) -> int | None:
        """Bytes stored, or None when there is no snapshot (for /workspace status)."""
        value = await self.redis.strlen(self._key(snapshot_id))
        return int(value) if value else None

"""Redis-backed remote snapshot store for sandbox workspace persistence.

Implements the upload/download/exists trio that `agents.sandbox.RemoteSnapshot`
requires (see `examples/sandbox/sandbox_agent_with_remote_snapshot.py` in the
openai-agents-python SDK) so a sandbox's workspace can be reseeded into a
fresh, disposable container on a later run in the same Discord thread — see
classes/sandbox_agent.py's build_sandbox_client/_create_sandbox_session for
how a snapshot id (the thread id) ties one Discord thread's runs together.

Unlike configManager/UserMemory, this stores BINARY tar data, so the Redis
client here is opened WITHOUT decode_responses (the other two classes decode
every value to str, which would corrupt a tar blob).
"""
import asyncio
import io
import os

from classes.redis_client import binary_client

DEFAULT_SNAPSHOT_MAX_BYTES = 50_000_000  # Redis here shares a PVC with settings/user-memory
DEFAULT_SNAPSHOT_TTL_SECONDS = 7 * 24 * 60 * 60  # abandoned threads don't grow Redis forever


class SnapshotTooLargeError(Exception):
    """Raised by upload() when the workspace tar exceeds sandbox_snapshot_max_bytes()."""


def _positive_int(raw: str | None, default: int) -> int:
    try:
        value = int(str(raw).strip())
        return value if value > 0 else default
    except (TypeError, ValueError):
        return default


def sandbox_snapshot_max_bytes() -> int:
    """Max size of one stored sandbox workspace snapshot
    (SANDBOX_SNAPSHOT_MAX_BYTES, default 50MB)."""
    return _positive_int(os.environ.get("SANDBOX_SNAPSHOT_MAX_BYTES"), DEFAULT_SNAPSHOT_MAX_BYTES)


def sandbox_snapshot_ttl_seconds() -> int:
    """How long an unused sandbox snapshot survives in Redis
    (SANDBOX_SNAPSHOT_TTL_SECONDS, default 7 days)."""
    return _positive_int(
        os.environ.get("SANDBOX_SNAPSHOT_TTL_SECONDS"), DEFAULT_SNAPSHOT_TTL_SECONDS
    )


class SandboxSnapshotStore:
    """Bound into `agents.sandbox.session.Dependencies` under a
    client_dependency_key (see classes.sandbox_agent.SANDBOX_SNAPSHOT_DEP_KEY);
    implements the three methods `RemoteSnapshot` requires (async — its
    `_require_client_method` awaits the result via `_maybe_await`, so no sync
    wrapper is needed here)."""

    NAMESPACE = "dcb"

    @property
    def redis(self):
        # The BINARY shared client (classes/redis_client.py): this store holds
        # tar bytes, unlike configManager/UserMemory, which decode to str.
        # sandbox_agent.py constructs a SandboxSnapshotStore at four separate
        # points in one run, so sharing the pool matters here too.
        return binary_client()

    def _key(self, snapshot_id: str) -> str:
        return f"{self.NAMESPACE}:sandbox_snapshot:{snapshot_id}"

    async def upload(self, snapshot_id: str, data: io.IOBase) -> None:
        # The tar stream can be large; a bare synchronous .read() here would
        # block the bot's single event loop (and the gateway heartbeat with
        # it), so it must go through a thread.
        raw = await asyncio.to_thread(data.read)
        if len(raw) > sandbox_snapshot_max_bytes():
            raise SnapshotTooLargeError(
                f"sandbox snapshot for {snapshot_id!r} is {len(raw)} bytes, over the "
                f"{sandbox_snapshot_max_bytes()}-byte limit"
            )
        await self.redis.set(self._key(snapshot_id), raw, ex=sandbox_snapshot_ttl_seconds())

    async def download(self, snapshot_id: str) -> io.IOBase:
        raw = await self.redis.get(self._key(snapshot_id))
        return io.BytesIO(raw or b"")

    async def exists(self, snapshot_id: str) -> bool:
        return bool(await self.redis.exists(self._key(snapshot_id)))

    async def delete(self, snapshot_id: str) -> None:
        """Drops a thread's saved workspace.

        Not part of the RemoteSnapshot protocol — this is the recovery path
        in classes/sandbox_agent.py's _create_sandbox_session: a snapshot that
        cannot be restored (corrupt/truncated tar) would otherwise fail every
        future run in that thread until its TTL expires, so the bad key is
        deleted and the run retried against the same id, which then starts
        empty and writes a good snapshot back on teardown.
        """
        await self.redis.delete(self._key(snapshot_id))

    async def ttl(self, snapshot_id: str) -> int:
        """Seconds until this thread's saved workspace expires, as Redis
        reports it: -2 when the key does not exist, -1 when it has no
        expiry. Used for the "resumable for N more days" line in the
        sandbox's closing note — reading the REAL remaining time rather than
        echoing sandbox_snapshot_ttl_seconds() means the note cannot promise
        a resume for a snapshot whose persist silently failed (persisting is
        best-effort, see _persist_sandbox_snapshot).
        """
        return int(await self.redis.ttl(self._key(snapshot_id)))

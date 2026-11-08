"""The server workspace's snapshot store (classes/workspace_snapshot_store.py).

Run from the repo root: `pytest core/tests/workspace_snapshot_store_tests.py`
"""
import io

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from classes import redis_client

from classes.workspace_snapshot_store import (
    WorkspaceSnapshotStore,
    SnapshotTooLargeError,
    snapshot_max_bytes,
    snapshot_ttl_seconds,
)

@pytest.fixture
def mock_redis(monkeypatch):
    mock_redis_instance = MagicMock()
    mock_redis_instance.get = AsyncMock()
    mock_redis_instance.set = AsyncMock()
    mock_redis_instance.exists = AsyncMock()
    monkeypatch.setenv('REDIS_HOST', 'localhost')
    monkeypatch.setattr('redis.asyncio.Redis', MagicMock(return_value=mock_redis_instance))
    # The clients are shared and cached (classes/redis_client.py), so the
    # cache must be dropped either side of the patch or the first test's
    # mock would be handed to every later one.
    redis_client.reset_clients()
    yield mock_redis_instance
    redis_client.reset_clients()


# ---------------------- env accessors ----------------------

def test_max_bytes_default(monkeypatch):
    monkeypatch.delenv("WORKSPACE_SNAPSHOT_MAX_BYTES", raising=False)
    assert snapshot_max_bytes() == 50_000_000


def test_max_bytes_override(monkeypatch):
    monkeypatch.setenv("WORKSPACE_SNAPSHOT_MAX_BYTES", "1000")
    assert snapshot_max_bytes() == 1000


def test_max_bytes_invalid_falls_back(monkeypatch):
    for bad in ("abc", "-1", "0", ""):
        monkeypatch.setenv("WORKSPACE_SNAPSHOT_MAX_BYTES", bad)
        assert snapshot_max_bytes() == 50_000_000, bad


def test_ttl_default(monkeypatch):
    monkeypatch.delenv("WORKSPACE_SNAPSHOT_TTL_SECONDS", raising=False)
    assert snapshot_ttl_seconds() == 7 * 24 * 60 * 60


def test_ttl_override(monkeypatch):
    monkeypatch.setenv("WORKSPACE_SNAPSHOT_TTL_SECONDS", "60")
    assert snapshot_ttl_seconds() == 60


def test_ttl_invalid_falls_back(monkeypatch):
    for bad in ("abc", "-1", "0"):
        monkeypatch.setenv("WORKSPACE_SNAPSHOT_TTL_SECONDS", bad)
        assert snapshot_ttl_seconds() == 7 * 24 * 60 * 60, bad


# ---------------------- upload ----------------------

@pytest.mark.asyncio
async def test_upload_stores_raw_bytes_with_ttl(mock_redis):
    store = WorkspaceSnapshotStore()
    await store.upload("guild-1", io.BytesIO(b"tarbytes"))

    mock_redis.set.assert_awaited_once()
    args, kwargs = mock_redis.set.call_args
    assert args[0] == "dcb:workspace:v1:snapshot:guild-1"
    assert args[1] == b"tarbytes"
    assert kwargs["ex"] == snapshot_ttl_seconds()


@pytest.mark.asyncio
async def test_upload_reads_the_stream_via_a_thread(mock_redis):
    # A bare synchronous data.read() would block the bot's single event
    # loop (and the gateway heartbeat with it); upload() must go through
    # asyncio.to_thread, never call data.read() inline.
    data = MagicMock()
    data.read = MagicMock(return_value=b"tarbytes")

    with patch("classes.workspace_snapshot_store.asyncio.to_thread",
               AsyncMock(return_value=b"tarbytes")) as to_thread:
        store = WorkspaceSnapshotStore()
        await store.upload("guild-1", data)

    to_thread.assert_awaited_once_with(data.read)


@pytest.mark.asyncio
async def test_upload_rejects_oversized_payload(mock_redis, monkeypatch):
    monkeypatch.setenv("WORKSPACE_SNAPSHOT_MAX_BYTES", "4")
    store = WorkspaceSnapshotStore()
    with pytest.raises(SnapshotTooLargeError):
        await store.upload("guild-1", io.BytesIO(b"too big"))
    mock_redis.set.assert_not_awaited()


@pytest.mark.asyncio
async def test_upload_accepts_payload_at_exactly_the_cap(mock_redis, monkeypatch):
    monkeypatch.setenv("WORKSPACE_SNAPSHOT_MAX_BYTES", "4")
    store = WorkspaceSnapshotStore()
    await store.upload("guild-1", io.BytesIO(b"1234"))
    mock_redis.set.assert_awaited_once()


# ---------------------- download ----------------------

@pytest.mark.asyncio
async def test_download_wraps_bytes_in_a_readable_stream(mock_redis):
    mock_redis.get.return_value = b"tarbytes"
    store = WorkspaceSnapshotStore()
    result = await store.download("guild-1")
    assert isinstance(result, io.IOBase)
    assert result.read() == b"tarbytes"
    mock_redis.get.assert_awaited_once_with("dcb:workspace:v1:snapshot:guild-1")


@pytest.mark.asyncio
async def test_download_missing_key_returns_an_empty_stream(mock_redis):
    mock_redis.get.return_value = None
    store = WorkspaceSnapshotStore()
    result = await store.download("guild-1")
    assert result.read() == b""


# ---------------------- exists ----------------------

@pytest.mark.asyncio
async def test_exists_true_when_key_present(mock_redis):
    mock_redis.exists.return_value = 1
    store = WorkspaceSnapshotStore()
    assert await store.exists("guild-1") is True
    mock_redis.exists.assert_awaited_once_with("dcb:workspace:v1:snapshot:guild-1")


@pytest.mark.asyncio
async def test_exists_false_when_key_absent(mock_redis):
    mock_redis.exists.return_value = 0
    store = WorkspaceSnapshotStore()
    assert await store.exists("guild-1") is False


# ---------------------- key namespacing ----------------------

def test_different_guilds_get_different_keys(mock_redis):
    store = WorkspaceSnapshotStore()
    assert store._key("1") != store._key("2")
    assert store._key("1") == "dcb:workspace:v1:snapshot:1"


# ---------------------- delete ----------------------

@pytest.mark.asyncio
async def test_delete_removes_the_key(mock_redis):
    mock_redis.delete = AsyncMock()
    store = WorkspaceSnapshotStore()
    await store.delete("42")
    mock_redis.delete.assert_awaited_once_with("dcb:workspace:v1:snapshot:42")


@pytest.mark.asyncio
async def test_upload_returns_the_size(mock_redis):
    assert await WorkspaceSnapshotStore().upload("guild-1", io.BytesIO(b"1234")) == 4


@pytest.mark.asyncio
async def test_size_is_none_without_a_snapshot(mock_redis):
    mock_redis.strlen = AsyncMock(return_value=0)
    assert await WorkspaceSnapshotStore().size("guild-1") is None
    mock_redis.strlen = AsyncMock(return_value=2048)
    assert await WorkspaceSnapshotStore().size("guild-1") == 2048

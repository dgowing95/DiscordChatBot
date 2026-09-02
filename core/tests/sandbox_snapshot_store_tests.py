import io
import os
import sys

# sandbox_snapshot_store is imported as core.classes.* here and classes.* in
# production (the app runs with cwd=core/); both resolve as namespace packages.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core.classes.sandbox_snapshot_store import (
    SandboxSnapshotStore,
    SnapshotTooLargeError,
    sandbox_snapshot_max_bytes,
    sandbox_snapshot_ttl_seconds,
)

# To run this pytest file from the command line, use:
# PYTHONPATH=$(pwd) pytest core/tests/sandbox_snapshot_store_tests.py


@pytest.fixture
def mock_redis(monkeypatch):
    mock_redis_instance = MagicMock()
    mock_redis_instance.get = AsyncMock()
    mock_redis_instance.set = AsyncMock()
    mock_redis_instance.exists = AsyncMock()
    monkeypatch.setenv('REDIS_HOST', 'localhost')
    monkeypatch.setattr('redis.asyncio.Redis', MagicMock(return_value=mock_redis_instance))
    return mock_redis_instance


# ---------------------- env accessors ----------------------

def test_max_bytes_default(monkeypatch):
    monkeypatch.delenv("SANDBOX_SNAPSHOT_MAX_BYTES", raising=False)
    assert sandbox_snapshot_max_bytes() == 50_000_000


def test_max_bytes_override(monkeypatch):
    monkeypatch.setenv("SANDBOX_SNAPSHOT_MAX_BYTES", "1000")
    assert sandbox_snapshot_max_bytes() == 1000


def test_max_bytes_invalid_falls_back(monkeypatch):
    for bad in ("abc", "-1", "0", ""):
        monkeypatch.setenv("SANDBOX_SNAPSHOT_MAX_BYTES", bad)
        assert sandbox_snapshot_max_bytes() == 50_000_000, bad


def test_ttl_default(monkeypatch):
    monkeypatch.delenv("SANDBOX_SNAPSHOT_TTL_SECONDS", raising=False)
    assert sandbox_snapshot_ttl_seconds() == 7 * 24 * 60 * 60


def test_ttl_override(monkeypatch):
    monkeypatch.setenv("SANDBOX_SNAPSHOT_TTL_SECONDS", "60")
    assert sandbox_snapshot_ttl_seconds() == 60


def test_ttl_invalid_falls_back(monkeypatch):
    for bad in ("abc", "-1", "0"):
        monkeypatch.setenv("SANDBOX_SNAPSHOT_TTL_SECONDS", bad)
        assert sandbox_snapshot_ttl_seconds() == 7 * 24 * 60 * 60, bad


# ---------------------- upload ----------------------

@pytest.mark.asyncio
async def test_upload_stores_raw_bytes_with_ttl(mock_redis):
    store = SandboxSnapshotStore()
    await store.upload("thread-1", io.BytesIO(b"tarbytes"))

    mock_redis.set.assert_awaited_once()
    args, kwargs = mock_redis.set.call_args
    assert args[0] == "dcb:sandbox_snapshot:thread-1"
    assert args[1] == b"tarbytes"
    assert kwargs["ex"] == sandbox_snapshot_ttl_seconds()


@pytest.mark.asyncio
async def test_upload_reads_the_stream_via_a_thread(mock_redis):
    # A bare synchronous data.read() would block the bot's single event
    # loop (and the gateway heartbeat with it); upload() must go through
    # asyncio.to_thread, never call data.read() inline.
    data = MagicMock()
    data.read = MagicMock(return_value=b"tarbytes")

    with patch("core.classes.sandbox_snapshot_store.asyncio.to_thread",
               AsyncMock(return_value=b"tarbytes")) as to_thread:
        store = SandboxSnapshotStore()
        await store.upload("thread-1", data)

    to_thread.assert_awaited_once_with(data.read)


@pytest.mark.asyncio
async def test_upload_rejects_oversized_payload(mock_redis, monkeypatch):
    monkeypatch.setenv("SANDBOX_SNAPSHOT_MAX_BYTES", "4")
    store = SandboxSnapshotStore()
    with pytest.raises(SnapshotTooLargeError):
        await store.upload("thread-1", io.BytesIO(b"too big"))
    mock_redis.set.assert_not_awaited()


@pytest.mark.asyncio
async def test_upload_accepts_payload_at_exactly_the_cap(mock_redis, monkeypatch):
    monkeypatch.setenv("SANDBOX_SNAPSHOT_MAX_BYTES", "4")
    store = SandboxSnapshotStore()
    await store.upload("thread-1", io.BytesIO(b"1234"))
    mock_redis.set.assert_awaited_once()


# ---------------------- download ----------------------

@pytest.mark.asyncio
async def test_download_wraps_bytes_in_a_readable_stream(mock_redis):
    mock_redis.get.return_value = b"tarbytes"
    store = SandboxSnapshotStore()
    result = await store.download("thread-1")
    assert isinstance(result, io.IOBase)
    assert result.read() == b"tarbytes"
    mock_redis.get.assert_awaited_once_with("dcb:sandbox_snapshot:thread-1")


@pytest.mark.asyncio
async def test_download_missing_key_returns_an_empty_stream(mock_redis):
    mock_redis.get.return_value = None
    store = SandboxSnapshotStore()
    result = await store.download("thread-1")
    assert result.read() == b""


# ---------------------- exists ----------------------

@pytest.mark.asyncio
async def test_exists_true_when_key_present(mock_redis):
    mock_redis.exists.return_value = 1
    store = SandboxSnapshotStore()
    assert await store.exists("thread-1") is True
    mock_redis.exists.assert_awaited_once_with("dcb:sandbox_snapshot:thread-1")


@pytest.mark.asyncio
async def test_exists_false_when_key_absent(mock_redis):
    mock_redis.exists.return_value = 0
    store = SandboxSnapshotStore()
    assert await store.exists("thread-1") is False


# ---------------------- key namespacing ----------------------

def test_different_threads_get_different_keys(mock_redis):
    store = SandboxSnapshotStore()
    assert store._key("1") != store._key("2")
    assert store._key("1") == "dcb:sandbox_snapshot:1"


# ---------------------- delete ----------------------

@pytest.mark.asyncio
async def test_delete_removes_the_thread_key(mock_redis):
    mock_redis.delete = AsyncMock()
    store = SandboxSnapshotStore()
    await store.delete("42")
    mock_redis.delete.assert_awaited_once_with("dcb:sandbox_snapshot:42")


@pytest.mark.asyncio
async def test_ttl_reports_the_remaining_seconds(mock_redis):
    mock_redis.ttl = AsyncMock(return_value=604800)
    store = SandboxSnapshotStore()
    assert await store.ttl("42") == 604800
    mock_redis.ttl.assert_awaited_once_with("dcb:sandbox_snapshot:42")


@pytest.mark.asyncio
async def test_ttl_passes_through_the_missing_key_sentinel(mock_redis):
    # Redis answers -2 for a key that does not exist; the caller
    # (sandbox_snapshot_remaining_seconds) turns that into "nothing saved".
    mock_redis.ttl = AsyncMock(return_value=-2)
    store = SandboxSnapshotStore()
    assert await store.ttl("42") == -2

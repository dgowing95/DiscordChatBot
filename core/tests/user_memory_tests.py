import os
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from classes import redis_client
from classes.user_memory import UserMemory

@pytest.fixture
def mock_redis(monkeypatch):
    mock_redis_instance = MagicMock()
    mock_redis_instance.get = AsyncMock()
    monkeypatch.setenv('REDIS_HOST', 'localhost')
    monkeypatch.setattr('redis.asyncio.Redis', MagicMock(return_value=mock_redis_instance))
    # The clients are shared and cached (classes/redis_client.py), so the
    # cache must be dropped either side of the patch or the first test's
    # mock would be handed to every later one.
    redis_client.reset_clients()
    yield mock_redis_instance
    redis_client.reset_clients()

@pytest.mark.asyncio
async def test_get_returns_deduped_set(mock_redis):
    # Simulate redis returning a JSON array with duplicates
    data = ['a', 'b', 'a', 'c']
    mock_redis.get.return_value = json.dumps(data)
    um = UserMemory(user_id=1, guild_id=2)
    result = set(await um.get())
    assert result == set(['a', 'b', 'c'])

@pytest.mark.asyncio
async def test_get_returns_none_when_no_data(mock_redis):
    mock_redis.get.return_value = None
    um = UserMemory(user_id=1, guild_id=2)
    result = await um.get()
    assert result is None

@pytest.mark.asyncio
async def test_get_returns_empty_set_when_empty_list(mock_redis):
    mock_redis.get.return_value = json.dumps([])
    um = UserMemory(user_id=1, guild_id=2)
    result = await um.get()
    assert result == list()

# To run this pytest file from the command line, use:
# PYTHONPATH=$(pwd) pytest core/tests/user_memory_tests.py
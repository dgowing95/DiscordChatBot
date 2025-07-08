import os
import json
import pytest
from unittest.mock import MagicMock, patch
from core.classes.user_memory import UserMemory

@pytest.fixture
def mock_redis(monkeypatch):
    mock_redis_instance = MagicMock()
    monkeypatch.setenv('REDIS_HOST', 'localhost')
    monkeypatch.setattr('redis.Redis', MagicMock(return_value=mock_redis_instance))
    return mock_redis_instance

def test_get_returns_deduped_set(mock_redis):
    # Simulate redis returning a JSON array with duplicates
    data = ['a', 'b', 'a', 'c']
    mock_redis.get.return_value = json.dumps(data)
    um = UserMemory(user_id=1, guild_id=2)
    result = set(um.get())
    assert result == set(['a', 'b', 'c'])

def test_get_returns_none_when_no_data(mock_redis):
    mock_redis.get.return_value = None
    um = UserMemory(user_id=1, guild_id=2)
    result = um.get()
    assert result is None

def test_get_returns_empty_set_when_empty_list(mock_redis):
    mock_redis.get.return_value = json.dumps([])
    um = UserMemory(user_id=1, guild_id=2)
    result = um.get()
    assert result == list()

# To run this pytest file from the command line, use:
# PYTHONPATH=$(pwd) pytest core/tests/user_memory_tests.py
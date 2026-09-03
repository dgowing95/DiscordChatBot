"""Shared Redis clients.

configManager, UserMemory and SandboxSnapshotStore each used to build their
own redis.asyncio.Redis - and so their own connection pool - in __init__.
They are constructed per message and per tool call (12 sites across the
codebase: two in TextLLMHandler.__init__ alone, one per memory tool, plus
four SandboxSnapshotStore() calls in sandbox_agent.py), and nothing ever
closed any of them. redis-py only tears a connection down when GC happens to
collect it, and only when a loop is still running, so the bot churned pools
and sockets for work that should reuse one.

Two clients, not one: the snapshot store holds binary tar data and must NOT
decode responses, while the settings and user-memory stores decode to str.

Same lazily-built singleton shape as _get_main_model_client() in
text_llm_handler.py, and safe for the same reason - the event loop is
single-threaded and there is no await between the check and the set.

REDIS_HOST is read here, the first time a client is actually needed, rather
than in each class's __init__. That is what lets `import main` succeed with no
Redis configuration at all, which in turn is what lets the test suite stop
calling os.environ.setdefault("REDIS_HOST", "localhost") - a leak that made
every later test pay a real connection timeout to a Redis that wasn't there.
"""
import os

import redis.asyncio as redis

_text_client = None
_binary_client = None


def _host() -> str:
    """The Redis host. Deliberately a KeyError when unset, exactly as the
    per-instance constructors were: callers already treat that as "no
    settings available" and it fails immediately instead of hanging."""
    return os.environ['REDIS_HOST']


def text_client():
    """Shared client for per-guild settings and user memories (str values)."""
    global _text_client
    if _text_client is None:
        _text_client = redis.Redis(
            host=_host(), port=6379, db=0, encoding="utf-8", decode_responses=True
        )
    return _text_client


def binary_client():
    """Shared client for sandbox workspace snapshots (raw tar bytes)."""
    global _binary_client
    if _binary_client is None:
        _binary_client = redis.Redis(host=_host(), port=6379, db=0)
    return _binary_client


def reset_clients() -> None:
    """Drop both cached clients so the next call rebuilds them.

    For tests, which patch redis.asyncio.Redis per test: without this the
    first test's client would be cached and handed to every later one.
    """
    global _text_client, _binary_client
    _text_client = None
    _binary_client = None

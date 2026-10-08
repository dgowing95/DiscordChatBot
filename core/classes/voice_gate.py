"""Which guilds the bot is in a voice call in (PURE, stdlib only).

While a guild is here, its text work stops: on_message ignores the guild's
messages (sandbox steering aside), queued messages and automation jobs are
dropped at dequeue, due schedules wait, and /generate_image refuses. The
bot then never talks in text and voice in one guild at once, and that
guild's llama.cpp time goes to the call.

In-process on purpose, never in Redis: a restart ends every call (the voice
sidecar drops its connections when core goes away), and state that outlived
the process could leave a guild silent for good.

voice_session.end() is the only place a guild leaves, and it also tears down
the discord.py voice client, so the two cannot drift apart.
"""

_active: dict[int, object] = {}


def enter(guild_id: int, session) -> None:
    _active[int(guild_id)] = session


def leave(guild_id: int, session=None) -> None:
    """Drop the guild; with `session`, only if that is still the one there
    (a late cleanup of an old session must not end a newer one)."""
    guild_id = int(guild_id)
    if session is None or _active.get(guild_id) is session:
        _active.pop(guild_id, None)


def active(guild_id) -> bool:
    try:
        return int(guild_id) in _active
    except (TypeError, ValueError):
        return False


def session(guild_id):
    try:
        return _active.get(int(guild_id))
    except (TypeError, ValueError):
        return None


def sessions() -> list:
    return list(_active.values())


def reset() -> None:
    """Tests only."""
    _active.clear()

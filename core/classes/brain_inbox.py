"""Questions the brain is waiting on. PURE (stdlib only).

When the brain calls `ask_user` (classes/brain.py) it posts the question in
the channel, as a reply to the requester, and waits here for their answer:
one waiter per (channel id, requester id). main.on_message (and, in a voice
call, VoiceSession.heard) hands the requester's next message in that channel
to the waiter instead of the persona.

Only the requester can answer: the key is their user id, so another member's
message in the same channel is answered as usual. A webhook's message has the
webhook as its author, so a webhook can answer only the questions its own
messages caused (how the dev stack and E2E drive the bot).

No mutating function here awaits, so each one is atomic on the one event
loop: a delivery can never land between a waiter's check and its removal.
"""
import asyncio

# deliver() outcomes.
ACCEPTED = "accepted"  # the waiter has the answer
EMPTY = "empty"  # nothing to read (attachments only); the waiter keeps waiting
NO_WAITER = "no_waiter"

# What a replaced waiter resolves to (a newer question from the same call or
# another call by the same requester in the same channel took its place).
REPLACED = None

_waiters: dict[tuple[int, int], asyncio.Future] = {}


def _key(channel_id, user_id) -> tuple[int, int]:
    return int(channel_id), int(user_id)


def ask(channel_id, user_id) -> asyncio.Future:
    """Register a waiter and return its future (the answer's text, or
    REPLACED). An older waiter for the same key resolves to REPLACED."""
    key = _key(channel_id, user_id)
    old = _waiters.get(key)
    if old is not None and not old.done():
        old.set_result(REPLACED)
    future = asyncio.get_running_loop().create_future()
    _waiters[key] = future
    return future


def waiting(channel_id, user_id) -> bool:
    """Whether a question to this person in this channel is open."""
    future = _waiters.get(_key(channel_id, user_id))
    return future is not None and not future.done()


def deliver(channel_id, user_id, text: str) -> str:
    """Hand a message to the waiter. Text-less messages are refused
    (EMPTY) and the question stays open."""
    key = _key(channel_id, user_id)
    future = _waiters.get(key)
    if future is None or future.done():
        return NO_WAITER
    text = (text or "").strip()
    if not text:
        return EMPTY
    future.set_result(text)
    del _waiters[key]
    return ACCEPTED


def cancel(channel_id, user_id, future: asyncio.Future) -> None:
    """Remove THIS waiter (called in a finally: a timeout, or the brain run
    being stopped mid-wait). A newer waiter under the same key is left
    alone. Without it a stale waiter would swallow the requester's next
    message."""
    key = _key(channel_id, user_id)
    if _waiters.get(key) is future:
        del _waiters[key]
    if not future.done():
        future.cancel()


def reset() -> None:
    """Forget every waiter (tests)."""
    _waiters.clear()

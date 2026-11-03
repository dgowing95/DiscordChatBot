"""Questions the brain is waiting on. PURE (stdlib only).

When the brain calls `ask_user` (classes/brain.py) it posts the question in
the channel, as a reply to the request, and waits here for the answer. One
waiter per question message. Answering is always deliberate
(maintainer, 2026-11-03): the requester presses one of the question's
buttons, types in its "Answer…" box (classes/brain_question.py), or uses
Discord's Reply on the question (main.on_message). An ordinary message is a
new request, never taken as an answer: the first version took the
requester's next message, so a new request sent instead of an answer was
swallowed with no reply.

The one exception is a voice call, where nobody can press Reply: there the
speaker's next utterance is the answer (VoiceSession.heard, through
voice_waiting), as with the wake phrase's follow-up window.

Only the requester can answer: every delivery names the person, and a
waiter takes only its own requester's.

No mutating function here awaits, so each one is atomic on the one event
loop: a delivery can never land between a waiter's check and its removal.
"""
import asyncio
from dataclasses import dataclass, field

# deliver() outcomes.
ACCEPTED = "accepted"  # the waiter has the answer
EMPTY = "empty"  # nothing to read (attachments only); the waiter keeps waiting
NOT_YOURS = "not_yours"  # someone other than the requester
NO_WAITER = "no_waiter"

# What a replaced waiter resolves to (a newer question to the same person in
# the same channel took its place: its buttons are dead).
REPLACED = None


@dataclass
class _Waiter:
    channel_id: int
    user_id: int
    voice: bool
    future: asyncio.Future = field(repr=False)


_waiters: dict[int, _Waiter] = {}  # question message id -> waiter


def ask(question_id, channel_id, user_id, *, voice: bool = False) -> asyncio.Future:
    """Register a waiter for the question message `question_id` and return
    its future (the answer's text, or REPLACED). An open question to the
    same person in the same channel resolves to REPLACED."""
    channel_id, user_id = int(channel_id), int(user_id)
    for old_id, old in list(_waiters.items()):
        if old.channel_id == channel_id and old.user_id == user_id:
            if not old.future.done():
                old.future.set_result(REPLACED)
            del _waiters[old_id]
    future = asyncio.get_running_loop().create_future()
    _waiters[int(question_id)] = _Waiter(channel_id, user_id, voice, future)
    return future


def waiting(question_id) -> bool:
    """Whether the question with this message id is still open."""
    waiter = _waiters.get(int(question_id))
    return waiter is not None and not waiter.future.done()


def requester(question_id) -> int | None:
    """Who may answer this question (None when it is not open)."""
    waiter = _waiters.get(int(question_id))
    return waiter.user_id if waiter is not None and not waiter.future.done() else None


def voice_waiting(channel_id, user_id) -> int | None:
    """The open voice question to this speaker in this channel, if any."""
    channel_id, user_id = int(channel_id), int(user_id)
    for question_id, waiter in _waiters.items():
        if (waiter.voice and waiter.channel_id == channel_id and waiter.user_id == user_id
                and not waiter.future.done()):
            return question_id
    return None


def deliver(question_id, user_id, text: str) -> str:
    """Hand an answer to the question's waiter. Only its requester's count;
    a text-less answer is refused (EMPTY) and the question stays open."""
    question_id = int(question_id)
    waiter = _waiters.get(question_id)
    if waiter is None or waiter.future.done():
        return NO_WAITER
    if int(user_id) != waiter.user_id:
        return NOT_YOURS
    text = (text or "").strip()
    if not text:
        return EMPTY
    waiter.future.set_result(text)
    del _waiters[question_id]
    return ACCEPTED


def cancel(question_id, future: asyncio.Future) -> None:
    """Remove THIS waiter (called in a finally: a timeout, or the brain run
    being stopped mid-wait). A newer waiter is left alone."""
    question_id = int(question_id)
    waiter = _waiters.get(question_id)
    if waiter is not None and waiter.future is future:
        del _waiters[question_id]
    if not future.done():
        future.cancel()


def reset() -> None:
    """Forget every waiter (tests)."""
    _waiters.clear()

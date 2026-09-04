"""In-process mailbox letting people talk to a sandbox agent mid-run.

A sandbox run happens inside a Discord thread (see classes/sandbox_agent.py's
ensure_sandbox_thread). While it runs, messages posted in that thread are
delivered here instead of going to the outer LLM, and the nested sandbox agent
picks them up — automatically, appended to its next shell command's output, or
explicitly via its check_thread_messages tool. That makes a run steerable
("actually make it blue") without restarting it.

Deliberately pure: no discord and no SDK imports, so every rule here is unit
testable without a Discord client. main.py's on_message does the Discord half
(one predicate call plus one deliver call), mirroring how sandbox_progress.py
splits its pure transcript from its Discord layer.

This is also the concurrency guard for snapshots. Two runs in one thread would
both persist to dcb:sandbox_snapshot:{thread_id} on teardown and the last one
to finish would silently clobber the other, so a thread with a run in flight
never starts a second one — the new request is forwarded into the running
sandbox instead (main.py returns early; run_code_sandbox has the same guard for
the paths that bypass on_message).

State lives in a module-level dict, which is correct here and not a shortcut:
the bot is one process running one asyncio event loop, and both the producer
(on_message) and the consumer (the nested tools) live in it. Nothing here is
persisted — a pending message is only meaningful to the run it was aimed at,
and that run dies with the process.

Known failure mode: because the registry is plain process memory, deregistered
in run_code_sandbox's `finally`, a worker torn down in a way that skips that
finally leaves the thread registered. Every later message posted there then
gets a 📨 reaction and goes nowhere, because main.py routes it to a run that no
longer exists. A process restart clears it. If a thread ever "goes deaf" —
reactions but no replies — that is what happened.
"""

# Caps on one thread's pending queue. A sandbox run can last minutes, and
# everything buffered here is eventually pasted into the nested model's
# context, so an unbounded queue would be both a context-blowout and a way
# for a chatty thread to derail a run. Module constants rather than env vars,
# matching MAX_ARTIFACT_FILES / MAX_ARTIFACT_BYTES in sandbox_agent.py.
MAX_PENDING_MESSAGES = 20
MAX_PENDING_CHARS = 4000
MAX_MESSAGE_CHARS = 1000  # per-message cap, applied before the total

# thread id -> list of (message_id, author, text) waiting to be drained.
# A thread is "active" iff it has an entry here at all (an empty list still
# means a run is in flight), so begin_run/end_run own the key's lifetime and
# drain() must never delete it.
_PENDING: dict[int, list[tuple[int, str, str]]] = {}

# thread id -> every message accepted for this run, in order, kept until the
# run ends. _PENDING is a queue that empties as the sandbox reads it; this is
# the record of what was said, which outlives the reading.
#
# It exists for the OUTER model, which otherwise cannot know the request
# changed: main.py's on_message routes a message in an active thread here
# instead of enqueuing it, and the outer agent's history was snapshotted
# before the run started. Without this it answered a red-fizzy-milk image
# with "looks like the sandbox went a bit rogue", having only ever seen
# "a glass of milk".
#
# Recorded in deliver() rather than drain() for two reasons: one call site
# instead of drain()'s four, and consume() removes a message from _PENDING
# because ask_user already received it — it steered the run just as much as
# any other, so it must survive here.
_SEEN: dict[int, list[tuple[str, str]]] = {}


def begin_run(thread_id: int) -> None:
    """Marks a sandbox run as in flight in this thread, so messages posted
    here are routed to it. Idempotent: a re-entry keeps anything already
    pending rather than dropping it."""
    _PENDING.setdefault(int(thread_id), [])
    _SEEN.setdefault(int(thread_id), [])


def end_run(thread_id: int) -> None:
    """Marks the run finished. Anything still undrained is discarded — the
    run it was addressed to is over, and holding it would leak it into an
    unrelated later run in the same thread. The same goes for the seen
    record, so read it with history() BEFORE calling this."""
    _PENDING.pop(int(thread_id), None)
    _SEEN.pop(int(thread_id), None)


def is_run_active(thread_id: int) -> bool:
    """Whether a sandbox run is currently in flight in this thread."""
    return int(thread_id) in _PENDING


def deliver(thread_id: int, message_id: int, author: str, text: str) -> bool:
    """Queues one thread message for the running sandbox agent.

    Returns True when it was accepted, False when there is no run in flight,
    the message is empty, or a cap is already reached — the caller uses that
    to acknowledge accurately (main.py reacts 📨 vs 🚫) rather than silently
    swallowing a message the model will never see.
    """
    pending = _PENDING.get(int(thread_id))
    if pending is None:
        return False
    text = (text or "").strip()
    if not text:
        return False
    if len(pending) >= MAX_PENDING_MESSAGES:
        return False
    if len(text) > MAX_MESSAGE_CHARS:
        text = text[: MAX_MESSAGE_CHARS - 1] + "…"
    if sum(len(t) for _, _, t in pending) + len(text) > MAX_PENDING_CHARS:
        return False
    author = str(author or "someone")
    pending.append((int(message_id), author, text))
    # Only messages that were actually ACCEPTED are recorded: a rejected one
    # never reaches the sandbox, so telling the outer model the request
    # changed because of it would be a lie in the opposite direction.
    _SEEN.setdefault(int(thread_id), []).append((author, text))
    return True


def consume(thread_id: int, message_id: int) -> None:
    """Drops one specific pending message.

    Needed because discord.py's Client.dispatch fans a message out to
    wait_for futures AND to on_message independently: a reply that ask_user
    was waiting for arrives through wait_for *and* gets queued here. Without
    this, the same message would be handed to the model twice — once as the
    answer to its question, once as an unrelated interjection.
    """
    pending = _PENDING.get(int(thread_id))
    if not pending:
        return
    message_id = int(message_id)
    pending[:] = [entry for entry in pending if entry[0] != message_id]


def drain(thread_id: int) -> str:
    """Pops everything pending for this thread, rendered for the model.

    Returns "" when there is nothing (or no run) — callers append the result
    unconditionally, so an empty string is what keeps a quiet thread from
    adding noise to every tool result.
    """
    pending = _PENDING.get(int(thread_id))
    if not pending:
        return ""
    lines = [f"[thread message from {author}]: {text}" for _, author, text in pending]
    pending.clear()
    return "\n".join(lines)


def history(thread_id: int) -> str:
    """Everything said to the run in flight in this thread so far, rendered
    for the OUTER model, or "" when nothing was said (or no run is active).

    Same rendering as drain() on purpose — both are quoting the same people
    to a model — but this one is non-destructive and cumulative, so it can be
    read once at the end of a run after drain() has already emptied the queue
    several times over.
    """
    seen = _SEEN.get(int(thread_id))
    if not seen:
        return ""
    return "\n".join(f"[thread message from {author}]: {text}" for author, text in seen)

"""Tracked ledger of what people say in a run's thread while it runs.

A workflow claims its thread (classes/workflows.py), and on_message records
each message there instead of treating it as a new request. Every accepted
message is an EVENT with an explicit stage, so none is lost or counted as
handled just because something read it:

    received -> presented -> addressed -> resolved

  - received:  accepted from Discord, nothing has taken it yet.
  - presented: taken by the workflow.
  - addressed: someone said how it will be handled, but not that it is done.
  - resolved:  a specific outcome: done, answered, noted, declined,
               superseded or deferred.

Deliberately pure: stdlib only, no discord and no SDK imports, so every rule
here is unit testable without a Discord client. main.py's on_message does
the Discord half (one deliver call plus a reaction). Every mutating function
here is synchronous: the bot is one asyncio event loop, so a function with no
await in it is atomic with respect to every other coroutine, which is what
makes claim_run a real check-and-set.

A thread with a run claimed never starts a second one: the message goes to
the running job instead (main.py returns early). State lives in a module-level
dict, which is correct here: the bot is one process running one event loop,
and both sides live in it, so a restart loses it.

Stale owners: claim_run records the task that claimed the run, and get_run
drops an entry whose owner task has already finished. That closes the old
failure mode where a worker torn down without reaching its `finally` left the
thread registered forever, so every later message there got a 📨 reaction
and went nowhere.
"""
import asyncio
import re
import time
from dataclasses import dataclass, field

# Caps on one run's UNHANDLED input (received or presented, not yet
# addressed). Everything here is pasted into the nested model's context on
# every call, so an unbounded backlog would be both a context blowout and a
# way for a chatty thread to derail a run. Oversized input is REJECTED, never
# truncated: a clipped requirement is worse than a visible "too long".
MAX_PENDING_MESSAGES = 20
MAX_PENDING_CHARS = 4000
MAX_MESSAGE_CHARS = 1000
# Bound on the whole retained record, handled or not. Only RESOLVED events
# are ever compacted away (oldest first); unresolved guidance — including
# ADDRESSED ones, accepted but never marked done — is never silently
# evicted. When the record is at this bound and nothing in it is resolved,
# deliver() rejects new input (FULL) instead, so the context the coordinator
# re-sends on every call stays bounded either way.
MAX_RETAINED_EVENTS = 60
MAX_AUTHOR_CHARS = 40

RECEIVED = "received"
PRESENTED = "presented"
ADDRESSED = "addressed"
RESOLVED = "resolved"

# The decisions that can be recorded for a message, and the stage
# each one moves an event to. will_apply is the only one that is not an
# outcome: it is a promise, and the run's report keeps it apart from "done".
DECISIONS = {
    "will_apply": ADDRESSED,
    "done": RESOLVED,
    "answered": RESOLVED,
    "noted": RESOLVED,
    "declined": RESOLVED,
    "superseded": RESOLVED,
    "deferred": RESOLVED,
}

# deliver() results. main.py maps each to a reaction/reply, and metrics.py
# counts them, so these are also bounded metric label values.
ACCEPTED = "accepted"
DUPLICATE = "duplicate"
NO_RUN = "no_run"
EMPTY = "empty"
ATTACHMENT_ONLY = "attachment_only"
TOO_LONG = "too_long"
FULL = "full"

# Run lifecycle: open from the claim to release_run, then CLOSED. A workflow
# releases its claim the moment it ends, so a later message is no longer its.
SETUP = "setup"
RUNNING = "running"
CLOSED = "closed"


def _clean_author(name) -> str:
    """A display name is user-controlled text that ends up in a model prompt,
    so it is flattened to one short line with no brackets or colons — it must
    not be able to impersonate the "[thread message ...]" framing."""
    name = re.sub(r"[\[\]:\n\r`]", " ", str(name or "someone"))
    name = " ".join(name.split()) or "someone"
    return name[:MAX_AUTHOR_CHARS]


@dataclass
class ThreadEvent:
    seq: int
    message_id: int | None
    author_id: int | None
    author_name: str
    text: str
    is_requester: bool
    received_at: float
    reply_to: int | None = None
    has_attachments: bool = False
    stage: str = RECEIVED
    outcome: str | None = None
    # The reply it got.
    reply: str = ""
    addressed_at: float | None = None


@dataclass
class RunLedger:
    thread_id: int
    requester_id: int | None = None
    owner: object | None = None
    state: str = SETUP
    events: list[ThreadEvent] = field(default_factory=list)
    # seq of the newest accepted event (0 = none yet). Monotonic.
    revision: int = 0
    compacted: int = 0
    _changed: asyncio.Event | None = None

    # -- signalling ---------------------------------------------------------

    @property
    def changed(self) -> asyncio.Event:
        """Set whenever input is accepted; waiters clear it themselves."""
        if self._changed is None:
            self._changed = asyncio.Event()
        return self._changed

    # -- intake -------------------------------------------------------------

    def _seen_message(self, message_id) -> bool:
        if not message_id:
            # 0/None are not real ids (a forwarded message can
            # lack one), so they never take part in de-duplication.
            return False
        return any(e.message_id == message_id for e in self.events)

    def deliver(self, message_id, author_id, author_name, text,
                reply_to=None, has_attachments: bool = False, stop_word: bool = False) -> str:
        """Records one thread message. Returns one of the delivery results
        above so the caller acknowledges accurately rather than implying the
        run will see a message it never will. A stop word from the requester
        (`stop_word`, the caller's check) is never refused for a full inbox:
        others' chatter must not take away their way to stop the run."""
        if self.state == CLOSED:
            return NO_RUN
        message_id = int(message_id) if message_id else None
        if self._seen_message(message_id):
            return DUPLICATE
        text = (text or "").strip()
        if not text:
            return ATTACHMENT_ONLY if has_attachments else EMPTY
        if len(text) > MAX_MESSAGE_CHARS:
            return TOO_LONG
        event = ThreadEvent(
            seq=0,
            message_id=message_id,
            author_id=int(author_id) if author_id is not None else None,
            author_name=_clean_author(author_name),
            text=text,
            is_requester=(author_id is not None and self.requester_id is not None
                          and int(author_id) == int(self.requester_id)),
            received_at=time.monotonic(),
            reply_to=int(reply_to) if reply_to else None,
            has_attachments=bool(has_attachments),
        )
        pending = self.unaddressed()
        if not (stop_word and event.is_requester):
            if len(pending) >= MAX_PENDING_MESSAGES:
                return FULL
            if sum(len(e.text) for e in pending) + len(text) > MAX_PENDING_CHARS:
                return FULL
            if (len(self.events) >= MAX_RETAINED_EVENTS
                    and all(e.stage != RESOLVED for e in self.events)):
                return FULL
        self.revision += 1
        event.seq = self.revision
        self.events.append(event)
        self._compact()
        self.changed.set()
        return ACCEPTED

    def _compact(self) -> None:
        excess = len(self.events) - MAX_RETAINED_EVENTS
        if excess <= 0:
            return
        keep = []
        for e in self.events:
            if excess > 0 and e.stage == RESOLVED:
                excess -= 1
                self.compacted += 1
                continue
            keep.append(e)
        self.events[:] = keep

    # -- queries ------------------------------------------------------------

    def get(self, seq: int) -> ThreadEvent | None:
        return next((e for e in self.events if e.seq == seq), None)

    def unpresented(self) -> list[ThreadEvent]:
        return [e for e in self.events if e.stage == RECEIVED]

    def unaddressed(self) -> list[ThreadEvent]:
        """Received or presented: what still waits for an answer."""
        return [e for e in self.events if e.stage in (RECEIVED, PRESENTED)]

    # -- transitions ----------------------------------------------------------

    def mark_presented(self, seqs) -> list[ThreadEvent]:
        shown = []
        for seq in seqs:
            e = self.get(seq)
            if e is None or e.stage != RECEIVED:
                continue
            e.stage = PRESENTED
            shown.append(e)
        return shown

    def record_decision(self, seqs, decision: str, reply: str = ""):
        """Applies one decision. Returns (updated, unknown):
        unknown are ids that were never presented to this run, which the
        tool reports back instead of pretending they were handled."""
        stage = DECISIONS[decision]
        now = time.monotonic()
        updated, unknown = [], []
        for seq in seqs:
            e = self.get(seq)
            if e is None or e.stage == RECEIVED:
                unknown.append(seq)
                continue
            e.stage = stage
            e.outcome = decision
            if reply:
                e.reply = reply
            if e.addressed_at is None:
                e.addressed_at = now
            updated.append(e)
        return updated, unknown


# thread id -> the ledger of the run claimed in it. A thread is "active" iff
# it has an entry here. claim_run/release_run own the key's lifetime.
_RUNS: dict[int, RunLedger] = {}


def _current_task():
    try:
        return asyncio.current_task()
    except RuntimeError:
        return None


def claim_run(thread_id: int, requester_id: int | None = None) -> RunLedger | None:
    """Atomically claims this thread for a new run. Returns the run's ledger
    (which doubles as the ownership token for release_run), or None when a
    run is already claimed here — the caller then forwards into it."""
    thread_id = int(thread_id)
    if get_run(thread_id) is not None:
        return None
    ledger = RunLedger(thread_id=thread_id, requester_id=requester_id, owner=_current_task())
    _RUNS[thread_id] = ledger
    return ledger


def release_run(thread_id: int, ledger: RunLedger) -> bool:
    """Ends the claim. Only the ledger claim_run returned can release it, so a
    late release from a stale or forwarded caller cannot drop somebody
    else's live run."""
    thread_id = int(thread_id)
    if _RUNS.get(thread_id) is not ledger:
        return False
    ledger.state = CLOSED
    del _RUNS[thread_id]
    return True


def get_run(thread_id: int) -> RunLedger | None:
    """The ledger of the run claimed in this thread, or None. Drops an entry
    whose owner task has finished without releasing it."""
    thread_id = int(thread_id)
    ledger = _RUNS.get(thread_id)
    if ledger is None:
        return None
    owner = ledger.owner
    if owner is not None and getattr(owner, "done", lambda: False)():
        ledger.state = CLOSED
        del _RUNS[thread_id]
        return None
    return ledger


def is_run_active(thread_id: int) -> bool:
    """Whether a sandbox run is currently claimed in this thread."""
    return get_run(thread_id) is not None


def deliver(thread_id: int, message_id, author_id, author_name, text,
            reply_to=None, has_attachments: bool = False, stop_word: bool = False) -> str:
    """Records one thread message for the run claimed in this thread; see
    RunLedger.deliver for the possible results."""
    ledger = get_run(thread_id)
    if ledger is None:
        return NO_RUN
    return ledger.deliver(message_id, author_id, author_name, text,
                          reply_to=reply_to, has_attachments=has_attachments,
                          stop_word=stop_word)

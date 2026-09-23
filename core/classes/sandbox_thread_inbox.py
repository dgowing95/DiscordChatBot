"""Tracked ledger of what people say to a sandbox run while it happens.

A sandbox run happens inside a Discord thread (see classes/sandbox_agent.py's
ensure_sandbox_thread). While it runs, messages posted in that thread are
recorded here instead of going to the outer LLM, and the conversation
coordinator (classes/sandbox_conversation.py) shows them to the nested model
as user input at its next model call. That makes a run steerable ("actually
make it blue") without restarting it.

This used to be a destructive inbox: messages were drained into whichever
tool result happened to run next, and reading one counted as handling it.
Nothing recorded whether the model understood a request, or whether it ever
saw it at all — pending input was simply discarded when the run ended, and
the outer model was told "the sandbox received these and adapted" either way.
So every accepted message is now an EVENT with an explicit stage:

    received -> presented -> addressed -> resolved

  - received:  accepted from Discord, not yet shown to the model.
  - presented: included in a model call that completed.
  - addressed: the model said how it will handle it (decision "will_apply")
               but has not said the change is made. Acceptance is not
               evidence of completion, so this stays distinct from resolved.
  - resolved:  a specific outcome — done, answered, noted, declined,
               superseded or deferred.

An event the model kept ignoring past the gate's refusal cap is marked
UNACKNOWLEDGED instead (see sandbox_conversation.MAX_GATE_REFUSALS): it stays
unresolved and is reported as such, but stops blocking the run.

Deliberately pure: stdlib only, no discord and no SDK imports, so every rule
here is unit testable without a Discord client. main.py's on_message does
the Discord half (one deliver call plus a reaction), and
sandbox_conversation.py does the runner half. Every mutating function here is
synchronous: the bot is one asyncio event loop, so a function with no await
in it is atomic with respect to every other coroutine — which is what makes
claim_run a real check-and-set rather than the old check, await, register.

This is also the concurrency guard for snapshots. Two runs in one thread would
both persist to dcb:sandbox_snapshot:{thread_id} on teardown and the last one
to finish would silently clobber the other, so a thread with a run claimed
never starts a second one — the new request is forwarded into the running
sandbox instead (main.py returns early; run_code_sandbox has the same guard
for the paths that bypass on_message).

State lives in a module-level dict, which is correct here and not a shortcut:
the bot is one process running one event loop, and both the producer
(on_message) and the consumer (the nested run) live in it. What must outlive
the run (unapplied follow-ups, an unanswered question) is persisted by
run_code_sandbox through classes/sandbox_conversation_store.py; an abrupt
crash before that point loses it, since this ledger is process memory.

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
# are ever compacted away (oldest first); unresolved guidance is never
# silently evicted — the pending caps above reject new input instead.
MAX_RETAINED_EVENTS = 60
# Messages that arrive after the finalization boundary (see finalize()).
MAX_FOLLOW_UPS = 10
MAX_AUTHOR_CHARS = 40

RECEIVED = "received"
PRESENTED = "presented"
ADDRESSED = "addressed"
RESOLVED = "resolved"

# The decisions the model can record with respond_to_updates, and the stage
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
UNACKNOWLEDGED = "unacknowledged"

# deliver() results. main.py maps each to a reaction/reply, and metrics.py
# counts them, so these are also bounded metric label values.
ACCEPTED = "accepted"
DUPLICATE = "duplicate"
NO_RUN = "no_run"
EMPTY = "empty"
ATTACHMENT_ONLY = "attachment_only"
TOO_LONG = "too_long"
FULL = "full"
FINISHING = "finishing"

# Run lifecycle. SETUP covers the awaits between claim and the first model
# call (snapshot lookup, container start); input accepted then is presented
# at the first call. FINALIZING starts at the finalization boundary: artifact
# collection, persistence and Discord delivery are under way and nothing new
# can be applied. The claim is held through all of it.
SETUP = "setup"
RUNNING = "running"
FINALIZING = "finalizing"
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
    # The model's own words about it (respond_to_updates' reply).
    reply: str = ""
    # Position in the model's input list where this event was first shown.
    # The SDK rebuilds that list every call and does not keep what a filter
    # added, so the coordinator re-inserts every presented event at its
    # anchor on every call — see sandbox_conversation.
    anchor: int | None = None
    presented_at: float | None = None
    addressed_at: float | None = None
    # Responses (not tool calls) whose gated actions were refused over this
    # event; see sandbox_conversation.gate.
    gate_refusals: int = 0
    last_refused_call: int | None = None
    # The open question this may be answering (see OpenQuestion). A candidate
    # only: the model decides whether it is an answer, a correction or new.
    question_id: int | None = None


@dataclass
class OpenQuestion:
    qid: int
    text: str
    blocking: bool
    default: str
    # Events with a higher seq arrived after the question was asked.
    after_seq: int
    asked_at: float
    message_id: int | None = None
    waiting: bool = False
    timed_out: bool = False


@dataclass
class RunLedger:
    thread_id: int
    requester_id: int | None = None
    owner: object | None = None
    state: str = SETUP
    events: list[ThreadEvent] = field(default_factory=list)
    follow_ups: list[ThreadEvent] = field(default_factory=list)
    question: OpenQuestion | None = None
    # seq of the newest accepted event (0 = none yet). Monotonic.
    revision: int = 0
    # Bumped whenever the model accepts or completes a change (will_apply /
    # done), so attachments chosen before it can be recognised as outdated.
    change_revision: int = 0
    # Set while a gated tool (a shell command) is executing, so main.py can
    # say "received, I'll look at it before my next step" instead of leaving
    # a bare reaction while a long command runs.
    busy_since: float | None = None
    last_receipt_notice: float = 0.0
    compacted: int = 0
    _next_qid: int = 1
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
            # 0/None are not real ids (run_code_sandbox's forward path can
            # lack one), so they never take part in de-duplication.
            return False
        return any(e.message_id == message_id for e in self.events) or any(
            e.message_id == message_id for e in self.follow_ups)

    def deliver(self, message_id, author_id, author_name, text,
                reply_to=None, has_attachments: bool = False) -> str:
        """Records one thread message. Returns one of the delivery results
        above so the caller acknowledges accurately rather than implying the
        run will see a message it never will."""
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
        if self.state == FINALIZING:
            # Past the boundary nothing can be applied any more. Kept, and
            # persisted by run_code_sandbox, so the next run in this thread
            # is offered it — never silently dropped, never claimed handled.
            if len(self.follow_ups) >= MAX_FOLLOW_UPS:
                return FULL
            self.follow_ups.append(event)
            return FINISHING
        pending = self.unaddressed()
        if len(pending) >= MAX_PENDING_MESSAGES:
            return FULL
        if sum(len(e.text) for e in pending) + len(text) > MAX_PENDING_CHARS:
            return FULL
        self.revision += 1
        event.seq = self.revision
        q = self.question
        if q is not None and (event.is_requester or
                              (q.message_id and event.reply_to == q.message_id)):
            event.question_id = q.qid
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
        """Received or presented, and not given up on: what still needs the
        model's attention before it may do more work."""
        return [e for e in self.events
                if e.stage in (RECEIVED, PRESENTED) and e.outcome != UNACKNOWLEDGED]

    def unresolved(self) -> list[ThreadEvent]:
        return [e for e in self.events if e.stage != RESOLVED]

    # -- transitions ----------------------------------------------------------

    def mark_presented(self, seqs, anchor: int) -> list[ThreadEvent]:
        now = time.monotonic()
        shown = []
        for seq in seqs:
            e = self.get(seq)
            if e is None or e.stage != RECEIVED:
                continue
            e.stage = PRESENTED
            e.anchor = anchor
            e.presented_at = now
            shown.append(e)
        return shown

    def record_decision(self, seqs, decision: str, reply: str = ""):
        """Applies one respond_to_updates decision. Returns (updated, unknown):
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
        if updated and decision in ("will_apply", "done"):
            self.change_revision += 1
        q = self.question
        if q is not None and any(e.question_id == q.qid for e in updated):
            # The model has now looked at a candidate answer and said what it
            # makes of it; if it still needs to know, it can ask again.
            self.question = None
        return updated, unknown

    def give_up_on(self, event: ThreadEvent) -> None:
        """Stops gating on an event the model keeps ignoring. It stays
        unresolved — reported as never acknowledged — but no longer blocks."""
        event.outcome = UNACKNOWLEDGED

    def open_question(self, text: str, blocking: bool, default: str = "") -> OpenQuestion:
        q = OpenQuestion(qid=self._next_qid, text=text, blocking=bool(blocking),
                         default=default or "", after_seq=self.revision,
                         asked_at=time.monotonic())
        self._next_qid += 1
        self.question = q
        return q

    def close_question(self, qid: int) -> None:
        if self.question is not None and self.question.qid == qid:
            self.question = None

    def has_input_after(self, seq: int) -> bool:
        return any(e.seq > seq for e in self.events)

    def finalize(self) -> None:
        """The finalization boundary. From here on deliver() files messages
        as follow-ups for a later run instead of accepting them into this
        one — see FINISHING. Synchronous, so the caller's "anything left?"
        check and this commit cannot be split by an arriving message."""
        if self.state in (SETUP, RUNNING):
            self.state = FINALIZING

    # -- reporting ------------------------------------------------------------

    def outcomes(self) -> list[dict]:
        """What happened to each message, for the outer model and the thread's
        closing note. `status` is one of: done, answered, noted, declined,
        superseded, deferred (the model's own outcome), accepted (will_apply,
        completion not confirmed), unacknowledged (the model never responded),
        unseen (never reached a completed model call) or late (arrived after
        the finalization boundary)."""
        rows = []
        for e in self.events:
            if e.outcome == UNACKNOWLEDGED:
                status = "unacknowledged"
            elif e.stage == RECEIVED:
                status = "unseen"
            elif e.stage == PRESENTED:
                status = "unacknowledged"
            elif e.outcome == "will_apply":
                status = "accepted"
            else:
                status = e.outcome or "noted"
            rows.append({"author": e.author_name, "text": e.text,
                         "status": status, "reply": e.reply})
        for e in self.follow_ups:
            rows.append({"author": e.author_name, "text": e.text,
                         "status": "late", "reply": ""})
        return rows


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
            reply_to=None, has_attachments: bool = False) -> str:
    """Records one thread message for the run claimed in this thread; see
    RunLedger.deliver for the possible results."""
    ledger = get_run(thread_id)
    if ledger is None:
        return NO_RUN
    return ledger.deliver(message_id, author_id, author_name, text,
                          reply_to=reply_to, has_attachments=has_attachments)

"""Conversation coordinator for one sandbox run.

Sits between the thread ledger (classes/sandbox_thread_inbox.py, pure state)
and the nested SandboxAgent's runner, and is what makes steering ENFORCED
rather than hoped for. The old design appended thread messages to whichever
tool result ran next and relied on the model to act on them; nothing stopped
it from reading "make it blue", ignoring it, and delivering a red chart that
the outer model was then told reflected the change. Four mechanisms replace
that, and correctness depends on none of them being remembered by the model:

1. ONE delivery path, at model-call boundaries. input_filter (installed as
   RunConfig.call_model_input_filter) shows every message as its own
   user-role item, after completed tool-call/result pairs, instead of inside
   a tool's output where it competed with a page of logs. The SDK rebuilds
   the input list from original input + generated items on every call and
   does NOT keep what a filter added (verified against openai-agents 0.22
   with its ScriptedModel: see sandbox_conversation_tests), so the ledger is
   authoritative and every presented message is re-inserted at the position
   it was first shown — its `anchor` — on every call. That keeps it in
   chronological place, exactly once, including across a continuation,
   whose input is the previous run's to_input_list() and therefore has the
   same prefix. The anchor is a plain index: that prefix is stable because
   nothing in this setup rewrites earlier items (no compaction capability,
   and Memory contributes instructions only, never input items).

2. Presentation is committed AFTER the call succeeds. The filter runs before
   the model call, which can time out; marking input "presented" there would
   let the gate believe the model saw something it never did. The filter
   only stages, and on_llm_end (an AgentHooks on the SandboxAgent — the
   run's `hooks=` slot is already SandboxProgressHooks) commits it.

3. Side effects are gated. While a presented message has had no
   respond_to_updates, or a message arrived after the current response was
   generated, gated tools (new shell commands, attach_file, previews) return
   a structured refusal instead of executing. The SDK runs one response's
   tool calls concurrently, so gated tools are serialized under a lock and
   each checks the gate when it gets its turn: a message landing mid-batch
   blocks every later action in that batch. Conversation tools
   (respond_to_updates, ask_user, say_in_thread) and write_stdin (observing
   or stopping a running process) are never gated, and a respond_to_updates
   in the SAME response counts — acknowledging and acting in one response
   costs no refusal, which matters on a 10-turn budget. The gate cannot make
   the model obey; it guarantees the message was considered and answered
   visibly. Whether the change was actually made is what "done" versus
   "will_apply" is for, and the outer report keeps them apart.

4. Finishing is gated too. A final answer with unanswered input (or with an
   outdated attached file next to its replacement) gets a continuation in
   the same live session — bounded by MAX_CONTINUATIONS and by the run's
   single cumulative turn budget — before the finalization boundary is
   committed. See run_sandbox_task in sandbox_agent.py.

Known limitation, accepted: a message that arrives DURING a model call cannot
interrupt it. Cancelling and reissuing an in-flight call was judged not safe
to ship without proof (the plan's Phase 1 rule), so the response finishes and
its now-stale actions are refused by the gate instead. Response latency to a
mid-inference message is therefore one model call plus one turn.

The Discord side is limited to posting replies through the thread object
passed in; everything else here is testable with a stub thread.
"""
import asyncio
import json
import logging
import re
import time

from classes import sandbox_thread_inbox as inbox
from classes.metrics import (
    inc_sandbox_continuation,
    inc_sandbox_question,
    inc_sandbox_stale_action_blocked,
    inc_sandbox_unresolved_input,
    observe_sandbox_input_presentation,
    observe_sandbox_input_response,
)

logger = logging.getLogger(__name__)

# How many times a gated tool may be refused over one ignored message before
# the gate gives up on it (marking it unacknowledged — reported, not
# silently dropped). Without a cap a model that never calls
# respond_to_updates would spend its whole turn budget being refused.
MAX_GATE_REFUSALS = 2
# Extra runner passes after a final answer that left input unanswered. Each
# costs at least one turn out of the same SANDBOX_MAX_TURNS budget.
MAX_CONTINUATIONS = 2
# A continuation is only worth starting with room for the acknowledgement
# and a new final message.
MIN_CONTINUATION_TURNS = 2
REPLY_SEND_ATTEMPTS = 2

RESPOND_TOOL_NAME = "respond_to_updates"
# Every gate refusal starts with this, so sandbox_progress can show a held
# command as held rather than as a command still running.
GATE_REFUSAL_PREFIX = "Not run"

# Loose spellings a small local model produces, mapped onto DECISIONS.
_DECISION_ALIASES = {
    "will apply": "will_apply", "willapply": "will_apply", "apply": "will_apply",
    "applying": "will_apply", "accept": "will_apply", "accepted": "will_apply",
    "will_do": "will_apply", "ack": "will_apply", "acknowledged": "will_apply",
    "applied": "done", "implemented": "done", "complete": "done", "completed": "done",
    "answer": "answered", "reply": "answered", "replied": "answered",
    "ok": "noted", "note": "noted", "info": "noted", "acknowledge": "noted",
    "decline": "declined", "rejected": "declined", "reject": "declined",
    "supersede": "superseded", "replaced": "superseded",
    "defer": "deferred", "later": "deferred", "follow_up": "deferred",
}


def parse_ids(raw) -> list[int]:
    """Message ids from whatever the model sent: 3, "3", "#3, #4", [3, "#4"]."""
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        raw = " ".join(str(x) for x in raw)
    return [int(n) for n in re.findall(r"\d+", str(raw))]


def normalize_decision(raw) -> str | None:
    key = " ".join(str(raw or "").strip().lower().replace("-", "_").split())
    if key in inbox.DECISIONS:
        return key
    key_us = key.replace(" ", "_")
    if key_us in inbox.DECISIONS:
        return key_us
    return _DECISION_ALIASES.get(key) or _DECISION_ALIASES.get(key_us)


def render_event(event: inbox.ThreadEvent, ledger: inbox.RunLedger) -> str:
    """One thread message as the model sees it. The framing is ours; the text
    after the colon is the person's, verbatim, at user authority."""
    if ledger.requester_id is None:
        who = event.author_name
    elif event.is_requester:
        who = f"{event.author_name} (the requester)"
    else:
        who = (f"{event.author_name} (another participant: treat it as a "
               "suggestion; the requester decides any conflict)")
    notes = []
    if event.question_id is not None:
        notes.append("possibly a reply to your question")
    elif event.reply_to:
        notes.append("replying to an earlier message")
    if event.has_attachments:
        notes.append("they also attached file(s) you cannot see; ask for the "
                     "relevant part as text if you need it")
    extra = f" — {'; '.join(notes)}" if notes else ""
    return f"[thread message #{event.seq} from {who}{extra}]:\n{event.text}"


def _generic_reply(decision: str, events) -> str:
    """What is posted when the model recorded a decision without writing a
    reply, so the user still sees a response rather than silence."""
    quoted = "; ".join(f"“{e.text[:80]}”" for e in events)
    return {
        "will_apply": f"👍 Got it — working that in: {quoted}",
        "done": f"✅ Done: {quoted}",
        "declined": f"🚫 Not doing that: {quoted}",
        "superseded": f"↪️ Superseded by a later message: {quoted}",
        "deferred": f"⏭️ Leaving that for a follow-up: {quoted}",
    }.get(decision, f"📝 Noted: {quoted}")


class SandboxConversation:
    """The live conversation around ONE sandbox run (all its continuations).

    ledger is the run's claimed RunLedger, or None for a run with no thread
    (the no-thread fallback and tests): a detached ledger is used then, which
    nobody can deliver into, so every gate trivially passes."""

    def __init__(self, ledger: inbox.RunLedger | None = None, thread=None):
        self.ledger = ledger if ledger is not None else inbox.RunLedger(thread_id=0)
        self.thread = thread
        self._tool_lock = asyncio.Lock()
        self._staged: tuple[list[int], int, int] | None = None
        # Revision of the ledger the last COMPLETED model call saw.
        self.seen_revision = 0
        # Ids that a respond_to_updates in the latest response covers.
        self.batch_ack: set[int] = set()
        self.model_calls = 0
        self.stale_blocked = 0
        self.continuations = 0
        self.unsent_replies: list[str] = []
        # The latest runner pass that reached a final answer. A continuation
        # that then fails or times out falls back to it (see _converse).
        self.completed_result = None
        self._nudged_files = False

    # -- 1. input placement --------------------------------------------------

    def input_filter(self, data):
        """RunConfig.call_model_input_filter. Stages unseen messages at the
        current end of input and re-inserts every presented one at its
        anchor. Never raises: the SDK fails the whole run on a filter error,
        and steering is not worth a run."""
        from agents.run_config import ModelInputData

        instructions = data.model_data.instructions
        items = list(data.model_data.input)
        try:
            anchor = len(items)
            staged = self.ledger.unpresented()
            self._staged = ([e.seq for e in staged], anchor, self.ledger.revision)
            placed: dict[int, list[inbox.ThreadEvent]] = {}
            for e in self.ledger.events:
                if e.stage != inbox.RECEIVED and e.anchor is not None:
                    placed.setdefault(min(e.anchor, anchor), []).append(e)
            for e in staged:
                placed.setdefault(anchor, []).append(e)
            if not placed:
                return ModelInputData(input=items, instructions=instructions)
            out = []
            for i in range(len(items) + 1):
                for e in sorted(placed.get(i, ()), key=lambda ev: ev.seq):
                    out.append({"role": "user", "content": render_event(e, self.ledger)})
                if i < len(items):
                    out.append(items[i])
            return ModelInputData(input=out, instructions=instructions)
        except Exception as e:
            logger.warning(f"Sandbox conversation: input filter failed, sending input unchanged: {e}")
            return ModelInputData(input=items, instructions=instructions)

    # -- 2. commit after a completed call --------------------------------------

    def on_model_response(self, response) -> None:
        """Called from the agent's on_llm_end: the call completed, so what
        was staged was really seen."""
        self.model_calls += 1
        if self._staged is not None:
            seqs, anchor, revision = self._staged
            self._staged = None
            for e in self.ledger.mark_presented(seqs, anchor):
                observe_sandbox_input_presentation(e.presented_at - e.received_at)
            self.seen_revision = revision
        acked: set[int] = set()
        for item in getattr(response, "output", None) or []:
            if getattr(item, "type", None) != "function_call":
                continue
            if getattr(item, "name", None) != RESPOND_TOOL_NAME:
                continue
            try:
                args = json.loads(getattr(item, "arguments", "") or "{}")
            except (TypeError, ValueError):
                continue
            ids = parse_ids(args.get("message_ids"))
            if not ids:
                # No ids means "all of them" in respond(), so it covers
                # everything the model has been shown.
                ids = [e.seq for e in self.ledger.unaddressed() if e.stage == inbox.PRESENTED]
            acked.update(ids)
        self.batch_ack = acked

    # -- 3. gating -----------------------------------------------------------

    def has_new_input(self) -> bool:
        """Whether a message is waiting that the model has not been shown."""
        return bool(self.ledger.unpresented())

    def gate(self, tool_name: str) -> str | None:
        """None if a gated tool may run now, else the refusal to return."""
        led = self.ledger
        q = led.question
        if q is not None and q.blocking and q.waiting:
            self._blocked()
            return (f"{GATE_REFUSAL_PREFIX}: you are waiting for the answer to your blocking "
                    "question. Nothing else runs until it is answered or times out.")
        unseen = [e for e in led.unaddressed() if e.stage == inbox.RECEIVED]
        if unseen:
            self._blocked()
            return (f"{GATE_REFUSAL_PREFIX} ({tool_name}): {len(unseen)} new thread message(s) arrived "
                    "after you chose this action. You will see them next — respond to "
                    "them with respond_to_updates, then redo this if it still makes sense.")
        ignored = []
        for e in led.unaddressed():
            if e.stage != inbox.PRESENTED or e.seq in self.batch_ack:
                continue
            # Counted once per model RESPONSE, not per refused call: counted
            # per call, one response with three gated tools would exhaust
            # the cap itself and run the third tool with nothing answered.
            if e.last_refused_call != self.model_calls:
                e.last_refused_call = self.model_calls
                e.gate_refusals += 1
            if e.gate_refusals > MAX_GATE_REFUSALS:
                logger.info(f"Sandbox conversation: giving up on message #{e.seq} "
                            f"after {MAX_GATE_REFUSALS} refusals")
                led.give_up_on(e)
                continue
            ignored.append(e)
        if ignored:
            self._blocked()
            ids = ", ".join(f"#{e.seq}" for e in ignored)
            return (f"{GATE_REFUSAL_PREFIX} ({tool_name}): thread message(s) {ids} need a response "
                    "first. Call respond_to_updates for them (in the same response as "
                    "this tool is fine), then carry on.")
        return None

    def _blocked(self) -> None:
        self.stale_blocked += 1
        inc_sandbox_stale_action_blocked()

    async def run_gated(self, tool_name: str, invoke, *, busy: bool = False):
        """Runs one gated tool invocation: serialized with the others, and
        refused instead of executed when the gate says so."""
        async with self._tool_lock:
            refusal = self.gate(tool_name)
            if refusal is not None:
                return refusal
            if busy:
                self.ledger.busy_since = time.monotonic()
            try:
                return await invoke()
            finally:
                if busy:
                    self.ledger.busy_since = None

    # -- acknowledgement --------------------------------------------------------

    async def _post(self, text: str) -> bool:
        if self.thread is None:
            return False
        for attempt in range(REPLY_SEND_ATTEMPTS):
            try:
                await self.thread.send(text[:1900])
                return True
            except Exception as e:
                logger.warning(f"Sandbox conversation: reply send failed "
                               f"(attempt {attempt + 1}/{REPLY_SEND_ATTEMPTS}): {e}")
        return False

    async def respond(self, message_ids, decision, reply: str = "") -> str:
        """respond_to_updates' implementation."""
        choice = normalize_decision(decision)
        if choice is None:
            return (f"Unknown decision {decision!r}. Use one of: "
                    f"{', '.join(inbox.DECISIONS)}. Nothing was recorded.")
        led = self.ledger
        ids = parse_ids(message_ids)
        if not ids:
            # No ids: everything shown and still open — plus, for "done",
            # what was accepted earlier and is now finished.
            stages = (inbox.PRESENTED, inbox.ADDRESSED) if choice == "done" else (inbox.PRESENTED,)
            ids = [e.seq for e in led.events
                   if e.stage in stages and e.outcome != inbox.UNACKNOWLEDGED]
            if not ids:
                return "There are no thread messages waiting for a response."
        updated, unknown = led.record_decision(ids, choice, (reply or "").strip())
        for e in updated:
            observe_sandbox_input_response(e.addressed_at - e.received_at)
        parts = []
        if updated:
            text = (reply or "").strip() or _generic_reply(choice, updated)
            if not text.startswith(("🐳", "👍", "✅", "🚫", "↪", "⏭", "📝")):
                text = f"🐳 {text}"
            if await self._post(text):
                parts.append(f"Recorded '{choice}' for "
                             f"{', '.join(f'#{e.seq}' for e in updated)} and posted your reply.")
            else:
                self.unsent_replies.append(text)
                parts.append(f"Recorded '{choice}' for "
                             f"{', '.join(f'#{e.seq}' for e in updated)}; the reply could not be "
                             "posted now and will be included in the closing message.")
            if choice == "will_apply":
                parts.append("Make the change, then mark it 'done' — or say in your "
                             "final message if you could not.")
        if unknown:
            parts.append(f"No message {', '.join(f'#{n}' for n in unknown)} has been "
                         "shown to you, so nothing was recorded for it.")
        left = [e for e in led.unaddressed() if e.stage == inbox.PRESENTED]
        if left:
            parts.append(f"Still waiting for a response: {', '.join(f'#{e.seq}' for e in left)}.")
        return " ".join(parts)

    # -- questions ---------------------------------------------------------------

    async def ask(self, question: str, *, blocking: bool, choices=None, default: str = "",
                  timeout: float) -> str:
        """ask_user's implementation. The question is registered BEFORE it is
        sent, so a reply that beats the send is still recognised; replies
        arrive through the ledger like any other message, exactly once."""
        led = self.ledger
        if self.thread is None:
            return "No interactive thread is available right now — proceed using your best judgement."
        question = (question or "").strip()
        if not question:
            return "The question was empty, so nothing was asked."
        q = led.question
        if q is not None and not q.timed_out:
            return (f"You already have an open question ({q.text[:120]!r}). Work with "
                    "that one first; a reply will arrive as a thread message.")
        mode = "blocking" if blocking else "optional"
        q = led.open_question(question, blocking=blocking, default=default)
        lines = [f"🤖 {question}"]
        opts = [str(c).strip() for c in (choices or []) if str(c).strip()][:5]
        if opts:
            lines.append("Options: " + " · ".join(opts))
        if default:
            lines.append(f"If I don't hear back, I'll go with: {default}")
        lines.append("-# Reply in this thread — no @mention needed."
                     + ("" if blocking else " I'll keep working meanwhile."))
        try:
            sent = await self.thread.send("\n".join(lines)[:1990])
            q.message_id = getattr(sent, "id", None)
        except Exception as e:
            logger.warning(f"Sandbox ask_user: failed to send question: {e}")
            led.close_question(q.qid)
            inc_sandbox_question(mode, "send_failed")
            return ("Could not reach the user (failed to send the question) — proceed "
                    "using your best judgement.")
        # A reply can land while the send is in flight; it was already tagged
        # (deliver() matches on the requester while the question is open).
        if not blocking:
            inc_sandbox_question(mode, "posted")
            return ("Question posted. Keep working on anything that does not depend on "
                    "it; a reply arrives as a thread message. Before the step it affects "
                    "(or your final message), use the reply"
                    + (f" or your stated default ({default})." if default else
                       " or state the assumption you made."))
        q.waiting = True
        try:
            got = await self.wait_for_input(q.after_seq, timeout)
        finally:
            q.waiting = False
        if got:
            inc_sandbox_question(mode, "replied")
            return ("New thread input arrived; you will see it next. Check whether it "
                    "answers your question — it may be a correction or a new request "
                    "instead — and call respond_to_updates for it.")
        q.timed_out = True
        inc_sandbox_question(mode, "timeout")
        if default:
            return (f"No reply within {int(timeout)}s. Go with your stated default "
                    f"({default}) and say so in your final message. A late reply will "
                    "still reach you as a thread message.")
        return (f"No reply within {int(timeout)}s. If you cannot continue without this, "
                "stop and say plainly in your final message that the task is blocked on "
                "it; otherwise make the most reasonable assumption and state it.")

    async def wait_for_input(self, after_seq: int, timeout: float) -> bool:
        """Waits until a message newer than after_seq is accepted, or timeout."""
        led = self.ledger
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            if led.has_input_after(after_seq):
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            led.changed.clear()
            if led.has_input_after(after_seq):
                return True
            try:
                await asyncio.wait_for(led.changed.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                return led.has_input_after(after_seq)

    # -- 4. finishing -------------------------------------------------------------

    def continuation_nudge(self, deliverables) -> str | None:
        """What to tell the model before letting it finish, or None when it
        may. Covers input it has not answered (including anything that
        arrived while it wrote the final answer) and, once per run, files
        attached before an accepted change sitting next to ones attached
        after it — both would be delivered."""
        parts = []
        waiting = self.ledger.unaddressed()
        if waiting:
            ids = ", ".join(f"#{e.seq}" for e in waiting)
            parts.append(
                f"Before you finish: thread message(s) {ids} have not been answered. "
                "Call respond_to_updates for them, make any change they ask for, then "
                "write your final message again.")
        if not self._nudged_files and deliverables:
            rev = self.ledger.change_revision
            old = [d for d in deliverables if d.get("change_revision", 0) < rev]
            new = [d for d in deliverables if d.get("change_revision", 0) >= rev]
            if rev and old and new:
                self._nudged_files = True
                parts.append(
                    f"You attached {', '.join(d['path'] for d in old)} before accepting a "
                    f"change and {', '.join(d['path'] for d in new)} after it, so both "
                    "would be sent. Remove any outdated file with "
                    "attach_file(path, remove=true) — or attach it again if it is still "
                    "right — then write your final message again.")
        return "\n\n".join(parts) or None

    def note_continuation(self) -> None:
        self.continuations += 1
        inc_sandbox_continuation()

    def finalize(self) -> None:
        """Commits the finalization boundary (see RunLedger.finalize)."""
        self.ledger.finalize()



def record_exit_metrics(ledger: inbox.RunLedger) -> None:
    """Counts what a run left unresolved. Called when the claim is released,
    after delivery, so messages that arrived as it finished are included."""
    for row in ledger.outcomes():
        if row["status"] in ("accepted", "unacknowledged", "unseen", "late"):
            inc_sandbox_unresolved_input(row["status"])


def make_agent_hooks(conversation: SandboxConversation):
    """The AgentHooks that commits presentation after each completed call."""
    from agents import AgentHooks

    class _ConversationHooks(AgentHooks):
        async def on_llm_end(self, context, agent, response):
            try:
                conversation.on_model_response(response)
            except Exception as e:
                logger.warning(f"Sandbox conversation: could not record a model response: {e}")

    return _ConversationHooks()

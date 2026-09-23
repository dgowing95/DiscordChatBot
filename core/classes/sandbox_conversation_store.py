"""Per-thread conversation record for sandbox runs, kept in Redis.

The workspace snapshot (classes/sandbox_snapshot_store.py) carries a
thread's FILES from one run to the next. This carries what was SAID that
still matters: messages that arrived too late to be applied, an unanswered
question, requests the sandbox accepted but never confirmed finished, and
how the run ended. Without it, "make it green too" posted as a run was
finishing was acknowledged with "not applied, ask again" and then forgotten.

Kept separate from the snapshot on purpose: either can fail without the
other, and each is reported independently. Nothing here implies the
workspace can be resumed — the resume preamble speaks only about messages.

Same thread scope as the snapshot (the key is the thread id) and the same
TTL, so the two expire together. Small and versioned: a record from an
unknown version is ignored rather than misread. Recovered messages are only
ever OFFERED to the next run as context; nothing is executed automatically.
"""
import json
import logging

from classes.redis_client import text_client
from classes.sandbox_snapshot_store import sandbox_snapshot_ttl_seconds

logger = logging.getLogger(__name__)

RECORD_VERSION = 1
MAX_RECORD_BYTES = 16_000
MAX_RECORD_ITEMS = 10
MAX_ITEM_CHARS = 500

# Ledger statuses that mean "said, but not (confirmed) applied".
_CARRY_STATUSES = {"late", "unseen", "unacknowledged", "accepted", "deferred"}


def _key(thread_id) -> str:
    return f"dcb:sandbox_conversation:{thread_id}"


def build_record(outcomes: list[dict], *, outcome: str, open_question: str | None = None) -> dict:
    """The record to save at the end of a run. Pure.

    outcomes is RunLedger.outcomes(); outcome is "ok" or the run's error."""
    carry = []
    for row in outcomes or []:
        if row.get("status") not in _CARRY_STATUSES:
            continue
        carry.append({
            "author": str(row.get("author", "someone"))[:40],
            "text": str(row.get("text", ""))[:MAX_ITEM_CHARS],
            "status": row.get("status"),
        })
    return {
        "version": RECORD_VERSION,
        "outcome": outcome,
        "open_question": (open_question or "")[:MAX_ITEM_CHARS] or None,
        "carry": carry[-MAX_RECORD_ITEMS:],
    }


def resume_preamble(record: dict | None) -> str:
    """What the next run in the thread is told about the previous one's loose
    ends, or "" when there are none. Pure. Worded as context, not orders:
    the new request is what the user asked for now."""
    if not record or record.get("version") != RECORD_VERSION:
        return ""
    lines = []
    labels = {
        "late": "arrived as it was finishing, not applied",
        "unseen": "arrived too late to be seen, not applied",
        "unacknowledged": "never answered, may not be applied",
        "accepted": "accepted, not confirmed finished",
        "deferred": "left for a follow-up",
    }
    for item in record.get("carry") or []:
        label = labels.get(item.get("status"), "not applied")
        lines.append(f"- {item.get('author', 'someone')} ({label}): {item.get('text', '')}")
    question = record.get("open_question")
    if question:
        lines.append(f"- Your unanswered question: {question}")
    if not lines:
        return ""
    return ("Loose ends from the previous run in this thread — check whether each "
            "still applies to the request below, and say what you did about them:\n"
            + "\n".join(lines) + "\n\n")


class SandboxConversationStore:
    """Load/save one thread's record. Every call is best-effort for the
    caller: failures raise, and run_code_sandbox logs and carries on."""

    @property
    def redis(self):
        return text_client()

    async def load(self, thread_id) -> dict | None:
        raw = await self.redis.get(_key(thread_id))
        if not raw:
            return None
        try:
            record = json.loads(raw)
        except (TypeError, ValueError):
            logger.warning(f"Sandbox conversation record for {thread_id} is not valid JSON; ignoring it")
            return None
        return record if isinstance(record, dict) else None

    async def save(self, thread_id, record: dict) -> None:
        raw = json.dumps(record, ensure_ascii=False)
        while len(raw.encode("utf-8")) > MAX_RECORD_BYTES and record.get("carry"):
            record = dict(record, carry=record["carry"][1:])
            raw = json.dumps(record, ensure_ascii=False)
        await self.redis.set(_key(thread_id), raw, ex=sandbox_snapshot_ttl_seconds())

    async def delete(self, thread_id) -> None:
        await self.redis.delete(_key(thread_id))

"""Which Discord messages go into a prompt: sliding vs anchored history.

Pure module (stdlib only), following the response_filter.py / message_queue.py
pattern: MessageHandler does the Discord fetches and formatting, but every
decision about WHICH messages to include lives here so it can be unit tested
without discord, Redis or an LLM.

Why anchored history exists
---------------------------
llama.cpp only skips prefill for the part of a prompt that is identical to a
cached one. The sliding window (newest N messages) drops the oldest message
every time a new one arrives, so the prompt's start changes on every call and
the whole history is prefilled again - the ~10 s median prefill seen in prod.

Anchored mode keeps the start of the window fixed and lets it grow:

    sliding:   A B C D -> B C D E -> C D E F
    anchored:  A B C D -> A B C D E -> A B C D E F   ...refresh ->  D E F G

and moves the anchor forward ("refresh") once enough new messages have
arrived, so the window stays bounded. Between refreshes each prompt is the
previous one plus a tail, which is what the server's cache can reuse.

State
-----
Per (guild, channel) the store keeps two ints: an EXCLUSIVE lower bound for
the window (`after_id`, passed straight to channel.history(after=...), which
excludes its argument) and the trigger id at the last refresh. Threads have
their own channel id, so a thread's window is independent of its parent's.

No message content is cached. Every build re-reads the window from Discord,
so edits and deletions are always current without any event handling - an
edit simply changes the prefix for that one call. The store only loses
optimisation state on eviction or restart (the next build is a cold refresh),
never correctness. Entries are fixed-size (two ints), so the channel-count cap
also bounds the bytes retained.

Concurrency
-----------
Builds for one channel run under that channel's lock (message_queue), so a
read-decide-commit never interleaves with another for the same key. Workers
can still pick messages up out of order, so a trigger OLDER than the last
refresh gets its own sliding snapshot and never commits: the anchor only ever
moves forward.
"""

import logging
import math
import os
from collections import OrderedDict
from dataclasses import dataclass

logger = logging.getLogger(__name__)

SLIDING = "sliding"
ANCHORED = "anchored"
MODES = (SLIDING, ANCHORED)

DEFAULT_HISTORY_LIMIT = 5
DEFAULT_REFRESH_MESSAGES = 10
# Tokens held back from the per-slot context for everything that is not
# channel history: the system prompt, tool schemas, tool calls/results added
# during the run, and the generated answer (reasoning included).
DEFAULT_RESERVE_TOKENS = 12000

# Channels remembered at once (LRU). Each entry is two ints.
MAX_CHANNELS = 4096

# Rough token estimate. ~3 characters per token is deliberately pessimistic
# for English (closer to 4), so the budget errs toward trimming a little early
# rather than overrunning the slot. Images cost a fixed amount: the real cost
# depends on the model's vision encoder and the image size.
CHARS_PER_TOKEN = 3
IMAGE_TOKENS = 1500
# Chat-template wrapping per message (role markers etc.).
MESSAGE_OVERHEAD_TOKENS = 8

# Decision kinds.
KEEP = "keep"          # reuse the anchored window as fetched
REFRESH = "refresh"    # take a fresh newest-N window and commit it as the anchor
SNAPSHOT = "snapshot"  # take a fresh newest-N window WITHOUT committing

# Refresh reasons (the discord_bot_history_refreshes_total label; bounded).
REASON_COLD = "cold"
REASON_THRESHOLD = "threshold"
REASON_BURST = "burst"
REASON_RESET = "reset"
REASON_TOKEN_BUDGET = "token_budget"
REASON_STALE = "stale_trigger"


def _env_positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def history_mode() -> str:
    """MSG_HISTORY_MODE: `sliding` (default) or `anchored`."""
    raw = os.environ.get("MSG_HISTORY_MODE", "").strip().lower()
    if not raw:
        return SLIDING
    if raw not in MODES:
        logger.warning(f"Unknown MSG_HISTORY_MODE {raw!r}; using {SLIDING!r}")
        return SLIDING
    return raw


def history_limit() -> int:
    """MSG_HISTORY_LIMIT: messages in a (refreshed) window, trigger included."""
    return _env_positive_int("MSG_HISTORY_LIMIT", DEFAULT_HISTORY_LIMIT)


def refresh_messages() -> int:
    """MSG_HISTORY_REFRESH_MESSAGES: new channel messages (bot replies
    included) since the last refresh that trigger the next one."""
    return _env_positive_int("MSG_HISTORY_REFRESH_MESSAGES", DEFAULT_REFRESH_MESSAGES)


def reserve_tokens() -> int:
    """MSG_HISTORY_RESERVE_TOKENS: per-slot tokens kept free for non-history use."""
    return _env_positive_int("MSG_HISTORY_RESERVE_TOKENS", DEFAULT_RESERVE_TOKENS)


def max_window(limit: int, refresh_every: int) -> int:
    """Most PRECEDING messages (trigger excluded) an anchored window can hold.

    A refresh keeps up to limit-1 preceding messages plus its trigger R. Later
    builds add the messages after R; once those plus the new trigger reach
    refresh_every, the build refreshes instead. So a kept window has at most
    (limit-1) + 1 + (refresh_every-2) preceding messages.
    """
    return max(limit + refresh_every - 2, 0)


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnchorState:
    after_id: int            # exclusive lower bound of the window
    refresh_trigger_id: int  # trigger of the build that last refreshed


@dataclass(frozen=True)
class Decision:
    kind: str
    reason: str = ""


def initial_decision(state: AnchorState | None, trigger_id: int) -> Decision | None:
    """What can be decided before fetching anything, or None to go fetch."""
    if state is None:
        return Decision(REFRESH, REASON_COLD)
    if trigger_id < state.refresh_trigger_id:
        # Picked up after a newer message already moved the anchor: build a
        # private window rather than rewinding shared state.
        return Decision(SNAPSHOT, REASON_STALE)
    return None


def window_decision(state: AnchorState, trigger_id: int, fetched: list[tuple[int, bool]],
                    limit: int, refresh_every: int) -> Decision:
    """Keep or refresh the anchored window.

    `fetched` is [(message_id, is_reset)] oldest first: the messages after
    state.after_id and before the trigger, fetched with a cap of
    max_window()+1 so an overflow is visible.
    """
    if len(fetched) > max_window(limit, refresh_every):
        return Decision(REFRESH, REASON_BURST)
    if any(is_reset for _, is_reset in fetched):
        return Decision(REFRESH, REASON_RESET)
    # Channel messages since the last refresh, counting this trigger.
    new = sum(1 for mid, _ in fetched if mid > state.refresh_trigger_id) + 1
    if new >= refresh_every:
        return Decision(REFRESH, REASON_THRESHOLD)
    return Decision(KEEP)


def state_after_refresh(included_ids: list[int], trigger_id: int) -> AnchorState:
    """The state to commit for a refreshed window of `included_ids`.

    The bound is one below the oldest included message; with nothing included
    (fresh channel, or a reset right before the trigger) it is one below the
    trigger, so the next build starts from the trigger and excludes whatever
    lies before it - a reset included.
    """
    oldest = min(included_ids) if included_ids else trigger_id
    return AnchorState(after_id=oldest - 1, refresh_trigger_id=trigger_id)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class HistoryStore:
    """Process-local LRU of AnchorState per (guild_id, channel_id)."""

    def __init__(self, max_channels: int = MAX_CHANNELS):
        self.max_channels = max_channels
        self._states: OrderedDict[tuple, AnchorState] = OrderedDict()

    def get(self, key) -> AnchorState | None:
        state = self._states.get(key)
        if state is not None:
            self._states.move_to_end(key)
        return state

    def commit(self, key, state: AnchorState) -> bool:
        """Store `state` unless it would move the anchor backwards.

        Returns True when stored. A state whose refresh trigger is not newer
        than the current one is refused, so an out-of-order build can never
        rewind a channel.
        """
        current = self._states.get(key)
        if current is not None and state.refresh_trigger_id <= current.refresh_trigger_id:
            return False
        self._states[key] = state
        self._states.move_to_end(key)
        while len(self._states) > self.max_channels:
            self._states.popitem(last=False)
        return True

    def clear(self) -> None:
        self._states.clear()

    def __len__(self) -> int:
        return len(self._states)


_store = HistoryStore()


def store() -> HistoryStore:
    """The process-wide store MessageHandler uses."""
    return _store


# ---------------------------------------------------------------------------
# Token budget
# ---------------------------------------------------------------------------


def estimate_tokens(entries: list[dict]) -> int:
    """Rough prompt tokens for chat entries ({'role', 'content'})."""
    total = 0
    for entry in entries:
        total += MESSAGE_OVERHEAD_TOKENS
        content = entry.get("content")
        if isinstance(content, str):
            total += math.ceil(len(content) / CHARS_PER_TOKEN)
            continue
        for part in content or []:
            if part.get("type") == "input_image":
                total += IMAGE_TOKENS
            else:
                total += math.ceil(len(str(part.get("text", ""))) / CHARS_PER_TOKEN)
    return total


def history_budget(slot_context: int | None, reserve: int) -> int | None:
    """Tokens available for history + trigger, or None when capacity is unknown.

    A reserve that would eat the whole slot (a small dev server with the prod
    default) falls back to half the slot rather than trimming all history.
    """
    if not slot_context or slot_context <= 0:
        return None
    if reserve >= slot_context:
        return slot_context // 2
    return slot_context - reserve


def trim_count(group_tokens: list[int], fixed_tokens: int, budget: int | None) -> int:
    """How many of the OLDEST groups to drop to fit `budget`.

    `group_tokens` is oldest first; `fixed_tokens` is what can never be dropped
    (the trigger and any status hint). No budget means no trimming. If the fixed
    part alone is over, every group goes and the prompt is sent anyway - the
    request itself is never cut.
    """
    if budget is None:
        return 0
    total = fixed_tokens + sum(group_tokens)
    drop = 0
    while total > budget and drop < len(group_tokens):
        total -= group_tokens[drop]
        drop += 1
    return drop

"""Message queue sizing, worker count, per-channel locks and the in-flight
task registry (used by main.py / message_handler.py).

Pure module (stdlib only), following the response_filter.py / metrics.py
pattern: main.py wires these into the discord client and the queue workers,
but the sizing/locking/registry logic lives here so it can be unit tested
without the discord/agents SDKs.

Concurrency model (see process_messages() / on_message() in main.py and
MessageHandler.handle_message() in message_handler.py):
  - N worker tasks (worker_count(), WORKER_COUNT env, default 2) share one
    bounded asyncio.Queue (queue_max_size(), QUEUE_MAX_SIZE env, default 10).
    A message stuck in a slow tool (sandbox / image generation) therefore
    pins only ONE worker; the others keep handling other channels.
  - one asyncio.Lock per channel id (get_channel_lock) guards only the two
    FAST phases of a handle: prompt building (build_messages) and the
    chunked send. This keeps same-channel chunk sends from interleaving and
    keeps every channel.history() snapshot free of half-sent replies, while
    the slow middle phase (LLM run + tool calls) runs UNLOCKED — so a free
    worker can answer a second message in the same channel while the first
    is stuck in a 10-minute sandbox. The lock is never held across an LLM
    or tool await, so no deadlock is possible.
  - the in-flight task registry (register_task_run / in_flight_hint) lets a
    newer message's prompt know that an older one is still being processed:
    MessageHandler appends the hint line to the prompt it builds. Slow tools
    register via ToolMetricsHooks in text_llm_handler.py.
  - when the queue is full, on_message() DROPS the message (counted in
    metrics; a mention gets a short "busy" reply) — the queue stays bounded
    so stale messages are never answered against stale history while a slow
    tool holds the bot up.
"""

import asyncio
import os
import sys
import time

DEFAULT_WORKER_COUNT = 2
DEFAULT_QUEUE_MAX_SIZE = 10

# The tools that can pin a worker for a long time (sandbox up to
# SANDBOX_TIMEOUT, image gen up to IMAGE_GEN_TIMEOUT + queue wait). Names
# must match the FunctionTool names in tool_functions.py. The registry is
# generic (register/unregister work for any label); this set only drives
# which tools get tracked automatically by the metrics hooks.
SLOW_TOOL_NAMES = frozenset({"run_code_sandbox", "generate_image", "edit_image"})

# Human-facing labels for the hint (emoji + short name).
TOOL_DISPLAY = {
    "run_code_sandbox": "🐳 code sandbox",
    "generate_image": "🎨 image generation",
    "edit_image": "🖌️ image editing",
}

# Source text (the tool's argument: sandbox task / image prompt) kept per
# in-flight run and shown in the hint, truncated.
MAX_SOURCE_CHARS = 120

# Recently-finished runs stay visible to follow-up prompts this long (a
# follow-up like "make it shorter" often lands seconds after the result was
# posted; without this, the prompt looks like the result never happened).
RECENT_DONE_SECONDS = 300.0

# Cap on the recently-done list per channel (oldest dropped).
MAX_RECENT_DONE = 5

# The repo is importable as both `classes.*` and `core.classes.*` (see
# AGENTS.md). If both names ended up in sys.modules the per-channel lock
# registry and the task registry would exist twice, so alias whichever name
# isn't ours yet to this module object (same trick as metrics.py).
if __name__ == "classes.message_queue":
    sys.modules.setdefault("core.classes.message_queue", sys.modules[__name__])
elif __name__ == "core.classes.message_queue":
    sys.modules.setdefault("classes.message_queue", sys.modules[__name__])


def _env_positive_int(name: str, default: int) -> int:
    """Env var `name` as an int >= 1; `default` when unset/empty/invalid."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def worker_count() -> int:
    """Queue worker tasks to run (WORKER_COUNT, default 2, min 1).

    Each worker handles one message at a time, so a message stuck in a slow
    tool (sandbox / image generation) no longer blocks messages from other
    channels. Same-channel chunk sends stay ordered via get_channel_lock().
    """
    return _env_positive_int("WORKER_COUNT", DEFAULT_WORKER_COUNT)


def queue_max_size() -> int:
    """Max messages waiting on the bounded queue (QUEUE_MAX_SIZE, default 10,
    min 1). Once full, on_message() drops new messages instead of letting
    the queue grow unbounded."""
    return _env_positive_int("QUEUE_MAX_SIZE", DEFAULT_QUEUE_MAX_SIZE)


def make_message_queue() -> asyncio.Queue:
    """The bounded message queue shared by all worker tasks."""
    return asyncio.Queue(maxsize=queue_max_size())


# One lock per channel id. Creating a lock needs no guard: the event loop is
# single-threaded and there is no await between the get and the set.
_channel_locks: dict[int, asyncio.Lock] = {}


def get_channel_lock(channel_id: int) -> asyncio.Lock:
    """The lock for one channel; the same id always gets the same lock.

    SCOPED lock (see module docstring): held only for the two fast phases
    of a handle (build_messages and the chunked send), never across the
    LLM run or tool calls. That serializes chunk sends (no interleaved
    replies, gentle on Discord rate limits) and keeps history snapshots
    consistent, while other workers can still answer the same channel while
    the LLM is busy on another message.
    """
    lock = _channel_locks.get(channel_id)
    if lock is None:
        lock = asyncio.Lock()
        _channel_locks[channel_id] = lock
    return lock


# ---------------------------------------------------------------------------
# In-flight task registry (long tools running in a channel)
# ---------------------------------------------------------------------------
#
# Because the LLM/tool phase of a handle is not channel-locked, a newer
# message in the same channel can build its prompt while an older message's
# slow tool is still running. The registry tracks those runs so the newer
# prompt can say "a previous request is still being processed (which tool,
# how long, what it was)", and follow-ups seconds after a finish know the
# result was already posted. All mutations happen on the single event loop
# with no await between read and write, so no lock of their own is needed.

# channel_id -> {run_key: {"label", "source", "started"}}
_in_flight: dict[int, dict[str, dict]] = {}
# channel_id -> [run_key, ...] in registration order (stable hint order)
_in_flight_order: dict[int, list[str]] = {}
# channel_id -> [{"label", "ended"}, ...] recently finished (ordered)
_recent_done: dict[int, list[dict]] = {}


def _truncate(text: str, limit: int = MAX_SOURCE_CHARS) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _format_elapsed(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def register_task_run(
    channel_id: int,
    label: str,
    source: str,
    run_key: str | None = None,
    started: float | None = None,
) -> bool:
    """Record one long tool run as in-flight in `channel_id`.

    `source` is the tool's argument (sandbox task text / image prompt),
    truncated to MAX_SOURCE_CHARS for prompt use. `run_key` (the SDK's
    tool_call_id) distinguishes concurrent runs of the same tool; without
    one, the truncated source is the key, so a duplicate register of the
    same run is a no-op. Returns True when a NEW entry was recorded
    (False: that run is already tracked).
    """
    key = run_key or _truncate(source) or "task"
    runs = _in_flight.get(channel_id)
    if runs is None:
        runs = {}
        _in_flight[channel_id] = runs
        _in_flight_order[channel_id] = []
    if key in runs:
        return False
    runs[key] = {
        "label": label,
        "source": _truncate(source),
        "started": started if started is not None else time.monotonic(),
    }
    _in_flight_order[channel_id].append(key)
    return True


def unregister_task_run(
    channel_id: int,
    run_key: str | None = None,
    finish: bool = True,
    ended: float | None = None,
) -> bool:
    """Remove an in-flight run from `channel_id`.

    `run_key` must match the one used in register_task_run (the hooks pass
    the same key to both). If no key is given, or the key is unknown, this
    is a no-op rather than a guess — removing the wrong run would clear
    someone else's hint. With `finish` (default) the run is also kept in
    the channel's recently-done list for RECENT_DONE_SECONDS. `ended`
    overrides the end timestamp (tests). Returns True when an entry was
    removed.
    """
    runs = _in_flight.get(channel_id)
    if not runs or run_key is None or run_key not in runs:
        return False
    entry = runs.pop(run_key)
    order = _in_flight_order.get(channel_id, [])
    if run_key in order:
        order.remove(run_key)
    if not runs:
        del _in_flight[channel_id]
        _in_flight_order.pop(channel_id, None)
    if finish:
        recent = _recent_done.setdefault(channel_id, [])
        recent.append({
            "label": entry["label"],
            "source": entry["source"],
            "ended": ended if ended is not None else time.monotonic(),
        })
        if len(recent) > MAX_RECENT_DONE:
            del recent[: len(recent) - MAX_RECENT_DONE]
    return True


def in_flight_hint(channel_id: int, now: float | None = None) -> str:
    """Prompt hint for a message being built in `channel_id`, or "".

    Lists tools still running (with how long they've been running and a
    truncated source) and runs that finished within RECENT_DONE_SECONDS
    (whose result was already sent to the channel). Called from
    MessageHandler.build_messages(); it only READS the registry (plus
    prunes stale recently-done entries), so it is safe to call either
    inside or outside the per-channel lock.
    """
    now = time.monotonic() if now is None else now
    lines: list[str] = []
    runs = _in_flight.get(channel_id)
    if runs:
        for key in _in_flight_order.get(channel_id, []):
            entry = runs.get(key)
            if entry is None:
                continue
            lines.append(
                f"- {entry['label']} running for {_format_elapsed(now - entry['started'])}"
                f" — {entry['source']}"
            )
    recent = _recent_done.get(channel_id)
    if recent:
        kept: list[dict] = []
        for record in recent:
            age = now - record["ended"]
            if age > RECENT_DONE_SECONDS:
                continue  # stale: drop it
            kept.append(record)
            lines.append(
                f"- {record['label']} finished {_format_elapsed(age)} ago — {record['source']}"
                f" (result already sent to the channel)"
            )
        if len(kept) != len(recent):
            _recent_done[channel_id] = kept
        if not kept:
            _recent_done.pop(channel_id, None)
    if not lines:
        return ""
    return ("Note: an earlier request in this channel is still being processed"
            " or finished very recently:\n" + "\n".join(lines))
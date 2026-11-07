"""Priority lanes over one LLM server's parallel slots.

Pure module (stdlib only). A self-hosted server (llama.cpp, Strata) answers
a fixed number of requests at once (its slots); more just queue inside it, in
arrival order. Lanes put that queue in core instead, ordered by what matters:
chat replies first, then the brain, then subagents, then background work. So
a long brain job never makes a chat reply wait behind its model calls.

One Lanes per server, keyed by llm_roles.host_key(), so chat only competes
where the persona runs: a brain on a remote API never waits on, or holds, the
local server's slots. A server is laned only when its /props reports slots
(main.py's refresh calls configure()); a remote API has no Lanes at all.

Rules (all unit-tested in llm_lanes_tests.py):

  * Capacity = slots. Chat may use any free slot. The other lanes share
    `slots - chat_reserved` between them; when that is <= 0 (a 1-slot
    server) they may still use the one slot whenever no chat call is
    running or waiting, so everything works in sequence, chat first.
  * When a slot frees, the highest-priority waiter goes next
    (chat > brain > subagent > background), first come first served within
    a lane.
  * A slot is held for ONE model call, never for a run or a tool, so a side
    call made inside a tool cannot deadlock against its own parent run.
  * Every mutation is free of `await`: the check and the take are atomic on
    the one event loop. Cancelling a waiter removes it; release is in
    `finally`.
"""
import asyncio
import contextlib
import contextvars
import os
import time
from collections import deque

CHAT, BRAIN, SUBAGENT, BACKGROUND = "chat", "brain", "subagent", "background"
PRIORITY = (CHAT, BRAIN, SUBAGENT, BACKGROUND)

# The lane of the work running in this task: a brain run sets BRAIN, and the
# side calls its tools make (research brief, image rewrite and check) inherit
# it, since asyncio tasks copy the context they are created in (D8).
current_lane: contextvars.ContextVar[str] = contextvars.ContextVar("llm_lane", default=CHAT)
# Seconds the latest model call in this task waited for its slot (LanedModel
# sets it). The SDK's on_llm_start hook fires before that wait, so
# ToolMetricsHooks subtracts it to keep discord_bot_llm_call_seconds model time.
last_wait: contextvars.ContextVar[float] = contextvars.ContextVar("llm_lane_wait", default=0.0)


class Lanes:
    def __init__(self, slots: int = 1, chat_reserved: int = 1):
        self.slots = 1
        self.chat_reserved = 0
        # Tokens one slot holds, from the server's /props (None: not reported).
        self.slot_context: int | None = None
        self._in_use = {lane: 0 for lane in PRIORITY}
        self._waiting = {lane: deque() for lane in PRIORITY}
        self.configure(slots, chat_reserved)

    def configure(self, slots: int, chat_reserved: int) -> None:
        """Set the slot count (a /props refresh can change it). Shrinking never
        revokes a held slot: new calls just wait until enough are released."""
        self.slots = max(1, int(slots))
        self.chat_reserved = min(max(0, int(chat_reserved)), self.slots)
        self._wake()

    # --- state -------------------------------------------------------------

    def _total(self) -> int:
        return sum(self._in_use.values())

    def _can_take(self, lane: str) -> bool:
        if self._total() >= self.slots:
            return False
        if lane == CHAT:
            return True
        shared = self.slots - self.chat_reserved
        others = self._total() - self._in_use[CHAT]
        if shared > 0:
            return others < shared
        # A 1-slot server (or every slot reserved): the rest go when chat is idle.
        return self._in_use[CHAT] == 0 and not self._waiting[CHAT]

    def _wake(self) -> None:
        """Hand free slots to waiters, highest priority first. A lane whose
        head cannot go (its share is full) does not block a lane that can."""
        progressed = True
        while progressed:
            progressed = False
            for lane in PRIORITY:
                queue = self._waiting[lane]
                while queue and queue[0].done():
                    queue.popleft()  # cancelled while waiting
                if queue and self._can_take(lane):
                    future = queue.popleft()
                    self._in_use[lane] += 1
                    future.set_result(None)
                    progressed = True
                    break

    def _release(self, lane: str) -> None:
        self._in_use[lane] -= 1
        self._wake()

    # --- API ---------------------------------------------------------------

    async def acquire(self, lane: str) -> float:
        """Take a slot for `lane`; returns the seconds spent waiting."""
        if lane not in self._in_use:
            raise ValueError(f"unknown lane {lane!r}")
        # Only when nobody of equal or higher priority is already waiting, so
        # a newcomer never jumps a queue.
        ahead = any(self._waiting[l] for l in PRIORITY[:PRIORITY.index(lane) + 1])
        if not ahead and self._can_take(lane):
            self._in_use[lane] += 1
            return 0.0
        started = time.monotonic()
        future = asyncio.get_running_loop().create_future()
        self._waiting[lane].append(future)
        try:
            await future
        except asyncio.CancelledError:
            if future.done() and not future.cancelled():
                # Granted at the same moment as cancelled: give it back.
                self._release(lane)
            else:
                with contextlib.suppress(ValueError):
                    self._waiting[lane].remove(future)
                self._wake()
            raise
        return time.monotonic() - started

    @contextlib.asynccontextmanager
    async def slot(self, lane: str):
        """`async with lanes.slot(lane) as waited:` one model call's slot."""
        waited = await self.acquire(lane)
        try:
            yield waited
        finally:
            self._release(lane)

    def snapshot(self) -> dict:
        """In use and waiting per lane (metrics)."""
        # Lengths, not an iteration: the metrics thread calls this while the
        # loop may be changing the deques. A cancelled waiter is removed at
        # once, so the length is the waiting count.
        return {lane: {"in_use": self._in_use[lane], "waiting": len(self._waiting[lane])}
                for lane in PRIORITY}


DEFAULT_CHAT_RESERVED = 1


def _env_int(name: str, env):
    try:
        return int(str(env.get(name, "") or "").strip())
    except ValueError:
        return None


def resolve_slots(props_slots: int | None, env=None) -> tuple[int, str]:
    """A self-hosted server's slot count and where it came from: its /props
    total_slots, else LLM_SLOTS (>= 1), else 1."""
    env = os.environ if env is None else env
    if props_slots and props_slots >= 1:
        return props_slots, "props"
    value = _env_int("LLM_SLOTS", env)
    if value is not None and value >= 1:
        return value, "env"
    return 1, "default"


def chat_reserved(env=None) -> int:
    """Slots kept for chat (LLM_CHAT_RESERVED, default 1; Lanes clamps it to
    the slot count)."""
    env = os.environ if env is None else env
    value = _env_int("LLM_CHAT_RESERVED", env)
    return value if value is not None and value >= 0 else DEFAULT_CHAT_RESERVED


# One process-wide registry, keyed by llm_roles.host_key().
_registry: dict[str, Lanes] = {}


def lanes_for(host_key: str) -> Lanes | None:
    """The host's Lanes, or None when it is not laned (a remote API)."""
    return _registry.get(host_key)


def configure_host(host_key: str, slots: int, chat_reserved: int,
                   slot_context: int | None = None) -> Lanes:
    """Lane a host (its /props answered), or update its slot count and the
    tokens one of its slots holds."""
    lanes = _registry.get(host_key)
    if lanes is None:
        lanes = _registry[host_key] = Lanes(slots, chat_reserved)
    else:
        lanes.configure(slots, chat_reserved)
    if slot_context:
        lanes.slot_context = slot_context
    return lanes


def slot_context_for(host_key: str) -> int | None:
    """Tokens one slot of this host holds, or None: a remote API (not laned)
    or a server that did not say. The context guard's limit
    (context_trim.ContextWatch)."""
    lanes = _registry.get(host_key)
    return None if lanes is None else lanes.slot_context


def all_lanes() -> dict[str, Lanes]:
    return dict(_registry)


def reset() -> None:
    """Forget every host (tests)."""
    _registry.clear()

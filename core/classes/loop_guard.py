"""Stops a brain run that is going round in circles. PURE (stdlib only).

The brain's turn and time limits are only backstops (100 turns, 30 minutes),
so this is what actually ends a stuck run. It watches the run's tool calls,
and a loop is either:

- the same tool with the same arguments `repeat_limit` times in a row, or
- the last WINDOW calls all repeats of calls already made in this run
  (nothing new is being tried).

The first loop earns a note at the next model call (note()); a second one
stops the run (should_stop()), and the caller reports what was done so far.

Honest long work never trips it: a command run again after an edit counts as
new when its arguments differ, and the window only fills with calls that
were all seen before.
"""
import json

WINDOW = 8
# Tools whose repeats are expected: polling a running command, and the
# brain's own bookkeeping. They never count towards a loop.
IGNORED = frozenset({"write_stdin", "report"})

REPEAT = "repeat"
STALE = "stale"


def _canonical(arguments) -> str:
    """Arguments as a stable string, so key order and spacing do not make
    the same call look new."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (TypeError, ValueError):
            return " ".join(arguments.split())
    try:
        return json.dumps(arguments, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(arguments)


class LoopGuard:
    def __init__(self, repeat_limit: int = 3, window: int = WINDOW):
        self.repeat_limit = max(2, repeat_limit)
        self.window = max(2, window)
        self._seen: set[tuple[str, str]] = set()
        self._recent: list[bool] = []  # per call: was it a repeat of an earlier one?
        self._last: tuple[str, str] | None = None
        self._run = 0
        self.loops = 0
        self.kind: str | None = None
        self._pending_note: str | None = None

    def record(self, tool: str, arguments) -> str | None:
        """Records one tool call. Returns the loop kind when this call makes
        a loop, else None."""
        if tool in IGNORED:
            return None
        key = (tool, _canonical(arguments))
        self._run = self._run + 1 if key == self._last else 1
        self._last = key
        self._recent.append(key in self._seen)
        self._recent = self._recent[-self.window:]
        self._seen.add(key)
        kind = None
        if self._run >= self.repeat_limit:
            kind = REPEAT
        elif len(self._recent) >= self.window and all(self._recent):
            kind = STALE
        if kind is None:
            return None
        self.loops += 1
        self.kind = kind
        # Start counting afresh, so the note gets a fair chance to work.
        self._run = 0
        self._recent = []
        if self.loops == 1:
            self._pending_note = (
                f"(You are repeating `{tool}` with the same arguments." if kind == REPEAT
                else "(Your last few tool calls repeat ones you already made."
            ) + (" Change your approach, or call `report` now with what you have"
                 " and what is still missing.)")
        return kind

    def note(self) -> str | None:
        """The warning to show at the next model call, once."""
        note, self._pending_note = self._pending_note, None
        return note

    def should_stop(self) -> bool:
        return self.loops >= 2

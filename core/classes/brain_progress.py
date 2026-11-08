"""The brain's status card ("🧠 Working on it") and its live progress.

Posted ONLY when the brain uses the server's workspace: its first shell
command, file attachment or message export (D18). A brain call that only
searches, saves a memory or posts a poll shows just that tool's own card, as
before; a status card on top of a poll would be noise. So nothing is sent
until the first workspace action (BrainProgressHooks.on_tool_start).

Two forms, chosen per server with /workspace progress:
  - off (the default): the card with the request (and its notes), relabelled
    when the call ends ("🧠 Done" / "🧠 Stopped"). While it works, one line
    says what the brain is doing in plain words ("⏳ Creating the script"),
    so a long job never looks frozen: a side call writes it
    (classes/brain_status.py) from the newest tool calls, at most every
    STATUS_INTERVAL_SECONDS. Until the first one answers (or when it fails),
    the line is the rule-written one helpers use (tool_cards.helper_line).
    The line goes when the call ends.
  - on: the same card with every exec_command and its output streamed in,
    a gap between each, edited in place.

How it hooks in:
  The OpenAI Agents SDK fires on_tool_start / on_tool_end for every
  function-tool call, the Shell capability's exec_command and write_stdin
  included (they are FunctionTools underneath). BrainProgressHooks is one of
  the brain run's hooks: on_tool_start carries the arguments (the command),
  on_tool_end the result (the command's output, formatted by the SDK). The
  Docker workspace runs with a PTY, so long commands are polled with
  write_stdin, each poll returning fresh output.

Discord rate limits:
  Sends are limited to 5 messages/5s per channel and EDITS to 5/minute per
  channel, so after the first message progress is edited at most every
  EDIT_INTERVAL seconds (15s, so at most 4 edits a minute), plus one final
  flush when the call ends. Every network failure is swallowed (progress
  must never break the brain's run) with a backoff before the next attempt; a
  deleted message falls back to a fresh send.

Rendering (streamed form):
  Each command in a ```sh block, its output (the newest OUTPUT_CHARS of it:
  a live log) in a plain block, then a small status line (✅ / ❌ exit N /
  ⏳ running). Nothing is dropped: when the commands outgrow one message the
  card goes on in more (card.LiveCard), and while the call runs only its last
  message is edited (EDIT_LAST), so the edit rate holds; the finish updates
  them all.

The pure parts (WorkTranscript, parse_exec_result) are unit-tested in
core/tests/brain_progress_tests.py.

A workflow's scouts and researchers have their own, smaller card
(HelperProgress): one each, posted when they start, showing only the last thing
they did. Their tools post nothing (common.QuietChannel). Two can run in one
thread, so each edits at most every HELPER_EDIT_SECONDS, which keeps the pair
under the 5 edits a minute.
"""
import asyncio
import logging
import json
import re
import time
from dataclasses import dataclass

from agents import RunHooks

from classes import card_kit as K
from classes.brain_status import plain_words
from classes.card import LiveCard
from classes.card_kit import CardSpec, Divider, Text
from classes.tool_cards import (
    BRAIN_DONE, BRAIN_NOT_FINISHED, BRAIN_STOPPED, BRAIN_WORKING,
    helper_ending, helper_line, work_task_text,
)

logger = logging.getLogger(__name__)

EDIT_INTERVAL_SECONDS = 15.0  # Discord allows 5 edits/minute per channel
HELPER_EDIT_SECONDS = 30.0    # two helpers share a thread's edits
BACKOFF_SECONDS = 25.0        # wait before retrying after a failed update
MAX_NOTES = 5                 # status notes kept
NOTE_CHARS = 300              # per-note cap
CMD_CHARS = 1500              # a command's text shown (a heredoc can be long)
OUTPUT_CHARS = 1500           # the newest output of a command (a live log)
PROGRESS_HINT = "live progress is on (/workspace progress)"
STATUS_INTERVAL_SECONDS = 30.0  # between status lines: two of the 5 edits a minute
STATUS_STEPS = 3                # newest tool calls the status line is written from

_EXIT_RE = re.compile(r"Process exited with code (-?\d+)")
_RUNNING_RE = re.compile(r"Process running with session ID (\d+)")
_OUTPUT_HEADER = "Output:"


def parse_exec_result(result) -> dict:
    """Parse one exec_command / write_stdin result from the SDK into
    {'output': str, 'exit_code': int | None, 'process_id': int | None}.

    The SDK formats results as:
        Chunk ID: …
        Wall time: …
        Process exited with code N     (or: Process running with session ID N)
        [Original token count: N]
        Output:
        <stdout/stderr>
    Anything that does not match is returned verbatim as 'output'.
    """
    text = result if isinstance(result, str) else str(result)
    exit_code = None
    process_id = None
    m = _EXIT_RE.search(text)
    if m:
        exit_code = int(m.group(1))
    m = _RUNNING_RE.search(text)
    if m:
        process_id = int(m.group(1))
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line == _OUTPUT_HEADER:
            text = "\n".join(lines[i + 1:])
            break
    return {"output": text.strip("\n"), "exit_code": exit_code, "process_id": process_id}


def _tool_args(context) -> dict:
    """The raw JSON tool_arguments of a ToolContext, as a dict ({} on
    anything unexpected — the hooks must never raise)."""
    raw = getattr(context, "tool_arguments", None)
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


@dataclass
class _Block:
    """One command the brain ran, plus everything that came back."""
    cmd: str
    stdin: str = ""
    output: str = ""
    exit_code: int | None = None
    process_id: int | None = None


def _fence(text: str, lang: str = "") -> str:
    # A ``` inside would end the block early.
    return f"```{lang}\n{text.replace('```', 'ˋˋˋ')}\n```"


def _tail(text: str, cap: int) -> str:
    """Keep the last `cap` chars (ellipsis marks what was cut)."""
    if len(text) <= cap:
        return text
    return "… " + text[-(cap - 2):]


def _head(text: str, cap: int) -> str:
    """Keep the first `cap` chars (ellipsis marks what was cut)."""
    if len(text) <= cap:
        return text
    return text[: cap - 2] + "… "


def _took(seconds: float) -> str:
    minutes = round(seconds / 60)
    return f"{max(1, round(seconds))} s" if minutes < 1 else f"{minutes} min"


class WorkTranscript:
    """The commands and notes behind the live card.

    Pure (no Discord/SDK interaction): append events, render a CardSpec.
    """

    def __init__(self, task: str, *, label: str = BRAIN_WORKING, title: str = "", files=(),
                 started: float | None = None, hint: str = ""):
        self._task = (task or "").strip()
        self._label = label
        self._title = title
        self._final: tuple | None = None  # (label, colour, seconds taken)
        self._files = list(files or ())
        self._started = started
        self._hint = hint
        self._blocks: list[_Block] = []
        self._notes: list[str] = []
        self._thinking = False
        self._status = ""

    # -- events -----------------------------------------------------------

    def add_note(self, text: str) -> None:
        note = " ".join((text or "").split())
        if not note:
            return
        self._notes.append(note[:NOTE_CHARS])
        del self._notes[:-MAX_NOTES]

    def add_command(self, cmd: str) -> _Block:
        block = _Block(cmd=_head((cmd or "").strip("\n"), CMD_CHARS))
        self._blocks.append(block)
        return block

    def _current_block(self) -> _Block:
        if not self._blocks:
            self._blocks.append(_Block(cmd=""))
        return self._blocks[-1]

    def add_input(self, chars: str) -> None:
        chars = (chars or "").strip("\n")
        if not chars:
            return
        block = self._current_block()
        block.stdin = _head(f"{block.stdin}\n{chars}".strip("\n"), CMD_CHARS)

    def add_output(self, output: str, exit_code: int | None = None,
                   process_id: int | None = None, block: _Block | None = None) -> None:
        """block: the command this output belongs to, when the caller knows
        it (BrainProgressHooks keys blocks by tool call); otherwise the
        latest one."""
        out = (output or "").strip("\n")
        if not out and exit_code is None and process_id is None:
            return
        block = block if block is not None else self._current_block()
        if out:
            block.output = _tail(f"{block.output}\n{out}".strip("\n"), OUTPUT_CHARS * 2)
        block.exit_code = exit_code
        block.process_id = process_id

    def set_thinking(self, thinking: bool) -> None:
        self._thinking = bool(thinking)

    def set_status(self, text: str | None) -> None:
        """The one line on what the call is doing now, as plain words (it
        is model or search text); None clears it."""
        self._status = plain_words(text)[:NOTE_CHARS]

    @property
    def status(self) -> str:
        return self._status

    @property
    def task(self) -> str:
        return self._task

    def set_final(self, label: str, colour: int, seconds: float | None = None) -> None:
        """The call is over: its closing label and colour."""
        self._final = (label, colour, seconds)

    # -- render -----------------------------------------------------------

    @staticmethod
    def _status_line(block: _Block) -> str:
        if block.exit_code is not None:
            return "-# ✅ Finished" if block.exit_code == 0 else f"-# ❌ Failed · exit {block.exit_code}"
        if block.process_id is not None or block.output:
            return "-# ⏳ Running…"
        return "-# ⏳ Starting…"

    def _block_text(self, block: _Block) -> str | None:
        if (not block.cmd and not block.stdin and not block.output
                and block.exit_code is None and block.process_id is None):
            return None
        parts = []
        if block.cmd:
            parts.append(_fence(block.cmd, "sh"))
        if block.stdin:
            parts.append("-# ⌨ typed in\n" + _fence(block.stdin, "sh"))
        if block.output:
            parts.append(_fence(_tail(block.output, OUTPUT_CHARS)))
        parts.append(self._status_line(block))
        return "\n".join(parts)

    def _head_label(self) -> str:
        if self._final is not None:
            label, _colour, seconds = self._final
            return f"{label} · took {_took(seconds)}" if seconds is not None else label
        if self._started:
            return f"{self._label} · started <t:{int(self._started)}:R>"
        return self._label

    def render(self) -> CardSpec:
        body = []
        if self._task:
            body.append(Text(work_task_text(self._task, self._files)))
        commands = [t for t in (self._block_text(b) for b in self._blocks) if t]
        if commands:
            body += [Divider(), *K.listing(commands)]
        if self._notes:
            body += [Divider(), Text(K.safe("\n".join(self._notes)))]
        footer = []
        if self._final is None and self._status:
            footer.append(f"⏳ {K.safe(self._status)}")
        if self._final is None and self._thinking:
            footer.append("💭 Thinking…")
        if self._final is None and self._hint:
            footer.append(self._hint)
        colour = self._final[1] if self._final is not None else K.WORKING
        return CardSpec(label=self._head_label(), title=self._title, body=body,
                        footer=" · ".join(footer), colour=colour, whole=True)


# The tools that use the server's workspace: the first of them posts the
# status card (D18).
WORKSPACE_TOOLS = frozenset({"exec_command", "write_stdin", "attach_file", "save_messages_to_file"})


class HelperProgress(RunHooks):
    """A scout's, researcher's or check's one card: what it is, its last tool
    call and its turn, out of `max_turns` (workflows._agent_run sets the run's
    real share). Like the hooks below it never raises, and a failed edit only
    waits for the next one.

    `label` says in plain words what this helper is ("🔬 Research · step 1 of
    3"), `title` names its job in a few words, `brief` is the whole of it (in
    small text: nothing is cut)."""

    def __init__(self, channel, label: str, *, max_turns: int, title: str = "", brief: str = "",
                 edit_interval: float = HELPER_EDIT_SECONDS):
        self.channel = channel
        self.label = label
        self.title = title
        self.brief = brief
        self.max_turns = max_turns
        self.edit_interval = edit_interval
        self.turn = 0
        self.line = ""
        self.card = LiveCard(channel)
        self._last_sent = 0.0
        self._finalized = False

    def spec(self, text: str, colour: int, footer: str | None = None) -> CardSpec:
        body = [Text(K.small(K.safe(self.brief)))] if self.brief else []
        if footer is None:
            footer = f"Turn {self.turn} of {self.max_turns}" if self.turn else ""
        return CardSpec(label=self.label, title=self.title,
                        body=body + [Divider(), Text(text)], footer=footer, colour=colour)

    async def _send(self, spec: CardSpec) -> None:
        self._last_sent = time.monotonic()
        try:
            await self.card.show(spec)
        except Exception as e:
            logger.warning(f"Helper progress: update failed: {e}")

    async def start(self) -> None:
        await self._send(self.spec("Starting…", K.WORKING))

    async def _update(self) -> None:
        if not self._finalized and time.monotonic() - self._last_sent >= self.edit_interval:
            await self._send(self.spec(self.line or "Thinking…", K.WORKING))

    async def on_llm_start(self, context, agent, system_prompt, input_items) -> None:
        self.turn += 1

    async def on_tool_start(self, context, agent, tool) -> None:
        try:
            line = helper_line(getattr(tool, "name", ""), _tool_args(context))
        except Exception as e:  # a progress error must never kill the run
            logger.warning(f"Helper progress: ignoring a bad tool call: {e}")
            return
        if line:
            self.line = line
            await self._update()

    async def finalize(self, ok: bool = True, note: str | None = None, found: int = 0) -> None:
        if self._finalized:
            return
        self._finalized = True
        footer = " · ".join(([f"Last: {self.line}"] if self.line else [])
                            + ([f"{self.turn} turns"] if self.turn else []))
        await self._send(self.spec(helper_ending(ok, note, found), K.OK if ok else K.FAIL, footer))


class BrainProgressHooks(RunHooks):
    """RunHooks for a brain call: posts the status card at its first
    workspace action and, with `stream` on (/workspace progress), mirrors
    each exec_command / write_stdin call and its output into it (edited in
    place, throttled to stay under Discord's edit rate limit).

    All hook methods are exception-safe — the SDK awaits them inside the
    run loop, and a raised hook would abort the brain's run.
    """

    def __init__(self, channel, task: str, *, stream: bool = False, files=(),
                 label: str = BRAIN_WORKING, title: str = "",
                 edit_interval: float = EDIT_INTERVAL_SECONDS, clock=time.time):
        self.channel = channel
        self.stream = stream
        self.clock = clock
        self.transcript = WorkTranscript(task, label=label, title=title, files=files,
                                            hint=PROGRESS_HINT if stream else "")
        self.edit_interval = edit_interval
        self.card = LiveCard(channel)
        self._started = False
        self._started_at = 0.0
        self._dirty = False
        self._last_flush = 0.0
        self._backoff_until = 0.0
        self._finalized = False
        # tool_call_id -> the transcript block its exec_command created.
        # One response's tool calls start concurrently, so the latest block
        # is not necessarily the one a finishing call belongs to.
        self._call_blocks: dict[str, _Block] = {}
        # The status line (off form only): the newest tool calls, whether one
        # came since the last line, and the task that writes it.
        self.status_interval = STATUS_INTERVAL_SECONDS
        self._steps: list[str] = []
        self._status_wanted = False
        self._status_task: asyncio.Task | None = None
        self._last_status = float("-inf")  # no line yet: the first comes at once
        # Whether the line is the side call's: until then each tool call
        # brings a fresh rule-written one, so a failing side call never
        # leaves an old step on the card.
        self._model_line = False

    @property
    def started(self) -> bool:
        """Whether the status card went out (the call used the workspace)."""
        return self._started

    # -- lifecycle -----------------------------------------------------------

    async def _start(self) -> None:
        if self._started or self._finalized or self.channel is None:
            return
        self._started = True
        self._started_at = self.clock()
        self.transcript._started = self._started_at
        await self._flush(force=True)

    async def finalize(self, ok: bool = True, note: str | None = None,
                       stopped: bool = True) -> None:
        """Flush the final state once the call is over. Nothing when the
        card never went out. `stopped` False: nothing stopped it, it only
        reported that it did not finish."""
        if self._finalized:
            return
        self._finalized = True
        await self._stop_status()
        if not self._started:
            return
        self.transcript.set_thinking(False)
        self.transcript.set_status(None)
        label, colour = ((BRAIN_DONE, K.OK) if ok else (BRAIN_STOPPED, K.FAIL) if stopped
                         else (BRAIN_NOT_FINISHED, K.WARN))
        self.transcript.set_final(label, colour, self.clock() - self._started_at)
        if note:
            self.transcript.add_note(note)
        await self._flush(force=True)

    # -- SDK hooks ----------------------------------------------------------

    async def on_llm_start(self, context, agent, system_prompt, input_items) -> None:
        if self.stream and self._started:
            self._note(self.transcript.set_thinking, True)

    async def on_llm_end(self, context, agent, response) -> None:
        if self.stream and self._started:
            self._note(self.transcript.set_thinking, False)

    async def on_tool_start(self, context, agent, tool) -> None:
        name = getattr(tool, "name", None)
        if not self.stream:
            self._record_step(name, context)
        if name in WORKSPACE_TOOLS:
            await self._start()
        if not self.stream:
            self._want_status()
            return
        args = _tool_args(context)
        if name == "exec_command":
            cmd = args.get("cmd")
            if cmd:
                try:
                    block = self.transcript.add_command(cmd)
                    self._dirty = True
                    call_id = getattr(context, "tool_call_id", None)
                    if isinstance(call_id, str) and call_id:
                        self._call_blocks[call_id] = block
                except Exception as e:  # a progress error must never kill the run
                    logger.warning(f"Brain progress: ignoring transcript error: {e}")
        elif name == "write_stdin":
            chars = args.get("chars") or ""
            if chars.strip("\n"):
                self._note(self.transcript.add_input, chars)
        await self._maybe_flush()

    async def on_tool_end(self, context, agent, tool, result) -> None:
        name = getattr(tool, "name", None)
        if not self.stream or name not in ("exec_command", "write_stdin"):
            return
        block = None
        if name == "exec_command":
            call_id = getattr(context, "tool_call_id", None)
            if isinstance(call_id, str):
                block = self._call_blocks.pop(call_id, None)
        parsed = parse_exec_result(result)
        if block is not None:
            parsed["block"] = block
        # A pure write_stdin poll with no fresh output and no exit adds
        # nothing — skip it (polls can arrive every 250ms).
        if name == "write_stdin" and not parsed["output"] and parsed["exit_code"] is None:
            return
        self._note(self.transcript.add_output, **parsed)
        await self._maybe_flush()

    # -- the status line (off form) ---------------------------------------------

    def _record_step(self, name, context) -> None:
        """Keeps the newest tool calls for the status line. Until the side
        call has written one, the newest also sets the rule-written line, so
        the card's first post already says what it is doing."""
        try:
            from classes.brain_status import step_text
            from classes.tool_cards import helper_line
            if not isinstance(name, str):
                return
            args = _tool_args(context)
            step = step_text(name, args)
            if step is None:
                return
            self._steps = (self._steps + [step])[-STATUS_STEPS:]
            self._status_wanted = True
            if not self._model_line:
                line = helper_line(name, args)
                if line:
                    # helper_line starts with an emoji: the card adds its own ⏳.
                    # Marked dirty: on a card already out, the status task shows it.
                    self._note(self.transcript.set_status, line.split(" ", 1)[-1])
        except Exception as e:  # a progress error must never kill the run
            logger.warning(f"Brain progress: ignoring status error: {e}")

    def _want_status(self) -> None:
        """Starts the task that writes the status line, unless it runs."""
        if (self._finalized or not self._started or not self._status_wanted
                or (self._status_task is not None and not self._status_task.done())):
            return
        try:
            self._status_task = asyncio.get_running_loop().create_task(self._status_loop())
        except Exception as e:
            logger.warning(f"Brain progress: could not start the status line: {e}")

    async def _status_loop(self) -> None:
        """While new tool calls come: wait out the interval, write a line
        from the newest ones, show it. Never raises (but is cancellable)."""
        try:
            from classes import brain_status
            while self._status_wanted and not self._finalized:
                wait = self._last_status + self.status_interval - time.monotonic()
                if wait > 0:
                    await asyncio.sleep(wait)
                self._status_wanted = False
                self._last_status = time.monotonic()
                line = await brain_status.describe(self.transcript.task, list(self._steps))
                if line and not self._finalized:
                    self._model_line = True
                    if line != self.transcript.status:
                        self._note(self.transcript.set_status, line)
                await self._show_status()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"Brain progress: status line failed: {e}")

    async def _show_status(self) -> None:
        """Puts a changed line on the card, waiting out the edit throttle
        (or a backoff) rather than dropping it: during one long command no
        other edit comes before the end, which clears the line."""
        for _ in range(3):
            if not self._dirty or self._finalized:
                return
            wait = (max(self._last_flush + self.edit_interval, self._backoff_until)
                    - time.monotonic())
            if wait > 0:
                await asyncio.sleep(wait)
            await self._maybe_flush()

    async def _stop_status(self) -> None:
        """Ends the status task, so its edit never lands on the final card."""
        task, self._status_task = self._status_task, None
        if task is None:
            return
        if not task.done():
            task.cancel()
        # Its own end only: a cancel of finalize itself still propagates.
        await asyncio.gather(task, return_exceptions=True)

    # -- internals ------------------------------------------------------------

    def _note(self, fn, /, *args, **kwargs) -> None:
        try:
            fn(*args, **kwargs)
            self._dirty = True
        except Exception as e:  # a progress error must never kill the run
            logger.warning(f"Brain progress: ignoring transcript error: {e}")

    async def _maybe_flush(self, force: bool = False) -> None:
        if self._finalized or not self._started or not self._dirty:
            return
        now = time.monotonic()
        if not force and now < self._backoff_until:
            return
        if not force and now - self._last_flush < self.edit_interval:
            return
        await self._flush(force=force)

    async def _flush(self, force: bool = False) -> None:
        try:
            spec = self.transcript.render()
        except Exception as e:
            logger.warning(f"Brain progress: render failed: {e}")
            return
        # Only a message whose content changed is edited: with the commands
        # kept whole (render), that is the newest one while it works; the
        # finish updates them all (their colour).
        if not await self.card.show(spec):
            # Usually a Discord rate limit (edits: 5/minute per channel) or
            # a network error — back off and keep the brain's run going.
            logger.warning("Brain progress: update failed, backing off")
            self._backoff_until = time.monotonic() + BACKOFF_SECONDS
            return
        self._last_flush = time.monotonic()
        self._dirty = False

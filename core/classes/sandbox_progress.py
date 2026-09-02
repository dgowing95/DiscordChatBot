"""Live progress for the run_code_sandbox tool.

Streams the sandbox agent's activity — every exec_command it runs and the
output it gets back — into a SINGLE Discord message that is edited in
place, so the channel can watch the sandbox work instead of going silent
for minutes.

How it hooks in:
  The OpenAI Agents SDK fires on_tool_start / on_tool_end lifecycle hooks
  for every function-tool invocation — including the sandbox Shell
  capability's exec_command and write_stdin (they are FunctionTools under
  the hood). We attach a RunHooks instance (SandboxProgressHooks) to the
  nested sandbox Runner.run: on_tool_start carries the tool arguments
  (the command), on_tool_end carries the tool result (the command output,
  formatted by the SDK). The Docker sandbox runs with a PTY, so
  long-running commands are polled with write_stdin and each poll returns
  up to ~10s of fresh output — that is what gets appended to the live
  message.

Discord rate limits:
  Sends are limited to 5 messages/5s per channel and EDITS to 5/minute
  per channel, so after the first message progress is ONE message edited
  at most every EDIT_INTERVAL seconds (15s → max 4 edits/min), plus one
  final flush when the run ends. Every network failure is swallowed
  (best-effort progress must never break the sandbox run) with a backoff
  before the next attempt; a deleted message falls back to a fresh send.

Rendering:
  ONE embed, styled like the static "Tool Usage" embed the tool sends
  when progress is off: title "🐳 Sandbox", description "Running in
  sandbox: {task}" (same wording as the static embed) and ONE FIELD PER
  COMMAND. One-line commands are the field NAME (bold, "$ cmd");
  heredocs/long commands use the name "⌨ Command" and carry their full
  text in the value. Output goes in a code fence (monospace, whitespace
  kept) followed by an "exit N" / "… still running" status line. The
  embed's accent colour is the overall state: cyan (idle/initial, the
  static embed's colour), yellow (latest command still running), red
  (latest command failed). Events are QUEUED in the transcript and the
  whole queue is rendered into every (throttled) edit, so many commands
  land in one edit; when the queue outgrows Discord's embed budget
  (25 fields / 6000 chars of embed text per message) the OLDEST fields
  are dropped first.

This is a per-guild opt-in: the /sandbox_progress_updates slash command
stores the setting in Redis (configManager, dcb namespace); when it is off
or unset (the default) the tool only sends the one static
"Running in sandbox" embed and no progress hooks are attached.

The pure parts (SandboxTranscript, parse_exec_result,
sandbox_progress_updates_enabled) are unit-tested in
core/tests/sandbox_progress_tests.py.
"""
import json
import re
import time
from dataclasses import dataclass, field

import discord
from agents import RunHooks

EDIT_INTERVAL_SECONDS = 15.0  # Discord allows 5 edits/minute per channel
BACKOFF_SECONDS = 25.0        # wait before retrying after a failed update
MAX_BODY_CHARS = 1850         # keeps the message body under Discord's 2000
MAX_NOTES = 5                 # status notes kept for the body
NOTE_CHARS = 200              # per-note cap

# Embed limits (Discord hard caps: 25 fields per embed, 256-char field
# names, 1024-char field values, 6000 chars of embed text in total).
FIELD_NAME_CHARS = 200        # one-line commands that fit in a field NAME
FIELD_VALUE_CHARS = 1024
FIELD_CMD_CHARS = 400         # heredoc command text kept in the value
MAX_FIELDS = 25
EMBED_CHAR_BUDGET = 5800      # stay a bit under the 6000 hard cap
DESCRIPTION_CHARS = 1000      # the task line in the embed description

# Accent colours: the single embed's state.
COLOR_TOOL = 0x00B0F4         # cyan — idle/initial, like the static embed
COLOR_RUNNING = 0xFEE75C      # yellow — latest command still running
COLOR_FAIL = 0xED4245         # red — latest command exited non-zero

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


def sandbox_progress_updates_enabled(raw) -> bool:
    """Whether a guild's sandbox progress updates are on.

    raw is the value from configManager.get_setting for the
    sandbox_progress_updates setting (set by the /sandbox_progress_updates
    slash command): None/False when unset — which DEFAULTS TO OFF —
    otherwise the stored string ("True"/"False").
    """
    if raw in (None, False):
        return False
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


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
class FieldSpec:
    """One Discord embed field, without the discord library (pure)."""
    name: str
    value: str


@dataclass
class EmbedSpec:
    """One Discord embed, described without the discord library (pure).

    SandboxProgressHooks turns this into a discord.Embed.
    """
    title: str
    color: int
    description: str | None = None
    fields: list[FieldSpec] = field(default_factory=list)


@dataclass
class _Block:
    """One command the sandbox ran, plus everything that came back."""
    cmd: str
    stdin: str = ""
    output: str = ""
    exit_code: int | None = None
    process_id: int | None = None


def _fence(text: str, lang: str = "") -> str:
    return f"```{lang}\n{text}\n```"


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


class SandboxTranscript:
    """The rolling queue of command/output blocks behind the live message.

    Pure (no Discord/SDK interaction): append events, render the tail.
    render_message() returns (body, EmbedSpec): the body (status notes /
    thinking line) always fits Discord's 2000-char message limit and the
    single embed always fits its 25-field / 6000-char budget — when the
    queue is bigger, the OLDEST fields are dropped.
    """

    def __init__(self, task: str, *, workspace_note: str = ""):
        self._task = " ".join((task or "").split())
        # One line above the task saying whether this run started from an
        # empty workspace or resumed the thread's saved one (built by
        # sandbox_agent.sandbox_workspace_note). Empty renders nothing, so
        # the no-thread fallback path looks exactly as it did before.
        self._workspace_note = " ".join((workspace_note or "").split())
        self._blocks: list[_Block] = []
        self._notes: list[str] = []
        self._thinking = False

    # -- events -----------------------------------------------------------

    def add_note(self, text: str) -> None:
        note = " ".join((text or "").split())
        if not note:
            return
        self._notes.append(note[:NOTE_CHARS])
        del self._notes[:-MAX_NOTES]

    def add_command(self, cmd: str) -> None:
        cmd = (cmd or "").strip("\n")
        if len(cmd) > FIELD_CMD_CHARS:
            cmd = _head(cmd, FIELD_CMD_CHARS)
        self._blocks.append(_Block(cmd=cmd))

    def _current_block(self) -> _Block:
        if not self._blocks:
            self._blocks.append(_Block(cmd=""))
        return self._blocks[-1]

    def add_input(self, chars: str) -> None:
        chars = (chars or "").strip("\n")
        if not chars:
            return
        block = self._current_block()
        block.stdin = f"{block.stdin}\n{chars}".strip("\n")
        if len(block.stdin) > FIELD_CMD_CHARS:
            block.stdin = _head(block.stdin, FIELD_CMD_CHARS)

    def add_output(self, output: str, exit_code: int | None = None,
                   process_id: int | None = None) -> None:
        out = (output or "").strip("\n")
        if not out and exit_code is None and process_id is None:
            return
        block = self._current_block()
        if out:
            block.output = f"{block.output}\n{out}".strip("\n")
            # the per-field value budget is enforced at render time; keep
            # the raw tail bounded here as well
            if len(block.output) > FIELD_VALUE_CHARS * 2:
                block.output = _tail(block.output, FIELD_VALUE_CHARS * 2)
        block.exit_code = exit_code
        block.process_id = process_id

    def set_thinking(self, thinking: bool) -> None:
        self._thinking = bool(thinking)

    # -- render -----------------------------------------------------------

    def render_message(self) -> tuple[str, EmbedSpec]:
        return self._render_body(), self._render_embed()

    def _render_body(self) -> str:
        lines = list(self._notes[-3:])
        if self._thinking:
            lines.append("💭 model thinking…")
        body = "\n".join(lines)
        # notes are capped, so this normally never triggers
        if len(body) > MAX_BODY_CHARS:
            body = body[: MAX_BODY_CHARS - 1] + "…"
        return body

    def _state_color(self) -> int:
        if self._blocks:
            last = self._blocks[-1]
            if last.exit_code is None:
                return COLOR_RUNNING
            if last.exit_code != 0:
                return COLOR_FAIL
        return COLOR_TOOL

    def _status_line(self, block: _Block) -> str:
        if block.exit_code is not None:
            if block.exit_code == 0:
                return "exit 0"
            return f"**exit {block.exit_code}**"
        if block.process_id is not None or block.output:
            return "… still running"
        return ""

    def _field_for_block(self, block: _Block) -> FieldSpec | None:
        if (not block.cmd and not block.stdin and not block.output
                and block.exit_code is None and block.process_id is None):
            return None
        one_line = (bool(block.cmd) and not block.stdin
                    and "\n" not in block.cmd
                    and len(block.cmd) <= FIELD_NAME_CHARS)
        if one_line:
            name = f"$ {block.cmd}"
            parts: list[str] = []
        elif block.cmd:
            name = "⌨ Command"
            parts = [_fence(_head(block.cmd, FIELD_CMD_CHARS), "sh")]
        else:
            name = "⌨ stdin"
            parts = []
        if block.stdin:
            parts.append(_fence(_head(block.stdin, FIELD_CMD_CHARS), "sh"))
        status = self._status_line(block)
        if block.output:
            # the output fence gets whatever of the 1024-char field budget
            # the command/stdin/status parts do not use (tail kept)
            used = sum(len(p) for p in parts) + len(status)
            separators = 2 * (len(parts) + 1)  # "\n\n" between parts
            room = FIELD_VALUE_CHARS - used - separators - 5  # 5 = fence
            parts.append(_fence(_tail(block.output, max(room, 40))))
        if status:
            parts.append(status)
        value = "\n\n".join(parts)
        # defensive: the caps above should already keep this under 1024
        return FieldSpec(name[:256], _tail(value, FIELD_VALUE_CHARS))

    def _render_embed(self) -> EmbedSpec:
        title = "🐳 Sandbox"
        description = f"Running in sandbox: {_tail(self._task, DESCRIPTION_CHARS)}"
        if self._workspace_note:
            description = f"{self._workspace_note}\n{description}"
        # newest first: keep whole fields while they fit Discord's embed
        # budget; older fields are evicted as a unit so a command never
        # ends up separated from its output
        fields: list[FieldSpec] = []
        total = len(title) + len(description)
        for block in reversed(self._blocks):
            f = self._field_for_block(block)
            if f is None:
                continue
            cost = len(f.name) + len(f.value)
            if len(fields) >= MAX_FIELDS or total + cost > EMBED_CHAR_BUDGET:
                break
            fields.append(f)
            total += cost
        fields.reverse()
        return EmbedSpec(title, self._state_color(), description, fields)


class SandboxProgressHooks(RunHooks):
    """RunHooks for the nested sandbox run: mirrors each exec_command /
    write_stdin call and its output into one Discord message (edited in
    place, throttled to stay under Discord's edit rate limit).

    All hook methods are exception-safe — the SDK awaits them inside the
    run loop, and a raised hook would abort the sandbox run.
    """

    def __init__(self, channel, task: str, *, workspace_note: str = "",
                 edit_interval: float = EDIT_INTERVAL_SECONDS):
        self.channel = channel
        self.transcript = SandboxTranscript(task, workspace_note=workspace_note)
        self.edit_interval = edit_interval
        self._message: discord.Message | None = None
        self._dirty = False
        self._last_flush = 0.0
        self._backoff_until = 0.0
        self._finalized = False

    # -- lifecycle (called by the tool, not by the SDK) -------------------

    async def start(self) -> None:
        """Send the initial progress message (before Runner.run starts).

        The embed mirrors the static "Tool Usage" embed sent when progress
        is off — the channel sees the same thing either way.
        """
        self.transcript.add_note("starting container…")
        await self._flush(force=True)

    async def finalize(self, note: str | None = None) -> None:
        """Flush the final state (after the run ended: done/timeout/error)."""
        if self._finalized:
            return
        self._finalized = True
        self.transcript.set_thinking(False)
        if note:
            self.transcript.add_note(note)
        await self._flush(force=True)

    # -- SDK hooks ----------------------------------------------------------

    async def on_llm_start(self, context, agent, system_prompt, input_items) -> None:
        self._note(self.transcript.set_thinking, True)

    async def on_llm_end(self, context, agent, response) -> None:
        self._note(self.transcript.set_thinking, False)

    async def on_tool_start(self, context, agent, tool) -> None:
        name = getattr(tool, "name", None)
        args = _tool_args(context)
        if name == "exec_command":
            cmd = args.get("cmd")
            if cmd:
                self._note(self.transcript.add_command, cmd)
        elif name == "write_stdin":
            chars = args.get("chars") or ""
            if chars.strip("\n"):
                self._note(self.transcript.add_input, chars)
        await self._maybe_flush()

    async def on_tool_end(self, context, agent, tool, result) -> None:
        name = getattr(tool, "name", None)
        if name not in ("exec_command", "write_stdin"):
            return
        parsed = parse_exec_result(result)
        # A pure write_stdin poll with no fresh output and no exit adds
        # nothing — skip it (polls can arrive every 250ms).
        if name == "write_stdin" and not parsed["output"] and parsed["exit_code"] is None:
            return
        self._note(self.transcript.add_output, **parsed)
        await self._maybe_flush()

    # -- internals ------------------------------------------------------------

    def _note(self, fn, /, *args, **kwargs) -> None:
        try:
            fn(*args, **kwargs)
            self._dirty = True
        except Exception as e:  # a progress error must never kill the run
            print(f"Sandbox progress: ignoring transcript error: {e}")

    async def _maybe_flush(self, force: bool = False) -> None:
        if self._finalized or not self._dirty:
            return
        now = time.monotonic()
        if not force and now < self._backoff_until:
            return
        if not force and now - self._last_flush < self.edit_interval:
            return
        await self._flush(force=force)

    def _build_embed(self, spec: EmbedSpec) -> discord.Embed:
        embed = discord.Embed(title=spec.title, description=spec.description,
                              color=spec.color)
        for f in spec.fields:
            embed.add_field(name=f.name, value=f.value, inline=False)
        return embed

    async def _flush(self, force: bool = False) -> None:
        try:
            content, spec = self.transcript.render_message()
        except Exception as e:
            print(f"Sandbox progress: render failed: {e}")
            return
        embed = self._build_embed(spec)
        try:
            if self._message is None:
                self._message = await self.channel.send(content, embeds=[embed])
            else:
                await self._message.edit(content=content, embeds=[embed])
        except discord.NotFound:
            # The message was deleted; fall back to a fresh send.
            self._message = None
            print("Sandbox progress: message deleted, will resend")
            return
        except Exception as e:
            # Usually a Discord rate limit (edits: 5/minute per channel) or
            # a network error — back off and keep the sandbox run going.
            print(f"Sandbox progress: update failed, backing off: {e}")
            self._backoff_until = time.monotonic() + BACKOFF_SECONDS
            return
        self._last_flush = time.monotonic()
        self._dirty = False

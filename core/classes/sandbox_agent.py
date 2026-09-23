"""Docker-backed code sandbox for the run_code_sandbox tool.

Uses the OpenAI Agents SDK sandbox feature (beta, `agents.sandbox`): a nested
SandboxAgent does the work inside a throwaway Docker container (shell +
filesystem capabilities, python image) and returns its final answer.

Lifecycle: every call creates a FRESH container, runs the nested agent loop,
then the container is stopped and deleted. When the run happens in a Discord
thread (see ensure_sandbox_thread/tool_functions.run_code_sandbox), the
workspace is persisted to Redis as a snapshot keyed by that thread's id
before teardown (session.stop(), see _persist_sandbox_snapshot) and restored
into the next fresh container for a later run in the SAME thread
(_create_sandbox_session) — so a thread's work can continue across calls even
though every individual container is still fully disposable. A one-off call
with no thread behaves exactly as before: nothing persists.

Requirements (see docker-compose.yaml / charts/dis-ai-bot):
  - a reachable Docker daemon (docker.sock mounted into the core container;
    `docker` + `websocket-client` Python packages in requirements.txt)
  - the sandbox image is pulled once onto the daemon, then cached

The heavy imports (docker SDK, agents.sandbox) stay inside the builder
functions so the pure helpers below are testable without them.
"""
import logging
import asyncio
import io
import json
import os
import posixpath
import re
import time
import uuid
from dataclasses import dataclass, field

import discord
# httpx2, not httpx: openai>=3 speaks HTTPX2 and does not translate a legacy
# httpx.Timeout (it lands inside the httpx2 client as an opaque object, not
# per-phase floats). Nothing else in core needs legacy httpx any more.
import httpx2
from openai import APITimeoutError
from agents import (
    AsyncOpenAI,
    MaxTurnsExceeded,
    ModelBehaviorError,
    ModelRefusalError,
    ModelSettings,
    ModelTimeoutError,
    OpenAIChatCompletionsModel,
    Runner,
    RunConfig,
    RunContextWrapper,
    function_tool,
)
from agents.sandbox.config import DEFAULT_PYTHON_SANDBOX_IMAGE

from classes import sandbox_thread_inbox
from classes.sandbox_conversation import (
    MAX_CONTINUATIONS,
    MIN_CONTINUATION_TURNS,
    SandboxConversation,
    make_agent_hooks,
)
from classes.llm_config import DEFAULT_LLM_API_KEY, DEFAULT_LLM_HOST, DEFAULT_MODEL, env_or

logger = logging.getLogger(__name__)

DEFAULT_MAX_TURNS = 10
DEFAULT_TIMEOUT_SECONDS = 600
DEFAULT_ASK_USER_TIMEOUT_SECONDS = 300
DEFAULT_PERSIST_TIMEOUT_SECONDS = 180  # generous: covers Memory's 2 extra model calls on stop()
DEFAULT_REQUEST_TIMEOUT_SECONDS = 180  # per-HTTP-request bound on one sandbox model call
DEFAULT_MAX_RETRIES = 2  # matches the OpenAI client's own default, but stated explicitly
SANDBOX_CONNECT_TIMEOUT_SECONDS = 10.0  # separate, short: a dead host must not eat the whole budget
SANDBOX_DELETE_TIMEOUT_SECONDS = 30  # bound on tearing down our own container
SANDBOX_RECOVERY_TIMEOUT_SECONDS = 30  # bound on best-effort artifact recovery after a timeout
# Dependency key the DockerSandboxClient's snapshot Dependencies are bound
# under (see build_sandbox_client) — must match what RemoteSnapshot(id=...,
# client_dependency_key=...) is constructed with in _create_sandbox_session.
SANDBOX_SNAPSHOT_DEP_KEY = "dcb.sandbox_snapshot"

# Directory (relative to the sandbox's own shell cwd) where the sandbox
# agent saves anything it means to return as a file. It is no longer the
# delivery boundary — the agent picks what is delivered with attach_file,
# and out/ is only where finished candidates live (plus the fallback swept
# when it attached nothing; see _deliver). Nothing left elsewhere is lost
# either: in a thread the whole workspace is snapshotted to Redis on
# teardown and restored for the next run there (_persist_sandbox_snapshot).
#
# This is intentionally relative, never a hardcoded path: _create_sandbox_session
# now explicitly calls session.start() (added for snapshot resume — see its
# docstring), which DOES materialize the SDK's nominal "/workspace" root and
# make it the session's tracked cwd — verified live against a real Docker
# daemon — even with the Shell-only capability we use (see build_sandbox_agent;
# Filesystem is still avoided for the apply_patch/ChatCompletions reason
# below). Before that call existed, an un-started session's exec_command
# calls ran with whatever cwd the image itself defaulted to instead (e.g. "/"
# for the default sandbox image) — this comment described that older
# behavior. Either way we never hardcode the path: we resolve `out/` by
# asking the live session for its own cwd (_sandbox_output_dir), so this
# stays correct regardless of what the actual root turns out to be. We do
# this resolution BEFORE
# Runner.run starts (not just post-hoc to collect artifacts afterward), so
# we can mkdir -p it before the model's first turn — but we deliberately
# never tell the model the resolved ABSOLUTE path (see SANDBOX_INSTRUCTIONS
# below): exec_command's own `workdir` argument is validated by the SDK's
# manifest system, which unconditionally rejects absolute paths
# (InvalidManifestPathError: "manifest path must be relative: ...") even
# when the path genuinely exists — confirmed in production, where a model
# told the absolute path reused it as `workdir` and the tool call failed
# outright. The model only ever gets the relative name; we resolve/track
# the absolute form ourselves for our own raw session.exec() calls, which
# bypass that manifest check entirely (see _collect_artifacts).
SANDBOX_OUTPUT_DIRNAME = "out"
MAX_ARTIFACT_FILES = 10
# Marker file touched at the start of every run, next to (never inside) the
# output dir, so the fallback sweep can tell this run's files from ones a
# resumed workspace snapshot restored — see _mark_run_start.
RUN_MARKER_NAME = ".dcb_run_start"
# Discord's own non-boosted upload cap is ~25MB; enforcing it here fails
# fast instead of reading a huge file into memory only for Discord to
# reject the upload.
MAX_ARTIFACT_BYTES = 25_000_000

# Instructions for the nested sandbox agent. On a first run in a thread the
# workspace is empty and the task text is its only input, so it must behave
# self-sufficiently; on a resumed run RESUMED_TASK_PREAMBLE tells it
# otherwise. It MUST rule out apply_patch by name: the SDK's default sandbox
# prompt (suppressed — see build_sandbox_agent) tells the model to use an
# apply_patch tool we do not expose, and a model that calls it aborts the
# whole run with ModelBehaviorError.
#
# Every instruction here has ONE home. The prompt reached ~5000 characters by
# accretion — each observed bug fixed by adding text and never removing any —
# at which point `out/` was explained in five places and two bullets
# contradicted each other outright. Duplicated guidance is worse than terse
# guidance on a small local model: copies drift apart, and a model given two
# versions of a rule follows neither reliably. If you need to say something
# here that is already said elsewhere, edit the existing home instead.
#
# {output_bullet} is filled in by build_sandbox_agent(out_dir): both
# variants name only the RELATIVE `out/` dirname, never the resolved
# absolute path, and both say WHY — exec_command's `workdir` rejects every
# absolute path, `/workspace` included (see the SANDBOX_OUTPUT_DIRNAME
# comment above). They used to claim instead that `/workspace` "does not
# exist", which was true before _create_sandbox_session called
# session.start() and is not now; the bullet ended up contradicting its own
# closing clause ("even though the directory is real").
SANDBOX_INSTRUCTIONS = """You work inside an isolated Linux sandbox (a minimal
Python container) with shell tools. What you receive is a REQUEST, not a
specification.

- You own every design and implementation decision the request does not
  explicitly pin down: visual style, dimensions, colours, libraries,
  algorithms, file structure, method. Decide, build it, look at the result.
  You can actually run code and see the output, so your judgement beats the
  caller's. Only details the USER stated are fixed.
- The only way to touch files is the shell: `exec_command` runs a command,
  `write_stdin` talks to a running process. There is no separate editing or
  patching tool. Create and modify files with heredocs
  (`cat > file.py << 'EOF' ... EOF`) or python one-liners, then run and
  verify what you wrote.
- The workspace starts empty unless the task says otherwise. Never assume
  files, packages or context exist unless the task provides them; install
  what you need (e.g. `pip install ...`).
- Do the work: write files, run commands, read the output, iterate until the
  task is done. If a step fails, read the error, fix it and retry.
- If the task includes code or data, create it exactly as given.
- For most ambiguity, make a reasonable assumption and state it. Use
  `ask_user` only for a choice that really changes the result:
  `blocking=false` plus a `default` lets you keep working meanwhile.
- People can message you in the thread while you work, as
  `[thread message #N from ...]`. Answer each FIRST with
  `respond_to_updates` (your other tools refuse to run until you do), then
  act on it straight away; mark it `done` once it really is. E.g.
  `respond_to_updates(message_ids="3", decision="will_apply", reply="Keeping
  the layout, making the bars blue.")` and re-render in the same response.
- For something the user will judge (a chart, a design), show an early
  draft with `send_preview_to_thread` rather than only the final result.
- {output_bullet}
- {budget_bullet}
- Once you have confirmed a result is correct, stop — do not re-verify what
  you have already verified.
- When done, end with a short message TO THE USER: it is posted in the thread
  beside the file(s) you attached, so write it to them. Say what you made, the
  key results (exact numbers where they matter), the significant choices you
  made, anything that failed or that you assumed, and anything you changed
  because they asked mid-run. It is FINAL:
  there is no turn after it and nothing you say later reaches anyone, so never
  describe work as still in progress, being generated, or about to arrive. If
  a file was required and you have not verified with `ls` that it exists, it
  does not exist — say so plainly as a failure instead of implying it is on
  its way."""

_OUTPUT_BULLET_RESOLVED = (
    "Use RELATIVE paths only: exec_command's `workdir` rejects every "
    "absolute path — `/workspace` included — even when the directory really "
    "exists. To return a FILE (a plot, a converted document, generated "
    "data), save it under `out/` from wherever you already are; it has "
    "already been created for you, no `cd` or `mkdir` needed. Then call "
    "`attach_file` on the finished file — ONE per thing the user asked for. "
    "Only attached files are sent, so drafts left behind cost nothing. "
    "Don't print file contents to stdout, and leave "
    "your working files in place: the rest of the workspace is saved and "
    "restored for the next run in this thread."
)
_OUTPUT_BULLET_FALLBACK = (
    "Use RELATIVE paths only. exec_command's `workdir` rejects every "
    "absolute path — `/workspace` included — even when the directory really "
    "exists. To return a FILE (a plot, a converted document, generated "
    "data), save it under the `out/` directory relative to your working "
    "directory (create it first: `mkdir -p out`), then call `attach_file` on "
    "the finished file — ONE per thing the user asked for. Only attached "
    "files are sent, so drafts cost nothing. Don't print file contents to "
    "stdout."
)

# {budget_bullet}: tells the model its hard turn/time budget so it can pace
# itself instead of getting cut off mid-task with nothing to show. A turn is
# one model response (which may include one or more tool calls); running
# out of either turns or wall-clock seconds ends the run immediately with
# no chance to write a final report (see run_sandbox_task's
# MaxTurnsExceeded/timeout handling) — anything not already saved under
# out/ at that point is lost. Filled in by build_sandbox_agent with the
# live sandbox_max_turns()/sandbox_timeout() values so this always matches
# what the caller actually enforces.
#
# The model has no internal clock, so it is TOLD the elapsed time rather
# than taught to measure it: _elapsed_note appends the running total to
# every shell result. The bullet used to carry the measurement procedure
# instead ("run `date +%s` as your very first command ... re-run it every
# few turns and subtract"), which asked a small local model to carry an
# epoch integer across turns and do arithmetic on it, spent a turn out of a
# budget that can be as low as 10, and fought RESUMED_TASK_PREAMBLE over
# which command gets to be "first".
_BUDGET_BULLET = (
    "You have at most {max_turns} turns (one response = one turn) and "
    "{timeout_seconds} seconds of wall-clock time for this whole task; "
    "whichever runs out first ends the run immediately, with no chance to "
    "write a final report. Every command tells you how much of that time "
    "you have used. Once roughly 80% is gone without a finished result, "
    "STOP iterating: save and attach your best current partial output, then "
    "write the final message describing what is done and what is missing, "
    "rather than losing everything to a forced cutoff."
)


def sandbox_enabled() -> bool:
    """True when the run_code_sandbox tool should be offered to the LLM.

    Controlled by SANDBOX_ENABLED (set from the helm chart's sandbox.enabled,
    or .env locally); defaults to enabled.
    """
    raw = os.environ.get("SANDBOX_ENABLED", "1")
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def sandbox_image() -> str:
    """Container image for the sandbox workspace (pulled once, then cached)."""
    return os.environ.get("SANDBOX_IMAGE", DEFAULT_PYTHON_SANDBOX_IMAGE)


def sandbox_model() -> str:
    """Model id for the nested sandbox agent (SANDBOX_MODEL; default: the
    main bot's MODEL — i.e. the local llama.cpp model unless SANDBOX_MODEL
    points the sandbox at a different OpenAI-compatible API, e.g.
    OpenRouter)."""
    return env_or("SANDBOX_MODEL", "MODEL", DEFAULT_MODEL)


def sandbox_llm_host() -> str:
    """Base URL of the LLM the sandbox agent talks to (SANDBOX_LLM_HOST;
    default: the main bot's LLM_HOST). The core appends /v1 itself, so this
    must not include it (OpenRouter: https://openrouter.ai/api)."""
    return env_or("SANDBOX_LLM_HOST", "LLM_HOST", DEFAULT_LLM_HOST)


def sandbox_llm_api_key() -> str:
    """API key for the sandbox agent's LLM (SANDBOX_LLM_API_KEY; default:
    the main bot's LLM_PASS placeholder)."""
    return env_or("SANDBOX_LLM_API_KEY", "LLM_PASS", DEFAULT_LLM_API_KEY)


def _positive_int(raw: str | None, default: int) -> int:
    try:
        value = int(str(raw).strip())
        return value if value > 0 else default
    except (TypeError, ValueError):
        return default


def _non_negative_int(raw: str | None, default: int) -> int:
    try:
        value = int(str(raw).strip())
        return value if value >= 0 else default
    except (TypeError, ValueError):
        return default


def sandbox_max_turns() -> int:
    """Max model turns for one sandbox task (SANDBOX_MAX_TURNS, default 10)."""
    return _positive_int(os.environ.get("SANDBOX_MAX_TURNS"), DEFAULT_MAX_TURNS)


def sandbox_timeout() -> int:
    """Wall-clock seconds before a sandbox task is stopped (SANDBOX_TIMEOUT,
    default 600)."""
    return _positive_int(os.environ.get("SANDBOX_TIMEOUT"), DEFAULT_TIMEOUT_SECONDS)


def sandbox_ask_user_timeout() -> int:
    """Max seconds the ask_user tool waits for a reply before telling the
    model to proceed on its own (SANDBOX_ASK_USER_TIMEOUT, default 300). Any
    single call also never waits past the run's own remaining
    sandbox_timeout() budget — see ask_user's `deadline` handling."""
    return _positive_int(os.environ.get("SANDBOX_ASK_USER_TIMEOUT"), DEFAULT_ASK_USER_TIMEOUT_SECONDS)


def sandbox_persist_timeout() -> int:
    """Wall-clock seconds allowed to persist a thread's workspace snapshot on
    session.stop() (SANDBOX_PERSIST_TIMEOUT_SECONDS, default 180). This runs
    AFTER the sandbox_timeout() envelope has already returned, and — when the
    Memory capability is active — is also where its phase-one/phase-two
    memory extraction runs (two extra local-model calls), so it needs its own,
    separate budget rather than none at all."""
    return _positive_int(
        os.environ.get("SANDBOX_PERSIST_TIMEOUT_SECONDS"), DEFAULT_PERSIST_TIMEOUT_SECONDS
    )


def sandbox_request_timeout() -> int:
    """Seconds of silence on one HTTP request to the sandbox's LLM before the
    client gives up on it (SANDBOX_REQUEST_TIMEOUT_SECONDS, default 180),
    down from the OpenAI client's unstated 600s default.

    This is the HTTP client's per-read timeout, not a wall-clock bound on the
    call: it catches a connection that hangs (see
    SANDBOX_CONNECT_TIMEOUT_SECONDS) and a server that accepts the request
    then sends nothing, but NOT a server that dribbles keep-alive bytes while
    it works — OpenRouter pads non-streaming responses with whitespace, and
    every pad byte resets this timer. That case is caught by the separate
    wall-clock bound on the whole model call, sandbox_model_call_timeout(),
    which is derived from this value rather than configured on its own.
    """
    return _positive_int(
        os.environ.get("SANDBOX_REQUEST_TIMEOUT_SECONDS"), DEFAULT_REQUEST_TIMEOUT_SECONDS
    )


def sandbox_model_call_timeout() -> float:
    """Wall-clock seconds one model call may take in total, all HTTP retries
    included: (1 + SANDBOX_MAX_RETRIES) x SANDBOX_REQUEST_TIMEOUT_SECONDS.

    Applied as ModelSettings.timeout, which the SDK's runner enforces by
    cancelling the whole call (it wraps the OpenAI client's own retry loop,
    so it does not add to the worst case). It is deliberately the same
    figure that was already the documented worst case for a silent server,
    so the only behaviour change is that a server which keeps the read
    timeout alive with padding is now bounded by it too, instead of only by
    sandbox_timeout(). Exceeding it raises agents.ModelTimeoutError, which
    run_sandbox_task lets propagate exactly like the client's own
    APITimeoutError already did: run_code_sandbox reports both as "the
    sandbox itself failed".
    """
    return float((1 + sandbox_max_retries()) * sandbox_request_timeout())


def sandbox_max_retries() -> int:
    """How many times the sandbox's LLM client retries a failed request
    (SANDBOX_MAX_RETRIES, default 2). 0 disables retries — which is why this
    uses _non_negative_int rather than _positive_int: worst-case latency is
    (1 + retries) x sandbox_request_timeout(), so being able to say "none"
    is the point."""
    return _non_negative_int(os.environ.get("SANDBOX_MAX_RETRIES"), DEFAULT_MAX_RETRIES)


def _tolerant_tool_invoke(original_invoke):
    """Wraps a shell tool's on_invoke_tool so a malformed tool call (missing
    a required argument) becomes a model-visible error turn instead of
    aborting the whole run.

    Needed because both shell tools are built with strict_json_schema=False
    (see build_sandbox_agent's docstring: the grammar-enforced strict mode
    isn't available over ChatCompletions), so nothing stops the sandbox LLM
    — often a small local model — from emitting a tool call with required
    arguments missing entirely (e.g. exec_command with no `cmd`). The
    underlying *Tool._invoke does `args_model.model_validate_json(raw_input)`
    with no guard, so that surfaces as a raw pydantic ValidationError.
    FunctionTool.on_invoke_tool's own contract (see agents.tool) is explicit
    that raising fails the whole run while returning a string is fed back to
    the model as a normal tool result — confirmed in production, where an
    empty exec_command call propagated all the way out of Runner.run as
    UserError and aborted an otherwise-working sandbox task. We opt into the
    string-result half of that contract here so the model can just retry.

    Deliberately narrow to ValidationError: transport/session failures
    (ExecTransportError, a dead container, ...) must still raise, or a
    genuinely broken sandbox would spend every remaining turn being told
    to "try again" instead of failing fast.
    """
    from pydantic import ValidationError

    async def _invoke(ctx, raw_input):
        try:
            return await original_invoke(ctx, raw_input)
        except ValidationError as e:
            missing = [
                ".".join(str(p) for p in err["loc"])
                for err in e.errors()
                if err["type"] == "missing"
            ]
            if missing:
                detail = f"missing required argument(s): {', '.join(missing)}"
            else:
                detail = "arguments did not match the tool's schema"
            return f"Tool call rejected: {detail}. Retry with a corrected call."

    return _invoke


def _elapsed_note(context: dict) -> str:
    """The time-budget marker appended to every shell result, or "" when
    this run has no deadline (the nested tools are also used in tests and
    on paths that never set one).

    This exists so _BUDGET_BULLET can state the budget instead of teaching
    the model to measure it — see the comment above _BUDGET_BULLET for what
    that cost. Rounded to whole seconds: the model is pacing itself against
    an 80% threshold, not timing anything.
    """
    deadline = context.get("deadline")
    total = context.get("timeout_seconds")
    if deadline is None or not total:
        return ""
    used = int(total - max(0.0, deadline - time.monotonic()))
    return f"[time used: {used}s of {total}s]"


def _with_elapsed_note(original_invoke):
    """Wraps a shell tool's on_invoke_tool so each command reports how much
    of the run's time budget has been spent.

    A small local model will not remember to check a clock, but it cannot
    avoid reading the output of the command it just ran.

    Exception-safe by design: a failure here degrades to "no marker", never
    breaks a shell call the run depends on.
    """
    async def _invoke(ctx, raw_input):
        result = await original_invoke(ctx, raw_input)
        try:
            context = ctx.context if isinstance(getattr(ctx, "context", None), dict) else {}
            note = _elapsed_note(context)
            return f"{result}\n\n{note}" if note else result
        except Exception as e:
            logger.warning(f"Sandbox: could not attach the time marker to a tool result: {e}")
            return result

    return _invoke


def _conversation(ctx):
    """The run's SandboxConversation from a tool's context, or None (tests
    and any path that never set one — every wrapper then passes through)."""
    context = getattr(ctx, "context", None)
    return context.get("conversation") if isinstance(context, dict) else None


def _gated_invoke(original_invoke, tool_name: str, *, busy: bool = False):
    """Wraps a tool's on_invoke_tool so it only runs when no thread input is
    waiting for an answer — see sandbox_conversation's "side effects are
    gated". A refused call returns the refusal as its result and has no
    side effect. busy marks the run as executing while it runs, which is
    what lets main.py say "received, I'll review this before my next step"
    for a message that lands during a long command.
    """
    async def _invoke(ctx, raw_input):
        conversation = _conversation(ctx)
        if conversation is None:
            return await original_invoke(ctx, raw_input)
        return await conversation.run_gated(
            tool_name, lambda: original_invoke(ctx, raw_input), busy=busy)

    return _invoke


# Longest a shell call keeps the model waiting without a conversation
# checkpoint. The model's own yield_time_ms (exec_command defaults to 10s,
# and it may ask for far longer) is honoured, but in slices of this size:
# between slices the wrapper checks for thread input and, if something
# arrived, hands control back with the command still running. Slicing inside
# ONE tool call is deliberate — clamping yield_time_ms instead would turn
# every command longer than this into an extra write_stdin polling turn out
# of a budget as small as 10.
SHELL_CHECKPOINT_SECONDS = 5.0
_RUNNING_SESSION_RE = re.compile(r"Process running with session ID (\d+)")


def _with_shell_checkpoints(exec_invoke, poll_invoke):
    """Wraps exec_command so a long command yields at SHELL_CHECKPOINT_SECONDS
    intervals when new thread input is waiting, instead of only when the
    model's full yield_time_ms is up.

    Works entirely through the two public tool invokes: exec_command with its
    yield clamped to one slice, then write_stdin polls (empty input) on the
    session id the SDK reports, until the command exits, the requested wait
    is used up, or someone posts in the thread. The slices come back merged
    into one SDK-shaped result (_merge_shell_chunks). Without write_stdin (no
    PTY) or without a conversation this is a plain pass-through.
    """
    async def _invoke(ctx, raw_input):
        conversation = _conversation(ctx)
        if conversation is None or poll_invoke is None:
            return await exec_invoke(ctx, raw_input)
        try:
            args = json.loads(raw_input or "{}")
        except (TypeError, ValueError):
            return await exec_invoke(ctx, raw_input)
        if not isinstance(args, dict):
            return await exec_invoke(ctx, raw_input)
        try:
            requested = float(args.get("yield_time_ms", 10_000)) / 1000
        except (TypeError, ValueError):
            return await exec_invoke(ctx, raw_input)
        slice_s = SHELL_CHECKPOINT_SECONDS
        if requested <= slice_s:
            return await exec_invoke(ctx, raw_input)
        started = time.monotonic()
        first = dict(args, yield_time_ms=int(slice_s * 1000))
        chunks = [await exec_invoke(ctx, json.dumps(first))]
        note = ""
        while True:
            match = _RUNNING_SESSION_RE.search(str(chunks[-1]))
            if match is None:
                break
            if conversation.has_new_input():
                note = (f"[The command is still running as session {match.group(1)}. New "
                        "thread input arrived, so control is back with you early; poll it "
                        "later with write_stdin.]")
                break
            remaining = requested - (time.monotonic() - started)
            if remaining <= 0.05:
                break
            poll = {"session_id": int(match.group(1)), "chars": "",
                    "yield_time_ms": int(min(slice_s, remaining) * 1000)}
            if args.get("max_output_tokens") is not None:
                poll["max_output_tokens"] = args["max_output_tokens"]
            chunks.append(await poll_invoke(ctx, json.dumps(poll)))
        merged = _merge_shell_chunks(chunks)
        return f"{merged}\n\n{note}" if note else merged

    return _invoke


_SDK_OUTPUT_SEPARATOR = "\nOutput:\n"


def _merge_shell_chunks(chunks) -> str:
    """One SDK-shaped result from several slices of the same command: the
    LAST slice's header (so the exit code / still-running line is current)
    and every slice's output, in order. Keeps the result readable for the
    model and parseable by sandbox_progress.parse_exec_result, which expects
    exactly one header. Anything not in the SDK's shape is joined verbatim."""
    chunks = [str(c) for c in chunks]
    if len(chunks) == 1:
        return chunks[0]
    header, outputs = None, []
    for chunk in chunks:
        head, sep, out = chunk.partition(_SDK_OUTPUT_SEPARATOR)
        if not sep:
            return "\n\n".join(chunks)
        header = head
        outputs.append(out)
    return header + _SDK_OUTPUT_SEPARATOR + "".join(outputs)


def _configure_shell_tools(toolset) -> None:
    """configure_tools callback (see agents.sandbox.capabilities.Shell):
    wraps both shell tools so malformed calls become model-visible errors
    (see _tolerant_tool_invoke), every result reports the time used
    (_with_elapsed_note), a long command yields early when someone posts in
    the thread (_with_shell_checkpoints), and a NEW command is refused while
    thread input is unanswered (_gated_invoke).

    write_stdin is deliberately not gated: observing or stopping a command
    that is already running is exactly what the model may need to do while
    it reviews new input. Only starting new work is held back.

    Order matters: the gate is outermost, so a refused command runs nothing
    at all; the elapsed note wraps the checkpointed call once, so the time
    marker appears once at the end rather than after every slice.
    """
    tolerant_exec = _tolerant_tool_invoke(toolset.exec_command.on_invoke_tool)
    tolerant_poll = None
    if toolset.write_stdin is not None:
        tolerant_poll = _tolerant_tool_invoke(toolset.write_stdin.on_invoke_tool)
        toolset.write_stdin.on_invoke_tool = _with_elapsed_note(tolerant_poll)
    toolset.exec_command.on_invoke_tool = _gated_invoke(
        _with_elapsed_note(_with_shell_checkpoints(tolerant_exec, tolerant_poll)),
        "exec_command", busy=True,
    )


def _thread_name(task: str) -> str:
    """A Discord thread name derived from the task (Discord caps thread
    names at 100 chars)."""
    prefix = "🐳 "
    name = " ".join((task or "").split())
    if not name:
        return f"{prefix}Sandbox"
    limit = 100 - len(prefix)
    if len(name) > limit:
        name = name[: limit - 1] + "…"
    return f"{prefix}{name}"


async def ensure_sandbox_thread(original_message, task: str):
    """Resolves the Discord channel a sandbox run should post to.

    Returns (channel, thread_created). If original_message is already inside
    a thread, that thread is reused as-is (thread_created=False) — this
    covers both a sandbox-created thread and any other thread the user is
    @mentioning the bot in; either way it's safe, since sandbox_snapshot_id_for
    simply starts fresh for a thread that never ran a sandbox task before.
    A thread previously started off this very message is likewise reused.
    Otherwise a new thread is created off original_message (thread_created=
    True). If thread creation isn't possible (e.g. the channel type doesn't
    support threads, or a permissions error), the failure is caught and the
    original channel is returned unchanged (thread_created=False) so the tool
    degrades to today's behavior instead of failing outright — see
    sandbox_snapshot_id_for for why that fallback also means "no snapshot".
    """
    channel = original_message.channel
    if isinstance(channel, discord.Thread):
        return channel, False
    # Reuse a thread already started off this message. Discord permits
    # exactly one, so a second create_thread here fails with 160004 ("a
    # thread has already been created for this message") and we would fall
    # back to the parent channel — running outside the thread the earlier
    # attempt lives in, and unsnapshotted, since a non-thread channel is
    # never a snapshot key. That is observed behavior, not a hypothetical:
    # it is what happened when the outer model retried a timed-out run.
    #
    # Residual gap, not closed by this or by run_code_sandbox's own
    # context["sandbox_thread"] check: the SDK runs one turn's tool calls
    # concurrently, so if the outer model emits TWO run_code_sandbox calls in a
    # single response both read no recorded thread and race. The loser gets
    # 160004 with `.thread` not yet in the guild cache (THREAD_CREATE has not
    # landed), so it still falls back to the parent channel — where in_thread
    # is False and the is_run_active guard cannot see it either.
    existing = getattr(original_message, "thread", None)
    if isinstance(existing, discord.Thread):
        return existing, False
    try:
        thread = await original_message.create_thread(name=_thread_name(task))
    except Exception as e:
        logger.warning(f"Sandbox: could not create a thread, falling back to the channel: {e}")
        return channel, False
    return thread, True


def sandbox_snapshot_id_for(channel) -> str | None:
    """The Redis snapshot id for a sandbox run's target channel, or None
    when there is no thread to key a snapshot on. A plain (non-thread)
    channel id is deliberately never used as a fallback key — that would
    wrongly link unrelated tasks posted in the same channel into one
    workspace lineage; no thread means no snapshot, matching the fully
    disposable behavior this tool had before threads existed."""
    return str(channel.id) if isinstance(channel, discord.Thread) else None


async def sandbox_snapshot_exists(snapshot_id: str | None) -> bool:
    """Whether a saved workspace exists for this snapshot id — i.e. whether
    session.start() will restore one rather than starting empty.

    The SDK gives the caller no way to ask after the fact: session.start()
    returns None, and state.snapshot_fingerprint is written on persist only,
    never on restore (verified against the pinned openai-agents version). But
    the restore condition is exactly `await snapshot.restorable(...)`, which
    for our RemoteSnapshot is exactly this store's exists() — so asking it
    ourselves, before the run starts, gives the same answer the SDK will act
    on moments later.

    Any error answers False: this only drives a status badge, and a wrong
    badge must never break a run.
    """
    if snapshot_id is None:
        return False
    try:
        from classes.sandbox_snapshot_store import SandboxSnapshotStore

        return bool(await SandboxSnapshotStore().exists(snapshot_id))
    except Exception as e:
        logger.warning(f"Sandbox: could not check for a saved workspace snapshot: {e}")
        return False


def sandbox_workspace_note(resumed: bool) -> str:
    """The one-line origin badge shown at the top of a run's embed, so the
    thread says plainly whether this run built on the last one or started
    over — otherwise the two are indistinguishable from the outside."""
    if resumed:
        return "♻️ **Resumed** — this thread's saved workspace and memory were restored."
    return "🆕 **Fresh sandbox** — empty workspace, nothing carried over."


async def sandbox_snapshot_remaining_seconds(snapshot_id: str | None) -> int | None:
    """How much longer this thread's saved workspace can be resumed, or None
    when there is nothing saved to resume.

    Read from Redis rather than computed from sandbox_snapshot_ttl_seconds()
    on purpose: persisting is best-effort and swallowed
    (_persist_sandbox_snapshot), so the configured TTL would happily promise
    a resume for a snapshot that was never written. Redis answers -2 for a
    missing key, which becomes None here — as does any error, since this
    only drives a closing note and must never break a run.
    """
    if snapshot_id is None:
        return None
    try:
        from classes.sandbox_snapshot_store import SandboxSnapshotStore

        remaining = await SandboxSnapshotStore().ttl(snapshot_id)
    except Exception as e:
        logger.warning(f"Sandbox: could not read the saved workspace's expiry: {e}")
        return None
    return remaining if remaining > 0 else None


def format_resume_window(seconds: int) -> str:
    """A rough, human-readable "for another ..." duration. Deliberately
    coarse — this is a reassurance in a closing note, not a countdown, so
    one unit is enough and rounding down never over-promises."""
    if seconds >= 172_800:  # 2 days
        return f"{seconds // 86_400} days"
    if seconds >= 86_400:
        return "1 day"
    if seconds >= 7_200:
        return f"{seconds // 3_600} hours"
    if seconds >= 3_600:
        return "1 hour"
    minutes = max(1, seconds // 60)
    return "1 minute" if minutes == 1 else f"{minutes} minutes"


def sandbox_closing_note(
    remaining_seconds: int | None, in_thread: bool, outcome: str = "",
) -> str:
    """The description of the embed posted when a sandbox run is over.

    `outcome` is a one-line reason the run ended, shown first and passed only
    when it did NOT finish normally (see tool_functions.finalize_notes). With
    live progress off this embed is the only end-of-run signal in the thread,
    so without it a run that timed out looks exactly like one that succeeded
    and simply produced no file.

    It marks the boundary between "messages here steer the running sandbox"
    and "messages here are ordinary chat again" — without it the only signal
    is the absence of a 📨 reaction, which is invisible until you try.

    It also states where a resume is possible, because that is not
    guessable: the snapshot is keyed on the thread's own id
    (sandbox_snapshot_id_for) and ensure_sandbox_thread creates a NEW thread
    for a request made anywhere else, so asking in the parent channel starts
    a fresh sandbox rather than reopening this one. (The one exception is
    invisible to the user: a retry within the same outer turn reuses the
    thread that turn already opened, instead of being pushed out into the
    parent channel.)

    One imprecision, accepted: remaining_seconds says A workspace is
    resumable, not that THIS run's workspace is. Persisting is best-effort
    and swallowed (_persist_sandbox_snapshot), so on a second-or-later run
    whose persist failed, the window shown belongs to the PREVIOUS run's
    snapshot. The note stays true — you can still resume, just to that
    earlier state — and the first run in a thread has no older key to fall
    back on, so it correctly reports that nothing was saved.
    """
    head = f"{outcome}\n" if outcome else ""
    if not in_thread:
        return head + "🐳 Sandbox closed. It left nothing behind — the next request starts fresh."
    if remaining_seconds is None:
        return head + ("🐳 Sandbox closed. Its workspace could **not** be saved, so a "
                       "follow-up here will start from scratch.")
    return head + (
        f"🐳 Sandbox closed — messages here are back to normal chat.\n"
        f"Its workspace was saved for another **{format_resume_window(remaining_seconds)}**: "
        f"@mention me *in this thread* to carry on from where it left off. "
        f"Asking anywhere else always starts a brand-new sandbox."
    )


RESUMED_TASK_PREAMBLE = (
    "This sandbox was restored from an earlier run in this same thread — the "
    "workspace is NOT empty. Run `ls -R .` first and build on what is already "
    "there instead of redoing it.\n\n"
)


def _ask_user_context(wrapper: RunContextWrapper[dict]) -> dict:
    return wrapper.context if isinstance(wrapper.context, dict) else {}


@function_tool
async def ask_user(
    wrapper: RunContextWrapper[dict],
    question: str,
    blocking: bool = True,
    choices: list[str] | None = None,
    default: str = "",
) -> str:
    """Ask the user who requested this task a question in the Discord thread.

    Args:
        question: A specific, self-contained question about a choice that
            changes the result. Not a request for approval of routine steps.
        blocking: True pauses all other work until they reply (or the wait
            times out) — only for information you truly cannot proceed
            without. False posts it and returns at once so you keep working.
        choices: Optional short suggested answers.
        default: What you will do if nobody answers. Strongly recommended
            when blocking is false.
    """
    ctx = _ask_user_context(wrapper)
    conversation = ctx.get("conversation")
    if ctx.get("thread") is None or conversation is None:
        return "No interactive thread is available right now — proceed using your best judgement."

    timeout = float(sandbox_ask_user_timeout())
    deadline = ctx.get("deadline")
    if deadline is not None:
        remaining = deadline - time.monotonic()
        # Never wait past this run's own remaining sandbox_timeout() budget —
        # a question asked late in a run must not outlive the outer timeout
        # that will cut the whole run off anyway.
        timeout = max(1.0, min(timeout, remaining))
    # One pipeline for answers: the reply is a thread message like any other,
    # recorded by the ledger and shown to the model at its next call. The old
    # tool waited on client.wait_for, a SECOND consumer of the same message
    # that needed de-duplicating against the inbox, accepted any requester
    # message as "the answer" even when it was a correction, and could miss
    # a reply that landed before its waiter was installed.
    return await conversation.ask(
        question, blocking=blocking, choices=choices, default=(default or "").strip(),
        timeout=timeout,
    )


@function_tool
async def respond_to_updates(
    wrapper: RunContextWrapper[dict],
    message_ids: str | int | list[str | int],
    decision: str,
    reply: str = "",
) -> str:
    """Respond to thread message(s) you have been shown. Your reply is posted
    in the thread. Call it before any other work — until you do, your other
    tools refuse to run — and again with decision "done" once an accepted
    change is really made.

    Args:
        message_ids: The message number(s), e.g. "3" or "3, 4". Several
            messages with the same decision can share one call.
        decision: One of: will_apply (you will make this change), done (the
            change is made), answered (you answered their question in
            reply), noted (information you have taken into account),
            declined (you will not, and reply says why), superseded
            (replaced by a later message), deferred (for a follow-up run).
        reply: One or two sentences TO THE USER saying how you are handling
            it, e.g. "Keeping the layout, switching the bars to blue."
    """
    # message_ids also takes a bare number or a list: small models send
    # either, and a pydantic rejection here would drop the acknowledgement
    # while on_model_response had already let the response's gated tools
    # through on the strength of it. parse_ids copes with all three shapes.
    conversation = _ask_user_context(wrapper).get("conversation")
    if conversation is None:
        return "No thread is attached to this run, so there is nothing to respond to."
    return await conversation.respond(message_ids, decision, reply)


@function_tool
async def say_in_thread(wrapper: RunContextWrapper[dict], text: str) -> str:
    """Post a short message to the Discord thread and carry on immediately;
    it does not wait for a reply. Use it when an approach fails and you are
    trying another — not to narrate every command, and not to answer thread
    messages (that is respond_to_updates).

    Args:
        text: The message to post. Plain text, a sentence or two.
    """
    ctx = _ask_user_context(wrapper)
    thread = ctx.get("thread")
    if thread is None:
        return "No thread is attached to this run — nothing was sent."
    text = (text or "").strip()
    if not text:
        return "Nothing to send (the message was empty)."
    try:
        await thread.send(f"🐳 {text[:1900]}")
    except Exception as e:
        logger.warning(f"Sandbox say_in_thread: failed to send: {e}")
        return "Could not post that to the thread — carry on with the task."
    return "Posted to the thread."


@function_tool
async def send_preview_to_thread(
    wrapper: RunContextWrapper[dict], path: str, caption: str = ""
) -> str:
    """Send a file to the Discord thread right now, without ending the task
    — for showing a partial or in-progress result (a draft plot, an
    intermediate file) the user can give feedback on.

    Args:
        path: Path to the file, relative to your current working directory
            (never an absolute path).
        caption: Optional short caption to send alongside the file.
    """
    ctx = _ask_user_context(wrapper)
    thread = ctx.get("thread")
    session = ctx.get("session")
    if thread is None or session is None:
        return "No thread is available to preview to right now."
    if path.startswith("/"):
        return "Use a path relative to your working directory, not an absolute one."

    try:
        result = await session.exec("cat", "--", path, shell=False)
    except Exception as e:
        logger.warning(f"Sandbox send_preview_to_thread: exec failed: {e}")
        return f"Could not read {path!r} to preview it."
    if not result.ok():
        return f"Could not read {path!r} — check the path and try again."
    data = result.stdout
    if len(data) > MAX_ARTIFACT_BYTES:
        return (
            f"{path!r} is too large to preview ({len(data)} bytes, over the "
            f"{MAX_ARTIFACT_BYTES}-byte limit)."
        )

    name = _attachment_name(path)
    try:
        await thread.send(content=caption or None, file=discord.File(io.BytesIO(data), filename=name))
    except Exception as e:
        logger.warning(f"Sandbox send_preview_to_thread: send failed: {e}")
        return f"Read {path!r} but could not send it to the thread."
    return f"Sent {name!r} to the thread as a preview."


@function_tool
async def attach_file(
    wrapper: RunContextWrapper[dict], path: str, caption: str = "", remove: bool = False,
) -> str:
    """Mark a finished file to be delivered to the user when the task ends.
    Only files you attach are sent. Attach the same path again to replace an
    earlier version of it; don't attach your intermediate attempts.

    Args:
        path: Path to the file, relative to your current working directory
            (never an absolute path).
        caption: Optional short caption sent alongside this file.
        remove: True un-attaches a file you attached earlier that is now
            outdated (e.g. after the user changed what they want).
    """
    ctx = _ask_user_context(wrapper)
    session = ctx.get("session")
    if session is None:
        return "No sandbox session is available, so nothing can be attached."
    path = (path or "").strip()
    if not path:
        return "No path given — nothing was attached."
    if path.startswith("/"):
        return "Use a path relative to your working directory, not an absolute one."

    deliverables = ctx.get("deliverables")
    name = _attachment_name(path)
    if remove:
        if deliverables is None:
            return "This run cannot attach files."
        before = len(deliverables)
        deliverables[:] = [entry for entry in deliverables if entry["path"] != path]
        if len(deliverables) == before:
            return f"{name!r} was not attached, so there was nothing to remove."
        return f"Removed {name!r}; it will not be sent."

    # stat, not `test -f` then `wc -c`: one call answers both "does it exist"
    # and "how big is it", and the size is what the caps below need. GNU
    # coreutils is already a hard assumption here — _collect_artifacts uses
    # `find -printf`, which is GNU findutils.
    try:
        result = await session.exec("stat", "-c", "%s", "--", path, shell=False)
    except Exception as e:
        logger.warning(f"Sandbox attach_file: stat failed for {path!r}: {e}")
        return f"Could not check {path!r}. Verify it exists with `ls`, then attach it again."
    if not result.ok():
        return (
            f"{path!r} does not exist, so it was NOT attached. Check the path "
            "with `ls` — nothing is sent to the user unless it is attached."
        )
    try:
        size = int(result.stdout.decode("utf-8", errors="replace").strip())
    except ValueError:
        return f"Could not read the size of {path!r}. Verify it with `ls -l` and try again."
    if size > MAX_ARTIFACT_BYTES:
        return (
            f"{path!r} is too large to send ({size} bytes, over the "
            f"{MAX_ARTIFACT_BYTES}-byte limit). Produce a smaller file and attach that."
        )

    if deliverables is None:
        return "This run cannot attach files."
    caption = (caption or "").strip()
    # Which accepted changes this selection already reflects: an attachment
    # older than the latest accepted change, sitting next to one newer than
    # it, is probably the pre-change variant — see continuation_nudge.
    conversation = ctx.get("conversation")
    change_revision = conversation.ledger.change_revision if conversation is not None else 0
    for entry in deliverables:
        # Re-attaching a path is how the model says "I fixed it", so the new
        # version replaces the old rather than the user receiving both.
        if entry["path"] == path:
            entry.update(size=size, caption=caption, change_revision=change_revision)
            return f"Replaced the attached {name!r}; it will be sent when you finish."
    if len(deliverables) >= MAX_ARTIFACT_FILES:
        return (
            f"You already have {MAX_ARTIFACT_FILES} files attached, which is the "
            "limit. Attach only the finished results the user asked for."
        )
    deliverables.append({"path": path, "size": size, "caption": caption,
                         "change_revision": change_revision})
    return f"Attached {name!r}; it will be sent to the user when you finish."


# Gated like a new shell command: choosing what is delivered, or showing a
# preview, on the basis of a plan the user has since changed is exactly the
# stale action the gate exists to stop. Wrapped once, here: the wrapper reads
# the run's conversation from the tool context at call time, so one module-
# level tool serves every run (and passes straight through without one).
attach_file.on_invoke_tool = _gated_invoke(attach_file.on_invoke_tool, "attach_file")
send_preview_to_thread.on_invoke_tool = _gated_invoke(
    send_preview_to_thread.on_invoke_tool, "send_preview_to_thread")


def build_sandbox_agent(out_dir: str | None, conversation=None) -> "object":
    """The nested SandboxAgent that does the work inside the sandbox.

    out_dir: whether the sandbox's real output path was already resolved
    and mkdir -p'd by the caller (see run_sandbox_task) — only used to pick
    between wording variants (both say "out/", never the absolute value:
    see SANDBOX_OUTPUT_DIRNAME comment for why). None means resolution
    failed and out/ hasn't been created yet, so the model is told to
    create it itself.

    Also tells the model its turn/time budget (sandbox_max_turns() /
    sandbox_timeout(), i.e. SANDBOX_MAX_TURNS / SANDBOX_TIMEOUT) so it can
    pace itself and save partial output before a forced cutoff — see
    _BUDGET_BULLET.

    Uses the same LLM as the main agent by default (MODEL / LLM_HOST /
    LLM_PASS); SANDBOX_MODEL / SANDBOX_LLM_HOST / SANDBOX_LLM_API_KEY point
    it at a different OpenAI-compatible API instead (e.g. an OpenRouter
    model — see .env.example / the chart's sandbox.model/llmHost/apiKey).
    No default manifest, so the workspace starts empty.

    Capabilities: Shell ONLY. The Filesystem capability adds `apply_patch`, a
    grammar (hosted) tool that the ChatCompletions API — what llama.cpp serves
    — does not support (the converter raises "Hosted tools are not supported").
    exec_command already gives full filesystem access (heredocs, cat, pip, …),
    which is the pattern the SDK's own docker_runner.py example uses.

    base_instructions is an empty string ON PURPOSE: with the default (None)
    the SDK injects its bundled computer-use prompt, which repeatedly tells
    the model to use the apply_patch tool — which we do not expose. A model
    that obeys it aborts the run with ModelBehaviorError ("Model produced
    apply_patch call without an apply_patch tool"). Empty string suppresses
    the default entirely; our instructions + the Shell capability's own
    instructions are all the model needs.

    Capabilities also include Memory (persistent, distilled memory of prior
    runs — see the module docstring): configured with live_update=False so
    its own capability requirement stays Shell-only (the default,
    live_update=True, requires Filesystem — the same apply_patch problem
    above), and with its phase-one/phase-two extraction models pointed at
    THIS SAME local/self-hosted LLM rather than the SDK's hosted-OpenAI
    defaults ("gpt-5.4-mini"/"gpt-5.5"), which this bot has no API key
    configured for and must never silently send sandbox content to.

    tools (ask_user, respond_to_updates, attach_file, ...) are in addition
    to the Shell capability's own tools — SandboxAgent is a dataclass
    subclass of Agent and accepts tools=[...] the same way. The thread tools
    no-op gracefully (return a string telling the model to proceed on its
    own) when no Discord thread context was passed into this run — see
    run_sandbox_task's nested_context.

    conversation: the run's SandboxConversation, whose AgentHooks commit
    "this input was seen" after each completed model call (see
    classes/sandbox_conversation.py). None builds an agent without it
    (tests; the prompt and tools are identical either way).
    """
    from agents.sandbox import MemoryGenerateConfig, MemoryReadConfig, SandboxAgent
    from agents.sandbox.capabilities import Memory, Shell

    output_bullet = _OUTPUT_BULLET_RESOLVED if out_dir is not None else _OUTPUT_BULLET_FALLBACK
    budget_bullet = _BUDGET_BULLET.format(
        max_turns=sandbox_max_turns(), timeout_seconds=sandbox_timeout()
    )
    sandbox_llm = OpenAIChatCompletionsModel(
        model=sandbox_model(),
        openai_client=AsyncOpenAI(
            base_url=sandbox_llm_host() + "/v1",
            api_key=sandbox_llm_api_key(),
            # Both stated rather than left to the client's defaults (600s,
            # 2 retries), which together let one silent request spend 30
            # minutes before the run reports anything. Connect gets its own
            # short budget so an unreachable host fails fast instead of
            # waiting out the read. Note the read timeout does not bound a
            # server that keeps sending keep-alive padding — that is what
            # ModelSettings.timeout below is for; see sandbox_request_timeout()
            # for what each does and does not catch.
            timeout=httpx2.Timeout(
                float(sandbox_request_timeout()), connect=SANDBOX_CONNECT_TIMEOUT_SECONDS
            ),
            max_retries=sandbox_max_retries(),
        ),
    )

    return SandboxAgent(
        name="Code Sandbox",
        model=sandbox_llm,
        instructions=SANDBOX_INSTRUCTIONS.format(
            output_bullet=output_bullet, budget_bullet=budget_bullet
        ),
        base_instructions="",
        capabilities=[
            Shell(configure_tools=_configure_shell_tools),
            Memory(
                read=MemoryReadConfig(live_update=False),
                generate=MemoryGenerateConfig(
                    phase_one_model=sandbox_llm,
                    phase_two_model=sandbox_llm,
                ),
            ),
        ],
        tools=[respond_to_updates, ask_user, attach_file, send_preview_to_thread,
               say_in_thread],
        hooks=make_agent_hooks(conversation) if conversation is not None else None,
        # Slightly cooler than the chat agent: code tasks want determinism.
        # timeout: the wall-clock bound on each model call that the client's
        # read timeout above cannot provide (see sandbox_model_call_timeout).
        model_settings=ModelSettings(temperature=0.5, timeout=sandbox_model_call_timeout()),
    )


@dataclass
class SandboxArtifact:
    """One file being delivered to the user: normally one the sandbox agent
    chose with attach_file, otherwise one swept out of SANDBOX_OUTPUT_DIR by
    the fallback path. caption is whatever the agent passed to attach_file
    (empty for a swept file), sent as the message text beside it."""
    name: str
    data: bytes
    caption: str = ""


@dataclass
class SandboxResult:
    """What run_sandbox_task returns: the agent's final message to the user
    (text) plus the files it chose to deliver with attach_file — or, when it
    chose none, whatever the out/ fallback sweep found (empty if nothing).

    text is written FOR the user (see SANDBOX_INSTRUCTIONS' final bullet):
    the caller posts it in the thread beside the files as well as handing it
    to the outer model.

    ok is False when the run was stopped before producing a final report:
    error identifies why, as one of "timeout" (wall-clock budget exceeded),
    "max_turns" (turn budget exceeded) or "model_error" (the nested model
    misbehaved, e.g. an invalid tool call or an outright refusal). In all
    three cases artifacts may still be populated (best-effort recovery — see
    run_sandbox_task) and text is empty since Runner.run never returned a
    final report. Every other failure (docker/session errors, UserError,
    ...) still raises instead of returning ok=False — those mean the
    sandbox itself is unusable, not that the task ran out of budget.

    resumed is the GROUND TRUTH of whether this run started from the
    thread's saved workspace, which can differ from what the caller predicted
    with sandbox_snapshot_exists(): a snapshot that existed but could not be
    restored is dropped and the run retried empty (see
    _create_sandbox_session), so the caller can correct a badge it already
    posted rather than leaving a wrong one standing.

    skipped_artifacts holds a human-readable reason for each file that was
    attached (or found under out/) but NOT fetched (over MAX_ARTIFACT_FILES /
    MAX_ARTIFACT_BYTES) — populated on both success and failure, empty when
    nothing was skipped, so the caller can tell the model what was dropped
    and why instead of it going unexplained.

    conversation is what happened to each thread message during the run —
    SandboxConversation's ledger outcomes (author, text, status, reply) — so
    the outer model is told what was actually handled rather than assuming
    every message was applied. unsent_replies are acknowledgements the model
    wrote that Discord refused at the time; the caller posts them with the
    closing note so none is silently lost."""
    text: str
    artifacts: list[SandboxArtifact] = field(default_factory=list)
    ok: bool = True
    error: str | None = None
    skipped_artifacts: list[str] = field(default_factory=list)
    resumed: bool = False
    conversation: list[dict] = field(default_factory=list)
    unsent_replies: list[str] = field(default_factory=list)


def build_sandbox_client() -> "object":
    """A fresh Docker sandbox client bound to the local daemon, wired to
    persist/restore per-thread workspace snapshots through Redis
    (classes.sandbox_snapshot_store.SandboxSnapshotStore) under
    SANDBOX_SNAPSHOT_DEP_KEY — see _create_sandbox_session for how a
    snapshot id ties a client.create() call to one Discord thread."""
    from docker import from_env as docker_from_env
    from agents.sandbox.sandboxes.docker import DockerSandboxClient
    from agents.sandbox.session import Dependencies

    from classes.sandbox_snapshot_store import SandboxSnapshotStore

    dependencies = Dependencies().bind_value(SANDBOX_SNAPSHOT_DEP_KEY, SandboxSnapshotStore())
    return DockerSandboxClient(docker_from_env(), dependencies=dependencies)


async def _create_sandbox_session(sandbox_client, snapshot_id: str | None,
                                  resumed: bool = False):
    """Creates AND STARTS one container for this run.

    The explicit session.start() call (distinct from the raw Docker
    container.start() that client.create() already does internally) matters
    for two things, both verified live against a real Docker daemon and the
    pinned SDK version (docker.from_env() + DockerSandboxClient, no mocks):
    it is what actually materializes the manifest root (/workspace) and sets
    it as the session's tracked working directory (a raw session.exec("pwd")
    on an un-started session returns the image's own default, e.g. "/", and
    /workspace never gets created at all) — and, when a snapshot is
    restorable, it is where hydration into that materialized root actually
    happens (create() alone does not hydrate anything; without start(), a
    "resumed" session came back completely empty in that same live check,
    even though the snapshot was uploaded correctly). Skipping this call
    (the original state of this function) meant BOTH the model's own work
    and any prior snapshot content lived somewhere session.stop()'s
    persist step never looks at, silently making resume/Memory persistence
    into no-ops despite the plumbing otherwise looking correct — caught by
    that live check, not by any mocked unit test.

    When snapshot_id is given (the run is happening in a Discord thread —
    see ensure_sandbox_thread/sandbox_snapshot_id_for), the session is seeded
    from that thread's saved workspace snapshot in Redis if one exists (a
    prior run in the same thread), via a RemoteSnapshot bound to the SAME id
    both here and in _persist_sandbox_snapshot — this is what lets "continue
    this in the thread" pick back up instead of starting from scratch.
    snapshot_id is None for a one-off run with no thread: passing
    snapshot=None to client.create() resolves to a true no-op snapshot
    (verified against the SDK: DockerSandboxClient.create's
    resolve_snapshot(None, ...) builds a NoopSnapshot, never a hidden local
    default) — today's fully disposable behavior, unchanged.

    We create the session ourselves — rather than letting RunConfig create
    one automatically — so we retain a live handle to it after Runner.run
    returns and can read files out of it before it is torn down. See
    build_sandbox_run_config for the other half of this.
    """
    from agents.sandbox.sandboxes.docker import DockerSandboxClientOptions
    from agents.sandbox.snapshot import RemoteSnapshot

    def _snapshot():
        if snapshot_id is None:
            return None
        return RemoteSnapshot(id=snapshot_id, client_dependency_key=SANDBOX_SNAPSHOT_DEP_KEY)

    async def _create():
        return await sandbox_client.create(
            options=DockerSandboxClientOptions(image=sandbox_image()),
            snapshot=_snapshot(),
        )

    session = await _create()
    try:
        await session.start()
    except Exception as e:
        # Only start() is guarded, and only when a restore was actually
        # expected: a Docker-daemon problem fails in create() above and is
        # therefore outside this branch, so a failure here with a restorable
        # snapshot is overwhelmingly the restore itself (a truncated or
        # corrupt tar surfaces from hydrate_workspace's untar, NOT as
        # SnapshotRestoreError — RemoteSnapshot only wraps download(), and
        # our store's download() never raises — so this cannot be narrowed
        # to a specific exception class).
        if not (resumed and snapshot_id is not None):
            raise
        logger.warning(f"Sandbox: could not restore this thread's saved workspace, starting fresh: {e}")
        await _delete_sandbox_session(sandbox_client, session)
        # Delete the bad key BEFORE retrying, for two reasons: it is what
        # makes the retry work at all (exists() is now False, so start()
        # stops trying to restore it), and it un-bricks the thread — every
        # later run would otherwise hit the same corrupt snapshot until its
        # TTL expired. Retrying with snapshot=None instead would "work" once
        # and then quietly disable persistence forever, since a NoopSnapshot
        # makes the teardown session.stop() save nothing.
        await _delete_sandbox_snapshot(snapshot_id)
        session = await _create()
        await session.start()
        return session, False
    return session, resumed


def build_sandbox_run_config(client, session, input_filter=None, group_id=None) -> RunConfig:
    """Run config that reuses an already-created container for the run.

    Passing a live `session` (rather than just `client`/`options`) makes
    the SDK treat it as caller-owned: RunConfig(sandbox=...) makes the SDK
    run the agent inside it, but it will NOT stop or delete the container
    afterward — that becomes our responsibility (see _delete_sandbox_session),
    which is what gives us the window to read artifacts out of it first.
    It is also what makes a continuation safe: a second Runner.run on the
    same caller-owned session skips session.start() (the SDK only starts a
    session it does not own if it is not already running) and never tears
    it down, so the workspace, cwd and any running process carry over.

    input_filter is SandboxConversation.input_filter — the single path by
    which thread messages reach the model (see sandbox_conversation.py).

    group_id must be the same for every pass of one run. The Memory
    capability files each Runner.run under a "rollout" keyed by it, and with
    none set the SDK makes a fresh id per call — so every continuation would
    become its own rollout and cost its own extraction model call inside the
    persist budget at teardown. One id appends the passes to one rollout.
    """
    from agents.sandbox import SandboxRunConfig

    return RunConfig(
        sandbox=SandboxRunConfig(client=client, session=session),
        call_model_input_filter=input_filter,
        group_id=group_id,
    )


async def _delete_sandbox_session(sandbox_client, session) -> None:
    """Tears down a container we own, tolerating our own cancellation.

    Wrapped in asyncio.shield so that if the caller of run_sandbox_task is
    itself cancelled (e.g. process shutdown), this delete keeps running in
    the background instead of being cancelled mid-teardown and orphaning
    the container; wait_for still bounds how long we wait for it here.
    """
    try:
        await asyncio.wait_for(
            asyncio.shield(sandbox_client.delete(session)),
            timeout=SANDBOX_DELETE_TIMEOUT_SECONDS,
        )
    except Exception as e:
        logger.warning(f"Sandbox: failed to delete session/container: {e}")


async def _persist_sandbox_snapshot(session, snapshot_id: str | None) -> None:
    """Persists the workspace back to Redis (via session.stop(), which
    archives the workspace through whatever snapshot the session was created
    with — see _create_sandbox_session) so a later run in the same thread can
    resume from it. A no-op when snapshot_id is None (no thread, or thread
    creation failed — see ensure_sandbox_thread/sandbox_snapshot_id_for):
    today's fully disposable behavior is unchanged in that case.

    Time-boxed and best-effort, same rationale as _delete_sandbox_session:
    when the Memory capability is active, session.stop() is also where its
    phase-one/phase-two extraction runs (two extra local-model calls) —
    entirely outside the sandbox_timeout() envelope already spent on the run
    itself — so a slow or failed persist must never block returning the
    sandbox's result to the user.
    """
    if snapshot_id is None:
        return
    try:
        await asyncio.wait_for(session.stop(), timeout=sandbox_persist_timeout())
    except Exception as e:
        logger.warning(f"Sandbox: failed to persist workspace snapshot {snapshot_id!r}: {e}")


async def _delete_sandbox_snapshot(snapshot_id: str | None) -> None:
    """Drops a thread's saved workspace from Redis. Used only by
    _create_sandbox_session's recovery path, when the snapshot turned out not
    to be restorable. Best-effort: if the delete itself fails the retry below
    will fail too and propagate, which is the correct outcome — but a failure
    here must not mask the original restore error in the logs."""
    if snapshot_id is None:
        return
    try:
        from classes.sandbox_snapshot_store import SandboxSnapshotStore

        await SandboxSnapshotStore().delete(snapshot_id)
    except Exception as e:
        logger.warning(f"Sandbox: failed to drop unrestorable snapshot {snapshot_id!r}: {e}")


def _parse_find_output(raw: str) -> list[tuple[str, int]]:
    """Parses `find OUT_DIR -maxdepth 3 -type f -printf '%s %p\\n'` output
    into (container_path, size_bytes) pairs. Pure: no Docker access."""
    entries: list[tuple[str, int]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        size_str, _, path = line.partition(" ")
        if not path:
            continue
        try:
            entries.append((path, int(size_str)))
        except ValueError:
            continue
    return entries


def _relative_artifact_name(container_path: str, out_dir: str) -> str:
    """The Discord-safe filename for a file found under out_dir: its path
    relative to out_dir, with subdirectory separators flattened (a Discord
    attachment name is a single path segment). Pure."""
    if container_path.startswith(out_dir):
        rel = container_path[len(out_dir):].lstrip("/")
    else:
        rel = container_path.lstrip("/")
    rel = rel.replace("/", "_")
    return rel or "artifact"


def _attachment_name(container_path: str) -> str:
    """The Discord-safe filename for a file the agent named itself: its
    basename, since a Discord attachment name is a single path segment. Pure.

    Distinct from _relative_artifact_name, which flattens a path relative to
    out/ so two files with the same basename in different subdirectories stay
    distinguishable in a blind sweep. Here the agent picked the file by name,
    so `out/milk.png` should arrive as `milk.png`, not `out_milk.png`.
    """
    name = posixpath.basename(container_path.rstrip("/"))
    return name or "artifact"


def _select_deliverables(
    entries: list[dict],
    *,
    max_files: int = MAX_ARTIFACT_FILES,
    max_bytes: int = MAX_ARTIFACT_BYTES,
) -> tuple[list[dict], list[str]]:
    """Applies the file-count/size caps to what the agent attached.

    The same caps _select_artifacts applies to a blind sweep, re-applied here
    because attach_file's own checks are per-call: it can reject one oversized
    file, but only this sees the running total. Returns (kept, skip_notes)
    with skip notes already named for the user (_attachment_name), unlike
    _select_artifacts, whose caller re-maps container paths. Pure.
    """
    kept: list[dict] = []
    skipped: list[str] = []
    total = 0
    for entry in entries:
        name = _attachment_name(entry["path"])
        size = int(entry.get("size") or 0)
        if len(kept) >= max_files:
            skipped.append(f"{name} (skipped: more than {max_files} attached files)")
            continue
        if size > max_bytes:
            skipped.append(f"{name} (skipped: {size} bytes over the {max_bytes}-byte limit)")
            continue
        if total + size > max_bytes:
            skipped.append(f"{name} (skipped: would exceed the {max_bytes}-byte total limit)")
            continue
        kept.append(entry)
        total += size
    return kept, skipped


def _select_artifacts(
    listed: list[tuple[str, int]],
    *,
    max_files: int = MAX_ARTIFACT_FILES,
    max_bytes: int = MAX_ARTIFACT_BYTES,
) -> tuple[list[str], list[str]]:
    """Applies the file-count/size caps to files found under out/.

    listed: (container_path, size_bytes) pairs, in the order `find`
    reported them. Returns (paths_to_fetch, skip_notes). Pure: runs on
    sizes already known from `find`, never touches Docker — lets the caps
    be unit-tested without a real session.
    """
    to_fetch: list[str] = []
    skipped: list[str] = []
    total = 0
    for path, size in listed:
        if len(to_fetch) >= max_files:
            skipped.append(f"{path} (skipped: more than {max_files} output files)")
            continue
        if size > max_bytes:
            skipped.append(f"{path} (skipped: {size} bytes over the {max_bytes}-byte limit)")
            continue
        if total + size > max_bytes:
            skipped.append(f"{path} (skipped: would exceed the {max_bytes}-byte total limit)")
            continue
        to_fetch.append(path)
        total += size
    return to_fetch, skipped


async def _sandbox_output_dir(session) -> str | None:
    """The absolute path to SANDBOX_OUTPUT_DIRNAME under the sandbox's own
    current working directory — queried live via `pwd` rather than assumed,
    since the same session.exec() call is what exec_command uses under the
    hood, so this always matches wherever the model actually is. Returns
    None if the cwd can't be determined (treated as "no artifacts").
    """
    res = await session.exec("pwd", shell=False)
    if not res.ok():
        return None
    cwd = res.stdout.decode("utf-8", errors="replace").strip()
    return posixpath.join(cwd, SANDBOX_OUTPUT_DIRNAME) if cwd else None


async def _collect_artifacts(
    session, out_dir: str | None, newer_than: str | None = None,
) -> tuple[list[SandboxArtifact], list[str]]:
    """Reads back whatever the sandbox agent saved under out_dir.

    This is the FALLBACK delivery path, used when the agent attached nothing
    with attach_file (or was cut off before it could). It is a blind sweep, so
    a model that iterated saves v1/v2/v3 here and the user receives all three
    — which is exactly why selection, not this, is the normal path now.

    newer_than is the run marker written by _mark_run_start: when given, only
    files modified since this run began are swept. A thread's workspace
    snapshot includes out/ and a resumed run only mkdir -p's it, so without
    the filter run #2 re-delivers run #1's files alongside its own. None (no
    marker, or the touch failed) keeps the unfiltered behavior — this is a
    fallback path and must never fail a run over its own filter.

    Returns (artifacts, skip_notes): skip_notes is one human-readable
    reason per output file that was found but not fetched (over
    MAX_ARTIFACT_FILES / MAX_ARTIFACT_BYTES — see _select_artifacts), with
    the container's absolute path replaced by the same relative, Discord-
    safe name used for artifacts (_relative_artifact_name) — the model must
    never see the sandbox's real absolute filesystem paths (see
    SANDBOX_OUTPUT_DIRNAME comment for why: it has previously reused one as
    an exec_command `workdir` argument and failed outright).

    out_dir is resolved once by the caller (run_sandbox_task, via
    _sandbox_output_dir) up front — before Runner.run starts — rather than
    re-queried here, so the same path used to tell the model where to save
    files is also the one we look in afterward.

    Uses `find`+`exec("cat", ...)` directly — not session.read(), and not
    the exec_command/write_stdin tool path the model's own turns go
    through. session.read() is out because it validates paths against the
    SDK's nominal manifest root ("/workspace"), which the Shell-only
    capability we use never actually creates (see SANDBOX_OUTPUT_DIRNAME) —
    it rejects our real, cwd-relative paths as "not relative to the
    manifest". exec_command is out because it truncates large output to a
    token budget and would burn the (often small, local) sandbox model's
    own context on raw file bytes for no reason. Plain exec("cat", ...) is
    the same primitive session.read() uses internally, just without the
    manifest-relative check that doesn't apply to our setup.
    """
    if out_dir is None:
        return [], []
    args = ["find", out_dir, "-maxdepth", "3", "-type", "f"]
    if newer_than:
        args += ["-newer", newer_than]
    res = await session.exec(*args, "-printf", "%s %p\n", shell=False)
    if not res.ok():
        return [], []  # a missing/empty out/ dir is not an error
    listed = _parse_find_output(res.stdout.decode("utf-8", errors="replace"))
    to_fetch, skipped = _select_artifacts(listed)
    skip_notes: list[str] = []
    for note in skipped:
        logger.info(f"Sandbox: {note}")
        # split on the " (skipped: " marker, not plain whitespace — a path
        # under out/ can itself contain spaces (e.g. "my plot.png")
        path, _, reason = note.partition(" (skipped: ")
        skip_notes.append(f"{_relative_artifact_name(path, out_dir)} (skipped: {reason}")

    artifacts: list[SandboxArtifact] = []
    for path in to_fetch:
        try:
            read_result = await session.exec("cat", "--", path, shell=False)
        except Exception as e:
            logger.warning(f"Sandbox: failed to read output file {path}: {e}")
            continue
        if not read_result.ok():
            logger.warning(f"Sandbox: failed to read output file {path}: exit {read_result.exit_code}")
            continue
        artifacts.append(
            SandboxArtifact(name=_relative_artifact_name(path, out_dir), data=read_result.stdout)
        )
    return artifacts, skip_notes


async def _collect_deliverables(
    session, entries: list[dict]
) -> tuple[list[SandboxArtifact], list[str]]:
    """Reads back the files the agent chose with attach_file, in the order it
    attached them.

    Same `exec("cat", ...)` primitive and the same reasons as
    _collect_artifacts (session.read() enforces a manifest root we don't use;
    exec_command truncates to a token budget) — it just reads a list the agent
    curated instead of everything it happened to leave lying around. Sizes
    were recorded at attach time, so the caps are applied without re-statting.
    """
    kept, skip_notes = _select_deliverables(entries)
    for note in skip_notes:
        logger.info(f"Sandbox: {note}")

    artifacts: list[SandboxArtifact] = []
    for entry in kept:
        path = entry["path"]
        try:
            read_result = await session.exec("cat", "--", path, shell=False)
        except Exception as e:
            logger.warning(f"Sandbox: failed to read attached file {path}: {e}")
            continue
        if not read_result.ok():
            logger.warning(f"Sandbox: failed to read attached file {path}: exit {read_result.exit_code}")
            continue
        artifacts.append(SandboxArtifact(
            name=_attachment_name(path),
            data=read_result.stdout,
            caption=entry.get("caption") or "",
        ))
    return artifacts, skip_notes


async def _mark_run_start(session, out_dir: str | None) -> str | None:
    """Touches this run's marker file and returns its path, or None when that
    wasn't possible.

    Placed BESIDE out_dir, never inside it, or the sweep would find the marker
    itself and deliver it to the user as an empty file. Touched after
    session.start() has already hydrated any restored snapshot, so everything
    that came out of that snapshot is older than it — tar preserves mtimes, so
    restored files keep the times they were archived with.
    """
    if out_dir is None:
        return None
    marker = posixpath.join(posixpath.dirname(out_dir), RUN_MARKER_NAME)
    try:
        res = await session.exec("touch", "--", marker, shell=False)
    except Exception as e:
        logger.warning(f"Sandbox: could not write the run marker: {e}")
        return None
    return marker if res.ok() else None


async def _deliver(
    session, deliverables: list[dict], out_dir: str | None, marker: str | None,
) -> tuple[list[SandboxArtifact], list[str]]:
    """What the user actually receives: the files the agent attached, or —
    only when it attached none — a swept fallback from out/.

    The fallback is what makes a cut-off run still worth something (a timeout
    that already produced a good file), and it is also the safety net for a
    model that finished without ever calling attach_file. It is not the normal
    path: sweeping delivers every iteration the model happened to leave in
    out/, which is the bug this selection exists to fix.
    """
    if deliverables:
        return await _collect_deliverables(session, deliverables)
    return await _collect_artifacts(session, out_dir, marker)


async def run_sandbox_task(
    task: str,
    progress_hooks=None,
    *,
    thread=None,
    client=None,
    requesting_user_id=None,
    resumed: bool = False,
    ledger=None,
) -> SandboxResult:
    """Run one self-contained task in a fresh Docker sandbox.

    Returns a SandboxResult: the agent's final message to the user (text)
    plus the files it attached (artifacts, possibly empty).
    progress_hooks is an optional agents.RunHooks instance (e.g.
    classes.sandbox_progress.SandboxProgressHooks) attached to the nested
    run so every tool call (exec_command/write_stdin) and its output can be
    observed — e.g. to stream them to Discord.

    thread/client/requesting_user_id are all optional (default None, which
    reproduces the exact original single-argument behavior: no thread
    context, no snapshot persistence, and the ask_user/send_preview_to_thread
    tools degrade to telling the model to proceed on its own). When given —
    thread is the Discord thread/channel resolved by
    tool_functions.run_code_sandbox via ensure_sandbox_thread, client is the
    discord.Client, and requesting_user_id is the id of the user who asked —
    they are threaded into the NESTED Runner.run's own `context=`, which the
    thread tools (see their definitions above) read via wrapper.context.

    ledger is the thread's claimed sandbox_thread_inbox.RunLedger (see
    run_code_sandbox): what people post in the thread during the run lands
    there and is steered into the run by a SandboxConversation built around
    it. None (no thread, tests) uses a detached ledger nobody can post to. thread also determines whether this run's workspace
    is persisted as a snapshot for a later run in the same thread — see
    sandbox_snapshot_id_for/_create_sandbox_session/_persist_sandbox_snapshot.

    resumed says whether the caller expects this thread's saved workspace to
    be restored (it asked sandbox_snapshot_exists before announcing it). It
    does two things: it prepends RESUMED_TASK_PREAMBLE so the nested model
    actually LOOKS at the restored files — a small local model will not infer
    "there is prior work here" from the Memory summary alone — and it enables
    _create_sandbox_session's recovery path for a snapshot that turns out not
    to be restorable. The value that comes back on SandboxResult.resumed is
    the ground truth, which differs from this argument exactly when that
    recovery fired.

    We create and own the container ourselves (rather than letting
    RunConfig(sandbox=...) manage it) so we can read artifacts out of it
    after the run finishes but before it is destroyed; the container is
    always deleted in `finally`, including on timeout or any other error —
    and, when a snapshot id applies, its workspace is persisted just before
    that (_persist_sandbox_snapshot), on every exit path (success, timeout,
    max_turns, model_error, or an uncaught infra error) since even a partial
    run's workspace may be worth resuming from.

    On timeout (the task outliving sandbox_timeout() seconds), we do NOT
    raise: the container is still alive at that point, and a task that ran
    out of time may already have produced and verified a good file, so we
    make a bounded, best-effort attempt to recover whatever is under the
    output dir before teardown and return SandboxResult(ok=False,
    error="timeout", artifacts=<recovered>) instead — recovery failures are
    swallowed, since it's strictly better-than-nothing on top of today's
    "discard everything" behavior. The same applies to MaxTurnsExceeded
    (error="max_turns") and ModelBehaviorError/ModelRefusalError
    (error="model_error") — see SandboxResult and _failure_result. Every
    OTHER error (a sandbox that's unavailable, UserError, a dead
    container/transport) still propagates to the caller unchanged: those
    are infra/configuration failures the sandbox itself can't recover from,
    not a task that simply ran out of budget.
    """
    snapshot_id = sandbox_snapshot_id_for(thread)
    conversation = SandboxConversation(ledger, thread)
    sandbox_client = build_sandbox_client()
    session, resumed = await _create_sandbox_session(sandbox_client, snapshot_id, resumed)
    if resumed:
        task = RESUMED_TASK_PREAMBLE + task
    try:
        out_dir = await _sandbox_output_dir(session)
        if out_dir is not None:
            await session.exec("mkdir", "-p", out_dir, shell=False)
        # After the mkdir, so the marker is never older than out/ itself, and
        # after any snapshot restore (session.start(), above) so every
        # restored file is older than it. See _mark_run_start.
        marker = await _mark_run_start(session, out_dir)
        agent = build_sandbox_agent(out_dir, conversation)
        nested_context = {
            "thread": thread,
            "client": client,
            "requesting_user_id": requesting_user_id,
            "session": session,
            # The run's conversation coordinator: the tools read it from
            # here to gate side effects, answer thread messages and ask
            # questions (see classes/sandbox_conversation.py).
            "conversation": conversation,
            # Appended to by the attach_file tool; read back below to decide
            # what is delivered. A list rather than a return value because the
            # agent chooses files DURING the run, including on the paths where
            # Runner.run never returns one (timeout, max_turns).
            "deliverables": [],
            "deadline": time.monotonic() + sandbox_timeout(),
            # Paired with "deadline" so _elapsed_note can report time used
            # as a fraction of the whole budget. Read from the context
            # rather than re-read from the env at each command, so the
            # figure the model sees always matches the deadline it is
            # actually racing.
            "timeout_seconds": sandbox_timeout(),
        }
        conversation.ledger.state = sandbox_thread_inbox.RUNNING
        run_config = build_sandbox_run_config(
            sandbox_client, session, input_filter=conversation.input_filter,
            group_id=f"sandbox-{uuid.uuid4().hex}")
        try:
            # ONE wait_for around every pass, so a continuation never
            # resets the wall-clock budget.
            run_result = await asyncio.wait_for(
                _converse(agent, task, run_config, progress_hooks, nested_context,
                          conversation),
                timeout=sandbox_timeout(),
            )
        except asyncio.TimeoutError:
            # Out of time during a continuation: the pass before it had
            # already produced a final answer, which still stands.
            run_result = conversation.completed_result
            if run_result is None:
                return await _failure_result(session, nested_context, out_dir, marker,
                                             "timeout", resumed=resumed)
            logger.warning("Sandbox: timed out during a continuation; delivering the "
                           "answer from before it")
            conversation.finalize()
        except MaxTurnsExceeded:
            return await _failure_result(session, nested_context, out_dir, marker,
                                         "max_turns", resumed=resumed)
        except (ModelBehaviorError, ModelRefusalError) as e:
            return await _failure_result(session, nested_context, out_dir, marker,
                                         "model_error", str(e), resumed=resumed)
        final_output = run_result.final_output
        text = final_output if isinstance(final_output, str) else str(final_output)
        artifacts, skip_notes = await _deliver(
            session, nested_context["deliverables"], out_dir, marker)
        return SandboxResult(text=text, artifacts=artifacts, skipped_artifacts=skip_notes,
                             resumed=resumed, conversation=conversation.ledger.outcomes(),
                             unsent_replies=list(conversation.unsent_replies))
    finally:
        # Idempotent, and a no-op when _converse already committed it: this
        # covers the exits that bypass it (an infra error mid-run), so no
        # message is accepted into a run that can no longer see it.
        conversation.finalize()
        await _persist_sandbox_snapshot(session, snapshot_id)
        await _delete_sandbox_session(sandbox_client, session)


async def _converse(agent, task, run_config, progress_hooks, nested_context,
                    conversation: SandboxConversation):
    """Runs the nested agent to a final answer that has accounted for the
    thread, then commits the finalization boundary.

    A final answer with thread input still unanswered — including input that
    arrived while it was being written — gets a CONTINUATION rather than
    being delivered: a further Runner.run in the same live session, fed the
    previous run's to_input_list() plus a short note saying what is still
    open (see SandboxConversation.continuation_nudge). Continuing replays
    history, never tools: completed tool calls are in that list as
    call/result pairs, so nothing is executed twice.

    Bounded three ways, so a model that never cooperates cannot loop: at most
    MAX_CONTINUATIONS extra passes; each pass gets only the turns left of ONE
    cumulative sandbox_max_turns() budget (counted from raw_responses), and
    one is only started with MIN_CONTINUATION_TURNS left; and the caller's
    single wait_for bounds the wall clock. When a bound stops it, the run
    finishes with the input reported as unresolved rather than hidden — and
    a continuation that fails (turns, model error, a model call's own
    timeout) or times out falls back
    to the previous pass's completed answer rather than to "stopped", since
    that answer was already final; see conversation.completed_result.

    The final check and conversation.finalize() run with no await between
    them, so a message cannot slip in after the check and be accepted into a
    run that will never read it: it becomes a follow-up instead.
    """
    turns_left = sandbox_max_turns()
    run_input = task
    try:
        while True:
            try:
                result = await Runner.run(
                    agent,
                    run_input,
                    max_turns=turns_left,
                    run_config=run_config,
                    hooks=progress_hooks,
                    context=nested_context,
                )
            except (MaxTurnsExceeded, ModelBehaviorError, ModelRefusalError,
                    ModelTimeoutError, APITimeoutError) as e:
                # A continuation that fails must not cost the answer the run had
                # already finished: deliver that one, with the message it was
                # continuing for reported as not applied by the ledger.
                previous = conversation.completed_result
                if previous is None:
                    raise
                logger.warning(f"Sandbox: continuation failed ({type(e).__name__}); "
                               "delivering the answer from before it")
                conversation.finalize()
                return previous
            conversation.completed_result = result
            turns_left -= len(result.raw_responses)
            nudge = conversation.continuation_nudge(nested_context.get("deliverables"))
            if (nudge is None or conversation.continuations >= MAX_CONTINUATIONS
                    or turns_left < MIN_CONTINUATION_TURNS):
                conversation.finalize()
                return result
            conversation.note_continuation()
            logger.info(f"Sandbox: continuing the run ({conversation.continuations}/"
                        f"{MAX_CONTINUATIONS}, {turns_left} turn(s) left) to account for thread input")
            run_input = result.to_input_list() + [{"role": "user", "content": f"[sandbox]: {nudge}"}]
    finally:
        # A call that raised or was cancelled (the caller's wait_for) never
        # reached on_llm_end, which is what normally stops the indicator.
        conversation.stop_typing()


async def _failure_result(
    session, nested_context: dict, out_dir: str | None, marker: str | None,
    error: str, detail: str | None = None,
    *, resumed: bool = False,
) -> SandboxResult:
    """Builds the SandboxResult for a run stopped before it produced a
    final report (timeout / max_turns / model_error — see SandboxResult).

    The container is still alive at this point, so we make a bounded,
    best-effort attempt to recover whatever the run had already produced
    before the caller tears it down: the files it had attached, or a swept
    fallback from out_dir if it was cut off before attaching any (see
    _deliver). Recovery failures are swallowed, since it's strictly
    better-than-nothing on top of returning no artifacts at all. `detail`
    (e.g. the ModelBehaviorError message) is logged only — never put in
    SandboxResult.text or shown to the outer model, which should be told to
    retry with a clearer task, not fed the raw error.

    A forced stop also finalizes the conversation first, so whatever was
    still unanswered is reported as incomplete instead of disappearing.
    """
    conversation = nested_context.get("conversation")
    if conversation is not None:
        conversation.finalize()
    logger.warning(f"Sandbox: run stopped ({error}), attempting best-effort artifact recovery"
          + (f": {detail}" if detail else ""))
    artifacts: list[SandboxArtifact] = []
    skip_notes: list[str] = []
    try:
        artifacts, skip_notes = await asyncio.wait_for(
            _deliver(session, nested_context.get("deliverables") or [], out_dir, marker),
            timeout=SANDBOX_RECOVERY_TIMEOUT_SECONDS,
        )
    except Exception as e:
        logger.warning(f"Sandbox: artifact recovery after {error} failed: {e}")
    return SandboxResult(
        text="", artifacts=artifacts, ok=False, error=error, skipped_artifacts=skip_notes,
        resumed=resumed,
        conversation=conversation.ledger.outcomes() if conversation is not None else [],
        unsent_replies=list(conversation.unsent_replies) if conversation is not None else [],
    )


# ---------------------------------------------------------------------------
# What the OUTER model is told about a finished run
# ---------------------------------------------------------------------------
#
# Pure string-building, kept here rather than inline in
# tool_functions.run_code_sandbox: it is the part that changes most often (every
# observed misbehaviour of the outer model gets fixed by rewording it) and the
# part hardest to review buried in 370 lines of orchestration. Keeping it pure
# also means it is testable without a Discord channel or a Docker daemon.

# One line naming how a run ended, shown to the USER: on the live-progress
# message's final state and as the first line of the closing embed. Only used
# when the run did NOT finish normally.
SANDBOX_OUTCOME_NOTES = {
    "timeout": "⏱ Stopped: the task timed out.",
    "max_turns": "🔁 Stopped: the task ran out of turns.",
    "model_error": "⚠️ Stopped: the sandbox's model misbehaved.",
}

# The same three reasons as a clause, for the sentence written to the MODEL.
_FAILURE_REASON = {
    "timeout": "it timed out",
    "max_turns": "it ran out of turns",
    "model_error": "the sandbox's model misbehaved",
}


def sandbox_outcome_note(error: str | None) -> str:
    """The user-facing one-liner for a run that did not finish normally."""
    return SANDBOX_OUTCOME_NOTES.get(error, "❌ Stopped.")


def _no_artifact_message(error: str | None) -> str:
    """What to tell the model when a stopped run produced nothing at all.

    Deliberately none of these says "retry". A retry starts the whole task
    again in a fresh container, throwing away the partial workspace teardown
    just persisted - and in practice the outer model answers "retry, more
    focused" by writing exactly the implementation spec run_code_sandbox's
    docstring exists to suppress (observed: a timed-out cow GIF retried with an
    invented canvas size, frame count and library choice). The resumable
    follow-up offered below is strictly better and costs nothing but asking.
    """
    return {
        "timeout": (
            f"The sandbox task ran out of time and was stopped at the "
            f"{sandbox_timeout()}s limit before it finished. Tell the user "
            "it timed out."
        ),
        "max_turns": (
            f"The sandbox task ran out of turns ({sandbox_max_turns()} max) "
            "before finishing. Tell the user it did not finish in the number "
            "of steps it had."
        ),
        "model_error": (
            "The sandbox's own model produced an invalid action mid-task "
            "and the run was stopped before finishing. Tell the user the "
            "sandbox could not complete this task."
        ),
    }.get(error,
          "The sandbox task was stopped before finishing. Tell the user the "
          "task failed.")


def sandbox_skip_note(skipped: list) -> str:
    """Files that were produced but not delivered (over the artifact caps)."""
    if not skipped:
        return ""
    return (
        f"\n\n{len(skipped)} output file(s) were NOT sent "
        f"(over the {MAX_ARTIFACT_FILES}-file / {MAX_ARTIFACT_BYTES}-byte "
        f"limit): {', '.join(skipped)}. Tell the user "
        "which file(s) could not be delivered and why; if it matters, "
        "retry producing a smaller or fewer files."
    )


# How each ledger status (sandbox_thread_inbox.RunLedger.outcomes) is put to
# the OUTER model. Only "done" and "answered" claim anything was handled;
# every other status says plainly what did NOT happen, because the old note
# ("the sandbox received these and adapted") let the outer model claim a
# change had been made when the nested model had ignored it or never seen it.
_STEERING_STATUS = {
    "done": "done",
    "answered": "answered",
    "noted": "taken into account",
    "accepted": "accepted, but the sandbox did not confirm it was finished",
    "declined": "declined by the sandbox",
    "superseded": "superseded by a later message",
    "deferred": "left for a follow-up in this thread",
    "unacknowledged": "never answered by the sandbox, so it may not have been applied",
    "unseen": "arrived too late for the sandbox to see it; NOT applied",
    "late": "arrived as the run was finishing; NOT applied (the user can ask again in this thread)",
}
_HANDLED_STATUSES = {"done", "answered", "noted"}


def sandbox_steering_note(steering) -> str:
    """What people said to the run while it happened, and what became of it.

    The outer model's history was built before the run started and on_message
    routes these to the sandbox instead of enqueuing them, so without this it
    cannot know the request changed. Observed without it: a mid-run "make the
    milk red" produced a red image the outer model called a mistake. So the
    note still says to treat HANDLED messages as part of the request — but
    only those, and it states the rest as not applied.

    steering is SandboxResult.conversation (a list of outcome rows).
    """
    if not steering:
        return ""
    lines = []
    handled = False
    for row in steering:
        status = row.get("status", "")
        handled = handled or status in _HANDLED_STATUSES
        text = " ".join(str(row.get("text", "")).split())
        if len(text) > 200:
            text = text[:199] + "…"
        line = f"- {row.get('author', 'someone')}: “{text}” → {_STEERING_STATUS.get(status, status)}"
        reply = " ".join(str(row.get("reply", "")).split())
        if reply:
            line += f" (sandbox said: “{reply[:200]}”)"
        lines.append(line)
    note = ("\n\nPeople posted in the sandbox's thread while it worked. What became "
            "of each message:\n" + "\n".join(lines) + "\n")
    if handled:
        note += ("Treat the ones marked done, answered or taken into account as part "
                 "of what was asked: do not call the result a mistake or offer to "
                 "undo it. ")
    note += ("Do not claim any other message was applied; if one matters, tell the "
             "user it was not done and that they can ask again in the thread.")
    return note


# Closing-embed wording for messages that did not get handled. Short: this
# sits under "Sandbox closed" in the thread, for the people who wrote them.
# "accepted" (the model replied will_apply but never said done) is left out
# on purpose: the writer already got a 👍 reply and can see the result, so a
# ⚠️ line for it read as noise in live testing. The outer model still hears
# about it via sandbox_steering_note.
_UNRESOLVED_LABELS = {
    "unacknowledged": "Never answered",
    "unseen": "Not seen in time",
    "late": "Arrived as it was finishing",
    "deferred": "Left for a follow-up",
}
MAX_UNRESOLVED_LINES = 5


def sandbox_unresolved_note(outcomes, in_thread: bool) -> str:
    """Lines for the closing embed naming each thread message the run did not
    apply (or did not confirm applying), or "" when every one was handled.

    The plan's rule is that acknowledged input must never silently disappear
    when a run ends; this is the user-facing half of that, and the outer
    model's half is sandbox_steering_note.
    """
    rows = [r for r in (outcomes or []) if r.get("status") in _UNRESOLVED_LABELS]
    if not rows:
        return ""
    lines = []
    for r in rows[:MAX_UNRESOLVED_LINES]:
        text = " ".join(str(r.get("text", "")).split())
        if len(text) > 80:
            text = text[:79] + "…"
        lines.append(f"⚠️ {_UNRESOLVED_LABELS[r['status']]}: “{text}”")
    if len(rows) > MAX_UNRESOLVED_LINES:
        lines.append(f"…and {len(rows) - MAX_UNRESOLVED_LINES} more.")
    if in_thread:
        lines.append("@mention me in this thread to carry on with these.")
    return "\n" + "\n".join(lines)


def sandbox_tool_result(
    result, *, sent_names: list, in_thread: bool, resumable: bool,
    steering=None, resume_correction: str = "",
) -> str:
    """The string run_code_sandbox hands back to the outer model.

    `resumable` must come from the closing embed's own answer (the live Redis
    TTL), not from the configured one, so what the model is told about picking
    the work back up can never disagree with what the user was just shown.
    """
    skip_note = sandbox_skip_note(result.skipped_artifacts)
    steering_note = sandbox_steering_note(steering)
    tail = skip_note + steering_note + resume_correction

    if not result.ok:
        if in_thread and resumable:
            unfinished = (
                " Its workspace WAS saved, so do NOT retry: a retry starts the "
                "whole task again from scratch in an empty sandbox, while the "
                "user simply asking again in THIS thread carries on from where "
                "it stopped. Offer them that, and do not add build instructions "
                "of your own when you do."
            )
        else:
            unfinished = (
                " Do not retry the same task unchanged; ask the user how they "
                "want to proceed."
            )
        if sent_names:
            return (
                f"The sandbox task did not finish "
                f"({_FAILURE_REASON.get(result.error, 'it failed')}), "
                f"but before it was stopped it had already produced and verified "
                f"{len(sent_names)} file(s), which were recovered and sent to the "
                f"channel: {', '.join(sent_names)}. Tell the user the file(s) were "
                "delivered despite the task not finishing, and do not claim they "
                "are pending." + unfinished + tail
            )
        return _no_artifact_message(result.error) + unfinished + tail

    if not sent_names:
        # Only warn when nothing was produced AT ALL. An artifact that was found
        # but failed to send is a different case, already logged, and claiming
        # "no files were found" for it would be false.
        no_file_note = (
            "\n\n(The sandbox attached no file for this run. If one was "
            "expected, it was not produced — tell the user that plainly. Do "
            "not claim a file exists, was sent, or is still being generated.)"
        ) if not result.artifacts else ""
        return result.text + no_file_note + tail

    return (
        f"{result.text}\n\n"
        f"The sandbox chose {len(sent_names)} file(s) to deliver and they are "
        f"already in the thread: {', '.join(sent_names)}. The text above is its "
        "own message to the user, posted there beside them — so the user has "
        "already read it. Add at most one short sentence of your own: do not "
        "repeat it, re-describe the files, or second-guess the result."
        + tail
    )

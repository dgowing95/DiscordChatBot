"""The brain's workspace tools: shell set-up, files in and out, and the
thread tools Tier 3 will use.

The brain (classes/brain.py) runs in the SERVER'S WORKSPACE
(classes/workspace_manager.py): one long-lived Docker container per server,
shared by every call there and kept between calls and core restarts. Each
call works in its own folder, jobs/<id>/; shared/ holds what the server
keeps. Nothing here persists anything: the manager snapshots the workspace.

What lives here:
  - the Shell capability's tool wrappers (_configure_shell_tools): the job
    folder as working directory, malformed calls as tool results, and the
    thread gate and checkpoints the steering code needs;
  - attach_file and the delivery helpers (_deliver: what the user receives);
  - writing chat files into in/ (_write_inputs);
  - the thread machinery Tier 3 reuses (ensure_sandbox_thread, ask_user,
    respond_to_updates, say_in_thread, send_preview_to_thread, _converse).
    Nothing calls those until phase 5; they are kept, with their tests, as
    the named exception to "delete, don't deprecate".

The heavy imports (docker SDK, agents.sandbox) stay inside functions so the
pure helpers below are testable without them.
"""
import logging
import io
import json
import os
import posixpath
import re
import time
from dataclasses import dataclass
from pathlib import PurePosixPath

import discord
from openai import APITimeoutError
from agents import (
    MaxTurnsExceeded,
    ModelBehaviorError,
    ModelRefusalError,
    ModelTimeoutError,
    Runner,
    RunConfig,
    RunContextWrapper,
    function_tool,
    set_tracing_disabled,
)
from classes import workspace_inputs
from classes.sandbox_conversation import (
    MAX_CONTINUATIONS,
    MIN_CONTINUATION_TURNS,
    SandboxConversation,
)

# No run is ever uploaded to OpenAI's trace dashboard (why: text_llm_handler.py).
set_tracing_disabled(True)

logger = logging.getLogger(__name__)

DEFAULT_ASK_USER_TIMEOUT_SECONDS = 300

# Directory (relative to the run's job folder, its working directory) where
# the brain saves anything it means to return as a file. It is not
# the delivery boundary — the agent picks what is delivered with attach_file,
# and out/ is only where finished candidates live (plus the fallback swept
# when it attached nothing; see _deliver).
#
# The model only ever gets RELATIVE names, never the absolute path:
# exec_command's own `workdir` argument is validated by the SDK's manifest
# system, which unconditionally rejects absolute paths
# (InvalidManifestPathError: "manifest path must be relative: ...") even
# when the path genuinely exists — confirmed in production, where a model
# told the absolute path reused it as `workdir` and the tool call failed
# outright. We track the absolute form ourselves for our own raw
# session.exec() calls, which bypass that check (see _collect_artifacts), and
# _in_job_dir makes every workdir the model gives relative to its job folder.
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


def _positive_int(raw: str | None, default: int) -> int:
    try:
        value = int(str(raw).strip())
        return value if value > 0 else default
    except (TypeError, ValueError):
        return default


def sandbox_ask_user_timeout() -> int:
    """Max seconds the ask_user tool waits for a reply before telling the
    model to proceed on its own (SANDBOX_ASK_USER_TIMEOUT, default 300). Any
    single call also never waits past the run's own remaining
    time budget — see ask_user's `deadline` handling."""
    return _positive_int(os.environ.get("SANDBOX_ASK_USER_TIMEOUT"), DEFAULT_ASK_USER_TIMEOUT_SECONDS)


def _tolerant_tool_invoke(original_invoke):
    """Wraps a shell tool's on_invoke_tool so a malformed tool call (missing
    a required argument) becomes a model-visible error turn instead of
    aborting the whole run.

    Needed because both shell tools are built with strict_json_schema=False
    (see brain.build_brain_agent's docstring: the grammar-enforced strict mode
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
    model and parseable by brain_progress.parse_exec_result, which expects
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


def _job_relative(context: dict, path: str) -> str | None:
    """`path` (relative, as the model wrote it) resolved against the run's job
    folder, still relative to the workspace root; None when it would leave
    the workspace. Without a job folder in the context (tests, a context
    built elsewhere) the path is returned as given."""
    job = context.get("job_dir") if isinstance(context, dict) else None
    if not job:
        return path
    resolved = posixpath.normpath(posixpath.join(job, path or "."))
    if resolved == ".." or resolved.startswith("../") or resolved.startswith("/"):
        return None
    return resolved


def _job_path(context: dict, path: str) -> str | None:
    """Like _job_relative, but absolute (for our own session.exec calls,
    whose working directory is the workspace root, not the job folder)."""
    relative = _job_relative(context, path)
    root = context.get("workspace_root") if isinstance(context, dict) else None
    if relative is None or not root or not context.get("job_dir"):
        return relative
    return posixpath.join(root, relative)


def _in_job_dir(original_invoke):
    """Wraps exec_command so the model works in its job folder: no `workdir`
    runs there, and a relative `workdir` is taken from there (so `out` is
    jobs/<id>/out and `shared`, a link to ../../shared, the server's kept
    files). The SDK's own default is the workspace root, which every call in
    the server shares.
    A workdir that would leave the workspace is refused as a tool result."""
    async def _invoke(ctx, raw_input):
        context = ctx.context if isinstance(getattr(ctx, "context", None), dict) else {}
        if not context.get("job_dir"):
            return await original_invoke(ctx, raw_input)
        try:
            args = json.loads(raw_input or "{}")
        except (TypeError, ValueError):
            return await original_invoke(ctx, raw_input)
        if not isinstance(args, dict):
            return await original_invoke(ctx, raw_input)
        workdir = args.get("workdir")
        if workdir is not None and not isinstance(workdir, str):
            return await original_invoke(ctx, raw_input)
        if workdir and workdir.startswith("/"):
            # Refused here, as a result the model can act on: the SDK's own
            # InvalidManifestPathError is raised, not returned.
            return ("Tool call rejected: use a workdir relative to your working directory "
                    "(e.g. \"out\"), not an absolute path.")
        resolved = _job_relative(context, workdir or ".")
        if resolved is None:
            return "Tool call rejected: that workdir is outside the workspace. Use a relative path."
        return await original_invoke(ctx, json.dumps(dict(args, workdir=resolved)))

    return _invoke


def _configure_shell_tools(toolset) -> None:
    """configure_tools callback (see agents.sandbox.capabilities.Shell):
    wraps both shell tools so commands run in the job folder (_in_job_dir),
    malformed calls become model-visible errors (see _tolerant_tool_invoke),
    a long command yields early when someone posts in the thread
    (_with_shell_checkpoints), and a NEW command is refused while thread
    input is unanswered (_gated_invoke). Without a thread conversation in
    the run context (every brain call until Tier 3) the last two pass
    straight through.

    write_stdin is deliberately not gated: observing or stopping a command
    that is already running is exactly what the model may need to do while
    it reviews new input. Only starting new work is held back.

    Order matters: the gate is outermost, so a refused command runs nothing
    at all.
    """
    tolerant_exec = _tolerant_tool_invoke(_in_job_dir(toolset.exec_command.on_invoke_tool))
    tolerant_poll = None
    if toolset.write_stdin is not None:
        tolerant_poll = _tolerant_tool_invoke(toolset.write_stdin.on_invoke_tool)
        toolset.write_stdin.on_invoke_tool = tolerant_poll
    toolset.exec_command.on_invoke_tool = _gated_invoke(
        _with_shell_checkpoints(tolerant_exec, tolerant_poll), "exec_command", busy=True,
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
    @mentioning the bot in; either way it's safe, since a thread that never
    ran a task simply gets a new, empty job folder.
    A thread previously started off this very message is likewise reused.
    Otherwise a new thread is created off original_message (thread_created=
    True). If thread creation isn't possible (e.g. the channel type doesn't
    support threads, or a permissions error), the failure is caught and the
    original channel is returned unchanged (thread_created=False) so the tool
    degrades instead of failing outright; that run gets a job folder of its
    own, which no follow-up can find.
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
        # Never wait past this run's own remaining time budget —
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
    where = _job_path(ctx, path)
    if where is None:
        return "That path is outside the workspace. Use one relative to your working directory."

    # Size first, as attach_file does, so a huge file is never read into
    # core only to be refused (the check after the read stays: it may grow).
    try:
        stat = await session.exec("stat", "-c", "%s", "--", where, shell=False)
        size = int(stat.stdout.decode("utf-8", errors="replace").strip()) if stat.ok() else None
    except Exception:
        size = None
    if size is not None and size > MAX_ARTIFACT_BYTES:
        return (f"{path!r} is too large to preview ({size} bytes, over the "
                f"{MAX_ARTIFACT_BYTES}-byte limit).")
    try:
        result = await session.exec("cat", "--", where, shell=False)
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
    # Kept absolute from here on: _collect_deliverables reads it with our own
    # session.exec, whose working directory is the workspace root. The
    # messages below use the model's own relative name: it must never be
    # handed an absolute path (see SANDBOX_OUTPUT_DIRNAME).
    relative = path
    path = _job_path(ctx, path)
    if path is None:
        return "That path is outside the workspace. Use one relative to your working directory."

    deliverables = ctx.get("deliverables")
    name = _attachment_name(path)
    if remove:
        if deliverables is None:
            return "This run cannot attach files."
        before = len(deliverables)
        deliverables[:] = [entry for entry in deliverables if entry["path"] != path]
        if len(deliverables) == before:
            return f"{name!r} was not attached, so there was nothing to remove."
        # Removing the last one is a choice too: without this, an empty list
        # reads as "chose nothing" and the sweep sends the file just removed.
        ctx["delivery_chosen"] = True
        return f"Removed {name!r}; it will not be sent."

    # stat, not `test -f` then `wc -c`: one call answers both "does it exist"
    # and "how big is it", and the size is what the caps below need. GNU
    # coreutils is already a hard assumption here — _collect_artifacts uses
    # `find -printf`, which is GNU findutils.
    try:
        result = await session.exec("stat", "-c", "%s", "--", path, shell=False)
    except Exception as e:
        logger.warning(f"Sandbox attach_file: stat failed for {path!r}: {e}")
        return f"Could not check {relative!r}. Verify it exists with `ls`, then attach it again."
    if not result.ok():
        return (
            f"{relative!r} does not exist, so it was NOT attached. Check the path "
            "with `ls` — nothing is sent to the user unless it is attached."
        )
    try:
        size = int(result.stdout.decode("utf-8", errors="replace").strip())
    except ValueError:
        return f"Could not read the size of {relative!r}. Verify it with `ls -l` and try again."
    if size > MAX_ARTIFACT_BYTES:
        return (
            f"{relative!r} is too large to send ({size} bytes, over the "
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


async def _resolve_requester(thread, user_id):
    """The person who asked for this run, as a guild Member, or None.

    The event tools need a Member, not an id, to check the requester's own
    permission. The bot runs without the members intent, so
    guild.get_member usually misses for someone who only typed a message,
    and the sandbox context has no message author to fall back on. One
    fetch_member here, before the run, is what makes that check work. A
    failure leaves None, which the event tools treat as "not allowed"."""
    guild = getattr(thread, "guild", None)
    if guild is None or user_id is None:
        return None
    member = guild.get_member(user_id)
    if member is not None:
        return member
    try:
        return await guild.fetch_member(user_id)
    except Exception as e:
        # A warning, not info: the event tools then refuse as if the person
        # lacked the permission, and this line is the only sign of why.
        logger.warning(f"Sandbox: could not look up the requester {user_id}: {e}")
        return None


@dataclass
class SandboxArtifact:
    """One file being delivered to the user: normally one the sandbox agent
    chose with attach_file, otherwise one swept out of SANDBOX_OUTPUT_DIR by
    the fallback path. caption is whatever the agent passed to attach_file
    (empty for a swept file), sent as the message text beside it."""
    name: str
    data: bytes
    caption: str = ""


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

    group_id is the same for every pass of one run, so the SDK files a run's
    continuations together rather than as separate runs.
    """
    from agents.sandbox import SandboxRunConfig

    return RunConfig(
        sandbox=SandboxRunConfig(client=client, session=session),
        call_model_input_filter=input_filter,
        group_id=group_id,
    )


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


async def _sandbox_output_dir(session, job_dir: str = "") -> str | None:
    """The absolute path to SANDBOX_OUTPUT_DIRNAME in the run's job folder
    (job_dir, relative to the workspace root). The root is asked of the live
    session (`pwd`: our own exec calls start there) rather than assumed.
    None if it can't be determined (treated as "no artifacts")."""
    res = await session.exec("pwd", shell=False)
    if not res.ok():
        return None
    cwd = res.stdout.decode("utf-8", errors="replace").strip()
    return posixpath.join(cwd, job_dir, SANDBOX_OUTPUT_DIRNAME) if cwd else None


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

    out_dir is resolved once by the caller (brain.run_brain, via
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
    curated instead of everything it happened to leave lying around.

    Sizes are taken again first (one stat each, at most MAX_ARTIFACT_FILES):
    the file may have been rewritten bigger since it was attached (a chart
    re-rendered without attaching it again), and the caps, the running total
    included, must see its size now, before anything is read into core. The
    check after the read stays as the backstop: it may still grow between.
    """
    current = [dict(entry, size=await _current_size(session, entry)) for entry in entries]
    kept, skip_notes = _select_deliverables(current)
    for note in skip_notes:
        logger.info(f"Sandbox: {note}")

    artifacts: list[SandboxArtifact] = []
    total = 0
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
        size = len(read_result.stdout)
        if size > MAX_ARTIFACT_BYTES or total + size > MAX_ARTIFACT_BYTES:
            name = _attachment_name(path)
            skip_notes.append(f"{name} (skipped: grew past the {MAX_ARTIFACT_BYTES}-byte limit)")
            logger.info(f"Sandbox: {name} grew past the size limit while it was being sent")
            continue
        total += size
        artifacts.append(SandboxArtifact(
            name=_attachment_name(path),
            data=read_result.stdout,
            caption=entry.get("caption") or "",
        ))
    return artifacts, skip_notes


async def _current_size(session, entry: dict) -> int:
    """The attached file's size now, or the one recorded at attach time
    when it can't be read (the read after it then decides)."""
    try:
        result = await session.exec("stat", "-c", "%s", "--", entry["path"], shell=False)
        if result.ok():
            return int(result.stdout.decode("utf-8", errors="replace").strip())
    except Exception:
        pass
    return int(entry.get("size") or 0)


async def _mark_run_start(session, out_dir: str | None) -> str | None:
    """Touches this run's marker file and returns its path, or None when that
    wasn't possible.

    Placed BESIDE out_dir, never inside it, or the sweep would find the marker
    itself and deliver it to the user as an empty file. Touched at the start
    of the run, so anything an earlier run in the same job folder left (a
    follow-up in the thread) is older than it — a restored snapshot keeps the
    times files were archived with.
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


async def _write_inputs(session, out_dir: str | None, inputs, problems):
    """Writes the chat files into in/. Returns (task preamble, names
    written, problems): the caller's problems plus every file that could not
    be written. Never raises: a file that cannot be written is named as not
    provided instead."""
    problems = list(problems)
    written = []
    in_dir = posixpath.join(posixpath.dirname(out_dir), workspace_inputs.INPUT_DIRNAME) if out_dir else None
    try:
        if in_dir is None or not (await session.exec("mkdir", "-p", in_dir, shell=False)).ok():
            raise RuntimeError("no workspace folder for the files")
    except Exception as e:
        logger.warning(f"Sandbox: could not make the input folder: {e}")
        in_dir = None
    for name, data in inputs:
        if in_dir is None:
            problems.append(f"{name} (could not be copied in)")
            continue
        try:
            # session.write streams through exec (no put_archive) and keeps
            # the path inside the workspace root, which in/ is.
            await session.write(PurePosixPath(in_dir, name), io.BytesIO(data))
            written.append((name, len(data)))
        except Exception as e:
            logger.warning(f"Sandbox: could not write input {name}: {e}")
            problems.append(f"{name} (could not be copied in)")
    if written:
        logger.info(f"Sandbox: gave the run {len(written)} file(s) in {in_dir}")
    return workspace_inputs.task_preamble(written, problems), [n for n, _ in written], problems


async def _deliver(
    session, deliverables: list[dict], out_dir: str | None, marker: str | None,
    chosen: bool = False,
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
    if chosen:  # it attached files and removed them all: send nothing
        return [], []
    return await _collect_artifacts(session, out_dir, marker)


async def _converse(agent, task, run_config, progress_hooks, nested_context,
                    conversation: SandboxConversation, *, max_turns: int = 10):
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
    cumulative max_turns budget (counted from raw_responses), and
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
    turns_left = max_turns
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

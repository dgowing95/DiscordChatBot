"""Tests for the brain's workspace tools (core/classes/workspace_tools.py):
shell wrappers, attach_file and delivery, input files, and the thread tools
Tier 3 will reuse. The brain itself is tested in brain_tests.py.

Run from the repo root:
    pytest core/tests/workspace_tools_tests.py
"""
import asyncio
import json
import os
import sys
import time

import discord
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from classes import workspace_tools as sandbox_agent
import classes.workspace_tools as prod_sandbox_agent
import classes.tool_functions as prod_tool_functions


@pytest.fixture(autouse=True)
def _no_conversation_record_redis():
    """No test here should reach a real Redis for a thread's conversation
    record."""
    store = MagicMock(load=AsyncMock(return_value=None), save=AsyncMock())
    with patch("classes.sandbox_conversation_store.SandboxConversationStore",
               return_value=store):
        yield store


# ---------------------- sandbox LLM (model / host / api key) ----------------------

def _clear_llm_env(monkeypatch):
    for var in ("BRAIN_MODEL", "BRAIN_LLM_HOST", "BRAIN_LLM_API_KEY",
                "LLM_MODEL", "LLM_HOST", "LLM_API_KEY"):
        monkeypatch.delenv(var, raising=False)


# ---------------------- tolerant shell tool wrapping ----------------------
# Regression (found live in production): the sandbox LLM is often a small
# local model, and both shell tools are built with strict_json_schema=False
# (no grammar enforcement over ChatCompletions), so a tool call missing a
# required argument (e.g. exec_command with no `cmd`) reaches
# args_model.model_validate_json unguarded and raises a bare pydantic
# ValidationError. Uncaught, that propagates out of Runner.run as UserError
# ("Error running tool exec_command: 1 validation error for
# ExecCommandArgs\ncmd\n  Field required") and aborts the whole sandbox
# task. _tolerant_tool_invoke must turn that into a string tool result
# instead, per FunctionTool.on_invoke_tool's own documented contract.

def _validation_error():
    from pydantic import BaseModel, ValidationError

    class _Args(BaseModel):
        cmd: str

    try:
        _Args.model_validate({})
    except ValidationError as e:
        return e
    raise AssertionError("expected ValidationError")


@pytest.mark.asyncio
async def test_tolerant_invoke_passes_through_on_success():
    original = AsyncMock(return_value="ok")
    wrapped = sandbox_agent._tolerant_tool_invoke(original)
    result = await wrapped("ctx", '{"cmd": "ls"}')
    assert result == "ok"
    original.assert_awaited_once_with("ctx", '{"cmd": "ls"}')


@pytest.mark.asyncio
async def test_tolerant_invoke_turns_missing_arg_into_tool_result():
    async def raiser(ctx, raw_input):
        raise _validation_error()

    wrapped = sandbox_agent._tolerant_tool_invoke(raiser)
    result = await wrapped("ctx", "{}")
    assert isinstance(result, str)
    assert "cmd" in result
    assert "retry" in result.lower()


@pytest.mark.asyncio
async def test_tolerant_invoke_reraises_non_validation_errors():
    # A dead container/transport failure must still abort the run rather
    # than being told "try again" for every remaining turn.
    async def raiser(ctx, raw_input):
        raise RuntimeError("session wedged")

    wrapped = sandbox_agent._tolerant_tool_invoke(raiser)
    with pytest.raises(RuntimeError):
        await wrapped("ctx", "{}")


def test_configure_shell_tools_wraps_both_tools():
    exec_tool = MagicMock()
    exec_tool.on_invoke_tool = AsyncMock(return_value="exec ok")
    write_tool = MagicMock()
    write_tool.on_invoke_tool = AsyncMock(return_value="write ok")
    toolset = MagicMock(exec_command=exec_tool, write_stdin=write_tool)

    sandbox_agent._configure_shell_tools(toolset)

    assert exec_tool.on_invoke_tool is not exec_tool  # sanity: reassigned
    assert write_tool.on_invoke_tool is not write_tool


def test_configure_shell_tools_handles_no_write_stdin():
    exec_tool = MagicMock()
    exec_tool.on_invoke_tool = AsyncMock(return_value="exec ok")
    toolset = MagicMock(exec_command=exec_tool, write_stdin=None)

    sandbox_agent._configure_shell_tools(toolset)  # must not raise
    assert toolset.write_stdin is None


# ---------------------- artifact helpers (pure) ----------------------

def test_parse_find_output_parses_size_and_path():
    raw = "1234 /workspace/out/plot.png\n56 /workspace/out/data.csv\n"
    assert sandbox_agent._parse_find_output(raw) == [
        ("/workspace/out/plot.png", 1234),
        ("/workspace/out/data.csv", 56),
    ]


def test_parse_find_output_ignores_blank_and_malformed_lines():
    raw = "1234 /workspace/out/plot.png\n\n   \nnotanumber /workspace/out/bad\n"
    assert sandbox_agent._parse_find_output(raw) == [("/workspace/out/plot.png", 1234)]


def test_relative_artifact_name_strips_out_dir_and_flattens_subdirs():
    assert sandbox_agent._relative_artifact_name("/out/plot.png", "/out") == "plot.png"
    assert sandbox_agent._relative_artifact_name(
        "/out/nested/data.csv", "/out"
    ) == "nested_data.csv"


def test_relative_artifact_name_falls_back_when_empty():
    assert sandbox_agent._relative_artifact_name("/out", "/out") == "artifact"


def test_attachment_name_uses_the_basename():
    # The agent named this file itself, so it should arrive under that name
    # rather than the sweep's flattened form (_relative_artifact_name).
    assert sandbox_agent._attachment_name("out/plot.png") == "plot.png"
    assert sandbox_agent._attachment_name("out/sub/plot.png") == "plot.png"
    assert sandbox_agent._attachment_name("plot.png") == "plot.png"


def test_attachment_name_falls_back_when_there_is_no_basename():
    assert sandbox_agent._attachment_name("out/") == "out"
    assert sandbox_agent._attachment_name("/") == "artifact"


def _entry(path, size, caption=""):
    return {"path": path, "size": size, "caption": caption}


def test_select_deliverables_keeps_attached_files_under_the_caps():
    kept, skipped = sandbox_agent._select_deliverables(
        [_entry("out/a.png", 10), _entry("out/b.png", 20)], max_files=10, max_bytes=1000)
    assert [e["path"] for e in kept] == ["out/a.png", "out/b.png"]
    assert skipped == []


def test_select_deliverables_names_skips_for_the_user():
    # Unlike _select_artifacts, whose caller re-maps container paths, these
    # notes already read as the user will see the file named.
    kept, skipped = sandbox_agent._select_deliverables(
        [_entry("out/big.png", 5000)], max_files=10, max_bytes=1000)
    assert kept == []
    assert len(skipped) == 1 and skipped[0].startswith("big.png (skipped:")


def test_select_deliverables_enforces_the_running_total():
    kept, skipped = sandbox_agent._select_deliverables(
        [_entry("out/a.png", 600), _entry("out/b.png", 600)], max_files=10, max_bytes=1000)
    assert [e["path"] for e in kept] == ["out/a.png"]
    assert "would exceed" in skipped[0]


def test_select_deliverables_enforces_the_file_count_cap():
    entries = [_entry(f"out/{i}.png", 1) for i in range(3)]
    kept, skipped = sandbox_agent._select_deliverables(entries, max_files=2, max_bytes=1000)
    assert len(kept) == 2
    assert skipped == ["2.png (skipped: more than 2 attached files)"]


def test_select_artifacts_accepts_files_under_the_caps():
    listed = [("/workspace/out/a.png", 100), ("/workspace/out/b.png", 200)]
    to_fetch, skipped = sandbox_agent._select_artifacts(listed, max_files=10, max_bytes=1000)
    assert to_fetch == ["/workspace/out/a.png", "/workspace/out/b.png"]
    assert skipped == []


def test_select_artifacts_rejects_oversized_file():
    listed = [("/workspace/out/huge.bin", 5000)]
    to_fetch, skipped = sandbox_agent._select_artifacts(listed, max_files=10, max_bytes=1000)
    assert to_fetch == []
    assert len(skipped) == 1 and "huge.bin" in skipped[0]


def test_select_artifacts_enforces_running_total_cap():
    listed = [("/workspace/out/a.bin", 600), ("/workspace/out/b.bin", 600)]
    to_fetch, skipped = sandbox_agent._select_artifacts(listed, max_files=10, max_bytes=1000)
    assert to_fetch == ["/workspace/out/a.bin"]
    assert len(skipped) == 1 and "b.bin" in skipped[0]


def test_select_artifacts_enforces_file_count_cap():
    listed = [(f"/workspace/out/{i}.txt", 1) for i in range(5)]
    to_fetch, skipped = sandbox_agent._select_artifacts(listed, max_files=2, max_bytes=1000)
    assert to_fetch == ["/workspace/out/0.txt", "/workspace/out/1.txt"]
    assert len(skipped) == 3


# ---------------------- run_sandbox_task ----------------------


def _exec_result(stdout: bytes, ok: bool = True):
    r = MagicMock(stdout=stdout, stderr=b"", exit_code=0 if ok else 1)
    r.ok.return_value = ok
    return r


@pytest.mark.asyncio
async def test_sandbox_output_dir_joins_live_cwd():
    session = MagicMock()
    session.exec = AsyncMock(return_value=_exec_result(b"/root\n"))
    assert await prod_sandbox_agent._sandbox_output_dir(session) == "/root/out"
    session.exec.assert_awaited_once_with("pwd", shell=False)


@pytest.mark.asyncio
async def test_sandbox_output_dir_none_when_pwd_fails():
    session = MagicMock()
    session.exec = AsyncMock(return_value=_exec_result(b"", ok=False))
    assert await prod_sandbox_agent._sandbox_output_dir(session) is None


@pytest.mark.asyncio
async def test_collect_artifacts_returns_empty_when_out_dir_unresolved():
    # out_dir is now resolved once by the caller and passed in — None
    # (pwd couldn't be queried) must short-circuit without touching exec.
    session = MagicMock()
    session.exec = AsyncMock()

    artifacts, skipped = await prod_sandbox_agent._collect_artifacts(session, None)
    assert artifacts == []
    assert skipped == []
    session.exec.assert_not_awaited()


@pytest.mark.asyncio
async def test_collect_artifacts_returns_empty_when_find_fails():
    session = MagicMock()
    session.exec = AsyncMock(return_value=_exec_result(b"", ok=False))

    artifacts, skipped = await prod_sandbox_agent._collect_artifacts(session, "/out")
    assert artifacts == []
    assert skipped == []


@pytest.mark.asyncio
async def test_collect_artifacts_reads_listed_files():
    session = MagicMock()
    # Regression: with the Shell-only capability, the SDK's nominal
    # "/workspace" root is never created, so files must be read via plain
    # exec("cat", ...), not session.read() (which validates paths against
    # that never-created "/workspace" root and rejects ours as "not
    # relative to the manifest").
    session.exec = AsyncMock(side_effect=[
        _exec_result(b"3 /out/a.txt\n"),
        _exec_result(b"hi!"),
    ])

    artifacts, skipped = await prod_sandbox_agent._collect_artifacts(session, "/out")
    assert len(artifacts) == 1
    assert artifacts[0].name == "a.txt"
    assert artifacts[0].data == b"hi!"
    assert skipped == []
    session.exec.assert_awaited_with("cat", "--", "/out/a.txt", shell=False)


@pytest.mark.asyncio
async def test_collect_artifacts_skips_file_that_fails_to_read():
    session = MagicMock()
    session.exec = AsyncMock(side_effect=[
        _exec_result(b"3 /out/a.txt\n"),
        _exec_result(b"", ok=False),  # cat fails
    ])

    artifacts, skipped = await prod_sandbox_agent._collect_artifacts(session, "/out")
    assert artifacts == []
    assert skipped == []


@pytest.mark.asyncio
async def test_collect_artifacts_returns_relative_skip_notes_for_oversized_file():
    # Regression: skip notes used to be built from the raw find path
    # (e.g. "/root/out/huge.bin (skipped: ...)"). The model must never see
    # the sandbox's absolute filesystem paths (see _relative_artifact_name
    # usage elsewhere) — the skip note now names the file the same way a
    # successfully-fetched artifact would be named.
    session = MagicMock()
    huge = b"x" * (sandbox_agent.MAX_ARTIFACT_BYTES + 1)
    session.exec = AsyncMock(return_value=_exec_result(
        f"{len(huge)} /root/out/huge.bin\n".encode()
    ))

    artifacts, skipped = await prod_sandbox_agent._collect_artifacts(session, "/root/out")
    assert artifacts == []
    assert len(skipped) == 1
    assert skipped[0].startswith("huge.bin ")
    assert "/root/out" not in skipped[0]


@pytest.mark.asyncio
async def test_collect_artifacts_skip_notes_handle_filenames_with_spaces():
    # Regression: skip notes used to be split on the first plain space,
    # which mangled any path containing one (find -printf reports the raw
    # path as-is, spaces included) — must split on the " (skipped: "
    # marker instead.
    session = MagicMock()
    huge = sandbox_agent.MAX_ARTIFACT_BYTES + 1
    session.exec = AsyncMock(return_value=_exec_result(
        f"{huge} /root/out/my plot.png\n".encode()
    ))

    artifacts, skipped = await prod_sandbox_agent._collect_artifacts(session, "/root/out")
    assert artifacts == []
    assert len(skipped) == 1
    assert skipped[0].startswith("my plot.png (skipped: ")


# ---------------------- tool registration in the LLM agent ----------------------


# ---------------------- run_code_sandbox tool behaviour ----------------------


_EMBED_PARAMS = ("channel", "description", "color", "title", "footer")


@pytest.mark.asyncio
async def test_collect_artifacts_filters_by_the_run_marker():
    # A thread's snapshot carries out/ with it, and a resumed run only
    # mkdir -p's it — so without the filter run #2 re-delivers run #1's
    # files. Restored files keep their archived mtimes, so they sort older
    # than the marker touched at the start of this run.
    session = MagicMock()
    session.exec = AsyncMock(return_value=_exec_result(b""))
    await prod_sandbox_agent._collect_artifacts(session, "/root/out", "/root/.dcb_run_start")
    assert session.exec.await_args.args == (
        "find", "/root/out", "-maxdepth", "3", "-type", "f",
        "-newer", "/root/.dcb_run_start", "-printf", "%s %p\n",
    )


@pytest.mark.asyncio
async def test_collect_artifacts_omits_the_filter_without_a_marker():
    # The marker is best-effort (see _mark_run_start); losing it must fall
    # back to the old unfiltered sweep, never to sweeping nothing.
    session = MagicMock()
    session.exec = AsyncMock(return_value=_exec_result(b""))
    await prod_sandbox_agent._collect_artifacts(session, "/root/out", None)
    assert "-newer" not in session.exec.await_args.args


@pytest.mark.asyncio
async def test_mark_run_start_touches_beside_the_output_dir():
    # Never INSIDE out/, or the sweep would find the marker itself and
    # deliver it to the user as an empty file.
    session = MagicMock()
    session.exec = AsyncMock(return_value=_exec_result(b""))
    marker = await prod_sandbox_agent._mark_run_start(session, "/root/out")
    assert marker == "/root/.dcb_run_start"
    session.exec.assert_awaited_once_with("touch", "--", "/root/.dcb_run_start", shell=False)


@pytest.mark.asyncio
async def test_mark_run_start_is_best_effort():
    session = MagicMock()
    session.exec = AsyncMock(side_effect=Exception("no touch"))
    assert await prod_sandbox_agent._mark_run_start(session, "/root/out") is None
    assert await prod_sandbox_agent._mark_run_start(MagicMock(), None) is None


@pytest.mark.asyncio
async def test_collect_deliverables_reads_only_what_was_attached():
    session = MagicMock()
    session.exec = AsyncMock(return_value=_exec_result(b"PNGDATA"))
    entries = [{"path": "out/final.png", "size": 7, "caption": "here you go"}]

    artifacts, skipped = await prod_sandbox_agent._collect_deliverables(session, entries)

    assert skipped == []
    assert [(a.name, a.data, a.caption) for a in artifacts] == [
        ("final.png", b"PNGDATA", "here you go")
    ]
    session.exec.assert_awaited_once_with("cat", "--", "out/final.png", shell=False)


@pytest.mark.asyncio
async def test_deliver_prefers_what_the_agent_attached():
    # The whole point of the fix: out/ may hold v1/v2/v3, but only the
    # attached file is delivered.
    session = MagicMock()
    session.exec = AsyncMock(return_value=_exec_result(b"PNGDATA"))
    sweep = AsyncMock(return_value=([], []))

    with patch.object(prod_sandbox_agent, "_collect_artifacts", sweep):
        artifacts, _ = await prod_sandbox_agent._deliver(
            session, [{"path": "out/v3.png", "size": 7, "caption": ""}], "/root/out", None)

    assert [a.name for a in artifacts] == ["v3.png"]
    sweep.assert_not_awaited()


@pytest.mark.asyncio
async def test_deliver_falls_back_to_the_sweep_when_nothing_was_attached():
    # The safety net for a run cut off before it could attach anything, and
    # for a model that finished without ever calling attach_file.
    recovered = [sandbox_agent.SandboxArtifact(name="plot.png", data=b"x")]
    sweep = AsyncMock(return_value=(recovered, []))

    with patch.object(prod_sandbox_agent, "_collect_artifacts", sweep):
        artifacts, _ = await prod_sandbox_agent._deliver(
            MagicMock(), [], "/root/out", "/root/.dcb_run_start")

    assert artifacts == recovered
    sweep.assert_awaited_once()
    assert sweep.await_args.args[1:] == ("/root/out", "/root/.dcb_run_start")


# ============================================================
# Threads, HITL (ask_user/send_preview_to_thread), multi-sandbox resume,
# persistent memory
# ============================================================

# ---------------------- ensure_sandbox_thread ----------------------

def _thread_mock(thread_id=555):
    thread = MagicMock(spec=discord.Thread)
    thread.id = thread_id
    return thread


@pytest.mark.asyncio
async def test_ensure_sandbox_thread_reuses_existing_thread():
    message = MagicMock()
    message.channel = _thread_mock()
    channel, created = await prod_sandbox_agent.ensure_sandbox_thread(message, "do a thing")
    assert channel is message.channel
    assert created is False
    message.create_thread.assert_not_called()


@pytest.mark.asyncio
async def test_ensure_sandbox_thread_creates_one_off_a_normal_channel():
    message = MagicMock()
    message.channel = MagicMock()  # not a discord.Thread
    new_thread = _thread_mock()
    message.create_thread = AsyncMock(return_value=new_thread)
    channel, created = await prod_sandbox_agent.ensure_sandbox_thread(message, "do a thing")
    assert channel is new_thread
    assert created is True
    message.create_thread.assert_awaited_once()
    assert message.create_thread.await_args.kwargs["name"].startswith("🐳 ")


@pytest.mark.asyncio
async def test_ensure_sandbox_thread_falls_back_when_creation_fails():
    # Some channel types (forum channels, DMs, ...) can't have a thread
    # created off a message at all; the tool must degrade, not fail.
    message = MagicMock()
    message.channel = MagicMock()
    message.create_thread = AsyncMock(side_effect=RuntimeError("can't thread here"))
    channel, created = await prod_sandbox_agent.ensure_sandbox_thread(message, "do a thing")
    assert channel is message.channel
    assert created is False


def test_thread_name_truncates_to_discord_limit():
    name = prod_sandbox_agent._thread_name("x" * 200)
    assert len(name) <= 100


def test_thread_name_falls_back_when_task_empty():
    assert prod_sandbox_agent._thread_name("") == "🐳 Sandbox"


# ---------------------- ask_user / sandbox_ask_user_timeout ----------------------

def test_ask_user_timeout_default(monkeypatch):
    monkeypatch.delenv("SANDBOX_ASK_USER_TIMEOUT", raising=False)
    assert sandbox_agent.sandbox_ask_user_timeout() == 300


def test_ask_user_timeout_override(monkeypatch):
    monkeypatch.setenv("SANDBOX_ASK_USER_TIMEOUT", "45")
    assert sandbox_agent.sandbox_ask_user_timeout() == 45


def _ask_user_ctx(**overrides):
    from agents.tool_context import ToolContext
    ctx = {
        "thread": _thread_mock(),
        "requesting_user_id": 42,
        "conversation": MagicMock(ask=AsyncMock(return_value="asked")),
    }
    ctx.update(overrides)
    return ToolContext(
        context=ctx,
        tool_name="ask_user",
        tool_call_id="a1",
        tool_arguments=json.dumps({"question": "png or jpg?"}),
    )


@pytest.mark.asyncio
async def test_ask_user_hands_the_question_to_the_conversation():
    ctx = _ask_user_ctx()
    result = await prod_sandbox_agent.ask_user.on_invoke_tool(
        ctx, json.dumps({"question": "png or jpg?", "blocking": False,
                         "choices": ["png", "jpg"], "default": "png"}))

    ask = ctx.context["conversation"].ask
    assert result == "asked"
    assert ask.await_args.args[0] == "png or jpg?"
    assert ask.await_args.kwargs["blocking"] is False
    assert ask.await_args.kwargs["choices"] == ["png", "jpg"]
    assert ask.await_args.kwargs["default"] == "png"


@pytest.mark.asyncio
async def test_ask_user_blocks_by_default():
    ctx = _ask_user_ctx()
    await prod_sandbox_agent.ask_user.on_invoke_tool(ctx, json.dumps({"question": "q"}))
    assert ctx.context["conversation"].ask.await_args.kwargs["blocking"] is True


@pytest.mark.asyncio
async def test_ask_user_without_thread_context_tells_model_to_proceed():
    # No thread available at all (the no-thread fallback) must degrade
    # gracefully, never raise.
    ctx = _ask_user_ctx(thread=None, conversation=None)
    result = await prod_sandbox_agent.ask_user.on_invoke_tool(ctx, json.dumps({"question": "q"}))
    assert "proceed using your best judgement" in result.lower()


@pytest.mark.asyncio
async def test_ask_user_uses_full_timeout_without_a_deadline(monkeypatch):
    monkeypatch.setenv("SANDBOX_ASK_USER_TIMEOUT", "45")
    ctx = _ask_user_ctx(deadline=None)

    await prod_sandbox_agent.ask_user.on_invoke_tool(ctx, json.dumps({"question": "q"}))

    assert ctx.context["conversation"].ask.await_args.kwargs["timeout"] == 45.0


# ---------------------- send_preview_to_thread ----------------------

def _preview_ctx(**overrides):
    from agents.tool_context import ToolContext
    ctx = {"thread": _thread_mock(), "session": MagicMock()}
    ctx.update(overrides)
    return ToolContext(
        context=ctx,
        tool_name="send_preview_to_thread",
        tool_call_id="p1",
        tool_arguments=json.dumps({"path": "out/plot.png"}),
    )


def _attach_ctx(**overrides):
    from agents.tool_context import ToolContext
    ctx = {"thread": _thread_mock(), "session": MagicMock(), "deliverables": []}
    ctx.update(overrides)
    return ToolContext(
        context=ctx,
        tool_name="attach_file",
        tool_call_id="a1",
        tool_arguments=json.dumps({"path": "out/plot.png"}),
    )


async def _attach(ctx, **args):
    return await prod_sandbox_agent.attach_file.on_invoke_tool(ctx, json.dumps(args))


@pytest.mark.asyncio
async def test_attach_file_records_the_file_for_delivery():
    # The fix for the core bug: a model that iterates leaves v1/v2/v3 under
    # out/, and only what it attaches is sent.
    session = MagicMock()
    session.exec = AsyncMock(return_value=_exec_result(b"512\n"))
    ctx = _attach_ctx(session=session)

    result = await _attach(ctx, path="out/plot.png", caption="the final one")

    assert ctx.context["deliverables"] == [
        {"path": "out/plot.png", "size": 512, "caption": "the final one", "change_revision": 0}
    ]
    assert "plot.png" in result
    session.exec.assert_awaited_once_with(
        "stat", "-c", "%s", "--", "out/plot.png", shell=False)


@pytest.mark.asyncio
async def test_attach_file_replaces_an_earlier_version_of_the_same_path():
    # "I fixed it, attach again" must not deliver both versions.
    session = MagicMock()
    session.exec = AsyncMock(return_value=_exec_result(b"20\n"))
    ctx = _attach_ctx(session=session, deliverables=[
        {"path": "out/plot.png", "size": 10, "caption": "draft"}])

    result = await _attach(ctx, path="out/plot.png", caption="fixed")

    assert ctx.context["deliverables"] == [
        {"path": "out/plot.png", "size": 20, "caption": "fixed", "change_revision": 0}
    ]
    assert "Replaced" in result


@pytest.mark.asyncio
async def test_attach_file_rejects_a_missing_file():
    # Attaching something that does not exist must be a model-visible error,
    # not a silent promise of a file the user never receives.
    session = MagicMock()
    session.exec = AsyncMock(return_value=_exec_result(b"", ok=False))
    ctx = _attach_ctx(session=session)

    result = await _attach(ctx, path="out/nope.png")

    assert ctx.context["deliverables"] == []
    assert "does not exist" in result


@pytest.mark.asyncio
async def test_attach_file_rejects_an_absolute_path():
    ctx = _attach_ctx()
    result = await _attach(ctx, path="/etc/passwd")
    assert ctx.context["deliverables"] == []
    assert "relative" in result


@pytest.mark.asyncio
async def test_attach_file_rejects_an_oversized_file():
    session = MagicMock()
    size = str(sandbox_agent.MAX_ARTIFACT_BYTES + 1).encode()
    session.exec = AsyncMock(return_value=_exec_result(size))
    ctx = _attach_ctx(session=session)

    result = await _attach(ctx, path="out/huge.bin")

    assert ctx.context["deliverables"] == []
    assert "too large" in result


@pytest.mark.asyncio
async def test_attach_file_refuses_past_the_file_cap():
    session = MagicMock()
    session.exec = AsyncMock(return_value=_exec_result(b"1\n"))
    full = [{"path": f"out/{i}.png", "size": 1, "caption": ""}
            for i in range(sandbox_agent.MAX_ARTIFACT_FILES)]
    ctx = _attach_ctx(session=session, deliverables=full)

    result = await _attach(ctx, path="out/extra.png")

    assert len(ctx.context["deliverables"]) == sandbox_agent.MAX_ARTIFACT_FILES
    assert "limit" in result


@pytest.mark.asyncio
async def test_send_preview_to_thread_sends_the_file():
    thread = _thread_mock()
    thread.send = AsyncMock()
    session = MagicMock()
    session.exec = AsyncMock(return_value=_exec_result(b"PNGDATA"))
    ctx = _preview_ctx(thread=thread, session=session)

    result = await prod_sandbox_agent.send_preview_to_thread.on_invoke_tool(
        ctx, json.dumps({"path": "out/plot.png", "caption": "in progress"})
    )

    thread.send.assert_awaited_once()
    _, kwargs = thread.send.await_args
    # basename, not the flattened out_plot.png: the agent named this file
    # itself, so it should arrive under the name it used (_attachment_name).
    assert kwargs["file"].filename == "plot.png"
    assert kwargs["content"] == "in progress"
    assert "sent" in result.lower()


@pytest.mark.asyncio
async def test_send_preview_to_thread_rejects_absolute_path():
    ctx = _preview_ctx()
    result = await prod_sandbox_agent.send_preview_to_thread.on_invoke_tool(
        ctx, json.dumps({"path": "/etc/passwd"})
    )
    assert "relative" in result.lower()


@pytest.mark.asyncio
async def test_send_preview_to_thread_reports_read_failure():
    session = MagicMock()
    session.exec = AsyncMock(return_value=_exec_result(b"", ok=False))
    ctx = _preview_ctx(session=session)
    result = await prod_sandbox_agent.send_preview_to_thread.on_invoke_tool(
        ctx, json.dumps({"path": "out/missing.png"})
    )
    assert "could not read" in result.lower()


@pytest.mark.asyncio
async def test_send_preview_to_thread_enforces_size_cap():
    session = MagicMock()
    huge = b"x" * (sandbox_agent.MAX_ARTIFACT_BYTES + 1)
    session.exec = AsyncMock(return_value=_exec_result(huge))
    ctx = _preview_ctx(session=session)
    result = await prod_sandbox_agent.send_preview_to_thread.on_invoke_tool(
        ctx, json.dumps({"path": "out/huge.bin"})
    )
    assert "too large" in result.lower()


@pytest.mark.asyncio
async def test_send_preview_to_thread_checks_the_size_before_reading():
    session = MagicMock()
    size = str(sandbox_agent.MAX_ARTIFACT_BYTES + 1).encode()
    session.exec = AsyncMock(return_value=_exec_result(size))
    ctx = _preview_ctx(session=session)
    result = await prod_sandbox_agent.send_preview_to_thread.on_invoke_tool(
        ctx, json.dumps({"path": "out/huge.bin"})
    )
    assert "too large" in result.lower()
    assert session.exec.await_count == 1  # stat only: the file is never read


@pytest.mark.asyncio
async def test_a_file_that_grew_after_it_was_attached_is_not_sent():
    session = MagicMock()
    session.exec = AsyncMock(return_value=_exec_result(b"x" * (sandbox_agent.MAX_ARTIFACT_BYTES + 1)))
    entries = [{"path": "/workspace/jobs/t1/out/plot.png", "size": 10, "caption": ""}]
    artifacts, notes = await prod_sandbox_agent._collect_deliverables(session, entries)
    assert artifacts == [] and "grew past" in notes[0]


@pytest.mark.asyncio
async def test_removing_every_attachment_sends_nothing_rather_than_sweeping():
    session = MagicMock()
    session.exec = AsyncMock(return_value=_exec_result(b"12"))
    ctx = _attach_ctx(session=session, job_dir="jobs/t1", workspace_root="/workspace")
    await _attach(ctx, path="out/plot.png")
    await _attach(ctx, path="out/plot.png", remove=True)
    assert ctx.context["delivery_chosen"] is True
    sweep = AsyncMock(return_value=(["swept"], []))
    with patch.object(prod_sandbox_agent, "_collect_artifacts", sweep):
        assert await prod_sandbox_agent._deliver(session, [], "/w/out", None, chosen=True) == ([], [])
        sweep.assert_not_awaited()
        assert (await prod_sandbox_agent._deliver(session, [], "/w/out", None))[0] == ["swept"]


@pytest.mark.asyncio
async def test_send_preview_to_thread_without_context_is_graceful():
    ctx = _preview_ctx(thread=None, session=None)
    result = await prod_sandbox_agent.send_preview_to_thread.on_invoke_tool(
        ctx, json.dumps({"path": "out/plot.png"})
    )
    assert "no thread" in result.lower()


# ---------------------- the brain's call timeout ----------------------


# ---------------------- run_code_sandbox tool: thread wiring ----------------------


# ---------------------- workspace origin: fresh vs resumed ----------------------


# ---------------------- resumed runs tell the nested model ----------------------


# ---------------------- unrestorable snapshot recovery ----------------------


# ---------------------- shell tools: gate and checkpoints ----------------------
# Thread messages no longer ride on shell output (one delivery path: the
# model input filter — see sandbox_conversation_tests). What the shell
# wrappers own now is refusing a NEW command while input is unanswered, and
# yielding a long command early when someone posts.

def _gate_conversation(thread_id=7):
    from classes import sandbox_thread_inbox as inbox
    ledger = inbox.RunLedger(thread_id=thread_id, requester_id=1)
    return prod_sandbox_agent.SandboxConversation(ledger, _thread_mock(thread_id=thread_id))


@pytest.mark.asyncio
async def test_a_new_command_is_refused_while_a_message_is_unanswered():
    conv = _gate_conversation()
    conv.ledger.deliver(1, 1, "ana", "make it blue")
    conv.ledger.mark_presented([1], anchor=1)
    exec_tool = MagicMock()
    inner = AsyncMock(return_value="exit 0")
    exec_tool.on_invoke_tool = inner
    toolset = MagicMock(exec_command=exec_tool, write_stdin=None)
    prod_sandbox_agent._configure_shell_tools(toolset)

    out = await exec_tool.on_invoke_tool(MagicMock(context={"conversation": conv}), '{"cmd": "ls"}')

    inner.assert_not_awaited()
    assert "Not run" in out and "#1" in out


@pytest.mark.asyncio
async def test_write_stdin_is_never_gated():
    # Observing or stopping a running command is what the model may need to
    # do while it reviews new input, so only NEW work is held back.
    conv = _gate_conversation()
    conv.ledger.deliver(1, 1, "ana", "make it blue")
    conv.ledger.mark_presented([1], anchor=1)
    exec_tool, stdin_tool = MagicMock(), MagicMock()
    exec_tool.on_invoke_tool = AsyncMock(return_value="exit 0")
    stdin_tool.on_invoke_tool = AsyncMock(return_value="polled")
    toolset = MagicMock(exec_command=exec_tool, write_stdin=stdin_tool)
    prod_sandbox_agent._configure_shell_tools(toolset)

    out = await stdin_tool.on_invoke_tool(
        MagicMock(context={"conversation": conv}), '{"session_id": 3}')

    assert out == "polled"


@pytest.mark.asyncio
async def test_a_rejected_tool_call_is_still_tolerated_under_the_gate():
    from pydantic import BaseModel, ValidationError

    class _Args(BaseModel):
        cmd: str

    try:
        _Args.model_validate_json("{}")
    except ValidationError as e:
        err = e
    exec_tool = MagicMock()
    exec_tool.on_invoke_tool = AsyncMock(side_effect=err)
    toolset = MagicMock(exec_command=exec_tool, write_stdin=None)
    prod_sandbox_agent._configure_shell_tools(toolset)

    out = await exec_tool.on_invoke_tool(
        MagicMock(context={"conversation": _gate_conversation()}), "{}")

    assert "Tool call rejected" in out


def _running(session_id=5, output="partial"):
    return (f"Chunk ID: x\nWall time: 5.0 seconds\nProcess running with session ID "
            f"{session_id}\nOutput:\n{output}")


@pytest.mark.asyncio
async def test_a_long_command_is_waited_on_in_checkpoint_slices(monkeypatch):
    monkeypatch.setattr(prod_sandbox_agent, "SHELL_CHECKPOINT_SECONDS", 0.05)
    conv = _gate_conversation()
    exec_invoke = AsyncMock(return_value=_running())
    polls = iter([_running(output="more"), "Chunk ID: y\nProcess exited with code 0\nOutput:\ndone"])
    poll_invoke = AsyncMock(side_effect=lambda ctx, raw: next(polls))
    wrapped = prod_sandbox_agent._with_shell_checkpoints(exec_invoke, poll_invoke)

    out = await wrapped(MagicMock(context={"conversation": conv}),
                        json.dumps({"cmd": "pip install x", "yield_time_ms": 60_000}))

    # the first slice is clamped to the checkpoint, the rest are polls
    assert json.loads(exec_invoke.await_args.args[1])["yield_time_ms"] == 50
    assert poll_invoke.await_count == 2
    assert json.loads(poll_invoke.await_args.args[1])["session_id"] == 5
    # merged into ONE SDK-shaped result: the last slice's status, all output
    from classes.brain_progress import parse_exec_result
    parsed = parse_exec_result(out)
    assert parsed == {"output": "partialmoredone", "exit_code": 0, "process_id": None}


@pytest.mark.asyncio
async def test_a_long_command_hands_back_early_when_someone_posts(monkeypatch):
    monkeypatch.setattr(prod_sandbox_agent, "SHELL_CHECKPOINT_SECONDS", 0.05)
    conv = _gate_conversation()
    conv.ledger.state = "running"
    exec_invoke = AsyncMock(return_value=_running())

    async def _poll(ctx, raw):
        conv.ledger.deliver(9, 1, "ana", "stop, use the other dataset")
        return _running(output="still going")

    poll_invoke = AsyncMock(side_effect=_poll)
    wrapped = prod_sandbox_agent._with_shell_checkpoints(exec_invoke, poll_invoke)

    out = await wrapped(MagicMock(context={"conversation": conv}),
                        json.dumps({"cmd": "make", "yield_time_ms": 600_000}))

    assert poll_invoke.await_count == 1
    assert "still running as session 5" in out


@pytest.mark.asyncio
async def test_a_short_command_is_not_sliced():
    exec_invoke = AsyncMock(return_value="exit 0")
    poll_invoke = AsyncMock()
    wrapped = prod_sandbox_agent._with_shell_checkpoints(exec_invoke, poll_invoke)

    raw = json.dumps({"cmd": "ls", "yield_time_ms": 2000})
    out = await wrapped(MagicMock(context={"conversation": _gate_conversation()}), raw)

    assert out == "exit 0"
    exec_invoke.assert_awaited_once()
    assert exec_invoke.await_args.args[1] == raw
    poll_invoke.assert_not_awaited()


# ---------------------- say_in_thread ----------------------

def _nested_tool_context(context):
    from agents.tool_context import ToolContext
    return ToolContext(context=context, tool_name="t", tool_call_id="t1", tool_arguments="{}")


@pytest.mark.asyncio
async def test_say_in_thread_posts_and_returns_immediately():
    thread = _thread_mock(thread_id=7)
    thread.send = AsyncMock()
    out = await prod_sandbox_agent.say_in_thread.on_invoke_tool(
        _nested_tool_context({"thread": thread}), json.dumps({"text": "trying another way"}))
    thread.send.assert_awaited_once()
    assert "trying another way" in thread.send.await_args.args[0]
    assert "Posted" in out


@pytest.mark.asyncio
async def test_say_in_thread_reports_a_send_failure_without_raising():
    thread = _thread_mock(thread_id=7)
    thread.send = AsyncMock(side_effect=RuntimeError("no perms"))
    out = await prod_sandbox_agent.say_in_thread.on_invoke_tool(
        _nested_tool_context({"thread": thread}), json.dumps({"text": "hello"}))
    assert "Could not post" in out


@pytest.mark.asyncio
async def test_say_in_thread_rejects_an_empty_message():
    thread = _thread_mock(thread_id=7)
    thread.send = AsyncMock()
    out = await prod_sandbox_agent.say_in_thread.on_invoke_tool(
        _nested_tool_context({"thread": thread}), json.dumps({"text": "   "}))
    thread.send.assert_not_awaited()
    assert "empty" in out


@pytest.mark.asyncio
async def test_say_in_thread_degrades_without_a_thread():
    out = await prod_sandbox_agent.say_in_thread.on_invoke_tool(
        _nested_tool_context({}), json.dumps({"text": "hello"}))
    assert "No thread" in out


# ---------------------- run_code_sandbox: origin badge & thread routing ----------------------

def _warm(warm, calls=None):
    """Patches the server workspaces so is_warm answers `warm` (calls, when
    given, collects each guild id asked about)."""
    def _is_warm(guild_id):
        if calls is not None:
            calls.append(guild_id)
        return warm(guild_id) if callable(warm) else warm
    fake = MagicMock(name="workspaces")
    fake.is_warm = MagicMock(side_effect=_is_warm)
    return patch("classes.workspace_manager.workspaces", return_value=fake)


# ---------------------- run_code_sandbox: one run per thread ----------------------


# ---------------------- run_code_sandbox: resume-mismatch correction ----------------------


# ---------------------- closing note ----------------------


# ---------------------- retrying a stopped run stays in the thread ----------------------
#
# Observed failure this covers: a sandbox run timed out after 600s, the outer
# model retried in the same turn, ensure_sandbox_thread asked Discord for a
# second thread off the same message and was refused with 160004 ("a thread
# has already been created for this message"), so the retry ran in the PARENT
# CHANNEL — away from the thread holding the work, and with no snapshot id,
# so the partial workspace the timeout had just saved was abandoned.


@pytest.mark.asyncio
async def test_ensure_sandbox_thread_reuses_a_thread_already_started_off_the_message():
    # Discord allows exactly one thread per message, so create_thread would
    # fail here; reusing is both correct and the only thing that works.
    message = MagicMock()
    message.channel = MagicMock()  # not a discord.Thread
    message.thread = _thread_mock(thread_id=777)
    message.create_thread = AsyncMock()
    channel, created = await prod_sandbox_agent.ensure_sandbox_thread(message, "do a thing")
    assert channel is message.thread
    assert created is False
    message.create_thread.assert_not_awaited()


# ---------------------- the closing embed says how the run ended ----------------------


# ---------------------- elapsed-time marker ----------------------
#
# The budget bullet used to teach the model to measure its own elapsed time
# ("run `date +%s` as your very first command ... re-run it every few turns
# and subtract"). That asked a 27B Q4 model to carry an epoch integer across
# turns and do arithmetic on it, spent one of a budget as small as 10 turns,
# and collided with the follow-up preamble's "run `ls -R .` first". We measure
# it and tell the model the answer instead, on the same carrier as thread
# messages and for the same reason.


def test_run_config_carries_the_input_filter_and_one_group_id():
    # One group id for every pass of a run: Memory files each Runner.run
    # under it, so a fresh id per continuation would cost an extra
    # extraction call at teardown.
    def _filter(data):
        return data

    config = prod_sandbox_agent.build_sandbox_run_config(
        MagicMock(), MagicMock(), input_filter=_filter, group_id="sandbox-abc")
    assert config.call_model_input_filter is _filter
    assert config.group_id == "sandbox-abc"


# ------------- the main agent's shared tools, in the sandbox -------------
# The sandbox agent gets the main agent's lookup and event tools as the same
# objects (tool_functions.shared_tools), not as copies of their code. The
# poll tools are left out. The two that change something (create_event,
# update_event) are gated like attach_file, through a copy, so the main
# agent's own tools are never gated.


@pytest.mark.asyncio
async def test_resolve_requester_prefers_the_cache_and_fails_soft():
    cached = MagicMock(name="cached")
    thread = MagicMock()
    thread.guild.get_member.return_value = cached
    thread.guild.fetch_member = AsyncMock()
    assert await prod_sandbox_agent._resolve_requester(thread, 7) is cached
    thread.guild.fetch_member.assert_not_awaited()

    thread.guild.get_member.return_value = None
    thread.guild.fetch_member = AsyncMock(side_effect=discord.NotFound(MagicMock(status=404), "gone"))
    assert await prod_sandbox_agent._resolve_requester(thread, 7) is None
    assert await prod_sandbox_agent._resolve_requester(None, 7) is None
    assert await prod_sandbox_agent._resolve_requester(thread, None) is None


@pytest.mark.asyncio
async def test_the_web_tools_work_in_a_run_with_no_thread():
    # A sandbox run may have no thread (its "channel" is None); the lookup
    # skips its notice instead of failing on None.send.
    from classes import web_research
    with patch.object(prod_tool_functions, "check_web_request",
                      new=AsyncMock(return_value=(True, ""))),          patch.object(web_research, "research", new=AsyncMock(return_value="brief")):
        out = await prod_tool_functions.web_search.on_invoke_tool(
            _nested_tool_context({"channel": None}), json.dumps({"search_request": "x"}))
    assert out == "brief"


# ---------------------- the server workspace: job folders ----------------------


@pytest.mark.parametrize("workdir, expected", [
    (None, "jobs/t1"), ("", "jobs/t1"), (".", "jobs/t1"), ("out", "jobs/t1/out"),
    ("../../shared", "shared"), ("../../shared/data", "shared/data"),
])
def test_job_relative_takes_paths_from_the_job_folder(workdir, expected):
    assert prod_sandbox_agent._job_relative({"job_dir": "jobs/t1"}, workdir) == expected


def test_job_relative_refuses_to_leave_the_workspace():
    assert prod_sandbox_agent._job_relative({"job_dir": "jobs/t1"}, "../../..") is None
    assert prod_sandbox_agent._job_relative({"job_dir": "jobs/t1"}, "../../../etc") is None


def test_job_paths_pass_through_without_a_job_folder():
    assert prod_sandbox_agent._job_relative({}, "out/a.png") == "out/a.png"
    assert prod_sandbox_agent._job_path({}, "out/a.png") == "out/a.png"
    ctx = {"job_dir": "jobs/t1", "workspace_root": "/workspace"}
    assert prod_sandbox_agent._job_path(ctx, "out/a.png") == "/workspace/jobs/t1/out/a.png"


def _shell_ctx(context):
    return MagicMock(context=context)


@pytest.mark.asyncio
async def test_commands_run_in_the_job_folder():
    seen = []

    async def _exec(ctx, raw):
        seen.append(json.loads(raw))
        return "ok"
    invoke = prod_sandbox_agent._in_job_dir(_exec)
    ctx = _shell_ctx({"job_dir": "jobs/t1"})
    await invoke(ctx, json.dumps({"cmd": "ls"}))
    await invoke(ctx, json.dumps({"cmd": "ls", "workdir": "out"}))
    await invoke(ctx, json.dumps({"cmd": "ls", "workdir": "../../shared"}))
    assert [a["workdir"] for a in seen] == ["jobs/t1", "jobs/t1/out", "shared"]
    assert all(a["cmd"] == "ls" for a in seen)


@pytest.mark.asyncio
async def test_a_workdir_outside_the_workspace_is_refused_as_a_tool_result():
    invoke = prod_sandbox_agent._in_job_dir(AsyncMock(return_value="ran"))
    out = await invoke(_shell_ctx({"job_dir": "jobs/t1"}),
                       json.dumps({"cmd": "ls", "workdir": "../../../etc"}))
    assert "outside the workspace" in out


@pytest.mark.asyncio
async def test_the_job_folder_wrapper_passes_through_without_one_or_on_odd_input():
    inner = AsyncMock(return_value="ran")
    invoke = prod_sandbox_agent._in_job_dir(inner)
    await invoke(_shell_ctx({}), '{"cmd": "ls"}')
    await invoke(_shell_ctx({"job_dir": "jobs/t1"}), "not json")
    assert [c.args[1] for c in inner.await_args_list] == ['{"cmd": "ls"}', "not json"]


@pytest.mark.asyncio
async def test_an_absolute_workdir_is_refused_as_a_result_not_an_error():
    # The SDK raises InvalidManifestPathError for it, which would end the run.
    inner = AsyncMock(return_value="ran")
    invoke = prod_sandbox_agent._in_job_dir(inner)
    out = await invoke(_shell_ctx({"job_dir": "jobs/t1"}), '{"cmd": "ls", "workdir": "/workspace"}')
    assert "relative" in out and not inner.await_count


@pytest.mark.asyncio
async def test_attach_file_reads_from_the_job_folder():
    session = MagicMock()
    session.exec = AsyncMock(return_value=_exec_result(b"12"))
    ctx = _attach_ctx(session=session, job_dir="jobs/t1", workspace_root="/workspace")
    await _attach(ctx, path="out/plot.png")
    assert session.exec.await_args.args[-1] == "/workspace/jobs/t1/out/plot.png"
    assert ctx.context["deliverables"][0]["path"] == "/workspace/jobs/t1/out/plot.png"
    # the name the user sees is still the file's own
    assert prod_sandbox_agent._attachment_name(ctx.context["deliverables"][0]["path"]) == "plot.png"


@pytest.mark.asyncio
async def test_attach_file_refuses_a_path_outside_the_workspace():
    ctx = _attach_ctx(job_dir="jobs/t1", workspace_root="/workspace")
    out = await _attach(ctx, path="../../../etc/passwd")
    assert "outside the workspace" in out
    assert ctx.context["deliverables"] == []


def _lease_runner():
    result = MagicMock()
    result.final_output = "done"
    return AsyncMock(return_value=result)


@pytest.mark.asyncio
async def test_attach_file_messages_never_show_the_absolute_path():
    """A model handed an absolute path reused it as exec_command's workdir,
    which the SDK refuses (see SANDBOX_OUTPUT_DIRNAME)."""
    session = MagicMock()
    session.exec = AsyncMock(return_value=_exec_result(b"", ok=False))
    ctx = _attach_ctx(session=session, job_dir="jobs/t1", workspace_root="/workspace")
    out = await _attach(ctx, path="out/nope.png")
    assert "'out/nope.png' does not exist" in out and "/workspace" not in out
    session.exec = AsyncMock(return_value=_exec_result(
        str(sandbox_agent.MAX_ARTIFACT_BYTES + 1).encode()))
    out = await _attach(ctx, path="out/huge.bin")
    assert "too large" in out and "/workspace" not in out



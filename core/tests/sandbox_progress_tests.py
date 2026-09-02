import json
import os
import sys

# sandbox_progress is imported as core.classes.* here and classes.* in
# production (the app runs with cwd=core/); both resolve as namespace packages.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from unittest.mock import AsyncMock, MagicMock

from core.classes.sandbox_progress import (
    COLOR_FAIL,
    COLOR_RUNNING,
    COLOR_TOOL,
    DESCRIPTION_CHARS,
    EMBED_CHAR_BUDGET,
    FIELD_VALUE_CHARS,
    MAX_FIELDS,
    SandboxProgressHooks,
    SandboxTranscript,
    parse_exec_result,
    sandbox_progress_updates_enabled,
)

# To run this pytest file from the command line, use:
# PYTHONPATH=$(pwd) pytest core/tests/sandbox_progress_tests.py


# ---------------------- parse_exec_result ----------------------

def test_parse_exec_result_exited():
    result = (
        "Chunk ID: ab12cd\n"
        "Wall time: 0.1234 seconds\n"
        "Process exited with code 0\n"
        "Output:\n"
        "hello world"
    )
    assert parse_exec_result(result) == {
        "output": "hello world", "exit_code": 0, "process_id": None,
    }


def test_parse_exec_result_still_running():
    result = (
        "Chunk ID: ef56ab\n"
        "Wall time: 10.0001 seconds\n"
        "Process running with session ID 7\n"
        "Output:\n"
        "partial\n"
        "more"
    )
    assert parse_exec_result(result) == {
        "output": "partial\nmore", "exit_code": None, "process_id": 7,
    }


def test_parse_exec_result_nonzero_exit_and_truncation():
    result = (
        "Chunk ID: aa00bb\n"
        "Wall time: 1.0000 seconds\n"
        "Process exited with code 2\n"
        "Original token count: 999\n"
        "Output:\n"
        "boom"
    )
    parsed = parse_exec_result(result)
    assert parsed["output"] == "boom"
    assert parsed["exit_code"] == 2
    assert parsed["process_id"] is None


def test_parse_exec_result_without_output_header():
    parsed = parse_exec_result("Command timed out after 10.000 seconds.")
    assert parsed == {
        "output": "Command timed out after 10.000 seconds.",
        "exit_code": None,
        "process_id": None,
    }


def test_parse_exec_result_empty_output():
    parsed = parse_exec_result("Chunk ID: x\nWall time: 0.1 s\nOutput:\n")
    assert parsed["output"] == ""


def test_parse_exec_result_non_string():
    parsed = parse_exec_result(object())
    assert parsed["exit_code"] is None
    assert parsed["process_id"] is None
    assert isinstance(parsed["output"], str)


# ---------------------- sandbox_progress_updates_enabled ----------------------

def test_progress_updates_default_off_when_unset():
    # the /sandbox_progress_updates setting is unset (configManager returns
    # False) or None: progress is OFF by default
    for raw in (None, False):
        assert sandbox_progress_updates_enabled(raw) is False, raw


@pytest.mark.parametrize("raw", ["True", "true", " TRUE ", "1", "yes", "on", "On"])
def test_progress_updates_on_values(raw):
    assert sandbox_progress_updates_enabled(raw) is True, raw


@pytest.mark.parametrize("raw", ["False", "false", "0", "no", "off", "garbage", ""])
def test_progress_updates_off_values(raw):
    assert sandbox_progress_updates_enabled(raw) is False, raw


# ---------------------- SandboxTranscript ----------------------

def _render(t):
    return t.render_message()


def _embed_chars(spec) -> int:
    total = len(spec.title) + len(spec.description or "")
    total += sum(len(f.name) + len(f.value) for f in spec.fields)
    return total


def test_transcript_initial_embed_mirrors_tool_embed():
    # no commands yet: just the "Tool Usage"-style embed, nothing in the
    # message body
    body, spec = _render(SandboxTranscript("compute 6*7"))
    assert body == ""
    assert spec.title == "🐳 Sandbox"
    assert spec.description == "Running in sandbox: compute 6*7"
    assert spec.color == COLOR_TOOL
    assert spec.fields == []


def test_transcript_one_field_per_command():
    t = SandboxTranscript("t")
    t.add_command("pip install matplotlib")
    t.add_output("Successfully installed", exit_code=0)
    t.add_command("python goat.py")
    t.add_output("done", exit_code=0)
    _, spec = _render(t)
    assert [f.name for f in spec.fields] == [
        "$ pip install matplotlib", "$ python goat.py",
    ]
    assert "Successfully installed" in spec.fields[0].value
    assert "exit 0" in spec.fields[0].value
    assert "done" in spec.fields[1].value
    assert "```" in spec.fields[0].value  # output in a code fence


def test_transcript_heredoc_field():
    t = SandboxTranscript("t")
    t.add_command("cat > x.py << 'EOF'\nprint(1)\nEOF")
    t.add_output("1", exit_code=0)
    _, spec = _render(t)
    f = spec.fields[0]
    assert f.name == "⌨ Command"
    assert "```sh" in f.value
    assert "cat > x.py" in f.value
    assert "print(1)" in f.value
    assert "1" in f.value
    assert "exit 0" in f.value


def test_transcript_state_colors():
    t = SandboxTranscript("t")
    t.add_command("sleep 30")
    _, spec = _render(t)
    assert spec.color == COLOR_RUNNING
    t.add_output("partial", process_id=4)
    _, spec = _render(t)
    assert spec.color == COLOR_RUNNING
    t.add_output("ok", exit_code=0)
    _, spec = _render(t)
    assert spec.color == COLOR_TOOL
    t.add_command("false")
    t.add_output("boom", exit_code=2)
    _, spec = _render(t)
    assert spec.color == COLOR_FAIL
    # a later successful command clears the failure colour
    t.add_command("true")
    t.add_output("", exit_code=0)
    _, spec = _render(t)
    assert spec.color == COLOR_TOOL


def test_transcript_field_value_capped():
    t = SandboxTranscript("t")
    t.add_command("echo hi")
    t.add_output("x\n" * 5000, exit_code=0)
    _, spec = _render(t)
    f = spec.fields[0]
    assert len(f.value) <= FIELD_VALUE_CHARS
    assert "… " in f.value
    assert "exit 0" in f.value  # the status line survives the cap


def test_transcript_huge_heredoc_capped():
    t = SandboxTranscript("t")
    t.add_command("cat > big.py << 'EOF'\n" + "line\n" * 3000 + "EOF")
    _, spec = _render(t)
    f = spec.fields[0]
    assert len(f.value) <= FIELD_VALUE_CHARS
    assert "… " in f.value


def test_transcript_stdin_shown_in_field():
    t = SandboxTranscript("t")
    t.add_command("read x")
    t.add_input("y\n")
    t.add_output("got it", exit_code=0)
    _, spec = _render(t)
    f = spec.fields[0]
    assert f.name == "⌨ Command"
    assert "read x" in f.value
    assert "y" in f.value
    assert "got it" in f.value


def test_transcript_stdin_without_command():
    t = SandboxTranscript("t")
    t.add_input("y\n")
    t.add_output("ok", exit_code=0)
    _, spec = _render(t)
    assert spec.fields[0].name == "⌨ stdin"
    assert "y" in spec.fields[0].value


def test_transcript_note_and_thinking_in_body():
    t = SandboxTranscript("t")
    t.add_note("starting container…")
    t.set_thinking(True)
    body, _ = _render(t)
    assert "starting container…" in body
    assert "model thinking" in body
    t.set_thinking(False)
    body, _ = _render(t)
    assert "model thinking" not in body


def test_transcript_skips_empty_output():
    t = SandboxTranscript("t")
    t.add_output("")
    _, spec = _render(t)
    assert spec.fields == []


def test_transcript_fields_fit_embed_budget():
    t = SandboxTranscript("t")
    for i in range(50):
        t.add_command(f"echo {i}")
        t.add_output("o" * 3000, exit_code=0)
    _, spec = _render(t)
    assert len(spec.fields) <= MAX_FIELDS
    assert _embed_chars(spec) <= EMBED_CHAR_BUDGET
    names = [f.name for f in spec.fields]
    assert "$ echo 49" in names
    assert "$ echo 0" not in names  # old fields are evicted
    assert all(len(f.value) <= FIELD_VALUE_CHARS for f in spec.fields)


def test_transcript_fields_chronological():
    t = SandboxTranscript("t")
    t.add_command("first")
    t.add_output("one", exit_code=0)
    t.add_command("second")
    t.add_output("two", exit_code=0)
    _, spec = _render(t)
    assert [f.name for f in spec.fields] == ["$ first", "$ second"]
    assert "one" in spec.fields[0].value
    assert "two" in spec.fields[1].value


def test_transcript_long_task_truncated_in_description():
    _, spec = _render(SandboxTranscript("x" * 5000))
    prefix = "Running in sandbox: "
    assert spec.description.startswith(prefix)
    assert len(spec.description) <= len(prefix) + DESCRIPTION_CHARS
    # the task tail is truncated, ellipsis marks what was cut
    assert spec.description[len(prefix):len(prefix) + 2] == "… "


# ---------------------- SandboxProgressHooks ----------------------

def _tool(name: str):
    tool = MagicMock()
    tool.name = name
    return tool


def _ctx(tool_arguments: str):
    ctx = MagicMock()
    ctx.tool_arguments = tool_arguments
    return ctx


def _channel_with_message():
    message = MagicMock()
    message.edit = AsyncMock()
    channel = MagicMock()
    channel.send = AsyncMock(return_value=message)
    return channel, message


def _last_flush(message):
    """(content, single discord.Embed) of the most recent edit."""
    last = message.edit.await_args_list[-1]
    content = last.kwargs.get("content", last.args[0] if last.args else "")
    embeds = last.kwargs.get("embeds") or []
    return content, (embeds[0] if embeds else None)


@pytest.mark.asyncio
async def test_start_sends_initial_embed_like_tool_embed():
    channel, message = _channel_with_message()
    hooks = SandboxProgressHooks(channel, "print 42", edit_interval=0)
    await hooks.start()
    channel.send.assert_awaited_once()
    sent = channel.send.await_args
    embeds = sent.kwargs.get("embeds") or []
    assert len(embeds) == 1
    assert embeds[0].title == "🐳 Sandbox"
    assert "Running in sandbox: print 42" in embeds[0].description
    assert embeds[0].color.value == COLOR_TOOL
    assert len(embeds[0].fields) == 0  # no commands yet
    content = sent.args[0]
    assert "starting container…" in content
    # the header lives in the embed now, not as cut-off emoji text
    assert "🐳" not in content


@pytest.mark.asyncio
async def test_tool_start_end_stream_command_and_output():
    channel, message = _channel_with_message()
    hooks = SandboxProgressHooks(channel, "t", edit_interval=0)
    await hooks.start()
    message.edit.reset_mock()

    await hooks.on_tool_start(
        _ctx(json.dumps({"cmd": "echo hi"})), None, _tool("exec_command"))
    await hooks.on_tool_end(
        _ctx(json.dumps({"cmd": "echo hi"})), None, _tool("exec_command"),
        "Chunk ID: x\nWall time: 0.1 s\nProcess exited with code 0\nOutput:\nhi")

    assert message.edit.await_count >= 1
    content, embed = _last_flush(message)
    assert content
    assert len(embed.fields) == 1
    assert embed.fields[0].name == "$ echo hi"
    assert "hi" in embed.fields[0].value
    assert "exit 0" in embed.fields[0].value
    assert embed.color.value == COLOR_TOOL


@pytest.mark.asyncio
async def test_throttle_delays_edits_until_finalize():
    channel, message = _channel_with_message()
    hooks = SandboxProgressHooks(channel, "t", edit_interval=3600)  # never due
    await hooks.start()
    message.edit.reset_mock()

    await hooks.on_tool_start(
        _ctx(json.dumps({"cmd": "echo a"})), None, _tool("exec_command"))
    message.edit.assert_not_awaited()  # still throttled

    await hooks.finalize("done")  # the final flush bypasses the throttle
    content, embed = _last_flush(message)
    assert embed.fields[0].name == "$ echo a"
    assert "done" in content


@pytest.mark.asyncio
async def test_finalize_is_idempotent():
    channel, message = _channel_with_message()
    hooks = SandboxProgressHooks(channel, "t", edit_interval=0)
    await hooks.start()
    message.edit.reset_mock()
    await hooks.finalize("done")
    count_after_first = message.edit.await_count
    await hooks.finalize("again")
    assert message.edit.await_count == count_after_first


@pytest.mark.asyncio
async def test_write_stdin_poll_without_output_is_skipped():
    channel, message = _channel_with_message()
    hooks = SandboxProgressHooks(channel, "t", edit_interval=0)
    await hooks.start()
    message.edit.reset_mock()

    await hooks.on_tool_end(
        _ctx(json.dumps({"session_id": 3, "chars": ""})), None,
        _tool("write_stdin"),
        "Chunk ID: x\nWall time: 0.3 s\nOutput:\n")
    message.edit.assert_not_awaited()


@pytest.mark.asyncio
async def test_write_stdin_with_output_is_appended():
    channel, message = _channel_with_message()
    hooks = SandboxProgressHooks(channel, "t", edit_interval=0)
    await hooks.start()

    await hooks.on_tool_end(
        _ctx(json.dumps({"session_id": 3, "chars": ""})), None,
        _tool("write_stdin"),
        "Chunk ID: x\nWall time: 0.3 s\nProcess exited with code 1\nOutput:\nerr")

    content, embed = _last_flush(message)
    assert len(embed.fields) == 1
    assert "err" in embed.fields[0].value
    assert "**exit 1**" in embed.fields[0].value
    assert embed.color.value == COLOR_FAIL
    # write_stdin never adds a new "$" command field
    assert not any(f.name.startswith("$ ") for f in embed.fields)


@pytest.mark.asyncio
async def test_write_stdin_chars_shown_as_input():
    channel, message = _channel_with_message()
    hooks = SandboxProgressHooks(channel, "t", edit_interval=0)
    await hooks.start()

    await hooks.on_tool_start(
        _ctx(json.dumps({"session_id": 3, "chars": "y\n"})), None,
        _tool("write_stdin"))
    _, embed = _last_flush(message)
    assert embed.fields[0].name == "⌨ stdin"
    assert "y" in embed.fields[0].value


@pytest.mark.asyncio
async def test_unrelated_tools_are_ignored():
    channel, message = _channel_with_message()
    hooks = SandboxProgressHooks(channel, "t", edit_interval=0)
    await hooks.start()
    message.edit.reset_mock()

    await hooks.on_tool_start(_ctx("{}"), None, _tool("web_search"))
    await hooks.on_tool_end(_ctx("{}"), None, _tool("web_search"), "results")
    message.edit.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_edit_backs_off_and_run_continues():
    channel, message = _channel_with_message()
    message.edit = AsyncMock(side_effect=Exception("rate limited"))
    hooks = SandboxProgressHooks(channel, "t", edit_interval=0)
    await hooks.start()

    # must not raise, even though every edit fails
    await hooks.on_tool_start(
        _ctx(json.dumps({"cmd": "echo a"})), None, _tool("exec_command"))
    await hooks.finalize("done")  # force-flush still attempted, still safe


@pytest.mark.asyncio
async def test_deleted_message_falls_back_to_send():
    import discord
    channel, message = _channel_with_message()
    not_found = discord.NotFound(MagicMock(status=404), "deleted")
    message.edit = AsyncMock(side_effect=not_found)
    hooks = SandboxProgressHooks(channel, "t", edit_interval=0)
    await hooks.start()

    await hooks.on_tool_start(
        _ctx(json.dumps({"cmd": "echo a"})), None, _tool("exec_command"))
    await hooks.finalize("done")

    assert channel.send.await_count == 2  # initial + resend after deletion
    last = channel.send.await_args
    content = last.args[0]
    embeds = last.kwargs.get("embeds") or []
    assert embeds[0].fields[0].name == "$ echo a"
    assert "done" in content


@pytest.mark.asyncio
async def test_send_failure_on_start_does_not_break_hooks():
    channel = MagicMock()
    channel.send = AsyncMock(side_effect=Exception("no permissions"))
    hooks = SandboxProgressHooks(channel, "t", edit_interval=0)
    await hooks.start()  # must not raise

    await hooks.on_tool_start(
        _ctx(json.dumps({"cmd": "echo a"})), None, _tool("exec_command"))
    await hooks.finalize("done")  # must not raise


@pytest.mark.asyncio
async def test_hooks_never_raise_on_garbage():
    channel, _ = _channel_with_message()
    hooks = SandboxProgressHooks(channel, "t", edit_interval=0)
    await hooks.on_tool_start(_ctx("not json"), None, _tool("exec_command"))
    await hooks.on_tool_start(_ctx("{}"), None, _tool("exec_command"))  # no cmd
    await hooks.on_tool_end(_ctx("{}"), None, _tool("exec_command"), 12345)
    await hooks.on_tool_start(None, None, _tool("web_search"))
    await hooks.on_llm_start(None, None, "sys", [])
    await hooks.on_llm_end(None, None, "resp")
    await hooks.finalize("done")


@pytest.mark.asyncio
async def test_message_stays_under_discord_limits():
    channel, message = _channel_with_message()
    hooks = SandboxProgressHooks(channel, "task " * 100, edit_interval=0)
    await hooks.start()
    for i in range(30):
        await hooks.on_tool_start(
            _ctx(json.dumps({"cmd": f"echo line-{i} " + "x" * 50})), None,
            _tool("exec_command"))
        await hooks.on_tool_end(
            _ctx(json.dumps({})), None, _tool("exec_command"),
            f"Chunk ID: {i}\nWall time: 0.1 s\nOutput:\n" + "out " * 100)
    content, embed = _last_flush(message)
    assert len(content) <= 2000
    assert len(embed.fields) <= MAX_FIELDS
    total = len(embed.title) + len(embed.description or "")
    total += sum(len(f.name) + len(f.value) for f in embed.fields)
    assert total <= EMBED_CHAR_BUDGET
    assert all(len(f.value) <= FIELD_VALUE_CHARS for f in embed.fields)
    # the newest command is still visible after eviction
    assert any("line-29" in f.name for f in embed.fields)


# ---------------------- workspace origin badge ----------------------

def test_transcript_has_no_workspace_note_by_default():
    # the no-thread fallback path passes nothing, and must look exactly as
    # it did before the badge existed
    _, spec = _render(SandboxTranscript("do a thing"))
    assert spec.description == "Running in sandbox: do a thing"


def test_transcript_puts_the_workspace_note_above_the_task():
    _, spec = _render(SandboxTranscript("do a thing", workspace_note="♻️ **Resumed**"))
    assert spec.description == "♻️ **Resumed**\nRunning in sandbox: do a thing"


def test_hooks_pass_the_workspace_note_into_the_transcript():
    hooks = SandboxProgressHooks(MagicMock(), "do a thing", workspace_note="🆕 **Fresh**")
    _, spec = hooks.transcript.render_message()
    assert spec.description.startswith("🆕 **Fresh**\n")

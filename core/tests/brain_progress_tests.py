import json
import os
import sys

# brain_progress is imported as classes.* here and in production (the app
# runs with cwd=core/).

import pytest
from unittest.mock import AsyncMock, MagicMock

from classes.brain_progress import (
    COLOR_FAIL,
    COLOR_RUNNING,
    COLOR_TOOL,
    DESCRIPTION_CHARS,
    EMBED_CHAR_BUDGET,
    FIELD_VALUE_CHARS,
    MAX_FIELDS,
    BrainProgressHooks,
    SandboxTranscript,
    parse_exec_result,
)
from classes.tool_embeds import (
    BRAIN_DONE, BRAIN_NOT_FINISHED, BRAIN_STOPPED, BRAIN_WORKING, COLOR_BRAIN, COLOR_DONE,
    COLOR_STOPPED, work_task_text,
)

# To run this pytest file from the command line, use:
# pytest core/tests/brain_progress_tests.py


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





# ---------------------- SandboxTranscript ----------------------

def _render(t):
    return t.render_message()


def _embed_chars(spec) -> int:
    total = len(spec.title) + len(spec.description or "") + len(spec.footer or "")
    total += sum(len(f.name) + len(f.value) for f in spec.fields)
    return total


def test_transcript_initial_embed_mirrors_tool_embed():
    # no commands yet: just what the static run embed shows, nothing in the
    # message body
    body, spec = _render(SandboxTranscript("compute 6*7"))
    assert body == ""
    assert spec.title == BRAIN_WORKING
    assert spec.description == "compute 6*7"
    assert spec.color == COLOR_TOOL == COLOR_BRAIN
    assert spec.fields == []
    assert spec.footer is None


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
    _, spec = _render(SandboxTranscript("word " * 1000))
    assert len(spec.description) <= DESCRIPTION_CHARS
    # the start of the task is kept, cut at a word; the ellipsis marks the cut
    assert spec.description.startswith("word word")
    assert spec.description.endswith("word…")


# ---------------------- BrainProgressHooks ----------------------

def _tool(name: str):
    tool = MagicMock()
    tool.name = name
    return tool


def _ctx(tool_arguments: str):
    ctx = MagicMock()
    ctx.tool_arguments = tool_arguments
    return ctx


async def _begin(hooks):
    """The first workspace action, which is what posts the status embed."""
    await hooks.on_tool_start(_ctx("{}"), None, _tool("attach_file"))


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
async def test_nothing_is_posted_until_the_workspace_is_used():
    # D18: a call that only searches or posts a poll shows only that tool's
    # own embed.
    channel, message = _channel_with_message()
    hooks = BrainProgressHooks(channel, "find the weather", stream=True, edit_interval=0)
    await hooks.on_llm_start(None, None, "sys", [])
    await hooks.on_tool_start(_ctx("{}"), None, _tool("web_search"))
    await hooks.on_tool_end(_ctx("{}"), None, _tool("web_search"), "results")
    await hooks.on_llm_end(None, None, "resp")
    await hooks.finalize(ok=True, note="done")
    channel.send.assert_not_awaited()
    assert hooks.started is False


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", ["exec_command", "attach_file", "save_messages_to_file"])
async def test_the_first_workspace_action_posts_the_embed(tool):
    channel, message = _channel_with_message()
    hooks = BrainProgressHooks(channel, "print 42", edit_interval=0)
    await hooks.on_tool_start(_ctx(json.dumps({"cmd": "ls"})), None, _tool(tool))
    channel.send.assert_awaited_once()
    embeds = channel.send.await_args.kwargs.get("embeds") or []
    assert len(embeds) == 1
    assert embeds[0].title == BRAIN_WORKING
    assert embeds[0].description == "print 42"
    assert embeds[0].color.value == COLOR_TOOL
    assert hooks.started is True


@pytest.mark.asyncio
async def test_without_streaming_the_embed_shows_no_commands_and_is_retitled_at_the_end():
    channel, message = _channel_with_message()
    hooks = BrainProgressHooks(channel, "chart it", stream=False, edit_interval=0)
    args = _ctx(json.dumps({"cmd": "python plot.py"}))
    await hooks.on_tool_start(args, None, _tool("exec_command"))
    await hooks.on_tool_end(args, None, _tool("exec_command"),
                            "Chunk ID: x\nProcess exited with code 0\nOutput:\nok")
    assert len(channel.send.await_args.kwargs["embeds"][0].fields) == 0
    await hooks.finalize(ok=True)
    _, embed = _last_flush(message)
    assert embed.title == BRAIN_DONE and embed.color.value == COLOR_DONE
    assert len(embed.fields) == 0


@pytest.mark.asyncio
async def test_a_stopped_call_is_titled_and_coloured_as_stopped():
    channel, message = _channel_with_message()
    hooks = BrainProgressHooks(channel, "t", stream=True, edit_interval=0)
    await _begin(hooks)
    await hooks.finalize(ok=False, note="⚠️ It ran out of time.")
    content, embed = _last_flush(message)
    assert embed.title == BRAIN_STOPPED and embed.color.value == COLOR_STOPPED
    assert "ran out of time" in content


@pytest.mark.asyncio
async def test_a_helper_nothing_stopped_is_not_called_stopped():
    channel, message = _channel_with_message()
    hooks = BrainProgressHooks(channel, "t", stream=True, edit_interval=0)
    await _begin(hooks)
    await hooks.finalize(ok=False, stopped=False)
    _, embed = _last_flush(message)
    assert embed.title == BRAIN_NOT_FINISHED


@pytest.mark.asyncio
async def test_tool_start_end_stream_command_and_output():
    channel, message = _channel_with_message()
    hooks = BrainProgressHooks(channel, "t", stream=True, edit_interval=0)
    await _begin(hooks)
    message.edit.reset_mock()

    await hooks.on_tool_start(
        _ctx(json.dumps({"cmd": "echo hi"})), None, _tool("exec_command"))
    await hooks.on_tool_end(
        _ctx(json.dumps({"cmd": "echo hi"})), None, _tool("exec_command"),
        "Chunk ID: x\nWall time: 0.1 s\nProcess exited with code 0\nOutput:\nhi")

    assert message.edit.await_count >= 1
    content, embed = _last_flush(message)
    assert len(embed.fields) == 1
    assert embed.fields[0].name == "$ echo hi"
    assert "hi" in embed.fields[0].value
    assert "exit 0" in embed.fields[0].value
    assert embed.color.value == COLOR_TOOL


@pytest.mark.asyncio
async def test_throttle_delays_edits_until_finalize():
    channel, message = _channel_with_message()
    hooks = BrainProgressHooks(channel, "t", stream=True, edit_interval=3600)  # never due
    await _begin(hooks)
    message.edit.reset_mock()

    await hooks.on_tool_start(
        _ctx(json.dumps({"cmd": "echo a"})), None, _tool("exec_command"))
    message.edit.assert_not_awaited()  # still throttled

    await hooks.finalize(note="done")  # the final flush bypasses the throttle
    content, embed = _last_flush(message)
    assert embed.fields[0].name == "$ echo a"
    assert "done" in content


@pytest.mark.asyncio
async def test_finalize_is_idempotent():
    channel, message = _channel_with_message()
    hooks = BrainProgressHooks(channel, "t", stream=True, edit_interval=0)
    await _begin(hooks)
    message.edit.reset_mock()
    await hooks.finalize(note="done")
    count_after_first = message.edit.await_count
    await hooks.finalize(note="again")
    assert message.edit.await_count == count_after_first


@pytest.mark.asyncio
async def test_write_stdin_poll_without_output_is_skipped():
    channel, message = _channel_with_message()
    hooks = BrainProgressHooks(channel, "t", stream=True, edit_interval=0)
    await _begin(hooks)
    message.edit.reset_mock()

    await hooks.on_tool_end(
        _ctx(json.dumps({"session_id": 3, "chars": ""})), None,
        _tool("write_stdin"),
        "Chunk ID: x\nWall time: 0.3 s\nOutput:\n")
    message.edit.assert_not_awaited()


@pytest.mark.asyncio
async def test_write_stdin_with_output_is_appended():
    channel, message = _channel_with_message()
    hooks = BrainProgressHooks(channel, "t", stream=True, edit_interval=0)
    await _begin(hooks)

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
    hooks = BrainProgressHooks(channel, "t", stream=True, edit_interval=0)
    await _begin(hooks)

    await hooks.on_tool_start(
        _ctx(json.dumps({"session_id": 3, "chars": "y\n"})), None,
        _tool("write_stdin"))
    _, embed = _last_flush(message)
    assert embed.fields[0].name == "⌨ stdin"
    assert "y" in embed.fields[0].value


@pytest.mark.asyncio
async def test_unrelated_tools_are_ignored():
    channel, message = _channel_with_message()
    hooks = BrainProgressHooks(channel, "t", stream=True, edit_interval=0)
    await _begin(hooks)
    message.edit.reset_mock()

    await hooks.on_tool_start(_ctx("{}"), None, _tool("web_search"))
    await hooks.on_tool_end(_ctx("{}"), None, _tool("web_search"), "results")
    message.edit.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_edit_backs_off_and_run_continues():
    channel, message = _channel_with_message()
    message.edit = AsyncMock(side_effect=Exception("rate limited"))
    hooks = BrainProgressHooks(channel, "t", stream=True, edit_interval=0)
    await _begin(hooks)

    # must not raise, even though every edit fails
    await hooks.on_tool_start(
        _ctx(json.dumps({"cmd": "echo a"})), None, _tool("exec_command"))
    await hooks.finalize(note="done")  # force-flush still attempted, still safe


@pytest.mark.asyncio
async def test_deleted_message_falls_back_to_send():
    import discord
    channel, message = _channel_with_message()
    not_found = discord.NotFound(MagicMock(status=404), "deleted")
    message.edit = AsyncMock(side_effect=not_found)
    hooks = BrainProgressHooks(channel, "t", stream=True, edit_interval=0)
    await _begin(hooks)

    await hooks.on_tool_start(
        _ctx(json.dumps({"cmd": "echo a"})), None, _tool("exec_command"))
    await hooks.finalize(note="done")

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
    hooks = BrainProgressHooks(channel, "t", stream=True, edit_interval=0)
    await _begin(hooks)  # must not raise

    await hooks.on_tool_start(
        _ctx(json.dumps({"cmd": "echo a"})), None, _tool("exec_command"))
    await hooks.finalize(note="done")  # must not raise


@pytest.mark.asyncio
async def test_hooks_never_raise_on_garbage():
    channel, _ = _channel_with_message()
    hooks = BrainProgressHooks(channel, "t", stream=True, edit_interval=0)
    await hooks.on_tool_start(_ctx("not json"), None, _tool("exec_command"))
    await hooks.on_tool_start(_ctx("{}"), None, _tool("exec_command"))  # no cmd
    await hooks.on_tool_end(_ctx("{}"), None, _tool("exec_command"), 12345)
    await hooks.on_tool_start(None, None, _tool("web_search"))
    await hooks.on_llm_start(None, None, "sys", [])
    await hooks.on_llm_end(None, None, "resp")
    await hooks.finalize(note="done")


@pytest.mark.asyncio
async def test_message_stays_under_discord_limits():
    channel, message = _channel_with_message()
    hooks = BrainProgressHooks(channel, "task " * 100, stream=True, edit_interval=0)
    await _begin(hooks)
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


# ---------------------- title, files and reply hint ----------------------

def test_transcript_shows_the_title_files_and_footer_it_is_given():
    _, spec = _render(SandboxTranscript("do a thing", title=BRAIN_WORKING,
                                        files=["a.csv", "b.png"], footer="reply here"))
    assert spec.title == BRAIN_WORKING
    assert spec.description == "do a thing\n📎 a.csv, b.png"
    assert spec.footer == "reply here"


def test_transcript_matches_the_static_embed_text():
    _, spec = _render(SandboxTranscript("do a thing", files=["a.csv"]))
    assert spec.description == work_task_text("do a thing", ["a.csv"], DESCRIPTION_CHARS)


def test_hooks_pass_the_files_into_the_embed():
    channel, _ = _channel_with_message()
    hooks = BrainProgressHooks(channel, "do a thing", files=["a.csv"])
    _, spec = hooks.transcript.render_message()
    embed = hooks._build_embed(spec)
    assert embed.title == BRAIN_WORKING
    assert embed.description.endswith("a.csv")


def test_the_footer_counts_against_the_embed_budget():
    big_footer = "f" * 2000
    t = SandboxTranscript("t", footer=big_footer)
    for i in range(20):
        t.add_command(f"cmd{i}")
        t.add_output("x" * 900, exit_code=0)
    _, spec = _render(t)
    assert _embed_chars(spec) <= 6000


def _call_ctx(call_id: str, cmd: str):
    ctx = _ctx(json.dumps({"cmd": cmd}))
    ctx.tool_call_id = call_id
    return ctx


"""classes/brain_progress.py: the brain's live status card. The pure
transcript is rendered to a card spec; the hooks post and edit it."""
import json

import pytest
from unittest.mock import AsyncMock, MagicMock

from classes import card_kit as K
from classes.brain_progress import (
    PROGRESS_HINT,
    BrainProgressHooks,
    WorkTranscript,
    parse_exec_result,
)
from classes.tool_cards import BRAIN_DONE, BRAIN_NOT_FINISHED, BRAIN_STOPPED, BRAIN_WORKING


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



# ---------------------- WorkTranscript ----------------------

def _texts(spec):
    return [b.content for c in K.split(spec) for b in c if isinstance(b, K.Text)]


def _all(spec) -> str:
    return "\n".join(_texts(spec))


def test_the_card_starts_with_the_label_and_the_quoted_request():
    spec = WorkTranscript("print 42", files=["a.csv"], started=1000).render()
    assert spec.label == f"{BRAIN_WORKING} · started <t:1000:R>"
    assert spec.colour == K.WORKING
    assert _texts(spec)[:2] == [f"-# {BRAIN_WORKING} · started <t:1000:R>",
                                "> print 42\n-# 📎 a.csv"]


def test_each_command_shows_its_text_output_and_how_it_went():
    t = WorkTranscript("t")
    t.add_output("hi", exit_code=0, block=t.add_command("echo hi"))
    t.add_output("boom", exit_code=2, block=t.add_command("false"))
    t.add_output("half", process_id=7, block=t.add_command("sleep 9"))
    text = _all(t.render())
    assert "```sh\necho hi\n```\n```\nhi\n```\n-# ✅ Finished" in text
    assert "-# ❌ Failed · exit 2" in text
    assert "```sh\nsleep 9\n```\n```\nhalf\n```\n-# ⏳ Running…" in text
    assert text.index("echo hi") < text.index("false") < text.index("sleep 9")


def test_a_heredoc_and_typed_input_are_shown_whole():
    t = WorkTranscript("t")
    t.add_command("cat > a.py <<'EOF'\nprint(1)\nEOF")
    t.add_input("y")
    text = _all(t.render())
    assert "```sh\ncat > a.py <<'EOF'\nprint(1)\nEOF\n```" in text
    assert "-# ⌨ typed in\n```sh\ny\n```" in text


def test_a_fence_in_the_output_never_ends_the_block_early():
    t = WorkTranscript("t")
    t.add_output("```\nnot code\n```", exit_code=0, block=t.add_command("cat README.md"))
    for text in _texts(t.render()):
        assert text.count("```") % 2 == 0


def test_output_shows_its_newest_part():
    t = WorkTranscript("t")
    t.add_output("old " * 1000 + "NEWEST", exit_code=0, block=t.add_command("noisy"))
    text = _all(t.render())
    assert "NEWEST" in text and text.count("old ") < 1000


def test_notes_and_thinking_and_the_hint():
    t = WorkTranscript("t", hint=PROGRESS_HINT)
    t.add_note("⚠️ Something went wrong.")
    t.set_thinking(True)
    spec = t.render()
    assert "⚠️ Something went wrong." in _all(spec)
    assert spec.footer == f"💭 Thinking… · {PROGRESS_HINT}"


def test_nothing_is_dropped_when_the_commands_outgrow_a_message():
    t = WorkTranscript("task " * 100)
    for i in range(30):
        t.add_output("out " * 200, exit_code=0, block=t.add_command(f"echo line-{i}"))
    cards = K.split(t.render())
    assert len(cards) > 1
    text = "\n".join(b.content for c in cards for b in c if isinstance(b, K.Text))
    assert all(f"echo line-{i}" in text for i in range(30))
    for c in cards:
        assert sum(len(b.content) for b in c if isinstance(b, K.Text)) <= K.MAX_TEXT


def test_the_end_says_how_it_went_and_how_long_it_took():
    t = WorkTranscript("t", started=1000)
    t.set_thinking(True)
    t.set_final(BRAIN_DONE, K.OK, 130)
    spec = t.render()
    assert spec.label == f"{BRAIN_DONE} · took 2 min" and spec.colour == K.OK
    assert spec.footer == ""


def test_the_request_is_shown_in_full():
    task = "word " * 600
    assert "word " * 599 in _all(WorkTranscript(task).render()).replace("\n> ", " ")


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
    """The first workspace action, which is what posts the status card."""
    await hooks.on_tool_start(_ctx("{}"), None, _tool("attach_file"))


def _channel_with_message():
    message = MagicMock()
    message.edit = AsyncMock()
    channel = MagicMock()
    channel.send = AsyncMock(return_value=message)
    return channel, message


def _view_text(view) -> str:
    return "\n".join(b.content for b in view.blocks if isinstance(b, K.Text))


def _last(message):
    """The view of the most recent edit."""
    return message.edit.await_args_list[-1].kwargs["view"]


@pytest.mark.asyncio
async def test_nothing_is_posted_until_the_workspace_is_used():
    # D18: a call that only searches or posts a poll shows only that tool's
    # own card.
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
async def test_the_first_workspace_action_posts_the_card(tool):
    channel, message = _channel_with_message()
    hooks = BrainProgressHooks(channel, "print 42", edit_interval=0, clock=lambda: 1000)
    await hooks.on_tool_start(_ctx(json.dumps({"cmd": "ls"})), None, _tool(tool))
    channel.send.assert_awaited_once()
    kwargs = channel.send.await_args.kwargs
    assert "content" not in kwargs and "embeds" not in kwargs
    view = kwargs["view"]
    assert view.kind == K.NOTICE and view.colour == K.WORKING
    assert _view_text(view).startswith(f"-# {BRAIN_WORKING} · started <t:1000:R>\n> print 42")
    assert hooks.started is True


@pytest.mark.asyncio
async def test_without_streaming_the_card_shows_no_commands_and_is_relabelled_at_the_end():
    channel, message = _channel_with_message()
    times = iter([1000, 1100])
    hooks = BrainProgressHooks(channel, "chart it", stream=False, edit_interval=0,
                               clock=lambda: next(times))
    args = _ctx(json.dumps({"cmd": "python plot.py"}))
    await hooks.on_tool_start(args, None, _tool("exec_command"))
    await hooks.on_tool_end(args, None, _tool("exec_command"), "Output:\nok")
    await hooks.finalize(ok=True)
    view = _last(message)
    assert "plot.py" not in _view_text(view)
    assert _view_text(view).startswith(f"-# {BRAIN_DONE} · took 2 min")
    assert view.colour == K.OK


@pytest.mark.asyncio
async def test_a_stopped_call_is_labelled_and_coloured_as_stopped():
    channel, message = _channel_with_message()
    hooks = BrainProgressHooks(channel, "t", edit_interval=0)
    await _begin(hooks)
    await hooks.finalize(ok=False, note="⚠️ It kept repeating itself.")
    view = _last(message)
    assert BRAIN_STOPPED in _view_text(view) and view.colour == K.FAIL
    assert "It kept repeating itself." in _view_text(view)


@pytest.mark.asyncio
async def test_a_helper_nothing_stopped_is_not_called_stopped():
    channel, message = _channel_with_message()
    hooks = BrainProgressHooks(channel, "t", edit_interval=0)
    await _begin(hooks)
    await hooks.finalize(ok=False, stopped=False)
    view = _last(message)
    assert BRAIN_NOT_FINISHED in _view_text(view) and view.colour == K.WARN


@pytest.mark.asyncio
async def test_tool_start_end_stream_command_and_output():
    channel, message = _channel_with_message()
    hooks = BrainProgressHooks(channel, "t", stream=True, edit_interval=0)
    await _begin(hooks)
    args = _ctx(json.dumps({"cmd": "echo hi"}))
    await hooks.on_tool_start(args, None, _tool("exec_command"))
    await hooks.on_tool_end(args, None, _tool("exec_command"),
                            "Process exited with code 0\nOutput:\nhi")
    text = _view_text(_last(message))
    assert "```sh\necho hi\n```\n```\nhi\n```\n-# ✅ Finished" in text
    assert PROGRESS_HINT in text


@pytest.mark.asyncio
async def test_throttle_delays_edits_until_finalize():
    channel, message = _channel_with_message()
    hooks = BrainProgressHooks(channel, "t", stream=True, edit_interval=3600)
    await _begin(hooks)
    for cmd in ("echo a", "echo b"):
        await hooks.on_tool_start(_ctx(json.dumps({"cmd": cmd})), None, _tool("exec_command"))
    message.edit.assert_not_awaited()
    await hooks.finalize()
    message.edit.assert_awaited_once()
    assert "echo b" in _view_text(_last(message))


@pytest.mark.asyncio
async def test_finalize_is_idempotent():
    channel, message = _channel_with_message()
    hooks = BrainProgressHooks(channel, "t", edit_interval=0)
    await _begin(hooks)
    await hooks.finalize()
    await hooks.finalize(ok=False)
    assert message.edit.await_count == 1


@pytest.mark.asyncio
async def test_write_stdin_poll_without_output_is_skipped_and_with_output_is_shown():
    channel, message = _channel_with_message()
    hooks = BrainProgressHooks(channel, "t", stream=True, edit_interval=0)
    await _begin(hooks)
    await hooks.on_tool_start(_ctx(json.dumps({"cmd": "sleep 5"})), None, _tool("exec_command"))
    edits = message.edit.await_count
    await hooks.on_tool_end(_ctx("{}"), None, _tool("write_stdin"), "Process running with session ID 3\nOutput:\n")
    assert message.edit.await_count == edits
    await hooks.on_tool_end(_ctx("{}"), None, _tool("write_stdin"), "Process exited with code 0\nOutput:\ndone")
    assert "done" in _view_text(_last(message))


@pytest.mark.asyncio
async def test_write_stdin_chars_shown_as_input():
    channel, message = _channel_with_message()
    hooks = BrainProgressHooks(channel, "t", stream=True, edit_interval=0)
    await _begin(hooks)
    await hooks.on_tool_start(_ctx(json.dumps({"cmd": "python"})), None, _tool("exec_command"))
    await hooks.on_tool_start(_ctx(json.dumps({"chars": "print(1)\n"})), None, _tool("write_stdin"))
    assert "-# ⌨ typed in\n```sh\nprint(1)\n```" in _view_text(_last(message))


@pytest.mark.asyncio
async def test_failed_edit_backs_off_and_run_continues():
    channel, message = _channel_with_message()
    message.edit = AsyncMock(side_effect=Exception("rate limited"))
    hooks = BrainProgressHooks(channel, "t", stream=True, edit_interval=0)
    await _begin(hooks)
    await hooks.on_tool_start(_ctx(json.dumps({"cmd": "echo a"})), None, _tool("exec_command"))
    assert hooks._backoff_until > 0
    await hooks.finalize(note="done")  # force-flush still attempted, still safe


@pytest.mark.asyncio
async def test_deleted_message_falls_back_to_send():
    import discord
    channel, message = _channel_with_message()
    message.edit = AsyncMock(side_effect=discord.NotFound(MagicMock(status=404), "deleted"))
    hooks = BrainProgressHooks(channel, "t", stream=True, edit_interval=0)
    await _begin(hooks)
    await hooks.on_tool_start(_ctx(json.dumps({"cmd": "echo a"})), None, _tool("exec_command"))
    await hooks.finalize(note="done")
    assert channel.send.await_count == 2  # the first, and again after the deletion
    text = _view_text(channel.send.await_args.kwargs["view"])
    assert "echo a" in text and "done" in text


@pytest.mark.asyncio
async def test_send_failure_on_start_does_not_break_hooks():
    channel = MagicMock()
    channel.send = AsyncMock(side_effect=Exception("no permissions"))
    hooks = BrainProgressHooks(channel, "t", stream=True, edit_interval=0)
    await _begin(hooks)  # must not raise
    await hooks.on_tool_start(_ctx(json.dumps({"cmd": "echo a"})), None, _tool("exec_command"))
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
async def test_a_long_run_spills_into_more_messages_and_edits_only_the_newest_while_working():
    sent = []

    async def send(**kwargs):
        message = MagicMock()
        message.edit = AsyncMock()
        sent.append(message)
        return message

    channel = MagicMock()
    channel.send = AsyncMock(side_effect=send)
    hooks = BrainProgressHooks(channel, "t", stream=True, edit_interval=0)
    await _begin(hooks)
    for i in range(30):
        args = _ctx(json.dumps({"cmd": f"echo line-{i}"}))
        await hooks.on_tool_start(args, None, _tool("exec_command"))
        await hooks.on_tool_end(args, None, _tool("exec_command"),
                                "Process exited with code 0\nOutput:\n" + "out " * 150)
    assert len(sent) > 1
    # once a message is full it is left alone while the run goes on
    before = sent[0].edit.await_count
    for i in range(30, 33):
        args = _ctx(json.dumps({"cmd": f"echo line-{i}"}))
        await hooks.on_tool_start(args, None, _tool("exec_command"))
    assert sent[0].edit.await_count == before
    await hooks.finalize(ok=True)
    # the finish recolours them all
    assert sent[0].edit.await_count == before + 1
    views = [m.edit.await_args.kwargs["view"] for m in sent if m.edit.await_args]
    assert all(v.colour == K.OK for v in views)
    shown = "\n".join(_view_text(m.edit.await_args.kwargs["view"]) if m.edit.await_args
                      else "" for m in sent) + "\n".join(
        _view_text(c.kwargs["view"]) for c in channel.send.await_args_list)
    assert all(f"echo line-{i}" in shown for i in range(33))


def test_hooks_pass_the_files_and_label_into_the_card():
    channel, _ = _channel_with_message()
    hooks = BrainProgressHooks(channel, "do a thing", files=["a.csv"],
                               label="🛠️ Build · step 2 of 3", title="Write it up")
    spec = hooks.transcript.render()
    assert spec.label == "🛠️ Build · step 2 of 3" and spec.title == "Write it up"
    assert _texts(spec)[2] == "> do a thing\n-# 📎 a.csv"


@pytest.mark.asyncio
async def test_two_commands_at_once_get_their_own_output():
    channel, message = _channel_with_message()
    hooks = BrainProgressHooks(channel, "t", stream=True, edit_interval=0)
    await _begin(hooks)
    first, second = _ctx(json.dumps({"cmd": "echo one"})), _ctx(json.dumps({"cmd": "echo two"}))
    first.tool_call_id, second.tool_call_id = "a", "b"
    await hooks.on_tool_start(first, None, _tool("exec_command"))
    await hooks.on_tool_start(second, None, _tool("exec_command"))
    await hooks.on_tool_end(first, None, _tool("exec_command"), "Process exited with code 0\nOutput:\nONE")
    text = _view_text(_last(message))
    assert text.index("echo one") < text.index("ONE") < text.index("echo two")


# ---------------------- the status line (off form) ----------------------

from unittest.mock import patch

from classes import brain_status


def _cmd(cmd: str):
    return _ctx(json.dumps({"cmd": cmd}))


@pytest.mark.asyncio
async def test_the_first_post_already_says_what_it_is_doing_in_words():
    channel, message = _channel_with_message()
    hooks = BrainProgressHooks(channel, "make a gif", edit_interval=0)
    await hooks.on_tool_start(_cmd("pip install pillow"), None, _tool("exec_command"))
    text = _view_text(channel.send.await_args.kwargs["view"])
    assert "⏳ Running pip" in text and "pip install pillow" not in text
    await hooks.finalize(ok=True)


@pytest.mark.asyncio
async def test_the_status_line_is_written_in_plain_words_and_goes_at_the_end():
    channel, message = _channel_with_message()
    hooks = BrainProgressHooks(channel, "make a gif of a bee", edit_interval=0)
    hooks.status_interval = 0
    complete = AsyncMock(return_value="installing python libraries.")
    with patch.object(brain_status._llm, "complete", complete):
        await hooks.on_tool_start(_cmd("pip install pillow"), None, _tool("exec_command"))
        await hooks._status_task
    assert "⏳ Installing python libraries" in _view_text(_last(message))
    asked = complete.await_args.args[0][1]["content"]
    assert "make a gif of a bee" in asked and "ran: pip install pillow" in asked
    await hooks.finalize(ok=True)
    assert "⏳" not in _view_text(_last(message))


@pytest.mark.asyncio
async def test_the_line_waits_out_its_interval_and_a_finished_call_writes_no_more():
    channel, message = _channel_with_message()
    hooks = BrainProgressHooks(channel, "t", edit_interval=0)
    complete = AsyncMock(return_value="Creating the script")
    with patch.object(brain_status._llm, "complete", complete):
        await hooks.on_tool_start(_cmd("cat > a.py << 'EOF'"), None, _tool("exec_command"))
        await hooks._status_task  # the first line comes at once
        await hooks.on_tool_start(_cmd("python a.py"), None, _tool("exec_command"))
        await hooks.on_tool_start(_cmd("python a.py"), None, _tool("exec_command"))
        await hooks.finalize(ok=True)  # within the 30 s: the next line never comes
    assert complete.await_count == 1
    assert hooks._status_task is None


@pytest.mark.asyncio
async def test_a_failed_line_keeps_the_one_there_was():
    channel, message = _channel_with_message()
    hooks = BrainProgressHooks(channel, "t", edit_interval=0)
    hooks.status_interval = 0
    await hooks.on_tool_start(_cmd("python a.py"), None, _tool("exec_command"))
    await hooks._status_task  # the side call fails (conftest)
    assert hooks.transcript._status == "Running a Python script"
    await hooks.finalize(ok=True)


@pytest.mark.asyncio
async def test_the_streamed_form_has_no_status_line_and_makes_no_side_call():
    channel, message = _channel_with_message()
    hooks = BrainProgressHooks(channel, "t", stream=True, edit_interval=0)
    complete = AsyncMock(return_value="Creating the script")
    with patch.object(brain_status._llm, "complete", complete):
        await hooks.on_tool_start(_cmd("python a.py"), None, _tool("exec_command"))
        await hooks.finalize(ok=True)
    complete.assert_not_awaited()
    assert "⏳ Running" not in _view_text(channel.send.await_args.kwargs["view"])


def test_the_status_prompt_stays_small_and_shows_the_shape():
    assert len(brain_status.PROMPT) <= 300, len(brain_status.PROMPT)
    assert "Installing python libraries" in brain_status.PROMPT


@pytest.mark.parametrize("raw, line", [
    ("Creating the script", "Creating the script"),
    ('"drawing the frames."', "Drawing the frames"),
    ("<think>hmm</think>\n- Building the GIF\n", "Building the GIF"),
    ("", None),
    ("I am now going to carefully install every single library this job needs", None),
])
def test_only_a_short_line_is_a_status(raw, line):
    assert brain_status.clean_status(raw) == line


def test_steps_read_as_what_was_done_and_polls_say_nothing():
    assert brain_status.step_text("exec_command", {"cmd": "pip install pillow"}) == \
        "ran: pip install pillow"
    assert brain_status.step_text("attach_file", {"path": "out/bee.gif"}) == \
        "sent the user the file out/bee.gif"
    assert brain_status.step_text("write_stdin", {"chars": ""}) is None
    assert brain_status.step_text("report", {}) is None
    assert len(brain_status.step_text("exec_command", {"cmd": "x" * 5000})) == brain_status.STEP_CHARS


def test_the_line_is_the_brains_work_on_the_background_lane(monkeypatch):
    monkeypatch.setenv("BRAIN_LLM_HOST", "https://brain.example")
    assert brain_status._config().host.startswith("https://brain.example")
    assert brain_status._llm.lane == "background"

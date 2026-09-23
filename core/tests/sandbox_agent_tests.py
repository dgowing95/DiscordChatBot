import asyncio
import json
import os
import sys
import time

# Imported as `classes.X` — the same name the app uses (it runs with cwd=core/,
# and pyproject.toml puts core/ on the test path). `sandbox_agent` and
# `prod_sandbox_agent` are therefore the SAME module object, so it no longer
# matters which one a patch targets; both aliases are kept only because the
# tests below already name them.

import discord
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from classes import sandbox_agent
import classes.sandbox_agent as prod_sandbox_agent
import classes.tool_functions as prod_tool_functions

# To run this pytest file from the command line, use:
# PYTHONPATH=$(pwd) pytest core/tests/sandbox_agent_tests.py


@pytest.fixture(autouse=True)
def _no_conversation_record_redis():
    """run_code_sandbox loads and saves a per-thread conversation record in
    Redis; no test here should reach a real one."""
    store = MagicMock(load=AsyncMock(return_value=None), save=AsyncMock())
    with patch("classes.sandbox_conversation_store.SandboxConversationStore",
               return_value=store):
        yield store


# ---------------------- sandbox_enabled ----------------------

def test_enabled_env_values(monkeypatch):
    cases = [
        ("1", True), ("true", True), ("True", True), ("on", True),
        ("0", False), ("false", False), ("no", False), ("off", False),
        ("", False),
    ]
    for value, expected in cases:
        monkeypatch.setenv("SANDBOX_ENABLED", value)
        assert sandbox_agent.sandbox_enabled() is expected, value


def test_enabled_defaults_on(monkeypatch):
    monkeypatch.delenv("SANDBOX_ENABLED", raising=False)
    assert sandbox_agent.sandbox_enabled() is True


def test_enabled_trims_whitespace(monkeypatch):
    monkeypatch.setenv("SANDBOX_ENABLED", "  0  ")
    assert sandbox_agent.sandbox_enabled() is False


# ---------------------- sandbox_image ----------------------

def test_image_default(monkeypatch):
    monkeypatch.delenv("SANDBOX_IMAGE", raising=False)
    assert sandbox_agent.sandbox_image() == "python:3.14-slim"


def test_image_override(monkeypatch):
    monkeypatch.setenv("SANDBOX_IMAGE", "python:3.13-slim")
    assert sandbox_agent.sandbox_image() == "python:3.13-slim"


# ---------------------- sandbox_max_turns / sandbox_timeout ----------------------

def test_max_turns_default(monkeypatch):
    monkeypatch.delenv("SANDBOX_MAX_TURNS", raising=False)
    assert sandbox_agent.sandbox_max_turns() == 10


def test_max_turns_override(monkeypatch):
    monkeypatch.setenv("SANDBOX_MAX_TURNS", "7")
    assert sandbox_agent.sandbox_max_turns() == 7


def test_max_turns_invalid_falls_back(monkeypatch):
    for bad in ("abc", "-3", "0", ""):
        monkeypatch.setenv("SANDBOX_MAX_TURNS", bad)
        assert sandbox_agent.sandbox_max_turns() == 10, bad


def test_timeout_default(monkeypatch):
    monkeypatch.delenv("SANDBOX_TIMEOUT", raising=False)
    assert sandbox_agent.sandbox_timeout() == 600


def test_timeout_override(monkeypatch):
    monkeypatch.setenv("SANDBOX_TIMEOUT", "42")
    assert sandbox_agent.sandbox_timeout() == 42


def test_timeout_invalid_falls_back(monkeypatch):
    for bad in ("abc", "-1", "0"):
        monkeypatch.setenv("SANDBOX_TIMEOUT", bad)
        assert sandbox_agent.sandbox_timeout() == 600, bad


# ---------------------- sandbox LLM (model / host / api key) ----------------------

def _clear_llm_env(monkeypatch):
    for var in ("SANDBOX_MODEL", "SANDBOX_LLM_HOST", "SANDBOX_LLM_API_KEY",
                "MODEL", "LLM_HOST", "LLM_PASS"):
        monkeypatch.delenv(var, raising=False)


def test_sandbox_llm_defaults_to_main_llm(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("MODEL", "orcarouter/Qwen3.8-27B-Uncensored-GGUF:Q4_K_M")
    monkeypatch.setenv("LLM_HOST", "http://llamacpp:8080")
    monkeypatch.setenv("LLM_PASS", "llamacpp")
    assert sandbox_agent.sandbox_model() == "orcarouter/Qwen3.8-27B-Uncensored-GGUF:Q4_K_M"
    assert sandbox_agent.sandbox_llm_host() == "http://llamacpp:8080"
    assert sandbox_agent.sandbox_llm_api_key() == "llamacpp"


def test_sandbox_llm_falls_back_to_incode_defaults(monkeypatch):
    _clear_llm_env(monkeypatch)
    assert sandbox_agent.sandbox_model() == "qwen3:4b"
    assert sandbox_agent.sandbox_llm_host() == "http://llamacpp:8080"
    assert sandbox_agent.sandbox_llm_api_key() == "llamacpp"


def test_sandbox_llm_openrouter_override(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("MODEL", "some/local-model")
    monkeypatch.setenv("SANDBOX_MODEL", "deepseek/deepseek-v4-flash-0731")
    monkeypatch.setenv("SANDBOX_LLM_HOST", "https://openrouter.ai/api")
    monkeypatch.setenv("SANDBOX_LLM_API_KEY", "sk-or-test")
    assert sandbox_agent.sandbox_model() == "deepseek/deepseek-v4-flash-0731"
    assert sandbox_agent.sandbox_llm_host() == "https://openrouter.ai/api"
    assert sandbox_agent.sandbox_llm_api_key() == "sk-or-test"


def test_sandbox_llm_empty_override_falls_back(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("MODEL", "some/local-model")
    for empty in ("", "  "):
        monkeypatch.setenv("SANDBOX_MODEL", empty)
        assert sandbox_agent.sandbox_model() == "some/local-model", repr(empty)


def test_build_sandbox_agent_uses_configured_llm(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("SANDBOX_MODEL", "deepseek/deepseek-v4-flash-0731")
    monkeypatch.setenv("SANDBOX_LLM_HOST", "https://openrouter.ai/api")
    monkeypatch.setenv("SANDBOX_LLM_API_KEY", "sk-or-test")
    agent = sandbox_agent.build_sandbox_agent(None)
    assert agent.model.model == "deepseek/deepseek-v4-flash-0731"
    # the client stores the base URL with /v1 appended by the core
    # (the openai client normalizes it to a trailing-slash URL)
    assert agent.model._client.base_url == "https://openrouter.ai/api/v1/"
    assert agent.model._client.api_key == "sk-or-test"


def test_build_sandbox_agent_defaults_to_main_llm(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("MODEL", "some/local-model")
    monkeypatch.setenv("LLM_HOST", "http://llamacpp:8080")
    monkeypatch.setenv("LLM_PASS", "llamacpp")
    agent = sandbox_agent.build_sandbox_agent(None)
    assert agent.model.model == "some/local-model"
    assert agent.model._client.base_url == "http://llamacpp:8080/v1/"
    assert agent.model._client.api_key == "llamacpp"


def test_build_sandbox_agent_suppresses_default_base_instructions(monkeypatch):
    # Regression: with the SDK default base prompt the model is told to
    # use an apply_patch tool we do not expose (Shell capability only),
    # and the run aborts with ModelBehaviorError("Model produced
    # apply_patch call without an apply_patch tool."). The empty string
    # suppresses the SDK default entirely.
    agent = sandbox_agent.build_sandbox_agent(None)
    assert agent.base_instructions == ""
    # and our instructions scope the file-touching tools positively, so the
    # model has no reason to reach for an editor. Naming apply_patch here
    # would be self-defeating: the failure mode is the model HALLUCINATING a
    # call to it, and a 27B Q4 model handed the name is likelier to emit it.
    assert "The only way to touch files is the shell" in agent.instructions
    assert "apply_patch" not in agent.instructions


def test_build_sandbox_agent_never_tells_model_an_absolute_output_path():
    # Regression (found live in production): exec_command's `workdir`
    # argument is validated by the SDK's manifest system, which rejects ANY
    # absolute path outright ("manifest path must be relative: ..."), even
    # a real, already-created one. A model told an absolute path reused it
    # as `workdir` and the tool call failed. The model must only ever see
    # the relative `out/` dirname, whether or not the real path was
    # resolved and mkdir -p'd server-side.
    for out_dir in ("/root/out", "/out", None):
        agent = sandbox_agent.build_sandbox_agent(out_dir)
        assert "/root/out" not in agent.instructions
        assert not any(
            line.strip().startswith("/") for line in agent.instructions.splitlines()
        )
        # The rule is now stated as the reason it actually exists (the SDK's
        # manifest check rejects absolute paths) rather than the stale claim
        # that /workspace does not exist — session.start() does materialize
        # it, so the old wording contradicted the same bullet's own closing
        # clause. See _OUTPUT_BULLET_RESOLVED.
        assert "use relative paths only" in agent.instructions.lower()
        assert "rejects every absolute path" in agent.instructions


def test_build_sandbox_agent_resolved_output_path_warns_against_workdir():
    agent = sandbox_agent.build_sandbox_agent("/root/out")
    assert "already been created for you" in agent.instructions
    assert "workdir" in agent.instructions
    assert "out/" in agent.instructions


def test_build_sandbox_agent_falls_back_when_output_path_unresolved():
    # No regression versus pre-fix behavior when pwd couldn't be queried.
    agent = sandbox_agent.build_sandbox_agent(None)
    assert "save it under the `out/` directory" in agent.instructions
    assert "mkdir -p out" in agent.instructions


def test_sandbox_instructions_explain_out_in_exactly_one_place():
    # The prompt reached ~5000 chars by accretion, with `out/` explained in
    # five separate bullets that had started to drift apart. One home per
    # concept is the rule this pins: `out/` mechanics belong to the output
    # bullet, and the remaining mentions are references, not restatements.
    agent = sandbox_agent.build_sandbox_agent("/root/out")
    assert agent.instructions.count("out/") <= 4
    # This ceiling is a deliberate brake, not a comfortable bound: the prompt
    # is ~3870 chars, so there is under 100 to spare. It is meant to fail the
    # next person who adds a paragraph instead of editing one — the exact
    # accretion that took this prompt to ~5000 chars with `out/` explained in
    # five places. Rewrite to fit; only raise it with a reason.
    #
    # Raised once, from 3800, when attach_file replaced "everything under
    # out/ is sent" as the delivery rule: the output bullet had to teach the
    # selection step, and the final-report bullet had to be re-aimed at the
    # user (it is now posted in the thread, not just handed to the caller).
    # Both were rewritten in place, not added to.
    assert len(agent.instructions) < 3950


def test_sandbox_instructions_tells_model_to_stop_after_verification():
    agent = sandbox_agent.build_sandbox_agent(None)
    assert "confirmed a result is correct, stop" in agent.instructions


def test_sandbox_instructions_tells_model_its_turn_and_time_budget(monkeypatch):
    monkeypatch.setenv("SANDBOX_MAX_TURNS", "7")
    monkeypatch.setenv("SANDBOX_TIMEOUT", "123")
    agent = sandbox_agent.build_sandbox_agent(None)
    assert "7 turns" in agent.instructions
    assert "123 seconds" in agent.instructions
    # The escape hatch has to name the delivery step, not just the saving:
    # once anything is attached the out/ fallback sweep no longer runs, so
    # "save it under out/" alone would not reach the user.
    assert "save and attach your best current partial output" in agent.instructions


def test_sandbox_instructions_do_not_teach_the_model_to_measure_time():
    # The budget bullet used to hand the model a procedure — run `date +%s`
    # first, re-run it, subtract — which asked a small local model to carry
    # an epoch integer across turns, spent one of as few as 10 turns, and
    # collided with RESUMED_TASK_PREAMBLE's "run `ls -R .` first". The
    # elapsed time is injected into every shell result instead
    # (_with_elapsed_note), so the prompt only has to say it is there.
    agent = sandbox_agent.build_sandbox_agent(None)
    assert "date +%s" not in agent.instructions
    assert "tells you how much of that time you have used" in agent.instructions


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


def test_build_sandbox_agent_registers_tolerant_shell_tools():
    agent = sandbox_agent.build_sandbox_agent(None)
    shell = agent.capabilities[0]
    assert shell.configure_tools is sandbox_agent._configure_shell_tools


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

def _patch_session_lifecycle(client=None, session=None, delete=None, out_dir=None):
    """Patches the client/session creation+teardown helpers so
    run_sandbox_task tests don't touch Docker. Also patches
    _sandbox_output_dir (default None, i.e. unresolved) so tests that don't
    care about the output-path resolution don't need session.exec to
    understand `pwd`/`mkdir -p`. Returns the patch context managers to use
    in a `with`."""
    client = client if client is not None else MagicMock(name="client")
    if session is None:
        session = MagicMock(name="session")
        # A real exec result, not a bare AsyncMock: an AsyncMock's own
        # return_value is another AsyncMock, so `res.ok()` would hand
        # production code a coroutine (truthy, never awaited) instead of a
        # bool — which is not how a live session behaves.
        session.exec = AsyncMock(return_value=_exec_result(b""))
    return (
        patch.object(prod_sandbox_agent, "build_sandbox_client", return_value=client),
        patch.object(prod_sandbox_agent, "_create_sandbox_session",
                     AsyncMock(return_value=(session, False))),
        patch.object(prod_sandbox_agent, "_delete_sandbox_session",
                     delete if delete is not None else AsyncMock()),
        patch.object(prod_sandbox_agent, "_sandbox_output_dir", AsyncMock(return_value=out_dir)),
    )


@pytest.mark.asyncio
async def test_run_sandbox_task_returns_final_output(monkeypatch):
    monkeypatch.setenv("SANDBOX_MAX_TURNS", "6")
    agent = MagicMock(name="agent")
    run_config = MagicMock(name="run_config")
    result = MagicMock()
    result.final_output = "done: 42"
    runner = MagicMock()
    runner.run = AsyncMock(return_value=result)
    p_client, p_session, p_delete, p_out_dir = _patch_session_lifecycle()

    with patch.object(prod_sandbox_agent, "Runner", runner), \
         patch.object(prod_sandbox_agent, "build_sandbox_agent", return_value=agent), \
         patch.object(prod_sandbox_agent, "build_sandbox_run_config", return_value=run_config), \
         patch.object(prod_sandbox_agent, "_collect_artifacts", AsyncMock(return_value=([], []))), \
         p_client, p_session, p_delete, p_out_dir:
        output = await prod_sandbox_agent.run_sandbox_task("print 42")

    assert output.text == "done: 42"
    assert output.artifacts == []
    runner.run.assert_awaited_once()
    args, kwargs = runner.run.call_args
    assert args[0] is agent
    assert args[1] == "print 42"
    assert kwargs["max_turns"] == 6
    assert kwargs["run_config"] is run_config


@pytest.mark.asyncio
async def test_run_sandbox_task_resolves_output_dir_before_run(monkeypatch):
    # Fix 3: the output path must be resolved and mkdir -p'd BEFORE
    # Runner.run starts, and the same path handed to build_sandbox_agent
    # so the model is told exactly where to save files.
    result = MagicMock()
    result.final_output = "done"
    runner = MagicMock()
    runner.run = AsyncMock(return_value=result)
    session = MagicMock(name="session")
    session.exec = AsyncMock(return_value=_exec_result(b"ok"))
    build_agent = MagicMock(name="build_sandbox_agent", return_value=MagicMock(name="agent"))
    p_client, p_session, p_delete, p_out_dir = _patch_session_lifecycle(
        session=session, out_dir="/root/out")

    with patch.object(prod_sandbox_agent, "Runner", runner), \
         patch.object(prod_sandbox_agent, "build_sandbox_agent", build_agent), \
         patch.object(prod_sandbox_agent, "build_sandbox_run_config"), \
         patch.object(prod_sandbox_agent, "_collect_artifacts", AsyncMock(return_value=([], []))), \
         p_client, p_session, p_delete, p_out_dir:
        await prod_sandbox_agent.run_sandbox_task("t")

    # mkdir first, then the run marker beside (never inside) out/ — see
    # _mark_run_start.
    assert [c.args for c in session.exec.await_args_list] == [
        ("mkdir", "-p", "/root/out"),
        ("touch", "--", "/root/.dcb_run_start"),
    ]
    build_agent.assert_called_once()
    assert build_agent.call_args.args[0] == "/root/out"
    assert isinstance(build_agent.call_args.args[1], prod_sandbox_agent.SandboxConversation)


@pytest.mark.asyncio
async def test_run_sandbox_task_skips_mkdir_when_output_dir_unresolved(monkeypatch):
    result = MagicMock()
    result.final_output = "done"
    runner = MagicMock()
    runner.run = AsyncMock(return_value=result)
    session = MagicMock(name="session")
    session.exec = AsyncMock()
    build_agent = MagicMock(name="build_sandbox_agent", return_value=MagicMock(name="agent"))
    p_client, p_session, p_delete, p_out_dir = _patch_session_lifecycle(
        session=session, out_dir=None)

    with patch.object(prod_sandbox_agent, "Runner", runner), \
         patch.object(prod_sandbox_agent, "build_sandbox_agent", build_agent), \
         patch.object(prod_sandbox_agent, "build_sandbox_run_config"), \
         patch.object(prod_sandbox_agent, "_collect_artifacts", AsyncMock(return_value=([], []))), \
         p_client, p_session, p_delete, p_out_dir:
        await prod_sandbox_agent.run_sandbox_task("t")

    session.exec.assert_not_awaited()
    build_agent.assert_called_once()
    assert build_agent.call_args.args[0] is None


@pytest.mark.asyncio
async def test_run_sandbox_task_non_string_output_coerced(monkeypatch):
    result = MagicMock()
    result.final_output = 1234
    runner = MagicMock()
    runner.run = AsyncMock(return_value=result)
    p_client, p_session, p_delete, p_out_dir = _patch_session_lifecycle()

    with patch.object(prod_sandbox_agent, "Runner", runner), \
         patch.object(prod_sandbox_agent, "build_sandbox_agent"), \
         patch.object(prod_sandbox_agent, "build_sandbox_run_config"), \
         patch.object(prod_sandbox_agent, "_collect_artifacts", AsyncMock(return_value=([], []))), \
         p_client, p_session, p_delete, p_out_dir:
        output = await prod_sandbox_agent.run_sandbox_task("t")
    assert output.text == "1234"


@pytest.mark.asyncio
async def test_run_sandbox_task_returns_collected_artifacts(monkeypatch):
    result = MagicMock()
    result.final_output = "done"
    runner = MagicMock()
    runner.run = AsyncMock(return_value=result)
    artifacts = [sandbox_agent.SandboxArtifact(name="plot.png", data=b"\x89PNG...")]
    p_client, p_session, p_delete, p_out_dir = _patch_session_lifecycle()

    with patch.object(prod_sandbox_agent, "Runner", runner), \
         patch.object(prod_sandbox_agent, "build_sandbox_agent"), \
         patch.object(prod_sandbox_agent, "build_sandbox_run_config"), \
         patch.object(prod_sandbox_agent, "_collect_artifacts", AsyncMock(return_value=(artifacts, []))), \
         p_client, p_session, p_delete, p_out_dir:
        output = await prod_sandbox_agent.run_sandbox_task("t")

    assert output.artifacts == artifacts


@pytest.mark.asyncio
async def test_run_sandbox_task_timeout_recovers_artifacts_instead_of_raising(monkeypatch):
    # Regression: a timeout used to raise asyncio.TimeoutError straight
    # through, discarding whatever the sandbox had already produced and
    # verified. It must now return SandboxResult(ok=False) with any
    # recovered artifacts, and still tear down the container.
    monkeypatch.setenv("SANDBOX_TIMEOUT", "1")

    async def slow_run(*args, **kwargs):
        await asyncio.sleep(5)

    runner = MagicMock()
    runner.run = slow_run
    delete = AsyncMock()
    artifacts = [sandbox_agent.SandboxArtifact(name="plot.png", data=b"\x89PNG...")]
    p_client, p_session, p_delete, p_out_dir = _patch_session_lifecycle(
        delete=delete, out_dir="/root/out")

    with patch.object(prod_sandbox_agent, "Runner", runner), \
         patch.object(prod_sandbox_agent, "build_sandbox_agent"), \
         patch.object(prod_sandbox_agent, "build_sandbox_run_config"), \
         patch.object(prod_sandbox_agent, "_collect_artifacts", AsyncMock(return_value=(artifacts, []))), \
         p_client, p_session, p_delete, p_out_dir:
        result = await prod_sandbox_agent.run_sandbox_task("sleep forever")

    assert result.ok is False
    assert result.error == "timeout"
    assert result.artifacts == artifacts
    # the container must never be left running, even on timeout
    delete.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_sandbox_task_timeout_recovery_failure_is_swallowed(monkeypatch):
    # Recovery itself must be best-effort: if it raises (e.g. the session
    # is wedged after cancellation), the timeout result still comes back
    # with empty artifacts instead of the recovery error propagating.
    monkeypatch.setenv("SANDBOX_TIMEOUT", "1")

    async def slow_run(*args, **kwargs):
        await asyncio.sleep(5)

    runner = MagicMock()
    runner.run = slow_run
    delete = AsyncMock()
    p_client, p_session, p_delete, p_out_dir = _patch_session_lifecycle(
        delete=delete, out_dir="/root/out")

    with patch.object(prod_sandbox_agent, "Runner", runner), \
         patch.object(prod_sandbox_agent, "build_sandbox_agent"), \
         patch.object(prod_sandbox_agent, "build_sandbox_run_config"), \
         patch.object(prod_sandbox_agent, "_collect_artifacts",
                      AsyncMock(side_effect=RuntimeError("session wedged"))), \
         p_client, p_session, p_delete, p_out_dir:
        result = await prod_sandbox_agent.run_sandbox_task("sleep forever")

    assert result.ok is False
    assert result.artifacts == []
    delete.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_sandbox_task_timeout_recovery_is_bounded(monkeypatch):
    # Recovery must never be able to extend the run: a hung recovery
    # attempt is cut off by SANDBOX_RECOVERY_TIMEOUT_SECONDS rather than
    # blocking teardown indefinitely.
    monkeypatch.setenv("SANDBOX_TIMEOUT", "1")
    monkeypatch.setattr(prod_sandbox_agent, "SANDBOX_RECOVERY_TIMEOUT_SECONDS", 1)

    async def slow_run(*args, **kwargs):
        await asyncio.sleep(5)

    async def hung_collect(*args, **kwargs):
        await asyncio.sleep(5)

    runner = MagicMock()
    runner.run = slow_run
    delete = AsyncMock()
    p_client, p_session, p_delete, p_out_dir = _patch_session_lifecycle(
        delete=delete, out_dir="/root/out")

    with patch.object(prod_sandbox_agent, "Runner", runner), \
         patch.object(prod_sandbox_agent, "build_sandbox_agent"), \
         patch.object(prod_sandbox_agent, "build_sandbox_run_config"), \
         patch.object(prod_sandbox_agent, "_collect_artifacts", hung_collect), \
         p_client, p_session, p_delete, p_out_dir:
        result = await asyncio.wait_for(
            prod_sandbox_agent.run_sandbox_task("sleep forever"), timeout=5,
        )

    assert result.ok is False
    assert result.artifacts == []
    delete.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_sandbox_task_deletes_session_on_run_exception(monkeypatch):
    runner = MagicMock()
    runner.run = AsyncMock(side_effect=RuntimeError("boom"))
    delete = AsyncMock()
    p_client, p_session, p_delete, p_out_dir = _patch_session_lifecycle(delete=delete)

    with patch.object(prod_sandbox_agent, "Runner", runner), \
         patch.object(prod_sandbox_agent, "build_sandbox_agent"), \
         patch.object(prod_sandbox_agent, "build_sandbox_run_config"), \
         p_client, p_session, p_delete, p_out_dir:
        with pytest.raises(RuntimeError):
            await prod_sandbox_agent.run_sandbox_task("t")

    delete.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_sandbox_task_returns_skipped_artifacts_on_success(monkeypatch):
    result = MagicMock()
    result.final_output = "done"
    runner = MagicMock()
    runner.run = AsyncMock(return_value=result)
    artifacts = [sandbox_agent.SandboxArtifact(name="a.png", data=b"data")]
    p_client, p_session, p_delete, p_out_dir = _patch_session_lifecycle()

    with patch.object(prod_sandbox_agent, "Runner", runner), \
         patch.object(prod_sandbox_agent, "build_sandbox_agent"), \
         patch.object(prod_sandbox_agent, "build_sandbox_run_config"), \
         patch.object(prod_sandbox_agent, "_collect_artifacts",
                      AsyncMock(return_value=(artifacts, ["huge.bin (skipped: too large)"]))), \
         p_client, p_session, p_delete, p_out_dir:
        output = await prod_sandbox_agent.run_sandbox_task("t")

    assert output.ok is True
    assert output.artifacts == artifacts
    assert output.skipped_artifacts == ["huge.bin (skipped: too large)"]


@pytest.mark.asyncio
async def test_run_sandbox_task_max_turns_recovers_artifacts_instead_of_raising(monkeypatch):
    # A run that exhausts its turn budget never returns a final report, so
    # (like a timeout) it must come back as SandboxResult(ok=False,
    # error="max_turns") with any recovered artifacts rather than the
    # MaxTurnsExceeded exception propagating to the caller.
    runner = MagicMock()
    runner.run = AsyncMock(side_effect=prod_sandbox_agent.MaxTurnsExceeded("out of turns"))
    delete = AsyncMock()
    artifacts = [sandbox_agent.SandboxArtifact(name="partial.txt", data=b"partial")]
    p_client, p_session, p_delete, p_out_dir = _patch_session_lifecycle(
        delete=delete, out_dir="/root/out")

    with patch.object(prod_sandbox_agent, "Runner", runner), \
         patch.object(prod_sandbox_agent, "build_sandbox_agent"), \
         patch.object(prod_sandbox_agent, "build_sandbox_run_config"), \
         patch.object(prod_sandbox_agent, "_collect_artifacts",
                      AsyncMock(return_value=(artifacts, []))), \
         p_client, p_session, p_delete, p_out_dir:
        result = await prod_sandbox_agent.run_sandbox_task("t")

    assert result.ok is False
    assert result.error == "max_turns"
    assert result.text == ""
    assert result.artifacts == artifacts
    delete.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_sandbox_task_model_behavior_error_recovers_artifacts_instead_of_raising(monkeypatch):
    runner = MagicMock()
    runner.run = AsyncMock(
        side_effect=prod_sandbox_agent.ModelBehaviorError("bad tool call"))
    delete = AsyncMock()
    p_client, p_session, p_delete, p_out_dir = _patch_session_lifecycle(
        delete=delete, out_dir="/root/out")

    with patch.object(prod_sandbox_agent, "Runner", runner), \
         patch.object(prod_sandbox_agent, "build_sandbox_agent"), \
         patch.object(prod_sandbox_agent, "build_sandbox_run_config"), \
         patch.object(prod_sandbox_agent, "_collect_artifacts",
                      AsyncMock(return_value=([], []))), \
         p_client, p_session, p_delete, p_out_dir:
        result = await prod_sandbox_agent.run_sandbox_task("t")

    assert result.ok is False
    assert result.error == "model_error"
    delete.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_sandbox_task_model_refusal_error_recovers_artifacts_instead_of_raising(monkeypatch):
    runner = MagicMock()
    runner.run = AsyncMock(side_effect=prod_sandbox_agent.ModelRefusalError("nope"))
    delete = AsyncMock()
    p_client, p_session, p_delete, p_out_dir = _patch_session_lifecycle(
        delete=delete, out_dir="/root/out")

    with patch.object(prod_sandbox_agent, "Runner", runner), \
         patch.object(prod_sandbox_agent, "build_sandbox_agent"), \
         patch.object(prod_sandbox_agent, "build_sandbox_run_config"), \
         patch.object(prod_sandbox_agent, "_collect_artifacts",
                      AsyncMock(return_value=([], []))), \
         p_client, p_session, p_delete, p_out_dir:
        result = await prod_sandbox_agent.run_sandbox_task("t")

    assert result.ok is False
    assert result.error == "model_error"
    delete.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_sandbox_task_still_raises_for_infra_errors(monkeypatch):
    # A dead container/transport or docker-level failure is not a "task
    # ran out of budget" case — it means the sandbox itself is unusable, so
    # it must still propagate to the caller (tool_functions' generic
    # "sandbox may be unavailable" handler), not come back as ok=False.
    runner = MagicMock()
    runner.run = AsyncMock(side_effect=RuntimeError("session wedged"))
    delete = AsyncMock()
    p_client, p_session, p_delete, p_out_dir = _patch_session_lifecycle(delete=delete)

    with patch.object(prod_sandbox_agent, "Runner", runner), \
         patch.object(prod_sandbox_agent, "build_sandbox_agent"), \
         patch.object(prod_sandbox_agent, "build_sandbox_run_config"), \
         p_client, p_session, p_delete, p_out_dir:
        with pytest.raises(RuntimeError):
            await prod_sandbox_agent.run_sandbox_task("t")

    delete.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_sandbox_task_forwards_progress_hooks(monkeypatch):
    hooks = object()
    agent = MagicMock(name="agent")
    run_config = MagicMock(name="run_config")
    result = MagicMock()
    result.final_output = "ok"
    runner = MagicMock()
    runner.run = AsyncMock(return_value=result)
    p_client, p_session, p_delete, p_out_dir = _patch_session_lifecycle()

    with patch.object(prod_sandbox_agent, "Runner", runner), \
         patch.object(prod_sandbox_agent, "build_sandbox_agent", return_value=agent), \
         patch.object(prod_sandbox_agent, "build_sandbox_run_config", return_value=run_config), \
         patch.object(prod_sandbox_agent, "_collect_artifacts", AsyncMock(return_value=([], []))), \
         p_client, p_session, p_delete, p_out_dir:
        await prod_sandbox_agent.run_sandbox_task("t", hooks)

    args, kwargs = runner.run.call_args
    assert kwargs["hooks"] is hooks


@pytest.mark.asyncio
async def test_delete_sandbox_session_swallows_errors():
    client = MagicMock()
    client.delete = AsyncMock(side_effect=RuntimeError("daemon gone"))
    # must not raise: cleanup failures are logged, never propagated
    await prod_sandbox_agent._delete_sandbox_session(client, MagicMock())
    client.delete.assert_awaited_once()


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

async def _agent_tool_names(monkeypatch, enabled_value):
    # TextLLMHandler.__init__ needs Redis; skip it and set what get_client()
    # reads directly.
    monkeypatch.setenv("SANDBOX_ENABLED", enabled_value)
    from classes.text_llm_handler import TextLLMHandler

    handler = TextLLMHandler.__new__(TextLLMHandler)
    handler.messages = []
    handler.guild_id = 0
    handler.original_message = MagicMock()
    handler.config = MagicMock()
    handler.user_memory = MagicMock()
    handler.system = "test"
    handler.model = "qwen3:4b"
    handler.options = {"temperature": 1.0}
    await handler.get_client()
    return [getattr(tool, "name", str(tool)) for tool in handler.agent.tools]


@pytest.mark.asyncio
async def test_run_code_sandbox_tool_registered_when_enabled(monkeypatch):
    names = await _agent_tool_names(monkeypatch, "1")
    assert "run_code_sandbox" in names
    # the other tools are unaffected
    assert "web_search" in names


@pytest.mark.asyncio
async def test_run_code_sandbox_tool_absent_when_disabled(monkeypatch):
    names = await _agent_tool_names(monkeypatch, "0")
    assert "run_code_sandbox" not in names
    assert "web_search" in names


# ---------------------- run_code_sandbox tool behaviour ----------------------

def _tool_context(message):
    from agents.tool_context import ToolContext
    return ToolContext(
        context={"original_message": message},
        tool_name="run_code_sandbox",
        tool_call_id="t1",
        tool_arguments="{}",
    )


def _embed_descriptions(embed):
    """Every description passed to send_tool_discord_embed, in order."""
    return [call.args[1] for call in embed.await_args_list]


def _embed_matching(embed, needle):
    """The single embed description containing `needle`.

    run_code_sandbox sends several embeds per call (the optional "started a
    thread" note, the static "Running in sandbox" one, and the closing
    note), so positional indexing into await_args_list is fragile — assert
    on the one you actually mean.
    """
    matches = [d for d in _embed_descriptions(embed) if needle in d]
    assert len(matches) == 1, (needle, _embed_descriptions(embed))
    return matches[0]


def _text_result(text, artifacts=None):
    """A SandboxResult with no artifacts, for tool tests that only care
    about the text path."""
    return sandbox_agent.SandboxResult(text=text, artifacts=artifacts or [])


def _timeout_result(artifacts=None):
    """A SandboxResult as returned by run_sandbox_task on timeout: no text
    (Runner.run never returned one), ok=False, possibly-recovered artifacts."""
    return sandbox_agent.SandboxResult(
        text="", artifacts=artifacts or [], ok=False, error="timeout")


def _max_turns_result(artifacts=None):
    return sandbox_agent.SandboxResult(
        text="", artifacts=artifacts or [], ok=False, error="max_turns")


def _model_error_result(artifacts=None):
    return sandbox_agent.SandboxResult(
        text="", artifacts=artifacts or [], ok=False, error="model_error")


def _fake_progress_factory():
    """Returns (factory, instances): a SandboxProgressHooks replacement
    that records every constructed hook."""
    instances = []

    def factory(channel, task, **kwargs):
        fake = MagicMock(name="sandbox-progress")
        fake.start = AsyncMock()
        fake.finalize = AsyncMock()
        instances.append(fake)
        return fake

    return factory, instances


def _assert_ran_with(task_api, task, progress):
    """Asserts task_api (the patched run_sandbox_task) was awaited with the
    given (task, progress_hooks) positionally, ignoring the thread/client/
    requesting_user_id keyword args threaded through on every call — these
    tests use plain MagicMock messages, for which ensure_sandbox_thread
    degrades to the original (non-thread) channel, see
    test_ensure_sandbox_thread_* below."""
    task_api.assert_awaited_once()
    args, kwargs = task_api.await_args
    assert args == (task, progress)


def _stub_config(get_value=False):
    """A configManager stub whose get_setting returns get_value
    (False = setting unset, like a fresh guild)."""
    cm = MagicMock(name="configManager")
    cm.return_value.get_setting = AsyncMock(return_value=get_value)
    return cm


@pytest.mark.asyncio
async def test_tool_runs_task_and_streams_progress_when_enabled(monkeypatch):
    message = MagicMock()
    task_api = AsyncMock(return_value=_text_result("done: 42"))
    factory, instances = _fake_progress_factory()

    with patch.object(prod_tool_functions, "check_web_request",
                      new=AsyncMock(return_value=(True, ""))), \
         patch("classes.config_manager.configManager", _stub_config("True")), \
         patch("classes.sandbox_progress.SandboxProgressHooks", side_effect=factory) as hook_cls, \
         patch.object(prod_tool_functions.Common, "send_tool_discord_embed", AsyncMock()) as embed, \
         patch.object(prod_sandbox_agent, "run_sandbox_task", task_api):
        result = await prod_tool_functions.run_code_sandbox.on_invoke_tool(
            _tool_context(message),
            json.dumps({"task": "print 42"}),
        )

    assert result.startswith("done: 42")
    # one live-progress message is started and passed into the sandbox run
    hook_cls.assert_called_once()
    assert len(instances) == 1
    instances[0].start.assert_awaited_once()
    _assert_ran_with(task_api, "print 42", instances[0])
    # the run finished, so the live message gets its final state
    instances[0].finalize.assert_awaited_once()
    assert "Done" in instances[0].finalize.await_args.args[0]
    # with progress on, the static embed is not sent (only the closing note)
    assert not [d for d in _embed_descriptions(embed) if "Running in sandbox" in d]


@pytest.mark.asyncio
async def test_tool_progress_disabled_by_default_sends_embed(monkeypatch):
    message = MagicMock()
    task_api = AsyncMock(return_value=_text_result("done: 42"))
    factory, instances = _fake_progress_factory()

    with patch.object(prod_tool_functions, "check_web_request",
                      new=AsyncMock(return_value=(True, ""))), \
         patch("classes.config_manager.configManager", _stub_config(False)), \
         patch("classes.sandbox_progress.SandboxProgressHooks", side_effect=factory) as hook_cls, \
         patch.object(prod_tool_functions.Common, "send_tool_discord_embed", AsyncMock()) as embed, \
         patch.object(prod_sandbox_agent, "run_sandbox_task", task_api):
        result = await prod_tool_functions.run_code_sandbox.on_invoke_tool(
            _tool_context(message),
            json.dumps({"task": "print 42"}),
        )

    assert result.startswith("done: 42")
    # progress is off by default: no hooks, just the static embed
    hook_cls.assert_not_called()
    assert instances == []
    assert "Running in sandbox: print 42" in _embed_matching(embed, "Running in sandbox")
    _assert_ran_with(task_api, "print 42", None)


@pytest.mark.asyncio
async def test_tool_progress_disabled_when_setting_false(monkeypatch):
    message = MagicMock()
    task_api = AsyncMock(return_value=_text_result("done: 42"))
    factory, instances = _fake_progress_factory()

    with patch.object(prod_tool_functions, "check_web_request",
                      new=AsyncMock(return_value=(True, ""))), \
         patch("classes.config_manager.configManager", _stub_config("False")), \
         patch("classes.sandbox_progress.SandboxProgressHooks", side_effect=factory) as hook_cls, \
         patch.object(prod_tool_functions.Common, "send_tool_discord_embed", AsyncMock()) as embed, \
         patch.object(prod_sandbox_agent, "run_sandbox_task", task_api):
        result = await prod_tool_functions.run_code_sandbox.on_invoke_tool(
            _tool_context(message),
            json.dumps({"task": "print 42"}),
        )

    assert result.startswith("done: 42")
    hook_cls.assert_not_called()
    assert "Running in sandbox" in _embed_matching(embed, "Running in sandbox")
    _assert_ran_with(task_api, "print 42", None)


@pytest.mark.asyncio
async def test_tool_progress_falls_back_to_off_on_config_error(monkeypatch):
    message = MagicMock()
    task_api = AsyncMock(return_value=_text_result("done: 42"))
    cm = MagicMock(name="configManager")
    cm.return_value.get_setting = AsyncMock(side_effect=Exception("redis down"))
    factory, instances = _fake_progress_factory()

    with patch.object(prod_tool_functions, "check_web_request",
                      new=AsyncMock(return_value=(True, ""))), \
         patch("classes.config_manager.configManager", cm), \
         patch("classes.sandbox_progress.SandboxProgressHooks", side_effect=factory) as hook_cls, \
         patch.object(prod_tool_functions.Common, "send_tool_discord_embed", AsyncMock()) as embed, \
         patch.object(prod_sandbox_agent, "run_sandbox_task", task_api):
        result = await prod_tool_functions.run_code_sandbox.on_invoke_tool(
            _tool_context(message),
            json.dumps({"task": "print 42"}),
        )

    assert result.startswith("done: 42")  # the run still happens
    hook_cls.assert_not_called()
    assert "Running in sandbox" in _embed_matching(embed, "Running in sandbox")
    _assert_ran_with(task_api, "print 42", None)


@pytest.mark.asyncio
async def test_tool_warns_the_outer_model_when_ok_but_no_files_were_produced(monkeypatch):
    # Regression: a nested run that finishes without raising (ok=True) but
    # never actually wrote anything under out/ used to hand the outer model
    # nothing but the (possibly overconfident) nested report, e.g. "I've
    # kicked off the render, it'll pop up shortly" — with no ground truth
    # telling the outer model no file was ever delivered. It must now be
    # told plainly so it doesn't relay a false "in progress" claim.
    message = MagicMock()
    task_api = AsyncMock(return_value=_text_result("On it! It'll pop up shortly."))

    with patch.object(prod_tool_functions, "check_web_request",
                      new=AsyncMock(return_value=(True, ""))), \
         patch.object(prod_tool_functions.Common, "send_tool_discord_embed", AsyncMock()), \
         patch.object(prod_sandbox_agent, "run_sandbox_task", task_api):
        result = await prod_tool_functions.run_code_sandbox.on_invoke_tool(
            _tool_context(message),
            json.dumps({"task": "make a gif"}),
        )

    assert "On it! It'll pop up shortly." in result
    assert "The sandbox attached no file" in result
    assert "Do not claim a file exists" in result


@pytest.mark.asyncio
async def test_tool_blocked_by_content_guard(monkeypatch):
    message = MagicMock()
    task_api = AsyncMock(return_value="should never run")

    with patch.object(prod_tool_functions, "check_web_request",
                      new=AsyncMock(return_value=(False, "malicious"))), \
         patch.object(prod_tool_functions.Common, "send_tool_discord_embed", AsyncMock()), \
         patch.object(prod_sandbox_agent, "run_sandbox_task", task_api):
        result = await prod_tool_functions.run_code_sandbox.on_invoke_tool(
            _tool_context(message),
            json.dumps({"task": "rm -rf /"}),
        )

    assert "blocked by the safety guard" in result
    task_api.assert_not_awaited()


@pytest.mark.asyncio
async def test_tool_reports_timeout_with_no_recovered_artifacts(monkeypatch):
    message = MagicMock()
    task_api = AsyncMock(return_value=_timeout_result())

    factory, instances = _fake_progress_factory()
    with patch.object(prod_tool_functions, "check_web_request",
                      new=AsyncMock(return_value=(True, ""))), \
         patch("classes.config_manager.configManager", _stub_config("True")), \
         patch("classes.sandbox_progress.SandboxProgressHooks", side_effect=factory), \
         patch.object(prod_tool_functions.Common, "send_tool_discord_embed", AsyncMock()), \
         patch.object(prod_sandbox_agent, "run_sandbox_task", task_api):
        result = await prod_tool_functions.run_code_sandbox.on_invoke_tool(
            _tool_context(message),
            json.dumps({"task": "sleep forever"}),
        )

    assert "timed out" in result
    assert "may retry" not in result.lower()
    assert "do not retry" in result.lower()
    # the live message is closed with a timeout note
    instances[0].finalize.assert_awaited_once()
    assert "timed out" in instances[0].finalize.await_args.args[0]


@pytest.mark.asyncio
async def test_tool_reports_timeout_with_recovered_artifact(monkeypatch):
    # Fix 1: a timeout that still produced a verified file must deliver it
    # and explicitly tell the model not to retry — this is the observed
    # failure mode where the outer model retried the same task three times
    # after a timeout that had actually already produced a good file.
    message = MagicMock()
    message.channel.send = AsyncMock()
    artifacts = [sandbox_agent.SandboxArtifact(name="out.gif", data=b"GIF89a")]
    task_api = AsyncMock(return_value=_timeout_result(artifacts))

    factory, instances = _fake_progress_factory()
    with patch.object(prod_tool_functions, "check_web_request",
                      new=AsyncMock(return_value=(True, ""))), \
         patch("classes.config_manager.configManager", _stub_config("True")), \
         patch("classes.sandbox_progress.SandboxProgressHooks", side_effect=factory), \
         patch.object(prod_tool_functions.Common, "send_tool_discord_embed", AsyncMock()), \
         patch.object(prod_sandbox_agent, "run_sandbox_task", task_api):
        result = await prod_tool_functions.run_code_sandbox.on_invoke_tool(
            _tool_context(message),
            json.dumps({"task": "make a gif"}),
        )

    message.channel.send.assert_awaited_once()
    assert "out.gif" in result
    assert "do not retry" in result.lower()
    assert "not claim" in result.lower() or "pending" in result.lower()
    instances[0].finalize.assert_awaited_once()
    assert "timed out" in instances[0].finalize.await_args.args[0]


@pytest.mark.asyncio
async def test_tool_reports_max_turns_exhausted_with_no_recovered_artifacts(monkeypatch):
    # The main gap this fixes: the outer model used to get the same
    # generic "may be unavailable" message whether the sandbox ran out of
    # turns, ran out of time, or the model misbehaved — with nothing to
    # tell it what to change before retrying.
    message = MagicMock()
    task_api = AsyncMock(return_value=_max_turns_result())

    factory, instances = _fake_progress_factory()
    with patch.object(prod_tool_functions, "check_web_request",
                      new=AsyncMock(return_value=(True, ""))), \
         patch("classes.config_manager.configManager", _stub_config("True")), \
         patch("classes.sandbox_progress.SandboxProgressHooks", side_effect=factory), \
         patch.object(prod_tool_functions.Common, "send_tool_discord_embed", AsyncMock()), \
         patch.object(prod_sandbox_agent, "run_sandbox_task", task_api):
        result = await prod_tool_functions.run_code_sandbox.on_invoke_tool(
            _tool_context(message),
            json.dumps({"task": "an overly ambitious task"}),
        )

    assert "ran out of turns" in result
    assert "do not retry" in result.lower()
    instances[0].finalize.assert_awaited_once()
    assert "turns" in instances[0].finalize.await_args.args[0]


@pytest.mark.asyncio
async def test_tool_reports_max_turns_with_recovered_artifact(monkeypatch):
    message = MagicMock()
    message.channel.send = AsyncMock()
    artifacts = [sandbox_agent.SandboxArtifact(name="partial.txt", data=b"partial")]
    task_api = AsyncMock(return_value=_max_turns_result(artifacts))

    factory, instances = _fake_progress_factory()
    with patch.object(prod_tool_functions, "check_web_request",
                      new=AsyncMock(return_value=(True, ""))), \
         patch("classes.config_manager.configManager", _stub_config("True")), \
         patch("classes.sandbox_progress.SandboxProgressHooks", side_effect=factory), \
         patch.object(prod_tool_functions.Common, "send_tool_discord_embed", AsyncMock()), \
         patch.object(prod_sandbox_agent, "run_sandbox_task", task_api):
        result = await prod_tool_functions.run_code_sandbox.on_invoke_tool(
            _tool_context(message),
            json.dumps({"task": "an overly ambitious task"}),
        )

    message.channel.send.assert_awaited_once()
    assert "partial.txt" in result
    assert "do not retry" in result.lower()


@pytest.mark.asyncio
async def test_tool_reports_model_error_with_no_recovered_artifacts(monkeypatch):
    message = MagicMock()
    task_api = AsyncMock(return_value=_model_error_result())

    factory, instances = _fake_progress_factory()
    with patch.object(prod_tool_functions, "check_web_request",
                      new=AsyncMock(return_value=(True, ""))), \
         patch("classes.config_manager.configManager", _stub_config("True")), \
         patch("classes.sandbox_progress.SandboxProgressHooks", side_effect=factory), \
         patch.object(prod_tool_functions.Common, "send_tool_discord_embed", AsyncMock()), \
         patch.object(prod_sandbox_agent, "run_sandbox_task", task_api):
        result = await prod_tool_functions.run_code_sandbox.on_invoke_tool(
            _tool_context(message),
            json.dumps({"task": "a confusing task"}),
        )

    assert "could not complete this task" in result
    # this one used to say "retry once with a more explicit, step-by-step
    # task description", which directly contradicted the docstring's ban on
    # the outer model writing step-by-step plans
    assert "step-by-step" not in result.lower()
    assert "do not retry" in result.lower()
    instances[0].finalize.assert_awaited_once()
    assert "misbehaved" in instances[0].finalize.await_args.args[0]


@pytest.mark.asyncio
async def test_tool_reports_skipped_artifacts_on_success(monkeypatch):
    message = MagicMock()
    message.channel.send = AsyncMock()
    artifacts = [sandbox_agent.SandboxArtifact(name="small.png", data=b"data")]
    result_obj = sandbox_agent.SandboxResult(
        text="done", artifacts=artifacts,
        skipped_artifacts=["huge.bin (skipped: 30000000 bytes over the 25000000-byte limit)"],
    )
    task_api = AsyncMock(return_value=result_obj)

    with patch.object(prod_tool_functions, "check_web_request",
                      new=AsyncMock(return_value=(True, ""))), \
         patch.object(prod_tool_functions.Common, "send_tool_discord_embed", AsyncMock()), \
         patch.object(prod_sandbox_agent, "run_sandbox_task", task_api):
        result = await prod_tool_functions.run_code_sandbox.on_invoke_tool(
            _tool_context(message),
            json.dumps({"task": "make two files"}),
        )

    assert "small.png" in result
    assert "huge.bin" in result
    assert "NOT sent" in result


@pytest.mark.asyncio
async def test_tool_reports_skipped_artifacts_on_timeout_with_no_recovered_artifact(monkeypatch):
    # skipped_artifacts must surface on the failure path too, not just on
    # success — a timeout can still have found (but not fetched, e.g. it
    # was oversized) a file under out/ during best-effort recovery.
    message = MagicMock()
    result_obj = sandbox_agent.SandboxResult(
        text="", artifacts=[], ok=False, error="timeout",
        skipped_artifacts=["huge.bin (skipped: too large)"],
    )
    task_api = AsyncMock(return_value=result_obj)

    factory, instances = _fake_progress_factory()
    with patch.object(prod_tool_functions, "check_web_request",
                      new=AsyncMock(return_value=(True, ""))), \
         patch("classes.config_manager.configManager", _stub_config("True")), \
         patch("classes.sandbox_progress.SandboxProgressHooks", side_effect=factory), \
         patch.object(prod_tool_functions.Common, "send_tool_discord_embed", AsyncMock()), \
         patch.object(prod_sandbox_agent, "run_sandbox_task", task_api):
        result = await prod_tool_functions.run_code_sandbox.on_invoke_tool(
            _tool_context(message),
            json.dumps({"task": "make a huge file slowly"}),
        )

    assert "timed out" in result
    assert "huge.bin" in result
    assert "NOT sent" in result


@pytest.mark.asyncio
async def test_tool_reports_failure(monkeypatch):
    message = MagicMock()

    factory, instances = _fake_progress_factory()
    with patch.object(prod_tool_functions, "check_web_request",
                      new=AsyncMock(return_value=(True, ""))), \
         patch("classes.config_manager.configManager", _stub_config("True")), \
         patch("classes.sandbox_progress.SandboxProgressHooks", side_effect=factory), \
         patch.object(prod_tool_functions.Common, "send_tool_discord_embed", AsyncMock()), \
         patch.object(prod_sandbox_agent, "run_sandbox_task",
                      new=AsyncMock(side_effect=Exception("no docker daemon"))):
        result = await prod_tool_functions.run_code_sandbox.on_invoke_tool(
            _tool_context(message),
            json.dumps({"task": "print 1"}),
        )

    assert "failed" in result
    # the live message is closed with a failure note
    instances[0].finalize.assert_awaited_once()
    assert "failed" in instances[0].finalize.await_args.args[0]


@pytest.mark.asyncio
async def test_tool_sends_artifacts_and_tells_llm_they_were_sent(monkeypatch):
    message = MagicMock()
    message.channel.send = AsyncMock()
    artifacts = [sandbox_agent.SandboxArtifact(name="plot.png", data=b"\x89PNG...")]
    task_api = AsyncMock(return_value=_text_result("done: 42", artifacts))

    with patch.object(prod_tool_functions, "check_web_request",
                      new=AsyncMock(return_value=(True, ""))), \
         patch.object(prod_tool_functions.Common, "send_tool_discord_embed", AsyncMock()), \
         patch.object(prod_sandbox_agent, "run_sandbox_task", task_api):
        result = await prod_tool_functions.run_code_sandbox.on_invoke_tool(
            _tool_context(message),
            json.dumps({"task": "plot something"}),
        )

    message.channel.send.assert_awaited_once()
    _, kwargs = message.channel.send.await_args
    assert kwargs["file"].filename == "plot.png"
    # The sandbox's own closing message rides on the file, so the user reads
    # what the agent that did the work actually said — not only the outer
    # model's second-hand version of it.
    assert kwargs["content"] == "done: 42"
    assert "done: 42" in result
    assert "plot.png" in result
    assert "already in the thread" in result
    assert "at most one short sentence" in result


@pytest.mark.asyncio
async def test_tool_returns_plain_text_when_an_artifact_fails_to_send(monkeypatch):
    message = MagicMock()
    message.channel.send = AsyncMock(side_effect=Exception("discord is down"))
    artifacts = [sandbox_agent.SandboxArtifact(name="plot.png", data=b"data")]
    task_api = AsyncMock(return_value=_text_result("done: 42", artifacts))

    with patch.object(prod_tool_functions, "check_web_request",
                      new=AsyncMock(return_value=(True, ""))), \
         patch.object(prod_tool_functions.Common, "send_tool_discord_embed", AsyncMock()), \
         patch.object(prod_sandbox_agent, "run_sandbox_task", task_api):
        result = await prod_tool_functions.run_code_sandbox.on_invoke_tool(
            _tool_context(message),
            json.dumps({"task": "plot something"}),
        )

    # the send failed, so the text result is returned as-is (no false claim
    # that a file was sent)
    assert result == "done: 42"


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


@pytest.mark.asyncio
async def test_run_sandbox_task_delivers_what_the_agent_attached(monkeypatch):
    # End to end through run_sandbox_task: the tool writes into the run
    # context DURING the run, and delivery reads it back afterwards.
    session = MagicMock(name="session")
    session.exec = AsyncMock(return_value=_exec_result(b"PNGDATA"))

    async def run(*args, **kwargs):
        kwargs["context"]["deliverables"].append(
            {"path": "out/final.png", "size": 7, "caption": ""})
        out = MagicMock()
        out.final_output = "done"
        return out

    runner = MagicMock()
    runner.run = run
    sweep = AsyncMock(return_value=([], []))
    p_client, p_session, p_delete, p_out_dir = _patch_session_lifecycle(
        session=session, out_dir="/root/out")

    with patch.object(prod_sandbox_agent, "Runner", runner), \
         patch.object(prod_sandbox_agent, "build_sandbox_agent"), \
         patch.object(prod_sandbox_agent, "build_sandbox_run_config"), \
         patch.object(prod_sandbox_agent, "_collect_artifacts", sweep), \
         p_client, p_session, p_delete, p_out_dir:
        result = await prod_sandbox_agent.run_sandbox_task("draw a glass of milk")

    assert [a.name for a in result.artifacts] == ["final.png"]
    sweep.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_sandbox_task_recovers_attachments_after_a_timeout(monkeypatch):
    # A run killed by the clock still delivers whatever it had already
    # attached, in preference to sweeping out/.
    monkeypatch.setenv("SANDBOX_TIMEOUT", "1")
    session = MagicMock(name="session")
    session.exec = AsyncMock(return_value=_exec_result(b"PNGDATA"))

    async def slow_run(*args, **kwargs):
        kwargs["context"]["deliverables"].append(
            {"path": "out/partial.png", "size": 7, "caption": ""})
        await asyncio.sleep(5)

    runner = MagicMock()
    runner.run = slow_run
    sweep = AsyncMock(return_value=([], []))
    p_client, p_session, p_delete, p_out_dir = _patch_session_lifecycle(
        session=session, out_dir="/root/out")

    with patch.object(prod_sandbox_agent, "Runner", runner), \
         patch.object(prod_sandbox_agent, "build_sandbox_agent"), \
         patch.object(prod_sandbox_agent, "build_sandbox_run_config"), \
         patch.object(prod_sandbox_agent, "_collect_artifacts", sweep), \
         p_client, p_session, p_delete, p_out_dir:
        result = await prod_sandbox_agent.run_sandbox_task("draw a glass of milk")

    assert result.ok is False and result.error == "timeout"
    assert [a.name for a in result.artifacts] == ["partial.png"]
    sweep.assert_not_awaited()


# ============================================================
# Threads, HITL (ask_user/send_preview_to_thread), multi-sandbox resume,
# persistent memory
# ============================================================

# ---------------------- ensure_sandbox_thread / sandbox_snapshot_id_for ----------------------

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


def test_snapshot_id_for_thread_is_its_id():
    thread = _thread_mock(thread_id=987)
    assert prod_sandbox_agent.sandbox_snapshot_id_for(thread) == "987"


def test_snapshot_id_for_non_thread_channel_is_none():
    # A plain channel must never be used as a fallback snapshot key — that
    # would wrongly link unrelated tasks posted in the same channel into one
    # workspace lineage (see ensure_sandbox_thread's fallback path).
    assert prod_sandbox_agent.sandbox_snapshot_id_for(MagicMock()) is None


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
async def test_ask_user_clamps_timeout_to_remaining_sandbox_budget(monkeypatch):
    # A question asked late in a run must not wait past the point the outer
    # sandbox_timeout() would cut the whole run off anyway.
    monkeypatch.setenv("SANDBOX_ASK_USER_TIMEOUT", "300")
    deadline = time.monotonic() + 5  # only ~5s left in the run's own budget
    ctx = _ask_user_ctx(deadline=deadline)

    await prod_sandbox_agent.ask_user.on_invoke_tool(ctx, json.dumps({"question": "q"}))

    assert ctx.context["conversation"].ask.await_args.kwargs["timeout"] <= 5.5


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
async def test_send_preview_to_thread_without_context_is_graceful():
    ctx = _preview_ctx(thread=None, session=None)
    result = await prod_sandbox_agent.send_preview_to_thread.on_invoke_tool(
        ctx, json.dumps({"path": "out/plot.png"})
    )
    assert "no thread" in result.lower()


# ---------------------- Memory capability (persistent memory) ----------------------

def test_build_sandbox_agent_includes_memory_capability_shell_only():
    from agents.sandbox.capabilities import Memory, Shell
    agent = sandbox_agent.build_sandbox_agent(None)
    assert isinstance(agent.capabilities[0], Shell)  # unchanged position (see existing test)
    assert isinstance(agent.capabilities[1], Memory)
    memory = agent.capabilities[1]
    # live_update=False keeps Memory's own requirement at Shell-only — the
    # default (True) needs Filesystem, which build_sandbox_agent avoids
    # entirely (apply_patch is unsupported over ChatCompletions).
    assert memory.read.live_update is False
    assert memory.required_capability_types() == {"shell"}


def test_build_sandbox_agent_memory_uses_the_sandbox_llm_not_hosted_openai(monkeypatch):
    # Regression: the SDK's Memory defaults point phase-one/phase-two
    # extraction at hosted OpenAI models ("gpt-5.4-mini"/"gpt-5.5"), which
    # this self-hosted bot has no chat API key for and must never silently
    # send sandbox workspace content to.
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("SANDBOX_MODEL", "deepseek/deepseek-v4-flash-0731")
    monkeypatch.setenv("SANDBOX_LLM_HOST", "https://openrouter.ai/api")
    monkeypatch.setenv("SANDBOX_LLM_API_KEY", "sk-or-test")
    agent = sandbox_agent.build_sandbox_agent(None)
    memory = agent.capabilities[1]
    assert memory.generate.phase_one_model is agent.model
    assert memory.generate.phase_two_model is agent.model
    assert memory.generate.phase_one_model.model == "deepseek/deepseek-v4-flash-0731"


def test_build_sandbox_agent_registers_hitl_tools():
    agent = sandbox_agent.build_sandbox_agent(None)
    names = {getattr(t, "name", str(t)) for t in agent.tools}
    assert "ask_user" in names
    assert "send_preview_to_thread" in names


def test_sandbox_instructions_no_longer_forbid_asking_questions():
    # Regression: the old "Do not ask questions" bullet directly
    # contradicted the new ask_user tool.
    agent = sandbox_agent.build_sandbox_agent(None)
    assert "do not ask questions" not in agent.instructions.lower()
    assert "ask_user" in agent.instructions
    assert "send_preview_to_thread" in agent.instructions


# ---------------------- sandbox_persist_timeout / _persist_sandbox_snapshot ----------------------

def test_persist_timeout_default(monkeypatch):
    monkeypatch.delenv("SANDBOX_PERSIST_TIMEOUT_SECONDS", raising=False)
    assert sandbox_agent.sandbox_persist_timeout() == 180


def test_persist_timeout_override(monkeypatch):
    monkeypatch.setenv("SANDBOX_PERSIST_TIMEOUT_SECONDS", "30")
    assert sandbox_agent.sandbox_persist_timeout() == 30


# ---------------------- sandbox_request_timeout / sandbox_max_retries ----------------------

def test_request_timeout_default(monkeypatch):
    monkeypatch.delenv("SANDBOX_REQUEST_TIMEOUT_SECONDS", raising=False)
    assert sandbox_agent.sandbox_request_timeout() == 180


def test_request_timeout_override(monkeypatch):
    monkeypatch.setenv("SANDBOX_REQUEST_TIMEOUT_SECONDS", "45")
    assert sandbox_agent.sandbox_request_timeout() == 45


def test_request_timeout_ignores_junk(monkeypatch):
    monkeypatch.setenv("SANDBOX_REQUEST_TIMEOUT_SECONDS", "nope")
    assert sandbox_agent.sandbox_request_timeout() == 180


def test_max_retries_default(monkeypatch):
    monkeypatch.delenv("SANDBOX_MAX_RETRIES", raising=False)
    assert sandbox_agent.sandbox_max_retries() == 2


def test_max_retries_override(monkeypatch):
    monkeypatch.setenv("SANDBOX_MAX_RETRIES", "5")
    assert sandbox_agent.sandbox_max_retries() == 5


def test_max_retries_zero_disables_retries(monkeypatch):
    """0 is a meaningful value here (unlike the _positive_int settings), so it
    must survive rather than falling back to the default."""
    monkeypatch.setenv("SANDBOX_MAX_RETRIES", "0")
    assert sandbox_agent.sandbox_max_retries() == 0


def test_max_retries_ignores_negative(monkeypatch):
    monkeypatch.setenv("SANDBOX_MAX_RETRIES", "-1")
    assert sandbox_agent.sandbox_max_retries() == 2


# ---------------------- sandbox_model_call_timeout ----------------------

def test_model_call_timeout_is_the_documented_worst_case(monkeypatch):
    """(1 + retries) x request timeout: the figure SANDBOX_MAX_RETRIES' docs
    already gave as the worst case, now enforced as a wall-clock bound."""
    monkeypatch.setenv("SANDBOX_REQUEST_TIMEOUT_SECONDS", "100")
    monkeypatch.setenv("SANDBOX_MAX_RETRIES", "2")
    assert sandbox_agent.sandbox_model_call_timeout() == 300.0
    monkeypatch.setenv("SANDBOX_MAX_RETRIES", "0")
    assert sandbox_agent.sandbox_model_call_timeout() == 100.0


def test_build_sandbox_agent_bounds_each_model_call(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("SANDBOX_REQUEST_TIMEOUT_SECONDS", "100")
    monkeypatch.setenv("SANDBOX_MAX_RETRIES", "1")
    agent = sandbox_agent.build_sandbox_agent(None)

    assert agent.model_settings.timeout == 200.0
    # Regression: openai>=3 talks HTTPX2 and stores a legacy httpx.Timeout as
    # one opaque object in every phase of the underlying client's timeout
    # instead of translating it. The per-phase budgets must be plain floats.
    inner = agent.model._client._client.timeout
    assert (inner.connect, inner.read) == (
        sandbox_agent.SANDBOX_CONNECT_TIMEOUT_SECONDS, 100.0)


@pytest.mark.asyncio
async def test_model_call_timeout_bounds_a_server_that_dribbles_padding():
    """The case the read timeout cannot catch (OpenRouter pads non-streaming
    responses with whitespace, resetting the read timer on every byte) is
    exactly what ModelSettings.timeout is here for. Pinned against a real
    chat-completions call so an SDK release that quietly made the timeout
    Responses-only would fail here rather than in prod."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    import httpx2
    from agents import (Agent, AsyncOpenAI, ModelSettings, ModelTimeoutError,
                        OpenAIChatCompletionsModel, RunConfig, Runner)

    class Dribble(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            try:
                while True:
                    self.wfile.write(b" ")
                    self.wfile.flush()
                    time.sleep(0.05)
            except OSError:
                pass  # client gave up: exactly what is being tested

    server = ThreadingHTTPServer(("127.0.0.1", 0), Dribble)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        model = OpenAIChatCompletionsModel(
            model="x",
            openai_client=AsyncOpenAI(
                base_url=f"http://127.0.0.1:{server.server_address[1]}/v1", api_key="x",
                # a read timeout shorter than the wall-clock one, to prove
                # the padding really does keep it from ever firing
                timeout=httpx2.Timeout(0.3, connect=1.0), max_retries=0,
            ),
        )
        agent = Agent(name="a", instructions="", model=model,
                      model_settings=ModelSettings(timeout=0.6))
        started = time.monotonic()
        with pytest.raises(ModelTimeoutError):
            await asyncio.wait_for(
                Runner.run(agent, "hi", max_turns=1,
                           run_config=RunConfig(tracing_disabled=True)),
                timeout=5,
            )
        assert time.monotonic() - started < 3
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.asyncio
async def test_persist_sandbox_snapshot_noop_without_snapshot_id():
    session = MagicMock()
    session.stop = AsyncMock()
    await prod_sandbox_agent._persist_sandbox_snapshot(session, None)
    session.stop.assert_not_awaited()


@pytest.mark.asyncio
async def test_persist_sandbox_snapshot_calls_stop_when_snapshot_id_given():
    session = MagicMock()
    session.stop = AsyncMock()
    await prod_sandbox_agent._persist_sandbox_snapshot(session, "123")
    session.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_persist_sandbox_snapshot_swallows_timeout(monkeypatch):
    # Must never extend how long a run can pin its worker: bounded by its
    # own budget, and a slow persist must never block returning the result.
    monkeypatch.setenv("SANDBOX_PERSIST_TIMEOUT_SECONDS", "1")

    async def slow_stop():
        await asyncio.sleep(5)

    session = MagicMock()
    session.stop = slow_stop
    await prod_sandbox_agent._persist_sandbox_snapshot(session, "123")  # must not raise


@pytest.mark.asyncio
async def test_persist_sandbox_snapshot_swallows_exception():
    session = MagicMock()
    session.stop = AsyncMock(side_effect=RuntimeError("redis down"))
    await prod_sandbox_agent._persist_sandbox_snapshot(session, "123")  # must not raise


# ---------------------- _create_sandbox_session snapshot wiring ----------------------

def _uncreated_session():
    """A session as client.create() would return it: NOT yet started."""
    session = MagicMock(name="session")
    session.start = AsyncMock()
    return session


@pytest.mark.asyncio
async def test_create_sandbox_session_passes_none_snapshot_when_no_thread():
    # No thread means no snapshot at all — never a fallback keyed on
    # something else (see sandbox_snapshot_id_for).
    client = MagicMock()
    client.create = AsyncMock(return_value=_uncreated_session())
    session, resumed = await prod_sandbox_agent._create_sandbox_session(client, None)
    assert client.create.await_args.kwargs["snapshot"] is None
    assert resumed is False


@pytest.mark.asyncio
async def test_create_sandbox_session_uses_deterministic_remote_snapshot_id():
    from agents.sandbox.snapshot import RemoteSnapshot
    client = MagicMock()
    client.create = AsyncMock(return_value=_uncreated_session())
    await prod_sandbox_agent._create_sandbox_session(client, "123456")  # noqa: F841
    snapshot = client.create.await_args.kwargs["snapshot"]
    assert isinstance(snapshot, RemoteSnapshot)
    assert snapshot.id == "123456"
    assert snapshot.client_dependency_key == prod_sandbox_agent.SANDBOX_SNAPSHOT_DEP_KEY


@pytest.mark.asyncio
async def test_create_sandbox_session_starts_the_session_before_returning():
    # Regression, found via a live check against a real Docker daemon: a
    # session that is only created() but never started() never materializes
    # /workspace and never hydrates from a snapshot on create() alone —
    # session.start() is what performs both, and skipping it silently made
    # resume (requirement 4) and Memory persistence (requirement 5) no-ops
    # despite the plumbing otherwise looking correct in every mocked test.
    client = MagicMock()
    session = _uncreated_session()
    client.create = AsyncMock(return_value=session)

    returned, resumed = await prod_sandbox_agent._create_sandbox_session(client, None)

    session.start.assert_awaited_once()
    assert returned is session
    assert resumed is False


@pytest.mark.asyncio
async def test_build_sandbox_client_binds_snapshot_store_dependency(monkeypatch):
    from classes.sandbox_snapshot_store import SandboxSnapshotStore

    monkeypatch.setenv("REDIS_HOST", "localhost")
    monkeypatch.setattr("redis.asyncio.Redis", MagicMock())
    with patch("docker.from_env", return_value=MagicMock()):
        client = prod_sandbox_agent.build_sandbox_client()
    resolved = await client._dependencies.require(
        prod_sandbox_agent.SANDBOX_SNAPSHOT_DEP_KEY, consumer="test"
    )
    assert isinstance(resolved, SandboxSnapshotStore)


# ---------------------- run_sandbox_task: persist-before-delete ordering ----------------------

@pytest.mark.asyncio
async def test_run_sandbox_task_persists_before_deleting_when_in_a_thread():
    thread = _thread_mock(thread_id=42)
    result = MagicMock()
    result.final_output = "done"
    runner = MagicMock()
    runner.run = AsyncMock(return_value=result)

    events = []
    session = MagicMock(name="session")
    session.exec = AsyncMock()

    async def _stop():
        events.append("stop")
    session.stop = _stop

    async def _delete(sandbox_client, sess):
        events.append("delete")

    p_client, p_session, p_delete, p_out_dir = _patch_session_lifecycle(
        session=session, delete=_delete)

    with patch.object(prod_sandbox_agent, "Runner", runner), \
         patch.object(prod_sandbox_agent, "build_sandbox_agent"), \
         patch.object(prod_sandbox_agent, "build_sandbox_run_config"), \
         patch.object(prod_sandbox_agent, "_collect_artifacts", AsyncMock(return_value=([], []))), \
         p_client, p_session, p_delete, p_out_dir:
        await prod_sandbox_agent.run_sandbox_task("t", thread=thread)

    assert events == ["stop", "delete"]


@pytest.mark.asyncio
async def test_run_sandbox_task_skips_persist_without_a_thread():
    result = MagicMock()
    result.final_output = "done"
    runner = MagicMock()
    runner.run = AsyncMock(return_value=result)
    session = MagicMock(name="session")
    session.exec = AsyncMock()
    session.stop = AsyncMock()
    p_client, p_session, p_delete, p_out_dir = _patch_session_lifecycle(session=session)

    with patch.object(prod_sandbox_agent, "Runner", runner), \
         patch.object(prod_sandbox_agent, "build_sandbox_agent"), \
         patch.object(prod_sandbox_agent, "build_sandbox_run_config"), \
         patch.object(prod_sandbox_agent, "_collect_artifacts", AsyncMock(return_value=([], []))), \
         p_client, p_session, p_delete, p_out_dir:
        await prod_sandbox_agent.run_sandbox_task("t")  # no thread kwarg

    session.stop.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_sandbox_task_persists_even_on_timeout(monkeypatch):
    # A partial run's workspace may still be worth resuming from.
    monkeypatch.setenv("SANDBOX_TIMEOUT", "1")
    thread = _thread_mock(thread_id=7)
    session = MagicMock(name="session")
    session.exec = AsyncMock(return_value=_exec_result(b""))
    session.stop = AsyncMock()

    async def slow_run(*args, **kwargs):
        await asyncio.sleep(5)

    runner = MagicMock()
    runner.run = slow_run
    p_client, p_session, p_delete, p_out_dir = _patch_session_lifecycle(
        session=session, out_dir="/root/out")

    with patch.object(prod_sandbox_agent, "Runner", runner), \
         patch.object(prod_sandbox_agent, "build_sandbox_agent"), \
         patch.object(prod_sandbox_agent, "build_sandbox_run_config"), \
         patch.object(prod_sandbox_agent, "_collect_artifacts", AsyncMock(return_value=([], []))), \
         p_client, p_session, p_delete, p_out_dir:
        result = await prod_sandbox_agent.run_sandbox_task("sleep forever", thread=thread)

    assert result.ok is False
    assert result.error == "timeout"
    session.stop.assert_awaited_once()


# ---------------------- run_code_sandbox tool: thread wiring ----------------------

@pytest.mark.asyncio
async def test_tool_forwards_thread_client_and_user_to_run_sandbox_task(monkeypatch):
    # message is a plain MagicMock (not a real discord.Thread), so
    # ensure_sandbox_thread degrades to the original channel (see
    # test_ensure_sandbox_thread_falls_back_when_creation_fails) — this test
    # only checks that whatever channel/client/user IS resolved reaches
    # run_sandbox_task, not the thread-creation path itself.
    message = MagicMock()
    message.author.id = 99
    task_api = AsyncMock(return_value=_text_result("done"))
    discord_client = MagicMock()

    ctx_dict = {
        "original_message": message,
        "discord_client": discord_client,
        "user_id": 99,
    }
    from agents.tool_context import ToolContext
    ctx = ToolContext(
        context=ctx_dict, tool_name="run_code_sandbox", tool_call_id="t1",
        tool_arguments="{}",
    )

    with patch.object(prod_tool_functions, "check_web_request",
                      new=AsyncMock(return_value=(True, ""))), \
         patch.object(prod_tool_functions.Common, "send_tool_discord_embed", AsyncMock()), \
         patch.object(prod_sandbox_agent, "run_sandbox_task", task_api):
        await prod_tool_functions.run_code_sandbox.on_invoke_tool(
            ctx, json.dumps({"task": "print 42"}),
        )

    task_api.assert_awaited_once()
    _, kwargs = task_api.await_args
    assert kwargs["client"] is discord_client
    assert kwargs["requesting_user_id"] == 99
    assert kwargs["thread"] is message.channel


@pytest.mark.asyncio
async def test_tool_records_resolved_channel_on_shared_context_for_final_reply(monkeypatch):
    # Regression: the outer agent's final reply (sent by MessageHandler after
    # this tool returns) was always posted to the original channel, even
    # when the sandbox's own output went to a thread. run_code_sandbox must
    # record the resolved channel back onto the shared run context so the
    # caller can redirect the final reply there too.
    message = MagicMock()
    message.author.id = 99
    new_thread = _thread_mock()
    task_api = AsyncMock(return_value=_text_result("done"))

    ctx_dict = {"original_message": message, "discord_client": MagicMock(), "user_id": 99}
    from agents.tool_context import ToolContext
    ctx = ToolContext(
        context=ctx_dict, tool_name="run_code_sandbox", tool_call_id="t1",
        tool_arguments="{}",
    )

    with patch.object(prod_tool_functions, "check_web_request",
                      new=AsyncMock(return_value=(True, ""))), \
         patch.object(prod_sandbox_agent, "ensure_sandbox_thread",
                      AsyncMock(return_value=(new_thread, True))), \
         patch.object(prod_tool_functions.Common, "send_tool_discord_embed", AsyncMock()), \
         patch.object(prod_sandbox_agent, "run_sandbox_task", task_api):
        await prod_tool_functions.run_code_sandbox.on_invoke_tool(
            ctx, json.dumps({"task": "print 42"}),
        )

    assert ctx_dict["sandbox_thread"] is new_thread


@pytest.mark.asyncio
async def test_tool_notifies_original_channel_when_a_new_thread_is_created():
    message = MagicMock()
    message.author.id = 1
    new_thread = _thread_mock()
    new_thread.mention = "<#555>"
    task_api = AsyncMock(return_value=_text_result("done"))

    with patch.object(prod_tool_functions, "check_web_request",
                      new=AsyncMock(return_value=(True, ""))), \
         patch.object(prod_sandbox_agent, "ensure_sandbox_thread",
                      AsyncMock(return_value=(new_thread, True))), \
         patch.object(prod_tool_functions.Common, "send_tool_discord_embed", AsyncMock()) as embed, \
         patch.object(prod_sandbox_agent, "run_sandbox_task", task_api):
        await prod_tool_functions.run_code_sandbox.on_invoke_tool(
            _tool_context(message), json.dumps({"task": "print 42"}),
        )

    # the "started a thread" note (to the ORIGINAL channel), then three to the
    # thread (these tests do not enable progress updates): the static "Running
    # in sandbox" embed, the "Thread Linked to Sandbox" notice telling people
    # they can steer the run from here, and the closing note.
    assert embed.await_count == 4
    first_call = embed.await_args_list[0]
    assert first_call.args[0] is message.channel
    assert "sandbox thread" in first_call.args[1]
    assert "<#555>" in first_call.args[1]


@pytest.mark.asyncio
async def test_tool_does_not_notify_when_reusing_an_existing_thread():
    message = MagicMock()
    message.author.id = 1
    existing_thread = message.channel  # ensure_sandbox_thread returns it unchanged
    task_api = AsyncMock(return_value=_text_result("done"))

    with patch.object(prod_tool_functions, "check_web_request",
                      new=AsyncMock(return_value=(True, ""))), \
         patch.object(prod_sandbox_agent, "ensure_sandbox_thread",
                      AsyncMock(return_value=(existing_thread, False))), \
         patch.object(prod_tool_functions.Common, "send_tool_discord_embed", AsyncMock()) as embed, \
         patch.object(prod_sandbox_agent, "run_sandbox_task", task_api):
        await prod_tool_functions.run_code_sandbox.on_invoke_tool(
            _tool_context(message), json.dumps({"task": "print 42"}),
        )

    # ensure_sandbox_thread returned a plain channel, not a discord.Thread, so
    # in_thread is False: the static "Running in sandbox" embed and the closing
    # note only. No "started a thread" note, and no "Thread Linked to Sandbox"
    # notice either — nothing would route messages from here to the run.
    assert embed.await_count == 2
    assert not [d for d in _embed_descriptions(embed) if "sandbox thread" in d]


# ---------------------- workspace origin: fresh vs resumed ----------------------

@pytest.fixture
def clean_inbox():
    """The thread inbox is module-level process state; a test that leaves a
    thread registered would change the next test's routing."""
    import classes.sandbox_thread_inbox as prod_inbox
    prod_inbox._RUNS.clear()
    yield prod_inbox
    prod_inbox._RUNS.clear()


@pytest.mark.asyncio
async def test_snapshot_exists_is_false_without_a_snapshot_id():
    # no thread means no snapshot id, and must not touch Redis at all
    with patch("classes.sandbox_snapshot_store.SandboxSnapshotStore") as store:
        assert await prod_sandbox_agent.sandbox_snapshot_exists(None) is False
    store.assert_not_called()


@pytest.mark.asyncio
async def test_snapshot_exists_reports_a_saved_workspace():
    store = MagicMock()
    store.exists = AsyncMock(return_value=True)
    with patch("classes.sandbox_snapshot_store.SandboxSnapshotStore", return_value=store):
        assert await prod_sandbox_agent.sandbox_snapshot_exists("42") is True
    store.exists.assert_awaited_once_with("42")


@pytest.mark.asyncio
async def test_snapshot_exists_reports_no_saved_workspace():
    store = MagicMock()
    store.exists = AsyncMock(return_value=False)
    with patch("classes.sandbox_snapshot_store.SandboxSnapshotStore", return_value=store):
        assert await prod_sandbox_agent.sandbox_snapshot_exists("42") is False


@pytest.mark.asyncio
async def test_snapshot_exists_answers_false_when_redis_is_unreachable():
    # this only drives a status badge - a wrong badge must never break a run
    store = MagicMock()
    store.exists = AsyncMock(side_effect=RuntimeError("redis down"))
    with patch("classes.sandbox_snapshot_store.SandboxSnapshotStore", return_value=store):
        assert await prod_sandbox_agent.sandbox_snapshot_exists("42") is False


def test_workspace_note_distinguishes_fresh_from_resumed():
    fresh = prod_sandbox_agent.sandbox_workspace_note(False)
    resumed = prod_sandbox_agent.sandbox_workspace_note(True)
    assert fresh != resumed
    assert "Fresh" in fresh
    assert "Resumed" in resumed


# ---------------------- resumed runs tell the nested model ----------------------

@pytest.mark.asyncio
async def test_run_sandbox_task_prepends_the_resumed_preamble():
    result = MagicMock()
    result.final_output = "done"
    runner = AsyncMock(return_value=result)
    patches = _patch_session_lifecycle()
    with patches[0], patches[2], patches[3], \
         patch.object(prod_sandbox_agent, "_persist_sandbox_snapshot", AsyncMock()), \
         patch.object(prod_sandbox_agent, "_create_sandbox_session",
                      AsyncMock(return_value=(MagicMock(exec=AsyncMock()), True))), \
         patch.object(prod_sandbox_agent, "_collect_artifacts",
                      AsyncMock(return_value=([], []))), \
         patch.object(prod_sandbox_agent.Runner, "run", runner):
        out = await prod_sandbox_agent.run_sandbox_task(
            "plot it", thread=_thread_mock(), resumed=True)

    sent_task = runner.await_args.args[1]
    assert sent_task.startswith(prod_sandbox_agent.RESUMED_TASK_PREAMBLE)
    assert sent_task.endswith("plot it")
    assert out.resumed is True


@pytest.mark.asyncio
async def test_run_sandbox_task_leaves_the_task_alone_when_not_resumed():
    result = MagicMock()
    result.final_output = "done"
    runner = AsyncMock(return_value=result)
    patches = _patch_session_lifecycle()
    with patches[0], patches[1], patches[2], patches[3], \
         patch.object(prod_sandbox_agent, "_collect_artifacts",
                      AsyncMock(return_value=([], []))), \
         patch.object(prod_sandbox_agent.Runner, "run", runner):
        out = await prod_sandbox_agent.run_sandbox_task("plot it")

    assert runner.await_args.args[1] == "plot it"
    assert out.resumed is False


# ---------------------- unrestorable snapshot recovery ----------------------

@pytest.mark.asyncio
async def test_create_session_recovers_by_dropping_an_unrestorable_snapshot():
    # A corrupt snapshot must not brick the thread: the bad key is deleted
    # and the run retried against the SAME id, so this run starts empty but
    # still persists a good snapshot back for the next one.
    from agents.sandbox.snapshot import RemoteSnapshot
    bad, good = _uncreated_session(), _uncreated_session()
    bad.start = AsyncMock(side_effect=RuntimeError("tar: unexpected EOF"))
    client = MagicMock()
    client.create = AsyncMock(side_effect=[bad, good])
    delete = AsyncMock()

    with patch.object(prod_sandbox_agent, "_delete_sandbox_session", AsyncMock()) as teardown, \
         patch.object(prod_sandbox_agent, "_delete_sandbox_snapshot", delete):
        session, resumed = await prod_sandbox_agent._create_sandbox_session(
            client, "123456", True)

    assert session is good
    assert resumed is False
    delete.assert_awaited_once_with("123456")
    teardown.assert_awaited_once()
    # the retry keeps the same snapshot id, so this run's work is saved back
    # to it on teardown (snapshot=None would make that a silent no-op)
    retry_snapshot = client.create.await_args_list[1].kwargs["snapshot"]
    assert isinstance(retry_snapshot, RemoteSnapshot)
    assert retry_snapshot.id == "123456"


@pytest.mark.asyncio
async def test_create_session_does_not_recover_when_no_restore_was_expected():
    # Nothing was being restored, so a start() failure is a real infra
    # problem and must surface rather than being retried away.
    session = _uncreated_session()
    session.start = AsyncMock(side_effect=RuntimeError("docker is down"))
    client = MagicMock()
    client.create = AsyncMock(return_value=session)
    delete = AsyncMock()

    with patch.object(prod_sandbox_agent, "_delete_sandbox_snapshot", delete), \
         pytest.raises(RuntimeError, match="docker is down"):
        await prod_sandbox_agent._create_sandbox_session(client, "123456", False)
    delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_create_session_propagates_a_failure_on_the_retry():
    bad, worse = _uncreated_session(), _uncreated_session()
    bad.start = AsyncMock(side_effect=RuntimeError("tar: unexpected EOF"))
    worse.start = AsyncMock(side_effect=RuntimeError("docker is down"))
    client = MagicMock()
    client.create = AsyncMock(side_effect=[bad, worse])

    with patch.object(prod_sandbox_agent, "_delete_sandbox_session", AsyncMock()), \
         patch.object(prod_sandbox_agent, "_delete_sandbox_snapshot", AsyncMock()), \
         pytest.raises(RuntimeError, match="docker is down"):
        await prod_sandbox_agent._create_sandbox_session(client, "123456", True)


@pytest.mark.asyncio
async def test_create_session_does_not_swallow_a_create_failure():
    # create() is deliberately outside the guard - a dead Docker daemon must
    # not be mistaken for a bad snapshot and cost a second container.
    client = MagicMock()
    client.create = AsyncMock(side_effect=RuntimeError("docker is down"))
    with pytest.raises(RuntimeError, match="docker is down"):
        await prod_sandbox_agent._create_sandbox_session(client, "123456", True)


@pytest.mark.asyncio
async def test_delete_sandbox_snapshot_is_a_no_op_without_an_id():
    with patch("classes.sandbox_snapshot_store.SandboxSnapshotStore") as store:
        await prod_sandbox_agent._delete_sandbox_snapshot(None)
    store.assert_not_called()


@pytest.mark.asyncio
async def test_delete_sandbox_snapshot_swallows_a_redis_failure():
    store = MagicMock()
    store.delete = AsyncMock(side_effect=RuntimeError("redis down"))
    with patch("classes.sandbox_snapshot_store.SandboxSnapshotStore", return_value=store):
        await prod_sandbox_agent._delete_sandbox_snapshot("42")  # must not raise


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
    from classes.sandbox_progress import parse_exec_result
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


def test_build_sandbox_agent_exposes_the_thread_io_tools():
    agent = sandbox_agent.build_sandbox_agent(None)
    names = {getattr(t, "name", None) for t in agent.tools}
    assert {"respond_to_updates", "say_in_thread"} <= names
    # one delivery path: the old polling tool must not come back
    assert "check_thread_messages" not in names


def test_build_sandbox_agent_attaches_the_conversation_hooks():
    conv = prod_sandbox_agent.SandboxConversation()
    agent = sandbox_agent.build_sandbox_agent(None, conv)
    assert agent.hooks is not None


# ---------------------- run_code_sandbox: origin badge & thread routing ----------------------

def _sandbox_tool_patches(result, *, resumed=False, thread=None, created=False):
    """The common patch set for run_code_sandbox tests that care about the
    Discord side: content guard open, thread resolution fixed, snapshot
    lookup fixed, run_sandbox_task stubbed."""
    thread = thread if thread is not None else _thread_mock(thread_id=555)
    return thread, (
        patch.object(prod_tool_functions, "check_web_request",
                     new=AsyncMock(return_value=(True, ""))),
        patch.object(prod_sandbox_agent, "ensure_sandbox_thread",
                     AsyncMock(return_value=(thread, created))),
        patch.object(prod_sandbox_agent, "sandbox_snapshot_exists",
                     AsyncMock(return_value=resumed)),
        patch.object(prod_sandbox_agent, "run_sandbox_task",
                     AsyncMock(return_value=result)),
    )


@pytest.mark.asyncio
async def test_tool_marks_a_fresh_sandbox_in_the_static_embed(clean_inbox):
    message = MagicMock()
    message.author.id = 1
    thread, patches = _sandbox_tool_patches(_text_result("done"), resumed=False)
    with patches[0], patches[1], patches[2], patches[3], \
         patch.object(prod_tool_functions.Common, "send_tool_discord_embed",
                      AsyncMock()) as embed:
        await prod_tool_functions.run_code_sandbox.on_invoke_tool(
            _tool_context(message), json.dumps({"task": "print 42"}))

    description = _embed_matching(embed, "Running in sandbox")
    assert description.startswith(prod_sandbox_agent.sandbox_workspace_note(False))
    assert "Running in sandbox: print 42" in description


@pytest.mark.asyncio
async def test_tool_marks_a_resumed_sandbox_in_the_static_embed(clean_inbox):
    message = MagicMock()
    message.author.id = 1
    thread, patches = _sandbox_tool_patches(_text_result("done"), resumed=True)
    with patches[0], patches[1], patches[2], patches[3], \
         patch.object(prod_tool_functions.Common, "send_tool_discord_embed",
                      AsyncMock()) as embed:
        await prod_tool_functions.run_code_sandbox.on_invoke_tool(
            _tool_context(message), json.dumps({"task": "print 42"}))

    assert _embed_matching(embed, "Running in sandbox").startswith(
        prod_sandbox_agent.sandbox_workspace_note(True))


@pytest.mark.asyncio
async def test_tool_truncates_a_very_long_task_in_the_static_embed(clean_inbox):
    from classes.sandbox_progress import DESCRIPTION_CHARS
    message = MagicMock()
    message.author.id = 1
    thread, patches = _sandbox_tool_patches(_text_result("done"))
    with patches[0], patches[1], patches[2], patches[3], \
         patch.object(prod_tool_functions.Common, "send_tool_discord_embed",
                      AsyncMock()) as embed:
        await prod_tool_functions.run_code_sandbox.on_invoke_tool(
            _tool_context(message), json.dumps({"task": "x" * (DESCRIPTION_CHARS + 500)}))

    description = _embed_matching(embed, "Running in sandbox")
    assert description.endswith("…")
    assert len(description) < DESCRIPTION_CHARS + 200


@pytest.mark.asyncio
async def test_tool_tells_the_sandbox_when_the_workspace_was_restored(clean_inbox):
    message = MagicMock()
    message.author.id = 1
    thread, patches = _sandbox_tool_patches(_text_result("done"), resumed=True)
    with patches[0], patches[1], patches[2], patches[3] as run, \
         patch.object(prod_tool_functions.Common, "send_tool_discord_embed", AsyncMock()):
        await prod_tool_functions.run_code_sandbox.on_invoke_tool(
            _tool_context(message), json.dumps({"task": "print 42"}))

    assert run.await_args.kwargs["resumed"] is True


@pytest.mark.asyncio
async def test_tool_passes_the_badge_to_the_live_progress_embed(clean_inbox):
    message = MagicMock()
    message.author.id = 1
    thread, patches = _sandbox_tool_patches(_text_result("done"), resumed=True)
    hooks = MagicMock()
    hooks.start = AsyncMock()
    hooks.finalize = AsyncMock()
    with patches[0], patches[1], patches[2], patches[3], \
         patch.object(prod_tool_functions.Common, "send_tool_discord_embed", AsyncMock()), \
         patch("classes.config_manager.configManager", _stub_config("True")), \
         patch("classes.sandbox_progress.SandboxProgressHooks",
               return_value=hooks) as hooks_cls:
        await prod_tool_functions.run_code_sandbox.on_invoke_tool(
            _tool_context(message), json.dumps({"task": "print 42"}))

    assert hooks_cls.call_args.kwargs["workspace_note"] == \
        prod_sandbox_agent.sandbox_workspace_note(True)


# ---------------------- run_code_sandbox: one run per thread ----------------------

@pytest.mark.asyncio
async def test_tool_registers_and_deregisters_the_run_in_its_thread(clean_inbox):
    message = MagicMock()
    message.author.id = 1
    thread, patches = _sandbox_tool_patches(_text_result("done"))
    seen = {}

    async def _record(*args, **kwargs):
        seen["active_during_run"] = clean_inbox.is_run_active(thread.id)
        return _text_result("done")

    with patches[0], patches[1], patches[2], \
         patch.object(prod_sandbox_agent, "run_sandbox_task", AsyncMock(side_effect=_record)), \
         patch.object(prod_tool_functions.Common, "send_tool_discord_embed", AsyncMock()):
        await prod_tool_functions.run_code_sandbox.on_invoke_tool(
            _tool_context(message), json.dumps({"task": "print 42"}))

    assert seen["active_during_run"] is True
    # must be deregistered afterwards, or every later message in the thread
    # would be routed to a run that no longer exists
    assert clean_inbox.is_run_active(thread.id) is False


@pytest.mark.asyncio
async def test_tool_deregisters_the_run_even_when_it_fails(clean_inbox):
    message = MagicMock()
    message.author.id = 1
    thread, patches = _sandbox_tool_patches(_text_result("done"))
    with patches[0], patches[1], patches[2], \
         patch.object(prod_sandbox_agent, "run_sandbox_task",
                      AsyncMock(side_effect=RuntimeError("docker is down"))), \
         patch.object(prod_tool_functions.Common, "send_tool_discord_embed", AsyncMock()):
        await prod_tool_functions.run_code_sandbox.on_invoke_tool(
            _tool_context(message), json.dumps({"task": "print 42"}))

    assert clean_inbox.is_run_active(thread.id) is False


@pytest.mark.asyncio
async def test_tool_forwards_to_the_run_already_going_in_this_thread(clean_inbox):
    # Two containers in one thread would race to persist the same snapshot
    # key on teardown, so a second run is never started.
    message = MagicMock()
    message.author.id = 1
    message.id = 99
    message.author.display_name = "ana"
    thread, patches = _sandbox_tool_patches(_text_result("done"))
    ledger = clean_inbox.claim_run(thread.id, 1)
    ledger.state = clean_inbox.RUNNING

    with patches[0], patches[1], patches[2], patches[3] as run, \
         patch.object(prod_tool_functions.Common, "send_tool_discord_embed",
                      AsyncMock()) as embed:
        out = await prod_tool_functions.run_code_sandbox.on_invoke_tool(
            _tool_context(message), json.dumps({"task": "make it blue"}))

    run.assert_not_awaited()
    embed.assert_not_awaited()
    assert "already running" in out
    [event] = ledger.events
    assert (event.author_name, event.text, event.is_requester) == ("ana", "make it blue", True)
    # the claim belongs to the run in flight, not to the forwarding call
    assert clean_inbox.get_run(thread.id) is ledger


@pytest.mark.asyncio
async def test_tool_says_so_when_the_running_sandbox_could_not_take_the_request(clean_inbox):
    message = MagicMock()
    message.author.id = 1
    message.id = 99
    message.author.display_name = "ana"
    thread, patches = _sandbox_tool_patches(_text_result("done"))
    ledger = clean_inbox.claim_run(thread.id, 1)
    ledger.finalize()

    with patches[0], patches[1], patches[2], patches[3], \
         patch.object(prod_tool_functions.Common, "send_tool_discord_embed", AsyncMock()):
        out = await prod_tool_functions.run_code_sandbox.on_invoke_tool(
            _tool_context(message), json.dumps({"task": "make it blue"}))

    assert "finishing" in out and "not be applied" in out
    assert ledger.follow_ups[0].text == "make it blue"


@pytest.mark.asyncio
async def test_tool_claims_the_thread_before_any_other_await(clean_inbox):
    # The old guard checked, awaited several things, then registered — so
    # two runs could both pass the check. The claim now lands before the
    # snapshot lookup (the first await after the thread is resolved).
    message = MagicMock()
    message.author.id = 1
    thread, patches = _sandbox_tool_patches(_text_result("done"))
    seen = {}

    async def _exists(snapshot_id):
        seen["claimed"] = clean_inbox.is_run_active(thread.id)
        return False

    with patches[0], patches[1], patches[3], \
         patch.object(prod_sandbox_agent, "sandbox_snapshot_exists", AsyncMock(side_effect=_exists)), \
         patch.object(prod_tool_functions.Common, "send_tool_discord_embed", AsyncMock()):
        await prod_tool_functions.run_code_sandbox.on_invoke_tool(
            _tool_context(message), json.dumps({"task": "print 42"}))

    assert seen["claimed"] is True


@pytest.mark.asyncio
async def test_tool_holds_the_claim_until_the_files_are_delivered(clean_inbox):
    # A message posted while files upload must still reach the sandbox's
    # ledger (as a follow-up), not fall through to the outer LLM.
    message = MagicMock()
    message.author.id = 1
    artifact = sandbox_agent.SandboxArtifact(name="plot.png", data=b"x")
    thread, patches = _sandbox_tool_patches(_text_result("here", [artifact]))
    seen = {}

    async def _send(*args, **kwargs):
        seen["active_while_sending"] = clean_inbox.is_run_active(thread.id)

    thread.send = AsyncMock(side_effect=_send)
    with patches[0], patches[1], patches[2], patches[3], \
         patch.object(prod_tool_functions.Common, "send_tool_discord_embed", AsyncMock()):
        await prod_tool_functions.run_code_sandbox.on_invoke_tool(
            _tool_context(message), json.dumps({"task": "plot"}))

    assert seen["active_while_sending"] is True
    assert clean_inbox.is_run_active(thread.id) is False


@pytest.mark.asyncio
async def test_tool_does_not_guard_a_plain_channel(clean_inbox):
    # The no-thread fallback targets a normal channel; registering it would
    # swallow every message in that channel for the length of the run.
    message = MagicMock()
    message.author.id = 1
    channel = MagicMock()  # not a discord.Thread
    channel.id = 777
    _, patches = _sandbox_tool_patches(_text_result("done"), thread=channel)

    with patches[0], patches[1], patches[2], patches[3] as run, \
         patch.object(prod_tool_functions.Common, "send_tool_discord_embed", AsyncMock()):
        await prod_tool_functions.run_code_sandbox.on_invoke_tool(
            _tool_context(message), json.dumps({"task": "print 42"}))

    run.assert_awaited_once()
    assert clean_inbox.is_run_active(777) is False


# ---------------------- run_code_sandbox: resume-mismatch correction ----------------------

@pytest.mark.asyncio
async def test_tool_posts_the_sandbox_message_even_with_no_files(clean_inbox):
    # The sandbox is the only party that knows what it did; if its message
    # only reached the outer model, the user would read a second-hand
    # version of it and nothing else.
    message = MagicMock()
    message.author.id = 1
    thread, patches = _sandbox_tool_patches(_text_result("no file this time, here is why"))
    thread.send = AsyncMock()

    with patches[0], patches[1], patches[2], patches[3], \
         patch.object(prod_tool_functions.Common, "send_tool_discord_embed", AsyncMock()):
        await prod_tool_functions.run_code_sandbox.on_invoke_tool(
            _tool_context(message), json.dumps({"task": "print 42"}))

    posted = [c.args[0] for c in thread.send.await_args_list if c.args]
    assert "no file this time, here is why" in posted


@pytest.mark.asyncio
async def test_tool_does_not_post_a_message_for_a_stopped_run(clean_inbox):
    # A stopped run has no report (Runner.run never returned one), so there
    # is nothing to post — the closing embed states what happened instead.
    message = MagicMock()
    message.author.id = 1
    thread, patches = _sandbox_tool_patches(_timeout_result())
    thread.send = AsyncMock()

    with patches[0], patches[1], patches[2], patches[3], \
         patch.object(prod_tool_functions.Common, "send_tool_discord_embed", AsyncMock()):
        await prod_tool_functions.run_code_sandbox.on_invoke_tool(
            _tool_context(message), json.dumps({"task": "print 42"}))

    thread.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_tool_uses_the_attached_caption_for_later_files(clean_inbox):
    # The report leads the first file; each further file carries whatever
    # caption the agent gave it in attach_file.
    message = MagicMock()
    message.author.id = 1
    artifacts = [
        sandbox_agent.SandboxArtifact(name="a.png", data=b"a", caption="ignored"),
        sandbox_agent.SandboxArtifact(name="b.png", data=b"b", caption="the second one"),
    ]
    thread, patches = _sandbox_tool_patches(_text_result("both charts", artifacts))
    thread.send = AsyncMock()

    with patches[0], patches[1], patches[2], patches[3], \
         patch.object(prod_tool_functions.Common, "send_tool_discord_embed", AsyncMock()):
        await prod_tool_functions.run_code_sandbox.on_invoke_tool(
            _tool_context(message), json.dumps({"task": "chart it"}))

    contents = [c.kwargs.get("content") for c in thread.send.await_args_list
                if "file" in c.kwargs]
    assert contents == ["both charts", "the second one"]


@pytest.mark.asyncio
async def test_tool_still_delivers_the_message_when_the_file_send_fails(clean_inbox):
    # The report must not go down with the attachment: losing both leaves
    # the user with only the closing embed.
    message = MagicMock()
    message.author.id = 1
    artifacts = [sandbox_agent.SandboxArtifact(name="a.png", data=b"a")]
    thread, patches = _sandbox_tool_patches(_text_result("here is the chart", artifacts))
    thread.send = AsyncMock(side_effect=[Exception("discord is down"), None])

    with patches[0], patches[1], patches[2], patches[3], \
         patch.object(prod_tool_functions.Common, "send_tool_discord_embed", AsyncMock()):
        await prod_tool_functions.run_code_sandbox.on_invoke_tool(
            _tool_context(message), json.dumps({"task": "chart it"}))

    assert thread.send.await_args.args[0] == "here is the chart"


@pytest.mark.asyncio
async def test_tool_tells_the_outer_model_what_the_user_said_mid_run(clean_inbox):
    # Regression: mid-run steering reached only the sandbox, so the outer
    # model answered a request it never saw change — a "make the milk red"
    # produced a red image it then called a mistake.
    message = MagicMock()
    message.author.id = 1
    thread, patches = _sandbox_tool_patches(_text_result("red fizzy milk"))
    thread.send = AsyncMock()

    async def run_with_steering(*args, **kwargs):
        clean_inbox.deliver(thread.id, 1, 1, "ana", "can you make the milk red?")
        clean_inbox.deliver(thread.id, 2, 1, "ana", "and also fizzy, add bubbles")
        ledger = clean_inbox.get_run(thread.id)
        ledger.mark_presented([1, 2], anchor=1)
        ledger.record_decision([1, 2], "done", "Red and fizzy now.")
        return _text_result("red fizzy milk")

    with patches[0], patches[1], patches[2], \
         patch.object(prod_sandbox_agent, "run_sandbox_task",
                      AsyncMock(side_effect=run_with_steering)), \
         patch.object(prod_tool_functions.Common, "send_tool_discord_embed", AsyncMock()):
        out = await prod_tool_functions.run_code_sandbox.on_invoke_tool(
            _tool_context(message), json.dumps({"task": "draw a glass of milk"}))

    assert "can you make the milk red?" in out
    assert "and also fizzy, add bubbles" in out
    assert "do not call the result a mistake" in out
    assert "Red and fizzy now." in out


@pytest.mark.asyncio
async def test_tool_does_not_claim_an_unanswered_message_was_applied(clean_inbox):
    # The old note said "the sandbox received these and adapted" for every
    # accepted message, even one the nested model ignored or never saw.
    message = MagicMock()
    message.author.id = 1
    thread, patches = _sandbox_tool_patches(_text_result("a red chart"))
    thread.send = AsyncMock()

    async def run_with_ignored_message(*args, **kwargs):
        clean_inbox.deliver(thread.id, 1, 1, "ana", "make it blue")
        clean_inbox.get_run(thread.id).mark_presented([1], anchor=1)
        return _text_result("a red chart")

    with patches[0], patches[1], patches[2], \
         patch.object(prod_sandbox_agent, "run_sandbox_task",
                      AsyncMock(side_effect=run_with_ignored_message)), \
         patch.object(prod_tool_functions.Common, "send_tool_discord_embed",
                      AsyncMock()) as embed:
        out = await prod_tool_functions.run_code_sandbox.on_invoke_tool(
            _tool_context(message), json.dumps({"task": "chart"}))

    assert "make it blue" in out
    assert "never answered" in out
    assert "adapted" not in out
    assert "do not call the result a mistake" not in out
    # and the thread's closing note names it for the user too
    assert "make it blue" in _embed_matching(embed, "Sandbox closed")


@pytest.mark.asyncio
async def test_tool_says_nothing_about_steering_when_nobody_interrupted(clean_inbox):
    message = MagicMock()
    message.author.id = 1
    thread, patches = _sandbox_tool_patches(_text_result("done"))
    thread.send = AsyncMock()

    with patches[0], patches[1], patches[2], patches[3], \
         patch.object(prod_tool_functions.Common, "send_tool_discord_embed", AsyncMock()):
        out = await prod_tool_functions.run_code_sandbox.on_invoke_tool(
            _tool_context(message), json.dumps({"task": "print 42"}))

    assert "People posted" not in out


@pytest.mark.asyncio
async def test_tool_reports_steering_on_a_stopped_run_too(clean_inbox):
    # A run that timed out after being steered still has to explain itself:
    # the outer model is about to tell the user what happened.
    message = MagicMock()
    message.author.id = 1
    thread, patches = _sandbox_tool_patches(_timeout_result())

    async def run_with_steering(*args, **kwargs):
        clean_inbox.deliver(thread.id, 1, 1, "ana", "make it red")
        return _timeout_result()

    with patches[0], patches[1], patches[2], \
         patch.object(prod_sandbox_agent, "run_sandbox_task",
                      AsyncMock(side_effect=run_with_steering)), \
         patch.object(prod_tool_functions.Common, "send_tool_discord_embed", AsyncMock()):
        out = await prod_tool_functions.run_code_sandbox.on_invoke_tool(
            _tool_context(message), json.dumps({"task": "draw a glass of milk"}))

    assert "make it red" in out


@pytest.mark.asyncio
async def test_tool_corrects_the_badge_when_the_workspace_could_not_be_restored(clean_inbox):
    # The badge already went out saying "Resumed"; the run reports it
    # actually started empty, so the record has to be put straight.
    message = MagicMock()
    message.author.id = 1
    result = sandbox_agent.SandboxResult(text="done", artifacts=[], resumed=False)
    thread, patches = _sandbox_tool_patches(result, resumed=True)
    thread.send = AsyncMock()

    with patches[0], patches[1], patches[2], patches[3], \
         patch.object(prod_tool_functions.Common, "send_tool_discord_embed", AsyncMock()):
        out = await prod_tool_functions.run_code_sandbox.on_invoke_tool(
            _tool_context(message), json.dumps({"task": "print 42"}))

    posted = [c.args[0] for c in thread.send.await_args_list if c.args]
    assert any("couldn't be restored" in text for text in posted)
    assert "could not be restored" in out


@pytest.mark.asyncio
async def test_tool_does_not_correct_the_badge_on_a_normal_resume(clean_inbox):
    message = MagicMock()
    message.author.id = 1
    result = sandbox_agent.SandboxResult(text="done", artifacts=[], resumed=True)
    thread, patches = _sandbox_tool_patches(result, resumed=True)
    thread.send = AsyncMock()

    with patches[0], patches[1], patches[2], patches[3], \
         patch.object(prod_tool_functions.Common, "send_tool_discord_embed", AsyncMock()):
        out = await prod_tool_functions.run_code_sandbox.on_invoke_tool(
            _tool_context(message), json.dumps({"task": "print 42"}))

    # "done" (the sandbox's own message) is expected here; a correction is not.
    posted = [c.args[0] for c in thread.send.await_args_list if c.args]
    assert not any("couldn't be restored" in text for text in posted)
    assert "could not be restored" not in out


@pytest.mark.asyncio
async def test_tool_does_not_correct_the_badge_on_a_fresh_run(clean_inbox):
    message = MagicMock()
    message.author.id = 1
    thread, patches = _sandbox_tool_patches(_text_result("done"), resumed=False)
    thread.send = AsyncMock()

    with patches[0], patches[1], patches[2], patches[3], \
         patch.object(prod_tool_functions.Common, "send_tool_discord_embed", AsyncMock()):
        await prod_tool_functions.run_code_sandbox.on_invoke_tool(
            _tool_context(message), json.dumps({"task": "print 42"}))

    posted = [c.args[0] for c in thread.send.await_args_list if c.args]
    assert not any("couldn't be restored" in text for text in posted)


# ---------------------- closing note ----------------------

@pytest.mark.parametrize("seconds,expected", [
    (604800, "7 days"),
    (172800, "2 days"),
    (172799, "1 day"),
    (86400, "1 day"),
    (86399, "23 hours"),
    (7200, "2 hours"),
    (3600, "1 hour"),
    (3599, "59 minutes"),
    (120, "2 minutes"),
    (30, "1 minute"),
])
def test_resume_window_is_rounded_down_to_one_unit(seconds, expected):
    # rounding down never over-promises how long a workspace will survive
    assert prod_sandbox_agent.format_resume_window(seconds) == expected


def test_closing_note_states_the_resume_window_and_where_it_works():
    note = prod_sandbox_agent.sandbox_closing_note(604800, True)
    assert "closed" in note
    assert "7 days" in note
    assert "in this thread" in note
    # the point of the note: a request anywhere else is guaranteed fresh
    assert "anywhere else" in note


def test_closing_note_admits_when_nothing_was_saved():
    note = prod_sandbox_agent.sandbox_closing_note(None, True)
    assert "not** be saved" in note or "not be saved" in note
    assert "7 days" not in note


def test_closing_note_claims_no_resume_outside_a_thread():
    # the no-thread fallback never snapshots, so it must not imply resume
    note = prod_sandbox_agent.sandbox_closing_note(None, False)
    assert "closed" in note
    assert "thread" not in note


@pytest.mark.asyncio
async def test_remaining_seconds_is_none_without_a_snapshot_id():
    with patch("classes.sandbox_snapshot_store.SandboxSnapshotStore") as store:
        assert await prod_sandbox_agent.sandbox_snapshot_remaining_seconds(None) is None
    store.assert_not_called()


@pytest.mark.asyncio
async def test_remaining_seconds_reports_the_live_redis_ttl():
    store = MagicMock()
    store.ttl = AsyncMock(return_value=12345)
    with patch("classes.sandbox_snapshot_store.SandboxSnapshotStore", return_value=store):
        assert await prod_sandbox_agent.sandbox_snapshot_remaining_seconds("42") == 12345
    store.ttl.assert_awaited_once_with("42")


@pytest.mark.asyncio
async def test_remaining_seconds_is_none_when_the_key_is_missing():
    # Redis answers -2 for a missing key. This is the case that matters:
    # persisting is best-effort, so the configured TTL would have promised a
    # resume for a snapshot that was never written.
    store = MagicMock()
    store.ttl = AsyncMock(return_value=-2)
    with patch("classes.sandbox_snapshot_store.SandboxSnapshotStore", return_value=store):
        assert await prod_sandbox_agent.sandbox_snapshot_remaining_seconds("42") is None


@pytest.mark.asyncio
async def test_remaining_seconds_is_none_when_redis_is_unreachable():
    store = MagicMock()
    store.ttl = AsyncMock(side_effect=RuntimeError("redis down"))
    with patch("classes.sandbox_snapshot_store.SandboxSnapshotStore", return_value=store):
        assert await prod_sandbox_agent.sandbox_snapshot_remaining_seconds("42") is None


@pytest.mark.asyncio
async def test_tool_posts_the_closing_note_after_a_successful_run(clean_inbox):
    message = MagicMock()
    message.author.id = 1
    thread, patches = _sandbox_tool_patches(_text_result("done"))
    with patches[0], patches[1], patches[2], patches[3], \
         patch.object(prod_sandbox_agent, "sandbox_snapshot_remaining_seconds",
                      AsyncMock(return_value=604800)), \
         patch.object(prod_tool_functions.Common, "send_tool_discord_embed",
                      AsyncMock()) as embed:
        await prod_tool_functions.run_code_sandbox.on_invoke_tool(
            _tool_context(message), json.dumps({"task": "print 42"}))

    closing = _embed_matching(embed, "Sandbox closed")
    assert "7 days" in closing
    # it must come after the run, not before it
    assert _embed_descriptions(embed)[-1] == closing


@pytest.mark.asyncio
async def test_tool_posts_the_closing_note_when_the_sandbox_is_unavailable(clean_inbox):
    # the "Running in sandbox" embed already went out, so the thread would
    # otherwise be left looking permanently mid-run
    message = MagicMock()
    message.author.id = 1
    thread, patches = _sandbox_tool_patches(_text_result("done"))
    with patches[0], patches[1], patches[2], \
         patch.object(prod_sandbox_agent, "run_sandbox_task",
                      AsyncMock(side_effect=RuntimeError("docker is down"))), \
         patch.object(prod_sandbox_agent, "sandbox_snapshot_remaining_seconds",
                      AsyncMock(return_value=None)), \
         patch.object(prod_tool_functions.Common, "send_tool_discord_embed",
                      AsyncMock()) as embed:
        await prod_tool_functions.run_code_sandbox.on_invoke_tool(
            _tool_context(message), json.dumps({"task": "print 42"}))

    assert _embed_matching(embed, "Sandbox closed")


@pytest.mark.asyncio
async def test_tool_posts_no_closing_note_when_forwarding_to_a_running_sandbox(clean_inbox):
    # nothing was opened, so there is nothing to close
    message = MagicMock()
    message.author.id = 1
    thread, patches = _sandbox_tool_patches(_text_result("done"))
    clean_inbox.claim_run(thread.id, 1)
    with patches[0], patches[1], patches[2], patches[3], \
         patch.object(prod_tool_functions.Common, "send_tool_discord_embed",
                      AsyncMock()) as embed:
        await prod_tool_functions.run_code_sandbox.on_invoke_tool(
            _tool_context(message), json.dumps({"task": "make it blue"}))

    embed.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_closing_note_failure_never_changes_what_the_tool_returns(clean_inbox):
    message = MagicMock()
    message.author.id = 1
    thread, patches = _sandbox_tool_patches(_text_result("done: 42"))
    with patches[0], patches[1], patches[2], patches[3], \
         patch.object(prod_sandbox_agent, "sandbox_snapshot_remaining_seconds",
                      AsyncMock(side_effect=RuntimeError("redis down"))), \
         patch.object(prod_tool_functions.Common, "send_tool_discord_embed", AsyncMock()):
        out = await prod_tool_functions.run_code_sandbox.on_invoke_tool(
            _tool_context(message), json.dumps({"task": "print 42"}))

    assert out.startswith("done: 42")


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


@pytest.mark.asyncio
async def test_tool_reuses_the_thread_this_turn_already_opened(clean_inbox):
    # The context dict is one object for the whole outer Runner.run, so the
    # thread the first call recorded is what a retry in the same turn must
    # use — without asking Discord for a second one.
    message = MagicMock()
    message.author.id = 1
    thread = _thread_mock(thread_id=888)
    ctx = _tool_context(message)
    ctx.context["sandbox_thread"] = thread

    with patch.object(prod_tool_functions, "check_web_request",
                      new=AsyncMock(return_value=(True, ""))), \
         patch.object(prod_sandbox_agent, "ensure_sandbox_thread",
                      AsyncMock()) as ensure, \
         patch.object(prod_sandbox_agent, "sandbox_snapshot_exists",
                      AsyncMock(return_value=True)) as exists, \
         patch.object(prod_sandbox_agent, "run_sandbox_task",
                      AsyncMock(return_value=_text_result("done"))), \
         patch.object(prod_tool_functions.Common, "send_tool_discord_embed", AsyncMock()) as embed:
        await prod_tool_functions.run_code_sandbox.on_invoke_tool(
            ctx, json.dumps({"task": "try again"}))

    ensure.assert_not_awaited()
    # and it is keyed on that thread, so the retry resumes its workspace
    exists.assert_awaited_once_with("888")
    # no "🧵 Started a sandbox thread" note — nothing new was opened
    assert not [d for d in _embed_descriptions(embed) if "Started a sandbox thread" in d]


@pytest.mark.asyncio
async def test_tool_ignores_a_non_thread_left_in_the_context(clean_inbox):
    # The context is seeded with sandbox_thread=None, and the no-thread
    # fallback path writes a plain channel there; neither may short-circuit
    # thread resolution.
    message = MagicMock()
    message.author.id = 1
    thread, patches = _sandbox_tool_patches(_text_result("done"))
    ctx = _tool_context(message)
    ctx.context["sandbox_thread"] = MagicMock()  # a plain channel, not a Thread

    with patches[0], patches[1] as ensure, patches[2], patches[3], \
         patch.object(prod_tool_functions.Common, "send_tool_discord_embed", AsyncMock()):
        await prod_tool_functions.run_code_sandbox.on_invoke_tool(
            ctx, json.dumps({"task": "print 42"}))

    ensure.assert_awaited_once()


# ---------------------- the closing embed says how the run ended ----------------------


def test_closing_note_leads_with_the_outcome_when_a_run_was_stopped():
    note = prod_sandbox_agent.sandbox_closing_note(604800, True, "⏱ Stopped: the task timed out.")
    # first line, so it reads as the headline rather than a footnote
    assert note.splitlines()[0] == "⏱ Stopped: the task timed out."
    assert "7 days" in note


def test_closing_note_omits_the_outcome_line_on_a_normal_finish():
    assert not prod_sandbox_agent.sandbox_closing_note(604800, True).startswith("\n")
    assert prod_sandbox_agent.sandbox_closing_note(604800, True) == \
        prod_sandbox_agent.sandbox_closing_note(604800, True, "")


@pytest.mark.asyncio
async def test_tool_puts_the_stop_reason_in_the_closing_embed(clean_inbox):
    # With live progress off this embed is the thread's ONLY end-of-run
    # signal: without the reason, a timeout looked exactly like a success
    # that happened to produce no file.
    message = MagicMock()
    message.author.id = 1
    thread, patches = _sandbox_tool_patches(_timeout_result())
    with patches[0], patches[1], patches[2], patches[3], \
         patch.object(prod_sandbox_agent, "sandbox_snapshot_remaining_seconds",
                      AsyncMock(return_value=604800)), \
         patch.object(prod_tool_functions.Common, "send_tool_discord_embed",
                      AsyncMock()) as embed:
        out = await prod_tool_functions.run_code_sandbox.on_invoke_tool(
            _tool_context(message), json.dumps({"task": "sleep forever"}))

    closing = _embed_matching(embed, "Sandbox closed")
    assert "timed out" in closing
    # and the model is pointed at the resume rather than at a retry
    assert "do NOT retry" in out
    assert "THIS thread" in out


@pytest.mark.asyncio
async def test_tool_does_not_offer_a_resume_that_was_never_saved(clean_inbox):
    # Persisting is best-effort. When it did not happen, the model must not
    # tell the user to carry on in the thread — there is nothing to carry on
    # from. This keeps the return string in step with the closing embed.
    message = MagicMock()
    message.author.id = 1
    thread, patches = _sandbox_tool_patches(_timeout_result())
    with patches[0], patches[1], patches[2], patches[3], \
         patch.object(prod_sandbox_agent, "sandbox_snapshot_remaining_seconds",
                      AsyncMock(return_value=None)), \
         patch.object(prod_tool_functions.Common, "send_tool_discord_embed", AsyncMock()):
        out = await prod_tool_functions.run_code_sandbox.on_invoke_tool(
            _tool_context(message), json.dumps({"task": "sleep forever"}))

    assert "carries on from where" not in out
    assert "do not retry the same task unchanged" in out.lower()


# ---------------------- elapsed-time marker ----------------------
#
# The budget bullet used to teach the model to measure its own elapsed time
# ("run `date +%s` as your very first command ... re-run it every few turns
# and subtract"). That asked a 27B Q4 model to carry an epoch integer across
# turns and do arithmetic on it, spent one of a budget as small as 10 turns,
# and collided with RESUMED_TASK_PREAMBLE's "run `ls -R .` first". We measure
# it and tell the model the answer instead, on the same carrier as thread
# messages and for the same reason.


def test_elapsed_note_reports_time_used_against_the_whole_budget():
    import time as _time
    ctx = {"deadline": _time.monotonic() + 1500.0, "timeout_seconds": 1800}
    note = prod_sandbox_agent._elapsed_note(ctx)
    assert note.startswith("[time used: 30")  # 300s, allowing for clock jitter
    assert note.endswith("s of 1800s]")


def test_elapsed_note_is_empty_without_a_deadline():
    # The nested tools also run on paths that never set one; a missing
    # budget must produce no marker rather than a wrong one.
    assert prod_sandbox_agent._elapsed_note({}) == ""
    assert prod_sandbox_agent._elapsed_note({"deadline": 1.0}) == ""
    assert prod_sandbox_agent._elapsed_note({"timeout_seconds": 600}) == ""


def test_elapsed_note_never_reports_more_than_the_budget():
    # Past the deadline the run is already being torn down; the marker must
    # not go on to claim a negative remaining budget.
    import time as _time
    ctx = {"deadline": _time.monotonic() - 500.0, "timeout_seconds": 600}
    assert prod_sandbox_agent._elapsed_note(ctx) == "[time used: 600s of 600s]"


@pytest.mark.asyncio
async def test_shell_results_carry_the_time_used():
    import time as _time
    wrapped = prod_sandbox_agent._with_elapsed_note(AsyncMock(return_value="exit 0"))
    ctx = MagicMock(context={"deadline": _time.monotonic() + 300.0,
                             "timeout_seconds": 600})
    out = await wrapped(ctx, "{}")
    assert out.startswith("exit 0")
    assert "of 600s]" in out


@pytest.mark.asyncio
async def test_a_time_marker_failure_never_breaks_a_shell_call():
    # Same contract as the inbox wrapper: the marker is an enhancement, and
    # the command the run depends on must survive its failure.
    wrapped = prod_sandbox_agent._with_elapsed_note(AsyncMock(return_value="exit 0"))
    exploding = MagicMock()
    type(exploding).context = property(lambda self: (_ for _ in ()).throw(RuntimeError("boom")))
    assert await wrapped(exploding, "{}") == "exit 0"



@pytest.mark.asyncio
async def test_the_run_context_carries_the_budget_the_deadline_was_built_from(
        monkeypatch):
    # _elapsed_note reports time used as a fraction of the whole budget, so
    # the total must come from the context rather than being re-read from
    # the env per command — otherwise a mid-run env change would show the
    # model a figure that does not match the deadline it is racing.
    monkeypatch.setenv("SANDBOX_TIMEOUT", "321")
    captured = {}

    async def _run(agent, task, **kwargs):
        captured.update(kwargs.get("context") or {})
        return MagicMock(final_output="done")

    patches = _patch_session_lifecycle()
    with patches[0], patches[1], patches[2], patches[3], \
         patch.object(prod_sandbox_agent.Runner, "run", _run):
        await prod_sandbox_agent.run_sandbox_task("print 42", None)

    assert captured["timeout_seconds"] == 321
    assert captured["deadline"] is not None


@pytest.mark.asyncio
async def test_a_timeout_during_a_continuation_delivers_the_finished_answer():
    # The first pass had already produced a final answer; running out of
    # time while continuing for a late message must not turn it into
    # "the task timed out".
    async def _converse(agent, task, run_config, hooks, context, conversation):
        conversation.completed_result = MagicMock(final_output="your chart")
        await asyncio.sleep(5)

    patches = _patch_session_lifecycle()
    with patches[0], patches[1], patches[2], patches[3],          patch.object(prod_sandbox_agent, "_converse", _converse),          patch.object(prod_sandbox_agent, "sandbox_timeout", return_value=0.05),          patch.object(prod_sandbox_agent, "build_sandbox_agent", return_value=MagicMock()),          patch.object(prod_sandbox_agent, "_collect_artifacts", AsyncMock(return_value=([], []))):
        result = await prod_sandbox_agent.run_sandbox_task("chart")

    assert result.ok is True
    assert result.text == "your chart"


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


@pytest.mark.asyncio
async def test_every_pass_of_a_run_shares_one_group_id():
    captured = []

    def _build(client, session, input_filter=None, group_id=None):
        captured.append(group_id)
        return MagicMock()

    patches = _patch_session_lifecycle()
    runner = MagicMock()
    runner.run = AsyncMock(return_value=MagicMock(final_output="done"))
    with patches[0], patches[1], patches[2], patches[3], \
         patch.object(prod_sandbox_agent, "Runner", runner), \
         patch.object(prod_sandbox_agent, "build_sandbox_run_config", side_effect=_build), \
         patch.object(prod_sandbox_agent, "build_sandbox_agent", return_value=MagicMock()), \
         patch.object(prod_sandbox_agent, "_collect_artifacts", AsyncMock(return_value=([], []))):
        await prod_sandbox_agent.run_sandbox_task("t")

    assert len(captured) == 1 and captured[0].startswith("sandbox-")


@pytest.mark.asyncio
async def test_a_message_during_the_closing_note_is_still_saved(clean_inbox, _no_conversation_record_redis):
    # The claim is held through the closing note, and ⏳ promised the
    # message would be kept — so the saved record must include it.
    message = MagicMock()
    message.author.id = 1
    thread, patches = _sandbox_tool_patches(_text_result("done"))
    thread.send = AsyncMock()

    async def _embed(channel, description, *args, **kwargs):
        if kwargs.get("title") == "Sandbox closed":
            clean_inbox.deliver(thread.id, 77, 1, "ana", "and a csv too")

    with patches[0], patches[1], patches[2], patches[3], \
         patch.object(prod_tool_functions.Common, "send_tool_discord_embed",
                      AsyncMock(side_effect=_embed)):
        await prod_tool_functions.run_code_sandbox.on_invoke_tool(
            _tool_context(message), json.dumps({"task": "print 42"}))

    snapshot_id, record = _no_conversation_record_redis.save.await_args.args
    assert [c["text"] for c in record["carry"]] == ["and a csv too"]


@pytest.mark.asyncio
async def test_a_message_during_the_record_save_is_still_saved(clean_inbox, _no_conversation_record_redis):
    # The save itself awaits Redis while the claim is still held, so a
    # message can land mid-save; the record must be re-saved with it.
    message = MagicMock()
    message.author.id = 1
    thread, patches = _sandbox_tool_patches(_text_result("done"))
    thread.send = AsyncMock()

    async def _save(snapshot_id, record):
        if _no_conversation_record_redis.save.await_count == 1:
            clean_inbox.deliver(thread.id, 78, 1, "ana", "zip it up")

    _no_conversation_record_redis.save.side_effect = _save
    with patches[0], patches[1], patches[2], patches[3]:
        await prod_tool_functions.run_code_sandbox.on_invoke_tool(
            _tool_context(message), json.dumps({"task": "print 42"}))

    assert _no_conversation_record_redis.save.await_count == 2
    snapshot_id, record = _no_conversation_record_redis.save.await_args.args
    assert [c["text"] for c in record["carry"]] == ["zip it up"]


def test_closing_note_lists_unhandled_messages_but_not_accepted_ones():
    # An accepted message already got a 👍 reply in the thread; a ⚠️ line
    # saying it was "not confirmed finished" read as noise in live testing.
    rows = [
        {"author": "ana", "text": "with a wizard hat", "status": "accepted", "reply": "on it"},
        {"author": "ana", "text": "and a cape", "status": "unacknowledged", "reply": ""},
    ]
    note = sandbox_agent.sandbox_unresolved_note(rows, in_thread=True)
    assert "wizard hat" not in note
    assert "Never answered: “and a cape”" in note
    assert sandbox_agent.sandbox_unresolved_note(rows[:1], in_thread=True) == ""


def test_steering_note_treats_an_accepted_message_as_handled():
    # The model replied it would apply it; the outer model must not hedge
    # about it or call the result a mistake.
    rows = [{"author": "ana", "text": "make it a silly cow", "status": "accepted",
             "reply": "on it"}]
    note = sandbox_agent.sandbox_steering_note(rows)
    assert "“make it a silly cow” → accepted by the sandbox" in note
    assert "do not call the result a mistake" in note
    assert "not confirm" not in note

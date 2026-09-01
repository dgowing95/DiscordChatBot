import asyncio
import json
import os
import sys

# sandbox_agent is imported as core.classes.* here and classes.* in
# production (the app runs with cwd=core/); both resolve as namespace packages.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core.classes import sandbox_agent
# The production-style import names: the modules the tools themselves use at
# call time (a different module object than core.classes.* — patch THESE).
import classes.sandbox_agent as prod_sandbox_agent
import classes.tool_functions as prod_tool_functions

# To run this pytest file from the command line, use:
# PYTHONPATH=$(pwd) pytest core/tests/sandbox_agent_tests.py


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
    # and our instructions make the tool set explicit so the model has no
    # reason to invent tools
    assert "ONLY tools are the shell tools" in agent.instructions
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
        assert "do not assume `/workspace` exists" in agent.instructions.lower()


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


def test_sandbox_instructions_tells_model_to_stop_after_verification():
    agent = sandbox_agent.build_sandbox_agent(None)
    assert "confirmed a result is correct, stop" in agent.instructions


def test_sandbox_instructions_tells_model_its_turn_and_time_budget(monkeypatch):
    monkeypatch.setenv("SANDBOX_MAX_TURNS", "7")
    monkeypatch.setenv("SANDBOX_TIMEOUT", "123")
    agent = sandbox_agent.build_sandbox_agent(None)
    assert "7 turns" in agent.instructions
    assert "123 seconds" in agent.instructions
    assert "save your best current partial output" in agent.instructions


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


def test_make_shell_tools_tolerant_wraps_both_tools():
    exec_tool = MagicMock()
    exec_tool.on_invoke_tool = AsyncMock(return_value="exec ok")
    write_tool = MagicMock()
    write_tool.on_invoke_tool = AsyncMock(return_value="write ok")
    toolset = MagicMock(exec_command=exec_tool, write_stdin=write_tool)

    sandbox_agent._make_shell_tools_tolerant(toolset)

    assert exec_tool.on_invoke_tool is not exec_tool  # sanity: reassigned
    assert write_tool.on_invoke_tool is not write_tool


def test_make_shell_tools_tolerant_handles_no_write_stdin():
    exec_tool = MagicMock()
    exec_tool.on_invoke_tool = AsyncMock(return_value="exec ok")
    toolset = MagicMock(exec_command=exec_tool, write_stdin=None)

    sandbox_agent._make_shell_tools_tolerant(toolset)  # must not raise
    assert toolset.write_stdin is None


def test_build_sandbox_agent_registers_tolerant_shell_tools():
    agent = sandbox_agent.build_sandbox_agent(None)
    shell = agent.capabilities[0]
    assert shell.configure_tools is sandbox_agent._make_shell_tools_tolerant


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
        session.exec = AsyncMock()
    return (
        patch.object(prod_sandbox_agent, "build_sandbox_client", return_value=client),
        patch.object(prod_sandbox_agent, "_create_sandbox_session", AsyncMock(return_value=session)),
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

    session.exec.assert_awaited_once_with("mkdir", "-p", "/root/out", shell=False)
    build_agent.assert_called_once_with("/root/out")


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
    build_agent.assert_called_once_with(None)


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
    from core.classes.text_llm_handler import TextLLMHandler

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

    assert result == "done: 42"
    # one live-progress message is started and passed into the sandbox run
    hook_cls.assert_called_once()
    assert len(instances) == 1
    instances[0].start.assert_awaited_once()
    task_api.assert_awaited_once_with("print 42", instances[0])
    # the run finished, so the live message gets its final state
    instances[0].finalize.assert_awaited_once()
    assert "Done" in instances[0].finalize.await_args.args[0]
    # with progress on, the static embed is not sent
    embed.assert_not_awaited()


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

    assert result == "done: 42"
    # progress is off by default: no hooks, just the static embed
    hook_cls.assert_not_called()
    assert instances == []
    embed.assert_awaited_once()
    assert "Running in sandbox: print 42" in embed.await_args.args[1]
    task_api.assert_awaited_once_with("print 42", None)


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

    assert result == "done: 42"
    hook_cls.assert_not_called()
    embed.assert_awaited_once()
    task_api.assert_awaited_once_with("print 42", None)


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

    assert result == "done: 42"  # the run still happens
    hook_cls.assert_not_called()
    embed.assert_awaited_once()
    task_api.assert_awaited_once_with("print 42", None)


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
    assert "may retry" in result
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
    assert "smaller" in result or "narrower" in result
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

    assert "retry once" in result.lower()
    assert "step-by-step" in result.lower()
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
    assert "done: 42" in result
    assert "plot.png" in result
    assert "already sent" in result


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

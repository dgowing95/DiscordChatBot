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
    agent = sandbox_agent.build_sandbox_agent()
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
    agent = sandbox_agent.build_sandbox_agent()
    assert agent.model.model == "some/local-model"
    assert agent.model._client.base_url == "http://llamacpp:8080/v1/"
    assert agent.model._client.api_key == "llamacpp"


def test_build_sandbox_agent_suppresses_default_base_instructions(monkeypatch):
    # Regression: with the SDK default base prompt the model is told to
    # use an apply_patch tool we do not expose (Shell capability only),
    # and the run aborts with ModelBehaviorError("Model produced
    # apply_patch call without an apply_patch tool."). The empty string
    # suppresses the SDK default entirely.
    agent = sandbox_agent.build_sandbox_agent()
    assert agent.base_instructions == ""
    # and our instructions make the tool set explicit so the model has no
    # reason to invent tools
    assert "ONLY tools are the shell tools" in sandbox_agent.SANDBOX_INSTRUCTIONS
    assert "apply_patch" not in sandbox_agent.SANDBOX_INSTRUCTIONS


# ---------------------- run_sandbox_task ----------------------

@pytest.mark.asyncio
async def test_run_sandbox_task_returns_final_output(monkeypatch):
    monkeypatch.setenv("SANDBOX_MAX_TURNS", "6")
    agent = MagicMock(name="agent")
    run_config = MagicMock(name="run_config")
    result = MagicMock()
    result.final_output = "done: 42"
    runner = MagicMock()
    runner.run = AsyncMock(return_value=result)

    with patch.object(prod_sandbox_agent, "Runner", runner), \
         patch.object(prod_sandbox_agent, "build_sandbox_agent", return_value=agent), \
         patch.object(prod_sandbox_agent, "build_sandbox_run_config", return_value=run_config):
        output = await prod_sandbox_agent.run_sandbox_task("print 42")

    assert output == "done: 42"
    runner.run.assert_awaited_once()
    args, kwargs = runner.run.call_args
    assert args[0] is agent
    assert args[1] == "print 42"
    assert kwargs["max_turns"] == 6
    assert kwargs["run_config"] is run_config


@pytest.mark.asyncio
async def test_run_sandbox_task_non_string_output_coerced(monkeypatch):
    result = MagicMock()
    result.final_output = 1234
    runner = MagicMock()
    runner.run = AsyncMock(return_value=result)

    with patch.object(prod_sandbox_agent, "Runner", runner), \
         patch.object(prod_sandbox_agent, "build_sandbox_agent"), \
         patch.object(prod_sandbox_agent, "build_sandbox_run_config"):
        assert await prod_sandbox_agent.run_sandbox_task("t") == "1234"


@pytest.mark.asyncio
async def test_run_sandbox_task_timeout_raises(monkeypatch):
    monkeypatch.setenv("SANDBOX_TIMEOUT", "1")

    async def slow_run(*args, **kwargs):
        await asyncio.sleep(5)

    runner = MagicMock()
    runner.run = slow_run

    with patch.object(prod_sandbox_agent, "Runner", runner), \
         patch.object(prod_sandbox_agent, "build_sandbox_agent"), \
         patch.object(prod_sandbox_agent, "build_sandbox_run_config"):
        with pytest.raises(asyncio.TimeoutError):
            await prod_sandbox_agent.run_sandbox_task("sleep forever")


@pytest.mark.asyncio
async def test_run_sandbox_task_forwards_progress_hooks(monkeypatch):
    hooks = object()
    agent = MagicMock(name="agent")
    run_config = MagicMock(name="run_config")
    result = MagicMock()
    result.final_output = "ok"
    runner = MagicMock()
    runner.run = AsyncMock(return_value=result)

    with patch.object(prod_sandbox_agent, "Runner", runner), \
         patch.object(prod_sandbox_agent, "build_sandbox_agent", return_value=agent), \
         patch.object(prod_sandbox_agent, "build_sandbox_run_config", return_value=run_config):
        await prod_sandbox_agent.run_sandbox_task("t", hooks)

    args, kwargs = runner.run.call_args
    assert kwargs["hooks"] is hooks


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
    task_api = AsyncMock(return_value="done: 42")
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
    task_api = AsyncMock(return_value="done: 42")
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
    task_api = AsyncMock(return_value="done: 42")
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
    task_api = AsyncMock(return_value="done: 42")
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
async def test_tool_reports_timeout(monkeypatch):
    message = MagicMock()

    factory, instances = _fake_progress_factory()
    with patch.object(prod_tool_functions, "check_web_request",
                      new=AsyncMock(return_value=(True, ""))), \
         patch("classes.config_manager.configManager", _stub_config("True")), \
         patch("classes.sandbox_progress.SandboxProgressHooks", side_effect=factory), \
         patch.object(prod_tool_functions.Common, "send_tool_discord_embed", AsyncMock()), \
         patch.object(prod_sandbox_agent, "run_sandbox_task",
                      new=AsyncMock(side_effect=asyncio.TimeoutError)):
        result = await prod_tool_functions.run_code_sandbox.on_invoke_tool(
            _tool_context(message),
            json.dumps({"task": "sleep forever"}),
        )

    assert "timed out" in result
    # the live message is closed with a timeout note
    instances[0].finalize.assert_awaited_once()
    assert "timed out" in instances[0].finalize.await_args.args[0]


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

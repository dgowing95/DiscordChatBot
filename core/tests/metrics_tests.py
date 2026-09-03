"""
Unit tests for core/classes/metrics.py (pure module: stdlib + prometheus_client only).

Run from the repo root:
    PYTHONPATH=$(pwd) pytest core/tests/metrics_tests.py

The metrics live in prometheus_client's global default REGISTRY, so values are
per-process; tests use a unique guild/user id to read exact values.
"""

import importlib
import os
import sys
import urllib.error
import urllib.request
from unittest.mock import MagicMock

# Same dual-import setup as image_generation_tests.py: the app imports
# classes.* (cwd = core/) while most tests import classes.*.

import classes.metrics as prod_metrics  # production-style import path
from classes import metrics as m


def test_no_double_registration_under_both_import_paths():
    # This module has import-time side effects (metric registration in the
    # global prometheus REGISTRY); both import paths must resolve to the same
    # module object (aliasing at the top of metrics.py).
    assert prod_metrics is m


GUILD = "987654321"
USER = "123456789"


def _counter_value(metric, **labels):
    return metric.labels(**labels)._value.get()


def test_metric_names():
    # Counter._name omits the "_total" suffix (the exposition layer appends
    # it — see the /metrics body assertions in test_metrics_http_server).
    assert m.messages_received_total._name == "discord_bot_messages_received"
    assert m.messages_processed_total._name == "discord_bot_messages_processed"
    assert m.llm_errors_total._name == "discord_bot_llm_errors"
    assert m.response_generation_seconds._name == "discord_bot_response_generation_seconds"
    assert m.tool_calls_total._name == "discord_bot_tool_calls"
    assert m.tool_duration_seconds._name == "discord_bot_tool_duration_seconds"
    assert m.tool_errors_total._name == "discord_bot_tool_errors"
    assert m.image_generation_seconds._name == "discord_bot_image_generation_seconds"
    assert m.message_queue_size._name == "discord_bot_message_queue_size"
    assert m.queue_drops_total._name == "discord_bot_message_queue_drops"


def test_message_counters_and_generation_histogram():
    before = _counter_value(m.messages_received_total, guild_id=GUILD, user_id=USER)
    m.inc_messages_received(GUILD, USER)
    assert _counter_value(m.messages_received_total, guild_id=GUILD, user_id=USER) == before + 1

    before = _counter_value(m.messages_processed_total, guild_id=GUILD, user_id=USER)
    m.inc_messages_processed(GUILD, USER)
    assert _counter_value(m.messages_processed_total, guild_id=GUILD, user_id=USER) == before + 1

    before = m.response_generation_seconds.labels(guild_id=GUILD)._sum.get()
    m.observe_response_generation(GUILD, 3.5)
    assert m.response_generation_seconds.labels(guild_id=GUILD)._sum.get() == before + 3.5

    before = _counter_value(m.llm_errors_total, guild_id=GUILD)
    m.inc_llm_error(GUILD)
    assert _counter_value(m.llm_errors_total, guild_id=GUILD) == before + 1


def test_tool_and_image_helpers():
    tool, guild = "web_search", GUILD

    before = _counter_value(m.tool_calls_total, tool=tool, guild_id=guild)
    m.inc_tool_call(tool, guild)
    assert _counter_value(m.tool_calls_total, tool=tool, guild_id=guild) == before + 1

    before = m.tool_duration_seconds.labels(tool=tool, guild_id=guild)._sum.get()
    m.observe_tool_duration(tool, guild, 0.25)
    assert m.tool_duration_seconds.labels(tool=tool, guild_id=guild)._sum.get() == before + 0.25

    before = _counter_value(m.tool_errors_total, tool=tool, guild_id=guild)
    m.inc_tool_error(tool, guild)
    assert _counter_value(m.tool_errors_total, tool=tool, guild_id=guild) == before + 1

    before = m.image_generation_seconds.labels(mode="text_to_image")._sum.get()
    m.observe_image_generation("text_to_image", 7.0)
    assert m.image_generation_seconds.labels(mode="text_to_image")._sum.get() == before + 7.0


def test_queue_gauge_and_drops():
    m.set_message_queue_size(4)
    assert m.message_queue_size._value.get() == 4
    m.set_message_queue_size(0)
    assert m.message_queue_size._value.get() == 0

    before = _counter_value(m.queue_drops_total, guild_id=GUILD)
    m.inc_queue_drop(GUILD)
    assert _counter_value(m.queue_drops_total, guild_id=GUILD) == before + 1


def test_guild_label_defaults():
    # Falsy guild ids become the "unknown" label instead of "0".
    m.inc_llm_error(0)
    assert m.llm_errors_total.labels(guild_id="unknown")._value.get() >= 1


def test_metrics_http_server():
    server = m.start_metrics_server(0)  # ephemeral port
    try:
        port = server.server_address[1]
        body = urllib.request.urlopen(
            f"http://127.0.0.1:{port}/metrics", timeout=5
        ).read().decode()
        for name in (
            "discord_bot_messages_received_total",
            "discord_bot_response_generation_seconds",
            "discord_bot_tool_calls_total",
            "discord_bot_image_generation_seconds",
            "discord_bot_message_queue_size",
        ):
            assert name in body, f"{name} missing from /metrics output"
        # Prometheus process collectors are registered by default.
        assert "python_info" in body or "process_" in body

        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/nope", timeout=5)
            raise AssertionError("expected 404 for non-/metrics path")
        except urllib.error.HTTPError as e:
            assert e.code == 404
    finally:
        server.shutdown()


def test_start_from_env_disabled(monkeypatch):
    monkeypatch.setenv("METRICS_PORT", "0")
    assert m.start_metrics_server_from_env() is None
    monkeypatch.setenv("METRICS_PORT", "")
    assert m.start_metrics_server_from_env() is None


# ---------------------- ToolMetricsHooks x in-flight registry ----------------------
#
# The slow tools (SLOW_TOOL_NAMES) register themselves in the per-channel
# in-flight registry (core/classes/message_queue.py) through the hooks, so a
# newer same-channel message's prompt gets the hint. These tests drive the
# REAL hooks (classes.text_llm_handler) against the REAL registry.


def _hook_context(channel_id, tool_name, tool_input, call_id):
    """A RunContextWrapper-like mock: the hooks read .tool_call_id /
    .tool_name / .tool_input off it and the user_info dict (carrying the
    original message) off .context."""
    message = MagicMock()
    message.channel = MagicMock()
    message.channel.id = channel_id
    context = MagicMock()
    context.context = {"original_message": message}
    context.tool_call_id = call_id
    context.tool_name = tool_name
    context.tool_input = tool_input
    return context


def _hook_tool(name):
    tool = MagicMock()
    tool.name = name
    return tool


def test_tool_hooks_register_slow_tool_in_flight():
    import asyncio
    from classes import message_queue as mq
    from classes import text_llm_handler as tlh

    hooks = tlh.ToolMetricsHooks(GUILD)
    ctx = _hook_context(300, "run_code_sandbox", {"task": "compute pi"}, "call_1")
    tool = _hook_tool("run_code_sandbox")

    asyncio.run(hooks.on_tool_start(ctx, None, tool))
    hint = mq.in_flight_hint(300)
    assert "🐳 code sandbox" in hint
    assert "compute pi" in hint
    assert "running for" in hint

    asyncio.run(hooks.on_tool_end(ctx, None, tool, "Done: 3.14159"))
    hint = mq.in_flight_hint(300)
    assert "running for" not in hint
    assert "finished" in hint  # recently-done note for follow-up prompts


def test_tool_hooks_failed_run_leaves_no_recently_done_note():
    import asyncio
    from classes import message_queue as mq
    from classes import text_llm_handler as tlh

    hooks = tlh.ToolMetricsHooks(GUILD)
    ctx = _hook_context(301, "generate_image", {"prompt": "a red fox"}, "call_2")
    tool = _hook_tool("generate_image")

    asyncio.run(hooks.on_tool_start(ctx, None, tool))
    asyncio.run(hooks.on_tool_end(
        ctx, None, tool,
        tlh.ToolMetricsHooks._SDK_FAILURE_PREFIX + " boom",))
    # the run vanished (no more "running"), and there is NO "finished" note —
    # the user never received a result to refer to
    assert mq.in_flight_hint(301) == ""


def test_tool_hooks_ignore_fast_tools():
    import asyncio
    from classes import message_queue as mq
    from classes import text_llm_handler as tlh

    hooks = tlh.ToolMetricsHooks(GUILD)
    ctx = _hook_context(302, "web_search", {"search_request": "python asyncio"}, "call_3")
    tool = _hook_tool("web_search")

    asyncio.run(hooks.on_tool_start(ctx, None, tool))
    asyncio.run(hooks.on_tool_end(ctx, None, tool, "results"))
    assert mq.in_flight_hint(302) == ""
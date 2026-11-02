"""LanedModel on the real agents-SDK Runner (classes/laned_model.py): the
slot per model call, the call timeout after the lane wait, side calls inside
tools, and a cancelled stream giving its slot back.

Run from the repo root: `pytest core/tests/laned_model_tests.py`
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from agents import Agent, ModelTimeoutError, RunConfig, Runner, function_tool
from agents.models.interface import Model, ModelTracing
from agents.testing import ModelStep, ScriptedModel, assistant_message, function_call

from classes import llm_lanes
from classes.laned_model import LanedModel
from classes.llm_roles import RoleConfig
from classes.side_llm import SideLLM

HOST = "http://llamacpp:8080"


@pytest.fixture(autouse=True)
def _lanes():
    llm_lanes.reset()
    yield
    llm_lanes.reset()


def _run_config():
    return RunConfig(tracing_disabled=True)


async def _settle():
    for _ in range(5):
        await asyncio.sleep(0)


def _call_args():
    return dict(system_instructions="", input="hi", model_settings=None, tools=[],
                output_schema=None, handoffs=[], tracing=ModelTracing.DISABLED)


@pytest.mark.asyncio
async def test_an_agent_on_a_laned_model_runs_to_its_answer():
    lanes = llm_lanes.configure_host(HOST, 1, 1)
    inner = ScriptedModel([ModelStep(output=[assistant_message("hello", item_id="m1")])])
    agent = Agent(name="a", instructions="", model=LanedModel(inner, HOST, 30, lane=llm_lanes.CHAT))
    result = await Runner.run(agent, "hi", run_config=_run_config())
    assert result.final_output == "hello"
    assert lanes.snapshot()[llm_lanes.CHAT]["in_use"] == 0


@pytest.mark.asyncio
async def test_an_unlaned_host_is_called_straight_away():
    inner = ScriptedModel([ModelStep(output=[assistant_message("hello", item_id="m1")])])
    agent = Agent(name="a", instructions="", model=LanedModel(inner, "https://remote.example", 30))
    assert (await Runner.run(agent, "hi", run_config=_run_config())).final_output == "hello"


def _side_llm():
    """A real SideLLM on HOST, its client faked: what a research brief does."""
    config = RoleConfig(role="research", model="m", host=HOST, api_key="k", reasoning_effort="medium",
                        call_timeout=5, host_key=HOST)
    side = SideLLM("test_side", lambda: config)
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="brief"))],
                               usage=None)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=AsyncMock(return_value=response))))
    side._get_client = lambda: client
    return side


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", [llm_lanes.CHAT, llm_lanes.BRAIN])
async def test_a_side_call_inside_a_tool_does_not_deadlock_on_one_slot(lane):
    """The run's slot is held per model call, not across its tools, so the
    tool's own side call gets the one slot."""
    lanes = llm_lanes.configure_host(HOST, 1, 1)
    side = _side_llm()
    seen = {}

    @function_tool
    async def look_up(query: str) -> str:
        """Look something up."""
        seen["lane"] = llm_lanes.current_lane.get()
        seen["in_use"] = lanes.snapshot()
        return await side.complete([{"role": "user", "content": query}], temperature=0, max_tokens=8)

    inner = ScriptedModel([
        ModelStep(output=[function_call("look_up", {"query": "x"}, call_id="c1")]),
        ModelStep(output=[assistant_message("done", item_id="m1")]),
    ])
    agent = Agent(name="a", instructions="", tools=[look_up],
                  model=LanedModel(inner, HOST, 30, lane=lane))
    token = llm_lanes.current_lane.set(lane)
    try:
        result = await asyncio.wait_for(Runner.run(agent, "hi", run_config=_run_config()), 5)
    finally:
        llm_lanes.current_lane.reset(token)
    assert result.final_output == "done"
    assert seen["lane"] == lane  # the tool's side call waits in its caller's lane
    assert all(v["in_use"] == 0 for v in seen["in_use"].values())  # nothing held during the tool
    assert all(v["in_use"] == 0 for v in lanes.snapshot().values())


class _SlowModel(Model):
    """get_response sleeps; stream_response yields once, then hangs."""

    def __init__(self, delay=0.0):
        self.delay = delay
        self.closed = False

    async def get_response(self, *args, **kwargs):
        await asyncio.sleep(self.delay)
        return "response"

    async def stream_response(self, *args, **kwargs):
        try:
            yield "first"
            await asyncio.sleep(3600)
            yield "never"
        finally:
            self.closed = True


@pytest.mark.asyncio
async def test_the_call_timeout_bounds_the_model_call():
    model = LanedModel(_SlowModel(delay=10), HOST, call_timeout=0.05)
    with pytest.raises(ModelTimeoutError):
        await model.get_response(**_call_args())
    llm_lanes.configure_host(HOST, 1, 1)
    with pytest.raises(ModelTimeoutError):
        await model.get_response(**_call_args())
    assert llm_lanes.lanes_for(HOST).snapshot()[llm_lanes.CHAT]["in_use"] == 0


@pytest.mark.asyncio
async def test_the_lane_wait_does_not_count_toward_the_call_timeout():
    lanes = llm_lanes.configure_host(HOST, 1, 1)
    await lanes.acquire(llm_lanes.CHAT)  # someone else's reply holds the slot
    model = LanedModel(_SlowModel(delay=0.05), HOST, call_timeout=0.2, lane=llm_lanes.BRAIN)
    call = asyncio.create_task(model.get_response(**_call_args()))
    await asyncio.sleep(0.4)  # twice the timeout spent waiting
    assert not call.done()
    lanes._release(llm_lanes.CHAT)
    assert await asyncio.wait_for(call, 1) == "response"


@pytest.mark.asyncio
async def test_the_lane_is_the_callers_when_none_is_given():
    lanes = llm_lanes.configure_host(HOST, 2, 1)
    model = LanedModel(_SlowModel(delay=0.2), HOST, call_timeout=5)
    token = llm_lanes.current_lane.set(llm_lanes.BACKGROUND)
    try:
        call = asyncio.create_task(model.get_response(**_call_args()))
        await _settle()
        assert lanes.snapshot()[llm_lanes.BACKGROUND]["in_use"] == 1
        await call
    finally:
        llm_lanes.current_lane.reset(token)


@pytest.mark.asyncio
async def test_a_closed_stream_gives_its_slot_back():
    lanes = llm_lanes.configure_host(HOST, 1, 1)
    inner = _SlowModel()
    stream = LanedModel(inner, HOST, call_timeout=30, lane=llm_lanes.CHAT).stream_response(**_call_args())
    assert await anext(stream) == "first"
    assert lanes.snapshot()[llm_lanes.CHAT]["in_use"] == 1
    await stream.aclose()  # a voice turn stopped mid-answer
    assert lanes.snapshot()[llm_lanes.CHAT]["in_use"] == 0
    assert inner.closed


@pytest.mark.asyncio
async def test_a_stream_that_stalls_times_out_and_releases():
    lanes = llm_lanes.configure_host(HOST, 1, 1)
    stream = LanedModel(_SlowModel(), HOST, call_timeout=0.1).stream_response(**_call_args())
    assert await anext(stream) == "first"
    with pytest.raises(ModelTimeoutError):
        await anext(stream)
    assert lanes.snapshot()[llm_lanes.CHAT]["in_use"] == 0


@pytest.mark.asyncio
async def test_a_cancelled_streamed_run_releases_its_slot():
    lanes = llm_lanes.configure_host(HOST, 1, 1)
    inner = _SlowModel()

    class _Streaming(_SlowModel):
        async def stream_response(self, *args, **kwargs):
            # A real model call: hangs mid-answer until cancelled.
            try:
                await asyncio.sleep(3600)
                yield None
            finally:
                inner.closed = True

    agent = Agent(name="a", instructions="",
                  model=LanedModel(_Streaming(), HOST, call_timeout=60, lane=llm_lanes.CHAT))
    result = Runner.run_streamed(agent, "hi", run_config=_run_config())

    async def consume():
        async for _ in result.stream_events():
            pass

    task = asyncio.create_task(consume())
    for _ in range(50):
        await asyncio.sleep(0.01)
        if lanes.snapshot()[llm_lanes.CHAT]["in_use"] == 1:
            break
    assert lanes.snapshot()[llm_lanes.CHAT]["in_use"] == 1
    result.cancel()
    await asyncio.wait_for(task, 5)
    for _ in range(50):
        if lanes.snapshot()[llm_lanes.CHAT]["in_use"] == 0:
            break
        await asyncio.sleep(0.01)
    assert lanes.snapshot()[llm_lanes.CHAT]["in_use"] == 0
    assert inner.closed


@pytest.mark.asyncio
async def test_the_model_time_metric_leaves_the_lane_wait_out():
    """on_llm_start fires before the lane wait; the wait has its own metric."""
    from prometheus_client import REGISTRY

    from classes.text_llm_handler import ToolMetricsHooks

    def total(name, **labels):
        return REGISTRY.get_sample_value(name, labels) or 0.0

    lanes = llm_lanes.configure_host(HOST, 1, 1)
    await lanes.acquire(llm_lanes.CHAT)  # another reply holds the one slot
    inner = ScriptedModel([ModelStep(output=[assistant_message("hello", item_id="m1")])])
    agent = Agent(name="a", instructions="", model=LanedModel(inner, HOST, 30, lane=llm_lanes.BRAIN))
    before_call = total("discord_bot_llm_call_seconds_sum", caller="main", outcome="ok")
    before_wait = total("discord_bot_llm_lane_wait_seconds_sum", lane="brain")
    run = asyncio.create_task(Runner.run(agent, "hi", hooks=ToolMetricsHooks(1), run_config=_run_config()))
    await asyncio.sleep(0.3)
    lanes._release(llm_lanes.CHAT)
    assert (await asyncio.wait_for(run, 5)).final_output == "hello"
    assert total("discord_bot_llm_lane_wait_seconds_sum", lane="brain") - before_wait >= 0.25
    assert total("discord_bot_llm_call_seconds_sum", caller="main", outcome="ok") - before_call < 0.2

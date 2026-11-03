"""Tests for the brain (core/classes/brain.py): its agent, its task, and one
brain call end to end on the REAL SDK Runner with agents.testing's
ScriptedModel, so an SDK upgrade that changes how a run stops at `report`,
how same-turn tool calls run, or how hooks and input filters are called
fails here rather than in production.

No Docker: the workspace lease and the run config are the two seams
(brain._acquire_workspace, brain._run_config), and the agent is a plain
Agent with the brain's real tools instead of the SandboxAgent (whose Shell
capability needs a live session).

Run from the repo root:
    pytest core/tests/brain_tests.py
"""
import asyncio
import contextlib
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agents import Agent, RunConfig, Runner, function_tool
from agents.testing import ModelStep, ScriptedModel, assistant_message, function_call

from classes import brain, brain_result


# --------------------------- fakes ---------------------------

class _Result:
    def __init__(self, stdout=b"", code=0):
        self.stdout, self.exit_code = stdout, code

    def ok(self):
        return self.exit_code == 0


class _Session:
    """The bits of a sandbox session brain.py and workspace_tools use."""

    def __init__(self, files=None, jobs=b""):
        self.files = dict(files or {})
        self.jobs = jobs
        self.commands = []

    async def exec(self, *args, shell=True, timeout=None):
        self.commands.append(args)
        cmd = args[0]
        if cmd == "pwd":
            return _Result(b"/workspace\n")
        if cmd == "stat":
            data = self.files.get(args[-1])
            return _Result(str(len(data)).encode()) if data is not None else _Result(code=1)
        if cmd == "cat":
            data = self.files.get(args[-1])
            return _Result(data) if data is not None else _Result(code=1)
        if cmd == "find":
            return _Result(b"")
        if isinstance(cmd, str) and cmd.startswith("ls -1dt"):
            return _Result(self.jobs)
        return _Result()

    async def write(self, path, data):
        self.files[str(path)] = data.read()


class _Lease:
    root = "/workspace"
    sandbox_client = None

    def __init__(self, session, restore_failed=False):
        self.session, self.restore_failed = session, restore_failed

    def job_dir(self, job_id):
        return f"jobs/{job_id}"

    def job_path(self, job_id):
        return f"/workspace/jobs/{job_id}"


def _channel():
    channel = MagicMock(id=77)
    channel.send = AsyncMock()
    channel.guild = None
    return channel


def _ctx(channel=None, **extra):
    ctx = {"guild_id": 1, "user_id": 5, "channel": channel or _channel(),
           "original_message": None, "automatic": False, "voice": False,
           "poll_tool_calls": 0, "redis_save_tool_calls": 0, "events_made": [],
           "data": ["prefers metric units"], "requester_name": "Ana",
           "persona_messages": [{"role": "user", "content": "Message from 'Ana': hi"}]}
    ctx.update(extra)
    return ctx


@contextlib.contextmanager
def _brain(model, session=None, extra_tools=(), restore_failed=False):
    """Runs brain calls against `model` (a ScriptedModel) with the brain's
    real report / ask_user / attach_file tools plus `extra_tools`."""
    from classes.workspace_tools import attach_file
    session = session or _Session()

    @contextlib.asynccontextmanager
    async def _acquire(guild_id):
        yield _Lease(session, restore_failed)

    def _agent(**kw):
        brain._guard_workspace_tools()
        return Agent(name="Brain", model=model, instructions=brain.BRAIN_INSTRUCTIONS,
                     tools=[*extra_tools, attach_file, brain.ask_user, brain.report],
                     tool_use_behavior={"stop_at_tool_names": ["report"]})

    def _config(lease, guard):
        return RunConfig(call_model_input_filter=brain.input_filter(guard, 6),
                         tracing_disabled=True)

    with patch.object(brain, "_acquire_workspace", _acquire), \
         patch.object(brain, "build_brain_agent", side_effect=_agent) as build, \
         patch.object(brain, "_run_config", _config), \
         patch("classes.workspace_commands.progress_enabled", AsyncMock(return_value=False)), \
         patch("classes.content_guard.check_web_request", AsyncMock(return_value=(True, ""))):
        yield SimpleNamespace(session=session, build=build)


def _report_call(call_id="r1", **args):
    args.setdefault("status", "done")
    args.setdefault("facts", ["Saved it."])
    return function_call("report", args, call_id=call_id)


# --------------------------- one call, end to end ---------------------------

@pytest.mark.asyncio
async def test_the_run_stops_at_report_and_posts_nothing_for_a_tool_only_job():
    model = ScriptedModel([ModelStep(output=[_report_call(facts=["It is 12°C in Oslo."])])])
    ctx = _ctx()
    with _brain(model):
        report = await brain.run_brain(ctx, "Look up the weather in Oslo")
    assert report.status == brain_result.DONE
    assert report.facts == ["It is 12°C in Oslo."]
    assert "12°C" in report.must_keep
    assert len(model.calls) == 1 and model.remaining_steps == 0
    # D18: no workspace action, so no status embed (and no files).
    ctx["channel"].send.assert_not_awaited()
    # The brain's keys are taken off the persona's context again.
    assert not set(brain._BRAIN_KEYS) & set(ctx)


@pytest.mark.asyncio
async def test_report_in_the_same_turn_as_other_calls_still_lets_them_run():
    ran = []

    @function_tool
    async def note(text: str) -> str:
        """Notes something."""
        ran.append(text)
        return "noted"

    model = ScriptedModel([ModelStep(output=[function_call("note", {"text": "a"}, call_id="n1"),
                                             _report_call("r1")])])
    with _brain(model, extra_tools=[note]):
        report = await brain.run_brain(_ctx(), "do it")
    assert ran == ["a"] and report.facts == ["Saved it."]
    assert len(model.calls) == 1


@pytest.mark.asyncio
async def test_a_plain_text_ending_still_reaches_the_persona():
    model = ScriptedModel([ModelStep(output=[assistant_message("Found it: https://x.example")])])
    with _brain(model):
        report = await brain.run_brain(_ctx(), "find it")
    assert report.status == brain_result.DONE
    assert report.facts == ["Found it: https://x.example"]


@pytest.mark.asyncio
async def test_the_task_carries_the_chat_the_requester_and_their_memories():
    model = ScriptedModel([ModelStep(output=[_report_call()])])
    with _brain(model, session=_Session(jobs=b"/workspace/jobs/c77-1-aa\n")):
        await brain.run_brain(_ctx(), "remember I'm vegan")
    task = json.dumps(model.first_call.input)
    assert "Message from 'Ana': hi" in task
    assert "Request: remember I'm vegan" in task
    assert "Asked by: Ana" in task and "prefers metric units" in task
    assert "already open): jobs/c77-" in task
    assert "../c77-1-aa/" in task
    assert "Current datetime:" in task


@pytest.mark.asyncio
async def test_an_attached_file_is_posted_in_the_channel_and_named_as_posted():
    session = _Session()
    model = ScriptedModel([
        ModelStep(output=[function_call("attach_file", {"path": "out/chart.png"}, call_id="a1")]),
        ModelStep(output=[_report_call(facts=["Made the chart."])]),
    ])
    ctx = _ctx()
    with _brain(model, session=session):
        # the file exists once the job folder is known: put it at every path
        session.files = _AnyPath(b"png!")
        report = await brain.run_brain(ctx, "chart it")
    assert report.files_posted == ["chart.png"]
    sent = [c for c in ctx["channel"].send.await_args_list if "file" in c.kwargs]
    assert len(sent) == 1 and sent[0].kwargs["file"].filename == "chart.png"
    text = brain_result.tool_result_for_persona(report)
    assert "Already posted in the channel: chart.png" in text


class _AnyPath(dict):
    """Every path exists, with the same bytes."""

    def __init__(self, data):
        super().__init__()
        self.data = data

    def get(self, key, default=None):
        return self.data


@pytest.mark.asyncio
async def test_running_out_of_turns_is_a_partial_report_with_its_files(monkeypatch):
    monkeypatch.setenv("BRAIN_MAX_TURNS", "1")
    session = _Session()
    session.files = _AnyPath(b"gif")
    model = ScriptedModel([
        ModelStep(output=[function_call("attach_file", {"path": "out/cow.gif"}, call_id="a1")]),
        ModelStep(output=[_report_call()]),
    ])
    with _brain(model, session=session):
        report = await brain.run_brain(_ctx(), "make a gif")
    assert report.status == brain_result.PARTIAL
    assert report.files_posted == ["cow.gif"]
    assert "ran out of steps" in report.note and "Do not ask for the same thing again" in report.note


@pytest.mark.asyncio
async def test_the_loop_guard_warns_then_stops_a_repeating_run():
    @function_tool
    async def lookup(q: str) -> str:
        """Looks something up."""
        return "nothing"

    steps = [ModelStep(output=[function_call("lookup", {"q": "x"}, call_id=f"l{i}")]) for i in range(6)]
    model = ScriptedModel(steps + [ModelStep(output=[_report_call()])])
    with _brain(model, extra_tools=[lookup]):
        report = await brain.run_brain(_ctx(), "find x")
    assert report.status == brain_result.PARTIAL
    assert "kept repeating itself" in report.note
    # the warning reached the model once, at the call after the third repeat
    assert "You are repeating `lookup`" in json.dumps(model.calls[3].input)
    assert "You are repeating" not in json.dumps(model.calls[2].input)


@pytest.mark.asyncio
async def test_a_poll_the_brain_posts_counts_for_the_persona_and_is_named_as_posted():
    from classes.tool_functions import create_poll
    model = ScriptedModel([
        ModelStep(output=[function_call("create_poll", {"question": "Pizza or tacos?",
                                                         "answers": ["Pizza", "Tacos"]}, call_id="p1")]),
        ModelStep(output=[_report_call(facts=["Posted the poll."])]),
    ])
    ctx = _ctx()
    with _brain(model, extra_tools=[create_poll]):
        report = await brain.run_brain(ctx, "make a poll")
    assert ctx["poll_tool_calls"] == 1  # the persona's own counter (D19)
    assert report.posted == ["a poll"]


@pytest.mark.asyncio
async def test_no_workspace_means_a_failed_report_and_no_run():
    from classes.workspace_manager import WorkspaceUnavailable

    @contextlib.asynccontextmanager
    async def _down(guild_id):
        raise WorkspaceUnavailable("docker is down")
        yield

    with patch.object(brain, "_acquire_workspace", _down), \
         patch.object(brain, "build_brain_agent") as build:
        report = await brain.run_brain(_ctx(), "search for x")
    assert report.status == brain_result.FAILED
    assert "unavailable" in report.note
    build.assert_not_called()


@pytest.mark.asyncio
async def test_an_unexpected_error_is_a_failed_report_not_an_exception():
    with patch.object(brain, "_run_brain", AsyncMock(side_effect=RuntimeError("boom"))):
        report = await brain.run_brain(_ctx(), "x")
    assert report.status == brain_result.FAILED


@pytest.mark.asyncio
async def test_the_content_guard_runs_once_at_the_first_workspace_action():
    session = _Session()
    session.files = _AnyPath(b"x")
    model = ScriptedModel([
        ModelStep(output=[function_call("attach_file", {"path": "out/a.txt"}, call_id="a1"),
                          function_call("attach_file", {"path": "out/b.txt"}, call_id="a2")]),
        ModelStep(output=[_report_call(status="failed", facts=["Blocked."])]),
    ])
    guard = AsyncMock(return_value=(False, "harmful"))
    with _brain(model, session=session), \
         patch("classes.content_guard.check_web_request", guard):
        report = await brain.run_brain(_ctx(), "something bad")
    guard.assert_awaited_once_with("something bad")
    outputs = json.dumps(model.calls[1].input)
    assert outputs.count("safety guard blocked") == 2
    assert report.files_posted == []


@pytest.mark.asyncio
async def test_a_search_never_pays_for_the_content_guard():
    guard = AsyncMock(return_value=(True, ""))
    model = ScriptedModel([ModelStep(output=[_report_call()])])
    with _brain(model), patch("classes.content_guard.check_web_request", guard):
        await brain.run_brain(_ctx(), "look it up")
    guard.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_lost_workspace_is_said_once_to_the_user_and_to_the_persona():
    model = ScriptedModel([ModelStep(output=[_report_call()])])
    ctx = _ctx()
    with _brain(model, restore_failed=True):
        report = await brain.run_brain(ctx, "x")
    assert "could not be restored" in report.note
    assert any("couldn't be restored" in str(c.args) for c in ctx["channel"].send.await_args_list)


@pytest.mark.asyncio
async def test_the_brain_runs_in_the_brain_lane():
    from classes import llm_lanes
    seen = []

    @function_tool
    async def where() -> str:
        """Says where."""
        seen.append(llm_lanes.current_lane.get())
        return "ok"

    model = ScriptedModel([ModelStep(output=[function_call("where", {}, call_id="w1")]),
                           ModelStep(output=[_report_call()])])
    with _brain(model, extra_tools=[where]):
        await brain.run_brain(_ctx(), "x")
    assert seen == [llm_lanes.BRAIN]
    assert llm_lanes.current_lane.get() != llm_lanes.BRAIN


# --------------------------- persona -> ask_brain -> persona ---------------------------

@pytest.mark.asyncio
async def test_the_persona_gets_the_brains_facts_through_ask_brain():
    from classes.tool_functions import ask_brain
    brain_model = ScriptedModel([ModelStep(output=[_report_call(
        facts=["The Core One is $1,199 (https://x.example)"])])])
    persona_model = ScriptedModel([
        ModelStep(output=[function_call("ask_brain", {"request": "cheapest printer?"}, call_id="b1")]),
        ModelStep(output=[assistant_message("Arr, the Core One at $1,199: https://x.example")]),
    ])
    persona = Agent(name="Assistant", model=persona_model, tools=[ask_brain])
    ctx = _ctx(brain_reports=[])
    with _brain(brain_model):
        result = await Runner.run(persona, "which printer is cheapest?", context=ctx,
                                  run_config=RunConfig(tracing_disabled=True))
    assert result.final_output.startswith("Arr")
    told = json.dumps(persona_model.calls[1].input)
    assert "Brain result: done." in told and "$1,199" in told
    assert "Must appear exactly in your reply" in told
    assert len(ctx["brain_reports"]) == 1
    assert brain_result.missing_tokens(result.final_output, ctx["brain_reports"][0]) == []


# --------------------------- the agent ---------------------------

def _clear_llm_env(monkeypatch):
    for var in ("BRAIN_MODEL", "BRAIN_LLM_HOST", "BRAIN_LLM_API_KEY",
                "LLM_MODEL", "LLM_HOST", "LLM_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def test_the_brain_agent_is_a_sandbox_agent_that_stops_at_report(monkeypatch):
    _clear_llm_env(monkeypatch)
    from agents.sandbox import SandboxAgent
    agent = brain.build_brain_agent()
    assert isinstance(agent, SandboxAgent)
    assert agent.tool_use_behavior == {"stop_at_tool_names": ["report"]}
    # The SDK's default base prompt pushes an apply_patch tool we do not have.
    assert agent.base_instructions == ""
    assert [type(c).__name__ for c in agent.capabilities] == ["Shell"]
    assert agent.model.lane == "brain"
    assert agent.tools[-1].name == "report"


def test_the_brain_uses_its_own_llm_when_configured(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("BRAIN_MODEL", "deepseek/deepseek-v4-flash-0731")
    monkeypatch.setenv("BRAIN_LLM_HOST", "https://openrouter.ai/api")
    monkeypatch.setenv("BRAIN_LLM_API_KEY", "sk-or-test")
    model = brain.build_brain_model()
    assert model.model == "deepseek/deepseek-v4-flash-0731"
    assert model._client.base_url == "https://openrouter.ai/api/v1/"
    assert model._client.api_key == "sk-or-test"


def test_the_brain_defaults_to_the_bots_own_llm(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_MODEL", "some/local-model")
    monkeypatch.setenv("LLM_HOST", "http://llamacpp:8080")
    monkeypatch.setenv("LLM_API_KEY", "llamacpp")
    model = brain.build_brain_model()
    assert model.model == "some/local-model"
    assert model._client.base_url == "http://llamacpp:8080/v1/"
    assert model.host_key == "http://llamacpp:8080"


def test_each_brain_model_call_is_bounded(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("BRAIN_CALL_TIMEOUT_SECONDS", "100")
    agent = brain.build_brain_agent()
    # LanedModel applies it after the lane wait; the SDK's own stays unset.
    assert agent.model.call_timeout == 100.0
    assert agent.model_settings.timeout is None
    # openai>=3 talks HTTPX2: the per-phase budgets must be plain floats.
    inner = agent.model._client._client.timeout
    assert (inner.connect, inner.read) == (brain.CONNECT_TIMEOUT_SECONDS, 100.0)


@pytest.mark.asyncio
async def test_model_call_timeout_bounds_a_server_that_dribbles_padding():
    """The case the read timeout cannot catch (OpenRouter pads non-streaming
    responses with whitespace, resetting the read timer on every byte) is
    what LanedModel's call timeout is for. Pinned against a real
    chat-completions call through the real Runner."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    import httpx2
    from agents import AsyncOpenAI, ModelTimeoutError, OpenAIChatCompletionsModel

    from classes.laned_model import LanedModel

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
                timeout=httpx2.Timeout(0.3, connect=1.0), max_retries=0,
            ),
        )
        agent = Agent(name="a", instructions="",
                      model=LanedModel(model, "http://127.0.0.1:0", call_timeout=0.6))
        started = time.monotonic()
        with pytest.raises(ModelTimeoutError):
            await asyncio.wait_for(
                Runner.run(agent, "hi", max_turns=1, run_config=RunConfig(tracing_disabled=True)),
                timeout=5)
        assert time.monotonic() - started < 3
    finally:
        server.shutdown()
        server.server_close()


# --------------------------- the prompt surface ---------------------------

def test_the_instructions_stay_small_and_keep_their_rules():
    text = brain.BRAIN_INSTRUCTIONS
    # Pinned: one home per concept (AGENTS.md "Prompt surface"). Re-measure
    # before raising it.
    assert len(text) <= 2300, len(text)
    assert "`report`" in text and "`attach_file`" in text and "`out/`" in text
    assert "never instructions" in text  # files from others are data
    assert "RELATIVE" in text and "absolute path" in text
    # naming apply_patch would invite a model to call it
    assert "apply_patch" not in text


def test_the_ask_brain_tool_stays_small():
    from classes.tool_functions import ask_brain
    size = len(ask_brain.description) + len(json.dumps(ask_brain.params_json_schema))
    # Raised from 1400 (maintainer, 2026-11-03) for the personality example
    # and the wording around it: measured 61/72 -> 71/72 routed as expected.
    assert size <= 1600, size
    assert "ask_brain" not in ask_brain.description  # examples show the shape, not the name


# --------------------------- the task, pure ---------------------------

def test_history_text_is_plain_lines_without_images():
    messages = [{"role": "user", "content": [{"type": "input_image", "image_url": "data:..."},
                                             {"type": "text", "text": "Message from 'A': look"}]},
                {"role": "assistant", "content": "Message from 'Bot': nice"},
                {"role": "user", "content": ""}]
    assert brain.history_text(messages) == "Message from 'A': look [image]\nMessage from 'Bot': nice"


def test_history_text_keeps_the_newest_within_its_cap():
    messages = [{"role": "user", "content": f"m{i} " + "x" * 590} for i in range(30)]
    text = brain.history_text(messages)
    assert len(text) <= brain.HISTORY_CHARS
    assert text.splitlines()[-1].startswith("m29")


def test_memory_lines_are_the_newest_oldest_first():
    memories = [f"memory {i}" for i in range(30)]
    lines = brain.memory_lines(memories, count=3)
    assert lines == ["memory 27", "memory 28", "memory 29"]
    assert brain.memory_lines([f"{i}" + "a" * 300 for i in range(10)], chars=700) == \
        [f"{i}" + "a" * 298 + "…" for i in (8, 9)]


def test_job_ids_are_per_channel_and_unique():
    a, b = brain.job_id_for(77), brain.job_id_for(77)
    assert a.startswith("c77-") and a != b



@pytest.mark.asyncio
async def test_two_ask_brain_calls_in_one_turn_keep_their_own_results():
    # The SDK runs one response's tool calls at once, and both brain calls
    # share the persona's context: they must take turns (ask_brain's lock).
    from classes.tool_functions import ask_brain
    brain_model = ScriptedModel([
        ModelStep(output=[_report_call("r1", facts=["Weather: 12°C."])]),
        ModelStep(output=[_report_call("r2", facts=["Posted the poll."])]),
    ])
    persona_model = ScriptedModel([
        ModelStep(output=[function_call("ask_brain", {"request": "weather"}, call_id="b1"),
                          function_call("ask_brain", {"request": "poll"}, call_id="b2")]),
        ModelStep(output=[assistant_message("done")]),
    ])
    persona = Agent(name="Assistant", model=persona_model, tools=[ask_brain])
    ctx = _ctx(brain_reports=[])
    with _brain(brain_model):
        await Runner.run(persona, "weather and a poll", context=ctx,
                         run_config=RunConfig(tracing_disabled=True))
    facts = sorted(r.facts[0] for r in ctx["brain_reports"])
    assert facts == ["Posted the poll.", "Weather: 12°C."]
    told = json.dumps(persona_model.calls[1].input, ensure_ascii=False)
    assert "Weather: 12°C." in told and "Posted the poll." in told


@pytest.mark.asyncio
async def test_a_report_without_facts_is_still_a_clean_report():
    model = ScriptedModel([ModelStep(output=[function_call("report", {"status": "done"}, call_id="r1")])])
    with _brain(model):
        report = await brain.run_brain(_ctx(), "x")
    assert report.status == brain_result.DONE and report.facts == []


@pytest.mark.asyncio
async def test_report_with_bad_arguments_never_becomes_a_fact():
    model = ScriptedModel([ModelStep(output=[function_call("report", {"facts": "x"}, call_id="r1")])])
    with _brain(model):
        report = await brain.run_brain(_ctx(), "x")
    assert "Reported." not in report.facts


@pytest.mark.asyncio
async def test_an_unmapped_error_stops_the_run_but_still_delivers_its_files():
    from agents import ModelTimeoutError
    session = _Session()
    session.files = _AnyPath(b"x")
    model = ScriptedModel([
        ModelStep(output=[function_call("attach_file", {"path": "out/a.txt"}, call_id="a1")]),
        ModelStep(error=ModelTimeoutError(5.0)),
    ])
    ctx = _ctx()
    with _brain(model, session=session):
        report = await brain.run_brain(ctx, "x")
    # Not a bare failure: the file it chose is sent and named, as for a timeout.
    assert report.status == brain_result.PARTIAL
    assert report.files_posted == ["a.txt"]
    assert "error" in report.note
    embeds = [c.kwargs["embeds"][0] for c in ctx["channel"].send.await_args_list if c.kwargs.get("embeds")]
    edits = ctx["channel"].send.return_value.edit.await_args_list
    last = edits[-1].kwargs["embeds"][0] if edits else embeds[-1]
    assert last.title == "🧠 Stopped"


@pytest.mark.asyncio
async def test_a_report_that_cannot_be_read_is_not_called_done():
    model = ScriptedModel([ModelStep(output=[function_call("report", "{not json", call_id="r1")])])
    with _brain(model):
        report = await brain.run_brain(_ctx(), "x")
    assert report.status == brain_result.PARTIAL
    assert "could not be read" in report.note


@pytest.mark.asyncio
async def test_report_arguments_that_miss_the_schema_are_kept():
    # facts sent as one string instead of a list fails the SDK's validation;
    # the report still carries it.
    model = ScriptedModel([ModelStep(output=[function_call(
        "report", {"status": "done", "facts": "Posted the poll."}, call_id="r1")])])
    with _brain(model):
        report = await brain.run_brain(_ctx(), "x")
    assert report.status == brain_result.DONE
    assert report.facts == ["Posted the poll."]


@pytest.mark.asyncio
async def test_files_wait_for_the_channel_lock_like_a_reply():
    import asyncio
    from types import SimpleNamespace
    from classes.message_queue import get_channel_lock
    channel = MagicMock()
    channel.id = 987654
    channel.send = AsyncMock()
    lock = get_channel_lock(channel.id)
    await lock.acquire()  # another reply is sending its chunks
    task = asyncio.create_task(brain._post_files(
        channel, [SimpleNamespace(name="chart.png", data=b"x", caption="")]))
    await asyncio.sleep(0.01)
    assert channel.send.await_count == 0
    lock.release()
    assert await task == ["chart.png"]


def test_brain_calls_share_one_model_client(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_MODEL", "some/local-model")
    assert brain.build_brain_model() is brain.build_brain_model()

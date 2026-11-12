import asyncio
import json
import logging
import os,aiohttp, discord, io, time
from classes.user_memory import UserMemory
from classes.metrics import (
    inc_brain_facts_dropped,
    inc_llm_error,
    inc_persona_escalation,
    inc_tool_call,
    inc_tool_error,
    observe_llm_call,
    observe_llm_completion_tokens,
    observe_llm_prompt_tokens,
    observe_tool_duration,
    set_context_window,
    set_llm_slots,
    set_slot_context,
)
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, FunctionTool, function_tool, RunContextWrapper, ModelSettings, RunHooks, set_tracing_disabled
from agents import ItemHelpers
from agents.models.interface import ModelTracing
from agents.run_error_handlers import RunErrorHandlerResult
from openai import BadRequestError, UnprocessableEntityError
from openai.types.responses import ResponseCreatedEvent, ResponseTextDeltaEvent
from classes.config_manager import configManager
from classes.response_filter import extract_reasoning_items, extract_thinking, strip_tool_call_text
from classes import llm_lanes
from classes.laned_model import LanedModel
from classes.llm_roles import PERSONA, distinct_hosts, parse_temperature, role_config
from classes.reply_policy import INSTRUCTION as DOUBLE_REPLY_INSTRUCTION
from classes.side_llm import NO_THINKING
from classes import voice_policy

# The agents SDK uploads every run (prompt, replies, tool calls - so
# Discord users' messages) to OpenAI's trace dashboard by default, whenever
# OPENAI_API_KEY is set (v3 names the moderation key CONTENT_GUARD_API_KEY, but
# an operator may still export the old one for something else). Off for good,
# in code: the SDK lets this call win over its
# OPENAI_AGENTS_DISABLE_TRACING env var and a run's own RunConfig.
# brain.py does the same for the nested agent.
set_tracing_disabled(True)

# Max turns for ONE reply from the persona (a turn = one model response,
# however many tool calls it carries). The persona has one tool, ask_brain, so
# a reply is at most: ask the brain, maybe ask again, answer. An overrun is
# still answered (_wrap_up), not failed.
DEFAULT_PERSONA_MAX_TURNS = 4


def persona_max_turns() -> int:
    """Max model turns for one reply (PERSONA_MAX_TURNS, default 4)."""
    try:
        value = int(str(os.environ.get("PERSONA_MAX_TURNS")).strip())
        return value if value > 0 else DEFAULT_PERSONA_MAX_TURNS
    except (TypeError, ValueError):
        return DEFAULT_PERSONA_MAX_TURNS


# Sent after the last allowed turn, when a reply runs out of turns. Before
# this the run raised MaxTurnsExceeded and the user got a bare ❌, even though
# the tools had already posted their results; a long chain (search, read,
# draw, the brain) is exactly the reply that runs out.
WRAP_UP_INSTRUCTION = (
    "(You have used every step you get for this reply, so no more tools can "
    "run. Write your reply to the user now from what you already have: what "
    "you found or made, and plainly what is still not done.)"
)


from classes.message_queue import (
    SLOW_TOOL_NAMES,
    TOOL_DISPLAY,
    register_task_run,
    unregister_task_run,
)


from classes.tool_functions import ask_brain, get_current_datetime

logger = logging.getLogger(__name__)


# The persona's host/model/key never change at runtime, so the AsyncOpenAI
# client (its own httpx connection pool) is built once and reused across every
# message instead of per-message. Safe without locking: no `await` between
# the check and the set, so no race is possible even with concurrent worker
# tasks (the event loop is single-threaded).
_main_model_client = None


def _get_main_model_client() -> LanedModel:
    """The persona's model, in the chat lane of its server.

    Every call is bounded by PERSONA_CALL_TIMEOUT_SECONDS (wall clock, the
    client's own retries included), applied by LanedModel once a slot is
    taken, so waiting behind other work never counts toward it. The client's
    own timeout is per-read: it catches a server that goes silent, but one
    that keeps the connection warm while it works (OpenRouter pads
    non-streaming responses) resets it forever, which only the wall-clock
    bound catches.
    """
    global _main_model_client
    if _main_model_client is None:
        config = role_config(PERSONA)
        _main_model_client = LanedModel(
            OpenAIChatCompletionsModel(
                model=config.model,
                openai_client=AsyncOpenAI(
                    base_url=config.host + "/v1",
                    api_key=config.client_api_key,
                    timeout=config.call_timeout,
                    default_headers=config.extra_headers or None,
                ),
            ),
            config.host_key, config.call_timeout, lane=llm_lanes.CHAT, role=config.role,
        )
    return _main_model_client


# Per-slot context of the LLM server (its /props), refreshed by main.py.
# None until the first successful read: the history token budget then does
# not trim at all rather than guessing a capacity.
_slot_context: int | None = None


def slot_context_tokens() -> int | None:
    """The last per-slot context read from the LLM server, or None."""
    return _slot_context


def parse_slot_context(props) -> int | None:
    """Per-slot context from a /props body (llama.cpp's, which Strata
    copies), or None.

    default_generation_settings.n_ctx is the SLOT context (the server's
    --ctx-size divided across --parallel slots, rounded up to its padding), not
    the configured total - the number one request can actually fill.
    """
    try:
        value = int(props["default_generation_settings"]["n_ctx"])
    except (KeyError, TypeError, ValueError):
        return None
    return value if value > 0 else None


def parse_total_slots(props) -> int | None:
    """Number of parallel slots from a /props body (total_slots), or None."""
    try:
        value = int(props["total_slots"])
    except (KeyError, TypeError, ValueError):
        return None
    return value if value > 0 else None


def _key_names(config) -> str:
    """The settings that give `config`'s role its key, for a log line."""
    if config.role == PERSONA:
        return "LLM_API_KEY (or PERSONA_LLM_API_KEY)"
    return f"{config.role.upper()}_LLM_API_KEY (or LLM_API_KEY)"


async def _read_props(config) -> dict | None:
    """One server's /props, with its role's key, or None (never raises).

    Only a body shaped like llama.cpp's (which Strata copies) counts: a
    remote API that answers 200 with something else is not self-hosted.
    """
    url = config.host + "/props"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=config.auth_headers(),
                                   timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    props = await response.json()
                    if isinstance(props, dict) and ("total_slots" in props
                                                    or "default_generation_settings" in props):
                        return props
                    logger.info(f"LLM server {config.host}: /props is not llama.cpp's; not laned")
                elif response.status in (401, 403):
                    # Not a slow start: the key is wrong, and (for the persona's
                    # server) the history budget stops trimming until it is fixed.
                    logger.warning(f"LLM server /props rejected the API key ({response.status}); "
                                   f"{_key_names(config)} must be the server's key")
                else:
                    logger.info(f"LLM server {config.host} /props returned {response.status}")
    except Exception as e:
        logger.info(f"Could not read LLM server /props at {url}: {e}")
    return None


def served_model_names(body) -> list[str]:
    """Model names in a /v1/models body, in either shape a server returns.

    llama.cpp sends both its own `models` list (name) and the OpenAI `data`
    list (id); Strata sends only `data`. Reading only `models`, as this did,
    made a Strata server look as if it had nothing loaded.
    """
    if not isinstance(body, dict):
        return []
    names = []
    for key in ("models", "data"):
        entries = body.get(key)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, dict):
                name = entry.get("name") or entry.get("id")
                if name and name not in names:
                    names.append(name)
    return names


class ToolMetricsHooks(RunHooks):
    """RunHooks that record per-tool metrics AND track the long tools in the
    in-flight registry (classes.message_queue).

    Attached to Runner.run instead of wrapping every tool. Tool-level error
    detection relies on the SDK's default failure handler: the tools in
    tool_functions.py catch their own exceptions and return friendly strings,
    so only unexpected tool exceptions reach the SDK handler (whose default
    result starts with the prefix below).

    In-flight tracking: the slow tools (SLOW_TOOL_NAMES) register themselves
    at start and unregister at end, so a NEWER message in the same channel
    (the per-channel lock only guards build+send) gets a hint in its prompt
    that the older one is still being processed.
    """

    _SDK_FAILURE_PREFIX = "An error occurred while running the tool"
    # The tool argument that identifies what a slow tool is doing (shown,
    # truncated, in the in-flight hint).
    _SOURCE_FIELDS = {
        "ask_brain": "request",
    }

    def __init__(self, guild_id, caller: str = "main"):
        """caller labels the model-call metrics: "main" for the persona,
        "brain" for a brain call (classes/brain.py attaches these hooks too,
        so the tools it runs are still counted)."""
        self.guild_id = guild_id
        self.caller = caller
        self._starts: dict[str, tuple[float, str]] = {}
        # Start of the model request in progress. One run's requests are
        # sequential, so one slot is enough; a request that raises never
        # reaches on_llm_end, so generate() closes it via abandon_llm_call().
        self._llm_started: float | None = None

    async def on_llm_start(self, context, agent, system_prompt, input_items) -> None:
        self._llm_started = time.monotonic()

    async def on_llm_end(self, context, agent, response) -> None:
        try:
            started, self._llm_started = self._llm_started, None
            if started is not None:
                observe_llm_call(self.caller, "ok", self._model_seconds(started))
            usage = getattr(response, "usage", None)
            observe_llm_completion_tokens(self.caller, getattr(usage, "output_tokens", None))
        except Exception as e:
            logger.warning(f"Metrics llm_end hook failed: {e}")

    def abandon_llm_call(self, outcome: str) -> None:
        """Record the request in progress, if any, as ended with `outcome`."""
        try:
            started, self._llm_started = self._llm_started, None
            if started is not None:
                observe_llm_call(self.caller, outcome, self._model_seconds(started))
        except Exception as e:
            logger.warning(f"Metrics llm abandon failed: {e}")

    @staticmethod
    def _model_seconds(started: float) -> float:
        """Seconds since on_llm_start minus the call's lane wait (which
        on_llm_start precedes, and which has its own metric). A call
        cancelled while still waiting counts its wait."""
        return max(0.0, time.monotonic() - started - llm_lanes.last_wait.get())

    def _key(self, context, tool) -> str:
        # ToolContext carries a tool_call_id; several concurrent calls of the
        # same tool are distinguished by it. Fall back to the tool name.
        return getattr(context, "tool_call_id", None) or getattr(tool, "name", "unknown")

    def _tool_name(self, context, tool) -> str:
        return getattr(context, "tool_name", None) or getattr(tool, "name", "unknown")

    def _task_source(self, context, name: str) -> str:
        """The tool argument shown in the in-flight hint (the request).

        A function tool's arguments are in `tool_arguments` (the JSON
        string); the SDK leaves `tool_input` empty for them. Reading only
        `tool_input` left every hint entry without its request, and a
        finished entry with no request reads as "the job you are answering
        is done": the persona then claimed work it never asked for (about a
        quarter of action requests in a busy channel, live on v3)."""
        raw = getattr(context, "tool_input", None)
        if raw is None:
            try:
                raw = json.loads(getattr(context, "tool_arguments", "") or "")
            except (TypeError, ValueError):
                raw = None
        field = self._SOURCE_FIELDS.get(name)
        if field is not None and isinstance(raw, dict):
            return str(raw.get(field) or "")
        if isinstance(raw, str):
            return raw
        return ""

    def _channel_id(self, context) -> int:
        # The user_info dict passed to Runner.run(context=...) carries the
        # original message; the hook's `context` is the RunContextWrapper,
        # so the dict lives on context.context.
        return (context.context.get("channel") or context.context["original_message"].channel).id

    # Hook failures must never abort the run (same rule as brain_progress).

    async def on_tool_start(self, context, agent, tool) -> None:
        name = self._tool_name(context, tool)
        try:
            self._starts[self._key(context, tool)] = (time.monotonic(), name)
        except Exception as e:
            logger.warning(f"Metrics tool_start hook failed: {e}")
        if name in SLOW_TOOL_NAMES:
            try:
                register_task_run(
                    self._channel_id(context),
                    TOOL_DISPLAY.get(name, name),
                    self._task_source(context, name),
                    run_key=self._key(context, tool),
                )
            except Exception as e:
                logger.warning(f"In-flight registry tool_start failed: {e}")

    async def on_tool_end(self, context, agent, tool, result) -> None:
        name = self._tool_name(context, tool)
        try:
            started = self._starts.pop(self._key(context, tool), None)
            inc_tool_call(name, self.guild_id)
            if started is not None:
                observe_tool_duration(name, self.guild_id, time.monotonic() - started[0])
            if isinstance(result, str) and result.startswith(self._SDK_FAILURE_PREFIX):
                inc_tool_error(name, self.guild_id)
        except Exception as e:
            logger.warning(f"Metrics tool_end hook failed: {e}")
        if name in SLOW_TOOL_NAMES:
            try:
                unregister_task_run(self._channel_id(context), run_key=self._key(context, tool))
            except Exception as e:
                logger.warning(f"In-flight registry tool_end failed: {e}")


# Whether a voice turn asks the server to skip thinking. Cleared for the
# process once a backend has rejected the option (see generate_streamed).
_voice_latch = {"send_no_thinking": True}


def voice_thinking_off() -> bool:
    return not voice_policy.settings()["thinking"] and _voice_latch["send_no_thinking"]


def persona_tools() -> list:
    """The persona's tools: exactly one, ask_brain, in chat, voice turns and
    automatic runs alike (maintainer, 2026-11-02). With one tool its choice is
    "answer from the chat, or ask the brain", which fixes its habit of trying
    jobs it could not do. Every other tool is the brain's, and what differs
    between chat, voice and automatic runs is decided there
    (classes/brain_tools.py)."""
    return [ask_brain]


def trailing_block(datetime: str, memories=None, name: str | None = None) -> str:
    """The per-message block after the history: the time and, when there
    are any, what is saved about the person asking.

    It goes LAST, after the history, so the cached prompt prefix (system
    prompt + the ask_brain schema + history) does not change when a memory
    does; the voice prefill sends the request without it for the same
    reason. role="user" (not "system"): some chat templates raise for a
    system message that isn't the very first one.

    The memories are what fixed recall (broken since 2025-08-27, when they
    were dropped from the prompt and no tool read them back): "what units do
    I like?" is answered from here, with no brain call. Capped like the
    brain's task (brain.MEMORY_COUNT newest, within brain.MEMORY_CHARS), by
    the same memory_lines. Saving or deleting one is brain work.
    """
    from classes.brain import memory_lines
    text = f"(Current datetime: {datetime})"
    lines = memory_lines(memories)
    if lines:
        who = name or "the person asking"
        text += f"\n(Saved about {who}:\n" + "\n".join(f"- {m}" for m in lines) + ")"
    return text


class TextLLMHandler:
    # Set by MessageHandler after eligibility is chosen under the build lock.
    allow_double_reply = False

    def __init__(self, messages, guild_id, original_message, client=None,
                 actor_id=None, channel=None, automatic=False, voice=False, request_text=None,
                 bot_name=None, followup=False, unprompted=False, no_tools=False):
        self.original_message = original_message
        self.messages = messages
        self.guild_id = guild_id
        # The discord.Client, forwarded to tool-run context as
        # "discord_client" (see generate()) and on into the brain
        # run's context. Nothing there waits on it any more (ask_user's
        # replies arrive through the thread ledger), so it is optional/None
        # for callers and tests.
        self.client = client
        self.config = configManager()
        self.actor_id = actor_id if actor_id is not None else original_message.author.id
        self.channel = channel if channel is not None else original_message.channel
        self.automatic = automatic
        # A follow-up run (automation_runner): the event tools then skip the
        # requester's own permission check (event_tools._checks_requester).
        self.followup = followup
        # A reply the bot was not asked for (the random reply chance): only
        # talk, no brain (_model_settings, and ask_brain refuses). One such
        # reply once spent three minutes searching people's messages, posting
        # a card for each search (prod, 2026-11-12).
        self.unprompted = unprompted
        # No tools at all: the reply when a workflow ends only voices its
        # result (classes/workflows.reply_done).
        self.no_tools = no_tools
        # A turn in a voice call: spoken instructions, thinking off, and no
        # source message (original_message is None; request_text is what the
        # speaker said, for tools that compare their result against it).
        self.voice = voice
        self.request_text = request_text
        # The bot's display name, told to the model in a voice call.
        self.bot_name = bot_name
        self.user_memory = None if automatic else UserMemory(self.actor_id, guild_id)
        # Filled in by generate(): the model's internal reasoning for this
        # run, which the caller sends to Discord behind a spoiler when
        # SHOW_THINKING is on. It cannot be recovered from the returned
        # answer - our llama.cpp server strips it out of the visible content
        # into reasoning_content - so it is handed over separately here
        # rather than by changing generate()'s string return (and its
        # "Error" sentinel).
        self.reasoning = ""
        # Filled in by generate(): whether this reply called the brain.
        self.used_brain = False
        # Skill reviews of long brain jobs, started after the reply is sent.
        self.skill_reviews = []

    def _after_run(self, user_info, reply) -> None:
        """Records whether the reply called the brain, and checks the reply
        kept the brain's must-keep facts (logged and counted, never
        retried). Never raises."""
        try:
            reports = user_info.get("brain_reports") or []
            self.used_brain = bool(reports)
            self.skill_reviews = list(user_info.get("skill_reviews") or [])
            failed = not isinstance(reply, str) or reply == "Error"
            # A run that failed before calling the brain answered nothing, so
            # counting it as "answered" would skew the routing metric; nor is
            # an unprompted reply a routing choice (it has no brain).
            if (reports or not failed) and not user_info.get("unprompted"):
                inc_persona_escalation("brain" if reports else "answered")
            # A spoken reply says "8 degrees", never "8°C" or a URL, so the
            # exact-token check would count every voice turn as dropping facts.
            if not isinstance(reply, str) or reply == "Error" or user_info.get("voice"):
                return
            from classes.brain_result import missing_tokens
            dropped = [t for r in reports for t in missing_tokens(reply, r)]
            if dropped:
                logger.warning(f"Persona reply left out brain facts: {dropped}")
                inc_brain_facts_dropped(len(dropped))
        except Exception as e:
            logger.warning(f"Could not check the reply against the brain's facts: {e}")


    @staticmethod
    async def check_model_ready(model: str):
        # llama.cpp has no pull endpoint: the llamacpp container downloads the model
        # on boot (LLAMA_ARG_HF_REPO into the LLAMA_CACHE volume). We only check that
        # the configured model is loaded (fail-soft: it may still be downloading).
        config = role_config(PERSONA)
        url = config.host + "/v1/models"
        logger.info(f"Checking model {model} on LLM host ({url})")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=config.auth_headers()) as response:
                    if response.status in (401, 403):
                        logger.warning(f"LLM server rejected the API key ({response.status}); {_key_names(config)} must be the server's key")
                        return
                    if response.status != 200:
                        logger.warning(f"LLM server not ready yet ({response.status}); model may still be downloading")
                        return
                    available = served_model_names(await response.json())
                    if model in available:
                        logger.info(f"Model {model} is available")
                    elif len(available) == 1:
                        # A one-model server (Strata) answers whatever name
                        # is requested, so this is not a failure to wait on.
                        logger.info(f"LLM server serves one model, {available[0]}; requests for {model} go to it")
                    else:
                        logger.warning(f"Model {model} not loaded yet (server has: {available})")
        except Exception as e:
            logger.warning(f"Could not reach LLM server at {url}: {e}")
      
    @staticmethod
    async def refresh_slot_context() -> int | None:
        """Re-read /props from every role's server (never raises); returns
        the persona's per-slot context.

        A server whose /props answers is self-hosted: it gets lanes
        (classes/llm_lanes.py) sized by its total_slots, else LLM_SLOTS, else
        1, and its total context (per-slot x slots) is published. A server
        that never answers (a remote API) is not laned; one that answered
        before keeps its lanes and last slot count through a failed read.

        The persona's per-slot context drives the history token budget. On
        failure the last value is kept for the budget but the gauge is marked
        unavailable, so a dashboard can tell stale from measured.
        """
        global _slot_context
        value = None
        for index, config in enumerate(distinct_hosts()):
            props = await _read_props(config)
            if props is None:
                continue
            per_slot = parse_slot_context(props)
            if index == 0:  # distinct_hosts() lists the persona's server first
                value = per_slot
            slots, source = llm_lanes.resolve_slots(parse_total_slots(props))
            lanes = llm_lanes.lanes_for(config.host_key)
            if lanes is None or lanes.slots != slots:
                logger.info(f"LLM server {config.host}: {slots} slot(s) ({source}), lanes on")
            llm_lanes.configure_host(config.host_key, slots, llm_lanes.chat_reserved(), per_slot)
            set_llm_slots(config.host_key, slots, source)
            set_context_window(config.host_key, per_slot * slots if per_slot else None)
        if value is None:
            set_slot_context(None)
            return _slot_context
        if value != _slot_context:
            logger.info(f"LLM per-slot context: {value} tokens")
        _slot_context = value
        set_slot_context(value)
        return value

    async def get_settings(self):
        self.system = await self.config.get_setting("system", self.guild_id) or "An AI Story Teller"
        self.model = role_config(PERSONA).model
        # parse_temperature, not `float(...) or 1.0`: 0.0 is falsy, so the
        # original turned a deliberate /temperature 0 back into 1.0.
        self.options = {
            "temperature": parse_temperature(
                await self.config.get_setting("temperature", self.guild_id)
            )
        }

    async def get_client(self):
        main_model_client = _get_main_model_client()
        voice = getattr(self, "voice", False)
        tools = [] if getattr(self, "no_tools", False) else persona_tools()
        instructions = self.system
        if voice:
            # After the personality, and the same for the whole call, so it
            # stays inside the prompt prefix llama.cpp keeps cached.
            instructions = f"{self.system}\n\n{voice_policy.voice_instructions(getattr(self, 'bot_name', None) or '')}"

        self.agent = Agent(
            name="Assistant",
            instructions=instructions,
            model=main_model_client,
            tools=tools,
            model_settings=self._model_settings(),
            # The SDK sets tool_choice back to "auto" after a tool call; an
            # unprompted reply's "none" must hold even if a server ignored it.
            reset_tool_choice=not getattr(self, "unprompted", False),
        )

    def _model_settings(self, no_thinking: bool | None = None) -> ModelSettings:
        """`no_thinking` overrides the voice latch for one request (the
        retry without the option, before it is known to work)."""
        if no_thinking is None:
            no_thinking = voice_thinking_off()
        settings = dict(
            temperature=self.options["temperature"],
            # No frequency_penalty: at 1.1 it pushed the model off tokens it
            # had just read, so tool arguments that repeat them (a date from
            # list_events, an id, a name) came out mangled - "2026-10-23T21:00
            # +08:35 UTC+8?" - in about a third of calls.
            top_p=1.0,
            # No timeout here: LanedModel applies PERSONA_CALL_TIMEOUT_SECONDS
            # after the lane wait, which the SDK's own would include.
        )
        if getattr(self, "unprompted", False):
            # A reply nobody asked for only talks. The tools are still sent,
            # refused, as in _wrap_up (llama.cpp keeps them in the prompt, so
            # its cached prefix holds; Strata leaves refused tools out, so
            # there it does not); tool-call text written anyway is stripped
            # (generate), and a real call is refused by ask_brain.
            settings["tool_choice"] = "none"
        if getattr(self, "voice", False) and no_thinking:
            # Off, not merely "low": a spoken reply waits for every reasoning
            # token before its first word. Same switch the image side calls
            # use (side_llm.NO_THINKING).
            settings["extra_body"] = NO_THINKING
        else:
            settings["reasoning"] = {"effort": role_config(PERSONA).reasoning_effort}
        return ModelSettings(**settings)

    async def _prepare(self) -> dict:
      """Settings, agent and run context: everything generate(),
      generate_streamed() and prefill() share. Built in exactly one place so
      a voice prefill sends the same prompt prefix as the turn after it."""
      await self.get_settings()
      user_info = {
        # The requester's saved memories: shown to the persona in the
        # trailing block, and to the brain in its task.
        "data": await self.user_memory.get() or [] if getattr(self, "user_memory", None) else [],
        "user_id": self.actor_id if hasattr(self, "actor_id") else self.original_message.author.id,
        "guild_id": self.guild_id,
        "original_message": self.original_message,
        "channel": self.channel if hasattr(self, "channel") else self.original_message.channel,
        "automatic": getattr(self, "automatic", False),
        "followup": getattr(self, "followup", False),
        "unprompted": getattr(self, "unprompted", False),
        "discord_client": self.client,
        "redis_save_tool_calls": 0,
        "personality_tool_calls": 0,
        "poll_tool_calls": 0,
        "events_made": [],
        "event_tool_failures": 0,
        # What a voice speaker said (no source message to read it from).
        "request_text": getattr(self, "request_text", None),
        # For the brain (classes/brain.py): what kind of run this is (its
        # tools differ), the history it is shown as text, and who asked.
        "voice": getattr(self, "voice", False),
        "persona_messages": self.messages,
        "requester_name": self._requester_name(),
        # One per ask_brain call, read back after the reply (_after_run).
        "brain_reports": [],
      }
      self.system = f"Answer as if you are {self.system}."
      await self.get_client()
      return user_info

    def _requester_name(self) -> str | None:
        """The asking person's display name, for the memory block and the
        brain's task. A voice turn and an automatic run have no message, so
        the member cache is tried (often a miss: no members intent)."""
        author = getattr(getattr(self, "original_message", None), "author", None)
        if author is not None:
            return getattr(author, "display_name", None) or getattr(author, "name", None)
        guild = getattr(getattr(self, "channel", None), "guild", None)
        actor = getattr(self, "actor_id", None)
        member = guild.get_member(actor) if guild is not None and actor is not None else None
        return getattr(member, "display_name", None)

    def _trailing(self, user_info, datetime: str) -> dict:
        return {"role": "user", "content": trailing_block(
            datetime, user_info.get("data"), user_info.get("requester_name"))}

    async def generate(self):
      user_info = await self._prepare()
      datetime = await get_current_datetime()
      # The datetime (and the person's saved memories) are appended as a
      # trailing message rather than folded into the system prompt: the
      # system prompt + growing history stays byte-identical across turns of
      # the same conversation, so llama.cpp's prefix cache only has to
      # prefill the new tail instead of the whole prompt every time. See
      # trailing_block.
      messages_for_run = self.messages + [self._trailing(user_info, datetime)]
      if self.allow_double_reply and not getattr(self, "automatic", False):
          messages_for_run.append({"role": "user", "content": DOUBLE_REPLY_INSTRUCTION})
      hooks = ToolMetricsHooks(self.guild_id)
      try:
         response = await Runner.run(self.agent, messages_for_run, context=user_info,
                                     max_turns=persona_max_turns(),
                                     hooks=hooks,
                                     error_handlers={"max_turns": self._wrap_up})
         logger.info('Response generated')
         logger.debug(response)
         final_output = response.final_output
         if getattr(self, "unprompted", False) and isinstance(final_output, str):
             # Tools refused (_model_settings): a call written out as text
             # is not a reply.
             final_output = strip_tool_call_text(final_output)
         self._capture_reasoning(response.new_items, final_output)
         self._record_prompt_tokens(response.raw_responses)
         self._after_run(user_info, final_output)
         return final_output
      except asyncio.CancelledError:
         hooks.abandon_llm_call("cancelled")
         raise
      except Exception as e:
         hooks.abandon_llm_call("error")
         self._after_run(user_info, None)
         logger.warning('Failed to get response from LLM: ' + str(e))
         # A run that died part-way (MaxTurnsExceeded after a few chained
         # tool calls is the common one) still reasoned before it broke, and
         # its tools have already posted their cards and files to the channel.
         # The SDK hangs the completed turns off the exception as
         # RunErrorDetails, so recover the reasoning from there rather than
         # leaving the user with a bare ❌ and no idea what happened.
         run_data = getattr(e, "run_data", None)
         self._capture_reasoning(getattr(run_data, "new_items", None), None)
         self._record_prompt_tokens(getattr(run_data, "raw_responses", None))
         inc_llm_error(self.guild_id)
         return "Error"

    async def generate_streamed(self, on_text, on_model_start=None, on_tool=None, hooks=None):
      """generate() for a voice turn: the answer is handed to `on_text` as it
      is written, so the first sentence can be spoken while the model is
      still writing the rest. Returns the final answer, or "Error".

      on_model_start() fires as each model call starts (one per turn of a
      tool-calling run), and on_tool(name, arguments) as the model calls a
      tool. Between them the caller can tell whether the model said anything
      before a tool call -- the spoken "hold on" line.

      Thinking is off for voice (see _model_settings). A backend that rejects
      that option gets one retry without it, and is not sent it again, the
      side_llm latch rule: only a retry that WORKED identifies the option.
      """
      user_info = await self._prepare()
      datetime = await get_current_datetime()
      # Same trailing block as generate(): everything before it is the
      # prompt prefix a voice prefill warmed (see prefill()).
      messages_for_run = self.messages + [self._trailing(user_info, datetime)]
      hooks = hooks or ToolMetricsHooks(self.guild_id)
      emitted = False
      retried_without_option = False
      for attempt in (1, 2):
        sent_no_thinking = self.agent.model_settings.extra_body is not None
        try:
          result = Runner.run_streamed(self.agent, messages_for_run, context=user_info,
                                       max_turns=persona_max_turns(), hooks=hooks)
          async for event in result.stream_events():
            if event.type == "raw_response_event":
              if isinstance(event.data, ResponseTextDeltaEvent) and event.data.delta:
                emitted = True
                await on_text(event.data.delta)
              elif isinstance(event.data, ResponseCreatedEvent) and on_model_start:
                await on_model_start()
            elif (event.type == "run_item_stream_event" and event.name == "tool_called"
                  and on_tool is not None):
              raw = getattr(event.item, "raw_item", None)
              await on_tool(getattr(raw, "name", "") or "", getattr(raw, "arguments", "") or "")
          final_output = result.final_output
          self._capture_reasoning(result.new_items, final_output)
          self._record_prompt_tokens(result.raw_responses)
          final_text = final_output if isinstance(final_output, str) else str(final_output or "")
          self._after_run(user_info, final_text)
          if retried_without_option:
            # Only now is the option known to be what the server refused;
            # a retry that failed too says nothing about it.
            _voice_latch["send_no_thinking"] = False
          return final_text
        except asyncio.CancelledError:
          hooks.abandon_llm_call("cancelled")
          raise
        except (BadRequestError, UnprocessableEntityError) as e:
          if attempt == 1 and sent_no_thinking and not emitted:
            logger.info(f"Voice: request rejected with thinking disabled ({e}); "
                        f"retrying without that option")
            retried_without_option = True
            self.agent = self.agent.clone(model_settings=self._model_settings(no_thinking=False))
            continue
          return self._streamed_failure(hooks, user_info, e)
        except Exception as e:
          return self._streamed_failure(hooks, user_info, e)

    async def _wrap_up(self, data):
      """The SDK's max_turns handler: one last model call, with tools
      refused, that writes the reply from what the run already did.

      Sent the same tools with tool_choice "none" rather than none at all, so
      the prompt prefix matches the turns before it and llama.cpp reuses its
      cache. Called on the model directly, not through Runner, so a tool call
      the model makes anyway is dropped instead of run, and one it writes out
      as text is stripped (strip_tool_call_text). Returns None (the
      run then raises MaxTurnsExceeded, the old ❌ path) when the call fails
      or writes no text.

      Text replies only: a voice turn speaks what streams, and this answer
      would not stream.
      """
      started = time.monotonic()
      outcome = "error"
      try:
        run_context = data.context
        response = await self.agent.model.get_response(
            system_instructions=await self.agent.get_system_prompt(run_context),
            input=list(data.run_data.history) + [{"role": "user", "content": WRAP_UP_INSTRUCTION}],
            model_settings=self.agent.model_settings.resolve(ModelSettings(tool_choice="none")),
            tools=await self.agent.get_all_tools(run_context),
            output_schema=None,
            handoffs=[],
            tracing=ModelTracing.DISABLED,
            previous_response_id=None,
            conversation_id=None,
            prompt=None,
        )
        self._record_prompt_tokens([response])
        text = strip_tool_call_text("".join(ItemHelpers.extract_last_text(item) or "" for item in response.output))
        outcome = "ok" if text else "empty"
      except asyncio.CancelledError:
        outcome = "cancelled"
        raise
      except Exception as e:
        logger.warning(f"Max turns wrap-up failed: {e}")
        return None
      finally:
        observe_llm_call("max_turns_wrap_up", outcome, time.monotonic() - started)
      if not text:
        return None
      logger.info("Reply ran out of turns; answered with a wrap-up")
      return RunErrorHandlerResult(final_output=text)

    def _streamed_failure(self, hooks, user_info, e) -> str:
      hooks.abandon_llm_call("error")
      self._after_run(user_info, None)
      logger.warning('Failed to get a streamed response from LLM: ' + str(e))
      inc_llm_error(self.guild_id)
      return "Error"

    async def prefill(self) -> None:
      """Warm llama.cpp's prompt cache with this conversation, so the next
      voice turn only has to process its tail.

      Sends exactly what generate_streamed() would send for these messages --
      same instructions, tools and model settings, built by the same
      _prepare() -- minus the trailing block (time, memories), and asks for one
      token. llama.cpp keeps the processed prompt in its slot; the real turn
      is that prompt plus a few tokens, so it is answered almost at once.
      Measured on the dev stack: a 2.4k-token voice prompt went from 1.0s of
      prompt processing to 0.3s, 52 tokens instead of 2427. Other guilds share
      the slot; llama.cpp's --cache-ram restores this prefix after them.
      """
      user_info = await self._prepare()
      run_context = RunContextWrapper(context=user_info)
      tools = await self.agent.get_all_tools(run_context)
      settings = self.agent.model_settings.resolve(ModelSettings(max_tokens=1))
      started = time.monotonic()
      outcome = "error"
      try:
        await self.agent.model.get_response(
            system_instructions=await self.agent.get_system_prompt(run_context),
            input=self.messages,
            model_settings=settings,
            tools=tools,
            output_schema=None,
            handoffs=[],
            tracing=ModelTracing.DISABLED,
        )
        outcome = "ok"
      except asyncio.CancelledError:
        outcome = "cancelled"
        raise
      finally:
        observe_llm_call("voice_prefill", outcome, time.monotonic() - started)

    def _record_prompt_tokens(self, raw_responses):
        """Record the prompt size of every model call in this run (never raises).

        One reply is several model calls - one per turn - and each sends the
        whole conversation so far, so each is its own sample of "how much of
        the context window did we need". Recording only the last (or only the
        aggregate on the run) would miss that a tool-heavy reply grows the
        prompt turn by turn.

        Called on the error path too, from RunErrorDetails.raw_responses: a run
        that died on MaxTurnsExceeded is exactly the one whose prompts got big.

        Fully guarded, same rule as _capture_reasoning: a surprise in the
        run-item shape must not turn a good answer into the "Error" sentinel.
        """
        try:
            for model_response in raw_responses or []:
                tokens = getattr(getattr(model_response, "usage", None), "input_tokens", None)
                if tokens:
                    observe_llm_prompt_tokens(self.guild_id, tokens)
        except Exception as e:
            logger.warning('Could not record prompt token usage: ' + str(e))

    def _capture_reasoning(self, items, final_output):
        """Store this run's reasoning on self.reasoning (never raises).

        Reasoning comes back one of two ways depending on the server's
        --reasoning-format: out of band as reasoning_content (our llama.cpp
        default, and the only source that survives into the run items), or
        inline as think tags in the answer itself. Try the run items first
        and fall back to the tags.

        Fully guarded: reasoning is a nice-to-have, so a surprise in the
        run-item shape must not turn a good answer into the "Error" sentinel
        (nor mask the real failure on the error path).
        """
        try:
            self.reasoning = extract_reasoning_items(items)
            if not self.reasoning and isinstance(final_output, str):
                self.reasoning = extract_thinking(final_output)
        except Exception as e:
            logger.warning('Could not extract reasoning from the response: ' + str(e))
            self.reasoning = ""
        logger.info(f'Reasoning captured: {len(self.reasoning)} chars')
      
  

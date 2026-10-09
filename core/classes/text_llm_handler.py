import asyncio
import logging
import math
import os,aiohttp, discord, io, time
from classes.user_memory import UserMemory
from classes.metrics import (
    inc_llm_error,
    inc_tool_call,
    inc_tool_error,
    observe_llm_call,
    observe_llm_completion_tokens,
    observe_llm_prompt_tokens,
    observe_tool_duration,
    set_slot_context,
)
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, FunctionTool, function_tool, RunContextWrapper, ModelSettings, RunHooks
from agents.models.interface import ModelTracing
from openai import BadRequestError, UnprocessableEntityError
from openai.types.responses import ResponseCreatedEvent, ResponseTextDeltaEvent
from classes.config_manager import configManager
from classes.response_filter import extract_reasoning_items, extract_thinking
from classes.llm_config import llm_api_key, llm_host, llm_model, parse_temperature
from classes.reply_policy import INSTRUCTION as DOUBLE_REPLY_INSTRUCTION
from classes.side_llm import NO_THINKING
from classes import voice_policy

# Max turns for ONE reply from the main agent (a turn = one model response,
# however many tool calls it carries). The SDK's own default is 10, which a
# chained tool run overruns easily - three sandbox calls plus the reasoning
# turns around them is already most of it - and overrunning raises
# MaxTurnsExceeded, which costs the user the whole answer.
DEFAULT_LLM_MAX_TURNS = 20


def llm_max_turns() -> int:
    """Max model turns for one reply (LLM_MAX_TURNS, default 20)."""
    try:
        value = int(str(os.environ.get("LLM_MAX_TURNS")).strip())
        return value if value > 0 else DEFAULT_LLM_MAX_TURNS
    except (TypeError, ValueError):
        return DEFAULT_LLM_MAX_TURNS


# Wall-clock seconds ONE model call (one turn's request, the OpenAI client's
# own retries included) may take before the run is abandoned. Applied as
# ModelSettings.timeout, which the SDK's runner enforces by cancelling the
# call. The client's own timeout is a per-read timeout: it catches a server
# that goes silent, but a server that keeps the connection warm while it
# works (OpenRouter pads non-streaming responses) resets it forever, and
# until now nothing bounded that at all. The default matches the client's
# 600s so a slow local generation is no more likely to be cut off than
# before; it only closes the unbounded case.
DEFAULT_LLM_CALL_TIMEOUT = 600.0


def llm_call_timeout() -> float:
    """Wall-clock seconds per model call (LLM_CALL_TIMEOUT_SECONDS, default 600)."""
    try:
        value = float(str(os.environ.get("LLM_CALL_TIMEOUT_SECONDS")).strip())
        # ModelSettings.timeout is validated as a finite positive float.
        return value if value > 0 and math.isfinite(value) else DEFAULT_LLM_CALL_TIMEOUT
    except (TypeError, ValueError):
        return DEFAULT_LLM_CALL_TIMEOUT



from classes.image_generation import image_generation_enabled

from classes.message_queue import (
    SLOW_TOOL_NAMES,
    TOOL_DISPLAY,
    register_task_run,
    unregister_task_run,
)

from classes.sandbox_agent import sandbox_enabled

from classes.tool_functions import (
    change_personality,
    check_polls,
    clear_memories,
    create_poll,
    fetch_url,
    generate_image,
    get_current_datetime,
    remove_memory,
    run_code_sandbox,
    store_memory,
    web_search,
)

logger = logging.getLogger(__name__)


# The LLM host/model/key never change at runtime, so the AsyncOpenAI client
# (its own httpx connection pool) is built once and reused across every
# message instead of per-message. Safe without locking: no `await` between
# the check and the set, so no race is possible even with concurrent worker
# tasks (the event loop is single-threaded).
_main_model_client = None


def _get_main_model_client() -> OpenAIChatCompletionsModel:
    global _main_model_client
    if _main_model_client is None:
        _main_model_client = OpenAIChatCompletionsModel(
            model=llm_model(),
            openai_client=AsyncOpenAI(
                base_url=llm_host() + "/v1",
                api_key=llm_api_key(),
            ),
        )
    return _main_model_client


# Per-slot context of the LLM server (llama.cpp /props), refreshed by main.py.
# None until the first successful read: the history token budget then does
# not trim at all rather than guessing a capacity.
_slot_context: int | None = None


def slot_context_tokens() -> int | None:
    """The last per-slot context read from the LLM server, or None."""
    return _slot_context


def parse_slot_context(props) -> int | None:
    """Per-slot context from a llama.cpp /props body, or None.

    default_generation_settings.n_ctx is the SLOT context (the server's
    --ctx-size divided across --parallel slots, rounded up to its padding), not
    the configured total - the number one request can actually fill.
    """
    try:
        value = int(props["default_generation_settings"]["n_ctx"])
    except (KeyError, TypeError, ValueError):
        return None
    return value if value > 0 else None


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
    that the older one is still being processed. A run that ends in the SDK
    failure prefix is unregistered WITHOUT the recently-done note — the user
    didn't receive a result to refer to.
    """

    _SDK_FAILURE_PREFIX = "An error occurred while running the tool"
    # The tool argument that identifies what a slow tool is doing (shown,
    # truncated, in the in-flight hint).
    _SOURCE_FIELDS = {
        "run_code_sandbox": "task",
        "generate_image": "prompt",
    }

    def __init__(self, guild_id):
        self.guild_id = guild_id
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
                observe_llm_call("main", "ok", time.monotonic() - started)
            usage = getattr(response, "usage", None)
            observe_llm_completion_tokens("main", getattr(usage, "output_tokens", None))
        except Exception as e:
            logger.warning(f"Metrics llm_end hook failed: {e}")

    def abandon_llm_call(self, outcome: str) -> None:
        """Record the request in progress, if any, as ended with `outcome`."""
        try:
            started, self._llm_started = self._llm_started, None
            if started is not None:
                observe_llm_call("main", outcome, time.monotonic() - started)
        except Exception as e:
            logger.warning(f"Metrics llm abandon failed: {e}")

    def _key(self, context, tool) -> str:
        # ToolContext carries a tool_call_id; several concurrent calls of the
        # same tool are distinguished by it. Fall back to the tool name.
        return getattr(context, "tool_call_id", None) or getattr(tool, "name", "unknown")

    def _tool_name(self, context, tool) -> str:
        return getattr(context, "tool_name", None) or getattr(tool, "name", "unknown")

    def _task_source(self, context, name: str) -> str:
        """The tool argument shown in the in-flight hint (task/prompt)."""
        raw = getattr(context, "tool_input", None)
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

    # Hook failures must never abort the run (same rule as sandbox_progress).

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
                failed = isinstance(result, str) and result.startswith(self._SDK_FAILURE_PREFIX)
                unregister_task_run(
                    self._channel_id(context),
                    run_key=self._key(context, tool),
                    # No "just finished" note when the run failed — the user
                    # didn't receive a result to refer to.
                    finish=not failed,
                )
            except Exception as e:
                logger.warning(f"In-flight registry tool_end failed: {e}")


# Whether a voice turn asks the server to skip thinking. Cleared for the
# process once a backend has rejected the option (see generate_streamed).
_voice_latch = {"send_no_thinking": True}


def voice_thinking_off() -> bool:
    return not voice_policy.settings()["thinking"] and _voice_latch["send_no_thinking"]


def agent_tools(automatic: bool = False, voice: bool = False) -> list:
    """The function tools the main agent gets. A function of its own so the
    /help sync test (core/tests/help_catalog_tests.py) can list them.

    A voice turn gets the same tools as a chat reply, except that it can
    leave the call instead of joining one."""
    tools = [
        web_search,
        fetch_url,
        change_personality,
        create_poll,
        check_polls,
    ]
    if not automatic:
        tools.extend((store_memory, remove_memory, clear_memories))
        from classes.automation_tools import automation_tools
        tools.extend(automation_tools())
    if not automatic and voice_policy.settings()["enabled"]:
        from classes.tool_functions import join_voice_channel, leave_voice_channel
        tools.append(leave_voice_channel if voice else join_voice_channel)
    # The image tool only exists when the diffusion service is enabled
    # (IMAGE_GEN_ENABLED; set from the helm chart's diffusion.enabled).
    if image_generation_enabled():
        tools.append(generate_image)

    # Sandbox tool (nested SandboxAgent in a throwaway Docker container);
    # needs the Docker socket mounted (SANDBOX_ENABLED; chart sandbox.enabled).
    if sandbox_enabled():
        tools.append(run_code_sandbox)
    return tools


class TextLLMHandler:
    # Set by MessageHandler after eligibility is chosen under the build lock.
    allow_double_reply = False

    def __init__(self, messages, guild_id, original_message, client=None,
                 actor_id=None, channel=None, automatic=False, voice=False, request_text=None,
                 bot_name=None):
        self.original_message = original_message
        self.messages = messages
        self.guild_id = guild_id
        # The discord.Client, forwarded to tool-run context as
        # "discord_client" (see generate()) and on into the nested sandbox
        # run's context. Nothing there waits on it any more (ask_user's
        # replies arrive through the thread ledger), so it is optional/None
        # for callers and tests.
        self.client = client
        self.config = configManager()
        self.actor_id = actor_id if actor_id is not None else original_message.author.id
        self.channel = channel if channel is not None else original_message.channel
        self.automatic = automatic
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
        # Filled in by generate() from the shared run context: the Discord
        # thread run_code_sandbox resolved/created for this run, if any. The
        # caller (MessageHandler) sends the final reply there instead of
        # self.message.channel so it lands next to the sandbox's own output
        # rather than outside the thread.
        self.sandbox_thread = None


    @staticmethod
    async def check_model_ready(model: str):
        # llama.cpp has no pull endpoint: the llamacpp container downloads the model
        # on boot (LLAMA_ARG_HF_REPO into the LLAMA_CACHE volume). We only check that
        # the configured model is loaded (fail-soft: it may still be downloading).
        url = llm_host() + "/v1/models"
        logger.info(f"Checking model {model} on LLM host ({url})")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        logger.warning(f"LLM server not ready yet ({response.status}); model may still be downloading")
                        return
                    data = await response.json()
                    available = [m.get("name") or m.get("id") for m in data.get("models", [])]
                    if model in available:
                        logger.info(f"Model {model} is available")
                    else:
                        logger.warning(f"Model {model} not loaded yet (server has: {available})")
        except Exception as e:
            logger.warning(f"Could not reach LLM server at {url}: {e}")
      
    @staticmethod
    async def refresh_slot_context() -> int | None:
        """Re-read the per-slot context from {LLM_HOST}/props (never raises).

        On failure the last value is kept for the token budget but the gauge
        is marked unavailable, so a dashboard can tell stale from measured.
        """
        global _slot_context
        url = llm_host() + "/props"
        value = None
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        value = parse_slot_context(await response.json())
                    else:
                        logger.info(f"LLM server /props returned {response.status}")
        except Exception as e:
            logger.info(f"Could not read LLM server /props at {url}: {e}")
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
        self.model = llm_model()
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
        tools = agent_tools(getattr(self, "automatic", False), voice)
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
        )

    def _model_settings(self, no_thinking: bool | None = None) -> ModelSettings:
        """`no_thinking` overrides the voice latch for one request (the
        retry without the option, before it is known to work)."""
        if no_thinking is None:
            no_thinking = voice_thinking_off()
        settings = dict(
            temperature=self.options["temperature"],
            frequency_penalty=1.1,
            top_p=1.0,
            timeout=llm_call_timeout(),
        )
        if getattr(self, "voice", False) and no_thinking:
            # Off, not merely "low": a spoken reply waits for every reasoning
            # token before its first word. Same switch the image side calls
            # use (side_llm.NO_THINKING).
            settings["extra_body"] = NO_THINKING
        else:
            settings["reasoning"] = {"effort": os.environ.get("REASONING_EFFORT", "medium")}
        return ModelSettings(**settings)

    async def _prepare(self) -> dict:
      """Settings, agent and run context: everything generate(),
      generate_streamed() and prefill() share. Built in exactly one place so
      a voice prefill sends the same prompt prefix as the turn after it."""
      await self.get_settings()
      user_info = {
        "data": await self.user_memory.get() or [] if getattr(self, "user_memory", None) else [],
        "user_id": self.actor_id if hasattr(self, "actor_id") else self.original_message.author.id,
        "guild_id": self.guild_id,
        "original_message": self.original_message,
        "channel": self.channel if hasattr(self, "channel") else self.original_message.channel,
        "automatic": getattr(self, "automatic", False),
        "discord_client": self.client,
        "redis_save_tool_calls": 0,
        "personality_tool_calls": 0,
        "poll_tool_calls": 0,
        # Set by run_code_sandbox (tool_functions.py) if/when it runs, to the
        # thread it resolved/created — read back below regardless of how the
        # run ends, since a tool call may have already mutated this dict
        # before a later turn raises (e.g. MaxTurnsExceeded).
        "sandbox_thread": None,
        # What a voice speaker said (no source message to read it from).
        "request_text": getattr(self, "request_text", None),
      }
      self.system = f"Answer as if you are {self.system}."
      await self.get_client()
      return user_info

    async def generate(self):
      user_info = await self._prepare()
      datetime = await get_current_datetime()
      # The datetime is appended as a trailing message rather than folded
      # into the system prompt: the system prompt + growing history stays
      # byte-identical across turns of the same conversation, so llama.cpp's
      # prefix cache only has to prefill the new tail instead of the whole
      # prompt every time. role="user" (not "system"): some chat templates
      # (e.g. this model's) raise a Jinja error ("System message must be at
      # the beginning") for any system message that isn't the very first one.
      messages_for_run = self.messages + [
          {"role": "user", "content": f"(Current datetime: {datetime})"}
      ]
      if self.allow_double_reply and not getattr(self, "automatic", False):
          messages_for_run.append({"role": "user", "content": DOUBLE_REPLY_INSTRUCTION})
      hooks = ToolMetricsHooks(self.guild_id)
      try:
         response = await Runner.run(self.agent, messages_for_run, context=user_info,
                                     max_turns=llm_max_turns(),
                                     hooks=hooks)
         logger.info('Response generated')
         logger.debug(response)
         final_output = response.final_output
         self._capture_reasoning(response.new_items, final_output)
         self._record_prompt_tokens(response.raw_responses)
         self.sandbox_thread = user_info.get("sandbox_thread")
         return final_output
      except asyncio.CancelledError:
         hooks.abandon_llm_call("cancelled")
         raise
      except Exception as e:
         hooks.abandon_llm_call("error")
         self.sandbox_thread = user_info.get("sandbox_thread")
         logger.warning('Failed to get response from LLM: ' + str(e))
         # A run that died part-way (MaxTurnsExceeded after a few chained
         # tool calls is the common one) still reasoned before it broke, and
         # its tools have already posted their embeds/files to the channel.
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
      # Same trailing datetime as generate(): everything before it is the
      # prompt prefix a voice prefill warmed (see prefill()).
      messages_for_run = self.messages + [
          {"role": "user", "content": f"(Current datetime: {datetime})"}
      ]
      hooks = hooks or ToolMetricsHooks(self.guild_id)
      emitted = False
      retried_without_option = False
      for attempt in (1, 2):
        sent_no_thinking = self.agent.model_settings.extra_body is not None
        try:
          result = Runner.run_streamed(self.agent, messages_for_run, context=user_info,
                                       max_turns=llm_max_turns(), hooks=hooks)
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
          self.sandbox_thread = user_info.get("sandbox_thread")
          if retried_without_option:
            # Only now is the option known to be what the server refused;
            # a retry that failed too says nothing about it.
            _voice_latch["send_no_thinking"] = False
          return final_output if isinstance(final_output, str) else str(final_output or "")
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

    def _streamed_failure(self, hooks, user_info, e) -> str:
      hooks.abandon_llm_call("error")
      self.sandbox_thread = user_info.get("sandbox_thread")
      logger.warning('Failed to get a streamed response from LLM: ' + str(e))
      inc_llm_error(self.guild_id)
      return "Error"

    async def prefill(self) -> None:
      """Warm llama.cpp's prompt cache with this conversation, so the next
      voice turn only has to process its tail.

      Sends exactly what generate_streamed() would send for these messages --
      same instructions, tools and model settings, built by the same
      _prepare() -- minus the trailing datetime message, and asks for one
      token. llama.cpp keeps the processed prompt in its slot; the real turn
      is that prompt plus a few tokens, so it is answered almost at once.
      Measured on the dev stack: a 2.4k-token voice prompt went from 1.0s of
      prompt processing to 0.3s, 52 tokens instead of 2427.
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
      
  

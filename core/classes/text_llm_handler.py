import os,aiohttp, discord, io, time
from classes.user_memory import UserMemory
from classes.metrics import inc_llm_error, inc_tool_call, inc_tool_error, observe_tool_duration
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, FunctionTool, function_tool, RunContextWrapper, ModelSettings, RunHooks
from classes.config_manager import configManager
from classes.response_filter import extract_reasoning_items, extract_thinking
from classes.llm_config import llm_api_key, llm_host, llm_model, parse_temperature

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



from classes.image_generation import image_generation_enabled

from classes.message_queue import (
    SLOW_TOOL_NAMES,
    TOOL_DISPLAY,
    register_task_run,
    unregister_task_run,
)

from classes.sandbox_agent import sandbox_enabled

from classes.tool_functions import *


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
        return context.context.get("original_message").channel.id

    # Hook failures must never abort the run (same rule as sandbox_progress).

    async def on_tool_start(self, context, agent, tool) -> None:
        name = self._tool_name(context, tool)
        try:
            self._starts[self._key(context, tool)] = (time.monotonic(), name)
        except Exception as e:
            print(f"Metrics tool_start hook failed: {e}")
        if name in SLOW_TOOL_NAMES:
            try:
                register_task_run(
                    self._channel_id(context),
                    TOOL_DISPLAY.get(name, name),
                    self._task_source(context, name),
                    run_key=self._key(context, tool),
                )
            except Exception as e:
                print(f"In-flight registry tool_start failed: {e}")

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
            print(f"Metrics tool_end hook failed: {e}")
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
                print(f"In-flight registry tool_end failed: {e}")

class TextLLMHandler:

    def __init__(self, messages, guild_id, original_message, client=None):
        self.original_message = original_message
        self.messages = messages
        self.guild_id = guild_id
        # The discord.Client, forwarded to tool-run context as
        # "discord_client" (see generate()) — needed by run_code_sandbox's
        # ask_user tool for client.wait_for(). Optional/None for callers
        # (and tests) that don't need sandbox HITL.
        self.client = client
        self.config = configManager()
        self.user_memory = UserMemory(original_message.author.id, guild_id)
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
        print(f"Checking model {model} on LLM host ({url})")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        print(f"LLM server not ready yet ({response.status}); model may still be downloading")
                        return
                    data = await response.json()
                    available = [m.get("name") or m.get("id") for m in data.get("models", [])]
                    if model in available:
                        print(f"Model {model} is available")
                    else:
                        print(f"Model {model} not loaded yet (server has: {available})")
        except Exception as e:
            print(f"Could not reach LLM server at {url}: {e}")
      
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
        tools = [
            web_search,
            fetch_url,
            store_memory,
            remove_memory,
            clear_memories,
            change_personality,
        ]
        # The image tool only exists when the diffusion service is enabled
        # (IMAGE_GEN_ENABLED; set from the helm chart's diffusion.enabled).
        if image_generation_enabled():
            tools.append(generate_image)

        # Sandbox tool (nested SandboxAgent in a throwaway Docker container);
        # needs the Docker socket mounted (SANDBOX_ENABLED; chart sandbox.enabled).
        if sandbox_enabled():
            tools.append(run_code_sandbox)

        self.agent = Agent(
            name="Assistant",
            instructions=self.system,
            model=main_model_client,
            tools=tools,
            model_settings=ModelSettings(
                temperature=self.options["temperature"],
                frequency_penalty=1.1,
                top_p=1.0,
                reasoning={"effort": os.environ.get("REASONING_EFFORT", "medium")}
            ),
        )

    async def generate(self):
      await self.get_settings()
      user_info = {
        "data": await self.user_memory.get() or [],
        "user_id": self.original_message.author.id,
        "guild_id": self.guild_id,
        "original_message": self.original_message,
        "discord_client": self.client,
        "redis_save_tool_calls": 0,
        "personality_tool_calls": 0,
        # Set by run_code_sandbox (tool_functions.py) if/when it runs, to the
        # thread it resolved/created — read back below regardless of how the
        # run ends, since a tool call may have already mutated this dict
        # before a later turn raises (e.g. MaxTurnsExceeded).
        "sandbox_thread": None,
      }
      datetime = await get_current_datetime()
      self.system = f"Answer as if you are {self.system}."
      await self.get_client()
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
      try:
         response = await Runner.run(self.agent, messages_for_run, context=user_info,
                                     max_turns=llm_max_turns(),
                                     hooks=ToolMetricsHooks(self.guild_id))
         print(f'Response generated')
         print(response)
         final_output = response.final_output
         self._capture_reasoning(response.new_items, final_output)
         self.sandbox_thread = user_info.get("sandbox_thread")
         return final_output
      except Exception as e:
         self.sandbox_thread = user_info.get("sandbox_thread")
         print('Failed to get response from LLM: ' + str(e))
         # A run that died part-way (MaxTurnsExceeded after a few chained
         # tool calls is the common one) still reasoned before it broke, and
         # its tools have already posted their embeds/files to the channel.
         # The SDK hangs the completed turns off the exception as
         # RunErrorDetails, so recover the reasoning from there rather than
         # leaving the user with a bare ❌ and no idea what happened.
         run_data = getattr(e, "run_data", None)
         self._capture_reasoning(getattr(run_data, "new_items", None), None)
         inc_llm_error(self.guild_id)
         return "Error"

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
            print('Could not extract reasoning from the response: ' + str(e))
            self.reasoning = ""
        print(f'Reasoning captured: {len(self.reasoning)} chars')
      
  

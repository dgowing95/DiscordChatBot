import os,aiohttp, discord, io, time
from classes.user_memory import UserMemory
from classes.metrics import inc_llm_error, inc_tool_call, inc_tool_error, observe_tool_duration
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, FunctionTool, function_tool, RunContextWrapper, ModelSettings, RunHooks
from classes.config_manager import configManager



from classes.image_generation import image_generation_enabled

from classes.sandbox_agent import sandbox_enabled

from classes.tool_functions import *


class ToolMetricsHooks(RunHooks):
    """RunHooks that record per-tool call counts, durations and failures.

    Attached to Runner.run instead of wrapping every tool. Tool-level error
    detection relies on the SDK's default failure handler: the tools in
    tool_functions.py catch their own exceptions and return friendly strings,
    so only unexpected tool exceptions reach the SDK handler (whose default
    result starts with the prefix below).
    """

    _SDK_FAILURE_PREFIX = "An error occurred while running the tool"

    def __init__(self, guild_id):
        self.guild_id = guild_id
        self._starts: dict[str, tuple[float, str]] = {}

    def _key(self, context, tool) -> str:
        # ToolContext carries a tool_call_id; several concurrent calls of the
        # same tool are distinguished by it. Fall back to the tool name.
        return getattr(context, "tool_call_id", None) or getattr(tool, "name", "unknown")

    def _tool_name(self, context, tool) -> str:
        return getattr(context, "tool_name", None) or getattr(tool, "name", "unknown")

    # Hook failures must never abort the run (same rule as sandbox_progress).

    async def on_tool_start(self, context, agent, tool) -> None:
        try:
            self._starts[self._key(context, tool)] = (time.monotonic(), self._tool_name(context, tool))
        except Exception as e:
            print(f"Metrics tool_start hook failed: {e}")

    async def on_tool_end(self, context, agent, tool, result) -> None:
        try:
            name = self._tool_name(context, tool)
            started = self._starts.pop(self._key(context, tool), None)
            inc_tool_call(name, self.guild_id)
            if started is not None:
                observe_tool_duration(name, self.guild_id, time.monotonic() - started[0])
            if isinstance(result, str) and result.startswith(self._SDK_FAILURE_PREFIX):
                inc_tool_error(name, self.guild_id)
        except Exception as e:
            print(f"Metrics tool_end hook failed: {e}")

class TextLLMHandler:

    def __init__(self, messages, guild_id, original_message, attachment_refs=None):
        self.original_message = original_message
        self.messages = messages
        self.guild_id = guild_id
        # Short labels -> real CDN URLs for attached images (see
        # MessageHandler._collect_attachment_refs). The edit_image tool
        # resolves a label here instead of the LLM copying a full signed URL.
        self.attachment_refs = attachment_refs or []
        self.config = configManager()
        self.user_memory = UserMemory(original_message.author.id, guild_id)
        self.get_settings()


    @staticmethod
    async def check_model_ready(model: str):
        # llama.cpp has no pull endpoint: the llamacpp container downloads the model
        # on boot (LLAMA_ARG_HF_REPO into the LLAMA_CACHE volume). We only check that
        # the configured model is loaded (fail-soft: it may still be downloading).
        url = os.environ.get("LLM_HOST", "http://llamacpp:8080") + "/v1/models"
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
      
    def get_settings(self):
        self.system = self.config.get_setting("system", self.guild_id) or "An AI Story Teller"
        self.model = os.environ.get("MODEL", "qwen3:4b")
        self.options = {
         "temperature": float(self.config.get_setting("temperature", self.guild_id)) or 1.0
        }

    async def get_client(self):
        main_model_client = OpenAIChatCompletionsModel(
            model=self.model,
            openai_client=AsyncOpenAI(
                base_url=os.environ.get("LLM_HOST", "http://llamacpp:8080") + "/v1",
                api_key=os.environ.get("LLM_PASS", "ollama")
            )
        )
        tools = [
            web_search,
            fetch_url,
            fetch_weather,
            store_memory,
            remove_memory,
            clear_memories,
            change_personality,
        ]
        # Image tools only exist when the diffusion service is enabled
        # (IMAGE_GEN_ENABLED; set from the helm chart's diffusion.enabled).
        if image_generation_enabled():
            tools.extend([generate_image, edit_image])

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
                reasoning={"effort": "low"}
            ),
        )

    async def generate(self):
      user_info = {
        "data": self.user_memory.get() or [],
        "user_id": self.original_message.author.id,
        "guild_id": self.guild_id,
        "original_message": self.original_message,
        "attachment_refs": self.attachment_refs,
        "redis_save_tool_calls": 0,
        "personality_tool_calls": 0,
      }
      datetime = await get_current_datetime()
      self.system = f"""
        The current datetime is {datetime}.
        Answer as if you are {self.system}.
      """
      await self.get_client()
      try:
         response = await Runner.run(self.agent, self.messages, context=user_info,
                                     hooks=ToolMetricsHooks(self.guild_id))
         print(f'Response generated')
         print(response)
         return response.final_output
      except Exception as e:
         print('Failed to get response from LLM: ' + str(e))
         inc_llm_error(self.guild_id)
         return "Error"
      
  

import os,aiohttp, discord, io
from classes.user_memory import UserMemory

from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, FunctionTool, function_tool, RunContextWrapper, ModelSettings
from classes.config_manager import configManager



from classes.tool_functions import *

class TextLLMHandler:

    def __init__(self, messages, guild_id, original_message):
        self.original_message = original_message
        self.messages = messages
        self.guild_id = guild_id
        self.config = configManager()
        self.user_memory = UserMemory(original_message.author.id, guild_id)
        self.get_settings()


    @staticmethod
    async def pull_model(model: str):
        print(f"Pulling model {model} from LLM host")

        payload = {"model": model, "stream": False}
        url = os.environ.get("LLM_HOST", "http://ollama:11434") + "/api/pull"

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    print(f"Failed to pull model {model}: {response.status}")
                    return
                
                data = await response.json()
                if data.get("error"):
                    print(f"Error pulling model {model}: {data['error']}")
                    return    
                print(f"Model {model} pulled successfully")
      
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
                base_url=os.environ.get("LLM_HOST", "http://ollama:11434") + "/v1",
                api_key=os.environ.get("LLM_PASS", "ollama")
            )
        )
        self.agent = Agent(
            name="Assistant",
            instructions=self.system,
            model=main_model_client,
            tools=[web_search, fetch_url, store_memory, change_personality, remove_memory, clear_memories],
            model_settings=ModelSettings(
                temperature=self.options["temperature"],
                frequency_penalty=1.1,
                top_p=1.0,
                max_tokens=5000
            ),
        )

    async def generate(self):
      user_info = {
        "data": self.user_memory.get() or [],
        "user_id": self.original_message.author.id,
        "guild_id": self.guild_id,
        "original_message": self.original_message,
        "redis_save_tool_calls": 0,
        "personality_tool_calls": 0
      }
      user_data_formatted = "\n".join(f"- {item}" for item in user_info["data"])
      datetime = await get_current_datetime()
      self.system = f"""
        Answer as if you are {self.system}
        Answer the most recent message only. Do not answer previous messages.
        The current datetime is {datetime}
        You know the following information about the user, but do not have to use it in your response:
        {user_data_formatted}
      """
      await self.get_client()
      try:
         response = await Runner.run(self.agent, self.messages, context=user_info)
         print(f'Response generated')
         print(response)
         return response.final_output
      except Exception as e:
         print('Failed to get response from LLM: ' + str(e))
         return "Error"
      
  

import os,aiohttp, discord, io

from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, FunctionTool, function_tool, RunContextWrapper
from classes.config_manager import configManager



from classes.tool_functions import *

class TextLLMHandler:

    def __init__(self, messages, guild_id, original_message):
        self.original_message = original_message
        self.messages = messages
        self.guild_id = guild_id
        self.config = configManager()
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
        self.model = os.environ.get("MODEL", "gemma3:4b")
        self.options = {
         "temperature": float(self.config.get_setting("temperature", self.guild_id)) or 1.0
        }

    async def get_client(self):
        tool_model_client = OpenAIChatCompletionsModel(
            model="qwen3:4b",
            openai_client=AsyncOpenAI(
                base_url=os.environ.get("LLM_HOST", "http://ollama:11434") + "/v1",
                api_key=os.environ.get("LLM_PASS", "ollama")
            )
        )
        main_model_client = OpenAIChatCompletionsModel(
            model=self.model,
            openai_client=AsyncOpenAI(
                base_url=os.environ.get("LLM_HOST", "http://ollama:11434") + "/v1",
                api_key=os.environ.get("LLM_PASS", "ollama")
            )
        )
        main_agent = Agent(
            name="Main Agent",
            instructions=self.system + ". Answer the most recent question in the conversation only.",
            handoff_description="Use this for responding to messages.",
            model=main_model_client,
        )
        tool_agent = Agent(
            name="Tool Agent",
            instructions="Use this if the prompt requires searching the internet.",
            model=tool_model_client,
            tools=[web_search, fetch_url]
        )
        self.agent = Agent(
            name="Assistant",
            instructions="You must transfer to the Main Agent or the Tool Agent",
            model=tool_model_client,
            handoffs=[main_agent,tool_agent]
        )

    async def generate(self):
      await self.get_client()
      discord_info = {
         'channel': self.original_message.channel,
         'user': self.original_message.author,
      }
      try:
         response = await Runner.run(self.agent, self.messages, context=discord_info)
         print(f'Response generated')
         print(response)
         return response.final_output
      except Exception as e:
         print('Failed to get response from LLM: ' + str(e))
         return "Error"
      
  

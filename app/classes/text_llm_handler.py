import os, asyncio, json, aiohttp
from openai import AsyncOpenAI
from urllib.parse import urlparse
from classes.config_manager import configManager

class TextLLMHandler:

    def __init__(self, messages, guild_id):
        self.messages = messages
        self.guild_id = guild_id
        self.config = configManager()
        self.get_settings()


    @staticmethod
    async def pull_model():
        model = os.environ.get("MODEL", "gemma3:4b")
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
      self.client = client = AsyncOpenAI(
          base_url=os.environ.get("LLM_HOST", "http://ollama:11434") + "/v1",
          api_key=os.environ.get("LLM_PASS", "ollama")
      )

    async def generate(self):
      await self.get_client()
      system = self.system + ". Reply with only your message, no prefixes or titles."
      msgs = [
           {"role": "system", "content": system + " /no_think"},
            *self.messages,
      ]

      try:
         response_stream = await self.client.chat.completions.create(
            model=self.model,
            messages=msgs,
            temperature=self.options["temperature"],
            stream=True
          )
         print(f'Stream response generated')
         return response_stream
      except Exception as e:
         print('Failed to get response from LLM: ' + str(e))
         return "Error"
      

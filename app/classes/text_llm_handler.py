import os, asyncio, json
from openai import AsyncOpenAI
from urllib.parse import urlparse
from classes.config_manager import configManager

class TextLLMHandler:

    def __init__(self, messages, guild_id):
        self.messages = messages
        self.guild_id = guild_id
        self.config = configManager()
        self.get_settings()

	
    def get_settings(self):
        self.system = self.config.get_setting("system", self.guild_id) or "An AI Story Teller"
        self.model = os.environ.get("MODEL", "gemma3:4b")
        self.options = {
         "temperature": float(self.config.get_setting("temperature", self.guild_id)) or 1.0
        }


    async def test_connection(self, host, port):
      try:
        _reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=5)
        writer.close()
        await writer.wait_closed()
        return True
      except:
        return False

    async def try_clients(self):
      client_configs = json.loads(os.environ.get("LLM_HOSTS", "[]"))
      print(client_configs)
      for llm_host in client_configs:
         print(f"Trying LLM host: {llm_host['base_url']}")
         url_components = urlparse(llm_host['base_url'])
         if await self.test_connection(url_components.hostname, url_components.port):
            return llm_host
        
      raise Exception("No LLM hosts available")

    async def get_client(self):
      valid_client = await self.try_clients()
      self.client = client = AsyncOpenAI(
          base_url=valid_client['base_url'],
          api_key=valid_client['api_key']
      )



    async def generate(self):
      await self.get_client()
      system = self.system + ". Reply with only your message, no prefixes or titles."
      msgs = [
           {"role": "system", "content": system},
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
      

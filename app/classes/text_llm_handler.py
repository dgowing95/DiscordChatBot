import os
from openai import AsyncOpenAI
from classes.config_manager import configManager

class TextLLMHandler:

    def __init__(self, messages, guild_id):
        self.messages = messages
        self.guild_id = guild_id
        self.config = configManager()
        self.get_settings()
        self.llm_text_host_url = os.environ['LLM_TEXT_HOST']
        self.llm_text_api_token = os.environ['TEXT_API_TOKEN']
        self.get_client()

	
    def get_settings(self):
        self.system = self.config.get_setting("system", self.guild_id) or "An AI Story Teller"
        self.model = "gemma3:12b"
        self.options = {
         "temperature": float(self.config.get_setting("temperature", self.guild_id)) or 1.0
        }

    def get_client(self):
       try:
         self.client = AsyncOpenAI(
            base_url=self.llm_text_host_url,
            api_key=self.llm_text_api_token
          )
       except Exception as e:
         print('Failed to connect to LLM Host: ' + str(e))


    async def generate(self):
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
         print(f'Message returned from Text LLM')
         return response_stream
      except Exception as e:
         print('Failed to get response from LLM: ' + str(e))
         return "Error"
      

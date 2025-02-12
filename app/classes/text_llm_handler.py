import re, json
from openai import AsyncOpenAI
from classes.config_manager import configManager

class TextLLMHandler:

    def __init__(self, messages):
        self.messages = messages
        self.system = configManager().get_setting("system")
        self.model = configManager().get_setting("model")
        self.llm_text_host_url = configManager().get_setting("llm_text_host")
        self.options = {
         "temperature": configManager().get_setting("temperature")
        }
        self.get_client()

    def get_client(self):
       try:
         self.client = AsyncOpenAI(
            base_url=self.llm_text_host_url,
            api_key=configManager().get_setting("text_api_token")
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
      

import re, json
from ollama import AsyncClient
from ollama import ChatResponse
from classes.config_manager import configManager

class ollamaHandler:

    def __init__(self, messages):
        self.messages = messages
        self.system = configManager().get_setting("system")
        self.model = configManager().get_setting("model")
        self.ollama_host = configManager().get_setting("ollama_host")
        self.options = {
         "temperature": configManager().get_setting("temperature")
        }
        self.get_client()

    def get_client(self):
       try:
         self.client = AsyncClient(host=self.ollama_host)
       except:
         print('Failed to connect to Ollama Host')


    async def generate(self):
      msgs = [
           {"role": "system", "content": self.system},
            *self.messages,
      ]

      print(json.dumps(msgs, indent=2))

      try:
         response: ChatResponse = await self.client.chat(model=self.model, messages=msgs, options=self.options)
         print(f'Message returned from Ollama')

         text = response['message']['content']
         cleaned_response = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
         cleaned_response = re.sub(r"^.*:", "", text, flags=re.DOTALL)
         return cleaned_response[0:1999]
      except Exception as e:
         print('Failed to get response from Ollama: ' + str(e))
         return "Error"
      

import re
from ollama import AsyncClient
from ollama import ChatResponse
from classes.config_manager import configManager

class ollamaHandler:

    def __init__(self, history, prompt, client_id):
        self.history = history
        self.prompt = prompt
        self.client_id = client_id
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

    def format_message_history(self):

       self.history.pop(0) # Remove current message
       self.history.reverse() # Reverse the order of the messages so the newest is first

       formatted_history = []
       for message in self.history:
          if len(message.content) == 0:
             continue
          
          formatted_history.append({
             'role': "assistant" if message.author.id == self.client_id else "user",
             'content': message.content
          })
       self.message_history = formatted_history
          

    async def generate(self):
      self.format_message_history()

      msgs = [
           {"role": "system", "content": self.system},
            *self.message_history,
           {
                'role': 'user',
                'content': self.prompt
           }
      ]
      try:
         response: ChatResponse = await self.client.chat(model=self.model, messages=msgs, options=self.options)
         print(f'Message returned from Ollama')

         text = response['message']['content']
         cleaned_response = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
         return cleaned_response[0:1999]
      except Exception as e:
         print('Failed to get response from Ollama: ' + str(e))
         return "Error"
      

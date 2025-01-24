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
        self.get_client()

    def get_client(self):
       self.client = AsyncClient(host=self.ollama_host)

    def format_message_history(self):

       self.history.pop(0) # Remove current message
       self.history.reverse() 

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

      response: ChatResponse = await self.client.chat(model=self.model, messages=msgs)
      print(f'Message returned')
      return response['message']['content']
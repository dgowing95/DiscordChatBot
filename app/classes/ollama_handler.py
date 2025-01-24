from ollama import chat
from ollama import ChatResponse
from classes.config_manager import configManager

class ollamaHandler:
    history = []
    message_history = []
    system = ""
    model = ""
    prompt = ""
    client_id = 0

    def __init__(self, history, prompt, client_id):
        self.history = history
        self.prompt = prompt
        self.client_id = client_id
        self.system = configManager().get_setting("system")
        self.model = configManager().get_setting("model")

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

      print(msgs)

      response: ChatResponse = chat(model=self.model, messages=msgs)
      return response['message']['content']
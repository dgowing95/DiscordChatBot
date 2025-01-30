import re
import base64
import requests
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
        self.vision_model = configManager().get_setting("vision_model")
        self.get_client()

    def get_client(self):
       self.client = AsyncClient(host=self.ollama_host)

    def format_message_history(self):

      self.history.pop(0) # Remove current message
      self.history.reverse() 

      formatted_history = []
      for message in self.history:
         if len(message.content) == 0 and not message.attachments:
            continue
         
         content = message.content
      

         formatted_history.append({
            'role': "assistant" if message.author.id == self.client_id else "user",
            'content': content
         })
      self.message_history = formatted_history
          

    async def generate(self):
      self.format_message_history()

      msgs = [
           {"role": "system", "content": self.system},
            *self.message_history,
           {
                'role': 'user',
                'content': self.prompt.content
           }
      ]
      
      attachment = self.get_image_base64(self.prompt)
      if attachment != "":
         attachment_msgs = [
            {"role": "system", "content": self.system},
            {
                'role': 'user',
                'content': attachment
           }
         ]
         print("The prompt contains an attachment")
         response: ChatResponse = await self.client.chat(model=self.vision_model, messages=attachment_msgs)
      else:
         response: ChatResponse = await self.client.chat(model=self.model, messages=msgs)
      
      print(f'Message returned')
      
      text = response['message']['content']
      cleaned_response = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
      return cleaned_response
   
    def get_image_base64(self, message):
      content = ""
      if message.attachments: 
         for attachment in message.attachments:
            response = requests.get(attachment.url)
            if response.status_code == 200:
               base64_data = base64.b64encode(response.content).decode('utf-8')
               base64_data = re.sub(r'^<.*?>', '', base64_data)
               content += f"This image needs to be described:\ndata:image/{attachment.url.split('.')[-1]};base64,{base64_data}\n"
      return content
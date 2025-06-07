import random, asyncio, re, json, time, os
from classes.text_llm_handler import TextLLMHandler

class MessageHandler:

    def __init__(self, message, client):
        self.message = message
        self.client = client
        self.text_response = ""
        self.discord_message_object = None


    async def build_messages(self):
       self.history = [message async for message in self.message.channel.history(limit=int(os.environ.get("MSG_HISTORY_LIMIT", 5)))]
       self.history.pop(0) # Remove current message

       formatted_history = []
       for message in self.history:
          
          if message.content.lower() == "!reset_history":
              break
          
          
          for embed in message.embeds:
              embed_dict = embed.to_dict()
              embed_dict.pop('fields', None)
              content = json.dumps(embed_dict)
              formatted_history.append({
                  'role': "assistant" if message.author.id == self.client.user.id else "user",
                  'content': f"Discord Emebed from '{message.author.name}' converted to JSON: {content}"
              })
            
          if len(message.content) == 0:
             continue        
          
          formatted_history.append({
             'role': "assistant" if message.author.id == self.client.user.id else "user",
             'content': f"Message from '{message.author.name}': {message.content.replace(f'<@{self.client.user.id}>', '').strip()}"
          })
       formatted_history.reverse()  # Reverse the history to have the oldest message first

       self.message.content = self.clean_message_content(self.message)
       formatted_history.append({
            'role': 'user',
            'content': f"Message from '{self.message.author.name}': {self.message.content}"
        })
       self.messages = formatted_history


    def should_process_message(self):
        if len(self.message.content) == 0 and len(self.message.embeds) == 0:
            return False
        if self.message.author == self.client.user:
            return False
        if self.message.content.lower() == "!reset_history":
            return False
        if self.client.user in self.message.mentions:
            return True
        if (random.randint(0,30) == 0):
            return True
        return False
    
    
    def clean_message_content(self, message):
        return message.content.replace(f'<@{self.client.user.id}>', '').strip()
    


    def filter_response(self, text_response):
        text_response.replace(f'<@{self.client.user.id}>', '').strip()
        text_response = re.sub(r"^<@.*:", "", text_response, flags=re.DOTALL)
        text_response = re.sub(r'\n\s*\n', '\n\n', text_response, flags=re.DOTALL)
        text_response = re.sub(r"Message from.*?:", "", text_response, flags=re.DOTALL)
        text_response = re.sub('<think>.*?</think>', '', text_response, flags=re.DOTALL)
        return text_response.strip()

    async def handle_message(self):
        print(f'Handling message: {self.message.content}')
        await self.build_messages()
        ollama = TextLLMHandler(self.messages, self.message.guild.id, self.message)
        response = await ollama.generate()

        if response == "Error":
            await self.message.add_reaction('❌')
            return

        response = self.filter_response(response)
        await self.message.reply(
            content=response
        )
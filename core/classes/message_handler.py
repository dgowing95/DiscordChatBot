import random, asyncio, re, json, time, os
from classes.text_llm_handler import TextLLMHandler
from classes.config_manager import configManager
from classes.response_filter import filter_response as clean_response

class MessageHandler:

    def __init__(self, message, client):
        self.message = message
        self.client = client
        self.text_response = ""
        self.discord_message_object = None
        self.config = configManager()


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
                  'content': f"Discord Embed from '{message.author.name}' converted to JSON: {content}"
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
        return self.random_chance_reply()
    
    def random_chance_reply(self):
        guild_id = self.message.guild.id if self.message.guild else 0
        chance = self.config.get_setting("response_chance", guild_id) or 5
        try:
            chance = min(max(float(chance), 0), 50)
        except (ValueError, TypeError):
            chance = 5
        return random.uniform(0, 100) < chance
    
    
    def clean_message_content(self, message):
        return message.content.replace(f'<@{self.client.user.id}>', '').strip()
    


    def filter_response(self, text_response):
        # Filter logic lives in classes.response_filter (pure, unit-tested);
        # we only inject the guild-specific bits (bot mention id).
        return clean_response(text_response, mention=str(self.client.user.id))


    async def handle_message_send(self, message_content):
        from textwrap import wrap
        chunks = wrap(message_content, 2000, break_long_words=False, replace_whitespace=False)
        for chunk in chunks:
            if len(chunk) == 0:
                continue
            await self.message.channel.send(chunk)
            await asyncio.sleep(1)
        


    async def handle_message(self):
        print(f'Handling message: {self.message.content}')
        await self.build_messages()
        ollama = TextLLMHandler(self.messages, self.message.guild.id, self.message)
        response = await ollama.generate()

        if response == "Error":
            await self.message.add_reaction('❌')
            return

        response = self.filter_response(response)
        await self.handle_message_send(response)

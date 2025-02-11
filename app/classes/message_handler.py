import random
from classes.text_llm_handler import TextLLMHandler

class MessageHandler:

    def __init__(self, message, client):
        self.message = message
        self.client = client
        self.response_exists = False


    async def build_messages(self):
       self.history = [message async for message in self.message.channel.history(limit=15)]
       self.history.pop(0) # Remove current message
       self.history.reverse() # Reverse the order of the messages so the newest is first

       formatted_history = []
       for message in self.history:
          if len(message.content) == 0:
             continue
          
          formatted_history.append({
             'role': "assistant" if message.author.id == self.client.user.id else "user",
             'content': f"Message from '{message.author.name}': {message.content.replace(f'<@{self.client.user.id}>', '').strip()}"
          })
       self.message.content = self.clean_message_content(self.message)
       formatted_history.append({
            'role': 'user',
            'content': f"Message from '{self.message.author.name}': {self.message.content}"
        })
       self.messages = formatted_history


    def should_process_message(self):
        if self.message.author == self.client.user:
            return False
        if self.client.user in self.message.mentions:
            return True
        if (random.randint(0,30) == 0):
            return True
        return False
    
    async def handle_text_input(self):
        ollama = TextLLMHandler(self.messages)
        response = await ollama.generate()
        self.response_exists = True
        return response
    
    def clean_message_content(self, message):
        return message.content.replace(f'<@{self.client.user.id}>', '').strip()

    
    async def handle_message(self):
        
        print(f'Handling message: {self.message.content}')
        await self.build_messages()

        # Check Message type
        if len(self.message.content) > 0:
            text_response = await self.handle_text_input()

        if text_response == "Error":
            self.response_exists = False

        if self.response_exists == False:
            await self.message.add_reaction('❌')
            return
        
        await self.message.reply(
            content=text_response
        )
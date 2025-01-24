import random
from classes.ollama_handler import ollamaHandler

class MessageHandler:
    message = False
    client = False
    history = []
    response_exists = False

    def __init__(self, message, client):
        self.message = message
        self.client = client

    async def get_message_history(self):
        self.history = [message async for message in self.message.channel.history(limit=50)]
        return self.history

    def should_process_message(self):
        if self.message.author == self.client.user:
            return False
        if self.client.user in self.message.mentions:
            return True
        if (random.randint(0,9) == 0):
            return True
        return False
    
    async def handle_text_input(self):
        ollama = ollamaHandler(self.history, self.message.content, self.client.user.id)
        response = await ollama.generate()
        self.response_exists = True
        return response

    
    async def handle_message(self):
        print(f'Handling message: {self.message.content}')
        await self.get_message_history()

        # Check Message type
        if len(self.message.content) > 0:
            text_response = await self.handle_text_input()

        if self.response_exists == False:
            return
        
        await self.message.reply(
            content=text_response
        )
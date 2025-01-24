import random
class MessageHandler:
    message = False
    client = False
    def __init__(self, message, client):
        self.message = message
        self.client = client

    def should_process_message(self):
        if self.message.author == self.client.user:
            return False
        if self.client.user in self.message.mentions:
            return True
        if (random.randint(0,9) == 0):
            return True
        return False
    
    async def handle_message(self):
        await self.message.reply(
            content="Hello"
        )
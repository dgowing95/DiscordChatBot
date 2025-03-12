import random, asyncio, re
from classes.text_llm_handler import TextLLMHandler

class MessageHandler:

    def __init__(self, message, client):
        self.message = message
        self.client = client
        self.text_response = ""
        self.discord_message_object = None


    async def build_messages(self):
       self.history = [message async for message in self.message.channel.history(limit=5)]
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
        if len(self.message.content) == 0:
            return False
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
        return response
    
    def clean_message_content(self, message):
        return message.content.replace(f'<@{self.client.user.id}>', '').strip()
    
    async def handle_text_stream(self, text_response_stream):
        async for chunk in text_response_stream:
            self.text_response += chunk.choices[0].delta.content or ""
            self.filter_response()
        print("Finished streaming chunks from LLM")

    async def handle_message_response(self, chunk_collect_task):
        self.discord_message_object = await self.message.reply(
            content=self.text_response or "..."
        )

        while chunk_collect_task.done() == False:
            await self.discord_message_object.edit(
                content=self.text_response or "..."
            )
            await asyncio.sleep(3)

        # Final update to the message
        await self.discord_message_object.edit(
            content=self.text_response[:1999]
        )
        print("Finished streaming text to discord message")

    def filter_response(self):
        self.text_response.replace(f'<@{self.client.user.id}>', '').strip()
        self.text_response = re.sub(r"^<@.*:", "", self.text_response, flags=re.DOTALL)
        self.text_response = re.sub(r'\n\s*\n', '\n\n', self.text_response, flags=re.DOTALL)
        self.text_response = re.sub(r"Message from.*?:", "", self.text_response, flags=re.DOTALL)

    async def handle_message(self):
        print(f'Handling message: {self.message.content}')
        await self.build_messages()
        text_response_stream = await self.handle_text_input()

        if text_response_stream == "Error":
            await self.message.add_reaction('❌')
            return

        async with asyncio.TaskGroup() as tg:
            chunk_collect_task = asyncio.create_task(
                self.handle_text_stream(text_response_stream)
            )
            message_response_task = asyncio.create_task(
                self.handle_message_response(chunk_collect_task)
            )
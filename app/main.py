import discord
import asyncio
import os
from classes.message_handler import MessageHandler
from classes.config_manager import configManager

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)
message_queue = asyncio.Queue()

@client.event
async def on_ready():
    print(f'Logged in as {client.user}')
    client.loop.create_task(process_messages())

@client.event
async def on_message(message):
    await message_queue.put(message)
    
async def process_messages():
    while True:
        message = await message_queue.get()
        handler = MessageHandler(message, client)

        if handler.should_process_message() == False:
            continue

        async with message.channel.typing():
            await handler.handle_message()
        message_queue.task_done()
    
    

token = configManager().get_setting("discord_token")
client.run(token)

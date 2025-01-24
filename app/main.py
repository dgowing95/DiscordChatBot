import discord
import os
from classes.message_handler import MessageHandler

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)


@client.event
async def on_ready():
    print(f'Logged in as {client.user}')

@client.event
async def on_message(message):
    print(message.content)
    handler = MessageHandler(message, client)

    if handler.should_process_message() == False:
        return

    await handler.handle_message()

client.run(os.environ["DISCORD_BOT_TOKEN"])
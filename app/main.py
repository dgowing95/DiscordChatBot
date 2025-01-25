import discord
import os
from classes.message_handler import MessageHandler
from classes.config_manager import configManager

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)


@client.event
async def on_ready():
    print(f'Logged in as {client.user}')

@client.event
async def on_message(message):
    handler = MessageHandler(message, client)

    if handler.should_process_message() == False:
        return

    async with message.channel.typing():
        await handler.handle_message()

token = configManager().get_setting("discord_token")
client.run(token)

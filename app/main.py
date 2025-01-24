import discord
import os
from classes.message_handler import MessageHandler

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)


@client.event
async def on_ready():
    print(f'Logged in as {client.user}')
    CLIENT_ID = client.user.id

@client.event
async def on_message(message):
    handler = MessageHandler(message, client)

    if handler.should_process_message() == False:
        return

    async with message.channel.typing():
        await handler.handle_message()

client.run(os.environ["DISCORD_BOT_TOKEN"])

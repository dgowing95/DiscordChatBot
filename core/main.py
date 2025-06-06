import discord
import asyncio
import os
import aiohttp
import io
from classes.message_handler import MessageHandler
from classes.text_llm_handler import TextLLMHandler
from classes.config_manager import configManager


intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)
message_queue = asyncio.Queue()
config = configManager()


async def register_commands():
    print("Registering commands")
    command_tree = discord.app_commands.CommandTree(client=client, fallback_to_global=True)

    @command_tree.command(name="system", description="Change the behaviour/personality of the bot")
    async def change_system(ctx, system: str):
        config.update_setting("system", system, ctx.guild.id)
        await ctx.response.send_message(content=f"System updated to: \"{system}\"")

    @command_tree.command(name="get_system", description="See the existing behaviour/personality of the bot")
    async def get_system(ctx):
        system = config.get_setting("system", ctx.guild.id)
        await ctx.response.send_message(content=f"System is currently: \"{system}\"")

    @command_tree.command(name="temperature", description="Change the randomness of responses, max of 2.0 is max random")
    async def change_temperature(ctx, temperature: float):
        config.update_setting("temperature", temperature, ctx.guild.id)
        await ctx.response.send_message(content=f"Temperature updated to: \"{temperature}\"")  
    
    synced_commands = await command_tree.sync()
    for synced_command in synced_commands:
        print(f"Command '{synced_command.name}' synced")

@client.event
async def on_ready():
    print(f'Logged in as {client.user}')
    model = os.environ.get("MODEL", "gemma3:4b")
    await TextLLMHandler.pull_model(model)
    await TextLLMHandler.pull_model("qwen3:4b")
    await register_commands()
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
        
        print("Picking up message from queue")
        try:
            async with message.channel.typing():
                await handler.handle_message()
                message_queue.task_done()
                print("Done with message from queue")
        except Exception as e:
            print("Error handling message: " + str(e))
            message_queue.task_done()
            print("Done with message from queue")


token = os.environ['DISCORD_TOKEN']
client.run(token)

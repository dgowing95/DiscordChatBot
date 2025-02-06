import discord
import asyncio
import os
from classes.message_handler import MessageHandler
from classes.config_manager import configManager

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)
message_queue = asyncio.Queue()


async def register_commands():
    print("Registering commands")
    command_tree = discord.app_commands.CommandTree(client=client, fallback_to_global=True)

    @command_tree.command(name="system", description="Change the behaviour/personality of the bot")
    async def change_system(ctx, system: str):
        configManager().update_setting("system", system)
        await ctx.response.send_message(content=f"System updated to: \"{system}\"")

    @command_tree.command(name="temperature", description="Change the randomness of responses, max of 2.0 is max random")
    async def change_temperature(ctx, temperature: float):
        configManager().update_setting("temperature", temperature)
        await ctx.response.send_message(content=f"Temperature updated to: \"{temperature}\"")

    @command_tree.command(name="model", description="Change the model used to generate reponses")
    @discord.app_commands.choices(choices=[
        discord.app_commands.Choice(name="dolphin-llama3:8b", value="dolphin-llama3:8b"),
        discord.app_commands.Choice(name="deepseek-r1:8b", value="deepseek-r1:8b"),
        discord.app_commands.Choice(name="deepseek-r1:14b", value="deepseek-r1:14b")
    ])
    async def change_model(ctx, choices: discord.app_commands.Choice[str]):
        configManager().update_setting("model", choices.value)
        await ctx.response.send_message(content=f"Model updated to: \"{choices.value}\"")        

    synced_commands = await command_tree.sync()
    for synced_command in synced_commands:
        print(f"Command '{synced_command.name}' synced")

@client.event
async def on_ready():
    print(f'Logged in as {client.user}')
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

        async with message.channel.typing():
            await handler.handle_message()
        message_queue.task_done()
    
    

token = configManager().get_setting("discord_token")
client.run(token)

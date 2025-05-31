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
image_task_queue = asyncio.Queue()
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
    
    @command_tree.command(name="make_image", description="Generate an image based on a prompt")
    async def make_image(ctx, prompt: str):
        await ctx.response.defer(ephemeral=False, thinking=True)
        await image_task_queue.put({'type': 'make', 'task_data': {'context': ctx, 'prompt': prompt}})
        print(f"Image generation task added to queue with prompt: {prompt}")

    @command_tree.command(name="modify_image", description="Modify an image based on a prompt")
    async def modify_image(ctx, prompt: str, attachment: discord.Attachment):
        await ctx.response.defer(ephemeral=False, thinking=True)
        await image_task_queue.put({'type': 'modify', 'task_data': {'context': ctx, 'prompt': prompt, 'attachment': attachment}})
        print(f"Image generation task added to queue with prompt: {prompt}")

    synced_commands = await command_tree.sync()
    for synced_command in synced_commands:
        print(f"Command '{synced_command.name}' synced")

@client.event
async def on_ready():
    print(f'Logged in as {client.user}')
    await TextLLMHandler.pull_model()
    await register_commands()
    client.loop.create_task(process_messages())
    client.loop.create_task(process_text_images())
    

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

        
async def process_text_images():
    while True:
        task = await image_task_queue.get()
        prompt = task['task_data']['prompt']
        ctx = task['task_data']['context']

        try:
            if task['type'] == 'make':
                print(f"Processing image generation for prompt: {prompt}")
                image_bytes = await generate_image_from_api(prompt)
            elif task['type'] == 'modify':
                print(f"Processing image modification for prompt: {prompt}")
                attachment = task['task_data']['attachment']
                image_bytes = await modify_image_from_api(prompt, attachment)

            file = discord.File(io.BytesIO(image_bytes), filename="image.png")
            await ctx.followup.send(file=file, content=prompt)

        except Exception as e:
            print(f"Error during image generation: {e}")
            await ctx.followup.send("Failed to generate image.")
        image_task_queue.task_done()

async def generate_image_from_api(prompt: str) -> bytes:
    async with aiohttp.ClientSession(auto_decompress=False) as session:
        async with session.post(
            f"http://{os.environ.get("DIFFUSION_URL", 5)}:8000/text-image",
            json={"prompt": prompt}
        ) as resp:
            if resp.status != 200:
                raise Exception(f"API failed with status {resp.status}")
            return await resp.read()
    

async def modify_image_from_api(prompt: str, image: discord.Attachment) -> bytes:
    async with aiohttp.ClientSession(auto_decompress=False) as session:
        image_bytes = await image.read()
        data = aiohttp.FormData()
        data.add_field('prompt', prompt)
        data.add_field('file', image_bytes)

        async with session.post(
            f"http://{os.environ.get("DIFFUSION_URL", 5)}:8000/image-image",
            data=data
        ) as resp:
            if resp.status != 200:
                raise Exception(f"API failed with status {resp.status}")
            return await resp.read()
    

token = os.environ['DISCORD_TOKEN']
client.run(token)

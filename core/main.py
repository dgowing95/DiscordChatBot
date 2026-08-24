import discord
import asyncio
import os
import aiohttp
import io
from classes.message_handler import MessageHandler
from classes.text_llm_handler import TextLLMHandler
from classes.config_manager import configManager
from classes.image_generation import generate_image_from_api, image_generation_enabled
from classes.sandbox_agent import sandbox_enabled


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
  
    @command_tree.command(name="chance", description="Change the chance (0-50%) that the bot replies without being mentioned, default 5%")
    async def change_chance(ctx, chance: discord.app_commands.Range[int, 0, 50]):
        config.update_setting("response_chance", chance, ctx.guild.id)
        await ctx.response.send_message(content=f"Response chance updated to: \"{chance}%\"")
  
    @command_tree.command(name="get_chance", description="See the current chance that the bot replies without being mentioned")
    async def get_chance(ctx):
        chance = config.get_setting("response_chance", ctx.guild.id) or 5
        await ctx.response.send_message(content=f"Response chance is currently: \"{chance}%\"")  
    
    # Only offered when the code sandbox is enabled (SANDBOX_ENABLED; set
    # from the helm chart's sandbox.enabled). Per-guild toggle, default off:
    # when true, run_code_sandbox streams the sandbox's commands and output
    # to the channel in a live-updating message; when false it only sends
    # the one static "Running in sandbox" embed.
    if sandbox_enabled():

        @command_tree.command(name="sandbox_progress_updates",
                              description="Enable/disable live progress updates (commands & output) for code sandbox runs in this server")
        async def change_sandbox_progress_updates(ctx, enabled: bool):
            config.update_setting("sandbox_progress_updates", str(enabled), ctx.guild.id)
            await ctx.response.send_message(content=f"Sandbox progress updates are now: \"{enabled}\"")
    
    # Only offered when the diffusion service is enabled (IMAGE_GEN_ENABLED;
    # set from the helm chart's diffusion.enabled).
    if image_generation_enabled():

        @command_tree.command(name="generate_image", description="Generate an image from a text prompt using the image service")
        async def generate_image_cmd(ctx, prompt: str):
            # Image generation is slow (queue + GPU): defer first (spinner) so
            # the command doesn't time out, then respond to the deferred
            # interaction via the webhook. ctx is a raw discord.Interaction
            # (the bot is a discord.Client), so: ctx.response.defer() to defer
            # and ctx.edit_original_response() to answer it later — NOT
            # ctx.response.edit_message(), which raises InteractionResponded.
            await ctx.response.defer()
            print(f"Slash command: generating image for prompt: {prompt}")
            try:
                image_bytes = await generate_image_from_api(prompt)
            except Exception as e:
                print(f"Image generation failed: {e}")
                await ctx.edit_original_response(
                    content="❌ Image generation failed — the image service may be down or busy. Try again later."
                )
                return
            await ctx.edit_original_response(
                content="🎨",
                attachments=[discord.File(io.BytesIO(image_bytes), filename="generated-image.png")],
            )

        @command_tree.command(name="edit_image", description="Edit an image with a text prompt (image-to-image)")
        async def edit_image_cmd(
            ctx,
            image: discord.Attachment,
            prompt: str,
            strength: discord.app_commands.Range[float, 0.1, 0.9] | None = None,
        ):
            await ctx.response.defer()
            print(f"Slash command: editing image with prompt: {prompt}")
            try:
                source = await image.read()
                image_bytes = await generate_image_from_api(
                    prompt, image=source, strength=strength
                )
            except Exception as e:
                print(f"Image editing failed: {e}")
                await ctx.edit_original_response(
                    content="❌ Image editing failed — the image service may be down or busy, or the image could not be read. Try again later."
                )
                return
            await ctx.edit_original_response(
                content="🎨",
                attachments=[discord.File(io.BytesIO(image_bytes), filename="edited-image.png")],
            )

    synced_commands = await command_tree.sync()
    for synced_command in synced_commands:
        print(f"Command '{synced_command.name}' synced")

@client.event
async def on_ready():
    print(f'Logged in as {client.user}')
    # Start the queue worker immediately so the bot still consumes messages
    # even if the model check or command sync fails (e.g. server not up yet
    # after a power cycle).
    client.loop.create_task(process_messages())
    try:
        model = os.environ.get("MODEL", "gemma3:4b")
        await TextLLMHandler.check_model_ready(model)
    except Exception as e:
        print(f"Failed to check model: {e}")
        return
    await register_commands()
    


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

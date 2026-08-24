import asyncio
import io

from agents import FunctionTool, function_tool,RunContextWrapper
from classes.common import Common
from classes.content_guard import check_web_request
import discord, aiohttp
from ddgs import DDGS
from bs4 import BeautifulSoup

@function_tool
async def fetch_weather(location: str) -> str:
    """Fetch the weather for a given location.

    Args:
        location: The location to fetch the weather for.
    """
    print(f"Fetching weather for location: {location}")
    return "15°C, clear skies"


async def add_emoji_to_message(message: discord.Message, emoji: str) -> None:
    try:
        await message.add_reaction(emoji)
        print(f"Added emoji {emoji} to message {message.id}")
    except Exception as e:
        print(f"Failed to add emoji {emoji} to message {message.id}: {e}")
    
@function_tool
async def web_search(wrapper: RunContextWrapper[dict], search_request: str) -> str:
    """Searches the internet for a given query.

    Args:
        search_request: The query to search for.
    """
    print(f"Searching the web for: {search_request}")

    allowed, reason = await check_web_request(search_request)
    if not allowed:
        print(f"Web search blocked by content guard: {reason}")
        return ("I can't perform that search — it was blocked by the safety "
                "guard. Please rephrase with a safe, non-harmful query.")

    await Common.send_tool_discord_embed(
        wrapper.context.get("original_message").channel,
        f"Searching the web for: {search_request}",
    )
    try:
        results = DDGS().text(search_request, max_results=5)
    except Exception as e:
        print(f"An error occurred while searching: {e}")
        return "Error fetching search results."
    return results
    
@function_tool
async def fetch_url(wrapper: RunContextWrapper[dict], url: str) -> str:
    """Fetches the content of a URL. Returns the text content of the page.

    Args:
        url: The URL to fetch.
    """
    print(f"Fetching content from URL: {url}")

    allowed, reason = await check_web_request(url)
    if not allowed:
        print(f"URL fetch blocked by content guard: {reason}")
        return ("I can't fetch that URL — it was blocked by the safety "
                "guard.")

    await Common.send_tool_discord_embed(
        wrapper.context.get("original_message").channel,
        f"Visiting URL: {url}",
    )
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers={"User-Agent": "dis-ai-bot"}) as response:
                html = await response.text()
    except Exception as e:
        print(f"An error occurred while fetching the URL: {e}")
        return "Error fetching URL content."

    soup = BeautifulSoup(html, features='html.parser')
    for script in soup(["script", "style"]):
        script.extract()  # remove all javascript and stylesheet code
    
    text = soup.body.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)
    print(f"Fetched content from {url} successfully.")
    return text

async def get_current_datetime() -> str:
    """Returns the current date and time."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    now = datetime.now(ZoneInfo("Europe/London"))
    now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"Current date and time: {now_formatted}")
    return now_formatted

@function_tool
async def store_memory(wrapper: RunContextWrapper[dict], data: str) -> str:
    """Stores a memory about the user. This could be anything from preferences to personal information.
    Returns True if successful, False otherwise.
    Args:
        data: The data to store. e.g. User's name, preferences, etc.
    """

    # Sometimes OpenAI repeats a tool call.
    times_called = wrapper.context.get("redis_save_tool_calls")
    if times_called > 0:
        err = f"Tool call limit reached: {times_called}. Not storing data."
        print(err)
        return "Data stored successfully."
    wrapper.context["redis_save_tool_calls"] += 1
    

    from classes.user_memory import UserMemory
    user_id = wrapper.context.get("user_id")
    guild_id = wrapper.context.get("guild_id")

    response_message = ""
    try:
        print(f"Storing data for user {user_id} in guild {guild_id}: {data}")
        user_memory = UserMemory(user_id, guild_id)
        user_memory.append(data)
        await add_emoji_to_message(wrapper.context.get("original_message"), "💾")
        await Common.send_tool_discord_embed(
            wrapper.context.get("original_message").channel,
            f"Stored data: {data}",
        )
        
        response_message = "Data stored successfully."
    except Exception as e:
        response_message = f"An error occurred while storing user data: {e}"
        print(response_message)

    return response_message

@function_tool
async def remove_memory(wrapper: RunContextWrapper[dict], data: str) -> str:
    """Removes a specific memory for the user.
    Args:
        data: The specific memory to remove.
    """
    
    from classes.user_memory import UserMemory
    user_id = wrapper.context.get("user_id")
    guild_id = wrapper.context.get("guild_id")
    try:
        user_memory = UserMemory(user_id, guild_id)
        removed = user_memory.remove(data)
        if removed:
            await add_emoji_to_message(wrapper.context.get("original_message"), "🗑️")
            return f"Removed memory: {data}"
        else:
            return "Memory not found."
    except Exception as e:
        print(f"An error occurred while removing user memory: {e}")
        return "Error removing user memory."

@function_tool
async def clear_memories(wrapper: RunContextWrapper[dict]) -> str:
    """Clears all memories for the user."""
    
    from classes.user_memory import UserMemory
    user_id = wrapper.context.get("user_id")
    guild_id = wrapper.context.get("guild_id")
    try:
        user_memory = UserMemory(user_id, guild_id)
        user_memory.clear()
        await add_emoji_to_message(wrapper.context.get("original_message"), "🧹")
        return "All memories cleared."
    except Exception as e:
        print(f"An error occurred while clearing user memories: {e}")
        return "Error clearing user memories."


@function_tool
async def generate_image(wrapper: RunContextWrapper[dict], prompt: str) -> str:
    """Generates an image from a text description and sends it to the channel.
    Use it when the user asks for art, illustrations, pictures or drawings.
    The image is sent automatically; never try to send it yourself.
    Args:
        prompt: The text description of the image to generate. The image
            model (Juggernaut XI) responds best to precise, specific
            prompts: put the main subject in the first sentence, then add
            the setting, the subject's action, secondary objects, colors,
            lighting, style/medium, mood and camera angle; name textures
            and materials explicitly (e.g. "coarse fur", "polished
            steel"); for people, describe their clothing and emphasize
            the emotion they should show; keep it tight - a couple of
            short sentences with no filler, since long prompts reduce
            adherence; append "high resolution"; for any text that must
            appear in the image, use a short phrase in quotes near the
            start (long text is often misspelled).
    """
    from classes.image_generation import generate_image_from_api

    message = wrapper.context.get("original_message")
    print(f"Generating image for prompt: {prompt}")
    await add_emoji_to_message(message, "🎨")
    await Common.send_tool_discord_embed(
        message.channel,
        f"Generating image: {prompt}",
    )
    try:
        image_bytes = await generate_image_from_api(prompt)
    except Exception as e:
        print(f"Image generation failed: {e}")
        return ("Image generation failed. Tell the user the image service is "
                "unavailable right now and do not retry.")
    try:
        await message.channel.send(
            file=discord.File(io.BytesIO(image_bytes), filename="generated-image.png")
        )
    except Exception as e:
        print(f"Image generated but failed to send to Discord: {e}")
        return "The image was generated but could not be sent to the channel."
    return ("Image generated and sent to the channel. The user can already see "
            "it; do not send the image again or describe it as if pending.")


@function_tool
async def edit_image(
    wrapper: RunContextWrapper[dict],
    prompt: str,
    image_ref: str = "latest",
    strength: float | None = None,
) -> str:
    """Edits an existing image with a text prompt (image-to-image) and sends
    the result to the channel. Use it when the user asks to modify, restyle
    or transform an image attached in the conversation.
    The edited image is sent automatically; never try to send it yourself.
    Args:
        prompt: What to change or create (e.g. "make it snowy").
        image_ref: Label of the image to edit, from the "Attached images"
            list in the message (e.g. "1"), or "latest" for the most recent
            image (default). Never paste or guess URLs — the bot resolves
            the label to the real image on its side.
        strength: Optional, 0-1 (exclusive). Higher = more changes, lower =
            closer to the original. Omit for a sensible default.
    """
    from classes.image_generation import generate_image_from_api

    # Resolve the short label to the real signed CDN URL (the LLM must never
    # copy the URL itself — it corrupts the 64-char hex signature).
    refs = wrapper.context.get("attachment_refs") or []
    if not refs:
        return ("No image is attached in this conversation. Ask the user to "
                "attach the image they want edited, then try again.")
    if image_ref in (None, "", "latest"):
        entry = refs[-1]
    else:
        entry = next((r for r in refs if r["ref"] == str(image_ref)), None)
        if entry is None:
            available = ", ".join(f"'{r['ref']}'" for r in refs)
            return (f"No attached image with label '{image_ref}'. "
                    f"Available image labels: {available}.")
    image_url = entry["url"]

    message = wrapper.context.get("original_message")
    print(f"Editing image [{entry['ref']}] {entry['filename']} with prompt: {prompt}")
    await add_emoji_to_message(message, "🎨")
    await Common.send_tool_discord_embed(
        message.channel,
        f"Editing image: {prompt}",
    )
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                image_url, timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status != 200:
                    return (f"Could not download the image to edit (HTTP {resp.status}). "
                            "Tell the user the image could not be fetched.")
                source = await resp.read()
    except Exception as e:
        print(f"Failed to download image {image_url}: {e}")
        return ("Could not download the image to edit. "
                "Tell the user the image could not be fetched.")
    try:
        image_bytes = await generate_image_from_api(prompt, image=source, strength=strength)
    except Exception as e:
        print(f"Image editing failed: {e}")
        return ("Image editing failed. Tell the user the image service is "
                "unavailable right now and do not retry.")
    try:
        await message.channel.send(
            file=discord.File(io.BytesIO(image_bytes), filename="edited-image.png")
        )
    except Exception as e:
        print(f"Image edited but failed to send to Discord: {e}")
        return "The image was edited but could not be sent to the channel."
    return ("The image was edited and sent to the channel. The user can already "
            "see it; do not send the image again or describe it as if pending.")


@function_tool
async def run_code_sandbox(wrapper: RunContextWrapper[dict], task: str) -> str:
    """Runs a task in an isolated code sandbox: a fresh, disposable Linux
    container (Python + shell) where code is written and actually executed.
    Use it when the answer depends on running code or commands, not just
    reasoning about it: writing or debugging a program, computing a value,
    processing or analyzing data, converting files, checking that a library
    or API behaves as expected. Do not use it for questions you can answer
    directly. The sandbox starts completely empty and is destroyed when the
    task finishes, so the task must be fully self-contained: include any
    code, data or context the sandbox needs, and state exactly what the
    final result must contain. When the guild has live progress updates
    enabled (/sandbox_progress_updates, default off), the commands the
    sandbox runs and their output are streamed to the channel in a
    live-updating message.
    Args:
        task: A precise, self-contained description of what to do in the
            sandbox, including any code or data involved and what the final
            answer should contain.
    """
    print(f"Running sandbox task: {task}")

    allowed, reason = await check_web_request(task)
    if not allowed:
        print(f"Sandbox task blocked by content guard: {reason}")
        return ("I can't run that in the sandbox — it was blocked by the safety "
                "guard. Please rephrase with a safe, non-harmful task.")

    from classes.config_manager import configManager
    from classes.sandbox_agent import run_sandbox_task
    from classes.sandbox_progress import (
        SandboxProgressHooks,
        sandbox_progress_updates_enabled,
    )

    channel = wrapper.context.get("original_message").channel
    progress = None
    # Per-guild toggle (/sandbox_progress_updates, default off). A config
    # read failure falls back to off — progress is a nice-to-have, not a
    # dependency of the run itself.
    try:
        raw = configManager().get_setting(
            "sandbox_progress_updates", wrapper.context.get("guild_id"))
    except Exception as e:
        print(f"Could not read sandbox_progress_updates setting: {e}")
        raw = False
    if sandbox_progress_updates_enabled(raw):
        # Live progress: one Discord message, edited in place, showing each
        # command the sandbox runs and its output (throttled for Discord's
        # 5-edits/minute limit). start() is exception-safe (swallows send
        # failures), so a progress problem never blocks the run.
        progress = SandboxProgressHooks(channel, task)
        await progress.start()
    else:
        await Common.send_tool_discord_embed(
            channel,
            f"Running in sandbox: {task}",
        )

    try:
        result = await run_sandbox_task(task, progress)
    except asyncio.TimeoutError:
        print("Sandbox task timed out")
        if progress is not None:
            await progress.finalize("⏱ Stopped: the task timed out.")
        return ("The sandbox task took too long and was stopped. Tell the user "
                "the task timed out; you may retry with a smaller or more "
                "focused task.")
    except Exception as e:
        print(f"Sandbox task failed: {e}")
        if progress is not None:
            await progress.finalize("❌ Stopped: the sandbox run failed.")
        return ("The sandbox task failed (the code sandbox may be unavailable). "
                "Tell the user the sandbox is not working right now and do "
                "not retry the same task.")
    if progress is not None:
        # give the live message its final state (it would otherwise sit on
        # the last "still running" / thinking snapshot)
        await progress.finalize("✅ Done.")
    return result


@function_tool
async def change_personality(wrapper: RunContextWrapper[dict], personality: str) -> bool:
    """Changes the personality of the bot. Returns True if successful, False otherwise.
    
    Args:
        personality: The new personality to set.
    """

    # Sometimes OpenAI repeats a tool call.
    times_called = wrapper.context.get("personality_tool_calls")
    if times_called > 0:
        err = f"Tool call limit reached: {times_called}. Not storing data."
        print(err)
        return True
    wrapper.context["personality_tool_calls"] += 1

    from classes.config_manager import configManager
    print(f"Changing personality to: {personality}")
    try:
        configmanager = configManager()
        configmanager.update_setting("system", personality, wrapper.context.get("guild_id"))
        print(f"Changed personality to: {personality}")

        embed = discord.Embed(title="Personality Updated",
                      description=personality)
        await wrapper.context.get("original_message").channel.send(embed=embed)
        return True
    except Exception as e:
        print(f"An error occurred while changing personality: {e}")
        return False

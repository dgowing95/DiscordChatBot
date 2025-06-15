from agents import FunctionTool, function_tool,RunContextWrapper
import discord, aiohttp
from duckduckgo_search import DDGS
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
    await add_emoji_to_message(wrapper.context.get("original_message"), "🌐")
    try:
        results = DDGS().text(search_request, max_results=5)
    except Exception as e:
        print(f"An error occurred while searching: {e}")
        return "Error fetching search results."
    
    # for result in results:
    #     print(f"Title: {result['title']}, URL: {result['href']}")
    print(results)
    return results
    
@function_tool
async def fetch_url(wrapper: RunContextWrapper[dict], url: str) -> str:
    """Fetches the content of a URL. Returns the text content of the page.

    Args:
        url: The URL to fetch.
    """
    print(f"Fetching content from URL: {url}")
    await add_emoji_to_message(wrapper.context.get("original_message"), "📄")
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

@function_tool
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
    Args:
        data: The data to store. e.g. User's name, preferences, etc.
    """
    await add_emoji_to_message(wrapper.context.get("original_message"), "💾")
    from classes.user_memory import UserMemory
    user_id = wrapper.context.get("user_id")
    guild_id = wrapper.context.get("guild_id")
    try:
        print(f"Storing data for user {user_id} in guild {guild_id}: {data}")
        user_memory = UserMemory(user_id, guild_id)
        user_memory.append(data)
        return "User data stored successfully."
    except Exception as e:
        print(f"An error occurred while storing user data: {e}")
        return "Error storing user data."


@function_tool
async def change_personality(wrapper: RunContextWrapper[dict], personality: str) -> str:
    """Changes the personality of the bot.
    
    Args:
        personality: The new personality to set.
    """
    from classes.config_manager import configManager
    print(f"Changing personality to: {personality}")
    try:
        configmanager = configManager()
        configmanager.update_setting("system", personality, wrapper.context.get("guild_id"))
        print(f"Changed personality to: {personality}")

        embed = discord.Embed(title="Personality Updated",
                      description=personality)
        await wrapper.context.get("original_message").channel.send(embed=embed)
        return "Success."
    except Exception as e:
        print(f"An error occurred while changing personality: {e}")
        return "Error changing personality."

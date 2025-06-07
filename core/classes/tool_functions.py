from agents import FunctionTool, function_tool,RunContextWrapper
import aiohttp
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
    
@function_tool
async def web_search(search_request: str) -> str:
    """Searches the internet for a given query.

    Args:
        search_request: The query to search for.
    """
    print(f"Searching the web for: {search_request}")

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
async def fetch_url(url: str) -> str:
    """Fetches the content of a URL. Returns the text content of the page.

    Args:
        url: The URL to fetch.
    """
    print(f"Fetching content from URL: {url}")
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
async def store_user_data(wrapper: RunContextWrapper[dict], data: str) -> str:
    """Stores user data in Redis.
    Args:
        data: The data to store. e.g. User's name, preferences, etc.
    """
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

from agents import FunctionTool, function_tool
import aiohttp
from duckduckgo_search import DDGS

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
    """Fetches the content of a URL.

    Args:
        url: The URL to fetch.
    """
    print(f"Fetching content from URL: {url}")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers={"User-Agent": "dis-ai-bot"}) as response:
                return await response.text()
    except Exception as e:
        print(f"An error occurred while fetching the URL: {e}")
        return "Error fetching URL content."
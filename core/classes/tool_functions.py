from agents import FunctionTool, function_tool

@function_tool
async def fetch_weather(location: str) -> str:
    """Fetch the weather for a given location.

    Args:
        location: The location to fetch the weather for.
    """
    print(f"Fetching weather for location: {location}")
    return "15°C, clear skies"
    

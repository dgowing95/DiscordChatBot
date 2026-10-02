"""What /help shows (PURE, stdlib only).

Hand-written on purpose: generating it with the LLM on every call would be
slow, and a small local model can invent or misdescribe features. To stop it
going stale, core/tests/help_catalog_tests.py fails when an agent tool
(text_llm_handler.agent_tools) or a slash command (main.register_commands)
has no entry here. Adding a tool or command therefore means adding a line.

Entries with `requires` are hidden when that feature is switched off for the
deployment (IMAGE_GEN_ENABLED / SANDBOX_ENABLED), so /help never offers
something the bot cannot do.
"""
from dataclasses import dataclass

from classes.whats_new import EMBED_COLOR

IMAGE = "image"
SANDBOX = "sandbox"


@dataclass(frozen=True)
class HelpEntry:
    name: str
    description: str
    # The agent tool names this entry explains (what the sync test checks).
    tools: tuple = ()
    # The slash command this entry IS ("" for an ask-me entry).
    command: str = ""
    requires: str = ""


# "Just ask me" - agent tools, described by what a person asks for rather
# than by tool name, since nobody calls them directly.
ASK_ENTRIES = (
    HelpEntry("🔎 Search the web", "Ask about news or anything recent and I'll look it up.",
              tools=("web_search",)),
    HelpEntry("🔗 Read a link", "Paste a URL and ask me to summarise or explain it.",
              tools=("fetch_url",)),
    HelpEntry("🧠 Remember things", "Tell me something to remember about you, ask me to forget "
              "it, or to forget everything. Memories are per server.",
              tools=("store_memory", "remove_memory", "clear_memories")),
    HelpEntry("🎭 Change personality", "Ask me to act differently, e.g. \"talk like a pirate\".",
              tools=("change_personality",)),
    HelpEntry("🎨 Make images", "Ask me to draw or generate a picture.",
              tools=("generate_image",), requires=IMAGE),
    HelpEntry("🐳 Run code", "Ask me to build, run or make something with code (scripts, "
              "charts, gifs...). I work in a thread you can reply in to steer me, and "
              "pick up where I left off when you @mention me there again.",
              tools=("run_code_sandbox",), requires=SANDBOX),
)

COMMAND_ENTRIES = (
    HelpEntry("/help", "Show this list.", command="help"),
    HelpEntry("/whats_new", "See the new features in the current version.", command="whats_new"),
    HelpEntry("/system", "Set my personality for this server.", command="system"),
    HelpEntry("/get_system", "Show my current personality.", command="get_system"),
    HelpEntry("/temperature", "How random my replies are (0 to 2).", command="temperature"),
    HelpEntry("/chance", "Chance (0-50%) I reply without being mentioned.", command="chance"),
    HelpEntry("/get_chance", "Show that chance.", command="get_chance"),
    HelpEntry("/generate_image", "Make an image straight from a prompt.",
              command="generate_image", requires=IMAGE),
    HelpEntry("/sandbox_progress_updates", "Turn live progress for code runs on or off.",
              command="sandbox_progress_updates", requires=SANDBOX),
)

TIPS = (
    "• @mention me to get a reply; otherwise I only join in now and then (see /chance).\n"
    "• Send `!reset_history` to make me forget the conversation above it.\n"
    "• I can see images you attach."
)

HELP_TITLE = "🤖 What I can do"


def _shown(entry: HelpEntry, image_enabled: bool, sandbox_enabled: bool) -> bool:
    if entry.requires == IMAGE:
        return image_enabled
    if entry.requires == SANDBOX:
        return sandbox_enabled
    return True


def help_embed_data(image_enabled: bool, sandbox_enabled: bool) -> dict:
    """The /help embed as plain data (see classes.common.embed_from_data).
    One field per group keeps it compact on mobile."""
    ask = [e for e in ASK_ENTRIES if _shown(e, image_enabled, sandbox_enabled)]
    commands = [e for e in COMMAND_ENTRIES if _shown(e, image_enabled, sandbox_enabled)]
    return {
        "title": HELP_TITLE,
        "color": EMBED_COLOR,
        "fields": [
            {"name": "Just ask me",
             "value": "\n".join(f"**{e.name}** — {e.description}" for e in ask)},
            {"name": "Slash commands",
             "value": "\n".join(f"`{e.name}` — {e.description}" for e in commands)},
            {"name": "Tips", "value": TIPS},
        ],
        "footer": "",
    }


def documented_tools() -> set:
    return {tool for entry in ASK_ENTRIES for tool in entry.tools}


def documented_commands() -> set:
    return {entry.command for entry in COMMAND_ENTRIES}

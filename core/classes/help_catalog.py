"""What /help shows (PURE, stdlib only).

Hand-written on purpose: generating it with the LLM on every call would be
slow, and a small local model can invent or misdescribe features. To stop it
going stale, core/tests/help_catalog_tests.py fails when a brain tool
(brain_tools.brain_tools: since v3 every tool is the brain's, and the persona
only has ask_brain) or a slash command (main.register_commands) has no entry
here. Adding a tool or command therefore means adding a line.

Entries with `requires` are hidden when that feature is switched off for the
deployment (IMAGE_GEN_ENABLED / EVENTS_ENABLED ...), so /help never offers
something the bot cannot do.
"""
from dataclasses import dataclass

from classes.whats_new import EMBED_COLOR, MAX_FIELD_VALUE

IMAGE = "image"
AUTOMATIONS = "automations"
VOICE = "voice"
EVENTS = "events"


@dataclass(frozen=True)
class HelpEntry:
    name: str
    description: str
    # The brain tool names this entry explains (what the sync test checks).
    tools: tuple = ()
    # The slash command this entry IS ("" for an ask-me entry).
    command: str = ""
    requires: str = ""


# "Just ask me" - the brain's tools, described by what a person asks for
# rather than by tool name, since nobody calls them directly.
ASK_ENTRIES = (
    HelpEntry("🔎 Search the web", "Ask about news or anything recent and I'll look it up, "
              "read the top results and tell you what they say, with links.",
              tools=("web_search",)),
    HelpEntry("🔗 Read a link", "Paste a link (a page, a YouTube video, a Reddit thread...) and "
              "ask me to summarise or explain it.",
              tools=("fetch_url",)),
    HelpEntry("🗂️ Find old messages", "Ask me to find something said earlier in the server or a "
              "channel, or paste a message link and ask what was being said around it.",
              tools=("search_messages", "read_messages_around")),
    HelpEntry("🧠 Remember things", "Tell me things to remember about you, or ask me to forget "
              "them. Memories are per server.",
              tools=("store_memory", "remove_memory", "clear_memories")),
    HelpEntry("🎭 Change personality", "Ask me to act differently, e.g. \"talk like a pirate\".",
              tools=("change_personality",)),
    HelpEntry("📊 Run polls", "Ask me for a poll on anything, then ask how it's going.",
              tools=("create_poll", "check_polls")),
    HelpEntry("📅 Events", "Ask me to make, move or cancel events, or what's coming up.",
              tools=("create_event", "list_events", "update_event"), requires=EVENTS),
    HelpEntry("🎨 Make images", "Ask me to draw a picture, or to change one I made "
              "(reply to it, or just say what to change).",
              tools=("generate_image",), requires=IMAGE),
    HelpEntry("💻 Code, files and charts", "Ask me to build or run something with code "
              "(scripts, charts, gifs, a file you attach...). The files come back here, and "
              "you can ask for changes after.",
              tools=("attach_file", "save_messages_to_file")),
    HelpEntry("⏰ Schedules and rules", "Ask me to schedule or follow up on something, or react to a word.",
              tools=("list_schedules", "get_schedule", "create_schedule", "schedule_followup",
                     "update_schedule", "delete_schedule",
                     "list_rules", "get_rule", "create_rule", "update_rule", "delete_rule"),
              requires=AUTOMATIONS),
    HelpEntry("🎙️ Talk in voice", "Ask me to join your voice channel, then say \"hey\" and my name "
              "to talk to me out loud. Ask me to leave when you're done.",
              tools=("join_voice_channel", "leave_voice_channel"), requires=VOICE),
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
    HelpEntry("/workspace", "The server's workspace for code tasks: status and reset (both "
              "need Manage Server), and live progress on or off.", command="workspace"),
    HelpEntry("/schedule", "Create, list, view, edit, delete, pause or resume schedules.",
              command="schedule", requires=AUTOMATIONS),
    HelpEntry("/rule", "Create, list, view, edit, delete, pause or resume message rules.",
              command="rule", requires=AUTOMATIONS),
    HelpEntry("/voice", "Join or leave a voice call, and set my wake phrase, voice and speed.",
              command="voice", requires=VOICE),
)

TIPS = (
    "• @mention me to get a reply; otherwise I only join in now and then (see /chance).\n"
    "• Send `!reset_history` to make me forget the conversation above it.\n"
    "• I can see images you attach."
)

HELP_TITLE = "🤖 What I can do"


def _shown(entry: HelpEntry, image_enabled: bool) -> bool:
    if entry.requires == IMAGE:
        return image_enabled
    if entry.requires == AUTOMATIONS:
        from classes.automation_policy import settings
        return settings()["enabled"]
    if entry.requires == EVENTS:
        from classes.event_format import events_enabled
        return events_enabled()
    if entry.requires == VOICE:
        from classes.voice_policy import settings
        return settings()["enabled"]
    return True


def _fields(name: str, lines: list) -> list:
    """`lines` as one field, or as several ("name (more)") when they would
    pass Discord's 1024-character field limit."""
    groups, current = [], []
    for line in lines:
        if current and len("\n".join(current + [line])) > MAX_FIELD_VALUE:
            groups.append(current)
            current = []
        current.append(line)
    if current:
        groups.append(current)
    return [{"name": name if i == 0 else f"{name} (more)", "value": "\n".join(group)}
            for i, group in enumerate(groups)]


def help_embed_data(image_enabled: bool) -> dict:
    """The /help embed as plain data (see classes.common.embed_from_data).
    One field per group keeps it compact on mobile; a group past Discord's
    field limit carries on in a second field."""
    ask = [e for e in ASK_ENTRIES if _shown(e, image_enabled)]
    commands = [e for e in COMMAND_ENTRIES if _shown(e, image_enabled)]
    return {
        "title": HELP_TITLE,
        "color": EMBED_COLOR,
        "fields": (_fields("Just ask me", [f"**{e.name}** — {e.description}" for e in ask])
                   + _fields("Slash commands", [f"`{e.name}` — {e.description}" for e in commands])
                   + [{"name": "Tips", "value": TIPS}]),
        "footer": "",
    }


# Brain tools that are its own bookkeeping, not something a person asks for.
INTERNAL_TOOLS = frozenset({"ask_user", "report"})


def documented_tools() -> set:
    return {tool for entry in ASK_ENTRIES for tool in entry.tools}


def documented_commands() -> set:
    return {entry.command for entry in COMMAND_ENTRIES}

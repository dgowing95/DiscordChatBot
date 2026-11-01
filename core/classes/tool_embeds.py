"""PURE (stdlib-only) how the embeds a tool posts while it works read.

Every one used to be titled "Tool Usage", with the action and its subject run
together in one sentence ("Visiting URL: https://..."). The title said nothing
and the sentence buried the part worth reading. So now:

- the TITLE names the action ("🔎 Web search", "🎨 Drawing an image"),
- the DESCRIPTION is only its subject (the query, the link, the prompt),
- the FOOTER, small and grey, carries any how-to hint,
- each kind of tool has its own accent colour, so a channel can be scanned.

The description stays self-contained on purpose: MessageHandler._format_group
sends the bot's own embeds back to the model as JSON in later prompts (minus
any fields), and the description is what tells it which tool it used on what.
Footers cost prompt tokens on every history message, so only the sandbox's
embeds have one.

Titles never carry user text, links or mentions: Discord renders none of those
in a title, and caps it at 256 characters. The sandbox run embed's titles and
reply hint live here; its closing embed (how a run ended, whether it can be
resumed) is built in sandbox_agent (sandbox_closing_embed, SANDBOX_STOP_TITLES),
beside the rules it reports on.
"""
import re
from urllib.parse import urlsplit

# Accent colours, one per kind of tool.
COLOR_WEB = 0x00B0F4       # cyan: searching and reading the web
COLOR_IMAGE = 0xEB459E     # pink: drawing
COLOR_MEMORY = 0x57F287    # green: remembering about a person
COLOR_MESSAGES = 0x5865F2  # blurple: reading this server's messages
COLOR_SANDBOX = 0x1ABC9C   # teal: the code sandbox at work
COLOR_PERSONA = 0x9B59B6   # purple: the bot's personality
COLOR_DONE = 0x99AAB5      # grey: a sandbox run that is over
COLOR_STOPPED = 0xED4245   # red: a sandbox run that was stopped

WEB_SEARCH = "🔎 Web search"
READING_PAGE = "🌐 Reading a page"
REMEMBERED = "💾 Remembered"
DRAWING = "🎨 Drawing an image"
EDITING = "🖌️ Editing an image"
SEARCHING_MESSAGES = "🔍 Searching messages"
SAVING_MESSAGES = "🗂️ Saving messages to a file"
READING_MESSAGES = "📜 Reading messages"
PERSONALITY = "🎭 Personality updated"
SANDBOX_STARTED = "🧵 Sandbox started"
SANDBOX_NEW = "🐳 Sandbox · new workspace"
SANDBOX_RESUMED = "🐳 Sandbox · resumed"

# Under the run's embed, in a thread only: messages there reach the run.
SANDBOX_REPLY_HINT = ("💬 Reply in this thread to change the task or ask a question. "
                      "No @mention needed.")

LINK_LABEL_CHARS = 80


def reading_page_title(part: int) -> str:
    return READING_PAGE if part <= 1 else f"{READING_PAGE} · part {part}"


def _plain(text: str) -> str:
    """Text safe as a masked link's label: brackets, which end the link
    early, and markdown characters removed. Not escaped: a backslash escape
    shows literally inside a link's label. An "_" between two letters
    ("Pluto_dwarf_planet") never formats and is kept; one at the edge of a
    word or beside another ("__main__") underlines, and no invisible
    character between them stops it (zero-width space, word joiner, soft
    hyphen and hair space all tried live), so it becomes the look-alike
    "ˍ" (U+02CD). The label is only shown: the link itself is untouched."""
    text = "".join(c for c in text if c not in "[]()<>\\*~|`")
    return re.sub(r"(?<![A-Za-z0-9])_|_(?![A-Za-z0-9])", "ˍ", text)


def link_label(url: str) -> str:
    """For a link, "example.com › gifs › some-page": where it goes, readable,
    without the query string. The raw URL when it does not parse."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return _plain(url[:LINK_LABEL_CHARS])
    host = (parts.hostname or "").removeprefix("www.")
    if not host:
        return _plain(url[:LINK_LABEL_CHARS])
    label = " › ".join([host] + [p for p in parts.path.split("/") if p])
    if len(label) > LINK_LABEL_CHARS:
        label = label[: LINK_LABEL_CHARS - 1] + "…"
    return _plain(label)


def page_link(url: str) -> str:
    """The page as a masked link: the readable label, clickable. Brackets in
    the URL are percent-encoded: a Wikipedia "Mercury_(planet)" link left
    them in, and Discord showed the whole thing as raw text (checked live;
    encoding only the ")" was not enough)."""
    target = url.replace(" ", "%20").replace("(", "%28").replace(")", "%29")
    return f"[{link_label(url)}]({target})"


def message_search_line(query: str, where: str, author: str = "") -> str:
    """What a message search looks for: “words” in #place · by someone."""
    line = f"“{query}” in {where}" if query else f"Newest messages in {where}"
    return line + (f" · by {author}" if author else "")


def trim(text: str, limit: int) -> str:
    """At most `limit` characters, cut at a word where there is one, with "…"."""
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    cut = text[: limit - 1]
    if " " in cut[limit // 2:]:
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip(",.;:") + "…"


def sandbox_task_text(task: str, files=(), limit: int = 1000) -> str:
    """The run embed's description: the task, and the chat files given to it."""
    text = trim(task, limit)
    if files:
        text += "\n📎 " + ", ".join(files)
    return text

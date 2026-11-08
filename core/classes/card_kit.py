"""PURE (stdlib only) the bot's cards: every status, tool and system message
it posts, as one design. classes/card.py turns a CardSpec into Discord's
layout components (a LayoutView: a Container of TextDisplays, Separators,
Sections, galleries and button rows) and sends it. How to build a new card,
or extend the kit, with examples: docs/cards.md.

How a card reads, top to bottom (only the body is required):

- the LABEL: a small grey line that says in plain words what this is
  ("🔎 Searching the web", "🔬 Research · step 1 of 3"). Its own block, so
  there is a gap before the title.
- the TITLE: `###` (or `##` for a panel such as /help).
- the BODY: blocks in order. Items in a list are split by a Gap (space, no
  line); groups by a Divider (a line).
- the FOOTER: a small grey line of details (turns, times, hints).

Colour says the STATE, not the kind of tool: the label and its emoji already
say what kind it is (decided with the maintainer, 2026-11-07).

A message holds at most 4000 characters of text and 40 components, so a
long card is SPLIT over several messages (split()): the same colour, the
label and title only on the first, the footer only on the last, and text cut
at a paragraph or line, never inside a code block. Nothing is cut short.

Mentions in a card's text notify (unlike in an embed), so card.py sends with
no mentions allowed unless a caller asks for them.

The bot reads its own messages back as history (MessageHandler._format_group),
where a card's container id says what it is (KIND_IDS): a NOTICE is shown to
the model as a status card, not as something it said; an ANNOUNCE (What's
New) is left out; a CARD (a question, a plan) reads as its text, as before.
"""
import re
from dataclasses import dataclass, field

# --- colours: the state of the thing ------------------------------------------
ACTIVE = 0x5865F2   # blurple: a notice, a panel, something happening now
WORKING = 0xFEE75C  # yellow: live work still going
OK = 0x57F287       # green: it worked / finished well
WARN = 0xFAA61A     # orange: partly done, busy, a heads-up
FAIL = 0xED4245     # red: failed, stopped, a destructive confirm
CLOSED = 0x80848E   # grey: over, answered, retired
ASK = 0xEB459E      # pink: waiting on a person

# --- Discord's limits (one message) -------------------------------------------
MAX_TEXT = 4000        # characters across every TextDisplay (Discord counts UTF-16 units)
TEXT_BUDGET = 3800     # what split() fills a message to: room to spare under MAX_TEXT
MAX_COMPONENTS = 40    # every component, nested ones included
MAX_BUTTON_LABEL = 80
MAX_URL = 512
MIN_SPLIT_ROOM = 200   # less room than this left in a card: start a new one

# --- what a card is, for reading it back --------------------------------------
NOTICE = "notice"
ANNOUNCE = "announce"
CARD = "card"
_MARKER_BASE = 0x0DCB0000  # well above the ids Discord numbers components with
KIND_IDS = {NOTICE: _MARKER_BASE + 1, ANNOUNCE: _MARKER_BASE + 2, CARD: _MARKER_BASE + 3}
_KINDS = {v: k for k, v in KIND_IDS.items()}


def kind_of_id(component_id) -> str | None:
    return _KINDS.get(component_id)


# --- blocks -------------------------------------------------------------------
@dataclass(frozen=True)
class Text:
    content: str


@dataclass(frozen=True)
class Gap:
    """Space between two items of a list, no line."""


@dataclass(frozen=True)
class Divider:
    """A line between two groups."""


@dataclass(frozen=True)
class Link:
    """A link button (grey, opens a page)."""
    label: str
    url: str
    emoji: str | None = None


@dataclass(frozen=True)
class Thumb:
    url: str


@dataclass(frozen=True)
class Section:
    """Up to 3 texts beside an accessory: a Link, a Thumb, or a ready-made
    discord button (opaque here)."""
    texts: tuple
    accessory: object


@dataclass(frozen=True)
class Buttons:
    """One row: Links, or ready-made discord buttons (opaque here)."""
    items: tuple


@dataclass(frozen=True)
class Gallery:
    """Pictures: "attachment://name" for a file sent with the message."""
    urls: tuple


@dataclass(frozen=True)
class FileRef:
    """A file sent with the message, shown inside the card."""
    name: str


@dataclass
class CardSpec:
    body: list = field(default_factory=list)
    label: str = ""
    title: str = ""
    footer: str = ""
    colour: int = ACTIVE
    kind: str = NOTICE
    heading: int = 3  # the title's size: 3 (###) for most, 2 for a panel
    # Keep each text whole when it fits one message: it starts the next
    # message instead of being cut to fill this one. For a live card (the
    # brain's commands): a text cut in two changes both messages each time it
    # grows, a whole one only its own.
    whole: bool = False


# --- the everyday cards -------------------------------------------------------

def _simple(label: str, text: str, colour: int, footer: str, title: str) -> CardSpec:
    return CardSpec(label=label, title=title, body=[Text(text)] if text else [],
                    footer=footer, colour=colour)


def info(label: str, text: str = "", footer: str = "", title: str = "") -> CardSpec:
    """A notice or a panel: something to know."""
    return _simple(label, text, ACTIVE, footer, title)


def done(label: str, text: str = "", footer: str = "", title: str = "") -> CardSpec:
    """It worked: a setting changed, something saved."""
    return _simple(label, text, OK, footer, title)


def heads_up(label: str, text: str = "", footer: str = "", title: str = "") -> CardSpec:
    """Not an error, but worth knowing: busy, too late, stopped by a restart."""
    return _simple(label, text, WARN, footer, title)


def error(text: str, label: str = "❌ Couldn't do that", footer: str = "", title: str = "") -> CardSpec:
    """It failed, or it is not allowed."""
    return _simple(label, text, FAIL, footer, title)


def finished(spec: CardSpec, footer: str | None = None, colour: int = CLOSED,
             label: str | None = None) -> CardSpec:
    """The same card with its buttons gone, in grey, so it never looks open
    (a card is never edited to no view: see classes/card.py)."""
    body = _trim_spacers([b for b in spec.body if not isinstance(b, Buttons)])
    return CardSpec(body=body, label=spec.label if label is None else label, title=spec.title,
                    footer=spec.footer if footer is None else footer, colour=colour,
                    kind=spec.kind, heading=spec.heading, whole=spec.whole)


def details(pairs) -> str:
    """Key/value lines: **Key** value."""
    return "\n".join(f"**{key}** {value}" for key, value in pairs if value not in (None, ""))


def listing(items) -> list:
    """Items of a list as blocks, a Gap between each."""
    blocks = []
    for item in items:
        if blocks:
            blocks.append(Gap())
        blocks.append(Text(item))
    return blocks


# --- markdown safety for text the bot did not write ---------------------------

def _escape_line(line: str) -> str:
    """A line that would turn into a heading, small text or a quote when it
    starts the text it is put in."""
    stripped = line.lstrip()
    if stripped.startswith(("#", "-#", ">")):
        return line[: len(line) - len(stripped)] + "\\" + stripped
    return line


def safe(text: str) -> str:
    """User or model text as it was written: no line turns into a heading,
    small text or a quote."""
    return "\n".join(_escape_line(line) for line in str(text or "").split("\n"))


def quote(text: str) -> str:
    """Text the bot did not write (a request, a memory, a personality), as a
    quote: every line, since "> " quotes only its own."""
    return "\n".join(f"> {_escape_line(line)}".rstrip() if line.strip() else ">"
                     for line in str(text or "").strip("\n").split("\n"))


def bold(text: str) -> str:
    text = " ".join(str(text or "").split()).replace("*", "\\*")
    return f"**{text}**" if text else ""


def small(text: str) -> str:
    """Small grey text: every line, since "-# " covers only its own."""
    return "\n".join(f"-# {line}" if line.strip() else "" for line in str(text or "").split("\n"))


# --- size of a block ----------------------------------------------------------

def units(text: str) -> int:
    """Text's length as Discord counts it: UTF-16 units (an emoji is two)."""
    return len(text.encode("utf-16-le")) // 2


def _chars(block) -> int:
    if isinstance(block, Text):
        return units(block.content)
    if isinstance(block, Section):
        return sum(units(t) for t in block.texts)
    return 0


def _components(block) -> int:
    if isinstance(block, Section):
        return 2 + len(block.texts)  # the section, its texts, its accessory
    if isinstance(block, Buttons):
        return 1 + len(block.items)
    return 1


def _spacer(block) -> bool:
    return isinstance(block, (Gap, Divider))


# --- splitting ------------------------------------------------------------------

def _fences_open(text: str) -> str | None:
    """The opening line of a code block left open at the end of `text`."""
    if text.count("```") % 2 == 0:
        return None
    lang = text.rsplit("```", 1)[1].split("\n", 1)[0].strip()
    return "```" + (lang if lang.isalnum() else "")


def split_text(text: str, limit: int) -> tuple[str, str]:
    """`text` cut so the first part is at most `limit` units (as Discord
    counts: units()): at a blank line, else a line, else a space, else
    anywhere. A code block open at the cut is closed there and opened again
    in the second part."""
    if units(text) <= limit:
        return text, ""
    room = limit  # in Python characters, which _cut slices by
    while True:
        head, tail = _cut(text, room)
        size = units(head)
        if size <= limit or room <= 1:
            return head, tail
        # Shrunk by how far over it went, as a ratio: characters and units
        # differ (an emoji is 1 and 2), so taking the units over off the
        # characters cut text of emoji into dozens of small messages.
        room = max(1, room * limit // size)


def _cut(text: str, limit: int) -> tuple[str, str]:
    """split_text's cut, in Python characters."""
    room = max(limit - 12, 1)  # for closing and reopening a code block
    window = text[:room]
    cut = -1
    for sep in ("\n\n", "\n", " "):
        cut = window.rfind(sep)
        if cut > room // 3:
            break
    if cut <= 0:
        cut = room
    head, tail = text[:cut].rstrip(), text[cut:].lstrip("\n").lstrip(" ")
    fence = _fences_open(head)
    if fence is not None:
        head += "\n```"
        tail = f"{fence}\n{tail}"
    return head, tail


def _header(spec: CardSpec) -> list:
    blocks = []
    if spec.label:
        blocks.append(Text(small(spec.label)))
    if spec.title:
        blocks.append(Text(f"{'#' * max(1, min(3, spec.heading))} {spec.title}"))
    return blocks


def _trim_spacers(blocks: list) -> list:
    while blocks and _spacer(blocks[0]):
        blocks.pop(0)
    while blocks and _spacer(blocks[-1]):
        blocks.pop()
    return blocks


def split(spec: CardSpec, max_text: int = TEXT_BUDGET, max_components: int = MAX_COMPONENTS) -> list[list]:
    """The card as one block list per message. Each fits Discord's limits
    (with its container); the header goes on the first, the footer on the
    last, and no text is dropped."""
    room_components = max_components - 1  # the container
    cards: list[list] = []
    current: list = []
    chars = comps = 0

    def flush():
        nonlocal current, chars, comps
        current = _trim_spacers(current)
        if current:
            cards.append(current)
        current, chars, comps = [], 0, 0

    def add(block):
        nonlocal chars, comps
        while True:
            size, count = _chars(block), _components(block)
            if chars + size <= max_text and comps + count <= room_components:
                if not (_spacer(block) and not current):
                    current.append(block)
                    chars += size
                    comps += count
                return
            if isinstance(block, Text) and comps + 1 <= room_components \
                    and max_text - chars >= MIN_SPLIT_ROOM \
                    and not (spec.whole and current and size <= max_text):
                head, tail = split_text(block.content, max_text - chars)
                if head:
                    current.append(Text(head))
                    chars += units(head)
                    comps += 1
                flush()
                block = Text(tail)
                if not tail:
                    return
                continue
            if not current:
                # Alone it still does not fit: a Text too long for one card
                # (handled above) or a block Discord would refuse anyway.
                if isinstance(block, Text):
                    head, tail = split_text(block.content, max_text)
                    current.append(Text(head))
                    flush()
                    block = Text(tail)
                    continue
                current.append(block)
                chars += size
                comps += count
                return
            flush()
            if _spacer(block):
                return

    for block in _header(spec) + list(spec.body):
        add(block)
    if spec.footer:
        add(Text(small(spec.footer)))
    flush()
    return cards or [[Text("​")]]


# --- reading a card back --------------------------------------------------------

def plain_line(line: str) -> str:
    """A card's line without its decoration (small text, heading, quote)."""
    line = line.strip()
    for prefix in ("-# ", "### ", "## ", "# ", ">>> ", "> "):
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    if line == ">":
        return ""
    return line


def plain(texts) -> str:
    """A card's texts as plain lines, for a prompt: no decoration, no blank lines."""
    lines = []
    for text in texts:
        for line in str(text or "").split("\n"):
            line = plain_line(line)
            if line:
                lines.append(line)
    return "\n".join(lines)


_CODE_BLOCK = re.compile(r"```.*?```", re.DOTALL)
# A live command's own status line (brain_progress): part of its log.
_LOG_LINE = re.compile(r"^-# (✅ Finished|❌ Failed|⏳ Running|⏳ Starting|⌨ typed in)")
# Most of one text of a notice that goes into a prompt. The card shows it all;
# the model only needs what the card was about (a status card's request, a
# helper's brief: the request is in the history already, as a message).
HISTORY_TEXT_CHARS = 1000


def notice_for_history(author: str, texts) -> str:
    """How a notice card reads in a prompt: plainly a status card, so the model
    does not take it for something it said (it copies its earlier replies).
    Code blocks and command status lines (a live run's commands and output)
    are left out, and a long text is cut (HISTORY_TEXT_CHARS): they would
    fill later prompts and push out the real conversation, as the embed they
    used to be in never did. "" when nothing is left (a message that only
    continued the commands)."""
    kept = []
    for text in texts:
        text = _CODE_BLOCK.sub("", str(text or ""))
        text = "\n".join(line for line in text.split("\n") if not _LOG_LINE.match(line.strip()))
        if len(text) > HISTORY_TEXT_CHARS:
            text = text[:HISTORY_TEXT_CHARS - 1].rstrip() + "…"
        kept.append(text)
    body = plain(kept)
    return f"[Status card posted by '{author}', not a chat message]\n{body}" if body else ""

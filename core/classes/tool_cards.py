"""PURE (stdlib-only) what the cards a tool posts while it works say.

Every one used to be an embed titled "Tool Usage", with the action and its
subject run together in one sentence ("Visiting URL: https://..."). Now each
is a card (classes/card_kit.py):

- its LABEL says in plain words what is happening ("🔎 Searching the web"),
- its body is only the subject (the query, the page, the prompt),
- its footer, small and grey, carries any detail or hint,
- its colour is the state (card_kit), not the kind of tool.

The body stays self-contained on purpose: MessageHandler._format_group sends
the bot's own notice cards back to the model as plain lines in later
prompts, and they are what tell it which tool it used on what.

Labels never carry links: a link reads badly in small text. The brain's
status card labels live here too (classes/brain_progress.py posts it).
"""
import re
from urllib.parse import urlsplit

from classes import card_kit as K
from classes.card_kit import CardSpec, Divider, Gallery, Link, Section, Text

WEB_SEARCH = "🔎 Searching the web"
WEB_SEARCHED = "🔎 Searched the web"
READING_PAGE = "🌐 Reading a page"
REMEMBERED = "💾 Remembered"
DRAWING = "🎨 Drawing an image"
EDITING = "🖌️ Editing an image"
IMAGE = "🎨 Image"
EDITED_IMAGE = "🖌️ Edited image"
SEARCHING_MESSAGES = "🔍 Searching this server's messages"
SAVING_MESSAGES = "🗂️ Saving messages to a file"
READING_MESSAGES = "📜 Reading messages"
PERSONALITY = "🎭 New personality"
SKILL_SAVED = "📘 New skill saved"
SKILL_UPDATED = "📗 Skill updated"
SKILL_USED = "📖 Using a skill"
SKILL_DELETED = "🗑️ Skill deleted"
SKILL_RETIRED = "🗑️ Skill retired"
SKILL_RESTORED = "♻️ Skill restored"
SKILL_REVERTED = "⏪ Skill reverted"
SKILL_FOOTER_REFLECTION = "after reviewing the last job"
SKILL_FOOTER_CURATOR = "daily tidy-up"

# The brain's status card, posted only once it uses the workspace (a command
# or a file): a search or a poll shows only its own card.
BRAIN_WORKING = "🧠 Working on it"
BRAIN_DONE = "🧠 Done"
BRAIN_STOPPED = "🧠 Stopped"
BRAIN_NOT_FINISHED = "🧠 Not finished"

IMAGE_WAIT = "⏳ This usually takes 10–30 seconds"
IMAGE_HINT = "Reply to this to change it"
GENERATED_IMAGE = "generated-image.png"

LINK_LABEL_CHARS = 80
HELPER_LINE_CHARS = 80


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


def _full_url(url: str) -> str:
    # fetch_url takes a bare "example.com/page" (web_policy.normalize_url
    # reads it as https), which would be neither a readable label nor a link.
    url = (url or "").strip()
    return url if "://" in url else "https://" + url


def link_target(url: str) -> str:
    """The URL as a masked link's target. Brackets are percent-encoded: a
    Wikipedia "Mercury_(planet)" link left them in, and Discord showed the
    whole thing as raw text (checked live; encoding only the ")" was not
    enough)."""
    return _full_url(url).replace(" ", "%20").replace("(", "%28").replace(")", "%29")


def page_link(url: str) -> str:
    """The page as a masked link: the readable label, clickable."""
    url = _full_url(url)
    return f"[{link_label(url)}]({link_target(url)})"


def host_link(url: str) -> str:
    """A short masked link: just the site's name."""
    url = _full_url(url)
    try:
        host = (urlsplit(url).hostname or url).removeprefix("www.")
    except ValueError:
        host = url
    return f"[{_plain(host)}]({link_target(url)})"


def message_search_line(query: str, where: str, author: str = "") -> str:
    """What a message search looks for: **“words”** in #place · from someone."""
    line = f"{K.bold(f'“{query}”')} in {where}" if query else f"Newest messages in {where}"
    return line + (f" · from {author}" if author else "")


def trim(text: str, limit: int) -> str:
    """At most `limit` characters, cut at a word where there is one, with "…".
    Only for one-line summaries of live work (a helper's last action);
    content is never trimmed."""
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    cut = text[: limit - 1]
    if " " in cut[limit // 2:]:
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip(",.;:") + "…"


# --- the cards ----------------------------------------------------------------

def web_search_card(query: str, sources=None, read: bool = True) -> CardSpec:
    """While searching (sources None), then what it read: the same card
    edited, with each source a link."""
    body = [Text(K.bold(query))]
    if sources is None:
        return CardSpec(label=WEB_SEARCH, body=body)
    if sources:
        what = "page" if read else "result"
        count = f"{'Read' if read else 'Found'} {len(sources)} {what}{'' if len(sources) == 1 else 's'}"
        body += [Divider(), Text(K.small(" · ".join([count] + [host_link(u) for _t, u in sources])))]
    else:
        body += [Divider(), Text(K.small("Nothing found"))]
    return CardSpec(label=WEB_SEARCHED, body=body)


def can_link(url: str) -> bool:
    """Whether Discord takes `url` for a link button: http(s), at most
    MAX_URL characters. It refuses the whole message otherwise."""
    return url.startswith(("https://", "http://")) and len(url) <= K.MAX_URL


def page_card(url: str, part: int = 1, title: str = "") -> CardSpec:
    """The page being read, with an Open button (when Discord would take the
    link for one). Its own title once read."""
    link = page_link(url)
    text = f"{K.bold(title)}\n{K.small(link)}" if title else link
    texts = (K.small(reading_page_title(part)), text)
    target = link_target(url)
    if not can_link(target):
        return CardSpec(body=[Text("\n".join(texts))])
    return CardSpec(body=[Section(texts, Link("Open", target))])


def memory_card(fact: str, who: str = "") -> CardSpec:
    label = f"{REMEMBERED} about {who}" if who else REMEMBERED
    return CardSpec(label=label, body=[Text(K.quote(fact))], colour=K.OK)


def personality_card(personality: str) -> CardSpec:
    return CardSpec(label=PERSONALITY, body=[Text(K.quote(personality))],
                    footer="Change it back any time with /system", colour=K.OK)


def message_search_card(label: str, line: str, footer: str = "") -> CardSpec:
    return CardSpec(label=label, body=[Text(line)], footer=footer)


def drawing_card(prompt: str, editing: bool = False) -> CardSpec:
    return CardSpec(label=EDITING if editing else DRAWING, body=[Text(K.safe(prompt))],
                    footer=IMAGE_WAIT, colour=K.WORKING)


def image_card(prompt: str, editing: bool = False, filename: str = GENERATED_IMAGE) -> CardSpec:
    """The finished picture inside its card, the prompt under it."""
    return CardSpec(label=EDITED_IMAGE if editing else IMAGE,
                    body=[Gallery((f"attachment://{filename}",))],
                    footer=f"{' '.join(str(prompt or '').split())}\n{IMAGE_HINT}", colour=K.OK)


_SKILL_COLOURS = {SKILL_SAVED: K.OK, SKILL_UPDATED: K.OK, SKILL_RESTORED: K.OK, SKILL_REVERTED: K.OK,
                  SKILL_DELETED: K.CLOSED, SKILL_RETIRED: K.CLOSED}


def skill_card(label: str, name: str, detail: str = "", footer: str | None = None,
               version: int | None = None) -> CardSpec:
    """A skill event: the label, the skill's name as the title, what is worth
    saying under it. Self-contained, since the bot's own notices come back to
    the model as history."""
    if version is not None:
        label = f"{label} · v{version}"
    detail = " ".join(str(detail or "").split())
    return CardSpec(label=label + (f" · {footer}" if footer else ""), title=name,
                    body=[Text(K.safe(detail[:1].upper() + detail[1:]))] if detail else [],
                    colour=_SKILL_COLOURS.get(label.split(" · ")[0], K.ACTIVE))


# --- a workflow helper's one line -------------------------------------------------

def _program(cmd: str) -> str:
    """What a shell command runs, in a word: "python3" for
    "cd /workspace/jobs/x && python3 - <<'EOF'"."""
    first = cmd.strip().splitlines()[0] if cmd.strip() else ""
    last = re.split(r"&&|\|\||;|\|", first)[-1].strip()
    words = [w for w in last.split() if "=" not in w or w.startswith("-")]
    return (words[0].rsplit("/", 1)[-1] if words else "") or "a command"


_PROGRAMS = {"python": "a Python script", "python3": "a Python script", "pip": "pip",
             "node": "a Node script", "bash": "a shell script", "sh": "a shell script",
             "curl": "a download", "wget": "a download"}


def helper_line(tool: str, args: dict) -> str | None:
    """One short line for a helper's status card: what it just started, in
    plain words. None for a call that says nothing new (a poll, its report)."""
    if tool in ("report", "write_stdin"):
        return None
    if tool == "web_search":
        line = f"🔎 Searching: {args.get('search_request') or ''}"
    elif tool == "fetch_url":
        url = str(args.get("url") or "")
        host = urlsplit(url if "://" in url else "https://" + url).hostname or url
        line = f"🌐 Reading {host.removeprefix('www.')}"
    elif tool == "search_messages":
        line = f"🔍 Searching messages: {args.get('query') or ''}"
    elif tool == "read_messages_around":
        line = "📜 Reading messages"
    elif tool == "save_messages_to_file":
        line = "🗂️ Saving messages"
    elif tool == "exec_command":
        program = _program(str(args.get("cmd") or ""))
        line = f"💻 Running {_PROGRAMS.get(program, program)}"
    elif tool == "attach_file":
        line = f"📎 Attaching {args.get('path') or ''}"
    elif tool == "view_image":
        line = "👀 Looking at a picture"
    else:
        line = f"🔧 {tool.replace('_', ' ')}"
    return trim(_plain(line), HELPER_LINE_CHARS)


def helper_ending(ok: bool, note: str | None, found: int) -> str:
    """How a helper ended: how it went and how many facts it kept."""
    head = note or ("✅ Finished" if ok else "⚠️ Not finished")
    return f"{head} · found {found} fact{'' if found == 1 else 's'}"


def work_task_text(task: str, files=()) -> str:
    """The status card's request, quoted, and the chat files given to it."""
    text = K.quote(task)
    if files:
        text += "\n" + K.small("📎 " + ", ".join(files))
    return text

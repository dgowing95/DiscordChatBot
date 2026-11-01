"""Discord message search and reading around a message: the query sent and
how results read in a prompt (PURE, stdlib only).

The search_messages tool (message_search_tools.py) calls Discord's Search
Guild Messages endpoint (GET /guilds/{id}/messages/search, opened to bots on
2026-03-19; it needs Read Message History and the Message Content intent).
discord.py does not wrap it, so the request and its response are plain
dicts, and everything that can be decided from them lives here, tested in
core/tests/message_search_tests.py.

Shapes worth knowing (Discord's docs; check them again if this breaks):
  * `messages` is an array of ARRAYS, one message each (they used to carry
    surrounding context; they no longer do).
  * a server Discord has not indexed yet answers 202 with code 110000 and no
    `messages`. discord.py treats every 2xx as success, so that arrives as a
    normal body, not an exception.
  * booleans must be the strings "true"/"false" (yarl refuses Python bools).
  * thread names come in `threads`: an archived thread is often not in
    discord.py's cache.

Reading around a message (read_messages_around) uses the ordinary
GET /channels/{id}/messages with `before=` / `after=`, plus the message
itself. Those messages have the same shape as search hits, so the same
helpers format them. Each list comes back newest first, whichever of
before/after was asked, so they are sorted by id here.
"""
import re
from datetime import datetime

DEFAULT_LIMIT = 10
MAX_LIMIT = 25
# One message's text is cut at this: enough to recognise it, and 25 of them
# still fit beside the rest of a prompt.
CONTENT_CHARS = 300
# Searches in one reply. Discord does not publish this endpoint's rate limit,
# and a model that finds nothing tends to retry with near-identical words.
MAX_CALLS = 3
INDEX_CODE = 110000

# Reading around a message: how many messages on each side by default, and
# at most (so one call shows at most 25, like a search). A model paging back
# through a long conversation calls it again and again, so it has its own
# cap per reply.
CONTEXT_DEFAULT = 5
CONTEXT_MAX_SIDE = 12
MAX_CONTEXT_CALLS = 4
_LINK = re.compile(r"https?://(?:(?:ptb|canary)\.)?discord(?:app)?\.com/channels/"
                   r"(\d+|@me)/(\d+)/(\d+)")
_BARE_ID = re.compile(r"^\d{15,25}$")


def clamp_limit(limit) -> int:
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        return DEFAULT_LIMIT
    return max(1, min(MAX_LIMIT, limit))


def search_params(query: str, channel_id=None, author_id=None, limit=DEFAULT_LIMIT,
                  include_nsfw: bool = False) -> list:
    """The query string, as (key, value) pairs. Best matches first when there
    are words to match, else newest first (listing someone's messages)."""
    params = [("limit", str(clamp_limit(limit))),
              ("include_nsfw", "true" if include_nsfw else "false")]
    query = (query or "").strip()[:1024]
    if query:
        params += [("content", query), ("sort_by", "relevance")]
    else:
        params += [("sort_by", "timestamp"), ("sort_order", "desc")]
    if channel_id:
        params.append(("channel_id", str(channel_id)))
    if author_id:
        params.append(("author_id", str(author_id)))
    return params


def not_indexed(body) -> bool:
    """Discord has not finished indexing the server: no results yet."""
    return not isinstance(body, dict) or body.get("code") == INDEX_CODE or "messages" not in body


def flatten(body: dict) -> list:
    """One message dict per hit, in Discord's order."""
    hits = []
    for group in body.get("messages") or []:
        if isinstance(group, list):
            chosen = next((m for m in group if isinstance(m, dict) and m.get("hit")), None)
            chosen = chosen or next((m for m in group if isinstance(m, dict)), None)
        else:
            chosen = group if isinstance(group, dict) else None
        if chosen:
            hits.append(chosen)
    return hits


def _author(message: dict) -> str:
    author = message.get("author") or {}
    member = message.get("member") or {}
    return (member.get("nick") or author.get("global_name") or author.get("username")
            or "someone")


def _when(message: dict) -> str:
    try:
        stamp = datetime.fromisoformat(str(message.get("timestamp")).replace("Z", "+00:00"))
    except ValueError:
        return "at an unknown time"
    return f"<t:{int(stamp.timestamp())}:f>"


def _content(message: dict) -> str:
    text = " ".join(str(message.get("content") or "").split())
    if len(text) > CONTENT_CHARS:
        text = text[:CONTENT_CHARS].rstrip() + "…"
    extras = [f"[attached: {a.get('filename')}]" for a in message.get("attachments") or []
              if a.get("filename")]
    extras += [f"[embed: {e.get('title')}]" for e in message.get("embeds") or [] if e.get("title")]
    poll = (message.get("poll") or {}).get("question") or {}
    if poll.get("text"):
        extras.append(f"[poll: {poll['text']}]")
    return " ".join(part for part in [text] + extras if part) or "(no text)"


def _reply_note(message: dict) -> str:
    replied = message.get("referenced_message")
    if isinstance(replied, dict):
        return f" (replying to {_author(replied)})"
    return ""


def message_link(guild_id, channel_id, message_id) -> str:
    return f"https://discord.com/channels/{guild_id}/{channel_id}/{message_id}"


def format_results(body: dict, guild_id, channel_names: dict, query: str = "",
                   where: str = "this server") -> str:
    """The results as the model reads them: one line per message with who,
    when (a Discord timestamp, so each reader sees their own time zone),
    where, a jump link and the text."""
    hits = flatten(body)
    names = dict(channel_names)
    for thread in body.get("threads") or []:
        if thread.get("id") and thread.get("name"):
            names.setdefault(str(thread["id"]), thread["name"])
    total = body.get("total_results", len(hits))
    if not hits:
        lines = [f"No messages found in {where}" + (f" for \"{query}\"." if query else ".")]
    else:
        order = "best matches first" if query else "newest first"
        lines = [f"{total} message{'s' if total != 1 else ''} found in {where}; showing "
                 f"{len(hits)}, {order}. Share a message's link so people can jump to it; "
                 f"to see the conversation around one, pass its link to read_messages_around."]
        for message in hits:
            channel_id = str(message.get("channel_id", ""))
            channel = names.get(channel_id)
            place = f"#{channel}" if channel else f"<#{channel_id}>"
            link = message_link(guild_id, channel_id, message.get("id"))
            lines.append(f"- {_author(message)}, {_when(message)}, in {place}: "
                         f"{_content(message)}\n  {link}")
    if body.get("doing_deep_historical_index"):
        lines.append("(Discord is still indexing this server's older messages, so some may "
                     "be missing.)")
    return "\n".join(lines)


def parse_message_link(text):
    """(guild id, channel id, message id) from a message link, or
    (None, None, message id) for a bare id, which means the channel being
    talked in. None when it is neither. A DM link (@me) is refused by the
    caller, so its guild comes back as 0."""
    raw = (text or "").strip().strip("<>")
    match = _LINK.search(raw)
    if match:
        guild = 0 if match.group(1) == "@me" else int(match.group(1))
        return guild, int(match.group(2)), int(match.group(3))
    if _BARE_ID.match(raw):
        return None, None, int(raw)
    return None


def clamp_side(value) -> int:
    """Messages wanted on one side: 0 to CONTEXT_MAX_SIDE."""
    try:
        value = int(value)
    except (TypeError, ValueError):
        return CONTEXT_DEFAULT
    return max(0, min(CONTEXT_MAX_SIDE, value))


def _sorted(messages) -> list:
    unique = {}
    for message in messages or []:
        if isinstance(message, dict) and str(message.get("id", "")).isdigit():
            unique[str(message["id"])] = message
    return sorted(unique.values(), key=lambda m: int(m["id"]))


def format_context(anchor: dict, older, newer, guild_id, channel_id, channel_name: str,
                   wanted_before: int, wanted_after: int) -> str:
    """The messages around `anchor`, oldest first, the anchor marked with ►
    and given its link. The last lines say how to read further back or
    forward (call again from the oldest or newest message), or that there is
    nothing more that way: the model only pages if the result tells it how."""
    anchor_id = str(anchor.get("id"))
    older = [m for m in _sorted(older) if int(m["id"]) < int(anchor_id)]
    newer = [m for m in _sorted(newer) if int(m["id"]) > int(anchor_id)]
    shown = older + [anchor] + newer
    lines = [f"Messages in #{channel_name} around the one asked for, oldest first "
             f"(► marks it):"]
    for message in shown:
        mark = "►" if str(message.get("id")) == anchor_id else "-"
        lines.append(f"{mark} {_author(message)}{_reply_note(message)}, {_when(message)}: "
                     f"{_content(message)}")
        if mark == "►":
            lines.append(f"  {message_link(guild_id, channel_id, anchor_id)}")
    first = message_link(guild_id, channel_id, shown[0].get("id"))
    last = message_link(guild_id, channel_id, shown[-1].get("id"))
    if wanted_before and len(older) < wanted_before:
        lines.append("Nothing earlier: this is the start of the channel.")
    else:
        lines.append(f"Earlier messages: call read_messages_around with message={first}, "
                     f"before={CONTEXT_MAX_SIDE}, after=0.")
    if wanted_after and len(newer) < wanted_after:
        lines.append("Nothing later: these are the channel's newest messages.")
    else:
        lines.append(f"Later messages: call read_messages_around with message={last}, "
                     f"before=0, after={CONTEXT_MAX_SIDE}.")
    return "\n".join(lines)


NOT_INDEXED_TEXT = ("Discord hasn't finished indexing this server's messages for search yet, "
                    "so nothing can be found right now. Tell the user to try again in a little "
                    "while.")
TOO_MANY_TEXT = ("The search limit for this reply has been reached. Answer with what the "
                 "earlier searches found.")
CONTEXT_TOO_MANY_TEXT = ("The limit for reading messages in this reply has been reached. "
                         "Answer with what has been read so far.")

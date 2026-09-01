"""Pure helpers for cleaning LLM output before it is sent to Discord.

A reasoning model delivers its internal thinking in one of two shapes, and
this module handles both:

  * Out of band, in the OpenAI-compatible ``reasoning_content`` field. This
    is what llama.cpp does by default (``--reasoning-format auto``) for a
    template with thinking enabled, and it is what our server actually
    returns: the visible ``content`` has no tags in it at all. The Agents
    SDK turns that field into a separate reasoning item on the run result,
    so it never reaches the final output text - see
    extract_reasoning_items, which pulls it back out.
  * Inline in the answer text, wrapped in an opening and a closing "think"
    tag (some models insert a tab after the angle bracket). This is what
    ``--reasoning-format none`` produces. strip_thinking removes those
    blocks from the visible answer and extract_thinking captures them.

Either way the captured reasoning is formatted as spoiler-hidden Discord
messages (format_thinking_for_discord) so it can still be sent, collapsed
by default.

Kept dependency-free (stdlib only) so it can be imported and tested
without the discord/agent SDKs - extract_reasoning_items duck-types the
SDK's run items rather than importing them.
"""
import re
from textwrap import wrap

# Building the tag fragments as plain strings keeps this file easy to read
# and lets us match the tag with or without the tab after the bracket.
_T = "\\" + "t?"          # optional tab, as a regex escape sequence
OPEN = "<" + _T + "think>"
CLOSE = "</" + _T + "think>"

# Complete OPEN ... CLOSE blocks (possibly several, interleaved with content).
_THINKING_BLOCK_RE = re.compile(OPEN + r".*?" + CLOSE, re.DOTALL | re.IGNORECASE)
# Same, but capturing the inner reasoning text (for extract_thinking).
_THINKING_BLOCK_CAPTURE_RE = re.compile(OPEN + r"(.*?)" + CLOSE, re.DOTALL | re.IGNORECASE)
# An opening tag with no matching close tag: drop it and everything after,
# since whatever follows is (by definition) still part of the reasoning
# (typical when a response is truncated mid-thought at max_tokens).
_UNCLOSED_THINKING_RE = re.compile(OPEN + r"[\s\S]*$", re.IGNORECASE)
# Same, but capturing the trailing reasoning text (for extract_thinking).
_UNCLOSED_THINKING_CAPTURE_RE = re.compile(OPEN + r"([\s\S]*)$", re.IGNORECASE)
# A close tag with no open tag: drop just the tag.
_LONELY_CLOSE_TAG_RE = re.compile(CLOSE, re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Remove reasoning blocks (open/close "think" tags) from an LLM response.

    Handles the cases the old single-pass regex got wrong:
      * several reasoning blocks interleaved with real content
      * repeated open/close pairs inside one block
      * an unterminated opening tag (response cut off mid-thought)
      * a stray closing tag with no matching open tag
      * tag variants with/without the tab after the bracket
    """
    text = _THINKING_BLOCK_RE.sub("", text)
    text = _UNCLOSED_THINKING_RE.sub("", text)
    text = _LONELY_CLOSE_TAG_RE.sub("", text)
    return text.strip()


def extract_thinking(text: str) -> str:
    """Collect the reasoning captured in open/close "think" tags, in order.

    Mirrors strip_thinking's tag handling (complete blocks, then an
    unterminated trailing open tag) but returns the captured text instead
    of discarding it, so it can still be sent to Discord (just hidden).
    """
    parts = [m.group(1) for m in _THINKING_BLOCK_CAPTURE_RE.finditer(text)]
    remainder = _THINKING_BLOCK_RE.sub("", text)
    unclosed = _UNCLOSED_THINKING_CAPTURE_RE.search(remainder)
    if unclosed:
        parts.append(unclosed.group(1))
    return "\n\n".join(p.strip() for p in parts if p.strip())


def extract_reasoning_items(items) -> str:
    """Collect reasoning text from an Agents SDK run's ``new_items``.

    When the server returns reasoning out of band (``reasoning_content``),
    the SDK's chat-completions converter drops it into its own reasoning
    item rather than into the assistant message, so it is absent from
    ``final_output`` and extract_thinking has nothing to find. This walks
    the run items and gathers it back.

    Every turn of a tool-calling run contributes its own reasoning item, so
    the result is joined in order rather than taken from the last one.

    Duck-typed on purpose (getattr, no SDK imports) to keep this module
    stdlib-only: an item is a reasoning item when its ``type`` says so, and
    its payload lives on ``raw_item`` in one of two normalized fields -
    ``content`` (per-block thinking text) or ``summary`` (where the
    converter puts ``reasoning_content``). Prefer ``content`` when both are
    populated; they are two views of the same thinking, not two halves.
    """
    parts: list = []
    for item in items or []:
        if getattr(item, "type", None) != "reasoning_item":
            continue
        raw = getattr(item, "raw_item", None)
        if raw is None:
            continue
        blocks = getattr(raw, "content", None) or getattr(raw, "summary", None) or []
        for block in blocks:
            text = getattr(block, "text", None)
            if text and text.strip():
                parts.append(text.strip())
    return "\n\n".join(parts)


# Discord spoiler markup (||...||) collapses text behind a "click to reveal"
# bar - the closest thing Discord has to a closed-by-default dropdown for a
# plain message. Wrapping it in a code block too keeps whitespace/formatting
# intact. Overhead per chunk: "||```\n" + "\n```||".
_SPOILER_CODE_PREFIX = "||```\n"
_SPOILER_CODE_SUFFIX = "\n```||"
_SPOILER_CODE_OVERHEAD = len(_SPOILER_CODE_PREFIX) + len(_SPOILER_CODE_SUFFIX)
DISCORD_CHUNK_SIZE = 2000
# A literal ``` inside the reasoning would close our code fence early; break
# it up with a zero-width space so it renders the same but can't terminate it.
_TRIPLE_BACKTICK_ESCAPE = "``​`"
# A tool-calling run reasons on every turn, so the collected thinking can be
# far longer than the single block a plain answer produces. Cap the follow-up
# messages so one long sandbox run cannot flood a channel; the tail is the
# part that explains the final answer, so keep that end and note the cut.
MAX_THINKING_CHUNKS = 4
_TRUNCATION_NOTE = "[... earlier reasoning truncated ...]\n"


def format_thinking_for_discord(thinking: str) -> list:
    """Wrap reasoning text into ready-to-send Discord message bodies.

    Each chunk is a spoiler-hidden code block, sized to fit Discord's
    2000-char message limit including the spoiler/code markup. At most
    MAX_THINKING_CHUNKS are returned (the last ones, with a note marking
    what was dropped). Returns [] when there is nothing to show.
    """
    thinking = thinking.strip()
    if not thinking:
        return []
    safe = thinking.replace("```", _TRIPLE_BACKTICK_ESCAPE)
    max_len = DISCORD_CHUNK_SIZE - _SPOILER_CODE_OVERHEAD
    chunks = [c for c in wrap(safe, max_len, break_long_words=False, replace_whitespace=False) if c]
    if len(chunks) > MAX_THINKING_CHUNKS:
        chunks = chunks[-MAX_THINKING_CHUNKS:]
        chunks[0] = _TRUNCATION_NOTE + chunks[0][: max_len - len(_TRUNCATION_NOTE)]
    return [_SPOILER_CODE_PREFIX + chunk + _SPOILER_CODE_SUFFIX for chunk in chunks]


def filter_response(text: str, mention: str = "") -> str:
    """Clean a full LLM response before it is split into Discord-sized chunks.

    Drops mention artifacts, normalises whitespace and removes reasoning
    blocks. Pure function - safe to unit test without a live client.
    """
    if mention:
        text = text.replace("<@" + str(mention) + ">", "")
        text = re.sub(r"^<@.*:", "", text, flags=re.DOTALL)
    text = re.sub(r"\n\s*\n", "\n\n", text, flags=re.DOTALL)
    text = re.sub(r"Message from.*?:", "", text, flags=re.DOTALL)
    text = strip_thinking(text)
    return text.strip()


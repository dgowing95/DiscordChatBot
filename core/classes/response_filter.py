"""Pure helpers for cleaning LLM output before it is sent to Discord.

LLM responses may contain internal reasoning, wrapped in an opening and a
closing "think" tag; some models insert a tab character
after the angle bracket. This module strips those blocks so only the
final answer reaches Discord.

Kept dependency-free (stdlib only) so it can be imported and tested
without the discord/agent SDKs.
"""
import re

# Building the tag fragments as plain strings keeps this file easy to read
# and lets us match the tag with or without the tab after the bracket.
_T = "\\" + "t?"          # optional tab, as a regex escape sequence
OPEN = "<" + _T + "think>"
CLOSE = "</" + _T + "think>"

# Complete OPEN ... CLOSE blocks (possibly several, interleaved with content).
_THINKING_BLOCK_RE = re.compile(OPEN + r".*?" + CLOSE, re.DOTALL | re.IGNORECASE)
# An opening tag with no matching close tag: drop it and everything after,
# since whatever follows is (by definition) still part of the reasoning
# (typical when a response is truncated mid-thought at max_tokens).
_UNCLOSED_THINKING_RE = re.compile(OPEN + r"[\s\S]*$", re.IGNORECASE)
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


import re
from textwrap import wrap

import pytest

from core.classes.response_filter import (
    strip_thinking,
    filter_response,
    extract_thinking,
    extract_reasoning_items,
    format_thinking_for_discord,
    DISCORD_CHUNK_SIZE as _FMT_CHUNK_SIZE,
    MAX_THINKING_CHUNKS,
)

# Tags are built via chr() so the literals in tests survive any editor or
# pipeline that mangles "tab-after-bracket" sequences.
_L, _G, _S, _T = chr(60), chr(62), chr(47), chr(9)


def _think_open(tab=True):
    return _L + (_T if tab else "") + "think" + _G


def _think_close(tab=True):
    return _L + _S + (_T if tab else "") + "think" + _G


def _think(inner, tab=True):
    return _think_open(tab) + inner + _think_close(tab)


# --------------------------- strip_thinking ---------------------------

def test_removes_basic_block_with_tab_variant():
    text = "hello " + _think("let me reason...", tab=True) + " world"
    assert strip_thinking(text) == "hello  world"


def test_removes_basic_block_plain_variant():
    text = "hello " + _think("let me reason...", tab=False) + " world"
    assert strip_thinking(text) == "hello  world"


def test_removes_multiple_blocks_mixed_variants():
    text = (
        "a" + _think("x", tab=True) + "b"
        + _think("y", tab=False) + "c"
        + _think("z", tab=True) + "d"
    )
    assert strip_thinking(text) == "abcd"


def test_unterminated_opening_tag_drops_trailing_reasoning():
    # Typical when the model is cut off mid-thought at max_tokens:
    # the answer came first, then an open tag with no close tag.
    text = "The answer is 42 " + _think_open(tab=True) + " but wait, let me double-check everything..."
    assert strip_thinking(text) == "The answer is 42"


def test_stray_closing_tag_without_opener_is_removed():
    text = "note " + _think_close(tab=True) + " rest"
    assert strip_thinking(text) == "note  rest"


def test_case_insensitive():
    text = "ans " + _L + _T + "THINK" + _G + "blah" + _L + _S + _T + "THINK" + _G + " end"
    assert strip_thinking(text) == "ans  end"
    text = "ans " + _L + _T + "ThInK" + _G + "blah" + _L + _S + _T + "ThInK" + _G + " end"
    assert strip_thinking(text) == "ans  end"


def test_idempotent():
    text = "intro " + _think("reasoning", tab=True) + "outro"
    once = strip_thinking(text)
    assert strip_thinking(once) == once


def test_text_without_thinking_untouched():
    text = "Just a normal reply with a  <span> html tag inside"
    assert strip_thinking(text) == text


# ------- the chunked-send scenario the regex used to fail on ----------

DISCORD_CHUNK_SIZE = 2000  # as used in MessageHandler.handle_message_send


def _chunk_like_handle_message_send(text):
    return [c for c in wrap(text, DISCORD_CHUNK_SIZE, break_long_words=False,
                            replace_whitespace=False) if len(c) > 0]


def test_long_response_chunks_contain_no_thinking_fragments():
    # A long response with several reasoning blocks and a long final answer;
    # ends up as multiple Discord messages when sent.
    answer = " ".join(f"word{j}" for j in range(400))  # ~2800 chars -> 2 chunks
    response = (
        _think("internal monologue " * 80, tab=True)
        + answer
        + _think("more " * 50, tab=True)
        + " closing " + _think_close(tab=True) + " paragraph"
        + _think_open(tab=True)  # deliberately unterminated at the very end
    )
    filtered = filter_response(response)

    # Every chunk (i.e. every Discord message) must be clean...
    for chunk in _chunk_like_handle_message_send(filtered):
        assert _think_open(tab=True) not in chunk
        assert _think_open(tab=False) not in chunk
        assert _think_close(tab=True) not in chunk
        assert _think_close(tab=False) not in chunk
        assert strip_thinking(chunk) == chunk  # no reasoning survives either variant

    # ...and no real content is lost.
    assert all(f"word{j}" in filtered for j in (0, 199, 399))
    assert " closing  paragraph" in filtered


# --------------------------- filter_response --------------------------

def test_filter_response_strips_mention_and_history_artifacts():
    text = "<@123456> hey" + _think("thinking", tab=True) + " real answer"
    assert filter_response(text, mention="123456") == "hey real answer"



def test_filter_response_whitespace_normalised_and_trimmed():
    assert filter_response("  a\n\n\nb  ") == "a\n\nb"


def test_filter_response_no_mention_still_clean():
    text = "hi " + _think("x", tab=True)
    assert filter_response(text) == "hi"


# --------------------------- extract_thinking --------------------------

def test_extract_thinking_returns_the_captured_reasoning():
    text = "hello " + _think("let me reason...", tab=True) + " world"
    assert extract_thinking(text) == "let me reason..."


def test_extract_thinking_joins_multiple_blocks_in_order():
    text = "a" + _think("first", tab=True) + "b" + _think("second", tab=False) + "c"
    assert extract_thinking(text) == "first\n\nsecond"


def test_extract_thinking_captures_unterminated_trailing_reasoning():
    text = "The answer is 42 " + _think_open(tab=True) + "still thinking..."
    assert extract_thinking(text) == "still thinking..."


def test_extract_thinking_empty_when_no_reasoning():
    assert extract_thinking("just a normal reply") == ""


def test_extract_thinking_ignores_stray_close_tag():
    text = "note " + _think_close(tab=True) + " rest"
    assert extract_thinking(text) == ""


# --------------------------- extract_reasoning_items --------------------------
#
# Mirrors the shapes the Agents SDK's chat-completions converter builds from
# a response: a ReasoningItem wrapping a ResponseReasoningItem whose
# `summary` holds the server's reasoning_content (llama.cpp's default), or
# whose `content` holds per-block thinking text.

class _Block:
    def __init__(self, text):
        self.text = text


class _RawReasoning:
    def __init__(self, summary=(), content=()):
        self.summary = [_Block(t) for t in summary]
        self.content = [_Block(t) for t in content]


class _Item:
    def __init__(self, type, raw_item=None):
        self.type = type
        self.raw_item = raw_item


def _reasoning_item(summary=(), content=()):
    return _Item("reasoning_item", _RawReasoning(summary, content))


def _message_item():
    return _Item("message_output_item", object())


def test_extract_reasoning_items_reads_reasoning_content_from_summary():
    items = [_reasoning_item(summary=["the model thought this"]), _message_item()]
    assert extract_reasoning_items(items) == "the model thought this"


def test_extract_reasoning_items_joins_every_turn_of_a_tool_run():
    """Each turn of a tool-calling run contributes its own reasoning item;
    taking only the final turn's would lose the rest."""
    items = [
        _reasoning_item(summary=["i should call the sandbox"]),
        _Item("tool_call_item", object()),
        _Item("tool_call_output_item", object()),
        _reasoning_item(summary=["the sandbox returned 42, so"]),
        _message_item(),
    ]
    assert extract_reasoning_items(items) == (
        "i should call the sandbox\n\nthe sandbox returned 42, so"
    )


def test_extract_reasoning_items_prefers_content_over_summary():
    # Two views of the same thinking - emitting both would duplicate it.
    items = [_reasoning_item(summary=["dupe"], content=["the real thinking"])]
    assert extract_reasoning_items(items) == "the real thinking"


def test_extract_reasoning_items_ignores_non_reasoning_items():
    assert extract_reasoning_items([_message_item(), _Item("tool_call_item")]) == ""


def test_extract_reasoning_items_empty_for_no_items():
    assert extract_reasoning_items([]) == ""
    assert extract_reasoning_items(None) == ""


def test_extract_reasoning_items_skips_blank_and_missing_payloads():
    items = [
        _Item("reasoning_item", None),      # no raw_item
        _reasoning_item(summary=["   "]),   # whitespace only
        _reasoning_item(),                  # no blocks at all
        _reasoning_item(summary=["kept"]),
    ]
    assert extract_reasoning_items(items) == "kept"


# --------------------------- format_thinking_for_discord --------------------------

def test_format_thinking_for_discord_empty_returns_no_messages():
    assert format_thinking_for_discord("") == []
    assert format_thinking_for_discord("   ") == []


def test_format_thinking_for_discord_wraps_in_spoiler_code_block():
    chunks = format_thinking_for_discord("some internal reasoning")
    assert len(chunks) == 1
    assert chunks[0] == "||```\nsome internal reasoning\n```||"


def test_format_thinking_for_discord_chunks_stay_under_discord_limit():
    # several chunks, but still under the cap, so nothing is dropped
    long_thinking = " ".join(f"word{j}" for j in range(700))
    chunks = format_thinking_for_discord(long_thinking)
    assert 1 < len(chunks) <= MAX_THINKING_CHUNKS
    for chunk in chunks:
        assert len(chunk) <= _FMT_CHUNK_SIZE
        assert chunk.startswith("||```\n") and chunk.endswith("\n```||")
    # no content lost
    rejoined = "".join(c[len("||```\n"):-len("\n```||")] for c in chunks)
    assert "word0" in rejoined and "word699" in rejoined


def test_format_thinking_for_discord_caps_chunks_and_keeps_the_tail():
    """A tool-calling run reasons every turn, so the collected thinking can
    run to many messages. Only the last MAX_THINKING_CHUNKS are sent (the
    tail explains the final answer), flagged so the cut is not silent."""
    long_thinking = " ".join(f"word{j}" for j in range(4000))
    chunks = format_thinking_for_discord(long_thinking)
    assert len(chunks) == MAX_THINKING_CHUNKS
    for chunk in chunks:
        assert len(chunk) <= _FMT_CHUNK_SIZE
    assert "truncated" in chunks[0]
    rejoined = "".join(c[len("||```\n"):-len("\n```||")] for c in chunks)
    assert "word3999" in rejoined   # the tail survives
    assert "word0" not in rejoined  # the head is the part dropped


def test_format_thinking_for_discord_escapes_triple_backticks():
    chunks = format_thinking_for_discord("here is ```code``` inline")
    assert len(chunks) == 1
    assert "```code```" not in chunks[0]
    assert "code" in chunks[0]

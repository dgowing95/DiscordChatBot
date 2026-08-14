import re
from textwrap import wrap

import pytest

from core.classes.response_filter import strip_thinking, filter_response

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

"""Tests for classes/tool_embeds.py: how tool embeds read.

Run from the repo root: pytest core/tests/tool_embeds_tests.py
"""
import pytest

from classes import tool_embeds as te


def test_titles_name_the_action_and_fit_discord():
    titles = [te.WEB_SEARCH, te.READING_PAGE, te.REMEMBERED, te.DRAWING, te.EDITING,
              te.SEARCHING_MESSAGES, te.SAVING_MESSAGES, te.READING_MESSAGES,
              te.PERSONALITY, te.SANDBOX_STARTED, te.SANDBOX_NEW, te.SANDBOX_RESUMED]
    assert "Tool Usage" not in titles
    assert len(set(titles)) == len(titles)
    assert all(len(t) <= 256 for t in titles)


def test_reading_page_title_names_a_later_part():
    assert te.reading_page_title(1) == te.READING_PAGE
    assert te.reading_page_title(3) == f"{te.READING_PAGE} · part 3"


@pytest.mark.parametrize("url, label", [
    ("https://www.example.com/gifs/some-page?x=1#top", "example.com › gifs › some-page"),
    ("https://example.com", "example.com"),
    ("https://example.com/", "example.com"),
    ("not a url", "not a url"),
])
def test_link_label_reads_where_a_link_goes(url, label):
    assert te.link_label(url) == label


def test_link_label_is_capped():
    label = te.link_label("https://example.com/" + "a" * 300)
    assert len(label) == te.LINK_LABEL_CHARS
    assert label.endswith("…")


def test_page_link_keeps_the_whole_url_as_the_target():
    # the model reads this back from history: the full link must survive
    url = "https://example.com/wiki/Thing_(disambiguation)?q=1"
    link = te.page_link(url)
    assert link == ("[example.com › wiki › Thing_disambiguation]"
                    "(https://example.com/wiki/Thing_%28disambiguation%29?q=1)")
    # a bracket left in the target made Discord show the link as raw text
    label, target = link[1:].split("](", 1)
    assert not set("()") & set(target[:-1]) and "]" not in label
    # and a backslash escape shows literally inside a link's label
    assert "\\" not in label


def test_message_search_line():
    assert te.message_search_line("movie night", "#general", "Alex") == \
        "“movie night” in #general · by Alex"
    assert te.message_search_line("", "this server", "Alex") == \
        "Newest messages in this server · by Alex"
    assert te.message_search_line("trip", "this server") == "“trip” in this server"


def test_trim_cuts_at_a_word():
    assert te.trim("short", 10) == "short"
    assert te.trim("one two three four five", 12) == "one two…"
    assert len(te.trim("x" * 50, 10)) == 10


def test_sandbox_task_text_lists_the_files():
    assert te.sandbox_task_text("chart it", ["a.csv", "b.png"]) == "chart it\n📎 a.csv, b.png"
    assert te.sandbox_task_text("chart it") == "chart it"

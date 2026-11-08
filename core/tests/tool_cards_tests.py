"""Tests for classes/tool_cards.py: what the cards a tool posts say."""
import pytest

from classes import card_kit as K
from classes import tool_cards as te


def _texts(spec):
    return [t for c in K.split(spec) for t in (b.content for b in c if isinstance(b, K.Text))]


def test_labels_name_the_action_in_plain_words():
    labels = [te.WEB_SEARCH, te.WEB_SEARCHED, te.READING_PAGE, te.REMEMBERED, te.DRAWING, te.EDITING,
              te.SEARCHING_MESSAGES, te.SAVING_MESSAGES, te.READING_MESSAGES,
              te.PERSONALITY, te.BRAIN_WORKING, te.BRAIN_DONE, te.BRAIN_STOPPED]
    assert "Tool Usage" not in labels
    assert len(set(labels)) == len(labels)
    assert all(" " in label for label in labels)  # an emoji and words, never the emoji alone


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


def test_link_label_keeps_a_double_underscore_from_underlining():
    # "__main__" underlined in a label, whatever invisible character sat
    # between the underscores, so they become the look-alike U+02CD
    label = te.link_label("https://docs.python.org/3/library/__main__.html")
    assert label == "docs.python.org › 3 › library › ˍˍmainˍˍ.html"
    # one between two letters never formats, so it stays as it is
    assert te.link_label("https://example.com/Pluto_dwarf_planet") == \
        "example.com › Pluto_dwarf_planet"
    # and the link itself keeps the real path
    assert te.page_link("https://docs.python.org/3/library/__main__.html").endswith(
        "(https://docs.python.org/3/library/__main__.html)")


def test_page_link_reads_a_bare_domain_as_https():
    # fetch_url accepts "example.com/page" (normalize_url adds https)
    assert te.page_link("example.com/wiki/Pluto") == \
        "[example.com › wiki › Pluto](https://example.com/wiki/Pluto)"


def test_message_search_line():
    assert te.message_search_line("movie night", "#general", "Alex") == \
        "**“movie night”** in #general · from Alex"
    assert te.message_search_line("", "this server", "Alex") == \
        "Newest messages in this server · from Alex"
    assert te.message_search_line("trip", "this server") == "**“trip”** in this server"


def test_trim_cuts_at_a_word():
    assert te.trim("short", 10) == "short"
    assert te.trim("one two three four five", 12) == "one two…"
    assert len(te.trim("x" * 50, 10)) == 10


def test_work_task_text_quotes_the_request_and_lists_the_files():
    assert te.work_task_text("chart it", ["a.csv", "b.png"]) == "> chart it\n-# 📎 a.csv, b.png"
    assert te.work_task_text("chart it") == "> chart it"


def test_the_web_search_card_then_says_what_it_read():
    assert _texts(te.web_search_card("keyboards")) == ["-# 🔎 Searching the web", "**keyboards**"]
    done = te.web_search_card("keyboards", [("A", "https://www.a.example/x"), ("B", "https://b.example")])
    assert done.label == te.WEB_SEARCHED
    assert _texts(done)[-1] == ("-# Read 2 pages · [a.example](https://www.a.example/x) · "
                                "[b.example](https://b.example)")
    assert _texts(te.web_search_card("x", [("A", "https://a.example")], read=False))[-1].startswith(
        "-# Found 1 result ·")
    assert _texts(te.web_search_card("x", []))[-1] == "-# Nothing found"


def test_the_page_card_has_an_open_button_and_later_the_pages_title():
    [section] = te.page_card("https://example.com/a(b)", 2).body
    assert isinstance(section, K.Section) and section.texts[0] == "-# 🌐 Reading a page · part 2"
    assert section.accessory == K.Link("Open", "https://example.com/a%28b%29")  # Discord adds the arrow
    [section] = te.page_card("https://example.com/a", title="A page").body
    assert section.texts[1] == "**A page**\n-# [example.com › a](https://example.com/a)"


def test_tool_cards_quote_what_they_were_given_and_say_how_it_went():
    memory = te.memory_card("# likes tea", "<@5>")
    assert memory.label == "💾 Remembered about <@5>" and memory.colour == K.OK
    assert _texts(memory)[-1] == "> \\# likes tea"
    assert te.personality_card("a pirate").colour == K.OK
    drawing = te.drawing_card("a cat", editing=True)
    assert drawing.label == te.EDITING and drawing.colour == K.WORKING
    image = te.image_card("a cat")
    assert image.body == [K.Gallery(("attachment://generated-image.png",))]
    assert image.footer == "a cat\nReply to this to change it" and image.colour == K.OK


def test_a_skill_card_names_the_skill_and_its_state():
    saved = te.skill_card(te.SKILL_SAVED, "weekly-digest", "summarise the week")
    assert saved.title == "weekly-digest" and saved.colour == K.OK
    assert _texts(saved)[-1] == "Summarise the week"
    updated = te.skill_card(te.SKILL_UPDATED, "weekly-digest", "groups links", te.SKILL_FOOTER_CURATOR,
                            version=4)
    assert updated.label == "📗 Skill updated · v4 · daily tidy-up" and updated.colour == K.OK
    assert te.skill_card(te.SKILL_RETIRED, "old").colour == K.CLOSED
    assert te.skill_card(te.SKILL_USED, "x").colour == K.ACTIVE


@pytest.mark.parametrize("tool, args, line", [
    ("exec_command", {"cmd": "cd /workspace/jobs/x && python3 - <<'EOF'\nprint(1)\nEOF"},
     "💻 Running a Python script"),
    ("exec_command", {"cmd": "ls -la"}, "💻 Running ls"),
    ("fetch_url", {"url": "https://www.ssb.no/a"}, "🌐 Reading ssb.no"),
    ("web_search", {"search_request": "Oslo population"}, "🔎 Searching: Oslo population"),
    ("view_image", {}, "👀 Looking at a picture"),
    ("some_tool", {}, "🔧 some tool"),
    ("report", {}, None),
])
def test_a_helpers_line_says_what_it_does_in_plain_words(tool, args, line):
    assert te.helper_line(tool, args) == line


def test_a_helpers_ending():
    assert te.helper_ending(True, None, 3) == "✅ Finished · found 3 facts"
    assert te.helper_ending(False, None, 1) == "⚠️ Not finished · found 1 fact"
    assert te.helper_ending(False, "⚠️ Stopped", 0) == "⚠️ Stopped · found 0 facts"


@pytest.mark.parametrize("url", ["https://example.com/" + "a" * 600, "ftp://example.com/file"])
def test_a_link_discord_would_refuse_as_a_button_still_gets_its_card(url):
    # Discord refuses the whole message for a link button over 512 characters
    # or not http(s): the card would be lost, not just the button.
    [block] = te.page_card(url).body
    assert isinstance(block, K.Text) and "Reading a page" in block.content
    assert te.can_link("https://example.com/a") and not te.can_link(url)


@pytest.mark.asyncio
async def test_a_failed_search_does_not_leave_its_card_searching(monkeypatch):
    import asyncio
    import json
    from unittest.mock import AsyncMock, MagicMock
    from agents.tool_context import ToolContext
    from card_checks import view_text
    from classes import tool_functions, web_research
    monkeypatch.setattr(tool_functions, "check_web_request", AsyncMock(return_value=(True, "")))
    monkeypatch.setattr(web_research, "research", AsyncMock(side_effect=RuntimeError("No results found.")))
    message = MagicMock()
    message.edit = AsyncMock()
    channel = MagicMock()
    channel.send = AsyncMock(return_value=message)
    ctx = ToolContext(context={"channel": channel}, tool_name="web_search", tool_call_id="t1",
                      tool_arguments="{}")
    out = await tool_functions.web_search.on_invoke_tool(ctx, json.dumps({"search_request": "keyboards"}))
    assert out == "Error fetching search results."
    await asyncio.sleep(0)  # the edit runs in the background
    await asyncio.sleep(0)
    edited = message.edit.await_args.kwargs["view"]
    assert edited.colour == K.CLOSED and "The search failed" in view_text(edited)

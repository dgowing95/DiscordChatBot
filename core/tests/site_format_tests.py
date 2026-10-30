"""Tests for how Reddit, Steam and YouTube read in a prompt
(core/classes/site_format.py). The fixtures in core/tests/fixtures/web/ are
real responses (Reddit's with the names and text replaced).

To run this pytest file from the command line, use:
    pytest core/tests/site_format_tests.py
"""
import json
from pathlib import Path

import pytest

from classes import site_format as sf

FIXTURES = Path(__file__).parent / "fixtures" / "web"


def _fixture(name):
    return (FIXTURES / name).read_text(encoding="utf-8")


def test_reddit_thread():
    title, text = sf.format_reddit_thread(_fixture("reddit_thread.rss"))
    assert title == "A tool I made"
    assert text.startswith("Reddit thread in r/examplesub: A tool I made\n"
                           "Posted by u/example_poster on 2026-10-24\n"
                           "Links to: https://example.com/tool")
    assert "- Fast: very.\n- Small: tiny." in text
    assert "Can't say more yet." in text          # entities decoded
    assert "submitted by" not in text and "[comments]" not in text
    assert "Comments (2 shown):" in text
    assert "- u/example_user: Nice work — does it run on Windows?" in text
    assert "- u/another_user: It does, I tried it." in text


def test_reddit_text_post_has_no_links_to_line():
    feed = _fixture("reddit_thread.rss").replace("https://example.com/tool",
                                                 "https://www.reddit.com/r/examplesub/comments/1abc234/a_tool_i_made/")
    assert "Links to:" not in sf.format_reddit_thread(feed)[1]


def test_reddit_thread_comment_limits(monkeypatch):
    monkeypatch.setattr(sf, "MAX_COMMENTS", 1)
    monkeypatch.setattr(sf, "COMMENT_CHARS", 10)
    text = sf.format_reddit_thread(_fixture("reddit_thread.rss"))[1]
    assert "Comments (1 shown of 2):" in text
    assert "- u/example_user: Nice work…" in text
    assert "another_user" not in text


def test_reddit_empty_feed_is_an_error():
    with pytest.raises(ValueError):
        sf.format_reddit_thread('<feed xmlns="http://www.w3.org/2005/Atom"><title>x</title></feed>')


def test_reddit_listing():
    title, text = sf.format_reddit_listing(_fixture("reddit_listing.rss"))
    assert title == "Reddit: r/examplesub"
    assert "(2 posts)" in text
    assert "- Weekly thread (u/example_poster, 2026-10-25)\n  https://www.reddit.com/r/examplesub/comments/1aaa111/weekly_thread/" in text
    assert "Share what you are working on." in text
    assert "- Some news (u/example_user, 2026-10-30)" in text


def _steam():
    return json.loads(_fixture("steam_appdetails.json"))["620"]["data"]


def test_steam_app():
    title, text = sf.format_steam_app(_steam(), _fixture("steam_store_reviews.html"))
    assert title == "Portal 2"
    lines = text.splitlines()
    assert lines[0] == "Steam: Portal 2"
    assert "Release date: 18 Apr, 2011" in lines
    assert "Developer: Valve" in lines
    assert "Genres: Action, Adventure" in lines
    assert "Platforms: Windows, Linux" in lines
    assert "Metacritic: 95" in lines
    assert "Recommendations: 392,550" in lines
    assert ("User reviews: Overwhelmingly Positive (98% of the 2,174 user reviews in the last 30 "
            "days are positive; 98% of the 174,718 user reviews in your language are positive)") in lines
    assert "About the game:" in lines
    assert "<" not in text


def test_steam_free_sale_and_unreleased():
    data = _steam()
    data["price_overview"] = {"final_formatted": "$4.99", "initial_formatted": "$9.99",
                              "discount_percent": 50}
    assert "Price: $4.99 (50% off, was $9.99)" in sf.format_steam_app(data)[1]
    data["is_free"] = True
    data["release_date"] = {"coming_soon": True, "date": "Coming soon"}
    text = sf.format_steam_app(data)[1]
    assert "Price: free to play" in text
    assert "Release date: not out yet (Coming soon)" in text
    assert "User reviews" not in text  # no store page fetched


def test_transcript_from_real_snippets():
    snippets = json.loads(_fixture("yt_transcript.json"))
    text = sf.format_transcript("A video", "A channel", "English", False, snippets)
    lines = text.splitlines()
    assert lines[:3] == ["YouTube video: A video", "Channel: A channel", "Captions: English"]
    assert lines[4].startswith("[0:01] All right, so here we are, in front of the elephants")


def test_transcript_blocks_and_hours():
    snippets = [{"text": f"word{i}", "start": i * 10.0} for i in range(8)] + \
               [{"text": "late", "start": 3725.0}, {"text": "  ", "start": 3730.0}]
    text = sf.format_transcript("T", "", "German", True, snippets)
    assert "Channel" not in text
    assert "Captions: German (auto-generated)" in text
    body = text.split("\n\n", 1)[1].splitlines()
    assert body == ["[0:00] word0 word1 word2", "[0:30] word3 word4 word5", "[1:00] word6 word7",
                    "[1:02:05] late"]


def test_no_captions():
    text = sf.format_no_captions("T", "C")
    assert text.startswith("YouTube video: T\nChannel: C\n\n")
    assert "no captions" in text


def test_html_to_text_drops_scripts():
    assert sf.html_to_text("<p>a<script>evil()</script></p><style>x{}</style><p>b</p>") == "a\n\nb"

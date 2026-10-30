"""Tests for the link-reading rules (core/classes/web_policy.py).

To run this pytest file from the command line, use:
    pytest core/tests/web_policy_tests.py
"""
import pytest

from classes import web_policy as wp


@pytest.mark.parametrize("url", [
    "https://youtu.be/dQw4w9WgXcQ?si=abc123",
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ&feature=share&t=42",
    "youtube.com/watch?v=dQw4w9WgXcQ",
    "https://m.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://music.youtube.com/watch?v=dQw4w9WgXcQ&list=x",
    "https://www.youtube.com/shorts/dQw4w9WgXcQ",
    "https://www.youtube.com/live/dQw4w9WgXcQ?feature=share",
    "https://www.youtube-nocookie.com/embed/dQw4w9WgXcQ",
])
def test_every_youtube_spelling_is_one_video(url):
    target = wp.route(url)
    assert target == wp.Route(wp.YOUTUBE, "dQw4w9WgXcQ", "https://www.youtube.com/watch?v=dQw4w9WgXcQ")


@pytest.mark.parametrize("url", [
    "https://www.youtube.com/watch?v=short",
    "https://www.youtube.com/@somechannel",
    "https://www.youtube.com/",
    "https://notyoutube.com/watch?v=dQw4w9WgXcQ",
])
def test_non_videos_are_plain_pages(url):
    assert wp.route(url).kind == wp.PAGE


@pytest.mark.parametrize("url, ident", [
    ("https://www.reddit.com/r/Python/comments/1abc234/some_title/", "r/Python/comments/1abc234"),
    ("https://old.reddit.com/r/Python/comments/1abc234/some_title/?utm_source=share", "r/Python/comments/1abc234"),
    ("https://new.reddit.com/r/Python/comments/1abc234", "r/Python/comments/1abc234"),
    ("https://m.reddit.com/r/Python/comments/1abc234/x/comment/zzz/", "r/Python/comments/1abc234"),
    ("https://redd.it/1abc234", "comments/1abc234"),
    ("https://www.reddit.com/comments/1abc234", "comments/1abc234"),
])
def test_reddit_threads(url, ident):
    target = wp.route(url)
    assert target.kind == wp.REDDIT_THREAD
    assert target.ident == ident
    assert target.url == f"https://www.reddit.com/{ident}/"


def test_reddit_spellings_share_a_cache_key():
    keys = {wp.cache_key(wp.route(u).url) for u in (
        "https://www.reddit.com/r/Python/comments/1abc234/some_title/",
        "https://old.reddit.com/r/Python/comments/1abc234/other_slug/?share_id=1",
    )}
    assert len(keys) == 1


@pytest.mark.parametrize("url, ident", [
    ("https://www.reddit.com/r/Python/", "r/Python"),
    ("https://www.reddit.com/r/Python/top/", "r/Python/top"),
])
def test_reddit_listings(url, ident):
    target = wp.route(url)
    assert (target.kind, target.ident) == (wp.REDDIT_LISTING, ident)


def test_reddit_share_links_and_other_pages():
    assert wp.route("https://www.reddit.com/r/Python/s/AbCdEf").kind == wp.REDDIT_SHARE
    assert wp.route("https://www.reddit.com/user/someone/").kind == wp.PAGE


def test_steam_apps():
    target = wp.route("https://store.steampowered.com/app/620/Portal_2/?snr=1_7_15")
    assert target == wp.Route(wp.STEAM_APP, "620", "https://store.steampowered.com/app/620/")
    assert wp.route("https://store.steampowered.com/search/?term=x").kind == wp.PAGE


def test_normalize_drops_tracking_and_fragment_and_sorts():
    assert wp.normalize_url("HTTPS://Example.COM:443/a?b=2&utm_source=x&a=1&fbclid=z#top") == \
        "https://example.com/a?a=1&b=2"
    assert wp.normalize_url("http://example.com:8080") == "http://example.com:8080/"
    assert wp.normalize_url("example.com/page") == "https://example.com/page"


@pytest.mark.parametrize("url", ["", "ftp://example.com/x", "file:///etc/passwd", "https://",
                                 "javascript:alert(1)", "http://[::1"])
def test_unreadable_urls(url):
    assert wp.route(url) is None


@pytest.mark.parametrize("address, public", [
    ("8.8.8.8", True),
    ("2606:4700:4700::1111", True),
    ("10.0.0.5", False),
    ("172.16.3.4", False),
    ("192.168.1.1", False),
    ("127.0.0.1", False),
    ("169.254.169.254", False),   # cloud metadata
    ("100.64.0.1", False),        # shared address space (CGNAT)
    ("0.0.0.0", False),
    ("::1", False),
    ("fd00::1", False),
    ("fe80::1%eth0", False),
    ("::ffff:10.0.0.1", False),   # IPv4 wrapped in IPv6
    ("::ffff:8.8.8.8", True),
    ("224.0.0.1", False),
    ("not-an-ip", False),
])
def test_is_public_ip(address, public):
    assert wp.is_public_ip(address) is public


def test_private_ip_literal():
    assert wp.private_ip_literal("http://127.0.0.1:6379/")
    assert wp.private_ip_literal("http://[::1]/")
    assert not wp.private_ip_literal("http://8.8.8.8/")
    assert not wp.private_ip_literal("http://redis:6379/")  # a name: the resolver checks it


def test_needs_browser():
    assert wp.needs_browser("", 400)
    assert wp.needs_browser("Loading…", 400)
    assert wp.needs_browser("x" * 500 + " You need to enable JavaScript to run this app.", 400)
    assert not wp.needs_browser("word " * 200, 400)
    # A long article that merely mentions JavaScript is not a script wall.
    assert not wp.needs_browser("enable javascript " + "word " * 600, 400)


def test_split_parts_cuts_at_line_breaks_and_loses_nothing():
    text = "\n".join(f"line {i} " + "x" * 40 for i in range(100))
    parts = wp.split_parts(text, 500)
    assert "".join(parts) == text
    assert all(len(p) <= 500 for p in parts)
    assert all(p.endswith("\n") for p in parts[:-1])


def test_split_parts_without_breaks_still_splits():
    parts = wp.split_parts("y" * 2500, 1000)
    assert [len(p) for p in parts] == [1000, 1000, 500]


def test_paginate_footers():
    text = "\n".join("z" * 99 for _ in range(30))  # 3000 chars
    first = wp.paginate(text, 1, 1000)
    assert first.endswith("[Part 1 of 3. Call fetch_url again with part=2 to read on.]")
    assert wp.paginate(text, 3, 1000).endswith("[Part 3 of 3: the end.]")
    assert wp.paginate(text, 4, 1000).startswith("There is no part 4: this page has 3 parts.")
    assert wp.paginate("short", 1, 1000) == "short"
    assert wp.paginate("short", 2, 1000).startswith("There is no part 2: this page has 1 part.")


def test_settings_defaults_and_bad_values(monkeypatch):
    for name in ("WEB_FETCH_PART_CHARS", "BROWSER_URL", "BROWSER_TIMEOUT", "BROWSER_SETTLE_SECONDS"):
        monkeypatch.delenv(name, raising=False)
    config = wp.settings()
    assert config["part_chars"] == 10_000
    assert config["browser_url"] == ""
    assert config["browser_wait"] == 50
    monkeypatch.setenv("WEB_FETCH_PART_CHARS", "-5")
    monkeypatch.setenv("BROWSER_URL", "http://browser:8000/")
    assert wp.settings()["part_chars"] == 10_000
    assert wp.settings()["browser_url"] == "http://browser:8000"

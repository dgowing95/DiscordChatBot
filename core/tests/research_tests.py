"""web_search's research: what it reads, the brief call, and every fallback.

research_format is pure; web_research is tested with search, the page reader
and the side LLM patched, so nothing here touches the network.

Run: pytest core/tests/research_tests.py
"""
import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from classes import research_format as rf
from classes import web_research


def _raw(*urls):
    return [{"title": f"Title {u}", "href": u, "body": f"snippet  for\n{u}"} for u in urls]


# ------------------------------ research_format ------------------------------

def test_clean_results_keeps_web_links_once_and_tidies_text():
    raw = _raw("https://a.example/1", "ftp://b.example/x", "https://a.example/1") + ["junk", {}]
    results = rf.clean_results(raw)
    assert [r.url for r in results] == ["https://a.example/1"]
    assert results[0].snippet == "snippet for https://a.example/1"


def test_a_result_with_no_title_is_named_by_its_link():
    [result] = rf.clean_results([{"href": "https://a.example/", "body": ""}])
    assert result.title == "https://a.example/"


def test_pick_pages_prefers_one_page_per_site_then_fills_in_order():
    results = rf.clean_results(_raw("https://a.example/1", "https://a.example/2",
                                    "https://b.example/1", "https://c.example/1"))
    assert [r.url for r in rf.pick_pages(results, 3)] == [
        "https://a.example/1", "https://b.example/1", "https://c.example/1"]
    only_one_site = rf.clean_results(_raw("https://a.example/1", "https://a.example/2"))
    assert [r.url for r in rf.pick_pages(only_one_site, 3)] == [
        "https://a.example/1", "https://a.example/2"]


def test_brief_messages_give_read_pages_cut_to_size_and_snippets_for_the_rest():
    results = rf.clean_results(_raw("https://a.example/", "https://b.example/"))
    messages = rf.brief_messages("what is it?", results, {"https://a.example/": "x" * 50},
                                 "2026-10-31", page_chars=10)
    assert messages[0] == {"role": "system", "content": rf.SYSTEM_PROMPT}
    user = messages[1]["content"]
    assert "Today's date: 2026-10-31" in user and "Question: what is it?" in user
    assert "[1] Title https://a.example/\nhttps://a.example/\n" + "x" * 10 + "\n" in user
    assert "x" * 11 not in user
    assert "[2] Title https://b.example/\nhttps://b.example/\n(search snippet only) snippet" in user


def test_brief_result_lists_the_real_sources_and_marks_unread_ones():
    results = rf.clean_results(_raw("https://a.example/", "https://b.example/"))
    text = rf.brief_result("q", " It is blue [1]. ", results, {"https://a.example/"})
    assert text.startswith("Web research on: q\n\nIt is blue [1].\n\nSources:\n")
    assert "[1] Title https://a.example/ - https://a.example/\n" in text
    assert "[2] Title https://b.example/ - https://b.example/ (snippet only)" in text
    assert "fetch_url" in text


def test_results_text_is_plain_and_says_why():
    results = rf.clean_results(_raw("https://a.example/"))
    text = rf.results_text("q", results, "none of the pages could be read")
    assert text.startswith("Search results for: q\n(none of the pages could be read)\n\n[1] ")
    assert "{" not in text  # not a Python repr any more


def test_settings_defaults_and_bounds(monkeypatch):
    for name in ("RESEARCH_BRIEF_ENABLED", "RESEARCH_PAGES", "RESEARCH_PAGE_CHARS",
                 "RESEARCH_FETCH_TIMEOUT", "RESEARCH_TIMEOUT"):
        monkeypatch.delenv(name, raising=False)
    assert rf.settings() == {"enabled": True, "pages": 3, "page_chars": 6000,
                             "fetch_timeout": 25.0, "timeout": 90.0}
    monkeypatch.setenv("RESEARCH_PAGES", "50")
    monkeypatch.setenv("RESEARCH_PAGE_CHARS", "nope")
    monkeypatch.setenv("RESEARCH_BRIEF_ENABLED", "off")
    config = rf.settings()
    assert config["pages"] == rf.MAX_PAGES
    assert config["page_chars"] == 6000
    assert config["enabled"] is False


def test_the_brief_model_falls_back_to_the_bots_own(monkeypatch):
    monkeypatch.delenv("RESEARCH_MODEL", raising=False)
    monkeypatch.setenv("MODEL", "main-model")
    assert rf.research_model() == "main-model"
    monkeypatch.setenv("RESEARCH_MODEL", "other")
    assert rf.research_model() == "other"


# -------------------------------- web_research --------------------------------

RESULTS = rf.clean_results(_raw("https://a.example/", "https://b.example/", "https://c.example/"))


async def _research(pages=None, brief="It is blue [1].", search=RESULTS, brief_error=None):
    complete = AsyncMock(return_value=brief, side_effect=brief_error)
    with patch.object(web_research, "search", AsyncMock(return_value=search)), \
         patch.object(web_research, "read_pages", AsyncMock(return_value=pages or {})) as read, \
         patch.object(web_research._llm, "complete", complete):
        text = await web_research.research("what colour?", "2026-10-31")
    return text, read, complete


@pytest.mark.asyncio
async def test_research_reads_the_top_pages_and_returns_the_brief(monkeypatch):
    monkeypatch.delenv("RESEARCH_BRIEF_ENABLED", raising=False)
    monkeypatch.setenv("RESEARCH_PAGES", "2")
    text, read, complete = await _research(pages={"https://a.example/": "a page"})
    assert [r.url for r in read.await_args.args[0]] == ["https://a.example/", "https://b.example/"]
    assert text.startswith("Web research on: what colour?\n\nIt is blue [1].")
    assert "[2] Title https://b.example/ - https://b.example/ (snippet only)" in text
    assert complete.await_args.kwargs["temperature"] == 0.2


@pytest.mark.asyncio
async def test_no_results_says_so_without_reading_anything():
    text, read, complete = await _research(search=[])
    assert "found nothing" in text
    read.assert_not_awaited()
    complete.assert_not_awaited()


@pytest.mark.asyncio
async def test_brief_switched_off_returns_the_results_alone(monkeypatch):
    monkeypatch.setenv("RESEARCH_BRIEF_ENABLED", "0")
    text, read, complete = await _research()
    assert text.startswith("Search results for: what colour?\n\n[1] ")
    read.assert_not_awaited()
    complete.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_page_read_skips_the_brief_call(monkeypatch):
    monkeypatch.delenv("RESEARCH_BRIEF_ENABLED", raising=False)
    text, _, complete = await _research(pages={})
    assert "(none of the pages could be read)" in text
    complete.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("kwargs", [{"brief_error": RuntimeError("down")}, {"brief": ""},
                                    {"brief": "<think>hmm</think>"}])
async def test_a_failed_or_empty_brief_falls_back_to_the_results(monkeypatch, kwargs):
    monkeypatch.delenv("RESEARCH_BRIEF_ENABLED", raising=False)
    text, _, _ = await _research(pages={"https://a.example/": "a page"}, **kwargs)
    assert text.startswith("Search results for: what colour?\n(the pages could not be summarised)")


@pytest.mark.asyncio
async def test_read_pages_leaves_out_slow_and_failed_pages():
    async def _read(url):
        if "slow" in url:
            await asyncio.sleep(5)
        return None if "bad" in url else f"text of {url}"

    results = rf.clean_results(_raw("https://ok.example/", "https://slow.example/",
                                    "https://bad.example/"))
    with patch.object(web_research, "_read_one", _read):
        pages = await web_research.read_pages(results, timeout=0.2)
    assert pages == {"https://ok.example/": "text of https://ok.example/"}


@pytest.mark.asyncio
async def test_read_one_turns_a_fetch_error_into_none():
    from classes import web_reader
    with patch.object(web_reader, "read", AsyncMock(side_effect=web_reader.FetchError("x", "no"))):
        assert await web_research._read_one("https://a.example/") is None

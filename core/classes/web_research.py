"""web_search's work: search, read the top results, write a brief.

The rules, the prompt and every string the model sees are in
research_format.py (pure, and why this replaced the old snippet-only search
is there too); this module only does the I/O. Fails soft at each step: no
pages read, or a brief that fails, still returns the search results.
"""
import asyncio
import logging

from ddgs import DDGS

from classes import research_format as rf
from classes.response_filter import strip_thinking
from classes.side_llm import SideLLM

logger = logging.getLogger(__name__)

_llm = SideLLM("web_research", rf.research_model, rf.research_llm_host,
               rf.research_llm_api_key, rf.research_timeout)


async def search(query: str) -> list:
    """Raises like DDGS does; the caller says so to the model."""
    raw = await asyncio.to_thread(DDGS().text, query, max_results=rf.SEARCH_RESULTS)
    return rf.clean_results(raw)


async def _read_one(url: str):
    from classes import web_reader
    try:
        return (await web_reader.read(url)).text
    except Exception as e:
        logger.info(f"Research: could not read {url}: {e}")
        return None


async def read_pages(results, timeout: float) -> dict:
    """{url: text} for the results that were read within `timeout`. A page
    still loading is left out; web_reader shields its load, so it still
    finishes into the cache for a later fetch_url."""
    tasks = {r.url: asyncio.create_task(_read_one(r.url)) for r in results}
    if not tasks:
        return {}
    done, pending = await asyncio.wait(tasks.values(), timeout=timeout)
    for task in pending:
        task.cancel()
    return {url: task.result() for url, task in tasks.items()
            if task in done and task.result()}


async def research(question: str, today: str) -> str:
    """The text web_search returns for `question`."""
    results = await search(question)
    if not results:
        return rf.no_results_text(question)
    config = rf.settings()
    if not config["enabled"]:
        return rf.results_text(question, results)
    pages = await read_pages(rf.pick_pages(results, config["pages"]), config["fetch_timeout"])
    if not pages:
        # A brief from snippets alone adds a call and says nothing new.
        return rf.results_text(question, results, "none of the pages could be read")
    try:
        content = await _llm.complete(
            rf.brief_messages(question, results, pages, today, config["page_chars"]),
            # Reporting, not writing: and a guild's /temperature must not reach it.
            temperature=0.2,
            # ~200 words asked for; the headroom is for a backend that ignores
            # the thinking switch.
            max_tokens=1024,
        )
    except Exception as e:
        logger.warning(f"Research brief failed ({e}); returning the search results")
        return rf.results_text(question, results, "the pages could not be summarised")
    brief = strip_thinking(content or "").strip()
    if not brief:
        return rf.results_text(question, results, "the pages could not be summarised")
    logger.info(f"Research brief for {question!r}: {len(pages)} pages read, {len(brief)} chars")
    return rf.brief_result(question, brief, results, set(pages))

"""web_search's work: search, read the top results, write a brief.

The rules, the prompt and every string the model sees are in
research_format.py (pure, and why this replaced the old snippet-only search
is there too); this module only does the I/O. Fails soft at each step: no
pages read, or a brief that fails, still returns the search results.

The search itself runs in a worker process (classes/search_worker.py), one
search at a time across the bot:
  - in a process of its own because the search library's native code once
    crashed the whole bot (search_worker's docstring); there, a crash costs
    one search, and the next search starts a new worker;
  - one at a time because that crash came with four searches at once, and
    because the search engines rate-limit parallel searches hard (six at
    once: up to five of six refused). Workflow helpers that search side by
    side wait a few seconds for each other instead.
"""
import asyncio
import json
import logging
import sys
from pathlib import Path

from classes import research_format as rf
from classes.response_filter import strip_thinking
from classes.side_llm import SideLLM

logger = logging.getLogger(__name__)

# The caller's lane: chat inside a reply, brain inside a brain call.
_llm = SideLLM("web_research", rf.research_config)


# One search, the worker's own wait on each engine included. Past it the
# worker is killed (it may be stuck) and the next search starts a new one.
SEARCH_TIMEOUT_SECONDS = 30
# A reply is one JSON line: the results' titles, links and snippets.
_REPLY_LIMIT = 4 * 1024 * 1024
_CORE_DIR = Path(__file__).resolve().parents[1]


class SearchWorker:
    """The worker process and the lock that keeps it to one search at a
    time. One per event loop (worker())."""

    def __init__(self, command=None):
        self.command = command or [sys.executable, "-m", "classes.search_worker"]
        self.lock = asyncio.Lock()
        self.proc = None
        self.loop = asyncio.get_running_loop()
        self._reaping: set = set()
        self._next_id = 0

    async def _process(self):
        if self.proc is None or self.proc.returncode is not None:
            self._drop()  # a worker that ended meanwhile is released, not just replaced
            self.proc = await asyncio.create_subprocess_exec(
                *self.command, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                cwd=str(_CORE_DIR), limit=_REPLY_LIMIT)
        return self.proc

    def _drop(self) -> None:
        """Kills the worker (it may be stuck, or a late reply would answer
        the next search) and waits for its end in the background, so its
        pipes are closed. Synchronous: it also runs while a search is being
        cancelled."""
        proc, self.proc = self.proc, None
        if proc is None:
            return
        if proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
        task = self.loop.create_task(self._reap(proc))
        self._reaping.add(task)
        task.add_done_callback(self._reaping.discard)

    @staticmethod
    async def _reap(proc) -> None:
        try:
            await asyncio.wait_for(proc.wait(), 10)
        except Exception as e:
            logger.warning(f"Search worker {proc.pid} did not end when killed: {e!r}")

    async def close(self) -> None:
        """Ends the worker and waits for it (tests; the bot's exit closes
        its stdin, which ends it too)."""
        if self.proc is not None and self.proc.returncode is None and self.proc.stdin:
            self.proc.stdin.close()
        self._drop()
        await asyncio.gather(*self._reaping, return_exceptions=True)

    async def search(self, query: str, max_results: int) -> list:
        """The engines' results. Raises RuntimeError when the search failed,
        took too long, the worker ended (a crash) or answered out of turn: the
        caller fails soft. Every reply carries its request's id, so a stray
        line on the pipe can never pass as the answer to another search."""
        async with self.lock:
            self._next_id += 1
            request_id = self._next_id
            try:
                proc = await self._process()
                request = json.dumps({"id": request_id, "query": query,
                                      "max_results": max_results})
                proc.stdin.write((request + "\n").encode())
                await proc.stdin.drain()
                line = await asyncio.wait_for(proc.stdout.readline(), SEARCH_TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                self._drop()
                raise RuntimeError(f"the search took over {SEARCH_TIMEOUT_SECONDS} seconds")
            except asyncio.CancelledError:
                self._drop()
                raise
            except (OSError, ValueError) as e:  # a pipe to a worker that ended
                self._drop()
                raise RuntimeError(f"the search worker stopped ({e})")
            if not line.endswith(b"\n"):  # nothing, or cut off mid-line: it ended
                code = proc.returncode if proc.returncode is not None else "unknown"
                logger.warning(f"Search worker ended (exit {code}); the next search starts a new one")
                self._drop()
                raise RuntimeError("the search worker stopped")
            try:
                reply = json.loads(line)
            except ValueError:
                reply = None
            if not isinstance(reply, dict) or reply.get("id") != request_id:
                # A line the worker did not mean as this reply: the pipe is out
                # of step, so a new worker answers the next search.
                logger.warning(f"Search worker answered out of turn: {line[:200]!r}")
                self._drop()
                raise RuntimeError("the search worker sent an unreadable reply")
            if not reply.get("ok"):
                raise RuntimeError(reply.get("error") or "the search failed")
            return reply.get("results") or []


_worker: SearchWorker | None = None


def worker() -> SearchWorker:
    """This event loop's search worker (tests run several loops)."""
    global _worker
    if _worker is None or _worker.loop is not asyncio.get_running_loop():
        old, _worker = _worker, None
        if old is not None and old.proc is not None and old.proc.returncode is None:
            try:  # its loop is gone, so no reap task can run for it
                old.proc.kill()
            except ProcessLookupError:  # it ended meanwhile
                pass
            except Exception as e:
                logger.warning(f"Could not kill the old search worker: {e!r}")
        _worker = SearchWorker()
    return _worker


async def search(query: str) -> list:
    """Raises when the search fails; the caller says so to the model."""
    return rf.clean_results(await worker().search(query, rf.SEARCH_RESULTS))


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


async def research(question: str, today: str, seen: dict | None = None) -> str:
    """The text web_search returns for `question`. `seen`, when given, is
    filled with what the search card shows afterwards: "sources", (title,
    url) pairs, and "read", whether they were read or only found."""
    seen = {} if seen is None else seen
    results = await search(question)
    seen.update(sources=[(r.title, r.url) for r in results], read=False)
    if not results:
        return rf.no_results_text(question)
    config = rf.settings()
    if not config["enabled"]:
        return rf.results_text(question, results)
    pages = await read_pages(rf.pick_pages(results, config["pages"]), config["fetch_timeout"])
    if pages:
        seen.update(sources=[(r.title, r.url) for r in results if r.url in pages], read=True)
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

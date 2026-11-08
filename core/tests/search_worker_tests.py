"""classes/search_worker.py and web_research.SearchWorker: the web search in
a process of its own, one at a time, so a native crash costs one search and
never the bot."""
import asyncio
import json
import sys

import pytest

from classes import search_worker, web_research

# Stands in for the worker: answers, fails, crashes or hangs by its query.
FAKE = r"""
import json, os, sys, time
for line in sys.stdin:
    request = json.loads(line)
    q, i = request["query"], request["id"]
    if q == "crash":
        os._exit(134)  # as a native crash would: no reply, the process gone
    if q == "slow":
        time.sleep(60)
    if q == "stray":  # a line the worker did not mean as a reply, then the reply
        print("primp: something printed", flush=True)
    if q == "half":  # cut off mid-line
        sys.stdout.write('{"id": ' + str(i) + ', "ok": tr'); sys.stdout.flush(); os._exit(1)
    if q == "fail":
        print(json.dumps({"id": i, "ok": False, "error": "DDGSException: No results found."}), flush=True)
        continue
    print(json.dumps({"id": i, "ok": True, "results": [{"title": q, "pid": os.getpid()}]}), flush=True)
"""


def _fake():
    return web_research.SearchWorker(command=[sys.executable, "-c", FAKE])


# --------------------------- the worker module ---------------------------

def test_a_request_line_gets_the_results():
    calls = []

    def text(query, max_results):
        calls.append((query, max_results))
        return [{"title": "Io", "href": "https://example.com/io", "body": "3,643 km"}]

    reply = search_worker.handle(json.dumps({"id": 7, "query": "io", "max_results": 5}), text)
    assert reply == {"id": 7, "ok": True,
                     "results": [{"title": "Io", "href": "https://example.com/io",
                                  "body": "3,643 km"}]}
    assert calls == [("io", 5)]


@pytest.mark.parametrize("line", ["not json", json.dumps({"query": "x"})])
def test_a_bad_request_is_an_error_reply_not_a_crash(line):
    reply = search_worker.handle(line, lambda query, max_results: [])
    assert reply["ok"] is False and reply["error"]


def test_a_failed_search_is_an_error_reply():
    def text(query, max_results):
        raise RuntimeError("No results found.")
    assert search_worker.handle(json.dumps({"id": 3, "query": "x", "max_results": 5}), text) == \
        {"id": 3, "ok": False, "error": "RuntimeError: No results found."}


@pytest.mark.asyncio
async def test_the_real_module_starts_from_the_bots_directory_and_answers():
    # A bad line needs no network: it shows `python -m classes.search_worker`
    # runs where the bot runs it, and speaks the protocol.
    worker = web_research.SearchWorker()
    proc = await worker._process()
    proc.stdin.write(b"not json\n")
    await proc.stdin.drain()
    reply = json.loads(await asyncio.wait_for(proc.stdout.readline(), 30))
    assert reply["ok"] is False
    await worker.close()


# --------------------------- SearchWorker ---------------------------

@pytest.mark.asyncio
async def test_searches_reuse_one_worker():
    worker = _fake()
    first = await worker.search("io", 5)
    second = await worker.search("europa", 5)
    assert [first[0]["title"], second[0]["title"]] == ["io", "europa"]
    assert first[0]["pid"] == second[0]["pid"]
    await worker.close()


@pytest.mark.asyncio
async def test_a_failed_search_raises_for_the_caller_and_keeps_the_worker():
    worker = _fake()
    with pytest.raises(RuntimeError, match="No results found"):
        await worker.search("fail", 5)
    assert worker.proc is not None and worker.proc.returncode is None
    await worker.close()


@pytest.mark.asyncio
async def test_a_crash_costs_one_search_and_the_next_starts_a_new_worker():
    worker = _fake()
    before = (await worker.search("io", 5))[0]["pid"]
    with pytest.raises(RuntimeError, match="worker stopped"):
        await worker.search("crash", 5)
    after = (await worker.search("io", 5))[0]["pid"]
    assert after != before
    await worker.close()


@pytest.mark.asyncio
async def test_a_stuck_search_is_killed_and_the_next_starts_a_new_worker(monkeypatch):
    monkeypatch.setattr(web_research, "SEARCH_TIMEOUT_SECONDS", 1)
    worker = _fake()
    with pytest.raises(RuntimeError, match="took over 1 seconds"):
        await worker.search("slow", 5)
    assert (await worker.search("io", 5))[0]["title"] == "io"
    await worker.close()


@pytest.mark.asyncio
async def test_searches_at_once_take_turns_and_each_gets_its_own_answer():
    worker = _fake()
    results = await asyncio.gather(*(worker.search(q, 5) for q in ("io", "europa", "ganymede")))
    assert [r[0]["title"] for r in results] == ["io", "europa", "ganymede"]
    await worker.close()


@pytest.mark.asyncio
async def test_a_cancelled_search_never_answers_the_next_one(monkeypatch):
    worker = _fake()
    slow = asyncio.create_task(worker.search("slow", 5))
    await asyncio.sleep(0.5)
    slow.cancel()
    with pytest.raises(asyncio.CancelledError):
        await slow
    assert (await worker.search("io", 5))[0]["title"] == "io"
    await worker.close()


@pytest.mark.asyncio
async def test_a_stray_line_never_passes_as_another_searchs_answer():
    worker = _fake()
    with pytest.raises(RuntimeError, match="unreadable reply"):
        await worker.search("stray", 5)
    # The real reply to "stray" is still in the old pipe: a new worker answers.
    assert (await worker.search("io", 5))[0]["title"] == "io"
    await worker.close()


@pytest.mark.asyncio
async def test_a_reply_cut_off_mid_line_is_a_stopped_worker():
    worker = _fake()
    with pytest.raises(RuntimeError, match="worker stopped"):
        await worker.search("half", 5)
    assert (await worker.search("io", 5))[0]["title"] == "io"
    await worker.close()


@pytest.mark.asyncio
async def test_a_worker_that_cannot_start_is_a_failed_search(monkeypatch):
    worker = web_research.SearchWorker(command=["no-such-program-dcb"])
    with pytest.raises(RuntimeError, match="worker stopped"):
        await worker.search("io", 5)
    await worker.close()


@pytest.mark.asyncio
async def test_the_real_worker_keeps_native_output_off_the_replies():
    # Output written straight to fd 1 (as native code would) must not reach
    # the bot's pipe: the next line read is the reply.
    worker = web_research.SearchWorker(command=[sys.executable, "-c", (
        "import os, sys\n"
        "from classes import search_worker\n"
        "orig = search_worker.handle\n"
        "def handle(line, text=None):\n"
        "    os.write(1, b'native noise\\n')\n"
        "    return orig(line, lambda q, max_results: [{'title': q}])\n"
        "search_worker.handle = handle\n"
        "search_worker.main()\n")])
    assert await worker.search("io", 5) == [{"title": "io"}]
    await worker.close()


@pytest.mark.asyncio
async def test_each_event_loop_gets_its_own_worker():
    first = web_research.worker()
    assert web_research.worker() is first
    assert first.loop is asyncio.get_running_loop()

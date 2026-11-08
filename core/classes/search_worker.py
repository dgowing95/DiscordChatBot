"""The web search itself, in a process of its own (web_research.search).

ddgs drives primp, a native (Rust) HTTP client, from threads of its own. In
one E2E run (2026-11-08) a native crash there ("free(): invalid pointer")
killed the whole bot, chat and voice included, while four searches ran at
once. It could not be reproduced (36 searches, six at a time), so the cause
is not known. In this process, a crash like that costs one search: the bot
sees the worker's end and starts a new one for the next search.

The protocol is one JSON object per line: {"id", "query", "max_results"} in
on stdin, {"id", "ok": true, "results": [...]} or {"id", "ok": false,
"error": "..."} out. One request at a time (the caller holds a lock), and
the id echoed, so the bot can tell a stray line from a reply. EOF on stdin,
which the bot's exit gives, ends it.

Replies go to a copy of the original stdout; file descriptor 1 itself is
pointed at stderr, so anything the library or its native code prints lands
in the bot's log, never between two replies.

Run as `python -m classes.search_worker` from core/ (the bot's working
directory). It imports nothing of the bot's, so it starts in well under a
second.
"""
import json
import os
import sys


def handle(line: str, text=None) -> dict:
    """One request line in, one reply out. `text` stands in for DDGS().text
    in tests."""
    request_id = None
    try:
        request = json.loads(line)
        request_id = request.get("id")
        if text is None:
            from ddgs import DDGS
            text = DDGS().text
        results = text(str(request["query"]), max_results=int(request["max_results"]))
        return {"id": request_id, "ok": True, "results": list(results or [])}
    except Exception as e:
        return {"id": request_id, "ok": False, "error": f"{type(e).__name__}: {e}"}


def main() -> None:
    # The pipe to the bot is a copy of fd 1; fd 1 (and sys.stdout) become
    # stderr, so not even native code can write between two replies.
    sys.stdout.flush()
    replies = os.fdopen(os.dup(1), "w", encoding="utf-8")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    for line in sys.stdin:
        if not line.strip():
            continue
        replies.write(json.dumps(handle(line), ensure_ascii=False) + "\n")
        replies.flush()


if __name__ == "__main__":
    main()

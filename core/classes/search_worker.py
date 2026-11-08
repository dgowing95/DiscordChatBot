"""The web search itself, in a process of its own (web_research.search).

ddgs drives primp, a native (Rust) HTTP client, from threads of its own. In
one E2E run (2026-11-08) a native crash there ("free(): invalid pointer")
killed the whole bot, chat and voice included, while four searches ran at
once. It could not be reproduced (36 searches, six at a time), so the cause
is not known. In this process, a crash like that costs one search: the bot
sees the worker's end and starts a new one for the next search.

The protocol is one JSON object per line: {"query", "max_results"} in on
stdin, {"ok": true, "results": [...]} or {"ok": false, "error": "..."} out on
stdout. One request at a time (the caller holds a lock). EOF on stdin, which
the bot's exit gives, ends it.

Run as `python -m classes.search_worker` from core/ (the bot's working
directory). It imports nothing of the bot's, so it starts in well under a
second.
"""
import json
import sys


def handle(line: str, text=None) -> dict:
    """One request line in, one reply out. `text` stands in for DDGS().text
    in tests."""
    try:
        request = json.loads(line)
        if text is None:
            from ddgs import DDGS
            text = DDGS().text
        results = text(str(request["query"]), max_results=int(request["max_results"]))
        return {"ok": True, "results": list(results or [])}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def main() -> None:
    # stdout is the protocol: anything the library prints goes to stderr
    # (the bot's log) instead of in between two replies.
    replies, sys.stdout = sys.stdout, sys.stderr
    for line in sys.stdin:
        if not line.strip():
            continue
        replies.write(json.dumps(handle(line), ensure_ascii=False) + "\n")
        replies.flush()


if __name__ == "__main__":
    main()

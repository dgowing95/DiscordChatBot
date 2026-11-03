"""Keeps a long brain run from filling its slot. PURE (stdlib only).

Every model call resends the whole run so far, and a run of tens of shell
commands soon carries pages of old output it no longer needs. This cuts the
OLD tool outputs down to their head and tail, keeping the newest
`keep_recent` whole. The system prompt and every message are left alone, and
every tool call keeps its output item, so the call/result pairs the API
needs stay intact.

Not the SDK's Compaction: that relies on the Responses API's
context_management, and llama.cpp and Strata serve Chat Completions only.

Cutting an old output changes the prompt from that point on, which costs the
server its cached prefix past it. So outputs are cut only once there are
more than keep_recent + BATCH of them, and then down to keep_recent: the
prefix changes once per BATCH new outputs, not on every call.
"""
HEAD_CHARS = 400
TAIL_CHARS = 400
BATCH = 4
MARKER = "\n[… {n} characters cut from this old output …]\n"

_OUTPUT_TYPES = ("function_call_output",)


def _text(output) -> str | None:
    if isinstance(output, str):
        return output
    return None


def cut(text: str, head: int = HEAD_CHARS, tail: int = TAIL_CHARS) -> str:
    """`text` with its middle replaced by a marker, when that makes it
    shorter."""
    removed = len(text) - head - tail
    marker = MARKER.format(n=removed)
    if removed <= len(marker):
        return text
    return text[:head] + marker + text[-tail:]


def trim_items(items: list, keep_recent: int) -> list:
    """`items` (the run's input list) with old tool outputs cut. Never
    changes the list it was given."""
    positions = [i for i, item in enumerate(items)
                 if isinstance(item, dict) and item.get("type") in _OUTPUT_TYPES
                 and _text(item.get("output")) is not None]
    if len(positions) <= keep_recent + BATCH:
        return list(items)
    # Cut everything but the newest keep_recent, rounded so the cut point
    # moves in steps of BATCH (see the module docstring).
    cut_count = ((len(positions) - keep_recent) // BATCH) * BATCH
    out = list(items)
    for i in positions[:cut_count]:
        item = out[i]
        out[i] = dict(item, output=cut(item["output"]))
    return out

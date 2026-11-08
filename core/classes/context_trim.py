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

Cutting by count alone does not keep a run inside its slot: a researcher on a
64k llama.cpp slot reached 60k tokens in 47 turns (v3 health check,
2026-11-07), and past the slot the server refuses the call and the run fails.
So a run on a self-hosted server also watches its prompt against that
server's per-slot context (ContextWatch, the limit read from its /props, so
no size is assumed; a remote API has none and is never squeezed). Past
SQUEEZE_SHARE every old output but the newest SQUEEZE_KEEP is cut, at once
and on every call (no BATCH wait: a few large outputs can fill a slot, and
the cached prefix matters less than the call fitting); past FINISH_SHARE the
run is told to finish. Both stay on once reached: letting the cut outputs
back in would only fill the slot again.
"""
HEAD_CHARS = 400
TAIL_CHARS = 400
BATCH = 4
MARKER = "\n[… {n} characters cut from this old output …]\n"

_OUTPUT_TYPES = ("function_call_output",)

SQUEEZE_SHARE = 0.75
FINISH_SHARE = 0.9
SQUEEZE_KEEP = 1

# Names no tool: the brain, a helper and the verifier each finish differently.
FINISH_NOTE = ("(This run is close to the most the model can read at once. Finish now with what "
               "you have, and say what is still missing.)")


class ContextWatch:
    """One run's prompt size against its server's per-slot context.
    `limit` None (a remote API, or a server whose /props gave none) means
    no limit: it never squeezes and never notes."""

    def __init__(self, limit: int | None):
        self.limit = limit if isinstance(limit, int) and limit > 0 else None
        self.squeezed = False
        self.finishing = False

    def seen(self, prompt_tokens) -> None:
        """The prompt tokens of the response just received."""
        if not isinstance(prompt_tokens, int) or self.limit is None:
            return
        share = prompt_tokens / self.limit
        self.squeezed = self.squeezed or share >= SQUEEZE_SHARE
        self.finishing = self.finishing or share >= FINISH_SHARE

    def keep_recent(self, normal: int) -> int:
        return min(normal, SQUEEZE_KEEP) if self.squeezed else normal

    def batch(self) -> int:
        """trim_items' batch: none once squeezed, so even two big outputs
        are cut."""
        return 1 if self.squeezed else BATCH

    def note(self) -> str | None:
        return FINISH_NOTE if self.finishing else None


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


def trim_items(items: list, keep_recent: int, batch: int = BATCH) -> list:
    """`items` (the run's input list) with old tool outputs cut. Never
    changes the list it was given. `batch` 1 cuts every old output now."""
    positions = [i for i, item in enumerate(items)
                 if isinstance(item, dict) and item.get("type") in _OUTPUT_TYPES
                 and _text(item.get("output")) is not None]
    batch = max(1, batch)
    if len(positions) <= keep_recent + (0 if batch == 1 else batch):
        return list(items)
    # Cut everything but the newest keep_recent, rounded so the cut point
    # moves in steps of batch (see the module docstring).
    cut_count = ((len(positions) - keep_recent) // batch) * batch
    out = list(items)
    for i in positions[:cut_count]:
        item = out[i]
        out[i] = dict(item, output=cut(item["output"]))
    return out

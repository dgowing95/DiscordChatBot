"""What model calls use: tokens and, when the provider says, dollars. Pure.

For the logs and the metrics only, never a card or a reply: the owner
compares models with it (maintainer, 2026-11-10).

A call's usage splits four ways: prompt tokens the provider served from
its cache, the rest of the prompt, the output's thinking (reasoning tokens)
and the rest of the output (the answer and its tool calls). Nothing here weights
them, because the prices are not fixed: OpenRouter billed one DeepSeek
model's cached input at about a quarter of the uncached price while its
model listing said 5/6, and output at about 13 times uncached input, and
those vary with the upstream provider it routes to. So the counts go out as
they are, with the cost the provider reports (OpenRouter's usage has
`cost`, in dollars, on every response; llama.cpp's has none).

WORKFLOW_MAX_TOKENS still counts raw tokens, in and out: it is a brake on a
run that goes round in circles, and one weight would be wrong for every
model but the one it was measured on. What a job costs is capped by
WORKFLOW_MAX_COST instead, on the provider's own figure.
"""
import math
from dataclasses import dataclass


@dataclass
class Tally:
    """Tokens and cost of one call, or of many added up."""
    input: int = 0    # prompt tokens, the cached ones included
    cached: int = 0   # of those, served from the provider's cache
    output: int = 0
    cost: float | None = None  # dollars; None: no call said
    reasoning: int = 0  # of the output, the thinking

    @property
    def total(self) -> int:
        return self.input + self.output

    def add(self, other: "Tally") -> None:
        self.input += other.input
        self.cached += other.cached
        self.output += other.output
        self.reasoning += other.reasoning
        if other.cost is not None:
            self.cost = (self.cost or 0.0) + other.cost

    def text(self) -> str:
        """For a log line, e.g. "5064316 token(s) (4800000 of 5000000 in
        cached, 64316 out, 50000 of them thinking), $0.0812"."""
        line = (f"{self.total} token(s) ({self.cached} of {self.input} in cached, "
                f"{self.output} out, {self.reasoning} of them thinking)")
        return line if self.cost is None else f"{line}, ${self.cost:.4f}"


def _get(obj, name):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _count(value) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


def _first(obj, *names) -> int:
    for name in names:
        value = _get(obj, name)
        if value is not None:
            return _count(value)
    return 0


def _dollars(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) and value >= 0 else None


def from_usage(usage, raw=None) -> Tally:
    """One call's Tally. `usage`: the agents SDK's Usage (input_tokens, ...),
    the OpenAI client's CompletionUsage (prompt_tokens, ...) or either as a
    dict. `raw`: the provider's own usage payload (the SDK's raw_usage), where
    the cost is; without it the cost is looked for on `usage`."""
    if usage is None:
        return Tally()
    inp = _first(usage, "input_tokens", "prompt_tokens")
    details = _get(usage, "input_tokens_details") or _get(usage, "prompt_tokens_details")
    cached = min(_count(_get(details, "cached_tokens")), inp)
    out = _first(usage, "output_tokens", "completion_tokens")
    out_details = _get(usage, "output_tokens_details") or _get(usage, "completion_tokens_details")
    reasoning = min(_count(_get(out_details, "reasoning_tokens")), out)
    cost = _dollars(_get(raw if raw is not None else usage, "cost"))
    return Tally(inp, cached, out, cost, reasoning)

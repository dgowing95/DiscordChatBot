"""The brain's tiers and limits. PURE (stdlib only).

Tier 2 is the brain working inline in the channel (classes/brain.py). Tier 3
(a long workflow in its own thread) does not exist yet, so effective_tier
always answers 2.

The limits are loop guards, not budgets (maintainer, 2026-11-02): a brain
call may take minutes. classes/loop_guard.py is the real guard; the turn and
time limits here only stop a run that gets past it.
"""
import os

TIER_BRAIN = 2

DEFAULT_MAX_TURNS = 100
DEFAULT_TIMEOUT_SECONDS = 1800
DEFAULT_REPEAT_LIMIT = 3
DEFAULT_TRIM_KEEP_RECENT = 6


def _env_int(name: str, default: int, env) -> int:
    """A positive int setting; anything else is the default. (The _env
    prefix is what settings_docs_tests looks for.)"""
    try:
        value = int(str(env.get(name)).strip())
        return value if value > 0 else default
    except (TypeError, ValueError):
        return default


def effective_tier(requested: int | None = None, **_flags) -> int:
    """The tier a brain call runs at. Always 2 until workflows exist."""
    return TIER_BRAIN


def brain_max_turns(env=os.environ) -> int:
    """Model turns one brain call may take (BRAIN_MAX_TURNS, default 100)."""
    return _env_int("BRAIN_MAX_TURNS", DEFAULT_MAX_TURNS, env)


def brain_timeout(env=os.environ) -> int:
    """Wall-clock seconds one brain call may take, lane waits included
    (BRAIN_TIMEOUT, default 1800)."""
    return _env_int("BRAIN_TIMEOUT", DEFAULT_TIMEOUT_SECONDS, env)


def repeat_limit(env=os.environ) -> int:
    """Identical tool calls in a row before the loop guard steps in
    (BRAIN_REPEAT_LIMIT, default 3, at least 2)."""
    return max(2, _env_int("BRAIN_REPEAT_LIMIT", DEFAULT_REPEAT_LIMIT, env))


def trim_keep_recent(env=os.environ) -> int:
    """Tool outputs kept whole in the brain's prompt; older ones are cut
    (BRAIN_TRIM_KEEP_RECENT, default 6)."""
    return _env_int("BRAIN_TRIM_KEEP_RECENT", DEFAULT_TRIM_KEEP_RECENT, env)

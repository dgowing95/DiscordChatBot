"""The brain's tiers and limits. PURE (stdlib only).

Tier 2 is the brain working inline in the channel (classes/brain.py). Tier 3
is a long workflow in its own thread (classes/workflows.py), which the brain
may start only where effective_tier allows it.

The limits are loop guards, not budgets (maintainer, 2026-11-02): a brain
call may take minutes. classes/loop_guard.py is the real guard; the turn and
time limits here only stop a run that gets past it.

A reply the bot was not asked for (the random reply chance, not a mention)
gets far tighter backstops (BRAIN_UNPROMPTED_*): nobody is waiting on it, so
it must never become a half-hour job (F11).

BRAIN_MAX_CONCURRENT caps brain calls at once across the process. Each one
on a queue worker gets a spare worker, so chat stays answered while the
brain works (classes/message_queue.py); the cap bounds how many spares. A
call that finds every slot busy for BRAIN_QUEUE_WAIT seconds is told the bot
is busy.

SAFETY_RULE is the one home of the rule every agent that does work is given
(the brain, and a workflow's manager, helpers, scouts and verifier), since
the content guard checks only search, links and pictures
(classes/content_guard.py). Each builder appends it to its prompt.
"""
import os

TIER_BRAIN = 2
TIER_WORKFLOW = 3

DEFAULT_MAX_TURNS = 100
DEFAULT_TIMEOUT_SECONDS = 1800
DEFAULT_REPEAT_LIMIT = 3
DEFAULT_TRIM_KEEP_RECENT = 6
DEFAULT_UNPROMPTED_MAX_TURNS = 40
DEFAULT_UNPROMPTED_TIMEOUT_SECONDS = 180
DEFAULT_ASK_TIMEOUT_SECONDS = 120
DEFAULT_MAX_CONCURRENT = 3
DEFAULT_QUEUE_WAIT_SECONDS = 120

# Its size is pinned (brain_tests.py). Ends the prompt, after the agent's own
# "Finish by calling ..." line, which "finish at once" points back to.
SAFETY_RULE = ("\n\nTurn down any job meant to cause harm: malware, breaking into or attacking "
               "a machine or an account, probing the network around this workspace, or spying "
               "on someone. Do none of it, and finish at once saying you won't help with that.")


def _env_int(name: str, default: int, env) -> int:
    """A positive int setting; anything else is the default. (The _env
    prefix is what settings_docs_tests looks for.)"""
    try:
        value = int(str(env.get(name)).strip())
        return value if value > 0 else default
    except (TypeError, ValueError):
        return default


def effective_tier(*, unprompted: bool = False, automatic: bool = False, voice: bool = False,
                   env=os.environ) -> int:
    """The highest tier a brain call may use: 3 when it may start a
    workflow (classes/workflows.py), else 2. Never from a reply nobody asked
    for (F11), never in a voice turn (not yet: a call has no thread to
    follow), and never from an automatic run (a schedule, a rule or a
    follow-up): nobody is there to approve its plan."""
    from classes import workflow_policy
    if unprompted or voice or automatic or not workflow_policy.enabled(env):
        return TIER_BRAIN
    return TIER_WORKFLOW


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


def unprompted_max_turns(env=os.environ) -> int:
    """Model turns for a brain call in an unprompted reply
    (BRAIN_UNPROMPTED_MAX_TURNS, default 40)."""
    return _env_int("BRAIN_UNPROMPTED_MAX_TURNS", DEFAULT_UNPROMPTED_MAX_TURNS, env)


def unprompted_timeout(env=os.environ) -> int:
    """Wall-clock seconds for a brain call in an unprompted reply
    (BRAIN_UNPROMPTED_TIMEOUT, default 180)."""
    return _env_int("BRAIN_UNPROMPTED_TIMEOUT", DEFAULT_UNPROMPTED_TIMEOUT_SECONDS, env)


def limits_for(*, unprompted: bool, env=os.environ) -> tuple[int, int]:
    """(max turns, timeout seconds) for one brain call."""
    if unprompted:
        return unprompted_max_turns(env), unprompted_timeout(env)
    return brain_max_turns(env), brain_timeout(env)


def ask_timeout(env=os.environ) -> int:
    """Seconds ask_user waits for the requester's answer
    (BRAIN_ASK_TIMEOUT, default 120). Counted inside the call's timeout."""
    return _env_int("BRAIN_ASK_TIMEOUT", DEFAULT_ASK_TIMEOUT_SECONDS, env)


def max_concurrent(env=os.environ) -> int:
    """Brain calls at once in this process (BRAIN_MAX_CONCURRENT, default 3)."""
    return _env_int("BRAIN_MAX_CONCURRENT", DEFAULT_MAX_CONCURRENT, env)


def queue_wait(env=os.environ) -> int:
    """Seconds a brain call waits for a free slot before it is told the bot
    is busy (BRAIN_QUEUE_WAIT, default 120)."""
    return _env_int("BRAIN_QUEUE_WAIT", DEFAULT_QUEUE_WAIT_SECONDS, env)

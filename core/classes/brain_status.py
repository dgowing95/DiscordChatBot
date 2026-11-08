"""The one-line status on the brain's "Working on it" card.

With /workspace progress off, the card used to show only the request until
the call ended, so a three-minute job looked frozen. Now it also shows what
the brain is doing, in plain words ("Installing python libraries", "Creating
the script"), never a raw command.

One small side call writes the line from the request and the brain's newest
tool calls (brain_progress.BrainProgressHooks decides when):
  - on the brain's own connection (it is the brain's work being described),
    with thinking off and a few tokens;
  - in the background lane, so it never holds a slot chat or the brain
    itself is waiting for, and bounded by STATUS_TIMEOUT_SECONDS, lane wait
    included: a late line is worth nothing;
  - fail-soft: a failure or an unusable answer gives None, and the card
    keeps the line it had (or the rule-written one, tool_cards.helper_line).
"""
import asyncio
import dataclasses
import json
import logging

from classes import llm_lanes
from classes.llm_roles import BRAIN, role_config
from classes.response_filter import strip_thinking
from classes.side_llm import SideLLM

logger = logging.getLogger(__name__)

STATUS_TIMEOUT_SECONDS = 20
MAX_TOKENS = 30
STEP_CHARS = 300      # of one tool call, as the side call sees it
TASK_CHARS = 400      # of the request
MAX_STATUS_CHARS = 60
MAX_STATUS_WORDS = 8

# ONE home; its size is pinned (brain_progress_tests.py). The shape is shown.
PROMPT = ("You write the one status line shown while a bot works on a job. In 2 to 6 words, "
          "say what it is doing now, starting with a word ending in -ing, e.g. \"Installing "
          "python libraries\" or \"Creating the script\". Write only the line.")


def _config():
    return dataclasses.replace(role_config(BRAIN), call_timeout=STATUS_TIMEOUT_SECONDS)


_llm = SideLLM("brain_status", _config, lane=llm_lanes.BACKGROUND)


def step_text(tool: str, args: dict) -> str | None:
    """One tool call as the side call reads it; None for one that says
    nothing new (a poll of a running command, the closing report)."""
    if tool in ("report", "write_stdin"):
        return None
    if tool == "exec_command":
        text = f"ran: {args.get('cmd') or ''}"
    elif tool == "attach_file":
        text = f"sent the user the file {args.get('path') or ''}"
    else:
        try:
            shown = json.dumps(args, ensure_ascii=False)
        except (TypeError, ValueError):
            shown = ""
        text = f"{tool} {shown}".strip()
    return text[:STEP_CHARS]


def status_messages(task: str, steps: list[str]) -> list[dict]:
    lines = "\n".join(f"- {step}" for step in steps)
    return [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": f"The job: {(task or '')[:TASK_CHARS]}\n"
                                    f"What it just did, newest last:\n{lines}"},
    ]


def clean_status(text: str) -> str | None:
    """The model's answer as a status line, or None when it is not one (too
    long, empty, more than a line's worth of words)."""
    for line in strip_thinking(text or "").splitlines():
        line = line.strip().strip("-*•\"'`").strip().rstrip(".!").strip()
        if not line:
            continue
        if len(line) > MAX_STATUS_CHARS or len(line.split()) > MAX_STATUS_WORDS:
            return None
        return line[0].upper() + line[1:]
    return None


async def describe(task: str, steps: list[str]) -> str | None:
    """The status line for the newest steps, or None. Never raises."""
    if not steps:
        return None
    try:
        text = await asyncio.wait_for(
            _llm.complete(status_messages(task, steps), temperature=0.3, max_tokens=MAX_TOKENS),
            STATUS_TIMEOUT_SECONDS)
        return clean_status(text)
    except Exception as e:
        logger.info(f"Brain status: no line this time ({e!r})")
        return None

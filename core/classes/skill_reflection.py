"""After a long brain job: is it worth keeping as a skill?

The brain is told to save skills itself (save_skill), but a model deep in a
job rarely stops to write a recipe. So after a job that took SKILLS_REFLECTION_
MIN_TOOL_CALLS or more tool calls and ended done or partial, and only if the
brain did not already save a skill, ONE small background call reads what
happened and answers with JSON: none, create or update.

It runs after the persona's reply has been sent (fire-and-forget, background
lane, thinking off), so it never delays an answer. Any failure is logged and
counted, never shown: it is a bonus on top of the reply.

The review is built where the run's result is in hand (brain._run_and_deliver)
and carried on the persona's run context (`skill_reviews`) to the message
handler, which starts it once the reply is out.
"""
import asyncio
import json
import logging
import re

from classes import llm_lanes, metrics, skill_format, tool_embeds
from classes.llm_roles import BRAIN, role_config
from classes.response_filter import strip_thinking
from classes.side_llm import SideLLM

logger = logging.getLogger(__name__)

TRANSCRIPT_CHARS = 6000
TOOL_OUTPUT_TAIL = 300
ARGS_CHARS = 300
REFLECTION_TIMEOUT_SECONDS = 90
MAX_TOKENS = 1500

# The one home of this prompt (AGENTS.md "Prompt surface"); its size is pinned
# by a test. The JSON shape is shown, not described.
REFLECTION_PROMPT = """You review a job a Discord bot's brain just finished and decide whether to keep it as a skill: a short recipe for a job that will come up again with different details.
Save one only when the job took several steps that were not obvious, and a similar request would go faster with the recipe. Do not save a one-off, a plain lookup or a job that failed. If a saved skill already covers it, answer "update" with that skill's name only when this job found a better way.
Answer with JSON only, either {"action": "none"} or
{"action": "create", "name": "weekly-tracker-chart", "description": "One line: what it does", "body": "The steps in order, and what to watch for", "note": ""}
("update" has the same fields; note says what changed.)"""


def should_reflect(*, status: str, tool_calls: int, skill_saved: bool, env=None) -> bool:
    cfg = skill_format.settings(env)
    return (cfg["enabled"] and cfg["reflection"] and not skill_saved
            and status in ("done", "partial") and tool_calls >= cfg["reflection_min_tool_calls"])


def _tail(text, limit: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else "…" + text[-limit:]


def _call_line(item) -> str:
    raw = getattr(item, "raw_item", None)
    name = getattr(raw, "name", None) or (raw.get("name") if isinstance(raw, dict) else "") or "tool"
    args = getattr(raw, "arguments", None) or (raw.get("arguments") if isinstance(raw, dict) else "") or ""
    try:
        parsed = json.loads(args) if isinstance(args, str) else args
    except ValueError:
        parsed = None
    if isinstance(parsed, dict) and isinstance(parsed.get("cmd"), str):
        return "$ " + _tail(parsed["cmd"], ARGS_CHARS)
    return f"{name}({_tail(args, ARGS_CHARS)})"


def condense(items, limit: int = TRANSCRIPT_CHARS) -> str:
    """The run as short text: each command (or tool call) and the tail of
    what it printed. Duck-typed on the SDK's run items (`type`, `raw_item`,
    `output`), so it is pure. The newest lines win when it is too long: the
    working version of a script is at the end, after its failed attempts."""
    lines = []
    for item in items or []:
        kind = getattr(item, "type", "")
        if kind == "tool_call_item":
            lines.append(_call_line(item))
        elif kind == "tool_call_output_item":
            lines.append("  → " + _tail(getattr(item, "output", ""), TOOL_OUTPUT_TAIL))
    text = "\n".join(lines)
    return text if len(text) <= limit else "…\n" + text[-limit:]


def build_review(*, request: str, status: str, facts, transcript: str, skills_used,
                 used_bodies: dict, existing_names, channel, guild_id, user_id, job_id: str) -> dict:
    return {"request": str(request or ""), "status": status, "facts": list(facts or []),
            "transcript": transcript, "skills_used": list(skills_used or []),
            "used_bodies": dict(used_bodies or {}), "existing": list(existing_names or []),
            "channel": channel, "guild_id": guild_id, "user_id": user_id, "job_id": job_id}


def review_messages(review: dict) -> list[dict]:
    parts = [f"Request: {review['request'].strip()[:1000]}", f"Result: {review['status']}"]
    if review.get("facts"):
        parts.append("Facts it reported:\n" + "\n".join(f"- {f}" for f in review["facts"][:8]))
    for name in review.get("skills_used", []):
        body = review.get("used_bodies", {}).get(name)
        parts.append(f"Skill it used, {name}:\n{(body or '')[:2000]}")
    if review.get("existing"):
        parts.append("Saved skill names: " + ", ".join(review["existing"][:60]))
    parts.append("What it did:\n" + review["transcript"])
    return [{"role": "system", "content": REFLECTION_PROMPT},
            {"role": "user", "content": "\n\n".join(parts)}]


def parse_decision(text: str) -> dict:
    """The model's answer as {"action": "none"} or a create/update with its
    fields. Anything unreadable is none."""
    text = strip_thinking(str(text or ""))
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {"action": "none"}
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return {"action": "none"}
    if not isinstance(data, dict) or data.get("action") not in ("create", "update"):
        return {"action": "none"}
    out = {"action": data["action"]}
    for key in ("name", "description", "body", "note"):
        out[key] = str(data.get(key) or "").strip()
    return out


def _config():
    import dataclasses
    return dataclasses.replace(role_config(BRAIN), call_timeout=REFLECTION_TIMEOUT_SECONDS)


_llm = SideLLM("skill_reflection", _config, lane=llm_lanes.BACKGROUND)


async def reflect(review: dict) -> str:
    """Runs one review. Returns what it did: none, create, update (or the
    store's `unchanged`) or failed. Never raises."""
    from classes.content_guard import check_web_request
    from classes.skill_store import SkillError, SkillStore
    from classes.skill_tools import announce, saved_embed
    try:
        text = await _llm.complete(review_messages(review), temperature=0.2, max_tokens=MAX_TOKENS)
        decision = parse_decision(text)
        if decision["action"] == "none":
            metrics.inc_skill_reflection("none")
            return "none"
        guild = review["guild_id"]
        store = SkillStore()
        name = skill_format.clean_name(decision["name"])
        exists = await store.get(guild, name)
        if decision["action"] == "update" and exists is None:
            decision["action"] = "create"
        if decision["action"] == "create" and exists is not None:
            decision["action"] = "update"
        allowed, _ = await check_web_request(f"{decision['description']}\n{decision['body']}")
        if not allowed:
            metrics.inc_skill_reflection("failed")
            return "failed"
        result = await store.save(guild, name, decision["description"], decision["body"],
                                  # an update keeps the skill's files: the review reads no files
                                  (exists or {}).get("files"),
                                  author_id=review.get("user_id") or 0, source="reflection",
                                  job_id=review.get("job_id", ""), note=decision["note"])
        if result == "unchanged":
            metrics.inc_skill_reflection("none")
            return "none"
        skill = await store.get(guild, name)
        title, detail = saved_embed(result, name, skill["version"], skill["description"], decision["note"])
        await announce(review.get("channel"), title, name, detail, tool_embeds.SKILL_FOOTER_REFLECTION)
        metrics.inc_skill_reflection(decision["action"])
        return decision["action"]
    except SkillError as e:
        logger.info(f"Skills: reflection's skill was refused: {e}")
    except Exception as e:
        logger.warning(f"Skills: reflection failed: {e}")
    metrics.inc_skill_reflection("failed")
    return "failed"


_running: set = set()


def start(reviews) -> None:
    """Starts each review in the background (called after the reply is
    sent). The tasks are kept in a set so they are not collected midway."""
    for review in reviews or []:
        task = asyncio.get_running_loop().create_task(reflect(review))
        _running.add(task)
        task.add_done_callback(_running.discard)

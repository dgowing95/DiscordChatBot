"""The daily tidy-up of each server's skills.

Skills pile up: the brain and the reflection save one per recurring job, and
two jobs that are really the same end up as two skills. Once every
SKILLS_CURATOR_INTERVAL_HOURS, for each server with SKILLS_CURATOR_MIN_SKILLS
or more live skills, ONE background-lane call reads them all and proposes at
most MAX_CHANGES changes:

  merge     two skills that do the same job become one (the kept one gets a
            new version with the combined steps; the other is soft-deleted)
  retire    soft-delete a skill nobody used in SKILLS_RETIRE_DAYS
  describe  a clearer one-line description

The model only proposes; this module decides. Code works out which skills
are stale (`stale`), and a retire of anything else is dropped, as is a merge
of a name that is not live, a skill named twice, or anything past the limit
(`parse_changes`). Every change is a new version or a soft delete with
source "curator", so /skills history shows it and /skills revert or restore
undoes it. Each is made only if the skill is still at the version the model
read (`expect_version`): a person's edit made meanwhile wins. And an undo
sticks: a skill a person restored, reverted or edited with /skills within
SKILLS_RETIRE_DAYS is `protected`, shown to the model as such and never
changed, so the next pass cannot retire again what someone just restored.

When a pass is due is kept in Redis (`dcb:skills:v1:{guild}:curator`: when
it last ran and a signature of what it saw), so a restart neither repeats a
pass nor skips one, and a Redis lock per guild keeps two processes from
running the same one. A server whose skills have not changed since the last
pass is skipped without a model call; the signature includes the stale set,
so a skill going stale with time makes a pass due again. The signature is
taken AFTER the pass's own changes, so they never make the next pass due.

Each change posts its embed (footer "tidy-up") in the channel where the skill
was last used, or saved; when that channel is gone the embed is skipped.
"""
import asyncio
import hashlib
import json
import logging
import random
import re
import secrets
import time

from classes import llm_lanes, metrics, skill_format, tool_embeds
from classes.llm_roles import BRAIN, role_config
from classes.response_filter import strip_thinking
from classes.side_llm import SideLLM

logger = logging.getLogger(__name__)

MAX_CHANGES = 3
CURATOR_TIMEOUT_SECONDS = 120
MAX_TOKENS = 3000
BODY_CHARS = 600
PROMPT_SKILLS_CHARS = 20000
LOCK_SECONDS = 600
TICK_SECONDS = 3600
FIRST_TICK_SECONDS = (60, 300)
DAY = 86400

# The one home of this prompt (AGENTS.md "Prompt surface"); its size is pinned
# by a test. The JSON shape is shown, not described.
CURATOR_PROMPT = """You tidy a Discord server's saved skills: short recipes the bot's brain reuses for jobs that come up again.
Make a change only where it clearly helps, at most 3:
- "merge": two skills do the same job. Keep the better name and write the combined steps.
- "retire": a skill marked unused that nobody will miss.
- "describe": a description that does not say what the skill is for.
Most days nothing needs changing. Answer with JSON only, for example
{"changes": [{"action": "merge", "keep": "weekly-chart", "drop": "week-chart", "description": "One line: what it does", "body": "The combined steps in order", "note": "merged week-chart"}, {"action": "retire", "name": "old-poll", "note": "not used since spring"}, {"action": "describe", "name": "bar-chart", "description": "Bar chart from pasted numbers, labels on the bars", "note": "clearer"}]}
or {"changes": []}."""


# ------------------------------- pure -------------------------------------

def _idle_since(skill: dict) -> float:
    return max(skill.get("last_used_at") or 0, skill.get("updated_at") or 0)


def stale(skills, now: float, days: int) -> set[str]:
    """Names neither used nor changed in `days`: the only ones a retire may
    name. A new version counts as a change, so an edited skill is not stale."""
    return {s["name"] for s in skills if now - _idle_since(s) >= days * DAY}


PERSON_SOURCES = ("command", "revert")


def protected(skills, now: float, days: int) -> set[str]:
    """Names a person acted on within `days`: restored, or whose current
    version is their /skills edit or revert. No change may touch them."""
    window = days * DAY
    return {s["name"] for s in skills
            if (s.get("restored_at") and now - s["restored_at"] < window)
            or (s.get("source") in PERSON_SOURCES and now - (s.get("updated_at") or 0) < window)}


def view(skills, now: float, days: int) -> tuple[set[str], set[str]]:
    """(stale, protected): what a pass may retire, and what it may not touch."""
    guarded = protected(skills, now, days)
    return stale(skills, now, days) - guarded, guarded


def signature(skills, stale_names) -> str:
    """What a pass saw: every live skill's name and version, and the stale
    set. Equal signatures mean nothing to do."""
    seen = sorted(f"{s['name']}@{s['version']}" for s in skills)
    raw = json.dumps([seen, sorted(stale_names)])
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def due(state: dict | None, sig: str, now: float, interval_hours: int) -> bool:
    if not state:
        return True
    return now - float(state.get("at") or 0) >= interval_hours * 3600 and state.get("sig") != sig


def skills_text(skills, stale_names, now: float, limit: int = PROMPT_SKILLS_CHARS,
                guarded=()) -> str:
    """The skills as the model reads them, best ranked first (so a cut drops
    the least used). Bodies are cut to BODY_CHARS."""
    parts = []
    used = 0
    for s in skill_format.rank(skills, now):
        idle = int((now - _idle_since(s)) // DAY)
        mark = (" [a person changed it lately: leave it]" if s["name"] in guarded
                else " [unused]" if s["name"] in stale_names else "")
        body = s["body"] if len(s["body"]) <= BODY_CHARS else s["body"][:BODY_CHARS] + "…"
        part = (f"## {s['name']}{mark}\n{s['description']}\n"
                f"used {s.get('uses', 0)}x, worked {s.get('successes', 0)}x, idle {idle} days\n{body}")
        if used + len(part) > limit:
            parts.append(f"({len(skills) - len(parts)} more not shown)")
            break
        parts.append(part)
        used += len(part)
    return "\n\n".join(parts)


def curator_messages(skills, stale_names, now: float, guarded=()) -> list[dict]:
    return [{"role": "system", "content": CURATOR_PROMPT},
            {"role": "user", "content": skills_text(skills, stale_names, now, guarded=guarded)}]


def _text(change, key) -> str:
    return str(change.get(key) or "").strip()


def parse_changes(text: str, live_names, stale_names, limit: int = MAX_CHANGES) -> list[dict]:
    """The model's answer as the changes that may be made, in its order.
    Anything unreadable is no change. Dropped: unknown actions, names that
    are not live, retires of skills that are not stale, a merge into itself,
    a skill named by an earlier change, and everything past `limit`."""
    text = strip_thinking(str(text or ""))
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return []
    changes = data.get("changes") if isinstance(data, dict) else None
    if not isinstance(changes, list):
        return []
    live, stale_set = set(live_names), set(stale_names)
    taken, out = set(), []
    for change in changes:
        if len(out) >= limit:
            break
        if not isinstance(change, dict):
            continue
        action = change.get("action")
        note = skill_format.one_line(_text(change, "note"), skill_format.NOTE_CHARS)
        if action == "merge":
            keep = skill_format.clean_name(change.get("keep"))
            drop = skill_format.clean_name(change.get("drop"))
            names = {keep, drop}
            ok = (keep != drop and names <= live and not names & taken
                  and _text(change, "description") and _text(change, "body"))
            if ok:
                out.append({"action": "merge", "keep": keep, "drop": drop, "note": note,
                            "description": _text(change, "description"), "body": _text(change, "body")})
        elif action == "retire":
            name = skill_format.clean_name(change.get("name"))
            names = {name}
            ok = name in live and name not in taken and name in stale_set
            if ok:
                out.append({"action": "retire", "name": name, "note": note})
        elif action == "describe":
            name = skill_format.clean_name(change.get("name"))
            names = {name}
            ok = name in live and name not in taken and bool(_text(change, "description"))
            if ok:
                out.append({"action": "describe", "name": name, "note": note,
                            "description": _text(change, "description")})
        else:
            continue
        if ok:
            taken |= names
    return out


# ------------------------------ one pass ----------------------------------

def _config():
    import dataclasses
    return dataclasses.replace(role_config(BRAIN), call_timeout=CURATOR_TIMEOUT_SECONDS)


_llm = SideLLM("skill_curator", _config, lane=llm_lanes.BACKGROUND)


def _state_key(guild) -> str:
    from classes.skill_store import PREFIX
    return f"{PREFIX}:{guild}:curator"


def _lock_key(guild) -> str:
    from classes.skill_store import PREFIX
    return f"{PREFIX}:{guild}:curator-lock"


_RELEASE = "if redis.call('GET', KEYS[1]) == ARGV[1] then return redis.call('DEL', KEYS[1]) end return 0"


class Curator:
    """One server's pass (`curate`) and the loop over every server
    (`run_forever`). `channel_of(id)` finds a channel for the embeds (the
    Discord client's get_channel); `actor_id` is the bot's own user id, so
    /skills history shows the bot made the change."""

    def __init__(self, store, channel_of, actor_id, *, llm=None, clock=time.time):
        self.store = store
        self.redis = store.redis
        self.channel_of = channel_of
        self.actor_id = actor_id
        self.llm = llm or _llm
        self.clock = clock

    async def _state(self, guild):
        raw = await self.redis.get(_state_key(guild))
        try:
            return json.loads(raw) if raw else None
        except ValueError:
            return None

    async def _remember(self, guild, sig: str) -> None:
        await self.redis.set(_state_key(guild), json.dumps({"at": self.clock(), "sig": sig}))

    async def curate(self, guild, force: bool = False) -> str:
        """Runs a pass for one server when it is due. Returns what happened:
        off, few, not due, busy, failed, or the number of changes made.
        Never raises."""
        cfg = skill_format.settings()
        if not (cfg["enabled"] and cfg["curator"]):
            return "off"
        try:
            skills = await self.store.list(guild)
            if len(skills) < cfg["curator_min_skills"]:
                return "few"
            now = self.clock()
            stale_names, guarded = view(skills, now, cfg["retire_days"])
            sig = signature(skills, stale_names)
            if not force and not due(await self._state(guild), sig, now, cfg["curator_interval_hours"]):
                return "not due"
            token = secrets.token_hex(8)
            if not await self.redis.set(_lock_key(guild), token, nx=True, ex=LOCK_SECONDS):
                return "busy"
        except Exception as e:
            logger.warning(f"Skills: tidy-up of guild {guild} could not start: {e}")
            return "failed"
        try:
            try:
                text = await self.llm.complete(curator_messages(skills, stale_names, now, guarded),
                                               temperature=0.2, max_tokens=MAX_TOKENS)
            except Exception as e:
                logger.warning(f"Skills: tidy-up of guild {guild} failed: {e}")
                # Remembered with no signature: tried again after the interval,
                # not every hour.
                await self._remember(guild, "")
                metrics.inc_skill_curator_change("failed")
                return "failed"
            by_name = {s["name"]: s for s in skills}
            # A protected skill is not offered as live: no change can name it.
            changes = parse_changes(text, set(by_name) - guarded, stale_names)
            made = 0
            for change in changes:
                made += await self._apply(guild, change, by_name, cfg)
            # After the changes, so the pass's own new versions never make the
            # next one due.
            after = await self.store.list(guild)
            await self._remember(guild, signature(after, view(after, self.clock(), cfg["retire_days"])[0]))
            logger.info(f"Skills: tidy-up of guild {guild}: {made} change(s) of {len(changes)} proposed")
            return str(made)
        except Exception as e:
            logger.warning(f"Skills: tidy-up of guild {guild} failed: {e}")
            return "failed"
        finally:
            try:
                await self.redis.eval(_RELEASE, 1, _lock_key(guild), token)
            except Exception as e:
                logger.warning(f"Skills: could not release the tidy-up lock of guild {guild}: {e}")

    async def _apply(self, guild, change: dict, by_name: dict, cfg) -> int:
        """Makes one change. Returns 1 when it was made, 0 when refused (a
        person changed the skill meanwhile, the guard said no, a limit)."""
        from classes.content_guard import check_web_request
        from classes.skill_store import SkillError
        kind = change["action"]
        try:
            if kind in ("merge", "describe"):
                text = "\n".join([change["description"], change.get("body", ""), change["note"]])
                allowed, _ = await check_web_request(text)
                if not allowed:
                    raise SkillError("the safety guard refused the new text")
            if kind == "merge":
                await self._merge(guild, by_name[change["keep"]], by_name[change["drop"]], change)
            elif kind == "retire":
                skill = by_name[change["name"]]
                await self.store.delete(guild, skill["name"], self.actor_id, expect_version=skill["version"])
                # Fixed wording: the model's note is not shown for a retire.
                await self._announce(skill, tool_embeds.SKILL_RETIRED,
                                     f"not used in {cfg['retire_days']} days (restore with /skills restore)")
            else:
                skill = by_name[change["name"]]
                await self._save(guild, skill, change["description"], skill["body"], skill["files"],
                                 change["note"] or "clearer description")
        except SkillError as e:
            logger.info(f"Skills: tidy-up left {change.get('name') or change.get('keep')} alone: {e}")
            metrics.inc_skill_curator_change("refused")
            return 0
        metrics.inc_skill_curator_change(kind)
        return 1

    async def _save(self, guild, skill, description, body, files, note) -> None:
        from classes.skill_store import UNCHANGED, SkillError
        result = await self.store.save(guild, skill["name"], description, body, files,
                                       author_id=self.actor_id, source="curator", note=note,
                                       expect_version=skill["version"])
        if result == UNCHANGED:
            raise SkillError("the proposed text is what it already says")
        fresh = await self.store.get(guild, skill["name"])
        await self._announce(skill, tool_embeds.SKILL_UPDATED, f"v{fresh['version']}: {note}")

    async def _merge(self, guild, keep, drop, change) -> None:
        """The kept skill first, then the other's delete: if the delete is
        refused, two skills overlap, but nothing is lost."""
        from classes.skill_store import SkillError, VersionError
        note = change["note"] or f"merged {drop['name']}"
        try:
            # Both skills' files; the kept one's win a name clash.
            await self._save(guild, keep, change["description"], change["body"],
                             {**drop["files"], **keep["files"]}, note)
        except VersionError:
            raise
        except SkillError:
            if not drop["files"]:
                raise
            # Over a file limit together: the kept skill's files only.
            await self._save(guild, keep, change["description"], change["body"], keep["files"], note)
        try:
            await self.store.delete(guild, drop["name"], self.actor_id, expect_version=drop["version"])
        except VersionError:
            logger.info(f"Skills: tidy-up merged into {keep['name']} but kept {drop['name']}: "
                        "it changed meanwhile")
            return
        await self._announce(drop, tool_embeds.SKILL_RETIRED,
                             f"merged into {keep['name']} (restore with /skills restore)")

    async def _announce(self, skill, title, detail) -> None:
        from classes.skill_tools import announce
        channel_id = skill.get("last_channel_id")
        channel = self.channel_of(channel_id) if channel_id else None
        if channel is None:
            logger.info(f"Skills: no channel for the tidy-up embed of {skill['name']}")
            return
        await announce(channel, title, skill["name"], detail, tool_embeds.SKILL_FOOTER_CURATOR)

    async def tick(self, guild_ids) -> None:
        """One round: the live-skill gauge, and a pass for each due server."""
        total = 0
        for guild in guild_ids:
            await self.curate(guild)
            try:  # after the pass, so its retires are counted
                total += await self.store.count_live(guild)
            except Exception as e:
                logger.warning(f"Skills: could not count guild {guild}'s skills: {e}")
        metrics.set_skills_live(total)

    async def run_forever(self, guild_ids) -> None:
        """`guild_ids()` is read each round, so joined and left servers are
        followed. The first round waits a few minutes after start, so a
        deploy's first boot does not land on top of the morning's chat."""
        await asyncio.sleep(random.uniform(*FIRST_TICK_SECONDS))
        while True:
            try:
                await self.tick(list(guild_ids()))
            except Exception:
                logger.exception("Skills: tidy-up round failed")
            await asyncio.sleep(TICK_SECONDS)

"""The brain's skill tools: use_skill, save_skill, search_skills.

A skill is how the brain solved a recurring job, kept per server in Redis
(classes/skill_store.py). The brain sees an index of them in its task
(skill_format.render_index) and opens one with use_skill.

use_skill writes the skill into the workspace, under skills/<name>/, FRESH
FROM THE STORE every time, so a file someone changed inside the container can
never become the skill (the trust rule: the store is the only source, the
workspace copy is a read-only convenience). It is written beside the final
folder and then moved into place, under a lock per skill, so two calls using
the same skill at once never see a half-written copy.

Every event posts its own notice card in the channel (tool_cards), like any
tool. The card helpers here (announce, RestoreButton) are also used by the
/skills commands, the reflection and the curator.
"""
import asyncio
import logging
import posixpath
import secrets
from pathlib import PurePosixPath
import io

import discord
from agents import function_tool, RunContextWrapper

from classes import card, card_kit, metrics, skill_format, tool_cards
from classes.content_guard import check_web_request
from classes.skill_store import QuotaError, SkillError, SkillStore, CREATED, UNCHANGED
from classes.workspace_lifecycle import SKILLS_DIR

logger = logging.getLogger(__name__)

# One lock per (guild, skill): the copy in the workspace is replaced as a unit.
_locks: dict[tuple, asyncio.Lock] = {}


def _lock(guild, name) -> asyncio.Lock:
    return _locks.setdefault((guild, name), asyncio.Lock())


RESTORE_TEMPLATE = r"skill:restore:(?P<name>[a-z0-9-]{3,48})"


class RestoreButton(discord.ui.DynamicItem[discord.ui.Button], template=RESTORE_TEMPLATE):
    """Restore, on a deleted or retired skill's card: what /skills restore
    does, for anyone in the server (no permission check, by decision: every
    change is a version and a delete is undoable). A DynamicItem registered
    once (main.on_ready), so it works on a card posted before a restart."""

    def __init__(self, name: str):
        super().__init__(discord.ui.Button(label="Restore", emoji="♻️",
                                           style=discord.ButtonStyle.success,
                                           custom_id=f"skill:restore:{name}"))
        self.name = name

    @classmethod
    async def from_custom_id(cls, interaction, item, match):
        return cls(match["name"])

    async def callback(self, interaction: discord.Interaction) -> None:
        # Registered whatever the setting, so an old card's button answers;
        # with skills off it restores nothing, like the hidden /skills.
        if not skill_format.enabled():
            await card.reply(interaction, card_kit.error("Skills are switched off.", "❌ Not restored"))
            return
        try:
            await SkillStore().restore(interaction.guild_id, self.name, interaction.user.id)
        except SkillError as e:
            await card.reply(interaction, card_kit.error(str(e), "❌ Not restored"))
            return
        except Exception as e:
            logger.warning(f"Skills: restore of {self.name} failed: {e!r}")
            await card.reply(interaction, card_kit.error(
                "Something went wrong. Try /skills restore in a moment.", "❌ Not restored"))
            return
        # The card it was on loses its button, and says who restored it.
        texts = card.texts_of(interaction.message)
        done = card.KitView([card_kit.Text(t) for t in texts]
                            + [card_kit.Text(card_kit.small(f"♻️ Restored by {interaction.user.mention}"))],
                            colour=card_kit.CLOSED)
        try:
            await interaction.response.edit_message(view=done, allowed_mentions=card.NO_MENTIONS)
        except Exception as e:
            logger.info(f"Skills: could not update the card after a restore: {e}")
        await announce(interaction.channel, tool_cards.SKILL_RESTORED, self.name)


async def announce(channel, label: str, name: str, detail: str = "", footer: str | None = None,
                   version: int | None = None) -> None:
    """Posts a skill event's card. Never raises: the card is a note, and a
    channel the bot cannot write to must not cost the skill change. A deleted
    or retired skill's card has a Restore button."""
    if channel is None:
        return
    spec = tool_cards.skill_card(label, name, detail, footer, version)
    if label in (tool_cards.SKILL_DELETED, tool_cards.SKILL_RETIRED):
        spec.body.append(card_kit.Buttons((RestoreButton(name),)))
    await card.notice(channel, spec)


def saved_card(result: str, name: str, version: int, description: str, note: str):
    """(label, detail, version) for a save's card: a new skill shows its
    description, an update its version and note."""
    if result == CREATED:
        return tool_cards.SKILL_SAVED, description, None
    return tool_cards.SKILL_UPDATED, note, version


def _ctx(wrapper) -> dict:
    return wrapper.context


def _scope(ctx: dict):
    guild = ctx.get("guild_id")
    if not guild:
        raise SkillError("Skills belong to a server; there is none here.")
    return guild


async def _guarded(description: str, body: str) -> str | None:
    """A refusal text when the content guard blocks the text, else None."""
    allowed, reason = await check_web_request(f"{description}\n{body}")
    if allowed:
        return None
    return f"Not saved: the safety guard blocked this skill ({reason or 'content refused'})."


async def write_to_workspace(ctx: dict, skill: dict) -> str | None:
    """Writes skills/<name>/ in the workspace from the store's copy. Returns
    a problem text, or None."""
    session, root = ctx.get("session"), ctx.get("workspace_root")
    if session is None or not root:
        return "no workspace to write the skill into"
    name = skill["name"]
    final = posixpath.join(root, SKILLS_DIR, name)
    temp = posixpath.join(root, SKILLS_DIR, f".{name}-{secrets.token_hex(4)}")
    files = {"SKILL.md": skill_format.render_skill_md(name, skill["description"], skill["body"]),
             **(skill.get("files") or {})}
    async with _lock(ctx.get("guild_id"), name):
        try:
            for path, text in files.items():
                target = PurePosixPath(temp, path)
                made = await session.exec("mkdir", "-p", "--", str(target.parent), shell=False)
                if not made.ok():
                    raise RuntimeError(f"could not make {target.parent}")
                await session.write(target, io.BytesIO(text.encode("utf-8")))
            # The old copy may be read-only (below), so make it writable to
            # remove it; then move the new one in and lock it.
            script = ('chmod -R u+w -- "$2" 2>/dev/null; rm -rf -- "$2" && mv -- "$1" "$2" '
                      '&& chmod -R a-w -- "$2"')
            moved = await session.exec("sh", "-c", script, "sh", temp, final, shell=False)
            if not moved.ok():
                raise RuntimeError("could not move the skill into place")
        except Exception as e:
            logger.warning(f"Skills: could not write {name} to the workspace: {e}")
            try:
                await session.exec("rm", "-rf", "--", temp, shell=False)
            except Exception:
                pass
            return "the workspace copy could not be written"
    return None


def _file_list(skill: dict) -> str:
    paths = sorted((skill.get("files") or {}))
    return ", ".join(paths) if paths else "none"


@function_tool
async def use_skill(wrapper: RunContextWrapper[dict], name: str) -> str:
    """Opens one of this server's saved skills: how a job like this was done
    before. Returns its steps and puts its files in skills/<name>/.

    Args:
        name: A name from the skills list in your task, e.g. "weekly-tracker-chart".
    """
    ctx = _ctx(wrapper)
    try:
        guild = _scope(ctx)
        store = SkillStore()
        wanted = skill_format.clean_name(name)
        skill = await store.get(guild, wanted)
        if skill is None:
            names = [s["name"] for s in await store.list(guild)]
            near = skill_format.closest_names(names, wanted)
            return (f"No skill named {name!r}." +
                    (f" Closest: {', '.join(near)}." if near else " This server has no skills yet."))
        problem = await write_to_workspace(ctx, skill)
        channel = ctx.get("channel")
        await store.record_use(guild, wanted, getattr(channel, "id", None))
        ctx.setdefault("skills_used", [])
        if wanted not in ctx["skills_used"]:
            ctx["skills_used"].append(wanted)
        metrics.inc_skill_used()
        await announce(channel, tool_cards.SKILL_USED, wanted, skill["description"])
        out = [f"Skill {wanted} (v{skill['version']}): {skill['description']}", "", skill["body"]]
        if problem:
            out += ["", f"Its files could not be copied ({problem}); follow the steps above."]
        else:
            extra = f", {_file_list(skill)}" if skill.get("files") else ""
            out += ["", f"Files (in {SKILLS_DIR}/{wanted}/, read-only; copy what you change): "
                        f"SKILL.md{extra}"]
        return "\n".join(out)
    except SkillError as e:
        return str(e)


async def _read_files(ctx: dict, paths) -> tuple[dict, list[str]]:
    """The workspace files named in save_skill, as {skills path: text}. Paths
    are relative to the brain's folder; each is stored under scripts/ by its
    base name."""
    from classes.workspace_tools import _job_path
    session = ctx.get("session")
    files, problems = {}, []
    for path in paths or []:
        absolute = _job_path(ctx, str(path).strip()) if session is not None else None
        if absolute is None:
            problems.append(f"{path}: not a path inside your folder")
            continue
        try:
            result = await session.exec("cat", "--", absolute, shell=False)
            if not result.ok():
                problems.append(f"{path}: not found")
                continue
            text = result.stdout.decode("utf-8")
        except UnicodeDecodeError:
            problems.append(f"{path}: not a text file")
            continue
        except Exception as e:
            logger.warning(f"Skills: could not read {path!r}: {e}")
            problems.append(f"{path}: could not be read")
            continue
        files[f"scripts/{posixpath.basename(absolute)}"] = text
    return files, problems


@function_tool
async def save_skill(wrapper: RunContextWrapper[dict], name: str, description: str, steps: str,
                     files: list[str] | None = None, note: str = "") -> str:
    """Saves how you did a job as a skill the whole server can reuse, or
    updates one with the same name.

    Args:
        name: Short, lower-case, hyphens: "weekly-tracker-chart".
        description: One line: what the skill does.
        steps: What to do, in order, and what to watch for.
        files: Paths in your folder to keep with it, e.g. the script that did the work.
            Leave out to keep an existing skill's files; [] removes them.
        note: When updating, what changed.
    """
    ctx = _ctx(wrapper)
    try:
        guild = _scope(ctx)
        store = SkillStore()
        clean = skill_format.clean_name(name)
        if files is None:
            # Left out: keep the skill's files. Its copy in skills/ is
            # read-only, so the brain usually could not pass them again.
            stored, problems = ((await store.get(guild, clean)) or {}).get("files") or {}, []
        else:
            stored, problems = await _read_files(ctx, files)
        if problems:
            return "Not saved: " + "; ".join(problems)
        refusal = await _guarded(description, steps)
        if refusal:
            return refusal
        author = ctx.get("user_id") or 0
        job_id = posixpath.basename(ctx.get("job_dir") or "")
        result = await store.save(guild, clean, description, steps, stored, author_id=author,
                                  source="brain", job_id=job_id, note=note,
                                  channel_id=getattr(ctx.get("channel"), "id", None))
    except QuotaError as e:
        return f"Not saved: {e}"
    except SkillError as e:
        return str(e)
    ctx["skill_saved"] = True
    if result == UNCHANGED:
        return f"{clean} is already saved exactly like this."
    skill = await store.get(guild, clean)
    label, detail, version = saved_card(result, clean, skill["version"], skill["description"],
                                         skill_format.one_line(note, skill_format.NOTE_CHARS))
    await announce(ctx.get("channel"), label, clean, detail, version=version)
    return f"Skill {clean} {result} (v{skill['version']})."


@function_tool
async def search_skills(wrapper: RunContextWrapper[dict], query: str) -> str:
    """Looks through this server's saved skills for ones that fit a job.

    Args:
        query: What the job is, in a few words.
    """
    ctx = _ctx(wrapper)
    try:
        guild = _scope(ctx)
    except SkillError as e:
        return str(e)
    found = skill_format.search(await SkillStore().list(guild), query)
    if not found:
        return "No saved skill matches that."
    text = "\n".join(f"- {s['name']}: {s['description']}" for s in found)
    # A workflow's manager searches but has no use_skill (brain_tools).
    return text if ctx.get("workflow_id") else text + "\n(use_skill opens one)"


def skill_tools() -> list:
    """The tools a brain call gets. All three always: a tool list that varied
    with the server's skill count would change the brain's cached prefix."""
    return [use_skill, save_skill, search_skills]

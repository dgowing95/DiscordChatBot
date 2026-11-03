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

Every event posts its own embed in the channel (tool_embeds), like any tool.
The embed helpers here (announce) are also used by the /skills commands, the
reflection and the curator.
"""
import asyncio
import logging
import posixpath
import secrets
from pathlib import PurePosixPath
import io

from agents import function_tool, RunContextWrapper

from classes import metrics, skill_format, tool_embeds
from classes.common import Common
from classes.content_guard import check_web_request
from classes.skill_store import QuotaError, SkillError, SkillStore, CREATED, UNCHANGED
from classes.workspace_lifecycle import SKILLS_DIR

logger = logging.getLogger(__name__)

# One lock per (guild, skill): the copy in the workspace is replaced as a unit.
_locks: dict[tuple, asyncio.Lock] = {}


def _lock(guild, name) -> asyncio.Lock:
    return _locks.setdefault((guild, name), asyncio.Lock())


async def announce(channel, title: str, name: str, detail: str = "", footer: str | None = None) -> None:
    """Posts a skill event's embed. Never raises: the embed is a note, and a
    channel the bot cannot write to must not cost the skill change."""
    if channel is None:
        return
    try:
        await Common.send_tool_discord_embed(
            channel, tool_embeds.skill_line(name, detail), tool_embeds.COLOR_SKILL, title, footer)
    except Exception as e:
        logger.warning(f"Skills: could not post the {title!r} embed: {e}")


def saved_embed(result: str, name: str, version: int, description: str, note: str):
    """(title, detail) for a save's embed: new skills show their description,
    updates show 'v3: note'."""
    if result == CREATED:
        return tool_embeds.SKILL_SAVED, description
    detail = f"v{version}: {note}" if note else f"v{version}"
    return tool_embeds.SKILL_UPDATED, detail


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
        await announce(channel, tool_embeds.SKILL_USED, wanted)
        out = [f"Skill {wanted} (v{skill['version']}): {skill['description']}", "", skill["body"]]
        if problem:
            out += ["", f"Its files could not be copied ({problem}); follow the steps above."]
        else:
            out += ["", f"Files (in {SKILLS_DIR}/{wanted}/, read-only; copy what you change): "
                        f"SKILL.md, {_file_list(skill)}"]
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
        note: When updating, what changed.
    """
    ctx = _ctx(wrapper)
    try:
        guild = _scope(ctx)
        stored, problems = await _read_files(ctx, files)
        if problems:
            return "Not saved: " + "; ".join(problems)
        refusal = await _guarded(description, steps)
        if refusal:
            return refusal
        author = ctx.get("user_id") or 0
        job_id = posixpath.basename(ctx.get("job_dir") or "")
        store = SkillStore()
        clean = skill_format.clean_name(name)
        result = await store.save(guild, clean, description, steps, stored, author_id=author,
                                  source="brain", job_id=job_id, note=note)
    except QuotaError as e:
        return f"Not saved: {e}"
    except SkillError as e:
        return str(e)
    ctx["skill_saved"] = True
    if result == UNCHANGED:
        return f"{clean} is already saved exactly like this."
    skill = await store.get(guild, clean)
    title, detail = saved_embed(result, clean, skill["version"], skill["description"],
                                skill_format.one_line(note, skill_format.NOTE_CHARS))
    await announce(ctx.get("channel"), title, clean, detail)
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
    return "\n".join(f"- {s['name']}: {s['description']}" for s in found) + "\n(use_skill opens one)"


def skill_tools() -> list:
    """The tools a brain call gets. All three always: a tool list that varied
    with the server's skill count would change the brain's cached prefix."""
    return [use_skill, save_skill, search_skills]

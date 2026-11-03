"""/skills: see and manage this server's saved skills.

Anyone in the server can (decision: skills belong to everyone, every change
is a new version and a delete is soft, so nothing is lost for good). Every
change posts its own embed in the channel, like the brain's tools do, so the
channel always shows who changed what; the command's own reply is private.
"""
import io
import logging

import discord
from discord import app_commands

from classes import skill_format, tool_embeds
from classes.content_guard import check_web_request
from classes.skill_store import SkillError, SkillStore
from classes.skill_tools import announce, saved_embed

logger = logging.getLogger(__name__)

MODAL_BODY_CHARS = 4000  # Discord's cap for a text input
SHOW_INLINE_CHARS = 1800
DEFAULT_UPLOAD_BYTES = 10 * 1024 * 1024  # Discord's limit without boosts


def _ago(ts) -> str:
    return f"<t:{int(ts)}:R>" if ts else "never"


def list_lines(skills) -> str:
    """The /skills list embed's text: one line per skill, best first."""
    import time
    lines = []
    for skill in skill_format.rank(skills, time.time()):
        used = skill.get("uses", 0)
        lines.append(f"**{skill['name']}** — {skill_format.one_line(skill['description'], 120)}\n"
                     f"-# v{skill['version']} · used {used}× · last {_ago(skill.get('last_used_at'))}")
    return "\n".join(lines)[:4000]


def history_lines(rows) -> str:
    lines = []
    for row in rows:
        who = f"<@{row['author_id']}>" if row.get("author_id") else "someone"
        note = f" — {row['note']}" if row.get("note") else ""
        mark = " (current)" if row.get("current") else ""
        lines.append(f"**v{row['v']}**{mark} · {_ago(row['created_at'])} · {who} · "
                     f"{row.get('source') or '?'}{note}")
    return "\n".join(lines)[:3900]


def show_text(skill: dict) -> str:
    text = skill_format.render_skill_md(skill["name"], skill["description"], skill["body"])
    for path, content in sorted((skill.get("files") or {}).items()):
        text += f"\n--- {path} ---\n{content}\n"
    return text


async def _names(ctx: discord.Interaction, current: str, *, deleted: bool = False):
    try:
        rows = await SkillStore().list(ctx.guild_id, include_deleted=deleted)
    except Exception as e:
        logger.warning(f"Skills: autocomplete failed: {e}")
        return []
    names = [r["name"] for r in rows if bool(r["deleted"]) == deleted]
    current = (current or "").lower()
    return [app_commands.Choice(name=n, value=n) for n in names if current in n][:25]


async def _live_names(ctx, current):
    return await _names(ctx, current)


async def _deleted_names(ctx, current):
    return await _names(ctx, current, deleted=True)


class EditModal(discord.ui.Modal, title="Edit skill"):
    description = discord.ui.TextInput(label="Description (one line)", max_length=skill_format.DESCRIPTION_CHARS)
    steps = discord.ui.TextInput(label="Steps", style=discord.TextStyle.paragraph, max_length=MODAL_BODY_CHARS)
    note = discord.ui.TextInput(label="What changed (optional)", required=False,
                                max_length=skill_format.NOTE_CHARS)

    def __init__(self, skill: dict):
        super().__init__()
        self.skill = skill
        self.description.default = skill["description"]
        self.steps.default = skill["body"]

    async def on_submit(self, ctx: discord.Interaction) -> None:
        name = self.skill["name"]
        store = SkillStore()
        try:
            allowed, reason = await check_web_request(f"{self.description.value}\n{self.steps.value}")
            if not allowed:
                raise SkillError(f"Not saved: the safety guard blocked this ({reason or 'content refused'}).")
            result = await store.save(ctx.guild_id, name, str(self.description.value), str(self.steps.value),
                                      self.skill.get("files") or {}, author_id=ctx.user.id,
                                      source="command", note=str(self.note.value or ""),
                                      channel_id=ctx.channel_id)
        except SkillError as e:
            await ctx.response.send_message(str(e), ephemeral=True)
            return
        if result == "unchanged":
            await ctx.response.send_message("Nothing changed.", ephemeral=True)
            return
        skill = await store.get(ctx.guild_id, name)
        title, detail = saved_embed(result, name, skill["version"], skill["description"],
                                    str(self.note.value or ""))
        await announce(ctx.channel, title, name, detail)
        await ctx.response.send_message(f"Saved {name} (v{skill['version']}).", ephemeral=True)


def register_skill_commands(tree) -> None:
    group = app_commands.Group(name="skills", guild_only=True,
                               description="The skills I have learned in this server")

    @group.command(name="list", description="The skills saved in this server")
    async def list_skills(ctx: discord.Interaction):
        skills = await SkillStore().list(ctx.guild_id)
        if not skills:
            await ctx.response.send_message("No skills saved yet. I save one when a job is worth "
                                            "repeating.", ephemeral=True)
            return
        embed = discord.Embed(title=f"📘 Skills ({len(skills)})", description=list_lines(skills),
                              color=tool_embeds.COLOR_SKILL)
        await ctx.response.send_message(embed=embed)

    @group.command(name="show", description="Read a skill")
    @app_commands.autocomplete(name=_live_names)
    async def show(ctx: discord.Interaction, name: str, version: int | None = None):
        skill = await SkillStore().get(ctx.guild_id, skill_format.clean_name(name), version)
        if skill is None:
            await ctx.response.send_message("No such skill (or version).", ephemeral=True)
            return
        text = show_text(skill)
        if len(text) <= SHOW_INLINE_CHARS:
            await ctx.response.send_message(f"```md\n{text}\n```", ephemeral=True)
        else:
            await ctx.response.send_message(
                f"{skill['name']} v{skill['version']}", ephemeral=True,
                file=discord.File(io.BytesIO(text.encode("utf-8")), filename=f"{skill['name']}.md"))

    @group.command(name="history", description="Who changed a skill, and when")
    @app_commands.autocomplete(name=_live_names)
    async def history(ctx: discord.Interaction, name: str):
        rows = await SkillStore().history(ctx.guild_id, skill_format.clean_name(name))
        if not rows:
            await ctx.response.send_message("No such skill.", ephemeral=True)
            return
        # An embed: up to SKILLS_MAX_VERSIONS lines is past a message's 2000.
        embed = discord.Embed(title=f"{skill_format.clean_name(name)}: history",
                              description=history_lines(rows), color=tool_embeds.COLOR_SKILL)
        await ctx.response.send_message(embed=embed, ephemeral=True)

    @group.command(name="edit", description="Change a skill's description and steps")
    @app_commands.autocomplete(name=_live_names)
    async def edit(ctx: discord.Interaction, name: str):
        skill = await SkillStore().get(ctx.guild_id, skill_format.clean_name(name))
        if skill is None:
            await ctx.response.send_message("No such skill.", ephemeral=True)
            return
        if len(skill["body"]) > MODAL_BODY_CHARS:
            await ctx.response.send_message(
                f"This skill is too long for the edit box ({MODAL_BODY_CHARS} characters). "
                "Ask me to update it instead.", ephemeral=True)
            return
        await ctx.response.send_modal(EditModal(skill))

    async def _change(ctx, name, action, title, detail=""):
        name = skill_format.clean_name(name)
        try:
            await action(name)
        except SkillError as e:
            await ctx.response.send_message(str(e), ephemeral=True)
            return
        await announce(ctx.channel, title, name, detail)
        await ctx.response.send_message("Done.", ephemeral=True)

    @group.command(name="delete", description="Delete a skill (it can be restored)")
    @app_commands.autocomplete(name=_live_names)
    async def delete(ctx: discord.Interaction, name: str):
        await _change(ctx, name, lambda n: SkillStore().delete(ctx.guild_id, n, ctx.user.id),
                      tool_embeds.SKILL_DELETED, "restore with /skills restore")

    @group.command(name="restore", description="Bring back a deleted skill")
    @app_commands.autocomplete(name=_deleted_names)
    async def restore(ctx: discord.Interaction, name: str):
        await _change(ctx, name, lambda n: SkillStore().restore(ctx.guild_id, n, ctx.user.id),
                      tool_embeds.SKILL_RESTORED)

    @group.command(name="revert", description="Go back to an earlier version of a skill")
    @app_commands.autocomplete(name=_live_names)
    async def revert(ctx: discord.Interaction, name: str, version: int):
        await _change(ctx, name, lambda n: SkillStore().revert(ctx.guild_id, n, version, ctx.user.id),
                      tool_embeds.SKILL_REVERTED, f"back to v{version}")

    @group.command(name="export", description="Download every skill, with its history")
    async def export(ctx: discord.Interaction):
        data = await SkillStore().export(ctx.guild_id)
        limit = getattr(ctx.guild, "filesize_limit", None)
        limit = limit if isinstance(limit, int) else DEFAULT_UPLOAD_BYTES
        if len(data) > limit:
            await ctx.response.send_message(
                f"The export is {len(data) // 1_000_000 + 1} MB, more than this server can upload "
                f"({limit // 1_000_000} MB). Delete old skills to make it smaller.", ephemeral=True)
            return
        await ctx.response.send_message("Every skill in this server, with its history.", ephemeral=True,
                                        file=discord.File(io.BytesIO(data), filename="skills.zip"))

    tree.add_command(group)

"""/skills: see and manage this server's saved skills.

Anyone in the server can (decision: skills belong to everyone, every change
is a new version and a delete is soft, so nothing is lost for good). Every
change posts its own card in the channel, like the brain's tools do, so the
channel always shows who changed what; the command's own reply is private.
"""
import logging

import discord
from discord import app_commands

from classes import card, card_kit as K, skill_format, tool_cards
from classes.skill_store import SkillError, SkillStore
from classes.skill_tools import announce, saved_card

logger = logging.getLogger(__name__)

MODAL_BODY_CHARS = 4000  # Discord's cap for a text input
SHOW_INLINE_CHARS = 1800
DEFAULT_UPLOAD_BYTES = 10 * 1024 * 1024  # Discord's limit without boosts


def _ago(ts) -> str:
    return f"<t:{int(ts)}:R>" if ts else "never"


def list_lines(skills) -> list[str]:
    """The /skills list card's entries, best first: the name and version, the
    description in full, then how it has been used."""
    import time
    lines = []
    for skill in skill_format.rank(skills, time.time()):
        used = skill.get("uses", 0)
        lines.append(f"**{skill['name']}** · v{skill['version']}\n{K.safe(skill['description'])}\n"
                     f"-# used {used}× · last {_ago(skill.get('last_used_at'))}")
    return lines


def list_card(skills) -> K.CardSpec:
    """Every skill, a gap between each: a long list goes over several cards."""
    return K.CardSpec(label=f"📘 Skills · {len(skills)}",
                      body=[K.Text("Best first. Read one with `/skills show`."), K.Divider(),
                            *K.listing(list_lines(skills))],
                      footer="/skills export downloads them all")


_SOURCES = {"brain": "saved by me", "reflection": "after reviewing a job", "curator": "daily tidy-up",
            "command": "edited with /skills", "revert": "reverted", "restore": "restored"}


def history_lines(rows) -> list[str]:
    lines = []
    for row in rows:
        who = f"<@{row['author_id']}>" if row.get("author_id") else ""
        mark = " · current" if row.get("current") else ""
        source = _SOURCES.get(row.get("source"), row.get("source") or "")
        meta = " · ".join(x for x in (_ago(row["created_at"]), who, source) if x)
        note = f"\n{K.safe(row['note'])}" if row.get("note") else ""
        lines.append(f"**v{row['v']}**{mark}{note}\n-# {meta}")
    return lines


def history_card(name: str, rows) -> K.CardSpec:
    return K.CardSpec(label="🕘 Skill history", title=name, body=K.listing(history_lines(rows)))


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
            result = await store.save(ctx.guild_id, name, str(self.description.value), str(self.steps.value),
                                      self.skill.get("files") or {}, author_id=ctx.user.id,
                                      source="command", note=str(self.note.value or ""),
                                      channel_id=ctx.channel_id)
        except SkillError as e:
            await card.reply(ctx, K.error(str(e), "❌ Not saved"))
            return
        if result == "unchanged":
            await card.reply(ctx, K.info("📘 Nothing changed", f"**{name}** is already saved like this."))
            return
        skill = await store.get(ctx.guild_id, name)
        label, detail, version = saved_card(result, name, skill["version"], skill["description"],
                                             str(self.note.value or ""))
        await announce(ctx.channel, label, name, detail, version=version)
        await card.reply(ctx, K.done("✅ Saved", f"**{name}** is now v{skill['version']}."))


def register_skill_commands(tree) -> None:
    group = app_commands.Group(name="skills", guild_only=True,
                               description="The skills I have learned in this server")

    @group.command(name="list", description="The skills saved in this server")
    async def list_skills(ctx: discord.Interaction):
        skills = await SkillStore().list(ctx.guild_id)
        if not skills:
            await card.reply(ctx, K.info("📘 Skills", "No skills saved yet. I save one when a job "
                                                     "is worth repeating."))
            return
        await card.reply(ctx, list_card(skills), ephemeral=False)

    @group.command(name="show", description="Read a skill")
    @app_commands.autocomplete(name=_live_names)
    async def show(ctx: discord.Interaction, name: str, version: int | None = None):
        skill = await SkillStore().get(ctx.guild_id, skill_format.clean_name(name), version)
        if skill is None:
            await card.reply(ctx, K.error("No such skill (or version).", "❌ Not found"))
            return
        text = show_text(skill)
        label = f"📘 Skill · v{skill['version']}"
        if len(text) <= SHOW_INLINE_CHARS:
            await card.reply(ctx, K.CardSpec(label=label, title=skill["name"],
                                             body=[K.Text(f"```md\n{text}\n```")]))
        else:
            filename = f"{skill['name']}.md"
            await card.reply(ctx, K.CardSpec(label=label, title=skill["name"],
                                             body=[K.FileRef(filename)],
                                             footer="Too long to show here, so it's attached"),
                             files={filename: text.encode("utf-8")})

    @group.command(name="history", description="Who changed a skill, and when")
    @app_commands.autocomplete(name=_live_names)
    async def history(ctx: discord.Interaction, name: str):
        rows = await SkillStore().history(ctx.guild_id, skill_format.clean_name(name))
        if not rows:
            await card.reply(ctx, K.error("No such skill.", "❌ Not found"))
            return
        await card.reply(ctx, history_card(skill_format.clean_name(name), rows))

    @group.command(name="edit", description="Change a skill's description and steps")
    @app_commands.autocomplete(name=_live_names)
    async def edit(ctx: discord.Interaction, name: str):
        skill = await SkillStore().get(ctx.guild_id, skill_format.clean_name(name))
        if skill is None:
            await card.reply(ctx, K.error("No such skill.", "❌ Not found"))
            return
        if len(skill["body"]) > MODAL_BODY_CHARS:
            await card.reply(ctx, K.heads_up(
                "✏️ Too long to edit here",
                f"This skill is longer than the edit box allows ({MODAL_BODY_CHARS} characters). "
                "Ask me to update it instead."))
            return
        await ctx.response.send_modal(EditModal(skill))

    async def _change(ctx, name, action, title, detail=""):
        name = skill_format.clean_name(name)
        try:
            await action(name)
        except SkillError as e:
            await card.reply(ctx, K.error(str(e)))
            return
        await announce(ctx.channel, title, name, detail)
        await card.reply(ctx, K.done("✅ Done", "The change is posted in the channel."))

    @group.command(name="delete", description="Delete a skill (it can be restored)")
    @app_commands.autocomplete(name=_live_names)
    async def delete(ctx: discord.Interaction, name: str):
        await _change(ctx, name, lambda n: SkillStore().delete(ctx.guild_id, n, ctx.user.id),
                      tool_cards.SKILL_DELETED, "Deleted with /skills delete.")

    @group.command(name="restore", description="Bring back a deleted skill")
    @app_commands.autocomplete(name=_deleted_names)
    async def restore(ctx: discord.Interaction, name: str):
        await _change(ctx, name, lambda n: SkillStore().restore(ctx.guild_id, n, ctx.user.id),
                      tool_cards.SKILL_RESTORED)

    @group.command(name="revert", description="Go back to an earlier version of a skill")
    @app_commands.autocomplete(name=_live_names)
    async def revert(ctx: discord.Interaction, name: str, version: int):
        await _change(ctx, name, lambda n: SkillStore().revert(ctx.guild_id, n, version, ctx.user.id),
                      tool_cards.SKILL_REVERTED, f"Back to v{version}.")

    @group.command(name="export", description="Download every skill, with its history")
    async def export(ctx: discord.Interaction):
        data = await SkillStore().export(ctx.guild_id)
        limit = getattr(ctx.guild, "filesize_limit", None)
        limit = limit if isinstance(limit, int) else DEFAULT_UPLOAD_BYTES
        if len(data) > limit:
            await card.reply(ctx, K.error(
                f"The export is {len(data) // 1_000_000 + 1} MB, more than this server can upload "
                f"({limit // 1_000_000} MB). Delete old skills to make it smaller.", "❌ Too big to export"))
            return
        await card.reply(ctx, K.CardSpec(label="📦 Skills export", body=[
            K.Text("Every skill in this server, with its history."), K.FileRef("skills.zip")]),
            files={"skills.zip": data})

    tree.add_command(group)

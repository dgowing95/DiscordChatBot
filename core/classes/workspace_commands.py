"""/workspace: the server workspace's status, reset, and live progress.

status and reset need Manage Server (reset wipes the whole server's files),
checked when the command runs: Discord's default_permissions can only gate a
whole group, and progress keeps the permission /sandbox_progress_updates had
(none). Every reply is ephemeral.

progress replaces /sandbox_progress_updates. Its setting is now
`workspace_progress`; a server's old `sandbox_progress_updates` value is
copied over the first time it is read, then deleted (progress_enabled).
"""
import logging
import time

import discord
from discord import app_commands

from classes import card, card_kit as K
from classes.config_manager import configManager

logger = logging.getLogger(__name__)

SETTING = "workspace_progress"
OLD_SETTING = "sandbox_progress_updates"


def _on(raw) -> bool:
    """A stored setting as on/off: unset (None/False) is OFF."""
    if raw in (None, False):
        return False
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


async def progress_enabled(guild_id) -> bool:
    """Whether runs in this server stream their commands and output live.
    A read failure is off: progress is a nice-to-have, never a dependency."""
    from classes import config_manager  # looked up per call (tests patch it)
    config = config_manager.configManager()
    try:
        # get_setting answers False for a key that is not set.
        raw = await config.get_setting(SETTING, guild_id)
        if raw in (None, False):
            old = await config.get_setting(OLD_SETTING, guild_id)
            if old not in (None, False):
                await config.update_setting(SETTING, str(_on(old)), guild_id)
                await config.delete_setting(OLD_SETTING, guild_id)
                raw = str(_on(old))
    except Exception as e:
        logger.warning(f"Could not read the {SETTING} setting: {e}")
        return False
    return _on(raw)


def _size(n) -> str:
    if not n:
        return "none"
    for unit in ("bytes", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "bytes" else f"{n:.1f} {unit}"
        n /= 1024


def status_card(status: dict, now: float | None = None) -> K.CardSpec:
    """What /workspace status says (pure): the state as the title, then the
    details."""
    now = time.time() if now is None else now
    lines = []
    if status.get("warm"):
        title = "🟢 Running"
        up = int((now - (status.get("started_at") or now)) // 60)
        busy = status.get("busy") or 0
        lines.append(f"**Up for** {up} min" + (f" · **in use by** {busy} task{'s' if busy != 1 else ''}"
                                                if busy else ""))
        if status.get("jobs") is not None:
            lines.append(f"**Task folders** {status['jobs']}")
        lead = ""
    elif status.get("waiting"):
        title = "🟡 Waiting"
        lead = "Still running from before the bot restarted; the next task picks it up."
    else:
        title = "⚪ Stopped"
        lead = "It starts on the next task that needs it."
    saved = status.get("last_snapshot_at")
    lines.append(f"**Last saved** <t:{int(saved)}:R> · {_size(status.get('snapshot_bytes'))}"
                 if saved else "**Last saved** not yet")
    body = ([K.Text(lead)] if lead else []) + [K.Text("\n".join(lines))]
    if status.get("snapshot_too_large"):
        body.append(K.Text("⚠️ It is too big to save. Delete files in shared/ or reset it."))
    return K.CardSpec(label=STATUS_LABEL, title=title, body=body)


STATUS_LABEL = "💻 This server's workspace"
RESET_LABEL = "🧹 Reset the workspace?"
RESET_TEXT = ("This deletes **every file** in this server's workspace, including what was kept "
              "in shared/. It cannot be undone.")
NEEDS_MANAGE = "You need the Manage Server permission."


def _can_manage(ctx: discord.Interaction) -> bool:
    perms = getattr(ctx, "permissions", None)
    return bool(perms is not None and perms.manage_guild)


def _confirm_card(user_id: int) -> K.CardSpec:
    """Reset (red) and Cancel. Only the person who asked can press them, and
    only with Manage Server (checked again on the press). The first press
    that counts ends it: a second one, landing before the card changed,
    would reset twice."""
    answered = []

    async def _already(ctx) -> bool:
        if answered:
            await card.reply(ctx, K.error("This was answered already.", "🧹 Already answered"))
            return True
        answered.append(True)
        return False

    async def confirm(ctx: discord.Interaction):
        if ctx.user.id != user_id:
            await card.reply(ctx, K.error("Only the person who asked can confirm.", "🔒 Not yours"))
            return
        if not _can_manage(ctx):
            await card.reply(ctx, K.error(NEEDS_MANAGE, "🔒 Not allowed"))
            return
        if await _already(ctx):
            return
        from classes.workspace_manager import workspaces
        await ctx.response.edit_message(view=card.views(K.CardSpec(
            label="🧹 Resetting…", body=[K.Text("Deleting the workspace's files.")],
            colour=K.WORKING))[0])
        try:
            when = await workspaces().reset(ctx.guild_id)
        except Exception as e:
            logger.warning(f"Workspace reset failed: {e}")
            await ctx.edit_original_response(view=card.views(K.error(
                "The reset failed. Try again in a minute.", "❌ Not reset"))[0])
            return
        await ctx.edit_original_response(view=card.views(
            K.done("🧹 Workspace reset", "Its files are gone and the next task starts empty.")
            if when == "now" else
            K.heads_up("🧹 Reset waiting", "The workspace is busy right now, so it resets as soon "
                                           "as it is free. New tasks wait for it."))[0])

    async def cancel(ctx: discord.Interaction):
        if await _already(ctx):
            return
        await ctx.response.edit_message(view=card.views(
            K.CardSpec(label="🧹 Reset cancelled", body=[K.Text("Nothing was reset.")],
                       colour=K.CLOSED))[0])

    return K.CardSpec(label=RESET_LABEL, body=[K.Text(RESET_TEXT), K.Buttons((
        card.button("Reset", discord.ButtonStyle.danger, confirm, "🧹"),
        card.button("Cancel", discord.ButtonStyle.secondary, cancel)))], colour=K.FAIL)


async def _status_panel(guild_id, user_id: int) -> K.CardSpec:
    """The status with its buttons: live progress on or off, and Reset…"""
    from classes.workspace_manager import workspaces
    try:
        spec = status_card(await workspaces().status(guild_id))
    except Exception as e:
        logger.warning(f"Workspace status failed: {e}")
        spec = K.error("Could not read the workspace's status.", STATUS_LABEL)
    on = await progress_enabled(guild_id)

    async def toggle(ctx: discord.Interaction):
        # Answered first: the status read below (a docker exec) can take
        # longer than Discord's 3 seconds, as /workspace status defers too.
        await ctx.response.defer()
        await configManager().update_setting(SETTING, str(not on), ctx.guild_id)
        await ctx.edit_original_response(view=card.views(
            await _status_panel(ctx.guild_id, user_id), timeout=PANEL_SECONDS)[0])

    async def reset(ctx: discord.Interaction):
        if not _can_manage(ctx):
            await card.reply(ctx, K.error(NEEDS_MANAGE, "🔒 Not allowed"))
            return
        await ctx.response.edit_message(view=card.views(_confirm_card(user_id),
                                                        timeout=PANEL_SECONDS)[0])

    spec.body += [K.Divider(), K.Buttons((
        card.button(f"Live progress: {'on' if on else 'off'}", discord.ButtonStyle.secondary,
                    toggle, "📺"),
        card.button("Reset…", discord.ButtonStyle.danger, reset, "🧹")))]
    return spec


PANEL_SECONDS = 300


def register_workspace_commands(tree) -> None:
    group = app_commands.Group(name="workspace", guild_only=True,
                               description="The server's workspace, where code tasks run")

    @group.command(name="status", description="Is the workspace running, and when was it last saved (Manage Server)")
    async def status(ctx: discord.Interaction):
        if not _can_manage(ctx):
            await card.reply(ctx, K.error(NEEDS_MANAGE, "🔒 Not allowed"))
            return
        await ctx.response.defer(ephemeral=True)
        await card.reply(ctx, await _status_panel(ctx.guild_id, ctx.user.id), timeout=PANEL_SECONDS)

    @group.command(name="reset", description="Delete every file in this server's workspace (Manage Server)")
    async def reset(ctx: discord.Interaction):
        if not _can_manage(ctx):
            await card.reply(ctx, K.error(NEEDS_MANAGE, "🔒 Not allowed"))
            return
        await card.reply(ctx, _confirm_card(ctx.user.id), timeout=PANEL_SECONDS)

    @group.command(name="progress", description="Show each task's commands and output live while it runs")
    @app_commands.describe(enabled="On: a task's message shows its commands as they run")
    async def progress(ctx: discord.Interaction, enabled: bool):
        await configManager().update_setting(SETTING, str(enabled), ctx.guild_id)
        await card.reply(ctx, K.done("📺 Live progress " + ("on" if enabled else "off"),
                                     "A task's status card now shows its commands as they run."
                                     if enabled else "A task's status card shows only the request "
                                                     "and how it went."))

    tree.add_command(group)

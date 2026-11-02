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


def status_text(status: dict, now: float | None = None) -> str:
    """What /workspace status says (pure)."""
    now = time.time() if now is None else now
    lines = []
    if status.get("warm"):
        up = int((now - (status.get("started_at") or now)) // 60)
        busy = status.get("busy") or 0
        lines.append(f"🟢 **Running** for {up} min" + (f", {busy} task(s) using it now" if busy else ""))
        if status.get("jobs") is not None:
            lines.append(f"Task folders: {status['jobs']}")
    else:
        lines.append("⚪ **Stopped**: it starts on the next task that needs it.")
    saved = status.get("last_snapshot_at")
    if saved:
        lines.append(f"Last saved <t:{int(saved)}:R>, {_size(status.get('snapshot_bytes'))}")
    else:
        lines.append("Not saved yet.")
    if status.get("snapshot_too_large"):
        lines.append("⚠️ It is too big to save. Delete files in shared/ or use `/workspace reset`.")
    return "\n".join(lines)


def _can_manage(ctx: discord.Interaction) -> bool:
    perms = getattr(ctx, "permissions", None)
    return bool(perms is not None and perms.manage_guild)


class _ResetConfirm(discord.ui.View):
    def __init__(self, user_id: int):
        super().__init__(timeout=60)
        self.user_id = user_id

    @discord.ui.button(label="Reset the workspace", style=discord.ButtonStyle.danger)
    async def confirm(self, ctx: discord.Interaction, button: discord.ui.Button):
        if ctx.user.id != self.user_id:
            await ctx.response.send_message("Only the person who asked can confirm.", ephemeral=True)
            return
        from classes.workspace_manager import workspaces
        self.stop()
        await ctx.response.edit_message(content="Resetting…", view=None)
        try:
            when = await workspaces().reset(ctx.guild_id)
        except Exception as e:
            logger.warning(f"Workspace reset failed: {e}")
            await ctx.edit_original_response(content="❌ The reset failed. Try again in a minute.")
            return
        await ctx.edit_original_response(content=(
            "🧹 The workspace was reset: its files are gone and the next task starts empty."
            if when == "now" else
            "🧹 Tasks are using the workspace now, so it resets as soon as they finish. "
            "New tasks wait for it."))

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.secondary)
    async def cancel(self, ctx: discord.Interaction, button: discord.ui.Button):
        self.stop()
        await ctx.response.edit_message(content="Nothing was reset.", view=None)


def register_workspace_commands(tree) -> None:
    group = app_commands.Group(name="workspace", guild_only=True,
                               description="The server's workspace, where code tasks run")

    @group.command(name="status", description="Is the workspace running, and when was it last saved (Manage Server)")
    async def status(ctx: discord.Interaction):
        if not _can_manage(ctx):
            await ctx.response.send_message("You need the Manage Server permission.", ephemeral=True)
            return
        from classes.workspace_manager import workspaces
        await ctx.response.defer(ephemeral=True)
        try:
            text = status_text(await workspaces().status(ctx.guild_id))
        except Exception as e:
            logger.warning(f"Workspace status failed: {e}")
            text = "❌ Could not read the workspace's status."
        await ctx.followup.send(text, ephemeral=True)

    @group.command(name="reset", description="Delete every file in this server's workspace (Manage Server)")
    async def reset(ctx: discord.Interaction):
        if not _can_manage(ctx):
            await ctx.response.send_message("You need the Manage Server permission.", ephemeral=True)
            return
        await ctx.response.send_message(
            "This deletes **every file** in this server's workspace, including what was kept "
            "in shared/. It cannot be undone.", view=_ResetConfirm(ctx.user.id), ephemeral=True)

    @group.command(name="progress", description="Show each task's commands and output live while it runs")
    @app_commands.describe(enabled="On: a task's message shows its commands as they run")
    async def progress(ctx: discord.Interaction, enabled: bool):
        await configManager().update_setting(SETTING, str(enabled), ctx.guild_id)
        await ctx.response.send_message(
            f"Live progress for tasks is now {'on' if enabled else 'off'}.", ephemeral=True)

    tree.add_command(group)

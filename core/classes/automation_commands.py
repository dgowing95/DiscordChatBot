"""Discord slash management for schedules and rules."""
from datetime import datetime
from typing import Literal
import discord
from discord import app_commands

from classes.automation_policy import settings, timing_text
from classes.automation_store import AutomationError, AutomationStore


def _access(ctx, channel):
    if ctx.guild is None:
        raise AutomationError("Automations work only in servers")
    if channel is None or channel.guild.id != ctx.guild.id:
        raise AutomationError("Choose a channel in this server")
    perms = channel.permissions_for(ctx.user)
    bot_member = ctx.guild.me
    bot_perms = channel.permissions_for(bot_member) if bot_member is not None else ctx.app_permissions
    if not (perms.view_channel and perms.send_messages and bot_perms.view_channel and bot_perms.send_messages):
        raise AutomationError("You and the bot must be able to view and send in that channel")


async def _show(ctx, kind, ident):
    if ctx.guild is None:
        raise AutomationError("Use this in a server")
    row = await AutomationStore().get(ctx.guild.id, ident)
    if not row or row["kind"] != kind:
        raise AutomationError("No matching entry in this server")
    channel = ctx.guild.get_channel_or_thread(row["channel_id"])
    _access(ctx, channel)
    return row


def _display(row):
    lines = [f"**{row['kind'].capitalize()} `{row['id']}`** in <#{row['channel_id']}> — {row['status']}"]
    if row["kind"] == "schedule":
        lines.append(f"Timing: {timing_text(row['timing'], row['timezone'])}")
        if row["status"] == "enabled" and row.get("next_run"):
            lines.append(f"Next run: <t:{int(row['next_run'])}:f> (<t:{int(row['next_run'])}:R>)")
    else:
        lines.append(f"Matches: `{row['pattern']}` ({row['match_mode']})")
    last = row.get("last_result")
    if last:
        detail = f" — {last['detail']}" if last.get("detail") else ""
        when = int(datetime.fromisoformat(last["at"]).timestamp())
        lines.append(f"Last result: {last['status']}{detail} (<t:{when}:R>)")
    else:
        lines.append("Last result: never run")
    lines.append(f"Created by <@{row['creator_id']}>, last edited by <@{row['editor_id']}>")
    lines.append(f"Action: {row['action']}")
    return "\n".join(lines)[:1900]


async def _respond(ctx, operation):
    try:
        result = await operation()
        await ctx.response.send_message(result, ephemeral=True)
    except (AutomationError, ValueError, KeyError) as exc:
        await ctx.response.send_message(f"❌ {exc}", ephemeral=True)


class ActionModal(discord.ui.Modal):
    action = discord.ui.TextInput(label="Action to perform", style=discord.TextStyle.paragraph, max_length=1500)

    def __init__(self, kind, channel, fields):
        super().__init__(title=f"Create {kind}")
        self.kind, self.channel, self.fields = kind, channel, fields

    async def on_submit(self, ctx):
        async def op():
            _access(ctx, self.channel)
            row = await AutomationStore().create(ctx.guild.id, self.channel.id, ctx.user.id,
                                                  self.kind, str(self.action), **self.fields)
            return _display(row)
        await _respond(ctx, op)


class EditModal(discord.ui.Modal):
    action = discord.ui.TextInput(label="New action", style=discord.TextStyle.paragraph, max_length=1500)

    def __init__(self, row, changes=None):
        super().__init__(title=f"Edit {row['kind']} {row['id']}")
        self.row = row
        self.changes = changes or {}
        self.action.default = row["action"]

    async def on_submit(self, ctx):
        async def op():
            await _show(ctx, self.row["kind"], self.row["id"])
            changes = {"action": str(self.action), **self.changes}
            row = await AutomationStore().update(ctx.guild.id, self.row["id"], ctx.user.id,
                                                 self.row["revision"], **changes)
            return _display(row)
        await _respond(ctx, op)


def register_automation_commands(tree):
    schedule = app_commands.Group(name="schedule", description="Manage scheduled actions")
    rule = app_commands.Group(name="rule", description="Manage message rules")

    @schedule.command(name="create", description="Create a timed action in this channel")
    async def schedule_create(ctx: discord.Interaction, timing_type: Literal["once", "interval", "daily", "weekly"], at: str = "", every: int = 0, start: str = "",
                              unit: Literal["hours", "days", "weeks"] = "hours", local_time: str = "", weekday: str = "",
                              timezone: str = ""):
        try:
            _access(ctx, ctx.channel)
        except AutomationError as exc:
            await ctx.response.send_message(f"❌ {exc}", ephemeral=True)
            return
        timing = {"type": timing_type}
        if timing_type == "once": timing["at"] = at
        elif timing_type == "interval": timing.update(every=every, unit=unit, **({"start": start} if start else {}))
        else: timing.update(time=local_time, **({"weekday": weekday.lower()} if timing_type == "weekly" else {}))
        await ctx.response.send_modal(ActionModal("schedule", ctx.channel,
                                                  {"timing": timing, "timezone": timezone or settings()["timezone"]}))

    @rule.command(name="create", description="Create a message rule in this channel")
    async def rule_create(ctx: discord.Interaction, pattern: str, match_mode: Literal["word", "substring"] = "word"):
        try:
            _access(ctx, ctx.channel)
        except AutomationError as exc:
            await ctx.response.send_message(f"❌ {exc}", ephemeral=True)
            return
        await ctx.response.send_modal(ActionModal("rule", ctx.channel,
                                                  {"pattern": pattern, "match_mode": match_mode}))

    for group, kind in ((schedule, "schedule"), (rule, "rule")):
        def callbacks(kind):
          async def list_entries(ctx: discord.Interaction):
            async def op():
                if ctx.guild is None: raise AutomationError("Use this in a server")
                rows = await AutomationStore().list(ctx.guild.id, kind)
                visible = []
                for row in rows:
                    channel = ctx.guild.get_channel_or_thread(row["channel_id"])
                    try: _access(ctx, channel)
                    except AutomationError: continue
                    outcome = (row.get("last_result") or {}).get("status", "never run")
                    next_at = (f" next <t:{int(row['next_run'])}:f>" if row["kind"] == "schedule"
                               and row["status"] == "enabled" else "")
                    visible.append(f"`{row['id']}` <#{row['channel_id']}> {row['status']} ({outcome}){next_at} — {row['action'][:60]}")
                return "\n".join(visible)[:1900] or "No entries you can access."
            await _respond(ctx, op)

          async def view_entry(ctx: discord.Interaction, entry_id: str):
            await _respond(ctx, lambda: _view(ctx, kind, entry_id))

          async def delete_entry(ctx: discord.Interaction, entry_id: str):
            await _respond(ctx, lambda: _set_status(ctx, kind, entry_id, "deleted"))

          async def pause_entry(ctx: discord.Interaction, entry_id: str):
            await _respond(ctx, lambda: _set_status(ctx, kind, entry_id, "paused"))

          async def resume_entry(ctx: discord.Interaction, entry_id: str):
            await _respond(ctx, lambda: _set_status(ctx, kind, entry_id, "enabled"))
          return list_entries, view_entry, delete_entry, pause_entry, resume_entry

        list_entries, view_entry, delete_entry, pause_entry, resume_entry = callbacks(kind)
        for name, callback, description in (
            ("list", list_entries, "List accessible entries"), ("view", view_entry, "Inspect an entry"),
            ("delete", delete_entry, "Delete an entry"),
            ("pause", pause_entry, "Pause an entry"), ("resume", resume_entry, "Resume an entry")):
            group.command(name=name, description=description)(callback)

    @schedule.command(name="edit", description="Edit schedule timing and action")
    async def schedule_edit(ctx: discord.Interaction, entry_id: str,
                            timing_type: Literal["keep", "once", "interval", "daily", "weekly"] = "keep",
                            at: str = "", every: int = 0, unit: Literal["hours", "days", "weeks"] = "hours",
                            start: str = "", local_time: str = "", weekday: str = "", timezone: str = ""):
        try:
            row = await _show(ctx, "schedule", entry_id)
            changes = {}
            if timing_type != "keep":
                timing = {"type": timing_type}
                if timing_type == "once": timing["at"] = at
                elif timing_type == "interval": timing.update(every=every, unit=unit, **({"start": start} if start else {}))
                else: timing.update(time=local_time, **({"weekday": weekday.lower()} if timing_type == "weekly" else {}))
                changes["timing"] = timing
            if timezone: changes["timezone"] = timezone
            await ctx.response.send_modal(EditModal(row, changes))
        except AutomationError as exc:
            await ctx.response.send_message(f"❌ {exc}", ephemeral=True)

    @rule.command(name="edit", description="Edit rule matching and action")
    async def rule_edit(ctx: discord.Interaction, entry_id: str, pattern: str = "",
                        match_mode: Literal["keep", "word", "substring"] = "keep"):
        try:
            row = await _show(ctx, "rule", entry_id)
            changes = {}
            if pattern: changes["pattern"] = pattern
            if match_mode != "keep": changes["match_mode"] = match_mode
            await ctx.response.send_modal(EditModal(row, changes))
        except AutomationError as exc:
            await ctx.response.send_message(f"❌ {exc}", ephemeral=True)
    tree.add_command(schedule)
    tree.add_command(rule)


async def _view(ctx, kind, ident):
    return _display(await _show(ctx, kind, ident))


async def _set_status(ctx, kind, ident, status):
    row = await _show(ctx, kind, ident)
    updated = await AutomationStore().update(ctx.guild.id, ident, ctx.user.id, row["revision"], status=status)
    return _display(updated) + "\nA run that already started may still finish."

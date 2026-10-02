"""Discord slash management for schedules and rules.

Creating and editing happen in forms (modals), not slash options: Discord
cannot hide one slash option based on another's value, so a single command
offered "every" on a one-off schedule. Instead `/schedule create` and
`/schedule edit` first ask for the schedule type with buttons, then open a
form holding only that type's fields, each with a description.
"""
from datetime import datetime
import discord
from discord import app_commands

from classes.automation_policy import (
    WEEKDAYS, form_values, settings, timing_from_form, timing_text,
)
from classes.automation_store import AutomationError, AutomationStore, QuotaError

SCHEDULE_TYPES = (("once", "Once"), ("interval", "Repeat every…"), ("daily", "Daily"), ("weekly", "Weekly"))
TYPE_TITLES = {"once": "one-off", "interval": "repeating", "daily": "daily", "weekly": "weekly"}
UNITS = (("Hours", "hours"), ("Days", "days"), ("Weeks", "weeks"))
MATCH_MODES = (("Whole word or phrase", "word", "\"cat\" matches \"my cat\" but not \"concatenate\""),
               ("Anywhere in the text", "substring", "\"cat\" also matches inside \"concatenate\""))
ERRORS = (AutomationError, ValueError, KeyError)


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


def _summary(row):
    detail = (timing_text(row["timing"], row["timezone"]) if row["kind"] == "schedule"
              else f"\"{row['pattern']}\"")
    return f"{row['id']} — {row['status']}, {detail}: {row['action']}"[:100]


async def _respond(ctx, operation):
    try:
        result = await operation()
        await ctx.response.send_message(result, ephemeral=True)
    except ERRORS as exc:
        await ctx.response.send_message(f"❌ {exc}", ephemeral=True)


def _text(form, name, label, description, values, placeholder=None, required=True, paragraph=False, max_length=None):
    field = discord.ui.TextInput(
        style=discord.TextStyle.paragraph if paragraph else discord.TextStyle.short,
        default=values.get(name) or None, placeholder=placeholder, required=required, max_length=max_length)
    form.add_item(discord.ui.Label(text=label, description=description, component=field))
    form.fields[name] = field


def _select(form, name, label, description, options, values):
    current = values.get(name)
    field = discord.ui.Select(options=[
        discord.SelectOption(label=text, value=value, description=hint, default=value == current)
        for text, value, *rest in options for hint in [rest[0] if rest else None]])
    form.add_item(discord.ui.Label(text=label, description=description, component=field))
    form.fields[name] = field


def _read(fields):
    return {name: (field.values[0] if field.values else "") if isinstance(field, discord.ui.Select)
            else str(field.value) for name, field in fields.items()}


class RetryView(discord.ui.View):
    """Shown with a form error, so a rejected form can be reopened with what
    was typed instead of starting again."""

    def __init__(self, reopen):
        super().__init__(timeout=600)
        self.reopen = reopen

    @discord.ui.button(label="Fix and try again", style=discord.ButtonStyle.primary)
    async def retry(self, ctx, button):
        await ctx.response.send_modal(self.reopen())


async def _submit(ctx, operation, reopen):
    try:
        result = await operation()
        await ctx.response.send_message(result, ephemeral=True)
    except QuotaError as exc:
        await ctx.response.send_message(f"❌ {exc}", ephemeral=True)
    except ERRORS as exc:
        await ctx.response.send_message(f"❌ {exc}", ephemeral=True, view=RetryView(reopen))


class ScheduleForm(discord.ui.Modal):
    """One schedule type's fields. With `row` it edits that schedule."""

    def __init__(self, kind, channel, row=None, values=None):
        verb = f"Edit schedule {row['id']}" if row else "New schedule"
        super().__init__(title=f"{verb}: {TYPE_TITLES[kind]}"[:45])
        self.kind, self.channel, self.row = kind, channel, row
        self.fields = {}
        values = values if values is not None else (form_values(row, kind) if row else {})
        min_hours = settings()["min_hours"]
        if kind == "once":
            _text(self, "when", "Date and time", "When it runs, as YYYY-MM-DD HH:MM (24-hour), in the timezone below",
                  values, placeholder="2026-10-03 09:00")
        elif kind == "interval":
            _text(self, "every", "Every", f"How many hours, days or weeks between runs (at least {min_hours} hours)",
                  values, placeholder="6", max_length=4)
            _select(self, "unit", "Unit", "Pick what 'Every' counts", UNITS, {"unit": "hours", **values})
            _text(self, "start", "First run (optional)",
                  "YYYY-MM-DD HH:MM. Leave empty to run first one interval from now", values,
                  placeholder="2026-10-03 09:00", required=False)
        else:
            if kind == "weekly":
                _select(self, "weekday", "Day", "The day of the week it runs",
                        [(day.capitalize(), day) for day in WEEKDAYS], values)
            _text(self, "time", "Time", "24-hour local time, like 09:00 or 17:30 (9am works too)",
                  values, placeholder="09:00", max_length=8)
        _text(self, "timezone", "Timezone", "IANA name, like Europe/London or America/New_York",
              {"timezone": values.get("timezone") or settings()["timezone"]}, max_length=64)
        _text(self, "action", "Action", "What the bot should do each time, as you would ask it in chat",
              values, placeholder="Provide today's Wordle answer", paragraph=True, max_length=1500)

    async def on_submit(self, ctx):
        values = _read(self.fields)

        async def operation():
            timing = timing_from_form(self.kind, values)
            store = AutomationStore()
            if self.row:
                await _show(ctx, "schedule", self.row["id"])
                row = await store.update(ctx.guild.id, self.row["id"], ctx.user.id, self.row["revision"],
                                         timing=timing, timezone=values["timezone"].strip(), action=values["action"])
            else:
                _access(ctx, self.channel)
                row = await store.create(ctx.guild.id, self.channel.id, ctx.user.id, "schedule", values["action"],
                                         timing=timing, timezone=values["timezone"].strip())
            return _display(row)
        await _submit(ctx, operation, lambda: ScheduleForm(self.kind, self.channel, self.row, values))


class ScheduleTypePicker(discord.ui.View):
    """The buttons /schedule create and /schedule edit show first. The
    current type of an edited schedule is highlighted."""

    def __init__(self, channel, row=None):
        super().__init__(timeout=600)
        current = row["timing"]["type"] if row else None
        for kind, label in SCHEDULE_TYPES:
            button = discord.ui.Button(
                label=f"{label} (current)" if kind == current else label,
                style=discord.ButtonStyle.primary if kind == current or not row else discord.ButtonStyle.secondary)
            button.callback = self._opener(kind, channel, row)
            self.add_item(button)

    @staticmethod
    def _opener(kind, channel, row):
        async def open_form(ctx):
            await ctx.response.send_modal(ScheduleForm(kind, channel, row))
        return open_form


class RuleForm(discord.ui.Modal):
    def __init__(self, channel, row=None, values=None):
        super().__init__(title=f"Edit rule {row['id']}" if row else "New message rule")
        self.channel, self.row = channel, row
        self.fields = {}
        values = values if values is not None else (
            {"pattern": row["pattern"], "match_mode": row["match_mode"], "action": row["action"]} if row
            else {"match_mode": "word"})
        _text(self, "pattern", "Word or phrase", "Text to look for in new messages in this channel (any case)",
              values, placeholder="wordle", max_length=200)
        _select(self, "match_mode", "How to match", "Whole words only, or anywhere in the text",
                MATCH_MODES, values)
        _text(self, "action", "Action", "What the bot should do when it matches, as you would ask it in chat",
              values, placeholder="Reply with an encouraging sentence about Wordle", paragraph=True, max_length=1500)

    async def on_submit(self, ctx):
        values = _read(self.fields)

        async def operation():
            store = AutomationStore()
            if self.row:
                await _show(ctx, "rule", self.row["id"])
                row = await store.update(ctx.guild.id, self.row["id"], ctx.user.id, self.row["revision"],
                                         pattern=values["pattern"].strip(), match_mode=values["match_mode"],
                                         action=values["action"])
            else:
                _access(ctx, self.channel)
                row = await store.create(ctx.guild.id, self.channel.id, ctx.user.id, "rule", values["action"],
                                         pattern=values["pattern"], match_mode=values["match_mode"])
            return _display(row)
        await _submit(ctx, operation, lambda: RuleForm(self.channel, self.row, values))


def _entry_autocomplete(kind):
    async def complete(ctx: discord.Interaction, current: str):
        if ctx.guild is None:
            return []
        choices = []
        try:
            rows = await AutomationStore().list(ctx.guild.id, kind)
        except Exception:
            return []
        for row in rows:
            try:
                _access(ctx, ctx.guild.get_channel_or_thread(row["channel_id"]))
            except AutomationError:
                continue
            label = _summary(row)
            if current.lower() in label.lower():
                choices.append(app_commands.Choice(name=label, value=row["id"]))
        return choices[:25]
    return complete


def register_automation_commands(tree):
    schedule = app_commands.Group(name="schedule", description="Manage scheduled actions")
    rule = app_commands.Group(name="rule", description="Manage message rules")

    @schedule.command(name="create", description="Create a timed action in this channel (opens a form)")
    async def schedule_create(ctx: discord.Interaction):
        try:
            _access(ctx, ctx.channel)
            await AutomationStore().check_quota(ctx.guild.id, "schedule")
        except AutomationError as exc:
            await ctx.response.send_message(f"❌ {exc}", ephemeral=True)
            return
        await ctx.response.send_message("What kind of schedule?", view=ScheduleTypePicker(ctx.channel),
                                        ephemeral=True)

    @rule.command(name="create", description="Create a message rule in this channel (opens a form)")
    async def rule_create(ctx: discord.Interaction):
        try:
            _access(ctx, ctx.channel)
            await AutomationStore().check_quota(ctx.guild.id, "rule")
        except AutomationError as exc:
            await ctx.response.send_message(f"❌ {exc}", ephemeral=True)
            return
        await ctx.response.send_modal(RuleForm(ctx.channel))

    for group, kind in ((schedule, "schedule"), (rule, "rule")):
        def callbacks(kind):
            async def list_entries(ctx: discord.Interaction):
                async def op():
                    if ctx.guild is None:
                        raise AutomationError("Use this in a server")
                    rows = await AutomationStore().list(ctx.guild.id, kind)
                    visible = []
                    for row in rows:
                        channel = ctx.guild.get_channel_or_thread(row["channel_id"])
                        try:
                            _access(ctx, channel)
                        except AutomationError:
                            continue
                        outcome = (row.get("last_result") or {}).get("status", "never run")
                        next_at = (f" next <t:{int(row['next_run'])}:f>" if row["kind"] == "schedule"
                                   and row["status"] == "enabled" else "")
                        visible.append(f"`{row['id']}` <#{row['channel_id']}> {row['status']} ({outcome}){next_at}"
                                       f" — {row['action'][:60]}")
                    return "\n".join(visible)[:1900] or "No entries you can access."
                await _respond(ctx, op)

            async def view_entry(ctx: discord.Interaction, entry: str):
                await _respond(ctx, lambda: _view(ctx, kind, entry))

            async def edit_entry(ctx: discord.Interaction, entry: str):
                try:
                    row = await _show(ctx, kind, entry)
                except ERRORS as exc:
                    await ctx.response.send_message(f"❌ {exc}", ephemeral=True)
                    return
                channel = ctx.guild.get_channel_or_thread(row["channel_id"])
                if kind == "rule":
                    await ctx.response.send_modal(RuleForm(channel, row))
                    return
                await ctx.response.send_message(
                    f"Editing schedule `{row['id']}` ({timing_text(row['timing'], row['timezone'])}).\n"
                    "Pick the current type to change its time or action, or another type to switch.",
                    view=ScheduleTypePicker(channel, row), ephemeral=True)

            async def delete_entry(ctx: discord.Interaction, entry: str):
                await _respond(ctx, lambda: _set_status(ctx, kind, entry, "deleted"))

            async def pause_entry(ctx: discord.Interaction, entry: str):
                await _respond(ctx, lambda: _set_status(ctx, kind, entry, "paused"))

            async def resume_entry(ctx: discord.Interaction, entry: str):
                await _respond(ctx, lambda: _set_status(ctx, kind, entry, "enabled"))
            return list_entries, view_entry, edit_entry, delete_entry, pause_entry, resume_entry

        list_entries, *by_entry = callbacks(kind)
        group.command(name="list", description=f"List the {kind}s you can access")(list_entries)
        for name, callback, description in zip(
                ("view", "edit", "delete", "pause", "resume"), by_entry,
                (f"Show a {kind}'s details", f"Edit a {kind} (opens a form)", f"Delete a {kind}",
                 f"Stop a {kind} until resumed", f"Turn a paused {kind} back on")):
            callback = app_commands.autocomplete(entry=_entry_autocomplete(kind))(callback)
            callback = app_commands.describe(entry=f"The {kind} - start typing to pick from the list")(callback)
            group.command(name=name, description=description)(callback)
    tree.add_command(schedule)
    tree.add_command(rule)


async def _view(ctx, kind, ident):
    return _display(await _show(ctx, kind, ident))


async def _set_status(ctx, kind, ident, status):
    row = await _show(ctx, kind, ident)
    updated = await AutomationStore().update(ctx.guild.id, ident, ctx.user.id, row["revision"], status=status)
    if status == "deleted":
        return f"Deleted {kind} `{ident}`. A run that already started may still finish."
    return _display(updated) + "\nA run that already started may still finish."

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

from classes import card, card_kit as K
from classes.automation_policy import (
    WEEKDAYS, form_values, schedule_text, settings, timing_from_form, timing_text,
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


_KIND_LABELS = {"schedule": "⏰ Schedule", "rule": "⚡ Rule"}
_STATUS_COLOURS = {"enabled": K.ACTIVE, "paused": K.CLOSED, "suspended": K.WARN, "deleted": K.CLOSED}
_STATUS_WORDS = {"enabled": "on", "paused": "paused", "suspended": "suspended", "deleted": "deleted"}
TITLE_CHARS = 120  # a longer action reads badly in a heading's large type


def _display(row, note: str = "") -> K.CardSpec:
    """One schedule or rule as a card: its action as the title (or the body,
    when long), then when and where it runs and how it last went."""
    kind = "Follow-up" if row.get("followup") else _KIND_LABELS[row["kind"]].split(" ", 1)[1]
    emoji = _KIND_LABELS[row["kind"]].split(" ", 1)[0]
    label = f"{emoji} {kind} · {row['id']} · {_STATUS_WORDS.get(row['status'], row['status'])}"
    pairs = []
    if row["kind"] == "schedule":
        pairs.append(("When", schedule_text(row)))
        if row["status"] == "enabled" and row.get("next_run"):
            pairs.append(("Next", f"<t:{int(row['next_run'])}:f> · <t:{int(row['next_run'])}:R>"))
    else:
        pairs.append(("Matches", f"`{row['pattern']}` ({row['match_mode']})"))
    pairs.append(("Where", f"<#{row['channel_id']}>"))
    last = row.get("last_result")
    if last:
        detail = f" — {last['detail']}" if last.get("detail") else ""
        when = int(datetime.fromisoformat(last["at"]).timestamp())
        pairs.append(("Last run", f"{last['status']}{detail} · <t:{when}:R>"))
    else:
        pairs.append(("Last run", "never"))
    action = " ".join(str(row["action"]).split())
    short = len(action) <= TITLE_CHARS
    body = ([] if short else [K.Text(K.safe(row["action"]))]) + [K.Text(K.details(pairs))]
    if note:
        body.append(K.Text(note))
    return K.CardSpec(label=label, title=K.safe(action) if short else "", body=body,
                      footer=f"Created by <@{row['creator_id']}> · last edited by <@{row['editor_id']}>",
                      colour=_STATUS_COLOURS.get(row["status"], K.ACTIVE))


ALREADY_STARTED = "A run that already started may still finish."
PANEL_SECONDS = 600


def _panel(kind, row, note: str = "") -> K.CardSpec:
    """An entry's card with its buttons: Pause or Resume, Edit, Delete. Each
    press checks access again, as the matching slash command does."""
    spec = _display(row, note)
    if row["status"] == "deleted":
        return spec
    ident = row["id"]

    def run(status):
        async def press(ctx):
            try:
                result = await _set_status(ctx, kind, ident, status)
            except ERRORS as exc:
                await card.reply(ctx, K.error(str(exc)))
                return
            await ctx.response.edit_message(view=card.views(result, timeout=PANEL_SECONDS)[0])
        return press

    async def edit(ctx):
        await _open_editor(ctx, kind, ident)

    pause = ("Resume", "▶️", "enabled") if row["status"] in ("paused", "suspended") else ("Pause", "⏸️", "paused")
    buttons = [card.button(pause[0], discord.ButtonStyle.secondary, run(pause[2]), pause[1])]
    if not row.get("followup"):
        buttons.append(card.button("Edit", discord.ButtonStyle.secondary, edit, "✏️"))
    buttons.append(card.button("Delete", discord.ButtonStyle.danger, run("deleted"), "🗑️"))
    spec.body += [K.Divider(), K.Buttons(tuple(buttons))]
    return spec


def _summary(row):
    detail = (schedule_text(row) if row["kind"] == "schedule"
              else f"\"{row['pattern']}\"")
    return f"{row['id']} — {row['status']}, {detail}: {row['action']}"[:100]


async def _respond(ctx, operation):
    """The command's private answer: the card `operation` returns, or the error."""
    try:
        result = await operation()
    except ERRORS as exc:
        await card.reply(ctx, K.error(str(exc)))
        return
    await card.reply(ctx, result, timeout=PANEL_SECONDS)


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


def retry_card(message: str, reopen) -> K.CardSpec:
    """A form's error, with a button that reopens the form with what was
    typed instead of starting again."""
    async def retry(ctx):
        await ctx.response.send_modal(reopen())
    return K.CardSpec(label="❌ Not saved", body=[K.Text(message), K.Buttons((
        card.button("Fix and try again", discord.ButtonStyle.primary, retry, "✏️"),))],
        colour=K.FAIL)


async def _submit(ctx, operation, reopen):
    try:
        result = await operation()
    except QuotaError as exc:
        await card.reply(ctx, K.error(str(exc), "❌ Not saved"))
        return
    except ERRORS as exc:
        await card.reply(ctx, retry_card(str(exc), reopen), timeout=PANEL_SECONDS)
        return
    await card.reply(ctx, result, timeout=PANEL_SECONDS)


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
            return _panel("schedule", row, "✅ Saved.")
        await _submit(ctx, operation, lambda: ScheduleForm(self.kind, self.channel, self.row, values))


def type_picker(channel, row=None) -> K.CardSpec:
    """The buttons /schedule create and /schedule edit show first. The
    current type of an edited schedule is highlighted."""
    current = row["timing"]["type"] if row else None

    def opener(kind):
        async def open_form(ctx):
            await ctx.response.send_modal(ScheduleForm(kind, channel, row))
        return open_form

    buttons = tuple(card.button(
        f"{label} (current)" if kind == current else label,
        discord.ButtonStyle.primary if kind == current or not row else discord.ButtonStyle.secondary,
        opener(kind)) for kind, label in SCHEDULE_TYPES)
    if row:
        return K.CardSpec(label=f"⏰ Editing schedule · {row['id']}",
                          body=[K.Text(f"Now: {timing_text(row['timing'], row['timezone'])}"),
                                K.Text("Pick the current type to change its time or action, or another "
                                       "type to switch."), K.Buttons(buttons)])
    return K.CardSpec(label="⏰ New schedule", title="What kind of schedule?",
                      body=[K.Buttons(buttons)])


async def _open_editor(ctx, kind, ident):
    """/schedule edit and /rule edit, and an entry's Edit button."""
    try:
        row = await _show(ctx, kind, ident)
    except ERRORS as exc:
        await card.reply(ctx, K.error(str(exc)))
        return
    channel = ctx.guild.get_channel_or_thread(row["channel_id"])
    if kind == "rule":
        await ctx.response.send_modal(RuleForm(channel, row))
        return
    if row.get("followup"):
        await card.reply(ctx, K.info("⏰ Follow-ups can't be edited",
                                     f"Delete `{row['id']}` with `/schedule delete` and ask me for a new one."))
        return
    await card.reply(ctx, type_picker(channel, row), timeout=PANEL_SECONDS)


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
            return _panel("rule", row, "✅ Saved.")
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
            await card.reply(ctx, K.error(str(exc)))
            return
        await card.reply(ctx, type_picker(ctx.channel), timeout=PANEL_SECONDS)

    @rule.command(name="create", description="Create a message rule in this channel (opens a form)")
    async def rule_create(ctx: discord.Interaction):
        try:
            _access(ctx, ctx.channel)
            await AutomationStore().check_quota(ctx.guild.id, "rule")
        except AutomationError as exc:
            await card.reply(ctx, K.error(str(exc)))
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
                        visible.append(_list_entry(row))
                    return K.CardSpec(label=f"{_KIND_LABELS[kind]}s · {len(visible)}",
                                      body=K.listing(visible) or [K.Text("No entries you can access.")],
                                      footer=f"See one with /{kind} view")
                await _respond(ctx, op)

            async def view_entry(ctx: discord.Interaction, entry: str):
                await _respond(ctx, lambda: _view(ctx, kind, entry))

            async def edit_entry(ctx: discord.Interaction, entry: str):
                await _open_editor(ctx, kind, entry)

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


def _list_entry(row) -> str:
    """One line of /schedule list or /rule list, and its details under it."""
    outcome = (row.get("last_result") or {}).get("status", "never run")
    bits = [f"`{row['id']}`", f"<#{row['channel_id']}>", _STATUS_WORDS.get(row["status"], row["status"]),
            f"last: {outcome}"]
    if row.get("followup"):
        bits.insert(1, "follow-up")
    if row["kind"] == "schedule" and row["status"] == "enabled" and row.get("next_run"):
        bits.append(f"next <t:{int(row['next_run'])}:R>")
    return f"{K.safe(' '.join(str(row['action']).split()))}\n-# " + " · ".join(bits)


async def _view(ctx, kind, ident):
    return _panel(kind, await _show(ctx, kind, ident))


async def _set_status(ctx, kind, ident, status):
    row = await _show(ctx, kind, ident)
    updated = await AutomationStore().update(ctx.guild.id, ident, ctx.user.id, row["revision"], status=status)
    if status == "deleted":
        return K.CardSpec(label=f"🗑️ {kind.capitalize()} deleted · {ident}",
                          body=[K.Text(K.safe(row["action"]))], footer=ALREADY_STARTED, colour=K.CLOSED)
    return _panel(kind, updated, f"-# {ALREADY_STARTED}")

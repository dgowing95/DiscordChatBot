"""Conversational management tools; slash commands use the same store."""
import json
from agents import function_tool, RunContextWrapper
from classes.automation_policy import describe, settings
from classes.automation_store import AutomationStore, AutomationError


def _scope(wrapper):
    context = wrapper.context
    if context.get("automatic") or not settings()["enabled"]:
        raise AutomationError("Automation management is unavailable in this run")
    if not context.get("guild_id") or not context.get("channel"):
        raise AutomationError("Use automations in a server channel")
    channel = context["channel"]
    actor = getattr(context.get("original_message"), "author", None)
    if actor is None or not channel.permissions_for(actor).send_messages or not channel.permissions_for(actor).view_channel:
        raise AutomationError("You cannot manage automations in this channel")
    bot = getattr(channel.guild, "me", None)
    if bot is not None and not channel.permissions_for(bot).send_messages:
        raise AutomationError("I cannot send messages in this channel")
    return context


async def _get(wrapper, kind, ident):
    c = _scope(wrapper)
    row = await AutomationStore().get(c["guild_id"], ident)
    if not row or row["kind"] != kind:
        raise AutomationError("Automation not found in this server")
    channel = (c["channel"] if c["channel"].id == row["channel_id"] else
               c["discord_client"].get_channel(row["channel_id"]) if c.get("discord_client") else None)
    guild = getattr(c["channel"], "guild", None)
    actor = guild.get_member(c["user_id"]) if guild else None
    if actor is None and getattr(c.get("original_message"), "author", None):
        author = c["original_message"].author
        if author.id == c["user_id"]:
            actor = author
    if channel is None or actor is None or not channel.permissions_for(actor).view_channel:
        raise AutomationError("You cannot access that channel")
    return row


@function_tool
async def list_schedules(wrapper: RunContextWrapper[dict]) -> str:
    """List schedules in this server that you can access."""
    c = _scope(wrapper)
    rows = await AutomationStore().list(c["guild_id"], "schedule")
    visible = []
    for row in rows:
        try:
            await _get(wrapper, "schedule", row["id"])
            visible.append(row)
        except AutomationError:
            pass
    return json.dumps([describe(row) for row in visible])


@function_tool
async def get_schedule(wrapper: RunContextWrapper[dict], schedule_id: str) -> str:
    """Inspect a schedule by its ID."""
    return json.dumps(describe(await _get(wrapper, "schedule", schedule_id)))


@function_tool
async def create_schedule(wrapper: RunContextWrapper[dict], action: str, timing_type: str,
                          at: str = "", every: int = 0, unit: str = "hours", start: str = "",
                          local_time: str = "", weekday: str = "", timezone: str = "") -> str:
    """Create a schedule in this channel. Use explicit timing: once needs at (ISO date/time with offset); interval needs every and unit (hours/days/weeks); daily needs local_time (HH:MM); weekly also needs weekday. Ask the user when timing is ambiguous. Defaults to Europe/London. Afterwards tell the user the timing_summary and next_run_local from the result."""
    c = _scope(wrapper)
    timing = {"type": timing_type}
    if timing_type == "once": timing["at"] = at
    elif timing_type == "interval": timing.update(every=every, unit=unit, **({"start": start} if start else {}))
    else: timing.update(time=local_time, **({"weekday": weekday.lower()} if timing_type == "weekly" else {}))
    row = await AutomationStore().create(c["guild_id"], c["channel"].id, c["user_id"], "schedule", action,
                                         timing=timing, timezone=timezone or None)
    return json.dumps(describe(row))


@function_tool
async def update_schedule(wrapper: RunContextWrapper[dict], schedule_id: str, revision: int,
                          action: str = "", status: str = "", timing_json: str = "", timezone: str = "") -> str:
    """Edit, pause, or resume a schedule. First inspect it to get its current revision. timing_json is a JSON object with the same explicit timing fields as create_schedule."""
    row = await _get(wrapper, "schedule", schedule_id)
    changes = dict(action=action) if action else {}
    if status: changes["status"] = status
    if timing_json: changes["timing"] = json.loads(timing_json)
    if timezone: changes["timezone"] = timezone
    return json.dumps(describe(await AutomationStore().update(row["guild_id"], schedule_id, wrapper.context["user_id"], revision, **changes)))


@function_tool
async def delete_schedule(wrapper: RunContextWrapper[dict], schedule_id: str, revision: int) -> str:
    """Delete a schedule after inspecting its current revision."""
    row = await _get(wrapper, "schedule", schedule_id)
    return json.dumps(describe(await AutomationStore().update(row["guild_id"], schedule_id, wrapper.context["user_id"], revision, status="deleted")))


@function_tool
async def list_rules(wrapper: RunContextWrapper[dict]) -> str:
    """List message rules in accessible server channels."""
    c = _scope(wrapper)
    rows = await AutomationStore().list(c["guild_id"], "rule")
    visible = []
    for row in rows:
        try:
            await _get(wrapper, "rule", row["id"])
            visible.append(row)
        except AutomationError:
            pass
    return json.dumps([describe(row) for row in visible])


@function_tool
async def get_rule(wrapper: RunContextWrapper[dict], rule_id: str) -> str:
    """Inspect a rule by its ID."""
    return json.dumps(describe(await _get(wrapper, "rule", rule_id)))


@function_tool
async def create_rule(wrapper: RunContextWrapper[dict], pattern: str, action: str, match_mode: str = "word") -> str:
    """Create a rule in this channel. match_mode is word (whole words/phrases) or substring (literal text)."""
    c = _scope(wrapper)
    return json.dumps(describe(await AutomationStore().create(c["guild_id"], c["channel"].id, c["user_id"], "rule", action,
                                                              pattern=pattern, match_mode=match_mode)))


@function_tool
async def update_rule(wrapper: RunContextWrapper[dict], rule_id: str, revision: int,
                      pattern: str = "", action: str = "", match_mode: str = "", status: str = "") -> str:
    """Edit, pause, or resume a rule after inspecting its current revision."""
    row = await _get(wrapper, "rule", rule_id)
    changes = {k: v for k, v in dict(pattern=pattern, action=action, match_mode=match_mode, status=status).items() if v}
    return json.dumps(describe(await AutomationStore().update(row["guild_id"], rule_id, wrapper.context["user_id"], revision, **changes)))


@function_tool
async def delete_rule(wrapper: RunContextWrapper[dict], rule_id: str, revision: int) -> str:
    """Delete a rule after inspecting its current revision."""
    row = await _get(wrapper, "rule", rule_id)
    return json.dumps(describe(await AutomationStore().update(row["guild_id"], rule_id, wrapper.context["user_id"], revision, status="deleted")))


def automation_tools():
    if not settings()["enabled"]:
        return []
    return [list_schedules, get_schedule, create_schedule, update_schedule, delete_schedule,
            list_rules, get_rule, create_rule, update_rule, delete_rule]

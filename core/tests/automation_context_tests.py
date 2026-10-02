from types import SimpleNamespace

import pytest

from classes.text_llm_handler import TextLLMHandler, agent_tools


def test_automatic_tool_set_excludes_memory_and_management(monkeypatch):
    monkeypatch.setenv("AUTOMATIONS_ENABLED", "1")
    monkeypatch.setenv("IMAGE_GEN_ENABLED", "1")
    monkeypatch.setenv("SANDBOX_ENABLED", "1")
    names = {tool.name for tool in agent_tools(automatic=True)}
    assert {"web_search", "fetch_url", "generate_image", "run_code_sandbox", "change_personality"} <= names
    assert not any(name.endswith("memory") or "memories" in name for name in names)
    assert not any("schedule" in name or "rule" in name for name in names)


def test_automatic_handler_needs_no_source_message():
    channel = SimpleNamespace(id=10)
    handler = TextLLMHandler([], 20, None, actor_id=30, channel=channel, automatic=True)
    assert handler.user_memory is None
    assert handler.original_message is None
    assert handler.actor_id == 30


def test_disabled_automations_hide_management_tools(monkeypatch):
    monkeypatch.setenv("AUTOMATIONS_ENABLED", "0")
    names = {tool.name for tool in agent_tools()}
    assert not any("schedule" in name or "rule" in name for name in names)


def test_slash_groups_offer_same_management_operations():
    from classes.automation_commands import register_automation_commands
    groups = []
    class Tree:
        def add_command(self, group): groups.append(group)
    register_automation_commands(Tree())
    assert {group.name for group in groups} == {"schedule", "rule"}
    for group in groups:
        assert {command.name for command in group.commands} == {
            "create", "list", "view", "edit", "delete", "pause", "resume"}


def _commands(group):
    return {command.name: command for command in group.commands}


def test_create_takes_no_slash_options_and_entries_autocomplete():
    from classes.automation_commands import register_automation_commands
    groups = []
    class Tree:
        def add_command(self, group): groups.append(group)
    register_automation_commands(Tree())
    for group in groups:
        commands = _commands(group)
        assert commands["create"].parameters == [] and commands["list"].parameters == []
        for name in ("view", "edit", "delete", "pause", "resume"):
            (entry,) = commands[name].parameters
            assert entry.name == "entry" and entry.autocomplete and entry.description


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["once", "interval", "daily", "weekly"])
async def test_schedule_forms_hold_only_their_fields_within_discord_limits(kind):
    from classes.automation_commands import ScheduleForm
    form = ScheduleForm(kind, channel=None)
    expected = {"once": {"when"}, "interval": {"every", "unit", "start"},
                "daily": {"time"}, "weekly": {"weekday", "time"}}[kind] | {"timezone", "action"}
    assert set(form.fields) == expected
    _check_limits(form)


@pytest.mark.asyncio
async def test_edit_form_is_prefilled_and_rule_form_fits():
    from classes.automation_commands import RuleForm, ScheduleForm
    row = {"id": "abc", "kind": "schedule", "timezone": "Europe/London", "action": "find it",
           "timing": {"type": "weekly", "weekday": "friday", "time": "09:00"}}
    form = ScheduleForm("weekly", channel=None, row=row)
    assert form.fields["time"].default == "09:00"
    assert form.fields["action"].default == "find it"
    assert [o.value for o in form.fields["weekday"].options if o.default] == ["friday"]
    _check_limits(form)
    _check_limits(RuleForm(channel=None))


def _check_limits(form):
    import discord
    assert len(form.title) <= 45
    labels = [item for item in form.children if isinstance(item, discord.ui.Label)]
    assert 0 < len(labels) <= 5 and len(labels) == len(form.children)
    for label in labels:
        assert len(label.text) <= 45 and len(label.description or "") <= 100


@pytest.mark.asyncio
async def test_create_on_a_full_server_explains_instead_of_opening_a_form(monkeypatch):
    from unittest.mock import AsyncMock
    from classes import automation_commands
    from classes.automation_store import QuotaError
    class Store:
        async def check_quota(self, guild, kind):
            raise QuotaError(f"This server already has 1 of 1 allowed {kind}s")
    monkeypatch.setattr(automation_commands, "AutomationStore", Store)
    groups = []
    class Tree:
        def add_command(self, group): groups.append(group)
    automation_commands.register_automation_commands(Tree())
    perms = SimpleNamespace(view_channel=True, send_messages=True)
    guild = SimpleNamespace(id=1, me=object())
    channel = SimpleNamespace(guild=guild, permissions_for=lambda member: perms)
    for group in groups:
        ctx = SimpleNamespace(guild=guild, channel=channel, user=object(),
                              response=SimpleNamespace(send_message=AsyncMock(), send_modal=AsyncMock()))
        await _commands(group)["create"].callback(ctx)
        ctx.response.send_modal.assert_not_called()
        args, kwargs = ctx.response.send_message.call_args
        assert "1 of 1 allowed" in args[0] and kwargs.get("ephemeral") and "view" not in kwargs

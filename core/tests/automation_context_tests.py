from types import SimpleNamespace

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

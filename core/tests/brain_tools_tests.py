"""Which tools the brain gets (core/classes/brain_tools.py), row by row of
its gating table, and that the persona has only ask_brain.

Run from the repo root:
    pytest core/tests/brain_tools_tests.py
"""
import pytest

from classes.brain_tools import brain_tools
from classes.text_llm_handler import persona_tools

ALWAYS = {"web_search", "fetch_url", "search_messages", "read_messages_around", "check_polls",
          "create_poll", "change_personality", "attach_file", "save_messages_to_file",
          "ask_user", "report"}
EVENTS = {"create_event", "list_events", "update_event"}
MEMORY = {"store_memory", "remove_memory", "clear_memories"}


@pytest.fixture
def all_on(monkeypatch):
    for name in ("IMAGE_GEN_ENABLED", "AUTOMATIONS_ENABLED", "VOICE_ENABLED", "EVENTS_ENABLED"):
        monkeypatch.setenv(name, "1")


def _names(**kw):
    return [tool.name for tool in brain_tools(**kw)]


def test_chat_gets_everything_and_joins_voice(all_on):
    names = set(_names())
    assert ALWAYS | EVENTS | MEMORY | {"generate_image", "join_voice_channel", "schedule_followup"} <= names
    assert "leave_voice_channel" not in names


def test_a_voice_turn_leaves_rather_than_joins(all_on):
    names = set(_names(voice=True))
    assert "leave_voice_channel" in names and "join_voice_channel" not in names
    assert MEMORY <= names


def test_an_automatic_run_has_no_memory_automations_or_voice(all_on):
    names = set(_names(automatic=True))
    assert ALWAYS | EVENTS | {"generate_image"} <= names
    assert not names & MEMORY
    assert not any("schedule" in n or "rule" in n for n in names)
    assert not any("voice" in n for n in names)


def test_switched_off_features_are_left_out(monkeypatch):
    for name in ("IMAGE_GEN_ENABLED", "AUTOMATIONS_ENABLED", "VOICE_ENABLED", "EVENTS_ENABLED"):
        monkeypatch.setenv(name, "0")
    names = set(_names())
    assert "generate_image" not in names and not names & EVENTS
    assert not any("voice" in n or "schedule" in n for n in names)
    assert ALWAYS | MEMORY <= names


def test_every_tool_is_one_object_and_the_order_is_fixed(all_on):
    first, second = brain_tools(), brain_tools()
    assert [id(t) for t in first] == [id(t) for t in second]
    assert len({t.name for t in first}) == len(first)


def test_the_brain_ends_with_report(all_on):
    assert _names()[-1] == "report"


def test_create_event_keeps_its_poll_then_follow_up_advice(all_on):
    # The old sandbox copy cut it (the sandbox had neither tool); the brain has both.
    (tool,) = [t for t in brain_tools() if t.name == "create_event"]
    assert "To find a time" in tool.description


def test_the_persona_has_exactly_ask_brain():
    assert [t.name for t in persona_tools()] == ["ask_brain"]


# --------------------------- Tier 3 ---------------------------

def test_start_workflow_is_offered_only_when_tier_3_is_allowed_and_never_in_voice(all_on):
    assert "start_workflow" not in _names()
    assert _names(workflows=True)[-2:] == ["start_workflow", "report"]  # before report
    assert "start_workflow" in _names(automatic=True, workflows=True)
    assert "start_workflow" not in _names(voice=True, workflows=True)


def test_a_workflows_agents_get_their_own_short_lists(all_on, monkeypatch):
    monkeypatch.setenv("SKILLS_ENABLED", "1")
    assert _names(mode="manager") == ["web_search", "fetch_url", "search_messages",
                                      "read_messages_around", "search_skills", "use_skill",
                                      "submit_plan"]
    assert _names(mode="review") == ["submit_review"]
    assert _names(mode="finish") == ["report"]
    assert _names(mode="researcher") == ["web_search", "fetch_url", "search_messages",
                                         "read_messages_around", "attach_file",
                                         "save_messages_to_file", "report"]
    assert _names(mode="implementer") == ["fetch_url", "generate_image", "attach_file", "report"]
    # Nothing that belongs to a conversation, and no way to start another.
    for mode in ("manager", "review", "finish", "researcher", "implementer"):
        names = set(_names(mode=mode))
        assert not names & (MEMORY | EVENTS | {"ask_user", "start_workflow", "create_poll",
                                               "change_personality"})


def test_an_unknown_mode_is_an_error():
    with pytest.raises(ValueError):
        brain_tools(mode="boss")

"""The buttons the cards carry beyond questions and plans: Restore on a
retired skill, Leave on a voice call, and a schedule's Pause / Resume /
Edit / Delete. Each does what its slash command does, checked again when
pressed."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from classes import automation_commands, card, card_kit as K
from card_checks import interaction, label_of, replied_text, view_text

GUILD = 1


# --------------------------- Restore ---------------------------

@pytest.mark.asyncio
async def test_restore_brings_the_skill_back_and_takes_its_button_off(monkeypatch):
    from classes import skill_tools
    store = MagicMock()
    store.restore = AsyncMock()
    monkeypatch.setattr(skill_tools, "SkillStore", lambda: store)
    retired = card.views(K.CardSpec(label="🗑️ Skill retired · daily tidy-up", title="old-one",
                                    body=[K.Text("Not used in 60 days.")]))[0]
    message = SimpleNamespace(components=[c for c in _components(retired)])
    channel = MagicMock()
    channel.send = AsyncMock()
    press = interaction(guild_id=GUILD, user=SimpleNamespace(id=9, mention="<@9>"),
                        message=message, channel=channel)
    button = skill_tools.RestoreButton("old-one")
    assert button.item.custom_id == "skill:restore:old-one"
    await button.callback(press)
    store.restore.assert_awaited_once_with(GUILD, "old-one", 9)
    done = press.response.edit_message.await_args.kwargs["view"]
    assert done.buttons == [] and done.colour == K.CLOSED
    assert "Not used in 60 days." in view_text(done) and "Restored by <@9>" in view_text(done)
    assert label_of(channel.send.await_args.kwargs["view"]) == "♻️ Skill restored"


@pytest.mark.asyncio
async def test_restore_of_a_skill_that_is_back_already_says_so(monkeypatch):
    from classes import skill_tools
    from classes.skill_store import SkillError
    store = MagicMock()
    store.restore = AsyncMock(side_effect=SkillError("No deleted skill named old-one in this server."))
    monkeypatch.setattr(skill_tools, "SkillStore", lambda: store)
    press = interaction(guild_id=GUILD, user=SimpleNamespace(id=9))
    await skill_tools.RestoreButton("old-one").callback(press)
    assert "No deleted skill" in replied_text(press)
    press.response.edit_message.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("off, error, says", [
    (True, None, "Skills are switched off."),
    (False, ConnectionError("redis down"), "Something went wrong")])
async def test_restore_refuses_with_skills_off_and_answers_when_it_fails(monkeypatch, off, error, says):
    from classes import skill_tools
    if off:
        monkeypatch.setenv("SKILLS_ENABLED", "0")
    store = MagicMock()
    store.restore = AsyncMock(side_effect=error)
    monkeypatch.setattr(skill_tools, "SkillStore", lambda: store)
    press = interaction(guild_id=GUILD, user=SimpleNamespace(id=9))
    await skill_tools.RestoreButton("old-one").callback(press)
    assert says in replied_text(press)
    press.response.edit_message.assert_not_awaited()
    if off:
        store.restore.assert_not_awaited()


def _components(view):
    from discord.components import _component_factory
    return [_component_factory(c, MagicMock()) for c in view.to_components()]


# --------------------------- Leave ---------------------------

@pytest.mark.asyncio
async def test_leave_ends_the_call_and_answers_when_there_is_none(monkeypatch):
    from classes import voice_commands, voice_gate
    session = MagicMock()
    session.end = AsyncMock()
    monkeypatch.setattr(voice_gate, "session", lambda guild_id: session)
    press = interaction(guild_id=GUILD)
    await voice_commands.LeaveButton().callback(press)
    session.end.assert_awaited_once()
    assert "Leaving the call" in replied_text(press)
    monkeypatch.setattr(voice_gate, "session", lambda guild_id: None)
    after = interaction(guild_id=GUILD)
    await voice_commands.LeaveButton().callback(after)
    assert "not in a voice call" in replied_text(after)


# --------------------------- a schedule's buttons ---------------------------

ROW = {"id": "s-1", "kind": "schedule", "status": "enabled", "channel_id": 77, "revision": 3,
       "timing": {"type": "daily", "time": "09:00"}, "timezone": "Europe/London", "next_run": 2_000_000_000,
       "creator_id": 5, "editor_id": 5, "action": "Post the daily digest"}


class Store:
    rows: dict = {}
    updates: list = []

    async def get(self, guild, ident):
        return self.rows.get(ident)

    async def update(self, guild, ident, user, revision, **changes):
        Store.updates.append((ident, user, changes))
        row = dict(self.rows[ident], **changes, revision=revision + 1)
        self.rows[ident] = row
        return row


def _press(allowed=True):
    perms = SimpleNamespace(view_channel=allowed, send_messages=allowed)
    channel = SimpleNamespace(guild=SimpleNamespace(id=GUILD), permissions_for=lambda member: perms)
    guild = SimpleNamespace(id=GUILD, me=object(), get_channel_or_thread=lambda cid: channel)
    return interaction(guild=guild, user=SimpleNamespace(id=5))


def _buttons(spec):
    view = card.views(spec)[0]
    return {b.label: b for b in view.buttons}


@pytest.fixture
def store(monkeypatch):
    Store.rows, Store.updates = {"s-1": dict(ROW)}, []
    monkeypatch.setattr(automation_commands, "AutomationStore", Store)
    return Store


@pytest.mark.asyncio
async def test_a_schedule_card_has_pause_edit_and_delete(store):
    buttons = _buttons(automation_commands._panel("schedule", ROW))
    assert list(buttons) == ["Pause", "Edit", "Delete"]
    assert "Post the daily digest" in view_text(card.views(automation_commands._panel("schedule", ROW))[0])


@pytest.mark.asyncio
async def test_pause_then_resume_from_the_card(store):
    press = _press()
    await _buttons(automation_commands._panel("schedule", ROW))["Pause"].callback(press)
    assert store.updates[-1] == ("s-1", 5, {"status": "paused"})
    paused = press.response.edit_message.await_args.kwargs["view"]
    assert [b.label for b in paused.buttons] == ["Resume", "Edit", "Delete"]
    assert paused.colour == K.CLOSED
    again = _press()
    await _buttons(automation_commands._panel("schedule", store.rows["s-1"]))["Resume"].callback(again)
    assert store.updates[-1][2] == {"status": "enabled"}


@pytest.mark.asyncio
async def test_delete_from_the_card_and_access_is_checked_again(store):
    refused = _press(allowed=False)
    await _buttons(automation_commands._panel("schedule", ROW))["Delete"].callback(refused)
    assert store.updates == [] and "must be able to view and send" in replied_text(refused)
    press = _press()
    await _buttons(automation_commands._panel("schedule", ROW))["Delete"].callback(press)
    assert store.updates[-1][2] == {"status": "deleted"}
    gone = press.response.edit_message.await_args.kwargs["view"]
    assert gone.buttons == [] and "deleted" in view_text(gone)


@pytest.mark.asyncio
async def test_edit_from_the_card_offers_the_types_and_a_rule_opens_its_form(store):
    press = _press()
    await _buttons(automation_commands._panel("schedule", ROW))["Edit"].callback(press)
    picker = press.response.send_message.await_args.kwargs["view"]
    assert [b.label for b in picker.buttons] == ["Once", "Repeat every…", "Daily (current)", "Weekly"]
    rule = dict(ROW, id="r-1", kind="rule", pattern="deploy", match_mode="word")
    store.rows["r-1"] = rule
    form = _press()
    await _buttons(automation_commands._panel("rule", rule))["Edit"].callback(form)
    assert isinstance(form.response.send_modal.await_args.args[0], automation_commands.RuleForm)


def test_a_follow_up_cannot_be_edited_from_its_card():
    spec = automation_commands._panel("schedule", dict(ROW, followup=True))
    assert list(_buttons(spec)) == ["Pause", "Delete"]

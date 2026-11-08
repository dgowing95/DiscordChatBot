"""/workspace status, reset and progress (classes/workspace_commands.py).

Run from the repo root: `pytest core/tests/workspace_commands_tests.py`
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from classes import card, card_kit as K, workspace_commands as wc
from card_checks import interaction, replied, replied_text, view_text


class FakeConfig:
    def __init__(self, **settings):
        self.data = dict(settings)

    async def get_setting(self, setting, guild_id):
        return self.data.get(setting) or False  # configManager's own "unset" answer

    async def update_setting(self, setting, value, guild_id):
        self.data[setting] = value

    async def delete_setting(self, setting, guild_id):
        self.data.pop(setting, None)


def _config(**settings):
    config = FakeConfig(**settings)
    return config, patch("classes.config_manager.configManager", return_value=config)


# ---------------------- progress setting ----------------------

@pytest.mark.asyncio
async def test_progress_is_off_by_default():
    config, p = _config()
    with p:
        assert await wc.progress_enabled(1) is False


@pytest.mark.asyncio
@pytest.mark.parametrize("raw, expected", [("True", True), ("on", True), ("1", True),
                                           ("False", False), ("garbage", False)])
async def test_progress_reads_the_new_setting(raw, expected):
    config, p = _config(workspace_progress=raw)
    with p:
        assert await wc.progress_enabled(1) is expected


@pytest.mark.asyncio
async def test_the_old_setting_is_copied_once_and_dropped():
    config, p = _config(sandbox_progress_updates="True")
    with p:
        assert await wc.progress_enabled(1) is True
    assert config.data == {"workspace_progress": "True"}


@pytest.mark.asyncio
async def test_the_new_setting_wins_over_an_old_one():
    config, p = _config(workspace_progress="False", sandbox_progress_updates="True")
    with p:
        assert await wc.progress_enabled(1) is False
    assert "sandbox_progress_updates" in config.data  # untouched: the new one is set


@pytest.mark.asyncio
async def test_a_config_failure_is_off():
    config = MagicMock()
    config.get_setting = AsyncMock(side_effect=RuntimeError("redis down"))
    with patch("classes.config_manager.configManager", return_value=config):
        assert await wc.progress_enabled(1) is False


# ---------------------- status card ----------------------

def _text(spec) -> str:
    return "\n".join(b.content for c in K.split(spec) for b in c if isinstance(b, K.Text))


def test_status_of_a_stopped_workspace_that_was_never_saved():
    spec = wc.status_card({"warm": False, "last_snapshot_at": None})
    assert spec.title == "⚪ Stopped" and "**Last saved** not yet" in _text(spec)


def test_status_of_a_workspace_waiting_after_a_restart():
    spec = wc.status_card({"warm": False, "waiting": True, "last_snapshot_at": 1})
    assert spec.title == "🟡 Waiting" and "Stopped" not in _text(spec)


def test_status_of_a_running_busy_workspace():
    spec = wc.status_card({"warm": True, "started_at": 1000, "busy": 2, "jobs": 4,
                           "last_snapshot_at": 1500, "snapshot_bytes": 3 * 1024 * 1024},
                          now=1000 + 600)
    text = _text(spec)
    assert spec.title == "🟢 Running"
    assert "**Up for** 10 min · **in use by** 2 tasks" in text
    assert "**Task folders** 4" in text
    assert "<t:1500:R>" in text and "3.0 MB" in text


def test_status_warns_when_the_workspace_is_too_big_to_save():
    spec = wc.status_card({"warm": False, "last_snapshot_at": 1, "snapshot_bytes": 10,
                           "snapshot_too_large": True})
    assert "too big to save" in _text(spec)


# ---------------------- the commands ----------------------

class FakeTree:
    def __init__(self):
        self.groups = []

    def add_command(self, group):
        self.groups.append(group)


def _commands():
    tree = FakeTree()
    wc.register_workspace_commands(tree)
    (group,) = tree.groups
    assert group.name == "workspace"
    return {c.name: c for c in group.commands}


def _interaction(manage=True):
    return interaction(guild_id=7, user=SimpleNamespace(id=99),
                       permissions=SimpleNamespace(manage_guild=manage))


def _button(view, label):
    return next(b for b in view.buttons if b.label.startswith(label))


def test_the_group_has_status_reset_and_progress():
    assert set(_commands()) == {"status", "reset", "progress"}


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["status", "reset"])
async def test_status_and_reset_need_manage_server(name):
    ctx = _interaction(manage=False)
    await _commands()[name].callback(ctx)
    assert "Manage Server" in replied_text(ctx)
    assert ctx.response.send_message.await_args.kwargs["ephemeral"] is True


@pytest.mark.asyncio
async def test_status_replies_privately_with_the_state_and_its_buttons():
    ctx = _interaction()
    fake = MagicMock()
    fake.status = AsyncMock(return_value={"warm": False, "last_snapshot_at": None})
    ctx.response.is_done = MagicMock(return_value=True)  # it deferred first
    config, p = _config()
    with p, patch("classes.workspace_manager.workspaces", return_value=fake):
        await _commands()["status"].callback(ctx)
    fake.status.assert_awaited_once_with(7)
    view = ctx.followup.send.await_args.kwargs["view"]
    assert "Stopped" in view_text(view)
    assert ctx.followup.send.await_args.kwargs["ephemeral"] is True
    assert [b.label for b in view.buttons] == ["Live progress: off", "Reset…"]


@pytest.mark.asyncio
async def test_the_panels_buttons_toggle_progress_and_ask_before_a_reset():
    fake = MagicMock()
    fake.status = AsyncMock(return_value={"warm": False, "last_snapshot_at": None})
    config, p = _config()
    with p, patch.object(wc, "configManager", return_value=config), \
            patch("classes.workspace_manager.workspaces", return_value=fake):
        panel = card.views(await wc._status_panel(7, 99))[0]
        press = _interaction()
        await _button(panel, "Live progress").callback(press)
        assert config.data["workspace_progress"] == "True"
        updated = press.response.edit_message.await_args.kwargs["view"]
        assert _button(updated, "Live progress").label == "Live progress: on"
        reset = _interaction(manage=False)
        await _button(panel, "Reset").callback(reset)
        reset.response.edit_message.assert_not_awaited()  # checked again on the press
        allowed = _interaction()
        await _button(panel, "Reset").callback(allowed)
    confirm = allowed.response.edit_message.await_args.kwargs["view"]
    assert "every file" in view_text(confirm)
    fake.reset.assert_not_called()


@pytest.mark.asyncio
async def test_reset_asks_for_confirmation_first():
    ctx = _interaction()
    fake = MagicMock()
    fake.reset = AsyncMock()
    with patch("classes.workspace_manager.workspaces", return_value=fake):
        await _commands()["reset"].callback(ctx)
    fake.reset.assert_not_awaited()
    assert ctx.response.send_message.await_args.kwargs["ephemeral"] is True
    view = replied(ctx)
    assert [b.label for b in view.buttons] == ["Reset", "Cancel"]


@pytest.mark.asyncio
async def test_the_confirm_button_resets_and_only_for_the_person_who_asked():
    view = card.views(wc._confirm_card(user_id=99))[0]
    fake = MagicMock()
    fake.reset = AsyncMock(return_value="now")
    stranger = _interaction()
    stranger.user = SimpleNamespace(id=5)
    unmanaged = _interaction(manage=False)
    with patch("classes.workspace_manager.workspaces", return_value=fake):
        await _button(view, "Reset").callback(stranger)
        await _button(view, "Reset").callback(unmanaged)
        fake.reset.assert_not_awaited()
        ctx = _interaction()
        await _button(view, "Reset").callback(ctx)
    fake.reset.assert_awaited_once_with(7)
    assert "Workspace reset" in view_text(ctx.edit_original_response.await_args.kwargs["view"])
    # A second press that lands before the card changed resets nothing more.
    again = _interaction()
    with patch("classes.workspace_manager.workspaces", return_value=fake):
        await _button(view, "Reset").callback(again)
        await _button(view, "Cancel").callback(again)
    fake.reset.assert_awaited_once()
    again.response.edit_message.assert_not_awaited()
    assert "answered already" in replied_text(again)


@pytest.mark.asyncio
async def test_progress_stores_the_setting():
    ctx = _interaction(manage=False)  # same permission as before: none
    config, p = _config()
    with p, patch.object(wc, "configManager", return_value=config):
        await _commands()["progress"].callback(ctx, True)
    assert config.data["workspace_progress"] == "True"
    assert ctx.response.send_message.await_args.kwargs["ephemeral"] is True
    assert "Live progress on" in replied_text(ctx)

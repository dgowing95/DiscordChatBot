"""/workspace status, reset and progress (classes/workspace_commands.py).

Run from the repo root: `pytest core/tests/workspace_commands_tests.py`
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from classes import workspace_commands as wc


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


# ---------------------- status text ----------------------

def test_status_of_a_stopped_workspace_that_was_never_saved():
    text = wc.status_text({"warm": False, "last_snapshot_at": None})
    assert "Stopped" in text and "Not saved yet" in text


def test_status_of_a_workspace_waiting_after_a_restart():
    text = wc.status_text({"warm": False, "waiting": True, "last_snapshot_at": 1})
    assert "Waiting" in text and "Stopped" not in text


def test_status_of_a_running_busy_workspace():
    text = wc.status_text({"warm": True, "started_at": 1000, "busy": 2, "jobs": 4,
                           "last_snapshot_at": 1500, "snapshot_bytes": 3 * 1024 * 1024},
                          now=1000 + 600)
    assert "Running** for 10 min" in text and "2 task(s)" in text
    assert "Task folders: 4" in text
    assert "<t:1500:R>" in text and "3.0 MB" in text


def test_status_warns_when_the_workspace_is_too_big_to_save():
    text = wc.status_text({"warm": False, "last_snapshot_at": 1, "snapshot_bytes": 10,
                           "snapshot_too_large": True})
    assert "too big to save" in text and "/workspace reset" in text


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
    ctx = MagicMock()
    ctx.guild_id = 7
    ctx.user.id = 99
    ctx.permissions = SimpleNamespace(manage_guild=manage)
    ctx.response.send_message = AsyncMock()
    ctx.response.defer = AsyncMock()
    ctx.followup.send = AsyncMock()
    return ctx


def test_the_group_has_status_reset_and_progress():
    assert set(_commands()) == {"status", "reset", "progress"}


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["status", "reset"])
async def test_status_and_reset_need_manage_server(name):
    ctx = _interaction(manage=False)
    await _commands()[name].callback(ctx)
    text = ctx.response.send_message.await_args.args[0]
    assert "Manage Server" in text
    assert ctx.response.send_message.await_args.kwargs["ephemeral"] is True


@pytest.mark.asyncio
async def test_status_replies_ephemerally_with_the_workspaces_state():
    ctx = _interaction()
    fake = MagicMock()
    fake.status = AsyncMock(return_value={"warm": False, "last_snapshot_at": None})
    with patch("classes.workspace_manager.workspaces", return_value=fake):
        await _commands()["status"].callback(ctx)
    fake.status.assert_awaited_once_with(7)
    assert "Stopped" in ctx.followup.send.await_args.args[0]
    assert ctx.followup.send.await_args.kwargs["ephemeral"] is True


@pytest.mark.asyncio
async def test_reset_asks_for_confirmation_first():
    ctx = _interaction()
    fake = MagicMock()
    fake.reset = AsyncMock()
    with patch("classes.workspace_manager.workspaces", return_value=fake):
        await _commands()["reset"].callback(ctx)
    fake.reset.assert_not_awaited()
    kwargs = ctx.response.send_message.await_args.kwargs
    assert kwargs["ephemeral"] is True and isinstance(kwargs["view"], wc._ResetConfirm)


@pytest.mark.asyncio
async def test_the_confirm_button_resets_and_only_for_the_person_who_asked():
    view = wc._ResetConfirm(user_id=99)
    fake = MagicMock()
    fake.reset = AsyncMock(return_value="now")
    stranger = _interaction()
    stranger.user.id = 5
    with patch("classes.workspace_manager.workspaces", return_value=fake):
        await view.confirm.callback(stranger)
        fake.reset.assert_not_awaited()
        ctx = _interaction()
        ctx.response.edit_message = AsyncMock()
        ctx.edit_original_response = AsyncMock()
        await view.confirm.callback(ctx)
    fake.reset.assert_awaited_once_with(7)
    assert "was reset" in ctx.edit_original_response.await_args.kwargs["content"]


@pytest.mark.asyncio
async def test_progress_stores_the_setting():
    ctx = _interaction(manage=False)  # same permission as before: none
    config, p = _config()
    with p, patch.object(wc, "configManager", return_value=config):
        await _commands()["progress"].callback(ctx, True)
    assert config.data["workspace_progress"] == "True"
    assert ctx.response.send_message.await_args.kwargs["ephemeral"] is True

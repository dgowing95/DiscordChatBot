import asyncio
from types import SimpleNamespace

import pytest

from classes import automation_runner


def message(channel=4, content="Wordle today", bot=False, webhook=None):
    return SimpleNamespace(guild=SimpleNamespace(id=1), channel=SimpleNamespace(id=channel),
                           author=SimpleNamespace(bot=bot), content=content, id=99, webhook_id=webhook)


@pytest.mark.asyncio
async def test_rules_match_exact_channel_and_combine_in_creation_order(monkeypatch):
    rows = [dict(id="a", guild_id=1, channel_id=4, status="enabled", pattern="wordle", match_mode="word", action="first"),
            dict(id="b", guild_id=1, channel_id=4, status="enabled", pattern="today", match_mode="word", action="second"),
            dict(id="c", guild_id=1, channel_id=5, status="enabled", pattern="wordle", match_mode="word", action="wrong channel")]
    class Store:
        async def list(self, guild, kind): return rows
        async def claim_rule(self, row, message_id): return (row["id"], "busy", "token")
    monkeypatch.setattr(automation_runner, "AutomationStore", Store)
    queue = asyncio.Queue(maxsize=2)
    assert await automation_runner.match_message(message(), queue)
    job = queue.get_nowait()
    assert [r["id"] for r in job.records] == ["a", "b"]
    job.renewal_task.cancel()
    await asyncio.gather(job.renewal_task, return_exceptions=True)
    assert not await automation_runner.match_message(message(channel=6), queue)
    assert not await automation_runner.match_message(message(bot=True), queue)
    assert not await automation_runner.match_message(message(webhook=3), queue)


@pytest.mark.asyncio
async def test_full_queue_does_not_claim_rule(monkeypatch):
    class Store:
        async def list(self, guild, kind):
            return [dict(id="a", guild_id=1, channel_id=4, status="enabled", pattern="wordle", match_mode="word")]
        async def claim_rule(self, row, message_id):
            raise AssertionError("Must not consume cooldown")
    monkeypatch.setattr(automation_runner, "AutomationStore", Store)
    queue = asyncio.Queue(maxsize=1)
    queue.put_nowait("busy")
    assert not await automation_runner.match_message(message(), queue)


@pytest.mark.asyncio
async def test_automatic_run_uses_saved_action_and_origin_channel(monkeypatch):
    sent = []
    finished = []
    row = dict(id="r1", guild_id=1, channel_id=4, status="enabled", revision=1,
               editor_id=8, kind="rule", action="find answer")
    class Store:
        async def get(self, guild, ident): return row
        async def finish(self, record, outcome, detail=""): finished.append(outcome)
        async def release(self, claim): pass
    class Channel:
        id = 4
        def permissions_for(self, member):
            return SimpleNamespace(view_channel=True, send_messages=True)
        async def send(self, text): sent.append(text)
    channel = Channel()
    member = SimpleNamespace(id=8)
    guild = SimpleNamespace(get_member=lambda user_id: member)
    client = SimpleNamespace(get_channel=lambda channel_id: channel,
                             get_guild=lambda guild_id: guild,
                             user=SimpleNamespace(id=42))
    args = []
    class Handler:
        def __init__(self, messages, guild_id, original_message, **kw):
            args.append((messages, guild_id, original_message, kw))
            self.reasoning = ""
            self.sandbox_thread = None
        async def generate(self): return "Done"
    monkeypatch.setattr(automation_runner, "AutomationStore", Store)
    monkeypatch.setattr(automation_runner, "TextLLMHandler", Handler)
    trigger = message()
    trigger.author.display_name = "Alice"
    job = automation_runner.AutomationJob([row], [("key", "busy", "token")], trigger, 0)
    await automation_runner.execute(job, client)
    assert sent == ["Done"]
    assert finished == ["completed"]
    assert args[0][0][0]["content"].find("find answer") >= 0
    assert args[0][2] is None
    assert args[0][3]["automatic"] is True
    assert args[0][3]["channel"] is channel


@pytest.mark.asyncio
async def test_editor_who_left_suspends_the_entry(monkeypatch):
    import discord
    sent, updates, finished = [], [], []
    row = dict(id="s1", guild_id=1, channel_id=4, status="enabled", revision=3,
               editor_id=8, kind="schedule", action="find answer", next_run=1.0,
               timezone="Europe/London")
    class Store:
        async def get(self, guild, ident): return row
        async def update(self, guild, ident, actor, revision, **changes): updates.append(changes)
        async def finish(self, record, outcome, detail=""): finished.append((outcome, detail))
        async def release(self, claim): pass
    class Channel:
        id = 4
        async def send(self, text): sent.append(text)
    async def fetch_member(user_id):
        raise discord.NotFound(SimpleNamespace(status=404, reason="Not Found"), "Unknown Member")
    guild = SimpleNamespace(get_member=lambda user_id: None, fetch_member=fetch_member)
    client = SimpleNamespace(get_channel=lambda channel_id: Channel(),
                             get_guild=lambda guild_id: guild)
    monkeypatch.setattr(automation_runner, "AutomationStore", Store)
    job = automation_runner.AutomationJob([row], [("key", "busy", "token")], None, 0)
    await automation_runner.execute(job, client)
    assert updates == [{"status": "suspended"}]
    assert finished[0][0] == "suspended" and "no longer" in finished[0][1]

"""
Tests for what stops in text while the bot is in a voice call
(core/classes/voice_gate.py and its wiring in main.py / automation_runner),
and for the discord.py side of the voice bridge (classes/voice_bridge.py).

Run from the repo root:
    pytest core/tests/voice_gate_tests.py
"""
import asyncio
import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from classes import automation_runner, voice_bridge, voice_gate

GUILD = 42


def _import_main():
    if "main" in sys.modules:
        return sys.modules["main"]
    import main as m
    return m


def _message(content="hi", guild_id=GUILD, channel_id=77, msg_id=1, mentions=None, bot=False):
    msg = MagicMock()
    msg.id = msg_id
    msg.content = content
    msg.embeds = []
    msg.attachments = []
    msg.guild = MagicMock()
    msg.guild.id = guild_id
    msg.author = MagicMock()
    msg.author.id = 777
    msg.author.bot = bot
    msg.channel = MagicMock()
    msg.channel.id = channel_id
    msg.mentions = mentions or []
    msg.reply = AsyncMock()
    msg.webhook_id = None
    return msg


@pytest.fixture(autouse=True)
def _clean():
    voice_gate.reset()
    main = _import_main()
    main._voice_notice_at.clear()
    yield
    voice_gate.reset()


# ---------------------------------------------------------------------------
# voice_gate (pure)
# ---------------------------------------------------------------------------

def test_gate_round_trip():
    session = object()
    voice_gate.enter(GUILD, session)
    assert voice_gate.active(GUILD) and voice_gate.active(str(GUILD))
    assert voice_gate.session(GUILD) is session
    voice_gate.leave(GUILD, object())  # someone else's session: no effect
    assert voice_gate.active(GUILD)
    voice_gate.leave(GUILD, session)
    assert not voice_gate.active(GUILD)
    assert not voice_gate.active(None) and not voice_gate.active("x")


# ---------------------------------------------------------------------------
# main.on_message / process_messages
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_messages_in_a_voice_guild_are_not_queued():
    main = _import_main()
    bot = MagicMock()
    queue = asyncio.Queue(maxsize=5)
    voice_gate.enter(GUILD, object())
    with patch.object(main, "client", bot), patch.object(main, "message_queue", queue), \
         patch.object(main.automation_runner, "match_message", AsyncMock(return_value=False)) as rules:
        await main.on_message(_message(mentions=[bot.user]))
    assert queue.empty()
    rules.assert_not_awaited()  # rules are text work too


@pytest.mark.asyncio
async def test_a_mention_gets_one_fixed_notice_per_minute():
    main = _import_main()
    bot = MagicMock()
    voice_gate.enter(GUILD, object())
    with patch.object(main, "client", bot), patch.object(main, "message_queue", asyncio.Queue()):
        first = _message(mentions=[bot.user], msg_id=1)
        second = _message(mentions=[bot.user], msg_id=2)
        plain = _message(msg_id=3, channel_id=78)
        for msg in (first, second, plain):
            await main.on_message(msg)
    first.reply.assert_awaited_once()
    assert "voice call" in first.reply.await_args.args[0]
    second.reply.assert_not_awaited()
    plain.reply.assert_not_awaited()


@pytest.mark.asyncio
async def test_other_guilds_are_unaffected():
    main = _import_main()
    bot = MagicMock()
    queue = asyncio.Queue(maxsize=5)
    voice_gate.enter(GUILD + 1, object())
    with patch.object(main, "client", bot), patch.object(main, "message_queue", queue), \
         patch.object(main.automation_runner, "match_message", AsyncMock(return_value=False)):
        await main.on_message(_message(mentions=[bot.user]))
    assert queue.qsize() == 1


@pytest.mark.asyncio
async def test_sandbox_steering_still_reaches_a_running_sandbox():
    main = _import_main()
    voice_gate.enter(GUILD, object())
    with patch.object(main.sandbox_thread_inbox, "is_run_active", return_value=True), \
         patch.object(main, "route_to_sandbox", AsyncMock()) as route:
        await main.on_message(_message(content="make it blue"))
    route.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_message_queued_before_the_call_is_dropped():
    main = _import_main()
    handled = []

    class Handler:
        def __init__(self, message, client):
            handled.append(message)

        async def handle_message(self):
            pass

    queue = asyncio.Queue()
    queue.put_nowait(_message())
    voice_gate.enter(GUILD, object())
    with patch.object(main, "message_queue", queue), patch.object(main, "MessageHandler", Handler):
        worker = asyncio.create_task(main.process_messages())
        await asyncio.wait_for(queue.join(), 2)
        worker.cancel()
    assert handled == []


@pytest.mark.asyncio
async def test_an_automation_job_queued_before_the_call_is_given_back():
    main = _import_main()
    job = automation_runner.AutomationJob([{"guild_id": GUILD, "kind": "rule"}], [("k", "l", "t")])
    queue = asyncio.Queue()
    queue.put_nowait(job)
    voice_gate.enter(GUILD, object())
    with patch.object(main, "message_queue", queue), \
         patch.object(main.automation_runner, "discard", AsyncMock()) as discard, \
         patch.object(main.automation_runner, "execute", AsyncMock()) as execute:
        worker = asyncio.create_task(main.process_messages())
        await asyncio.wait_for(queue.join(), 2)
        worker.cancel()
    discard.assert_awaited_once_with(job)
    execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_discard_releases_the_claims():
    renewal = MagicMock()
    job = automation_runner.AutomationJob([{"guild_id": GUILD, "kind": "rule"}],
                                          [("k1", "l1", "t1"), ("k2", "l2", "t2")], renewal_task=renewal)
    store = MagicMock()
    store.cancel_claim = AsyncMock()
    with patch.object(automation_runner, "AutomationStore", return_value=store):
        await automation_runner.discard(job)
    renewal.cancel.assert_called_once()
    assert [c.kwargs["rule"] for c in store.cancel_claim.await_args_list] == [True, True]


@pytest.mark.asyncio
async def test_due_schedules_wait_while_their_guild_is_in_a_call(monkeypatch):
    monkeypatch.setenv("AUTOMATIONS_ENABLED", "1")
    store = MagicMock()
    store.purge_terminal = AsyncMock()
    store.due = AsyncMock(return_value=[f"{GUILD}:abc"])
    store.get = AsyncMock()
    voice_gate.enter(GUILD, object())
    with patch.object(automation_runner, "AutomationStore", return_value=store):
        await automation_runner.poll_schedules(asyncio.Queue())
    store.get.assert_not_awaited()  # not claimed: still due once the call ends


@pytest.mark.asyncio
async def test_generate_image_command_refuses_during_a_call(monkeypatch):
    main = _import_main()
    monkeypatch.setenv("IMAGE_GEN_ENABLED", "1")
    commands = {}

    class Tree:
        def __init__(self, *a, **k):
            pass

        def command(self, name, description=""):
            def register(fn):
                commands[name] = fn
                return fn
            return register

        def add_command(self, command):
            pass

        async def sync(self):
            return []

    monkeypatch.setattr(main.discord.app_commands, "CommandTree", Tree)
    await main.register_commands()
    voice_gate.enter(GUILD, object())
    ctx = MagicMock()
    ctx.guild_id = GUILD
    ctx.response.send_message = AsyncMock()
    ctx.response.defer = AsyncMock()
    await commands["generate_image"](ctx, "a cat")
    ctx.response.send_message.assert_awaited_once()
    ctx.response.defer.assert_not_awaited()


@pytest.mark.asyncio
async def test_dev_inject_is_off_by_default(monkeypatch):
    main = _import_main()
    monkeypatch.delenv("VOICE_DEV_INJECT", raising=False)
    assert await main.voice_dev_inject(_message(content="!voice_say hey sparky hi")) is False


# ---------------------------------------------------------------------------
# voice_bridge: discord.py's side of the handshake
# ---------------------------------------------------------------------------

class Bridge:
    def __init__(self, ok=True):
        self.sent = []
        self.ok = ok
        self.protocols = {}

    async def send(self, message):
        self.sent.append(message)
        return self.ok


def _protocol(bridge):
    guild = MagicMock()
    guild.id = GUILD
    guild.me = None
    channel = MagicMock()
    channel.id = 500
    channel.guild = guild
    channel._get_voice_client_key.return_value = (GUILD, "guild_id")
    client = MagicMock()
    client._connection = MagicMock()
    with patch.object(voice_bridge, "get_bridge", return_value=bridge):
        protocol = voice_bridge.SidecarVoiceProtocol(client, channel)
    return protocol, client


@pytest.mark.asyncio
async def test_connect_waits_for_the_sidecar():
    bridge = Bridge()
    protocol, client = _protocol(bridge)
    connecting = asyncio.create_task(protocol.connect(timeout=1, reconnect=True))
    await asyncio.sleep(0)
    assert bridge.sent[0] == {"type": "join", "guild_id": str(GUILD), "channel_id": "500"}
    protocol.sidecar_ready()
    await connecting
    client._connection._remove_voice_client.assert_not_called()


@pytest.mark.asyncio
async def test_a_failed_connect_cleans_up_after_itself():
    # Otherwise guild.voice_client stays set and every later join says
    # "Already connected".
    protocol, client = _protocol(Bridge(ok=False))
    with pytest.raises(voice_bridge.VoiceUnavailable):
        await protocol.connect(timeout=1, reconnect=True)
    client._connection._remove_voice_client.assert_called_once_with(GUILD)


@pytest.mark.asyncio
async def test_a_sidecar_that_gives_up_fails_the_connect():
    bridge = Bridge()
    protocol, client = _protocol(bridge)
    connecting = asyncio.create_task(protocol.connect(timeout=1, reconnect=True))
    await asyncio.sleep(0)
    protocol.sidecar_gone("disconnected")
    with pytest.raises(voice_bridge.VoiceUnavailable):
        await connecting
    client._connection._remove_voice_client.assert_called_once()


@pytest.mark.asyncio
async def test_a_timeout_is_left_to_discord_py():
    # discord.py calls disconnect(force=True) on TimeoutError, which cleans up.
    bridge = Bridge()
    protocol, client = _protocol(bridge)
    with pytest.raises(asyncio.TimeoutError):
        await protocol.connect(timeout=0.01, reconnect=True)
    with patch.object(voice_bridge, "LEAVE_TIMEOUT", 0.01):
        await protocol.disconnect(force=True)
    client._connection._remove_voice_client.assert_called_once()


@pytest.mark.asyncio
async def test_gateway_events_are_forwarded_and_a_kick_cleans_up():
    bridge = Bridge()
    protocol, client = _protocol(bridge)
    await protocol.on_voice_server_update({"token": "t", "endpoint": "e"})
    await protocol.on_voice_state_update({"channel_id": None, "session_id": "s"})
    assert [m["t"] for m in bridge.sent] == ["VOICE_SERVER_UPDATE", "VOICE_STATE_UPDATE"]
    client._connection._remove_voice_client.assert_called_once()


@pytest.mark.asyncio
async def test_sidecar_payloads_go_out_on_discord_py_gateway():
    guild = MagicMock()
    guild.change_voice_state = AsyncMock()
    client = MagicMock()
    client.get_guild.return_value = guild
    bridge = voice_bridge.VoiceBridge("ws://x", client)
    await bridge._dispatch(json.dumps({"type": "payload", "guild_id": str(GUILD),
                                       "d": {"channel_id": "500", "self_mute": False, "self_deaf": False}}))
    kwargs = guild.change_voice_state.await_args.kwargs
    assert kwargs["channel"].id == 500 and kwargs["self_deaf"] is False


@pytest.mark.asyncio
async def test_an_utterance_does_not_block_the_read_loop():
    # "hey sparky, leave" waits for the sidecar's `disconnected`, which comes
    # in through this same loop.
    gate_open = asyncio.Event()
    session = SimpleNamespace(on_utterance=AsyncMock(side_effect=lambda m: gate_open.wait()))
    voice_gate.enter(GUILD, session)
    bridge = voice_bridge.VoiceBridge("ws://x", MagicMock())
    await asyncio.wait_for(bridge._dispatch(json.dumps(
        {"type": "utterance", "guild_id": str(GUILD), "user_id": "7", "text": "hi"})), 0.5)
    gate_open.set()
    await asyncio.sleep(0)
    session.on_utterance.assert_awaited_once()

"""
Tests for core/classes/voice_session.py: what a call does with each
utterance, the order things are spoken in, and how a call ends.

The voice sidecar is a FakeBridge that records what it is sent, and the LLM
is a FakeHandler whose streamed answer each test scripts.

Run from the repo root:
    pytest core/tests/voice_session_tests.py
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from classes import voice_gate, voice_session
from classes import text_llm_handler

GUILD = 42


class FakeBridge:
    def __init__(self):
        self.sent = []
        self.connected = True

    async def send(self, message):
        self.sent.append(message)
        return True

    def spoken(self):
        return [m["text"] for m in self.sent if m["type"] == "speak"]

    def kinds(self):
        return [m["type"] for m in self.sent]


def _member(member_id, name="Ana", bot=False):
    return SimpleNamespace(id=member_id, display_name=name, bot=bot)


def _session(monkeypatch, members=None, **env):
    monkeypatch.setenv("VOICE_ENABLED", "1")
    for name, value in env.items():
        monkeypatch.setenv(name, str(value))
    guild = MagicMock()
    guild.id = GUILD
    guild.me = _member(1, "Sparky", bot=True)
    guild.voice_client = None
    voice_channel = MagicMock()
    voice_channel.id = 500
    voice_channel.guild = guild
    voice_channel.members = members if members is not None else [_member(7), guild.me]
    guild.get_member = lambda i: next((m for m in voice_channel.members if m.id == i), None)
    bridge = FakeBridge()
    session = voice_session.VoiceSession(MagicMock(), bridge, voice_channel, _member(7))
    session.custom_wake_phrase = "hey sparky"
    session.text_channel = MagicMock()
    session.text_channel.send = AsyncMock()
    voice_gate.enter(GUILD, session)
    return session, bridge


@pytest.fixture(autouse=True)
def _clean_gate():
    voice_gate.reset()
    yield
    voice_gate.reset()


class FakeHandler:
    """Plays a script of ("text", delta) / ("model",) / ("tool", name, args)
    steps through the generate_streamed callbacks."""

    script = []
    answer = "The answer."
    instances = []

    def __init__(self, messages, guild_id, original_message, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        FakeHandler.instances.append(self)

    async def generate_streamed(self, on_text, on_model_start=None, on_tool=None, hooks=None):
        for step in self.script:
            if step[0] == "text":
                await on_text(step[1])
            elif step[0] == "model":
                await on_model_start()
            elif step[0] == "tool":
                await on_tool(step[1], step[2])
            await asyncio.sleep(0)
        return self.answer

    async def prefill(self):
        FakeHandler.prefills += 1


FakeHandler.prefills = 0


@pytest.fixture
def fake_llm():
    FakeHandler.instances = []
    FakeHandler.prefills = 0
    with patch.object(text_llm_handler, "TextLLMHandler", FakeHandler):
        yield FakeHandler


async def _settle(session):
    """Let queued turns run to completion."""
    for _ in range(50):
        await asyncio.sleep(0)
        if session.turns.empty() and not session.turn_running:
            break
    await asyncio.sleep(0.01)


async def _run(session):
    session._worker = asyncio.create_task(session._run_turns())


@pytest.mark.asyncio
async def test_chatter_is_heard_but_not_answered(monkeypatch, fake_llm):
    session, bridge = _session(monkeypatch)
    await _run(session)
    await session.heard(7, "Ana", "I think we should order pizza tonight")
    await _settle(session)
    assert bridge.spoken() == []
    assert fake_llm.instances == []
    assert session.history.messages() == [{"role": "user", "content": "[Ana]: I think we should order pizza tonight"}]
    session._worker.cancel()


@pytest.mark.asyncio
async def test_a_request_is_answered_sentence_by_sentence(monkeypatch, fake_llm):
    session, bridge = _session(monkeypatch)
    fake_llm.script = [("model",), ("text", "Twelve times twelve "), ("text", "is 144. Easy! "), ("text", "Anything else")]
    fake_llm.answer = "Twelve times twelve is 144. Easy! Anything else"
    await _run(session)
    await session.heard(7, "Ana", "Hey Sparky, what is twelve times twelve?")
    await _settle(session)
    assert bridge.spoken() == ["Twelve times twelve is 144.", "Easy!", "Anything else"]
    handler = fake_llm.instances[0]
    # The turn sees the whole conversation, the request included, and runs
    # as the speaker in the session's text thread.
    assert handler.messages[-1]["content"] == "[Ana]: Hey Sparky, what is twelve times twelve?"
    assert handler.kwargs["actor_id"] == 7 and handler.kwargs["voice"] is True
    assert handler.kwargs["channel"] is session.text_channel
    assert session.history.messages()[-1] == {"role": "assistant", "content": fake_llm.answer}
    session._worker.cancel()


@pytest.mark.asyncio
async def test_the_models_own_line_before_a_tool_is_spoken_first(monkeypatch, fake_llm):
    session, bridge = _session(monkeypatch)
    fake_llm.script = [("model",), ("text", "Let me look that up."), ("tool", "web_search", '{"q": "f1"}'),
                       ("model",), ("text", "Verstappen won it.")]
    fake_llm.answer = "Verstappen won it."
    hold_on = AsyncMock(return_value="should not be used")
    with patch.object(voice_session._hold_on_llm, "complete", hold_on):
        await _run(session)
        await session.heard(7, "Ana", "hey sparky who won the race")
        await _settle(session)
    assert bridge.spoken() == ["Let me look that up.", "Verstappen won it."]
    hold_on.assert_not_awaited()
    session._worker.cancel()


@pytest.mark.asyncio
async def test_a_silent_tool_call_gets_a_written_line_before_the_answer(monkeypatch, fake_llm):
    session, bridge = _session(monkeypatch)
    fake_llm.script = [("model",), ("tool", "web_search", '{"search_request": "f1 winner"}'),
                       ("model",), ("text", "Verstappen won it.")]
    fake_llm.answer = "Verstappen won it."

    async def slow_line(messages, **kwargs):
        # Slower than the answer: the answer must still wait its turn.
        await asyncio.sleep(0.05)
        assert "f1 winner" in messages[0]["content"]
        return "Hang on, checking the results!"

    with patch.object(voice_session._hold_on_llm, "complete", side_effect=slow_line), \
         patch.object(voice_session.configManager, "get_setting", AsyncMock(return_value="a pirate")):
        await _run(session)
        await session.heard(7, "Ana", "hey sparky who won the race")
        await _settle(session)
        await asyncio.sleep(0.1)
    assert bridge.spoken() == ["Hang on, checking the results!", "Verstappen won it."]
    session._worker.cancel()


@pytest.mark.asyncio
async def test_a_failed_hold_on_line_falls_back_to_a_plain_one(monkeypatch, fake_llm):
    session, bridge = _session(monkeypatch)
    fake_llm.script = [("model",), ("tool", "generate_image", '{"prompt": "a cat"}'), ("model",),
                       ("text", "Done, it's in the thread.")]
    fake_llm.answer = "Done, it's in the thread."
    with patch.object(voice_session._hold_on_llm, "complete", AsyncMock(side_effect=RuntimeError("down"))), \
         patch.object(voice_session.configManager, "get_setting", AsyncMock(return_value=None)):
        await _run(session)
        await session.heard(7, "Ana", "hey sparky draw a cat")
        await _settle(session)
        await asyncio.sleep(0.05)
    assert bridge.spoken() == ["Hold on, I'm making a picture.", "Done, it's in the thread."]
    session._worker.cancel()


@pytest.mark.asyncio
async def test_the_greeting_is_written_by_the_model_then_says_the_wake_phrase(monkeypatch):
    session, bridge = _session(monkeypatch)
    write = AsyncMock(return_value="Ahoy, **Ana**! Ready for adventure?")
    with patch.object(voice_session._greeting_llm, "complete", write),          patch.object(voice_session.configManager, "get_setting", AsyncMock(return_value="a pirate")):
        await session._greet()
    assert bridge.spoken() == ["Ahoy, Ana! Ready for adventure?", "Say hey sparky when you need me."]
    prompt = write.call_args.args[0][0]["content"]
    assert "a pirate, called Sparky" in prompt and "with Ana" in prompt


@pytest.mark.asyncio
async def test_a_failed_greeting_still_says_hello(monkeypatch):
    session, bridge = _session(monkeypatch)
    with patch.object(voice_session._greeting_llm, "complete", AsyncMock(side_effect=RuntimeError("down"))),          patch.object(voice_session.configManager, "get_setting", AsyncMock(return_value=None)):
        await session._greet()
    assert bridge.spoken() == ["Hi everyone!", "Say hey sparky when you need me."]


@pytest.mark.asyncio
async def test_an_error_is_said_out_loud(monkeypatch, fake_llm):
    session, bridge = _session(monkeypatch)
    fake_llm.script = []
    fake_llm.answer = "Error"
    await _run(session)
    await session.heard(7, "Ana", "hey sparky do something")
    await _settle(session)
    assert bridge.spoken() == ["Sorry, something went wrong with that one."]
    assert session.history.messages()[-1]["role"] == "user"  # no fake answer recorded
    session._worker.cancel()


@pytest.mark.asyncio
async def test_the_wake_phrase_alone_chimes_and_opens_a_follow_up(monkeypatch, fake_llm):
    session, bridge = _session(monkeypatch, VOICE_FOLLOWUP_SECONDS=8)
    session.settings["followup_seconds"] = 8
    fake_llm.script = [("text", "Why did the cat sit on the computer?")]
    await _run(session)
    await session.heard(7, "Ana", "Hey Sparky.")
    assert bridge.kinds() == ["chime"]
    await session.heard(8, "Bo", "unrelated chatter from someone else")
    await session.heard(7, "Ana", "tell me a joke about cats")
    await _settle(session)
    assert len(fake_llm.instances) == 1
    assert fake_llm.instances[0].messages[-1]["content"] == "[Ana]: tell me a joke about cats"
    session._worker.cancel()


@pytest.mark.asyncio
async def test_a_request_chimes_before_the_answer(monkeypatch, fake_llm):
    session, bridge = _session(monkeypatch)
    fake_llm.script = [("text", "It's sunny.")]
    fake_llm.answer = "It's sunny."
    await _run(session)
    await session.heard(7, "Ana", "hey sparky what's the weather")
    await _settle(session)
    assert bridge.kinds() == ["chime", "speak"]
    session._worker.cancel()


@pytest.mark.asyncio
async def test_renaming_the_bot_mid_call_changes_the_default_phrase(monkeypatch, fake_llm):
    session, bridge = _session(monkeypatch)
    session.custom_wake_phrase = ""
    await _run(session)
    await session.heard(7, "Ana", "hey sparky, hello")
    await _settle(session)
    session.guild.me.display_name = "Nugget"
    await session.heard(7, "Ana", "hey sparky, hello again")
    await session.heard(7, "Ana", "hey nugget, hello")
    await _settle(session)
    assert len(fake_llm.instances) == 2
    assert [i.kwargs["bot_name"] for i in fake_llm.instances] == ["Sparky", "Nugget"]
    session._worker.cancel()


@pytest.mark.asyncio
async def test_the_follow_up_window_closes(monkeypatch, fake_llm):
    session, bridge = _session(monkeypatch)
    session.settings["followup_seconds"] = 0
    await _run(session)
    await session.heard(7, "Ana", "Hey Sparky.")
    await session.heard(7, "Ana", "tell me a joke")
    await _settle(session)
    assert fake_llm.instances == []
    session._worker.cancel()


@pytest.mark.asyncio
async def test_stop_stops_speaking_without_the_llm(monkeypatch, fake_llm):
    session, bridge = _session(monkeypatch)
    await session.heard(7, "Ana", "hey sparky, stop")
    assert bridge.kinds() == ["stop"]
    assert fake_llm.instances == []


@pytest.mark.asyncio
async def test_leave_ends_the_call_without_the_llm(monkeypatch, fake_llm):
    session, bridge = _session(monkeypatch)
    voice_client = MagicMock()
    voice_client.disconnect = AsyncMock()
    session.guild.voice_client = voice_client
    await session.heard(7, "Ana", "Hey Sparky, please leave the call")
    assert fake_llm.instances == []
    assert not voice_gate.active(GUILD)
    voice_client.disconnect.assert_awaited_once()
    session.text_channel.send.assert_awaited()
    # Ending twice is harmless.
    await session.end("again")
    voice_client.disconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_full_turn_queue_says_so_once(monkeypatch, fake_llm):
    session, bridge = _session(monkeypatch)
    # No worker: requests pile up.
    for _ in range(4):
        await session.heard(7, "Ana", "hey sparky what's up")
    assert session.turns.full()
    assert bridge.spoken() == ["One moment, I'm still on the last one."]


@pytest.mark.asyncio
async def test_bots_in_the_call_are_not_listened_to(monkeypatch, fake_llm):
    other_bot = _member(9, "Music", bot=True)
    session, bridge = _session(monkeypatch, members=[_member(7), other_bot])
    await session.on_utterance({"user_id": "9", "text": "hey sparky play a song", "stt_seconds": 0.2})
    assert session.history.messages() == []


@pytest.mark.asyncio
async def test_the_call_ends_when_everyone_has_left(monkeypatch, fake_llm):
    session, bridge = _session(monkeypatch, members=[])
    session.settings["idle_leave_seconds"] = 0.01
    session.members_changed()
    await asyncio.sleep(0.05)
    assert session.ending and not voice_gate.active(GUILD)


@pytest.mark.asyncio
async def test_someone_coming_back_keeps_the_call(monkeypatch, fake_llm):
    session, bridge = _session(monkeypatch, members=[])
    session.settings["idle_leave_seconds"] = 0.05
    session.members_changed()
    session.voice_channel.members = [_member(7)]
    session.members_changed()
    await asyncio.sleep(0.1)
    assert not session.ending and voice_gate.active(GUILD)


@pytest.mark.asyncio
async def test_a_leave_tool_call_leaves_after_the_turn(monkeypatch, fake_llm):
    session, bridge = _session(monkeypatch)
    session.guild.voice_client = None
    fake_llm.script = [("text", "Bye everyone!")]
    fake_llm.answer = "Bye everyone!"

    original = FakeHandler.generate_streamed

    async def leaving(self, *args, **kwargs):
        session.leave_after_turn = True  # what leave_voice_channel does
        return await original(self, *args, **kwargs)

    with patch.object(FakeHandler, "generate_streamed", leaving), \
         patch.object(voice_session, "GOODBYE_WAIT_SECONDS", 0.01):
        await _run(session)
        await session.heard(7, "Ana", "hey sparky we are all done here, thanks")
        await asyncio.sleep(0.1)
    assert bridge.spoken() == ["Bye everyone!"]
    assert session.ending and not voice_gate.active(GUILD)


@pytest.mark.asyncio
async def test_an_old_session_cannot_clear_a_newer_one(monkeypatch, fake_llm):
    old, _ = _session(monkeypatch)
    new, _ = _session(monkeypatch)
    old.guild.voice_client = None
    await old.end()
    assert voice_gate.session(GUILD) is new


@pytest.mark.asyncio
async def test_chatter_warms_the_cache_but_not_during_a_turn(monkeypatch, fake_llm):
    session, bridge = _session(monkeypatch)
    session.settings["prefill_min_seconds"] = 0
    await session.heard(7, "Ana", "some chatter")
    await asyncio.sleep(0.4)
    assert fake_llm.prefills == 1
    session.turn_running = True
    await session.heard(7, "Ana", "more chatter")
    await asyncio.sleep(0.4)
    assert fake_llm.prefills == 1


@pytest.mark.asyncio
async def test_start_refuses_when_voice_is_off(monkeypatch):
    monkeypatch.setenv("VOICE_ENABLED", "0")
    with pytest.raises(voice_session.VoiceError, match="switched off"):
        await voice_session.start(MagicMock(), MagicMock(), MagicMock(), MagicMock())


@pytest.mark.asyncio
async def test_start_refuses_a_second_call(monkeypatch):
    session, _ = _session(monkeypatch)
    other = MagicMock()
    other.id = 501
    other.guild = session.guild
    with patch.object(voice_session, "get_bridge", return_value=FakeBridge()):
        with pytest.raises(voice_session.VoiceError, match="already in a call"):
            await voice_session.start(MagicMock(), other, MagicMock(), MagicMock())


@pytest.mark.asyncio
async def test_a_failed_join_leaves_the_gate(monkeypatch):
    monkeypatch.setenv("VOICE_ENABLED", "1")
    guild = MagicMock()
    guild.id = GUILD
    guild.voice_client = None
    channel = MagicMock()
    channel.guild = guild
    channel.permissions_for.return_value = SimpleNamespace(connect=True, speak=True)
    channel.connect = AsyncMock(side_effect=asyncio.TimeoutError())
    with patch.object(voice_session, "get_bridge", return_value=FakeBridge()), \
         patch.object(voice_session, "_session_channel", AsyncMock(return_value=MagicMock())):
        with pytest.raises(voice_session.VoiceError, match="timed out"):
            await voice_session.start(MagicMock(), channel, MagicMock(), _member(7))
    assert not voice_gate.active(GUILD)


def test_main_argument():
    assert voice_session._main_argument('{"search_request": "f1 winner"}') == "f1 winner"
    assert voice_session._main_argument("not json") == ""
    assert voice_session._main_argument("[1, 2]") == ""

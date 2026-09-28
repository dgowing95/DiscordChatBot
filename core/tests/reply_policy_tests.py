import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from classes import reply_policy as rp
from classes import message_queue as mq
from message_handler_tests import (
    _generate_handler, _handled, _handler, _llm_message, _reasoning_llm,
    _run_result, _run_with_llm,
)


@pytest.fixture(autouse=True)
def isolated_policy(monkeypatch):
    monkeypatch.setattr(rp, "policy", rp.ReplyPolicy())
    monkeypatch.setenv("DOUBLE_REPLY_CHANCE", "1")
    monkeypatch.setenv("DOUBLE_REPLY_COOLDOWN_REPLIES", "10")
    monkeypatch.setattr(mq, "_channel_locks", {})


@pytest.mark.parametrize("raw,expected", [
    ("0", 0), ("1", 1), ("0.25", 0.25), ("nan", .08), ("inf", .08),
    ("-1", .08), ("1.1", .08), ("bad", .08), ("", .08),
])
def test_chance_validation(monkeypatch, raw, expected):
    monkeypatch.setenv("DOUBLE_REPLY_CHANCE", raw)
    assert rp.double_reply_chance() == expected


@pytest.mark.parametrize("raw,expected", [
    ("0", 0), ("3", 3), ("-1", 10), ("1.5", 10), ("bad", 10), ("", 10),
])
def test_cooldown_validation(monkeypatch, raw, expected):
    monkeypatch.setenv("DOUBLE_REPLY_COOLDOWN_REPLIES", raw)
    assert rp.cooldown_replies() == expected


def test_defaults_and_probability(monkeypatch):
    monkeypatch.delenv("DOUBLE_REPLY_CHANCE")
    monkeypatch.delenv("DOUBLE_REPLY_COOLDOWN_REPLIES")
    assert rp.cooldown_replies() == 10
    with patch.object(rp.random, "random", return_value=.079):
        assert rp.policy.eligible(1)
    with patch.object(rp.random, "random", return_value=.08):
        assert not rp.policy.eligible(1)
    monkeypatch.setenv("DOUBLE_REPLY_CHANCE", "0")
    with patch.object(rp.random, "random", return_value=0):
        assert not rp.policy.eligible(1)


def test_cooldown_and_channel_isolation():
    rp.policy.started_pair(1)
    assert rp.policy.eligible(2)
    for _ in range(10):
        assert not rp.policy.eligible(1)
        rp.policy.sent_single(1)
    assert rp.policy.eligible(1)


def test_state_is_bounded_and_recent_channels_survive():
    policy = rp.ReplyPolicy(max_channels=2)
    policy.started_pair(1)
    policy.started_pair(2)
    assert not policy.available(1)
    policy.available(3)
    assert not policy.available(1)
    assert policy.available(2)  # oldest entry was evicted


@pytest.mark.parametrize("text,allowed,expected", [
    ("Damn\n<message_break>\nYeah it is", True, ["Damn", "Yeah it is"]),
    ("Damn\n<message_break>\nYeah it is", False, ["Damn\n\nYeah it is"]),
    ("first\nsecond\n\nthird", True, ["first\nsecond\n\nthird"]),
    ("<message_break>\nhello", True, ["hello"]),
    ("hello\n<message_break>", True, ["hello"]),
    ("a\n<message_break>\nb\n<message_break>\nc", True, ["a\n\nb\n\nc"]),
    ("x" * 301 + "\n<message_break>\ny", True, ["x" * 301 + "\n\ny"]),
    ("x" * 300 + "\n<message_break>\ny", True, ["x" * 300, "y"]),
    ("inline <message_break> example", True, ["inline <message_break> example"]),
    ("```\n<message_break>\n```\n<message_break>\nend", True,
     ["```\n<message_break>\n```\n\nend"]),
    ("~~~~python\n<message_break>\n~~~~\n<message_break>\nend", True,
     ["~~~~python\n<message_break>\n~~~~\n\nend"]),
    ("```python\n<message_break>", True, ["```python\n<message_break>"]),
    ("", True, [""]),
])
def test_split_and_safe_fallback(text, allowed, expected):
    assert rp.reply_parts(text, allowed=allowed) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("allowed", [True, False])
async def test_generate_instruction_preserves_personality_and_history(allowed):
    result = _run_result("Damn\n<message_break>\nYeah it is")
    handler = _generate_handler(result)
    handler.allow_double_reply = allowed
    original = list(handler.messages)
    with patch("classes.text_llm_handler.Runner.run", AsyncMock(return_value=result)) as run, \
         patch("classes.text_llm_handler.get_current_datetime", AsyncMock(return_value="now")):
        assert await handler.generate() == result.final_output
    assert handler.system == "Answer as if you are a bot."
    assert handler.messages == original
    submitted = run.call_args.args[1]
    assert submitted[:len(original)] == original
    assert submitted[len(original)]["content"] == "(Current datetime: now)"
    assert (rp.INSTRUCTION in [m["content"] for m in submitted]) == allowed


@pytest.mark.asyncio
async def test_pair_plain_sends_pause_reasoning_and_cooldown():
    sends = []
    with patch("classes.message_handler.asyncio.sleep", AsyncMock()) as sleep, \
         patch("classes.message_handler.SHOW_THINKING", True):
        handler = await _run_with_llm(
            _reasoning_llm("thought", "Damn\n<message_break>\nYeah it is"), 900, sends)
    assert sends[:2] == ["Damn", "Yeah it is"]
    assert sends[2].startswith("-# Reasoning")
    assert "thought" in sends[3]
    assert all(not call.kwargs for call in handler.message.channel.send.await_args_list)
    sleep.assert_awaited_once()
    assert .8 <= sleep.await_args.args[0] <= 1.5
    assert rp.policy._remaining[900] == 10


@pytest.mark.asyncio
async def test_concurrent_generations_recheck_cooldown_and_serialize_pair():
    ready = asyncio.Event()
    started = []
    sends = []

    class LLM:
        def __init__(self, messages, guild_id, original_message, client=None):
            self.mid = original_message.id
            self.reasoning = ""

        async def generate(self):
            assert self.allow_double_reply
            started.append(self.mid)
            if len(started) == 2:
                ready.set()
            await ready.wait()
            return f"first {self.mid}\n<message_break>\nsecond {self.mid}"

    handlers = []
    for mid in (1, 2):
        handler = _handled(_handler(), _llm_message(mid, 901),
                           lambda h: setattr(h, "messages", []))
        handler.message.channel.send = AsyncMock(side_effect=lambda s: sends.append(s))
        handlers.append(handler)
    real_sleep = asyncio.sleep

    async def yield_during_pause(seconds):
        assert mq.get_channel_lock(901).locked()
        await real_sleep(0)

    with patch("classes.message_handler.TextLLMHandler", LLM), \
         patch("classes.message_handler.asyncio.sleep", yield_during_pause):
        await asyncio.gather(*(h.handle_message() for h in handlers))
    assert len(sends) == 3
    pair_id = sends[0][-1]
    assert sends[1] == f"second {pair_id}"
    assert "\n\n" in sends[2]
    assert rp.policy._remaining[901] == 9


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_at,remaining", [(1, 0), (2, 10)])
async def test_send_failure_never_replays_pair(fail_at, remaining):
    handler = _handled(_handler(), _llm_message(1, 902),
                       lambda h: setattr(h, "messages", []))
    effects = [None] * (fail_at - 1) + [RuntimeError("send failed")]
    handler.message.channel.send = AsyncMock(side_effect=effects)
    with patch("classes.message_handler.TextLLMHandler", _reasoning_llm("", "a\n<message_break>\nb")), \
         patch("classes.message_handler.asyncio.sleep", AsyncMock()):
        with pytest.raises(RuntimeError, match="send failed"):
            await handler.handle_message()
    assert handler.message.channel.send.await_count == fail_at
    assert rp.policy._remaining[902] == remaining


@pytest.mark.asyncio
async def test_sandbox_combines_pair_in_thread():
    thread = MagicMock(id=904, send=AsyncMock())
    base = _reasoning_llm("", "result\n<message_break>\ndetails")

    class LLM(base):
        sandbox_thread = thread

    sends = []
    await _run_with_llm(LLM, 903, sends)
    assert sends == []
    thread.send.assert_awaited_once_with("result\n\ndetails")


@pytest.mark.asyncio
async def test_disabled_combines_unsolicited_pair(monkeypatch):
    monkeypatch.setenv("DOUBLE_REPLY_CHANCE", "0")
    sends = []
    await _run_with_llm(_reasoning_llm("", "a\n<message_break>\nb"), 905, sends)
    assert sends == ["a\n\nb"]


@pytest.mark.asyncio
async def test_long_reply_counts_once_and_error_does_not_count():
    rp.policy.started_pair(906)
    with patch("classes.message_handler.asyncio.sleep", AsyncMock()):
        await _run_with_llm(_reasoning_llm("", "word " * 800), 906, [])
    assert rp.policy._remaining[906] == 9
    await _run_with_llm(_reasoning_llm("", "Error"), 906, [])
    assert rp.policy._remaining[906] == 9

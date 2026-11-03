"""The persona's side of v3 (core/classes/text_llm_handler.py): one tool,
the saved-memories block after the history, and the cached prompt prefix
that block must never change.

Run from the repo root:
    pytest core/tests/persona_tests.py
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from classes import text_llm_handler as tlh
from classes.text_llm_handler import TextLLMHandler, trailing_block


def test_the_trailing_block_has_the_time_and_the_newest_memories():
    memories = [f"fact {i}" for i in range(25)]
    block = trailing_block("2026-11-02T20:00:00+00:00", memories, "Ana")
    assert block.startswith("(Current datetime: 2026-11-02T20:00:00+00:00)")
    assert "(Saved about Ana:" in block
    assert "- fact 24" in block and "- fact 5" in block and "- fact 4" not in block


def test_the_trailing_block_is_just_the_time_without_memories():
    assert trailing_block("now") == "(Current datetime: now)"
    assert trailing_block("now", [], "Ana") == "(Current datetime: now)"
    assert "the person asking" in trailing_block("now", ["likes tea"])


def test_the_memory_block_is_capped():
    block = trailing_block("now", ["x" * 299 for _ in range(30)])
    from classes.brain import MEMORY_CHARS
    assert len(block) < MEMORY_CHARS + 200


def _handler(memories, automatic=False):
    handler = TextLLMHandler.__new__(TextLLMHandler)  # __init__ needs Redis
    handler.messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]
    handler.guild_id = 1
    handler.reasoning = ""
    handler.client = MagicMock()
    handler.automatic = automatic
    handler.voice = False
    author = MagicMock(display_name="Ana")
    handler.original_message = None if automatic else MagicMock(author=author)
    handler.actor_id = 5
    handler.channel = MagicMock()
    handler.followup = False
    handler.request_text = None
    handler.user_memory = None if automatic else MagicMock(get=AsyncMock(return_value=memories))
    handler.get_settings = AsyncMock()
    handler.get_client = AsyncMock()
    handler.agent = MagicMock()
    handler.system = "a bot"
    return handler


async def _sent(handler):
    result = MagicMock(final_output="ok", new_items=[], raw_responses=[])
    with patch.object(tlh, "Runner") as runner, \
         patch.object(tlh, "get_current_datetime", AsyncMock(return_value="now")):
        runner.run = AsyncMock(return_value=result)
        await handler.generate()
    args = runner.run.await_args
    return args.args[1], args.kwargs["context"]


@pytest.mark.asyncio
async def test_memories_go_after_the_history_and_the_prefix_never_changes():
    sent_a, ctx = await _sent(_handler(["likes tea"]))
    sent_b, _ = await _sent(_handler(["likes tea", "has a cat"]))
    assert sent_a[:-1] == sent_b[:-1] == _handler([]).messages
    assert "- likes tea" in sent_a[-1]["content"] and "Saved about Ana" in sent_a[-1]["content"]
    assert "- has a cat" in sent_b[-1]["content"]
    assert sent_a[-1]["role"] == "user"
    # the brain gets the same context: the history, who asked, their memories
    assert ctx["persona_messages"] is not None and ctx["requester_name"] == "Ana"
    assert ctx["data"] == ["likes tea"]


@pytest.mark.asyncio
async def test_an_automatic_run_shows_no_memories():
    sent, ctx = await _sent(_handler(None, automatic=True))
    assert sent[-1]["content"] == "(Current datetime: now)"
    assert ctx["automatic"] is True and ctx["data"] == []


@pytest.mark.asyncio
async def test_the_reply_is_checked_against_the_brains_facts(caplog):
    from classes.brain_result import BrainReport
    handler = _handler([])
    result = MagicMock(final_output="It costs about 1200.", new_items=[], raw_responses=[])

    async def _run(agent, messages, context=None, **kw):
        context["brain_reports"].append(BrainReport(must_keep=["$1,199"]))
        return result

    with patch.object(tlh, "Runner") as runner, \
         patch.object(tlh, "get_current_datetime", AsyncMock(return_value="now")), \
         patch.object(tlh, "inc_brain_facts_dropped") as dropped:
        runner.run = AsyncMock(side_effect=_run)
        await handler.generate()
    assert handler.used_brain is True
    dropped.assert_called_once_with(1)
    assert "left out brain facts" in caplog.text


@pytest.mark.asyncio
async def test_a_reply_without_the_brain_is_counted_as_answered():
    handler = _handler([])
    with patch.object(tlh, "inc_persona_escalation") as escalation:
        await _sent(handler)
    escalation.assert_called_once_with("answered")
    assert handler.used_brain is False


@pytest.mark.asyncio
async def test_the_persona_agent_has_only_ask_brain():
    handler = TextLLMHandler.__new__(TextLLMHandler)
    handler.system = "a bot"
    handler.options = {"temperature": 1.0}
    for voice in (False, True):
        handler.voice = voice
        await handler.get_client()
        assert [t.name for t in handler.agent.tools] == ["ask_brain"]


def test_a_spoken_reply_is_not_checked_for_exact_facts():
    # Speech says "8 degrees", never "8°C": a voice turn would always "drop" it.
    from classes.brain_result import BrainReport
    handler = _handler([])
    info = {"voice": True, "brain_reports": [BrainReport(must_keep=["8°C"])]}
    with patch.object(tlh, "inc_brain_facts_dropped") as dropped:
        handler._after_run(info, "About 8 degrees.")
    dropped.assert_not_called()
    assert handler.used_brain is True

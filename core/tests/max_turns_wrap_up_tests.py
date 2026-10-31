"""A reply that runs out of turns is answered with a wrap-up, not a ❌.

Run against the REAL SDK Runner with agents.testing.ScriptedModel, like
sandbox_conversation_tests.py: the point is to catch an SDK upgrade that
changes how the max_turns handler is called or what it is given.

Run: pytest core/tests/max_turns_wrap_up_tests.py
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agents import Agent, ModelSettings, function_tool
from agents.testing import ScriptedModel, assistant_message, function_call

from classes.text_llm_handler import WRAP_UP_INSTRUCTION, TextLLMHandler


def _handler(model, executed, max_turns=2):
    @function_tool
    async def look(what: str) -> str:
        """Look something up."""
        executed.append(what)
        return f"found {what}"

    handler = TextLLMHandler.__new__(TextLLMHandler)  # __init__ needs Redis
    handler.messages = [{"role": "user", "content": "find it and draw it"}]
    handler.guild_id = 1
    handler.reasoning = ""
    handler.client = MagicMock()
    handler.system = "a bot"
    handler.original_message = MagicMock()
    handler.user_memory = MagicMock()
    handler.user_memory.get = AsyncMock(return_value=[])
    handler.get_settings = AsyncMock()
    agent = Agent(name="t", model=model, tools=[look], model_settings=ModelSettings())

    async def _get_client():
        handler.agent = agent
    handler.get_client = _get_client
    return handler


def _looks(n):
    return [[function_call("look", {"what": f"thing {i}"}, call_id=f"c{i}")] for i in range(n)]


async def _generate(handler, max_turns=2):
    with patch("classes.text_llm_handler.get_current_datetime", AsyncMock(return_value="now")), \
         patch("classes.text_llm_handler.llm_max_turns", return_value=max_turns), \
         patch("classes.text_llm_handler.inc_llm_error", MagicMock()) as errors:
        answer = await handler.generate()
    return answer, errors


@pytest.mark.asyncio
async def test_running_out_of_turns_gives_the_wrap_up_answer():
    executed = []
    model = ScriptedModel(_looks(2) + [[assistant_message("I found two things; the drawing is not done.")]])
    answer, errors = await _generate(_handler(model, executed))

    assert answer == "I found two things; the drawing is not done."
    assert executed == ["thing 0", "thing 1"]
    errors.assert_not_called()

    last = model.last_call
    # The work so far is in the prompt, then the instruction last.
    assert last.input[-1] == {"role": "user", "content": WRAP_UP_INSTRUCTION}
    assert any(isinstance(i, dict) and i.get("type") == "function_call_output" for i in last.input)
    # Same tools as every other turn (the cached prefix), but refused.
    assert [t.name for t in last.tools] == ["look"]
    assert last.model_settings.tool_choice == "none"


@pytest.mark.asyncio
async def test_a_tool_call_in_the_wrap_up_is_never_run():
    executed = []
    model = ScriptedModel(_looks(3))
    answer, errors = await _generate(_handler(model, executed))

    # No text came back, so the old ❌ path stands, and the third look never ran.
    assert answer == "Error"
    assert executed == ["thing 0", "thing 1"]
    errors.assert_called_once()


@pytest.mark.asyncio
async def test_a_failed_wrap_up_falls_back_to_the_error_sentinel():
    executed = []
    model = ScriptedModel(_looks(2))
    handler = _handler(model, executed)
    original = model.get_response
    calls = {"n": 0}

    async def _flaky(*args, **kwargs):
        # The two run turns as scripted; the wrap-up call fails.
        calls["n"] += 1
        if calls["n"] > 2:
            raise RuntimeError("server gone")
        return await original(*args, **kwargs)

    model.get_response = _flaky
    answer, errors = await _generate(handler)

    assert answer == "Error"
    assert executed == ["thing 0", "thing 1"]
    errors.assert_called_once()


def test_strip_tool_call_text():
    from classes.response_filter import strip_tool_call_text
    assert strip_tool_call_text('Done.\n<tool_call>\n{"name": "x"}\n</tool_call>') == "Done."
    assert strip_tool_call_text('Done. <TOOL_CALL>{"name": "x"') == "Done."  # cut off
    assert strip_tool_call_text("<tool_call>{}</tool_call>") == ""
    assert strip_tool_call_text(None) == ""


@pytest.mark.asyncio
async def test_a_tool_call_written_as_text_is_stripped_from_the_wrap_up():
    # Seen live on Strata: tool_choice "none" takes the tools out of the
    # template, and the model writes its call into the answer instead.
    executed = []
    fake = '<tool_call>\n{"name": "look", "arguments": {"what": "more"}}\n</tool_call>'
    model = ScriptedModel(_looks(2) + [[assistant_message(f"Here is what I found.\n{fake}")]])
    answer, _ = await _generate(_handler(model, executed))
    assert answer == "Here is what I found."

    model = ScriptedModel(_looks(2) + [[assistant_message(fake)]])
    answer, errors = await _generate(_handler(model, []))
    assert answer == "Error"
    errors.assert_called_once()

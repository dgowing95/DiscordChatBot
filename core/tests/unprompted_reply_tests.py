"""A reply the bot was not asked for (the random reply chance) only talks:
the persona's tools are sent but refused, and tool-call text it writes
anyway is stripped (classes/text_llm_handler.py). ask_brain's own refusal is
in brain_concurrency_tests.py.

Run against the real SDK Runner with agents.testing.ScriptedModel, like
max_turns_wrap_up_tests.py."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agents.testing import ScriptedModel, assistant_message

from classes.text_llm_handler import TextLLMHandler


def _handler(unprompted: bool) -> TextLLMHandler:
    handler = TextLLMHandler.__new__(TextLLMHandler)  # __init__ needs Redis
    handler.unprompted = unprompted
    handler.options = {"temperature": 1.0}
    handler.system = "a bot"
    return handler


@pytest.mark.parametrize("unprompted, choice", [(True, "none"), (False, None)])
def test_an_unprompted_reply_refuses_its_tools(unprompted, choice):
    assert _handler(unprompted)._model_settings().tool_choice == choice


@pytest.mark.asyncio
@pytest.mark.parametrize("unprompted", [True, False])
async def test_the_refusal_holds_after_a_tool_call(unprompted):
    # The SDK puts tool_choice back to "auto" after a tool call unless told not to.
    handler = _handler(unprompted)
    with patch("classes.text_llm_handler._get_main_model_client", return_value=ScriptedModel([])):
        await handler.get_client()
    assert handler.agent.reset_tool_choice is (not unprompted)
    assert [t.name for t in handler.agent.tools] == ["ask_brain"]  # still sent: same request


@pytest.mark.asyncio
@pytest.mark.parametrize("unprompted, answer", [
    (True, "ha, same"),
    (False, 'ha, same <tool_call>{"name": "ask_brain"}</tool_call>'),
])
async def test_tool_call_text_in_an_unprompted_reply_is_stripped(unprompted, answer):
    from agents import Agent, ModelSettings
    model = ScriptedModel([[assistant_message('ha, same <tool_call>{"name": "ask_brain"}</tool_call>')]])
    handler = _handler(unprompted)
    handler.messages = [{"role": "user", "content": "lol"}]
    handler.guild_id = 1
    handler.reasoning = ""
    handler.client = MagicMock()
    handler.original_message = MagicMock()
    handler.user_memory = MagicMock(get=AsyncMock(return_value=[]))
    handler.get_settings = AsyncMock()

    async def _get_client():
        handler.agent = Agent(name="t", model=model, model_settings=ModelSettings())
    handler.get_client = _get_client
    with patch("classes.text_llm_handler.get_current_datetime", AsyncMock(return_value="now")):
        assert await handler.generate() == answer


@pytest.mark.asyncio
async def test_an_unprompted_reply_that_was_only_a_tool_call_sends_nothing():
    # Strata takes refused tools out of the template, and the model may then
    # write only <tool_call> text: stripped to "", which must post nothing.
    from classes import message_queue
    from message_handler_tests import _handled, _handler, _llm_message

    class _FakeLLM:
        def __init__(self, messages, guild_id, original_message, client=None, unprompted=False):
            self.reasoning = ""

        async def generate(self):
            return ""

    message = _llm_message(72)
    message_queue.mark_unprompted(message.id)
    handler = _handled(_handler(), message,
                       lambda h: setattr(h, "messages", [{"role": "user", "content": "hi"}]))
    handler.message.channel.send = AsyncMock()
    with patch("classes.message_handler.TextLLMHandler", _FakeLLM), \
         patch("classes.workspace_manager.workspaces", return_value=MagicMock()):
        await handler.handle_message()
    handler.message.channel.send.assert_not_awaited()
    assert ('❌',) not in [c.args for c in handler.message.add_reaction.call_args_list]

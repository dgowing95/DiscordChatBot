"""The brain's question to the requester, in the persona's own voice.

The brain writes plain, functional questions ("Which chart type: bar or
line?"); posted as they are, the bot would suddenly drop its personality
mid-conversation. So one small side call (the persona's own connection, chat
lane, thinking off) rewrites it with the server's personality as the system
message. Any failure posts the brain's question unchanged: a question in the
wrong voice beats no question.
"""
import dataclasses
import logging

from classes import llm_lanes
from classes.llm_roles import PERSONA, role_config
from classes.response_filter import strip_thinking
from classes.side_llm import SideLLM

logger = logging.getLogger(__name__)

# Short: the brain run is waiting on it, and the persona's server may be
# busy with chat (which it shares this lane with).
QUESTION_TIMEOUT_SECONDS = 20
MAX_TOKENS = 150
MAX_QUESTION_CHARS = 500

PROMPT = ("Ask this question in your own voice, in one or two sentences, and nothing else. "
          "Keep every option and detail it mentions:\n{question}")


def _config():
    """The persona's connection (not side_job_config, which falls back to
    the brain's), with the short wait a question allows."""
    return dataclasses.replace(role_config(PERSONA), call_timeout=QUESTION_TIMEOUT_SECONDS)


_llm = SideLLM("brain_question", _config, lane=llm_lanes.CHAT)


def question_messages(personality: str, question: str) -> list[dict]:
    return [
        {"role": "system", "content": f"Answer as if you are {personality}."},
        {"role": "user", "content": PROMPT.format(question=question[:MAX_QUESTION_CHARS])},
    ]


async def voice_question(guild_id, question: str) -> str:
    """The question as the persona would ask it; the question itself when
    that fails."""
    question = (question or "").strip()
    if not question:
        return question
    try:
        from classes.config_manager import configManager
        personality = await configManager().get_setting("system", guild_id) or "An AI Story Teller"
        text = await _llm.complete(question_messages(personality, question),
                                   temperature=0.7, max_tokens=MAX_TOKENS)
        text = strip_thinking(text or "").strip()
        if text:
            return text[:1500]
    except Exception as e:
        logger.info(f"Brain question: rewording failed ({e}); asking it as written")
    return question[:1500]

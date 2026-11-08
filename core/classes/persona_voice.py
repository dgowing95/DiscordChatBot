"""What the brain says straight to people, in the persona's own voice.

The brain writes plain, functional text: questions ("Which chart type: bar or
line?") and a workflow manager's replies to messages in its thread ("Noted:
the remaining steps use euros."). Posted as they are, the bot would suddenly
drop its personality mid-conversation. So one small side call each (the
persona's own connection, chat lane, thinking off) rewrites it with the
server's personality as the system message. Any failure posts the brain's
text unchanged: a message in the wrong voice beats no message.

Status cards (a timeout, "No more changes", an interrupted workflow) are not
rewritten: they are fixed notices, not something the brain said.
"""
import asyncio
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
# A manager's reply is cut to 400 characters before it is posted; reworded,
# it may run a little longer.
REPLY_MAX_TOKENS = 250
MAX_REPLY_CHARS = 600

PROMPT = ("Ask this question in your own voice, in one or two sentences, and nothing else. "
          "Keep every option and detail it mentions:\n{question}")
REPLY_PROMPT = ("Say this to the person in your own voice, in a few sentences at most, and "
                "nothing else. Keep every fact, number, name and link it has:\n{reply}")


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


async def _personality(guild_id) -> str:
    from classes.config_manager import configManager
    return await configManager().get_setting("system", guild_id) or "An AI Story Teller"


async def voice_question(guild_id, question: str) -> str:
    """The question as the persona would ask it; the question itself when
    that fails."""
    question = (question or "").strip()
    if not question:
        return question
    try:
        personality = await _personality(guild_id)
        text = await _llm.complete(question_messages(personality, question),
                                   temperature=0.7, max_tokens=MAX_TOKENS)
        text = strip_thinking(text or "").strip()
        if text:
            return text[:1500]
    except Exception as e:
        logger.info(f"Brain question: rewording failed ({e}); asking it as written")
    return question[:1500]


def reply_messages(personality: str, reply: str) -> list[dict]:
    return [
        {"role": "system", "content": f"Answer as if you are {personality}."},
        {"role": "user", "content": REPLY_PROMPT.format(reply=reply[:MAX_REPLY_CHARS])},
    ]


async def _voice_reply(personality: str, reply: str) -> str:
    try:
        text = await _llm.complete(reply_messages(personality, reply),
                                   temperature=0.7, max_tokens=REPLY_MAX_TOKENS)
        text = strip_thinking(text or "").strip()
        if text:
            return text[:MAX_REPLY_CHARS]
    except Exception as e:
        logger.info(f"Persona voice: rewording a reply failed ({e}); posting it as written")
    return reply


async def voice_replies(guild_id, replies: list[str]) -> list[str]:
    """Each reply as the persona would say it, side by side; a reply itself
    when its rewording fails (or there is no personality to read)."""
    replies = [(r or "").strip() for r in replies]
    if not any(replies):
        return replies
    try:
        personality = await _personality(guild_id)
    except Exception as e:
        logger.info(f"Persona voice: no personality ({e}); posting replies as written")
        return replies
    return list(await asyncio.gather(*(_voice_reply(personality, r) if r else asyncio.sleep(0, r)
                                       for r in replies)))

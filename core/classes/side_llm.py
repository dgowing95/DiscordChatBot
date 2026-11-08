"""One small, non-agent LLM call made on the side of a reply.

The research brief (web_research.py), the image prompt rewrite
(image_prompt.py), the generated-image check (image_review.py) and the voice
hold-on line and greeting (voice_session.py) each make a single chat
completion against an OpenAI-compatible server -- usually the bot's own
llama.cpp, optionally a different one per job (llm_roles.side_job_config).
They share the awkward parts, which live here:

  * the client is built once per job and reused (each AsyncOpenAI carries its
    own httpx connection pool, and the settings never change at runtime);
  * the call takes a slot of its lane on a self-hosted server
    (classes/llm_lanes.py). The lane is the job's own when given (the voice
    lines are chat), else the lane of the work making the call: a research
    brief inside a brain run waits as brain work, inside a chat reply as
    chat (llm_lanes.current_lane). The slot is taken per call, so a side call
    inside a tool never waits on its own parent run;
  * thinking is turned off. The bot's own model is typically a reasoning
    model (prod runs Qwen3), and a <think> block on a "reply with JSON" job
    runs to hundreds of tokens before the answer starts. Left on, it either
    overruns max_tokens -- an unterminated block, which strip_thinking reduces
    to nothing, so the job falls soft on EVERY call -- or it spends the whole
    timeout generating reasoning nobody reads. Two fields say it, because
    servers disagree on how: llama.cpp forwards chat_template_kwargs into the
    chat template (Qwen3 honours enable_thinking) and ignores `reasoning`;
    OpenRouter ignores chat_template_kwargs and reads `reasoning`. With only
    the first, DeepSeek on OpenRouter spent all 1024 tokens of the image
    rewrite and check thinking and returned nothing, so both fell soft on
    every call. A backend that rejects the request with either field (a
    400/422; neither is in the OpenAI schema) gets one retry without them,
    and is not asked again;
  * every call is timed into discord_bot_llm_call_seconds{caller=...}, after
    the lane wait (which goes to discord_bot_llm_lane_wait_seconds).

Callers own their prompts, parsing and fallbacks: complete() raises like the
client does, and each caller decides what failing soft means for it.
"""
import asyncio
import logging
import time

from openai import AsyncOpenAI, BadRequestError, UnprocessableEntityError

from classes import llm_lanes
from classes.metrics import observe_lane_wait, observe_llm_call, observe_llm_completion_tokens

logger = logging.getLogger(__name__)

NO_THINKING = {
    "chat_template_kwargs": {"enable_thinking": False},  # llama.cpp (the chat template)
    "reasoning": {"enabled": False},  # OpenRouter
}


class SideLLM:
    """A lazily built client for one side job, plus its thinking-off latch.

    `config` is a callable returning the job's llm_roles.RoleConfig, read
    when the client is first built, so tests (and a process that sets env
    vars late) see current values. Its call_timeout is the client's timeout.
    `lane` None means the caller's lane, read per call.
    """

    def __init__(self, caller: str, config, lane: str | None = None):
        self.caller = caller
        self._config_source = config
        self.lane = lane
        self._config = None
        self._client = None
        self.send_no_thinking = True

    def reset(self) -> None:
        """Forget the client and the latch (tests; nothing else needs it)."""
        self._config = None
        self._client = None
        self.send_no_thinking = True

    def config(self):
        if self._config is None:
            self._config = self._config_source()
        return self._config

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            config = self.config()
            self._client = AsyncOpenAI(
                base_url=config.host + "/v1",
                api_key=config.client_api_key,
                timeout=config.call_timeout,
                default_headers=config.extra_headers or None,
            )
        return self._client

    async def complete(self, messages, *, temperature: float, max_tokens: int) -> str:
        """The answer's text. Raises when the call fails."""
        # Read once: whether THIS call sent the option decides whether its own
        # failure says anything about it. Re-reading the shared latch after an
        # await let a concurrent call (an edit is checked twice at once) flip
        # it mid-call and turn this call's rejection into a plain failure.
        no_thinking = self.send_no_thinking
        try:
            return await self._complete(messages, temperature, max_tokens, no_thinking)
        except (BadRequestError, UnprocessableEntityError) as e:
            # The server refused the request itself, which is what a backend
            # that does not accept NO_THINKING's fields does (they are not
            # part of the OpenAI schema). Anything else -- a 5xx, a rate limit, the
            # backend being down or slow -- says nothing about the option, and
            # a retry that then happened to work would wrongly latch thinking
            # back on for the whole process.
            if not no_thinking:
                raise
            logger.info(f"{self.caller}: request rejected with thinking disabled ({e}); "
                        f"retrying without that option")
            content = await self._complete(messages, temperature, max_tokens, False)
            # Latched only now the retry has actually worked, which is what
            # identifies the option as the culprit -- thinking left on is what
            # breaks these jobs, so it must not be re-enabled on a guess.
            logger.info(f"{self.caller}: this backend rejects the thinking-disabled "
                        f"option; not sending it again")
            self.send_no_thinking = False
            return content

    async def _complete(self, messages, temperature, max_tokens, no_thinking) -> str:
        client = self._get_client()
        lanes = llm_lanes.lanes_for(self.config().host_key)
        if lanes is None:
            return await self._request(client, messages, temperature, max_tokens, no_thinking)
        lane = self.lane or llm_lanes.current_lane.get()
        async with lanes.slot(lane) as waited:
            observe_lane_wait(lane, waited)
            return await self._request(client, messages, temperature, max_tokens, no_thinking)

    async def _request(self, client, messages, temperature, max_tokens, no_thinking) -> str:
        extra = {"extra_body": NO_THINKING} if no_thinking else {}
        started = time.monotonic()
        outcome = "error"
        try:
            response = await client.chat.completions.create(
                model=self.config().model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **extra,
            )
            outcome = "ok"
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        finally:
            observe_llm_call(self.caller, outcome, time.monotonic() - started)
        usage = getattr(response, "usage", None)
        observe_llm_completion_tokens(self.caller, getattr(usage, "completion_tokens", None))
        return response.choices[0].message.content

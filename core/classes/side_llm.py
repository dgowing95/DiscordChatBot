"""One small, non-agent LLM call made on the side of a reply.

The image prompt rewrite (image_prompt.py) and the generated-image check
(image_review.py) each make a single chat completion against an
OpenAI-compatible server -- usually the bot's own llama.cpp, optionally a
different one per job. They share the awkward parts, which live here:

  * the client is built once per job and reused (each AsyncOpenAI carries its
    own httpx connection pool, and the settings never change at runtime);
  * thinking is turned off. The bot's own model is typically a reasoning
    model (prod runs Qwen3), and a <think> block on a "reply with JSON" job
    runs to hundreds of tokens before the answer starts. Left on, it either
    overruns max_tokens -- an unterminated block, which strip_thinking reduces
    to nothing, so the job falls soft on EVERY call -- or it spends the whole
    timeout generating reasoning nobody reads. llama.cpp forwards
    chat_template_kwargs into the chat template, and Qwen3 honours
    enable_thinking. A backend that rejects the field (it is not part of the
    OpenAI schema) gets one retry without it, and is not asked again;
  * every call is timed into discord_bot_llm_call_seconds{caller=...}.

Callers own their prompts, parsing and fallbacks: complete() raises like the
client does, and each caller decides what failing soft means for it.
"""
import asyncio
import logging
import time

# APITimeoutError subclasses APIConnectionError, so the one name covers the
# whole "backend is unreachable or too slow" class.
from openai import APIConnectionError, AsyncOpenAI

from classes.metrics import observe_llm_call, observe_llm_completion_tokens

logger = logging.getLogger(__name__)

NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}


class SideLLM:
    """A lazily built client for one side job, plus its thinking-off latch.

    The connection settings are callables, read when the client is first
    built, so tests (and a process that sets env vars late) see current values.
    """

    def __init__(self, caller: str, model, host, api_key, timeout):
        self.caller = caller
        self._model = model
        self._host = host
        self._api_key = api_key
        self._timeout = timeout
        self._client = None
        self.send_no_thinking = True

    def reset(self) -> None:
        """Forget the client and the latch (tests; nothing else needs it)."""
        self._client = None
        self.send_no_thinking = True

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                base_url=self._host() + "/v1",
                api_key=self._api_key(),
                timeout=self._timeout(),
            )
        return self._client

    async def complete(self, messages, *, temperature: float, max_tokens: int) -> str:
        """The answer's text. Raises when the call fails."""
        try:
            return await self._complete(messages, temperature, max_tokens, self.send_no_thinking)
        except APIConnectionError:
            # The backend is down or slow. The thinking option is not the
            # cause, so retrying would only spend the timeout a second time.
            raise
        except Exception as e:
            if not self.send_no_thinking:
                raise
            # Could be the backend rejecting chat_template_kwargs rather than
            # anything wrong with the call. Try once without it.
            logger.info(f"{self.caller}: call failed with thinking disabled ({e}); "
                        f"retrying without that option")
            content = await self._complete(messages, temperature, max_tokens, False)
            # Latched only now the retry has actually worked, which is what
            # identifies the option as the culprit. Latching on the failure
            # instead would mean one bad minute re-enabled thinking for the
            # whole process -- and thinking left on is what breaks these jobs.
            logger.info(f"{self.caller}: this backend rejects the thinking-disabled "
                        f"option; not sending it again")
            self.send_no_thinking = False
            return content

    async def _complete(self, messages, temperature, max_tokens, no_thinking) -> str:
        extra = {"extra_body": NO_THINKING} if no_thinking else {}
        started = time.monotonic()
        outcome = "error"
        try:
            response = await self._get_client().chat.completions.create(
                model=self._model(),
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

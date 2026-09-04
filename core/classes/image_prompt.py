"""Rewrites a plain-language image request into a diffusion-model prompt.

The image model is Stable Diffusion XL, conditioned on CLIP. It wants something
quite unlike what a chat model writes unprompted:

  * the subject first, because CLIP weights the opening tokens most heavily;
  * comma-separated descriptive clauses rather than narrative sentences;
  * negations expressed as a SEPARATE negative prompt, since "no people" in the
    positive prompt reads to CLIP as "people".

That knowledge used to live in the generate_image tool docstring, which meant
the main agent had to apply it while also holding a per-guild personality, and
which the /generate_image slash command bypassed entirely -- it sends whatever
the user typed. Putting it in a dedicated LLM call instead covers both paths
and keeps the main agent's job to "describe what the user asked for".

Fails soft, the way content_guard does: any error, timeout or unparseable
answer returns the original request unchanged, so a rewriter that is down
degrades to the previous behaviour instead of costing the user their image.

Environment variables:
    IMAGE_PROMPT_REWRITE_ENABLED  "0"/"false"/"no"/"off" to skip the rewrite
                                  and send the request through as-is
    IMAGE_PROMPT_MODEL            model id (default: the bot's MODEL)
    IMAGE_PROMPT_LLM_HOST         API base, /v1 appended (default: LLM_HOST)
    IMAGE_PROMPT_LLM_API_KEY      API key (default: LLM_PASS)
    IMAGE_PROMPT_TIMEOUT          seconds to wait for the rewrite (default 60)
"""
import json
import logging
import os
import re

# APITimeoutError subclasses APIConnectionError, so the one name covers the
# whole "backend is unreachable or too slow" class.
from openai import APIConnectionError, AsyncOpenAI

from classes.llm_config import (
    DEFAULT_LLM_API_KEY,
    DEFAULT_LLM_HOST,
    DEFAULT_MODEL,
    env_or,
)
from classes.response_filter import strip_thinking

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 60

SYSTEM_PROMPT = """You rewrite an image request into a prompt for Stable Diffusion XL.

Reply with ONLY a JSON object and nothing else — no prose, no code fence:
{"prompt": "...", "negative_prompt": "..."}

"prompt":
- Open with the main subject and what it is doing. The first words carry the
  most weight, so nothing vague goes at the front.
- Follow it with comma-separated clauses, roughly in this order: the subject's
  position and orientation, the setting, other objects present, materials and
  textures, lighting, colour, camera angle, and last the style or medium.
- State any spatial relationship plainly and say the important one twice in
  different words ("standing upright, vertical like a pillar").
- Name materials and textures outright: "coarse fur", "polished steel".
- Describe only what IS in the picture. "no X", "without X" and "not X" belong
  in negative_prompt, never here — the image model reads them as X.
- Any words that must be legible in the image go in double quotes near the
  front, and keep them to two or three words.
- 60 words maximum. No parentheses, no brackets, no hyphenated compounds and
  no +/- weighting marks: the encoder reads those as syntax.
- Add detail to what was asked for. Do not add objects nobody asked for.

"negative_prompt":
- Comma-separated terms, never sentences.
- Every negation from the request, stated positively: "no people" becomes
  "people".
- Then the failure modes that would specifically spoil THIS image, e.g. "extra
  fingers" when hands are visible, "bent walls" for architecture.
- Use "" when there is nothing specific to exclude. A general quality baseline
  is added later, so do not write one here."""

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

# The bot's own model is typically a reasoning model (prod runs Qwen3), and a
# <think> block on a "rewrite this as JSON" request runs to hundreds of tokens
# before the answer starts. Left on, it either overruns max_tokens -- leaving an
# unterminated block, which strip_thinking reduces to nothing and parse_rewrite
# then rejects, so the rewrite would fall soft on EVERY call -- or it spends the
# whole timeout generating reasoning nobody reads.
#
# llama.cpp forwards chat_template_kwargs into the chat template, and Qwen3's
# honours enable_thinking. Backends that reject the field (it is not part of the
# OpenAI schema) get one retry without it, and are not asked again.
_NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}
_send_no_thinking = True

# Built once and reused, like text_llm_handler's main model client: the
# connection settings never change at runtime, and each AsyncOpenAI carries its
# own httpx connection pool.
_client = None


def rewrite_enabled() -> bool:
    """True when a request should be rewritten before it reaches the service."""
    raw = os.environ.get("IMAGE_PROMPT_REWRITE_ENABLED", "1")
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def rewrite_model() -> str:
    """Model id for the rewrite (IMAGE_PROMPT_MODEL; default: the bot's MODEL).

    Split out so a deployment can point the rewrite at a different
    OpenAI-compatible API than the chat model, the way the sandbox agent does.
    """
    return env_or("IMAGE_PROMPT_MODEL", "MODEL", DEFAULT_MODEL)


def rewrite_llm_host() -> str:
    """API base for the rewrite (IMAGE_PROMPT_LLM_HOST; default: LLM_HOST).
    /v1 is appended here, so the value must not include it."""
    return env_or("IMAGE_PROMPT_LLM_HOST", "LLM_HOST", DEFAULT_LLM_HOST)


def rewrite_llm_api_key() -> str:
    """API key for the rewrite (IMAGE_PROMPT_LLM_API_KEY; default: LLM_PASS)."""
    return env_or("IMAGE_PROMPT_LLM_API_KEY", "LLM_PASS", DEFAULT_LLM_API_KEY)


def rewrite_timeout() -> float:
    raw = os.environ.get("IMAGE_PROMPT_TIMEOUT", "").strip()
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_TIMEOUT
    return value if value > 0 else DEFAULT_TIMEOUT


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            base_url=rewrite_llm_host() + "/v1",
            api_key=rewrite_llm_api_key(),
            timeout=rewrite_timeout(),
        )
    return _client


def parse_rewrite(content: str) -> tuple[str, str] | None:
    """(prompt, negative_prompt) from the model's answer, or None if unusable.

    The local model is a reasoning model, so the JSON arrives after a <think>
    block; strip_thinking removes it. What is left may still be fenced or have
    a sentence wrapped around it, hence matching the outermost {...} rather
    than parsing the whole string.
    """
    match = _JSON_OBJECT_RE.search(strip_thinking(content or ""))
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    prompt = str(data.get("prompt") or "").strip()
    if not prompt:
        return None
    return prompt, str(data.get("negative_prompt") or "").strip()


async def _complete(request: str, no_thinking: bool) -> str:
    """One chat completion for the rewrite. Raises like the client does."""
    extra = {"extra_body": _NO_THINKING} if no_thinking else {}
    response = await _get_client().chat.completions.create(
        model=rewrite_model(),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": request},
        ],
        # Low but not zero: this is a formatting job, not a creative one, and a
        # guild's /temperature setting must not reach it.
        temperature=0.3,
        # Roomy on purpose. The answer is ~80 tokens; the headroom is so that a
        # backend which ignores _NO_THINKING still has space to finish its
        # <think> block AND the JSON after it, rather than being cut mid-think.
        max_tokens=2048,
        **extra,
    )
    return response.choices[0].message.content


async def build_image_prompt(request: str) -> tuple[str, str]:
    """Rewrite `request` into (prompt, negative_prompt) for the image service.

    Returns (request, "") unchanged when the rewrite is disabled or fails.
    """
    global _send_no_thinking
    request = (request or "").strip()
    if not request or not rewrite_enabled():
        return request, ""
    try:
        content = await _complete(request, _send_no_thinking)
    except APIConnectionError as e:
        # The backend is down or slow. The thinking option is not the cause, so
        # retrying would only spend the timeout a second time.
        logger.warning(f"Image prompt rewrite failed ({e}); using the request as-is")
        return request, ""
    except Exception as e:
        if not _send_no_thinking:
            logger.warning(f"Image prompt rewrite failed ({e}); using the request as-is")
            return request, ""
        # Could be the backend rejecting chat_template_kwargs rather than
        # anything wrong with the call. Try once without it.
        logger.info(f"Image prompt rewrite failed with thinking disabled ({e}); "
                    f"retrying without that option")
        try:
            content = await _complete(request, False)
        except Exception as retry_error:
            logger.warning(f"Image prompt rewrite failed ({retry_error}); "
                           f"using the request as-is")
            return request, ""
        # Latched only now the retry has actually worked, which is what
        # identifies the option as the culprit. Latching on the failure instead
        # would mean one bad minute re-enabled thinking for the whole process --
        # and thinking left on is the thing that breaks this feature.
        logger.info("Image prompt rewrite: this backend rejects the "
                    "thinking-disabled option; not sending it again")
        _send_no_thinking = False
    parsed = parse_rewrite(content)
    if parsed is None:
        logger.warning(f"Image prompt rewrite returned no usable JSON "
                       f"({(content or '')[:200]!r}); using the request as-is")
        return request, ""
    prompt, negative = parsed
    logger.info(f"Image prompt rewritten:\n  from: {request}\n    to: {prompt}"
                f"\n   neg: {negative}")
    return prompt, negative

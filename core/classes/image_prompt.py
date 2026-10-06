"""Rewrites a plain-language image request into a diffusion-model prompt.

Only for SD/SDXL, which are conditioned on CLIP and want something quite
unlike what a chat model writes unprompted:

  * the subject first, because CLIP weights the opening tokens most heavily;
  * comma-separated descriptive clauses rather than narrative sentences;
  * negations expressed as a SEPARATE negative prompt, since "no people" in the
    positive prompt reads to CLIP as "people".

A model that reads plain sentences (FLUX.2 [klein], whose text encoder is an
LLM) gets the request as written instead: the diffusion service says which kind
it is on /health, and image_generation.create_image skips this for "natural".

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

from classes.llm_config import (
    DEFAULT_LLM_API_KEY,
    DEFAULT_LLM_HOST,
    DEFAULT_MODEL,
    env_or,
)
from classes.response_filter import strip_thinking
from classes.side_llm import SideLLM

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 60

# No literal "always exclude these" examples: the local model copied the ones
# this used to carry ("people, extra fingers, bent walls") into nearly every
# negative prompt -- "people" included, for pictures OF people. The worked
# example below is a subject nobody asks for, and clean_negative drops any
# term the prompt itself asks for, should one leak through.
SYSTEM_PROMPT = """You rewrite an image request into a prompt for Stable Diffusion XL.

Reply with ONLY a JSON object and nothing else — no prose, no code fence:
{"prompt": "...", "negative_prompt": "..."}

"prompt":
- Open with whatever is unusual, exaggerated or newly changed about the
  subject, then the subject and what it is doing. The first words carry the
  most weight, and the image model drifts back to the ordinary version of
  anything it hears about late.
- Follow it with comma-separated clauses, roughly in this order: position
  and orientation, the setting, other objects present, materials and
  textures, lighting, colour, camera angle, and last the style or medium.
- Say the most important detail twice, in different words.
- Name materials and textures outright: "coarse fur", "polished steel".
- Describe only what IS in the picture. "no X", "without X" and "not X"
  belong in negative_prompt, never here — the image model reads them as X.
- Any words that must be legible in the image go in double quotes near the
  front, and keep them to two or three words.
- 60 words maximum. No parentheses, no brackets, no hyphenated compounds and
  no +/- weighting marks: the encoder reads those as syntax.
- Add detail to what was asked for. Do not add objects nobody asked for.

"negative_prompt": at most 8 comma-separated terms, never sentences:
- the opposite of each unusual or exaggerated detail, so the image model
  cannot drift back to the ordinary version;
- anything the request says must not appear, stated positively.
Never list something the prompt asks for. Use "" when nothing applies; a
general quality baseline is added later.

Example request: a lighthouse made entirely of glass on a stormy cliff, no boats
{"prompt": "lighthouse made entirely of clear glass, transparent glass tower on a stormy sea cliff, crashing waves below, dark storm clouds, glowing lamp at the top, rain streaks, dramatic lighting, cinematic wide shot, digital painting", "negative_prompt": "brick lighthouse, stone tower, opaque walls, boats, ships"}"""

# What survives clean_negative. The instructions ask for 8; this is the
# backstop for a model that loops -- prod logged negatives of 262 to 624 CLIP
# tokens, which the encoder truncates at 77 anyway.
MAX_NEGATIVE_TERMS = 10

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_WORD_RE = re.compile(r"[a-z0-9]+")
# Words that say nothing about WHAT a term excludes, so they never make a
# term look different from the prompt.
_FILLER = frozenset({"a", "an", "the", "of", "and", "or", "with", "in", "on",
                     "at", "to", "for", "from", "by", "very", "too", "her", "his"})
_NEGATING = ("no ", "not ", "without ", "never ")


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


# Its own client and thinking-off latch, separate from image_review's: the two
# jobs can point at different backends.
_llm = SideLLM("image_prompt", rewrite_model, rewrite_llm_host,
               rewrite_llm_api_key, rewrite_timeout)


def _content_words(text: str) -> set:
    """Lower-cased words minus filler, with a plural "s" dropped so
    "helmets" and "helmet" compare equal."""
    words = set()
    for word in _WORD_RE.findall(text.lower()):
        if word in _FILLER:
            continue
        words.add(word[:-1] if len(word) > 3 and word.endswith("s") else word)
    return words


def clean_negative(negative: str, prompt: str) -> str:
    """The rewriter's negative prompt with what actually hurt removed.

    Each term is dropped when it is:
      * a repeat;
      * negated ("no orange sphere body") -- a negative of a negative, which
        is how the model restated the prompt it had just written;
      * made only of words the prompt itself uses ("helmet, visor, arms" for
        a soldier in a visored helmet). A term that differs by any word stays,
        which is what keeps the useful opposites: "thick arms" against a
        prompt asking for "thin stick arms".
    Then capped at MAX_NEGATIVE_TERMS.
    """
    prompt_words = _content_words(prompt)
    kept, seen = [], set()
    for raw in (negative or "").split(","):
        term = " ".join(raw.split())
        key = term.lower()
        if not term or key in seen:
            continue
        seen.add(key)
        if key.startswith(_NEGATING):
            continue
        words = _content_words(term)
        if not words or words <= prompt_words:
            continue
        kept.append(term)
        if len(kept) == MAX_NEGATIVE_TERMS:
            break
    return ", ".join(kept)


def parse_rewrite(content: str) -> tuple[str, str] | None:
    """(prompt, negative_prompt) from the model's answer, or None if unusable.

    The local model is a reasoning model, so the JSON may arrive after a
    <think> block; strip_thinking removes it. What is left may still be fenced
    or have a sentence wrapped around it, hence matching the outermost {...}
    rather than parsing the whole string.
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
    return prompt, clean_negative(str(data.get("negative_prompt") or ""), prompt)


async def build_image_prompt(request: str) -> tuple[str, str]:
    """Rewrite `request` into (prompt, negative_prompt) for the image service.

    Returns (request, "") unchanged when the rewrite is disabled or fails.
    """
    request = (request or "").strip()
    if not request or not rewrite_enabled():
        return request, ""
    try:
        content = await _llm.complete(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": request},
            ],
            # Low but not zero: this is a formatting job, not a creative one,
            # and a guild's /temperature setting must not reach it.
            temperature=0.3,
            # The answer is ~100 tokens. The headroom is for a backend that
            # ignores the thinking switch; past it, a model stuck repeating
            # negative terms is cut off and the request goes through as-is.
            max_tokens=1024,
        )
    except Exception as e:
        logger.warning(f"Image prompt rewrite failed ({e}); using the request as-is")
        return request, ""
    parsed = parse_rewrite(content)
    if parsed is None:
        logger.warning(f"Image prompt rewrite returned no usable JSON "
                       f"({(content or '')[:200]!r}); using the request as-is")
        return request, ""
    prompt, negative = parsed
    logger.info(f"Image prompt rewritten:\n  from: {request}\n    to: {prompt}"
                f"\n   neg: {negative}")
    return prompt, negative

"""Looks at a generated image before it is posted.

The bot used to post every image unseen and then caption it from the prompt
it had written itself. When the image model missed the point -- arms asked to
be "really really small and skinny" that came out ordinary, a "colossal"
soldier who came out life-sized -- the caption claimed the very thing the
picture lacked ("stick arms + pizza phone"). Two fixes ride on one call:

  * the caption: what the picture actually shows goes back to the outer model
    in generate_image's return string, so it describes that instead;
  * a retry: what is missing goes into one more attempt that stresses it
    (image_generation.create_image), and the better of the two is posted.

One vision chat completion (two for an edit; see review_image) against an
OpenAI-compatible server -- by default the bot's own llama.cpp, whose model
has its vision projector loaded. Fails soft: any error or unparseable answer
returns None, and the image is posted unchecked rather than not at all.

Environment variables:
    IMAGE_REVIEW_ENABLED      "0"/"false"/"no"/"off" to post images unchecked
    IMAGE_REVIEW_RETRIES      extra attempts when something asked for is
                              missing (default 1, max 2; 0 = check only)
    IMAGE_REVIEW_MODEL        model id (default: the bot's MODEL)
    IMAGE_REVIEW_LLM_HOST     API base, /v1 appended (default: LLM_HOST)
    IMAGE_REVIEW_LLM_API_KEY  API key (default: LLM_PASS)
    IMAGE_REVIEW_TIMEOUT      seconds to wait for the check (default 60)
"""
import asyncio
import base64
import io
import json
import logging
import os
import re
from dataclasses import dataclass

from classes.llm_config import (
    DEFAULT_LLM_API_KEY,
    DEFAULT_LLM_HOST,
    DEFAULT_MODEL,
    env_or,
)
from classes.metrics import inc_image_review
from classes.response_filter import strip_thinking
from classes.side_llm import SideLLM

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 60
DEFAULT_RETRIES = 1
# Each retry is a whole extra generation (12-25 s on prod's card) plus another
# check, inside one chat reply.
MAX_RETRIES = 2
# Long side of the copy the reviewer sees. Vision tokens grow with the pixel
# count; at 640 px body proportions, scale and a few words of UI text are still
# legible, at about a third of the tokens a 1024 px image costs -- and the full
# 1024 px judged worse, not better, in the measurement under review_image.
REVIEW_MAX_SIDE = 640
MAX_MISSING = 4

# Strict on purpose, and measured that way on prod's model. A plainer version
# of this prompt passed 5 of 6 checks of pictures that missed the request:
# klein drew dark lines on ordinary arms for "arms like sticks", and the checker
# read the lines as thin arms; a "female soldier" with a man's face passed both
# times. This version caught all 6 and still passed the pictures that were
# right (a fatter edit, a fox, a round body, an obese woman on the phone).
SYSTEM_PROMPT = """You check a generated picture against what was asked for, as strictly as the person who asked would.

Reply with ONLY a JSON object and nothing else — no prose, no code fence:
{"shows": "...", "missing": ["..."]}

"shows": one or two plain sentences on what the picture actually shows: who
the main subject is (a man or a woman, if a person), their body shape as it
really looks — how thick their arms and legs are next to their body, how big
they are next to what is around them — what they are doing, the setting, the
style, and any readable text. Judge shape and size from the outline of the
body, not from lines or textures drawn on it. Describe what you see, not what
was asked for.

"missing": each thing that was asked for but is absent, wrong, or only hinted
at, written as what SHOULD be visible ("very thin arms", "a woman", "the words
LV 99"), at most 4. An unusual or exaggerated detail is the point of a
request: if it is not obvious at first glance, it is missing. Leave out small
details and matters of taste. [] only when everything important is plainly
there."""

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class ImageReview:
    shows: str
    missing: tuple = ()

    @property
    def matches(self) -> bool:
        return not self.missing


def review_enabled() -> bool:
    raw = os.environ.get("IMAGE_REVIEW_ENABLED", "1")
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def review_retries() -> int:
    """Extra attempts after a miss (IMAGE_REVIEW_RETRIES), clamped to 0..MAX_RETRIES."""
    try:
        value = int(os.environ.get("IMAGE_REVIEW_RETRIES", DEFAULT_RETRIES))
    except (TypeError, ValueError):
        return DEFAULT_RETRIES
    return max(0, min(MAX_RETRIES, value))


def review_model() -> str:
    return env_or("IMAGE_REVIEW_MODEL", "MODEL", DEFAULT_MODEL)


def review_llm_host() -> str:
    """/v1 is appended here, so the value must not include it."""
    return env_or("IMAGE_REVIEW_LLM_HOST", "LLM_HOST", DEFAULT_LLM_HOST)


def review_llm_api_key() -> str:
    return env_or("IMAGE_REVIEW_LLM_API_KEY", "LLM_PASS", DEFAULT_LLM_API_KEY)


def review_timeout() -> float:
    raw = os.environ.get("IMAGE_REVIEW_TIMEOUT", "").strip()
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_TIMEOUT
    return value if value > 0 else DEFAULT_TIMEOUT


_llm = SideLLM("image_review", review_model, review_llm_host,
               review_llm_api_key, review_timeout)


def encode_for_review(png: bytes) -> str:
    """The image as a JPEG data URL, its long side at most REVIEW_MAX_SIDE."""
    from PIL import Image

    with Image.open(io.BytesIO(png)) as image:
        image = image.convert("RGB")
        image.thumbnail((REVIEW_MAX_SIDE, REVIEW_MAX_SIDE))
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def review_request_text(user_request: str, description: str, is_edit: bool) -> str:
    """What the reviewer is told was asked for.

    The person's own words come first: the model that wrote `description` can
    misread them (Big Sally: "make her BIGGER" became "colossal, towering"), and
    a check against its description alone would pass that misreading.
    """
    lines = []
    if (user_request or "").strip():
        lines.append(f"The person asked: {user_request.strip()}")
    lines.append(f"The picture was described to the image model as: {description.strip()}")
    if is_edit:
        lines.append("This is an edit. The FIRST picture is the one the person wanted "
                     "changed and the SECOND is the result. Judge the second against the "
                     "first: the requested change must be clearly visible between them. "
                     "\"shows\" describes the second picture only.")
    return "\n".join(lines)


def parse_review(content: str) -> ImageReview | None:
    """The reviewer's answer, or None when it is unusable."""
    match = _JSON_OBJECT_RE.search(strip_thinking(content or ""))
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    shows = str(data.get("shows") or "").strip()
    if not shows:
        return None
    raw_missing = data.get("missing") or []
    if isinstance(raw_missing, str):
        raw_missing = [raw_missing]
    if not isinstance(raw_missing, list):
        return None
    # Strings only: a model saying "nothing" as [null] must not become a miss
    # called "None", with a retry and a caption admitting it is not there.
    missing = tuple(item.strip() for item in raw_missing
                    if isinstance(item, str) and item.strip())
    return ImageReview(shows=shows, missing=missing[:MAX_MISSING])


def prefer_second(first: ImageReview | None, second: ImageReview | None) -> bool:
    """Whether the second attempt should be posted instead of the first.

    A checked attempt beats an unchecked one, since only a checked one can be
    described truthfully. Otherwise the one missing fewer things wins, and a
    tie goes to the second: it was made with the misses stressed.
    """
    if second is None:
        return first is None
    if first is None:
        return True
    return len(second.missing) <= len(first.missing)


def merge_reviews(reviews) -> ImageReview | None:
    """One verdict from several looks at the same picture: the first
    description, and everything any look found missing (deduplicated)."""
    found = [review for review in reviews if review is not None]
    if not found:
        return None
    missing, seen = [], set()
    for review in found:
        for item in review.missing:
            if item.lower() not in seen:
                seen.add(item.lower())
                missing.append(item)
    return ImageReview(shows=found[0].shows, missing=tuple(missing[:MAX_MISSING]))


async def review_image(png: bytes, user_request: str, description: str,
                       reference: bytes | None = None) -> ImageReview | None:
    """What the image shows and what it is missing, or None if unchecked.

    For an edit, `reference` is the picture that was edited, and the result is
    looked at twice -- on its own, and next to the original -- because each
    look misses what the other catches. Measured with the bot's model on five
    edits whose answer was known: shown only the result, a "much fatter" edit
    that had barely changed passed (it can see she is fat, not that she got
    FATTER -- live, that one was captioned "cranked up the size"), and so did
    an unchanged copy. Shown both, those were caught, but arms that had only
    had dark lines drawn on them passed every time: the model counted the lines
    as the change. A new picture gets the single look.
    """
    if not review_enabled():
        return None
    looks = [_look(png, user_request, description, None)]
    if reference is not None:
        looks.append(_look(png, user_request, description, reference))
    review = merge_reviews(await asyncio.gather(*looks))
    if review is None:
        inc_image_review("error")
        return None
    inc_image_review("match" if review.matches else "mismatch")
    logger.info(f"Image review: shows={review.shows!r} missing={list(review.missing)}")
    return review


async def _look(png: bytes, user_request: str, description: str,
                reference: bytes | None) -> ImageReview | None:
    """One vision call: the result alone, or the original then the result."""
    try:
        images = [png] if reference is None else [reference, png]
        data_urls = [await asyncio.to_thread(encode_for_review, image) for image in images]
        text = review_request_text(user_request, description, reference is not None)
        content = await _llm.complete(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [{"type": "text", "text": text}] + [
                    {"type": "image_url", "image_url": {"url": url}} for url in data_urls
                ]},
            ],
            # Greedy. Measured on prod's model with four saved pictures, four
            # checks each: at 0.2 the same picture of ordinary arms passed in
            # 3 of 8 checks; at 0 all 16 verdicts were right.
            temperature=0.0,
            max_tokens=1024,
        )
    except Exception as e:
        logger.warning(f"Image review failed ({e})")
        return None
    review = parse_review(content)
    if review is None:
        logger.warning(f"Image review returned no usable JSON ({(content or '')[:200]!r})")
    return review

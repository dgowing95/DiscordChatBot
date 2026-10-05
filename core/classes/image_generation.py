"""Making an image: the diffusion-service client, and create_image(), which
both entry points (the generate_image tool and the /generate_image slash
command) go through.

The service runs in its own pod/container (see diffusionservice/). What it can
do depends on the model it was started with, and it says so on /health:
"prompt_style" (SD/SDXL want an LLM rewrite into tag-list form, image_prompt.py;
FLUX.2 [klein] reads plain sentences) and "edits" (whether it takes a reference
image). service_capabilities() reads that.

create_image() then:
  1. builds the prompt (the rewrite only for "sdxl");
  2. generates;
  3. looks at the result (image_review.py) -- what it shows, what is missing;
  4. when something asked for is missing, tries again with exactly that
     stressed (IMAGE_REVIEW_RETRIES, default once), and keeps the better one.
It posts nothing: the caller posts the one image it returns, and tells the
outer model what that image actually shows.
"""
import base64
import dataclasses
import logging
import os
import time
from dataclasses import dataclass

import aiohttp

from classes import image_prompt, image_review
from classes.metrics import observe_image_generation

logger = logging.getLogger(__name__)

# Image generation is slow (queue wait + GPU generation); be generous.
GENERATION_TIMEOUT = int(os.environ.get("IMAGE_GEN_TIMEOUT", 300))
CAPABILITIES_TTL_SECONDS = 300
HEALTH_TIMEOUT_SECONDS = 10
# What every generated image is posted as, and so how an image the bot made
# can be told apart from the other images it posts (sandbox artifacts, charts),
# which are not offered for editing.
GENERATED_IMAGE_FILENAME = "generated-image.png"
# How far back in the channel to look for the picture to edit when the request
# does not reply to one.
EDIT_LOOKBACK_MESSAGES = 20


def image_generation_enabled() -> bool:
    """True when the generate_image tool should be offered to the LLM.

    Controlled by IMAGE_GEN_ENABLED (set from the helm chart's
    diffusion.enabled, or .env locally); defaults to enabled for local dev.
    """
    raw = os.environ.get("IMAGE_GEN_ENABLED", "1")
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def diffusion_base_url() -> str:
    """Base URL of the diffusion service (no trailing slash)."""
    return os.environ.get("DIFFUSION_URL", "http://diffusion:8000").rstrip("/")


# ---------------------- what the service can do ----------------------

@dataclass(frozen=True)
class ServiceCapabilities:
    prompt_style: str = "sdxl"
    edits: bool = False


# What a service that predates /health reporting these (or cannot be reached)
# is treated as: SDXL prompts, no editing -- how every image was made before.
FALLBACK_CAPABILITIES = ServiceCapabilities()

_capabilities = None  # (ServiceCapabilities, monotonic time read)


def parse_capabilities(data) -> ServiceCapabilities | None:
    """A ready service's /health body, or None when it is not one."""
    if not isinstance(data, dict) or data.get("status") != "ready":
        return None
    return ServiceCapabilities(
        prompt_style="natural" if data.get("prompt_style") == "natural" else "sdxl",
        edits=data.get("edits") is True,
    )


async def service_capabilities() -> ServiceCapabilities:
    """What the running model wants and can do, from the service's /health.

    Only a successful read is cached. /health answers 503 for the minutes the
    model takes to load after every restart, and caching the fallback through
    that would send a FLUX.2 model SDXL tag-list prompts, with editing off, for
    five minutes after each deploy.
    """
    global _capabilities
    now = time.monotonic()
    if _capabilities is not None and now - _capabilities[1] < CAPABILITIES_TTL_SECONDS:
        return _capabilities[0]
    data = None
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{diffusion_base_url()}/health",
                timeout=aiohttp.ClientTimeout(total=HEALTH_TIMEOUT_SECONDS),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
    except Exception as e:
        logger.warning(f"Could not read the image service's /health ({e!r})")
    capabilities = parse_capabilities(data)
    if capabilities is None:
        return FALLBACK_CAPABILITIES
    _capabilities = (capabilities, now)
    return capabilities


# ---------------------- one call to the service ----------------------

async def generate_image_from_api(prompt: str, negative_prompt: str = "",
                                  reference: bytes | None = None) -> bytes:
    """Ask the diffusion service for a PNG.

    `negative_prompt` lists what must NOT appear. The service merges it in
    front of its own IMAGE_NEGATIVE_PROMPT baseline and ignores it where it has
    no effect (distilled models, FLUX.2). `reference` is an image to edit; only
    send one when service_capabilities().edits is true.

    Raises on HTTP errors or connection failures; the callers turn that into a
    friendly message."""
    payload = {"prompt": prompt, "negative_prompt": negative_prompt or ""}
    if reference is not None:
        payload["reference_image"] = base64.b64encode(reference).decode()
    # Timed from the caller's perspective: queue wait + generation. Observed
    # in finally so timeouts/HTTP errors are measured too.
    start = time.monotonic()
    try:
        async with aiohttp.ClientSession(auto_decompress=False) as session:
            async with session.post(
                f"{diffusion_base_url()}/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=GENERATION_TIMEOUT),
            ) as resp:
                if resp.status != 200:
                    detail = (await resp.text())[:200]
                    raise Exception(f"diffusion service returned {resp.status}: {detail}")
                return await resp.read()
    finally:
        observe_image_generation("edit" if reference is not None else "text_to_image",
                                 time.monotonic() - start)


# ---------------------- make, look, maybe once more ----------------------

@dataclass(frozen=True)
class ImageResult:
    png: bytes
    prompt: str             # what the image model was given for THIS image
    review: object          # image_review.ImageReview, or None when unchecked
    attempts: int           # generations made, this one included
    edited: bool = False    # made from a reference image


def stress_missing(description: str, missing) -> str:
    """The description for a retry, with what the last attempt missed made
    explicit. A plain sentence rather than a label: FLUX.2 renders text well
    enough that "MOST IMPORTANT:" could end up written in the picture."""
    return f"{description.rstrip()} Make sure the picture clearly shows: {'; '.join(missing)}."


async def create_image(description: str, user_request: str = "", reference: bytes | None = None,
                       on_first_prompt=None) -> ImageResult:
    """Make one image for `description`, checked and retried (module docstring).

    `user_request` is the person's own words, which the check reads alongside
    the description. `reference` is an image to edit; a retry edits the same
    original again, not the attempt that missed. `on_first_prompt`, if given,
    is awaited with the first prompt just before generating, so the caller can
    show it while the image is made.

    Raises only when the FIRST generation fails; a failed retry posts the
    first attempt instead.
    """
    capabilities = await service_capabilities()
    is_edit = reference is not None
    best = None
    request = description
    attempts = 0
    retries = image_review.review_retries()
    for attempt in range(1 + retries):
        if capabilities.prompt_style == "sdxl":
            prompt, negative = await image_prompt.build_image_prompt(request)
        else:
            prompt, negative = request.strip(), ""
        if attempt == 0 and on_first_prompt is not None:
            await on_first_prompt(prompt)
        try:
            png = await generate_image_from_api(prompt, negative, reference)
        except Exception as e:
            if best is None:
                raise
            logger.warning(f"Image retry failed ({e}); keeping the first attempt")
            break
        attempts += 1
        review = await image_review.review_image(png, user_request, description, is_edit)
        current = ImageResult(png, prompt, review, attempts, is_edit)
        if best is None or image_review.prefer_second(best.review, review):
            best = current
        if review is None or review.matches or attempt == retries:
            break
        logger.info(f"Image attempt {attempts} is missing {list(review.missing)}; trying again")
        request = stress_missing(description, review.missing)
    return dataclasses.replace(best, attempts=attempts)


def tool_result_text(result: ImageResult) -> str:
    """What the outer model is told once the image is posted.

    This, not the prompt it wrote, is what its caption must come from: it used
    to caption from the prompt and so claimed exactly what the image model had
    missed.
    """
    sent = ("Edited image sent to the channel" if result.edited
            else "Image generated and sent to the channel")
    sent += "; the user can already see it, so do not send it again."
    review = result.review
    if review is None:
        return (f"{sent} Nobody has looked at it, so do not claim that any "
                f"particular detail is in it.")
    text = f"{sent} What it actually shows (checked by looking at it): {review.shows}"
    if review.missing:
        tries = "attempt" if result.attempts == 1 else "attempts"
        text += (f" Still not right after {result.attempts} {tries}: "
                 f"{'; '.join(review.missing)}. Say plainly that this did not come "
                 f"out; do not claim it did.")
    return text + " Describe the image only as shown here."


# ---------------------- which picture to edit ----------------------

def generated_attachment(message, bot_id):
    """The bot's own generated image on `message`, or None."""
    if message is None or bot_id is None:
        return None
    if getattr(getattr(message, "author", None), "id", None) != bot_id:
        return None
    for attachment in getattr(message, "attachments", None) or []:
        if getattr(attachment, "filename", None) == GENERATED_IMAGE_FILENAME:
            return attachment
    return None


def pick_edit_source(replied_to, recent, bot_id):
    """The attachment to edit: the bot-made image the request replies to, else
    the newest bot-made image in `recent` (newest first), else None.

    Only images the bot generated qualify. Never an uploaded photo -- that
    would make "make him fatter" work on a photo of a real person -- and never
    the bot's other images (sandbox output, charts)."""
    found = generated_attachment(replied_to, bot_id)
    if found is not None:
        return found
    for message in recent:
        found = generated_attachment(message, bot_id)
        if found is not None:
            return found
    return None


async def find_edit_source(channel, trigger, bot_id) -> bytes | None:
    """The bytes of the image to edit (pick_edit_source), or None.

    `trigger` is the message asking for the edit; None for automatic runs,
    which then look at the channel's newest messages."""
    replied_to = None
    reply = getattr(trigger, "reference", None)
    if reply is not None and getattr(reply, "message_id", None):
        resolved = getattr(reply, "resolved", None)
        # A deleted original resolves to a stand-in with no attachments.
        replied_to = resolved if getattr(resolved, "attachments", None) is not None else None
        if replied_to is None:
            try:
                replied_to = await channel.fetch_message(reply.message_id)
            except Exception as e:
                logger.info(f"Could not fetch the replied-to message ({e!r})")
    try:
        history = (channel.history(limit=EDIT_LOOKBACK_MESSAGES, before=trigger)
                   if trigger is not None else channel.history(limit=EDIT_LOOKBACK_MESSAGES))
        recent = [message async for message in history]
    except Exception as e:
        logger.warning(f"Could not read channel history for an image to edit ({e!r})")
        recent = []
    attachment = pick_edit_source(replied_to, recent, bot_id)
    if attachment is None:
        return None
    try:
        return await attachment.read()
    except Exception as e:
        logger.warning(f"Could not download the image to edit ({e!r})")
        return None

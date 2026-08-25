"""Client for the standalone image-generation (diffusion) service.

The service runs in its own pod/container (see diffusionservice/); this module
only knows whether the generate_image tool is enabled and how to ask the
service for an image.
"""
import base64
import os
import time

import aiohttp

from classes.metrics import observe_image_generation

# Image generation is slow (queue wait + GPU generation); be generous.
GENERATION_TIMEOUT = int(os.environ.get("IMAGE_GEN_TIMEOUT", 300))

# Discord attachment cap; the service resizes the source anyway, so anything
# bigger than this is wasted bandwidth rather than better quality.
MAX_SOURCE_IMAGE_BYTES = 10 * 1024 * 1024


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


async def generate_image_from_api(
    prompt: str,
    image: bytes | None = None,
    strength: float | None = None,
) -> bytes:
    """Ask the diffusion service for a PNG.

    Without `image`: text-to-image. With `image` (raw image bytes):
    image-to-image; `strength` (0-1, exclusive) controls how far the result
    moves from the source (service default when omitted).

    Raises on HTTP errors or connection failures; the caller (the
    generate_image / edit_image tools) turns that into a friendly message
    for the LLM."""
    payload = {"prompt": prompt}
    if image is not None:
        if len(image) > MAX_SOURCE_IMAGE_BYTES:
            raise Exception("source image is larger than 10MB")
        payload["image"] = base64.b64encode(image).decode()
        if strength is not None:
            payload["strength"] = strength
    # Timed from the caller's perspective: queue wait + generation. Observed
    # in finally so timeouts/HTTP errors are measured too; the tool layer
    # turns those into friendly LLM-facing strings.
    mode = "image_to_image" if image is not None else "text_to_image"
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
        observe_image_generation(mode, time.monotonic() - start)

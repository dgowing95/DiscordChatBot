"""Client for the standalone image-generation (diffusion) service.

The service runs in its own pod/container (see diffusionservice/); this module
only knows whether the generate_image tool is enabled and how to ask the
service for an image.
"""
import os
import time

import aiohttp

from classes.metrics import observe_image_generation

# Image generation is slow (queue wait + GPU generation); be generous.
GENERATION_TIMEOUT = int(os.environ.get("IMAGE_GEN_TIMEOUT", 300))


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


async def generate_image_from_api(prompt: str) -> bytes:
    """Ask the diffusion service for a PNG (text-to-image).

    Raises on HTTP errors or connection failures; the caller (the
    generate_image tool) turns that into a friendly message for the LLM."""
    payload = {"prompt": prompt}
    # Timed from the caller's perspective: queue wait + generation. Observed
    # in finally so timeouts/HTTP errors are measured too; the tool layer
    # turns those into friendly LLM-facing strings.
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
        observe_image_generation("text_to_image", time.monotonic() - start)

"""Safety guard for the web_search, fetch_url and run_code_sandbox tools.

Blocks requests that are classified as unsafe *before* the bot touches the
internet. Uses the free OpenAI Moderations endpoint
(https://developers.openai.com/api/reference/resources/moderations); the
response contains a ``categories`` object of detected unsafe categories and
a URL/query is considered unsafe if *any* of them is ``true``.

The free moderation endpoint is aggressively rate-limited (HTTP 429), so:
  * 429 / 5xx responses are retried with exponential backoff (honouring the
    ``Retry-After`` header when present), up to ``CONTENT_GUARD_MAX_RETRIES``
    times, and
  * verdicts for the same input are cached in memory for
    ``MODERATION_CACHE_TTL`` seconds so repeated lookups don't burn
    rate-limit budget.

If the moderation call itself keeps failing, we fail OPEN (allow, and log)
so a dead endpoint doesn't break the web tools entirely; disable with
``CONTENT_GUARD_ENABLED=0``.

Environment variables:
    OPENAI_API_KEY             API key for the moderation call (required
                               unless the guard is disabled)
    OPENAI_BASE_URL            API base, default https://api.openai.com/v1
                               (any OpenAI-compatible /v1/moderations endpoint
                               works)
    MODERATION_MODEL           model name to send, e.g. "text-moderation-latest";
                               if unset the parameter is omitted and the server
                               default is used
    MODERATION_CACHE_TTL       seconds a verdict stays cached (default 600,
                               0 disables the cache)
    CONTENT_GUARD_MAX_RETRIES  retries after initial attempt on 429/5xx
                               (default 2)
    CONTENT_GUARD_RETRY_BACKOFF
                               base backoff seconds, doubled per retry
                               (default 2)
    CONTENT_GUARD_ENABLED      set to "0"/"false"/"no" to disable the guard
    CONTENT_GUARD_DEBUG        set to "0"/"false" to silence debug logging
                               (default: on)
"""

import logging
import asyncio
import os
import time

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _debug_log(*args):
    """Guard debug logging, gated on CONTENT_GUARD_DEBUG (default: on).

    The env var is a documented part of the interface (AGENTS.md, .env.example),
    so it is kept and mapped onto this module's logger level rather than
    removed: on -> DEBUG is emitted, off -> it is not.
    """
    if _debug_enabled():
        logger.debug(' '.join(str(a) for a in args))


def _debug_enabled() -> bool:
    return os.environ.get("CONTENT_GUARD_DEBUG", "1").strip().lower() not in (
        "0", "false", "no", "off")


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
        return value if value >= 0 else default
    except ValueError:
        return default


def _categories(result: dict) -> dict:
    categories = result.get("categories")
    return categories if isinstance(categories, dict) else {}


def _is_unsafe(moderation_result: dict) -> bool:
    """True if the moderation result flags the input as unsafe.

    Any ``true`` value in ``categories`` counts, plus the overall
    ``flagged`` flag.
    """
    if not isinstance(moderation_result, dict):
        return False
    if moderation_result.get("flagged"):
        return True
    return any(value is True for value in _categories(moderation_result).values())


def _flagged_categories(result: dict) -> list[str]:
    return [k for k, v in _categories(result).items() if v is True]


# ---------------------------------------------------------------------------
# Verdict cache (avoids burning rate-limit budget on repeated inputs)
# ---------------------------------------------------------------------------

_verdict_cache: dict[str, tuple[bool, float]] = {}


def _cache_lookup(text: str) -> bool | None:
    """Cached verdict for ``text`` (True/False), or None on miss/expiry."""
    entry = _verdict_cache.get(text)
    if entry is None:
        return None
    verdict, expires_at = entry
    if expires_at < time.monotonic():
        _verdict_cache.pop(text, None)
        return None
    _debug_log(f"cache hit: {text[:200]!r} -> {verdict}")
    return verdict


def _cache_store(text: str, verdict: bool):
    ttl = _env_float("MODERATION_CACHE_TTL", 600.0)
    if ttl <= 0:
        return
    if len(_verdict_cache) > 1000:  # crude bound: drop everything
        _verdict_cache.clear()
    now = time.monotonic()
    _verdict_cache[text] = (verdict, now + ttl)


def _retry_delay(attempt: int) -> float:
    base = _env_float("CONTENT_GUARD_RETRY_BACKOFF", 2.0)
    return base * (2 ** attempt)


# ---------------------------------------------------------------------------
# Moderation call
# ---------------------------------------------------------------------------

async def _post_moderation(base_url: str, api_key: str, payload: dict):
    """One moderation POST. Returns (http_status, json_data, headers), or
    None on network-level failure."""
    import aiohttp

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                base_url + "/moderations",
                json=payload,
                headers={"Authorization": "Bearer " + api_key},
            ) as response:
                data = await response.json()
                return response.status, data, dict(response.headers)
    except Exception as e:
        _debug_log(f"network error: {e}")
        return None


async def moderate_with_openai(text: str) -> bool:
    """Call the OpenAI Moderations endpoint for a search query / URL.

    Returns True (allow) if no unsafe category is detected, or if the
    moderation call fails (fail open — logged).
    """
    cached = _cache_lookup(text)
    if cached is not None:
        return cached

    base_url = os.environ.get("OPENAI_BASE_URL", _DEFAULT_BASE_URL).rstrip("/")
    api_key = os.environ.get("OPENAI_API_KEY", "")

    if not api_key:
        logger.warning("Content guard: OPENAI_API_KEY is not set; failing open.")
        return True

    model = os.environ.get("MODERATION_MODEL", "").strip()
    _debug_log(
        f"moderating request (host={base_url}, model={model or 'server default'}): "
        f"{text[:200]!r}"
    )
    payload: dict = {"input": (text or "")}
    if model:
        payload["model"] = model

    max_retries = int(_env_float("CONTENT_GUARD_MAX_RETRIES", 2.0))

    attempt = 0
    while True:
        outcome = await _post_moderation(base_url, api_key, payload)

        if outcome is None:
            logger.warning("Content guard: moderation call failed (network error); "
                  "failing open.")
            return True

        status, data, headers = outcome

        if status == 200:
            try:
                result = data["results"][0]
            except Exception:
                _debug_log(f"unexpected moderation response: {data!r}")
                logger.warning(f"Content guard: unexpected moderation response "
                      f"({data}); failing open.")
                return True

            if _is_unsafe(result):
                _debug_log(f"detected categories: {_flagged_categories(result)}")
                logger.info(f"Content guard: flagged UNSAFE: {text[:200]!r}")
                _cache_store(text, False)
                return False

            logger.info(f"Content guard: flagged SAFE: {text[:200]!r}")
            _cache_store(text, True)
            return True

        # Non-200: retry 429/5xx with backoff, fail open for everything else.
        if status in _RETRYABLE_STATUS and attempt < max_retries:
            delay = _retry_from_headers(headers) or _retry_delay(attempt)
            _debug_log(f"HTTP {status} on attempt {attempt + 1}; "
                       f"retrying {attempt + 1}/{max_retries} in {delay:.0f}s")
            await asyncio.sleep(delay)
            attempt += 1
            continue

        logger.warning(f"Content guard: moderation endpoint returned HTTP {status}: "
              f"{data!r}; failing open.")
        return True


def _retry_from_headers(headers) -> float | None:
    """Numeric Retry-After header (seconds), capped at 60s, else None."""
    retry_after = (headers or {}).get("Retry-After", "")
    if not retry_after:
        return None
    try:
        return min(float(retry_after), 60.0)
    except ValueError:
        return None  # an HTTP-date Retry-After; use computed backoff


def guard_disabled() -> bool:
    value = os.environ.get("CONTENT_GUARD_ENABLED", "1").strip().lower()
    return value in ("0", "false", "no", "off")


async def check_web_request(text: str) -> tuple[bool, str]:
    """Run the guard on a web_search query or fetch_url target.

    Returns (allowed, reason). If blocked, reason explains why (safe to
    show to the user); if allowed, reason is "".
    """
    if guard_disabled():
        return True, ""

    if not await moderate_with_openai(text):
        return False, "The request was flagged as unsafe by the safety filter."

    return True, ""

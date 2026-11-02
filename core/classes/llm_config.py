"""LLM connection settings shared by the main agent and the nested sandbox agent.

Pure module (stdlib only), following the response_filter.py / message_queue.py
pattern. It is separate from text_llm_handler.py rather than living there
because sandbox_agent.py needs the same defaults and text_llm_handler.py
already imports sandbox_agent (`sandbox_enabled`), so putting them there would
be a circular import.

These defaults are the single source of truth. They used to be repeated at
every call site and had drifted apart: main.py's startup readiness check
defaulted MODEL to "gemma3:4b" while text_llm_handler.py requested "qwen3:4b",
so with MODEL unset the bot verified that one model was loaded and then asked
for a different one; and LLM_PASS defaulted to "ollama" here (a leftover from
before the llama.cpp switch) while the chart and .env.example use "llamacpp".
"""
import logging
import os
import re

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "qwen3:4b"
DEFAULT_LLM_HOST = "http://llamacpp:8080"
DEFAULT_LLM_API_KEY = "llamacpp"
DEFAULT_TEMPERATURE = 1.0


def env_or(primary: str, fallback: str, default: str) -> str:
    """First non-empty of: env var `primary`, env var `fallback`, `default`.

    How every secondary LLM caller overrides the main bot's connection one
    variable at a time — the sandbox agent (SANDBOX_MODEL / SANDBOX_LLM_HOST /
    SANDBOX_LLM_API_KEY) and the image prompt rewriter (IMAGE_PROMPT_*) both
    fall back to MODEL / LLM_HOST / LLM_PASS when their own var is unset.
    """
    for name in (primary, fallback):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return default


def llm_model() -> str:
    """The model name the bot requests (MODEL)."""
    return os.environ.get("MODEL") or DEFAULT_MODEL


def llm_host() -> str:
    """Base URL of the LLM server (LLM_HOST); callers append /v1 themselves."""
    return os.environ.get("LLM_HOST") or DEFAULT_LLM_HOST


def llm_api_key() -> str:
    """The LLM server's API key (LLM_PASS). For llama.cpp, which does not
    authenticate, it is only a placeholder: the OpenAI client refuses to
    start without a non-empty one. Strata does check it."""
    return os.environ.get("LLM_PASS") or DEFAULT_LLM_API_KEY


# An HTTP header name (RFC 9110 token characters).
_HEADER_NAME = re.compile(r"^[A-Za-z0-9!#$%&'*+.^_`|~-]+$")


def llm_extra_headers() -> dict:
    """Extra headers sent with every request to LLM_HOST (LLM_EXTRA_HEADERS).

    Comma-separated `name=value` pairs, e.g.
    `LLM_EXTRA_HEADERS=x-opencode-session=my-bot`. For a hosted API that
    needs more than a key: OpenCode Go refuses every request without an
    `x-opencode-session` header (400 MissingSessionID), which is what the
    E2E job runs against. Unset (the default, and llama.cpp/Strata) sends
    nothing extra. A malformed entry, or an Authorization header (LLM_PASS
    sets that one), is skipped with a warning rather than failing a reply.
    """
    headers = {}
    for entry in os.environ.get("LLM_EXTRA_HEADERS", "").split(","):
        entry = entry.strip()
        if not entry:
            continue
        name, sep, value = entry.partition("=")
        name, value = name.strip(), value.strip()
        if not sep or not value or not _HEADER_NAME.match(name):
            logger.warning(f"LLM_EXTRA_HEADERS: skipping {name or entry!r}, expected name=value")
            continue
        if name.lower() == "authorization":
            logger.warning("LLM_EXTRA_HEADERS: skipping Authorization; LLM_PASS sets it")
            continue
        headers[name] = value
    return headers


def extra_headers_for(host: str) -> dict:
    """llm_extra_headers() for a client of `host`, when that is LLM_HOST.

    The side jobs and the sandbox agent can each be given a server of their
    own (IMAGE_PROMPT_LLM_HOST, SANDBOX_LLM_HOST, ...). The extra headers
    belong to LLM_HOST's server, so a job pointed elsewhere sends none; one
    that falls back to LLM_HOST sends them like the main agent does.
    """
    if host.rstrip("/") != llm_host().rstrip("/"):
        return {}
    return llm_extra_headers()


def llm_auth_headers() -> dict:
    """Headers for core's own HTTP calls to the LLM server (/props,
    /v1/models), which go around the OpenAI client and so would otherwise
    send no key (or LLM_EXTRA_HEADERS). A server that checks the key answers
    them 401 without it; llama.cpp ignores it."""
    return {**llm_extra_headers(), "Authorization": f"Bearer {llm_api_key()}"}


def parse_temperature(raw) -> float:
    """A guild's configured temperature, as returned by
    configManager.get_setting — which yields False (not None) when unset.

    Written out explicitly rather than as `float(raw) or 1.0`, which was the
    original: 0.0 is falsy, so that expression silently turned a deliberate
    `/temperature 0` back into 1.0. An unparseable value also falls back here
    instead of raising out of generate() and costing the whole reply.
    """
    if raw is None or raw is False or raw == "":
        return DEFAULT_TEMPERATURE
    try:
        return float(raw)
    except (TypeError, ValueError):
        return DEFAULT_TEMPERATURE

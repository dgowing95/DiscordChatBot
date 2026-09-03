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
import os

DEFAULT_MODEL = "qwen3:4b"
DEFAULT_LLM_HOST = "http://llamacpp:8080"
DEFAULT_LLM_API_KEY = "llamacpp"
DEFAULT_TEMPERATURE = 1.0


def llm_model() -> str:
    """The model name the bot requests (MODEL)."""
    return os.environ.get("MODEL") or DEFAULT_MODEL


def llm_host() -> str:
    """Base URL of the LLM server (LLM_HOST); callers append /v1 themselves."""
    return os.environ.get("LLM_HOST") or DEFAULT_LLM_HOST


def llm_api_key() -> str:
    """Placeholder key (LLM_PASS): llama.cpp does not authenticate, but the
    OpenAI client refuses to start without a non-empty one."""
    return os.environ.get("LLM_PASS") or DEFAULT_LLM_API_KEY


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

"""Settings renamed or removed in v3, and the startup check that refuses them.

Pure module (stdlib only). v3 renamed many settings and keeps no aliases for
the old names, so a stale .env or values file would otherwise run quietly on
defaults: a brain pointed at a remote API, say, would lose its key and fall
back to the local server. Instead main.py calls check() before anything else
and exits, naming each old setting and what replaced it.

REMOVED is the one list. docs/upgrading-to-v3.md documents every row (its
env table is tested against this map), the chart's templates/_removed.tpl
does the same for values keys, and e2e/start_stack.sh unsets every name here
before starting core. A phase that renames a setting adds its row here.
"""
import os

# Old env name -> its replacement, or "removed: <why>" when nothing replaces it.
REMOVED = {
    # LLM server, roles, persona
    "MODEL": "LLM_MODEL",
    "LLM_PASS": "LLM_API_KEY",
    "REASONING_EFFORT": "PERSONA_REASONING_EFFORT",
    "LLM_MAX_TURNS": "PERSONA_MAX_TURNS",
    "LLM_CALL_TIMEOUT_SECONDS": "PERSONA_CALL_TIMEOUT_SECONDS",
    "LLM_CONTEXT_LENGTH": "removed: the context-window gauge is read from the LLM server's /props",
    "SANDBOX_MODEL": "BRAIN_MODEL",
    "SANDBOX_LLM_HOST": "BRAIN_LLM_HOST",
    "SANDBOX_LLM_API_KEY": "BRAIN_LLM_API_KEY",
    "SANDBOX_REQUEST_TIMEOUT_SECONDS": "BRAIN_CALL_TIMEOUT_SECONDS",
    "SANDBOX_MAX_RETRIES": "removed: the brain's client retries twice; BRAIN_CALL_TIMEOUT_SECONDS bounds one call",
    # The server workspace (phase 2)
    "SANDBOX_ENABLED": "removed: Docker is required, so the sandbox is always on",
    "SANDBOX_IMAGE": "WORKSPACE_IMAGE",
    "SANDBOX_MEMORY_LIMIT": "WORKSPACE_MEMORY_LIMIT",
    "SANDBOX_SNAPSHOT_MAX_BYTES": "WORKSPACE_SNAPSHOT_MAX_BYTES",
    "SANDBOX_SNAPSHOT_TTL_SECONDS": "WORKSPACE_SNAPSHOT_TTL_SECONDS",
    "SANDBOX_PERSIST_TIMEOUT_SECONDS": "WORKSPACE_SNAPSHOT_TIMEOUT_SECONDS",
    "SANDBOX_INPUT_MAX_BYTES": "WORKSPACE_INPUT_MAX_BYTES",
    # The brain (phase 3): new defaults, so an old value must be chosen again
    "SANDBOX_MAX_TURNS": "BRAIN_MAX_TURNS",
    "SANDBOX_TIMEOUT": "BRAIN_TIMEOUT",
    # Workflows (phase 5)
    "SANDBOX_ASK_USER_TIMEOUT": "WORKFLOW_ASK_TIMEOUT",
    # A reply the bot was not asked for has no brain since v3.4 (no limits to set)
    "BRAIN_UNPROMPTED_MAX_TURNS": "removed: a reply the bot was not asked for no longer uses the brain",
    "BRAIN_UNPROMPTED_TIMEOUT": "removed: a reply the bot was not asked for no longer uses the brain",
    # Queue
    "WORKER_COUNT": "QUEUE_WORKERS",
    # Content guard
    "OPENAI_API_KEY": "CONTENT_GUARD_API_KEY",
    "OPENAI_BASE_URL": "CONTENT_GUARD_BASE_URL",
    "MODERATION_MODEL": "CONTENT_GUARD_MODEL",
    "MODERATION_CACHE_TTL": "CONTENT_GUARD_CACHE_TTL_SECONDS",
    # docker-compose (.env reaches core through env_file, so a stale one is caught here too)
    "LLAMA_ARG_CACHE_TYPE_K": "LLAMA_CACHE_TYPE_K",
    "LLAMA_ARG_CACHE_TYPE_V": "LLAMA_CACHE_TYPE_V",
}

UPGRADE_DOC = "docs/upgrading-to-v3.md"


def find_removed(env=None) -> list[tuple[str, str]]:
    """(old, replacement) for every REMOVED name set in `env`, in REMOVED's order.

    "Set" means present and non-empty: compose writes `NAME=` for a variable
    it passes through unset, and an empty value configures nothing.
    """
    env = os.environ if env is None else env
    return [(old, new) for old, new in REMOVED.items() if str(env.get(old, "")).strip()]


def format_error(found: list[tuple[str, str]]) -> str:
    lines = ["These settings were renamed or removed in v3; update your .env or values file:"]
    for old, new in found:
        lines.append(f"  {old}: {new}" if new.startswith("removed") else f"  {old} -> {new}")
    lines.append(f"See {UPGRADE_DOC}.")
    return "\n".join(lines)


def missing_required(env=None) -> list[str]:
    """Required settings that are unset or empty."""
    env = os.environ if env is None else env
    return [name for name in ("LLM_MODEL",) if not str(env.get(name, "")).strip()]


def check(env=None) -> str | None:
    """The startup error to print before exiting, or None when the settings are fine."""
    problems = []
    found = find_removed(env)
    if found:
        problems.append(format_error(found))
    for name in missing_required(env):
        problems.append(f"{name} is required (the model the bot requests). See {UPGRADE_DOC}.")
    return "\n".join(problems) or None

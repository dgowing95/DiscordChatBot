"""Which model, server and key each LLM role uses.

Pure module (stdlib only). One place answers "which model/host/key/effort/
timeout does role X use", for the persona (the personality that answers in
chat), the brain (the agent in the server's workspace) and, from Tier 3, subagents;
and for the side jobs (research brief, image prompt rewrite, image check).

Every role falls back to the bot's own LLM server (LLM_MODEL / LLM_HOST /
LLM_API_KEY); a subagent falls back to the brain; a side job falls back to the
role that calls it. First non-empty value wins.

The host falls back first; the model and key then follow it:

  * The model and key only come from the parent (a subagent's brain, a side
    job's caller) when the job is on the parent's server. A side job pointed
    back at LLM_HOST while the brain is on OpenRouter asks for LLM_MODEL with
    LLM_API_KEY, not for the brain's hosted model.

Two rules keep a local key from leaking to someone else's server:

  * A key only falls back to a key for the SAME host. A brain pointed at
    OpenRouter with no BRAIN_LLM_API_KEY gets no key (and a startup warning),
    never the local Strata key. v2 fell back regardless of host.
  * LLM_EXTRA_HEADERS belong to LLM_HOST's server: a role or side job sends
    them only when its host is LLM_HOST.

Hosts are compared by host_key() (scheme, host, port, path), which is also
how llm_lanes keys its per-server slots.

It also holds the temperature parser, which every caller of the bot's own
model shares, and DEFAULT_TEMPERATURE.
"""
import logging
import math
import os
import re
from dataclasses import dataclass, field
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

PERSONA, BRAIN, SUBAGENT = "persona", "brain", "subagent"

DEFAULT_LLM_HOST = "http://llamacpp:8080"
# llama.cpp does not authenticate, but the OpenAI client refuses an empty key.
DEFAULT_LLM_API_KEY = "llamacpp"
NO_KEY = "none"  # what an OpenAI client is given for a role with no key
DEFAULT_REASONING_EFFORT = "medium"
DEFAULT_CALL_TIMEOUT = 600.0
DEFAULT_TEMPERATURE = 1.0

_PREFIX = {PERSONA: "PERSONA_", BRAIN: "BRAIN_", SUBAGENT: "SUBAGENT_"}
SIDE_JOB_PREFIXES = ("RESEARCH_", "IMAGE_PROMPT_", "IMAGE_REVIEW_")
_ROLE_SUFFIXES = ("MODEL", "LLM_HOST", "LLM_API_KEY", "REASONING_EFFORT", "CALL_TIMEOUT_SECONDS")
_SIDE_JOB_SUFFIXES = ("MODEL", "LLM_HOST", "LLM_API_KEY", "TIMEOUT")


def env_names() -> set[str]:
    """Every setting this module reads, for settings_docs_tests (the names
    are built from prefixes, which its source scan cannot see)."""
    names = {"LLM_MODEL", "LLM_HOST", "LLM_API_KEY", "LLM_EXTRA_HEADERS"}
    names.update(prefix + suffix for prefix in _PREFIX.values() for suffix in _ROLE_SUFFIXES)
    names.update(prefix + suffix for prefix in SIDE_JOB_PREFIXES for suffix in _SIDE_JOB_SUFFIXES)
    return names


@dataclass(frozen=True)
class RoleConfig:
    role: str
    model: str
    host: str  # base URL; callers append /v1
    api_key: str
    reasoning_effort: str
    call_timeout: float  # wall-clock bound for one model call
    host_key: str  # normalised host: which lane pool it uses
    extra_headers: dict = field(default_factory=dict, hash=False, compare=False)

    @property
    def client_api_key(self) -> str:
        """The key for an OpenAI client, which refuses an empty one: a role
        with no key (another server, none set) sends the word "none", never
        a key meant for a different server."""
        return self.api_key or NO_KEY

    def auth_headers(self) -> dict:
        """Headers for core's own HTTP calls to this host (/props, /v1/models),
        which go around the OpenAI client and so would otherwise send no key."""
        if not self.api_key:
            return dict(self.extra_headers)
        return {**self.extra_headers, "Authorization": f"Bearer {self.api_key}"}


def host_key(url: str) -> str:
    """A server's identity: scheme, host and port (the scheme's default when
    unstated) and path, without a trailing slash, lower-cased where URLs are
    case-insensitive. `http://llamacpp:8080/` == `http://LLAMACPP:8080`."""
    parts = urlsplit((url or "").strip())
    scheme = (parts.scheme or "http").lower()
    try:
        port = parts.port
    except ValueError:
        port = None
    port = port or {"http": 80, "https": 443}.get(scheme, 0)
    host = (parts.hostname or "").lower()
    return f"{scheme}://{host}:{port}{parts.path.rstrip('/')}"


def _get(env, name: str) -> str:
    return str(env.get(name, "") or "").strip()


def _timeout(env, name: str, default: float) -> float:
    try:
        value = float(_get(env, name))
    except ValueError:
        return default
    return value if value > 0 and math.isfinite(value) else default


def llm_host(env=None) -> str:
    env = os.environ if env is None else env
    return _get(env, "LLM_HOST") or DEFAULT_LLM_HOST


# An HTTP header name (RFC 9110 token characters).
_HEADER_NAME = re.compile(r"^[A-Za-z0-9!#$%&'*+.^_`|~-]+$")
# Each skipped entry is warned about once per setting value: role configs are
# resolved on every reply, which would repeat the same warning per message.
_warned: set = set()


def _warn_once(raw: str, message: str) -> None:
    if (raw, message) not in _warned:
        _warned.add((raw, message))
        logger.warning(message)


def llm_extra_headers(env=None) -> dict:
    """Extra headers for requests to LLM_HOST (LLM_EXTRA_HEADERS).

    Comma-separated `name=value` pairs, e.g.
    `LLM_EXTRA_HEADERS=x-opencode-session=my-bot`. For a hosted API that
    needs more than a key: OpenCode Go refuses every request without an
    `x-opencode-session` header (400 MissingSessionID), which is what the
    E2E job runs against. Unset (the default, and llama.cpp/Strata) sends
    nothing extra. A malformed entry, or an Authorization header (LLM_API_KEY
    sets that one), is skipped with a warning rather than failing a reply.
    """
    env = os.environ if env is None else env
    headers = {}
    raw = _get(env, "LLM_EXTRA_HEADERS")
    for position, entry in enumerate(raw.split(","), 1):
        entry = entry.strip()
        if not entry:
            continue
        name, sep, value = entry.partition("=")
        name, value = name.strip(), value.strip()
        # The value must be printable ASCII too: a pasted line break or a
        # curly quote would otherwise make httpx refuse to encode EVERY
        # request, failing every reply instead of skipping one header.
        if (not sep or not value or not _HEADER_NAME.match(name)
                or not (value.isascii() and value.isprintable())):
            # Only the position: a malformed entry may be a pasted key
            # ("Authorization Bearer sk-..."), which must not reach the logs.
            _warn_once(raw, f"LLM_EXTRA_HEADERS: skipping entry {position}, expected name=value")
            continue
        if name.lower() == "authorization":
            _warn_once(raw, "LLM_EXTRA_HEADERS: skipping Authorization; LLM_API_KEY sets it")
            continue
        headers[name] = value
    return headers


def _resolve(role: str, env, model: str, host: str, key: str, parent: "RoleConfig | None",
             effort: str, call_timeout: float) -> RoleConfig:
    """Fill a role's empty fields: from `parent` (a subagent's brain, a side
    job's caller) or else LLM_*. The key follows the same-host rule."""
    own_host = llm_host(env)
    base_host = parent.host if parent else own_host
    host = host or base_host
    hkey = host_key(host)
    on_parents_server = parent is not None and hkey == parent.host_key
    if not key:
        if on_parents_server:
            key = parent.api_key
        elif hkey == host_key(own_host):
            key = _get(env, "LLM_API_KEY") or DEFAULT_LLM_API_KEY
    model = model or (parent.model if on_parents_server else _get(env, "LLM_MODEL"))
    headers = llm_extra_headers(env) if hkey == host_key(own_host) else {}
    return RoleConfig(role=role, model=model, host=host, api_key=key, reasoning_effort=effort,
                      call_timeout=call_timeout, host_key=hkey, extra_headers=headers)


def role_config(role: str, env=None) -> RoleConfig:
    """The persona's, brain's or a subagent's connection (fallback table in
    the module docstring; the subagent falls back to the brain)."""
    env = os.environ if env is None else env
    prefix = _PREFIX[role]
    parent = role_config(BRAIN, env) if role == SUBAGENT else None
    effort = _get(env, prefix + "REASONING_EFFORT") or (
        parent.reasoning_effort if parent else DEFAULT_REASONING_EFFORT)
    call_timeout = _timeout(env, prefix + "CALL_TIMEOUT_SECONDS",
                            parent.call_timeout if parent else DEFAULT_CALL_TIMEOUT)
    return _resolve(role, env, _get(env, prefix + "MODEL"), _get(env, prefix + "LLM_HOST"),
                    _get(env, prefix + "LLM_API_KEY"), parent, effort, call_timeout)


def side_job_config(prefix: str, caller_role: str = BRAIN, default_timeout: float = 60.0,
                    env=None) -> RoleConfig:
    """A side job's connection (prefix RESEARCH_, IMAGE_PROMPT_ or
    IMAGE_REVIEW_): its own {prefix}MODEL / LLM_HOST / LLM_API_KEY /
    TIMEOUT, else the calling role's. Its role field is the prefix's job name."""
    if prefix not in SIDE_JOB_PREFIXES:
        raise ValueError(f"unknown side job {prefix!r}: add it to SIDE_JOB_PREFIXES")
    env = os.environ if env is None else env
    parent = role_config(caller_role, env)
    return _resolve(prefix.rstrip("_").lower(), env, _get(env, prefix + "MODEL"),
                    _get(env, prefix + "LLM_HOST"), _get(env, prefix + "LLM_API_KEY"), parent,
                    parent.reasoning_effort, _timeout(env, prefix + "TIMEOUT", default_timeout))


def role_warnings(env=None) -> list[str]:
    """Startup warnings for role and side-job settings that will not work as
    meant: a server of their own (not LLM_HOST's, not inherited) with no key
    or no model of its own."""
    env = os.environ if env is None else env
    own = host_key(llm_host(env))
    brain = role_config(BRAIN, env)
    checks = [(prefix, role_config(role, env), brain if role == SUBAGENT else None, role)
              for role, prefix in _PREFIX.items()]
    checks += [(prefix, side_job_config(prefix, env=env), brain, prefix.rstrip("_").lower())
               for prefix in SIDE_JOB_PREFIXES]
    warnings = []
    for prefix, config, parent, name in checks:
        if not _get(env, prefix + "LLM_HOST") or config.host_key == own:
            continue
        if parent is not None and config.host_key == parent.host_key:
            continue  # the parent's server: its key and model are inherited
        if not config.api_key:
            warnings.append(f"{prefix}LLM_HOST points at another server but {prefix}LLM_API_KEY is empty: "
                            f"{name} sends no key (the bot's own key is never sent to another server)")
        if not _get(env, prefix + "MODEL"):
            warnings.append(f"{prefix}LLM_HOST points at another server but {prefix}MODEL is empty: "
                            f"{name} will ask it for LLM_MODEL ({config.model or 'unset'})")
    return warnings


def distinct_hosts(env=None) -> list[RoleConfig]:
    """One config per distinct role host (persona, brain, subagent order):
    the servers whose /props the slot refresh reads, each with its own key."""
    seen, out = set(), []
    for role in (PERSONA, BRAIN, SUBAGENT):
        config = role_config(role, env)
        if config.host_key not in seen:
            seen.add(config.host_key)
            out.append(config)
    return out


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

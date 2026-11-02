"""Which model / server / key each LLM role and side job uses (classes/llm_roles.py).

Run from the repo root: `pytest core/tests/llm_roles_tests.py`
"""
import pytest

from classes import llm_roles
from classes.llm_roles import BRAIN, PERSONA, SUBAGENT, host_key, role_config, side_job_config

LOCAL = "http://llamacpp:8080"
REMOTE = "https://openrouter.ai/api"


def _env(**values):
    return {"LLM_MODEL": "local-model", **values}


# ---------------------- host_key ----------------------

@pytest.mark.parametrize("a, b", [
    ("http://llamacpp:8080/", "http://llamacpp:8080"),
    ("http://LLAMACPP:8080", "http://llamacpp:8080"),
    ("https://openrouter.ai/api/", "https://openrouter.ai:443/api"),
    ("http://strata", "http://strata:80"),
])
def test_host_key_normalises(a, b):
    assert host_key(a) == host_key(b)


def test_host_key_keeps_what_tells_servers_apart():
    assert host_key("http://llamacpp:8080") != host_key("http://llamacpp:8081")
    assert host_key("https://api.example/zen/go") != host_key("https://api.example/other")
    assert host_key("http://x:8080") != host_key("https://x:8080")


# ---------------------- fallbacks ----------------------

def test_every_role_falls_back_to_the_bots_own_llm():
    env = _env(LLM_HOST=LOCAL, LLM_API_KEY="local-key")
    for role in (PERSONA, BRAIN, SUBAGENT):
        config = role_config(role, env)
        assert (config.model, config.host, config.api_key) == ("local-model", LOCAL, "local-key"), role
        assert config.reasoning_effort == "medium"
        assert config.call_timeout == 600.0


def test_in_code_defaults():
    config = role_config(PERSONA, {})
    assert config.model == ""  # LLM_MODEL is required: settings_check stops startup without it
    assert config.host == llm_roles.DEFAULT_LLM_HOST
    assert config.api_key == llm_roles.DEFAULT_LLM_API_KEY


@pytest.mark.parametrize("role", [PERSONA, BRAIN])
def test_a_roles_own_settings_win(role):
    prefix = role.upper() + "_"
    env = _env(**{prefix + "MODEL": "own-model", prefix + "LLM_HOST": REMOTE,
                  prefix + "LLM_API_KEY": "own-key", prefix + "REASONING_EFFORT": "high",
                  prefix + "CALL_TIMEOUT_SECONDS": "45"})
    config = role_config(role, env)
    assert (config.model, config.host, config.api_key) == ("own-model", REMOTE, "own-key")
    assert (config.reasoning_effort, config.call_timeout) == ("high", 45.0)


def test_empty_or_blank_values_fall_back():
    env = _env(BRAIN_MODEL="  ", BRAIN_LLM_HOST="", BRAIN_CALL_TIMEOUT_SECONDS="")
    config = role_config(BRAIN, env)
    assert (config.model, config.host, config.call_timeout) == ("local-model", llm_roles.DEFAULT_LLM_HOST, 600.0)


@pytest.mark.parametrize("junk", ["nope", "0", "-5", "inf", "nan"])
def test_unusable_call_timeouts_fall_back(junk):
    assert role_config(PERSONA, _env(PERSONA_CALL_TIMEOUT_SECONDS=junk)).call_timeout == 600.0


def test_a_subagent_falls_back_to_the_brain():
    env = _env(BRAIN_MODEL="brain-model", BRAIN_LLM_HOST=REMOTE, BRAIN_LLM_API_KEY="brain-key",
               BRAIN_REASONING_EFFORT="low", BRAIN_CALL_TIMEOUT_SECONDS="30")
    config = role_config(SUBAGENT, env)
    assert (config.model, config.host, config.api_key) == ("brain-model", REMOTE, "brain-key")
    assert (config.reasoning_effort, config.call_timeout) == ("low", 30.0)


# ---------------------- the same-host key rule ----------------------

def test_a_remote_brain_without_its_own_key_gets_none():
    """v2 sent LLM_PASS to whatever SANDBOX_LLM_HOST was: a local Strata key
    to a third party."""
    env = _env(LLM_HOST=LOCAL, LLM_API_KEY="local-strata-key", BRAIN_LLM_HOST=REMOTE)
    config = role_config(BRAIN, env)
    assert config.api_key == ""
    assert config.client_api_key == llm_roles.NO_KEY
    assert "Authorization" not in config.auth_headers()


def test_a_brain_on_the_same_host_spelled_differently_keeps_the_key():
    env = _env(LLM_HOST=LOCAL, LLM_API_KEY="local-key", BRAIN_LLM_HOST="http://LLAMACPP:8080/")
    assert role_config(BRAIN, env).api_key == "local-key"


def test_remote_persona_and_local_brain_keep_their_own_keys():
    env = _env(LLM_HOST=LOCAL, LLM_API_KEY="local-key",
               PERSONA_LLM_HOST=REMOTE, PERSONA_MODEL="remote-model", PERSONA_LLM_API_KEY="persona-key")
    persona, brain = role_config(PERSONA, env), role_config(BRAIN, env)
    assert (persona.host, persona.api_key, persona.model) == (REMOTE, "persona-key", "remote-model")
    assert (brain.host, brain.api_key, brain.model) == (LOCAL, "local-key", "local-model")
    assert persona.host_key != brain.host_key


def test_extra_headers_only_go_to_llm_host():
    env = _env(LLM_HOST=LOCAL, LLM_EXTRA_HEADERS="x-session=s1", BRAIN_LLM_HOST=REMOTE)
    assert role_config(PERSONA, env).extra_headers == {"x-session": "s1"}
    assert role_config(BRAIN, env).extra_headers == {}
    assert role_config(PERSONA, env).auth_headers()["x-session"] == "s1"


# ---------------------- side jobs ----------------------

def test_side_jobs_fall_back_to_the_brain():
    env = _env(LLM_HOST=LOCAL, BRAIN_MODEL="brain-model", BRAIN_LLM_HOST=REMOTE, BRAIN_LLM_API_KEY="brain-key")
    for prefix in llm_roles.SIDE_JOB_PREFIXES:
        config = side_job_config(prefix, env=env)
        assert (config.model, config.host, config.api_key) == ("brain-model", REMOTE, "brain-key"), prefix


def test_a_side_job_on_its_own_host_never_gets_the_brains_key():
    env = _env(LLM_HOST=LOCAL, LLM_API_KEY="local-key", IMAGE_REVIEW_LLM_HOST=REMOTE)
    config = side_job_config("IMAGE_REVIEW_", env=env)
    assert config.host == REMOTE and config.api_key == ""


def test_a_side_job_back_on_llm_host_gets_the_local_key():
    # The brain is remote with its own key; the job points back at LLM_HOST.
    env = _env(LLM_HOST=LOCAL, LLM_API_KEY="local-key", BRAIN_LLM_HOST=REMOTE,
               BRAIN_LLM_API_KEY="brain-key", RESEARCH_LLM_HOST=LOCAL)
    assert side_job_config("RESEARCH_", env=env).api_key == "local-key"


def test_side_job_timeouts():
    assert side_job_config("RESEARCH_", default_timeout=90.0, env=_env()).call_timeout == 90.0
    assert side_job_config("RESEARCH_", env=_env(RESEARCH_TIMEOUT="12")).call_timeout == 12.0


def test_an_unknown_side_job_is_refused():
    with pytest.raises(ValueError):
        side_job_config("SOMETHING_", env=_env())


# ---------------------- warnings and hosts ----------------------

def test_role_warnings_name_a_remote_role_without_key_or_model():
    env = _env(LLM_HOST=LOCAL, BRAIN_LLM_HOST=REMOTE)
    warnings = llm_roles.role_warnings(env)
    assert any("BRAIN_LLM_API_KEY is empty" in w for w in warnings)
    assert any("BRAIN_MODEL is empty" in w for w in warnings)
    # The subagent inherits the brain's host but sets none of its own: no warning.
    assert not any(w.startswith("SUBAGENT_") for w in warnings)


def test_no_warnings_for_a_fully_set_remote_role_or_a_local_one():
    assert llm_roles.role_warnings(_env(LLM_HOST=LOCAL)) == []
    env = _env(LLM_HOST=LOCAL, BRAIN_LLM_HOST=REMOTE, BRAIN_MODEL="m", BRAIN_LLM_API_KEY="k")
    assert llm_roles.role_warnings(env) == []
    # A role "on another host" that is really LLM_HOST spelled differently.
    assert llm_roles.role_warnings(_env(LLM_HOST=LOCAL, BRAIN_LLM_HOST=LOCAL + "/")) == []


def test_distinct_hosts_lists_the_persona_first_once_per_server():
    env = _env(LLM_HOST=LOCAL, BRAIN_LLM_HOST=REMOTE, BRAIN_LLM_API_KEY="k")
    hosts = llm_roles.distinct_hosts(env)
    assert [c.role for c in hosts] == [PERSONA, BRAIN]
    assert [c.host for c in hosts] == [LOCAL, REMOTE]
    assert len(llm_roles.distinct_hosts(_env(LLM_HOST=LOCAL))) == 1


def test_env_names_cover_every_role_and_side_job():
    names = llm_roles.env_names()
    for name in ("LLM_MODEL", "PERSONA_LLM_API_KEY", "BRAIN_REASONING_EFFORT", "SUBAGENT_MODEL",
                 "RESEARCH_TIMEOUT", "IMAGE_REVIEW_LLM_HOST", "PERSONA_CALL_TIMEOUT_SECONDS"):
        assert name in names, name


@pytest.mark.parametrize("raw, expected", [
    (None, 1.0), (False, 1.0), ("", 1.0), ("0", 0.0), (0.0, 0.0), ("0.7", 0.7), ("junk", 1.0),
])
def test_parse_temperature(raw, expected):
    assert llm_roles.parse_temperature(raw) == expected

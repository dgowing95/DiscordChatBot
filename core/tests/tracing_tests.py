"""Agent runs are never uploaded to OpenAI's trace dashboard.

The agents SDK traces by default whenever OPENAI_API_KEY is set (it is, for
the moderation check), and a trace holds the prompt: Discord users'
messages. text_llm_handler and sandbox_agent switch it off at import, and
nothing may switch it back on.

To run this pytest file from the command line, use:
    pytest core/tests/tracing_tests.py
"""
from agents import RunConfig, trace
from agents.tracing.traces import NoOpTrace


def test_tracing_is_off_even_when_the_environment_asks_for_it(monkeypatch):
    monkeypatch.setenv("OPENAI_AGENTS_DISABLE_TRACING", "0")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    import classes.sandbox_agent  # noqa: F401
    import classes.text_llm_handler  # noqa: F401
    assert isinstance(trace("a run"), NoOpTrace)
    assert isinstance(trace("a run", disabled=RunConfig().tracing_disabled), NoOpTrace)

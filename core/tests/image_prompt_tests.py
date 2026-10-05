import httpx2
import pytest
from openai import APITimeoutError
from unittest.mock import AsyncMock, MagicMock, patch

# Imported as `classes.X`, the same name the app uses (it runs with cwd=core/,
# and pyproject.toml puts core/ on the test path).
from classes import image_prompt

# To run this pytest file from the command line, use:
# PYTHONPATH=$(pwd) pytest core/tests/image_prompt_tests.py


def _mock_client(content):
    """An AsyncOpenAI stand-in whose completion returns `content`."""
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]

    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)
    return client


@pytest.fixture(autouse=True)
def _reset_module_state():
    """image_prompt's SideLLM keeps two things for the life of the process: its
    AsyncOpenAI client (one connection pool) and whether the backend accepted
    the thinking-disabled option. Reset both, so an env change in one test is
    not read from a client another already built, and a test that trips the
    retry does not leave it tripped for the next."""
    image_prompt._llm.reset()
    yield
    image_prompt._llm.reset()


# ---------------------- parse_rewrite ----------------------

def test_parse_plain_json():
    assert image_prompt.parse_rewrite(
        '{"prompt": "a red fox", "negative_prompt": "cartoon"}'
    ) == ("a red fox", "cartoon")


def test_parse_strips_reasoning_block():
    """The local model is a reasoning model; the JSON follows a <think> block."""
    content = ('<think>The user wants a fox. Foxes are orange.</think>\n'
               '{"prompt": "a red fox", "negative_prompt": ""}')
    assert image_prompt.parse_rewrite(content) == ("a red fox", "")


def test_parse_ignores_fence_and_surrounding_prose():
    content = ('Here you go:\n```json\n'
               '{"prompt": "a red fox", "negative_prompt": "blurry"}\n```\nHope that helps!')
    assert image_prompt.parse_rewrite(content) == ("a red fox", "blurry")


def test_parse_defaults_missing_negative_prompt():
    assert image_prompt.parse_rewrite('{"prompt": "a red fox"}') == ("a red fox", "")


@pytest.mark.parametrize("content", [
    "",
    None,
    "no json here at all",
    "{not valid json}",
    '["a", "list"]',
    '{"negative_prompt": "cartoon"}',   # no prompt at all
    '{"prompt": "   "}',                # blank prompt
])
def test_parse_rejects_unusable(content):
    assert image_prompt.parse_rewrite(content) is None


def test_parse_cleans_the_negative_prompt():
    content = '{"prompt": "a knight in a visored helmet", "negative_prompt": "helmet, knights, cartoon"}'
    assert image_prompt.parse_rewrite(content) == ("a knight in a visored helmet", "cartoon")


# ---------------------- clean_negative ----------------------
# The inputs are what prod's rewriter actually produced (core logs, 2026-10-04/05).

def test_clean_negative_drops_what_the_prompt_asks_for():
    """A soldier in a narrow-visor helmet got "helmet, visor, face, arms" in
    her negative prompt -- the image model was told to avoid its own subject."""
    prompt = ("massive round female soldier, stern unamused face, narrow visor slit "
              "helmet, full tactical military gear, heavy armored plate mail")
    negative = "thin, skinny, helmet, visor, face, arms, wings, plate mail, gear"
    assert image_prompt.clean_negative(negative, prompt) == "thin, skinny, arms, wings"


def test_clean_negative_keeps_the_opposite_of_the_key_detail():
    """The term that matters most for "stick arms" shares the noun with it."""
    prompt = "extremely thin stick arms, obese woman at a kitchen table"
    assert image_prompt.clean_negative("thick arms, muscular arms, chubby arms", prompt) == \
        "thick arms, muscular arms, chubby arms"


def test_clean_negative_drops_negated_terms():
    """It restated the prompt as "no orange sphere body, no narrow visor slit helmet, ..."."""
    prompt = "massive round soldier, giant orange sphere body"
    negative = "no orange sphere body, not cinematic, without rim lighting, blurry"
    assert image_prompt.clean_negative(negative, prompt) == "blurry"


def test_clean_negative_ignores_plurals_and_case():
    assert image_prompt.clean_negative("Helmets, BOOTS", "a soldier with a helmet") == "BOOTS"


def test_clean_negative_dedupes_and_caps_a_runaway_list():
    """Prod logged negatives of 262-624 CLIP tokens: the model looping through
    "bad crust, bad cheese, bad pepperoni, ..."."""
    negative = ", ".join(["blurry", "blurry"] + [f"term{i}" for i in range(40)])
    kept = image_prompt.clean_negative(negative, "a red fox").split(", ")
    assert kept[:2] == ["blurry", "term0"]
    assert len(kept) == image_prompt.MAX_NEGATIVE_TERMS


def test_clean_negative_handles_empty_input():
    assert image_prompt.clean_negative("", "a red fox") == ""
    assert image_prompt.clean_negative(" , ,, ", "a red fox") == ""


def test_system_prompt_has_no_copyable_exclusion_list():
    """The model pasted these example terms into nearly every negative prompt,
    "people" included -- for pictures of people."""
    for literal in ('"people"', "extra fingers", "bent walls"):
        assert literal not in image_prompt.SYSTEM_PROMPT


# ---------------------- build_image_prompt ----------------------

@pytest.mark.asyncio
async def test_build_returns_rewritten_pair(monkeypatch):
    monkeypatch.delenv("IMAGE_PROMPT_REWRITE_ENABLED", raising=False)
    client = _mock_client('{"prompt": "a red fox, sharp fur", "negative_prompt": "cartoon"}')
    with patch.object(image_prompt._llm, "_get_client", return_value=client):
        assert await image_prompt.build_image_prompt("draw me a fox") == (
            "a red fox, sharp fur", "cartoon")

    kwargs = client.chat.completions.create.await_args.kwargs
    assert kwargs["messages"][0]["role"] == "system"
    assert kwargs["messages"][1] == {"role": "user", "content": "draw me a fox"}


@pytest.mark.asyncio
async def test_build_skips_the_call_when_disabled(monkeypatch):
    monkeypatch.setenv("IMAGE_PROMPT_REWRITE_ENABLED", "0")
    client = _mock_client('{"prompt": "should not be used"}')
    with patch.object(image_prompt._llm, "_get_client", return_value=client):
        assert await image_prompt.build_image_prompt("draw me a fox") == ("draw me a fox", "")
    client.chat.completions.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_build_disables_thinking_on_the_first_try(monkeypatch):
    """Prod's model is a reasoning model. Left thinking, it spends the whole
    token budget before the JSON and the rewrite falls soft on every call."""
    monkeypatch.delenv("IMAGE_PROMPT_REWRITE_ENABLED", raising=False)
    client = _mock_client('{"prompt": "a red fox"}')
    with patch.object(image_prompt._llm, "_get_client", return_value=client):
        await image_prompt.build_image_prompt("draw me a fox")

    kwargs = client.chat.completions.create.await_args.kwargs
    assert kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


@pytest.mark.asyncio
async def test_build_retries_once_without_the_thinking_option(monkeypatch):
    """chat_template_kwargs is not in the OpenAI schema, so a backend may
    reject it. That must cost one retry per process, not one per image."""
    monkeypatch.delenv("IMAGE_PROMPT_REWRITE_ENABLED", raising=False)
    client = _mock_client('{"prompt": "a red fox"}')
    ok = client.chat.completions.create.return_value
    client.chat.completions.create = AsyncMock(
        side_effect=[Exception("unknown field chat_template_kwargs"), ok, ok])

    with patch.object(image_prompt._llm, "_get_client", return_value=client):
        assert await image_prompt.build_image_prompt("draw me a fox") == ("a red fox", "")
        # The second call goes straight through without the option.
        assert await image_prompt.build_image_prompt("draw me a fox") == ("a red fox", "")

    calls = client.chat.completions.create.await_args_list
    assert len(calls) == 3
    assert "extra_body" in calls[0].kwargs
    assert "extra_body" not in calls[1].kwargs
    assert "extra_body" not in calls[2].kwargs
    assert image_prompt._llm.send_no_thinking is False


@pytest.mark.asyncio
async def test_build_falls_back_when_the_call_fails(monkeypatch):
    """Fail-soft, like content_guard: a rewriter that is down must cost the
    prompt quality, not the image."""
    monkeypatch.delenv("IMAGE_PROMPT_REWRITE_ENABLED", raising=False)
    client = _mock_client("")
    timeout = APITimeoutError(request=httpx2.Request("POST", "http://llamacpp:8080/v1"))
    client.chat.completions.create = AsyncMock(side_effect=timeout)
    with patch.object(image_prompt._llm, "_get_client", return_value=client):
        assert await image_prompt.build_image_prompt("draw me a fox") == ("draw me a fox", "")

    # A slow backend is not a rejected option: no retry (which would just spend
    # the timeout twice) and, crucially, thinking stays disabled for next time.
    assert client.chat.completions.create.await_count == 1
    assert image_prompt._llm.send_no_thinking is True


@pytest.mark.asyncio
async def test_build_keeps_disabling_thinking_when_the_retry_also_fails(monkeypatch):
    """Only a retry that SUCCEEDS identifies the option as the culprit. A
    backend having a bad minute must not re-enable thinking for the process --
    thinking left on is what breaks this feature."""
    monkeypatch.delenv("IMAGE_PROMPT_REWRITE_ENABLED", raising=False)
    client = _mock_client("")
    client.chat.completions.create = AsyncMock(side_effect=Exception("upstream 500"))
    with patch.object(image_prompt._llm, "_get_client", return_value=client):
        assert await image_prompt.build_image_prompt("draw me a fox") == ("draw me a fox", "")

    assert client.chat.completions.create.await_count == 2
    assert image_prompt._llm.send_no_thinking is True


@pytest.mark.asyncio
async def test_build_falls_back_on_unusable_output(monkeypatch):
    monkeypatch.delenv("IMAGE_PROMPT_REWRITE_ENABLED", raising=False)
    client = _mock_client("I'd rather not answer in JSON.")
    with patch.object(image_prompt._llm, "_get_client", return_value=client):
        assert await image_prompt.build_image_prompt("draw me a fox") == ("draw me a fox", "")


@pytest.mark.asyncio
async def test_build_ignores_an_empty_request(monkeypatch):
    monkeypatch.delenv("IMAGE_PROMPT_REWRITE_ENABLED", raising=False)
    client = _mock_client('{"prompt": "should not be used"}')
    with patch.object(image_prompt._llm, "_get_client", return_value=client):
        assert await image_prompt.build_image_prompt("   ") == ("", "")
    client.chat.completions.create.assert_not_awaited()


# ---------------------- connection settings ----------------------

def test_rewrite_settings_fall_back_to_the_bots_own(monkeypatch):
    for name in ("IMAGE_PROMPT_MODEL", "IMAGE_PROMPT_LLM_HOST", "IMAGE_PROMPT_LLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MODEL", "qwen3:27b")
    monkeypatch.setenv("LLM_HOST", "http://llamacpp:8080")
    monkeypatch.setenv("LLM_PASS", "llamacpp")

    assert image_prompt.rewrite_model() == "qwen3:27b"
    assert image_prompt.rewrite_llm_host() == "http://llamacpp:8080"
    assert image_prompt.rewrite_llm_api_key() == "llamacpp"


def test_rewrite_settings_override_the_bots_own(monkeypatch):
    """So the rewrite can run on OpenRouter while chat stays local, exactly as
    SANDBOX_MODEL/SANDBOX_LLM_HOST already allow for the sandbox agent."""
    monkeypatch.setenv("MODEL", "qwen3:27b")
    monkeypatch.setenv("LLM_HOST", "http://llamacpp:8080")
    monkeypatch.setenv("IMAGE_PROMPT_MODEL", "deepseek/deepseek-v4-flash-0731")
    monkeypatch.setenv("IMAGE_PROMPT_LLM_HOST", "https://openrouter.ai/api")
    monkeypatch.setenv("IMAGE_PROMPT_LLM_API_KEY", "sk-or-test")

    assert image_prompt.rewrite_model() == "deepseek/deepseek-v4-flash-0731"
    assert image_prompt.rewrite_llm_host() == "https://openrouter.ai/api"
    assert image_prompt.rewrite_llm_api_key() == "sk-or-test"


def test_rewrite_timeout_falls_back_on_junk(monkeypatch):
    monkeypatch.setenv("IMAGE_PROMPT_TIMEOUT", "90")
    assert image_prompt.rewrite_timeout() == 90
    for bad in ("", "   ", "not-a-number", "0", "-5"):
        monkeypatch.setenv("IMAGE_PROMPT_TIMEOUT", bad)
        assert image_prompt.rewrite_timeout() == image_prompt.DEFAULT_TIMEOUT


def test_rewrite_enabled_env_values(monkeypatch):
    for raw, expected in [("1", True), ("true", True), ("yes", True), ("ON", True),
                          ("0", False), ("false", False), ("no", False),
                          ("off", False), ("", False)]:
        monkeypatch.setenv("IMAGE_PROMPT_REWRITE_ENABLED", raw)
        assert image_prompt.rewrite_enabled() is expected, raw
    monkeypatch.delenv("IMAGE_PROMPT_REWRITE_ENABLED", raising=False)
    assert image_prompt.rewrite_enabled() is True

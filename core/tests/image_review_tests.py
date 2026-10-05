import base64
import io

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from PIL import Image

# Imported as `classes.X`, the same name the app uses (it runs with cwd=core/,
# and pyproject.toml puts core/ on the test path).
from classes import image_review
from classes.image_review import ImageReview

# To run this pytest file from the command line, use:
# PYTHONPATH=$(pwd) pytest core/tests/image_review_tests.py


def _png(size=(1024, 1024)):
    buf = io.BytesIO()
    Image.new("RGB", size, (200, 40, 40)).save(buf, format="PNG")
    return buf.getvalue()


def _mock_client(content):
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
    image_review._llm.reset()
    yield
    image_review._llm.reset()


# ---------------------- parse_review ----------------------

def test_parse_a_full_answer():
    review = image_review.parse_review(
        '{"shows": "An obese woman with ordinary arms.", "missing": ["very thin arms"]}')
    assert review == ImageReview("An obese woman with ordinary arms.", ("very thin arms",))
    assert review.matches is False


def test_parse_nothing_missing_is_a_match():
    review = image_review.parse_review('<think>hm</think>{"shows": "A red fox.", "missing": []}')
    assert review.matches is True


def test_parse_accepts_a_single_string_and_caps_the_list():
    assert image_review.parse_review('{"shows": "x", "missing": "a hat"}').missing == ("a hat",)
    many = image_review.parse_review('{"shows": "x", "missing": ["a", "b", "c", "d", "e", " "]}')
    assert many.missing == ("a", "b", "c", "d")


@pytest.mark.parametrize("content", [
    "", None, "no json", "{broken", '["list"]',
    '{"missing": ["a"]}',                     # no description of the image
    '{"shows": "x", "missing": {"a": 1}}',
])
def test_parse_rejects_unusable(content):
    assert image_review.parse_review(content) is None


# ---------------------- prefer_second ----------------------

def test_fewer_misses_wins():
    one, two = ImageReview("x", ("a",)), ImageReview("x", ("a", "b"))
    assert image_review.prefer_second(two, one) is True
    assert image_review.prefer_second(one, two) is False


def test_a_tie_goes_to_the_retry():
    """It was made with the misses stressed."""
    assert image_review.prefer_second(ImageReview("x", ("a",)), ImageReview("y", ("b",))) is True


def test_a_checked_attempt_beats_an_unchecked_one():
    """Only a checked image can be described truthfully."""
    checked = ImageReview("x", ("a", "b", "c"))
    assert image_review.prefer_second(None, checked) is True
    assert image_review.prefer_second(checked, None) is False
    assert image_review.prefer_second(None, None) is True


# ---------------------- what the reviewer is told ----------------------

def test_request_text_leads_with_the_persons_own_words():
    text = image_review.review_request_text("make big sally BIGGER", "a colossal soldier", False)
    assert text.index("make big sally BIGGER") < text.index("a colossal soldier")
    assert "edit" not in text


def test_request_text_without_a_person_uses_the_description():
    text = image_review.review_request_text("", "a red fox", True)
    assert "asked" not in text and "a red fox" in text
    assert "change is clearly visible" in text


def test_encode_for_review_shrinks_to_a_jpeg():
    data_url = image_review.encode_for_review(_png((1024, 768)))
    assert data_url.startswith("data:image/jpeg;base64,")
    image = Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1])))
    assert image.format == "JPEG"
    assert max(image.size) == image_review.REVIEW_MAX_SIDE
    assert image.size == (640, 480)


# ---------------------- review_image ----------------------

@pytest.mark.asyncio
async def test_review_sends_the_image_and_returns_the_verdict(monkeypatch):
    monkeypatch.delenv("IMAGE_REVIEW_ENABLED", raising=False)
    client = _mock_client('{"shows": "A red fox in snow.", "missing": []}')
    with patch.object(image_review._llm, "_get_client", return_value=client), \
         patch.object(image_review, "inc_image_review") as counted:
        review = await image_review.review_image(_png(), "draw a fox", "a red fox", False)

    assert review == ImageReview("A red fox in snow.", ())
    counted.assert_called_once_with("match")
    kwargs = client.chat.completions.create.await_args.kwargs
    parts = kwargs["messages"][1]["content"]
    assert parts[0]["type"] == "text" and "draw a fox" in parts[0]["text"]
    assert parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    # thinking off, like the rewrite: a reasoning model would spend the budget first
    assert kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


@pytest.mark.asyncio
async def test_review_counts_a_miss(monkeypatch):
    monkeypatch.delenv("IMAGE_REVIEW_ENABLED", raising=False)
    client = _mock_client('{"shows": "A woman.", "missing": ["very thin arms"]}')
    with patch.object(image_review._llm, "_get_client", return_value=client), \
         patch.object(image_review, "inc_image_review") as counted:
        review = await image_review.review_image(_png(), "", "a woman with stick arms")
    assert review.missing == ("very thin arms",)
    counted.assert_called_once_with("mismatch")


@pytest.mark.asyncio
async def test_review_off_makes_no_call(monkeypatch):
    monkeypatch.setenv("IMAGE_REVIEW_ENABLED", "0")
    client = _mock_client("{}")
    with patch.object(image_review._llm, "_get_client", return_value=client):
        assert await image_review.review_image(_png(), "", "a red fox") is None
    client.chat.completions.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_review_fails_soft(monkeypatch):
    """A reviewer that is down costs the check, never the image."""
    monkeypatch.delenv("IMAGE_REVIEW_ENABLED", raising=False)
    client = _mock_client("")
    client.chat.completions.create = AsyncMock(side_effect=Exception("upstream 500"))
    with patch.object(image_review._llm, "_get_client", return_value=client), \
         patch.object(image_review, "inc_image_review") as counted:
        assert await image_review.review_image(_png(), "", "a red fox") is None
    counted.assert_called_once_with("error")


@pytest.mark.asyncio
async def test_review_of_an_unreadable_image_fails_soft(monkeypatch):
    monkeypatch.delenv("IMAGE_REVIEW_ENABLED", raising=False)
    client = _mock_client('{"shows": "x", "missing": []}')
    with patch.object(image_review._llm, "_get_client", return_value=client):
        assert await image_review.review_image(b"not a png", "", "a red fox") is None
    client.chat.completions.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_review_with_unusable_output_fails_soft(monkeypatch):
    monkeypatch.delenv("IMAGE_REVIEW_ENABLED", raising=False)
    client = _mock_client("It looks great!")
    with patch.object(image_review._llm, "_get_client", return_value=client):
        assert await image_review.review_image(_png(), "", "a red fox") is None


# ---------------------- settings ----------------------

def test_retries_default_to_one_and_are_clamped(monkeypatch):
    monkeypatch.delenv("IMAGE_REVIEW_RETRIES", raising=False)
    assert image_review.review_retries() == 1
    for raw, expected in [("0", 0), ("2", 2), ("9", 2), ("-3", 0), ("junk", 1)]:
        monkeypatch.setenv("IMAGE_REVIEW_RETRIES", raw)
        assert image_review.review_retries() == expected, raw


def test_review_settings_fall_back_to_the_bots_own(monkeypatch):
    for name in ("IMAGE_REVIEW_MODEL", "IMAGE_REVIEW_LLM_HOST", "IMAGE_REVIEW_LLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MODEL", "qwen3:27b")
    monkeypatch.setenv("LLM_HOST", "http://llamacpp:8080")
    monkeypatch.setenv("LLM_PASS", "llamacpp")
    assert image_review.review_model() == "qwen3:27b"
    assert image_review.review_llm_host() == "http://llamacpp:8080"
    assert image_review.review_llm_api_key() == "llamacpp"


def test_review_settings_override_the_bots_own(monkeypatch):
    monkeypatch.setenv("MODEL", "qwen3:27b")
    monkeypatch.setenv("IMAGE_REVIEW_MODEL", "some/vision-model")
    monkeypatch.setenv("IMAGE_REVIEW_LLM_HOST", "https://openrouter.ai/api")
    assert image_review.review_model() == "some/vision-model"
    assert image_review.review_llm_host() == "https://openrouter.ai/api"


def test_review_timeout_falls_back_on_junk(monkeypatch):
    monkeypatch.setenv("IMAGE_REVIEW_TIMEOUT", "45")
    assert image_review.review_timeout() == 45
    for bad in ("", "nope", "0", "-1"):
        monkeypatch.setenv("IMAGE_REVIEW_TIMEOUT", bad)
        assert image_review.review_timeout() == image_review.DEFAULT_TIMEOUT

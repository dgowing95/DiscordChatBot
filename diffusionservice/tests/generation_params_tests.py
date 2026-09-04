"""Tests for the diffusion service's stdlib-only half.

main.py itself is never imported here (it needs torch/diffusers, which CI does
not install) -- generation_params.py exists precisely so the guidance and
negative-prompt rules can be checked without a GPU.
"""
import pytest

from generation_params import (
    env_flag,
    env_optional_float,
    is_distilled,
    merge_negative_prompt,
    resolve_guidance,
    sanitize_for_compel,
)


# ---------------------- is_distilled ----------------------

@pytest.mark.parametrize("model_id", [
    "stabilityai/sd-turbo",
    "stabilityai/sdxl-turbo",
    "ByteDance/SDXL-Lightning",
    "latent-consistency/lcm-lora-sdxl",
    "ByteDance/Hyper-SD",
    "SOME/UPPERCASE-TURBO",
])
def test_distilled_models_are_detected(model_id):
    assert is_distilled(model_id) is True


@pytest.mark.parametrize("model_id", [
    "RunDiffusion/Juggernaut-XL-v9",
    "Lykon/DreamShaper",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "",
    None,
])
def test_ordinary_models_are_not_distilled(model_id):
    assert is_distilled(model_id) is False


# ---------------------- resolve_guidance ----------------------

def test_guidance_unset_defers_to_the_pipeline():
    """None means "omit guidance_scale", not "use some number" -- so an unset
    IMAGE_GUIDANCE keeps whatever the pipeline class has always applied."""
    assert resolve_guidance("RunDiffusion/Juggernaut-XL-v9") is None


def test_guidance_env_default_is_used():
    assert resolve_guidance("RunDiffusion/Juggernaut-XL-v9", None, "5.0") == 5.0


def test_guidance_request_beats_env():
    assert resolve_guidance("RunDiffusion/Juggernaut-XL-v9", 7.5, "5.0") == 7.5


def test_guidance_junk_falls_through():
    assert resolve_guidance("Lykon/DreamShaper", "not-a-number", "6") == 6.0
    assert resolve_guidance("Lykon/DreamShaper", "not-a-number", "also-junk") is None


def test_distilled_guidance_is_pinned_to_zero():
    """sd-turbo is trained without classifier-free guidance; the service used
    to leave it at the pipeline's 7.5, which washes the image out."""
    assert resolve_guidance("stabilityai/sd-turbo") == 0.0
    assert resolve_guidance("stabilityai/sd-turbo", 7.5, "5.0") == 0.0


# ---------------------- merge_negative_prompt ----------------------

def test_negative_request_comes_before_the_baseline():
    """CLIP weights earlier tokens more, and the scene-specific terms matter
    more than the boilerplate."""
    assert merge_negative_prompt("cartoon", "blurry, watermark") == "cartoon, blurry, watermark"


def test_negative_handles_either_side_missing():
    assert merge_negative_prompt("cartoon", "") == "cartoon"
    assert merge_negative_prompt(None, "blurry") == "blurry"
    assert merge_negative_prompt(None, None) == ""
    assert merge_negative_prompt("  ", "   ") == ""


def test_negative_drops_a_duplicate_baseline():
    assert merge_negative_prompt("blurry", "BLURRY") == "blurry"


def test_negative_trims_stray_commas():
    assert merge_negative_prompt("cartoon,", " blurry ") == "cartoon, blurry"


def test_negative_is_empty_for_distilled_models():
    """diffusers skips the unconditional branch below CFG 1, so a negative
    prompt there is inert and must not be sent."""
    assert merge_negative_prompt("cartoon", "blurry", distilled=True) == ""


# ---------------------- sanitize_for_compel ----------------------

def test_sanitize_leaves_ordinary_prose_alone():
    text = "a red fox standing in tall grass, golden hour, 85mm lens"
    assert sanitize_for_compel(text) == text


def test_sanitize_strips_weights_and_grouping():
    assert sanitize_for_compel("a (red)1.3 fox, blurry++") == "a red fox, blurry"


def test_sanitize_survives_unbalanced_parentheses():
    """An unbalanced paren raises out of compel's parser, which would silently
    drop the request back onto the truncating code path."""
    assert sanitize_for_compel("unbalanced ( paren here") == "unbalanced paren here"


def test_sanitize_removes_method_call_syntax():
    out = sanitize_for_compel('("a cat", "a dog").blend(1, 0.5) in a field')
    assert ".blend" not in out and "(" not in out and ")" not in out
    assert "in a field" in out


def test_sanitize_strips_weights_before_punctuation():
    """Checked against compel's own parser: it reads "blurry++, deformed--" as
    a 1.21 and a 0.81 weight, and a comma-separated negative prompt is where an
    LLM is most likely to write one."""
    assert sanitize_for_compel("blurry++, deformed--, extra limbs") ==         "blurry, deformed, extra limbs"


def test_sanitize_keeps_quoted_text():
    """Words that must be legible go in quotes, and compel's parser is happy
    with them -- balanced or not -- so they stay."""
    assert sanitize_for_compel('red arrow with bold white text "the slot"') ==         'red arrow with bold white text "the slot"'


def test_sanitize_splits_hyphenated_compounds():
    """compel reads the hyphen in "low-quality" as a down-weight on "low"."""
    assert sanitize_for_compel("high-resolution state-of-the-art") == \
        "high resolution state of the art"


def test_sanitize_handles_empty_input():
    assert sanitize_for_compel("") == ""
    assert sanitize_for_compel(None) == ""


# ---------------------- env helpers ----------------------

def test_env_optional_float_is_none_when_unusable(monkeypatch):
    monkeypatch.setenv("IMAGE_GUIDANCE", "5.5")
    assert env_optional_float("IMAGE_GUIDANCE") == 5.5
    for bad in ("", "   ", "high"):
        monkeypatch.setenv("IMAGE_GUIDANCE", bad)
        assert env_optional_float("IMAGE_GUIDANCE") is None
    monkeypatch.delenv("IMAGE_GUIDANCE")
    assert env_optional_float("IMAGE_GUIDANCE") is None


def test_env_flag_matches_the_charts_true_false(monkeypatch):
    """The Helm configmap renders booleans as "true"/"false" strings."""
    for raw, expected in [("true", True), ("1", True), ("on", True),
                          ("false", False), ("0", False), ("off", False), ("", False)]:
        monkeypatch.setenv("IMAGE_LONG_PROMPT", raw)
        assert env_flag("IMAGE_LONG_PROMPT", True) is expected, raw
    monkeypatch.delenv("IMAGE_LONG_PROMPT")
    assert env_flag("IMAGE_LONG_PROMPT", True) is True

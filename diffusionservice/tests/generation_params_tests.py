"""Tests for the diffusion service's stdlib-only half.

main.py itself is never imported here (it needs torch/diffusers, which CI does
not install) -- generation_params.py exists precisely so the guidance and
negative-prompt rules can be checked without a GPU.
"""
import pytest

from generation_params import (
    component_dtypes,
    env_choice,
    env_flag,
    env_optional_float,
    is_distilled,
    merge_negative_prompt,
    pipeline_family,
    prompt_style,
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


def test_klein_guidance_is_pinned_to_one():
    """The klein pipeline defaults to 4.0 and warns on every call when a
    distilled model gets more than 1; 0.0 is the SD family's pin, not its."""
    klein = "black-forest-labs/FLUX.2-klein-4B"
    assert resolve_guidance(klein, None, None, family="flux2", distilled=True) == 1.0
    assert resolve_guidance(klein, 4.0, "3.5", family="flux2", distilled=True) == 1.0


def test_undistilled_klein_takes_the_configured_guidance():
    base = "black-forest-labs/FLUX.2-klein-base-4B"
    assert resolve_guidance(base, None, "4.0", family="flux2", distilled=False) == 4.0
    assert resolve_guidance(base, None, None, family="flux2", distilled=False) is None


# ---------------------- families ----------------------

def test_klein_is_its_own_family():
    assert pipeline_family("Flux2KleinPipeline") == "flux2"


@pytest.mark.parametrize("class_name", [
    "StableDiffusionXLPipeline", "StableDiffusionPipeline", "", None,
])
def test_everything_else_keeps_the_sd_behaviour(class_name):
    """Unknown or unreadable model_index.json must load exactly as before."""
    assert pipeline_family(class_name) == "sd"


def test_prompt_style_follows_the_family():
    assert prompt_style("flux2") == "natural"
    assert prompt_style("sd") == "sdxl"


def test_distilled_flag_from_model_index_counts():
    """klein's id has none of the SD-family markers; its model_index.json says so instead."""
    assert is_distilled("black-forest-labs/FLUX.2-klein-4B", flagged=True) is True
    assert is_distilled("black-forest-labs/FLUX.2-klein-base-4B", flagged=False) is False


# ---------------------- component_dtypes ----------------------

def test_sd_family_dtypes_are_unchanged():
    assert component_dtypes("sd", True) == {"default": "float16"}
    assert component_dtypes("sd", False) == {"default": "float16"}


def test_klein_uses_bf16_where_the_card_has_it():
    assert component_dtypes("flux2", True) == {"default": "bfloat16"}


def test_klein_text_encoder_computes_fp32_without_bf16():
    """Turing (prod's RTX 2070): the Qwen3 encoder's activations sit at a
    quarter of float16's range, so it gets float32 while the rest is float16."""
    assert component_dtypes("flux2", False) == {"default": "float16", "text_encoder": "float32"}


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


def test_env_choice_only_accepts_known_values(monkeypatch):
    monkeypatch.setenv("IMAGE_QUANTIZE", " NF4 ")
    assert env_choice("IMAGE_QUANTIZE", ("none", "nf4"), "none") == "nf4"
    for bad in ("", "int8", "yes"):
        monkeypatch.setenv("IMAGE_QUANTIZE", bad)
        assert env_choice("IMAGE_QUANTIZE", ("none", "nf4"), "none") == "none"
    monkeypatch.delenv("IMAGE_QUANTIZE")
    assert env_choice("IMAGE_QUANTIZE", ("none", "nf4"), "none") == "none"


def test_env_flag_matches_the_charts_true_false(monkeypatch):
    """The Helm configmap renders booleans as "true"/"false" strings."""
    for raw, expected in [("true", True), ("1", True), ("on", True),
                          ("false", False), ("0", False), ("off", False), ("", False)]:
        monkeypatch.setenv("IMAGE_LONG_PROMPT", raw)
        assert env_flag("IMAGE_LONG_PROMPT", True) is expected, raw
    monkeypatch.delenv("IMAGE_LONG_PROMPT")
    assert env_flag("IMAGE_LONG_PROMPT", True) is True

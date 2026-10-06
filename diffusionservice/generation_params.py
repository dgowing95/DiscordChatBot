"""Generation-parameter policy for the diffusion service.

Pure module (stdlib only), following the core/classes/llm_config.py pattern: it
holds every decision that can be made from strings and env vars alone, so the
rules below can be unit-tested without importing torch/diffusers. That matters
because CI installs neither (pyproject.toml's testpaths reach this directory,
but core/requirements-dev.txt has no torch) -- main.py itself is still never
imported by the suite.

main.py keeps everything that needs the loaded pipeline object.
"""
import os
import re

# Distilled few-step models (sd-turbo, SDXL-Lightning, LCM, Hyper-SD) are
# trained to produce an image in 1-8 steps WITHOUT classifier-free guidance.
# Two consequences, both handled here rather than left to the caller:
#   * running them at the pipeline's default CFG (7.5 for SD1.5, 5.0 for SDXL)
#     washes the output out -- which is what the service did for every
#     sd-turbo deployment, since it never passed guidance_scale at all;
#   * diffusers skips the unconditional branch entirely at guidance <= 1, so a
#     negative prompt on such a model is silently inert and must not be sent.
_DISTILLED_MARKERS = ("turbo", "lightning", "lcm", "hyper")

# Pipeline families, told apart by the `_class_name` in the repo's
# model_index.json (read before loading, so these choices can be made up front):
#   sd     SD 1.5 / SDXL. CLIP-conditioned: tag-list prompts, negative prompts,
#          compel for prompts past 77 tokens. Also the fallback for anything
#          unrecognised, which is how every model loaded before this existed.
#   flux2  FLUX.2 [klein]. A Qwen3 text encoder: plain sentences, no negative
#          prompt, and the pipeline takes reference images, i.e. it can edit.
_FLUX2_CLASS_PREFIX = "Flux2Klein"

# Compel's prompt syntax. Parentheses/brackets group and weight, a trailing
# run of + or - up/down-weights the preceding word, and .blend()/.and()/
# .swap()/.pow() are method calls on a parenthesised group. None of that is
# meant by a user (or an LLM) writing plain English, and an UNBALANCED paren
# raises out of the parser -- which would silently drop us back onto the
# truncating code path. So it is all stripped before encoding.
_COMPEL_METHOD_CALL = re.compile(r"\.(?:blend|and|swap|pow)\s*\([^)]*\)", re.IGNORECASE)
_PAREN_WEIGHT = re.compile(r"[)\]]\s*\d+(?:\.\d+)?")
# Grouping/escape characters, stripped outright (str.translate, so the
# backslash needs no regex escaping of its own).
_GROUPING_CHARS = "()[]" + chr(92)
_GROUPING_TABLE = str.maketrans(_GROUPING_CHARS, " " * len(_GROUPING_CHARS))
_INTRA_WORD_HYPHEN = re.compile(r"(?<=[A-Za-z])-(?=[A-Za-z])")
# The lookahead covers punctuation, not just whitespace: compel reads the "++"
# in "blurry++, deformed--" as a 1.21 and a 0.81 weight, and a comma-separated
# negative prompt is exactly where an LLM is most likely to write one.
_WEIGHT_SUFFIX = re.compile(r"(?<=\w)[+-]+(?=[\s,.;:!?]|$)")
_LONE_OPERATOR = re.compile(r"(?:(?<=\s)|^)[+-]+(?=[\s,.;:!?]|$)")


def env_positive_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def env_optional_int(name: str):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def env_optional_float(name: str):
    """A float env var, or None when unset/blank/unparseable.

    None is meaningful for IMAGE_GUIDANCE (see resolve_guidance), so this
    deliberately does NOT collapse to a numeric default.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def env_str(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def env_flag(name: str, default: bool = True) -> bool:
    """A 0/1-style switch, matching image_generation.py's falsy set."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def env_choice(name: str, choices, default: str) -> str:
    """A lower-cased env var when it is one of `choices`, else `default`."""
    raw = os.environ.get(name, "").strip().lower()
    return raw if raw in choices else default


def pipeline_family(class_name) -> str:
    """"flux2" for FLUX.2 [klein], "sd" for everything else (see _FLUX2_CLASS_PREFIX)."""
    return "flux2" if (class_name or "").startswith(_FLUX2_CLASS_PREFIX) else "sd"


def prompt_style(family: str) -> str:
    """What kind of prompt the family wants, as reported on /health so core
    knows whether to run its SDXL prompt rewrite.

    "natural": plain sentences, which is what the bot's model already writes.
    "sdxl": comma-separated clauses plus a negative prompt, which needs the
    rewrite."""
    return "natural" if family == "flux2" else "sdxl"


def component_dtypes(family: str, bf16_supported: bool) -> dict:
    """Torch dtype names per pipeline component ("default" covers the rest).

    SD/SDXL stay on float16, as they always have. FLUX.2 [klein] is published
    in bfloat16, so it gets that wherever the card has it natively. Where it
    does not (prod's RTX 2070 is Turing), the model runs in float16 and its
    Qwen3 text encoder computes in float32. Measured on that card, the
    encoder's hidden states reach ~17,000 -- a quarter of float16's 65,504
    ceiling, too little headroom for a prompt nobody has tried yet, and an
    overflow there is a black image. float32 costs ~1 GB more VRAM while the
    encoder is loaded and no measurable time (a 4-step image took 12-14 s
    either way).
    """
    if family != "flux2":
        return {"default": "float16"}
    if bf16_supported:
        return {"default": "bfloat16"}
    return {"default": "float16", "text_encoder": "float32"}


def is_distilled(model_id: str, flagged: bool = False) -> bool:
    """True when the model is a distilled few-step model.

    `flagged` is the repo's own model_index.json `is_distilled` (FLUX.2
    [klein] sets it); the name markers cover the SD-family models, which
    have no such field."""
    return bool(flagged) or any(marker in (model_id or "").lower() for marker in _DISTILLED_MARKERS)


def resolve_guidance(model_id: str, requested=None, env_default=None,
                     family: str = "sd", distilled=None):
    """The guidance_scale to use, or None to leave it to the pipeline.

    Precedence: distilled models are pinned whatever was asked, then the
    per-request value, then IMAGE_GUIDANCE, then None. The pin is 0.0 for the
    SD family (diffusers skips the unconditional branch below 1) and 1.0 for
    FLUX.2 [klein], whose pipeline defaults to 4.0 and logs a warning on every
    call when a distilled model is given more than 1.

    None means "omit guidance_scale from the pipeline kwargs entirely" rather
    than any particular number, so an unset IMAGE_GUIDANCE reproduces today's
    behaviour exactly. There is deliberately no SD1.5/SDXL default table here:
    a repo id ("Lykon/DreamShaper") does not reveal the family -- only
    isinstance(pipe, StableDiffusionXLPipeline) does, and that lives on the
    other side of the torch import this module exists to avoid.
    """
    if distilled is None:
        distilled = is_distilled(model_id)
    if distilled:
        return 1.0 if family == "flux2" else 0.0
    for value in (requested, env_default):
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def merge_negative_prompt(requested=None, env_default=None, distilled: bool = False) -> str:
    """The negative prompt to send: the per-request one, then the deployment
    baseline (IMAGE_NEGATIVE_PROMPT).

    Request first, because CLIP weights earlier tokens more heavily and the
    scene-specific terms matter more than the boilerplate. Empty for distilled
    models, where a negative prompt is inert (see _DISTILLED_MARKERS).
    """
    if distilled:
        return ""
    parts: list[str] = []
    seen: set[str] = set()
    for raw in (requested, env_default):
        cleaned = (raw or "").strip().strip(",").strip()
        if cleaned and cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            parts.append(cleaned)
    return ", ".join(parts)


def sanitize_for_compel(text: str) -> str:
    """Strip compel's weighting syntax out of free text (see _COMPEL_*).

    Done in the service rather than in the prompt rewriter's instructions so
    the slash-command path -- raw user input, never seen by an LLM -- and any
    future caller are covered too.
    """
    if not text:
        return ""
    text = _COMPEL_METHOD_CALL.sub(" ", text)
    text = _PAREN_WEIGHT.sub(" ", text)
    text = text.translate(_GROUPING_TABLE)
    text = _INTRA_WORD_HYPHEN.sub(" ", text)
    text = _WEIGHT_SUFFIX.sub("", text)
    text = _LONE_OPERATOR.sub(" ", text)
    return " ".join(text.split())

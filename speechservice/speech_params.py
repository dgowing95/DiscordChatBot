"""Speech-service policy that needs no model: env parsing, voice specs, and
which transcripts to throw away.

Pure module (stdlib only), the diffusionservice/generation_params.py pattern:
CI installs none of faster-whisper, kokoro-onnx or numpy, so everything that
can be decided from strings and numbers lives here and is tested in
speechservice/tests/. main.py keeps everything that touches a model.
"""
import os
import re

MIN_SPEED = 0.5
MAX_SPEED = 2.0

# A voice spec is one Kokoro voice id, or a weighted blend of several:
#   "af_heart"                 one voice
#   "af_bella:0.6,am_adam:0.4" a blend; weights are normalised to sum to 1
# Blends are the cheap way to a voice nobody else's bot has: Kokoro's style
# vectors interpolate cleanly.
MAX_BLEND_VOICES = 4
_VOICE_ID = re.compile(r"^[a-z]{2}_[a-z0-9]+$")

# Whisper's own drop rule for a segment that is probably not speech: the
# model thinks it is silence AND it is not confident in what it wrote. Either
# alone throws away quiet real speech. This is what turns the classic
# "Thank you." it hears in room noise into nothing.
NO_SPEECH_PROB = 0.6
LOW_AVG_LOGPROB = -1.0


def env_positive_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def env_str(name: str, default: str) -> str:
    value = (os.environ.get(name) or "").strip()
    return value or default


def clamp_speed(value) -> float:
    """A playback speed Kokoro handles well; anything unparseable is 1.0."""
    try:
        speed = float(value)
    except (TypeError, ValueError):
        return 1.0
    if speed != speed:  # NaN
        return 1.0
    return min(max(speed, MIN_SPEED), MAX_SPEED)


def parse_voice_spec(spec: str, known) -> list[tuple[str, float]]:
    """[(voice_id, weight)] for a voice spec, weights summing to 1.

    Raises ValueError with a message fit to show a user: an unknown voice, a
    bad weight or too many voices.
    """
    spec = (spec or "").strip().lower()
    if not spec:
        raise ValueError("No voice given")
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if not parts:
        raise ValueError("No voice given")
    if len(parts) > MAX_BLEND_VOICES:
        raise ValueError(f"A blend can mix at most {MAX_BLEND_VOICES} voices")
    known = set(known)
    weights: dict[str, float] = {}
    for part in parts:
        name, _, raw_weight = part.partition(":")
        name = name.strip()
        if not _VOICE_ID.match(name) or name not in known:
            raise ValueError(f"Unknown voice '{name}'")
        try:
            weight = float(raw_weight) if raw_weight.strip() else 1.0
        except ValueError:
            raise ValueError(f"Bad weight for '{name}'") from None
        if not weight > 0 or weight == float("inf"):
            raise ValueError(f"Bad weight for '{name}'")
        weights[name] = weights.get(name, 0.0) + weight
    total = sum(weights.values())
    return [(name, weight / total) for name, weight in weights.items()]


# Kokoro voice ids start with a language letter; the phonemizer must be told
# the same language or a French voice reads English phonemes (and vice versa).
# A blend speaks the language of its first voice.
_VOICE_LANGS = {
    "a": "en-us", "b": "en-gb", "e": "es", "f": "fr-fr", "h": "hi",
    "i": "it", "j": "ja", "p": "pt-br", "z": "cmn",
}


def voice_lang(blend: list[tuple[str, float]]) -> str:
    """The espeak language for a parsed voice spec (en-us when unknown)."""
    return _VOICE_LANGS.get(blend[0][0][:1], "en-us") if blend else "en-us"


def keep_segment(no_speech_prob: float, avg_logprob: float) -> bool:
    """Whether a transcribed segment is speech rather than hallucinated noise."""
    return not (no_speech_prob > NO_SPEECH_PROB and avg_logprob < LOW_AVG_LOGPROB)


def join_segments(texts) -> str:
    """The kept segments as one transcript."""
    return " ".join(t.strip() for t in texts if t and t.strip())

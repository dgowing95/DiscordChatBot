"""Tests for the speech service's stdlib-only half.

main.py itself is never imported here (it needs faster-whisper, kokoro-onnx
and numpy, which CI does not install) -- speech_params.py exists so the voice
spec and transcript rules can be checked without the models.
"""
import pytest

from speech_params import (
    MAX_SPEED,
    MIN_SPEED,
    clamp_speed,
    env_positive_int,
    join_segments,
    keep_segment,
    parse_voice_spec,
    voice_lang,
)

VOICES = ["af_heart", "af_bella", "am_adam", "bf_emma", "bm_george"]


def test_single_voice():
    assert parse_voice_spec("af_heart", VOICES) == [("af_heart", 1.0)]


def test_voice_spec_is_case_and_space_insensitive():
    assert parse_voice_spec("  AF_Heart ", VOICES) == [("af_heart", 1.0)]


def test_blend_weights_are_normalised():
    blend = dict(parse_voice_spec("af_bella:3, am_adam:1", VOICES))
    assert blend == {"af_bella": 0.75, "am_adam": 0.25}


def test_blend_without_weights_is_even():
    blend = dict(parse_voice_spec("af_bella,am_adam", VOICES))
    assert blend == {"af_bella": 0.5, "am_adam": 0.5}


def test_repeated_voice_adds_up():
    assert parse_voice_spec("af_bella:1,af_bella:1", VOICES) == [("af_bella", 1.0)]


@pytest.mark.parametrize("spec, message", [
    ("", "No voice"),
    ("zz_nobody", "Unknown voice"),
    ("af_heart:abc", "Bad weight"),
    ("af_heart:0", "Bad weight"),
    ("af_heart:-1", "Bad weight"),
    ("af_heart:nan", "Bad weight"),
    ("af_heart:inf", "Bad weight"),
    ("af_heart,af_bella,am_adam,bf_emma,bm_george", "at most"),
    ("../etc/passwd", "Unknown voice"),
])
def test_bad_specs_are_refused(spec, message):
    with pytest.raises(ValueError, match=message):
        parse_voice_spec(spec, VOICES)


@pytest.mark.parametrize("raw, expected", [
    (1.0, 1.0), ("1.25", 1.25), (0.1, MIN_SPEED), (9, MAX_SPEED),
    ("fast", 1.0), (None, 1.0), (float("nan"), 1.0),
])
def test_clamp_speed(raw, expected):
    assert clamp_speed(raw) == expected


def test_noise_segment_is_dropped():
    # Confident it is silence AND unsure of its words: Whisper's "Thank you."
    assert keep_segment(no_speech_prob=0.9, avg_logprob=-1.4) is False


@pytest.mark.parametrize("no_speech, logprob", [(0.9, -0.3), (0.1, -1.5), (0.1, -0.2)])
def test_speech_segments_are_kept(no_speech, logprob):
    # Either signal alone is not enough: quiet real speech trips one of them.
    assert keep_segment(no_speech, logprob) is True


def test_join_segments():
    assert join_segments([" Hey Sparky,", "", " what time is it? "]) == "Hey Sparky, what time is it?"


def test_env_positive_int(monkeypatch):
    monkeypatch.setenv("X_THREADS", "8")
    assert env_positive_int("X_THREADS", 4) == 8
    monkeypatch.setenv("X_THREADS", "0")
    assert env_positive_int("X_THREADS", 4) == 4
    monkeypatch.setenv("X_THREADS", "lots")
    assert env_positive_int("X_THREADS", 4) == 4


@pytest.mark.parametrize("blend, lang", [
    ([("af_heart", 1.0)], "en-us"),
    ([("bm_george", 1.0)], "en-gb"),
    ([("ff_siwis", 0.5), ("af_heart", 0.5)], "fr-fr"),
    ([("qx_unknown", 1.0)], "en-us"),
    ([], "en-us"),
])
def test_voice_lang_follows_the_first_voice(blend, lang):
    assert voice_lang(blend) == lang

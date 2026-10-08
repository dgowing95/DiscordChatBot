"""
Tests for core/classes/voice_policy.py (pure): the wake phrase, spoken
commands, the cache-friendly history window and text-to-speech cleanup.

Run from the repo root:
    pytest core/tests/voice_policy_tests.py
"""
import pytest

from classes import voice_policy as vp

SPARKY = "hey sparky"


# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------

def test_settings_defaults(monkeypatch):
    for name in ("VOICE_ENABLED", "VOICE_IDLE_LEAVE_SECONDS", "VOICE_HISTORY_LIMIT",
                 "VOICE_THINKING", "VOICE_PREFILL_MIN_SECONDS", "VOICE_DEV_INJECT"):
        monkeypatch.delenv(name, raising=False)
    s = vp.settings()
    assert s["enabled"] is False
    assert s["idle_leave_seconds"] == 30
    assert s["history_limit"] == 10 and s["history_refresh"] == 10
    assert s["thinking"] is False
    assert s["prefill_min_seconds"] == 5
    assert s["dev_inject"] is False


@pytest.mark.parametrize("raw, expected", [("1", 1), ("45", 45), ("0", 30), ("-3", 30), ("soon", 30)])
def test_idle_leave_seconds_falls_back_on_nonsense(monkeypatch, raw, expected):
    monkeypatch.setenv("VOICE_IDLE_LEAVE_SECONDS", raw)
    assert vp.settings()["idle_leave_seconds"] == expected


# ---------------------------------------------------------------------------
# wake phrase
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("display_name, phrase", [
    ("Sparky", "hey sparky"),
    ("Sparky the Bot", "hey sparky"),
    ("CleverHelperBot", "hey clever helper bot"),
    ("🤖", "hey bot"),
    ("", "hey bot"),
])
def test_default_wake_phrase(display_name, phrase):
    assert vp.default_wake_phrase(display_name) == phrase


@pytest.mark.parametrize("heard, remainder", [
    ("Hey, Sparky! What's the weather?", "What's the weather?"),
    ("hay sparkie what time is it", "what time is it"),          # Whisper spellings
    ("A Sparky, tell me a joke.", "tell me a joke."),            # "hey" heard as "a"
    ("um okay so hey sparky can you help", "can you help"),      # filler before it
    ("Heysparky, hello", "hello"),                               # glued together
    ("hey, Sparki. leave", "leave"),
    ("Hey Sparky.", ""),                                         # the phrase alone
])
def test_wake_phrase_is_heard(heard, remainder):
    assert vp.match_wake(heard, SPARKY) == (True, remainder)


@pytest.mark.parametrize("heard", [
    "hey barky what's up",           # a different name
    "hey marky",
    "sparky what's up",               # no greeting
    "I told sparky yesterday, hey",   # the name, but not addressed
    "",
])
def test_other_speech_is_not_a_wake(heard):
    assert vp.match_wake(heard, SPARKY)[0] is False


def test_multi_word_names():
    phrase = vp.default_wake_phrase("CleverHelperBot")
    assert vp.match_wake("Hey clever helper bot who won the race", phrase) == (True, "who won the race")
    assert vp.match_wake("Hey clever helper, Bot leave.", phrase)[0] is True
    # Whisper sometimes drops the leading "hey"; a multi-word name said in
    # full still counts, where a one-word name alone does not.
    assert vp.match_wake("clever helper bot draw me a cat", phrase) == (True, "draw me a cat")
    assert vp.match_wake("sparky draw me a cat", "hey sparky")[0] is False
    # ...but only opening the utterance: further in, it is talk ABOUT the bot.
    assert vp.match_wake("I think clever helper bot is broken", phrase)[0] is False
    assert vp.match_wake("did clever helper bot crash", phrase)[0] is False


@pytest.mark.parametrize("heard", [
    "Hey Taba, how are you?",       # a short name's vowel misheard
    "hey tooba what time is it",
    "Hey, Tuba.",
])
def test_a_short_name_survives_a_wrong_vowel(heard):
    assert vp.match_wake(heard, "hey tubba")[0] is True


@pytest.mark.parametrize("heard", [
    "hey gubba",                    # a different consonant
    "hey tab",
    "hey tubby",
    "hey tubbles",
    "and Taba, what's up",          # one-word name with no greeting
])
def test_a_short_name_is_not_loose(heard):
    assert vp.match_wake(heard, "hey tubba")[0] is False


@pytest.mark.parametrize("heard", ["Hey Sparkydev, hi.", "Okay Sparkidev, say test.", "Hey, Sporky Dev, how are you?"])
def test_name_words_split_or_joined(heard):
    assert vp.match_wake(heard, "hey sparky dev")[0] is True


def test_custom_phrase_without_a_greeting():
    assert vp.match_wake("Computer, lights on", "computer") == (True, "lights on")


@pytest.mark.parametrize("phrase, expected", [
    ("Hey, Sparky!", "hey sparky"),
    ("  OK   Computer ", "ok computer"),
])
def test_validate_wake_phrase(phrase, expected):
    assert vp.validate_wake_phrase(phrase) == expected


@pytest.mark.parametrize("phrase", ["", "!!!", "one two three four five six", "x" * 70])
def test_validate_wake_phrase_refuses(phrase):
    with pytest.raises(ValueError):
        vp.validate_wake_phrase(phrase)


# ---------------------------------------------------------------------------
# spoken commands
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("said, command", [
    ("leave", vp.LEAVE),
    ("please leave the call", vp.LEAVE),
    ("you can leave now", vp.LEAVE),
    ("Goodbye!", vp.LEAVE),
    ("go away", vp.LEAVE),
    ("stop.", vp.STOP),
    ("shut up", vp.STOP),
    ("be quiet please", vp.STOP),
    ("stop it", vp.STOP),
    ("OK, that's enough", vp.STOP),
    ("stop the music and tell me a joke", None),  # a real request
    ("what's the weather", None),
    ("", None),
])
def test_parse_command(said, command):
    assert vp.parse_command(said) == command


# ---------------------------------------------------------------------------
# history window
# ---------------------------------------------------------------------------

def _clip(n):
    return vp.utterance_message("Ana", f"clip {n}")


def test_utterance_message_names_the_speaker():
    assert vp.utterance_message("Ana", "hi") == {"role": "user", "content": "[Ana]: hi"}


def test_history_is_anchored_until_it_rebases():
    history = vp.VoiceHistory(limit=3, refresh=2)
    for n in range(5):
        assert history.add(_clip(n)) is False
    first = history.messages()
    # Growing to limit + refresh keeps the start fixed: every prompt so far
    # is the previous one plus a tail (what llama.cpp's cache needs).
    assert first[0]["content"] == "[Ana]: clip 0" and len(first) == 5
    assert history.add(_clip(5)) is True
    assert [m["content"] for m in history.messages()] == ["[Ana]: clip 3", "[Ana]: clip 4", "[Ana]: clip 5"]
    assert history.rebases == 1


def test_history_counts_utterances_not_replies():
    history = vp.VoiceHistory(limit=2, refresh=1)
    history.add(_clip(0))
    history.add({"role": "assistant", "content": "answer"})
    history.add(_clip(1))
    history.add({"role": "assistant", "content": "answer 2"})
    assert len(history.messages()) == 4          # 2 utterances: under the limit
    history.add(_clip(2))
    history.add(_clip(3))                       # 4 utterances > 2 + 1: rebase
    assert [m["content"] for m in history.messages()] == ["[Ana]: clip 2", "[Ana]: clip 3"]


def test_history_messages_are_copies():
    history = vp.VoiceHistory(limit=5, refresh=5)
    history.add(_clip(0))
    history.messages()[0]["content"] = "changed"
    assert history.messages()[0]["content"] == "[Ana]: clip 0"


# ---------------------------------------------------------------------------
# text -> speech
# ---------------------------------------------------------------------------

def test_speakable_strips_markup():
    text = ("# Title\n- **bold** item 🎉\n```py\nprint(1)\n```\n"
            "See [the docs](http://a.b) or https://x.y/z <@123>")
    assert vp.speakable(text) == "Title bold item See the docs or a link"


def test_speakable_keeps_plain_text():
    assert vp.speakable("It's 14.5 degrees, isn't it?") == "It's 14.5 degrees, isn't it?"


def _stream(text, splitter=None):
    splitter = splitter or vp.SentenceSplitter()
    out = []
    for ch in text:
        out += splitter.feed(ch)
    return out, splitter.flush()


def test_sentences_are_released_as_they_finish():
    out, rest = _stream("Ahoy! It's 14.5 degrees in London, e.g. rainy. Bring a coat")
    assert out == ["Ahoy!", "It's 14.5 degrees in London, e.g. rainy."]
    assert rest == ["Bring a coat"]


def test_a_long_sentence_is_cut_at_a_clause():
    long = "This is a very long sentence that keeps going, with clauses, and more clauses, " \
           "and yet more words to pass the limit of characters; then it ends"
    out, rest = _stream(long, vp.SentenceSplitter(long_chars=60))
    assert out and all(len(s) < 100 for s in out)
    assert " ".join(out + rest) == long


def test_newlines_end_sentences_and_markup_is_dropped():
    out, rest = _stream("**Sure**\n- one\n- two\n")
    assert out == ["Sure", "one", "two"]
    assert rest == []


def test_voice_names():
    assert vp.voice_names("af_bella:0.6, AM_adam:0.4") == ["af_bella", "am_adam"]
    assert vp.voice_names("") == []


def test_hold_on_messages_carry_the_tool_and_request():
    messages = vp.hold_on_messages("a pirate", vp.tool_action("web_search"), "f1 winner", "who won?")
    content = messages[0]["content"]
    assert "a pirate" in content and "searching the web" in content
    assert "f1 winner" in content and "who won?" in content
    assert vp.tool_action("made_up_tool") == "using made up tool"


def test_voice_instructions_fit_in_one_home():
    # One place for what the voice model is told (AGENTS.md "Prompt
    # surface"); keep it short -- it is in every voice prompt.
    assert len(vp.voice_instructions("x" * 200)) < 1000


def test_voice_instructions_name_the_bot():
    assert "people call you Sparky the Bot;" in vp.voice_instructions("  Sparky  the Bot ")
    assert "{name}" not in vp.voice_instructions("")
    assert "voice call;" in vp.voice_instructions("")

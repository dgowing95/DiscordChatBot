"""What the bot does with speech in a voice channel (PURE, stdlib only).

Everything here can be decided from strings and env vars alone, so it is
unit-tested in core/tests/voice_policy_tests.py without Discord, the voice
sidecar or the LLM. voice_session.py does the I/O.

The rules worth knowing before changing them:

  * The wake phrase is found in the TRANSCRIPT, not in the audio. Wake-word
    engines need a model trained per phrase, and the phrase is per-guild
    setting (/voice wake_word), so every utterance is transcribed anyway and
    the phrase is matched in the text. Whisper spells names creatively
    ("Hey, Sparkie." / "hay sparky" / "A Sparky,"), so words are compared by a
    rough phonetic key and a similarity ratio, never exactly.
  * Leaving and stopping are matched here, deterministically, rather than
    asked of the LLM: "hey sparky, leave" must work every time and at once,
    including while a slow turn is still running.
  * The history window is ANCHORED, not sliding. A window that slides by one
    clip changes its first message on every turn, so llama.cpp could reuse
    none of its cached prompt; this one keeps its start fixed and grows,
    then rebases (see VoiceHistory).
"""
import os
import re
import unicodedata

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _env_number(name: str, default, minimum, cast=int):
    try:
        value = cast(str(os.environ.get(name)).strip())
    except (TypeError, ValueError):
        return default
    if value != value or value < minimum:  # NaN or too small
        return default
    return value


def settings() -> dict:
    """The VOICE_* settings, read on every call (cheap; tests patch env)."""
    return {
        "enabled": _env_flag("VOICE_ENABLED", False),
        "bridge_url": (os.environ.get("VOICE_BRIDGE_URL") or "ws://voice:8765").strip(),
        "speech_url": (os.environ.get("SPEECH_URL") or "http://speech:8000").strip(),
        "idle_leave_seconds": _env_number("VOICE_IDLE_LEAVE_SECONDS", 30, 1),
        "history_limit": _env_number("VOICE_HISTORY_LIMIT", 10, 1),
        "history_refresh": _env_number("VOICE_HISTORY_REFRESH", 10, 1),
        "followup_seconds": _env_number("VOICE_FOLLOWUP_SECONDS", 8, 0, float),
        "thinking": _env_flag("VOICE_THINKING", False),
        "default_voice": (os.environ.get("VOICE_DEFAULT_VOICE") or "af_heart").strip(),
        "prefill_min_seconds": _env_number("VOICE_PREFILL_MIN_SECONDS", 5, 0, float),
        "dev_inject": _env_flag("VOICE_DEV_INJECT", False),
    }


# ---------------------------------------------------------------------------
# Wake phrase
# ---------------------------------------------------------------------------

# Words Whisper writes for a spoken "hey". A wake phrase that starts with any
# of them accepts any other in its place.
GREETINGS = frozenset({"hey", "hay", "hi", "hei", "heya", "hiya", "a", "eh", "ay", "ok", "okay", "yo", "hello"})
# How far into an utterance the phrase may start ("um, okay so hey sparky").
MAX_LEADING_WORDS = 3
# Similarity of the phonetic keys of phrase and heard words (_ratio).
WAKE_THRESHOLD = 0.8
MAX_WAKE_PHRASE_CHARS = 60
MAX_WAKE_PHRASE_WORDS = 5

_WORD = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*")


def words(text: str) -> list[str]:
    """Lower-case words, punctuation dropped ("Hey, Sparky!" -> hey, sparky)."""
    return _WORD.findall(_fold(text))


def _fold(text: str) -> str:
    """Lower case with accents removed (Whisper writes "Sparké" sometimes)."""
    text = unicodedata.normalize("NFKD", text or "")
    return "".join(c for c in text if not unicodedata.combining(c)).lower()


def _key(word: str) -> str:
    """A rough phonetic key: doubled letters collapsed and the common
    spellings of a final "ee" sound unified, so sparky / sparkie / sparki match
    each other but barky still does not match sparky."""
    word = word.replace("ph", "f").replace("ck", "k")
    word = re.sub(r"(.)\1+", r"\1", word)
    return re.sub(r"(ie|ey|ee|i|y)$", "y", word)


def default_wake_phrase(display_name: str) -> str:
    """"hey <first word of the bot's display name>": "Sparky the Bot" is
    called "hey sparky", not "hey sparky the bot". A CamelCase name is split
    the way people say it (and Whisper writes it): "CleverHelperBot" is
    "hey clever helper bot"."""
    first = (display_name or "").split()[0] if (display_name or "").split() else ""
    name = words(re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", first))
    return "hey " + (" ".join(name[:3]) if name else "bot")


def validate_wake_phrase(phrase: str) -> str:
    """The phrase normalised to plain words; raises ValueError to show a user."""
    normalised = " ".join(words(phrase))
    if not normalised:
        raise ValueError("The wake phrase needs at least one word.")
    if len(normalised) > MAX_WAKE_PHRASE_CHARS or len(normalised.split()) > MAX_WAKE_PHRASE_WORDS:
        raise ValueError(f"Keep the wake phrase to {MAX_WAKE_PHRASE_WORDS} words or fewer.")
    return normalised


def _ratio(heard: list[str], target: list[str]) -> float:
    """Spelling similarity of two word lists, 0..1: one minus the edit
    distance between their phonetic keys, over the longer key's length.

    Spaces are ignored, since Whisper splits and joins names as it likes
    ("sparkydev" / "sparky dev"). Swapping one vowel for another costs a
    quarter of an edit: the vowel is what Whisper gets wrong most in a short
    name ("lulu" written "lala"), and at full cost a four-letter name could
    not lose a single letter. A different consonant costs a whole edit, so
    "hey barky" stays apart from "hey sparky".
    """
    a = "".join(_key(w) for w in heard)
    b = "".join(_key(w) for w in target)
    if not a or not b:
        return 0.0
    previous = [float(j) for j in range(len(b) + 1)]
    for i, x in enumerate(a, 1):
        current = [float(i)]
        for j, y in enumerate(b, 1):
            swap = 0.0 if x == y else VOWEL_SWAP_COST if x in _VOWELS and y in _VOWELS else 1.0
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + swap))
        previous = current
    return 1 - previous[-1] / max(len(a), len(b))


_VOWELS = frozenset("aeiou")
VOWEL_SWAP_COST = 0.25


# Greetings Whisper may glue onto the name ("heysparky"). Two letters or
# more, so an ordinary word starting with "a" is not read as one.
_GLUED_GREETINGS = tuple(sorted((g for g in GREETINGS if len(g) > 1), key=len, reverse=True))


def _similar(heard: list[str], target: list[str], at_start: bool = True) -> float:
    """How alike the heard words are to the phrase, 0..1.

    For a phrase that starts with a greeting, three cases, each explicit
    (an earlier single ratio over everything let the score depend on how
    long the name was):

      * a greeting was heard: it is interchangeable with any other greeting
        and left out, so only the NAME is compared -- scored together, "hey"
        matching "hey" carried "hey barky" over the threshold for "hey sparky";
      * the greeting is glued to the name ("heysparky"): compared whole;
      * no greeting at all: accepted only for a name of two or more words
        that OPENS the utterance, compared name to name. The mic or Whisper
        clips a leading "hey" often enough (seen live) that requiring it
        misses real requests; but a one-word name on its own ("sparky,
        what's up") or a name further in ("I think clever helper bot is
        broken") is too easily talk ABOUT the bot.
    """
    if target[0] not in GREETINGS or len(target) == 1:
        return _ratio(heard, target)
    name = target[1:]
    if not heard:
        return 0.0
    if heard[0] in GREETINGS:
        return _ratio(heard[1:], name) if len(heard) > 1 else 0.0
    if heard[0].startswith(_GLUED_GREETINGS) and len(heard) < len(target):
        return _ratio(heard, target)
    if at_start and len(name) >= 2 and len(heard) == len(name):
        return _ratio(heard, name)
    return 0.0


def match_wake(transcript: str, phrase: str) -> tuple[bool, str]:
    """(matched, what was said after the phrase).

    The phrase may start within the first few words, and may be heard as one
    word more or fewer ("heysparky", "hey spar kee"). The remainder keeps the
    transcript's own spelling and punctuation, minus the separator after the
    phrase.
    """
    target = words(phrase)
    if not target:
        return False, ""
    spans = [(m.group(0), m.end()) for m in _WORD.finditer(_fold(transcript))]
    heard = [w for w, _ in spans]
    best = (0.0, None)
    for start in range(0, min(MAX_LEADING_WORDS, len(heard)) + 1):
        for length in (len(target), len(target) - 1, len(target) + 1):
            if length < 1 or start + length > len(heard):
                continue
            score = _similar(heard[start:start + length], target, at_start=start == 0)
            if score > best[0]:
                best = (score, start + length)
    if best[0] < WAKE_THRESHOLD:
        return False, ""
    end = spans[best[1] - 1][1]
    return True, transcript[end:].lstrip(" ,.!?;:-—").strip()


# ---------------------------------------------------------------------------
# Spoken commands handled without the LLM
# ---------------------------------------------------------------------------

LEAVE = "leave"
STOP = "stop"

_FILLER = frozenset({"please", "now", "can", "could", "would", "you", "will", "just", "the", "this",
                     "call", "channel", "voice", "chat", "vc", "ok", "okay", "right", "thanks", "thank"})
_LEAVE_COMMANDS = frozenset({"leave", "go away", "go", "disconnect", "get out", "bye", "goodbye",
                             "bye bye", "hang up", "go home", "log off", "leave us"})
_STOP_COMMANDS = frozenset({"stop", "shut up", "be quiet", "quiet", "stop talking", "silence",
                            "enough", "cancel", "hush", "shush", "nevermind", "never mind"})


def parse_command(remainder: str) -> str | None:
    """LEAVE / STOP when the request is only that command, else None.

    Short exact intents only: "stop the music and tell me a joke" is a request
    for the LLM, "please leave the call" is not.
    """
    said = " ".join(w for w in words(remainder) if w not in _FILLER)
    if said in _LEAVE_COMMANDS:
        return LEAVE
    if said in _STOP_COMMANDS:
        return STOP
    return None


# ---------------------------------------------------------------------------
# History (the cache-friendly window)
# ---------------------------------------------------------------------------


def utterance_message(name: str, text: str) -> dict:
    """One heard utterance as a prompt message, labelled with the speaker."""
    return {"role": "user", "content": f"[{name}]: {text}"}


class VoiceHistory:
    """The session's conversation as prompt messages, anchored for caching.

    Every transcript goes in (not only wake-phrase ones), so the model hears
    what was being discussed when someone asks "what do you think?". The
    window keeps its first message fixed and grows from `limit` to
    `limit + refresh` utterances, then rebases to the newest `limit`. Between
    rebases every prompt is the previous one plus a tail, so llama.cpp only
    processes the tail; a rebase costs one cold prefill per `refresh` clips.
    """

    def __init__(self, limit: int, refresh: int):
        self.limit = limit
        self.refresh = refresh
        self.items: list[dict] = []
        self.rebases = 0

    def add(self, message: dict) -> bool:
        """Append a message; True when that rebased the window."""
        self.items.append(message)
        user_indexes = [i for i, m in enumerate(self.items) if m["role"] == "user"]
        if len(user_indexes) <= self.limit + self.refresh:
            return False
        self.items = self.items[user_indexes[-self.limit]:]
        self.rebases += 1
        return True

    def messages(self) -> list[dict]:
        return [dict(m) for m in self.items]


def voice_names(spec: str) -> list[str]:
    """The voice ids in a voice setting ("af_bella:0.6,am_adam:0.4" ->
    af_bella, am_adam). The speech service validates the weights."""
    return [part.split(":", 1)[0].strip().lower() for part in (spec or "").split(",") if part.strip()]


# ---------------------------------------------------------------------------
# Text -> speech
# ---------------------------------------------------------------------------

_CODE_BLOCK = re.compile(r"```.*?(```|$)", re.DOTALL)
_INLINE_CODE = re.compile(r"`([^`]*)`")
_MD_LINK = re.compile(r"\[([^\]]+)\]\((?:[^)]+)\)")
_URL = re.compile(r"<?https?://\S+>?")
_DISCORD_TOKEN = re.compile(r"<(?:@[!&]?|#|a?:\w+:)\d+>")
_LINE_MARKUP = re.compile(r"^\s*(?:#{1,6}\s+|>\s?|[-*+]\s+|\d+[.)]\s+)", re.MULTILINE)
_EMPHASIS = re.compile(r"(\*\*|__|\*|_|~~|\|\|)(?=\S)(.+?)(?<=\S)\1")


def speakable(text: str) -> str:
    """Markdown, links, code and emoji removed: what reads well out loud."""
    text = _CODE_BLOCK.sub(" ", text or "")
    text = _INLINE_CODE.sub(r"\1", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _URL.sub("a link", text)
    text = _DISCORD_TOKEN.sub("", text)
    text = _LINE_MARKUP.sub("", text)
    for _ in range(2):  # nested emphasis
        text = _EMPHASIS.sub(r"\2", text)
    text = "".join(c for c in text if unicodedata.category(c) not in ("So", "Sk", "Cs", "Co")
                   and c not in "‍️")
    return re.sub(r"\s+", " ", text).strip()


# A sentence ends at . ! ? or an ellipsis followed by a space, or at a line
# break. Not after a single capital letter or a common abbreviation, and a
# decimal point is never followed by a space.
_SENTENCE_END = re.compile(r"(?<=[.!?…])[\"')\]]*\s+|\n+")
_ABBREVIATIONS = ("e.g.", "i.e.", "mr.", "mrs.", "ms.", "dr.", "st.", "vs.", "etc.", "approx.")
_CLAUSE_END = re.compile(r"[,;:—–]\s+")
# Past this a sentence is spoken at its last clause break instead, so a long
# first sentence does not hold back the first audio.
LONG_SENTENCE_CHARS = 120


class SentenceSplitter:
    """Turns streamed text deltas into speakable sentences as they complete."""

    def __init__(self, long_chars: int = LONG_SENTENCE_CHARS):
        self.buffer = ""
        self.long_chars = long_chars

    def feed(self, delta: str) -> list[str]:
        self.buffer += delta or ""
        out = []
        while True:
            cut = self._sentence_cut()
            if cut is None and len(self.buffer) > self.long_chars:
                cut = self._clause_cut()
            if cut is None:
                break
            sentence, self.buffer = self.buffer[:cut], self.buffer[cut:]
            spoken = speakable(sentence)
            if spoken:
                out.append(spoken)
        return out

    def flush(self) -> list[str]:
        spoken = speakable(self.buffer)
        self.buffer = ""
        return [spoken] if spoken else []

    def _sentence_cut(self):
        for match in _SENTENCE_END.finditer(self.buffer):
            before = self.buffer[:match.start()].rstrip("\"')]")
            last_word = before.rsplit(None, 1)[-1].lower() if before.split() else ""
            if last_word.endswith(_ABBREVIATIONS) or re.fullmatch(r"[a-z]\.", last_word):
                continue
            if self.buffer[:match.start()].strip() == "":
                continue
            return match.end()
        return None

    def _clause_cut(self):
        cuts = [m.end() for m in _CLAUSE_END.finditer(self.buffer) if m.start() >= 20]
        return cuts[-1] if cuts else None


# ---------------------------------------------------------------------------
# What the voice model is told
# ---------------------------------------------------------------------------

# Appended to the personality system prompt for voice turns only, so it is
# the same for the whole session (the cached prefix depends on that). One
# home for every voice-specific instruction: add guidance here, not in a
# second copy elsewhere (see AGENTS.md "Prompt surface").
#
# The bot's name is in it because a call is the one place the model hears it:
# in chat the mention is stripped. Without it, told only that people address
# it by name, the model answered "Hey <name>, how are you?" with "I don't see
# anyone addressing me directly here". Only a transcript that starts with the
# wake phrase becomes a turn, so the newest message is ALWAYS meant for it,
# and saying so is what stops that second-guessing.
VOICE_INSTRUCTIONS = (
    "You are talking out loud in a Discord voice call{name}; everything you write is spoken by a "
    "text-to-speech voice. Stay in character. Reply in one to three short, natural spoken "
    "sentences. No markdown, lists, emoji, code or links. Messages labelled [Name]: are "
    "transcripts of what people in the call said, and may contain recognition mistakes, your own "
    "name included. The newest one is said to you; the earlier ones are the conversation around "
    "it. Before you call a tool, first say one short sentence about what you are about to do, in "
    "your own words. Pictures and code results appear in the text thread for this call, so say "
    "they are there rather than describing every detail."
)


def voice_instructions(bot_name: str = "") -> str:
    """VOICE_INSTRUCTIONS for a bot called `bot_name` (its display name)."""
    name = " ".join((bot_name or "").split())[:80]
    return VOICE_INSTRUCTIONS.format(name=f", where people call you {name}" if name else "")

# Asked of the side model when the voice model called a tool without saying
# anything first, so the call is never silent and the line is never canned.
HOLD_ON_PROMPT = (
    "You are {persona}, in a voice call. You are about to use a tool: {tool}{detail}. "
    "The person asked: \"{request}\". Write ONE short spoken sentence (under 15 words) telling "
    "them to hold on while you do it, in character. Plain text only."
)


def hold_on_messages(persona: str, tool: str, detail: str, request: str) -> list[dict]:
    """Chat messages for the side-LLM hold-on line."""
    detail = f" ({detail[:120]})" if detail else ""
    return [{"role": "user", "content": HOLD_ON_PROMPT.format(
        persona=persona[:300], tool=tool, detail=detail, request=request[:300])}]


# Asked of the side model as the bot joins a call: the hello, in character
# and different each time. How to wake the bot is said after it, by code,
# so it is always there and always right.
GREETING_PROMPT = (
    "You are {persona}{name}, joining a Discord voice call{people}. Write ONE short spoken "
    "greeting (under 12 words) as you arrive, in character. Don't explain how to talk to you. "
    "Plain text only."
)


def greeting_messages(persona: str, bot_name: str, people: list[str]) -> list[dict]:
    """Chat messages for the side-LLM greeting when the bot joins a call."""
    name = f", called {bot_name[:80]}" if bot_name else ""
    people = f" with {', '.join(p[:40] for p in people[:8])}" if people else ""
    return [{"role": "user", "content": GREETING_PROMPT.format(
        persona=persona[:300], name=name, people=people)}]


# Readable tool names for the hold-on prompt and its last-resort fallback.
TOOL_ACTIONS = {
    "web_search": "searching the web",
    "fetch_url": "reading a web page",
    "generate_image": "making a picture",
    "run_code_sandbox": "running some code",
    "store_memory": "remembering that",
    "remove_memory": "forgetting that",
    "clear_memories": "clearing my memories",
    "change_personality": "changing my personality",
    "leave_voice_channel": "leaving the call",
}


def tool_action(name: str) -> str:
    return TOOL_ACTIONS.get(name, "using " + name.replace("_", " "))

"""A voice call in one guild: hear, decide, answer out loud.

One VoiceSession per guild, created by start() (the /voice join command and
the join_voice_channel tool both come here) and ended by end(), the only way
out: it clears the voice gate (voice_gate.py) and the discord.py voice client
together, so the two never disagree about whether the bot is in a call.

What happens to each utterance the voice sidecar transcribes (on_utterance):

  1. It joins the conversation history, whoever said it, so the model knows
     what was being talked about when it is finally asked something.
  2. The wake phrase is looked for in it (voice_policy.match_wake). Nothing
     else gets an answer: people talking to each other are left alone.
  3. "leave" / "stop" after the wake phrase are handled here, at once, never
     by the LLM. The wake phrase on its own plays a chime and lets that
     speaker's next utterance count as the request (VOICE_FOLLOWUP_SECONDS).
  4. A request becomes a TURN. It opens with the same chime, so whoever
     asked hears at once that they were heard while the model works on it.
     The voice agent then runs streamed, each finished
     sentence is sent to the sidecar while the model writes the next, and
     tools run as in chat, posting their pictures and code results in the
     session's text thread.

Before a tool runs the bot says a short line about what it is doing. The
model is asked to write one itself (voice_policy.VOICE_INSTRUCTIONS); when it calls a tool
without saying anything, a small side call writes the line instead, so it is
never canned and never silent. Lines and sentences are spoken strictly in
order (_OrderedSpeech), so the answer never plays over its own "hold on".

Between turns the history is prefilled into llama.cpp (TextLLMHandler.
prefill), so a turn's prompt is already processed up to its last few tokens
when the wake phrase is heard.
"""
import asyncio
import json
import logging
import time
import uuid

import discord

from classes import voice_gate, voice_policy
from classes.config_manager import configManager
from classes.llm_config import llm_api_key, llm_host, llm_model
from classes.metrics import (
    inc_voice_turn,
    inc_voice_utterance,
    observe_voice_first_audio,
    observe_voice_speech,
    set_voice_sessions,
)
from classes.side_llm import SideLLM
from classes.voice_bridge import CONNECT_TIMEOUT, SidecarVoiceProtocol, VoiceUnavailable, get_bridge
from classes.voice_policy import (
    LEAVE,
    STOP,
    SentenceSplitter,
    VoiceHistory,
    default_wake_phrase,
    hold_on_messages,
    match_wake,
    parse_command,
    speakable,
    tool_action,
    utterance_message,
)

logger = logging.getLogger(__name__)

# How long a session waits for its last sentences to finish playing before
# leaving (a spoken goodbye), and a hold-on line may take to be written.
GOODBYE_WAIT_SECONDS = 15
HOLD_ON_TIMEOUT_SECONDS = 8
# A busy notice is spoken at most this often.
BUSY_NOTICE_SECONDS = 10

_hold_on_llm = SideLLM("voice_hold_on", llm_model, llm_host, llm_api_key, lambda: HOLD_ON_TIMEOUT_SECONDS)


class VoiceError(Exception):
    """Why the bot cannot join; the message is shown to the user."""


async def wake_phrase_for(guild) -> str:
    stored = await configManager().get_setting("voice_wake_word", guild.id)
    return stored or default_wake_phrase(getattr(guild.me, "display_name", "") or "")


async def start(client, voice_channel, text_channel, requested_by) -> "VoiceSession":
    """Join `voice_channel` and start a session; raises VoiceError."""
    if not voice_policy.settings()["enabled"]:
        raise VoiceError("Voice is switched off on this bot.")
    bridge = get_bridge()
    if bridge is None or not bridge.connected:
        raise VoiceError("The voice service isn't reachable right now. Try again in a minute.")
    guild = voice_channel.guild
    existing = voice_gate.session(guild.id)
    if existing is not None:
        if existing.voice_channel.id == voice_channel.id:
            raise VoiceError(f"I'm already in {voice_channel.mention}.")
        raise VoiceError(f"I'm already in a call in {existing.voice_channel.mention}. "
                         f"Use /voice leave first.")
    if guild.voice_client is not None:
        raise VoiceError("I'm already connected to voice in this server.")
    permissions = voice_channel.permissions_for(guild.me)
    if not (permissions.connect and permissions.speak):
        raise VoiceError(f"I need the Connect and Speak permissions in {voice_channel.mention}.")

    session = VoiceSession(client, bridge, voice_channel, requested_by)
    # Entered first: from here on nothing new starts in text for this guild.
    voice_gate.enter(guild.id, session)
    set_voice_sessions(len(voice_gate.sessions()))
    try:
        session.text_channel = await _session_channel(text_channel, voice_channel)
        await voice_channel.connect(cls=SidecarVoiceProtocol, timeout=CONNECT_TIMEOUT,
                                    self_deaf=False, self_mute=False)
    except BaseException as e:
        voice_gate.leave(guild.id, session)
        set_voice_sessions(len(voice_gate.sessions()))
        if isinstance(e, asyncio.TimeoutError):
            raise VoiceError("Joining the voice channel timed out.") from None
        if isinstance(e, (VoiceUnavailable, discord.ClientException)):
            raise VoiceError(str(e)) from None
        raise
    await session.begin()
    return session


async def _session_channel(text_channel, voice_channel):
    """Where the call's text goes: a thread off the channel the join was
    asked from, so pictures and sandbox runs (which need a thread for their
    workspace and steering) land together. Falls back to the voice channel's
    own chat, which cannot hold threads, when no thread can be made."""
    if isinstance(text_channel, discord.Thread):
        return text_channel
    if isinstance(text_channel, discord.TextChannel):
        name = f"🎙️ Voice: {voice_channel.name}"[:100]
        # The bot's own still-active thread for this voice channel is reused,
        # so calls do not leave a trail of threads, and a sandbox workspace
        # (keyed by thread) carries over to the next call.
        for thread in text_channel.threads:
            if thread.name == name and thread.owner_id == text_channel.guild.me.id and not thread.locked:
                return thread
        permissions = text_channel.permissions_for(text_channel.guild.me)
        if permissions.create_public_threads and permissions.send_messages_in_threads:
            try:
                return await text_channel.create_thread(
                    name=name,
                    type=discord.ChannelType.public_thread,
                    auto_archive_duration=1440,
                )
            except discord.HTTPException as e:
                logger.warning(f"Voice: could not open a session thread: {e}")
    return voice_channel


class _OrderedSpeech:
    """A turn's spoken output, sent to the sidecar strictly in order.

    Items are sentences or pending hold-on lines (tasks that resolve to a
    sentence), so an answer streamed while its hold-on line is still being
    written waits for that line instead of overtaking it."""

    def __init__(self, send):
        self._send = send
        self._queue: asyncio.Queue = asyncio.Queue()
        self._task = asyncio.create_task(self._run())

    def put(self, item) -> None:
        self._queue.put_nowait(item)

    async def close(self) -> None:
        self._queue.put_nowait(None)
        await self._task

    async def _run(self) -> None:
        while True:
            item = await self._queue.get()
            if item is None:
                return
            try:
                text = await item if asyncio.isfuture(item) or asyncio.iscoroutine(item) else item
            except Exception as e:
                logger.warning(f"Voice: a hold-on line failed: {e}")
                continue
            if text:
                await self._send(text)


class VoiceSession:
    def __init__(self, client, bridge, voice_channel, requested_by):
        self.client = client
        self.bridge = bridge
        self.guild = voice_channel.guild
        self.voice_channel = voice_channel
        self.text_channel = None
        self.requested_by_id = getattr(requested_by, "id", None)
        self.settings = voice_policy.settings()
        self.history = VoiceHistory(self.settings["history_limit"], self.settings["history_refresh"])
        # A phrase set with /voice wake_word; empty means "hey <name>".
        self.custom_wake_phrase = ""
        self.voice = None
        self.speed = 1.0
        self.ending = False
        # Set by the leave_voice_channel tool: leave once this turn is spoken.
        self.leave_after_turn = False
        self.turns: asyncio.Queue = asyncio.Queue(maxsize=2)
        self.turn_running = False
        self._current_turn = None
        self._muted_turns: set[str] = set()
        self._pending_speech: dict[str, int] = {}
        self._speech_done: dict[str, asyncio.Event] = {}
        self._followup_user = None
        self._followup_until = 0.0
        self._last_prefill = 0.0
        self._last_busy_notice = 0.0
        self._worker = None
        self._idle_task = None
        self._prefill_task = None

    @property
    def guild_key(self) -> str:
        return str(self.guild.id)

    @property
    def bot_name(self) -> str:
        return getattr(self.guild.me, "display_name", "") or ""

    @property
    def wake_phrase(self) -> str:
        """Read on every utterance, so renaming the bot changes the default
        phrase in a call already running (discord.py keeps guild.me current)."""
        return self.custom_wake_phrase or default_wake_phrase(self.bot_name)

    # -- lifecycle ------------------------------------------------------------

    async def begin(self) -> None:
        await self.refresh_settings()
        self._worker = asyncio.create_task(self._run_turns())
        await self._post(
            f"🎙️ I'm in {self.voice_channel.mention}. Say **\"{self.wake_phrase}\"** and then what "
            f"you'd like. Pictures and code results will appear here. Say "
            f"\"{self.wake_phrase}, leave\" or use `/voice leave` to end the call.")
        await self._speak_line(f"Hi! Say {self.wake_phrase} when you need me.")
        self.members_changed()

    async def refresh_settings(self) -> None:
        """Re-read the per-guild voice settings."""
        config = configManager()
        self.custom_wake_phrase = await config.get_setting("voice_wake_word", self.guild.id) or ""
        self.voice = await config.get_setting("voice_name", self.guild.id) or self.settings["default_voice"]
        try:
            self.speed = float(await config.get_setting("voice_speed", self.guild.id) or 1.0)
        except (TypeError, ValueError):
            self.speed = 1.0
        await self._configure_sidecar()

    async def _configure_sidecar(self) -> None:
        bots = [str(m.id) for m in self.voice_channel.members if m.bot]
        await self.bridge.send({"type": "config", "guild_id": self.guild_key,
                                "ignore_user_ids": bots})

    async def end(self, note: str | None = None) -> None:
        """Leave the call. Safe to call more than once and from anywhere.

        A turn already running is not cancelled -- a sandbox run or an image
        in progress finishes and still posts in the thread -- it just stops
        being spoken. Queued turns are dropped."""
        if self.ending:
            return
        self.ending = True
        voice_gate.leave(self.guild.id, self)
        set_voice_sessions(len(voice_gate.sessions()))
        current = asyncio.current_task()
        for task in (self._worker, self._idle_task, self._prefill_task):
            # Not the task running this (the worker ends the call itself after
            # a leave_voice_channel turn): cancelling it would cut the
            # disconnect below short.
            if task is not None and not task.done() and task is not current:
                task.cancel()
        voice_client = self.guild.voice_client
        if voice_client is not None:
            try:
                await voice_client.disconnect(force=True)
            except Exception as e:
                logger.warning(f"Voice: leaving guild {self.guild.id} was not clean: {e}")
        if note:
            await self._post(f"👋 {note}")
        logger.info(f"Voice: session in guild {self.guild.id} ended ({note or 'asked'})")

    def moved(self, channel) -> None:
        """Someone dragged the bot to another channel: the call goes with it."""
        if channel is not None and channel.id != self.voice_channel.id:
            self.voice_channel = channel
            asyncio.create_task(self._configure_sidecar())
        self.members_changed()

    def members_changed(self) -> None:
        """Someone joined or left the call: leave once nobody is left."""
        if self.ending:
            return
        humans = [m for m in self.voice_channel.members if not m.bot]
        if humans:
            if self._idle_task is not None:
                self._idle_task.cancel()
                self._idle_task = None
        elif self._idle_task is None or self._idle_task.done():
            self._idle_task = asyncio.create_task(self._leave_when_alone())

    async def _leave_when_alone(self) -> None:
        await asyncio.sleep(self.settings["idle_leave_seconds"])
        if not [m for m in self.voice_channel.members if not m.bot]:
            await self.end("Everyone left the call, so I left too.")

    # -- hearing ----------------------------------------------------------------

    async def on_utterance(self, message: dict) -> None:
        observe_voice_speech("stt", message.get("stt_seconds"))
        try:
            member = self.guild.get_member(int(message["user_id"]))
        except (KeyError, TypeError, ValueError):
            return
        if member is None or member.bot:
            return
        await self.heard(member.id, member.display_name, message.get("text") or "")

    async def heard(self, user_id: int, name: str, text: str) -> None:
        """One transcribed utterance (also the dev-inject entry point)."""
        text = (text or "").strip()
        if not text or self.ending:
            return
        self.history.add(utterance_message(name, text))
        # DEBUG, not INFO: this is everything anyone in the call says.
        logger.debug(f"Voice: heard {name}: {text}")
        now = time.monotonic()
        matched, rest = match_wake(text, self.wake_phrase)
        if not matched and self._followup_user == user_id and now < self._followup_until:
            matched, rest = True, text
        if not matched:
            inc_voice_utterance("ignored")
            self._schedule_prefill()
            return
        command = parse_command(rest)
        if command == LEAVE:
            inc_voice_utterance("leave")
            await self.end("Left the call when asked.")
            return
        if command == STOP:
            inc_voice_utterance("stop")
            await self.stop_speaking()
            self._schedule_prefill()
            return
        if not rest:
            inc_voice_utterance("wake_only")
            self._followup_user = user_id
            self._followup_until = now + self.settings["followup_seconds"]
            await self._chime()
            self._schedule_prefill()
            return
        self._followup_user = None
        inc_voice_utterance("request")
        logger.info(f"Voice: request in guild {self.guild.id}")
        try:
            self.turns.put_nowait((user_id, text, now))
        except asyncio.QueueFull:
            inc_voice_turn("busy")
            if now - self._last_busy_notice > BUSY_NOTICE_SECONDS:
                self._last_busy_notice = now
                await self._speak_line("One moment, I'm still on the last one.")

    def on_speech(self, message: dict) -> None:
        """Playback progress from the sidecar."""
        state = message.get("state")
        if state == "started":
            observe_voice_speech("tts", message.get("synth_seconds"))
            return
        if state not in ("done", "failed"):
            return
        turn_id = message.get("id")
        if state == "failed":
            logger.warning(f"Voice: speech failed in guild {self.guild.id}: {message.get('error')}")
        if turn_id in self._pending_speech:
            self._pending_speech[turn_id] -= 1
            if self._pending_speech[turn_id] <= 0:
                self._pending_speech.pop(turn_id, None)
                event = self._speech_done.pop(turn_id, None)
                if event is not None:
                    event.set()

    async def stop_speaking(self) -> None:
        """Stop talking (the turn's work, a sandbox run say, carries on)."""
        if self._current_turn is not None:
            self._muted_turns.add(self._current_turn)
        await self.bridge.send({"type": "stop", "guild_id": self.guild_key})

    # -- answering --------------------------------------------------------------

    async def _run_turns(self) -> None:
        while True:
            user_id, text, heard_at = await self.turns.get()
            if self.ending:
                continue
            self.turn_running = True
            if self._prefill_task is not None:
                self._prefill_task.cancel()
            turn = asyncio.create_task(self._turn(user_id, text, heard_at))
            try:
                # Shielded: end() cancels this worker, but a turn already
                # running is left to finish (its tools may be mid-way).
                await asyncio.shield(turn)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Voice: turn failed")
            finally:
                self.turn_running = False
            if self.leave_after_turn and not self.ending:
                await self.end("Left the call when asked.")
                return
            self._schedule_prefill()

    async def _turn(self, user_id: int, text: str, heard_at: float) -> None:
        from classes.text_llm_handler import TextLLMHandler

        turn_id = uuid.uuid4().hex[:12]
        self._current_turn = turn_id
        first_audio = []
        # Played when the turn starts, not when the request is heard: the
        # sidecar plays one FIFO, so a chime sent while the last answer is
        # still being streamed would land in the middle of it.
        await self._chime()

        async def send(sentence: str) -> None:
            if self.ending or turn_id in self._muted_turns:
                return
            logger.debug(f"Voice: saying {sentence}")
            if await self._send_speech(turn_id, sentence) and not first_audio:
                first_audio.append(True)
                observe_voice_first_audio(time.monotonic() - heard_at)
                logger.info(f"Voice: first sentence {time.monotonic() - heard_at:.2f}s after the request")

        speech = _OrderedSpeech(send)
        splitter = SentenceSplitter()
        spoke = {"this_call": False}

        async def on_text(delta: str) -> None:
            for sentence in splitter.feed(delta):
                spoke["this_call"] = True
                speech.put(sentence)

        async def on_model_start() -> None:
            for sentence in splitter.flush():
                speech.put(sentence)
            spoke["this_call"] = False

        async def on_tool(name: str, arguments: str) -> None:
            for sentence in splitter.flush():
                spoke["this_call"] = True
                speech.put(sentence)
            if not spoke["this_call"]:
                speech.put(asyncio.create_task(self._hold_on_line(name, arguments, text)))
            # One line per model call, however many tools it calls at once.
            spoke["this_call"] = True

        handler = TextLLMHandler(self.history.messages(), self.guild.id, None, client=self.client,
                                 actor_id=user_id, channel=self.text_channel, voice=True,
                                 request_text=text, bot_name=self.bot_name)
        try:
            answer = await handler.generate_streamed(on_text, on_model_start, on_tool)
            for sentence in splitter.flush():
                speech.put(sentence)
            if answer == "Error":
                inc_voice_turn("error")
                speech.put("Sorry, something went wrong with that one.")
            else:
                inc_voice_turn("answered")
                self.history.add({"role": "assistant", "content": answer.strip() or "..."})
        finally:
            await speech.close()
        if self.leave_after_turn:
            await self._wait_spoken(turn_id, GOODBYE_WAIT_SECONDS)

    async def _hold_on_line(self, tool: str, arguments: str, request: str) -> str:
        """One short in-character line for a tool the model called silently."""
        detail = _main_argument(arguments)
        try:
            persona = await configManager().get_setting("system", self.guild.id) or "An AI Story Teller"
            line = await _hold_on_llm.complete(hold_on_messages(persona, tool_action(tool), detail, request),
                                               temperature=0.9, max_tokens=40)
            line = speakable(line or "")
            if line:
                return line
        except Exception as e:
            logger.info(f"Voice: hold-on line failed ({e}); using the plain one")
        return f"Hold on, I'm {tool_action(tool)}."

    async def _send_speech(self, turn_id: str, text: str) -> bool:
        sent = await self.bridge.send({"type": "speak", "guild_id": self.guild_key, "id": turn_id,
                                       "text": text[:1000], "voice": self.voice, "speed": self.speed})
        if sent:
            self._pending_speech[turn_id] = self._pending_speech.get(turn_id, 0) + 1
            self._speech_done.setdefault(turn_id, asyncio.Event())
        return sent

    async def _wait_spoken(self, turn_id: str, timeout: float) -> None:
        event = self._speech_done.get(turn_id)
        if event is None:
            return
        try:
            await asyncio.wait_for(event.wait(), timeout)
        except asyncio.TimeoutError:
            pass

    async def _chime(self) -> None:
        """The "I heard you" cue, played by the sidecar with no TTS round trip."""
        if not self.ending:
            await self.bridge.send({"type": "chime", "guild_id": self.guild_key})

    async def _speak_line(self, text: str) -> None:
        """A one-off line outside any turn (greeting, busy notice)."""
        if not self.ending:
            await self._send_speech("line-" + uuid.uuid4().hex[:8], text)

    async def speak_sample(self) -> None:
        await self._speak_line("This is how I sound now.")

    # -- keeping llama.cpp's cache warm ----------------------------------------

    def _schedule_prefill(self) -> None:
        if self.ending or self.turn_running or not self.turns.empty():
            return
        if self._prefill_task is not None and not self._prefill_task.done():
            # The running one reads the history when its delay is up, so it
            # already includes this utterance.
            return
        self._prefill_task = asyncio.create_task(self._prefill_soon())

    async def _prefill_soon(self) -> None:
        from classes.text_llm_handler import TextLLMHandler

        # Debounced (several people talking at once) and rate-limited: every
        # prefill is a llama.cpp job, and its checkpoints take host-RAM cache
        # space the text guilds share.
        wait = max(0.3, self._last_prefill + self.settings["prefill_min_seconds"] - time.monotonic())
        await asyncio.sleep(wait)
        if self.ending or self.turn_running or not self.turns.empty():
            return
        self._last_prefill = time.monotonic()
        handler = TextLLMHandler(self.history.messages(), self.guild.id, None, client=self.client,
                                 actor_id=self.requested_by_id or 0, channel=self.text_channel,
                                 voice=True, bot_name=self.bot_name)
        try:
            await handler.prefill()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.info(f"Voice: prefill failed: {e}")

    # -- text ---------------------------------------------------------------------

    async def _post(self, content: str) -> None:
        if self.text_channel is None:
            return
        try:
            await self.text_channel.send(content)
        except Exception as e:
            logger.warning(f"Voice: could not post to the session channel: {e}")


def _main_argument(arguments: str) -> str:
    """The argument that says what a tool call is about (query, prompt...)."""
    try:
        parsed = json.loads(arguments or "{}")
    except ValueError:
        return ""
    if not isinstance(parsed, dict):
        return ""
    for value in parsed.values():
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""

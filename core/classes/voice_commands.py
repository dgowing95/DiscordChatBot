"""/voice: join and leave calls, and set the wake phrase and the voice."""
import logging
import time

import aiohttp
import discord
from discord import app_commands

from classes import card, card_kit as K

from classes import voice_gate, voice_session
from classes.config_manager import configManager
from classes.voice_policy import settings, validate_wake_phrase, voice_names

logger = logging.getLogger(__name__)

MIN_SPEED = 0.5
MAX_SPEED = 2.0
_VOICES_TTL = 600
# After a failed lookup: autocomplete calls this on every keystroke, and each
# would otherwise wait out the timeout again.
_VOICES_RETRY_SECONDS = 30
_voices_cache: tuple[float, list[str]] = (0.0, [])
_voices_failed_at = float("-inf")


async def available_voices() -> list[str]:
    """The speech service's voice ids (cached; empty when unreachable)."""
    global _voices_cache, _voices_failed_at
    fetched_at, voices = _voices_cache
    if voices and time.monotonic() - fetched_at < _VOICES_TTL:
        return voices
    if time.monotonic() - _voices_failed_at < _VOICES_RETRY_SECONDS:
        return voices
    url = settings()["speech_url"].rstrip("/") + "/voices"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    voices = list((await response.json()).get("voices") or [])
                    _voices_cache = (time.monotonic(), voices)
                    return voices
                logger.info(f"Voice: could not list voices at {url}: HTTP {response.status}")
    except Exception as e:
        logger.info(f"Voice: could not list voices at {url}: {e}")
    _voices_failed_at = time.monotonic()
    return voices


async def _voice_autocomplete(ctx: discord.Interaction, current: str):
    # A blend is typed by hand ("af_bella:0.6,am_adam:0.4"); only the part
    # after the last comma is completed.
    head, _, tail = current.rpartition(",")
    prefix = f"{head}," if head else ""
    return [app_commands.Choice(name=prefix + v, value=prefix + v)
            for v in await available_voices() if tail.strip().lower() in v][:25]


VOICE_LABEL = "🎙️ Voice call"
VOICE_SETTINGS = "🎙️ Voice settings"


class LeaveButton(discord.ui.DynamicItem[discord.ui.Button], template=r"voice:leave"):
    """Leave, on a call's card: what /voice leave does. A DynamicItem
    registered once (main.on_ready), so it answers on a card from before a
    restart too (by then there is no call)."""

    def __init__(self):
        super().__init__(discord.ui.Button(label="Leave", emoji="👋", custom_id="voice:leave",
                                           style=discord.ButtonStyle.secondary))

    @classmethod
    async def from_custom_id(cls, interaction, item, match):
        return cls()

    async def callback(self, interaction: discord.Interaction) -> None:
        session = voice_gate.session(interaction.guild_id)
        if session is None:
            await card.reply(interaction, K.info(VOICE_LABEL, "I'm not in a voice call."))
            return
        await card.reply(interaction, K.info(VOICE_LABEL, "👋 Leaving the call."), ephemeral=False)
        await session.end()


def call_card(channel_mention: str, wake_phrase: str, where: str = "") -> K.CardSpec:
    """The call's card: where the bot is, how to talk to it, and Leave."""
    text = f'Say **"{K.safe(wake_phrase)}"** and then what you\'d like.{where}'
    return K.CardSpec(body=[K.Section((K.small(VOICE_LABEL), f"### I'm in {channel_mention}", text),
                                      LeaveButton())])


def register_voice_commands(tree, client) -> None:
    group = app_commands.Group(name="voice", description="Talk with me in a voice channel",
                               guild_only=True)

    @group.command(name="join", description="Join a voice channel and talk out loud (the one you're in by default)")
    @app_commands.describe(channel="The voice channel to join")
    async def join(ctx: discord.Interaction, channel: discord.VoiceChannel | None = None):
        target = channel or getattr(getattr(ctx.user, "voice", None), "channel", None)
        if target is None:
            await card.reply(ctx, K.heads_up(VOICE_LABEL, "Join a voice channel first, or pick one."))
            return
        await ctx.response.defer()
        try:
            session = await voice_session.start(client, target, ctx.channel, ctx.user)
        except voice_session.VoiceError as e:
            await card.reply(ctx, K.error(str(e), "❌ Couldn't join the call"), ephemeral=False)
            return
        where = (f" Pictures and code results go in {session.text_channel.mention}."
                 if session.text_channel is not None and session.text_channel.id != ctx.channel_id else "")
        await card.reply(ctx, call_card(target.mention, session.wake_phrase, where), ephemeral=False)

    @group.command(name="leave", description="Leave the voice call")
    async def leave(ctx: discord.Interaction):
        session = voice_gate.session(ctx.guild_id)
        if session is None:
            await card.reply(ctx, K.info(VOICE_LABEL, "I'm not in a voice call."))
            return
        await card.reply(ctx, K.info(VOICE_LABEL, "👋 Leaving the call."), ephemeral=False)
        await session.end()

    @group.command(name="wake_word", description="Change the phrase that gets my attention in voice (default: \"hey <my name>\")")
    @app_commands.describe(phrase="For example: hey sparky. \"default\" goes back to hey <my name>")
    async def wake_word(ctx: discord.Interaction, phrase: str):
        if phrase.strip().lower() == "default":
            # Stored empty, which reads back as unset.
            await configManager().update_setting("voice_wake_word", "", ctx.guild_id)
            phrase = await voice_session.wake_phrase_for(ctx.guild)
            session = voice_gate.session(ctx.guild_id)
            if session is not None:
                await session.refresh_settings()
            await card.reply(ctx, K.done(VOICE_SETTINGS, f'**Wake phrase** is back to "{K.safe(phrase)}"',
                                         footer="the default"), ephemeral=False)
            return
        try:
            phrase = validate_wake_phrase(phrase)
        except ValueError as e:
            await card.reply(ctx, K.error(str(e), "❌ Not changed"))
            return
        await configManager().update_setting("voice_wake_word", phrase, ctx.guild_id)
        session = voice_gate.session(ctx.guild_id)
        if session is not None:
            await session.refresh_settings()
        await card.reply(ctx, K.done(VOICE_SETTINGS, f'**Wake phrase** is now "{K.safe(phrase)}"'),
                         ephemeral=False)

    @group.command(name="voice", description="Change my speaking voice (blend voices like af_bella:0.6,am_adam:0.4)")
    @app_commands.describe(name="A voice, or a blend of up to four")
    @app_commands.autocomplete(name=_voice_autocomplete)
    async def voice(ctx: discord.Interaction, name: str):
        names = voice_names(name)
        known = await available_voices()
        unknown = [n for n in names if known and n not in known]
        if not names or unknown or len(names) > 4:
            reason = ("A blend can mix at most four voices." if len(names) > 4
                      else f"Unknown voice: {', '.join(unknown) or name}.")
            await card.reply(ctx, K.error(f"{reason} Start typing to pick from the list.", "❌ Not changed"))
            return
        await configManager().update_setting("voice_name", name.strip().lower(), ctx.guild_id)
        await card.reply(ctx, K.done(VOICE_SETTINGS, f"**Voice** is now `{name.strip().lower()}`"),
                         ephemeral=False)
        session = voice_gate.session(ctx.guild_id)
        if session is not None:
            await session.refresh_settings()
            await session.speak_sample()

    @group.command(name="speed", description="Change how fast I talk (0.5 to 2, default 1)")
    async def speed(ctx: discord.Interaction, speed: app_commands.Range[float, MIN_SPEED, MAX_SPEED]):
        await configManager().update_setting("voice_speed", str(speed), ctx.guild_id)
        session = voice_gate.session(ctx.guild_id)
        if session is not None:
            await session.refresh_settings()
        await card.reply(ctx, K.done(VOICE_SETTINGS, f"**Speaking speed** is now **{speed}**",
                                     footer="0.5 is slow, 2 is fast"), ephemeral=False)

    @group.command(name="settings", description="Show my voice settings for this server")
    async def show_settings(ctx: discord.Interaction):
        config = configManager()
        phrase = await voice_session.wake_phrase_for(ctx.guild)
        name = await config.get_setting("voice_name", ctx.guild_id) or settings()["default_voice"]
        speed_value = await config.get_setting("voice_speed", ctx.guild_id) or "1.0"
        session = voice_gate.session(ctx.guild_id)
        status = f"in {session.voice_channel.mention}" if session is not None else "not in a call"
        await card.reply(ctx, K.info(VOICE_SETTINGS, K.details([
            ("Wake phrase", f'"{K.safe(phrase)}"'), ("Voice", f"`{name}`"), ("Speed", speed_value),
            ("Right now", status)])))

    tree.add_command(group)

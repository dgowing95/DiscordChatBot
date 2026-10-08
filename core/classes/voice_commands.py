"""/voice: join and leave calls, and set the wake phrase and the voice."""
import logging
import time

import aiohttp
import discord
from discord import app_commands

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


def register_voice_commands(tree, client) -> None:
    group = app_commands.Group(name="voice", description="Talk with me in a voice channel",
                               guild_only=True)

    @group.command(name="join", description="Join a voice channel and talk out loud (the one you're in by default)")
    @app_commands.describe(channel="The voice channel to join")
    async def join(ctx: discord.Interaction, channel: discord.VoiceChannel | None = None):
        target = channel or getattr(getattr(ctx.user, "voice", None), "channel", None)
        if target is None:
            await ctx.response.send_message("Join a voice channel first, or pick one.", ephemeral=True)
            return
        await ctx.response.defer()
        try:
            session = await voice_session.start(client, target, ctx.channel, ctx.user)
        except voice_session.VoiceError as e:
            await ctx.followup.send(f"❌ {e}")
            return
        where = (f" Pictures and code results go in {session.text_channel.mention}."
                 if session.text_channel is not None and session.text_channel.id != ctx.channel_id else "")
        await ctx.followup.send(
            f"🎙️ Joined {target.mention}. Say **\"{session.wake_phrase}\"** to talk to me.{where}")

    @group.command(name="leave", description="Leave the voice call")
    async def leave(ctx: discord.Interaction):
        session = voice_gate.session(ctx.guild_id)
        if session is None:
            await ctx.response.send_message("I'm not in a voice call.", ephemeral=True)
            return
        await ctx.response.send_message("👋 Leaving the call.")
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
            await ctx.response.send_message(f"Wake phrase is back to the default: \"{phrase}\"")
            return
        try:
            phrase = validate_wake_phrase(phrase)
        except ValueError as e:
            await ctx.response.send_message(f"❌ {e}", ephemeral=True)
            return
        await configManager().update_setting("voice_wake_word", phrase, ctx.guild_id)
        session = voice_gate.session(ctx.guild_id)
        if session is not None:
            await session.refresh_settings()
        await ctx.response.send_message(f"Wake phrase is now: \"{phrase}\"")

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
            await ctx.response.send_message(f"❌ {reason} Start typing to pick from the list.",
                                            ephemeral=True)
            return
        await configManager().update_setting("voice_name", name.strip().lower(), ctx.guild_id)
        await ctx.response.send_message(f"Voice is now: `{name.strip().lower()}`")
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
        await ctx.response.send_message(f"Speaking speed is now: {speed}")

    @group.command(name="settings", description="Show my voice settings for this server")
    async def show_settings(ctx: discord.Interaction):
        config = configManager()
        phrase = await voice_session.wake_phrase_for(ctx.guild)
        name = await config.get_setting("voice_name", ctx.guild_id) or settings()["default_voice"]
        speed_value = await config.get_setting("voice_speed", ctx.guild_id) or "1.0"
        session = voice_gate.session(ctx.guild_id)
        status = f"in {session.voice_channel.mention}" if session is not None else "not in a call"
        await ctx.response.send_message(
            f"Wake phrase: \"{phrase}\"\nVoice: `{name}`\nSpeed: {speed_value}\nRight now: {status}",
            ephemeral=True)

    tree.add_command(group)

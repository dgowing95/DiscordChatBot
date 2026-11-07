import logging

import discord

# Discord's hard cap on an embed description's length. A tool call whose
# description (e.g. a long task string) exceeds this raises
# discord.HTTPException from channel.send() below, with no other guard
# anywhere in the tool-calling path.
EMBED_DESCRIPTION_MAX_CHARS = 4096

class Common:
    """
    Common class for shared functionality.
    """

    @staticmethod
    def tool_embed(description, color=0x00b0f4, title=None, footer=None):
        if len(description) > EMBED_DESCRIPTION_MAX_CHARS:
            description = description[: EMBED_DESCRIPTION_MAX_CHARS - 2] + "… "
        embed = discord.Embed(title=title, description=description, color=color)
        if footer:
            embed.set_footer(text=footer)
        return embed

    @staticmethod
    async def send_tool_discord_embed(channel, description, color=0x00b0f4,
                                      title=None, footer=None):
        """
        Sends a tool's embed and returns the sent message (for
        edit_tool_discord_embed).

        The title names the action and the description is only its subject;
        both, and the colours, come from classes/tool_embeds.py, which says
        why. There is no default title: the old shared "Tool Usage" said
        nothing about which tool ran.
        """
        return await channel.send(embed=Common.tool_embed(description, color, title, footer))

    @staticmethod
    async def edit_tool_discord_embed(message, description, color=0x00b0f4,
                                      title=None, footer=None):
        """Replaces an embed sent above. Never raises: the embed is a progress
        note, and failing to update it must not cost the tool's result."""
        try:
            await message.edit(embed=Common.tool_embed(description, color, title, footer))
        except Exception as e:
            logging.getLogger(__name__).warning(f"Could not update a tool embed: {e}")


logger = logging.getLogger(__name__)


class _NoMessage:
    """What a dropped embed "sent", for a tool that edits its embed later."""

    async def edit(self, **kwargs) -> None:
        pass


class QuietChannel:
    """The channel a workflow's scouts and researchers run with: a tool's own
    embed goes nowhere, and everything else is the real channel. Each posts
    one embed per tool call, dozens in a run, which buried the thread. Their
    one status embed (brain_progress.HelperProgress) says what they are doing."""

    def __init__(self, channel):
        self._channel = channel

    def __getattr__(self, name):
        return getattr(self._channel, name)

    async def send(self, *args, **kwargs):
        if (not args and kwargs.get("embed") is not None
                and not kwargs.get("file") and not kwargs.get("files")):
            return _NoMessage()
        return await self._channel.send(*args, **kwargs)


def embed_from_data(data: dict) -> discord.Embed:
    """A discord.Embed from the plain dicts the pure modules build
    (whats_new.notes_embed_data, help_catalog.help_embed_data), which keep
    those modules free of the discord import and unit-testable."""
    embed = discord.Embed(title=data.get("title"), color=data.get("color"))
    for field in data.get("fields", ()):
        embed.add_field(name=field["name"], value=field["value"], inline=False)
    if data.get("footer"):
        embed.set_footer(text=data["footer"])
    return embed

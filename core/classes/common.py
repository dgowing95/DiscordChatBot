import io
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

# Discord's cap on attachments per message.
MAX_FILES_PER_MESSAGE = 10


async def send_with_files(channel, content: str | None, files) -> None:
    """Sends `content` with `files` ((name, bytes) pairs) attached: ten to a
    message, the text on the first. Several files on one message can pass
    the server's upload limit when each alone did not, so a message that is
    refused is resent one file per message (the text going alone first).
    Each discord.File is built here, per send: one is used up by sending it.
    Only a text-only send (no files) can raise; past that, failures are logged.
    """
    files = list(files or [])
    if not files:
        if content:
            await channel.send(content)
        return
    for start in range(0, len(files), MAX_FILES_PER_MESSAGE):
        group = files[start:start + MAX_FILES_PER_MESSAGE]
        text = content if start == 0 else None
        try:
            await channel.send(content=text, files=[
                discord.File(io.BytesIO(data), filename=name) for name, data in group])
        except Exception as e:
            logger.info(f"Sending {len(group)} file(s) together failed ({e!r}); sending one by one")
            if text:
                try:
                    await channel.send(text)
                except Exception as e:
                    logger.warning(f"Could not send the text with the files: {e!r}")
            for name, data in group:
                try:
                    await channel.send(file=discord.File(io.BytesIO(data), filename=name))
                except Exception as e:
                    logger.warning(f"Could not send {name}: {e}")


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

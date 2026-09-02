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
    async def send_tool_discord_embed(channel, description, color=0x00b0f4,
                                      title="Tool Usage"):
        """
        Sends a Discord embed with the given title, description, and color.

        title defaults to "Tool Usage" — every caller that announces a tool
        ABOUT to run leaves it alone. It is overridden only where "Tool
        Usage" would be actively wrong, e.g. the sandbox's closing note,
        which reports on a run that has already finished.
        """
        if len(description) > EMBED_DESCRIPTION_MAX_CHARS:
            description = description[: EMBED_DESCRIPTION_MAX_CHARS - 2] + "… "

        embed = discord.Embed(
                title=title,
                description=description,
                color=color
        )
        await channel.send(embed=embed)


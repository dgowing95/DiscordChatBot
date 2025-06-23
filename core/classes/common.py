import discord

class Common:
    """
    Common class for shared functionality.
    """

    @staticmethod
    async def send_tool_discord_embed(channel, description, color=0x00b0f4):
        """
        Sends a Discord embed with the given title, description, and color.
        """

        embed = discord.Embed(
                title="Tool Usage",
                description=description,
                color=color
        )
        await channel.send(embed=embed)


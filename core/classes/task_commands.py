"""/task: start a Tier 3 workflow directly (classes/workflows.py).

It posts the request publicly ("🧵 Workflow: ..."), which is the message the
workflow's thread opens from, then starts the workflow. The brain can start
one too (start_workflow); /task is the way to ask for one outright. When it
cannot start (one already runs in the server, threads are not allowed
here), the public message says why.
"""
import logging

import discord
from discord import app_commands

logger = logging.getLogger(__name__)

MAX_REQUEST_CHARS = 1800


def register_task_commands(tree, client) -> None:
    @tree.command(name="task", description="Start a long job in its own thread: a plan you approve, then the work")
    @app_commands.guild_only()
    @app_commands.describe(request="What you want done, with every detail")
    async def task(ctx: discord.Interaction, request: app_commands.Range[str, 1, MAX_REQUEST_CHARS]):
        from classes import workflows
        await ctx.response.send_message(f"🧵 Workflow: {request}",
                                        allowed_mentions=discord.AllowedMentions.none())
        message = await ctx.original_response()
        run_ctx = {"guild_id": ctx.guild_id, "user_id": ctx.user.id,
                   "requester_name": getattr(ctx.user, "display_name", None) or ctx.user.name,
                   "channel": ctx.channel, "discord_client": client}
        result = await workflows.start_workflow(run_ctx, request, trigger=workflows.TRIGGER_SLASH,
                                                message=message)
        if not result.ok:
            try:
                await message.edit(content=f"🧵 Workflow: {request}\n-# ❌ Not started. "
                                           f"{_plain(result.text)}"[:2000])
            except Exception as e:
                logger.warning(f"/task: could not say why it did not start: {e}")


def _plain(text: str) -> str:
    """The refusal written for the persona, minus its instructions to it."""
    for tail in (" Tell them where it is.", " Say so.", " Do what you can in this reply."):
        text = text.replace(tail, "")
    return text

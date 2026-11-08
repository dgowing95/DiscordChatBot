"""/task: start a Tier 3 workflow directly (classes/workflows.py).

It posts the request publicly as a card ("🧵 Big job"), the message the
workflow's thread opens from, then starts the workflow. The brain can start
one too (start_workflow); /task is the way to ask for one outright. When it
cannot start (one already runs in the server, threads are not allowed
here), the public message says why.
"""
import logging

import discord
from discord import app_commands

from classes import card, card_kit as K

logger = logging.getLogger(__name__)

MAX_REQUEST_CHARS = 1800
PLAN_COMING = "A plan will appear in the thread for you to approve."
VOICE_BUSY_LABEL = "🎙️ In a voice call"
VOICE_BUSY_REPLY = "I'm in a voice call right now, so I can't start a big job. Try again after /voice leave."


def register_task_commands(tree, client) -> None:
    @tree.command(name="task", description="Start a long job in its own thread: a plan you approve, then the work")
    @app_commands.guild_only()
    @app_commands.describe(request="What you want done, with every detail")
    async def task(ctx: discord.Interaction, request: app_commands.Range[str, 1, MAX_REQUEST_CHARS]):
        from classes import voice_gate, workflows
        # Text stops while the bot is in a call here (classes/voice_gate.py),
        # so no new workflow either. One already running carries on in its
        # thread: its agents are on lanes below the call's.
        if voice_gate.active(ctx.guild_id):
            await card.reply(ctx, K.heads_up(VOICE_BUSY_LABEL, VOICE_BUSY_REPLY))
            return
        sent = await card.reply(ctx, workflows.starter_card(request, PLAN_COMING), ephemeral=False)
        message = sent[0]
        run_ctx = {"guild_id": ctx.guild_id, "user_id": ctx.user.id,
                   "requester_name": getattr(ctx.user, "display_name", None) or ctx.user.name,
                   "channel": ctx.channel, "discord_client": client}
        result = await workflows.start_workflow(run_ctx, request, trigger=workflows.TRIGGER_SLASH,
                                                message=message)
        if not result.ok:
            try:
                # The reason first (it carries the link to the workflow
                # already running), so it is on the first message even when
                # a long request takes more than one.
                spec = K.CardSpec(label="🧵 Big job · not started", colour=K.FAIL, body=[
                    K.Text(f"❌ {_plain(result.text)}"), K.Divider(), K.Text(K.quote(request))])
                views = card.views(spec)
                for part, view in zip(sent, views):
                    await part.edit(view=view, allowed_mentions=card.NO_MENTIONS)
                for extra in sent[len(views):]:  # no "a plan will appear" left up
                    await extra.delete()
                for view in views[len(sent):]:  # longer now: the rest of the request
                    await ctx.followup.send(view=view, allowed_mentions=card.NO_MENTIONS, wait=True)
            except Exception as e:
                logger.warning(f"/task: could not say why it did not start: {e}")


def _plain(text: str) -> str:
    """The refusal written for the persona, minus its instructions to it."""
    for tail in (" Tell them where it is.", " Say so.", " Do what you can in this reply."):
        text = text.replace(tail, "")
    return text

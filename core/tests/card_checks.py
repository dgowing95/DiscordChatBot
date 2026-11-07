"""Helpers for tests that check what card was sent (classes/card.py). Not a
test module itself; import it by its bare name, like any sibling helper."""
from unittest.mock import AsyncMock, MagicMock

import discord


def view_text(view) -> str:
    """Every text of a sent view, in order."""
    return "\n".join(item.content for item in view.walk_children()
                     if isinstance(item, discord.ui.TextDisplay))


def label_of(view) -> str:
    """A card's label without its details: "📘 New skill saved" from
    "-# 📘 New skill saved · v1"."""
    for item in view.walk_children():
        if isinstance(item, discord.ui.TextDisplay):
            return item.content.removeprefix("-# ").split(" · ")[0]
    return ""


def sent_views(mock) -> list:
    """The views every call of an AsyncMock send/send_message/edit carried."""
    return [c.kwargs["view"] for c in mock.call_args_list if c.kwargs.get("view") is not None]


def sent_text(mock) -> str:
    return "\n".join(view_text(v) for v in sent_views(mock))


def labels(channel) -> list:
    return [label_of(v) for v in sent_views(channel.send)]


def interaction(**attrs):
    """A slash command's or a button's interaction, as card.reply uses it."""
    ctx = MagicMock()
    ctx.response.send_message = AsyncMock()
    ctx.response.send_modal = AsyncMock()
    ctx.response.edit_message = AsyncMock()
    ctx.response.defer = AsyncMock()
    ctx.response.is_done = MagicMock(return_value=False)
    ctx.original_response = AsyncMock()
    ctx.edit_original_response = AsyncMock()
    ctx.followup.send = AsyncMock()
    for key, value in attrs.items():
        setattr(ctx, key, value)
    return ctx


def replied(ctx):
    """The view of the interaction's first answer (or its follow-up)."""
    for mock in (ctx.response.send_message, ctx.followup.send):
        if mock.call_args is not None and mock.call_args.kwargs.get("view") is not None:
            return mock.call_args.kwargs["view"]
    return None


def replied_text(ctx) -> str:
    view = replied(ctx)
    return view_text(view) if view is not None else ""

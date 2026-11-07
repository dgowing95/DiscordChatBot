"""The bot's cards in Discord: classes/card_kit.py's CardSpec as Discord's
layout components (a LayoutView: a Container of TextDisplays, Separators,
Sections, galleries and button rows), and how they are sent and edited.

The layout has rules of its own, and this is where they are kept:

- **No message content or embeds beside a card.** Everything is in it, a
  ping too.
- **A mention in a card notifies** (one in an embed never did), so
  send_card() allows none unless the caller names who to ping.
- **Never edit one to `view=None`:** that would leave an empty message, which
  Discord refuses. A card that is over is edited to its finished form.
- **4000 characters and 40 components per message.** A longer card goes over
  several messages (card_kit.split); LiveCard keeps them updated.
- **Files a card shows** ("attachment://name") must be sent with the message
  that shows them, and stay real attachments (the image-edit and chat-file
  code find them by name).
- **Its text is not message.content.** text_of() reads it back, and kind_of()
  says what it is (its container's id), for the history.

Buttons come in rows of at most 5, in four colours: blurple, grey, green, red.
"""
import asyncio
import io
import logging

import discord

from classes import card_kit as K
from classes.card_kit import CardSpec

logger = logging.getLogger(__name__)

NO_MENTIONS = discord.AllowedMentions.none()


def _accessory(item):
    if isinstance(item, K.Link):
        return discord.ui.Button(label=item.label[:K.MAX_BUTTON_LABEL], url=item.url,
                                 emoji=item.emoji, style=discord.ButtonStyle.link)
    if isinstance(item, K.Thumb):
        return discord.ui.Thumbnail(item.url)
    return item  # a ready-made discord button


def _item(block):
    if isinstance(block, K.Text):
        return discord.ui.TextDisplay(block.content)
    if isinstance(block, K.Gap):
        return discord.ui.Separator(visible=False)
    if isinstance(block, K.Divider):
        return discord.ui.Separator(visible=True)
    if isinstance(block, K.Section):
        return discord.ui.Section(*[discord.ui.TextDisplay(t) for t in block.texts],
                                  accessory=_accessory(block.accessory))
    if isinstance(block, K.Buttons):
        return discord.ui.ActionRow(*[_accessory(item) for item in block.items])
    if isinstance(block, K.Gallery):
        return discord.ui.MediaGallery(*[discord.MediaGalleryItem(url) for url in block.urls])
    if isinstance(block, K.FileRef):
        return discord.ui.File(f"attachment://{block.name}")
    if isinstance(block, K.Thumb):
        return discord.ui.Thumbnail(block.url)
    return block  # already a discord item


def safe_filename(name: str) -> str:
    """A file name Discord keeps unchanged on upload (it rewrites spaces and
    other characters, which would break a card's attachment:// link)."""
    import re
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name or "").strip()).strip("._") or "file"
    return base[-100:]


def files_shown(blocks) -> set:
    """The attachments these blocks show ("attachment://name")."""
    names = set()
    for block in blocks:
        if isinstance(block, K.FileRef):
            names.add(block.name)
        elif isinstance(block, K.Gallery):
            names.update(u.removeprefix("attachment://") for u in block.urls
                         if u.startswith("attachment://"))
    return names


class KitView(discord.ui.LayoutView):
    """One message of a card: one Container holding the blocks."""

    def __init__(self, blocks, *, colour: int = K.ACTIVE, kind: str = K.NOTICE,
                 timeout: float | None = None):
        super().__init__(timeout=timeout)
        self.kind = kind
        self.colour = colour
        self.blocks = list(blocks)
        self.box = discord.ui.Container(accent_colour=colour, id=K.KIND_IDS.get(kind))
        for block in self.blocks:
            self.box.add_item(_item(block))
        self.add_item(self.box)

    @property
    def buttons(self) -> list:
        """The buttons a person presses (not link buttons)."""
        return [item for item in self.walk_children()
                if isinstance(item, (discord.ui.Button, discord.ui.DynamicItem))
                and getattr(item, "url", None) is None]

    def file_names(self) -> set:
        return files_shown(self.blocks)


def views(spec: CardSpec, timeout: float | None = None) -> list[KitView]:
    """The card as one view per message."""
    return [KitView(blocks, colour=spec.colour, kind=spec.kind, timeout=timeout)
            for blocks in K.split(spec)]


def _files_for(view: KitView, files: dict) -> list:
    """Fresh discord.Files for what this view shows: one is used up by a send."""
    return [discord.File(io.BytesIO(files[name]), filename=name)
            for name in sorted(view.file_names()) if name in files]


async def _send_one(target, view, *, files, mentions, reference, mention_author,
                    ephemeral, first, interaction=None):
    kwargs = {"view": view, "allowed_mentions": mentions}
    if files:
        kwargs["files"] = files
    if interaction if interaction is not None else isinstance(target, discord.Interaction):
        if first and not target.response.is_done():
            await target.response.send_message(ephemeral=ephemeral, **kwargs)
            return await target.original_response()
        return await target.followup.send(ephemeral=ephemeral, wait=True, **kwargs)
    if first and reference is not None:
        kwargs["reference"] = reference
        if mention_author is not None:
            kwargs["mention_author"] = mention_author
    return await target.send(**kwargs)


async def send_card(target, spec: CardSpec, *, files: dict | None = None,
                    mentions: discord.AllowedMentions | None = None, reference=None,
                    mention_author: bool | None = None, ephemeral: bool = False,
                    timeout: float | None = None, interaction: bool | None = None) -> list:
    """Sends the card: to a channel (anything with .send) or as the answer to
    an interaction (its continuations as follow-ups). `files` is
    {name: bytes}, each sent with the message that shows it. Returns the
    messages sent. Raises what the first send raises; a continuation that
    fails is logged."""
    files = dict(files or {})
    mentions = mentions or NO_MENTIONS
    sent = []
    for index, view in enumerate(views(spec, timeout)):
        try:
            sent.append(await _send_one(
                target, view, files=_files_for(view, files), mentions=mentions,
                reference=reference, mention_author=mention_author,
                ephemeral=ephemeral, first=index == 0, interaction=interaction))
        except Exception:
            if index == 0:
                raise
            logger.warning("Could not send the rest of a card", exc_info=True)
            break
    return sent


async def reply(interaction, spec: CardSpec, *, ephemeral: bool = True, **kwargs) -> list:
    """A slash command's or a button's answer as a card (private by default)."""
    return await send_card(interaction, spec, ephemeral=ephemeral, interaction=True, **kwargs)


async def edit_message(message, view: KitView, files: list | None = None) -> bool:
    """Replaces a message's card. Never raises: False when it failed."""
    try:
        if files:
            await message.edit(view=view, attachments=files)
        else:
            await message.edit(view=view)
        return True
    except Exception as e:
        logger.info(f"Could not update a card: {e!r}")
        return False


class LiveCard:
    """A card that changes (a status, a plan): sent once, then edited in
    place. When it outgrows its messages, the new part is sent as more
    messages; only the messages whose content changed are edited."""

    def __init__(self, target, *, mentions: discord.AllowedMentions | None = None,
                 reference=None, mention_author: bool | None = None):
        self.target = target
        self.mentions = mentions or NO_MENTIONS
        self.reference = reference
        self.mention_author = mention_author
        self.messages: list = []
        self._shown: list = []

    @property
    def message(self):
        return self.messages[0] if self.messages else None

    async def show(self, spec: CardSpec, timeout: float | None = None,
                   files: dict | None = None, edit_last_only: bool = False) -> bool:
        """Sends or updates the card. Never raises: False when something
        could not be sent or edited (the next call tries again; a message
        deleted meanwhile is sent anew). `edit_last_only`: leave the earlier
        messages as they are (a live card's full, finished part), to keep
        under Discord's edit rate."""
        files = dict(files or {})
        ok = True
        built = views(spec, timeout)
        if len(built) < len(self.messages):
            # Shorter than before (a footer or a button row that had spilled
            # over is gone): the messages it no longer needs must not stay up
            # saying "Thinking…" or offering a dead Cancel.
            ok = await self._drop(len(built), spec.colour)
        for index, view in enumerate(built):
            payload = view.to_components()
            if index < len(self.messages):
                if self._shown[index] == payload or (
                        edit_last_only and index < len(self.messages) - 1):
                    continue
                attachments = _files_for(view, files) if view.file_names() else None
                try:
                    if attachments:
                        await self.messages[index].edit(view=view, attachments=attachments)
                    else:
                        await self.messages[index].edit(view=view)
                    self._shown[index] = payload
                except discord.NotFound:
                    logger.info("A card's message was deleted; it is sent anew next time")
                    self.forget()
                    return False
                except Exception as e:
                    logger.info(f"Could not update a card: {e!r}")
                    ok = False
                continue
            try:
                message = await _send_one(
                    self.target, view, files=_files_for(view, files), mentions=self.mentions,
                    reference=self.reference, mention_author=self.mention_author,
                    ephemeral=False, first=not self.messages)
            except Exception as e:
                logger.info(f"Could not send a card: {e!r}")
                return False
            self.messages.append(message)
            self._shown.append(payload)
        return ok

    async def _drop(self, keep: int, colour: int) -> bool:
        """Removes the messages past the first `keep`: deleted, or (when that
        fails) emptied to a small grey line, since a card is never edited to
        no view. True when all went."""
        ok = True
        for message in self.messages[keep:]:
            try:
                await message.delete()
            except discord.NotFound:
                pass
            except Exception as e:
                logger.info(f"Could not delete a card's extra message ({e!r}); emptying it")
                try:
                    await message.edit(view=KitView([K.Text(K.small("⋯"))], colour=K.CLOSED))
                except Exception as e:
                    logger.info(f"Could not empty a card's extra message: {e!r}")
                    ok = False
        del self.messages[keep:], self._shown[keep:]
        return ok

    def forget(self) -> None:
        """The messages are gone (deleted): the next show() sends anew."""
        self.messages, self._shown = [], []

    @property
    def dropped(self) -> bool:
        """Nothing really posted: no message, or a QuietChannel kept it."""
        return not self.messages or getattr(self.messages[0], "dropped", False) is True

    def show_soon(self, spec: CardSpec) -> None:
        """show() in the background: an edit that only adds detail (the pages
        a search read) must not hold up the tool when Discord rate-limits
        edits (discord.py waits out a 429 inside the call)."""
        task = asyncio.create_task(self.show(spec))
        _background.add(task)
        task.add_done_callback(_background.discard)


_background: set = set()


async def notice(channel, spec: CardSpec) -> LiveCard | None:
    """Posts a tool's notice card where the run is. Never raises; None when
    there is no channel (a sandbox run may have none: its tests, and any
    caller that passes none). The LiveCard can be updated later."""
    if channel is None:
        return None
    live = LiveCard(channel)
    await live.show(spec)
    return live


# --- a card waiting on a person (a question, a plan) ----------------------------

class Card(KitView):
    """A card with buttons that wait on a person, in one message. `rows` are
    lists of buttons, plain or discord.ui.DynamicItem; `label` is the small
    line on top, `outcome` a line under the text (how it ended) and `note`
    the small line under the buttons. Its callers keep what it holds short
    (a question is at most 1800 characters)."""

    def __init__(self, text: str, *, note: str = "", rows=(), colour: int = K.ASK,
                 timeout: float | None = None, label: str = "", kind: str = K.CARD,
                 outcome: str = ""):
        self.text = str(text)
        self.note = str(note or "")
        self.label = str(label or "")
        self.outcome = str(outcome or "")
        blocks = ([K.Text(K.small(self.label))] if self.label else []) + [K.Text(self.text)]
        if self.outcome:
            blocks.append(K.Text(self.outcome))
        blocks += [K.Buttons(tuple(row)) for row in rows if row]
        if self.note:
            blocks.append(K.Text(K.small(self.note)))
        super().__init__(blocks, colour=colour, kind=kind, timeout=timeout)

    def add_row(self, buttons) -> None:
        """Another row of buttons, above the note."""
        if not buttons:
            return
        row = discord.ui.ActionRow(*buttons)
        if self.note:
            note = self.box.children[-1]
            self.box.remove_item(note)
            self.box.add_item(row)
            self.box.add_item(note)
        else:
            self.box.add_item(row)

    def finished(self, outcome: str = "") -> "Card":
        """This card with its buttons and note gone, in grey, so it never
        looks open, and `outcome` (how it ended) under its text."""
        return Card(self.text, colour=K.CLOSED, label=self.label, kind=self.kind,
                    outcome=outcome or self.outcome)


def button(label: str, style: discord.ButtonStyle, callback,
           emoji: str | None = None) -> discord.ui.Button:
    item = discord.ui.Button(label=label, style=style, emoji=emoji)
    item.callback = callback
    return item


# --- reading a card back ----------------------------------------------------------

def _texts(items, out: list) -> None:
    for item in items:
        if isinstance(item, discord.components.TextDisplay):
            out.append(item.content)
        else:
            _texts(getattr(item, "children", None) or [], out)


def texts_of(message) -> list[str]:
    """Every TextDisplay of a card posted earlier, in order ([] for any other
    message)."""
    out: list = []
    components = getattr(message, "components", None)
    if isinstance(components, (list, tuple)):
        _texts(components, out)
    return out


def text_of(message) -> str:
    """The text in a card posted earlier ("" for any other message)."""
    return "\n".join(texts_of(message))


def kind_of(message) -> str | None:
    """What a card posted earlier is (card_kit.NOTICE, ANNOUNCE, CARD), from
    its container's id; None for anything else."""
    components = getattr(message, "components", None)
    if not isinstance(components, (list, tuple)):
        return None
    for component in components:
        kind = K.kind_of_id(getattr(component, "id", None))
        if kind:
            return kind
    return None

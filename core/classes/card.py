"""A message as one card: text with its buttons inside it, for anything that
waits on a person (a question, a plan to approve).

A classic embed cannot hold buttons, which then float loose under it, so this
uses Discord's newer layout (a LayoutView: a Container of TextDisplays and
ActionRows). That layout has rules of its own, and Card is where they are kept:

- **No message content or embeds beside it.** Everything is in the card, a
  ping too, and a mention in it still notifies.
- **Never edit one to `view=None`:** that would leave an empty message, which
  Discord refuses. finished() is the same card with its buttons gone.
- **4000 characters over the whole message**, so the text is cut to fit.
- **Its text is not message.content.** text_of() reads it back, or the bot
  would not see in the history what it had asked. The message tools read the
  raw API, so they have their own (message_search_format._card_text).

Buttons come in rows of at most 5, in four colours: blurple, grey, green, red.
"""
import discord

from classes.tool_embeds import COLOR_DONE

MAX_CARD_CHARS = 3900
COLOR_CARD = 0x5865F2  # blurple: waiting on a person


class Card(discord.ui.LayoutView):
    """`rows` are lists of buttons, plain or discord.ui.DynamicItem. `note` is
    a small line under the text."""

    def __init__(self, text: str, *, note: str = "", rows=(), colour: int = COLOR_CARD,
                 timeout: float | None = None):
        super().__init__(timeout=timeout)
        self.note = str(note or "")[:400]
        self.text = str(text)[:MAX_CARD_CHARS - len(self.note)]
        self.box = discord.ui.Container(accent_colour=colour)
        self.box.add_item(discord.ui.TextDisplay(self.text))
        if self.note:
            self.box.add_item(discord.ui.TextDisplay(f"-# {self.note}"))
        self.add_item(self.box)
        for row in rows:
            self.add_row(row)

    def add_row(self, buttons) -> None:
        if buttons:
            row = discord.ui.ActionRow()
            for button in buttons:
                row.add_item(button)
            self.box.add_item(row)

    @property
    def buttons(self) -> list:
        return [item for item in self.walk_children()
                if isinstance(item, (discord.ui.Button, discord.ui.DynamicItem))]

    def finished(self, note: str = "") -> "Card":
        """This card with its buttons gone, in grey, so it never looks open.
        Its own text, whatever has changed since it was posted."""
        return Card(self.text, note=note or self.note, colour=COLOR_DONE)


def button(label: str, style: discord.ButtonStyle, callback) -> discord.ui.Button:
    item = discord.ui.Button(label=label, style=style)
    item.callback = callback
    return item


def text_of(message) -> str:
    """The text in a card posted earlier ("" for any other message)."""
    lines = []

    def walk(items) -> None:
        for item in items:
            if isinstance(item, discord.components.TextDisplay):
                lines.append(item.content)
            else:
                walk(getattr(item, "children", None) or [])

    components = getattr(message, "components", None)
    if isinstance(components, (list, tuple)):
        walk(components)
    return "\n".join(lines)

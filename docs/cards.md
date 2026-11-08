# Cards: building the bot's messages

For developers. Every message the bot posts that is not a chat reply (a tool's notice, a
status, a panel, a question, a slash command's answer) is a **card**: one Discord
Components V2 container with one look. This guide shows how to make a new one, and how to
extend the kit when what you need is not there yet. The rules themselves are in
`core/classes/card_kit.py`'s docstring and the "Cards" section of `AGENTS.md`.

| File | What it holds |
|---|---|
| `core/classes/card_kit.py` | **Pure** (stdlib only, unit-tested): colours, limits, the blocks, `CardSpec`, the everyday builders, markdown safety, splitting, reading a card back for history |
| `core/classes/card.py` | The Discord side: `views`, `send_card`, `reply`, `LiveCard`, `notice`, `Card` (questions and plans), `button`, reading a posted card (`texts_of`, `kind_of`) |
| `core/classes/tool_cards.py` | The labels and cards of tools and skills (`web_search_card`, `page_card`, `skill_card`, ...): the place for a new tool's card |
| `core/tests/card_checks.py` | Test helpers: what card was sent (`view_text`, `label_of`, `sent_views`, `interaction`, `replied_text`) |

Build the card as data (a `CardSpec`) in pure code where you can, and send it with `card.py`.
Never send a `discord.Embed`, and never put a card and message content in one message:
Discord allows neither beside Components V2.

## How a card reads

```
-# 🔎 Searching the web          <- label: small, plain words, emoji first   (spec.label)
### keyboards under £100         <- title: optional                          (spec.title)
body blocks...                    <- Text, Gap, Divider, Section, Gallery, FileRef, Buttons
-# Read 3 pages · example.com     <- footer: small details, hints, times     (spec.footer)
```

- The **label** says what this is in plain words ("🔬 Research · step 1 of 3", never "s1").
  Details after ` · ` are fine.
- **Colour is the state**, not the kind of thing. The label's emoji already says the kind.

  | Constant | Colour | Use for |
  |---|---|---|
  | `ACTIVE` | blurple | a notice, a panel, something happening now |
  | `WORKING` | yellow | live work still going |
  | `OK` | green | it worked, finished well |
  | `WARN` | orange | partly done, busy, a heads-up |
  | `FAIL` | red | failed, not allowed, a destructive confirm |
  | `CLOSED` | grey | over, answered, paused, retired |
  | `ASK` | pink | waiting on a person |

- **Blocks** (all in `card_kit`):

  | Block | Becomes | Notes |
  |---|---|---|
  | `Text(content)` | a text display | markdown; at most 4000 characters a message, all texts together |
  | `Gap()` | space, no line | between items of a list (`listing(items)` adds them) |
  | `Divider()` | a line | between groups |
  | `Section(texts, accessory)` | up to 3 texts beside a `Link`, a `Thumb` or a button | the page card's Open button, /help's avatar |
  | `Link(label, url, emoji="")` | a link button | check `tool_cards.can_link(url)` first: Discord refuses the **whole message** for a URL over 512 characters or not http(s) |
  | `Buttons(items)` | one row of up to 5 buttons | `Link`s, or ready-made `discord.ui.Button` / `DynamicItem` |
  | `Gallery(urls)` | pictures | `"attachment://name.png"` for a file sent with the message |
  | `FileRef(name)` | a file shown in the card | the file must be sent with the message (below) |

- **Everyday builders** set the colour for you: `info(label, text)`, `done(...)`,
  `heads_up(...)`, `error(text)` (label "❌ Couldn't do that"). `details([(key, value), ...])`
  gives `**Key** value` lines and skips empty values. `finished(spec, footer)` gives the same
  card in grey with its buttons gone.

## Recipes

### A one-off notice (a tool says what it did)

```python
from classes import card, card_kit as K

await card.notice(ctx.get("channel"), K.info("📊 Making a poll", K.bold(K.safe(question))))
```

`card.notice` never raises and does nothing when there is no channel, so a tool never fails
because of its card. Put a new tool's label and card builder in `tool_cards.py`, next to the
others, and test the builder there (`tool_cards_tests.py`).

### A card that changes (a status, a search that then lists what it read)

```python
live = await card.notice(channel, web_search_card(query))  # a LiveCard, or None
...
if live is not None:
    live.show_soon(web_search_card(query, sources))  # in the background: never holds the tool up
```

`LiveCard.show(spec)` sends the card the first time and edits it after that. Only the
messages whose content changed are edited. It grows into more messages when the card gets
longer, and takes down the ones it no longer needs when it gets shorter. It never raises:
`False` means "try again next time". Updates run one at a time, so two hooks updating at
once can never post the card twice. For a card edited many times a minute (live command
output), pass `edit_last_only=True` while it runs, which edits only the newest message and
keeps under Discord's edit rate. Pass it `False` for the last update.

### A slash command's or a button's answer

```python
await card.reply(interaction, K.done("✅ Saved", "The new prompt is in use."))     # private
await card.reply(interaction, K.info("🎭 Personality", text), ephemeral=False)    # public
```

`reply` answers the interaction, and any further messages of a long card go as follow-ups.
A button that changes its own card edits it instead:
`await interaction.response.edit_message(view=card.views(spec)[0])`.

### A panel with buttons (status, a schedule)

Buttons live **inside** the card, as a `Buttons` row, after a `Divider`. A panel button
does what its slash command does and **checks permission again when pressed**: the panel
may have been open for a while, or forwarded. See `workspace_commands._status_panel` and
`automation_commands._panel`.

```python
async def _pause(press: discord.Interaction):
    if not allowed(press):                                # check again, every press
        await card.reply(press, K.error("You need Manage Server for that."))
        return
    await press.response.edit_message(view=card.views(_panel(row_after))[0])

spec.body += [K.Divider(), K.Buttons((card.button("Pause", discord.ButtonStyle.secondary, _pause, "⏸️"),))]
await card.reply(interaction, spec, timeout=300)          # the buttons stop after 5 minutes
```

**Buttons that must still work after a restart** (Restore on a deleted skill, Leave on a
call, Approve on a plan) are `discord.ui.DynamicItem`s with a `custom_id` that carries what
they act on (`skill:restore:<name>`). Register the class once in `main.on_ready`
(`client.add_dynamic_items(...)`). `skill_tools.RestoreButton` is a short example.

When a card's buttons are done with, never edit it to no view: show `K.finished(spec, "How
it ended")`, or `Card.finished(outcome)` for a question.

### A question or a plan (waiting on a person)

Use `card.Card(text, label=..., note=..., rows=[[buttons...]])`. It defaults to pink `ASK`
and kind `CARD`, so it reads back in history as its text (`brain_question.QuestionView` and
the workflow plan are the examples). Ping only the person it waits on:
`mentions=discord.AllowedMentions(users=[discord.Object(user_id)])`.

### Files and pictures inside a card

```python
spec = K.CardSpec(label="📎 chart.png", body=[K.Text(caption), K.Gallery(("attachment://chart.png",))])
await card.send_card(channel, spec, files={"chart.png": png_bytes})
```

`files` is `{name: bytes}`. Each file is sent with the message that shows it. Name files
with `card.safe_filename(name)`, because Discord rewrites spaces and other characters, which
would break the `attachment://` link.

**A card's files are not in `message.attachments`**: Discord lists them only in the card's
File and Gallery components. To read a message's files, the bot's own included, use
`card.files_of(message)` (attachment-like: `filename`, `size`, `url`, `content_type`,
`read()`; a gallery picture has no `size`). The image edit's `generated-image.png`, the
brain's chat files and the pictures in the history all go through it. Raw API messages (the
message search) have `message_search_format._file_names`.

### Long content

Never cut text to fit. `CardSpec` can hold any length: `card_kit.split` puts it over several
messages, each under Discord's 4000 characters (counted in UTF-16, as Discord counts them, so
an emoji is 2) and 40 components. The label and title go on the first message, the footer on
the last, and text is cut at a paragraph or a line. A code block cut in two is closed and
reopened.

### Text the bot did not write

Wrap anything from a person, a web page or a model in `K.safe(text)`, or `K.quote(text)` for
a quoted request, so a line starting `#`, `-#` or `>` never turns into a heading, small text
or a quote. Mentions in a card **do** ping, unlike in an embed. `send_card` and `LiveCard`
allow none unless you pass `mentions`.

## How the bot reads its own cards back

The container's `id` marks what a card is (`KIND_IDS`), and Discord keeps it:

| `kind` | In the model's history |
|---|---|
| `NOTICE` (default) | `[Status card posted by 'Bot', not a chat message]` plus its text, with command output and status lines left out (`notice_for_history`) |
| `ANNOUNCE` | left out (What's New) |
| `CARD` | its text, as something the bot said (a question, a plan) |

Choose the kind by how the model should see it later. A new kind needs an id in `KIND_IDS`
and a branch in `MessageHandler._format_group`. Workflow helpers run on a `QuietChannel`,
which drops `NOTICE` cards (with no files), so a helper's tools post nothing.

## Testing a card

- Test the pure builder: `K.split(spec)` gives the blocks of each message.
- Test what was sent with `card_checks`:

  ```python
  from card_checks import interaction, label_of, replied_text, sent_views, view_text

  press = interaction(guild_id=1, user=SimpleNamespace(id=5))
  await my_button.callback(press)
  assert "Paused" in view_text(press.response.edit_message.await_args.kwargs["view"])
  ```

- `core/tests/panel_buttons_tests.py` shows buttons tested end to end with fakes.
- To see a card in Discord before wiring it in, post it through a test webhook (never a
  production channel, and never commit the URL). Add `?wait=true&with_components=true` to the
  webhook URL and send `{"flags": 32768, "components": card.views(spec)[0].to_components()}`
  (one POST per message of a long card). 32768 is the Components V2 flag. A plain webhook
  can't send interactive buttons, so swap them for `Link`s in the preview.

## Extending the kit

- **A new block**: add a frozen dataclass to `card_kit.py`, count its components and text in
  `_components` / `_chars` (so `split` keeps messages under the limits), turn it into a
  `discord.ui` item in `card._item`, and test both. `card_tests.test_every_part_of_the_kit_builds`
  builds every block.
- **A new colour or state**: add it to the table at the top of `card_kit.py` and to this
  guide. Colour must keep meaning the state.
- **A new everyday builder**: put it beside `info` / `done` in `card_kit.py` if many places
  need it, or in the feature's own module if only one does.

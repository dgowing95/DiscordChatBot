"""How the brain's question to the requester looks in Discord.

The question (already in the persona's voice, classes/persona_voice.py) is
posted as a reply to the request with buttons: one per choice the brain
offered, "Answer…" (a pop-up text box, for anything else) and "You decide".
Discord's own Reply on the question also answers it (main.on_message).
Answering is deliberate on purpose: an ordinary message stays a new request
(why: classes/brain_inbox.py).

Only the requester can use the buttons; anyone else is told so privately.
When the question closes (answered, no answer in time, or replaced by a
newer one) its buttons go and a short line says how it ended, so a stale
question never looks open.

It is a card (classes/card.py), with the choices on one row and "Answer…" and
"You decide" always on the next: there can be 6 buttons, and a row holds 5.
"""
import asyncio
import logging
from dataclasses import dataclass
import time

import discord

from classes import brain_inbox, card_kit as K
from classes.card import Card, button, reply

logger = logging.getLogger(__name__)

MAX_OPTIONS = 4
OPTION_CHARS = 80  # Discord's button label limit
ANSWER_CHARS = 1000
MAX_QUESTION_CHARS = 1800
HEADING_CHARS = 120  # a longer question reads badly in a heading's large type
# The small line above a question, so it is clear why it was asked.
HEADING_CHAT = "Needs more information"
HEADING_PLAN = "Planning needs more information"
HEADING_STEP = "This step needs more information"
YOU_DECIDE = "You decide"
# What the "You decide" button hands the brain (classes/brain.py maps it).
YOU_DECIDE_ANSWER = "(you decide)"


ANSWERED = "✅ **{answer}**"
TIMED_OUT = "⌛ No answer in time, so I went with my best guess."
CLOSED = "Closed."


def clean_options(options) -> list[str]:
    """The brain's choices as button labels: text only, no repeats, at most
    MAX_OPTIONS, each cut to Discord's limit."""
    labels = []
    for option in options or []:
        label = " ".join(str(option or "").split())[:OPTION_CHARS]
        if label and label not in labels and label != YOU_DECIDE:
            labels.append(label)
    return labels[:MAX_OPTIONS]


def question_text(text: str) -> str:
    """The question as the card shows it: in large type when it is short."""
    text = K.safe(text)
    return f"### {text}" if len(text) <= HEADING_CHARS and "\n" not in text else text


def question_label(label: str, user_id: int) -> str:
    """The small line on top: why it was asked, and who is asked."""
    return f"❓ {label or HEADING_CHAT} · for <@{user_id}>"


class AnswerModal(discord.ui.Modal, title="Your answer"):
    answer = discord.ui.TextInput(label="Answer", style=discord.TextStyle.paragraph,
                                  max_length=ANSWER_CHARS)

    def __init__(self, view: "QuestionView"):
        super().__init__(timeout=view.timeout)
        self.view = view

    async def on_submit(self, interaction: discord.Interaction) -> None:
        await self.view.answer(interaction, str(self.answer.value), str(self.answer.value))


class QuestionView(Card):
    """The question and its buttons. `question_id` is set once it is posted."""

    def __init__(self, user_id: int, options, timeout: float, text: str = "", note: str = "",
                 label: str = ""):
        # The note under the buttons: who can answer, and when it closes (a
        # countdown Discord keeps up to date). `note` adds to it.
        closes = f"closes <t:{int(time.time() + timeout)}:R>" if timeout else ""
        footer = " · ".join(x for x in ("Only the person asked can answer", closes, note) if x)
        super().__init__(question_text(text), label=question_label(label, user_id), note=footer,
                         timeout=timeout)
        self.user_id = int(user_id)
        self.question_id = None
        self.closed = False
        # A button or the box is being handled: close() (run by ask_user the
        # moment the answer arrives) waits for it instead of editing the
        # message at the same time with a different line.
        self.handling = asyncio.Event()
        self.handled = False
        self.add_row([button(label, discord.ButtonStyle.success, self._choose(label))
                      for label in clean_options(options)])
        self.add_row([button("Answer…", discord.ButtonStyle.secondary, self._open_box),
                      button(YOU_DECIDE, discord.ButtonStyle.primary,
                             self._choose(YOU_DECIDE_ANSWER, shown=YOU_DECIDE))])

    def _choose(self, answer: str, shown: str | None = None):
        async def _callback(interaction: discord.Interaction) -> None:
            await self.answer(interaction, answer, shown or answer)
        return _callback

    async def _open_box(self, interaction: discord.Interaction) -> None:
        await interaction.response.send_modal(AnswerModal(self))

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id == self.user_id:
            return True
        await reply(interaction, K.error("Only the person who asked can answer this.",
                                         "🔒 Not your question"))
        return False

    async def answer(self, interaction: discord.Interaction, answer: str, shown: str) -> None:
        self.handled = True
        try:
            await self._answer(interaction, answer, shown)
        finally:
            self.handling.set()

    async def _answer(self, interaction: discord.Interaction, answer: str, shown: str) -> None:
        if self.question_id is None:
            outcome = brain_inbox.NO_WAITER
        else:
            outcome = brain_inbox.deliver(self.question_id, interaction.user.id, answer)
        if outcome == brain_inbox.ACCEPTED:
            self.stop()
            await interaction.response.edit_message(
                view=self.finished(ANSWERED.format(answer=shown[:200])))
            # Only once the edit worked: otherwise close() (in ask_user)
            # still takes the dead buttons off.
            self.closed = True
        elif outcome == brain_inbox.EMPTY:
            await reply(interaction, K.heads_up("✏️ Answer in words", "I need an answer in words, please."))
        else:
            await reply(interaction, K.info("❓ This question is closed"))


async def post(ctx: dict, text: str, options, timeout: float, label: str = HEADING_CHAT):
    """Posts the question (a reply to the request, or a mention when there is
    no source message). Returns (message, view), or (None, None)."""
    message = ctx.get("original_message")
    view = QuestionView(ctx["user_id"], options, timeout, text=str(text)[:MAX_QUESTION_CHARS],
                        note="" if message is not None else f"<@{ctx['user_id']}>", label=label)
    # The model writes the question, so a name in it must not ping anyone else.
    mentions = discord.AllowedMentions(users=[discord.Object(ctx["user_id"])], roles=False,
                                       everyone=False)
    try:
        if message is not None:
            # A request deleted meanwhile still gets its question, just
            # without the reply link (Discord refuses a reply to a missing
            # message by default).
            sent = await ctx["channel"].send(
                view=view, mention_author=True, allowed_mentions=mentions,
                reference=message.to_reference(fail_if_not_exists=False))
        else:
            sent = await ctx["channel"].send(view=view, allowed_mentions=mentions)
    except Exception as e:
        logger.warning(f"Brain: could not post its question: {e}")
        view.stop()
        return None, None
    view.question_id = sent.id
    return sent, view


# How ask() ended.
SKIPPED = "skipped"      # not posted, or no time left to wait
TIMEOUT = "timeout"      # posted, no answer in time
REPLACED = "replaced"    # a newer question took its place
ANSWERED_STATE = "answered"


@dataclass
class Asked:
    state: str
    answer: str = ""
    wait: float = 0.0  # the seconds it waited for (0: it never waited)


async def ask(ctx: dict, question: str, options, wait_left, label: str = HEADING_CHAT, *,
              on_asked=None, before_wait=None) -> Asked:
    """The one way a question reaches the requester: the brain's ask_user and
    a workflow's ask_requester. Reworded in the persona's voice, posted,
    registered (brain_inbox), waited on, and closed with how it ended.
    `wait_left()` is how long it may still wait (None or <= 0: not at all).
    It is read before the post and again after it: the reword and the post
    take time too. `on_asked(message, future)` runs once the answer can
    come, `before_wait(text)` (say, speaking it) before the wait, both inside
    the waiter's lifetime, so a run stopped meanwhile still takes it off."""
    from classes.persona_voice import voice_question

    wait = wait_left()
    if wait is None or wait <= 0:
        return Asked(SKIPPED)
    text = await voice_question(ctx.get("guild_id"), question)
    message, view = await post(ctx, text, options, wait, label)
    if message is None:
        return Asked(SKIPPED)
    wait = wait_left()
    if wait is None or wait <= 0:
        await close(message, view, TIMED_OUT)
        return Asked(TIMEOUT)
    # Registered after the post (the waiter is keyed by the question's
    # message), and before anything else awaits.
    future = brain_inbox.ask(message.id, ctx["channel"].id, ctx["user_id"],
                             voice=bool(ctx.get("voice")))
    logger.info(f"Asked the requester, waiting up to {wait:.0f}s")
    line = CLOSED
    try:
        if on_asked is not None:
            on_asked(message, future)
        if before_wait is not None:
            await before_wait(text)
        answer = await asyncio.wait_for(future, wait)
    except asyncio.TimeoutError:
        line = TIMED_OUT
        return Asked(TIMEOUT, wait=wait)
    finally:
        # Also on a cancel: a waiter left behind would take an answer nobody
        # is waiting for.
        brain_inbox.cancel(message.id, future)
        if future.done() and not future.cancelled() and future.result():
            said = future.result()
            said = YOU_DECIDE if said == YOU_DECIDE_ANSWER else said
            line = ANSWERED.format(answer=said[:200])
        await close(message, view, line)
    if answer is brain_inbox.REPLACED:
        return Asked(REPLACED, wait=wait)
    return Asked(ANSWERED_STATE, answer or "", wait)


async def close(message, view: QuestionView, line: str) -> None:
    """Takes the buttons off a question that ended without a button press
    (a Reply, a voice answer, no answer, replaced). Fail-soft."""
    view.stop()
    if view.handled and not view.closed:
        # An answer is being shown by its own interaction: let it finish.
        try:
            await asyncio.wait_for(view.handling.wait(), 5)
        except asyncio.TimeoutError:
            pass
    if view.closed:
        return
    view.closed = True
    try:
        # Bounded: this runs in the finally of a brain call that may be
        # being stopped.
        await asyncio.wait_for(message.edit(view=view.finished(line)), 10)
    except Exception as e:
        logger.info(f"Brain: could not close its question: {e}")

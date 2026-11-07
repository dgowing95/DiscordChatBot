"""The bot's cards in Discord (classes/card.py), built with the real
discord.py classes, since their rules are what it is for: a question or a
plan with its buttons inside, a card read back from a message, and how the
bot's own cards come back in its history."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import discord
import pytest
from discord.components import _component_factory

from classes import brain_question, card_kit as K, workflow_views
from classes.card import Card, KitView, button, kind_of, text_of, views
from classes.message_handler import MessageHandler


def _button(label="Go"):
    return discord.ui.Button(label=label, style=discord.ButtonStyle.success)


def _shape(view):
    box = view.to_components()[0]
    return box, [c["type"] for c in box["components"]]


def test_the_text_and_buttons_are_inside_one_box_with_the_note_last():
    view = Card("Hello", note="small", label="❓ Asked", rows=[[_button("A"), _button("B")], [_button("C")]])
    box, types = _shape(view)
    assert len(view.to_components()) == 1 and box["type"] == 17
    assert types == [10, 10, 1, 1, 10]  # the label, the text, two rows of buttons, the note
    assert box["components"][0]["content"] == "-# ❓ Asked"
    assert box["components"][-1]["content"] == "-# small"
    assert box["id"] == K.KIND_IDS[K.CARD] and box["accent_color"] == K.ASK
    assert view.has_components_v2()
    assert [b.label for b in view.buttons] == ["A", "B", "C"]


def test_a_row_added_later_goes_above_the_note():
    view = Card("Hello", note="small")
    view.add_row([_button("A")])
    _box, types = _shape(view)
    assert types == [10, 1, 10]


def test_an_empty_row_adds_nothing():
    _box, types = _shape(Card("Hello", rows=[[], ()]))
    assert types == [10]


def test_finished_keeps_its_text_loses_its_buttons_and_note_and_says_how_it_ended():
    view = Card("Hello", note="Press one", label="❓ Asked", rows=[[_button()]])
    finished = view.finished("✅ Done")
    assert finished.text == "Hello" and finished.outcome == "✅ Done" and finished.buttons == []
    box, types = _shape(finished)
    assert types == [10, 10, 10] and box["components"][2]["content"] == "✅ Done"
    assert box["accent_color"] == K.CLOSED
    assert finished.finished().outcome == "✅ Done"


@pytest.mark.asyncio
async def test_a_buttons_callback_is_the_one_given():
    pressed = []

    async def _pressed(interaction):
        pressed.append(interaction)

    item = button("Go", discord.ButtonStyle.primary, _pressed)
    await Card("Hello", rows=[[item]]).buttons[0].callback("press")
    assert pressed == ["press"]


def test_a_card_with_only_persistent_buttons_survives_a_restart():
    view = Card("Plan", rows=[workflow_views.approval_buttons("abc")])
    assert view.is_persistent() and view.timeout is None
    assert len(view.buttons) == 3
    ids = [b["custom_id"] for b in view.to_components()[0]["components"][1]["components"]]
    assert ids == ["workflow:abc:approve", "workflow:abc:edit", "workflow:abc:cancel"]


def test_link_buttons_are_not_buttons_a_person_answers_with():
    view = KitView([K.Text("x"), K.Buttons((K.Link("Open", "https://a.example"),))])
    assert view.buttons == []


def test_every_part_of_the_kit_builds():
    spec = K.CardSpec(label="l", title="t", footer="f", body=[
        K.Text("x"), K.Gap(), K.Divider(),
        K.Section(("a", "b"), K.Link("Open", "https://a.example", "↗️")),
        K.Section(("a",), K.Thumb("https://a.example/p.png")),
        K.Gallery(("attachment://p.png",)), K.FileRef("f.csv"),
        K.Buttons((K.Link("Go", "https://a.example"),))])
    [view] = views(spec)
    types = [c["type"] for c in view.to_components()[0]["components"]]
    assert types == [10, 10, 10, 14, 14, 9, 9, 12, 13, 1, 10]
    assert view.file_names() == {"p.png", "f.csv"}


def _posted(view):
    """A card as Discord gives it back on a message."""
    state = MagicMock()
    return SimpleNamespace(
        content="", components=[_component_factory(c, state) for c in view.to_components()])


def test_text_of_reads_a_card_back_sections_included():
    message = _posted(Card("### Which one?", note="<@5>", rows=[[_button()]]))
    assert text_of(message) == "### Which one?\n-# <@5>"
    [view] = views(K.CardSpec(body=[K.Section(("a", "b"), K.Link("Open", "https://a.example"))]))
    assert text_of(_posted(view)) == "a\nb"


def test_kind_of_says_what_a_card_is():
    assert kind_of(_posted(views(K.info("x"))[0])) == K.NOTICE
    assert kind_of(_posted(views(K.CardSpec(kind=K.ANNOUNCE))[0])) == K.ANNOUNCE
    assert kind_of(_posted(Card("q"))) == K.CARD
    assert kind_of(SimpleNamespace(content="hi", components=[])) is None


@pytest.mark.parametrize("message", [
    SimpleNamespace(content="hi", components=[]), SimpleNamespace(content="hi"), MagicMock()])
def test_text_of_is_empty_for_any_other_message(message):
    assert text_of(message) == ""


def _handler():
    handler = MessageHandler.__new__(MessageHandler)
    handler.client = SimpleNamespace(user=SimpleNamespace(id=42))
    return handler


def _from_bot(view):
    message = _posted(view)
    message.author = SimpleNamespace(id=42, name="bot")
    message.attachments, message.embeds = [], []
    message.poll, message.type = None, discord.MessageType.default
    return message


def test_the_bot_sees_in_the_history_what_it_asked():
    entries = _handler()._format_group(_from_bot(Card("### Which one?", rows=[[_button()]])), [])
    assert "Which one?" in entries[0]["content"] and entries[0]["role"] == "assistant"
    assert entries[0]["content"].startswith("Message from 'bot'")


def test_a_status_card_reads_as_one_not_as_something_the_bot_said():
    [view] = views(K.info("🔎 Searching the web", "**keyboards**"))
    [entry] = _handler()._format_group(_from_bot(view), [])
    assert entry["role"] == "assistant"
    assert entry["content"] == ("[Status card posted by 'bot', not a chat message]\n"
                                "🔎 Searching the web\n**keyboards**")


def test_the_announcement_is_left_out_of_the_history():
    [view] = views(K.CardSpec(label="✨ What's new", body=[K.Text("x")], kind=K.ANNOUNCE))
    assert _handler()._format_group(_from_bot(view), []) == []


def test_someone_elses_card_reads_as_their_message():
    [view] = views(K.info("🔎 Searching the web", "x"))
    message = _from_bot(view)
    message.author = SimpleNamespace(id=7, name="otherbot")
    [entry] = _handler()._format_group(message, [])
    assert entry["content"].startswith("Message from 'otherbot'")


def test_a_question_has_green_choices_then_answer_and_you_decide_on_their_own_row():
    view = brain_question.QuestionView(5, ["A", "B", "C", "D"], 60, text="Which?")
    rows = [c["components"] for c in view.to_components()[0]["components"] if c["type"] == 1]
    assert [len(r) for r in rows] == [4, 2]
    assert {b["style"] for b in rows[0]} == {discord.ButtonStyle.success.value}
    assert [(b["label"], b["style"]) for b in rows[1]] == [
        ("Answer…", discord.ButtonStyle.secondary.value),
        ("You decide", discord.ButtonStyle.primary.value)]


def test_a_question_says_why_and_who_and_when_it_closes_and_keeps_that_when_closed():
    view = brain_question.QuestionView(5, [], 60, text="Which?", label="Planning needs more")
    assert view.label == "❓ Planning needs more · for <@5>"
    assert view.text == "### Which?"
    assert view.note.startswith("Only the person asked can answer · closes <t:")
    closed = view.finished("✅ That")
    assert closed.label == view.label and closed.text == view.text and closed.outcome == "✅ That"


def test_a_short_question_is_a_heading_and_a_long_one_is_not():
    assert brain_question.question_text("Which?") == "### Which?"
    assert brain_question.question_text("x" * 200) == "x" * 200
    assert brain_question.question_text("One\nTwo") == "One\nTwo"
    assert brain_question.question_text("# shout") == "### \\# shout"


@pytest.mark.asyncio
async def test_send_card_answers_an_interaction_then_follows_up_and_keeps_files_with_their_card():
    interaction = MagicMock(spec=discord.Interaction)
    interaction.response.is_done = MagicMock(side_effect=[False, True])
    interaction.response.send_message = AsyncMock()
    interaction.original_response = AsyncMock(return_value="first")
    interaction.followup.send = AsyncMock(return_value="second")
    from classes.card import send_card
    spec = K.CardSpec(label="📘 Skills", body=[K.Text("word " * 900), K.FileRef("a.md")])
    sent = await send_card(interaction, spec, files={"a.md": b"x"}, ephemeral=True)
    assert sent == ["first", "second"]
    first = interaction.response.send_message.await_args.kwargs
    second = interaction.followup.send.await_args.kwargs
    assert first["ephemeral"] and second["ephemeral"] and second["wait"]
    assert "files" not in first and [f.filename for f in second["files"]] == ["a.md"]


def _progress_views():
    from classes.brain_progress import PROGRESS_HINT, SandboxTranscript
    t = SandboxTranscript("chart the messages", files=["a.csv"], started=1000, hint=PROGRESS_HINT)
    for i in range(12):
        block = t.add_command(f"python step_{i}.py")
        t.add_output("OUTPUT_LINE\n" * 60, 0, block=block)
    t.add_note("⚠️ It kept repeating itself.")
    return views(t.render())


def test_a_live_runs_commands_and_output_stay_out_of_the_prompt():
    built = _progress_views()
    assert len(built) > 2
    entries = [e for v in built for e in _handler()._format_group(_from_bot(v), [])]
    text = "\n".join(e["content"] for e in entries)
    assert "OUTPUT_LINE" not in text and "step_3.py" not in text and "Finished" not in text
    assert "chart the messages" in text and "It kept repeating itself." in text
    # a message that only went on with the commands adds nothing at all
    assert len(entries) < len(built)


@pytest.mark.asyncio
async def test_a_card_that_gets_shorter_takes_down_the_messages_it_no_longer_needs():
    from classes.card import LiveCard
    sent = []

    async def send(**kwargs):
        message = MagicMock()
        message.edit, message.delete = AsyncMock(), AsyncMock()
        sent.append(message)
        return message

    channel = MagicMock()
    channel.send = AsyncMock(side_effect=send)
    live = LiveCard(channel)
    long_one = K.CardSpec(label="🧠 Working", body=[K.Text("word " * 900)], footer="💭 Thinking…")
    await live.show(long_one)
    assert len(sent) == 2
    await live.show(K.CardSpec(label="🧠 Done", body=[K.Text("word " * 100)], colour=K.OK))
    sent[1].delete.assert_awaited_once()
    assert len(live.messages) == 1
    # when it cannot be deleted, it is emptied to a grey line instead
    sent[0].delete = AsyncMock(side_effect=RuntimeError("no permission"))
    live.messages.append(sent[0])
    live._shown.append(None)
    await live._drop(1, K.OK)
    assert sent[0].edit.await_args.kwargs["view"].colour == K.CLOSED

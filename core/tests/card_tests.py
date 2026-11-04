"""The card a question or a plan is posted as (classes/card.py), built with the
real discord.py classes, since their rules are what it is for."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import discord
import pytest
from discord.components import _component_factory

from classes import brain_question, tool_embeds, workflow_views
from classes.card import MAX_CARD_CHARS, Card, button, text_of
from classes.message_handler import MessageHandler


def _button(label="Go"):
    return discord.ui.Button(label=label, style=discord.ButtonStyle.success)


def _shape(view):
    box = view.to_components()[0]
    return box, [c["type"] for c in box["components"]]


def test_the_text_and_buttons_are_inside_one_box():
    view = Card("Hello", note="small", rows=[[_button("A"), _button("B")], [_button("C")]])
    box, types = _shape(view)
    assert len(view.to_components()) == 1 and box["type"] == 17
    assert types == [10, 10, 1, 1]  # the text, the note, two rows of buttons
    assert box["components"][1]["content"] == "-# small"
    assert view.has_components_v2()
    assert [b.label for b in view.buttons] == ["A", "B", "C"]


def test_an_empty_row_adds_nothing():
    _box, types = _shape(Card("Hello", rows=[[], ()]))
    assert types == [10]


def test_it_fits_a_message_whatever_it_is_given():
    view = Card("x" * 9000, note="y" * 900)
    assert view.content_length() <= 4000 and len(view.text) + len(view.note) <= MAX_CARD_CHARS


def test_finished_keeps_its_own_text_and_loses_its_buttons():
    view = Card("Hello", note="Press one", rows=[[_button()]])
    finished = view.finished("✅ Done")
    assert finished.text == "Hello" and finished.note == "✅ Done" and finished.buttons == []
    assert view.finished().note == "Press one"
    assert finished.to_components()[0]["accent_color"] == tool_embeds.COLOR_DONE
    assert finished.finished().text == "Hello"


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


def _posted(view):
    """A card as Discord gives it back on a message."""
    state = MagicMock()
    return SimpleNamespace(
        content="", components=[_component_factory(c, state) for c in view.to_components()])


def test_text_of_reads_a_card_back():
    message = _posted(Card("### ❓ Which one?", note="<@5>", rows=[[_button()]]))
    assert text_of(message) == "### ❓ Which one?\n-# <@5>"


@pytest.mark.parametrize("message", [
    SimpleNamespace(content="hi", components=[]), SimpleNamespace(content="hi"), MagicMock()])
def test_text_of_is_empty_for_any_other_message(message):
    assert text_of(message) == ""


def test_the_bot_sees_in_the_history_what_it_asked():
    handler = MessageHandler.__new__(MessageHandler)
    handler.client = SimpleNamespace(user=SimpleNamespace(id=42))
    message = _posted(Card("### ❓ Which one?", rows=[[_button()]]))
    message.author = SimpleNamespace(id=42, name="bot")
    message.attachments, message.embeds = [], []
    message.poll, message.type = None, discord.MessageType.default
    entries = handler._format_group(message, [])
    assert "Which one?" in entries[0]["content"] and entries[0]["role"] == "assistant"


def test_a_question_has_green_choices_then_answer_and_you_decide_on_their_own_row():
    view = brain_question.QuestionView(5, ["A", "B", "C", "D"], 60, text="Which?")
    _box, types = _shape(view)
    rows = [c["components"] for c in view.to_components()[0]["components"] if c["type"] == 1]
    assert [len(r) for r in rows] == [4, 2]
    assert {b["style"] for b in rows[0]} == {discord.ButtonStyle.success.value}
    assert [(b["label"], b["style"]) for b in rows[1]] == [
        ("Answer…", discord.ButtonStyle.secondary.value),
        ("You decide", discord.ButtonStyle.primary.value)]


def test_a_label_sits_small_above_the_question_and_stays_when_it_closes():
    view = brain_question.QuestionView(5, [], 60, text="Which?", label="Planning needs more")
    assert view.text == "-# Planning needs more\n### ❓ Which?"
    assert view.finished("✅ That").text == view.text


def test_a_short_question_is_a_heading_and_a_long_one_is_not():
    assert brain_question.question_text("Which?") == "### ❓ Which?"
    assert brain_question.question_text("x" * 200).startswith("❓ x")
    assert brain_question.question_text("One\nTwo") == "❓ One\nTwo"

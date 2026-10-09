"""Tests for core/classes/poll_format.py (pure): the create_poll tool's input
checks and how polls read in a prompt.

To run this pytest file from the command line, use:
    pytest core/tests/poll_format_tests.py
"""
from datetime import datetime, timedelta, timezone

from classes import poll_format
from classes.poll_format import format_poll, format_poll_result, validate

END = datetime(2026, 10, 10, 14, 0, 37, tzinfo=timezone.utc)


def test_validate_cleans_answers():
    question, answers, hours, error = validate("  Lunch?  ", [" Pizza", "pizza", "", "Sushi "], 24)
    assert error == ""
    assert question == "Lunch?"
    assert answers == ["Pizza", "Sushi"]
    assert hours == 24


def test_validate_clamps_hours():
    assert validate("Q", ["a", "b"], 0)[2] == 1
    assert validate("Q", ["a", "b"], 10_000)[2] == poll_format.MAX_HOURS
    assert validate("Q", ["a", "b"], "soon")[2] == 24


def test_validate_needs_two_to_ten_options():
    assert "2 to 10" in validate("Q", ["only"], 24)[3]
    assert "2 to 10" in validate("Q", ["a", "a"], 24)[3]  # duplicates do not count
    assert "2 to 10" in validate("Q", [str(i) for i in range(11)], 24)[3]
    assert validate("Q", [str(i) for i in range(10)], 24)[3] == ""


def test_validate_reports_too_long_text_instead_of_cutting_it():
    long_answer = "x" * (poll_format.MAX_ANSWER + 1)
    _, answers, _, error = validate("Q", ["ok", long_answer], 24)
    assert long_answer in answers  # not truncated
    assert f'"{long_answer}"' in error
    assert "call create_poll again" in error
    assert "question" in validate("q" * (poll_format.MAX_QUESTION + 1), ["a", "b"], 24)[3]
    assert "empty" in validate("  ", ["a", "b"], 24)[3]


def test_open_poll_with_voters():
    text = format_poll("bot", "Where should we eat?",
                       [("Pizza", 2, ["alice", "bob"]), ("Sushi", 1, ["carol"]), ("Tacos", 0, None)],
                       3, False, False, END)
    assert text == (
        "Message from 'bot': Poll \"Where should we eat?\" (one choice each, open until "
        "2026-10-10 14:00 UTC, 3 votes)\n"
        "- Pizza: 2 (alice, bob)\n"
        "- Sushi: 1 (carol)\n"
        "- Tacos: 0")


def test_end_time_is_absolute_so_the_text_is_stable():
    # The same poll formats identically whatever the time is now, and a
    # timezone-shifted expiry reads the same in UTC.
    shifted = END.astimezone(timezone(timedelta(hours=5)))
    assert format_poll("a", "Q", [], 0, False, False, END) == \
        format_poll("a", "Q", [], 0, False, False, shifted)


def test_ended_multiple_choice_poll_without_voter_names():
    text = format_poll("alice", "Games?", [("Chess", 1, None), ("Go", 4, None)],
                       5, True, True, END)
    assert "(several choices each, ended, 5 votes)" in text
    assert "- Chess: 1\n- Go: 4" in text
    assert "until" not in text


def test_voters_past_the_cap_become_more():
    names = [f"u{i}" for i in range(poll_format.VOTER_NAMES_PER_ANSWER)]
    text = format_poll("a", "Q", [("Yes", 40, names)], 40, False, False, None)
    assert text.endswith(f"u{poll_format.VOTER_NAMES_PER_ANSWER - 1}, +15 more)")
    assert "(one choice each, open, 40 votes)" in text


def test_poll_result_with_a_winner():
    fields = {"poll_question_text": "Lunch?", "victor_answer_text": "Pizza",
              "victor_answer_votes": "3", "total_votes": "5"}
    assert format_poll_result("bot", fields) == \
        "Message from 'bot': Poll ended: \"Lunch?\" - winner: Pizza with 3 of 5 votes"


def test_poll_result_tie_and_no_votes():
    assert "no single winner (4 votes)" in format_poll_result(
        "bot", {"poll_question_text": "Lunch?", "total_votes": "4"})
    assert "no votes" in format_poll_result(
        "bot", {"poll_question_text": "Lunch?", "total_votes": "0"})


def test_a_copied_poll_label_is_stripped_from_replies():
    """The model copies a history entry's label into its reply now and then.
    Polls use the one label response_filter strips; a "Poll from" label of
    their own once leaked straight into a reply."""
    from classes.response_filter import filter_response
    label = format_poll("Bubba", "Q", [], 0, False, False, END).split(" Poll ")[0]
    assert filter_response(f"{label} Poll's up, go vote!") == "Poll's up, go vote!"

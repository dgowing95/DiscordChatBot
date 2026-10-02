from datetime import datetime, timezone

import pytest

from datetime import timedelta

from classes.automation_policy import (
    anchor_timing, describe, form_values, matches, next_run, parse_clock, resolve_timezone, timing_from_form,
    validate_schedule,
)

UTC = timezone.utc


def at(value):
    return datetime.fromisoformat(value).replace(tzinfo=UTC)


def test_london_clock_skips_gap_and_uses_first_fold():
    timing = {"type": "daily", "time": "01:30"}
    assert next_run(timing, "Europe/London", at("2026-03-29T00:00:00")) == at("2026-03-30T00:30:00")
    assert next_run(timing, "Europe/London", at("2026-10-25T00:00:00")) == at("2026-10-25T00:30:00")
    assert next_run(timing, "Europe/London", at("2026-10-25T00:30:00")) == at("2026-10-26T01:30:00")


def test_interval_uses_elapsed_days_and_minimum():
    timing = {"type": "interval", "every": 1, "unit": "days", "start": "2026-03-28T09:00:00+00:00"}
    assert next_run(timing, "Europe/London", at("2026-03-29T08:00:00")) == at("2026-03-29T09:00:00")
    with pytest.raises(ValueError, match="at least 4 hours apart"):
        validate_schedule({"type": "interval", "every": 3, "unit": "hours"}, "Europe/London", 4, at("2026-01-01T00:00:00"))


def test_one_off_can_be_soon_and_local():
    assert validate_schedule({"type": "once", "at": "2026-01-01T01:00:00"}, "Europe/London", 4,
                             at("2026-01-01T00:00:00")) == at("2026-01-01T01:00:00")


def test_word_phrase_and_unicode_boundaries():
    assert matches("STRASSE", "Straße is here")
    assert matches("New   York", "welcome to new york!")
    assert not matches("cat", "concatenate")
    assert matches("cat", "concatenate", "substring")
    assert not matches("cat", "cat_2")


def test_interval_without_start_does_not_drift_after_a_run():
    created = at("2026-01-01T09:00:00")
    timing = anchor_timing({"type": "interval", "every": 1, "unit": "days"}, created)
    assert next_run(timing, "Europe/London", created) == at("2026-01-02T09:00:00")
    # The runner asks for the first slot at least the minimum gap after a
    # run finishes; an unanchored interval answered "minimum + one interval".
    finished = at("2026-01-02T09:00:30")
    after = finished + timedelta(hours=4) - timedelta(microseconds=1)
    assert next_run(timing, "Europe/London", after) == at("2026-01-03T09:00:00")
    assert anchor_timing(timing, at("2026-06-01T00:00:00")) == timing


def test_describe_spells_out_next_run_in_local_time():
    row = {"kind": "schedule", "status": "enabled", "timezone": "Europe/London",
           "timing": {"type": "daily", "time": "09:00"},
           "next_run": at("2026-07-01T08:00:00").timestamp()}
    described = describe(row)
    assert described["timing_summary"] == "daily at 09:00 (Europe/London)"
    assert described["next_run_local"] == "2026-07-01T09:00+01:00 (Europe/London)"


def test_unknown_timezone_is_a_clear_error():
    with pytest.raises(ValueError, match="Unknown timezone"):
        validate_schedule({"type": "daily", "time": "09:00"}, "Mars/Base", 4, at("2026-01-01T00:00:00"))


@pytest.mark.parametrize("text, clock", [("9", "09:00"), ("9:30", "09:30"), ("17:05", "17:05"),
                                         ("9am", "09:00"), ("12am", "00:00"), ("12:30 pm", "12:30"), ("9:15PM", "21:15")])
def test_parse_clock_accepts_common_forms(text, clock):
    assert parse_clock(text) == clock


@pytest.mark.parametrize("text", ["", "noon", "25:00", "9:75", "13pm"])
def test_parse_clock_explains_bad_input(text):
    with pytest.raises(ValueError, match="time"):
        parse_clock(text)


def test_form_fields_become_timing_for_each_type():
    assert timing_from_form("once", {"when": "2026-10-03 09:00"}) == {"type": "once", "at": "2026-10-03T09:00:00"}
    assert timing_from_form("interval", {"every": "6", "unit": "hours", "start": ""}) == {
        "type": "interval", "every": 6, "unit": "hours"}
    assert timing_from_form("daily", {"time": "9am"}) == {"type": "daily", "time": "09:00"}
    assert timing_from_form("weekly", {"weekday": "friday", "time": "17:30"}) == {
        "type": "weekly", "weekday": "friday", "time": "17:30"}
    with pytest.raises(ValueError, match="whole number"):
        timing_from_form("interval", {"every": "six", "unit": "hours"})
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        timing_from_form("once", {"when": "tomorrow"})
    with pytest.raises(ValueError, match="day of the week"):
        timing_from_form("weekly", {"weekday": "", "time": "09:00"})


def test_form_values_prefill_same_type_and_keep_action_when_switching():
    row = {"timezone": "Europe/London", "action": "go",
           "timing": {"type": "interval", "every": 6, "unit": "hours", "start": "2026-07-01T08:00:00+00:00"}}
    assert form_values(row, "interval") == {"timezone": "Europe/London", "action": "go", "every": "6",
                                            "unit": "hours", "start": "2026-07-01 09:00"}
    assert form_values(row, "daily") == {"timezone": "Europe/London", "action": "go"}
    # A prefilled form submitted unchanged must keep the same timing.
    again = timing_from_form("interval", form_values(row, "interval"))
    assert next_run(again, "Europe/London", at("2026-07-02T00:00:00")) == next_run(
        row["timing"], "Europe/London", at("2026-07-02T00:00:00"))


@pytest.mark.parametrize("typed, canonical", [
    ("Europe/London", "Europe/London"), ("europe/london", "Europe/London"), ("  Europe/London ", "Europe/London"),
    ("america/new york", "America/New_York"), ("London", "Europe/London"), ("tokyo", "Asia/Tokyo"), ("utc", "UTC")])
def test_resolve_timezone_forgives_case_spaces_and_bare_cities(typed, canonical):
    assert resolve_timezone(typed) == canonical


@pytest.mark.parametrize("typed, suggestion", [("Europe/Londn", "Europe/London"), ("Londn", "Europe/London")])
def test_resolve_timezone_suggests_close_names(typed, suggestion):
    with pytest.raises(ValueError, match=f"Did you mean .*{suggestion}"):
        resolve_timezone(typed)


@pytest.mark.parametrize("typed", ["Mars/Base", "../../etc/passwd", ""])
def test_resolve_timezone_rejects_unknown_names(typed):
    with pytest.raises(ValueError, match="Unknown timezone"):
        resolve_timezone(typed)

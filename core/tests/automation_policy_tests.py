from datetime import datetime, timezone

import pytest

from datetime import timedelta

from classes.automation_policy import anchor_timing, describe, matches, next_run, validate_schedule

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
    with pytest.raises(ValueError, match="at least 4 hours"):
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

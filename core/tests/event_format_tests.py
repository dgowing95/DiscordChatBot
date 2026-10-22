"""Tests for core/classes/event_format.py (pure: no Discord, no network).

To run this pytest file from the command line, use:
    pytest core/tests/event_format_tests.py
"""
from datetime import datetime, timedelta, timezone

import pytest

from classes.event_format import (
    DEFAULT_LENGTH, MAX_NAME, discord_timestamp, events_enabled, format_event, validate,
)

UTC = timezone.utc
NOW = datetime(2026, 10, 22, 12, 0, tzinfo=UTC)
TZ = "Europe/London"


def _new(**fields):
    base = dict(name="Game night", start="2026-10-24T19:00", tz_name=TZ, now=NOW, needs_end=False)
    base.update(fields)
    return validate(**base)


def test_naive_start_is_read_in_the_server_timezone():
    fields, error = _new()
    assert not error
    # 24 Oct 2026 is still British Summer Time (UTC+1).
    assert fields["start"] == datetime(2026, 10, 24, 18, 0, tzinfo=UTC)
    assert "end" not in fields


def test_offset_is_respected():
    fields, _ = _new(start="2026-10-24T19:00-04:00")
    assert fields["start"] == datetime(2026, 10, 24, 23, 0, tzinfo=UTC)


def test_an_event_with_a_location_gets_a_default_end():
    fields, error = _new(location="The pub", needs_end=True)
    assert not error
    assert fields["end"] - fields["start"] == DEFAULT_LENGTH
    assert fields["default_end"] is True
    assert fields["location"] == "The pub"


def test_past_start_far_future_and_bad_end_are_errors():
    for kwargs in (dict(start="2026-10-21T19:00"), dict(start="2028-10-24T19:00"),
                   dict(end="2026-10-24T18:00"), dict(start="next friday")):
        fields, error = _new(**kwargs)
        assert fields == {} and error.startswith("No event was made:") and "call again" in error


def test_a_skipped_wall_time_is_an_error():
    # The clocks go forward at 01:00 on 29 March 2026 in London.
    _, error = _new(start="2026-03-29T01:30", now=datetime(2026, 3, 1, tzinfo=UTC))
    assert "does not exist" in error


def test_name_and_start_are_required_for_a_new_event_and_limits_hold():
    assert "needs a name" in _new(name="  ")[1]
    assert "needs a start" in _new(start="")[1]
    assert "limit is 100" in _new(name="x" * (MAX_NAME + 1))[1]
    assert "limit is 1000" in _new(description="x" * 1001)[1]
    fields, _ = _new(name="  Game \n night  ")
    assert fields["name"] == "Game night"


def test_moving_the_start_keeps_the_length():
    current = {"start": datetime(2026, 10, 24, 18, 0, tzinfo=UTC),
               "end": datetime(2026, 10, 24, 21, 0, tzinfo=UTC)}
    fields, error = validate(start="2026-10-24T21:00", tz_name=TZ, now=NOW, needs_end=True,
                             current=current)
    assert not error
    assert fields["start"] == datetime(2026, 10, 24, 20, 0, tzinfo=UTC)
    assert fields["end"] == datetime(2026, 10, 24, 23, 0, tzinfo=UTC)
    assert "name" not in fields


def test_a_change_with_nothing_returns_no_fields():
    current = {"start": datetime(2026, 10, 24, 18, 0, tzinfo=UTC), "end": None}
    assert validate(tz_name=TZ, now=NOW, needs_end=False, current=current) == ({}, "")


def test_format_event_uses_absolute_times_and_names():
    start = datetime(2026, 10, 24, 18, 0, tzinfo=UTC)
    text = format_event({"id": 5, "name": "Game night", "status": "scheduled", "start": start,
                         "end": start + timedelta(hours=2), "where": "the Games voice channel",
                         "interested": 2, "names": ["alice", "bob"], "by_bot": True,
                         "description": ""})
    assert 'Event 5: "Game night" (scheduled)' in text
    assert "2026-10-24 18:00 UTC" in text and discord_timestamp(start) in text
    assert "Interested: 2 (alice, bob)" in text
    assert "update_event can change" in text


def test_events_switch(monkeypatch):
    monkeypatch.delenv("EVENTS_ENABLED", raising=False)
    assert events_enabled()
    monkeypatch.setenv("EVENTS_ENABLED", "0")
    assert not events_enabled()


def test_garbled_spots_tool_call_markup_only():
    from classes.event_format import garbled
    assert garbled("Game night", "2026-10-24T21:00", "") == ""
    assert garbled('Say "hi" at the door', "Bring snacks <3") == ""
    assert garbled("</parameter>> 5pm")
    assert garbled('20"start">2026-10-23')
    assert garbled("<parameter=start>2026")


def test_an_offset_no_timezone_uses_is_refused():
    """Seen live under a frequency penalty: "+08:35", "+01:39"."""
    from classes.event_format import parse_time
    for text in ("2026-10-23T21:00+08:35", "2026-10-23T21:00+01:39", "2026-10-23T21:00+15:00"):
        with pytest.raises(ValueError, match="offset no timezone uses"):
            parse_time(text, "Europe/London", "start")
    assert parse_time("2026-10-23T21:00+05:45", "Europe/London", "start").hour == 15
    assert parse_time("2026-10-23T21:00Z", "Europe/London", "start").hour == 21

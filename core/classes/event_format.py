"""Discord scheduled events: input checks and how an event reads in a
prompt (PURE, stdlib only).

The same split as poll_format: event_tools makes the Discord calls, this
decides what is valid and what the model is told. Nothing here raises to
the caller; a problem comes back as text telling the model what to fix.
"""
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import os
import re

UTC = timezone.utc

# Discord's limits for a scheduled event.
MAX_NAME = 100
MAX_DESCRIPTION = 1000
MAX_LOCATION = 100

# How far ahead an event may start. Discord allows more; a year is already
# past anything a chat asks for, and catches a mistyped year.
MAX_AHEAD = timedelta(days=366)

# An event outside a voice channel must have an end; this is it when the
# request gives none.
DEFAULT_LENGTH = timedelta(hours=2)


def events_enabled() -> bool:
    """EVENTS_ENABLED (default on); off removes the event tools."""
    return os.getenv("EVENTS_ENABLED", "1").lower() not in ("0", "false", "off")


def discord_timestamp(value: datetime, style: str = "F") -> str:
    """<t:unix:F>, which Discord shows in each reader's own timezone."""
    return f"<t:{int(value.timestamp())}:{style}>"


def utc_text(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")


# Tool-call markup inside an argument. Seen live: the model's call came out
# with one field's text running into the next ("...20\"start\">2026-10-23T..."
# and "</parameter>" in a description), which wrote junk into an event and
# filled voice_channel with half a reply.
_GARBLED = re.compile(r"</?parameter|</?function|</?tool_call|\"\w+\">")


def garbled(*values: str) -> str:
    """"" when the arguments look clean, else what to tell the model."""
    if any(_GARBLED.search(value or "") for value in values):
        return ("Nothing was done: the call's arguments ran into each other (one field holds "
                "tool-call text such as </parameter>). Call again with each value on its own, "
                "plainly, e.g. start=\"2026-10-24T21:00\".")
    return ""


def parse_time(text: str, tz_name: str, field: str) -> datetime:
    """ISO text -> an aware UTC datetime. A time without an offset is read
    in `tz_name`, the timezone the bot's "Current datetime" is given in.
    A wall time the clocks skip is an error."""
    try:
        value = datetime.fromisoformat(text.strip())
    except ValueError:
        raise ValueError(f"Could not read the {field} time {text!r}; use an ISO date and time "
                         f"like 2026-10-24T19:00 (read as {tz_name} unless it has an offset).")
    offset = value.utcoffset()
    # Real offsets are whole quarter hours within +-14h. Seen live: a model
    # under a frequency penalty wrote "+08:35" and "+01:39", which parse
    # and would quietly move the event.
    if offset is not None and (offset.total_seconds() % 900 or abs(offset) > timedelta(hours=14)):
        raise ValueError(f"The {field} time {text!r} has an offset no timezone uses. Leave the "
                         f"offset out (it is then read as {tz_name}) and call again.")
    if value.tzinfo is None:
        zone = ZoneInfo(tz_name)
        local = value.replace(tzinfo=zone, fold=0)
        if local.astimezone(UTC).astimezone(zone).replace(tzinfo=None) != value:
            raise ValueError(f"The {field} time {text!r} does not exist in {tz_name} "
                             "(the clocks skip it); pick another time.")
        value = local
    return value.astimezone(UTC)


def _text(value: str, limit: int, label: str, single_line: bool = True) -> str:
    value = " ".join(value.split()) if single_line else value.strip()
    if len(value) > limit:
        raise ValueError(f"The {label} is {len(value)} characters; the limit is {limit}. Shorten it.")
    return value


def validate(*, name: str = "", start: str = "", end: str = "", description: str = "",
             location: str = "", tz_name: str, now: datetime, needs_end: bool,
             current: dict | None = None) -> tuple[dict, str]:
    """Check an event's fields -> (fields, error).

    For a new event (`current` None) a name and start are required. For a
    change, `current` holds the event's "start" and "end", empty arguments
    mean "keep", and only changed values are returned; moving the start
    without a new end keeps the event's length, so "move it to 9pm" does
    not leave the end before the start.

    `fields` may hold name, description, location, start and end (aware
    UTC), plus "default_end" when the end was filled in because an event
    outside a voice channel needs one."""
    try:
        fields = {}
        if name.strip():
            fields["name"] = _text(name, MAX_NAME, "name")
        elif current is None:
            raise ValueError("An event needs a name.")
        if description.strip():
            fields["description"] = _text(description, MAX_DESCRIPTION, "description", single_line=False)
        if location.strip():
            fields["location"] = _text(location, MAX_LOCATION, "location")

        start_at = parse_time(start, tz_name, "start") if start.strip() else None
        end_at = parse_time(end, tz_name, "end") if end.strip() else None
        if start_at is None and current is None:
            raise ValueError("An event needs a start time.")
        if start_at is not None:
            if start_at <= now:
                raise ValueError(f"The start time {utc_text(start_at)} is in the past; "
                                 f"it is now {utc_text(now)}.")
            if start_at - now > MAX_AHEAD:
                raise ValueError("The start time is more than a year away; check the year.")
            fields["start"] = start_at
        old_start = (current or {}).get("start")
        old_end = (current or {}).get("end")
        if end_at is None and start_at is not None and old_start and old_end:
            end_at = start_at + (old_end - old_start)
        effective_start = start_at or old_start
        effective_end = end_at or old_end
        if effective_end is None and needs_end:
            end_at = effective_end = effective_start + DEFAULT_LENGTH
            fields["default_end"] = True
        if effective_end is not None and effective_end <= effective_start:
            raise ValueError(f"The end time {utc_text(effective_end)} must be after the start "
                             f"{utc_text(effective_start)}.")
        if end_at is not None:
            fields["end"] = end_at
        return fields, ""
    except ValueError as e:
        return {}, f"Nothing was changed: {e} Fix it and call again."


def format_event(event: dict) -> str:
    """One event for list_events. Times are absolute (UTC and a Discord
    timestamp), like poll_format, so the text does not change between
    builds; the timestamp is what to show people.

    `event`: id, name, status, start, end (or None), where, interested
    (count), names (list or None when the lookup failed), by_bot,
    description."""
    lines = [f"Event {event['id']}: \"{event['name']}\" ({event['status']})"]
    when = f"Starts {utc_text(event['start'])}, show it as {discord_timestamp(event['start'])}"
    if event.get("end"):
        when += f"; ends {utc_text(event['end'])}"
    lines.append(when)
    lines.append(f"Where: {event['where']}")
    if event.get("description"):
        lines.append(f"Details: {event['description'][:200]}")
    interested = f"Interested: {event['interested']}"
    if event.get("names"):
        interested += " (" + ", ".join(event["names"]) + ")"
    lines.append(interested)
    lines.append("Made by me, so update_event can change or cancel it." if event.get("by_bot")
                 else "Made by someone else; I cannot change it.")
    return "\n".join(lines)

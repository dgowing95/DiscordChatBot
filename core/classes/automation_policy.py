"""Pure validation, recurrence and message matching for server automations."""
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import os
import re

UTC = timezone.utc
WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


def settings():
    def number(name, default, minimum=0):
        value = int(os.getenv(name, str(default)))
        if value < minimum:
            raise ValueError(f"{name} must be at least {minimum}")
        return value
    tz = os.getenv("AUTOMATIONS_TIMEZONE", "Europe/London")
    ZoneInfo(tz)
    return {
        "enabled": os.getenv("AUTOMATIONS_ENABLED", "1").lower() not in ("0", "false", "off"),
        "timezone": tz,
        "min_hours": number("SCHEDULE_MIN_INTERVAL_HOURS", 4, 1),
        "schedule_limit": number("SCHEDULE_MAX_PER_GUILD", 1),
        "rule_limit": number("RULE_MAX_PER_GUILD", 4),
        "cooldown": number("RULE_COOLDOWN_SECONDS", 60),
    }


def _local_occurrence(day, clock, zone):
    hour, minute = map(int, clock.split(":"))
    if hour > 23 or minute > 59:
        raise ValueError("Time must be HH:MM")
    local = datetime(day.year, day.month, day.day, hour, minute, tzinfo=zone, fold=0)
    utc = local.astimezone(UTC)
    # A spring-forward wall time round-trips to a different clock time.
    if utc.astimezone(zone).replace(tzinfo=None) != local.replace(tzinfo=None):
        return None
    return utc


def next_run(timing, tz_name, after):
    """First occurrence strictly after `after` (aware UTC datetime)."""
    zone = ZoneInfo(tz_name)
    kind = timing["type"]
    if kind == "once":
        value = datetime.fromisoformat(timing["at"])
        if value.tzinfo is None:
            value = value.replace(tzinfo=zone, fold=0)
            if value.astimezone(UTC).astimezone(zone).replace(tzinfo=None) != value.replace(tzinfo=None):
                raise ValueError("One-off time does not exist in this timezone")
        value = value.astimezone(UTC)
        return value if value > after else None
    if kind == "interval":
        hours = int(timing["every"]) * {"hours": 1, "days": 24, "weeks": 168}[timing["unit"]]
        if hours <= 0:
            raise ValueError("Interval must be positive")
        start = datetime.fromisoformat(timing.get("start") or after.isoformat())
        if start.tzinfo is None:
            start = start.replace(tzinfo=zone, fold=0)
        start = start.astimezone(UTC)
        if start > after:
            return start
        step = timedelta(hours=hours)
        return start + step * ((after - start) // step + 1)
    if kind not in ("daily", "weekly"):
        raise ValueError("Unknown schedule type")
    if kind == "weekly" and timing.get("weekday") not in WEEKDAYS:
        raise ValueError("Invalid weekday")
    today = after.astimezone(zone).date()
    for offset in range(9):
        day = today + timedelta(days=offset)
        if kind == "weekly" and WEEKDAYS[day.weekday()] != timing["weekday"]:
            continue
        value = _local_occurrence(day, timing["time"], zone)
        if value is not None and value > after:
            return value
    raise ValueError("No next occurrence")


def anchor_timing(timing, now):
    """Pin an interval's grid to `now` when no start was given. Without a
    stored anchor every recalculation re-anchors at its own `after`, so each
    run lands a full interval past the minimum gap (a daily interval ran every
    28 hours) and every edit restarted the clock."""
    if timing.get("type") == "interval" and not timing.get("start"):
        return {**timing, "start": now.astimezone(UTC).isoformat()}
    return timing


def validate_schedule(timing, tz_name, min_hours, now):
    try:
        ZoneInfo(tz_name)
    except (ZoneInfoNotFoundError, ValueError):
        raise ValueError(f"Unknown timezone {tz_name!r}; use an IANA name like Europe/London")
    if timing["type"] == "once":
        value = next_run(timing, tz_name, now)
        if value is None:
            raise ValueError("One-off time must be in the future")
    else:
        value = next_run(timing, tz_name, now)
        later = next_run(timing, tz_name, value)
        if later - value < timedelta(hours=min_hours):
            raise ValueError(f"Recurring executions must be at least {min_hours} hours apart")
        if timing["type"] in ("daily", "weekly") and min_hours > (24 if timing["type"] == "daily" else 168):
            raise ValueError("Minimum interval exceeds this recurrence")
    return value


def matches(pattern, text, mode="word"):
    needle = " ".join(pattern.casefold().split())
    haystack = " ".join(text.casefold().split())
    if not needle:
        raise ValueError("Pattern cannot be empty")
    if mode == "substring":
        return needle in haystack
    if mode != "word":
        raise ValueError("Mode must be word or substring")
    return re.search(r"(?<!\w)" + re.escape(needle) + r"(?!\w)", haystack, re.UNICODE) is not None


def timing_text(timing, tz_name):
    kind = timing.get("type")
    if kind == "once":
        text = f"once at {timing.get('at')}"
    elif kind == "interval":
        text = f"every {timing.get('every')} {timing.get('unit')}"
    elif kind == "daily":
        text = f"daily at {timing.get('time')}"
    elif kind == "weekly":
        text = f"every {str(timing.get('weekday', '')).capitalize()} at {timing.get('time')}"
    else:
        text = str(kind)
    return f"{text} ({tz_name})"


def describe(row):
    """A copy of a record with its timing and next run spelled out, so a
    person or the model can report them without converting epoch seconds."""
    out = dict(row)
    if row.get("kind") == "schedule":
        out["timing_summary"] = timing_text(row["timing"], row["timezone"])
        if row.get("status") == "enabled" and row.get("next_run"):
            local = datetime.fromtimestamp(row["next_run"], UTC).astimezone(ZoneInfo(row["timezone"]))
            out["next_run_local"] = f"{local.isoformat(timespec='minutes')} ({row['timezone']})"
    return out

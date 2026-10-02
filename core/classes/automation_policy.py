"""Pure validation, recurrence and message matching for server automations."""
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, available_timezones
import difflib
import functools
import os
import re

UTC = timezone.utc
WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


@functools.lru_cache(maxsize=1)
def _zone_names():
    """lower-cased name -> canonical name, and lower-cased city -> names."""
    by_name, by_city = {}, {}
    for name in available_timezones():
        by_name[name.lower()] = name
        if "/" in name:
            by_city.setdefault(name.rsplit("/", 1)[1].lower(), []).append(name)
    return by_name, by_city


def resolve_timezone(text):
    """A typed timezone -> its canonical IANA name. Capitals, surrounding
    spaces and spaces for underscores are forgiven ('europe/london',
    'america/new york'), and a bare city resolves when exactly one zone has
    it ('London'). Anything else is an error that suggests close names."""
    cleaned = "_".join(str(text or "").split()).lower()
    by_name, by_city = _zone_names()
    if cleaned in by_name:
        return by_name[cleaned]
    if len(by_city.get(cleaned, ())) == 1:
        return by_city[cleaned][0]
    suggestions = [by_name[name] for name in difflib.get_close_matches(cleaned, list(by_name), n=3, cutoff=0.6)]
    for city in difflib.get_close_matches(cleaned, list(by_city), n=3, cutoff=0.75):
        suggestions.extend(name for name in by_city[city] if name not in suggestions)
    hint = f" Did you mean {', '.join(suggestions[:3])}?" if suggestions else ""
    raise ValueError(f"Unknown timezone {str(text).strip()!r}; use an IANA name like Europe/London or America/New_York.{hint}")


def settings():
    def number(name, default, minimum=0):
        value = int(os.getenv(name, str(default)))
        if value < minimum:
            raise ValueError(f"{name} must be at least {minimum}")
        return value
    tz = resolve_timezone(os.getenv("AUTOMATIONS_TIMEZONE", "Europe/London"))
    return {
        "enabled": os.getenv("AUTOMATIONS_ENABLED", "1").lower() not in ("0", "false", "off"),
        "timezone": tz,
        "min_hours": number("SCHEDULE_MIN_INTERVAL_HOURS", 4, 1),
        "schedule_limit": number("SCHEDULE_MAX_PER_GUILD", 1),
        "rule_limit": number("RULE_MAX_PER_GUILD", 4),
        "cooldown": number("RULE_COOLDOWN_SECONDS", 60),
    }


def parse_clock(text):
    """'9', '9:30', '09:30', '9am', '9:30pm' -> 'HH:MM' (24-hour)."""
    match = re.fullmatch(r"\s*(\d{1,2})(?::(\d{2}))?\s*([ap]\.?m\.?)?\s*", text.lower())
    if not match:
        raise ValueError(f"Could not read the time {text!r}; use 24-hour HH:MM like 09:00 or 17:30")
    hour, minute, half = int(match[1]), int(match[2] or 0), match[3]
    if half:
        if not 1 <= hour <= 12:
            raise ValueError(f"Could not read the time {text!r}; with am/pm the hour must be 1-12")
        hour = hour % 12 + (12 if half.startswith("p") else 0)
    if hour > 23 or minute > 59:
        raise ValueError(f"Could not read the time {text!r}; use 24-hour HH:MM like 09:00 or 17:30")
    return f"{hour:02d}:{minute:02d}"


def parse_when(text):
    """'2026-10-03 09:00' (or ISO with T / an offset) -> ISO string; naive
    values are read in the schedule's timezone later."""
    try:
        return datetime.fromisoformat(text.strip()).isoformat()
    except ValueError:
        raise ValueError(f"Could not read the date and time {text!r}; use YYYY-MM-DD HH:MM like 2026-10-03 09:00")


def _local_occurrence(day, clock, zone):
    hour, minute = map(int, parse_clock(clock).split(":"))
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


# How late a run may start and still count as on time when the next run is
# chosen. Without it, an interval equal to the minimum gap (the shortest the
# form allows) skipped every other slot: a run due at T starts a few seconds
# late, T + interval is then a few seconds short of the gap, and the next
# slot goes. Queue waits and the 15 second poll fit well inside it.
ON_TIME_GRACE = timedelta(minutes=15)


def following_run(timing, tz_name, occurrence, started, now, min_hours):
    """The run after one that was due at `occurrence` and started at
    `started`. It is the next slot after this occurrence (so missed slots
    are never replayed one by one) that is also at least the minimum gap
    after the run actually started, less ON_TIME_GRACE (so a late catch-up
    run is not followed straight away by the next slot), and never in the
    past."""
    earliest = started + timedelta(hours=min_hours) - ON_TIME_GRACE
    after = max(occurrence, earliest - timedelta(microseconds=1), now)
    return next_run(timing, tz_name, after)


def validate_schedule(timing, tz_name, min_hours, now):
    tz_name = resolve_timezone(tz_name)
    if timing["type"] == "once":
        value = next_run(timing, tz_name, now)
        if value is None:
            raise ValueError("One-off time must be in the future")
    else:
        value = next_run(timing, tz_name, now)
        later = next_run(timing, tz_name, value)
        if later - value < timedelta(hours=min_hours):
            raise ValueError(f"Repeating schedules must be at least {min_hours} hours apart")
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


def timing_from_form(kind, values):
    """Turn the text a person typed into a schedule form into a timing dict,
    with an error message that says which field to fix."""
    if kind == "once":
        return {"type": "once", "at": parse_when(values.get("when", ""))}
    if kind == "interval":
        raw = values.get("every", "").strip()
        if not raw.isdigit() or int(raw) < 1:
            raise ValueError(f"'Every' must be a whole number of at least 1, like 6 (got {raw!r})")
        unit = values.get("unit") or "hours"
        if unit not in ("hours", "days", "weeks"):
            raise ValueError("Choose hours, days or weeks")
        timing = {"type": "interval", "every": int(raw), "unit": unit}
        if values.get("start", "").strip():
            timing["start"] = parse_when(values["start"])
        return timing
    if kind == "daily":
        return {"type": "daily", "time": parse_clock(values.get("time", ""))}
    if kind == "weekly":
        if values.get("weekday") not in WEEKDAYS:
            raise ValueError("Choose a day of the week")
        return {"type": "weekly", "weekday": values["weekday"], "time": parse_clock(values.get("time", ""))}
    raise ValueError("Unknown schedule type")


def form_values(row, kind):
    """The form fields for `kind`, prefilled from an existing schedule. A
    different kind keeps only the timezone and action."""
    values = {"timezone": row.get("timezone", ""), "action": row.get("action", "")}
    timing = row.get("timing") or {}
    if timing.get("type") != kind:
        return values
    zone = ZoneInfo(row["timezone"])

    def local(text):
        value = datetime.fromisoformat(text)
        if value.tzinfo is not None:
            value = value.astimezone(zone)
        return value.strftime("%Y-%m-%d %H:%M")

    if kind == "once":
        values["when"] = local(timing["at"])
    elif kind == "interval":
        values.update(every=str(timing["every"]), unit=timing["unit"])
        if timing.get("start"):
            values["start"] = local(timing["start"])
    else:
        values["time"] = timing.get("time", "")
        if kind == "weekly":
            values["weekday"] = timing.get("weekday", "")
    return values


def quota_message(kind, count, limit):
    """What someone is told when a server has no room for another entry."""
    if limit == 0:
        return f"{kind.capitalize()}s are turned off on this server."
    freed = " A one-off schedule frees its slot once it has run." if kind == "schedule" else ""
    return (f"This server already has {count} of {limit} allowed {kind}s (paused ones count too). "
            f"Delete one with `/{kind} delete` (see `/{kind} list`) to make room.{freed}")

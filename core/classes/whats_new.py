"""What's New announcements: the per-version feature notes (PURE, stdlib only).

The notes live in ONE fixed file, core/whats_new.md, that each feature PR
rewrites. They cannot be named after their version: auto-tag.yaml mints
vX.Y only after the merge. Identity therefore comes from the build instead -
release.yaml bakes the tag into the image as APP_VERSION, and blanks the
file in the build context when it did not change since the previous tag, so
a release without features never re-ships the last release's notes.

That blanking is what makes "a version a guild missed is skipped" hold: the
bot stores only the LATEST version each guild has seen, so a guild that slept
through vX.Y is never shown vX.Y's notes under vX.Y+1.

File format: one `## Feature name` heading per feature, its body (what it
does, and a "How to use:" line) below it. Each feature becomes one embed
field. Text before the first heading and HTML comments are ignored, so an
empty file, or one holding only the template comment, means "no features".
"""
import os
import re
from pathlib import Path

# Discord's embed limits. Exceeding any of them makes channel.send() raise,
# so the shipped file is checked against them in core/tests/whats_new_tests.py.
MAX_FIELDS = 25
MAX_FIELD_NAME = 256
MAX_FIELD_VALUE = 1024
MAX_EMBED_TOTAL = 6000

EMBED_COLOR = 0x5865F2

# The marker that keeps this embed out of prompt history
# (MessageHandler._format_group): a release announcement is not part of the
# conversation, and letting it in would shift anchored-mode prompt prefixes.
ANNOUNCEMENT_FOOTER = "Use /help to see everything I can do"

DEFAULT_NOTES_PATH = Path(__file__).resolve().parent.parent / "whats_new.md"

_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)

# (path, mtime) -> parsed notes. The file is read once per process in
# practice; keyed on mtime so a bind-mounted dev edit is picked up.
_cache: dict = {}


def app_version() -> str:
    """The release tag baked into the image at build time ("" when unknown,
    e.g. local compose or a PR verification build)."""
    return os.environ.get("APP_VERSION", "").strip()


def whats_new_enabled() -> bool:
    """WHATS_NEW_ENABLED (default on) switches the automatic announcement.
    /whats_new keeps working either way - it only answers when asked."""
    raw = os.environ.get("WHATS_NEW_ENABLED", "1")
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def parse_notes(text: str) -> list[tuple[str, str]]:
    """`## Name` sections -> [(name, body)], in file order. Sections with an
    empty body are dropped (a heading alone tells the user nothing)."""
    text = _COMMENT_RE.sub("", text)
    notes = []
    name = None
    body: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            if name and "\n".join(body).strip():
                notes.append((name, "\n".join(body).strip()))
            name, body = line[3:].strip(), []
        elif name is not None:
            body.append(line)
    if name and "\n".join(body).strip():
        notes.append((name, "\n".join(body).strip()))
    return notes


def load_notes(path=None) -> list[tuple[str, str]]:
    """The parsed notes file; [] when it is missing or empty."""
    path = Path(path) if path else DEFAULT_NOTES_PATH
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return []
    key = (str(path), mtime)
    if key not in _cache:
        _cache.clear()
        _cache[key] = parse_notes(path.read_text(encoding="utf-8"))
    return _cache[key]


def limit_problems(notes: list[tuple[str, str]], version: str = "v99.999") -> list[str]:
    """Every way these notes would break a Discord embed ([] = fine)."""
    problems = []
    if len(notes) > MAX_FIELDS:
        problems.append(f"{len(notes)} features, max {MAX_FIELDS}")
    for name, value in notes:
        if len(name) > MAX_FIELD_NAME:
            problems.append(f"heading {name[:40]!r}... is over {MAX_FIELD_NAME} chars")
        if len(value) > MAX_FIELD_VALUE:
            problems.append(f"section {name!r} is {len(value)} chars, max {MAX_FIELD_VALUE}")
    data = notes_embed_data(version, notes)
    total = len(data["title"]) + len(data["footer"]) + sum(
        len(f["name"]) + len(f["value"]) for f in data["fields"])
    if total > MAX_EMBED_TOTAL:
        problems.append(f"embed is {total} chars in total, max {MAX_EMBED_TOTAL}")
    return problems


def notes_embed_data(version: str, notes: list[tuple[str, str]]) -> dict:
    """The announcement as plain data; classes.common.embed_from_data turns
    it into a discord.Embed."""
    return {
        "title": f"✨ What's new in {version}" if version else "✨ What's new",
        "color": EMBED_COLOR,
        "fields": [{"name": name, "value": value} for name, value in notes],
        "footer": ANNOUNCEMENT_FOOTER,
    }


def seen_key(guild_id) -> str:
    """Redis key holding the latest version announced in a guild."""
    return f"dcb:{guild_id}:whats_new_seen"

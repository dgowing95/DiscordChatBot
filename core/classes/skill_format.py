"""Skills: what a saved skill looks like, and the rules it must follow. PURE.

A skill is how the brain solved a recurring job, kept per server so it can be
used again. The layout is agentskills.io's, so a skill is portable: a
SKILL.md (front matter with `name` and `description`, then a Markdown body)
plus a few small text files under scripts/, references/ or assets/.

This module has no I/O: names, limits (SKILLS_* settings), the SKILL.md
parser and renderer, validation, the index shown to the brain and its
ordering. The store (skill_store.py) keeps them in Redis, outside the
container, so a workspace reset or a lost container never loses one.

The index (render_index) goes in the brain's task, after the request, so the
system prompt and tool schemas stay a stable cached prefix. It is ordered by
rank(): skills that worked and were used often come first, and a skill not
used for a long time sinks. When a server has more than SKILLS_INDEX_MAX the
rest are found with search_skills.
"""
import os
import posixpath
import re

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,46}[a-z0-9]$")
DESCRIPTION_CHARS = 200
NOTE_CHARS = 200
SKILL_FILE = "SKILL.md"
FILE_DIRS = ("scripts", "references", "assets")

DEFAULT_MAX_PER_GUILD = 100
DEFAULT_MAX_VERSIONS = 50
DEFAULT_MAX_BYTES = 65536
DEFAULT_MAX_BODY_CHARS = 8000
DEFAULT_MAX_FILES = 10
DEFAULT_INDEX_MAX = 20
DEFAULT_REFLECTION_MIN_TOOL_CALLS = 6
DEFAULT_CURATOR_INTERVAL_HOURS = 24
DEFAULT_CURATOR_MIN_SKILLS = 5
DEFAULT_RETIRE_DAYS = 60


def _flag(value, default: bool) -> bool:
    text = str(value if value is not None else "").strip().lower()
    if not text:
        return default
    return text not in ("0", "false", "no", "off")


def _int(env, name: str, default: int, minimum: int) -> int:
    try:
        return max(minimum, int(str(env.get(name, "")).strip()))
    except ValueError:
        return default


def settings(env=None) -> dict:
    """The SKILLS_* settings. env None = the process environment."""
    env = os.environ if env is None else env
    return {
        "enabled": _flag(env.get("SKILLS_ENABLED"), True),
        "max_per_guild": _int(env, "SKILLS_MAX_PER_GUILD", DEFAULT_MAX_PER_GUILD, 0),
        "max_versions": _int(env, "SKILLS_MAX_VERSIONS", DEFAULT_MAX_VERSIONS, 2),
        "max_bytes": _int(env, "SKILLS_MAX_BYTES", DEFAULT_MAX_BYTES, 1024),
        "max_body_chars": _int(env, "SKILLS_MAX_BODY_CHARS", DEFAULT_MAX_BODY_CHARS, 200),
        "max_files": _int(env, "SKILLS_MAX_FILES", DEFAULT_MAX_FILES, 0),
        "index_max": _int(env, "SKILLS_INDEX_MAX", DEFAULT_INDEX_MAX, 1),
        "reflection": _flag(env.get("SKILLS_REFLECTION_ENABLED"), True),
        "reflection_min_tool_calls": _int(env, "SKILLS_REFLECTION_MIN_TOOL_CALLS",
                                          DEFAULT_REFLECTION_MIN_TOOL_CALLS, 1),
        "curator": _flag(env.get("SKILLS_CURATOR_ENABLED"), True),
        "curator_interval_hours": _int(env, "SKILLS_CURATOR_INTERVAL_HOURS",
                                       DEFAULT_CURATOR_INTERVAL_HOURS, 1),
        "curator_min_skills": _int(env, "SKILLS_CURATOR_MIN_SKILLS", DEFAULT_CURATOR_MIN_SKILLS, 2),
        "retire_days": _int(env, "SKILLS_RETIRE_DAYS", DEFAULT_RETIRE_DAYS, 1),
    }


def enabled(env=None) -> bool:
    return settings(env)["enabled"]


def clean_name(name) -> str:
    """The name as typed, lower-cased with spaces and underscores as hyphens.
    Not validated: validate() says what is wrong."""
    return re.sub(r"[\s_]+", "-", str(name or "").strip().lower())


def one_line(text, limit: int) -> str:
    return " ".join(str(text or "").split())[:limit]


def parse_skill_md(text: str) -> tuple[dict, str]:
    """(front matter, body) of a SKILL.md. The front matter is `key: value`
    lines between two `---` fences (a small parser, not YAML: skills use two
    plain keys). No fences = no front matter."""
    text = str(text or "").replace("\r\n", "\n")
    if not text.startswith("---\n"):
        return {}, text.strip()
    end = text.find("\n---", 4)
    if end < 0:
        return {}, text.strip()
    meta = {}
    for line in text[4:end].split("\n"):
        key, sep, value = line.partition(":")
        if sep and key.strip():
            meta[key.strip().lower()] = value.strip().strip("\"'")
    rest = text[end + 4:]
    return meta, rest.lstrip("\n").strip()


def render_skill_md(name: str, description: str, body: str) -> str:
    return f"---\nname: {name}\ndescription: {one_line(description, DESCRIPTION_CHARS)}\n---\n\n{str(body).strip()}\n"


def _safe_path(path) -> str:
    """The problem with a file path inside a skill, or ''."""
    path = str(path or "")
    if not path or path != path.strip():
        return "empty or padded path"
    if "\\" in path or path.startswith("/") or ".." in path.split("/"):
        return "must be relative, with no '..' or backslashes"
    clean = posixpath.normpath(path)
    if clean != path:
        return "not a clean path"
    if clean.split("/")[0] not in FILE_DIRS or "/" not in clean:
        return "must be under " + ", ".join(d + "/" for d in FILE_DIRS)
    return ""


def validate(name, description, body, files=None, *, env=None) -> list[str]:
    """Everything wrong with a skill, in words for the brain / the person
    editing. Empty = fine. Used by the brain's tool, /skills edit, reflection
    and the curator alike."""
    cfg = settings(env)
    problems = []
    if not NAME_RE.match(str(name or "")):
        problems.append("name: 3-48 characters, lower-case letters, digits and hyphens, "
                        "starting and ending with a letter or digit")
    description = str(description or "")
    if not description.strip():
        problems.append("description: one line saying what the skill does")
    elif "\n" in description.strip() or len(description) > DESCRIPTION_CHARS:
        problems.append(f"description: one line of at most {DESCRIPTION_CHARS} characters")
    body = str(body or "")
    if not body.strip():
        problems.append("body: the steps to follow")
    elif len(body) > cfg["max_body_chars"]:
        problems.append(f"body: at most {cfg['max_body_chars']} characters (it has {len(body)})")
    files = files or {}
    if len(files) > cfg["max_files"]:
        problems.append(f"files: at most {cfg['max_files']}")
    total = len(body.encode("utf-8")) + len(str(description).encode("utf-8"))
    for path, text in files.items():
        bad = _safe_path(path)
        if bad:
            problems.append(f"file {path!r}: {bad}")
        if not isinstance(text, str):
            problems.append(f"file {path!r}: text only")
            continue
        if "\x00" in text:
            problems.append(f"file {path!r}: text only")
        total += len(text.encode("utf-8"))
    if total > cfg["max_bytes"]:
        problems.append(f"too big: {total} bytes, at most {cfg['max_bytes']} all together")
    return problems


def rank(skills, now: float) -> list:
    """Skills in index order. Score = 3 per success + 1 per use, minus one
    point per 30 days since last use (or since it was saved), so a proven
    skill outranks a merely used one and an abandoned one sinks. Ties go to
    the newer, then the name, so the order never flips between builds (the
    index is part of the brain's prompt)."""
    def score(skill):
        seen = skill.get("last_used_at") or skill.get("updated_at") or 0
        idle_days = max(0.0, (now - seen) / 86400)
        return (3 * skill.get("successes", 0) + skill.get("uses", 0) - idle_days / 30,
                skill.get("updated_at", 0))
    return sorted(skills, key=lambda s: (tuple(-v for v in score(s)), s.get("name", "")))


def render_index(skills, limit: int, now: float) -> str:
    """The brain's list of this server's skills ('' when there are none)."""
    ordered = rank(skills, now)
    if not ordered:
        return ""
    lines = ["Skills saved in this server (use_skill to open one):"]
    for skill in ordered[:limit]:
        lines.append(f"- {skill['name']}: {one_line(skill.get('description'), DESCRIPTION_CHARS)}")
    if len(ordered) > limit:
        lines.append(f"({len(ordered) - limit} more: search_skills)")
    return "\n".join(lines)


def search(skills, query: str, limit: int = 5) -> list:
    """Skills whose name, description or body share words with the query,
    best match first (name hits count most)."""
    words = {w for w in re.findall(r"[a-z0-9]+", str(query or "").lower()) if len(w) > 2}
    if not words:
        return []
    scored = []
    for skill in skills:
        name = set(re.findall(r"[a-z0-9]+", skill.get("name", "")))
        described = set(re.findall(r"[a-z0-9]+", str(skill.get("description", "")).lower()))
        body = set(re.findall(r"[a-z0-9]+", str(skill.get("body", "")).lower()))
        score = 3 * len(words & name) + 2 * len(words & described) + len(words & body)
        if score:
            scored.append((-score, skill.get("name", ""), skill))
    scored.sort(key=lambda row: row[:2])
    return [row[2] for row in scored[:limit]]


def closest_names(names, wanted: str, limit: int = 5) -> list[str]:
    """The saved names nearest to a mistyped one."""
    import difflib
    wanted = clean_name(wanted)
    close = difflib.get_close_matches(wanted, list(names), n=limit, cutoff=0.4)
    others = [n for n in sorted(names) if n not in close and wanted and wanted in n]
    return (close + others)[:limit]

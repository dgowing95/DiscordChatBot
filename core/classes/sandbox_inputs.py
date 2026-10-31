"""Which chat files a sandbox run is given, and how each side is told (PURE,
stdlib only).

The sandbox used to receive nothing but the `task` string, so anything it was
meant to work ON had to be squeezed through text: an uploaded CSV pasted in by
the outer model (when it could see it at all -- non-image attachments never
reached the prompt), and a picture the bot had just drawn not at all. So
"make a GIF from that picture" or "chart this spreadsheet" could not work.

Now run_code_sandbox copies files into the workspace's `in/` folder before the
run starts:

  * always, the attachments of the message that asked and of the message it
    replies to -- the user plainly meant those;
  * any other file the outer model names in the tool's `files` argument,
    found by name among the channel's recent messages (newest first). That is
    how the bot's own pictures are reached: every one is called
    GENERATED_IMAGE_NAME, so naming it gets the newest.

Caps: MAX_INPUT_FILES files, and SANDBOX_INPUT_MAX_BYTES (default 25MB) in
total, kept well under SANDBOX_SNAPSHOT_MAX_BYTES since `in/` is saved with
the workspace for the next run in the thread. What was left out, and why, is
told to both the sandbox model (so it does not go looking) and the outer one
(so it can tell the user).

tool_functions.py does the downloading and sandbox_agent.run_sandbox_task the
writing; core/tests/sandbox_inputs_tests.py covers this module.
"""
import os
import re
from collections import namedtuple

INPUT_DIRNAME = "in"
MAX_INPUT_FILES = 10
DEFAULT_MAX_BYTES = 25_000_000
GENERATED_IMAGE_NAME = "generated-image.png"

# name: as Discord shows it; size: bytes; ref: the attachment itself, carried
# through untouched so the caller can download what was chosen.
Candidate = namedtuple("Candidate", "name size ref")

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def max_input_bytes() -> int:
    try:
        value = int(os.environ.get("SANDBOX_INPUT_MAX_BYTES", "").strip())
    except ValueError:
        return DEFAULT_MAX_BYTES
    return value if value > 0 else DEFAULT_MAX_BYTES


def safe_name(filename: str) -> str:
    """One path segment the shell is happy with: no directories, no spaces."""
    base = (filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    cleaned = _UNSAFE.sub("_", base).strip("._")
    return cleaned or "file"


def unique_names(names) -> list:
    """safe_name of each, with -2, -3... added where two would collide."""
    out, used = [], set()
    for name in names:
        candidate = safe_name(name)
        stem, dot, ext = candidate.rpartition(".")
        if not dot or not stem:
            stem, ext = candidate, ""
        n = 1
        while candidate.lower() in used:
            n += 1
            candidate = f"{stem}-{n}.{ext}" if ext else f"{stem}-{n}"
        used.add(candidate.lower())
        out.append(candidate)
    return out


def human_size(size) -> str:
    size = int(size or 0)
    if size < 1000:
        return f"{size} B"
    if size < 1_000_000:
        return f"{size / 1000:.0f} KB"
    return f"{size / 1_000_000:.1f} MB"


def _matches(candidate, wanted: str) -> bool:
    want = (wanted or "").strip().lower()
    return bool(want) and want in (candidate.name.lower(), safe_name(candidate.name).lower())


def select(given, recent, requested, max_bytes: int):
    """(chosen, missing, skipped).

    given: the trigger's and its reply target's attachments, always taken.
    recent: the channel's recent attachments, NEWEST first, searched by name
    for each `requested` one that `given` does not already hold.
    missing: requested names found nowhere. skipped: "name (reason)" for files
    over the caps.
    """
    wanted, missing = [], []
    for candidate in given:
        if candidate not in wanted:
            wanted.append(candidate)
    for name in requested or []:
        if not (name or "").strip():
            continue
        if any(_matches(c, name) for c in wanted):
            continue
        found = next((c for c in recent if _matches(c, name)), None)
        if found is None:
            missing.append(name.strip())
        else:
            wanted.append(found)

    chosen, skipped, total = [], [], 0
    for candidate in wanted:
        if len(chosen) >= MAX_INPUT_FILES:
            skipped.append(f"{candidate.name} (more than {MAX_INPUT_FILES} files)")
        elif total + (candidate.size or 0) > max_bytes:
            skipped.append(f"{candidate.name} (over the {human_size(max_bytes)} limit)")
        else:
            chosen.append(candidate)
            total += candidate.size or 0
    return chosen, missing, skipped


def task_preamble(written, problems) -> str:
    """Put before the task: what is in `in/`, and what could not be given.
    written: [(name, size)]. Empty when there is nothing to say."""
    lines = []
    if written:
        files = ", ".join(f"{name} ({human_size(size)})" for name, size in written)
        lines.append(f"Files for this task are already in {INPUT_DIRNAME}/: {files}.")
    if problems:
        lines.append("Asked for but not provided: " + "; ".join(problems) + ".")
    return ("\n".join(lines) + "\n\n") if lines else ""


def tool_note(names, problems) -> str:
    """Added to what the outer model is told about the run."""
    parts = []
    if names:
        parts.append("Files given to the sandbox: " + ", ".join(names) + ".")
    if problems:
        parts.append("Not given (tell the user): " + "; ".join(problems) + ".")
    return ("\n\n" + " ".join(parts)) if parts else ""


def attachment_note(attachments) -> str:
    """How a user message's attachments read in the prompt, so the model
    knows a file is there and what to call it. [(filename, size)]."""
    if not attachments:
        return ""
    return "\n[Attached: " + ", ".join(f"{name} ({human_size(size)})"
                                       for name, size in attachments) + "]"

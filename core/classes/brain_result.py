"""What the brain hands back to the persona, and how the persona is told it.

PURE (stdlib only), so the wording and the fact checks are unit-tested
without the SDK.

The brain (classes/brain.py) finishes by calling its `report` tool. The
arguments land here as a BrainReport, and tool_result_for_persona turns that
into the string ask_brain returns. That string is the lever that changes what
the persona writes, so it is shaped as FACTS TO PASS ON, not prose to
paraphrase: the main risk of splitting the bot in two is a persona that
rewords a price or drops a link, so the exact tokens it must keep are listed
separately (must_keep), and missing_tokens checks the reply for them after.

What was already posted (files, a poll, an image) is listed too, because a
persona that does not know tells the user the picture is "coming".
"""
import posixpath
import re
from dataclasses import dataclass, field

DONE = "done"
PARTIAL = "partial"
NEEDS_INPUT = "needs_input"
FAILED = "failed"
STATUSES = (DONE, PARTIAL, NEEDS_INPUT, FAILED)
# Set by run_brain when the brain handed the job to a Tier 3 workflow
# (start_workflow), never from report's own arguments: a model writing it
# there has started nothing.
STARTED_WORKFLOW = "started_workflow"

# Caps on what the persona is handed: a brain that writes an essay into
# `facts` would otherwise fill the persona's slot with it.
MAX_FACTS = 20
MAX_FACT_CHARS = 600
MAX_MUST_KEEP = 20
MAX_TOKEN_CHARS = 300
MAX_TEXT_CHARS = 1500


@dataclass
class BrainReport:
    status: str = DONE
    facts: list[str] = field(default_factory=list)
    # Exact strings the reply must contain: URLs, numbers, names, file names.
    must_keep: list[str] = field(default_factory=list)
    # Files already posted in the channel (filled from what was sent, never
    # from the brain's own claim).
    files_posted: list[str] = field(default_factory=list)
    # Other things already posted: "a poll", "an image", "an event".
    posted: list[str] = field(default_factory=list)
    say_to_user: str = ""
    # Why it stopped, inputs left out, the workspace being unavailable...
    note: str = ""
    # The workflow's thread, when the job went to one (STARTED_WORKFLOW).
    thread_url: str = ""


def _clean(text, limit: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _strings(value, limit: int, count: int) -> list[str]:
    """A list of short strings from whatever the model sent: a list, one
    string (split on lines), or nothing. Empty entries and repeats go."""
    if value is None:
        return []
    if isinstance(value, str):
        value = value.splitlines()
    if not isinstance(value, (list, tuple)):
        value = [value]
    out = []
    for item in value:
        text = _clean(item, limit).lstrip("-•* ").strip()
        if text and text not in out:
            out.append(text)
        if len(out) >= count:
            break
    return out


def parse_report_args(raw: dict | None) -> BrainReport:
    """A BrainReport from the `report` tool's arguments. Lenient: a missing
    or odd field becomes empty, an unknown status becomes done (the model
    called report, so it thinks it finished)."""
    raw = raw if isinstance(raw, dict) else {}
    status = str(raw.get("status") or DONE).strip().lower()
    if status not in STATUSES:
        status = DONE
    facts = _strings(raw.get("facts"), MAX_FACT_CHARS, MAX_FACTS)
    must_keep = _strings(raw.get("must_keep"), MAX_TOKEN_CHARS, MAX_MUST_KEEP)
    for token in auto_must_keep(facts):
        if token not in must_keep and len(must_keep) < MAX_MUST_KEEP:
            must_keep.append(token)
    return BrainReport(
        status=status, facts=facts, must_keep=must_keep,
        say_to_user=_clean(raw.get("say_to_user"), MAX_TEXT_CHARS),
    )


def report_from_text(text: str) -> BrainReport:
    """The fallback when the brain ended with plain text instead of calling
    report: the text, as facts (one per paragraph)."""
    paragraphs = [p for p in re.split(r"\n\s*\n", str(text or "")) if p.strip()]
    facts = _strings(paragraphs, MAX_FACT_CHARS, MAX_FACTS)
    return BrainReport(status=DONE if facts else PARTIAL, facts=facts,
                       must_keep=auto_must_keep(facts)[:MAX_MUST_KEEP])


_URL = re.compile(r"https?://[^\s<>()\"']+")
_CODE = re.compile(r"`([^`\n]{1,200})`")
# A number with a currency sign or a unit after it: "$1,199", "42 km",
# "3.5%". Bare numbers are left out: "1" in "step 1" is not worth keeping.
_AMOUNT = re.compile(
    r"(?:[$£€¥]\s?\d[\d,]*(?:\.\d+)?)"
    r"|(?:\d[\d,]*(?:\.\d+)?\s?(?:%|°[CF]?|km|kg|mb|gb|tb|ms|mph|kph|cm|mm|kb|m|s|g|l)\b)",
    re.IGNORECASE)
_FILE = re.compile(r"\b[\w.-]+\.(?:png|jpe?g|gif|webp|svg|csv|json|txt|md|pdf|py|zip|html|mp4|mp3|wav|xlsx)\b",
                   re.IGNORECASE)


def auto_must_keep(facts) -> list[str]:
    """The exact tokens in `facts` a reply should not reword: URLs, amounts
    with a unit or currency, `code`, file names. In order, no repeats."""
    out: list[str] = []

    def _add(token: str) -> None:
        token = token.strip().rstrip(".,;:!?")
        if token and token not in out:
            out.append(token)

    for fact in facts or []:
        for match in _URL.finditer(fact):
            _add(match.group(0))
        for match in _CODE.finditer(fact):
            _add(match.group(1))
        for match in _AMOUNT.finditer(fact):
            _add(match.group(0))
        for match in _FILE.finditer(fact):
            if not any(match.group(0) in url for url in out if url.startswith("http")):
                _add(match.group(0))
    return out


def _join(items) -> str:
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


_STATUS_LINE = {
    DONE: "Brain result: done.",
    PARTIAL: "Brain result: only partly done.",
    NEEDS_INPUT: "Brain result: it needs more from the user.",
    FAILED: "Brain result: it could not do this.",
    STARTED_WORKFLOW: "Brain result: this is a big job, so it started a workflow in its own thread.",
}


def tool_result_for_persona(report: BrainReport) -> str:
    """The string ask_brain returns to the persona."""
    lines = [_STATUS_LINE.get(report.status, _STATUS_LINE[DONE])]
    if report.status == STARTED_WORKFLOW:
        lines.append(f"It runs in {report.thread_url}: the plan is posted there first. Tell "
                     "them it has started and where. It is not done; never say it is. "
                     "You will be told when it ends.")
    posted = list(report.files_posted) + list(report.posted)
    if posted:
        lines.append(f"Already posted in the channel: {_join(posted)}. "
                     "Do not describe them as coming later.")
    if report.facts:
        lines.append("Facts to pass on (keep exact):")
        lines.extend(f"- {fact}" for fact in report.facts)
    if report.must_keep:
        lines.append("Must appear exactly in your reply: " + ", ".join(report.must_keep))
    if report.say_to_user:
        lines.append(f"Tell the user: {report.say_to_user}")
    if report.note:
        lines.append(f"Note: {report.note}")
    if report.status in (PARTIAL, FAILED):
        lines.append("Say plainly what is not done. Do not ask the brain to try the same "
                     "thing again in this reply.")
    if report.facts or report.say_to_user:
        lines.append("Write your reply in your own voice. Keep every fact above; do not "
                     "add facts.")
    return "\n".join(lines)


# Dashes and spaces a reply may write differently from the brain's fact
# without changing it: "8-10°C" and "8–10°C" (seen live) are the same range.
_SAME = str.maketrans({"‐": "-", "‑": "-", "‒": "-", "–": "-",
                       "—": "-", "−": "-", " ": " ", " ": " ",
                       " ": " "})


# Markdown bold/italic and code marks are dropped too: "feels like **~15°C**"
# keeps "feels like ~15°C" (seen live). Not "_", which URLs and names contain.
_SAME.update({ord("*"): None, ord("`"): None})


def _fold(text: str) -> str:
    return text.translate(_SAME).casefold()


def missing_tokens(reply: str, report: BrainReport) -> list[str]:
    """must_keep tokens the reply left out. Case-insensitive, any dash
    matches any other, * and ` (Markdown) are ignored, and a URL counts as kept with or without a trailing
    slash. The name of a file already posted is not needed in the reply:
    the file is right there."""
    text = _fold(reply or "")
    posted = {_fold(posixpath.basename(str(name))) for name in report.files_posted}
    missing = []
    for token in report.must_keep:
        folded = _fold(token)
        if folded in posted:
            continue
        if folded in text or folded.rstrip("/") in text:
            continue
        missing.append(token)
    return missing

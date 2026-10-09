"""Polls (PURE, stdlib only): the create_poll tool's input checks, and how a
poll reads in a prompt.

A poll message has no text and no embeds, so before this the history build
left it out of the prompt entirely and the bot could not see a poll it had
made, let alone its votes. `format_poll` renders it as one entry, voter names
included (fetched in MessageHandler.prepare_messages).

The poll's end is written as an absolute time, never "ends in 3h": the entry
must read the same on every build while nobody votes, or llama.cpp loses its
cached prompt prefix from that message onward on every reply.
"""
from datetime import datetime, timezone

# Discord's own limits for a poll.
MAX_QUESTION = 300
MAX_ANSWER = 55
MIN_ANSWERS = 2
MAX_ANSWERS = 10
MAX_HOURS = 768  # 32 days

# Voter names shown per answer; the rest become "+N more".
VOTER_NAMES_PER_ANSWER = 25


def validate(question, answers, duration_hours):
    """(question, answers, hours, error). `error` is "" when the poll can be
    posted; otherwise it tells the model what to fix and to call again.

    Nothing is truncated: an option cut off mid-word is worse than a retry."""
    question = (question or "").strip()
    seen = set()
    cleaned = []
    for answer in answers or []:
        text = str(answer).strip()
        if text and text.casefold() not in seen:
            seen.add(text.casefold())
            cleaned.append(text)
    try:
        hours = int(duration_hours)
    except (TypeError, ValueError):
        hours = 24
    hours = min(max(hours, 1), MAX_HOURS)

    problems = []
    if not question:
        problems.append("the question is empty")
    elif len(question) > MAX_QUESTION:
        problems.append(f"the question is over {MAX_QUESTION} characters")
    if not MIN_ANSWERS <= len(cleaned) <= MAX_ANSWERS:
        problems.append(f"give {MIN_ANSWERS} to {MAX_ANSWERS} different options "
                        f"(got {len(cleaned)})")
    too_long = [a for a in cleaned if len(a) > MAX_ANSWER]
    if too_long:
        listed = ", ".join(f'"{a}"' for a in too_long)
        problems.append(f"shorten these options to {MAX_ANSWER} characters or fewer: {listed}")
    if problems:
        return question, cleaned, hours, (
            "No poll was posted: " + "; ".join(problems) + ". Fix that and call create_poll again.")
    return question, cleaned, hours, ""


def _when(moment):
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _votes(n):
    return f"{n} vote" if n == 1 else f"{n} votes"


def format_poll(author, question, answers, total, multiple, finalized, expires_at):
    """One poll as prompt text.

    `answers` is a list of (text, count, voters): voters is a list of names,
    or None when they could not be looked up (counts only)."""
    details = ["several choices each" if multiple else "one choice each"]
    if finalized:
        details.append("ended")
    elif isinstance(expires_at, datetime):
        details.append(f"open until {_when(expires_at)}")
    else:
        details.append("open")
    details.append(_votes(total))
    # The same "Message from" label as every other history entry: the model
    # sometimes copies a label into its reply, and response_filter strips
    # only that one. A "Poll from" label of its own leaked into replies.
    lines = [f"Message from '{author}': Poll \"{question}\" ({', '.join(details)})"]
    for text, count, voters in answers:
        line = f"- {text}: {count}"
        if voters:
            names = list(voters)[:VOTER_NAMES_PER_ANSWER]
            more = max(count, len(voters)) - len(names)
            listed = ", ".join(names) + (f", +{more} more" if more > 0 else "")
            line += f" ({listed})"
        lines.append(line)
    return "\n".join(lines)


def format_poll_result(author, fields):
    """Discord's "poll ended" system message, from its embed's fields (the
    names discord.py's Poll._update_results_from_message reads). The generic
    embed path drops fields, which would leave nothing of the result."""
    question = fields.get("poll_question_text") or "a poll"
    try:
        total = int(fields.get("total_votes") or 0)
    except ValueError:
        total = 0
    winner = fields.get("victor_answer_text")
    if winner:
        try:
            votes = int(fields.get("victor_answer_votes") or 0)
        except ValueError:
            votes = 0
        outcome = f"winner: {winner} with {votes} of {_votes(total)}"
    elif total:
        outcome = f"no single winner ({_votes(total)})"
    else:
        outcome = "no votes"
    return f"Message from '{author}': Poll ended: \"{question}\" - {outcome}"

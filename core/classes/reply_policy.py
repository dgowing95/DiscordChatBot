"""Occasional conversational pairs, independent of Discord and the LLM SDK.

Eligibility is an opportunity, never a requirement. The send lock must guard
the cooldown recheck and updates: generations can finish in a different order
from their prompts. State is process-local; restarts/eviction reset cooldowns.
"""
from collections import OrderedDict
import math
import os
import random
import re

DEFAULT_CHANCE = 0.08
DEFAULT_COOLDOWN = 10
MAX_PART_CHARS = 300
MARKER = "<message_break>"
INSTRUCTION = (
    "You may deliver this reply as two short chat messages if that suits the "
    "conversation. A quick reaction followed by a related thought can work well. "
    "One message is the default; use two only when the pause adds something. "
    "Keep both consistent with your personality and the conversation. "
    f"Separate them with a standalone {MARKER} line. Use one message for "
    "detailed explanations, code, or task reports."
)


def double_reply_chance() -> float:
    try:
        value = float(os.environ.get("DOUBLE_REPLY_CHANCE", DEFAULT_CHANCE))
        if math.isfinite(value) and 0 <= value <= 1:
            return value
    except (ValueError, TypeError):
        pass
    return DEFAULT_CHANCE


def cooldown_replies() -> int:
    try:
        value = int(os.environ.get("DOUBLE_REPLY_COOLDOWN_REPLIES", DEFAULT_COOLDOWN))
        if value >= 0:
            return value
    except (ValueError, TypeError):
        pass
    return DEFAULT_COOLDOWN


class ReplyPolicy:
    def __init__(self, max_channels=4096):
        self._remaining: OrderedDict[int, int] = OrderedDict()
        self.max_channels = max_channels

    def _touch(self, channel_id):
        self._remaining.setdefault(channel_id, 0)
        self._remaining.move_to_end(channel_id)
        while len(self._remaining) > self.max_channels:
            self._remaining.popitem(last=False)

    def available(self, channel_id) -> bool:
        self._touch(channel_id)
        return self._remaining[channel_id] == 0

    def eligible(self, channel_id) -> bool:
        return self.available(channel_id) and random.random() < double_reply_chance()

    def started_pair(self, channel_id):
        """Commit after the first send succeeds, even if the second later fails."""
        self._touch(channel_id)
        self._remaining[channel_id] = cooldown_replies()

    def sent_single(self, channel_id):
        """Count one fully delivered reply, regardless of its Discord chunk count."""
        self._touch(channel_id)
        self._remaining[channel_id] = max(0, self._remaining[channel_id] - 1)


policy = ReplyPolicy()


def reply_parts(text: str, *, allowed: bool) -> list[str]:
    """Parse standalone separators outside fences; fall back without losing text.

Normal newlines and literal separators in fenced examples retain their meaning.
Invalid or disallowed separators become paragraph breaks, never visible controls.
"""
    parts = [""]
    fence = None
    has_code = False
    for line in text.splitlines(keepends=True):
        match = re.match(r"^[ \t]*(`{3,}|~{3,})", line)
        if fence:
            if re.fullmatch(r"[ \t]*" + re.escape(fence[0]) +
                            "{" + str(len(fence)) + r",}[ \t\r\n]*", line):
                fence = None
        elif match:
            fence = match.group(1)
            has_code = True
        elif line.strip() == MARKER:
            parts.append("")
            continue
        parts[-1] += line
    parts = [part.strip() for part in parts]
    if allowed and not has_code and len(parts) == 2 and all(
        0 < len(part) <= MAX_PART_CHARS for part in parts
    ):
        return parts
    return ["\n\n".join(part for part in parts if part)]

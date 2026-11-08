import logging

from classes import card_kit

logger = logging.getLogger(__name__)


class _NoMessage:
    """What a dropped notice "sent", for a tool that edits its card later.
    `dropped` tells a tool that must still deliver something (a picture) to
    send it anew."""

    dropped = True

    async def edit(self, **kwargs) -> None:
        pass


class QuietChannel:
    """The channel a workflow's scouts and researchers run with: a tool's own
    notice card goes nowhere, and everything else (a question, a file) is the
    real channel. Each posts one notice per tool call, dozens in a run, which
    buried the thread. Their one status card (brain_progress.HelperProgress)
    says what they are doing."""

    def __init__(self, channel):
        self._channel = channel

    def __getattr__(self, name):
        return getattr(self._channel, name)

    async def send(self, *args, **kwargs):
        view = kwargs.get("view")
        if (not args and getattr(view, "kind", None) == card_kit.NOTICE
                and not kwargs.get("file") and not kwargs.get("files")):
            return _NoMessage()
        return await self._channel.send(*args, **kwargs)



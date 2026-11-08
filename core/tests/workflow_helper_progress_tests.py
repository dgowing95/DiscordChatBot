"""What a workflow's scouts and researchers show in the thread: one card each,
and tools that post nothing. Run against a fake channel, with the clock moved by
hand, since the edit limit is the point."""
from types import SimpleNamespace

import pytest

from classes import brain_progress, card, card_kit as K, tool_cards
from classes.brain_progress import HelperProgress
from classes.common import QuietChannel
from card_checks import view_text


class Message:
    def __init__(self):
        self.edits = []

    async def edit(self, **kwargs):
        self.edits.append(kwargs)


class Channel:
    id = 500

    def __init__(self):
        self.sent = []

    async def send(self, *args, **kwargs):
        message = Message()
        message.args, message.kwargs = args, kwargs
        self.sent.append(message)
        return message


def _call(args="{}"):
    return SimpleNamespace(tool_arguments=args)


def _tool(name):
    return SimpleNamespace(name=name)


@pytest.mark.parametrize("tool,args,expected", [
    ("web_search", {"search_request": "ORR tables"}, "🔎 Searching: ORR tables"),
    ("fetch_url", {"url": "https://www.example.com/a/very/long/path?x=1"}, "🌐 Reading example.com"),
    ("fetch_url", {"url": "example.com/page"}, "🌐 Reading example.com"),
    ("exec_command", {"cmd": "head sales.csv\nwc -l sales.csv"}, "💻 Running head"),
    ("exec_command", {}, "💻 Running a command"),
    ("read_messages_around", {}, "📜 Reading messages"),
    ("something_new", {}, "🔧 something new"),
    ("report", {"facts": ["x"]}, None),
    ("write_stdin", {"chars": ""}, None),
])
def test_a_helper_line_says_what_it_started(tool, args, expected):
    assert tool_cards.helper_line(tool, args) == expected


def test_a_helper_line_is_short_whatever_it_is_given():
    line = tool_cards.helper_line("web_search", {"search_request": "word " * 500})
    assert len(line) <= tool_cards.HELPER_LINE_CHARS and line.endswith("…")


def test_a_helper_line_cannot_format_the_card():
    line = tool_cards.helper_line("web_search", {"search_request": "`date` **x**"})
    assert "`" not in line and "*" not in line


@pytest.mark.asyncio
async def test_one_message_edited_no_more_often_than_the_limit(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(brain_progress.time, "monotonic", lambda: now[0])
    channel = Channel()
    progress = HelperProgress(channel, "🧭 Looking around before planning", title="Rail",
                              max_turns=12)
    await progress.start()
    for tick, name in enumerate(["web_search", "fetch_url", "fetch_url"]):
        now[0] += 5
        await progress.on_llm_start(None, None, None, [])
        await progress.on_tool_start(_call(), None, _tool(name))
    assert len(channel.sent) == 1 and channel.sent[0].edits == []
    first = view_text(channel.sent[0].kwargs["view"])
    assert first.startswith("-# 🧭 Looking around before planning\n### Rail") and "Starting…" in first
    now[0] += 30
    await progress.on_llm_start(None, None, None, [])
    await progress.on_tool_start(_call('{"url": "https://a.com"}'), None, _tool("fetch_url"))
    edits = channel.sent[0].edits
    assert len(edits) == 1
    text = view_text(edits[0]["view"])
    assert "🌐 Reading a.com" in text and text.endswith("-# Turn 4 of 12")
    assert edits[0]["view"].colour == K.WORKING


@pytest.mark.asyncio
async def test_the_ending_is_never_held_back_and_comes_once(monkeypatch):
    channel = Channel()
    progress = HelperProgress(channel, "🔬 Research · step 1 of 2", title="Find prices",
                              brief="Find the price of every printer", max_turns=12)
    await progress.start()
    progress.line = "🌐 Reading a.com"
    await progress.finalize(ok=True, found=2)
    await progress.finalize(ok=False, found=9)
    assert len(channel.sent[0].edits) == 1
    view = channel.sent[0].edits[0]["view"]
    text = view_text(view)
    assert "✅ Finished · found 2 facts" in text and "Last: 🌐 Reading a.com" in text
    assert "Find the price of every printer" in text  # the whole brief, small
    assert view.colour == K.OK


@pytest.mark.asyncio
async def test_a_failed_update_never_reaches_the_run():
    class Broken:
        async def send(self, *args, **kwargs):
            raise RuntimeError("no")

    progress = HelperProgress(Broken(), "🔬 Research", max_turns=12, edit_interval=0)
    await progress.start()
    await progress.on_tool_start(_call("not json"), None, _tool("web_search"))
    await progress.finalize()


@pytest.mark.asyncio
async def test_a_long_brief_is_shown_whole_over_several_messages():
    channel = Channel()
    progress = HelperProgress(channel, "🔬 Research", brief="word " * 1500, max_turns=12)
    await progress.start()
    views = [m.kwargs["view"] for m in channel.sent]
    assert len(views) > 1 and all(v.content_length() <= K.MAX_TEXT for v in views)
    assert sum(view_text(v).count("word") for v in views) == 1500


@pytest.mark.asyncio
async def test_the_quiet_channel_drops_a_tool_notice_and_nothing_else():
    channel = Channel()
    quiet = QuietChannel(channel)
    live = await card.notice(quiet, K.info("🔎 Searching the web", "text"))
    await live.show(K.info("🔎 Searched the web", "more"))
    assert channel.sent == [] and quiet.id == 500 and live.dropped
    await quiet.send("plain text")
    await quiet.send(view=card.views(K.info("x"))[0], files=["a.png"])
    assert len(channel.sent) == 2

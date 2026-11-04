"""What a workflow's scouts and researchers show in the thread: one embed each,
and tools that post nothing. Run against a fake channel, with the clock moved by
hand, since the edit limit is the point."""
from types import SimpleNamespace

import discord
import pytest

from classes import brain_progress, tool_embeds
from classes.brain_progress import HelperProgress
from classes.common import Common, QuietChannel


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
    ("exec_command", {"cmd": "head sales.csv\nwc -l sales.csv"}, "💻 head sales.csv"),
    ("exec_command", {}, "💻"),
    ("read_messages_around", {}, "📜 Reading messages"),
    ("something_new", {}, "🔧 something_new"),
    ("report", {"facts": ["x"]}, None),
    ("write_stdin", {"chars": ""}, None),
])
def test_a_helper_line_says_what_it_started(tool, args, expected):
    assert tool_embeds.helper_line(tool, args) == expected


def test_a_helper_line_is_short_whatever_it_is_given():
    line = tool_embeds.helper_line("web_search", {"search_request": "word " * 500})
    assert len(line) <= tool_embeds.HELPER_LINE_CHARS and line.endswith("…")


def test_a_helper_line_cannot_format_the_embed():
    assert "`" not in tool_embeds.helper_line("exec_command", {"cmd": "echo `date` **x**"})


def test_the_ending_says_how_it_went_how_many_and_the_last_call():
    assert tool_embeds.helper_ending(True, None, 3, "🌐 Reading a.com") == (
        "✅ Done · 3 facts\nLast: 🌐 Reading a.com")
    assert tool_embeds.helper_ending(False, "⚠️ Out of steps.", 1, "") == "⚠️ Out of steps. · 1 fact"
    assert tool_embeds.helper_ending(False, None, 0, "") == "⚠️ Not finished · 0 facts"


@pytest.mark.asyncio
async def test_one_message_edited_no_more_often_than_the_limit(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(brain_progress.time, "monotonic", lambda: now[0])
    channel = Channel()
    progress = HelperProgress(channel, "🔎 Rail", max_turns=12)
    await progress.start()
    for tick, name in enumerate(["web_search", "fetch_url", "fetch_url"]):
        now[0] += 5
        await progress.on_llm_start(None, None, None, [])
        await progress.on_tool_start(_call(), None, _tool(name))
    assert len(channel.sent) == 1 and channel.sent[0].edits == []
    now[0] += 30
    await progress.on_llm_start(None, None, None, [])
    await progress.on_tool_start(_call('{"url": "https://a.com"}'), None, _tool("fetch_url"))
    edits = channel.sent[0].edits
    assert len(edits) == 1 and edits[0]["embed"].description == "🌐 Reading a.com"
    assert edits[0]["embed"].footer.text == "Turn 4 of 12"


@pytest.mark.asyncio
async def test_the_ending_is_never_held_back_and_comes_once(monkeypatch):
    channel = Channel()
    progress = HelperProgress(channel, "🔎 Rail", max_turns=12)
    await progress.start()
    await progress.finalize(ok=True, found=2)
    await progress.finalize(ok=False, found=9)
    assert len(channel.sent[0].edits) == 1
    assert channel.sent[0].edits[0]["embed"].description == "✅ Done · 2 facts"


@pytest.mark.asyncio
async def test_a_failed_update_never_reaches_the_run():
    class Broken:
        async def send(self, *args, **kwargs):
            raise RuntimeError("no")

    progress = HelperProgress(Broken(), "🔎 Rail", max_turns=12, edit_interval=0)
    await progress.start()
    await progress.on_tool_start(_call("not json"), None, _tool("web_search"))
    await progress.finalize()


def test_a_long_title_fits_discord():
    progress = HelperProgress(Channel(), "x" * 900, max_turns=12)
    assert len(progress.title) <= 256


@pytest.mark.asyncio
async def test_the_quiet_channel_drops_a_tool_embed_and_nothing_else():
    channel = Channel()
    quiet = QuietChannel(channel)
    posted = await Common.send_tool_discord_embed(quiet, "text", title="Web")
    await Common.edit_tool_discord_embed(posted, "more", title="Web")
    assert channel.sent == [] and quiet.id == 500
    await quiet.send("plain text")
    await quiet.send(embed=discord.Embed(), file="a.png")
    assert len(channel.sent) == 2

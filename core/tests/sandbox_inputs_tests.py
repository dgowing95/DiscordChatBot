"""Files from the chat handed to a sandbox run: which ones, the caps, what
each side is told, and the download / write halves around that.

Run: pytest core/tests/sandbox_inputs_tests.py
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from classes import sandbox_inputs as si


def _c(name, size=10, ref=None):
    return si.Candidate(name, size, ref or name)


# ------------------------------- pure rules -------------------------------

def test_safe_name_keeps_one_plain_path_segment():
    assert si.safe_name("../../etc/passwd") == "passwd"
    assert si.safe_name("C:\\Users\\me\\my data (1).csv") == "my_data_1_.csv"
    assert si.safe_name("...") == "file"


def test_unique_names_numbers_collisions_before_the_extension():
    assert si.unique_names(["a.png", "A.png", "a.png", "notes"]) == [
        "a.png", "A-2.png", "a-3.png", "notes"]


def test_given_files_are_always_taken_and_named_ones_found_newest_first():
    older = _c("generated-image.png", ref="old")
    newer = _c("generated-image.png", ref="new")
    chosen, missing, skipped = si.select(
        given=[_c("data.csv")], recent=[newer, older],
        requested=["generated-image.png", "data.csv", "nope.txt", ""], max_bytes=1000)
    assert [c.ref for c in chosen] == ["data.csv", "new"]
    assert missing == ["nope.txt"]
    assert skipped == []


def test_names_match_case_blind_and_by_their_safe_form():
    chosen, missing, _ = si.select([], [_c("My Data.CSV")], ["my data.csv", "My_Data.CSV"], 1000)
    assert [c.name for c in chosen] == ["My Data.CSV"]
    assert missing == []


def test_caps_on_count_and_total_size():
    many = [_c(f"f{i}.txt", ref=i) for i in range(si.MAX_INPUT_FILES + 1)]
    chosen, _, skipped = si.select(many, [], [], max_bytes=10_000)
    assert len(chosen) == si.MAX_INPUT_FILES
    assert skipped == [f"f{si.MAX_INPUT_FILES}.txt (more than {si.MAX_INPUT_FILES} files)"]

    chosen, _, skipped = si.select([_c("big.bin", 900), _c("small.txt", 50), _c("more.bin", 100)],
                                   [], [], max_bytes=1000)
    assert [c.name for c in chosen] == ["big.bin", "small.txt"]
    assert skipped == ["more.bin (over the 1 KB limit)"]


def test_max_input_bytes(monkeypatch):
    monkeypatch.delenv("SANDBOX_INPUT_MAX_BYTES", raising=False)
    assert si.max_input_bytes() == si.DEFAULT_MAX_BYTES
    monkeypatch.setenv("SANDBOX_INPUT_MAX_BYTES", "500")
    assert si.max_input_bytes() == 500
    monkeypatch.setenv("SANDBOX_INPUT_MAX_BYTES", "-1")
    assert si.max_input_bytes() == si.DEFAULT_MAX_BYTES


def test_what_each_side_is_told():
    assert si.task_preamble([], []) == ""
    assert si.task_preamble([("a.csv", 12_000)], ["b.png (not found in the chat)"]) == (
        "Files for this task are already in in/: a.csv (12 KB).\n"
        "Asked for but not provided: b.png (not found in the chat).\n\n")
    assert si.tool_note([], []) == ""
    assert si.tool_note(["a.csv"], ["b.png (x)"]) == (
        "\n\nFiles given to the sandbox: a.csv. Not given (tell the user): b.png (x).")


def test_attachment_note():
    assert si.attachment_note([]) == ""
    assert si.attachment_note([("a.csv", 999), ("b.png", 2_500_000)]) == (
        "\n[Attached: a.csv (999 B), b.png (2.5 MB)]")


# ------------------------- download (tool_functions) -------------------------

def _attachment(name, data=b"x", size=None, fail=False):
    a = MagicMock()
    a.filename = name
    a.size = len(data) if size is None else size
    a.read = AsyncMock(side_effect=RuntimeError("cdn down") if fail else None, return_value=data)
    return a


class _Channel:
    def __init__(self, messages=(), fetched=None):
        self._messages = list(messages)
        self.fetch_message = AsyncMock(return_value=fetched)

    def history(self, limit):
        async def _gen():
            for m in self._messages[:limit]:
                yield m
        return _gen()


@pytest.mark.asyncio
async def test_input_files_take_the_trigger_its_reply_and_named_ones():
    from classes.tool_functions import _sandbox_input_files

    reply = SimpleNamespace(attachments=[_attachment("photo.png", b"P")])
    trigger = SimpleNamespace(attachments=[_attachment("data.csv", b"C")],
                              reference=SimpleNamespace(message_id=5, resolved=reply))
    bot_image = SimpleNamespace(attachments=[_attachment("generated-image.png", b"G")])
    wrapper = SimpleNamespace(context={"original_message": trigger,
                                       "channel": _Channel([bot_image])})

    inputs, problems, names = await _sandbox_input_files(
        wrapper, ["generated-image.png", "missing.txt"])
    assert inputs == [("data.csv", b"C"), ("photo.png", b"P"), ("generated-image.png", b"G")]
    assert names == ["data.csv", "photo.png", "generated-image.png"]
    assert problems == ["missing.txt (not found in the chat)"]


@pytest.mark.asyncio
async def test_input_files_fetch_an_unresolved_reply_and_report_failed_downloads():
    from classes.tool_functions import _sandbox_input_files

    fetched = SimpleNamespace(attachments=[_attachment("broken.zip", fail=True)])
    trigger = SimpleNamespace(attachments=[],
                              reference=SimpleNamespace(message_id=5, resolved=None))
    channel = _Channel(fetched=fetched)
    wrapper = SimpleNamespace(context={"original_message": trigger, "channel": channel})

    inputs, problems, names = await _sandbox_input_files(wrapper, None)
    channel.fetch_message.assert_awaited_once_with(5)
    assert inputs == [] and names == []
    assert problems == ["broken.zip (could not be downloaded)"]


@pytest.mark.asyncio
async def test_no_message_and_no_names_gives_nothing():
    from classes.tool_functions import _sandbox_input_files

    wrapper = SimpleNamespace(context={"original_message": None, "channel": _Channel()})
    assert await _sandbox_input_files(wrapper, []) == ([], [], [])


# --------------------------- write (sandbox_agent) ---------------------------

def _session(mkdir_ok=True, write_error=None):
    session = MagicMock()
    session.exec = AsyncMock(return_value=MagicMock(ok=MagicMock(return_value=mkdir_ok)))
    session.write = AsyncMock(side_effect=write_error)
    return session


@pytest.mark.asyncio
async def test_write_inputs_puts_files_beside_out_and_names_them():
    from classes.sandbox_agent import _write_inputs

    session = _session()
    preamble, given, problems = await _write_inputs(
        session, "/workspace/out", [("a.csv", b"123")], ["b (x)"])
    session.exec.assert_awaited_once_with("mkdir", "-p", "/workspace/in", shell=False)
    path, stream = session.write.await_args.args
    assert str(path) == "/workspace/in/a.csv" and stream.read() == b"123"
    assert preamble == ("Files for this task are already in in/: a.csv (3 B).\n"
                        "Asked for but not provided: b (x).\n\n")
    assert given == ["a.csv"] and problems == ["b (x)"]


@pytest.mark.asyncio
@pytest.mark.parametrize("out_dir,session", [
    (None, _session()), ("/workspace/out", _session(mkdir_ok=False)),
    ("/workspace/out", _session(write_error=RuntimeError("denied")))])
async def test_a_file_that_cannot_be_written_is_named_as_not_provided(out_dir, session):
    from classes.sandbox_agent import _write_inputs

    preamble, given, problems = await _write_inputs(session, out_dir, [("a.csv", b"1")], [])
    assert preamble == "Asked for but not provided: a.csv (could not be copied in).\n\n"
    # What the outer model is told comes from these, so a failed write is
    # never reported as given.
    assert given == [] and problems == ["a.csv (could not be copied in)"]

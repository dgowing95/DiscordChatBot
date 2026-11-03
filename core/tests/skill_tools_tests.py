"""The brain's skill tools, the reflection and the /skills commands.

The store is the in-memory one from skill_store_tests; the workspace is a
fake session that records what was written.
"""
import io
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agents.tool_context import ToolContext

from classes import skill_commands, skill_reflection, skill_tools, tool_embeds
from classes.skill_store import SkillStore
from skill_store_tests import FakeRedis

GUILD = 11


class Result:
    def __init__(self, ok=True, stdout=b""):
        self._ok, self.stdout = ok, stdout

    def ok(self):
        return self._ok


class FakeSession:
    def __init__(self, files=None):
        self.written = {}
        self.commands = []
        self.files = files or {}

    async def exec(self, *args, shell=False):
        self.commands.append(args)
        if args[0] == "cat":
            path = args[-1]
            return Result(path in self.files, self.files.get(path, "").encode())
        return Result()

    async def write(self, path, data):
        self.written[str(path)] = data.read().decode()


@pytest.fixture
def store(monkeypatch):
    shared = SkillStore(FakeRedis())
    monkeypatch.setattr(skill_tools, "SkillStore", lambda: shared)
    monkeypatch.setattr(skill_commands, "SkillStore", lambda: shared)
    monkeypatch.setattr("classes.skill_store.SkillStore", lambda: shared)  # imported inside functions
    return shared


@pytest.fixture
def allowed():
    with patch.object(skill_tools, "check_web_request", AsyncMock(return_value=(True, ""))), \
         patch.object(skill_commands, "check_web_request", AsyncMock(return_value=(True, ""))):
        yield


@pytest.fixture
def ctx():
    channel = MagicMock(id=77)
    channel.send = AsyncMock()
    return {"guild_id": GUILD, "channel": channel, "user_id": 5, "session": FakeSession(),
            "workspace_root": "/workspace", "job_dir": "jobs/c77-1-abc"}


async def call(tool, context, **args):
    return await tool.on_invoke_tool(
        ToolContext(context=context, tool_name=tool.name, tool_call_id="t1", tool_arguments="{}"),
        json.dumps(args))


def titles(channel):
    return [c.kwargs["embed"].title for c in channel.send.call_args_list]


@pytest.mark.asyncio
async def test_save_skill_stores_posts_an_embed_and_marks_the_call(store, ctx, allowed):
    ctx["session"].files["/workspace/jobs/c77-1-abc/chart.py"] = "print('hi')"
    out = await call(skill_tools.save_skill, ctx, name="Weekly Chart", description="Charts tracker embeds",
                     steps="1. save the embeds\n2. run chart.py", files=["chart.py"])
    assert "created" in out and "v1" in out
    skill = await store.get(GUILD, "weekly-chart")
    assert skill["files"] == {"scripts/chart.py": "print('hi')"} and skill["updated_by"] == 5
    assert ctx["skill_saved"] is True
    assert titles(ctx["channel"]) == [tool_embeds.SKILL_SAVED]
    # No files given: the update keeps the script.
    out = await call(skill_tools.save_skill, ctx, name="weekly-chart", description="Charts tracker embeds",
                     steps="1. new steps", note="simpler")
    assert "updated" in out
    assert titles(ctx["channel"])[-1] == tool_embeds.SKILL_UPDATED
    assert (await store.get(GUILD, "weekly-chart"))["files"] == {"scripts/chart.py": "print('hi')"}
    # An empty list removes them.
    await call(skill_tools.save_skill, ctx, name="weekly-chart", description="Charts tracker embeds",
               steps="1. new steps", files=[], note="no script")
    assert (await store.get(GUILD, "weekly-chart"))["files"] == {}


@pytest.mark.asyncio
async def test_save_skill_unchanged_posts_nothing(store, ctx, allowed):
    args = dict(name="a-skill", description="d", steps="do it")
    await call(skill_tools.save_skill, ctx, **args)
    ctx["channel"].send.reset_mock()
    assert "already saved" in await call(skill_tools.save_skill, ctx, **args)
    ctx["channel"].send.assert_not_called()


@pytest.mark.asyncio
async def test_save_skill_tells_the_brain_what_to_fix(store, ctx, allowed):
    out = await call(skill_tools.save_skill, ctx, name="x", description="d", steps="do it")
    assert out.startswith("Not saved") and "name" in out
    out = await call(skill_tools.save_skill, ctx, name="good-name", description="d", steps="s",
                     files=["missing.py", "../escape.py"])
    assert "missing.py: not found" in out and "escape.py" in out
    assert await store.list(GUILD) == []
    ctx["channel"].send.assert_not_called()


@pytest.mark.asyncio
async def test_save_skill_obeys_the_content_guard(store, ctx):
    with patch.object(skill_tools, "check_web_request", AsyncMock(return_value=(False, "flagged"))):
        out = await call(skill_tools.save_skill, ctx, name="bad-skill", description="d", steps="s")
    assert "safety guard" in out and await store.list(GUILD) == []


@pytest.mark.asyncio
async def test_use_skill_writes_it_fresh_and_records_the_use(store, ctx, allowed):
    await store.save(GUILD, "weekly-chart", "Charts things", "1. do it", {"scripts/run.py": "x=1"},
                     author_id=1)
    out = await call(skill_tools.use_skill, ctx, name="weekly-chart")
    assert "1. do it" in out and "scripts/run.py" in out
    written = ctx["session"].written
    assert set(written) == {s for s in written if "/workspace/skills/.weekly-chart-" in s}
    assert any(p.endswith("/SKILL.md") for p in written) and any(p.endswith("/scripts/run.py") for p in written)
    assert "name: weekly-chart" in next(v for p, v in written.items() if p.endswith("SKILL.md"))
    script = next(c for c in ctx["session"].commands if c[0] == "sh")
    assert script[4].startswith("/workspace/skills/.weekly-chart-") and script[5] == "/workspace/skills/weekly-chart"
    assert "chmod -R a-w" in script[2]
    skill = await store.get(GUILD, "weekly-chart")
    assert skill["uses"] == 1 and skill["last_channel_id"] == 77
    assert ctx["skills_used"] == ["weekly-chart"]
    assert titles(ctx["channel"]) == [tool_embeds.SKILL_USED]


@pytest.mark.asyncio
async def test_use_skill_still_returns_the_steps_when_the_copy_fails(store, ctx, allowed):
    await store.save(GUILD, "weekly-chart", "d", "1. do it", author_id=1)
    ctx["session"].write = AsyncMock(side_effect=RuntimeError("no disk"))
    out = await call(skill_tools.use_skill, ctx, name="weekly-chart")
    assert "1. do it" in out and "could not be copied" in out


@pytest.mark.asyncio
async def test_use_skill_without_files_lists_only_skill_md(store, ctx, allowed):
    await store.save(GUILD, "weekly-chart", "d", "1. do it", author_id=1)
    out = await call(skill_tools.use_skill, ctx, name="weekly-chart")
    assert out.rstrip().endswith("SKILL.md") and "none" not in out


@pytest.mark.asyncio
async def test_use_skill_unknown_name_offers_the_closest(store, ctx, allowed):
    await store.save(GUILD, "weekly-chart", "d", "s", author_id=1)
    assert "Closest: weekly-chart" in await call(skill_tools.use_skill, ctx, name="weekly-chrat")
    assert "no skills yet" in await call(skill_tools.use_skill, {**ctx, "guild_id": 99}, name="x")


@pytest.mark.asyncio
async def test_search_skills(store, ctx, allowed):
    await store.save(GUILD, "weekly-chart", "Chart tracker embeds", "s", author_id=1)
    assert "weekly-chart" in await call(skill_tools.search_skills, ctx, query="tracker chart")
    assert "No saved skill" in await call(skill_tools.search_skills, ctx, query="zebra")


@pytest.mark.asyncio
async def test_no_guild_is_refused(store, ctx, allowed):
    out = await call(skill_tools.use_skill, {**ctx, "guild_id": None}, name="x")
    assert "server" in out


# ----------------------------- the brain wiring -----------------------------

def test_brain_tools_include_skills_unless_switched_off(monkeypatch):
    from classes.brain_tools import brain_tools
    names = [t.name for t in brain_tools()]
    assert {"use_skill", "save_skill", "search_skills"} <= set(names) and names[-1] == "report"
    monkeypatch.setenv("SKILLS_ENABLED", "0")
    assert not {"use_skill", "save_skill", "search_skills"} & {t.name for t in brain_tools()}


def test_skills_instructions_are_small_and_off_when_disabled(monkeypatch):
    from classes import brain
    assert len(brain.SKILLS_INSTRUCTIONS) <= 400, len(brain.SKILLS_INSTRUCTIONS)
    assert brain.instructions().endswith(brain.SKILLS_INSTRUCTIONS)
    monkeypatch.setenv("SKILLS_ENABLED", "0")
    assert brain.instructions() == brain.BRAIN_INSTRUCTIONS


def test_the_index_goes_in_the_task_between_the_request_and_the_folder():
    from classes import brain
    task = brain.build_task({}, "make a chart", job_dir="jobs/x", skills_index="Skills saved: - a-skill: d",
                            datetime="now")
    assert task.index("Request:") < task.index("Skills saved") < task.index("Your folder") < task.index("Current datetime")
    assert "Skills saved" not in brain.build_task({}, "make a chart", job_dir="jobs/x")


@pytest.mark.asyncio
async def test_a_done_call_that_used_a_skill_counts_a_success(store):
    from classes import brain
    await store.save(GUILD, "weekly-chart", "d", "s", author_id=1)
    with patch("classes.skill_store.SkillStore", lambda: store):
        await brain._record_skill_successes({"guild_id": GUILD, "skills_used": ["weekly-chart"]})
        await brain._record_skill_successes({"guild_id": GUILD, "skills_used": []})
    assert (await store.get(GUILD, "weekly-chart"))["successes"] == 1


@pytest.mark.asyncio
async def test_the_skills_index_lists_the_servers_skills(store):
    from classes import brain
    await store.save(GUILD, "weekly-chart", "Charts things", "s", author_id=1)
    with patch("classes.skill_store.SkillStore", lambda: store):
        text = await brain._skills_index(GUILD)
        assert text.startswith("Skills saved in this server") and "- weekly-chart: Charts things" in text
        assert await brain._skills_index(None) == ""
        assert await brain._skills_index(999) == ""


# ------------------------------- reflection --------------------------------

def test_should_reflect_rules():
    ok = dict(status="done", tool_calls=6, skill_saved=False, env={})
    assert skill_reflection.should_reflect(**ok)
    assert skill_reflection.should_reflect(**{**ok, "status": "partial"})
    assert not skill_reflection.should_reflect(**{**ok, "status": "failed"})
    assert not skill_reflection.should_reflect(**{**ok, "tool_calls": 5})
    assert not skill_reflection.should_reflect(**{**ok, "skill_saved": True})
    assert not skill_reflection.should_reflect(**{**ok, "env": {"SKILLS_REFLECTION_ENABLED": "0"}})
    assert not skill_reflection.should_reflect(**{**ok, "env": {"SKILLS_ENABLED": "0"}})
    assert skill_reflection.should_reflect(**{**ok, "tool_calls": 3, "env": {"SKILLS_REFLECTION_MIN_TOOL_CALLS": "3"}})


def _item(kind, **raw):
    return SimpleNamespace(type=kind, raw_item=SimpleNamespace(**raw) if raw else None, output=raw.get("output", ""))


def test_condense_shows_commands_and_output_tails():
    items = [_item("tool_call_item", name="exec_command", arguments=json.dumps({"cmd": "python chart.py"})),
             SimpleNamespace(type="tool_call_output_item", raw_item=None, output="x" * 1000 + "done"),
             _item("tool_call_item", name="web_search", arguments=json.dumps({"search_request": "q"})),
             SimpleNamespace(type="message_output_item")]
    text = skill_reflection.condense(items)
    lines = text.split("\n")
    assert lines[0] == "$ python chart.py"
    assert lines[1].startswith("  → …") and lines[1].endswith("done") and len(lines[1]) < 330
    assert lines[2].startswith("web_search(")


def test_condense_keeps_the_newest_when_long():
    items = []
    for i in range(200):
        items.append(_item("tool_call_item", name="exec_command", arguments=json.dumps({"cmd": f"step {i}"})))
    text = skill_reflection.condense(items, limit=500)
    assert text.startswith("…") and "step 199" in text and "step 0\n" not in text and len(text) <= 505


def test_parse_decision():
    good = '<think>hm</think>{"action": "create", "name": "a-skill", "description": "d", "body": "b"}'
    assert skill_reflection.parse_decision(good) == {"action": "create", "name": "a-skill", "description": "d",
                                                     "body": "b", "note": ""}
    assert skill_reflection.parse_decision('{"action": "none"}') == {"action": "none"}
    for bad in ("", "no json", '{"action": "delete"}', '{"action": ', "[1]"):
        assert skill_reflection.parse_decision(bad) == {"action": "none"}


def test_the_prompt_stays_small_and_shows_the_json():
    assert len(skill_reflection.REFLECTION_PROMPT) <= 1000, len(skill_reflection.REFLECTION_PROMPT)
    assert '"action": "none"' in skill_reflection.REFLECTION_PROMPT


def _review(channel):
    return skill_reflection.build_review(
        request="chart it", status="done", facts=["made it"], transcript="$ python x", skills_used=[],
        used_bodies={}, existing_names=["other-skill"], channel=channel, guild_id=GUILD, user_id=5,
        job_id="c77-1-abc")


def test_review_messages_carry_the_job():
    msgs = skill_reflection.review_messages(_review(None))
    assert msgs[0]["role"] == "system" and "Request: chart it" in msgs[1]["content"]
    assert "other-skill" in msgs[1]["content"] and "$ python x" in msgs[1]["content"]


@pytest.mark.asyncio
async def test_reflect_creates_a_skill_and_posts_the_embed(store, ctx, allowed):
    answer = json.dumps({"action": "create", "name": "chart-it", "description": "Charts", "body": "1. do"})
    with patch.object(skill_reflection._llm, "complete", AsyncMock(return_value=answer)):
        assert await skill_reflection.reflect(_review(ctx["channel"])) == "create"
    skill = await store.get(GUILD, "chart-it")
    assert skill["source"] == "reflection" and skill["updated_by"] == 5
    embed = ctx["channel"].send.call_args.kwargs["embed"]
    assert embed.title == tool_embeds.SKILL_SAVED and embed.footer.text == tool_embeds.SKILL_FOOTER_REFLECTION


@pytest.mark.asyncio
async def test_reflect_none_or_failure_posts_nothing(store, ctx, allowed):
    for llm in (AsyncMock(return_value='{"action": "none"}'), AsyncMock(side_effect=RuntimeError("down")),
                AsyncMock(return_value='{"action": "create", "name": "x", "description": "d", "body": "b"}')):
        with patch.object(skill_reflection._llm, "complete", llm):
            assert await skill_reflection.reflect(_review(ctx["channel"])) in ("none", "failed")
    ctx["channel"].send.assert_not_called()
    assert await store.list(GUILD) == []


@pytest.mark.asyncio
async def test_reflect_update_of_an_existing_name(store, ctx, allowed):
    await store.save(GUILD, "chart-it", "Charts", "1. old", {"scripts/run.py": "x=1"}, author_id=1)
    answer = json.dumps({"action": "create", "name": "chart-it", "description": "Charts", "body": "1. new",
                         "note": "faster"})
    with patch.object(skill_reflection._llm, "complete", AsyncMock(return_value=answer)):
        assert await skill_reflection.reflect(_review(ctx["channel"])) == "update"
    updated = await store.get(GUILD, "chart-it")
    assert updated["body"] == "1. new" and updated["files"] == {"scripts/run.py": "x=1"}
    assert ctx["channel"].send.call_args.kwargs["embed"].title == tool_embeds.SKILL_UPDATED


@pytest.mark.asyncio
async def test_a_queued_review_is_left_on_the_context_only_for_long_jobs(store, ctx):
    from classes import brain, brain_result
    out = brain_result.BrainReport(status="done", facts=["f"])
    result = SimpleNamespace(new_items=[])
    base = {"guild_id": GUILD, "channel": ctx["channel"], "user_id": 5, "brain_request": "r",
            "job_dir": "jobs/x", "skills_used": []}
    with patch("classes.skill_store.SkillStore", lambda: store):
        short, long_ = dict(base), dict(base)
        await brain._queue_skill_review(short, out, SimpleNamespace(tool_calls=2), "t", result)
        await brain._queue_skill_review(long_, out, SimpleNamespace(tool_calls=9), "t", result)
        saved = dict(base, skill_saved=True)
        await brain._queue_skill_review(saved, out, SimpleNamespace(tool_calls=9), "t", result)
    assert "skill_reviews" not in short and "skill_reviews" not in saved
    assert long_["skill_reviews"][0]["request"] == "r"


# -------------------------------- /skills ---------------------------------

def _interaction(channel):
    ctx = MagicMock()
    ctx.guild_id, ctx.channel, ctx.user = GUILD, channel, SimpleNamespace(id=9)
    ctx.response.send_message = AsyncMock()
    ctx.response.send_modal = AsyncMock()
    ctx.guild = SimpleNamespace(filesize_limit=10 * 1024 * 1024)
    return ctx


def _commands():
    tree = MagicMock()
    skill_commands.register_skill_commands(tree)
    group = tree.add_command.call_args.args[0]
    return {c.name: c.callback for c in group.commands}


def test_the_group_has_every_command_and_is_guild_only():
    tree = MagicMock()
    skill_commands.register_skill_commands(tree)
    group = tree.add_command.call_args.args[0]
    assert group.name == "skills" and group.guild_only
    assert {c.name for c in group.commands} == {"list", "show", "history", "edit", "delete", "restore",
                                                "revert", "export"}


@pytest.mark.asyncio
async def test_list_show_and_history(store, ctx, allowed):
    cmds = _commands()
    one = _interaction(ctx["channel"])
    await cmds["list"](one)
    assert "No skills" in one.response.send_message.call_args.args[0]
    await store.save(GUILD, "weekly-chart", "Charts things", "1. do it", author_id=3)
    await store.save(GUILD, "weekly-chart", "Charts things", "1. do it better", author_id=4, note="better")
    two = _interaction(ctx["channel"])
    await cmds["list"](two)
    assert "weekly-chart" in two.response.send_message.call_args.kwargs["embed"].description
    three = _interaction(ctx["channel"])
    await cmds["show"](three, "Weekly Chart", None)
    assert "1. do it better" in three.response.send_message.call_args.args[0]
    four = _interaction(ctx["channel"])
    await cmds["show"](four, "weekly-chart", 1)
    assert "1. do it\n" in four.response.send_message.call_args.args[0]
    five = _interaction(ctx["channel"])
    await cmds["history"](five, "weekly-chart")
    text = five.response.send_message.call_args.kwargs["embed"].description
    assert "v2" in text and "(current)" in text and "<@4>" in text and "better" in text
    six = _interaction(ctx["channel"])
    await cmds["show"](six, "nope", None)
    assert "No such" in six.response.send_message.call_args.args[0]


@pytest.mark.asyncio
async def test_a_long_skill_is_shown_as_a_file(store, ctx, allowed):
    await store.save(GUILD, "long-one", "d", "x" * 5000, author_id=1)
    one = _interaction(ctx["channel"])
    await _commands()["show"](one, "long-one", None)
    assert one.response.send_message.call_args.kwargs["file"].filename == "long-one.md"


@pytest.mark.asyncio
async def test_delete_restore_revert_each_post_their_embed(store, ctx, allowed):
    cmds = _commands()
    await store.save(GUILD, "weekly-chart", "d", "first", author_id=1)
    await store.save(GUILD, "weekly-chart", "d", "second", author_id=1)
    seen = []
    for name, args, title in (("revert", ("weekly-chart", 1), tool_embeds.SKILL_REVERTED),
                              ("delete", ("weekly-chart",), tool_embeds.SKILL_DELETED),
                              ("restore", ("weekly-chart",), tool_embeds.SKILL_RESTORED)):
        interaction = _interaction(ctx["channel"])
        await cmds[name](interaction, *args)
        assert interaction.response.send_message.call_args.args == ("Done.",)
        seen.append(titles(ctx["channel"])[-1])
        assert seen[-1] == title
    assert (await store.get(GUILD, "weekly-chart"))["body"] == "first"
    bad = _interaction(ctx["channel"])
    await cmds["delete"](bad, "nope")
    assert "No skill named" in bad.response.send_message.call_args.args[0]
    assert len(titles(ctx["channel"])) == 3


@pytest.mark.asyncio
async def test_edit_opens_a_filled_modal_and_saves_with_a_public_embed(store, ctx, allowed):
    await store.save(GUILD, "weekly-chart", "Charts", "1. old", {"scripts/a.py": "x"}, author_id=1)
    one = _interaction(ctx["channel"])
    await _commands()["edit"](one, "weekly-chart")
    modal = one.response.send_modal.call_args.args[0]
    assert modal.steps.default == "1. old" and modal.description.default == "Charts"
    modal.description._value = "Charts better"
    modal.steps._value = "1. new"
    modal.note._value = "tidier"
    submit = _interaction(ctx["channel"])
    await modal.on_submit(submit)
    skill = await store.get(GUILD, "weekly-chart")
    assert skill["body"] == "1. new" and skill["files"] == {"scripts/a.py": "x"} and skill["source"] == "command"
    assert titles(ctx["channel"]) == [tool_embeds.SKILL_UPDATED]
    again = _interaction(ctx["channel"])
    await modal.on_submit(again)
    assert "Nothing changed" in again.response.send_message.call_args.args[0]


@pytest.mark.asyncio
async def test_edit_refuses_a_body_too_big_for_the_box(store, ctx, allowed):
    await store.save(GUILD, "huge-one", "d", "x" * 5000, author_id=1)
    one = _interaction(ctx["channel"])
    await _commands()["edit"](one, "huge-one")
    one.response.send_modal.assert_not_called()
    assert "too long" in one.response.send_message.call_args.args[0]


@pytest.mark.asyncio
async def test_export_sends_a_zip(store, ctx, allowed):
    await store.save(GUILD, "weekly-chart", "d", "s", author_id=1)
    one = _interaction(ctx["channel"])
    await _commands()["export"](one)
    assert one.response.send_message.call_args.kwargs["file"].filename == "skills.zip"
    small = _interaction(ctx["channel"])
    small.guild = SimpleNamespace(filesize_limit=10)
    await _commands()["export"](small)
    assert "file" not in small.response.send_message.call_args.kwargs
    assert "more than this server can upload" in small.response.send_message.call_args.args[0]


@pytest.mark.asyncio
async def test_history_of_many_versions_fits_discord(store, ctx, allowed):
    for i in range(50):
        await store.save(GUILD, "weekly-chart", "d", f"step {i}", author_id=123456789012345678,
                         note="a fairly long note about what changed here")
    one = _interaction(ctx["channel"])
    await _commands()["history"](one, "weekly-chart")
    assert len(one.response.send_message.call_args.kwargs["embed"].description) <= 4096


@pytest.mark.asyncio
async def test_list_reads_every_skill_in_one_round_trip(store, ctx, allowed):
    for name in ("one-skill", "two-skill", "three-skill"):
        await store.save(GUILD, name, "d", "s", author_id=1)
    store.redis.mgets = 0
    assert len(await store.list(GUILD)) == 3 and store.redis.mgets == 1


@pytest.mark.asyncio
async def test_autocomplete_offers_live_names_and_restore_offers_deleted(store, ctx, allowed):
    await store.save(GUILD, "weekly-chart", "d", "s", author_id=1)
    await store.save(GUILD, "old-poll", "d", "s", author_id=1)
    await store.delete(GUILD, "old-poll", 1)
    interaction = _interaction(ctx["channel"])
    assert [c.value for c in await skill_commands._live_names(interaction, "week")] == ["weekly-chart"]
    assert [c.value for c in await skill_commands._deleted_names(interaction, "")] == ["old-poll"]

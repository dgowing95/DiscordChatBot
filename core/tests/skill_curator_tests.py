"""The daily skills tidy-up: its pure rules, and passes against the
in-memory Redis from skill_store_tests."""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from classes import skill_curator as sc
from classes import tool_embeds
from classes.skill_store import SkillError, SkillStore, VersionError
from skill_store_tests import FakeRedis

GUILD = 11
BOT = 999
DAY = 86400


def _skill(name, *, version=1, uses=0, last_used_at=None, updated_at=0.0, body="steps", files=None,
           channel=None):
    return {"name": name, "description": f"{name} things", "body": body, "files": files or {},
            "version": version, "uses": uses, "successes": 0, "last_used_at": last_used_at,
            "updated_at": updated_at, "last_channel_id": channel}


# --------------------------------- pure -----------------------------------

def test_stale_means_neither_used_nor_changed_for_the_days():
    now = 100 * DAY
    skills = [_skill("old-one", updated_at=0), _skill("used-lately", updated_at=0, last_used_at=95 * DAY),
              _skill("edited-lately", updated_at=90 * DAY)]
    assert sc.stale(skills, now, 60) == {"old-one"}


def test_the_signature_follows_versions_and_the_stale_set():
    a = [_skill("one-skill"), _skill("two-skill")]
    assert sc.signature(a, set()) == sc.signature(list(reversed(a)), set())
    assert sc.signature(a, set()) != sc.signature([_skill("one-skill", version=2), _skill("two-skill")], set())
    assert sc.signature(a, set()) != sc.signature(a, {"one-skill"})


def test_due():
    assert sc.due(None, "s", 0, 24)
    assert not sc.due({"at": 0, "sig": "old"}, "new", 23 * 3600, 24)
    assert not sc.due({"at": 0, "sig": "same"}, "same", 48 * 3600, 24)
    assert sc.due({"at": 0, "sig": "old"}, "new", 24 * 3600, 24)


def test_skills_text_cuts_long_bodies_and_marks_unused():
    text = sc.skills_text([_skill("long-one", body="x" * 2000), _skill("old-one")], {"old-one"}, 0)
    assert "## old-one [unused]" in text and "x" * sc.BODY_CHARS + "…" in text
    assert "x" * (sc.BODY_CHARS + 1) not in text
    many = [_skill(f"skill-{i:03}", body="y" * 500) for i in range(100)]
    cut = sc.skills_text(many, set(), 0, limit=5000)
    assert len(cut) < 6000 and "more not shown" in cut


def _answer(*changes):
    return json.dumps({"changes": list(changes)})


LIVE = {"weekly-chart", "week-chart", "old-poll", "bar-chart"}


def test_parse_keeps_good_changes():
    text = _answer(
        {"action": "merge", "keep": "weekly-chart", "drop": "Week Chart", "description": "d", "body": "b",
         "note": "n"},
        {"action": "retire", "name": "old-poll", "note": "unused"},
        {"action": "describe", "name": "bar-chart", "description": "better", "note": ""})
    out = sc.parse_changes(text, LIVE, {"old-poll"})
    assert [c["action"] for c in out] == ["merge", "retire", "describe"]
    assert out[0]["drop"] == "week-chart"


@pytest.mark.parametrize("change", [
    {"action": "retire", "name": "bar-chart"},                     # not stale
    {"action": "retire", "name": "nope-skill"},                    # not live
    {"action": "merge", "keep": "bar-chart", "drop": "bar-chart", "description": "d", "body": "b"},
    {"action": "merge", "keep": "bar-chart", "drop": "nope-skill", "description": "d", "body": "b"},
    {"action": "merge", "keep": "bar-chart", "drop": "week-chart", "description": "d", "body": ""},
    {"action": "describe", "name": "bar-chart", "description": ""},
    {"action": "delete-everything"},
    "not a dict",
])
def test_parse_drops_what_is_not_allowed(change):
    assert sc.parse_changes(_answer(change), LIVE, {"old-poll"}) == []


def test_parse_names_each_skill_once_and_stops_at_the_limit():
    text = _answer({"action": "describe", "name": "bar-chart", "description": "a"},
                   {"action": "merge", "keep": "bar-chart", "drop": "week-chart", "description": "d",
                    "body": "b"},
                   {"action": "describe", "name": "week-chart", "description": "b"},
                   {"action": "describe", "name": "weekly-chart", "description": "c"},
                   {"action": "retire", "name": "old-poll"})
    out = sc.parse_changes(text, LIVE, {"old-poll"})
    assert [c.get("name") for c in out] == ["bar-chart", "week-chart", "weekly-chart"]


@pytest.mark.parametrize("text", ["", "no json here", "{broken", '{"changes": "all"}',
                                  "<think>hmm {\"changes\": [1]}</think>{\"changes\": []}"])
def test_parse_unreadable_is_no_change(text):
    assert sc.parse_changes(text, LIVE, set()) == []


def test_the_prompt_stays_small_and_shows_the_shape():
    assert len(sc.CURATOR_PROMPT) <= 1000, len(sc.CURATOR_PROMPT)
    assert '{"changes": []}' in sc.CURATOR_PROMPT


# -------------------------------- passes ----------------------------------

class Clock:
    def __init__(self, t=100 * DAY):
        self.t = t

    def __call__(self):
        return self.t


class FakeLLM:
    def __init__(self, answer="", side_effect=None):
        self.answer, self.side_effect, self.calls = answer, side_effect, 0

    async def complete(self, messages, **kwargs):
        self.calls += 1
        self.messages = messages
        if self.side_effect:
            result = await self.side_effect()
            if isinstance(result, Exception):
                raise result
        return self.answer


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def store(clock):
    return SkillStore(FakeRedis(), clock=clock)


@pytest.fixture
def channels():
    made = {}

    def channel_of(cid):
        if cid == 404:
            return None
        if cid not in made:
            made[cid] = MagicMock(id=cid)
            made[cid].send = AsyncMock()
        return made[cid]
    return made, channel_of


@pytest.fixture(autouse=True)
def allowed():
    with patch("classes.content_guard.check_web_request", AsyncMock(return_value=(True, ""))):
        yield


def embeds(channel):
    return [(c.kwargs["embed"].title, c.kwargs["embed"].footer.text) for c in channel.send.call_args_list]


async def _seed(store, clock, *, old=("old-poll",), channel=77):
    """Five skills; `old` ones saved 90 days ago and never used."""
    now = clock.t
    for name in ("weekly-chart", "week-chart", "bar-chart", "pie-chart", "old-poll"):
        clock.t = now - 90 * DAY if name in old else now
        await store.save(GUILD, name, f"{name} things", f"steps for {name}",
                         {f"scripts/{name}.py": "x"}, author_id=1, channel_id=channel)
    clock.t = now


def _curator(store, channels, llm, clock):
    return sc.Curator(store, channels[1], BOT, llm=llm, clock=clock)


@pytest.mark.asyncio
async def test_a_server_with_few_skills_gets_no_pass(store, channels, clock):
    await store.save(GUILD, "one-skill", "d", "s", author_id=1)
    llm = FakeLLM(_answer())
    assert await _curator(store, channels, llm, clock).curate(GUILD) == "few"
    assert llm.calls == 0


@pytest.mark.asyncio
async def test_the_switch_turns_it_off(store, channels, clock, monkeypatch):
    monkeypatch.setenv("SKILLS_CURATOR_ENABLED", "0")
    await _seed(store, clock)
    assert await _curator(store, channels, FakeLLM(_answer()), clock).curate(GUILD) == "off"


@pytest.mark.asyncio
async def test_merge_keeps_one_with_both_files_and_retires_the_other(store, channels, clock):
    await _seed(store, clock)
    llm = FakeLLM(_answer({"action": "merge", "keep": "weekly-chart", "drop": "week-chart",
                           "description": "Weekly chart of anything", "body": "combined steps",
                           "note": "merged week-chart"}))
    assert await _curator(store, channels, llm, clock).curate(GUILD) == "1"
    kept = await store.get(GUILD, "weekly-chart")
    assert kept["body"] == "combined steps" and kept["version"] == 2 and kept["source"] == "curator"
    assert kept["files"] == {"scripts/weekly-chart.py": "x", "scripts/week-chart.py": "x"}
    assert kept["updated_by"] == BOT
    assert await store.get(GUILD, "week-chart") is None
    assert (await store.get_any(GUILD, "week-chart"))["deleted"]
    assert embeds(channels[0][77]) == [(tool_embeds.SKILL_UPDATED, tool_embeds.SKILL_FOOTER_CURATOR),
                                       (tool_embeds.SKILL_RETIRED, tool_embeds.SKILL_FOOTER_CURATOR)]
    # What the model read: every skill, the old one marked.
    assert "## old-poll [unused]" in llm.messages[1]["content"]


@pytest.mark.asyncio
async def test_merge_over_the_file_limit_keeps_only_the_kept_files(store, channels, clock, monkeypatch):
    monkeypatch.setenv("SKILLS_MAX_FILES", "1")
    await _seed(store, clock)
    llm = FakeLLM(_answer({"action": "merge", "keep": "weekly-chart", "drop": "week-chart",
                           "description": "d", "body": "b"}))
    assert await _curator(store, channels, llm, clock).curate(GUILD) == "1"
    assert (await store.get(GUILD, "weekly-chart"))["files"] == {"scripts/weekly-chart.py": "x"}


@pytest.mark.asyncio
async def test_retire_and_describe(store, channels, clock):
    await _seed(store, clock)
    llm = FakeLLM(_answer({"action": "retire", "name": "old-poll", "note": ""},
                          {"action": "retire", "name": "bar-chart"},  # not stale: dropped
                          {"action": "describe", "name": "pie-chart", "description": "Pie chart of shares",
                           "note": "clearer"}))
    assert await _curator(store, channels, llm, clock).curate(GUILD) == "2"
    assert await store.get(GUILD, "old-poll") is None and await store.get(GUILD, "bar-chart")
    pie = await store.get(GUILD, "pie-chart")
    assert pie["description"] == "Pie chart of shares" and pie["body"] == "steps for pie-chart"
    assert pie["files"] == {"scripts/pie-chart.py": "x"}
    lines = [c.kwargs["embed"].description for c in channels[0][77].send.call_args_list]
    assert "not used in 60 days" in lines[0] and "/skills restore" in lines[0]
    assert "v2: clearer" in lines[1]


@pytest.mark.asyncio
async def test_a_persons_edit_meanwhile_wins(store, channels, clock):
    await _seed(store, clock)

    async def person_edits():
        await store.save(GUILD, "pie-chart", "pie-chart things", "a person's steps", author_id=5)
    llm = FakeLLM(_answer({"action": "describe", "name": "pie-chart", "description": "machine words"}),
                  side_effect=person_edits)
    assert await _curator(store, channels, llm, clock).curate(GUILD) == "0"
    pie = await store.get(GUILD, "pie-chart")
    assert pie["body"] == "a person's steps" and pie["description"] == "pie-chart things"
    assert 77 not in channels[0] or not channels[0][77].send.called


@pytest.mark.asyncio
async def test_a_merge_whose_drop_changed_keeps_both(store, channels, clock):
    await _seed(store, clock)

    async def person_edits():
        await store.save(GUILD, "week-chart", "d", "a person's steps", author_id=5)
    llm = FakeLLM(_answer({"action": "merge", "keep": "weekly-chart", "drop": "week-chart",
                           "description": "d", "body": "b"}), side_effect=person_edits)
    assert await _curator(store, channels, llm, clock).curate(GUILD) == "0"
    assert (await store.get(GUILD, "week-chart"))["body"] == "a person's steps"
    assert (await store.get(GUILD, "weekly-chart"))["body"] == "b"  # updated first: nothing lost


@pytest.mark.asyncio
async def test_the_guard_can_refuse_new_text(store, channels, clock):
    await _seed(store, clock)
    llm = FakeLLM(_answer({"action": "describe", "name": "pie-chart", "description": "bad words"}))
    with patch("classes.content_guard.check_web_request", AsyncMock(return_value=(False, "flagged"))):
        assert await _curator(store, channels, llm, clock).curate(GUILD) == "0"
    assert (await store.get(GUILD, "pie-chart"))["version"] == 1


@pytest.mark.asyncio
async def test_a_pass_is_not_repeated_until_something_changes(store, channels, clock):
    await _seed(store, clock, old=())
    llm = FakeLLM(_answer({"action": "describe", "name": "pie-chart", "description": "clearer"}))
    curator = _curator(store, channels, llm, clock)
    assert await curator.curate(GUILD) == "1"
    clock.t += 25 * 3600
    # Its own change made a new version, but the signature was taken after it.
    assert await curator.curate(GUILD) == "not due"
    clock.t += 70 * DAY  # every skill goes stale: the stale set changed
    assert await curator.curate(GUILD) == "1" and llm.calls == 2


@pytest.mark.asyncio
async def test_a_failed_call_waits_for_the_interval(store, channels, clock):
    await _seed(store, clock)

    async def boom():
        return RuntimeError("server down")
    llm = FakeLLM(side_effect=boom)
    curator = _curator(store, channels, llm, clock)
    assert await curator.curate(GUILD) == "failed"
    clock.t += 3600
    assert await curator.curate(GUILD) == "not due"
    clock.t += 24 * 3600
    assert await curator.curate(GUILD) == "failed" and llm.calls == 2
    assert not await store.redis.get(sc._lock_key(GUILD))  # released


@pytest.mark.asyncio
async def test_a_pass_already_running_elsewhere_is_left_alone(store, channels, clock):
    await _seed(store, clock)
    await store.redis.set(sc._lock_key(GUILD), "other-process")
    llm = FakeLLM(_answer())
    assert await _curator(store, channels, llm, clock).curate(GUILD) == "busy"
    assert llm.calls == 0 and await store.redis.get(sc._lock_key(GUILD)) == "other-process"


@pytest.mark.asyncio
async def test_a_gone_channel_skips_the_embed_not_the_change(store, channels, clock):
    await _seed(store, clock, channel=404)
    llm = FakeLLM(_answer({"action": "retire", "name": "old-poll"}))
    assert await _curator(store, channels, llm, clock).curate(GUILD) == "1"
    assert await store.get(GUILD, "old-poll") is None


@pytest.mark.asyncio
async def test_tick_sets_the_live_gauge(store, channels, clock):
    await _seed(store, clock)
    await store.save(12, "other-server", "d", "s", author_id=1)
    with patch.object(sc.metrics, "set_skills_live") as gauge:
        await _curator(store, channels, FakeLLM(_answer()), clock).tick([GUILD, 12])
    gauge.assert_called_once_with(6)


# ------------------------------ store additions ---------------------------

@pytest.mark.asyncio
async def test_save_keeps_the_first_channel_until_a_use_sets_one(store):
    await store.save(GUILD, "a-skill", "d", "s", author_id=1, channel_id=5)
    await store.save(GUILD, "a-skill", "d", "s2", author_id=1, channel_id=6)
    assert (await store.get(GUILD, "a-skill"))["last_channel_id"] == 5
    await store.record_use(GUILD, "a-skill", 7)
    assert (await store.get(GUILD, "a-skill"))["last_channel_id"] == 7


@pytest.mark.asyncio
async def test_expect_version_on_save_and_delete(store):
    await store.save(GUILD, "a-skill", "d", "s", author_id=1)
    with pytest.raises(VersionError):
        await store.save(GUILD, "a-skill", "d", "s2", author_id=1, expect_version=2)
    await store.save(GUILD, "a-skill", "d", "s2", author_id=1, expect_version=1)
    with pytest.raises(VersionError):
        await store.delete(GUILD, "a-skill", 1, expect_version=1)
    await store.delete(GUILD, "a-skill", 1, expect_version=2)
    with pytest.raises(SkillError):
        await store.save(GUILD, "a-skill", "d", "s3", author_id=1, expect_version=2)  # deleted

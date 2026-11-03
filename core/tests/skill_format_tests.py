"""Skills: names, limits, SKILL.md, validation, index and ranking (pure)."""
from classes import skill_format as sf

DAY = 86400


def test_settings_defaults_and_overrides():
    cfg = sf.settings({})
    assert cfg["enabled"] and cfg["max_per_guild"] == 100 and cfg["index_max"] == 20
    cfg = sf.settings({"SKILLS_ENABLED": "0", "SKILLS_MAX_FILES": "3", "SKILLS_INDEX_MAX": "x"})
    assert not cfg["enabled"] and cfg["max_files"] == 3 and cfg["index_max"] == 20


def test_clean_name():
    assert sf.clean_name("  Weekly Tracker_Chart ") == "weekly-tracker-chart"


def test_validate_good_and_bad_names():
    assert sf.validate("weekly-chart", "A chart.", "1. Do it") == []
    for bad in ("", "a", "-ab", "ab-", "Has Caps", "x" * 49, "under_score"):
        assert any(p.startswith("name") for p in sf.validate(bad, "d", "b")), bad


def test_validate_description_and_body():
    assert any("description" in p for p in sf.validate("good-name", "", "b"))
    assert any("description" in p for p in sf.validate("good-name", "two\nlines", "b"))
    assert any("description" in p for p in sf.validate("good-name", "x" * 201, "b"))
    assert any("body" in p for p in sf.validate("good-name", "d", "  "))
    assert any("body" in p for p in sf.validate("good-name", "d", "x" * 8001))


def test_validate_files():
    ok = {"scripts/run.py": "print(1)", "references/notes.md": "hi"}
    assert sf.validate("good-name", "d", "b", ok) == []
    for path in ("run.py", "/etc/passwd", "scripts/../x", "../scripts/a", "scripts\\a", "other/a.txt",
                 "scripts", "scripts//a"):
        assert sf.validate("good-name", "d", "b", {path: "x"}), path
    assert sf.validate("good-name", "d", "b", {"scripts/a": b"bytes"})
    assert sf.validate("good-name", "d", "b", {"scripts/a": "nul\x00"})


def test_validate_counts_and_total_size():
    many = {f"scripts/{i}.py": "x" for i in range(11)}
    assert any("files" in p for p in sf.validate("good-name", "d", "b", many))
    big = {"assets/a.txt": "x" * 70000}
    assert any("too big" in p for p in sf.validate("good-name", "d", "b", big))


def test_skill_md_round_trip():
    text = sf.render_skill_md("my-skill", "Does a thing.", "1. one\n2. two")
    meta, body = sf.parse_skill_md(text)
    assert meta == {"name": "my-skill", "description": "Does a thing."}
    assert body == "1. one\n2. two"


def test_parse_without_front_matter():
    assert sf.parse_skill_md("just text") == ({}, "just text")
    assert sf.parse_skill_md("---\nname: a\nno end") == ({}, "---\nname: a\nno end")


def _skill(name, uses=0, successes=0, last=None, updated=0, description="d"):
    return {"name": name, "uses": uses, "successes": successes, "last_used_at": last,
            "updated_at": updated, "description": description}


def test_rank_prefers_successes_then_uses_and_sinks_idle():
    now = 1000 * DAY
    skills = [_skill("idle", uses=5, last=now - 400 * DAY),
              _skill("used", uses=3, last=now),
              _skill("proven", uses=3, successes=2, last=now),
              _skill("new", updated=now)]
    assert [s["name"] for s in sf.rank(skills, now)] == ["proven", "used", "new", "idle"]


def test_rank_is_stable_on_ties():
    now = 100 * DAY
    skills = [_skill("b", updated=5), _skill("a", updated=5)]
    assert [s["name"] for s in sf.rank(skills, now)] == ["a", "b"]
    assert [s["name"] for s in sf.rank(list(reversed(skills)), now)] == ["a", "b"]


def test_render_index_caps_and_points_at_search():
    now = 10 * DAY
    skills = [_skill(f"skill-{i:02}", uses=i, description=f"does {i}") for i in range(5)]
    text = sf.render_index(skills, 3, now)
    lines = text.split("\n")
    assert lines[0].startswith("Skills saved in this server")
    assert lines[1] == "- skill-04: does 4"
    assert lines[-1] == "(2 more: search_skills)"
    assert sf.render_index([], 3, now) == ""
    assert "more" not in sf.render_index(skills, 5, now)


def test_search_orders_by_match():
    skills = [{"name": "weekly-chart", "description": "Chart tracker embeds", "body": "use matplotlib"},
              {"name": "poll-report", "description": "Summarise a poll", "body": "chart the votes"},
              {"name": "other", "description": "Unrelated", "body": "nothing"}]
    assert [s["name"] for s in sf.search(skills, "weekly chart")] == ["weekly-chart", "poll-report"]
    assert sf.search(skills, "a") == []


def test_closest_names():
    assert sf.closest_names(["weekly-chart", "poll-report"], "Weekly Chrat")[0] == "weekly-chart"
    assert sf.closest_names([], "x") == []

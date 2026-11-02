"""Tests for classes/settings_check.py: the v3 old-name guard.

Run from the repo root: pytest core/tests/settings_check_tests.py
"""
import re
from pathlib import Path

from classes import settings_check
from classes.settings_check import REMOVED, check, find_removed, format_error, missing_required

ROOT = Path(__file__).resolve().parents[2]


def test_nothing_found_in_a_v3_env():
    env = {"LLM_MODEL": "some/model", "QUEUE_WORKERS": "2", "PERSONA_REASONING_EFFORT": "low"}
    assert find_removed(env) == []
    assert check(env) is None


def test_old_names_are_found_in_order_with_their_replacement():
    env = {"LLM_MODEL": "m", "WORKER_COUNT": "2", "MODEL": "x", "LLM_CONTEXT_LENGTH": "80000"}
    found = find_removed(env)
    assert found == [
        ("MODEL", "LLM_MODEL"),
        ("LLM_CONTEXT_LENGTH", REMOVED["LLM_CONTEXT_LENGTH"]),
        ("WORKER_COUNT", "QUEUE_WORKERS"),
    ]


def test_empty_old_names_are_ignored():
    # compose writes NAME= for a variable passed through unset.
    assert find_removed({"LLM_MODEL": "m", "WORKER_COUNT": "", "LLM_PASS": "  "}) == []


def test_error_names_each_old_setting_and_its_replacement():
    text = format_error([("WORKER_COUNT", "QUEUE_WORKERS"), ("LLM_CONTEXT_LENGTH", REMOVED["LLM_CONTEXT_LENGTH"])])
    assert "WORKER_COUNT -> QUEUE_WORKERS" in text
    assert "LLM_CONTEXT_LENGTH: removed" in text
    assert settings_check.UPGRADE_DOC in text


def test_llm_model_is_required():
    assert missing_required({}) == ["LLM_MODEL"]
    assert missing_required({"LLM_MODEL": "  "}) == ["LLM_MODEL"]
    assert missing_required({"LLM_MODEL": "m"}) == []
    assert "LLM_MODEL is required" in check({})


def test_check_reports_both_problems():
    text = check({"WORKER_COUNT": "2"})
    assert "WORKER_COUNT -> QUEUE_WORKERS" in text and "LLM_MODEL is required" in text


def test_replacements_are_not_themselves_removed():
    for old, new in REMOVED.items():
        assert new not in REMOVED, (old, new)
        assert new.startswith("removed: ") or re.fullmatch(r"[A-Z][A-Z0-9_]+", new), (old, new)


def test_upgrade_doc_lists_exactly_the_removed_names():
    """docs/upgrading-to-v3.md's env table and REMOVED must agree both ways."""
    doc = (ROOT / settings_check.UPGRADE_DOC).read_text(encoding="utf-8")
    section = doc.split("## Environment variables", 1)[1].split("\n## ", 1)[0]
    # The rename table is the one headed "| Old | New |" (the section also has
    # a table of default changes, which are not renames).
    table = section.split("| Old | New |", 1)[1].split("\n\n", 1)[0]
    documented = set()
    for line in table.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 2 or not cells[0].startswith("`"):
            continue
        documented.update(re.findall(r"`([A-Z][A-Z0-9_]+)`", cells[0]))
    assert documented == set(REMOVED), (
        f"missing from the doc: {sorted(set(REMOVED) - documented)}; "
        f"not in REMOVED: {sorted(documented - set(REMOVED))}")

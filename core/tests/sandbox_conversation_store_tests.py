import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from classes import sandbox_conversation_store as store_mod
from classes.sandbox_conversation_store import (
    SandboxConversationStore,
    build_record,
    resume_preamble,
)

# To run this pytest file from the command line, use:
# PYTHONPATH=$(pwd) pytest core/tests/sandbox_conversation_store_tests.py


def _rows(*pairs):
    return [{"author": "ana", "text": text, "status": status, "reply": ""}
            for text, status in pairs]


def test_the_record_keeps_only_what_was_not_applied():
    record = build_record(_rows(("blue", "done"), ("csv too", "late"),
                                ("bigger", "accepted"), ("hi", "noted")),
                          outcome="ok", open_question="png or jpg?")
    assert record["version"] == store_mod.RECORD_VERSION
    assert [c["text"] for c in record["carry"]] == ["csv too", "bigger"]
    assert record["open_question"] == "png or jpg?"


def test_the_record_is_bounded():
    rows = _rows(*[(f"m{i}", "late") for i in range(store_mod.MAX_RECORD_ITEMS + 5)])
    record = build_record(rows, outcome="ok")
    assert len(record["carry"]) == store_mod.MAX_RECORD_ITEMS
    assert record["carry"][-1]["text"] == f"m{store_mod.MAX_RECORD_ITEMS + 4}"


def test_the_preamble_offers_loose_ends_as_context():
    record = build_record(_rows(("csv too", "late")), outcome="timeout",
                          open_question="png or jpg?")
    text = resume_preamble(record)
    assert "csv too" in text and "not applied" in text
    assert "png or jpg?" in text
    # context for the model, never a claim about the workspace
    assert "workspace" not in text.lower()


def test_no_preamble_without_loose_ends_or_for_an_unknown_version():
    assert resume_preamble(None) == ""
    assert resume_preamble(build_record(_rows(("blue", "done")), outcome="ok")) == ""
    assert resume_preamble({"version": 999, "carry": [{"text": "x", "status": "late"}]}) == ""


@pytest.mark.asyncio
async def test_save_uses_the_snapshot_ttl_and_trims_to_size(monkeypatch):
    monkeypatch.setenv("SANDBOX_SNAPSHOT_TTL_SECONDS", "123")
    monkeypatch.setattr(store_mod, "MAX_RECORD_BYTES", 400)
    redis = MagicMock(set=AsyncMock())
    record = build_record(_rows(*[("x" * 100, "late") for _ in range(8)]), outcome="ok")
    with patch.object(store_mod, "text_client", return_value=redis):
        await SandboxConversationStore().save(55, record)

    key, raw = redis.set.await_args.args
    assert key == "dcb:sandbox_conversation:55"
    assert redis.set.await_args.kwargs["ex"] == 123
    assert len(raw.encode("utf-8")) <= 400
    assert json.loads(raw)["carry"]  # trimmed from the oldest, not emptied


@pytest.mark.asyncio
async def test_load_ignores_a_corrupt_record():
    redis = MagicMock(get=AsyncMock(return_value="{not json"))
    with patch.object(store_mod, "text_client", return_value=redis):
        assert await SandboxConversationStore().load(55) is None


@pytest.mark.asyncio
async def test_load_round_trips():
    record = build_record(_rows(("csv", "late")), outcome="ok")
    redis = MagicMock(get=AsyncMock(return_value=json.dumps(record)))
    with patch.object(store_mod, "text_client", return_value=redis):
        assert await SandboxConversationStore().load(55) == record

"""core's own HTTP calls to the LLM server (/props, /v1/models), against the
response shapes of both servers the chart can run: llama.cpp and Strata.

Run from the repo root: `pytest core/tests/llm_server_tests.py`
"""
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from classes import text_llm_handler as tlh
from classes.llm_config import llm_auth_headers
from classes.text_llm_handler import parse_slot_context, served_model_names

# Trimmed from real responses.
LLAMACPP_MODELS = {
    "models": [{"name": "ggml-org/Qwen3.8-27B-GGUF:Q4_K_M", "model": "ggml-org/Qwen3.8-27B-GGUF:Q4_K_M"}],
    "object": "list",
    "data": [{"id": "ggml-org/Qwen3.8-27B-GGUF:Q4_K_M", "object": "model"}],
}
STRATA_MODELS = {
    "object": "list",
    "data": [{"id": "qwen3.8-flash-next-iq3_xxs", "object": "model",
              "status": {"value": "loaded"}, "meta": {"n_ctx": 32768}}],
}
STRATA_PROPS = {
    "default_generation_settings": {"n_ctx": 32768, "params": {"temperature": 1.0}},
    "total_slots": 2,
    "model_alias": "qwen3.8-flash-next-iq3_xxs",
    "build_info": "Strata 0.1.48",
}


def test_auth_header_sends_llm_pass(monkeypatch):
    monkeypatch.setenv("LLM_PASS", "secret-key")
    assert llm_auth_headers() == {"Authorization": "Bearer secret-key"}


def test_auth_header_falls_back_to_placeholder(monkeypatch):
    monkeypatch.delenv("LLM_PASS", raising=False)
    assert llm_auth_headers() == {"Authorization": "Bearer llamacpp"}


def test_strata_props_give_the_per_slot_context():
    assert parse_slot_context(STRATA_PROPS) == 32768


@pytest.mark.parametrize("body, expected", [
    (LLAMACPP_MODELS, ["ggml-org/Qwen3.8-27B-GGUF:Q4_K_M"]),
    (STRATA_MODELS, ["qwen3.8-flash-next-iq3_xxs"]),
    ({"data": [{"id": "a"}, {"id": "b"}], "models": [{"name": "a"}]}, ["a", "b"]),
    ({}, []), (None, []), ({"data": "nope"}, []), ({"data": [None, {}, {"id": ""}]}, []),
])
def test_served_model_names(body, expected):
    assert served_model_names(body) == expected


def _session(status=200, body=None):
    response = MagicMock()
    response.status = status
    response.json = AsyncMock(return_value=body)
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=False)
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    session.get = MagicMock(return_value=response)
    return session


@pytest.mark.asyncio
async def test_side_calls_send_the_key(monkeypatch):
    monkeypatch.setenv("LLM_PASS", "secret-key")
    monkeypatch.setattr(tlh, "_slot_context", None)
    expected = {"Authorization": "Bearer secret-key"}
    for call, body in ((tlh.TextLLMHandler.refresh_slot_context, STRATA_PROPS),
                       (lambda: tlh.TextLLMHandler.check_model_ready("any"), STRATA_MODELS)):
        session = _session(body=body)
        with patch.object(tlh.aiohttp, "ClientSession", return_value=session):
            await call()
        assert session.get.call_args.kwargs["headers"] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("model, body, level, text", [
    ("ggml-org/Qwen3.8-27B-GGUF:Q4_K_M", LLAMACPP_MODELS, logging.INFO, "is available"),
    # A one-model server answers any name: not something to wait for.
    ("qwen3.8-flash-next", STRATA_MODELS, logging.INFO, "serves one model, qwen3.8-flash-next-iq3_xxs"),
    ("wanted", {"data": [{"id": "a"}, {"id": "b"}]}, logging.WARNING, "not loaded yet"),
])
async def test_check_model_ready_logs(caplog, model, body, level, text):
    with patch.object(tlh.aiohttp, "ClientSession", return_value=_session(body=body)), \
         caplog.at_level(logging.INFO, logger=tlh.logger.name):
        await tlh.TextLLMHandler.check_model_ready(model)
    record = caplog.records[-1]
    assert record.levelno == level and text in record.getMessage()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403])
async def test_a_rejected_key_is_named_not_mistaken_for_a_slow_start(caplog, monkeypatch, status):
    monkeypatch.setattr(tlh, "_slot_context", None)
    for call in (tlh.TextLLMHandler.refresh_slot_context,
                 lambda: tlh.TextLLMHandler.check_model_ready("any")):
        caplog.clear()
        with patch.object(tlh.aiohttp, "ClientSession", return_value=_session(status=status)), \
             caplog.at_level(logging.INFO, logger=tlh.logger.name):
            await call()
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("rejected the API key" in w for w in warnings)
        assert not any("still be downloading" in r.getMessage() for r in caplog.records)

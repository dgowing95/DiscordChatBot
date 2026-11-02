"""core's own HTTP calls to the LLM server (/props, /v1/models), against the
response shapes of both servers the chart can run: llama.cpp and Strata.

Run from the repo root: `pytest core/tests/llm_server_tests.py`
"""
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from classes import llm_lanes
from classes import text_llm_handler as tlh
from classes.llm_roles import PERSONA, llm_extra_headers, role_config, side_job_config
from classes.side_llm import SideLLM
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


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("LLM_HOST", "PERSONA_LLM_HOST", "BRAIN_LLM_HOST", "SUBAGENT_LLM_HOST",
                 "PERSONA_LLM_API_KEY", "BRAIN_LLM_API_KEY", "LLM_SLOTS", "LLM_CHAT_RESERVED"):
        monkeypatch.delenv(name, raising=False)
    llm_lanes.reset()
    yield
    llm_lanes.reset()


def _auth_headers():
    return role_config(PERSONA).auth_headers()


def test_auth_header_sends_llm_api_key(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "secret-key")
    monkeypatch.delenv("LLM_EXTRA_HEADERS", raising=False)
    assert _auth_headers() == {"Authorization": "Bearer secret-key"}


def test_auth_header_falls_back_to_placeholder(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_EXTRA_HEADERS", raising=False)
    assert _auth_headers() == {"Authorization": "Bearer llamacpp"}


@pytest.mark.parametrize("raw, expected", [
    ("", {}),
    ("x-opencode-session=e2e-run-1", {"x-opencode-session": "e2e-run-1"}),
    (" a=1 , b = two words ,", {"a": "1", "b": "two words"}),
    ("a=x=y", {"a": "x=y"}),
    # Malformed entries are skipped, the rest kept.
    ("novalue, =x, bad name=1, ok=1", {"ok": "1"}),
    # Values httpx could not encode: a control character, non-ASCII.
    ("a=line\nbreak, b=“quoted”, c=em—dash, ok=1", {"ok": "1"}),
    # LLM_API_KEY owns Authorization.
    ("Authorization=Bearer other, ok=1", {"ok": "1"}),
])
def test_extra_headers_parse(monkeypatch, raw, expected):
    monkeypatch.setenv("LLM_EXTRA_HEADERS", raw)
    assert llm_extra_headers() == expected


def test_malformed_extra_header_is_not_logged(monkeypatch, caplog):
    monkeypatch.setenv("LLM_EXTRA_HEADERS", "ok=1, Authorization Bearer sk-live-secret")
    with caplog.at_level(logging.WARNING, logger="classes.llm_roles"):
        assert llm_extra_headers() == {"ok": "1"}
    assert "entry 2" in caplog.text
    assert "sk-live-secret" not in caplog.text


def test_extra_headers_go_with_the_auth_header(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "secret-key")
    monkeypatch.setenv("LLM_EXTRA_HEADERS", "x-opencode-session=s1")
    assert _auth_headers() == {"x-opencode-session": "s1", "Authorization": "Bearer secret-key"}


def test_extra_headers_only_for_llm_host(monkeypatch):
    monkeypatch.setenv("LLM_HOST", "https://api.example/zen/go")
    monkeypatch.setenv("LLM_EXTRA_HEADERS", "x-opencode-session=s1")
    monkeypatch.setenv("RESEARCH_LLM_HOST", "https://api.example/zen/go/")
    assert side_job_config("RESEARCH_").extra_headers == {"x-opencode-session": "s1"}
    monkeypatch.setenv("RESEARCH_LLM_HOST", "http://other:8080")
    assert side_job_config("RESEARCH_").extra_headers == {}


def test_clients_send_the_extra_headers(monkeypatch):
    monkeypatch.setenv("LLM_HOST", "https://api.example/zen/go")
    monkeypatch.setenv("LLM_EXTRA_HEADERS", "x-opencode-session=s1")
    monkeypatch.setattr(tlh, "_main_model_client", None)
    main = tlh._get_main_model_client()._client
    assert main.default_headers["x-opencode-session"] == "s1"
    monkeypatch.setattr(tlh, "_main_model_client", None)

    same_host = SideLLM("t", lambda: side_job_config("RESEARCH_"))
    assert same_host._get_client().default_headers["x-opencode-session"] == "s1"
    monkeypatch.setenv("RESEARCH_LLM_HOST", "http://other:8080")
    elsewhere = SideLLM("t", lambda: side_job_config("RESEARCH_"))
    assert "x-opencode-session" not in elsewhere._get_client().default_headers


def test_no_extra_headers_by_default(monkeypatch):
    monkeypatch.delenv("LLM_EXTRA_HEADERS", raising=False)
    monkeypatch.setattr(tlh, "_main_model_client", None)
    main = tlh._get_main_model_client()._client
    assert not any(k.lower().startswith("x-opencode") for k in main.default_headers)
    monkeypatch.setattr(tlh, "_main_model_client", None)


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
    monkeypatch.setenv("LLM_API_KEY", "secret-key")
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


@pytest.mark.parametrize("props, expected", [
    (STRATA_PROPS, 2), ({"total_slots": "4"}, 4), ({"total_slots": 0}, None),
    ({}, None), (None, None), ({"total_slots": "x"}, None),
])
def test_parse_total_slots(props, expected):
    assert tlh.parse_total_slots(props) == expected


def _per_url_session(responses):
    """A ClientSession whose get() answers by URL: {url: (status, body)}."""
    def get(url, **kwargs):
        status, body = responses.get(url, (404, None))
        response = MagicMock()
        response.status = status
        response.json = AsyncMock(return_value=body)
        response.__aenter__ = AsyncMock(return_value=response)
        response.__aexit__ = AsyncMock(return_value=False)
        get.calls.append((url, kwargs.get("headers")))
        return response
    get.calls = []
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    session.get = get
    return session


@pytest.mark.asyncio
async def test_refresh_lanes_each_self_hosted_server_with_its_own_key(monkeypatch):
    """Local persona (llama.cpp, 4 slots) and a brain on a remote API: the
    local server is laned, the remote one is not, and each /props read
    carries its own role's key."""
    monkeypatch.setenv("LLM_HOST", "http://llamacpp:8080")
    monkeypatch.setenv("LLM_API_KEY", "local-key")
    monkeypatch.setenv("BRAIN_LLM_HOST", "https://remote.example/api")
    monkeypatch.setenv("BRAIN_LLM_API_KEY", "remote-key")
    monkeypatch.setattr(tlh, "_slot_context", None)
    props = {"default_generation_settings": {"n_ctx": 40000}, "total_slots": 4}
    session = _per_url_session({"http://llamacpp:8080/props": (200, props)})
    with patch.object(tlh.aiohttp, "ClientSession", return_value=session):
        assert await tlh.TextLLMHandler.refresh_slot_context() == 40000
    local = llm_lanes.lanes_for("http://llamacpp:8080")
    assert (local.slots, local.chat_reserved) == (4, 1)
    assert llm_lanes.lanes_for("https://remote.example:443/api") is None
    headers = dict(session.get.calls)
    assert headers["http://llamacpp:8080/props"]["Authorization"] == "Bearer local-key"
    assert headers["https://remote.example/api/props"]["Authorization"] == "Bearer remote-key"


@pytest.mark.asyncio
async def test_a_laned_server_stays_laned_through_a_failed_read(monkeypatch):
    monkeypatch.setenv("LLM_HOST", "http://llamacpp:8080")
    monkeypatch.setattr(tlh, "_slot_context", None)
    props = {"default_generation_settings": {"n_ctx": 20000}, "total_slots": 2}
    with patch.object(tlh.aiohttp, "ClientSession",
                      return_value=_per_url_session({"http://llamacpp:8080/props": (200, props)})):
        await tlh.TextLLMHandler.refresh_slot_context()
    with patch.object(tlh.aiohttp, "ClientSession", return_value=_per_url_session({})):
        assert await tlh.TextLLMHandler.refresh_slot_context() == 20000  # last value kept
    assert llm_lanes.lanes_for("http://llamacpp:8080").slots == 2


@pytest.mark.asyncio
async def test_slots_fall_back_to_llm_slots_when_props_has_none(monkeypatch):
    monkeypatch.setenv("LLM_HOST", "http://llamacpp:8080")
    monkeypatch.setenv("LLM_SLOTS", "3")
    monkeypatch.setenv("LLM_CHAT_RESERVED", "2")
    monkeypatch.setattr(tlh, "_slot_context", None)
    props = {"default_generation_settings": {"n_ctx": 20000}}
    with patch.object(tlh.aiohttp, "ClientSession",
                      return_value=_per_url_session({"http://llamacpp:8080/props": (200, props)})):
        await tlh.TextLLMHandler.refresh_slot_context()
    lanes = llm_lanes.lanes_for("http://llamacpp:8080")
    assert (lanes.slots, lanes.chat_reserved) == (3, 2)


@pytest.mark.asyncio
async def test_a_200_that_is_not_llamacpp_props_is_not_laned(monkeypatch):
    monkeypatch.setenv("LLM_HOST", "https://api.example/zen/go")
    monkeypatch.setattr(tlh, "_slot_context", None)
    with patch.object(tlh.aiohttp, "ClientSession",
                      return_value=_per_url_session({"https://api.example/zen/go/props": (200, {"ok": 1})})):
        assert await tlh.TextLLMHandler.refresh_slot_context() is None
    assert llm_lanes.all_lanes() == {}

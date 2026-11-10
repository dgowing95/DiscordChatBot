"""What model calls use (classes/token_usage.py), and where it is counted:
the metrics (classes/metrics.py) and the model wrapper (classes/laned_model.py)."""
from types import SimpleNamespace

import pytest
from agents import ModelSettings
from agents.usage import InputTokensDetails, OutputTokensDetails, Usage
from prometheus_client import REGISTRY

from classes import token_usage
from classes.laned_model import LanedModel
from classes.metrics import observe_llm_usage
from laned_model_tests import _call_args


def _sdk_usage(inp=3339, cached=3328, out=13):
    return Usage(requests=1, input_tokens=inp, output_tokens=out, total_tokens=inp + out,
                 input_tokens_details=InputTokensDetails(cached_tokens=cached, cache_write_tokens=0),
                 output_tokens_details=OutputTokensDetails(reasoning_tokens=0))


def test_the_sdks_usage_and_the_providers_cost():
    # As OpenRouter answered (2026-11-10): the cost is in its own payload.
    tally = token_usage.from_usage(_sdk_usage(), {"prompt_tokens": 3339, "cost": 4.0198e-05})
    assert (tally.input, tally.cached, tally.output) == (3339, 3328, 13)
    assert tally.cost == pytest.approx(4.0198e-05)
    assert tally.total == 3352


def test_the_openai_clients_usage_with_its_cost():
    usage = {"prompt_tokens": 100, "completion_tokens": 20,
             "prompt_tokens_details": {"cached_tokens": 60}, "cost": 0.002}
    tally = token_usage.from_usage(usage)
    assert (tally.input, tally.cached, tally.output, tally.cost) == (100, 60, 20, 0.002)


def test_a_server_that_says_less_counts_as_nothing_not_an_error():
    assert token_usage.from_usage(None) == token_usage.Tally()
    tally = token_usage.from_usage(SimpleNamespace(input_tokens=50, output_tokens=None))
    assert (tally.input, tally.cached, tally.output, tally.cost) == (50, 0, 0, None)
    # No cost from llama.cpp, nonsense ignored, cached never above the prompt.
    odd = {"prompt_tokens": 10, "completion_tokens": True,
           "prompt_tokens_details": {"cached_tokens": 99}, "cost": "free"}
    assert token_usage.from_usage(odd) == token_usage.Tally(10, 10, 0, None)
    assert token_usage.from_usage({"prompt_tokens": 1, "cost": float("nan")}).cost is None


def test_a_tally_adds_up_and_reads_in_a_log_line():
    total = token_usage.Tally()
    assert total.text() == "0 token(s) (0 of 0 in cached, 0 out)"
    total.add(token_usage.Tally(1000, 800, 50))
    assert total.cost is None  # nobody said
    total.add(token_usage.Tally(2000, 1900, 30, 0.0125))
    total.add(token_usage.Tally(10, 0, 5, 0.0005))
    assert (total.input, total.cached, total.output) == (3010, 2700, 85)
    assert total.text() == "3095 token(s) (2700 of 3010 in cached, 85 out), $0.0130"


def _value(name, **labels):
    return REGISTRY.get_sample_value(name, labels) or 0.0


def test_the_metrics_split_input_cached_and_output_by_role_and_model():
    labels = {"role": "brain", "model": "test/split"}
    observe_llm_usage("brain", "test/split", token_usage.Tally(1000, 800, 50, 0.01))
    observe_llm_usage("brain", "test/split", token_usage.Tally(10, 0, 5))
    assert _value("discord_bot_llm_tokens_total", kind="input", **labels) == 210
    assert _value("discord_bot_llm_tokens_total", kind="cached", **labels) == 800
    assert _value("discord_bot_llm_tokens_total", kind="output", **labels) == 55
    assert _value("discord_bot_llm_cost_dollars_total", **labels) == pytest.approx(0.01)


class _Inner:
    model = "test/laned"

    def __init__(self):
        self.settings = []

    async def get_response(self, system_instructions, input, model_settings, *args, **kwargs):
        self.settings.append(model_settings)
        return SimpleNamespace(output=[], usage=_sdk_usage(200, 150, 10),
                               raw_usage={"cost": 0.003})


@pytest.mark.asyncio
async def test_every_call_through_the_wrapper_is_counted_with_its_cost():
    inner = _Inner()
    model = LanedModel(inner, "https://remote.example", 30, role="subagent")
    args = _call_args()
    args["model_settings"] = ModelSettings(temperature=0.5)
    await model.get_response(**args)
    # The SDK keeps the provider's payload (the cost) only when asked.
    assert inner.settings[0].preserve_raw_usage is True
    assert inner.settings[0].temperature == 0.5
    labels = {"role": "subagent", "model": "test/laned"}
    assert _value("discord_bot_llm_tokens_total", kind="cached", **labels) == 150
    assert _value("discord_bot_llm_tokens_total", kind="input", **labels) == 50
    assert _value("discord_bot_llm_cost_dollars_total", **labels) == pytest.approx(0.003)

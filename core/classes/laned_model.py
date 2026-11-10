"""An agents-SDK Model that takes its server's lane slot for every call.

LanedModel wraps the real model (an OpenAIChatCompletionsModel) and, before
each call, takes a slot of its lane on that server (classes/llm_lanes.py), so
chat replies go ahead of brain work when the server is busy. A server with no
Lanes (a remote API; its /props never answered) is called straight away.

It also applies the role's call timeout itself, AFTER the slot is taken. The
SDK applies ModelSettings.timeout around the whole get_response() (and the
whole stream, first event included: agents/run_internal/model_retry.py), so
left to the SDK the time spent waiting for a slot would count toward one
call's timeout, and a chat reply busy for ten minutes would time out a brain
call that never started. So agents built on a LanedModel leave
ModelSettings.timeout unset, and a call that runs past call_timeout raises
agents.ModelTimeoutError, the same error the SDK raises. Lane wait does count
toward a run's own wall clock (BRAIN_TIMEOUT): that one is a loop guard.

A slot is held for one model call, never across a tool, so a side call made
inside a tool (a research brief) cannot deadlock against its own parent run.

Every call through the wrapper is bounded, including the ones core makes on
the model directly (the max-turns wrap-up, the voice prefill), which the SDK
never wrapped.

A reply cut off before it said anything (finish_reason "length" with no
text or tool call) is asked for again, once, in the same slot: a reasoning
model can spend its whole output budget thinking. Seen live on the brain
(a hosted DeepSeek thought for 185 s on "draw a lighthouse" and returned
nothing, which the SDK raises as ModelBehaviorError and which ended the
run); the same request a minute later went through. Not for streamed
calls: by then part of the reply may already have been spoken.

Every call's tokens (and cost, when the provider sends it) go to the metrics
here (classes/token_usage.py), so the calls core makes on the model directly
are counted too. The SDK keeps the provider's own usage payload, where
OpenRouter puts the cost, only when asked (preserve_raw_usage), so every
call asks; it changes nothing in the request.
"""
import asyncio
import logging
import time

from agents import ModelBehaviorError, ModelSettings, ModelTimeoutError
from agents.models.interface import Model

from classes import llm_lanes, token_usage
from classes.metrics import inc_empty_retry, observe_lane_wait, observe_llm_usage

logger = logging.getLogger(__name__)


def _cut_off_empty(error: ModelBehaviorError) -> bool:
    """The SDK's error for a reply truncated before any text or tool call
    (agents/models/openai_chatcompletions.py). Matched on its wording: the
    SDK has no subclass for it."""
    return "finish_reason='length'" in str(error)


def _with_raw_usage(settings):
    """`settings` asking the SDK to keep the provider's usage payload."""
    try:
        return settings.resolve(ModelSettings(preserve_raw_usage=True))
    except Exception:
        return settings


class LanedModel(Model):
    """`inner` called through `host_key`'s lanes, bounded by `call_timeout`.

    `lane` None means the lane of the work making the call
    (llm_lanes.current_lane), read per call. `role` labels its token metrics.
    """

    def __init__(self, inner: Model, host_key: str, call_timeout: float, lane: str | None = None,
                 role: str = "unknown"):
        self.inner = inner
        self.host_key = host_key
        self.call_timeout = call_timeout
        self.lane = lane
        self.role = role

    def __getattr__(self, name):
        # Everything else (the model name, the client) is the inner model's.
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)

    def _lane(self) -> str:
        return self.lane or llm_lanes.current_lane.get()

    def _count(self, usage, raw=None) -> None:
        try:
            observe_llm_usage(self.role, getattr(self.inner, "model", "unknown"),
                              token_usage.from_usage(usage, raw))
        except Exception as e:
            logger.warning(f"Token metrics failed: {e}")

    async def _bounded(self, awaitable):
        try:
            async with asyncio.timeout(self.call_timeout) as scope:
                return await awaitable
        except TimeoutError:
            if scope.expired():
                raise ModelTimeoutError(self.call_timeout) from None
            raise

    async def get_response(self, system_instructions, input, model_settings, tools,
                           output_schema, handoffs, tracing, *, previous_response_id=None,
                           conversation_id=None, prompt=None):
        model_settings = _with_raw_usage(model_settings)

        def attempt():
            return self._bounded(self.inner.get_response(
                system_instructions, input, model_settings, tools, output_schema, handoffs,
                tracing, previous_response_id=previous_response_id,
                conversation_id=conversation_id, prompt=prompt,
            ))

        async def call():
            started = time.monotonic()
            try:
                response = await attempt()
            except ModelBehaviorError as e:
                if not _cut_off_empty(e):
                    raise
                logger.warning(f"Model call ({self._lane()} lane) was cut off after "
                               f"{time.monotonic() - started:.0f}s before saying anything; "
                               f"asking once more")
                inc_empty_retry(self._lane())
                response = await attempt()
            self._count(getattr(response, "usage", None), getattr(response, "raw_usage", None))
            return response

        llm_lanes.last_wait.set(0.0)
        lanes = llm_lanes.lanes_for(self.host_key)
        if lanes is None:
            return await call()
        lane = self._lane()
        async with lanes.slot(lane) as waited:
            observe_lane_wait(lane, waited)
            llm_lanes.last_wait.set(waited)
            return await call()

    async def stream_response(self, system_instructions, input, model_settings, tools,
                              output_schema, handoffs, tracing, *, previous_response_id=None,
                              conversation_id=None, prompt=None):
        model_settings = _with_raw_usage(model_settings)
        lanes = llm_lanes.lanes_for(self.host_key)
        lane = self._lane()
        slot = lanes.slot(lane) if lanes is not None else None
        llm_lanes.last_wait.set(0.0)
        if slot is not None:
            waited = await slot.__aenter__()
            observe_lane_wait(lane, waited)
            llm_lanes.last_wait.set(waited)
        try:
            stream = self.inner.stream_response(
                system_instructions, input, model_settings, tools, output_schema, handoffs,
                tracing, previous_response_id=previous_response_id,
                conversation_id=conversation_id, prompt=prompt,
            )
            # One deadline for the whole stream from the moment the slot is
            # taken, as the SDK's own was: each wait for the next event is
            # bounded by what is left of it.
            deadline = time.monotonic() + self.call_timeout
            try:
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise ModelTimeoutError(self.call_timeout)
                    try:
                        async with asyncio.timeout(remaining) as scope:
                            event = await anext(stream)
                    except StopAsyncIteration:
                        return
                    except TimeoutError:
                        if scope.expired():
                            raise ModelTimeoutError(self.call_timeout) from None
                        raise
                    if getattr(event, "type", None) == "response.completed":
                        # The streamed response's usage; its cost is not kept here.
                        self._count(getattr(getattr(event, "response", None), "usage", None))
                    yield event
            finally:
                await stream.aclose()
        finally:
            # Also on an early aclose() (a cancelled voice turn).
            if slot is not None:
                await slot.__aexit__(None, None, None)

    def get_retry_advice(self, request):
        return self.inner.get_retry_advice(request)

    async def close(self) -> None:
        await self.inner.close()

    async def _cleanup_on_run_end(self, owner: object) -> None:
        await self.inner._cleanup_on_run_end(owner)

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
toward a run's own wall clock (SANDBOX_TIMEOUT): that one is a loop guard.

A slot is held for one model call, never across a tool, so a side call made
inside a tool (a research brief) cannot deadlock against its own parent run.

Every call through the wrapper is bounded, including the ones core makes on
the model directly (the max-turns wrap-up, the voice prefill), which the SDK
never wrapped.
"""
import asyncio
import time

from agents import ModelTimeoutError
from agents.models.interface import Model

from classes import llm_lanes
from classes.metrics import observe_lane_wait


class LanedModel(Model):
    """`inner` called through `host_key`'s lanes, bounded by `call_timeout`.

    `lane` None means the lane of the work making the call
    (llm_lanes.current_lane), read per call.
    """

    def __init__(self, inner: Model, host_key: str, call_timeout: float, lane: str | None = None):
        self.inner = inner
        self.host_key = host_key
        self.call_timeout = call_timeout
        self.lane = lane

    def __getattr__(self, name):
        # Everything else (the model name, the client) is the inner model's.
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)

    def _lane(self) -> str:
        return self.lane or llm_lanes.current_lane.get()

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
        def call():
            return self._bounded(self.inner.get_response(
                system_instructions, input, model_settings, tools, output_schema, handoffs,
                tracing, previous_response_id=previous_response_id,
                conversation_id=conversation_id, prompt=prompt,
            ))

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
            # The deadline covers the model's time only: each wait for the next
            # event is bounded by what is left, but the consumer's own work
            # between events (speaking a sentence) is not counted.
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

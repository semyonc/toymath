#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""openrouter.py - the OpenAI Agents SDK backend, reached through OpenRouter.

This is an extraction of the original `do!` runner, not a rewrite: the same
authentication, the same model construction, the same empty-response retry,
the same tool-not-found steering, and the same Langfuse/OpenInference
integration. What is new is that a run is addressable - the asyncio task and
its loop are reachable from the run handle, so Jupyter Stop can actually stop
the provider call instead of leaving a daemon thread talking to OpenRouter
after the cell has ended.

LANDMINE (see AGENTS.md): the OpenInference instrumentor rides the Agents-SDK
tracing pipeline, so `build_model` may only disable that pipeline when
observability is inactive. That coupling lives here and nowhere else.
"""
import json
import logging
import os

import observability
import tactic_registry
from agent_backends.base import (COMPLETED, FAILED, INTERRUPTED, USER,
                                 AgentOutcome, DoAgentError, ThreadedRunHandle,
                                 ToolArgumentError, UnknownToolError)
from model_config import DEFAULT_MODEL, MODEL_VAR

log = logging.getLogger('toymath.agent.openrouter')

OPENROUTER_BASE_URL = 'https://openrouter.ai/api/v1'
API_KEY_VAR = 'OPEN_ROUTER'
DEFAULT_MAX_TURNS = 64
# providers intermittently return a zero-token, empty assistant message;
# ask again rather than let it end the derivation
EMPTY_RESPONSE_RETRIES = 3


# ---------------------------------------------------------------------------
# model construction
# ---------------------------------------------------------------------------

def _is_empty_model_response(response):
    """True when the model returned nothing actionable: no tool call and no
    text. Providers do this intermittently (a zero-token completion), and
    the Agents SDK reads it as the final answer."""
    for item in (getattr(response, 'output', None) or []):
        kind = getattr(item, 'type', None)
        if kind is not None and kind != 'message':
            return False  # a tool call, or anything else to act on
        for part in (getattr(item, 'content', None) or []):
            if getattr(part, 'refusal', None):
                return False
            text = getattr(part, 'text', None)
            if text and text.strip():
                return False
    return True


def _retrying_model(base_cls):
    """`base_cls` with empty completions retried.

    A provider that returns an empty assistant message ends the run where it
    stands, and the harness then presents whatever step came last as the
    answer - a half-finished derivation looking like a result. That is a
    transport hiccup, not a decision by the agent, so ask again instead of
    accepting it. Bounded, and the last response is returned either way so a
    genuinely mute model still terminates.
    """

    class RetryEmptyResponses(base_cls):
        async def get_response(self, *args, **kwargs):
            import asyncio
            response = None
            for attempt in range(EMPTY_RESPONSE_RETRIES + 1):
                response = await super().get_response(*args, **kwargs)
                if not _is_empty_model_response(response):
                    return response
                if attempt < EMPTY_RESPONSE_RETRIES:
                    await asyncio.sleep(0.5 * (2 ** attempt))
            return response

    return RetryEmptyResponses


def build_model(model_name=None):
    """OpenRouter-backed chat-completions model for the Agents SDK."""
    key = os.environ.get(API_KEY_VAR)
    if not key:
        raise DoAgentError(
            f'{API_KEY_VAR} is not set - put the OpenRouter key in .env')
    from openai import AsyncOpenAI
    from agents import OpenAIChatCompletionsModel, set_tracing_disabled
    # Leave the Agents SDK tracing ENABLED only while the OpenInference
    # instrumentor is attached: it rides that pipeline, so disabling it
    # would silently starve Langfuse. Otherwise disable it, so no traces are
    # shipped to OpenAI's backend - including the case where Langfuse is
    # active but the instrumentor could not attach.
    set_tracing_disabled(not observability.instrumented())
    client = AsyncOpenAI(base_url=OPENROUTER_BASE_URL, api_key=key)
    model = _retrying_model(OpenAIChatCompletionsModel)(
        model=model_name or os.environ.get(MODEL_VAR, DEFAULT_MODEL),
        openai_client=client)
    # kept for cancellation containment: if task cancellation does not
    # release the HTTP stream, the client we own can still be closed
    model.toymath_client = client
    return model


# ---------------------------------------------------------------------------
# tool conversion
# ---------------------------------------------------------------------------

def function_tools(dispatcher):
    """Agents SDK tools generated from the canonical bindings.

    The schema is the canonical one - this adapter converts representation
    only, so the model sees the same names, descriptions, and arguments it
    would see through any other backend.
    """
    from agents import FunctionTool
    return [FunctionTool(name=binding.name,
                         description=binding.description,
                         params_json_schema=binding.input_schema,
                         on_invoke_tool=_invoker(dispatcher, binding.name),
                         strict_json_schema=True)
            for binding in dispatcher.bindings.values()]


def _invoker(dispatcher, name):
    async def invoke(context, arguments):
        import asyncio
        # the handlers are synchronous and may block (a tactic, a sandboxed
        # figure): keep them off the event loop so cancellation stays live
        return await asyncio.to_thread(_dispatch, dispatcher, name, arguments)
    return invoke


def _dispatch(dispatcher, name, arguments):
    """Tool errors are steering, not run failures: a malformed call comes
    back to the model as text so it can repair, exactly as the SDK's own
    failure handler did."""
    try:
        return dispatcher.dispatch(name, arguments)
    except (ToolArgumentError, UnknownToolError) as exc:
        return json.dumps({'ok': False, 'op': name, 'error': str(exc)},
                          ensure_ascii=False)
    except Exception as exc:  # pragma: no cover - handler bug
        log.exception('tool %s failed', name)
        return ('An error occurred while running the tool. Please try '
                f'again. Error: {exc}')


def tool_error_message(dispatcher, fargs):
    """Steer a hallucinated tool name back in one turn.

    A model naming a TACTIC as a tool must be redirected to run_tactic,
    never abort the whole derivation (live: a hallucinated `integrate_table`
    call killed a run with three green steps).
    """
    if getattr(fargs, 'kind', None) != 'tool_not_found':
        return fargs.default_message
    spec = tactic_registry.BY_NAME.get(fargs.tool_name)
    if spec is not None:
        return (f"'{fargs.tool_name}' is a tactic, not a tool. Call "
                f"run_tactic with tactic='{fargs.tool_name}' and its "
                f"ordered string arguments (load_skill "
                f"'{spec.skill}' first if that skill is not loaded).")
    return (f"Tool '{fargs.tool_name}' does not exist. Use only the "
            'fixed tools: ' + ', '.join(dispatcher.names) + '.')


# ---------------------------------------------------------------------------
# backend
# ---------------------------------------------------------------------------

class OpenRouterBackend(object):
    """Runs one instruction through the OpenAI Agents SDK."""

    name = 'openrouter'

    def __init__(self, model=None, model_name=None, providers=(),
                 max_turns=DEFAULT_MAX_TURNS):
        # `model` is an already-built (or injected/scripted) SDK Model; when
        # absent one is built on `start`, so a missing key still raises on
        # the caller's thread instead of inside the worker.
        self.model = model
        self.model_name = model_name
        self.providers = tuple(providers or ())
        self.max_turns = max_turns

    def describe_model(self):
        if self.model is not None:
            return type(self.model).__name__
        return self.model_name or os.environ.get(MODEL_VAR, DEFAULT_MODEL)

    def start(self, request):
        model = (self.model if self.model is not None
                 else build_model(model_name=self.model_name))
        return OpenRouterRunHandle(request, model, self.providers,
                                   self.max_turns).start()


class OpenRouterRunHandle(ThreadedRunHandle):
    """The Agents SDK run, with its asyncio task reachable for cancellation."""

    def __init__(self, request, model, providers=(),
                 max_turns=DEFAULT_MAX_TURNS):
        super(OpenRouterRunHandle, self).__init__(request)
        self.model = model
        self.providers = tuple(providers or ())
        self.max_turns = max_turns
        self._loop = None
        self._task = None

    # -- execution ---------------------------------------------------------
    def run(self):
        import asyncio
        from agents.exceptions import MaxTurnsExceeded
        request = self.request
        metadata = dict(request.trace_metadata or {})
        metadata.update(backend=OpenRouterBackend.name,
                        model=(request.model
                               or type(self.model).__name__),
                        providers=list(self.providers),
                        max_turns=self.max_turns)
        # trace_run must be entered on this thread: the instrumentor's root
        # span inherits the active OTEL context here so the whole agent
        # trace nests under one Langfuse observation.
        with observability.trace_run(request.instruction,
                                     metadata=metadata) as span:
            try:
                result = asyncio.run(self._drive())
            except MaxTurnsExceeded as exc:
                # a harness limit with its own established semantics: the
                # finalizer decides whether a result committed on the last
                # turn still counts (see agent_do.finalize_session)
                return AgentOutcome(status=FAILED, error=str(exc),
                                    metadata={'turn_limit': True,
                                              'max_turns': self.max_turns})
            except asyncio.CancelledError:
                return AgentOutcome(status=INTERRUPTED)
            final = str(result.final_output or '').strip()
            observability.set_output(span, final or None)
            return AgentOutcome(status=COMPLETED, final_text=final)

    async def _drive(self):
        import asyncio
        from agents import Runner
        task = asyncio.ensure_future(
            Runner.run(self._agent(), self.request.instruction,
                       max_turns=self.max_turns,
                       run_config=self._run_config()))
        with self._lock:
            self._loop = asyncio.get_running_loop()
            self._task = task
            already_cancelled = self._cancel_requested
        if already_cancelled:
            task.cancel()       # cancel() arrived before the task existed
        return await task

    def _agent(self):
        from agents import Agent
        return Agent(name='toymath',
                     instructions=self.request.developer_instructions,
                     tools=function_tools(self.request.dispatcher),
                     model=self.model)

    def _run_config(self):
        from agents import ModelSettings, RunConfig
        settings = None
        if self.providers:
            settings = ModelSettings(extra_body={
                'provider': {
                    'order': list(self.providers),
                    'allow_fallbacks': False,
                },
            })
        dispatcher = self.request.dispatcher
        return RunConfig(
            tool_not_found_behavior='return_error_to_model',
            tool_error_formatter=lambda fargs: tool_error_message(dispatcher,
                                                                  fargs),
            model_settings=settings)

    # -- cancellation ------------------------------------------------------
    def request_cancel(self, reason=USER):
        with self._lock:
            loop, task = self._loop, self._task
        if loop is None or task is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(task.cancel)
        except RuntimeError:  # the loop finished first
            log.debug('agent loop already closed', exc_info=True)

    def abandon(self, reason=USER):
        """Past the grace deadline: drop the HTTP client we own so a stuck
        stream cannot hold the worker open indefinitely. The session is
        already closed, so nothing the worker does afterwards can land."""
        client = getattr(self.model, 'toymath_client', None)
        with self._lock:
            loop = self._loop
        if client is None or loop is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(
                lambda: loop.create_task(client.close()))
        except RuntimeError:
            log.debug('could not close the provider client', exc_info=True)

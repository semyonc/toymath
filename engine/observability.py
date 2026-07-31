#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""observability.py - optional Langfuse tracing for the do! agent endpoint.

Off by default. Set TOYMATH_OBSERVABILITY=on (with LANGFUSE_PUBLIC_KEY /
LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL in the environment or .env) to send
one trace per do! run to Langfuse: the agent turns, every LLM generation
with token usage and latency, and every trusted-primitive tool call.

Design notes (hard-won, see the module tests):

- The tracing leg is *observability only*. It never touches the ledger, the
  numeric oracle, or a step record. A Langfuse outage or misconfiguration is
  downgraded to a logged warning and the derivation runs unchanged - an
  unchecked derivation must never be the price of a tracing failure.
- We use the OpenInference OpenAI-Agents instrumentor, which registers an
  OpenTelemetry trace processor with the Agents SDK via
  ``set_trace_processors`` (exclusive by default). Two consequences the
  caller must respect:
    * The Agents SDK's own tracing MUST be left enabled while the
      instrumentor is attached - a stray ``set_tracing_disabled(True)``
      silently starves the processor.
    * The exclusive processor replaces the default OpenAI backend exporter,
      so traces are not shipped to OpenAI's platform; only Langfuse.
  Hence two separate states. ``active()`` means the Langfuse client is up,
  which is all a provider-neutral span needs. ``instrumented()`` means that
  processor actually attached, and ONLY that may leave Agents-SDK tracing
  on: active-but-not-instrumented would ship traces to OpenAI's backend
  instead of Langfuse. The instrumentor needs the Agents SDK installed, so
  a Codex-only installation traces through the client alone.
- Backends other than the Agents SDK get no automatic nesting: OTel context
  is thread-local, and a tool callback runs on a different thread from the
  one that opened the run span. ``capture_context`` + ``child_span`` carry
  it across explicitly.
- ``get_client()`` must run before ``instrument()``: constructing the
  Langfuse client registers its TracerProvider as the global OTEL provider,
  which the instrumentor then binds its tracer to.
"""
import base64
import contextlib
import logging
import os
import threading

log = logging.getLogger('toymath.observability')

ENABLE_VAR = 'TOYMATH_OBSERVABILITY'
_CRED_VARS = ('LANGFUSE_PUBLIC_KEY', 'LANGFUSE_SECRET_KEY')
_TRUTHY = {'1', 'true', 'on', 'yes', 'enable', 'enabled'}

_lock = threading.Lock()
_state = {'setup': False, 'active': False, 'client': None,
          'instrumented': False}


def _truthy(value):
    return bool(value) and value.strip().lower() in _TRUTHY


def is_enabled():
    """Whether the operator asked for tracing via the env toggle - a
    request, independent of whether deps/credentials actually let it start."""
    return _truthy(os.environ.get(ENABLE_VAR))


def active():
    """Whether tracing is set up and exporting. Only meaningful after
    setup() has run at least once."""
    return _state['active']


def instrumented():
    """Whether the OpenInference Agents-SDK processor actually attached.

    The ONLY predicate that may leave Agents-SDK tracing enabled: with the
    processor absent, that pipeline would export to OpenAI's backend rather
    than to Langfuse.
    """
    return _state['instrumented']


def setup(client=None):
    """Wire up Langfuse + the OpenInference Agents instrumentor once.

    Idempotent and thread-safe. Returns whether tracing is active. Any
    failure (missing deps, bad credentials, unreachable host at import) is
    logged and downgraded to inactive - never raised - so a misconfigured
    Langfuse can never break a derivation. ``client`` is a test seam; in
    production it is built from the environment.
    """
    with _lock:
        if _state['setup']:
            return _state['active']
        try:
            if client is None and not is_enabled():
                return False
            if client is None:
                client = _build_client()
            try:
                _instrument()
                _state['instrumented'] = True
            except Exception as exc:
                # A Codex-only installation has no Agents SDK to instrument.
                # That costs the nested LLM/tool spans, not the trace: the
                # provider-neutral run span still exports through the client.
                log.info('do! observability: the OpenAI-Agents instrumentor '
                         'is unavailable (%s); tracing per-run spans only',
                         exc)
            _state['client'] = client
            _state['active'] = True
            host = (os.environ.get('LANGFUSE_BASE_URL')
                    or os.environ.get('LANGFUSE_HOST') or 'cloud.langfuse.com')
            log.info('do! observability enabled - tracing to Langfuse (%s)',
                     host)
        except Exception as exc:  # deps/creds/host - never fatal to do!
            _state['active'] = False
            log.warning('do! observability was requested (%s) but could not '
                        'start: %s. Continuing without tracing.',
                        ENABLE_VAR, exc)
        finally:
            _state['setup'] = True
    return _state['active']


def _build_client():
    missing = [v for v in _CRED_VARS if not os.environ.get(v)]
    if missing:
        raise ValueError('missing Langfuse credentials: '
                         + ', '.join(missing))
    from langfuse import get_client
    return get_client()


def _instrument():
    from openinference.instrumentation.openai_agents import (
        OpenAIAgentsInstrumentor)
    instrumentor = OpenAIAgentsInstrumentor()
    # instrument() replaces the Agents SDK processors; only do it once.
    if not getattr(instrumentor, 'is_instrumented_by_opentelemetry', False):
        instrumentor.instrument()


@contextlib.contextmanager
def trace_run(instruction, metadata=None, session_id=None, tags=None,
              name='do!'):
    """Nest one do! run's whole agent trace under a single Langfuse
    observation carrying the instruction, mode metadata, and (later) the
    final answer. A no-op context manager yielding None when inactive.

    Must be entered on the same thread that runs the agent loop: the
    OpenInference root span inherits the active OTEL context, so the
    instrumentor's spans nest under this observation only when it is the
    current span on that thread.
    """
    client = _state['client']
    if not _state['active'] or client is None:
        yield None
        return
    stack = contextlib.ExitStack()
    try:
        prop = _propagate(session_id=session_id, tags=tags)
        if prop is not None:
            stack.enter_context(prop)
        span = stack.enter_context(client.start_as_current_observation(
            name=name, as_type='agent', input=instruction,
            metadata=metadata or {}))
    except Exception:  # tracing must never break the run
        log.debug('trace_run could not open an observation', exc_info=True)
        stack.close()  # unwind anything already entered
        yield None
        return
    try:
        yield span
    finally:
        try:
            stack.close()
        except Exception:
            log.debug('trace_run could not close its observation',
                      exc_info=True)


def otlp_trace_target():
    """Langfuse's OTLP/HTTP trace endpoint and auth header, or None.

    For a subprocess that owns its own OTel pipeline (the Codex runtime)
    and must be told where to ship. Deliberately a pure reading of the
    environment rather than of `_state`: the target has to be resolvable
    before `setup()` has run, and in a process where it never will.

    Returns `(endpoint, {header: value})`. The header carries a
    credential - keep it out of anything that reaches a command line, a
    log, or a trace.
    """
    if not is_enabled():
        return None
    if [v for v in _CRED_VARS if not os.environ.get(v)]:
        return None
    base = (os.environ.get('LANGFUSE_BASE_URL')
            or os.environ.get('LANGFUSE_HOST') or 'https://cloud.langfuse.com')
    base = base.strip().strip('"').strip("'").rstrip('/')
    if '://' not in base:
        base = 'https://' + base
    token = base64.b64encode(
        (f"{os.environ['LANGFUSE_PUBLIC_KEY']}:"
         f"{os.environ['LANGFUSE_SECRET_KEY']}").encode('utf-8')).decode()
    return base + '/api/public/otel/v1/traces', {
        'Authorization': 'Basic ' + token}


def traceparent():
    """A W3C trace-context carrier for this thread's current span, or None.

    The counterpart to `capture_context` for a subprocess rather than a
    thread: handed to a runtime that speaks W3C trace context, it makes
    that runtime's own spans children of this run instead of a separate
    trace. Keys are `traceparent` (and `tracestate` when set).
    """
    if not _state['active']:
        return None
    try:
        from opentelemetry.trace.propagation.tracecontext import (
            TraceContextTextMapPropagator)
        carrier = {}
        TraceContextTextMapPropagator().inject(carrier)
        if not carrier:
            return None
        parent = _plain_flags(carrier.get('traceparent'))
        if parent:
            carrier['traceparent'] = parent
        return carrier
    except Exception:
        log.debug('could not build a trace carrier', exc_info=True)
        return None


def _plain_flags(traceparent):
    """`traceparent` with only the sampled bit left in its flags.

    LANDMINE, measured against Codex 0.144.4: this SDK emits flags `03`
    (sampled + the W3C level-2 random-trace-id hint), and that runtime's
    carrier parser rejects the whole header rather than ignoring the bit
    it does not know - silently, so the spans simply start their own
    trace instead of joining ours. The spec says unknown flags must be
    ignored; a strict reader is not ours to fix, and the hint is advisory,
    so it is dropped. Same header with `01` nests correctly.
    """
    if not traceparent:
        return None
    fields = traceparent.split('-')
    if len(fields) != 4 or fields[0] != '00':
        return traceparent           # not a shape we understand; pass through
    try:
        flags = int(fields[3], 16)
    except ValueError:
        return traceparent
    fields[3] = '01' if flags & 0x01 else '00'
    return '-'.join(fields)


def capture_context():
    """Snapshot the current OTel context so another thread can nest under it.

    Needed because tool callbacks run on transport worker threads while the
    run span was opened on the backend's own thread; without this, a child
    observation would start a second, orphaned trace.
    """
    if not _state['active']:
        return None
    try:
        from opentelemetry import context as otel_context
        return otel_context.get_current()
    except Exception:
        log.debug('could not capture the tracing context', exc_info=True)
        return None


@contextlib.contextmanager
def child_span(name, context=None, input=None, metadata=None,
               as_type='tool'):
    """One nested observation, for backends the instrumentor cannot see.

    A no-op yielding None when tracing is inactive. Not used on the
    Agents-SDK path, where the instrumentor already emits tool spans.
    """
    client = _state['client']
    if not _state['active'] or client is None:
        yield None
        return
    stack = contextlib.ExitStack()
    token = None
    try:
        if context is not None:
            from opentelemetry import context as otel_context
            token = otel_context.attach(context)
        span = stack.enter_context(client.start_as_current_observation(
            name=name, as_type=as_type, input=input,
            metadata=metadata or {}))
    except Exception:
        log.debug('child_span could not open an observation', exc_info=True)
        stack.close()
        _detach(token)
        yield None
        return
    try:
        yield span
    finally:
        try:
            stack.close()
        finally:
            _detach(token)


def _detach(token):
    if token is None:
        return
    try:
        from opentelemetry import context as otel_context
        otel_context.detach(token)
    except Exception:
        log.debug('could not detach the tracing context', exc_info=True)


def _propagate(session_id=None, tags=None):
    kwargs = {}
    if session_id:
        kwargs['session_id'] = session_id
    if tags:
        kwargs['tags'] = list(tags)
    if not kwargs:
        return None
    try:
        from langfuse import propagate_attributes
        return propagate_attributes(**kwargs)
    except Exception:
        log.debug('propagate_attributes unavailable', exc_info=True)
        return None


def set_output(span, output):
    """Attach the run's final answer to the wrapping observation, if any."""
    if span is None or output is None:
        return
    try:
        span.update(output=output)
    except Exception:
        log.debug('could not set trace output', exc_info=True)


def set_metadata(span, metadata):
    """Add provider-side facts (status, counts) to an observation.

    Never account identifiers, login challenges, or credentials: those must
    not exist in a trace at all.
    """
    if span is None or not metadata:
        return
    try:
        span.update(metadata=dict(metadata))
    except Exception:
        log.debug('could not set trace metadata', exc_info=True)


def flush():
    """Force a final export at the end of a run. Never raises."""
    client = _state['client']
    if _state['active'] and client is not None:
        try:
            client.flush()
        except Exception:
            log.debug('Langfuse flush failed', exc_info=True)


def _reset_for_tests():
    """Drop cached setup state (not the global OTEL instrumentation)."""
    with _lock:
        _state.update(setup=False, active=False, client=None,
                      instrumented=False)

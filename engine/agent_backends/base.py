#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""base.py - the provider-neutral request/outcome/cancellation contract.

One do! run is: a `DoSession`, a prompt, a canonical `ToolBinding` list, and
a backend that hands those to some provider. This module owns the vocabulary
the orchestrator and every backend share, and nothing else - no provider
client, no ledger access, no mathematics.

Cancellation is part of that contract, not notebook polish. An agent that
keeps calling tactics after the user pressed Stop can append ledger steps,
consume provider quota, and designate a result after the cell has visibly
ended. So a run is addressable: one `CancellationToken` per run, one handle
that can be cancelled idempotently, and a bounded wait that never returns to
the prompt while an uncontained worker could still mutate the session.
"""
import json
import logging
import threading
import time
from dataclasses import dataclass, field, replace

log = logging.getLogger('toymath.agent')

# --- run outcome statuses --------------------------------------------------
COMPLETED = 'completed'
INTERRUPTED = 'interrupted'
BUDGET_EXHAUSTED = 'budget_exhausted'
CAPABILITY_VIOLATION = 'capability_violation'
FAILED = 'failed'

# --- cancellation reasons --------------------------------------------------
USER = 'user'
BUDGET = 'budget'
CAPABILITY = 'capability'

_STATUS_BY_REASON = {
    USER: INTERRUPTED,
    BUDGET: BUDGET_EXHAUSTED,
    CAPABILITY: CAPABILITY_VIOLATION,
}

# how long a cancelled run may take to acknowledge before ToyMath stops
# waiting and contains it instead. A cleanup deadline, not a run timeout.
DEFAULT_GRACE_PERIOD = 3.0
POLL_INTERVAL = 0.05


class DoAgentError(Exception):
    """Configuration/runtime error of the do! endpoint (not a math error)."""


class ToolArgumentError(ValueError):
    """A tool call whose arguments do not satisfy the binding's schema."""


class UnknownToolError(Exception):
    """A tool call naming something outside the canonical binding list."""


def status_for_reason(reason):
    """Map a cancellation reason to the outcome status it produces."""
    return _STATUS_BY_REASON.get(reason, INTERRUPTED)


# ---------------------------------------------------------------------------
# cancellation
# ---------------------------------------------------------------------------

class CancellationToken(object):
    """One run's stop signal: thread-safe, idempotent, first reason wins.

    Listeners registered with `add_listener` run exactly once, on whichever
    thread cancels first (the kernel thread for a Jupyter interrupt, a tool
    worker for a budget stop). The orchestrator uses them to close the
    session and cancel the provider call from one place regardless of who
    noticed first.
    """

    def __init__(self):
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._reason = None
        self._listeners = []

    @property
    def cancelled(self):
        return self._event.is_set()

    @property
    def reason(self):
        return self._reason

    def cancel(self, reason=USER):
        """Cancel the run. Returns True only for the call that won."""
        with self._lock:
            if self._reason is not None:
                return False
            self._reason = reason or USER
            listeners = list(self._listeners)
        self._event.set()
        for listener in listeners:
            try:
                listener(self._reason)
            except Exception:  # a listener must never break cancellation
                log.debug('cancellation listener failed', exc_info=True)
        return True

    def add_listener(self, listener):
        """Register a `listener(reason)`; fires immediately if already
        cancelled, so registration can never lose a race."""
        with self._lock:
            reason = self._reason
            if reason is None:
                self._listeners.append(listener)
        if reason is not None:
            listener(reason)

    def wait(self, timeout=None):
        return self._event.wait(timeout)


# ---------------------------------------------------------------------------
# the canonical tool surface
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolBinding:
    """One model-visible ToyMath tool, in no provider's dialect.

    Backends convert this to their own tool representation and nothing
    else: the name, description, schema, and handler have exactly one
    definition per run.
    """
    name: str
    description: str
    input_schema: dict
    invoke: object          # Callable[[dict], str]


@dataclass(frozen=True)
class AgentBudget:
    """Provider-neutral harness limits. `None` means unlimited.

    Deliberately not expressed in Agents-SDK turns: a Codex dynamic tool
    call and an SDK turn do not count the same events, and pretending they
    do would mislead. Turn limits stay an OpenRouter implementation detail.
    """
    max_tool_calls: int | None = None
    max_seconds: float | None = None


def validate_arguments(schema, payload):
    """Validate/normalize decoded tool arguments against a binding schema.

    Returns the keyword payload to call the handler with. Missing optional
    arguments fall back to their schema default, and an explicit null is
    read as omission - the model-visible schema has no nullable field.
    """
    if not isinstance(payload, dict):
        raise ToolArgumentError('arguments must be a JSON object')
    properties = schema.get('properties') or {}
    required = set(schema.get('required') or ())
    arguments = {}
    for name, spec in properties.items():
        value = payload.get(name)
        if name in payload and value is not None:
            arguments[name] = _coerce(name, spec, value)
        elif 'default' in spec:
            arguments[name] = spec['default']
        elif name in required:
            raise ToolArgumentError(f'missing required argument {name!r}')
    return arguments


def _coerce(name, spec, value):
    kind = spec.get('type')
    if kind == 'string':
        if not isinstance(value, str):
            raise ToolArgumentError(f'{name!r} must be a string')
        return value
    if kind == 'array':
        if not isinstance(value, list):
            raise ToolArgumentError(f'{name!r} must be an array')
        if ((spec.get('items') or {}).get('type') == 'string'
                and not all(isinstance(item, str) for item in value)):
            raise ToolArgumentError(f'{name!r} must contain only strings')
        return list(value)
    return value


class ToolDispatcher(object):
    """Executes canonical bindings for one run, under one policy.

    The cancellation token is consulted before validation, immediately
    before the handler runs, and after it returns, so no new tool work
    starts once Stop has been observed and a late result cannot be
    presented as if the run were still live. A committed ledger step is
    never rolled back - it won the session lock before closure or it never
    happened.
    """

    def __init__(self, bindings, cancellation=None, budget=None,
                 serialize=False):
        self.bindings = {binding.name: binding for binding in bindings}
        self.cancellation = (cancellation if cancellation is not None
                             else CancellationToken())
        self.budget = budget if budget is not None else AgentBudget()
        # Codex serializes: tactic provenance depends on step order and
        # concurrency buys nothing for this narrow surface. The Agents SDK
        # keeps its thread-pool concurrency (the ledger lock is the guard).
        self._serial = threading.Lock() if serialize else None
        self._count_lock = threading.Lock()
        self.calls = 0

    @property
    def names(self):
        return tuple(self.bindings)

    def dispatch(self, name, arguments):
        """Run one tool call and return its string result.

        `arguments` is a decoded object or its raw JSON text. Raises
        UnknownToolError for a name outside the surface and
        ToolArgumentError for a malformed call - both are transport-level
        failures, unlike a ToyMath record that ran and answered
        `{"ok": false}`.
        """
        binding = self.bindings.get(name)
        if binding is None:
            raise UnknownToolError(name)
        if self.cancellation.cancelled:
            return self._stopped(name)
        arguments = _decode(arguments)
        payload = validate_arguments(binding.input_schema, arguments)
        if not self._admit():
            return self._stopped(name)
        if self.cancellation.cancelled:
            return self._stopped(name)
        if self._serial is not None:
            with self._serial:
                # a queued call waited here while the run was stopped: the
                # checks above were made before the lock, so this is the
                # only place that can see a cancellation that landed while
                # the previous handler was still running
                if self.cancellation.cancelled:
                    return self._stopped(name)
                result = binding.invoke(payload)
        else:
            result = binding.invoke(payload)
        if self.cancellation.cancelled:
            # the handler finished after Stop: any step it committed before
            # session closure stands, but its answer never reaches the model
            return self._stopped(name)
        return result

    def _admit(self):
        """Count this call against the tool budget; False when exhausted."""
        limit = self.budget.max_tool_calls
        with self._count_lock:
            self.calls += 1
            over = limit is not None and self.calls > limit
        if over:
            self.cancellation.cancel(BUDGET)
        return not over

    def _stopped(self, name):
        reason = self.cancellation.reason or USER
        return json.dumps({
            'ok': False, 'op': name, 'cancelled': True, 'reason': reason,
            'error': ('this run was stopped (' + reason + '); the session '
                      'is closed to further steps'),
        }, ensure_ascii=False)


def _decode(arguments):
    if isinstance(arguments, (bytes, bytearray)):
        arguments = arguments.decode('utf-8', 'replace')
    if isinstance(arguments, str):
        text = arguments.strip()
        if not text:
            return {}
        try:
            return json.loads(text)
        except ValueError as exc:
            raise ToolArgumentError(f'arguments are not valid JSON: {exc}')
    return arguments


# ---------------------------------------------------------------------------
# request / outcome / handle
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AgentRequest:
    """Everything a backend needs, in no provider's vocabulary."""
    instruction: str
    developer_instructions: str
    dispatcher: ToolDispatcher
    cancellation: CancellationToken
    model: str | None = None
    budget: AgentBudget = AgentBudget()
    trace_metadata: dict = field(default_factory=dict)

    @property
    def bindings(self):
        return tuple(self.dispatcher.bindings.values())


@dataclass(frozen=True)
class AgentOutcome:
    """How one provider run ended. Never the mathematics - the ledger
    holds that, and `final_text` is unverified narrative."""
    status: str = COMPLETED
    final_text: str = ''
    error: str | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def cancelled(self):
        return self.status in (INTERRUPTED, BUDGET_EXHAUSTED,
                               CAPABILITY_VIOLATION)


class ThreadedRunHandle(object):
    """Base run handle for a backend that drives its provider on a private
    worker thread. Subclasses implement `run`, and may implement
    `request_cancel` (cooperative stop) and `abandon` (containment once the
    grace deadline passes).

    The worker stays a daemon thread, but it is no longer treated as safely
    disposable: the orchestrator closes the session before returning, so a
    worker that outlives its cell cannot mutate anything.
    """

    def __init__(self, request, name='toymath-do'):
        self.request = request
        self.cancellation = request.cancellation
        self._done = threading.Event()
        self._outcome = None
        self._lock = threading.RLock()
        self._cancel_requested = False
        self._abandoned = False
        self._thread = threading.Thread(target=self._main, name=name,
                                        daemon=True)

    # -- subclass hooks ----------------------------------------------------
    def run(self):  # pragma: no cover - abstract
        raise NotImplementedError

    def request_cancel(self, reason):
        """Ask the provider to stop this run. Must be safe to call from
        another thread, and is only ever called once."""

    def abandon(self, reason):
        """Contain a run that missed the cancellation grace deadline."""

    # -- handle protocol ---------------------------------------------------
    def start(self):
        try:
            self._thread.start()
        except KeyboardInterrupt:
            # Thread.start() blocks until the worker signals, so a Stop
            # pressed in that instant is delivered here - with the run
            # already executing. Never lose it: the token is set now, and
            # the orchestrator's listener closes the session as soon as it
            # registers against an already-cancelled token.
            self.cancel(USER)
        return self

    def wait(self, timeout=None):
        """Return the outcome, or None if it is still running."""
        if not self._done.wait(timeout):
            return None
        return self._outcome

    def cancel(self, reason=USER):
        """Idempotent, scoped to this run only."""
        with self._lock:
            if self._cancel_requested:
                return False
            self._cancel_requested = True
        self.cancellation.cancel(reason)
        try:
            self.request_cancel(reason)
        except Exception:  # containment continues regardless
            log.debug('provider cancellation failed', exc_info=True)
        return True

    def contain(self, reason=USER):
        """Escalate past the grace deadline. Idempotent."""
        with self._lock:
            if self._abandoned:
                return False
            self._abandoned = True
        try:
            self.abandon(reason)
        except Exception:
            log.debug('run containment failed', exc_info=True)
        return True

    @property
    def cancel_requested(self):
        with self._lock:
            return self._cancel_requested

    @property
    def running(self):
        return not self._done.is_set()

    def _main(self):
        try:
            outcome = self.run()
            if outcome is None:
                outcome = AgentOutcome(
                    status=FAILED, error='backend returned no outcome')
        except BaseException as exc:  # surfaced as a failed run, never lost
            outcome = AgentOutcome(status=FAILED,
                                   error=f'{type(exc).__name__}: {exc}')
        self._outcome = outcome
        self._done.set()


def wait_interruptibly(handle, cancellation, grace_period=None,
                       poll=POLL_INTERVAL, budget=None, clock=time.monotonic):
    """Wait for a run, honouring Jupyter Stop and the wall-clock budget.

    Replaces the unbounded `thread.join()`: the kernel thread polls, so a
    KeyboardInterrupt is delivered promptly, and the run is then given only
    a bounded grace period to acknowledge before it is contained. If the
    provider finished before cancellation was accepted, the completed
    outcome stands; if cancellation won, no late completion can change it.
    """
    grace = DEFAULT_GRACE_PERIOD if grace_period is None else grace_period
    limit = (budget or AgentBudget()).max_seconds
    deadline = clock() + limit if limit else None
    cancel_deadline = None
    while True:
        # the whole body is guarded: a kernel interrupt is delivered at
        # whatever bytecode the polling thread happens to be executing, not
        # only inside the wait
        try:
            # deadlines are settled before the wait, not after it: a Stop
            # delivered on every poll would otherwise abort the iteration
            # at `handle.wait` each time and never reach containment at all
            if (deadline is not None and clock() >= deadline
                    and not cancellation.cancelled):
                handle.cancel(BUDGET)
            # whoever stopped the run - Stop, the tool budget, the wall
            # clock, a capability violation - gets the same bounded cleanup.
            # Armed exactly once: renewing it every poll would let a
            # provider that ignores cancellation stall here forever.
            if cancellation.cancelled and cancel_deadline is None:
                cancel_deadline = clock() + grace
            if cancel_deadline is not None and clock() >= cancel_deadline:
                reason = cancellation.reason or USER
                handle.contain(reason)
                return AgentOutcome(status=status_for_reason(reason),
                                    error=('the run did not acknowledge '
                                           'cancellation within '
                                           f'{grace:g}s and was contained'),
                                    metadata={'grace_exceeded': True})
            outcome = handle.wait(poll)
            if outcome is not None:
                return _reconcile(outcome, cancellation)
        except KeyboardInterrupt:
            finished = handle.wait(0)
            if finished is not None:
                return finished         # the provider won the race
            handle.cancel(USER)
            if cancel_deadline is None:  # Stop pressed twice must not extend
                cancel_deadline = clock() + grace


def _reconcile(outcome, cancellation):
    """A cancelled run stays cancelled, whatever the worker finally said."""
    if not cancellation.cancelled:
        return outcome
    if outcome.status == CAPABILITY_VIOLATION:
        return outcome              # the stronger verdict survives
    status = status_for_reason(cancellation.reason)
    if outcome.status == status:
        return outcome
    return replace(outcome, status=status, final_text='')

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""agent_config.py - which agent backend a notebook runs on, and why.

Routing is notebook-local, exactly like model selection: one kernel can run
`do!` against OpenRouter while another uses the user's own Codex
subscription. Resolution happens once, before a run.

The one rule that matters more than convenience: a run never fails over.
A Codex rate limit must not silently create OpenRouter charges, and an
OpenRouter outage must not silently consume a user's Codex allowance. When
the selected backend cannot run, the cell says so.
"""
import os
import re
from dataclasses import dataclass, field

from agent_backends.base import DoAgentError

AUTO = 'auto'
OPENROUTER = 'openrouter'
CODEX = 'codex'
BACKENDS = (AUTO, OPENROUTER, CODEX)
SELECTABLE = (OPENROUTER, CODEX)

BACKEND_VAR = 'TOYMATH_AGENT_BACKEND'
OPENROUTER_KEY_VAR = 'OPEN_ROUTER'

#: Codex is an opt-in experiment while its dynamic-tool surface is
#: experimental upstream; the label is part of the UI, not decoration.
EXPERIMENTAL = (CODEX,)


@dataclass(frozen=True)
class AgentRoute:
    """One notebook's agent routing: backend, model, provider order."""
    backend: str = AUTO
    model: str | None = None
    providers: tuple = field(default_factory=tuple)

    def with_backend(self, backend):
        # the model is backend-specific: an OpenRouter id means nothing to
        # Codex, so switching returns to the new backend's own default
        return AgentRoute(backend=backend, model=default_model(backend),
                          providers=())

    def with_model(self, model, providers=()):
        return AgentRoute(backend=self.backend, model=model,
                          providers=tuple(providers))

    @property
    def experimental(self):
        return self.backend in EXPERIMENTAL


@dataclass(frozen=True)
class Resolution:
    backend: str
    reason: str


def check(backend):
    if backend not in BACKENDS:
        raise ValueError(
            f'unknown backend {backend!r}; use ' + ', '.join(BACKENDS))
    return backend


def default_model(backend):
    """The model a backend uses when the notebook has not chosen one."""
    if backend == CODEX:
        # prefer the Codex runtime/account default over a model id hard-coded
        # into ToyMath
        return os.environ.get('TOYMATH_CODEX_MODEL') or None
    if backend == OPENROUTER:
        import model_config
        return model_config.current_model()
    return None


#: Last observed managed-account state, so display can agree with routing
#: without starting a Codex runtime of its own. None means "not yet looked".
_codex_account = {'usable': None}


def note_codex_account(status):
    """Remember what an account read actually said.

    `preview` must not start a runtime, but it must not contradict `resolve`
    either: once anything has genuinely observed the account - a `login!`, a
    `login! status`, a run that probed - display agrees with routing.
    """
    _codex_account['usable'] = (None if status is None
                                else bool(status.usable))
    return status


def forget_codex_account():
    """Drop the cached observation (sign-out, or a test)."""
    _codex_account['usable'] = None


def _codex_usable(probe=True):
    """Whether a managed ChatGPT Codex account is authenticated here.

    Never raises and never starts an interactive login: auto-selection may
    consult this, but it must not surprise the user with a browser window
    or an error from an unrelated backend. With `probe=False` it answers
    only from what has already been observed, so display costs nothing.
    """
    try:
        from agent_backends import codex
        if not codex.available():
            return False
        if not probe:
            return bool(_codex_account['usable'])
        # `usable`, not `logged_in`: an API-key or Bedrock account must not
        # be auto-selected into spending on a ToyMath derivation
        return note_codex_account(codex.account_status()).usable
    except Exception:
        return False


#: `login!` options and the type label the completer shows for each. This is
#: the one list: `MathShell.exec_login` validates against its keys, so a new
#: option cannot appear in the popup without being accepted, or the reverse.
#: The bare `login!` (browser sign-in) takes no argument and so completes to
#: nothing.
LOGIN_ACTIONS = {
    'device': 'sign-in mode',
    'status': 'account query',
    'logout': 'account action',
}

_BACKEND_COMPLETION_RE = re.compile(r'^[ \t]*backend![ \t]+(.*)$')
_LOGIN_COMPLETION_RE = re.compile(r'^[ \t]*login![ \t]+(.*)$')


def complete_backend_command(code, cursor_pos):
    """Return a Jupyter ``complete_reply`` for ``backend!`` or ``None``."""
    return _complete_words(
        code, cursor_pos, _BACKEND_COMPLETION_RE,
        {name: ('backend (experimental)' if name in EXPERIMENTAL
                else 'backend') for name in BACKENDS})


def complete_login_command(code, cursor_pos):
    """Return a Jupyter ``complete_reply`` for ``login!`` or ``None``."""
    return _complete_words(code, cursor_pos, _LOGIN_COMPLETION_RE,
                           LOGIN_ACTIONS)


def _complete_words(code, cursor_pos, pattern, options):
    """One reply for a fixed word list after a `name! ` cell prefix.

    `options` maps each completion to the type label the popup shows. The
    returned range replaces only the token at the cursor, so accepting a
    completion leaves a replayable cell behind.
    """
    cursor_pos = max(0, min(cursor_pos, len(code)))
    line_start = code.rfind('\n', 0, cursor_pos) + 1
    line = code[line_start:cursor_pos]
    match = pattern.match(line)
    if match is None:
        return None
    token = match.group(1)
    leading = len(token) - len(token.lstrip())
    matches = [name for name in options
               if name.startswith(token.strip().lower())]
    return {
        'matches': matches,
        'cursor_start': line_start + match.start(1) + leading,
        'cursor_end': cursor_pos,
        'metadata': {
            '_jupyter_types_experimental': [
                {'text': name, 'type': options[name]} for name in matches
            ],
        },
        'status': 'ok',
    }


def codex_models():
    """The models this Codex account offers.

    Dynamic, unlike the OpenRouter catalog: `models.yaml` stays a static
    OpenRouter file rather than becoming a mixed catalog.
    """
    from agent_backends import codex
    codex.require_available()
    return codex.list_models()


def resolve(route, probe=True):
    """Resolve `route` to the backend one run will use, with its reason."""
    backend = check(getattr(route, 'backend', route) or AUTO)
    if backend != AUTO:
        return Resolution(backend, 'selected in this notebook')
    configured = (os.environ.get(BACKEND_VAR) or '').strip().lower()
    if configured in SELECTABLE:
        return Resolution(configured, f'{BACKEND_VAR} is set to {configured}')
    if os.environ.get(OPENROUTER_KEY_VAR):
        # existing installations keep working exactly as before
        return Resolution(OPENROUTER,
                          f'{OPENROUTER_KEY_VAR} is configured')
    if _codex_usable(probe=probe):
        return Resolution(CODEX, 'a Codex account is signed in on this '
                                 'machine')
    raise DoAgentError(
        'no agent backend is configured: run `login!` to use your own Codex '
        f'account (experimental), or put an OpenRouter key in '
        f'.env as {OPENROUTER_KEY_VAR}')


def preview(route):
    """Resolution for display, without starting a Codex runtime.

    Answers from the cached account observation rather than re-probing, so
    a signed-in user is not told the notebook is unconfigured while `do!`
    would in fact route to Codex.
    """
    try:
        return resolve(route, probe=False)
    except DoAgentError:
        return Resolution(AUTO, 'no backend is configured yet')


def describe(route):
    """A short, honest routing summary: backend, why, model."""
    resolution = preview(route)
    model = route.model or default_model(resolution.backend)
    return {
        'backend': resolution.backend,
        'reason': resolution.reason,
        'model': model,
        'providers': tuple(route.providers),
        'experimental': resolution.backend in EXPERIMENTAL,
    }

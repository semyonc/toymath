#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""codex.py - the experimental personal-Codex backend.

Opt-in and local: the user signs into their own Codex account, and the run
consumes their own entitlement. ToyMath never handles a raw ChatGPT token,
never proxies model traffic, and never shares an account.

The trust boundary is unchanged. Codex chooses strategy and calls tools; the
tactic registry, the independent numeric oracle, and the ledger remain the
only authority for a mechanically checked result. Codex prose is narrative
and can never become a designated value.

Capability containment is the part that needs saying out loud. Codex is
normally a coding agent, so the thread is created with an empty working
directory outside the repository, a read-only sandbox, no network, and every
available capability switch off. The pinned 0.144.4 runtime still leaves
three native tools visible - `update_plan`, `request_user_input`, and
`view_image` - and offers no endpoint that reports the effective tool list.
ToyMath accepts that residual exposure for this experimental backend under
four independent controls: the generated role policy, the isolated runtime,
an exact model-visible tool-set contract test, and this client, which
refuses every tool call outside the canonical surface and invalidates any
run that reaches for a native one.
"""
import json
import logging
import os
import re
import threading
import tomllib

from agent_backends.base import (CAPABILITY, CAPABILITY_VIOLATION, COMPLETED,
                                 FAILED, INTERRUPTED, USER, AgentOutcome,
                                 DoAgentError, ThreadedRunHandle,
                                 ToolArgumentError, UnknownToolError)
from agent_backends import codex_transport
from agent_backends.codex_transport import (AppServerTransport, CodexError,
                                            CodexUnavailable)

log = logging.getLogger('toymath.agent.codex')

# The native tools the pinned runtime leaves visible with a clean dedicated
# home. Visible, and behaviourally prohibited: a run that invokes one is
# interrupted and produces no result. Anything beyond this set fails closed
# rather than being quietly accepted.
RESIDUAL_NATIVE_TOOLS = ('update_plan', 'request_user_input', 'view_image')

# the exact SDK/runtime pair the containment contract was validated against
PINNED_SDK_VERSION = '0.144.4'

#: ToyMath is installed from a clone, not PyPI, so the extra is named
#: against the project directory - `toymath[codex]` would not resolve.
INSTALL_HINT = 'uv pip install ".[codex]" from the ToyMath checkout'
HOME_VAR = 'TOYMATH_CODEX_HOME'
MODEL_VAR = 'TOYMATH_CODEX_MODEL'
POLICY_FILE = 'AGENTS.md'
POLICY_HEADER = '# ToyMath Codex role'
CONFIG_FILE = 'config.toml'
WORKDIR = 'workdir'

#: Marks a directory as ToyMath's own Codex home. `ensure_home` overwrites
#: the policy file, so it must be able to tell its home from a directory
#: that merely got named in TOYMATH_CODEX_HOME by mistake.
OWNER_MARKER = '.toymath-codex-home'
OWNER_NOTE = ('This directory is the ToyMath Codex home. It is created and\n'
              'rewritten by ToyMath; do not keep anything of your own here.\n')

#: TOML bare keys, the only server names a dotted `--config` override can
#: address unambiguously
_BARE_KEY = re.compile(r'^[A-Za-z0-9_-]+$')

#: An interrupted login cleans up on the Stop path, where the transport's
#: ordinary request timeout would hold the kernel far too long.
LOGIN_CANCEL_TIMEOUT = 5.0

# Every capability switch the pinned runtime offers, off. The residual three
# survive this list - that is the documented, accepted exception - but shell,
# patch, web, apps, plugins, and subagents do not.
#
# MCP is deliberately NOT here: it has no global switch in this runtime.
# Measured against the real binary, `mcp_servers={}` does not clear an
# inherited table (overrides merge into the config rather than replacing it),
# and only `mcp_servers.<name>.enabled=false` works - so MCP is contained by
# `mcp_overrides()`, per server, by name.
CONTAINMENT_OVERRIDES = (
    'approval_policy="never"',
    'sandbox_mode="read-only"',
    'web_search="disabled"',
    'tools.web_search=false',
    'tools.view_image=false',
    'apps._default.enabled=false',
    'features.shell_tool=false',
    'features.unified_exec=false',
    'features.apps=false',
    'features.multi_agent=false',
    'features.hooks=false',
    'features.plugins=false',
    'features.remote_plugin=false',
    'features.image_generation=false',
    'features.in_app_browser=false',
    'features.computer_use=false',
    'features.browser_use=false',
    'features.workspace_dependencies=false',
    'features.skill_mcp_dependency_install=false',
    'features.tool_suggest=false',
    # the runtime has its own OpenTelemetry pipeline (`none | statsig |
    # otlp-http | otlp-grpc`); a ToyMath thread ships nothing anywhere
    # unless that is an explicit, separate decision
    'otel.exporter="none"',
)

_runtime_lock = threading.RLock()
_runtime = {'transport': None}


# ---------------------------------------------------------------------------
# availability
# ---------------------------------------------------------------------------

def runtime_binary():
    """The Codex binary pinned by the optional extra.

    Deliberately never the `codex` on PATH: the contract is with the
    runtime the extra pins and the containment tests were run against, not
    with whatever version the user's CLI happens to be.
    """
    try:
        import codex_cli_bin
    except ImportError:
        return None
    try:
        return str(codex_cli_bin.bundled_codex_path())
    except Exception:
        return None


def available():
    return runtime_binary() is not None


def require_available():
    if not available():
        raise DoAgentError(
            'the experimental Codex backend needs its optional extra: '
            + INSTALL_HINT)


# ---------------------------------------------------------------------------
# the dedicated ToyMath Codex home
# ---------------------------------------------------------------------------

def home_path():
    """Where ToyMath keeps its own Codex home.

    Never `~/.codex`: the general CLI/desktop home carries MCP servers,
    plugins, apps, and project instructions that would leak capabilities
    into a math agent. ToyMath creates its own, and never reads or copies
    the other one - including its `auth.json`. `TOYMATH_CODEX_HOME` may
    move it, but not onto the general home.
    """
    configured = os.environ.get(HOME_VAR)
    if not configured:
        return os.path.join(os.path.expanduser('~'), '.toymath', 'codex-home')
    home = os.path.abspath(os.path.expanduser(configured))
    if _same_path(home, general_home_path()):
        raise DoAgentError(
            f'{HOME_VAR} points at the general Codex home ({home}). ToyMath '
            'keeps a separate one so it neither inherits that home\'s '
            'capabilities nor overwrites its configuration - choose another '
            'directory.')
    return home


def general_home_path():
    """The Codex CLI/desktop home ToyMath must stay out of."""
    configured = os.environ.get('CODEX_HOME')
    if configured:
        return os.path.abspath(os.path.expanduser(configured))
    return os.path.join(os.path.expanduser('~'), '.codex')


def _same_path(left, right):
    try:
        return os.path.realpath(left) == os.path.realpath(right)
    except OSError:                                # pragma: no cover
        return os.path.normcase(left) == os.path.normcase(right)


def role_policy(tool_names=None):
    """The generated role policy, with or without the tool allowlist.

    Both forms reach the model on every run: the durable one from the
    dedicated home's global `AGENTS.md`, the per-thread one as
    `developerInstructions`. So only the per-thread form may enumerate.

    LANDMINE: the allowlist depends on the *session* - `plot` and `tikz`
    exist only when the figure sandboxes resolved - while the home file is
    written once, by whichever command happens to start the kernel's
    runtime. Baking names into it produced a global instruction listing
    seven tools while the thread offered nine, and a model that read both
    obeyed the stricter one: it refused to draw a diagram it had a working
    `tikz` tool for. The durable form therefore states the role and the
    prohibitions - which never vary - and defers the list to the thread.

    This is guidance to a model, not a security boundary. The boundaries
    are the isolated runtime, the exact tool-set contract, and a client
    that refuses anything outside the surface.
    """
    allowed = ('the ToyMath dynamic tools supplied with each thread'
               if tool_names is None else
               'the client-provided ToyMath dynamic tools: '
               + ', '.join(tool_names))
    forbidden = ', '.join(RESIDUAL_NATIVE_TOOLS)
    return f"""# ToyMath Codex role

You are the reasoning component of the ToyMath symbolic mathematics system.
Use only {allowed}.
Never call any other tool, including {forbidden}, shell, filesystem, web,
MCP, app, connector, or subagent tools.

Which ToyMath tools exist is decided per thread and stated there; treat that
list as authoritative. If it cannot perform an action, report the
limitation; do not try an alternative capability. Only mechanically checked
ToyMath tool results may be presented as derived mathematics. Model prose is
explanatory and is not verified.
"""


def ensure_home(home=None):
    """Create the dedicated home, its empty working directory, and the
    generated policy. Returns (home, cwd).

    Refuses to adopt a directory ToyMath did not create: `ensure_home`
    overwrites `AGENTS.md`, and a misdirected `TOYMATH_CODEX_HOME` must not
    silently rewrite somebody else's instructions.
    """
    home = home or home_path()
    _claim_home(home)
    workdir = os.path.join(home, WORKDIR)
    os.makedirs(home, mode=0o700, exist_ok=True)
    os.makedirs(workdir, mode=0o700, exist_ok=True)
    _write_if_changed(os.path.join(home, OWNER_MARKER), OWNER_NOTE)
    # the durable form: no allowlist, because this file outlives the session
    # that would define one (see role_policy)
    _write_if_changed(os.path.join(home, POLICY_FILE), role_policy())
    return home, workdir


def _write_if_changed(path, text):
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            if handle.read() == text:
                return
    except OSError:
        pass
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write(text)


def _claim_home(home):
    """Raise unless `home` is ToyMath's to write into."""
    if not os.path.exists(home):
        return
    if not os.path.isdir(home):
        raise DoAgentError(f'the Codex home {home} is not a directory')
    if _adoptable(home):
        return
    raise DoAgentError(
        f'{home} already exists and was not created by ToyMath. The Codex '
        'backend keeps its own home - it rewrites AGENTS.md there and would '
        'inherit that directory\'s configuration - so point '
        f'{HOME_VAR} at a new directory instead.')


def _adoptable(home):
    try:
        entries = set(os.listdir(home))
    except OSError as exc:
        raise DoAgentError(f'the Codex home {home} is unreadable: {exc}')
    if not entries or OWNER_MARKER in entries:
        return True                    # empty, or already marked as ours
    # a home created before the marker existed: our own generated policy
    # next to the working directory we make
    if WORKDIR not in entries or POLICY_FILE not in entries:
        return False
    try:
        with open(os.path.join(home, POLICY_FILE), encoding='utf-8') as handle:
            return handle.read().startswith(POLICY_HEADER)
    except OSError:
        return False


def mcp_overrides(home):
    """Config overrides that disable every MCP server this home configures.

    The containment claim has to be enforced, not asserted. Measured against
    the pinned runtime, a home carrying one `mcp_servers` entry adds four
    model-visible tools (the server's own, plus `list_mcp_resources`,
    `list_mcp_resource_templates`, and `read_mcp_resource`), and a
    whole-table override does not remove them - only naming each server
    does. So they are enumerated here and switched off individually.

    A server whose name cannot be expressed as a dotted override key is a
    hard error: leaving one enabled is not an option.
    """
    names = _configured_mcp_servers(home)
    unsafe = [name for name in names if not _BARE_KEY.match(name)]
    if unsafe:
        raise DoAgentError(
            'the Codex home configures MCP servers ToyMath cannot reliably '
            'disable (' + ', '.join(repr(n) for n in unsafe) + '); remove '
            f'them from {os.path.join(home, CONFIG_FILE)}')
    return tuple(f'mcp_servers.{name}.enabled=false' for name in sorted(names))


def _configured_mcp_servers(home):
    path = os.path.join(home, CONFIG_FILE)
    try:
        with open(path, 'rb') as handle:
            config = tomllib.load(handle)
    except FileNotFoundError:
        return ()
    except (OSError, tomllib.TOMLDecodeError) as exc:
        # unreadable config means an unknown tool surface; fail closed
        raise DoAgentError(f'the Codex home config {path} could not be read '
                           f'({exc}), so its MCP servers cannot be disabled')
    servers = config.get('mcp_servers')
    return tuple(servers) if isinstance(servers, dict) else ()


# ---------------------------------------------------------------------------
# the kernel-lifetime runtime
# ---------------------------------------------------------------------------

def open_transport(home=None, binary=None):
    """A started app-server transport against the dedicated home."""
    binary = binary or runtime_binary()
    if binary is None:
        require_available()
    home, workdir = ensure_home(home=home)
    transport = AppServerTransport(
        binary=binary, home=home, cwd=workdir,
        config_overrides=CONTAINMENT_OVERRIDES + mcp_overrides(home))
    return transport.start()


def runtime(factory=None):
    """The shared runtime for this kernel, created lazily.

    One process means one login state and immediate cancellation routing.
    A poisoned or dead runtime is replaced rather than reused.
    """
    with _runtime_lock:
        transport = _runtime['transport']
        if transport is not None and not transport.poisoned and (
                getattr(transport, 'running', True)):
            return transport
        if transport is not None:
            close_runtime()
        maker = factory or open_transport
        _runtime['transport'] = maker()
        return _runtime['transport']


def set_runtime(transport):
    """Install a runtime (the offline tests inject their transcript one)."""
    with _runtime_lock:
        _runtime['transport'] = transport
        return transport


def close_runtime():
    with _runtime_lock:
        transport = _runtime['transport']
        _runtime['transport'] = None
    if transport is not None:
        try:
            transport.close()
        except Exception:
            pass


def poison_runtime(reason='unresponsive'):
    with _runtime_lock:
        transport = _runtime['transport']
        _runtime['transport'] = None
    if transport is not None:
        try:
            transport.poison(reason)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# managed authentication (never a raw token)
# ---------------------------------------------------------------------------

def account_status():
    """The account this machine is signed into, or a logged-out status."""
    try:
        return runtime().account_read()
    except (CodexUnavailable, CodexError) as exc:
        raise DoAgentError(f'Codex is not reachable: {exc}')


def require_managed_account(status):
    """Refuse anything but the managed ChatGPT subscription.

    The runtime is happy to run on an API-key or Amazon Bedrock account,
    both of which report `logged_in`. Those bill a different credential -
    possibly an organization's - and this backend exists to spend the
    user's own ChatGPT entitlement. A wrong account fails loudly here
    rather than quietly costing money somewhere else.
    """
    if status.usable:
        return status
    if not status.logged_in:
        raise DoAgentError(
            'no Codex account is signed in on this machine - run `login!` '
            '(or `login! device`), or select the OpenRouter backend')
    raise DoAgentError(
        f'this Codex home is signed in with {status.auth_mode!r} '
        'authentication; the ToyMath Codex backend runs only on a managed '
        'ChatGPT account, so that credential is never spent here. Run '
        '`login! out` and then `login!`, or select the OpenRouter backend')


def login(mode='chatgpt', on_challenge=None, timeout=300.0,
          cancellation=None):
    """Run one managed ChatGPT login and return the resulting status.

    ToyMath sees a login challenge, a completion notification, and an
    account status - never an access or refresh token. The experimental
    externally managed `chatgptAuthTokens` mode is deliberately not
    implemented: ToyMath does not own the user's ChatGPT auth lifecycle.

    An interrupted login is cancelled through `account/login/cancel` so no
    challenge is left pending on the app-server.
    """
    transport = runtime()
    queue = transport.notifications()
    try:
        challenge = transport.login_start(mode)
    except (CodexUnavailable, CodexError) as exc:
        transport.release(queue)
        raise DoAgentError(f'could not start the Codex login: {exc}')
    if on_challenge is not None:
        on_challenge(challenge)
    try:
        if cancellation is not None:
            cancellation.add_listener(
                lambda reason: _cancel_login(transport, challenge.login_id))
        return transport.await_login(challenge.login_id, timeout=timeout,
                                     queue=queue)
    except KeyboardInterrupt:
        _cancel_login(transport, challenge.login_id)
        raise
    except (CodexUnavailable, CodexError) as exc:
        _cancel_login(transport, challenge.login_id)
        raise DoAgentError(f'the Codex login did not complete: {exc}')
    finally:
        transport.release(queue)


def _cancel_login(transport, login_id):
    """Discard a pending challenge, on a deadline of its own.

    This runs on the Stop path, where the ordinary request timeout would
    hold the kernel for a minute. A runtime that will not answer within
    LOGIN_CANCEL_TIMEOUT is not merely slow - it still holds an open
    challenge - so it is poisoned and replaced rather than trusted again.
    """
    try:
        transport.login_cancel(login_id, timeout=LOGIN_CANCEL_TIMEOUT)
    except CodexUnavailable:
        log.warning('the Codex runtime did not acknowledge the login '
                    'cancellation; replacing it')
        poison_runtime('login cancellation was not acknowledged')
    except Exception:          # the challenge is discarded either way
        pass


def logout():
    try:
        transport = runtime()
        transport.logout()
        return transport.account_read()
    except (CodexUnavailable, CodexError) as exc:
        raise DoAgentError(f'could not sign out of Codex: {exc}')


def list_models():
    try:
        return runtime().list_models()
    except (CodexUnavailable, CodexError) as exc:
        raise DoAgentError(f'could not list Codex models: {exc}')


# ---------------------------------------------------------------------------
# tool serialization: the canonical bindings in app-server dialect
# ---------------------------------------------------------------------------

def dynamic_tools(dispatcher):
    """Serialize the canonical bindings as app-server `dynamicTools`.

    Representation conversion only - the name, description, and schema are
    the ones every other backend advertises, so the model cannot see a
    different tool surface depending on who is running it.
    """
    return [{'type': 'function',
             'name': binding.name,
             'description': binding.description,
             'inputSchema': binding.input_schema}
            for binding in dispatcher.bindings.values()]


def expected_model_tools(dispatcher):
    """The exact model-visible tool set this backend will accept.

    The version contract: the canonical ToyMath tools plus the three
    reviewed residual native names. A new native tool, an inherited MCP
    helper, or a disappearing restriction fails this check - and the Codex
    backend is disabled - until the exception is reviewed again.
    """
    return tuple(dispatcher.names) + RESIDUAL_NATIVE_TOOLS


def capture_config_overrides(base_url, key_var='TOYMATH_CAPTURE_KEY'):
    """Runtime overrides that point the model provider at a loopback capture.

    The app-server has no endpoint that reports the complete model-visible
    tool list, so the only way to hold the containment claim to account is
    to look at the request the runtime actually sends. The version-contract
    test starts a local server, runs one turn against it, and compares the
    outgoing `tools` array with `expected_model_tools`.
    """
    return (
        'model="toymath-capture-model"',
        'model_provider="toymath_capture"',
        'model_providers.toymath_capture.name="ToyMath Capture"',
        f'model_providers.toymath_capture.base_url="{base_url}"',
        f'model_providers.toymath_capture.env_key="{key_var}"',
        'model_providers.toymath_capture.wire_api="responses"',
        'model_providers.toymath_capture.requires_openai_auth=false',
        'model_providers.toymath_capture.request_max_retries=0',
        'model_providers.toymath_capture.stream_max_retries=0',
    )


def tool_call_result(dispatcher, params):
    """Answer one `item/tool/call` request.

    The `success` flag is about the transport, not the mathematics. A
    ToyMath record that ran and answered `{"ok": false}` - a refused tactic,
    a provenance error, a check that disagreed - is a *successful* tool
    call whose content lets the model repair. Only an unknown tool, an
    invalid wire shape, or an unexpected handler failure is `success:
    false`.
    """
    if not isinstance(params, dict):
        return _failed('malformed item/tool/call: params must be an object')
    name = params.get('tool')
    if not isinstance(name, str) or not name:
        return _failed('malformed item/tool/call: no tool name')
    namespace = params.get('namespace')
    if namespace:
        return _failed(f'unknown tool namespace {namespace!r}; ToyMath '
                       'exposes only top-level function tools')
    try:
        text = dispatcher.dispatch(name, params.get('arguments'))
    except UnknownToolError:
        return _failed(
            f"'{name}' is not a ToyMath tool. Use only: "
            + ', '.join(dispatcher.names) + '.')
    except ToolArgumentError as exc:
        return _failed(f'{name}: {exc}')
    except Exception as exc:  # pragma: no cover - handler bug
        return _failed(f'{name} failed: {type(exc).__name__}: {exc}')
    return {'success': True,
            'contentItems': [{'type': 'inputText', 'text': text}]}


def _failed(message):
    return {'success': False,
            'contentItems': [{'type': 'inputText', 'text': message}]}


# ---------------------------------------------------------------------------
# backend
# ---------------------------------------------------------------------------

class CodexBackend(object):
    """Runs one instruction through the user's own Codex entitlement."""

    name = 'codex'
    #: provenance depends on step order, and concurrency buys nothing for
    #: this narrow surface
    serialize_tools = True

    def __init__(self, model=None, transport=None):
        self.model = model or os.environ.get(MODEL_VAR) or None
        self.transport = transport

    def describe_model(self):
        # with no explicit choice, the account's own default is used rather
        # than a model id hard-coded into ToyMath
        return self.model or 'codex default'

    def start(self, request):
        names = tuple(request.dispatcher.names)
        transport = self.transport
        if transport is None:
            require_available()
            transport = runtime()
        require_managed_account(transport.account_read())
        policy = role_policy(names)
        thread = codex_transport.CodexThreadRequest(
            instruction=request.instruction,
            # the same generated policy the dedicated home carries, so the
            # durable and per-thread instructions cannot drift apart
            developer_instructions=policy + '\n\n'
            + request.developer_instructions,
            dynamic_tools=tuple(dynamic_tools(request.dispatcher)),
            model=self.model)
        return CodexRunHandle(request, transport, thread).start()


class CodexRunHandle(ThreadedRunHandle):
    """One Codex turn, addressable for interruption and containment."""

    def __init__(self, request, transport, thread):
        super(CodexRunHandle, self).__init__(request, name='toymath-codex')
        self.transport = transport
        self.thread = thread
        self.thread_id = None
        self.turn_id = None
        self.violation = None
        self._trace_context = None

    def run(self):
        import observability
        metadata = dict(self.request.trace_metadata or {})
        metadata.update(backend=CodexBackend.name,
                        model=self.thread.model or 'default')
        with observability.trace_run(self.request.instruction,
                                     metadata=metadata) as span:
            # tool callbacks arrive on transport worker threads, so the
            # tracing context has to travel with them explicitly
            self._trace_context = observability.capture_context()
            outcome = self.transport.run_thread(
                self.thread, on_tool_call=self._tool_call,
                on_native_tool=self._native_tool, on_turn=self._turn_started)
            observability.set_output(span, outcome.final_text or None)
            observability.set_metadata(span, {
                'status': outcome.status,
                'tool_calls': len(outcome.tool_calls),
                'native_tool_calls': list(outcome.native_tool_calls),
            })
            return self._translate(outcome)

    # -- callbacks ---------------------------------------------------------
    def _turn_started(self, thread_id, turn_id):
        with self._lock:
            self.thread_id, self.turn_id = thread_id, turn_id
        if self.cancellation.cancelled:
            # Stop arrived while the thread was still being created
            self.request_cancel(self.cancellation.reason)

    def _tool_call(self, params):
        # The OpenInference instrumentor only sees Agents-SDK runs, so the
        # Codex path emits its own tool observations through the neutral
        # seam. Observability only: a tracing failure never reaches the
        # dispatcher, the ledger, or the oracle.
        import observability
        name = (params or {}).get('tool') if isinstance(params, dict) else None
        with observability.child_span(f'tool:{name or "?"}',
                                      context=self._trace_context,
                                      input=_trace_arguments(params)) as span:
            reply = tool_call_result(self.request.dispatcher, params)
            observability.set_output(span, _trace_output(reply))
            return reply

    def _native_tool(self, name):
        """A residual native tool was invoked. The run is over: interrupt
        the turn, close the session, and discard the final response. The
        steps already committed stay - they were mechanically checked - but
        the run designates nothing."""
        with self._lock:
            if self.violation is None:
                self.violation = name
        self.cancel(CAPABILITY)
        return True

    def _translate(self, outcome):
        metadata = {'thread': outcome.thread_id, 'turn': outcome.turn_id,
                    'tool_calls': list(outcome.tool_calls),
                    'native_tool_calls': list(outcome.native_tool_calls)}
        if self.violation is not None:
            return AgentOutcome(
                status=CAPABILITY_VIOLATION, metadata=metadata,
                error=(f'the model invoked the native {self.violation!r} '
                       'capability, which is outside the ToyMath tool '
                       'surface; the run was interrupted and designates no '
                       'result'))
        if outcome.status == codex_transport.TURN_INTERRUPTED:
            return AgentOutcome(status=INTERRUPTED, metadata=metadata)
        if outcome.status == codex_transport.TURN_FAILED:
            return AgentOutcome(status=FAILED, metadata=metadata,
                                error=outcome.error or 'the Codex turn failed')
        return AgentOutcome(status=COMPLETED, final_text=outcome.final_text,
                            metadata=metadata)

    # -- cancellation ------------------------------------------------------
    def request_cancel(self, reason=USER):
        with self._lock:
            thread_id, turn_id = self.thread_id, self.turn_id
        try:
            self.transport.interrupt_turn(thread_id, turn_id)
        except Exception:
            log.debug('turn/interrupt failed', exc_info=True)

    def abandon(self, reason=USER):
        """The turn missed its grace deadline. A shared runtime is poisoned
        and terminated; the next command lazily creates a fresh one. The
        session is already closed, so a late message lands nowhere."""
        transport = self.transport
        with _runtime_lock:
            shared = _runtime['transport'] is transport
        if shared:
            poison_runtime(f'cancellation was not acknowledged ({reason})')
        else:
            try:
                transport.close()
            except Exception:
                log.debug('could not close the Codex transport',
                          exc_info=True)


def _json(payload):  # pragma: no cover - debugging convenience
    return json.dumps(payload, ensure_ascii=False, default=str)


_TRACE_LIMIT = 2000


def _trace_arguments(params):
    """Tool arguments for a trace: mathematics, bounded, never credentials."""
    if not isinstance(params, dict):
        return None
    return _json(params.get('arguments'))[:_TRACE_LIMIT]


def _trace_output(reply):
    items = (reply or {}).get('contentItems') or []
    text = ''.join(item.get('text', '') for item in items)
    return text[:_TRACE_LIMIT] or None

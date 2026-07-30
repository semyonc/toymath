#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""codex_transport.py - the Codex app-server boundary, and its offline double.

ToyMath talks to a locally running Codex app-server over its documented
JSON-RPC protocol: managed ChatGPT authentication, thread/turn lifecycle,
dynamic tool calls, and turn interruption. Everything above this module -
the tool surface, the ledger, the oracle - is provider-neutral, and
everything below it is one pinned Codex runtime.

The seam exists for two reasons. First, ToyMath must never depend on
private SDK internals; the supported surface is the protocol. Second,
cancellation has to be routed by an independent reader: the pinned SDK
client answers `item/tool/call` synchronously on its sole JSON-RPC reader
thread, so a slow ToyMath callback (a figure render, a long tactic) would
delay `turn/interrupt` by exactly as long as the callback runs. The spike
measured that directly - a three-second tool call delayed interruption by
three seconds.

`TranscriptTransport` is the offline double: it replays a scripted turn so
the whole Codex path - tool dispatch, ledger writes, provenance,
interruption - can be tested with no app-server, no account, and no
network.
"""
import json
import logging
import os
import subprocess
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field

log = logging.getLogger('toymath.agent.codex')

# turn statuses, as the app-server reports them
TURN_COMPLETED = 'completed'
TURN_INTERRUPTED = 'interrupted'
TURN_FAILED = 'failed'


class CodexUnavailable(Exception):
    """The pinned Codex runtime or its required capability is missing."""


#: the one account type this backend runs on. The runtime's `Account` union
#: also admits `apiKey` and `amazonBedrock`, which bill a different, possibly
#: organizational credential - this backend is a personal-subscription seam
#: and must never spend one of those on a math derivation.
MANAGED_AUTH_MODE = 'chatgpt'


@dataclass(frozen=True)
class CodexAccountStatus:
    """What ToyMath is allowed to know about the user's Codex account.

    Deliberately not the credential: managed authentication means the
    app-server owns the OAuth tokens, refreshes them, and never hands them
    over. ToyMath keeps no account identifier or email either, so an
    exception, a log line, or a trace cannot leak one.
    """
    logged_in: bool = False
    auth_mode: str | None = None      # 'chatgpt' | 'apiKey' | 'amazonBedrock'
    plan_type: str | None = None
    requires_openai_auth: bool = False

    @property
    def usable(self):
        """Signed in *and* on the managed ChatGPT subscription.

        `logged_in` alone is not enough: an API-key or Bedrock account is
        also logged in, and running on one would silently consume a
        credential the user never pointed at ToyMath.
        """
        return self.logged_in and self.auth_mode == MANAGED_AUTH_MODE


@dataclass(frozen=True)
class LoginChallenge:
    """A one-time managed login prompt: a URL the user opens, never a token."""
    login_id: str
    kind: str                            # 'chatgpt' | 'chatgptDeviceCode'
    auth_url: str | None = None
    user_code: str | None = None
    verification_uri: str | None = None


@dataclass(frozen=True)
class CodexModel:
    id: str
    display_name: str | None = None
    default: bool = False


@dataclass(frozen=True)
class CodexThreadRequest:
    """One ephemeral, contained thread and the turn it runs."""
    instruction: str
    developer_instructions: str
    dynamic_tools: tuple = ()
    model: str | None = None
    cwd: str | None = None


@dataclass(frozen=True)
class CodexTurnOutcome:
    status: str = TURN_COMPLETED
    final_text: str = ''
    thread_id: str | None = None
    turn_id: str | None = None
    tool_calls: tuple = ()          # ToyMath tools invoked, in order
    native_tool_calls: tuple = ()   # residual native tools the model reached
    error: str | None = None
    metadata: dict = field(default_factory=dict)


class CodexTransport(object):
    """The app-server operations ToyMath uses. Implementations must not
    require, expose, or persist a raw ChatGPT token.

    `run_thread(request, on_tool_call, on_native_tool)` starts an ephemeral
    thread, runs one turn, and services tool calls through the callbacks:

    - `on_tool_call(params)` answers an `item/tool/call` with the
      documented `{"success": bool, "contentItems": [...]}` payload;
    - `on_native_tool(name)` reports a residual native tool the model
      invoked, and returns True when the run must be abandoned.

    `interrupt_turn` must be safe to call from another thread while a tool
    callback is still running - that is the whole point of the seam.
    """

    name = 'codex'
    #: a runtime marked unusable is replaced, never reused
    poisoned = False
    running = True

    def account_read(self):
        raise NotImplementedError

    def login_start(self, mode='chatgpt'):
        raise NotImplementedError

    def login_cancel(self, login_id, timeout=None):
        raise NotImplementedError

    def await_login(self, login_id, timeout=None, queue=None):
        raise NotImplementedError

    def logout(self):
        raise NotImplementedError

    def list_models(self):
        raise NotImplementedError

    def notifications(self):
        """Subscribe to server notifications before a request that will
        produce them, so a completion cannot be missed in the gap."""
        return None

    def release(self, queue):
        pass

    def run_thread(self, request, on_tool_call, on_native_tool=None,
                   on_turn=None):
        raise NotImplementedError

    def interrupt_turn(self, thread_id, turn_id):
        raise NotImplementedError

    def close(self):
        pass

    def poison(self, reason='unresponsive'):
        self.poisoned = True
        self.close()


# Thread items that are ordinary agent activity. Anything else is a native
# capability reaching outside the ToyMath surface - including an item type
# this version of ToyMath has never seen, which fails closed rather than
# being waved through.
BENIGN_ITEMS = frozenset((
    'userMessage', 'agentMessage', 'reasoning', 'dynamicToolCall',
    'sleep', 'contextCompaction', 'enteredReviewMode', 'exitedReviewMode',
))

#: thread item type -> the native capability it represents
NATIVE_ITEM_CAPABILITY = {
    'plan': 'update_plan',
    'imageView': 'view_image',
    'commandExecution': 'shell',
    'fileChange': 'apply_patch',
    'mcpToolCall': 'mcp',
    'webSearch': 'web_search',
    'subAgentActivity': 'subagent',
    'collabAgentToolCall': 'subagent',
    'imageGeneration': 'image_generation',
    'hookPrompt': 'hook',
}


def native_capability(item_type):
    """The native capability an emitted thread item reveals, or None."""
    if not item_type or item_type in BENIGN_ITEMS:
        return None
    return NATIVE_ITEM_CAPABILITY.get(item_type, item_type)


class ServerRefusal:
    """A server request ToyMath answers with a JSON-RPC error, not a result.

    For requests whose only valid responses would grant something (an OAuth
    access token, an attestation) or that have no refusal shape at all.
    """

    __slots__ = ('text',)

    def __init__(self, text):
        self.text = text


#: JSON-RPC application error for a request a contained thread may not make
METHOD_REFUSED = -32001

#: server request -> the capability it reveals, for the run's verdict. Taken
#: from the pinned runtime's own `ServerRequest` schema; an unlisted method
#: still invalidates the run, under its own name.
SERVER_REQUEST_CAPABILITY = {
    'item/commandExecution/requestApproval': 'shell',
    'item/fileChange/requestApproval': 'apply_patch',
    'item/tool/requestUserInput': 'request_user_input',
    'mcpServer/elicitation/request': 'mcp',
    'item/permissions/requestApproval': 'permissions',
    'account/chatgptAuthTokens/refresh': 'chatgpt_auth_tokens',
    'attestation/generate': 'attestation',
    'applyPatchApproval': 'apply_patch',
    'execCommandApproval': 'shell',
}

#: the refusal each method's own response schema accepts. Getting this shape
#: wrong would leave the runtime waiting or retrying instead of stopping.
#: A method absent here has no refusal shape at all - `requestUserInput`
#: must carry answers, a token refresh must carry an access token - so it is
#: answered with a JSON-RPC error instead.
DECLINE_RESPONSES = {
    'item/commandExecution/requestApproval': {'decision': 'decline'},
    'item/fileChange/requestApproval': {'decision': 'decline'},
    'mcpServer/elicitation/request': {'action': 'decline'},
    'item/permissions/requestApproval': {'permissions': {}},  # grants nothing
    'applyPatchApproval': {'decision': 'denied'},
    'execCommandApproval': {'decision': 'denied'},
}


def serve_request(method, params, on_tool_call, on_native_tool=None):
    """Answer one server request, refusing everything but a ToyMath tool.

    `item/tool/call` is the only server request a contained ToyMath thread
    has any business making. Every other one means a capability the
    containment review never cleared was reached, so it both gets the
    protocol's own refusal shape - a generic `{}` is not a valid response
    for most of them - and invalidates the run.
    """
    if method == 'item/tool/call':
        return on_tool_call(params or {})
    if on_native_tool is not None:
        on_native_tool(SERVER_REQUEST_CAPABILITY.get(method) or method)
    return DECLINE_RESPONSES.get(
        method,
        ServerRefusal(f'{method} is not available to a ToyMath thread'))


def check_login_mode(mode):
    """Only Codex-managed ChatGPT sign-in. An API key, an externally
    managed token, or any other mode is refused before it is sent - ToyMath
    does not own the user's ChatGPT auth lifecycle and has no reason to
    handle a raw credential."""
    if mode not in ('chatgpt', 'chatgptDeviceCode'):
        raise ValueError(
            f'unsupported login mode {mode!r}: ToyMath uses only '
            'Codex-managed ChatGPT authentication')
    return mode


class AppServerTransport(CodexTransport):
    """The documented app-server JSON-RPC protocol over the pinned runtime.

    Newline-delimited JSON on stdio: `{id, method, params}` requests,
    `{id, result|error}` responses, and notifications without an id. A
    private reader thread does nothing but classify and route messages, so
    a `turn/interrupt` response is delivered while a ToyMath tool callback
    is still running. Server-initiated requests (`item/tool/call` above
    all) are answered on worker threads for exactly that reason.

    The pinned SDK's own client answers server requests inline on its
    reader thread, which made a three-second tool call delay interruption
    by three seconds in the spike. That is why ToyMath speaks the protocol
    itself rather than reusing that client for execution.
    """

    name = 'codex'
    #: how long a plain request may take before it is treated as a fault
    REQUEST_TIMEOUT = 60.0

    def __init__(self, binary=None, home=None, cwd=None,
                 config_overrides=(), env=None, client_name='toymath',
                 client_version='0.1.0'):
        self.binary = binary
        self.home = home
        self.cwd = cwd
        self.config_overrides = tuple(config_overrides)
        self.extra_env = dict(env or {})
        self.client_name = client_name
        self.client_version = client_version
        self.poisoned = False
        self._process = None
        self._write_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._pending = {}
        self._notification_waiters = []
        self._stderr = deque(maxlen=200)
        self._reader = None
        self._stderr_thread = None
        self._request_handler = None
        self._closed = threading.Event()

    # -- process lifecycle -------------------------------------------------
    def start(self):
        if self._process is not None:
            return self
        if not self.binary:
            raise CodexUnavailable('no Codex runtime binary was resolved')
        command = [str(self.binary)]
        for override in self.config_overrides:
            command.extend(['--config', override])
        command.extend(['app-server', '--listen', 'stdio://'])
        # a scrubbed environment: the runtime gets its dedicated home and
        # nothing else this process happens to be holding
        env = {key: value for key, value in os.environ.items()
               if key in ('PATH', 'HOME', 'LANG', 'LC_ALL', 'TMPDIR',
                          'USER', 'LOGNAME', 'SHELL', 'TERM',
                          'SYSTEMROOT', 'APPDATA', 'LOCALAPPDATA')}
        if self.home:
            env['CODEX_HOME'] = str(self.home)
        env.update(self.extra_env)
        try:
            self._process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, encoding='utf-8',
                bufsize=1, cwd=self.cwd, env=env)
        except OSError as exc:
            raise CodexUnavailable(f'could not start the Codex runtime: {exc}')
        self._reader = threading.Thread(target=self._reader_loop,
                                        name='toymath-codex-reader',
                                        daemon=True)
        self._reader.start()
        self._stderr_thread = threading.Thread(target=self._drain_stderr,
                                               name='toymath-codex-stderr',
                                               daemon=True)
        self._stderr_thread.start()
        self.initialize()
        return self

    def initialize(self):
        result = self.request('initialize', {
            'clientInfo': {'name': self.client_name,
                           'title': 'ToyMath',
                           'version': self.client_version},
            # dynamic tools are an experimental app-server surface
            'capabilities': {'experimentalApi': True},
        })
        self.notify('initialized', None)
        return result

    def close(self):
        process, self._process = self._process, None
        self._closed.set()
        if process is None:
            return
        try:
            if process.stdin:
                process.stdin.close()
        except Exception:
            pass
        try:
            process.terminate()
            process.wait(timeout=2)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass
        self._fail_pending(CodexUnavailable('the Codex runtime was closed'))

    def poison(self, reason='unresponsive'):
        """Mark a shared runtime unusable and terminate it. The next command
        lazily creates a fresh one."""
        self.poisoned = True
        log.warning('the Codex runtime was poisoned (%s)', reason)
        self.close()

    @property
    def running(self):
        return self._process is not None and self._process.poll() is None

    def set_request_handler(self, handler):
        """`handler(method, params) -> payload` for server-initiated
        requests. Invoked on a worker thread, never on the reader."""
        self._request_handler = handler

    # -- JSON-RPC ----------------------------------------------------------
    def request(self, method, params=None, timeout=None):
        request_id = str(uuid.uuid4())
        waiter = {'event': threading.Event(), 'result': None, 'error': None}
        with self._state_lock:
            self._pending[request_id] = waiter
        message = {'id': request_id, 'method': method}
        if params is not None:
            message['params'] = params
        self._write(message)
        deadline = self.REQUEST_TIMEOUT if timeout is None else timeout
        if not waiter['event'].wait(deadline):
            with self._state_lock:
                self._pending.pop(request_id, None)
            raise CodexUnavailable(
                f'{method} did not answer within {deadline:g}s')
        if waiter['error'] is not None:
            raise CodexError(method, waiter['error'])
        return waiter['result']

    def notify(self, method, params=None):
        message = {'method': method}
        if params is not None:
            message['params'] = params
        self._write(message)

    def notifications(self):
        """A queue receiving every server notification, for one consumer."""
        queue = _NotificationQueue()
        with self._state_lock:
            self._notification_waiters.append(queue)
        return queue

    def release(self, queue):
        with self._state_lock:
            if queue in self._notification_waiters:
                self._notification_waiters.remove(queue)

    def _write(self, message):
        process = self._process
        if process is None or process.stdin is None:
            raise CodexUnavailable('the Codex runtime is not running')
        with self._write_lock:
            try:
                process.stdin.write(json.dumps(message) + '\n')
                process.stdin.flush()
            except (OSError, ValueError) as exc:
                raise CodexUnavailable(f'the Codex runtime closed: {exc}')

    def _reader_loop(self):
        process = self._process
        try:
            while True:
                line = process.stdout.readline()
                if not line:
                    raise CodexUnavailable(
                        'the Codex runtime closed its output. '
                        + self._stderr_tail())
                try:
                    message = json.loads(line)
                except ValueError:
                    continue                      # a stray non-protocol line
                if not isinstance(message, dict):
                    continue
                if 'method' in message and 'id' in message:
                    self._serve_request(message)  # never inline: see above
                elif 'method' in message:
                    self._route_notification(message)
                else:
                    self._route_response(message)
        except BaseException as exc:              # transport is finished
            if not self._closed.is_set():
                log.debug('codex reader stopped: %s', exc)
            self._fail_pending(exc)

    def _serve_request(self, message):
        handler = self._request_handler
        threading.Thread(
            target=self._answer, args=(message, handler),
            name='toymath-codex-request', daemon=True).start()

    def _answer(self, message, handler):
        method = message.get('method')
        payload = {}
        try:
            if method == 'currentTime/read':
                payload = {'currentTimeAt': int(time.time())}
            elif handler is not None:
                payload = handler(method, message.get('params')) or {}
        except Exception:
            log.debug('server request %s failed', method, exc_info=True)
            payload = ServerRefusal(f'{method} failed on the ToyMath client')
        reply = {'id': message['id']}
        if isinstance(payload, ServerRefusal):
            reply['error'] = {'code': METHOD_REFUSED, 'message': payload.text}
        else:
            reply['result'] = payload
        try:
            self._write(reply)
        except CodexUnavailable:
            pass

    def _route_notification(self, message):
        with self._state_lock:
            queues = list(self._notification_waiters)
        for queue in queues:
            queue.put(message)

    def _route_response(self, message):
        request_id = message.get('id')
        with self._state_lock:
            waiter = self._pending.pop(request_id, None)
        if waiter is None:
            return
        waiter['error'] = message.get('error')
        waiter['result'] = message.get('result')
        waiter['event'].set()

    def _fail_pending(self, exc):
        with self._state_lock:
            pending, self._pending = self._pending, {}
            queues = list(self._notification_waiters)
        for waiter in pending.values():
            waiter['error'] = {'message': str(exc)}
            waiter['event'].set()
        for queue in queues:
            queue.close()

    def _drain_stderr(self):
        process = self._process
        if process is None or process.stderr is None:
            return
        try:
            for line in process.stderr:
                self._stderr.append(line.rstrip())
        except Exception:
            pass

    def _stderr_tail(self, limit=20):
        return ' | '.join(list(self._stderr)[-limit:])[:1500]

    # -- account (managed ChatGPT only) ------------------------------------
    def account_read(self):
        result = self.request('account/read', {}) or {}
        account = result.get('account') or {}
        # deliberately drops `email`: ToyMath keeps no account identifier,
        # so no log line, exception, or trace can leak one
        return CodexAccountStatus(
            logged_in=bool(account),
            auth_mode=account.get('type'),
            plan_type=account.get('planType'),
            requires_openai_auth=bool(result.get('requiresOpenaiAuth')))

    def login_start(self, mode='chatgpt'):
        check_login_mode(mode)
        result = self.request('account/login/start', {'type': mode}) or {}
        return LoginChallenge(
            login_id=result.get('loginId') or '',
            kind=result.get('type') or mode,
            auth_url=result.get('authUrl'),
            user_code=result.get('userCode'),
            verification_uri=result.get('verificationUrl'))

    def login_cancel(self, login_id, timeout=None):
        self.request('account/login/cancel', {'loginId': login_id},
                     timeout=timeout)

    def await_login(self, login_id, timeout=None, queue=None):
        """Wait for the managed flow's completion notification.

        Never handles a token: the app-server owns the OAuth exchange and
        only reports whether it succeeded.
        """
        owned = queue is None
        queue = queue if queue is not None else self.notifications()
        deadline = None if timeout is None else time.monotonic() + timeout
        try:
            while True:
                remaining = (None if deadline is None
                             else max(0.0, deadline - time.monotonic()))
                if remaining == 0.0:
                    raise CodexUnavailable('the login flow timed out')
                message = queue.get(timeout=remaining or 1.0)
                if message is None:
                    continue
                if message.get('method') != 'account/login/completed':
                    continue
                params = message.get('params') or {}
                if params.get('loginId') not in (None, login_id):
                    continue
                if not params.get('success'):
                    raise CodexError('account/login/start',
                                     {'message': params.get('error')
                                      or 'the login was not completed'})
                return self.account_read()
        finally:
            if owned:
                self.release(queue)

    def logout(self):
        self.request('account/logout', None)

    # -- execution ---------------------------------------------------------
    def thread_params(self, request):
        """One ephemeral, contained thread.

        Ephemeral so no hidden conversational state carries between do!
        cells; an isolated empty `cwd` because that - not ephemerality - is
        what keeps a project `AGENTS.md` out of the instruction chain (the
        spike saw a marker file in one thread's cwd reach the model
        request); read-only with approvals off so nothing can be executed
        or written even if a native capability were reachable.
        """
        params = {
            'approvalPolicy': 'never',
            'sandbox': 'read-only',
            'cwd': request.cwd or self.cwd,
            'ephemeral': True,
            'developerInstructions': request.developer_instructions,
            'dynamicTools': list(request.dynamic_tools),
        }
        if request.model:
            params['model'] = request.model
        return params

    def run_thread(self, request, on_tool_call, on_native_tool=None,
                   on_turn=None):
        queue = self.notifications()
        self.set_request_handler(
            lambda method, params: serve_request(method, params, on_tool_call,
                                                 on_native_tool))
        thread_id = turn_id = None
        try:
            started = self.request('thread/start',
                                   self.thread_params(request)) or {}
            thread_id = ((started.get('thread') or {}).get('id'))
            turn = self.request('turn/start', {
                'threadId': thread_id,
                'input': [{'type': 'text', 'text': request.instruction}],
            }) or {}
            turn_id = ((turn.get('turn') or {}).get('id'))
            if on_turn is not None:
                on_turn(thread_id, turn_id)
            return self._await_turn(queue, thread_id, turn_id, on_tool_call,
                                    on_native_tool)
        except (CodexUnavailable, CodexError) as exc:
            return CodexTurnOutcome(status=TURN_FAILED, thread_id=thread_id,
                                    turn_id=turn_id, error=str(exc))
        finally:
            self.release(queue)
            self.set_request_handler(None)

    def _await_turn(self, queue, thread_id, turn_id, on_tool_call,
                    on_native_tool):
        final, tools, native = '', [], []
        while True:
            message = queue.get(timeout=1.0)
            if message is None:
                continue
            method = message.get('method')
            params = message.get('params') or {}
            if method == 'item/completed':
                item = params.get('item') or {}
                kind = item.get('type')
                if kind == 'agentMessage':
                    final = item.get('text') or final
                elif kind == 'dynamicToolCall':
                    tools.append(item.get('tool'))
                else:
                    name = native_capability(kind)
                    if name is not None:
                        native.append(name)
                        if (on_native_tool is not None
                                and on_native_tool(name)):
                            return CodexTurnOutcome(
                                status=TURN_INTERRUPTED, thread_id=thread_id,
                                turn_id=turn_id, tool_calls=tuple(tools),
                                native_tool_calls=tuple(native))
            elif method == 'turn/completed':
                turn = params.get('turn') or {}
                if turn.get('id') not in (None, turn_id):
                    continue
                return CodexTurnOutcome(
                    status=turn.get('status') or TURN_COMPLETED,
                    final_text=final, thread_id=thread_id, turn_id=turn_id,
                    tool_calls=tuple(tools), native_tool_calls=tuple(native),
                    error=(turn.get('error') or {}).get('message')
                    if isinstance(turn.get('error'), dict) else None)
            elif method == 'error':
                if params.get('turnId') not in (None, turn_id):
                    continue
                if params.get('willRetry'):
                    continue          # the runtime is retrying; not final
                error = params.get('error')
                message = (error.get('message') if isinstance(error, dict)
                           else error)
                return CodexTurnOutcome(
                    status=TURN_FAILED, thread_id=thread_id, turn_id=turn_id,
                    tool_calls=tuple(tools), native_tool_calls=tuple(native),
                    error=str(message or 'the Codex turn failed'))

    def interrupt_turn(self, thread_id, turn_id):
        """Idempotent by construction: the app-server answers an already
        interrupted turn the same way, and a dead transport is already
        stopped."""
        if not thread_id or not turn_id:
            return
        try:
            self.request('turn/interrupt',
                         {'threadId': thread_id, 'turnId': turn_id},
                         timeout=5.0)
        except (CodexUnavailable, CodexError) as exc:
            log.debug('turn/interrupt was not acknowledged: %s', exc)

    def list_models(self):
        result = self.request('model/list', {}) or {}
        return tuple(CodexModel(id=row.get('id') or row.get('model') or '',
                                display_name=row.get('displayName'),
                                default=bool(row.get('isDefault')))
                     for row in (result.get('data') or [])
                     if not row.get('hidden'))


class CodexError(Exception):
    """An app-server request answered with a JSON-RPC error."""

    def __init__(self, method, error):
        self.method = method
        self.error = error or {}
        message = (self.error.get('message') if isinstance(error, dict)
                   else None)
        super(CodexError, self).__init__(
            f'{method}: {message or error or "unknown error"}')


class _NotificationQueue(object):
    """A tiny closable queue: notifications for one waiting consumer."""

    def __init__(self):
        self._items = deque()
        self._ready = threading.Condition()
        self._closed = False

    def put(self, message):
        with self._ready:
            self._items.append(message)
            self._ready.notify()

    def close(self):
        with self._ready:
            self._closed = True
            self._ready.notify_all()

    def get(self, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._ready:
            while not self._items:
                if self._closed:
                    raise CodexUnavailable('the Codex runtime closed')
                remaining = (None if deadline is None
                             else deadline - time.monotonic())
                if remaining is not None and remaining <= 0:
                    return None
                self._ready.wait(remaining)
            return self._items.popleft()


class TranscriptTransport(CodexTransport):
    """Replays a scripted turn instead of reaching a Codex app-server.

    A transcript entry is one of:

    - `{'tool': NAME, 'arguments': {...}}`  - a dynamic tool call;
    - `{'native': NAME}`                    - a residual native tool call;
    - `{'message': TEXT}`                   - the final agent message;
    - `{'block': True}`                     - wait here until interrupted.
    """

    name = 'transcript'

    def __init__(self, transcript=(), account=None, models=(),
                 thread_id='t1', turn_id='r1'):
        self.transcript = list(transcript)
        self.account = account or CodexAccountStatus(
            logged_in=True, auth_mode='chatgpt', plan_type='plus')
        self.models = tuple(models) or (CodexModel('gpt-5.6-luna',
                                                   'GPT-5.6 Luna', True),)
        self.thread_id = thread_id
        self.turn_id = turn_id
        self.requests = []
        self.interrupts = []
        self.logins = []
        self.cancelled_login = None
        self.cancel_timeout = None
        #: set to an Event to script a login that never completes
        self.pending_login = None
        self.closed = False
        self._interrupted = threading.Event()
        self._released = threading.Event()

    # -- account -----------------------------------------------------------
    def account_read(self):
        return self.account

    def login_start(self, mode='chatgpt'):
        check_login_mode(mode)
        challenge = LoginChallenge(
            login_id=f'login-{len(self.logins) + 1}', kind=mode,
            auth_url=('https://auth.openai.test/activate'
                      if mode == 'chatgpt' else None),
            user_code='ABCD-EFGH' if mode != 'chatgpt' else None,
            verification_uri=('https://auth.openai.test/device'
                              if mode != 'chatgpt' else None))
        self.logins.append(challenge.login_id)
        return challenge

    def login_cancel(self, login_id, timeout=None):
        self.cancelled_login = login_id
        self.cancel_timeout = timeout

    def await_login(self, login_id, timeout=None, queue=None):
        if self.pending_login is not None:
            # the scripted flow never completes on its own: the test drives
            # the interrupt that must cancel it
            self.pending_login.wait(timeout or 10)
            raise CodexUnavailable('the login flow timed out')
        self.account = CodexAccountStatus(logged_in=True, auth_mode='chatgpt',
                                          plan_type='plus')
        return self.account

    def logout(self):
        self.account = CodexAccountStatus()

    def list_models(self):
        return self.models

    # -- execution ---------------------------------------------------------
    def run_thread(self, request, on_tool_call, on_native_tool=None,
                   on_turn=None):
        self.requests.append(request)
        if on_turn is not None:
            on_turn(self.thread_id, self.turn_id)
        called, native, final = [], [], ''
        for index, entry in enumerate(self.transcript):
            if self._interrupted.is_set():
                return self._interrupted_outcome(called, native, final)
            if entry.get('block'):
                self._released.wait(10)
                if self._interrupted.is_set():
                    return self._interrupted_outcome(called, native, final)
                continue
            if 'native' in entry:
                native.append(entry['native'])
                if on_native_tool is not None and on_native_tool(
                        entry['native']):
                    return CodexTurnOutcome(
                        status=TURN_INTERRUPTED, thread_id=self.thread_id,
                        turn_id=self.turn_id, tool_calls=tuple(called),
                        native_tool_calls=tuple(native))
                continue
            if 'tool' in entry:
                called.append(entry['tool'])
                on_tool_call({'tool': entry['tool'],
                              'arguments': entry.get('arguments', {}),
                              'callId': f'call-{index + 1}',
                              'threadId': self.thread_id,
                              'turnId': self.turn_id})
                continue
            final = entry.get('message', final)
        if self._interrupted.is_set():
            return self._interrupted_outcome(called, native, final)
        return CodexTurnOutcome(status=TURN_COMPLETED, final_text=final,
                                thread_id=self.thread_id,
                                turn_id=self.turn_id,
                                tool_calls=tuple(called),
                                native_tool_calls=tuple(native))

    def _interrupted_outcome(self, called, native, final):
        return CodexTurnOutcome(status=TURN_INTERRUPTED, final_text='',
                                thread_id=self.thread_id,
                                turn_id=self.turn_id,
                                tool_calls=tuple(called),
                                native_tool_calls=tuple(native))

    def interrupt_turn(self, thread_id, turn_id):
        self.interrupts.append((thread_id, turn_id))
        self._interrupted.set()
        self._released.set()

    def close(self):
        self.closed = True
        self._released.set()


class StubbornTransport(TranscriptTransport):
    """A transport that accepts `turn/interrupt` and never acts on it -
    the case where ToyMath must contain the run itself."""

    def interrupt_turn(self, thread_id, turn_id):
        self.interrupts.append((thread_id, turn_id))

# -*- coding: utf-8 -*-

import base64
import html as _html
import inspect
import re

from notation import Notation, Symbol
from LatexParser import MathParser
from processor import MathProcessor
from LatexWriter import LaTexWriter
from replicator import Replicator
from prolog import PrologModel
from ledger import Ledger
import agent_config
import cell_input
import model_config
import prompt_commands
from cell_input import split_lines

from IPython.display import HTML, Javascript
from engine import display

BACKREF_RE = re.compile(r'\[\[\s*(\d+)\s*\]\]')
# a cell that starts with `name!` (letter/underscore start) may be a
# prompt-command; only a *registered* name diverts, so math like `n! + 1`
# (factorial) still parses normally
CMD_PREFIX_RE = re.compile(r'^([A-Za-z_]\w*)!')
# any `name!` token anywhere in a cell — used to spot an inline expr command
# ({diff! {int! x^3}}); the registry check filters out factorials
EXPR_TOKEN_RE = re.compile(r'([A-Za-z_][\w-]*)!')


def _display_latex(latex):
    """Derived rich-view spelling; ledger records remain byte-identical."""
    import primitives
    return primitives.display_latex(latex)


def _display_math_spans(text):
    parts = text.split('$')
    return ''.join(f'${_display_latex(part)}$' if i % 2 else part
                   for i, part in enumerate(parts))


def _open_browser(url):
    """Hand a one-time sign-in URL to the OS browser. Never raises."""
    if not url:
        return False
    try:
        import webbrowser
        return bool(webbrowser.open(url))
    except Exception:
        return False


def _display_replacing(obj):
    """Display `obj` after clearing the cell's earlier output.

    Used to retract a spent login challenge before the notebook is saved.
    The kernel's display handler clears on this keyword; other handlers
    (console, embedders) simply display, which is why the challenge text
    also says it is one-time.
    """
    try:
        display(obj, clear_output=True)
    except TypeError:
        display(obj)


def _replace_cell_output(obj):
    """Show `obj` in place of everything this cell has displayed so far.

    Returns True only when the swap actually happened. The kernel clears
    with `wait=True`, so the frontend drops the old output at the moment
    the replacement arrives and the reader sees no flicker. A console or
    embedding handler has no clear at all (plain `IPython.display.display`
    rejects the keyword), and there the caller must leave what it already
    printed alone rather than print it a second time.
    """
    try:
        display(obj, clear_output=True)
        return True
    except TypeError:
        return False


class _LiveLog:
    """The cell's streamed step lines, kept so the finish can fold them.

    The stream is what a reader watches while a long run works, and it is
    worth having. But the verified-chain table rendered afterwards from the
    ledger records says the same thing more completely and is the actual
    artifact — streaming is decoration, never the record. Leaving both
    expanded buries the record under its own rehearsal: one measured cell
    had 60 streamed lines above the table that restates them.

    So the lines are collected as they stream and re-emitted once, collapsed,
    at the moment the cell's evidence renders. Only a run that CLOSED folds:
    with no certified result the stream is the only account of what happened,
    and hiding it would hide the diagnosis.

    One log per CELL, not per run: an argument-resolving sub-run streams
    before the outer run starts, and folding per run would clear lines that
    the fold does not contain.
    """

    def __init__(self, shell):
        self._shell = shell
        self._records = []
        self._spent = False

    def disable(self):
        """Give up folding for the rest of this cell.

        The swap clears the cell wholesale, so it is only ever safe as the
        FIRST thing published after the streaming stops. A path that must
        show evidence mid-cell — resolving inline commands inside another
        command's argument, which renders the inner chains and only then
        runs the outer instruction — says so here, and the later fold
        becomes a no-op instead of erasing what it does not contain.
        """
        self._spent = True

    def __call__(self, step):
        """Stream one step and remember it. Never raises: rendering a step
        must not be able to fail the derivation that produced it."""
        try:
            html = self._shell.render_do_step(step)
        except Exception:
            return
        self._records.append((step, html))
        try:
            display(HTML(html))
        except Exception:
            pass

    def _summary(self):
        steps = notes = checked = flagged = 0
        for step, _ in self._records:
            if step.get('op') in ('comment', 'branch'):
                notes += 1
                continue
            steps += 1
            if (step.get('check') or {}).get('status') in ('agree', 'exact'):
                checked += 1
            else:
                flagged += 1
        parts = [f'{steps} step{"s" if steps != 1 else ""}',
                 f'{checked} checked']
        if flagged:
            parts.append(f'{flagged} not checked')
        if notes:
            parts.append(f'{notes} note{"s" if notes != 1 else ""}')
        return 'agent turns &mdash; ' + ' &middot; '.join(parts)

    def fold(self, closed):
        """Collapse the streamed lines into one expandable group.

        `closed` is the caller's answer to "did this cell produce a verified
        result?" — the table only supersedes the log when it exists.
        """
        if self._spent or not closed or len(self._records) < 2:
            return False
        body = ''.join(html for _, html in self._records)
        folded = HTML(
            f'<details style="margin:2px 0"><summary style="cursor:pointer;'
            f'color:#888">{self._summary()} '
            f'<span style="font-size:90%">(click to expand)</span></summary>'
            f'<div style="margin:4px 0 4px 1em;border-left:2px solid #eee;'
            f'padding-left:.8em">{body}</div></details>')
        if not _replace_cell_output(folded):
            return False
        self._records = []
        return True


def _notify_route(handler, route):
    """Send routing changes to the frontend.

    The callback now takes an `AgentRoute`; a two-argument
    `(model, providers)` handler from before the backend seam keeps
    working, so embedders do not break on the transition.
    """
    try:
        positional = [p for p in inspect.signature(handler).parameters.values()
                      if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    except (TypeError, ValueError):      # a builtin or C callable
        positional = []
    if len(positional) >= 2:
        handler(route.model, route.providers)
    else:
        handler(route)


class MathShell(object):

    def __init__(self):
        self.history = {}
        self._live_log = _LiveLog(self)
        self.parsedNotation = Notation()
        self.processor = MathProcessor(model=PrologModel())
        # Discover both command systems before constructing the lexer. Bang
        # words become COMMAND tokens only when this registry admits them;
        # every other bang is mathematical factorial syntax.
        self.commands = prompt_commands.load_commands()
        self.parser = MathParser(
            self.parsedNotation, command_names=self._command_names())
        self.processor.trace = self.trace_step
        self.history = {}
        self.execution_history = {}
        self.echo_mode = False
        self.current_echo = False
        self.trace = False
        self.trace_mode = False
        self.trace_output = None
        self.show_quotes = False
        # notebook-wide derivation ledger fed by do! cells
        self.ledger = Ledger()
        # Agent routing is notebook-local: backend!/model! change this shell
        # without mutating process-wide environment variables or other
        # kernels.
        # No model until one is chosen: each backend supplies its own
        # default, so an OpenRouter model id can never be handed to Codex
        # (or the reverse) just because auto-resolution moved.
        self.route = agent_config.AgentRoute(backend=agent_config.AUTO)
        self.model_change_handler = None
        # commands! reloads the discoverable prompt-command registry.

    def _command_names(self):
        return (set(self.processor.actions) | set(self.commands)
                | set(prompt_commands.RESERVED))

    def preview_cell(self, code):
        """Rendered segments for a cell's input, or None to keep it raw.

        Read-only: the frontend calls this while the user edits, so it must
        never touch the history, the ledger, or the live notation graph.

        A `[[n]]` backreference renders as the formula it stands for — the
        expression the cell will actually run on. A fresh kernel has no
        history yet, so until that result exists the reference renders as
        itself and the cell still reads as the command it is.
        """
        names = self._command_names()
        # only a command that hands its argument to the agent may carry prose
        prose = set(self.commands) | {'do'}
        if BACKREF_RE.search(code):
            try:
                resolved = cell_input.preview(self.resolve_backrefs(code),
                                              names, prose)
            except Exception:
                resolved = None
            if resolved:
                return resolved
        return cell_input.preview(code, names, prose)

    def trace_step(self, sym, notation, index):
        if self.trace:
            writer = LaTexWriter(notation)
            if self.trace_output != '':
                self.trace_output += ' \\\\'
            self.trace_output += f'{{{writer(sym)}}} \\tag{{{index}}}'

    def add_action(self, name, instance):
        self.processor.actions[name] = instance
        self.parser.command_names = frozenset(self._command_names())

    def set_echo(self, current_echo=None, echo_mode=None):
        if current_echo is not None:
            self.current_echo = current_echo
        if echo_mode is not None:
            self.echo_mode = echo_mode

    def set_trace(self, trace_mode):
        self.trace = trace_mode

    def set_show_quotes(self, show_quotes):
        self.show_quotes = show_quotes

    def exec(self, code, execution_count, add_to_history=False, cell_id=None):
        # one live log per CELL: an argument-resolving sub-run streams before
        # the outer run starts, and the fold has to own both.
        self._live_log = _LiveLog(self)
        stripped = code.strip()
        m = CMD_PREFIX_RE.match(stripped)
        if m and self.dispatch_command(m.group(1), stripped[m.end():].strip(),
                                       execution_count, add_to_history):
            return
        if self.has_expr_command(stripped):
            self.exec_composite(stripped, execution_count, add_to_history)
            return
        lines = [line for line in split_lines(code)]
        for index, line in enumerate(lines):
            last = index == len(lines) - 1
            self.exec_stmt(line, execution_count, add_to_history and last, last)

    def dispatch_command(self, name, rest, execution_count, add_to_history):
        """Handle a `name!` cell prefix. Returns True when it was a command
        (do!, a registered prompt-command, or the commands!/help! listing),
        False when `name` is not a command so the cell is ordinary math
        (e.g. `n!` factorial)."""
        if name in ('commands', 'help'):
            self.show_commands()
            return True
        if name == 'model':
            self.exec_model(rest)
            return True
        if name == 'login':
            self.exec_login(rest)
            return True
        if name == 'backend':
            self.exec_backend(rest)
            return True
        if name == 'do':
            # the free-form agent endpoint: rest is the instruction verbatim
            self.exec_do(rest, execution_count, add_to_history)
            return True
        cmd = self.commands.get(name)
        if cmd is None:
            return False
        if cmd.expr:
            # an expr command composes inline; let exec_composite handle it
            # (even whole-cell `int! x^3`) so behaviour is uniform
            return False
        if not rest:
            self._do_error(f'{name}! needs an argument')
            return True
        resolved = self._resolve_prompt_argument(cmd, rest)
        if resolved is None:
            return True
        instruction = prompt_commands.render(cmd, resolved)
        proof_goal = resolved if cmd.mode == 'prove' else None
        self.exec_do(instruction, execution_count, add_to_history,
                     proof_goal=proof_goal)
        return True

    def has_expr_command(self, text):
        """True when the cell contains a registered expr command anywhere,
        so it must be evaluated as a composite expression."""
        return any(name in self.commands and self.commands[name].expr
                   for name in EXPR_TOKEN_RE.findall(text))

    def show_commands(self):
        """Render the discoverable command list. Reloads the registry first
        so a newly-added commands/*.md file goes live without a restart."""
        self.commands = prompt_commands.load_commands()
        self.parser.command_names = frozenset(self._command_names())
        rows = ['<tr><td style="padding:2px 14px 2px 0;vertical-align:top">'
                f'<code>{_html.escape(c.name)}!</code></td>'
                f'<td style="color:#444">{_html.escape(c.description)}'
                + (f' <code>[{_html.escape(prompt_commands.signature(c))}]'
                   '</code>' if prompt_commands.signature(c) else '')
                + (f' <code>[direct: {_html.escape(c.direct)}]</code>'
                   if c.direct else '')
                + '</td></tr>'
                for c in sorted(self.commands.values())]
        table = ('<table>' + ''.join(rows) + '</table>') if rows else (
            '<div style="color:#888">no commands defined yet '
            '(add <code>commands/&lt;name&gt;.md</code> files)</div>')
        display(HTML('<div><b>notebook commands</b>' + table
                     + '<div style="color:#888;margin-top:4px">plus '
                     '<code>do!</code> (free-form instruction), '
                     '<code>model!</code> (agent model), '
                     '<code>backend!</code> (agent backend), '
                     '<code>login!</code> (Codex account), and '
                     '<code>commands!</code> (this list)</div></div>'))

    # Compatibility for embedders and the kernel comm, which read the model
    # routing directly. The route is the single source of truth.
    @property
    def model_name(self):
        return self.route.model

    @property
    def model_providers(self):
        return self.route.providers

    @property
    def backend_name(self):
        """The backend this notebook would run on right now (never starts a
        Codex runtime to find out)."""
        return agent_config.preview(self.route).backend

    def _model_status_html(self):
        routing = agent_config.describe(self.route)
        model = routing['model'] or f'{routing["backend"]} default'
        if routing['backend'] == agent_config.CODEX:
            detail = 'your own Codex account'
        elif routing['endpoint']:
            # ahead of the provider branch: provider order is an OpenRouter
            # extension and is not sent here, so advertising it would
            # describe routing this run does not get
            detail = ('OpenAI-compatible endpoint <code>'
                      + _html.escape(routing['endpoint']) + '</code>')
            if routing['providers']:
                detail += ' &mdash; provider order does not apply here'
        elif routing['providers']:
            detail = ('providers <code>'
                      + _html.escape(', '.join(routing['providers']))
                      + '</code>; fallbacks disabled')
        else:
            detail = 'OpenRouter default provider routing'
        flag = (' <span style="color:#b65c00">(experimental)</span>'
                if routing['experimental'] else '')
        return ('<div><strong>agent backend:</strong> <code>'
                + _html.escape(routing['backend']) + '</code>' + flag
                + ' &mdash; ' + _html.escape(routing['reason'])
                + '</div><div><strong>agent model:</strong> <code>'
                + _html.escape(model) + '</code> &mdash; ' + detail
                + '</div>')

    def _route_changed(self):
        """Notify the frontend and render the effective routing."""
        if self.model_change_handler is not None:
            _notify_route(self.model_change_handler, self.route)
        display(HTML(self._model_status_html()))

    def _set_model(self, model, providers=()):
        """Set notebook-local agent routing and render its effective value."""
        self.route = self.route.with_model(model, providers)
        self._route_changed()

    def _note_account(self, status):
        """Record an observed Codex account and republish the routing.

        On `backend! auto` the effective backend is a function of the login
        state, so signing in or out can change where the next `do!` runs
        without the route object changing at all. The toolbar would
        otherwise keep advertising the pre-login answer.
        """
        before = self.backend_name
        agent_config.note_codex_account(status)
        if (self.backend_name != before
                and self.model_change_handler is not None):
            _notify_route(self.model_change_handler, self.route)
        return status

    def exec_backend(self, arguments):
        """Handle ``backend! [auto|openrouter|codex]`` for this notebook."""
        name = (arguments or '').strip().lower()
        if not name:
            display(HTML(self._model_status_html()
                         + '<div style="color:#666">Use <code>backend! '
                         'auto</code>, <code>backend! openrouter</code>, or '
                         '<code>backend! codex</code> (experimental, needs '
                         '<code>login!</code>).</div>'))
            return
        try:
            agent_config.check(name)
        except ValueError as e:
            display(HTML('<div style="color:#c00">backend! error: '
                         + _html.escape(str(e)) + '</div>'))
            return
        # Switching is explicit and never automatic: a run must not move a
        # user's spending or quota to the other provider by surprise.
        self.route = self.route.with_backend(name)
        self._route_changed()

    def exec_model(self, arguments):
        """Handle ``model! [MODEL[, PROVIDER...]]`` for this notebook.

        The catalog is backend-aware: OpenRouter reads `models.yaml` and its
        provider order, while Codex reports its own models and takes no
        provider argument."""
        if self.backend_name == agent_config.CODEX:
            self._exec_codex_model(arguments)
            return
        try:
            endpoints = model_config.load_model_config()
        except model_config.ModelConfigError as e:
            display(HTML('<div style="color:#c00">model! error: '
                         + _html.escape(str(e)) + '</div>'))
            return

        if arguments:
            parts = [part.strip() for part in arguments.split(',')]
            if not parts[0] or any(not part for part in parts[1:]):
                display(HTML('<div style="color:#c00">model! error: use '
                             '<code>model! MODEL[, PROVIDER...]</code>'
                             '</div>'))
                return
            model = parts[0]
            # Explicit providers override the configured provider order. A
            # bare model name picks up its optional routing from models.yaml.
            if len(parts) > 1:
                providers = tuple(dict.fromkeys(parts[1:]))
            else:
                endpoint = model_config.find_model(endpoints, model)
                providers = endpoint.providers if endpoint else ()
            self._set_model(model, providers)
            return

        display(HTML(self._model_status_html()
                     + '<div style="color:#666">Type '
                     '<code>model! </code> and press <kbd>Tab</kbd> or '
                     '<kbd>Ctrl</kbd>+<kbd>Space</kbd> to choose from '
                     '<code>engine/models.yaml</code>.</div>'))

    def _exec_codex_model(self, arguments):
        """``model!`` while Codex is selected: its own catalog, no providers."""
        import agent_do
        if ',' in (arguments or ''):
            display(HTML('<div style="color:#c00">model! error: the Codex '
                         'backend has no provider routing; use '
                         '<code>model! MODEL</code></div>'))
            return
        try:
            models = agent_config.codex_models()
        except agent_do.DoAgentError as e:
            display(HTML('<div style="color:#c00">model! error: '
                         + _html.escape(str(e)) + '</div>'))
            return
        name = (arguments or '').strip()
        if name:
            known = [m.id for m in models]
            if known and name not in known:
                display(HTML('<div style="color:#c00">model! error: '
                             + _html.escape(name) + ' is not offered by '
                             'this Codex account &mdash; choose from <code>'
                             + _html.escape(', '.join(known))
                             + '</code></div>'))
                return
            self._set_model(name, ())
            return
        listing = ', '.join(f'<code>{_html.escape(m.id)}</code>'
                            for m in models)
        display(HTML(self._model_status_html()
                     + '<div style="color:#666">Codex models: ' + listing
                     + '</div>'))

    # ------------------------------------------------------------------
    # login! — managed Codex authentication
    # ------------------------------------------------------------------

    _LOGIN_USAGE = ('<div style="color:#666">use <code>login!</code>, '
                    '<code>login! device</code>, <code>login! status</code>, '
                    'or <code>login! logout</code></div>')

    def exec_login(self, arguments):
        """Handle ``login! [device|status|logout]``.

        Authentication and backend choice are separate operations: signing
        in never changes which backend this notebook runs on.
        """
        from agent_backends import codex
        import agent_do
        action = (arguments or '').strip().lower()
        # the same list the completer offers: an option cannot appear in the
        # popup without being accepted here, or the reverse
        if action and action not in agent_config.LOGIN_ACTIONS:
            display(HTML('<div style="color:#c00">login! error: unknown '
                         f'option {_html.escape(action)}</div>'
                         + self._LOGIN_USAGE))
            return
        try:
            if action == 'status':
                self._render_account(self._note_account(
                    codex.account_status()))
            elif action == 'logout':
                self._render_account(self._note_account(codex.logout()),
                                     signed_out=True)
            else:
                mode = 'chatgptDeviceCode' if action == 'device' else 'chatgpt'
                status = codex.login(mode,
                                     on_challenge=self._render_challenge)
                # replace: the spent challenge leaves the cell output here
                self._render_account(self._note_account(status), replace=True)
        except agent_do.DoAgentError as e:
            _display_replacing(HTML('<div style="color:#c00">login! error: '
                                    + _html.escape(str(e)) + '</div>'))
        except KeyboardInterrupt:
            # the pending challenge was cancelled through the app-server,
            # and is cleared from the notebook as well
            _display_replacing(HTML(
                '<div style="color:#b65c00">login cancelled — the '
                'pending sign-in was discarded.</div>'))

    @staticmethod
    def _render_challenge(challenge):
        """Show a one-time sign-in prompt, keeping it out of the saved file.

        Never a token - the app-server owns the OAuth exchange - but a
        challenge is still a short-lived secret, and Jupyter persists cell
        output into the `.ipynb`. So the browser flow hands the URL to the
        OS browser instead of printing it, and the device code is wiped
        from the cell by `_render_account` once the flow ends.
        """
        if challenge.kind == 'chatgptDeviceCode':
            # this code has to be readable: the user types it. It is cleared
            # from the output as soon as the sign-in completes.
            display(HTML(
                '<div>Open <a href="'
                + _html.escape(challenge.verification_uri or '', quote=True)
                + '" target="_blank">'
                + _html.escape(challenge.verification_uri or '')
                + '</a> and enter the code <code>'
                + _html.escape(challenge.user_code or '')
                + '</code>. Waiting for the sign-in to complete…</div>'))
            return
        url = challenge.auth_url or ''
        if _open_browser(url):
            display(HTML(
                '<div>A Codex sign-in page was opened in your browser. '
                'Waiting for the sign-in to complete…<br>'
                '<span style="color:#888">The one-time link is not stored '
                'in this notebook.</span></div>'))
            return
        # headless or no browser available: the link is the only way in, so
        # show it and say plainly that it lands in the saved output
        display(HTML(
            '<div>Sign in to Codex: <a href="'
            + _html.escape(url, quote=True)
            + '" target="_blank">open the authorization page</a>. '
            'Waiting for the sign-in to complete…<br>'
            '<span style="color:#888">No browser could be opened, so this '
            'one-time link is shown here; it is cleared when the sign-in '
            'ends.</span></div>'))

    @staticmethod
    def _render_account(status, signed_out=False, replace=False):
        """Render the account status. `replace` wipes the cell first, so a
        spent sign-in challenge never reaches the saved notebook."""
        if not status.logged_in:
            body = ('<div>Codex: <strong>signed out</strong>'
                    + ('.' if signed_out else ' — run <code>login!</code> to '
                       'use your own Codex account.') + '</div>')
        elif not status.usable:
            # logged in, but on a credential this backend refuses to spend
            body = ('<div>Codex: signed in with '
                    f'<strong>{_html.escape(status.auth_mode or "unknown")}'
                    '</strong> authentication, which ToyMath does not use — '
                    'run <code>login! out</code> then <code>login!</code> for '
                    'a managed ChatGPT account.</div>')
        else:
            plan = (f' ({_html.escape(status.plan_type)} plan)'
                    if status.plan_type else '')
            body = ('<div>Codex: <strong>signed in</strong> with '
                    f'{_html.escape(status.auth_mode)} '
                    f'authentication{plan}. Usage follows that account\'s '
                    'own plan and limits.</div>')
        _display_replacing(HTML(body)) if replace else display(HTML(body))

    def exec_stmt(self, code, execution_count, add_to_history, do_output):
        self.current_echo = False
        self.trace = self.trace_mode
        self.trace_output = ''
        self.show_quotes = False
        try:
            sym = self.parser.parse(code)
            outsym, notation = self.process(sym, self.parsedNotation)
            if not outsym == Notation.NONE and do_output:
                output = self.output(outsym, notation, execution_count, add_to_history)
                if self.echo_mode or self.current_echo:
                    output = self.output(sym, self.parsedNotation, execution_count, False) + ' \\Rightarrow ' + output
                display(HTML('$' + output + '$'))
        except Exception as e:
            if self.trace:
                display(HTML('$\\color{red}{\\text2{Error: }\\textit{' + e.args[0] + '}}$'))
            else:
                raise
        if self.trace and self.trace_output != '':
            display(HTML('$\\begin{alignat}{3}' + self.trace_output + '\\end{alignat}$'))

    # ------------------------------------------------------------------
    # do! agent endpoint
    # ------------------------------------------------------------------

    def resolve_backrefs(self, text):
        """Inline [[n]] references as the rendered LaTeX of prior cell
        results; raises ValueError on an undefined reference."""
        import primitives

        def repl(m):
            key = m.group(1)
            sym = self.execution_history.get(key)
            if sym is None:
                raise ValueError(
                    f'[[{key}]] does not reference a previous result')
            notation = self.history.get(sym, self.parsedNotation)
            return primitives.write_latex(sym, notation)
        return BACKREF_RE.sub(repl, text)

    _DO_MARKS = {'agree': 'ok', 'exact': 'ok', 'skipped': '??',
                 'disagree': 'XX', 'domain-differs': 'D!'}

    def render_do_step(self, step):
        if step['op'] == 'comment':
            # tex2jax_ignore keeps MathJax from typesetting note prose
            # (stray $...$ or \commands in agent notes stay literal)
            return (f"<div class=\"tex2jax_ignore\" style=\"color:#666\">"
                    f"<code>{step['id']}</code> "
                    f"<em>{_html.escape(step['args']['text'])}</em></div>")
        if step['op'] == 'branch':
            # Exploration topology is annotation, not checked mathematics.
            # The later presentation generation will fold dead branches; for
            # now make the explicit source/reason visible as it is recorded.
            source = _html.escape(step['args']['from'])
            reason = _html.escape(step['args']['reason'])
            return (f'<div class="tex2jax_ignore" style="color:#666">'
                    f'<code>{step["id"]}</code> '
                    f'<strong>branch from {source}</strong> '
                    f'&mdash; <em>{reason}</em></div>')
        check = step['check'].get('status', '?')
        mark = self._DO_MARKS.get(check, '?')
        if mark in ('XX', '?'):
            style = ' style="color:#c00"'
        elif mark == 'D!':
            # conditional, not wrong: the result holds on the common domain
            style = ' style="color:#b65c00"'
        else:
            style = ''
        edge = step.get('exploration') or {}
        if edge:
            branch = (f' (resumes from {_html.escape(edge.get("from", "?"))}'
                      f' via {_html.escape(edge.get("marker", "?"))})')
        else:
            branch = ('' if step.get('continues') in (True, None)
                      else ' (new chain; no marker)')
        note = ''
        if step['op'] == 'apply_both_sides':
            a = step['args']
            note = f" {_html.escape(a['op'] + ' ' + a['arg'])}"
        lines = [f"<div{style}><code>{step['id']}#{step['hash']} "
                 f"[{mark}]{branch} {step['op']}{note}</code> "
                 f"&nbsp;${_display_latex(step['input'])} "
                 f"\\;\\Longrightarrow\\; "
                 f"{_display_latex(step['result'])}$</div>"]
        for a in step['assumptions']:
            lines.append(f'<div style="margin-left:2em;color:#888">'
                         f'assumes {self._assumption_html(a)}</div>')
        return ''.join(lines)

    _CHECK_COLORS = {'agree': '#176b2c', 'exact': '#176b2c',
                     'skipped': '#888', 'domain-differs': '#b65c00',
                     'disagree': '#c00'}

    def render_do_chain(self, steps, topology=None, all_steps=None):
        """End-of-run summary table of a run's verified chain, generated
        from the ledger records themselves — the agent is told never to
        retype it. Returns None only when there is nothing checked to
        show (no transforming steps): a single-step run still renders its
        one-row table, because that row is the only kernel-rendered place
        the check verdict appears — without it, "verified" is agent
        prose, which is exactly what this table exists to replace (live:
        a one-step FTC cell showed its result with no receipt).
        Marker-classified dead paths become collapsed, expandable rows
        instead of bare branch labels."""
        transforming = [s for s in steps if s.get('result') is not None]
        all_transforming = [
            s for s in (all_steps if all_steps is not None else steps)
            if s.get('result') is not None]

        def result_cells(step, nested=False):
            # tolerate minimal step dicts: rendering must never fail a
            # derivation (real recorded steps always carry check/assumptions)
            check = (step.get('check') or {}).get('status', '?')
            color = self._CHECK_COLORS.get(check, '#c00')
            edge = step.get('exploration') or {}
            if edge:
                lineage = (
                    '<div class="tex2jax_ignore" '
                    'style="color:#888;font-size:85%">resumed from '
                    f'{_html.escape(edge.get("from", "?"))} via '
                    f'{_html.escape(edge.get("marker", "?"))}</div>')
            elif step.get('continues') is False and not (
                    topology and (topology.get('parents')
                                  or {}).get(step['id'])):
                lineage = ('<div class="tex2jax_ignore" '
                           'style="color:#888;font-size:85%">'
                           'new chain; no marker</div>')
            else:
                lineage = ''
            note = ''
            if step['op'] == 'apply_both_sides':
                a = step['args']
                note = ' ' + _html.escape(a['op'] + ' ' + a['arg'])
            assum = ''
            if step.get('assumptions'):
                assum = (f' <span style="color:#888">+'
                         f'{len(step["assumptions"])} assum.</span>')
            cell = 'padding:2px 12px 2px 0;text-align:left;' \
                   'vertical-align:top'
            if nested:
                return (
                    '<div style="display:grid;grid-template-columns:'
                    '5em 11em minmax(12em,1fr) 9em;column-gap:12px;'
                    'padding:2px 0">'
                    f'<div><code>{step["id"]}</code>{lineage}</div>'
                    f'<div><code>{_html.escape(step["op"])}{note}</code>'
                    '</div>'
                    f'<div>${_display_latex(step["result"])}$</div>'
                    f'<div style="color:{color}">{check}{assum}</div>'
                    '</div>')
            return (
                '<tr>'
                f'<td style="{cell}"><code>{step["id"]}</code>'
                f'{lineage}</td>'
                f'<td style="{cell}"><code>'
                f'{_html.escape(step["op"])}{note}</code></td>'
                f'<td style="{cell}">${_display_latex(step["result"])}$</td>'
                f'<td style="{cell};color:{color}">{check}{assum}</td>'
                '</tr>')

        topology = topology or {}
        paths_by_insertion = {}
        folded = set()
        current_ids = {s['id'] for s in transforming}
        all_ids = {s['id'] for s in all_transforming}
        for path in topology.get('abandoned_paths') or []:
            ids = [sid for sid in path.get('steps', [])
                   if sid in all_ids]
            if not ids:
                continue
            insertion = next((sid for sid in ids if sid in current_ids), None)
            if insertion is None and path.get('continues_at') in current_ids:
                insertion = path['continues_at']
            paths_by_insertion.setdefault(insertion, []).append(
                dict(path, steps=ids))
            folded.update(sid for sid in ids if sid in current_ids)
        if not transforming and not paths_by_insertion:
            return None
        by_id = {s['id']: s for s in all_transforming}
        rows = []

        def path_row(path):
            count = len(path['steps'])
            target = (f'; resumed as {path["continues_at"]}'
                      if path.get('continues_at') else '')
            body = ''.join(result_cells(by_id[sid], nested=True)
                           for sid in path['steps'])
            cell = ('padding:3px 12px 3px 0;text-align:left;'
                    'vertical-align:top')
            return (
                f'<tr><td colspan="4" style="{cell}">'
                '<details><summary><span class="tex2jax_ignore">'
                '<strong>abandoned path from '
                f'{_html.escape(path["source"])}</strong> &mdash; '
                f'{_html.escape(path["reason"])} '
                f'({count} checked step{"s" if count != 1 else ""}'
                f'{_html.escape(target)})</span></summary>'
                f'{body}</details></td></tr>')

        for step in transforming:
            for path in paths_by_insertion.pop(step['id'], []):
                rows.append(path_row(path))
            if step['id'] in folded:
                continue
            rows.append(result_cells(step))
        for paths in paths_by_insertion.values():
            for path in paths:
                rows.append(path_row(path))
        head = 'padding:2px 12px 2px 0;text-align:left;' \
               'border-bottom:1px solid #8884'
        header = ('<tr>'
                  + ''.join(f'<th style="{head}">{h}</th>'
                            for h in ('step', 'move', 'result', 'check'))
                  + '</tr>')
        return ('<div style="margin-top:4px"><strong>verified chain'
                '</strong> <span style="color:#888">&mdash; rendered '
                'from the selected ledger spine</span></div>'
                '<table style="border-collapse:collapse">'
                + header + ''.join(rows) + '</table>')

    @staticmethod
    def _figure_html(figure):
        """One sandbox figure as HTML, by kind.

        png  - a raster, inlined as a data URI.
        svg  - TikZ output: fonts already inlined, so it still renders
               offline. Delivered as a data-URI <img>, never as inline
               markup: an untrusted notebook's text/html outputs go
               through JupyterLab's sanitizer, which strips <svg> and
               leaves the glyph text as debris, while a data-URI <img>
               is allowlisted. Image context also means the SVG cannot
               script or fetch, in trusted notebooks too.
        html - plotly, which needs its own <script> to run. JupyterLab
               strips scripts from cell output, so it only survives inside
               an iframe, which is a separate browsing context. srcdoc is
               attribute-escaped; the sandbox attribute keeps the figure
               from reaching back into the notebook page.
        """
        kind = figure.get('kind', 'png')
        data = figure.get('data') or ''
        if kind == 'svg':
            payload = base64.b64encode(data.encode('utf-8')).decode('ascii')
            return (f'<div style="max-width:640px;overflow-x:auto">'
                    f'<img src="data:image/svg+xml;base64,{payload}" '
                    f'style="max-width:640px"/></div>')
        if kind == 'html':
            height = int(figure.get('height') or 520)
            return (f'<iframe sandbox="allow-scripts" '
                    f'srcdoc="{_html.escape(data, quote=True)}" '
                    f'style="width:100%;max-width:760px;height:{height}px;'
                    f'border:none"></iframe>')
        return (f'<div><img src="data:image/png;base64,{data}" '
                f'style="max-width:640px"/></div>')

    @staticmethod
    def _show_run_narrative(res):
        if not res.get('summary'):
            return
        label = ('<strong>agent narrative — unverified:</strong> '
                 if res.get('summary_unverified') else '')
        display(HTML(f'<div class="tex2jax_ignore"><em>{label}'
                     f'{_html.escape(res["summary"])}</em></div>'))

    @classmethod
    def _show_run_figures(cls, res):
        """Render run-local illustrations after returning to the kernel thread.

        Provider tools execute on worker threads. Publishing Jupyter output
        there is best-effort and can lose or mis-parent the message, so the
        session buffers successful figures and the final failed attempt for
        this deterministic render pass.
        """
        for illustration in res.get('figures') or []:
            try:
                parts = [cls._figure_html(figure)
                         for figure in illustration.get('figures') or []]
                parts.append(f'<div style="color:#888"><em>'
                             f'{_html.escape(illustration.get("caption", ""))}'
                             f'</em> &mdash; illustration, not machine-checked'
                             f'</div>')
                display(HTML(''.join(parts)))
            except Exception as exc:
                cls._do_error('figure could not be displayed: ' + str(exc))

        failure = res.get('figure_error')
        if failure:
            kind = ('TikZ figure' if failure.get('kind') == 'tikz'
                    else 'plot')
            caption = failure.get('caption') or 'uncaptioned illustration'
            error = failure.get('error') or 'the renderer returned no figure'
            display(HTML(
                f'<div style="color:#b65c00"><strong>{kind} failed:</strong> '
                f'<em>{_html.escape(caption)}</em> &mdash; the mechanically '
                f'checked mathematics is unaffected.'
                f'<pre style="white-space:pre-wrap;margin:.4em 0">'
                f'{_html.escape(error)}</pre></div>'))

    @staticmethod
    def _show_run_premises(res):
        # where this run's checking starts: inputs it stated rather than
        # derived. Without them a laundered assertion is indistinguishable
        # from a derivation.
        if not res.get('premises'):
            return
        import primitives
        stated = ', '.join(
            f'\\({primitives.display_latex(p["input"])}\\)'
            for p in res['premises'])
        count = len(res['premises'])
        display(HTML(
            f'<div style="color:#888">rests on {count} stated '
            f'premise{"s" if count != 1 else ""}, not derived here: '
            f'{stated}</div>'))

    def _show_assumptions(self, assumptions):
        # alternative case hypotheses are listed apart: they hold one
        # at a time, never together
        if not assumptions:
            return
        import primitives
        split = {i for pair in primitives.exclusive_hypotheses(
            assumptions) for i in pair}
        for label, wanted in (('assumptions', False),
                              ('alternative cases', True)):
            shown = [a for i, a in enumerate(assumptions)
                     if (i in split) is wanted]
            if not shown:
                continue
            asm = '; '.join(self._assumption_html(a) for a in shown)
            display(HTML(f'<div style="color:#888">{label}: '
                         f'{asm}</div>'))

    @staticmethod
    def _assumption_html(assumption):
        """One assumption as HTML: with a `display` field, prose stays
        prose (escaped) and only the inline $...$ spans reach MathJax;
        a bare `text` keeps the historical whole-line math wrapping."""
        display = assumption.get('display')
        if display is None:
            return f'${_display_latex(assumption["text"])}$'
        parts = _display_math_spans(display).split('$')
        return ''.join(f'${seg}$' if i % 2 else _html.escape(seg)
                       for i, seg in enumerate(parts))

    @staticmethod
    def render_do_claim(claim):
        verdict = claim.get('verdict', 'open')
        colors = {'established': '#176b2c', 'conditional': '#8a5a00',
                  'supported': '#8a5a00', 'open': '#b00020'}
        conclusion = claim.get('conclusion') or {}
        detail = ''
        if verdict != 'open':
            detail = (f" &mdash; {len(conclusion.get('steps', []))} "
                      f"checked step(s), "
                      f"{len(conclusion.get('assumptions', []))} "
                      f"assumption(s)")
        else:
            detail = (' &mdash; no mechanically checked closing chain '
                      'was recorded')
        return (f'<div style="color:{colors.get(verdict, "#444")}">'
                f'<strong>CLAIM {claim["id"]}: '
                f'{_html.escape(verdict.upper())}</strong>{detail}<br>'
                f'${_display_latex(claim["statement"])}$</div>')

    @staticmethod
    def _do_error(message):
        display(HTML(f'<div style="color:#c00">do! error: '
                     f'{_html.escape(message)}</div>'))

    _CANCEL_LABELS = {'budget_exhausted': 'budget exhausted',
                      'capability_violation': 'capability violation'}

    @classmethod
    def _do_cancelled(cls, res):
        """Amber notice for a stopped run: what was checked is kept."""
        label = cls._CANCEL_LABELS.get(res.get('status'), 'cancelled')
        steps = len([s for s in (res.get('steps') or [])
                     if s.get('result') is not None])
        kept = (f'{steps} mechanically checked step'
                f'{"s" if steps != 1 else ""} preserved' if steps
                else 'nothing was recorded before it stopped')
        display(HTML(
            f'<div style="color:#b65c00"><strong>{_html.escape(label)}'
            f'</strong> &mdash; {kept}. This cell designates no result and '
            'is not available to <code>[[n]]</code>.</div>'))

    @classmethod
    def _do_partial_result(cls, res):
        """A verified value the run reached before it was stopped. Shown as
        preserved work, never as the cell's answer."""
        partial = res.get('partial_result')
        if not partial:
            return
        provenance = res.get('partial_provenance') or {}
        source = provenance.get('step') or provenance.get('claim') or '?'
        display(HTML(
            '<div style="color:#b65c00">partial result from '
            f'<code>{_html.escape(str(source))}</code> '
            f'({_html.escape(str(provenance.get("method", "?")))}) '
            '&mdash; verified before the stop, not designated as the '
            f'answer:</div><div>${_display_latex(partial)}$</div>'))

    def exec_do(self, instruction, execution_count, add_to_history,
                proof_goal=None):
        import agent_do
        if not instruction:
            self._do_error('empty instruction')
            return
        try:
            instruction = self.resolve_backrefs(instruction)
            if proof_goal is not None:
                proof_goal = self.resolve_backrefs(proof_goal)
        except ValueError as e:
            self._do_error(str(e))
            return

        on_step = self._live_log

        try:
            res = agent_do.run_instruction(instruction, ledger=self.ledger,
                                           on_step=on_step,
                                           proof_goal=proof_goal,
                                           route=self.route)
        except agent_do.DoAgentError as e:
            self._do_error(str(e))
            return
        # Rendered before anything else is published, because folding the
        # log replaces the cell's output wholesale: a notice shown first
        # would be swept away with the lines it was meant to survive. Pure
        # rendering off the records, so computing it early changes nothing.
        chain = self.render_do_chain(
            res.get('steps') or [], res.get('branch_topology'),
            all_steps=self.ledger.steps)
        self._live_log.fold(bool(
            chain and res.get('ok') and res.get('final_result')
            and not res.get('cancelled')))
        if res.get('cancelled'):
            # not a failure: the work that was mechanically checked before
            # the interrupt is kept and still replays. Amber, not red — and
            # the cell deliberately produces no chainable output below.
            self._do_cancelled(res)
        elif not res['ok']:
            self._do_error(res.get('error', 'agent failed'))
        elif res.get('turn_limit_reached'):
            # finished on its last turn: the result is designated and
            # verified, but the run had no room left. Say so - a derivation
            # that only just fit is worth knowing about before the next one.
            display(HTML(
                '<div style="color:#a60">do! note: the turn limit was '
                'reached; the result below was committed and verified on '
                'the final turn, so only the closing narrative is '
                'missing.</div>'))
        for claim in res.get('claims', []):
            display(HTML(self.render_do_claim(claim)))
        if chain:
            display(HTML(chain))
        self._show_run_figures(res)
        self._show_run_narrative(res)
        self._show_run_premises(res)
        self._show_assumptions(res['assumptions'])
        if res.get('cancelled'):
            self._do_partial_result(res)
            return          # no output history, no [[n]] backreference
        open_prov = res.get('final_provenance') or {}
        if not res.get('final_result') and open_prov.get('source') == 'open':
            # run-level open outcome: no certified result exists, and no
            # fallback value may stand in for one
            display(HTML(
                '<div style="color:#b00020"><strong>outcome: open — no '
                'certified result in this session.</strong> '
                + _html.escape(open_prov.get('reason', ''))
                + ' <em>(unverified reason)</em></div>'))
        if res['final_result']:
            provenance = res.get('final_provenance') or {}
            if provenance.get('status') == 'unverified':
                reason = provenance.get(
                    'reason', 'not established by a ledger step')
                display(HTML(
                    '<div style="color:#b65c00"><strong>unverified final '
                    'value:</strong> ' + _html.escape(reason) + '</div>'))
            try:
                sym = self.parser.parse(res['final_result'])
                output = self.output(sym, self.parsedNotation,
                                     execution_count, add_to_history)
                display(HTML('$' + output + '$'))
            except Exception:
                # unparseable result: still shown above, just not chainable
                self._do_error('final result could not be parsed for '
                               '[[n]] chaining')

    def exec_composite(self, code, execution_count, add_to_history):
        """Evaluate a cell with inline expr commands ({diff! {int! x^3}}):
        resolve each command to its verified do! result inner-to-outer, then
        combine the results with the expand primitive so the arithmetic glue
        is oracle-checked (no extra LLM call proves the composition)."""
        import agent_do
        import expr_commands
        import primitives
        try:
            text = self.resolve_backrefs(code)
        except ValueError as e:
            self._do_error(str(e))
            return
        try:
            # ellipsis may legitimately appear inside a command argument
            # (the agent interprets it via sum_from_ellipsis); the glue
            # expand() below still rejects it outside command results
            sym, notation = primitives.parse_latex(text, allow_ellipsis=True)
        except primitives.PrimitiveError as e:
            self._do_error(str(e))
            return

        on_step = self._live_log

        def run_selected(instruction, **kwargs):
            kwargs['route'] = self.route
            return agent_do.run_instruction(instruction, **kwargs)

        resolver = expr_commands.ExprResolver(
            notation, Notation(), self.commands, self.ledger, on_step,
            run_selected)
        try:
            root = resolver(sym)
        except (expr_commands.ExprCommandError, agent_do.DoAgentError) as e:
            self._do_error(str(e))
            return

        composite = primitives.write_latex(root, resolver.output_notation)
        # The cell's ledger evidence is rendered HERE, on the kernel
        # thread, from the records themselves: the per-step streaming
        # above is best-effort (a backend may dispatch tool callbacks off
        # the kernel thread and lose the displays), and a cell whose only
        # visible artifact is its final value reads as an assertion.
        chains = [self.render_do_chain(run.get('steps') or [],
                                       run.get('branch_topology'),
                                       all_steps=self.ledger.steps)
                  for run in resolver.subruns]
        # Folded before any of it is published, because the swap replaces
        # the whole cell. Every sub-run must have closed: one that ended
        # open makes its streamed lines the only account of why.
        self._live_log.fold(bool(
            any(chains)
            and all(run.get('final_result') for run in resolver.subruns)))
        for run, chain in zip(resolver.subruns, chains):
            if chain:
                display(HTML(chain))
            self._show_run_narrative(run)
            self._show_run_premises(run)
        assumptions = []
        for run in resolver.subruns:
            for a in run.get('assumptions', []):
                if a not in assumptions:
                    assumptions.append(a)
        for drec in resolver.direct_records:
            for a in drec.get('assumptions', []):
                if a not in assumptions:
                    assumptions.append(a)
        from tactic_registry import _same_spelling
        singles = ([r.get('final_result') for r in resolver.subruns]
                   + [r.get('result') for r in resolver.direct_records])
        if (len(singles) == 1 and singles[0]
                and _same_spelling(composite, singles[0])):
            # the whole cell IS one command: identity composition needs no
            # oracle-checked glue, and the designated result keeps its own
            # spelling instead of being re-expanded (live: expand respelled
            # \frac{\pi}{2|ab|} into a stacked fraction)
            final = singles[0]
        else:
            # verified glue: the numeric oracle proves the composition
            from tactics import core as core_tactics
            rec = core_tactics.expand(composite)
            if rec.get('ok'):
                step = self.ledger.record(rec)
                on_step(step)
                for a in rec.get('assumptions', []):
                    if a not in assumptions:
                        assumptions.append(a)
                final = rec['result']
            else:
                # expand could not combine (rare) - show the resolved
                # composite, but be honest that the glue was not
                # oracle-checked
                self._do_error('composition not verified: '
                               + rec.get('error', 'expand failed'))
                final = composite
        self._show_assumptions(assumptions)
        if final:
            try:
                outsym = self.parser.parse(final)
                output = self.output(outsym, self.parsedNotation,
                                     execution_count, add_to_history)
                display(HTML('$' + output + '$'))
            except Exception:
                self._do_error('final result could not be parsed for '
                               '[[n]] chaining')

    def _resolve_prompt_argument(self, cmd, argument):
        """Resolve inline value commands inside a whole-derivation input.

        ``expr`` historically answered two different questions: may a
        command's result be spliced, and may its argument contain splices.
        Explicit input/output contracts separate them.  A derivation command
        such as ``solve!`` stays non-inline, while its typed mathematical
        input may now contain expression-producing commands.  The assembled
        input gets the same checked ``expand`` glue as an ordinary composite
        cell before it is handed to the agent.
        """
        import agent_do
        import expr_commands
        import primitives

        try:
            text = self.resolve_backrefs(argument)
        except ValueError as e:
            self._do_error(str(e))
            return None

        if not self.has_expr_command(text):
            try:
                expr_commands.validate_command_input(cmd, text)
            except (expr_commands.ExprCommandError,
                    primitives.PrimitiveError) as e:
                self._do_error(str(e))
                return None
            return text

        try:
            sym, notation = primitives.parse_latex(text,
                                                   allow_ellipsis=True)
        except primitives.PrimitiveError as e:
            self._do_error(str(e))
            return None

        on_step = self._live_log

        def run_selected(instruction, **kwargs):
            kwargs['route'] = self.route
            return agent_do.run_instruction(instruction, **kwargs)

        resolver = expr_commands.ExprResolver(
            notation, Notation(), self.commands, self.ledger, on_step,
            run_selected)
        try:
            root = resolver(sym)
            composite = primitives.write_latex(
                root, resolver.output_notation)
            # Check the assembled top-level shape before recording glue.  A
            # relation-valued inner result is valid for solve!/prove!, but
            # must never become an integration or simplification input.
            expr_commands.validate_command_input(cmd, composite)

            from tactic_registry import _same_spelling
            singles = ([r.get('final_result') for r in resolver.subruns]
                       + [r.get('result')
                          for r in resolver.direct_records])
            glue = None
            if (len(singles) == 1 and singles[0]
                    and _same_spelling(composite, singles[0])):
                resolved = singles[0]
            else:
                from tactics import core as core_tactics
                glue = core_tactics.expand(composite)
                if not glue.get('ok'):
                    raise expr_commands.ExprCommandError(
                        'composition not verified: '
                        + glue.get('error', 'expand failed'))
                step = self.ledger.record(glue)
                on_step(step)
                resolved = glue['result']
            expr_commands.validate_command_input(cmd, resolved)
        except (expr_commands.ExprCommandError,
                agent_do.DoAgentError,
                primitives.PrimitiveError) as e:
            self._do_error(str(e))
            return None

        # Provider callbacks may run off the kernel thread, so reproduce the
        # end-of-composite evidence here after every inner run finishes.
        # This is mid-cell — the outer instruction has not run yet — so the
        # cell forfeits folding rather than let that later fold clear these.
        self._live_log.disable()
        for run in resolver.subruns:
            chain = self.render_do_chain(run.get('steps') or [],
                                         run.get('branch_topology'),
                                         all_steps=self.ledger.steps)
            if chain:
                display(HTML(chain))
            self._show_run_narrative(run)
            self._show_run_premises(run)
        assumptions = []
        for run in resolver.subruns:
            for item in run.get('assumptions', []):
                if item not in assumptions:
                    assumptions.append(item)
        for record in resolver.direct_records + ([glue] if glue else []):
            for item in record.get('assumptions', []):
                if item not in assumptions:
                    assumptions.append(item)
        self._show_assumptions(assumptions)
        return resolved

    def process(self, sym, notation):
        return self.processor(sym, notation, self.execution_history, self.history)

    def output(self, outsym, notation, execution_count, add_to_history):
        if add_to_history:
            # snapshot into a private graph: parser.parse() clears
            # parsedNotation in place, which would strand the stored symbol
            # and make later [[n]] references print its raw name (_nNN)
            snapshot = Notation()
            outsym = Replicator(notation, snapshot)(outsym)
            notation = snapshot
            self.execution_history[str(execution_count)] = outsym
            self.history[outsym] = snapshot
        if self.show_quotes:
            return LaTexWriter(notation, show_quotes=True)(outsym)
        # Notebook output is also future [[n]] input.  Use the validated
        # pretty writer so repeated parse/write hops stay stable instead of
        # accumulating one transparent brace layer per hop.
        import primitives
        return primitives.write_latex(outsym, notation)

    def clear(self):
        self.processor.prologModel.clear()

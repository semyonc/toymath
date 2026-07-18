# -*- coding: utf-8 -*-

import html as _html
import re

from notation import Notation, Symbol
from LatexParser import MathParser
from processor import MathProcessor
from LatexWriter import LaTexWriter
from replicator import Replicator
from prolog import PrologModel
from ledger import Ledger
import prompt_commands

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

def split_lines(self, code):
    bracket = 0
    buffer = ''
    for codePoint in code:
        if codePoint == '{':
            bracket += 1
        elif codePoint == '}':
            bracket -= 1
        elif codePoint == '\n' and bracket == 0:
            yield buffer
            buffer = ''
            continue
        elif codePoint == '\r':
            continue
        buffer += codePoint
    if buffer != '':
        yield buffer


class MathShell(object):

    def __init__(self):
        self.history = {}
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
        # commands! reloads the discoverable prompt-command registry.

    def _command_names(self):
        return (set(self.processor.actions) | set(self.commands)
                | set(prompt_commands.RESERVED))

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
        stripped = code.strip()
        m = CMD_PREFIX_RE.match(stripped)
        if m and self.dispatch_command(m.group(1), stripped[m.end():].strip(),
                                       execution_count, add_to_history):
            return
        if self.has_expr_command(stripped):
            self.exec_composite(stripped, execution_count, add_to_history)
            return
        lines = [line for line in split_lines(self, code)]
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
        instruction = prompt_commands.render(cmd, rest)
        proof_goal = rest if cmd.mode == 'prove' else None
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
                + (f' <code>[direct: {_html.escape(c.direct)}]</code>'
                   if c.direct else '')
                + '</td></tr>'
                for c in sorted(self.commands.values())]
        table = ('<table>' + ''.join(rows) + '</table>') if rows else (
            '<div style="color:#888">no commands defined yet '
            '(add <code>commands/&lt;name&gt;.md</code> files)</div>')
        display(HTML('<div><b>notebook commands</b>' + table
                     + '<div style="color:#888;margin-top:4px">plus '
                     '<code>do!</code> (free-form instruction) and '
                     '<code>commands!</code> (this list)</div></div>'))

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
        check = step['check'].get('status', '?')
        mark = self._DO_MARKS.get(check, '?')
        if mark in ('XX', '?'):
            style = ' style="color:#c00"'
        elif mark == 'D!':
            # conditional, not wrong: the result holds on the common domain
            style = ' style="color:#b65c00"'
        else:
            style = ''
        branch = '' if step.get('continues') in (True, None) else ' (branch)'
        note = ''
        if step['op'] == 'apply_both_sides':
            a = step['args']
            note = f" {_html.escape(a['op'] + ' ' + a['arg'])}"
        lines = [f"<div{style}><code>{step['id']}#{step['hash']} "
                 f"[{mark}]{branch} {step['op']}{note}</code> "
                 f"&nbsp;${step['input']} \\;\\Longrightarrow\\; "
                 f"{step['result']}$</div>"]
        for a in step['assumptions']:
            lines.append(f'<div style="margin-left:2em;color:#888">'
                         f'assumes {self._assumption_html(a)}</div>')
        return ''.join(lines)

    _CHECK_COLORS = {'agree': '#176b2c', 'exact': '#176b2c',
                     'skipped': '#888', 'domain-differs': '#b65c00',
                     'disagree': '#c00'}

    def render_do_chain(self, steps):
        """End-of-run summary table of a run's verified chain, generated
        from the ledger records themselves — the agent is told never to
        retype it. Returns None when a table would add nothing (fewer
        than two transforming steps)."""
        rows = []
        for step in steps:
            if step.get('result') is None:
                continue  # comments are strategy notes, not chain links
            check = step['check'].get('status', '?')
            color = self._CHECK_COLORS.get(check, '#c00')
            branch = ('<div style="color:#888;font-size:85%">(branch)'
                      '</div>'
                      if step.get('continues') is False else '')
            note = ''
            if step['op'] == 'apply_both_sides':
                a = step['args']
                note = ' ' + _html.escape(a['op'] + ' ' + a['arg'])
            assum = ''
            if step['assumptions']:
                assum = (f' <span style="color:#888">+'
                         f'{len(step["assumptions"])} assum.</span>')
            cell = 'padding:2px 12px 2px 0;text-align:left;' \
                   'vertical-align:top'
            rows.append(
                '<tr>'
                f'<td style="{cell}"><code>{step["id"]}</code>'
                f'{branch}</td>'
                f'<td style="{cell}"><code>'
                f'{_html.escape(step["op"])}{note}</code></td>'
                f'<td style="{cell}">${step["result"]}$</td>'
                f'<td style="{cell};color:{color}">{check}{assum}</td>'
                '</tr>')
        if len(rows) < 2:
            return None
        head = 'padding:2px 12px 2px 0;text-align:left;' \
               'border-bottom:1px solid #8884'
        header = ('<tr>'
                  + ''.join(f'<th style="{head}">{h}</th>'
                            for h in ('step', 'move', 'result', 'check'))
                  + '</tr>')
        return ('<div style="margin-top:4px"><strong>verified chain'
                '</strong> <span style="color:#888">&mdash; rendered '
                'from the ledger</span></div>'
                '<table style="border-collapse:collapse">'
                + header + ''.join(rows) + '</table>')

    @staticmethod
    def _figure_html(figure):
        """One sandbox figure as HTML, by kind.

        png  - a raster, inlined as a data URI.
        svg  - TikZ output: inert markup with its fonts already inlined,
               so it drops straight in and still renders offline.
        html - plotly, which needs its own <script> to run. JupyterLab
               strips scripts from cell output, so it only survives inside
               an iframe, which is a separate browsing context. srcdoc is
               attribute-escaped; the sandbox attribute keeps the figure
               from reaching back into the notebook page.
        """
        kind = figure.get('kind', 'png')
        data = figure.get('data') or ''
        if kind == 'svg':
            # tex2jax_ignore: the glyphs are already typeset, MathJax must
            # not re-scan them for $...$ delimiters
            return (f'<div class="tex2jax_ignore" '
                    f'style="max-width:640px;overflow-x:auto">{data}</div>')
        if kind == 'html':
            height = int(figure.get('height') or 520)
            return (f'<iframe sandbox="allow-scripts" '
                    f'srcdoc="{_html.escape(data, quote=True)}" '
                    f'style="width:100%;max-width:760px;height:{height}px;'
                    f'border:none"></iframe>')
        return (f'<div><img src="data:image/png;base64,{data}" '
                f'style="max-width:640px"/></div>')

    @staticmethod
    def _assumption_html(assumption):
        """One assumption as HTML: with a `display` field, prose stays
        prose (escaped) and only the inline $...$ spans reach MathJax;
        a bare `text` keeps the historical whole-line math wrapping."""
        display = assumption.get('display')
        if display is None:
            return f'${assumption["text"]}$'
        parts = display.split('$')
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
                f'${claim["statement"]}$</div>')

    @staticmethod
    def _do_error(message):
        display(HTML(f'<div style="color:#c00">do! error: '
                     f'{_html.escape(message)}</div>'))

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

        def on_step(step):
            try:
                display(HTML(self.render_do_step(step)))
            except Exception:
                pass  # rendering must never fail the derivation step

        def on_plot(caption, figures):
            try:
                parts = [self._figure_html(f) for f in figures]
                parts.append(f'<div style="color:#888"><em>'
                             f'{_html.escape(caption)}</em> '
                             f'&mdash; illustration, not machine-checked'
                             f'</div>')
                display(HTML(''.join(parts)))
            except Exception:
                pass  # rendering must never fail the plot call

        try:
            res = agent_do.run_instruction(instruction, ledger=self.ledger,
                                           on_step=on_step,
                                           on_plot=on_plot,
                                           proof_goal=proof_goal)
        except agent_do.DoAgentError as e:
            self._do_error(str(e))
            return
        if not res['ok']:
            self._do_error(res.get('error', 'agent failed'))
        for claim in res.get('claims', []):
            display(HTML(self.render_do_claim(claim)))
        chain = self.render_do_chain(res.get('steps') or [])
        if chain:
            display(HTML(chain))
        if res.get('summary'):
            label = ('<strong>agent narrative — unverified:</strong> '
                     if res.get('summary_unverified') else '')
            display(HTML(f'<div class="tex2jax_ignore"><em>{label}'
                         f'{_html.escape(res["summary"])}</em></div>'))
        if res['assumptions']:
            asm = '; '.join(self._assumption_html(a)
                            for a in res['assumptions'])
            display(HTML(f'<div style="color:#888">assumptions: '
                         f'{asm}</div>'))
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

        def on_step(step):
            try:
                display(HTML(self.render_do_step(step)))
            except Exception:
                pass  # rendering must never fail a derivation step

        resolver = expr_commands.ExprResolver(
            notation, Notation(), self.commands, self.ledger, on_step,
            agent_do.run_instruction)
        try:
            root = resolver(sym)
        except (expr_commands.ExprCommandError, agent_do.DoAgentError) as e:
            self._do_error(str(e))
            return

        composite = primitives.write_latex(root, resolver.output_notation)
        # verified glue: the numeric oracle proves the composition
        from tactics import core as core_tactics
        rec = core_tactics.expand(composite)
        assumptions = []
        for run in resolver.subruns:
            for a in run.get('assumptions', []):
                if a not in assumptions:
                    assumptions.append(a)
        for drec in resolver.direct_records:
            for a in drec.get('assumptions', []):
                if a not in assumptions:
                    assumptions.append(a)
        if rec.get('ok'):
            step = self.ledger.record(rec)
            on_step(step)
            for a in rec.get('assumptions', []):
                if a not in assumptions:
                    assumptions.append(a)
            final = rec['result']
        else:
            # expand could not combine (rare) - show the resolved composite,
            # but be honest that the glue was not oracle-checked
            self._do_error('composition not verified: '
                           + rec.get('error', 'expand failed'))
            final = composite
        if assumptions:
            asm = '; '.join(f'${a["text"]}$' for a in assumptions)
            display(HTML(f'<div style="color:#888">assumptions: {asm}</div>'))
        if final:
            try:
                outsym = self.parser.parse(final)
                output = self.output(outsym, self.parsedNotation,
                                     execution_count, add_to_history)
                display(HTML('$' + output + '$'))
            except Exception:
                self._do_error('final result could not be parsed for '
                               '[[n]] chaining')

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

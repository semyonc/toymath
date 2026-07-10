# -*- coding: utf-8 -*-

import html as _html
import re

from notation import Notation, Symbol
from LatexParser import MathParser
from processor import MathProcessor
from LatexWriter import LaTexWriter
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
        self.parser = MathParser(self.parsedNotation)
        self.processor = MathProcessor(model=PrologModel())
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
        # discoverable do!-style commands from the repo commands/ directory;
        # commands! reloads this registry so newly-added files go live
        self.commands = prompt_commands.load_commands()

    def trace_step(self, sym, notation, index):
        if self.trace:
            writer = LaTexWriter(notation)
            if self.trace_output != '':
                self.trace_output += ' \\\\'
            self.trace_output += f'{{{writer(sym)}}} \\tag{{{index}}}'

    def add_action(self, name, instance):
        self.processor.actions[name] = instance

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
        if not rest:
            self._do_error(f'{name}! needs an argument')
            return True
        instruction = prompt_commands.render(cmd, rest)
        self.exec_do(instruction, execution_count, add_to_history)
        return True

    def show_commands(self):
        """Render the discoverable command list. Reloads the registry first
        so a newly-added commands/*.md file goes live without a restart."""
        self.commands = prompt_commands.load_commands()
        rows = ['<tr><td style="padding:2px 14px 2px 0;vertical-align:top">'
                f'<code>{_html.escape(c.name)}!</code></td>'
                f'<td style="color:#444">{_html.escape(c.description)}</td>'
                '</tr>'
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
        def repl(m):
            key = m.group(1)
            sym = self.execution_history.get(key)
            if sym is None:
                raise ValueError(
                    f'[[{key}]] does not reference a previous result')
            notation = self.history.get(sym, self.parsedNotation)
            return LaTexWriter(notation)(sym)
        return BACKREF_RE.sub(repl, text)

    _DO_MARKS = {'agree': 'ok', 'exact': 'ok', 'skipped': '??',
                 'disagree': 'XX'}

    def render_do_step(self, step):
        check = step['check'].get('status', '?')
        mark = self._DO_MARKS.get(check, '?')
        style = ' style="color:#c00"' if mark in ('XX', '?') else ''
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
                         f'assumes ${a["text"]}$</div>')
        return ''.join(lines)

    @staticmethod
    def _do_error(message):
        display(HTML(f'<div style="color:#c00">do! error: '
                     f'{_html.escape(message)}</div>'))

    def exec_do(self, instruction, execution_count, add_to_history):
        import agent_do
        if not instruction:
            self._do_error('empty instruction')
            return
        try:
            instruction = self.resolve_backrefs(instruction)
        except ValueError as e:
            self._do_error(str(e))
            return

        def on_step(step):
            try:
                display(HTML(self.render_do_step(step)))
            except Exception:
                pass  # rendering must never fail the derivation step

        def on_plot(caption, images):
            try:
                parts = [f'<div><img src="data:image/png;base64,{b64}" '
                         f'style="max-width:640px"/></div>'
                         for b64 in images]
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
                                           on_plot=on_plot)
        except agent_do.DoAgentError as e:
            self._do_error(str(e))
            return
        if not res['ok']:
            self._do_error(res.get('error', 'agent failed'))
        if res.get('summary'):
            display(HTML(f'<div><em>{_html.escape(res["summary"])}'
                         f'</em></div>'))
        if res['assumptions']:
            asm = '; '.join(f'${a["text"]}$' for a in res['assumptions'])
            display(HTML(f'<div style="color:#888">assumptions: '
                         f'{asm}</div>'))
        if res['final_result']:
            try:
                sym = self.parser.parse(res['final_result'])
                output = self.output(sym, self.parsedNotation,
                                     execution_count, add_to_history)
                display(HTML('$' + output + '$'))
            except Exception:
                # unparseable result: still shown above, just not chainable
                self._do_error('final result could not be parsed for '
                               '[[n]] chaining')

    def process(self, sym, notation):
        return self.processor(sym, notation, self.execution_history, self.history)

    def output(self, outsym, notation, execution_count, add_to_history):
        if add_to_history:
            self.execution_history[str(execution_count)] = outsym
            self.history[outsym] = notation
        writer = LaTexWriter(notation, show_quotes=self.show_quotes)
        result = writer(outsym)
        return result

    def clear(self):
        self.processor.prologModel.clear()

# -*- coding: utf-8 -*-

import html as _html
import re

from notation import Notation, Symbol
from LatexParser import MathParser
from processor import MathProcessor
from LatexWriter import LaTexWriter
from prolog import PrologModel
from ledger import Ledger

from IPython.display import HTML, Javascript
from engine import display

BACKREF_RE = re.compile(r'\[\[\s*(\d+)\s*\]\]')

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
        if stripped.startswith('do!'):
            # agent endpoint: the whole rest of the cell is the instruction
            self.exec_do(stripped[len('do!'):].strip(), execution_count,
                         add_to_history)
            return
        lines = [line for line in split_lines(self, code)]
        for index, line in enumerate(lines):
            last = index == len(lines) - 1
            self.exec_stmt(line, execution_count, add_to_history and last, last)

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

        try:
            res = agent_do.run_instruction(instruction, ledger=self.ledger,
                                           on_step=on_step)
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

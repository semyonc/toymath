# -*- coding: utf-8 -*-

from notation import Notation, Symbol
from LatexParser import MathParser
from processor import MathProcessor
from LatexWriter import LaTexWriter

from IPython.display import HTML, Javascript
from engine import display


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
        self.processor = MathProcessor()
        self.processor.trace = self.trace_step
        self.history = {}
        self.execution_history = {}
        self.echo_mode = False
        self.current_echo = False
        self.trace = False
        self.trace_mode = False
        self.trace_output = None

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

    def exec(self, code, execution_count, add_to_history=False, cell_id=None):
        lines = [line for line in split_lines(self, code)]
        for index, line in enumerate(lines):
            last = index == len(lines) - 1
            self.exec_stmt(line, execution_count, add_to_history and last, last)

    def exec_stmt(self, code, execution_count, add_to_history, do_output):
        self.current_echo = False
        self.trace = self.trace_mode
        self.trace_output = ''
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

    def process(self, sym, notation):
        return self.processor(sym, notation, self.execution_history, self.history)

    def output(self, outsym, notation, execution_count, add_to_history):
        if add_to_history:
            self.execution_history[execution_count] = outsym
            self.history[outsym] = notation
        writer = LaTexWriter(notation)
        result = writer(outsym)
        return result

    def clear(self):
        self.processor.prologModel.clear()

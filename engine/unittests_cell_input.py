#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the rendered view of a cell's input (cell_input.py).

The contract under test is conservative: a cell renders only when the engine
reads it as a formula and the rendered spelling parses back to the same
expression. Everything else keeps its source, so a cell can never show
something other than what it runs.
"""
import time
import unittest
from unittest import mock

import cell_input
import primitives
import prompt_commands
from processor import MathProcessor
from prolog import PrologModel


def command_names():
    """The bang words a live shell knows: actions, commands, built-ins."""
    return (set(MathProcessor(model=PrologModel()).actions)
            | set(prompt_commands.load_commands())
            | set(prompt_commands.RESERVED))


class TestSplitLines(unittest.TestCase):
    def test_brace_group_keeps_its_newlines(self):
        cell = 'x + 1\n\\frac{a\n+ b}{c}\ny'
        self.assertEqual(list(cell_input.split_lines(cell)),
                         ['x + 1', '\\frac{a\n+ b}{c}', 'y'])


class TestProseVeto(unittest.TestCase):
    def test_the_parser_cannot_do_this_job(self):
        # documents why the veto is lexical: the parser reads prose as a
        # product of one-letter symbols and accepts it
        sym, notation = primitives.parse_latex('the derivative')
        self.assertTrue(primitives.write_latex(sym, notation))
        self.assertTrue(cell_input.has_prose('the derivative'))

    def test_macros_and_braced_names_are_not_prose(self):
        self.assertFalse(cell_input.has_prose(
            r'\int \frac{\sin x d x}{\sin ^3 x+\cos ^3 x}'))
        self.assertFalse(cell_input.has_prose(r'\operatorname{child}(#X)'))

    def test_function_names_survive_macro_stripping(self):
        self.assertFalse(cell_input.has_prose(r'(sin x + cos x)^2 - 1'))
        self.assertTrue(cell_input.has_prose('solve equation x^3 + 1 = 0'))


class TestPreview(unittest.TestCase):
    def setUp(self):
        self.names = command_names()

    def preview(self, code):
        return cell_input.preview(code, self.names)   # no prose commands

    def test_plain_formula_renders_in_the_engines_own_spelling(self):
        # the ToyMath dialect takes a parenthesised second operand; the
        # rendered form is what the engine understood, not what was typed
        segments = self.preview(r'\int \frac {dx} (x^{\frac 1 2} + 1)')
        self.assertEqual([s['kind'] for s in segments], ['math'])
        self.assertEqual(segments[0]['latex'],
                         r'\int\frac {dx} {(x^{\frac {1} {2}}+1)}')

    def test_command_cell_renders_as_label_and_formula(self):
        segments = self.preview(r'int! \int x^2 dx')
        self.assertEqual(segments[0], {'kind': 'command', 'text': 'int!'})
        self.assertEqual(segments[1]['kind'], 'math')
        self.assertIn(r'\int', segments[1]['latex'])

    def test_legacy_action_cell_also_renders(self):
        segments = self.preview(r'mul! (x+1)(x-1)')
        self.assertEqual(segments[0], {'kind': 'command', 'text': 'mul!'})
        self.assertEqual(segments[1]['latex'], '(x+1)(x-1)')

    def test_factorial_is_not_a_command_prefix(self):
        segments = self.preview('n! + 1')
        self.assertEqual([s['kind'] for s in segments], ['math'])
        self.assertEqual(segments[0]['latex'], 'n!+1')

    def test_multi_line_cell_renders_every_statement(self):
        segments = self.preview('x^2\ny^3')
        self.assertEqual([s['kind'] for s in segments],
                         ['math', 'break', 'math'])
        self.assertEqual(segments[0]['latex'], 'x^{2}')
        self.assertEqual(segments[2]['latex'], 'y^{3}')

    def test_a_reference_with_no_result_yet_renders_as_itself(self):
        # a fresh kernel has no history, and a cell that dropped back to raw
        # source next to its rendered siblings reads as an unsupported command
        segments = self.preview('simplify! [[1]]')
        self.assertEqual(segments, [{'kind': 'command', 'text': 'simplify!'},
                                    {'kind': 'ref', 'text': '[[1]]'}])

    def test_a_reference_inside_a_formula_keeps_the_source(self):
        # `x^2 +` is not a formula on its own, and what the reference stands
        # for is exactly what is missing: an unresolved one renders only
        # where it is a whole statement
        self.assertIsNone(self.preview('int! x^2 + [[3]]'))

    def test_a_reference_does_not_rescue_an_unparsable_cell(self):
        self.assertIsNone(self.preview('diff! [x] [[3]]'))

    def test_configuration_commands_keep_their_source(self):
        for cell in ('model! xiaomi/mimo-v2.5', 'backend! codex',
                     'login! status', 'commands!', 'help!'):
            self.assertIsNone(self.preview(cell), cell)

    def test_prose_cell_keeps_its_source(self):
        self.assertIsNone(self.preview('solve equation x^3 + 1 = 0'))

    def test_unparsable_cell_keeps_its_source(self):
        self.assertIsNone(self.preview(r'\frac{1}{'))

    def test_empty_cell_renders_nothing(self):
        self.assertIsNone(self.preview('   \n  '))
        self.assertIsNone(self.preview('rules!'))

    def test_a_line_the_writer_cannot_spell_back_keeps_its_source(self):
        # the writer drops the outer command of `track! {goal! ...}`; the
        # round-trip guard catches the loss instead of rendering a formula
        # that says something else than the cell runs
        cell = r'track! {goal! \operatorname{child}(#Q)}'
        sym, notation = primitives.parse_latex(cell, command_names=self.names)
        self.assertNotIn('track', primitives.write_latex(sym, notation))
        self.assertIsNone(self.preview(cell))

    def test_round_trip_guard_rejects_a_lossy_writer(self):
        with mock.patch.object(primitives, 'write_latex',
                               return_value='x^{3}'):
            self.assertIsNone(self.preview('x^2'))


def spell(segments):
    """A whole segmentation as one comparable line.

    Formulas come back in the engine's spelling between brackets, so a case
    records both where a formula was found and what it was understood to be.
    """
    if segments is None:
        return None
    parts = []
    for segment in segments:
        if segment['kind'] == 'math':
            parts.append('[' + segment['latex'] + ']')
        elif segment['kind'] in ('command', 'ref'):
            parts.append('<' + segment['text'] + '>')
        elif segment['kind'] == 'break':
            parts.append(' / ')
        else:
            parts.append(segment['text'])
    return ''.join(parts)


# Prompts as they are actually typed — no `$…$` anywhere — against the
# segmentation each one should get. The first block is taken from the
# repository's own notebooks; the second is adversarial, and is where the
# scan is meant to find *nothing*.
PROSE_CASES = [
    ('differentiate x³−3x, find where the derivative is zero',
     'differentiate [x^{3}-3x], find where the derivative is zero'),
    ('solve equation x^3 + 1 = 0',
     'solve equation [{x^{3}+1}=0]'),
    (r'plot f(x)=x \sin \frac{1}{x}',
     'plot [f(x)=x \\sin\\frac {1} {x}]'),
    ('show that (sin x + cos x)² − 1 = sin(2x)',
     'show that [{(sinx+cosx)^{2}-1}=sin(2x)]'),
    (r'Establish \sum_{k=0}^{\infty} (\frac{1}{2})^k = 2 with verified tactics',
     'Establish [\\sum_{k=0}^\\infty ( \\frac {1} {2})^k=2] with verified '
     'tactics'),
    # the article survives, and the measurement is one fragment
    ('render a 2×1 rectangle cut into pieces',
     'render a [2 \\times 1] rectangle cut into pieces'),
    # a formula wrapping across one line break is still one formula
    ('the points p_n = \\prod_{k=1}^{n}\n\\frac{2k-1}{2k} for later',
     'the points [p_n= \\prod_{k=1}^n \\frac {2k-1} {2k}] for later'),
    # `as a` is prose the scan has to give back
    (r'Re-derive \lim_{n \to \infty} \frac{n}{2^n} = 0 as a chain',
     'Re-derive [\\lim_{n \\to\\infty}\\frac {n} {2^n}=0] as a chain'),
    # a result reference inside prose keeps its own chip
    (r'prove [[2]] is \int \frac{dx}{x}',
     'prove <[[2]]> is [\\int\\frac {dx} {x}]'),

    # --- adversarial: nothing here is a formula ------------------------
    ('draw an example of the commutative diagram', None),
    ('plot same inverted function', None),
    ('use integrate_by_parts, then the limit chain', None),
    ('the derivative is zero at the origin', None),
    ('compare a to b and note it is done', None),
    # a range is not a value: rendering `n = 1` would drop the `..20`
    ('the points for n = 1..20 exactly', None),
    # a version number is not an equation
    ('install pandas 2.1 and matplotlib 3.9', None),
]


class TestProseSegmentation(unittest.TestCase):
    """The heuristic that finds formulas inside an unescaped prompt."""

    def test_corpus(self):
        for prompt, expected in PROSE_CASES:
            with self.subTest(prompt=prompt):
                self.assertEqual(spell(cell_input.prose_segments(prompt)),
                                 expected)

    def test_nothing_the_prompt_says_is_dropped(self):
        """Every character is either prose or inside a formula's source.

        This is what makes a wrong span boundary survivable: the scan may
        typeset less of a sentence than it should, but it can never make part
        of the prompt disappear from the view.
        """
        for prompt, _ in PROSE_CASES:
            with self.subTest(prompt=prompt):
                rebuilt, position = [], 0
                for start, end, _latex in cell_input.formula_spans(prompt):
                    rebuilt.append(prompt[position:start])
                    rebuilt.append(prompt[start:end])
                    position = end
                rebuilt.append(prompt[position:])
                self.assertEqual(''.join(rebuilt), prompt)

    def test_an_incomplete_formula_leaves_its_operator_in_the_prose(self):
        # `x^2 +` cannot render as a formula, and the `+` is not swallowed:
        # it stays visible in the sentence
        self.assertEqual(spell(cell_input.prose_segments('add x^2 + so on')),
                         'add [x^{2}] + so on')

    def test_macro_tier_alone_leaves_bare_formulas_as_prose(self):
        macro = 'differentiate x³−3x and \\frac{1}{x}'
        self.assertEqual(cell_input.prose_segments(macro, bare_seeds=False),
                         [{'kind': 'text', 'text': 'differentiate x³−3x and '},
                          {'kind': 'math', 'latex': '\\frac {1} {x}'}])
        self.assertEqual(
            spell(cell_input.prose_segments(macro, bare_seeds=True)),
            'differentiate [x^{3}-3x] and [\\frac {1} {x}]')

    def test_debris_from_an_unreadable_construct_is_not_a_formula(self):
        # ToyMath's parser has no `aligned`; the scan must not render the
        # bare `\begin` it can still parse as an opaque symbol
        prompt = ('prove \\begin{aligned} a &= b \\end{aligned} if n > 1')
        self.assertEqual(spell(cell_input.prose_segments(prompt)),
                         'prove \\begin{aligned} a &= b \\end{aligned} if '
                         '[n \\gt 1]')

    def test_a_long_prompt_stays_fast(self):
        # the backtracking retries parses; a pathological prompt must not
        # turn a keystroke-free render into a visible pause
        prompt = ('prove \\begin{aligned}\n& \\int \\frac{d x}'
                  '{(a+b \\cos x)^n}=\\frac{A \\sin x}{(a+b \\cos x)^{n-1}}'
                  '+ \\\\\n& +B \\int \\frac{d x}{(a+b \\cos x)^{n-1}}'
                  '\\end{aligned} if n > 1. Find A,B and C') * 4
        start = time.monotonic()
        cell_input.prose_segments(prompt)
        self.assertLess(time.monotonic() - start, 2.0)


class TestDoPromptPreview(unittest.TestCase):
    """Only a command that hands its argument to the agent may carry prose."""

    def setUp(self):
        self.names = command_names()
        self.prose = set(prompt_commands.load_commands()) | {'do'}

    def preview(self, code):
        return cell_input.preview(code, self.names, self.prose)

    def test_do_prompt_renders_prose_with_its_formulas(self):
        segments = self.preview(r'do! show that (sin x + cos x)² − 1 = sin(2x)')
        self.assertEqual(segments[0], {'kind': 'command', 'text': 'do!'})
        self.assertEqual(spell(segments),
                         '<do!>show that [{(sinx+cosx)^{2}-1}=sin(2x)]')

    def test_a_prompt_with_no_formula_keeps_its_source(self):
        self.assertIsNone(
            self.preview('do! draw an example of the commutative diagram'))

    def test_a_command_argument_is_read_as_a_formula_first(self):
        # `int! \int x^2 dx` is one formula, not a sentence containing one
        segments = self.preview(r'int! \int x^2 dx')
        self.assertEqual([s['kind'] for s in segments], ['command', 'math'])

    def test_a_command_argument_that_is_prose_falls_back_to_the_scan(self):
        segments = self.preview(r'prove! show that x^2 \ge 0')
        self.assertEqual(spell(segments), '<prove!>show that [x^{2}\\ge 0]')

    def test_a_rewrite_action_is_never_prose_scanned(self):
        # `mul!` runs its argument through the fixed-point engine as one
        # expression; reading it as a sentence would describe it wrongly
        self.assertIsNone(self.preview('mul! expand the product (x+1)(x-1)'))

    def test_a_plain_cell_is_never_prose_scanned(self):
        self.assertIsNone(self.preview('solve equation x^3 + 1 = 0'))

    def test_configuration_commands_are_never_prose_scanned(self):
        for cell in ('model! z-ai/glm-5.2', 'backend! codex', 'login! status'):
            self.assertIsNone(self.preview(cell), cell)


class TestShellPreview(unittest.TestCase):
    def setUp(self):
        from mathShell import MathShell
        self.shell = MathShell()

    def test_preview_is_read_only(self):
        before = len(self.shell.history), len(self.shell.execution_history)
        self.shell.preview_cell(r'int! \int x^2 dx')
        self.assertEqual(
            (len(self.shell.history), len(self.shell.execution_history)),
            before)

    def test_backreference_renders_as_the_formula_it_stands_for(self):
        import engine
        engine.setHandler(lambda *objs, **kw: None)
        self.shell.exec('x^2 + 1', 1, add_to_history=True)
        segments = self.shell.preview_cell('int! [[1]]')
        self.assertEqual(segments[0]['text'], 'int!')
        self.assertIn('x^{2}', segments[1]['latex'])

    def test_unresolved_backreference_still_renders_the_command(self):
        self.assertEqual(self.shell.preview_cell('int! [[7]]'),
                         [{'kind': 'command', 'text': 'int!'},
                          {'kind': 'ref', 'text': '[[7]]'}])


class TestRenderComm(unittest.TestCase):
    """The kernel side of the comm: one reply per request, never a raise."""

    def setUp(self):
        from mathShell import MathShell
        from toymathkernel import MathKernel
        self.kernel = mock.Mock(spec=['mathShell', 'log'])
        self.kernel.mathShell = MathShell()
        self.render = lambda comm, msg: MathKernel._render_cell(
            self.kernel, comm, msg)

    @staticmethod
    def message(code, ident='7'):
        return {'content': {'data': {'id': ident, 'code': code}}}

    def test_reply_carries_the_request_id_and_segments(self):
        comm = mock.Mock(spec=['send'])
        self.render(comm, self.message('x^2'))
        payload = comm.send.call_args[0][0]
        self.assertEqual(payload['id'], '7')
        self.assertEqual(payload['segments'][0]['latex'], 'x^{2}')

    def test_a_cell_that_does_not_render_still_gets_a_reply(self):
        comm = mock.Mock(spec=['send'])
        self.render(comm, self.message('do! find the derivative'))
        self.assertIsNone(comm.send.call_args[0][0]['segments'])

    def test_a_failing_preview_answers_instead_of_raising(self):
        comm = mock.Mock(spec=['send'])
        with mock.patch.object(type(self.kernel.mathShell), 'preview_cell',
                               side_effect=RuntimeError('boom')):
            self.render(comm, self.message('x^2'))
        self.assertIsNone(comm.send.call_args[0][0]['segments'])


if __name__ == '__main__':
    unittest.main()

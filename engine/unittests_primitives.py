#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the agent-scoped verified-derivation primitives."""
import os
import json
import tempfile
import unittest

from notation import Notation
from LatexParser import MathParser
from polyrat import (Poly, RatFunc, NotInFragment, to_ratfunc,
                     ratfunc_to_notation)
import primitives as P
import ledger as ledger_module
from tactics import core as Core
from tactics import differentiation as Differentiation
from tactics import equations as Equations
from tactics import finite_operators as FiniteOperators
from tactics import integration as Integration
from tactics import limits as Limits
from tactics import matrices as Matrices
from ledger import Ledger


def rf(latex):
    n = Notation()
    s = MathParser(n).parse(latex)
    return to_ratfunc(s, n)


def canon(latex):
    r = rf(latex)
    out = Notation()
    from LatexWriter import LaTexWriter
    return ' '.join(LaTexWriter(out)(ratfunc_to_notation(r, out)).split())


class TestPolyRat(unittest.TestCase):
    def test_expand_product(self):
        self.assertEqual(rf('(x+1)(x-1)'), rf('x^2 - 1'))

    def test_binomial_cube(self):
        self.assertEqual(rf('(x+1)^3'), rf('x^3 + 3x^2 + 3x + 1'))

    def test_equal_denominators(self):
        # probe failure of the old rewrite rules: a/b + c/b must not cross-multiply
        self.assertEqual(rf('\\frac{a}{b} + \\frac{c}{b}'),
                         rf('\\frac{a+c}{b}'))

    def test_cancellation_to_one(self):
        r = rf('\\frac{x}{x+1} + \\frac{1}{x+1}')
        self.assertTrue(r.is_const())
        self.assertEqual(r.const_value(), 1)

    def test_univariate_gcd(self):
        self.assertEqual(rf('\\frac{x^2-1}{x+1}'), rf('x - 1'))

    def test_monomial_gcd(self):
        self.assertEqual(rf('\\frac{3x^2}{x^3}'), rf('\\frac{3}{x}'))

    def test_merge_like_factors(self):
        self.assertEqual(rf('x^2 x'), rf('x^3'))

    def test_negative_power(self):
        self.assertEqual(rf('x^{-2}'), rf('\\frac{1}{x^2}'))

    def test_numeric_fractions(self):
        r = rf('\\frac{1}{2} + \\frac{1}{3}')
        self.assertEqual(r.const_value().numerator, 5)
        self.assertEqual(r.const_value().denominator, 6)

    def test_sign_normalization(self):
        self.assertEqual(rf('\\frac{x}{-y}'), rf('-\\frac{x}{y}'))

    def test_functions_not_in_fragment(self):
        with self.assertRaises(NotInFragment):
            rf('\\sin x')

    def test_zero_denominator(self):
        with self.assertRaises(ZeroDivisionError):
            rf('\\frac{x}{0}')

    def test_multivariate_exact_division_cancels(self):
        # exact trial division in the constructor: complete cancellation
        # when one side divides the other, even multivariate
        r = rf('\\frac{x^2-y^2}{x+y}')
        self.assertTrue(r.is_poly())
        self.assertEqual(r, rf('x-y'))
        self.assertEqual(rf('\\frac{x+y}{(x+y)^2}'),
                         rf('\\frac{1}{x+y}'))

    def test_partial_common_factor_stays(self):
        # no full multivariate GCD: partial factors stay uncancelled but
        # cross-multiplying equality still decides exactly
        r = rf('\\frac{(x+y)(x-z)}{(x+y)(x-w)}')
        self.assertFalse(r.is_poly())
        self.assertEqual(r, rf('\\frac{x-z}{x-w}'))

    def test_deterministic_output(self):
        self.assertEqual(canon('(x+1)(x-2)'), canon('(x-2)(x+1)'))

    def test_idempotent_output(self):
        once = canon('(x+1)^2')
        self.assertEqual(once, canon(once))


class TestSubstitute(unittest.TestCase):
    def test_numeric(self):
        r = Core.substitute('x^2 + 2x + 1', 'x', '3')
        self.assertTrue(r['ok'])
        self.assertEqual(Core.evaluate(r['result'])['result'], '16')

    def test_symbolic(self):
        r = Core.substitute('x^2 + y', 'x', 'a+b')
        self.assertTrue(r['ok'])
        self.assertEqual(Core.equal_exprs(r['result'],
                                       'a^2 + 2ab + b^2 + y')['verdict'],
                         'yes')

    def test_check_agrees(self):
        r = Core.substitute('x^3 - x', 'x', 'y+1')
        self.assertEqual(r['check']['status'], 'agree')

    def test_missing_variable(self):
        r = Core.substitute('x + 1', 'z', '2')
        self.assertFalse(r['ok'])


class TestApplyBothSides(unittest.TestCase):
    def test_subtract(self):
        r = Core.apply_both_sides('2x + 3 = 7', '-', '3')
        self.assertTrue(r['ok'])
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(Core.expand(r['result'])['result'], '2x = 4')

    def test_divide_constant_no_assumption(self):
        r = Core.apply_both_sides('2x = 4', '/', '2')
        self.assertTrue(r['ok'])
        self.assertEqual(r['assumptions'], [])

    def test_divide_symbol_records_assumption(self):
        r = Core.apply_both_sides('x y = 1', '/', 'y')
        self.assertTrue(r['ok'])
        self.assertEqual(len(r['assumptions']), 1)
        self.assertIn('\\ne 0', r['assumptions'][0]['text'])

    def test_divide_by_zero_rejected(self):
        r = Core.apply_both_sides('x = 1', '/', '0')
        self.assertFalse(r['ok'])

    def test_multiply_by_zero_rejected(self):
        r = Core.apply_both_sides('x = 1', '*', '0')
        self.assertFalse(r['ok'])

    def test_not_an_equation(self):
        r = Core.apply_both_sides('x + 1', '+', '2')
        self.assertFalse(r['ok'])

    def test_multiply_symbolic_records_assumption(self):
        r = Core.apply_both_sides('\\frac{2x+1}{x+1} = 3', '*', 'x+1')
        self.assertTrue(r['ok'])
        self.assertEqual(len(r['assumptions']), 1)
        self.assertIn('\\ne 0', r['assumptions'][0]['text'])

    def test_multiply_parenthesizes_sums(self):
        r = Core.apply_both_sides('x + 1 = y', '*', 'a + b')
        self.assertTrue(r['ok'])
        self.assertEqual(r['check']['status'], 'agree')

    def test_power(self):
        r = Core.apply_both_sides('x = 2', '^', '2')
        self.assertTrue(r['ok'])
        self.assertEqual(r['check']['status'], 'agree')


class TestExpandCollectEvaluate(unittest.TestCase):
    def test_expand(self):
        r = Core.expand('(x+1)(x-2)')
        self.assertEqual(r['result'], 'x^{2}-x-2')
        self.assertEqual(r['check']['status'], 'agree')

    def test_expand_equation(self):
        r = Core.expand('{2}x+{3} - {3} = {7} - {3}')
        self.assertEqual(r['result'], '2x = 4')

    def test_expand_trig_via_atoms(self):
        # outside the fragment, expand canonicalizes over opaque atoms
        r = Core.expand('\\sin x + 1')
        self.assertTrue(r['ok'])
        self.assertEqual(r['check']['status'], 'agree')

    def test_expand_merges_atom_terms(self):
        r = Core.expand('2 \\sin x + 3 \\sin x')
        self.assertTrue(r['ok'])
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(Core.equal_exprs(r['result'],
                                       '5 \\sin x')['verdict'], 'yes')

    def test_expand_cancels_atom_terms(self):
        r = Core.expand('x \\sin x - x \\sin x + 1')
        self.assertEqual(r['result'], '1')

    def test_expand_canonicalizes_rational_function_arguments(self):
        r = Core.expand(
            r'\frac {{1}} {{2}}x^{{2}}\ln({{4}}+'
            r'(x^{{{{2}}}})^{{{{2}}}})-x^{{2}}+C+{2}'
            r'\arctan\left(\frac{(x^{{{{2}}}})}{{{2}}}\right)')
        self.assertTrue(r['ok'])
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(
            r['result'],
            r'\frac {1} {2}x^{2}\ln (x^{4}+4)-x^{2}+C'
            r'+2 \arctan\left ( \frac {1} {2}x^{2}\right )')

    def test_expand_merges_atoms_after_argument_canonicalization(self):
        r = Core.expand(r'\ln(4+(x^2)^2)-\ln(x^4+4)')
        self.assertTrue(r['ok'])
        self.assertEqual(r['result'], '0')
        self.assertEqual(r['check']['status'], 'agree')

    def test_expand_argument_canonicalization_is_idempotent(self):
        r1 = Core.expand(r'\sin(2(x+x))+\sin(4x)')
        self.assertTrue(r1['ok'])
        self.assertEqual(r1['check']['status'], 'agree')
        r2 = Core.expand(r1['result'])
        self.assertTrue(r2['ok'])
        self.assertEqual(r2['result'], r1['result'])

    def test_expand_finishes_by_parts_assembly(self):
        r = Core.expand('x \\left(-\\cos\\left(x\\right)\\right) '
                     '- \\left(-\\sin\\left(x\\right) + C\\right)')
        self.assertTrue(r['ok'])
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(
            Core.equal_exprs(r['result'],
                          '-x \\cos(x) + \\sin(x) - C')['verdict'], 'yes')

    def test_equal_via_shared_atoms(self):
        r = Core.equal_exprs('2\\sin x + \\cos x - \\cos x', '\\sin(x) + \\sin x')
        self.assertEqual(r['verdict'], 'yes')
        self.assertIn('atoms', r['method'])

    def test_atom_inequality_not_trusted(self):
        # distinct atoms may still be related; canonical inequality must
        # fall through to the oracle, which decides correctly
        r = Core.equal_exprs('\\sin^2 x + \\cos^2 x', '1')
        self.assertEqual(r['verdict'], 'yes')
        self.assertIn('numeric', r['method'])

    def test_collect(self):
        r = Core.collect('x^2 + 2x + a x + a', 'x')
        self.assertTrue(r['ok'])
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(Core.equal_exprs(r['result'],
                                       'x^2 + (a+2)x + a')['verdict'], 'yes')

    def test_collect_equation(self):
        r = Core.collect('a x + 2x + 1 = b x', 'x')
        self.assertTrue(r['ok'])
        self.assertEqual(r['check']['status'], 'agree')

    def test_collect_over_atoms(self):
        r = Core.collect('x \\sin x + x \\cos x', 'x')
        self.assertTrue(r['ok'])
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(r['result'], '( \\sin x+ \\cos x)x')

    def test_collect_over_atoms_mixed_powers(self):
        r = Core.collect('2 e^x + x e^x + x^2', 'x')
        self.assertTrue(r['ok'])
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(Core.equal_exprs(
            r['result'], 'x^2 + e^x x + 2 e^x')['verdict'], 'yes')

    def test_collect_equation_over_atoms(self):
        r = Core.collect('x \\sin x + x \\cos x = 0', 'x')
        self.assertTrue(r['ok'])
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(r['result'], '( \\sin x+ \\cos x)x = 0')

    def test_collect_var_only_inside_atoms_rejected(self):
        r = Core.collect('\\sin x + \\cos x', 'x')
        self.assertFalse(r['ok'])
        self.assertIn('does not occur', r['error'])

    def test_atom_output_drops_redundant_parens(self):
        self.assertEqual(Core.expand('2 \\sin x + 3 \\sin x')['result'],
                         '5 \\sin x')
        self.assertEqual(Core.expand('x \\sin x + 2 x \\sin x')['result'],
                         '3x \\sin x')
        self.assertEqual(Core.expand('\\sin x + \\cos x - \\sin x')['result'],
                         '\\cos x')

    def test_atom_output_keeps_needed_parens(self):
        # a powered atom prints in standard \sin^{2}x form where the
        # position is unambiguous...
        r = Core.expand('(\\sin x)^2 + 1')
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(r['result'], '\\sin^{2}x+1')
        # A following function head supplies an unambiguous argument boundary,
        # so the non-trailing wrapper is redundant too.
        r = Core.expand('\\sin^2 x \\cos^2 x')
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(r['result'], '\\sin^{2}x \\cos^{2}x')

    def test_atom_integral_keeps_thin_space(self):
        r = Core.expand('x \\sin x + \\int x \\, dx - x \\sin x')
        self.assertEqual(r['result'], '\\int x \\, dx')

    def test_evaluate_fractions(self):
        r = Core.evaluate('\\frac{2}{3} + \\frac{1}{6}')
        self.assertTrue(r['exact'])
        self.assertEqual(r['result'], '\\frac {5} {6}')

    def test_evaluate_free_vars_rejected(self):
        r = Core.evaluate('x + 1')
        self.assertFalse(r['ok'])

    def test_evaluate_equation_holds(self):
        r = Core.evaluate('2(2) + 3 = 7')
        self.assertTrue(r['ok'])
        self.assertTrue(r['holds'])

    def test_evaluate_equation_fails(self):
        r = Core.evaluate('2(3) + 3 = 7')
        self.assertTrue(r['ok'])
        self.assertFalse(r['holds'])


class TestAtomPowers(unittest.TestCase):
    # gen 8: \sin^{n} x enters the atom layer as atom(\sin x)^n, so both
    # power spellings meet in one canonical form

    def test_power_forms_share_one_atom(self):
        r = Core.expand('(\\sin x)^2 - \\sin^2 x')
        self.assertTrue(r['ok'])
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(r['result'], '0')

    def test_equal_power_forms_is_canonical(self):
        r = Core.equal_exprs('\\sin^2 x', '(\\sin x)^2')
        self.assertEqual(r['verdict'], 'yes')
        self.assertIn('atoms', r['method'])

    def test_powers_merge_in_products(self):
        r = Core.expand('\\sin^2 x \\sin x')
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(r['result'], '\\sin^{3}x')

    def test_like_powered_terms_merge(self):
        r = Core.expand('2\\sin^2 x + 3\\sin^2 x')
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(r['result'], '5 \\sin^{2}x')

    def test_collect_powered_atoms(self):
        r = Core.collect('x \\sin^2 x + x \\cos^2 x', 'x')
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(r['result'], '( \\sin^{2}x+ \\cos^{2}x)x')

    def test_grouped_argument_power(self):
        r = Core.expand('\\sin^2(x+1) - (\\sin(x+1))^2')
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(r['result'], '0')

    def test_distinct_powered_atoms_not_collapsed(self):
        # soundness: sin^2 and cos^2 stay distinct atoms; no fake identity
        r = Core.expand('\\sin^2 x + \\cos^2 x')
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(r['result'], '\\sin^{2}x+ \\cos^{2}x')

    def test_inverse_function_power_stays_opaque(self):
        # \sin^{-1} keeps its arcsin reading: one opaque atom, no power
        r = Core.expand('\\sin^{-1} x + \\sin^{-1} x')
        self.assertTrue(r['ok'])
        self.assertEqual(r['check']['status'], 'agree')
        self.assertIn('\\sin^{-', r['result'])

    def test_adjacent_function_atoms_drop_safe_nontrailing_parens(self):
        r = Core.expand('2(\\sin x)\\cos x')
        self.assertEqual(r['result'], '2 \\sin x \\cos x')
        self.assertEqual(r['check']['status'], 'agree')

    def test_adjacent_powered_function_atoms_drop_safe_parens(self):
        r = Core.expand('(\\sin x)^2\\cos^2 x')
        self.assertEqual(r['result'], '\\sin^{2}x \\cos^{2}x')
        self.assertEqual(r['check']['status'], 'agree')


class TestDifferentiate(unittest.TestCase):
    def check(self, expr, var='x'):
        r = Differentiation.differentiate(expr, var)
        self.assertTrue(r['ok'], r.get('error'))
        self.assertEqual(r['check']['status'], 'agree',
                         f'{expr}: {r["result"]} check={r["check"]}')
        return r

    def test_polynomial(self):
        r = self.check('x^3 + 2x')
        self.assertEqual(r['result'], '3x^{2}+2')

    def test_rational(self):
        self.check('\\frac{1}{x}')

    def test_product_rule(self):
        self.check('x \\sin x')

    def test_quotient_rule(self):
        r = self.check('\\frac{\\sin x}{x}')
        self.assertEqual(
            r['result'],
            '\\frac {x \\cos\\left (x \\right )- \\sin\\left (x \\right )} '
            '{x^{2}}',
        )

    def test_chain_exp(self):
        r = self.check('e^{x^2}')
        self.assertEqual(r['result'], '2xe^{\\left (x^{2}\\right )}')

    def test_chain_ln(self):
        self.check('\\ln(x^2 + 1)')

    def test_func_power(self):
        self.check('\\sin^2 x')

    def test_sqrt(self):
        self.check('\\sqrt{x}')

    def test_tan(self):
        self.check('\\tan x')

    def test_sin_2x(self):
        self.check('\\sin 2x')

    def test_partial(self):
        r = Differentiation.differentiate('x^2 y + y', 'y')
        self.assertTrue(r['ok'])
        self.assertEqual(Core.equal_exprs(r['result'], 'x^2 + 1')['verdict'],
                         'yes')


class TestRewrite(unittest.TestCase):
    def test_diff_squares(self):
        r = Core.rewrite('x^2 - y^2', 'diff_squares')
        self.assertTrue(r['ok'])
        self.assertEqual(r['check']['status'], 'agree')

    def test_backward(self):
        r = Core.rewrite('(x + y)(x - y)', 'diff_squares',
                      direction='backward')
        self.assertTrue(r['ok'])
        self.assertEqual(Core.equal_exprs(r['result'], 'x^2 - y^2')['verdict'],
                         'yes')

    def test_no_match(self):
        r = Core.rewrite('x + 1', 'diff_squares')
        self.assertFalse(r['ok'])

    def test_unknown_lemma(self):
        r = Core.rewrite('x^2 - y^2', 'nope')
        self.assertFalse(r['ok'])

    def test_square_of_sum(self):
        r = Core.rewrite('(u + v)^2', 'square_of_sum')
        self.assertTrue(r['ok'])
        self.assertEqual(r['check']['status'], 'agree')


class TestInequalities(unittest.TestCase):
    def test_subtract_keeps_relation(self):
        r = Core.apply_both_sides('2x + 3 < 7', '-', '3')
        self.assertTrue(r['ok'])
        self.assertIn('\\lt', r['result'])
        self.assertEqual(Core.expand(r['result'])['result'], '2x \\lt 4')

    def test_divide_positive_keeps(self):
        r = Core.apply_both_sides('2x \\lt 4', '/', '2')
        self.assertTrue(r['ok'])
        self.assertIn('\\lt', r['result'])

    def test_divide_negative_flips(self):
        r = Core.apply_both_sides('-2x \\le 4', '/', '-2')
        self.assertTrue(r['ok'])
        self.assertIn('\\ge', r['result'])
        self.assertEqual(Core.expand(r['result'])['result'], 'x \\ge -2')

    def test_multiply_unknown_sign_rejected(self):
        r = Core.apply_both_sides('x y > 4', '*', 'y')
        self.assertFalse(r['ok'])

    def test_power_rejected(self):
        r = Core.apply_both_sides('x \\ge 3', '^', '2')
        self.assertFalse(r['ok'])

    def test_ne_divide_records_assumption(self):
        r = Core.apply_both_sides('x y \\ne 4', '/', 'y')
        self.assertTrue(r['ok'])
        self.assertEqual(len(r['assumptions']), 1)

    def test_evaluate_inequality_holds(self):
        self.assertTrue(Core.evaluate('3 \\le 4')['holds'])
        self.assertFalse(Core.evaluate('5 \\lt 4')['holds'])
        self.assertTrue(Core.evaluate('2(3) \\gt 5')['holds'])


class TestSubtermRewrite(unittest.TestCase):
    def test_inside_sum(self):
        r = Core.rewrite('3 + (x^2 - y^2)', 'diff_squares')
        self.assertTrue(r['ok'])
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(r['at'], 'x^{2}-y^{2}')

    def test_inside_fraction(self):
        r = Core.rewrite('\\frac{x^2 - y^2}{x + y}', 'diff_squares')
        self.assertTrue(r['ok'])
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(Core.equal_exprs(r['result'],
                                       'x - y')['verdict'], 'yes')

    def test_backward_inside(self):
        r = Core.rewrite('1 + (a+b)(a-b)', 'diff_squares', 'backward')
        self.assertTrue(r['ok'])
        self.assertEqual(Core.equal_exprs(r['result'],
                                       '1 + a^2 - b^2')['verdict'], 'yes')


class TestNumericPowerRewrite(unittest.TestCase):
    # gen 8: a^n pattern terms may bind perfect n-th power monomials

    def test_square_literal(self):
        r = Core.rewrite('x^2 - 4', 'diff_squares')
        self.assertTrue(r['ok'])
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(r['result'], '(x+2)(x-2)')
        self.assertEqual(r['numeric'], {'b': '2'})

    def test_literal_on_the_left(self):
        r = Core.rewrite('4 - x^2', 'diff_squares')
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(r['result'], '(2+x)(2-x)')
        self.assertEqual(r['numeric'], {'a': '2'})

    def test_monomial_roots(self):
        r = Core.rewrite('4x^2 - 9', 'diff_squares')
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(r['result'], '(2x+3)(2x-3)')

    def test_even_symbolic_powers(self):
        r = Core.rewrite('x^4 - y^4', 'diff_squares')
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(r['result'], '(x^{2}+y^{2})(x^{2}-y^{2})')

    def test_cube_literals(self):
        r = Core.rewrite('x^3 - 8', 'diff_cubes')
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(r['numeric'], {'b': '2'})
        r = Core.rewrite('x^3 + 27', 'sum_cubes')
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(r['numeric'], {'b': '3'})

    def test_fraction_square(self):
        r = Core.rewrite('x^2 - \\frac{1}{4}', 'diff_squares')
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(r['numeric']['b'], '\\frac {1} {2}')

    def test_imperfect_powers_still_refused(self):
        for expr in ('x^2 - 3', 'x^2 - y', 'x^2 - 4x'):
            r = Core.rewrite(expr, 'diff_squares')
            self.assertFalse(r['ok'], expr)

    def test_structural_match_keeps_priority(self):
        r = Core.rewrite('x^2 - y^2', 'diff_squares')
        self.assertEqual(r['result'], '(x+y)(x-y)')
        self.assertNotIn('numeric', r)

    def test_numeric_subterm(self):
        r = Core.rewrite('\\frac{x^2 - 4}{x + 2}', 'diff_squares')
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(r['at'], 'x^{2}-4')
        self.assertEqual(Core.equal_exprs(r['result'], 'x - 2')['verdict'],
                         'yes')

    def test_equation_rewrite_is_checked(self):
        # relation-aware _checked: per-side oracle instead of skipped
        r = Core.rewrite('x^2 - 4 = 0', 'diff_squares')
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(r['result'], '(x+2)(x-2)=0')


class TestRewriteAtSelector(unittest.TestCase):
    # gen 41: `at` (target latex or 1-based index) picks among several
    # matching subterms; default stays first-match

    TWO = '\\frac{x^2-y^2}{x+y} + \\frac{a^2-b^2}{a+b}'

    def test_default_keeps_first_match(self):
        r = Core.rewrite(self.TWO, 'diff_squares')
        self.assertEqual(r['at'], 'x^{2}-y^{2}')
        self.assertEqual(r['matches'], 2)  # wrapper twins not double-counted

    def test_select_second_by_latex(self):
        r = Core.rewrite(self.TWO, 'diff_squares', at='a^2-b^2')
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(r['at'], 'a^{2}-b^{2}')
        self.assertIn('(a+b)(a-b)', r['result'])
        self.assertIn('x^{2}-y^{2}', r['result'])

    def test_at_compares_modulo_spelling(self):
        r = Core.rewrite(self.TWO, 'diff_squares', at='a^{2} - b^{2}')
        self.assertTrue(r['ok'])
        self.assertEqual(r['at'], 'a^{2}-b^{2}')

    def test_select_occurrence_by_index(self):
        r = Core.rewrite('(x^2-4)+3(x^2-4)', 'diff_squares', at='2')
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(r['result'], '(x^{2}-4)+3(x+2)(x-2)')

    def test_variant_position_behind_structural_match(self):
        # the base pattern matches elsewhere; `at` still reaches the
        # numeric-variant position
        r = Core.rewrite('(a^2-b^2)+(x^2-4)', 'diff_squares', at='x^2-4')
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(r['numeric'], {'b': '2'})
        self.assertIn('(x+2)(x-2)', r['result'])

    def test_backward_with_index(self):
        r = Core.rewrite('(x+y)(x-y) + (a+b)(a-b)', 'diff_squares',
                         'backward', at='2')
        self.assertEqual(r['at'], '(a+b)(a-b)')
        self.assertIn('(x+y)(x-y)', r['result'])

    def test_relation_side_selection(self):
        r = Core.rewrite('x^2-y^2 = a^2-b^2', 'diff_squares', at='2')
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(r['at'], 'a^{2}-b^{2}')

    def test_index_out_of_range_lists_positions(self):
        r = Core.rewrite(self.TWO, 'diff_squares', at='9')
        self.assertFalse(r['ok'])
        self.assertIn('out of range', r['error'])
        self.assertIn('1. x^{2}-y^{2}', r['error'])
        self.assertIn('2. a^{2}-b^{2}', r['error'])

    def test_latex_without_match_lists_positions(self):
        r = Core.rewrite(self.TWO, 'diff_squares', at='u^2-v^2')
        self.assertFalse(r['ok'])
        self.assertIn('does not match at', r['error'])
        self.assertIn('2. a^{2}-b^{2}', r['error'])

    def test_unparseable_at_refused(self):
        r = Core.rewrite(self.TWO, 'diff_squares', at='\\frac{')
        self.assertFalse(r['ok'])
        self.assertIn('at:', r['error'])

    def test_at_recorded_in_args_for_replay(self):
        r = Core.rewrite(self.TWO, 'diff_squares', at='2')
        self.assertEqual(r['args'].get('at'), '2')
        again = Core.rewrite(**{'expr': r['args']['expr'],
                                'lemma_name': r['args']['lemma'],
                                'direction': r['args']['direction'],
                                'at': r['args']['at']})
        self.assertEqual(again['result'], r['result'])

    def test_at_selecting_the_root(self):
        r = Core.rewrite('x^2 - y^2', 'diff_squares', at='x^2 - y^2')
        self.assertEqual(r['result'], '(x+y)(x-y)')
        r = Core.rewrite('x^2 - y^2', 'diff_squares', at='1')
        self.assertEqual(r['result'], '(x+y)(x-y)')


class TestCollectRational(unittest.TestCase):
    def test_collect_num_and_den(self):
        r = Core.collect('\\frac{ax + bx + 1}{x + cx}', 'x')
        self.assertTrue(r['ok'])
        self.assertEqual(r['check']['status'], 'agree')


class TestPrettyOutput(unittest.TestCase):
    def test_no_value_braces(self):
        self.assertEqual(Core.expand('(x+1)(x-2)')['result'], 'x^{2}-x-2')

    def test_index_dims_keep_braces(self):
        r = Core.expand('x^{12} x')
        self.assertIn('x^{13}', r['result'])

    def test_repeated_index_groups_collapse_to_one_layer(self):
        for source, expected in (
                ('x^{{{3}}}', 'x^{3}'),
                ('C_{{{1}}}', 'C_{1}'),
                ('C_{{{1}}}^{2}', 'C_{1}^{2}'),
                ('2e^{{2}x}', '2e^{2x}')):
            with self.subTest(source=source):
                sym, notation = P.parse_latex(source)
                self.assertEqual(P.write_latex(sym, notation), expected)

    def test_pretty_reparses_equal(self):
        # every pretty result must parse back to an equal expression
        for expr in ['(x+1)^3', '\\frac{2}{3} + \\frac{1}{6}',
                     '2 \\cdot x \\cdot (x+1)']:
            r = Core.expand(expr) if 'frac' not in expr else Core.evaluate(expr)
            self.assertEqual(
                Core.equal_exprs(r['result'], expr)['verdict'], 'yes', expr)


class TestFactor(unittest.TestCase):
    def test_quadratic_two_roots(self):
        r = Core.factor_quadratic('x^2 - 5x + 6', 'x')
        self.assertTrue(r['ok'])
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(sorted(r['roots']), ['2', '3'])

    def test_quadratic_perfect_square(self):
        r = Core.factor_quadratic('x^2 - 6x + 9', 'x')
        self.assertTrue(r['ok'])
        self.assertIn('^{2}', r['result'])

    def test_quadratic_leading_coeff(self):
        r = Core.factor_quadratic('2x^2 + 5x - 3', 'x')
        self.assertTrue(r['ok'])
        self.assertEqual(r['check']['status'], 'agree')

    def test_quadratic_irrational_rejected(self):
        r = Core.factor_quadratic('x^2 - 2', 'x')
        self.assertFalse(r['ok'])

    def test_quadratic_complex_rejected(self):
        r = Core.factor_quadratic('x^2 + x + 1', 'x')
        self.assertFalse(r['ok'])

    def test_gcd(self):
        r = Core.factor_gcd('6x^2 + 9x')
        self.assertTrue(r['ok'])
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(Core.equal_exprs(r['result'],
                                       '3x(2x+3)')['verdict'], 'yes')

    def test_gcd_negative_leading(self):
        r = Core.factor_gcd('-2x^2 - 4x')
        self.assertTrue(r['ok'])
        self.assertEqual(r['check']['status'], 'agree')

    def test_gcd_nothing_to_factor(self):
        r = Core.factor_gcd('x + 1')
        self.assertFalse(r['ok'])

    def test_quadratic_factors_equation_side(self):
        r = Core.factor_quadratic('x^2 + 6x + 9 = 4', 'x')
        self.assertTrue(r['ok'], r.get('error'))
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(r['factored_sides'], ['lhs'])
        self.assertEqual(r['roots_by_side'], {'lhs': ['-3', '-3']})
        self.assertEqual(Core.equal_exprs(r['result'], '(x+3)^2=4')['verdict'],
                         'yes')

    def test_quadratic_factors_both_relation_sides(self):
        r = Core.factor_quadratic('x^2 - 1 = x^2 - 4x + 4', 'x')
        self.assertTrue(r['ok'], r.get('error'))
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(r['factored_sides'], ['lhs', 'rhs'])

    def test_gcd_factors_inequality_side(self):
        r = Core.factor_gcd('6x^2 + 9x \\lt 3')
        self.assertTrue(r['ok'], r.get('error'))
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(r['factored_sides'], ['lhs'])
        self.assertEqual(Core.equal_exprs(r['result'],
                                      '3x(2x+3) \\lt 3')['verdict'], 'yes')

    def test_relation_refuses_when_neither_side_factors(self):
        r = Core.factor_quadratic('x^2 + 1 = 4', 'x')
        self.assertFalse(r['ok'])
        self.assertIn('neither side', r['error'])


class TestQuadraticRoots(unittest.TestCase):
    def test_expression_returns_complete_ordered_solution(self):
        r = Equations.quadratic_roots('3x^2-3', 'x')
        self.assertTrue(r['ok'], r.get('error'))
        self.assertEqual(r['result'], r'x=-1 \lor x=1')
        self.assertEqual(r['solutions'], ['-1', '1'])
        self.assertEqual(r['check']['status'], 'agree')
        self.assertIn('Vieta', r['check']['method'])

    def test_equality_moves_sides_before_finding_roots(self):
        r = Equations.quadratic_roots('2x^2+5x=3', 'x')
        self.assertTrue(r['ok'], r.get('error'))
        self.assertEqual(r['solutions'], ['-3', r'\frac{1}{2}'])

    def test_repeated_root_is_one_solution(self):
        r = Equations.quadratic_roots('x^2-6x+9', 'x')
        self.assertTrue(r['ok'], r.get('error'))
        self.assertEqual(r['result'], 'x=3')
        self.assertEqual(r['solutions'], ['3'])

    def test_unsupported_root_classes_refuse(self):
        irrational = Equations.quadratic_roots('x^2-2', 'x')
        self.assertFalse(irrational['ok'])
        self.assertIn('not rational', irrational['error'])
        complex_roots = Equations.quadratic_roots('x^2+x+1', 'x')
        self.assertFalse(complex_roots['ok'])
        self.assertIn('no real roots', complex_roots['error'])
        linear = Equations.quadratic_roots('2x+1', 'x')
        self.assertFalse(linear['ok'])
        self.assertIn('not quadratic', linear['error'])

    def test_coefficients_must_be_constant_and_relation_equality(self):
        symbolic = Equations.quadratic_roots('ax^2+x+1', 'x')
        self.assertFalse(symbolic['ok'])
        self.assertIn('coefficients must be constants', symbolic['error'])
        inequality = Equations.quadratic_roots('x^2-1 \\lt 0', 'x')
        self.assertFalse(inequality['ok'])
        self.assertIn('requires an equality', inequality['error'])

    def test_independent_check_catches_bad_candidate(self):
        check = Equations._quadratic_roots_check(
            'x^2-1', 'x', ['-1', '2'])
        self.assertEqual(check['status'], 'disagree')

    def test_solution_metadata_is_replay_checked(self):
        ledger = Ledger()
        ledger.record(Equations.quadratic_roots('x^2-1', 'x'))
        self.assertEqual(ledger.replay()['status'], 'verified')
        ledger.steps[0]['solutions'][1] = '2'
        replay = ledger.replay()
        self.assertEqual(replay['status'], 'failed')
        self.assertIn('solution metadata', replay['reason'])


class TestParsingEdges(unittest.TestCase):
    def test_cdot_chain(self):
        r = Core.expand('2 \\cdot x \\cdot (x+1)')
        self.assertTrue(r['ok'])
        self.assertEqual(r['result'], '2x^{2}+2x')

    def test_star_chain(self):
        # '*' and \cdot are grammar-level product separators (P_LIST),
        # so chains parse without preprocessing
        self.assertEqual(Core.evaluate('2*3*4')['result'], '24')
        self.assertEqual(Core.equal_exprs('a \\cdot b * c', 'c b a')['verdict'],
                         'yes')

    def test_substitute_into_equation(self):
        r = Core.substitute('x^2 - 6x + 5 = 0', 'x', '5')
        self.assertTrue(r['ok'])
        self.assertEqual(r['check']['status'], 'agree')
        self.assertTrue(Core.evaluate(r['result'])['holds'])

    def test_substitute_wrong_candidate(self):
        r = Core.substitute('x^2 - 6x + 5 = 0', 'x', '4')
        self.assertTrue(r['ok'])
        self.assertFalse(Core.evaluate(r['result'])['holds'])

    def test_compressed_frac_digit_arguments(self):
        # TeX argument scanning: \frac12 means \frac{1}{2}. Agents type
        # this constantly; the lexer fuses '12' into one number and the
        # parse fails, so the TeX reading is retried on syntax error only
        # (a live prove! derailed on this spelling).
        self.assertEqual(Core.equal_exprs('\\frac12',
                                          '\\frac{1}{2}')['verdict'], 'yes')
        self.assertEqual(Core.equal_exprs('\\frac12\\cdot\\frac34',
                                          '\\frac{3}{8}')['verdict'], 'yes')
        # trailing digits stay ordinary factors: \frac123 = (1/2)*3
        self.assertEqual(Core.equal_exprs('\\frac123',
                                          '\\frac{3}{2}')['verdict'], 'yes')
        # spaced and half-braced spellings keep parsing
        self.assertEqual(Core.equal_exprs('\\frac 1 2',
                                          '\\frac12')['verdict'], 'yes')
        self.assertEqual(Core.equal_exprs('\\frac1{2n}',
                                          '\\frac{1}{2n}')['verdict'], 'yes')
        # token-per-argument dialect spellings parse first try and keep
        # their meaning: the retry never reinterprets a valid parse
        self.assertEqual(Core.equal_exprs('\\frac 13 15',
                                          '\\frac{13}{15}')['verdict'], 'yes')
        self.assertEqual(Core.equal_exprs('\\frac13 15',
                                          '\\frac{13}{15}')['verdict'], 'yes')


class TestIntegration(unittest.TestCase):
    def ok(self, rec):
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['check']['status'], 'agree',
                         f"{rec.get('result')} check={rec['check']}")
        return rec

    def test_power_rule_monomial(self):
        r = self.ok(Integration.integrate_power_rule('x^2', 'x'))
        self.assertEqual(r['result'], '\\frac {1} {3}x^{3} + C')

    def test_power_rule_polynomial(self):
        r = self.ok(Integration.integrate_power_rule('3x^2 + 2x + 1', 'x'))
        self.assertEqual(r['result'], 'x^{3}+x^{2}+x + C')

    def test_power_rule_strips_integral_wrapper(self):
        r = self.ok(Integration.integrate_power_rule('\\int x^2 \\, dx', 'x'))
        self.assertEqual(r['result'], '\\frac {1} {3}x^{3} + C')

    def test_power_rule_negative_power(self):
        r = self.ok(Integration.integrate_power_rule('\\frac{1}{x^2}', 'x'))
        self.assertEqual(r['result'], '-\\frac{1}{x} + C')

    def test_power_rule_symbolic_coefficient(self):
        self.ok(Integration.integrate_power_rule('a x', 'x'))

    def test_power_rule_refuses_log_case(self):
        r = Integration.integrate_power_rule('\\frac{1}{x}', 'x')
        self.assertFalse(r['ok'])
        self.assertIn('integrate_table', r['error'])

    def test_power_rule_refuses_trig(self):
        r = Integration.integrate_power_rule('\\sin x', 'x')
        self.assertFalse(r['ok'])

    def test_wrong_integration_variable(self):
        r = Integration.integrate_power_rule('\\int x^2 \\, dx', 'y')
        self.assertFalse(r['ok'])

    def test_definite_integral_refused(self):
        r = Integration.integrate_power_rule('\\int_0^1 x^2 \\, dx', 'x')
        self.assertFalse(r['ok'])

    def test_table_log_records_assumption(self):
        r = self.ok(Integration.integrate_table('\\frac{1}{x}', 'x'))
        self.assertIn('\\ln', r['result'])
        self.assertEqual(len(r['assumptions']), 1)

    def test_table_sin(self):
        r = self.ok(Integration.integrate_table('\\sin x', 'x'))
        self.assertEqual(r['result'], '-\\cos\\left(x\\right) + C')

    def test_table_constant_multiple(self):
        self.ok(Integration.integrate_table('2 \\cos x', 'x'))

    def test_table_exp(self):
        r = self.ok(Integration.integrate_table('e^x', 'x'))
        self.assertEqual(r['result'], 'e^{x} + C')

    def test_table_mixed_sum(self):
        self.ok(Integration.integrate_table('x + \\sin x', 'x'))

    def test_table_refuses_product(self):
        r = Integration.integrate_table('x \\sin x', 'x')
        self.assertFalse(r['ok'])
        self.assertIn('integrate_by_parts', r['error'])

    def test_by_parts_x_sin(self):
        r = self.ok(Integration.integrate_by_parts('x \\sin x', 'x', 'x', '\\sin x'))
        self.assertEqual(r['v'], '-\\cos\\left(x\\right)')
        self.assertEqual(r['du'], '1')
        self.assertIn('remaining_integral', r)
        # the remaining integral must feed back into integrate_table
        r2 = self.ok(Integration.integrate_table(r['remaining_integral'], 'x'))
        self.assertEqual(r2['result'], '-\\sin\\left(x\\right) + C')

    def test_by_parts_rejects_wrong_split(self):
        r = Integration.integrate_by_parts('x \\sin x', 'x', 'x', '\\cos x')
        self.assertFalse(r['ok'])

    def test_by_parts_requires_u_with_var(self):
        r = Integration.integrate_by_parts('a \\sin x', 'x', 'a', '\\sin x')
        self.assertFalse(r['ok'])

    def test_fresh_constant_avoids_collision(self):
        r = self.ok(Integration.integrate_power_rule('C x', 'x'))
        self.assertEqual(r['constant'], 'K')
        self.assertIn('+ K', r['result'])

    def test_final_answer_verifiable_by_existing_primitives(self):
        d = Differentiation.differentiate('-x \\cos(x) + \\sin(x)', 'x')
        self.assertTrue(d['ok'])
        self.assertEqual(
            Core.equal_exprs(d['result'], 'x \\sin x')['verdict'], 'yes')

    def test_substitute_tactic_chain(self):
        # ∫ 2x cos(x²) dx via u = x²
        r1 = self.ok(Integration.integrate_substitute('2x \\cos(x^2)', 'x',
                                            'x^2', 'u', '\\cos(u)'))
        self.assertEqual(r1['result'], '\\int \\cos(u) \\, d u')
        self.assertEqual(r1['back_substitute'],
                         {'var': 'u', 'value': 'x^2'})
        r2 = self.ok(Integration.integrate_table(r1['result'], 'u'))
        r3 = Core.substitute(r2['result'], 'u', 'x^2')
        self.assertTrue(r3['ok'])
        d = Differentiation.differentiate('\\sin(x^2)', 'x')
        self.assertEqual(
            Core.equal_exprs(d['result'], '2x \\cos(x^2)')['verdict'], 'yes')

    def test_substitute_tactic_rejects_wrong_rewrite(self):
        r = Integration.integrate_substitute('2x \\cos(x^2)', 'x', 'x^2', 'u',
                                   '\\sin(u)')
        self.assertFalse(r['ok'])

    def test_substitute_tactic_rejects_var_leak(self):
        r = Integration.integrate_substitute('2x \\cos(x^2)', 'x', 'x^2', 'u',
                                   'x \\cos(u)')
        self.assertFalse(r['ok'])

    def test_substitute_tactic_rejects_colliding_uvar(self):
        r = Integration.integrate_substitute('2u \\cos(u^2)', 'u', 'u^2', 'u',
                                   '\\cos(u)')
        self.assertFalse(r['ok'])

    def test_substitute_tactic_linear_inner(self):
        r1 = self.ok(Integration.integrate_substitute('\\int e^{3x} \\, dx', 'x',
                                            '3x', 'u', '\\frac{e^u}{3}'))
        r2 = self.ok(Integration.integrate_table(r1['result'], 'u'))
        self.assertEqual(
            Core.equal_exprs(r2['result'].replace(' + C', ''),
                          '\\frac{e^{u}}{3}')['verdict'], 'yes')

    def test_table_constant_denominator(self):
        r = self.ok(Integration.integrate_table('\\frac{\\sin u}{3}', 'u'))
        self.assertIn('\\cos', r['result'])

    def test_ledger_replay_with_integration(self):
        path = os.path.join(tempfile.mkdtemp(), 'session.json')
        ledger = Ledger(path)
        ledger.record(Integration.integrate_by_parts('x e^x', 'x', 'x', 'e^x'))
        ledger.record(Integration.integrate_table('e^x', 'x'))
        ledger.save()
        self.assertEqual(Ledger(path).replay()['status'], 'verified')

    def test_markdown_render(self):
        path = os.path.join(tempfile.mkdtemp(), 'session.json')
        ledger = Ledger(path)
        ledger.record(Integration.integrate_table('\\frac{1}{x}', 'x'))
        md = ledger.render_markdown()
        self.assertIn('# Verified derivation', md)
        self.assertIn('assumptions', md)
        self.assertIn('\\Longrightarrow', md)


class TestEqual(unittest.TestCase):
    def test_canonical_yes(self):
        self.assertEqual(Core.equal_exprs('(x+1)^2',
                                       'x^2 + 2x + 1')['verdict'], 'yes')

    def test_multivariate_cancellation_via_cross_multiply(self):
        # (x+y)(x-y)/(x+y) does not cancel (multivariate GCD is
        # monomial-level) but must still compare equal to x-y
        self.assertEqual(
            Core.equal_exprs('\\frac{(x+y)(x-y)}{x+y}', 'x - y')['verdict'],
            'yes')

    def test_canonical_no(self):
        self.assertEqual(Core.equal_exprs('(x+1)^2', 'x^2 + 1')['verdict'],
                         'no')

    def test_trig_identity_numeric(self):
        r = Core.equal_exprs('\\sin^2 x', '1 - \\cos^2 x')
        self.assertEqual(r['verdict'], 'yes')
        self.assertIn('numeric', r['method'])

    def test_trig_non_identity(self):
        self.assertEqual(Core.equal_exprs('\\sin(2x)', '2 \\sin x')['verdict'],
                         'no')

    def test_double_angle(self):
        self.assertEqual(Core.equal_exprs('\\sin(2x)',
                                       '2 \\sin x \\cos x')['verdict'], 'yes')

    def test_equations(self):
        self.assertEqual(Core.equal_exprs('x = 2', 'x = 2')['verdict'], 'yes')
        self.assertEqual(Core.equal_exprs('x = 2', 'x = 3')['verdict'], 'no')
        self.assertEqual(Core.equal_exprs('x = 2', 'x + 2')['verdict'], 'no')


class TestDomainAwareOracle(unittest.TestCase):
    # gen 9: a sample point where exactly one side is defined is a
    # definedness witness — the sides differ as real functions

    def test_log_square_is_not_two_log(self):
        r = Core.equal_exprs('\\ln(x^2)', '2 \\ln x')
        self.assertEqual(r['verdict'], 'no')
        self.assertIn('domain mismatch', r['method'])
        self.assertIn('counterexample', r)
        # ...but the agent learns equality may hold on a restricted domain
        self.assertIn('restricted domain', r['note'])

    def test_log_product_is_not_sum_of_logs(self):
        r = Core.equal_exprs('\\ln(xy)', '\\ln x + \\ln y')
        self.assertEqual(r['verdict'], 'no')
        self.assertIn('domain mismatch', r['method'])

    def test_sqrt_times_sqrt_is_not_x(self):
        r = Core.equal_exprs('\\sqrt{x}\\sqrt{x}', 'x')
        self.assertEqual(r['verdict'], 'no')
        self.assertIn('domain mismatch', r['method'])

    def test_common_restricted_domain_is_reported(self):
        # both sides live on x > 0: yes, with the caveat recorded
        r = Core.equal_exprs('\\frac{x}{\\sqrt{x}}', '\\sqrt{x}')
        self.assertEqual(r['verdict'], 'yes')
        self.assertIn('both sides are defined', r['note'])

    def test_everywhere_identities_have_no_caveat(self):
        r = Core.equal_exprs('\\sin^2 x + \\cos^2 x', '1')
        self.assertEqual(r['verdict'], 'yes')
        self.assertNotIn('note', r)

    def test_value_counterexample_still_wins(self):
        # both sides defined at x < 0, values differ: plain 'no'
        r = Core.equal_exprs('\\sqrt{x^2}', 'x')
        self.assertEqual(r['verdict'], 'no')
        self.assertEqual(r['method'], 'numeric-oracle')

    def test_spot_check_status_fields(self):
        c = P.numeric_spot_check('\\ln(x^2)', '2 \\ln x')
        self.assertEqual(c['status'], 'domain-differs')
        self.assertEqual(c['defined'], 'lhs')
        self.assertGreater(c['mismatches'], 0)
        self.assertGreater(c['common_samples'], 0)

    def test_merge_prefers_domain_differs(self):
        c = Core._merge_checks({'status': 'agree', 'samples': 8},
                            {'status': 'domain-differs', 'mismatches': 3,
                             'common_samples': 5, 'defined': 'rhs',
                             'point': {'x': -1.0}})
        self.assertEqual(c['status'], 'domain-differs')


class TestSubtermRelax(unittest.TestCase):
    # gen 9: subterm-rewrite surgery results relax like root results

    def test_factored_subterm_prints_clean(self):
        r = Core.rewrite('(x^{2}+4)(x^{2}-4)', 'diff_squares')
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(r['result'], '(x^{2}+4)(x+2)(x-2)')

    def test_rewrite_inside_function_argument(self):
        r = Core.rewrite('\\sin(x^{2}-4)', 'diff_squares')
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(r['result'], '\\sin ((x+2)(x-2))')

    def test_verbatim_powers_keep_single_braces(self):
        s, n = P.parse_latex('x^{2}+x^{12}')
        self.assertEqual(P.write_latex(s, n), 'x^{2}+x^{12}')

    def test_explicit_cdot_is_notation_until_transformed(self):
        # a \cdot product round-trips (series terms like 1/(1*2) must not
        # display as 1/(12) or fold to 1/2)...
        s, n = P.parse_latex('\\frac{1}{1 \\cdot 2}')
        self.assertEqual(P.write_latex(s, n).replace(' ', ''),
                         '\\frac{1}{1\\cdot2}')
        # ...but explicit canonicalization still computes it
        self.assertEqual(Core.expand('1 \\cdot 2')['result'], '2')
        self.assertEqual(Core.evaluate('1 \\cdot 2')['result'], '2')
        eq = Core.equal_exprs('1 \\cdot 2', '2')
        self.assertEqual(eq['verdict'], 'yes')

    def test_root_and_equation_outputs_unchanged(self):
        self.assertEqual(Core.rewrite('x^2 - 4', 'diff_squares')['result'],
                         '(x+2)(x-2)')
        self.assertEqual(Core.rewrite('x^4 - 16 = 0', 'diff_squares')['result'],
                         '(x^{2}+4)(x^{2}-4)=0')


class TestRelationSystemsAndProductSlash(unittest.TestCase):
    def test_comma_system_is_a_list_of_relations(self):
        from replicator import Replicator

        sym, notation = P.parse_latex('x+y=3,x-y=1')
        system = notation.getf(sym, Notation.C_LIST)
        self.assertIsNotNone(system)
        self.assertEqual(len(system.args), 2)
        self.assertTrue(all(notation.getf(item, Notation.COMP) is not None
                            for item in system.args))

        copied_notation = Notation()
        copied = Replicator(notation, copied_notation)(sym)
        written = P.write_latex(copied, copied_notation)
        self.assertTrue(P.same_expression(written, 'x+y=3,x-y=1'))

    def test_scalar_comma_list_is_unchanged(self):
        sym, notation = P.parse_latex('x,y')
        comma = notation.getf(sym, Notation.C_LIST)
        self.assertIsNotNone(comma)
        self.assertTrue(all(notation.getf(item, Notation.COMP) is None
                            for item in comma.args))
        self.assertEqual(P.write_latex(sym, notation), 'x,y')

    def test_product_left_of_slash_parses_and_round_trips(self):
        for latex in ('n(n-1)/2', 'n*n/2', 'n\\cdot(n-1)/2'):
            with self.subTest(latex=latex):
                sym, notation = P.parse_latex(latex)
                slash = notation.getf(sym, Notation.SLASH)
                self.assertIsNotNone(slash)
                self.assertIsNotNone(
                    notation.getf(slash.args[0], Notation.P_LIST))
                written = P.write_latex(sym, notation)
                self.assertTrue(P.same_expression(written, latex))

    def test_natural_parity_exponent_is_usable(self):
        expr = '(-1)^{n(n-1)/2}'
        sym, notation = P.parse_latex(expr)
        self.assertTrue(P.same_expression(P.write_latex(sym, notation), expr))
        self.assertEqual(
            Core.equal_exprs(expr, '(-1)^{\\frac{n(n-1)}{2}}')['verdict'],
            'yes')
        replaced = Core.substitute(expr, 'n', '4')
        self.assertEqual(replaced['check']['status'], 'agree')
        self.assertEqual(Core.evaluate(replaced['result'])['result'], '1.0')

    def test_system_substitution_is_checked_per_relation(self):
        for system in ('x+y=3,x-y=1',
                       '\\cases{x+y=3 \\cr x-y=1}'):
            with self.subTest(system=system):
                rec = Core.substitute(system, 'x', '3-y')
                self.assertTrue(rec['ok'])
                self.assertEqual(rec['check']['status'], 'agree')

    def test_apply_both_sides_maps_over_system(self):
        comma = Core.apply_both_sides('x+y=3,x-y=1', '-', 'y')
        cases = Core.apply_both_sides(
            '\\cases{x+y=3 \\cr x-y=1}', '/', '2')
        for rec in (comma, cases):
            self.assertTrue(rec['ok'])
            self.assertEqual(rec['check']['status'], 'agree')
        self.assertIn(',', comma['result'])
        self.assertIn('\\cases', cases['result'])
        symbolic = Core.apply_both_sides('xy=1,x+y=2', '/', 'y')
        self.assertEqual(symbolic['check']['status'], 'agree')
        self.assertEqual(symbolic['assumptions'],
                         [{'text': 'y \\ne 0', 'nonzero': 'y'}])

    def test_non_system_collections_are_not_relations(self):
        rec = Core.apply_both_sides('x=1,y', '+', '1')
        self.assertFalse(rec['ok'])
        self.assertIn('relation system', rec['error'])
        piecewise = Core.apply_both_sides(
            '\\cases{x & x\\ge0 \\cr -x & x\\lt0}', '+', '1')
        self.assertFalse(piecewise['ok'])
        self.assertIn('relation system', piecewise['error'])


class TestCombinatorialNotation(unittest.TestCase):
    def parse(self, latex):
        notation = Notation()
        sym = MathParser(notation).parse(latex)
        return sym, notation

    def test_factorial_is_a_postfix_node_not_a_command(self):
        sym, notation = self.parse('n!')
        factorial = notation.getf(sym, Notation.FACTORIAL)
        self.assertIsNotNone(factorial)
        self.assertEqual(factorial.args[0].name, 'n')

        # Juxtaposed letters remain mathematical factors: xy! = x(y!).
        sym, notation = self.parse('xy!')
        product = notation.getf(sym, Notation.P_LIST)
        self.assertIsNotNone(product)
        self.assertEqual(product.args[0].name, 'x')
        self.assertIsNotNone(
            notation.getf(product.args[1], Notation.FACTORIAL))

        # Word-like legacy/notebook commands retain their existing node.
        sym, notation = self.parse('mul! (x+1)(x-1)')
        self.assertTrue(notation.get(sym).sym.props.get('command'))

    def test_factorial_and_index_order_is_structural(self):
        before, n1 = self.parse('n!^2')
        before_index = n1.getf(before, Notation.INDEX)
        self.assertIsNotNone(before_index)
        self.assertIsNotNone(
            n1.getf(before_index.args[0], Notation.FACTORIAL))

        after, n2 = self.parse('n^2!')
        after_factorial = n2.getf(after, Notation.FACTORIAL)
        self.assertIsNotNone(after_factorial)
        self.assertIsNotNone(
            n2.getf(after_factorial.args[0], Notation.INDEX))

        double, n3 = self.parse('n!!')
        outer = n3.getf(double, Notation.FACTORIAL)
        self.assertIsNotNone(n3.getf(outer.args[0], Notation.FACTORIAL))

    def test_binom_trailing_power_binds_to_whole_coefficient(self):
        sym, notation = self.parse('\\binom{n+1}{k}^2')
        index = notation.getf(sym, Notation.INDEX)
        self.assertIsNotNone(index)
        binom = notation.getf(index.args[0], Notation.BINOM)
        self.assertIsNotNone(binom)
        self.assertIsNone(notation.getf(binom.args[1], Notation.INDEX))

    def test_writer_and_replicator_round_trip(self):
        from LatexWriter import LaTexWriter
        from replicator import Replicator

        cases = (
            '5!', '(2n)!', 'n!!', 'n!^2', 'n^2!',
            '\\binom{n+1}{k}^2', '\\binom{n}{k}!',
            '\\frac{(2n)!}{2^{2n}(n!)^2}',
        )
        for latex in cases:
            with self.subTest(latex=latex):
                sym, notation = self.parse(latex)
                output = Notation()
                copied = Replicator(notation, output)(sym)
                written = LaTexWriter(output)(copied)
                reparsed, reparsed_notation = self.parse(written)
                rewritten = LaTexWriter(reparsed_notation)(reparsed)
                self.assertTrue(P.same_expression(written, rewritten),
                                written)
        sym, notation = self.parse('\\binom{n+1}{k}')
        self.assertEqual(P.write_latex(sym, notation),
                         '\\binom{n+1}{k}')

    def test_closed_integer_oracle(self):
        factorial, fn = self.parse('5!')
        binom, bn = self.parse('\\binom{6}{2}')
        self.assertEqual(P.numeric_eval(factorial, fn, {}), 120.0)
        self.assertEqual(P.numeric_eval(binom, bn, {}), 15.0)
        self.assertEqual(Core.equal_exprs('5!', '120')['verdict'], 'yes')
        self.assertEqual(
            Core.equal_exprs('\\binom{6}{2}', '15')['verdict'], 'yes')

    def test_oracle_keeps_discrete_domain(self):
        for latex in ('(-1)!', '(\\frac{1}{2})!', '\\binom{2}{3}'):
            with self.subTest(latex=latex):
                sym, notation = self.parse(latex)
                with self.assertRaises(ValueError):
                    P.numeric_eval(sym, notation, {})
        sym, notation = self.parse('171!')
        with self.assertRaises(OverflowError):
            P.numeric_eval(sym, notation, {})


# classic non-commuting pair: AB = diag(1,0), BA = diag(0,1)
MAT_A = '\\pmatrix{0 & 1 \\cr 0 & 0}'
MAT_B = '\\pmatrix{0 & 0 \\cr 1 & 0}'


class TestMatrixParsing(unittest.TestCase):
    # gen 10: \begin{pmatrix}/\begin{matrix} normalize to the grammar's
    # plain-TeX commands. Gen 38 moves that normalization into the shared
    # parser and adds the non-alignment AMS matrix environment family.

    def test_pmatrix_env(self):
        s, n = P.parse_latex('\\begin{pmatrix} 1 & 2 \\\\ 3 & 4 \\end{pmatrix}')
        self.assertEqual(P.write_latex(s, n), '\\pmatrix{1 & 2 \\cr 3 & 4}')

    def test_matrix_env(self):
        s, n = P.parse_latex('\\begin{matrix} x & 2x \\\\ 1 & x \\end{matrix}')
        self.assertEqual(P.write_latex(s, n), '\\matrix{x & 2x \\cr 1 & x}')

    def test_nested_envs(self):
        s, n = P.parse_latex('\\begin{pmatrix} \\begin{matrix} 1 \\end{matrix}'
                             ' & 2 \\\\ 3 & 4 \\end{pmatrix}')
        self.assertEqual(P.write_latex(s, n),
                         '\\pmatrix{\\matrix{1} & 2 \\cr 3 & 4}')

    def test_ams_matrix_env_family(self):
        for name in ('bmatrix', 'Bmatrix', 'vmatrix', 'Vmatrix',
                     'smallmatrix'):
            with self.subTest(name=name):
                source = (f'\\begin{{{name}}}1 & 2 \\\\ 3 & 4'
                          f'\\end{{{name}}}')
                s, n = P.parse_latex(source)
                out = P.write_latex(s, n)
                self.assertEqual(
                    out,
                    f'\\begin{{{name}}} 1 & 2 \\\\ 3 & 4 '
                    f'\\end{{{name}}}',
                )
                s2, n2 = P.parse_latex(out)
                self.assertEqual(P.write_latex(s2, n2), out)

    def test_plain_bmatrix_command_canonicalizes_to_environment(self):
        s, n = P.parse_latex('\\bmatrix{1 & 2 \\cr 3 & 4}')
        self.assertEqual(
            P.write_latex(s, n),
            '\\begin{bmatrix} 1 & 2 \\\\ 3 & 4 \\end{bmatrix}',
        )

    def test_mixed_nested_matrix_environments(self):
        source = ('\\begin{bmatrix}\\begin{vmatrix}1\\end{vmatrix} & 2 '
                  '\\\\ 3 & 4\\end{bmatrix}')
        s, n = P.parse_latex(source)
        out = P.write_latex(s, n)
        self.assertIn('\\begin{bmatrix}', out)
        self.assertIn('\\begin{vmatrix}', out)
        P.parse_latex(out)

    def test_cases_environment_uses_shared_normalizer(self):
        s, n = P.parse_latex(
            '\\begin{cases}x & x>0 \\\\ -x & x<0\\end{cases}')
        self.assertEqual(
            P.write_latex(s, n),
            '\\cases{x & x \\gt 0 \\cr -x & x \\lt 0}',
        )

    def test_plain_tex_form_roundtrips(self):
        s, n = P.parse_latex(MAT_A)
        out = P.write_latex(s, n)
        self.assertEqual(out, MAT_A)
        s2, n2 = P.parse_latex(out)
        self.assertEqual(P.write_latex(s2, n2), out)

    def test_substitute_into_cells(self):
        r = Core.substitute('\\pmatrix{x & 2x \\cr 1 & x}', 'x', '3')
        self.assertTrue(r['ok'])
        self.assertEqual(r['check']['status'], 'agree')
        self.assertIn('\\pmatrix', r['result'])

    def test_substitute_into_bmatrix_cells(self):
        r = Core.substitute(
            '\\begin{bmatrix}x & 2x \\\\ 1 & x\\end{bmatrix}',
            'x', '3')
        self.assertTrue(r['ok'])
        self.assertEqual(r['check']['status'], 'agree')
        self.assertIn('\\begin{bmatrix}', r['result'])


class TestNoncommutativeAtoms(unittest.TestCase):
    # gen 10: products with >= 2 matrix-valued factors atomize as ONE
    # ordered word, so commutative polyrat can never prove AB = BA

    def test_commutator_does_not_vanish(self):
        r = Core.expand(f'{MAT_A}{MAT_B} - {MAT_B}{MAT_A}')
        self.assertTrue(r['ok'])
        self.assertNotEqual(r['result'], '0')
        self.assertEqual(r['check']['status'], 'agree')

    def test_same_word_still_collects(self):
        r = Core.expand(f'{MAT_A}{MAT_B} + {MAT_A}{MAT_B}')
        self.assertTrue(r['result'].startswith('2'))
        self.assertEqual(r['check']['status'], 'agree')

    def test_scalars_commute_out(self):
        r = Core.expand(f'{MAT_A} x - x {MAT_A}')
        self.assertEqual(r['result'], '0')
        self.assertEqual(r['check']['status'], 'agree')

    def test_matrix_sum_power_stays_opaque(self):
        r = Core.expand(f'({MAT_A} + {MAT_B})^2')
        self.assertIn('^{2}', r['result'])   # no fabricated 2AB expansion
        self.assertEqual(r['check']['status'], 'agree')

    def test_single_matrix_power_collects(self):
        r = Core.expand(f'{MAT_A}^2 + {MAT_A}^2')
        self.assertTrue(r['result'].startswith('2'))
        self.assertEqual(r['check']['status'], 'agree')

    def test_scalar_division_works(self):
        r = Core.expand(f'\\frac{{{MAT_A}}}{{2}} + \\frac{{{MAT_A}}}{{2}}')
        self.assertEqual(r['result'], MAT_A)
        self.assertEqual(r['check']['status'], 'agree')

    def test_division_by_matrix_stays_opaque(self):
        r = Core.expand(f'\\frac{{1}}{{{MAT_A}}} {MAT_A}')
        self.assertNotEqual(r['result'], '1')   # A^{-1}A is not scalar 1

    def test_vec_scalar_collect(self):
        r = Core.expand('2\\vec v + 3\\vec v')
        self.assertEqual(r['result'], '5 \\vec v')

    def test_vec_words_do_not_commute(self):
        r = Core.expand('\\vec u \\vec v - \\vec v \\vec u')
        self.assertNotEqual(r['result'], '0')

    def test_collect_matrix_coefficients(self):
        r = Core.collect(f'x {MAT_A} + x {MAT_B}', 'x')
        self.assertEqual(r['check']['status'], 'agree')
        self.assertTrue(r['result'].endswith('x'))

    def test_evaluate_refuses_matrices(self):
        r = Core.evaluate(f'{MAT_A} + {MAT_A}')
        self.assertFalse(r['ok'])
        self.assertIn('matrix', r['error'])


class TestMatrixOracle(unittest.TestCase):
    # gen 10: the oracle evaluates literal matrices with ORDERED
    # multiplication — it can disprove commutation, not just skip

    def test_equal_disproves_commutation(self):
        r = Core.equal_exprs(f'{MAT_A}{MAT_B}', f'{MAT_B}{MAT_A}')
        self.assertEqual(r['verdict'], 'no')
        self.assertEqual(r['lhs'], [[1.0, 0.0], [0.0, 0.0]])
        self.assertEqual(r['rhs'], [[0.0, 0.0], [0.0, 1.0]])

    def test_equal_confirms_product(self):
        r = Core.equal_exprs(f'{MAT_A}{MAT_B}', '\\pmatrix{1 & 0 \\cr 0 & 0}')
        self.assertEqual(r['verdict'], 'yes')

    def test_equal_same_word_canonical(self):
        r = Core.equal_exprs(f'{MAT_A}{MAT_B}', f'({MAT_A})({MAT_B})')
        self.assertEqual(r['verdict'], 'yes')

    def test_scaling_vs_addition(self):
        r = Core.equal_exprs(f'2{MAT_A}', f'{MAT_A}+{MAT_A}')
        self.assertEqual(r['verdict'], 'yes')

    def test_vmatrix_is_a_matrix_literal_not_absolute_value(self):
        expr = '\\begin{vmatrix}1 & 2 \\\\ 3 & 4\\end{vmatrix}'
        sym, notation = P.parse_latex(expr)
        self.assertEqual(P.numeric_eval(sym, notation, {}),
                         [[1.0, 2.0], [3.0, 4.0]])

    def test_shape_mismatch_is_no(self):
        r = Core.equal_exprs('\\pmatrix{1 & 2 \\cr 3 & 4}', '\\pmatrix{1 & 2}')
        self.assertEqual(r['verdict'], 'no')

    def test_cancelled_matrix_agrees_with_scalar_zero(self):
        c = P.numeric_spot_check(f'{MAT_A} - {MAT_A}', '0')
        self.assertEqual(c['status'], 'agree')

    def test_symbolic_vectors_stay_skipped(self):
        r = Core.equal_exprs('\\vec u \\vec v', '\\vec v \\vec u')
        self.assertEqual(r['verdict'], 'unknown')


class TestLedger(unittest.TestCase):
    def test_record_replay(self):
        path = os.path.join(tempfile.mkdtemp(), 'session.json')
        ledger = Ledger(path)
        r1 = Core.apply_both_sides('2x + 3 = 7', '-', '3')
        ledger.record(r1)
        r2 = Core.expand(r1['result'])
        ledger.record(r2)
        ledger.save()

        ledger2 = Ledger(path)
        self.assertEqual(len(ledger2.steps), 2)
        self.assertTrue(ledger2.steps[1]['continues'])
        rep = ledger2.replay()
        self.assertEqual(rep['status'], 'verified')

    def test_assumptions_accumulate(self):
        path = os.path.join(tempfile.mkdtemp(), 'session.json')
        ledger = Ledger(path)
        ledger.record(Core.apply_both_sides('x y = 1', '/', 'y'))
        ledger.save()
        self.assertEqual(len(Ledger(path).assumptions), 1)

    def test_relation_system_step_replays(self):
        path = os.path.join(tempfile.mkdtemp(), 'system.json')
        ledger = Ledger(path)
        ledger.record(Core.apply_both_sides('x+y=3,x-y=1', '-', 'y'))
        ledger.save()
        self.assertEqual(Ledger(path).replay()['status'], 'verified')

    def test_branch_detection(self):
        path = os.path.join(tempfile.mkdtemp(), 'session.json')
        ledger = Ledger(path)
        ledger.record(Core.expand('(x+1)^2'))
        ledger.record(Core.expand('(y+1)^2'))
        self.assertFalse(ledger.steps[1]['continues'])

    def test_v1_session_upgrades_in_memory(self):
        path = os.path.join(tempfile.mkdtemp(), 'session.json')
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump({'version': 1, 'steps': [], 'assumptions': []}, fh)
        ledger = Ledger(path)
        self.assertEqual(ledger.data['version'], 2)
        self.assertEqual(ledger.claims, [])
        self.assertEqual(ledger.selections, [])
        self.assertEqual(ledger.replay()['status'], 'verified')


class TestGoalAwareLedger(unittest.TestCase):
    CLAIM = r'\lim_{x \to 0} \frac{e^x-1}{x} = 1'

    def test_checked_chain_closes_claim_conditionally(self):
        ledger = Ledger()
        claim = ledger.record_claim(self.CLAIM)
        first = ledger.record(Limits.limit_lhopital(
            r'\lim_{x \to 0} \frac{e^x-1}{x}'), goal=claim['id'])
        second = ledger.record(Limits.limit_substitute(first['result']),
                               goal=claim['id'])
        closed = ledger.conclude(claim['id'], [first['id'], second['id']])
        self.assertEqual(closed['verdict'], 'conditional')
        self.assertEqual(closed['conclusion']['endpoint'], '1')
        self.assertGreater(len(closed['conclusion']['assumptions']), 0)
        self.assertEqual(ledger.replay()['open_claims'], 0)
        self.assertIn('CONDITIONAL', ledger.render())

    def test_unrelated_zero_cannot_close_target(self):
        ledger = Ledger()
        claim = ledger.record_claim(
            r'\lim_{n \to \infty} \frac{n}{2^n} = 0')
        unrelated = ledger.record(Limits.limit_table(
            r'\lim_{n \to \infty} \frac{1}{n}'), goal=claim['id'])
        with self.assertRaisesRegex(ValueError, 'does not close claim'):
            ledger.conclude(claim['id'], [unrelated['id']])
        self.assertEqual(claim['verdict'], 'open')
        self.assertIn('OPEN', ledger.render_markdown())

    def test_equivalent_noop_cannot_establish_false_relation(self):
        ledger = Ledger()
        claim = ledger.record_claim('x = 2')
        step = ledger.record(Core.expand('x = 2'), goal=claim['id'])
        with self.assertRaisesRegex(ValueError, 'does not close claim'):
            ledger.conclude(claim['id'], [step['id']])

    def test_true_relation_endpoint_can_close_identity(self):
        ledger = Ledger()
        claim = ledger.record_claim('(x+1)^2 = x^2+2x+1')
        step = ledger.record(Core.expand(claim['statement']), goal=claim['id'])
        closed = ledger.conclude(claim['id'], [step['id']])
        self.assertEqual(closed['verdict'], 'established')
        self.assertEqual(closed['conclusion']['closure'],
                         'true-relation-endpoint')

    def test_concluded_claim_defines_spine_without_result_selection(self):
        ledger = Ledger()
        claim = ledger.record_claim('(x+1)^2 = x^2+2x+1')
        source = ledger.record(Core.expand('(x+1)^2'), goal=claim['id'])
        dead = ledger.record(Core.substitute(
            source['result'], 'x', '1'), goal=claim['id'])
        marker = ledger.record_branch(
            source['id'], 'numeric evaluation was not a symbolic chain',
            goal=claim['id'])
        ledger.record(Core.expand(source['result']), goal=claim['id'])
        ledger.conclude(claim['id'], [source['id']])
        topology = ledger.presentation_topology()
        self.assertEqual(topology['selected_goal'], claim['id'])
        self.assertEqual(topology['spine'], [source['id']])
        self.assertEqual(topology['abandoned_paths'][0]['marker'],
                         marker['id'])
        self.assertEqual(topology['abandoned_paths'][0]['steps'],
                         [dead['id']])

    def test_claim_must_be_a_relation(self):
        ledger = Ledger()
        with self.assertRaisesRegex(ValueError, 'top-level relation'):
            ledger.record_claim('x^2 + 1')

    def test_equivalent_open_claim_is_reused(self):
        ledger = Ledger()
        first = ledger.record_claim('3x^{2}-3=0')
        again = ledger.record_claim('3x^2-3=0')
        self.assertEqual(again['id'], first['id'])
        self.assertEqual(len(ledger.claims), 1)

    def test_concluded_claim_is_refocused_and_strengthened(self):
        # A repeated statement must focus the SAME claim even after a
        # conclusion: a duplicate id would strand every later step under a
        # goal that can never close the original claim (live 64-turn
        # livelock). A repeated conclude then replaces the closing chain.
        ledger = Ledger()
        claim = ledger.record_claim(self.CLAIM)
        first = ledger.record(Limits.limit_lhopital(
            r'\lim_{x \to 0} \frac{e^x-1}{x}'), goal=claim['id'])
        second = ledger.record(Limits.limit_substitute(first['result']),
                               goal=claim['id'])
        ledger.conclude(claim['id'], [first['id'], second['id']])
        again = ledger.record_claim(self.CLAIM)
        self.assertEqual(again['id'], claim['id'])
        self.assertEqual(len(ledger.claims), 1)
        third = ledger.record(Limits.limit_lhopital(
            r'\lim_{x \to 0} \frac{e^x-1}{x}'), goal=claim['id'])
        fourth = ledger.record(Limits.limit_substitute(third['result']),
                               goal=claim['id'])
        closed = ledger.conclude(claim['id'], [third['id'], fourth['id']])
        self.assertEqual(closed['conclusion']['steps'],
                         [third['id'], fourth['id']])
        self.assertEqual(ledger.replay()['status'], 'verified')

    def test_goal_ownership_and_connectivity_are_enforced(self):
        ledger = Ledger()
        claim = ledger.record_claim(r'\lim_{x \to 0} x = 0')
        no_goal = ledger.record(Limits.limit_substitute(r'\lim_{x \to 0} x'))
        with self.assertRaisesRegex(ValueError, 'belongs to goal'):
            ledger.conclude(claim['id'], [no_goal['id']])

        first = ledger.record(Core.expand('(x+1)^2'), goal=claim['id'])
        last = ledger.record(Limits.limit_substitute(r'\lim_{x \to 0} x'),
                             goal=claim['id'])
        with self.assertRaisesRegex(ValueError, 'does not continue'):
            ledger.conclude(claim['id'], [first['id'], last['id']])

    def test_replay_rejects_tampered_claim_provenance(self):
        ledger = Ledger()
        claim = ledger.record_claim(r'\lim_{x \to 0} x = 0')
        step = ledger.record(Limits.limit_substitute(r'\lim_{x \to 0} x'),
                             goal=claim['id'])
        ledger.conclude(claim['id'], [step['id']])
        claim['conclusion']['endpoint'] = '999'
        replay = ledger.replay()
        self.assertEqual(replay['status'], 'failed')
        self.assertIn('claim replay failed', replay['reason'])


class TestOracleCatchesLies(unittest.TestCase):
    def test_disagree_on_wrong_identity(self):
        c = P.numeric_spot_check('(x+1)^2', 'x^2 + 1')
        self.assertEqual(c['status'], 'disagree')

    def test_respects_assumptions(self):
        c = P.numeric_spot_check('\\frac{x y}{y}', 'x',
                                 assumptions=[{'text': 'y \\ne 0',
                                               'nonzero': 'y'}])
        self.assertEqual(c['status'], 'agree')


class TestAbsoluteValue(unittest.TestCase):
    # gen 12: |...| was parsed as a transparent bracket, so BOTH the
    # symbolic path AND the oracle silently read |x| as x — a soundness
    # hole where the two trust legs shared the blind spot. |...| is now an
    # opaque atom and the oracle computes real |.|.

    def test_abs_is_not_its_argument(self):
        r = Core.equal_exprs('|x|', 'x')
        self.assertEqual(r['verdict'], 'no')

    def test_abs_of_negative_constant(self):
        self.assertEqual(Core.equal_exprs('|{-5}|', '5')['verdict'], 'yes')
        self.assertEqual(Core.equal_exprs('|{-5}|', '-5')['verdict'], 'no')

    def test_expand_keeps_abs_opaque_and_checks(self):
        r = Core.expand('|x|')
        self.assertTrue(r['ok'])
        self.assertEqual(r['result'], '|x|')
        self.assertEqual(r['check']['status'], 'agree')

    def test_like_abs_terms_collect_over_atoms(self):
        r = Core.expand('2|x| + |x|')
        self.assertEqual(r['result'], '3|x|')
        self.assertEqual(r['check']['status'], 'agree')

    def test_square_of_abs_equals_square(self):
        # |x^2| == x^2 because x^2 >= 0; the oracle decides this correctly
        self.assertEqual(Core.equal_exprs('|x^2|', 'x^2')['verdict'], 'yes')

    def test_left_right_bars_also_opaque(self):
        r = Core.equal_exprs('\\left|x-1\\right|', 'x-1')
        self.assertEqual(r['verdict'], 'no')

    def test_evaluate_constant_abs(self):
        r = Core.evaluate('|{-5}|')
        self.assertTrue(r['ok'])
        self.assertEqual(float(r['result']), 5.0)

    def test_differentiate_abs_refuses(self):
        r = Differentiation.differentiate('|x|', 'x')
        self.assertFalse(r['ok'])
        self.assertIn('absolute value', r['error'])

    def test_oracle_evaluates_real_abs(self):
        # the numeric leg alone must see |.|, independent of the symbolic path
        c = P.numeric_spot_check('|x|', 'x')
        self.assertEqual(c['status'], 'disagree')


class TestBigOperatorScoping(unittest.TestCase):
    def test_indexed_operators_keep_their_bodies_in_one_atom(self):
        cases = (
            ('2\\lim_{x \\to 0} \\frac{\\sin x}{x}'
             ' - \\lim_{x \\to 0} \\frac{\\sin x}{x}', '\\lim_'),
            ('2\\sum_{k=0}^{n} x^k - \\sum_{k=0}^{n} x^k', '\\sum_'),
            ('2\\prod_{j=1}^{m} (j+x) - \\prod_{j=1}^{m} (j+x)',
             '\\prod_'),
            ('2\\int_0^1 x^2 \\, dx - \\int_0^1 x^2 \\, dx',
             '\\int_'),
        )
        for expr, operator in cases:
            with self.subTest(expr=expr):
                r = Core.expand(expr)
                self.assertTrue(r['ok'], r.get('error'))
                self.assertEqual(r['opaque_atoms'], 1)
                self.assertIn(operator, r['result'])
                self.assertTrue(r['result'].startswith(operator))
                self.assertNotIn('2' + operator, r['result'])

    def test_binder_aware_free_symbols(self):
        cases = (
            ('\\lim_{x \\to a} \\frac{\\sin x}{x}', {'a'}),
            ('\\sum_{k=0}^{n} x^k', {'n', 'x'}),
            ('\\prod_{j=1}^{m} (j+x)', {'m', 'x'}),
            ('\\int_0^1 x^2 y \\, dx', {'y'}),
        )
        for expr, expected in cases:
            with self.subTest(expr=expr):
                sym, notation = P.parse_latex(expr)
                self.assertEqual(P.free_symbols(sym, notation), expected)

    def test_substitution_refuses_bound_variables(self):
        for expr, var in (
                ('\\lim_{x \\to 0} \\frac{\\sin x}{x}', 'x'),
                ('\\sum_{k=0}^{n} x^k', 'k'),
                ('\\prod_{j=1}^{m} (j+x)', 'j'),
                ('\\int_0^1 x^2 \\, dx', 'x')):
            with self.subTest(expr=expr):
                r = Core.substitute(expr, var, '3')
                self.assertFalse(r['ok'])
                self.assertIn('bound variable', r['error'])

    def test_substitution_still_enters_free_part_of_binder(self):
        r = Core.substitute('\\sum_{k=0}^{n} x^k', 'x', 'y')
        self.assertTrue(r['ok'], r.get('error'))
        self.assertIn('y^k', r['result'])
        self.assertIn('k=0', r['result'])

    def test_substitution_refuses_capture_from_replacement(self):
        r = Core.substitute('\\sum_{k=0}^{n} x^k', 'x', 'k')
        self.assertFalse(r['ok'])
        self.assertIn('would capture bound variable', r['error'])

    def test_infinity_is_not_a_sampled_ring_variable(self):
        r = Core.expand('\\infty - \\infty')
        self.assertFalse(r['ok'])
        self.assertIn('infinity outside', r['error'])
        check = P.numeric_spot_check('\\infty', '\\infty')
        self.assertEqual(check['status'], 'skipped')

    def test_infinity_is_allowed_as_a_big_operator_bound(self):
        r = Core.expand('2\\sum_{k=0}^{\\infty} x^k'
                     ' - \\sum_{k=0}^{\\infty} x^k')
        self.assertTrue(r['ok'], r.get('error'))
        self.assertEqual(r['opaque_atoms'], 1)
        self.assertIn('\\infty', r['result'])


class TestSubscriptedVariables(unittest.TestCase):
    # gen 15: x_{1}-style names are atomic variables INDEPENDENT of their
    # base. The oracle samples them (before: every check on C_{1}-style
    # constants was skipped as unevaluable), and substitute must not capture
    # them through the base (x := 2 turning x_{1} into 2_{1}).

    def test_distinct_subscripts_are_distinct_variables(self):
        r = Core.equal_exprs('x_{1}', 'x_{2}')
        self.assertEqual(r['verdict'], 'no')     # was 'unknown'

    def test_subscripted_terms_commute_over_atoms(self):
        r = Core.equal_exprs('x_{1}+x_{2}', 'x_{2}+x_{1}')
        self.assertEqual(r['verdict'], 'yes')

    def test_constant_difference_is_oracle_checked(self):
        # the composite-glue shape: independent integration constants
        r = Core.expand('(\\frac{x^3}{3}+C_{1})-(\\frac{x^3}{3}+C_{2})')
        self.assertTrue(r['ok'])
        self.assertEqual(r['result'].replace(' ', ''), 'C_{1}-C_{2}')
        self.assertEqual(r['check']['status'], 'agree')  # was 'skipped'

    def test_free_symbols_key_is_the_subscripted_name(self):
        s, n = P.parse_latex('x + x_{1}')
        self.assertEqual(P.free_symbols(s, n), {'x', 'x_{1}'})

    def test_substitute_does_not_capture_subscripted_base(self):
        r = Core.substitute('x + x_{1}', 'x', '2')
        self.assertTrue(r['ok'])
        flat = r['result'].replace(' ', '')
        self.assertIn('x_{1}', flat)
        # no other subscripted-1 name may appear (a captured base would
        # print as (2)_{1} or 2_{1})
        self.assertNotIn('_{', flat.replace('x_{1}', ''))
        self.assertEqual(r['check']['status'], 'agree')

    def test_substitute_refuses_when_only_subscripted_occurs(self):
        r = Core.substitute('x_{1}', 'x', '2')
        self.assertFalse(r['ok'])
        self.assertIn('does not occur', r['error'])

    def test_powered_subscripted_variable_evaluates(self):
        c = P.numeric_spot_check('C_{1}^{2}', 'C_{1} C_{1}')
        self.assertEqual(c['status'], 'agree')

    def test_powered_subscripted_variable_uses_one_atom(self):
        r = Core.expand('C_{1}^{2}-C_{1}C_{1}')
        self.assertEqual(r['result'], '0')
        self.assertEqual(r['check']['status'], 'agree')

    def test_like_powered_subscripted_terms_collect(self):
        r = Core.expand('2C_{1}^{2}+3C_{1}^{2}')
        self.assertEqual(r['result'], '5C_{1}^{2}')
        self.assertEqual(r['check']['status'], 'agree')

    def test_symbolic_subscript_stays_outside_the_oracle(self):
        # x_i is not a numeral subscript: oracle ignorance, never a verdict
        r = Core.equal_exprs('x_i', 'x_j')
        self.assertEqual(r['verdict'], 'unknown')


class TestIntegrateRewrite(unittest.TestCase):
    def test_equal_integrand_accepted(self):
        r = Integration.integrate_rewrite(
            '\\int \\frac{1}{1-x^2} \\, d x', 'x',
            '\\frac{1}{2(1-x)} + \\frac{1}{2(1+x)}')
        self.assertTrue(r['ok'], r.get('error'))
        self.assertEqual(r['check']['status'], 'agree')
        self.assertIn('\\int', r['result'])
        self.assertIn('equal?', r['check']['method'])

    def test_unequal_integrand_refused(self):
        r = Integration.integrate_rewrite('\\frac{1}{1-x^2}', 'x',
                                '\\frac{1}{1-x}')
        self.assertFalse(r['ok'])
        self.assertIn('not mechanically equal', r['error'])

    def test_negative_integrand_stays_consumable(self):
        # a leading minus must be parenthesized, or the result parses as
        # `\int` minus the rest and downstream tactics cannot strip it
        r = Integration.integrate_rewrite('-\\frac{2}{x^{2}}', 'x',
                                '-\\frac{2}{x^{2}}')
        self.assertTrue(r['ok'], r.get('error'))
        nxt = Integration.integrate_power_rule(r['result'], 'x')
        self.assertTrue(nxt['ok'], nxt.get('error'))

    def test_substitute_negative_integrand_stays_consumable(self):
        # same emit rule for integrate_substitute (regression: it used to
        # produce `\int -\frac{1}{16u} \, d u`)
        r = Integration.integrate_substitute('\\frac{1}{16(1-x)}', 'x', '1-x', 'u',
                                   '-\\frac{1}{16u}')
        self.assertTrue(r['ok'], r.get('error'))
        nxt = Integration.integrate_table(r['result'], 'u')
        self.assertTrue(nxt['ok'], nxt.get('error'))


class TestIntegrateLinearity(unittest.TestCase):
    def test_splits_signed_sum(self):
        r = Integration.integrate_linearity('\\int (x^2 - \\sin x + 3) \\, d x', 'x')
        self.assertTrue(r['ok'], r.get('error'))
        self.assertEqual(r['check']['status'], 'exact')
        self.assertEqual(len(r['integrals']), 3)
        self.assertEqual(r['result'].count('\\int'), 3)
        # every piece is independently attackable
        for piece in r['integrals']:
            P.parse_latex(piece)

    def test_refuses_non_sum(self):
        r = Integration.integrate_linearity('\\int x^2 \\, d x', 'x')
        self.assertFalse(r['ok'])
        self.assertIn('not a top-level sum', r['error'])

    def test_abs_is_not_a_sum(self):
        # |...| is an absolute value, never a transparent bracket
        r = Integration.integrate_linearity('\\int \\left|x+1\\right| \\, d x', 'x')
        self.assertFalse(r['ok'])


class TestIntegrateAssemble(unittest.TestCase):
    def test_assembles_signed_pieces_and_checks_derivatives(self):
        r = Integration.integrate_assemble(
            '\\int (x - \\sin x) \\, d x', 'x',
            ['\\frac{1}{2}x^2', '-\\cos x'])
        self.assertTrue(r['ok'], r.get('error'))
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(r['constant'], 'C')
        self.assertEqual([t['sign'] for t in r['linearity_terms']],
                         [1, -1])
        d = Differentiation.differentiate(r['result'], 'x')
        self.assertTrue(d['ok'], d.get('error'))
        self.assertEqual(Core.equal_exprs(d['result'], 'x-\\sin x')['verdict'],
                         'yes')

    def test_rejects_wrong_piece(self):
        r = Integration.integrate_assemble(
            '\\int (x - \\sin x) \\, d x', 'x',
            ['\\frac{1}{2}x^2', '\\cos x'])
        self.assertFalse(r['ok'])
        self.assertIn('piece 2', r['error'])

    def test_rejects_missing_piece(self):
        r = Integration.integrate_assemble('\\int (x+x^2) \\, d x', 'x', ['x^2/2'])
        self.assertFalse(r['ok'])
        self.assertIn('expected 2', r['error'])


class TestPartialFractionsIntegral(unittest.TestCase):
    def test_reported_integral_full_derivation(self):
        """\\int x^2/(1-x^2)^3 dx - the stress case that motivated
        integrate_rewrite + integrate_linearity: every step verified,
        final answer confirmed by exact differentiation."""
        I = '\\int \\frac{x^2}{(1-x^2)^3} \\, d x'
        pf = ('-\\frac{1}{16(1-x)} - \\frac{1}{16(1-x)^2} '
              '+ \\frac{1}{8(1-x)^3} - \\frac{1}{16(1+x)} '
              '- \\frac{1}{16(1+x)^2} + \\frac{1}{8(1+x)^3}')
        r1 = Integration.integrate_rewrite(I, 'x', pf)
        self.assertTrue(r1['ok'], r1.get('error'))
        r2 = Integration.integrate_linearity(r1['result'], 'x')
        self.assertTrue(r2['ok'], r2.get('error'))
        self.assertEqual(len(r2['integrals']), 6)
        pieces = []
        for i, piece in enumerate(r2['integrals']):
            u_expr = '1-x' if i < 3 else '1+x'
            du = -1 if i < 3 else 1
            coeff = 16 if i in (0, 1, 3, 4) else 8
            pw = [1, 2, 3][i % 3]
            mono = 'u' if pw == 1 else f'u^{{{pw}}}'
            new_i = (f'-\\frac{{1}}{{{coeff}{mono}}}' if du < 0
                     else f'\\frac{{1}}{{{coeff}{mono}}}')
            s = Integration.integrate_substitute(piece, 'x', u_expr, 'u', new_i)
            self.assertTrue(s['ok'], f'piece {i}: {s.get("error")}')
            if pw == 1:
                t = Integration.integrate_table(s['result'], 'u')
            else:
                t = Integration.integrate_power_rule(s['result'], 'u')
            self.assertTrue(t['ok'], f'piece {i}: {t.get("error")}')
            z = Core.substitute(t['result'], t['constant'], '0')
            self.assertTrue(z['ok'], f'piece {i}: {z.get("error")}')
            b = Core.substitute(Core.expand(z['result'])['result'], 'u', u_expr)
            self.assertTrue(b['ok'], f'piece {i}: {b.get("error")}')
            pieces.append(b['result'])
        final = Integration.integrate_assemble(r1['result'], 'x', pieces)
        self.assertTrue(final['ok'], final.get('error'))
        self.assertEqual(final['check']['status'], 'agree')
        # independent confirmation: d/dx(answer) == integrand, exactly
        d = Differentiation.differentiate(final['result'], 'x')
        self.assertTrue(d['ok'], d.get('error'))
        self.assertNotIn('\\ln', d['result'])
        self.assertNotIn('\\frac{0}', d['result'])
        # Dropping zero-multiplied log terms exposes a wider written domain;
        # cleanup stays honest by retaining the domain-differs signal.
        self.assertEqual(d['check']['status'], 'domain-differs')
        eq = Core.equal_exprs(d['result'], '\\frac{x^2}{(1-x^2)^3}')
        self.assertEqual(eq['verdict'], 'yes')
        self.assertIn('canonical', eq['method'])


class TestVGroupDifferentiate(unittest.TestCase):
    def test_left_right_parens_are_transparent(self):
        r = Differentiation.differentiate('\\ln\\left(1-x\\right)', 'x')
        self.assertTrue(r['ok'], r.get('error'))
        self.assertEqual(r['check']['status'], 'agree')

    def test_abs_vgroup_still_refused(self):
        r = Differentiation.differentiate('\\left|x\\right|', 'x')
        self.assertFalse(r['ok'])
        self.assertIn('absolute value', r['error'])


class TestDerivativeCheckNearPoles(unittest.TestCase):
    def test_truncation_error_is_not_a_counterexample(self):
        # near x = -1 the central difference of 1/(1+x)^2-shaped terms is
        # dominated by truncation error; a correct symbolic derivative
        # must not be reported as disagree (a live agent redid 17 steps
        # over this)
        c = Differentiation._derivative_check('\\frac{1}{(1+x)^2}',
                                '-\\frac{2}{(1+x)^3}', 'x')
        self.assertEqual(c['status'], 'agree')

    def test_wrong_derivative_still_caught_near_pole(self):
        c = Differentiation._derivative_check('\\ln\\left(1-x\\right)',
                                '\\frac{1}{1-x}', 'x')  # sign is wrong
        self.assertEqual(c['status'], 'disagree')

    def test_wrong_derivative_still_caught(self):
        c = Differentiation._derivative_check('x^3', '2x^2', 'x')
        self.assertEqual(c['status'], 'disagree')


class TestTextbookDifferential(unittest.TestCase):
    def test_dx_in_numerator(self):
        r = Integration.integrate_power_rule('\\int \\frac{dx}{x^2}', 'x')
        self.assertTrue(r['ok'], r.get('error'))
        self.assertEqual(r['result'], '-\\frac{1}{x} + C')

    def test_factor_before_differential(self):
        r = Integration.integrate_power_rule('\\int \\frac{x^2 dx}{x^4}', 'x')
        self.assertTrue(r['ok'], r.get('error'))
        self.assertEqual(r['result'], '-\\frac{1}{x} + C')

    def test_wrong_variable_refused(self):
        r = Integration.integrate_power_rule('\\int \\frac{dy}{y^2}', 'x')
        self.assertFalse(r['ok'])
        self.assertIn('not with respect to', r['error'])

    def test_substitution_on_textbook_form(self):
        # the stress cell: \int dx/(x^{1/2}+x^{1/3}), u = x^{1/6}
        r = Integration.integrate_substitute(
            '\\int \\frac {dx} {(x^{\\frac 1 2} + x^{\\frac 1 3})}', 'x',
            'x^{\\frac{1}{6}}', 'u', '\\frac{6u^5}{u^3+u^2}')
        self.assertTrue(r['ok'], r.get('error'))
        self.assertEqual(r['check']['status'], 'agree')

    def test_trailing_differential_form_unaffected(self):
        r = Integration.integrate_power_rule('\\int \\frac{1}{x^2} \\, d x', 'x')
        self.assertTrue(r['ok'], r.get('error'))
        self.assertEqual(r['result'], '-\\frac{1}{x} + C')


class TestPuiseuxFold(unittest.TestCase):
    """expand folds rational-exponent powers of plain variables via
    t = x^{1/q} into integer polyrat and maps back (records x > 0)."""

    def ok(self, rec, status='agree'):
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['check']['status'], status,
                         f"{rec.get('result')} check={rec['check']}")
        return rec

    def test_reported_expression_canonicalizes(self):
        E = ('\\frac{(x^{\\frac{1}{6}})^{4}}'
             '{x x^{\\frac{1}{6}} + x}')
        r = self.ok(Core.expand(E))
        self.assertEqual(r['result'].replace(' ', ''),
                         '\\frac{1}{x^{\\frac{1}{2}}+x^{\\frac{1}{3}}}')
        self.assertIn({'text': 'x > 0', 'nonzero': 'x'}, r['assumptions'])
        eq = Core.equal_exprs(r['result'],
                           '\\frac{1}{x^{\\frac{1}{2}}+x^{\\frac{1}{3}}}')
        self.assertEqual(eq['verdict'], 'yes')

    def test_power_of_root_folds(self):
        r = self.ok(Core.expand('(x^{\\frac{1}{6}})^{4}'))
        self.assertEqual(r['result'].replace(' ', ''),
                         'x^{\\frac{2}{3}}')

    def test_bare_variable_merges(self):
        r = self.ok(Core.expand('x \\cdot x^{\\frac{1}{6}}'))
        self.assertEqual(r['result'].replace(' ', ''),
                         'x^{\\frac{7}{6}}')

    def test_domain_extension_is_flagged(self):
        # (x^{1/6})^6 = x only for x >= 0: the fold is right, the oracle
        # honestly reports the domain extension
        r = Core.expand('(x^{\\frac{1}{6}})^{6}')
        self.assertTrue(r['ok'], r.get('error'))
        self.assertEqual(r['result'], 'x')
        self.assertEqual(r['check']['status'], 'domain-differs')

    def test_composite_base_never_folds(self):
        # (x^2)^{1/2} is |x|; the fold applies only to plain-variable bases
        r = Core.expand('(x^{2})^{\\frac{1}{2}}')
        self.assertTrue(r['ok'], r.get('error'))
        self.assertIn('\\frac', r['result'])  # exponent stays fractional

    def test_negative_fractional_power_skips(self):
        r = Core.expand('x^{-\\frac{1}{2}} x')
        self.assertTrue(r['ok'], r.get('error'))  # unchanged, no fold

    def test_atoms_coexist_with_fold(self):
        r = self.ok(Core.expand('2 x^{\\frac{1}{2}} \\sin y '
                             '+ 3 x^{\\frac{1}{2}} \\sin y'))
        self.assertEqual(r['result'].replace(' ', ''),
                         '5x^{\\frac{1}{2}}\\siny')

    def test_integer_only_expressions_untouched(self):
        r = self.ok(Core.expand('(x+1)(x-2)'))
        self.assertEqual(r['result'], 'x^{2}-x-2')
        self.assertEqual(r['assumptions'], [])


class TestLimitTactics(unittest.TestCase):
    def ok(self, rec, status='agree'):
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['check']['status'], status, rec.get('check'))
        return rec

    def test_opaque_binder_sum_keeps_plus_separator(self):
        # Stress regression: deep atom canonicalization used to turn the
        # mixed body ``sin(x)/x + x^2`` into a product while the opaque
        # limit made the ordinary oracle skip the transformation.
        r = Core.expand('\\lim_{x \\to 0} (\\frac{\\sin x}{x}+x^2)')
        self.assertTrue(r['ok'], r.get('error'))
        self.assertIn('+x^{2}', r['result'].replace(' ', ''))

    def test_one_sided_limit_round_trip_and_scope(self):
        for text, marker in [
                ('\\lim_{x \\to 0^-} \\frac{1}{x}', '^-'),
                ('\\lim_{x \\to 0^{+}} \\frac{1}{x}', '^+')]:
            sym, notation = P.parse_latex(text)
            out = P.write_latex(sym, notation).replace(' ', '')
            self.assertIn(marker, out)
            self.assertEqual(P.free_symbols(sym, notation), set())
            P.parse_latex(out)

    def test_continuity_substitution(self):
        r = self.ok(Limits.limit_substitute(
            '\\lim_{x \\to 2} (x^2+1)'))
        self.assertEqual(r['result'], '5')
        self.assertIn('continuous', r['assumptions'][0]['text'])

    def test_continuity_substitution_refuses_pole(self):
        r = Limits.limit_substitute('\\lim_{x \\to 0} \\frac{1}{x}')
        self.assertFalse(r['ok'])
        self.assertIn('approach oracle', r['error'])

    def test_rewrite_body_preserves_direction(self):
        r = self.ok(Limits.limit_rewrite(
            '\\lim_{x \\to 1^+} \\frac{x^2-1}{x-1}', 'x+1'))
        self.assertIn('^{+}', r['result'])
        self.assertIn('x+1', r['result'].replace(' ', ''))

    def test_standard_table_and_rational_infinity(self):
        constant = self.ok(Limits.limit_table(
            '\\lim_{x \\to \\infty} 7'), status='exact')
        self.assertEqual(constant['result'], '7')
        r = self.ok(Limits.limit_table(
            '\\lim_{x \\to 0} \\frac{\\sin x}{x}'))
        self.assertEqual(r['result'], '1')
        self.assertEqual(r['rule'], 'sin(x)/x')
        r = self.ok(Limits.limit_table(
            '\\lim_{x \\to \\infty} '
            '\\frac{3x^2+1}{x^2-2}'))
        self.assertEqual(r['result'], '3')

    def test_table_refuses_unknown_form(self):
        r = Limits.limit_table('\\lim_{x \\to 0} \\frac{\\tan x}{x}')
        self.assertFalse(r['ok'])
        self.assertIn('no table rule', r['error'])

    def test_rightarrow_binder_normalizes(self):
        # \rightarrow lexes to the canonical \to: the binder is a real
        # comparison (n stays bound) and every limit tactic accepts it
        r = self.ok(Limits.limit_table(
            '\\lim_{n \\rightarrow \\infty} \\frac{3}{2^n}'))
        self.assertEqual(r['result'], '0')
        s, n = P.parse_latex('\\lim_{n \\rightarrow \\infty} \\frac{3}{2^n}')
        self.assertEqual(P.free_symbols(s, n), set())
        self.assertIn('\\to', P.write_latex(s, n))

    def test_geometric_decay_at_infinity(self):
        for latex in (
                '\\lim_{n \\to \\infty} \\left(\\frac{1}{2}\\right)^n',
                '\\lim_{n \\to \\infty} \\frac{1}{2^n \\ln(2)}',
                '\\lim_{n \\to \\infty} \\frac{1}{\\ln(2) 2^n}',
                '\\lim_{n \\to \\infty} \\frac{3}{2^n}'):
            r = self.ok(Limits.limit_table(latex))
            self.assertEqual(r['result'], '0', latex)
            self.assertEqual(r['rule'], 'geometric decay at infinity')

    def test_geometric_decay_stays_narrow(self):
        for latex in (
                # variable outside the decaying power
                '\\lim_{n \\to \\infty} \\frac{n}{2^n}',
                # growth, not decay
                '\\lim_{n \\to \\infty} 2^n',
                '\\lim_{n \\to \\infty} \\frac{1}{(\\frac{1}{2})^{n}}',
                # decay only at +infinity
                '\\lim_{n \\to -\\infty} \\left(\\frac{1}{2}\\right)^n'):
            r = Limits.limit_table(latex)
            self.assertFalse(r['ok'], latex)
            self.assertIn('no table rule', r['error'])

    def test_lhopital_assumption_display_and_render(self):
        r = self.ok(Limits.limit_lhopital(
            '\\lim_{x \\to 0} \\frac{e^x-1}{x}'))
        displays = [a.get('display') for a in r['assumptions']]
        self.assertTrue(all(displays))
        # mixed record: math inside $...$ spans, prose outside them
        self.assertIn(' are differentiable', displays[0])
        self.assertNotIn('differentiable', ''.join(
            displays[0].split('$')[1::2]))
        ledger = Ledger()
        ledger.record(r)
        md = ledger.render_markdown()
        # prose is no longer swallowed by a whole-line math wrapping
        self.assertIn('are differentiable in a punctured neighborhood', md)
        self.assertNotIn('the approach point$', md)

    def test_lhopital_zero_over_zero(self):
        r = self.ok(Limits.limit_lhopital(
            '\\lim_{x \\to 0} \\frac{e^x-1}{x}'))
        self.assertEqual(r['indeterminate_form'], '0/0')
        self.assertIn('e^x', r['result'])
        self.assertTrue(r['assumptions'])

    def test_lhopital_infinity_over_infinity(self):
        r = self.ok(Limits.limit_lhopital(
            '\\lim_{x \\to \\infty} '
            '\\frac{x^2+1}{x^2-1}'))
        self.assertEqual(r['indeterminate_form'], 'infinity/infinity')

    def test_lhopital_refuses_non_indeterminate_quotient(self):
        r = Limits.limit_lhopital(
            '\\lim_{x \\to 0} \\frac{x+1}{x+2}')
        self.assertFalse(r['ok'])
        self.assertIn('indeterminate form', r['error'])

    def test_linearity_and_checked_assembly(self):
        expr = '\\lim_{x \\to 0} (\\frac{\\sin x}{x}+x^2)'
        split = self.ok(Limits.limit_linearity(expr), status='exact')
        self.assertEqual(len(split['terms']), 2)
        assembled = self.ok(Limits.limit_assemble(expr, ['1', '0']))
        self.assertEqual(assembled['result'], '1')
        bad = Limits.limit_assemble(expr, ['0', '1'])
        self.assertFalse(bad['ok'])
        self.assertIn('piece 1', bad['error'])

    def test_limit_steps_replay(self):
        ledger = Ledger()
        ledger.record(Limits.limit_lhopital(
            '\\lim_{x \\to 0} \\frac{e^x-1}{x}'))
        ledger.record(Limits.limit_substitute(ledger.last_result()))
        self.assertEqual(ledger.last_result(), '1')
        self.assertEqual(ledger.replay()['status'], 'verified')


class TestIndexItemRoundTrip(unittest.TestCase):
    def test_raw_frac_power_is_braced(self):
        # the kernel Preprocessor puts raw FracValue in INDEX power slots;
        # the writer used to emit x^\frac{1}{2}, which its own grammar
        # cannot re-parse (regression: int! [[n]] on \int dx/(x^{1/2}+...))
        from value import FracValue
        from LatexWriter import LaTexWriter
        from notation import Symbol as Sym
        n = Notation()
        idx = n.setf(Notation.INDEX,
                     (Sym('x'), (None, None, FracValue(1, 2), None)))
        out = LaTexWriter(n)(idx)
        self.assertEqual(out, 'x^{\\frac{1}{2}}')
        P.parse_latex(out)

    def test_plain_and_grouped_dims_unchanged(self):
        from LatexWriter import LaTexWriter
        from notation import Symbol as Sym
        from value import IntegerValue
        n = Notation()
        idx = n.setf(Notation.INDEX,
                     (Sym('x'), (None, None, IntegerValue(2), None)))
        self.assertEqual(LaTexWriter(n)(idx), 'x^{2}')
        s, n2 = P.parse_latex('x^{\\frac{1}{2}} + x^y')
        P.parse_latex(P.write_latex(s, n2))


class TestLedgerComment(unittest.TestCase):
    def test_note_is_transparent_to_the_chain(self):
        ledger = Ledger()
        r1 = Core.apply_both_sides('2x + 3 = 7', '-', '3')
        ledger.record(r1)
        note = ledger.record_comment('now tidy both sides')
        self.assertEqual(note['op'], 'comment')
        self.assertIsNone(note['result'])
        r2 = Core.expand(r1['result'])
        step = ledger.record(r2)
        # continuity is judged against the last real result, not the note
        self.assertTrue(step['continues'])
        self.assertEqual(ledger.last_result(), r2['result'])

    def test_replay_skips_notes(self):
        ledger = Ledger()
        ledger.record_comment('plan: expand the square')
        ledger.record(Core.expand('(x+1)^2'))
        rep = ledger.replay()
        self.assertEqual(rep['status'], 'verified')

    def test_renderings_show_notes(self):
        ledger = Ledger()
        ledger.record_comment('strategy: partial fractions')
        self.assertIn('note: strategy: partial fractions',
                      ledger.render())
        self.assertIn('strategy: partial fractions',
                      ledger.render_markdown())

    def test_empty_note_refused(self):
        ledger = Ledger()
        with self.assertRaises(ValueError):
            ledger.record_comment('   ')


class TestLedgerBranchMarker(unittest.TestCase):
    def marker_ledger(self):
        ledger = Ledger()
        source = ledger.record(Core.expand('(x+1)^2'))
        marker = ledger.record_branch(
            source['id'], 'the substitution route stalled')
        return ledger, source, marker

    def test_marker_is_structured_non_provenance_and_replays(self):
        ledger, source, marker = self.marker_ledger()
        self.assertEqual(marker['op'], 'branch')
        self.assertEqual(marker['args'], {
            'from': source['id'],
            'reason': 'the substitution route stalled',
        })
        self.assertIsNone(marker['result'])
        self.assertEqual(marker['check']['status'], 'note')
        self.assertEqual(ledger.last_result(), source['result'])
        self.assertEqual(ledger.replay()['status'], 'verified')
        self.assertIn('branch from s1 (awaiting continuation): '
                      'the substitution route stalled',
                      ledger.render())
        self.assertIn('*branch from `s1`*', ledger.render_markdown())

    def test_marker_round_trips_in_a_saved_session(self):
        path = os.path.join(tempfile.mkdtemp(), 'branch.json')
        ledger = Ledger(path)
        source = ledger.record(Core.expand('(x+1)^2'))
        ledger.record_branch(source['id'], 'try the factored route')
        ledger.save()
        loaded = Ledger(path)
        self.assertEqual(loaded.steps[1]['args']['from'], 's1')
        self.assertEqual(loaded.replay()['status'], 'verified')

    def test_invalid_source_reason_and_goal_are_refused(self):
        ledger = Ledger()
        with self.assertRaisesRegex(ValueError, 'unknown branch source'):
            ledger.record_branch('s999', 'resume')
        note = ledger.record_comment('ordinary note')
        with self.assertRaisesRegex(ValueError, 'not a transforming step'):
            ledger.record_branch(note['id'], 'resume')
        note['result'] = 'forged result'
        with self.assertRaisesRegex(ValueError, 'not a transforming step'):
            ledger.record_branch(note['id'], 'resume')
        note['result'] = None
        source = ledger.record(Core.expand('(x+1)^2'))
        with self.assertRaisesRegex(ValueError, 'needs a reason'):
            ledger.record_branch(source['id'], '  ')

        claim = ledger.record_claim('x=x')
        with self.assertRaisesRegex(ValueError, 'belongs to goal'):
            ledger.record_branch(source['id'], 'resume', goal=claim['id'])

    def test_replay_rejects_tampered_source_and_reason(self):
        ledger, _source, marker = self.marker_ledger()
        marker['args']['from'] = 's999'
        replay = ledger.replay()
        self.assertEqual(replay['status'], 'failed')
        self.assertIn('branch marker invalid', replay['reason'])

        ledger, _source, marker = self.marker_ledger()
        marker['args']['reason'] = 'silently changed'
        replay = ledger.replay()
        self.assertEqual(replay['status'], 'failed')
        self.assertIn('hash mismatch', replay['reason'])

    def test_marker_cannot_close_a_claim(self):
        ledger = Ledger()
        claim = ledger.record_claim('x=x')
        source = ledger.record(Core.expand('x=x'), goal=claim['id'])
        marker = ledger.record_branch(
            source['id'], 'try another route', goal=claim['id'])
        with self.assertRaisesRegex(ValueError, 'not a transforming step'):
            ledger.conclude(claim['id'], [marker['id']])

    def test_next_transform_persists_edge_and_selection_defines_spine(self):
        ledger = Ledger()
        source = ledger.record(Core.expand('(x+1)^2'))
        dead = ledger.record(Core.substitute(source['result'], 'x', '1'))
        marker = ledger.record_branch(
            source['id'], 'the numeric detour does not answer the goal')
        resumed = ledger.record(Core.factor_quadratic(
            source['result'], 'x'))
        self.assertEqual(resumed['exploration']['marker'], marker['id'])
        self.assertEqual(resumed['exploration']['from'], source['id'])
        self.assertFalse(resumed['continues'])  # chronological, not topology

        selection = ledger.record_selection(resumed['result'], {
            'status': 'verified', 'source': 'ledger',
            'step': resumed['id'], 'method': 'exact-result',
        })
        topology = ledger.presentation_topology()
        self.assertEqual(selection['id'], 'r1')
        self.assertEqual(topology['spine'], [source['id'], resumed['id']])
        self.assertEqual(topology['abandoned_paths'], [{
            'marker': marker['id'], 'source': source['id'],
            'continues_at': resumed['id'],
            'reason': 'the numeric detour does not answer the goal',
            'steps': [dead['id']],
        }])
        md = ledger.render_markdown()
        self.assertIn('<details>', md)
        self.assertIn('Abandoned path from <code>s1</code>', md)
        self.assertIn('resumed as <code>s4</code>', md)
        self.assertIn('**s2**', md)  # checked work stays expandable
        self.assertIn('Selected final result `r1` from `s4`', md)
        self.assertEqual(ledger.replay()['status'], 'verified')

    def test_branch_target_must_resume_source_and_markers_do_not_stack(self):
        ledger = Ledger()
        source = ledger.record(Core.expand('(x+1)^2'))
        marker = ledger.record_branch(source['id'], 'try another route')
        with self.assertRaisesRegex(ValueError, 'input does not resume'):
            ledger.record(Core.expand('(y+1)^2'))
        with self.assertRaisesRegex(ValueError, 'still needs'):
            ledger.record_branch(source['id'], 'stack another marker')
        # A partial saved session with an unresolved marker remains replayable.
        self.assertEqual(ledger.replay()['status'], 'verified')
        resumed = ledger.record(Core.factor_quadratic(
            source['result'], 'x'))
        self.assertEqual(resumed['exploration']['marker'], marker['id'])

    def test_replay_rejects_tampered_edge_and_selection(self):
        ledger = Ledger()
        source = ledger.record(Core.expand('(x+1)^2'))
        ledger.record(Core.substitute(source['result'], 'x', '1'))
        ledger.record_branch(source['id'], 'discard the value-only route')
        resumed = ledger.record(Core.factor_quadratic(
            source['result'], 'x'))
        selection = ledger.record_selection(resumed['result'], {
            'status': 'verified', 'source': 'ledger',
            'step': resumed['id'], 'method': 'exact-result',
        })
        resumed['exploration']['from'] = 's2'
        replay = ledger.replay()
        self.assertEqual(replay['status'], 'failed')
        self.assertIn('edge metadata', replay['reason'])

        resumed['exploration'] = ledger._branch_edge(
            ledger.steps[2], resumed)
        selection['provenance']['step'] = 's2'
        replay = ledger.replay()
        self.assertEqual(replay['status'], 'failed')
        self.assertIn('selection hash mismatch', replay['reason'])

    def test_legacy_marker_without_persisted_target_edge_still_replays(self):
        ledger = Ledger()
        source = ledger.record(Core.expand('(x+1)^2'))
        ledger.record(Core.substitute(source['result'], 'x', '1'))
        marker = ledger.record_branch(source['id'], 'legacy resume')
        resumed = ledger.record(Core.factor_quadratic(
            source['result'], 'x'))
        marker['hash'] = ledger_module._legacy_branch_hash(
            source['id'], marker['args']['reason'])
        resumed.pop('exploration')
        self.assertEqual(ledger.replay()['status'], 'verified')
        edge = ledger.branch_edges()[0]
        self.assertEqual((edge['from'], edge['to'], edge['persisted']),
                         ('s1', 's4', False))

    def test_dead_path_assumptions_do_not_condition_selected_spine(self):
        ledger = Ledger()
        source = ledger.record(Core.expand('xy=1'))
        dead = ledger.record(Core.apply_both_sides(
            source['result'], '/', 'y'))
        self.assertTrue(dead['assumptions'])
        ledger.record_branch(source['id'], 'division is unnecessary')
        resumed = ledger.record(Core.apply_both_sides(
            source['result'], '-', '1'))
        ledger.record_selection(resumed['result'], {
            'status': 'verified', 'source': 'ledger',
            'step': resumed['id'], 'method': 'exact-result',
        })
        topology = ledger.presentation_topology()
        self.assertTrue(ledger.assumptions)  # artifact preserves dead work
        self.assertEqual(topology['spine_assumptions'], [])


TELESCOPING_LIMIT = ('\\lim _{n \\rightarrow \\infty}\\left['
                     '\\frac{1}{1 \\cdot 2}+\\frac{1}{2 \\cdot 3}'
                     '+\\ldots+\\frac{1}{n(n+1)}\\right]')
TELESCOPING_SUM_FORM = '\\sum_{k=1}^{n} \\frac{1}{k(k+1)}'


class TestEllipsisGuard(unittest.TestCase):
    def test_primitives_reject_ellipsis_with_steering_error(self):
        for rec in (Core.expand('1 + 2 + \\ldots + n'),
                    Core.equal_exprs('1 + \\cdots + n', '\\frac{n(n+1)}{2}'),
                    Limits.limit_linearity(TELESCOPING_LIMIT),
                    Limits.limit_table(TELESCOPING_LIMIT)):
            self.assertFalse(rec.get('ok'))
            self.assertIn('sum_from_ellipsis', rec['error'])

    def test_cdot_and_dot_are_not_ellipsis(self):
        self.assertEqual(Core.evaluate('2 \\cdot 3')['result'], '6')

    def test_sum_from_ellipsis_is_the_one_door(self):
        rec = FiniteOperators.sum_from_ellipsis(TELESCOPING_LIMIT, TELESCOPING_SUM_FORM)
        self.assertTrue(rec['ok'])


class TestSumOracle(unittest.TestCase):
    def test_finite_sum_evaluates_numerically(self):
        self.assertEqual(
            Core.equal_exprs('\\sum_{k=1}^{5} k', '15')['verdict'], 'yes')

    def test_empty_sum_convention(self):
        self.assertEqual(
            Core.equal_exprs('\\sum_{k=3}^{2} k', '0')['verdict'], 'yes')

    def test_symbolic_summand_with_constant_factor(self):
        self.assertEqual(
            Core.equal_exprs('\\sum_{k=1}^{4} 2k', '20')['verdict'], 'yes')


class TestSumFromEllipsis(unittest.TestCase):
    def test_interprets_inside_limit_binder(self):
        rec = FiniteOperators.sum_from_ellipsis(TELESCOPING_LIMIT, TELESCOPING_SUM_FORM)
        self.assertTrue(rec['ok'])
        self.assertIn('\\lim', rec['result'])
        self.assertIn('\\sum', rec['result'])
        self.assertEqual(rec['check']['status'], 'exact')
        self.assertEqual(len(rec['assumptions']), 1)

    def test_interprets_bare_ellipsis_sum(self):
        rec = FiniteOperators.sum_from_ellipsis(
            '\\frac{1}{1 \\cdot 2}+\\frac{1}{2 \\cdot 3}'
            '+\\ldots+\\frac{1}{n(n+1)}', TELESCOPING_SUM_FORM)
        self.assertTrue(rec['ok'])
        self.assertNotIn('\\lim', rec['result'])

    def test_wrong_summand_rejected(self):
        rec = FiniteOperators.sum_from_ellipsis(TELESCOPING_LIMIT,
                                  '\\sum_{k=1}^{n} \\frac{1}{k^2}')
        self.assertFalse(rec['ok'])
        self.assertIn('does not match', rec['error'])

    def test_single_leading_term_rejected(self):
        rec = FiniteOperators.sum_from_ellipsis('\\frac{1}{2} + \\ldots + \\frac{1}{n}',
                                  '\\sum_{k=2}^{n} \\frac{1}{k}')
        self.assertFalse(rec['ok'])
        self.assertIn('two displayed leading terms', rec['error'])

    def test_minus_terms_rejected(self):
        rec = FiniteOperators.sum_from_ellipsis('1 - 2 + \\ldots - n',
                                  '\\sum_{k=1}^{n} k')
        self.assertFalse(rec['ok'])

    def test_stray_variable_rejected(self):
        rec = FiniteOperators.sum_from_ellipsis(
            '\\frac{1}{1 \\cdot 2}+\\frac{1}{2 \\cdot 3}'
            '+\\ldots+\\frac{1}{n(n+1)}',
            '\\sum_{k=1}^{m} \\frac{1}{k(k+1)}')
        self.assertFalse(rec['ok'])

    def test_pattern_continuation_is_an_assumption(self):
        rec = FiniteOperators.sum_from_ellipsis(TELESCOPING_LIMIT, TELESCOPING_SUM_FORM)
        self.assertIn('\\ldots', rec['assumptions'][0]['text'])
        self.assertIn('continues the pattern',
                      rec['assumptions'][0]['display'])

    def test_lim_wrapped_sum_form_accepted_when_binder_matches(self):
        rec = FiniteOperators.sum_from_ellipsis(
            TELESCOPING_LIMIT,
            '\\lim_{n \\to \\infty} ' + TELESCOPING_SUM_FORM)
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertIn('\\lim', rec['result'])
        self.assertIn('\\sum', rec['result'])

    def test_lim_wrapped_sum_form_refused_for_bare_expression(self):
        rec = FiniteOperators.sum_from_ellipsis(
            '\\frac{1}{1 \\cdot 2}+\\frac{1}{2 \\cdot 3}'
            '+\\ldots+\\frac{1}{n(n+1)}',
            '\\lim_{n \\to \\infty} ' + TELESCOPING_SUM_FORM)
        self.assertFalse(rec['ok'])
        self.assertIn('the expression has none', rec['error'])


class TestSumRewriteTelescope(unittest.TestCase):
    def test_sum_rewrite_partial_fractions(self):
        rec = FiniteOperators.sum_rewrite('\\sum_{k=1}^{n} \\frac{1}{k(k+1)}',
                            '\\frac{1}{k} - \\frac{1}{k+1}')
        self.assertTrue(rec['ok'])
        self.assertIn('\\sum', rec['result'])

    def test_sum_rewrite_keeps_limit_binder(self):
        rec = FiniteOperators.sum_rewrite(
            '\\lim_{n \\to \\infty} \\sum_{k=1}^{n} \\frac{1}{k(k+1)}',
            '\\frac{1}{k} - \\frac{1}{k+1}')
        self.assertTrue(rec['ok'])
        self.assertIn('\\lim', rec['result'])

    def test_sum_rewrite_rejects_inequivalent_summand(self):
        rec = FiniteOperators.sum_rewrite('\\sum_{k=1}^{n} \\frac{1}{k(k+1)}',
                            '\\frac{1}{k}')
        self.assertFalse(rec['ok'])

    def test_telescope_closes_the_sum(self):
        rec = FiniteOperators.sum_telescope('\\sum_{k=1}^{n} \\frac{1}{k(k+1)}',
                              '\\frac{1}{k}')
        self.assertTrue(rec['ok'])
        self.assertEqual(rec['check']['status'], 'agree')
        self.assertEqual(
            Core.equal_exprs(rec['result'], '\\frac{n}{n+1}')['verdict'], 'yes')
        self.assertTrue(rec['assumptions'])

    def test_telescope_carries_limit_binder(self):
        rec = FiniteOperators.sum_telescope(
            '\\lim_{n \\to \\infty} \\sum_{k=1}^{n} \\frac{1}{k(k+1)}',
            '\\frac{1}{k}')
        self.assertTrue(rec['ok'])
        self.assertIn('\\lim', rec['result'])

    def test_telescope_rejects_wrong_term(self):
        rec = FiniteOperators.sum_telescope('\\sum_{k=1}^{n} \\frac{1}{k(k+1)}',
                              '\\frac{1}{k+1}')
        self.assertFalse(rec['ok'])

    def test_infinite_bounds_refused_with_partial_sum_steering(self):
        # sampling \infty as a finite variable once "verified" r^{\infty+1};
        # infinite bounds must be refused at the door (gen-21 invariant)
        rec = FiniteOperators.sum_telescope('\\sum_{k=0}^{\\infty} r^k',
                              '\\frac{r^k}{1-r}')
        self.assertFalse(rec['ok'])
        self.assertIn('partial sums', rec['error'])
        rec = FiniteOperators.sum_from_ellipsis(
            '1 + r + \\ldots + r^n', '\\sum_{k=0}^{\\infty} r^k')
        self.assertFalse(rec['ok'])
        rec = FiniteOperators.sum_rewrite('\\sum_{k=0}^{\\infty} r^k', 'r^k')
        self.assertFalse(rec['ok'])

    def test_telescope_subsumes_table_sums(self):
        # Faulhaber via agent-proposed negated f: no sum_table needed
        rec = FiniteOperators.sum_telescope('\\sum_{k=1}^{n} k', '-\\frac{k(k-1)}{2}')
        self.assertTrue(rec['ok'])
        self.assertEqual(
            Core.equal_exprs(rec['result'],
                          '\\frac{n(n+1)}{2}')['verdict'], 'yes')

    def test_telescope_numeric_upper_bound(self):
        rec = FiniteOperators.sum_telescope('\\sum_{k=1}^{9} \\frac{1}{k(k+1)}',
                              '\\frac{1}{k}')
        self.assertTrue(rec['ok'])
        self.assertEqual(
            Core.equal_exprs(rec['result'], '\\frac{9}{10}')['verdict'], 'yes')

    def test_full_chain_closes_the_series_limit_and_replays(self):
        ledger = Ledger()
        ledger.record(FiniteOperators.sum_from_ellipsis(TELESCOPING_LIMIT,
                                          TELESCOPING_SUM_FORM))
        ledger.record(FiniteOperators.sum_telescope(ledger.last_result(), '\\frac{1}{k}'))
        ledger.record(Limits.limit_table(ledger.last_result()))
        self.assertEqual(ledger.last_result(), '1')
        self.assertEqual(ledger.replay()['status'], 'verified')


WALLIS_LIMIT = ('\\lim _{n \\rightarrow \\infty}\\left(\\frac{1}{2} \\cdot '
                '\\frac{3}{4} \\ldots \\frac{2 n-1}{2 n}\\right)')
WALLIS_PROD_FORM = '\\prod_{k=1}^{n} \\frac{2k-1}{2k}'
WALLIS_EXPLICIT = '\\lim_{n \\to \\infty} \\prod_{k=1}^{n} \\frac{2k-1}{2k}'
WALLIS_UPPER = '\\frac{1}{\\sqrt{2n+1}}'


class TestProdOracle(unittest.TestCase):
    def test_finite_product_evaluates_numerically(self):
        self.assertEqual(
            Core.equal_exprs('\\prod_{k=1}^{4} k', '24')['verdict'], 'yes')

    def test_empty_product_convention(self):
        self.assertEqual(
            Core.equal_exprs('\\prod_{k=3}^{2} k', '1')['verdict'], 'yes')

    def test_product_with_constant_factor(self):
        self.assertEqual(
            Core.equal_exprs('2 \\prod_{k=1}^{3} k', '12')['verdict'], 'yes')


class TestProdFromEllipsis(unittest.TestCase):
    def test_interprets_inside_limit_binder(self):
        rec = FiniteOperators.prod_from_ellipsis(WALLIS_LIMIT, WALLIS_PROD_FORM)
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertIn('\\lim', rec['result'])
        self.assertIn('\\prod', rec['result'])
        self.assertEqual(rec['check']['status'], 'exact')
        self.assertEqual(len(rec['assumptions']), 1)
        self.assertIn('continues the pattern',
                      rec['assumptions'][0]['display'])

    def test_interprets_bare_ellipsis_product(self):
        rec = FiniteOperators.prod_from_ellipsis(
            '\\frac{1}{2} \\cdot \\frac{3}{4} \\ldots \\frac{2n-1}{2n}',
            WALLIS_PROD_FORM)
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertNotIn('\\lim', rec['result'])

    def test_wrong_factor_rejected(self):
        rec = FiniteOperators.prod_from_ellipsis(WALLIS_LIMIT,
                                   '\\prod_{k=1}^{n} \\frac{k}{k+1}')
        self.assertFalse(rec['ok'])
        self.assertIn('does not match', rec['error'])

    def test_single_leading_factor_rejected(self):
        rec = FiniteOperators.prod_from_ellipsis(
            '\\frac{1}{2} \\ldots \\frac{2n-1}{2n}', WALLIS_PROD_FORM)
        self.assertFalse(rec['ok'])
        self.assertIn('two displayed leading factors', rec['error'])

    def test_sum_body_rejected(self):
        rec = FiniteOperators.prod_from_ellipsis(
            '\\frac{1}{2}+\\frac{3}{4}+\\ldots+\\frac{2n-1}{2n}',
            WALLIS_PROD_FORM)
        self.assertFalse(rec['ok'])

    def test_stray_variable_rejected(self):
        rec = FiniteOperators.prod_from_ellipsis(
            WALLIS_LIMIT, '\\prod_{k=1}^{m} \\frac{2k-1}{2k}')
        self.assertFalse(rec['ok'])

    def test_infinite_bounds_refused(self):
        rec = FiniteOperators.prod_from_ellipsis(
            '\\frac{1}{2} \\cdot \\frac{3}{4} \\ldots \\frac{2n-1}{2n}',
            '\\prod_{k=1}^{\\infty} \\frac{2k-1}{2k}')
        self.assertFalse(rec['ok'])

    def test_gate_error_names_both_doors(self):
        rec = Core.expand(
            '\\frac{1}{2} \\cdot \\frac{3}{4} \\ldots \\frac{2n-1}{2n}')
        self.assertFalse(rec['ok'])
        self.assertIn('prod_from_ellipsis', rec['error'])
        self.assertIn('sum_from_ellipsis', rec['error'])

    def test_compressed_frac_spelling_accepted(self):
        # the exact first call of the live Wallis prove! agent: compressed
        # \frac12 spellings inside the full \lim expression must parse
        rec = FiniteOperators.prod_from_ellipsis(
            '\\lim_{n\\to\\infty}\\left(\\frac12\\cdot\\frac34\\ldots'
            '\\frac{2n-1}{2n}\\right)',
            '\\prod_{k=1}^{n}\\frac{2k-1}{2k}')
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertIn('\\lim', rec['result'])
        self.assertIn('\\prod', rec['result'])

    def test_lim_wrapped_prod_form_accepted_when_binder_matches(self):
        # agents naturally re-type the whole limit as the proposal; a
        # matching \lim wrapper is a redundant spelling, not an error
        rec = FiniteOperators.prod_from_ellipsis(
            WALLIS_LIMIT, '\\lim_{n \\to \\infty} ' + WALLIS_PROD_FORM)
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['check']['status'], 'exact')
        self.assertIn('\\lim', rec['result'])

    def test_lim_wrapped_prod_form_refused_on_binder_mismatch(self):
        rec = FiniteOperators.prod_from_ellipsis(
            WALLIS_LIMIT,
            '\\lim_{m \\to \\infty} \\prod_{k=1}^{n} \\frac{2k-1}{2k}')
        self.assertFalse(rec['ok'])
        self.assertIn('differs from the expression', rec['error'])

    def test_lim_wrapped_prod_form_refused_for_bare_expression(self):
        rec = FiniteOperators.prod_from_ellipsis(
            '\\frac{1}{2} \\cdot \\frac{3}{4} \\ldots \\frac{2n-1}{2n}',
            '\\lim_{n \\to \\infty} ' + WALLIS_PROD_FORM)
        self.assertFalse(rec['ok'])
        self.assertIn('the expression has none', rec['error'])


GEOMETRIC_SERIES = '\\sum_{k=0}^{\\infty} (\\frac{1}{2})^k'


class TestSeriesPartialSums(unittest.TestCase):
    def test_sum_door_is_exact_with_existence_assumption(self):
        rec = FiniteOperators.series_partial_sums(GEOMETRIC_SERIES)
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertIn('\\lim_{n \\to \\infty}', rec['result'])
        self.assertIn('\\sum_{k=0}^{n}', rec['result'])
        self.assertEqual(rec['check']['status'], 'exact')
        self.assertEqual(len(rec['assumptions']), 1)
        self.assertIn('when that limit exists',
                      rec['assumptions'][0]['text'])

    def test_product_door(self):
        rec = FiniteOperators.series_partial_sums(
            '\\prod_{k=1}^{\\infty} \\frac{2k-1}{2k}')
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertIn('\\prod_{k=1}^{n}', rec['result'])
        self.assertIn('partial products', rec['assumptions'][0]['display'])

    def test_finite_bounds_refused(self):
        rec = FiniteOperators.series_partial_sums('\\sum_{k=0}^{n} r^k')
        self.assertFalse(rec['ok'])
        self.assertIn('already finite', rec['error'])

    def test_lim_wrapper_refused(self):
        rec = FiniteOperators.series_partial_sums(
            '\\lim_{m \\to \\infty} ' + GEOMETRIC_SERIES)
        self.assertFalse(rec['ok'])
        self.assertIn('bare infinite', rec['error'])

    def test_fresh_bound_avoids_free_symbols(self):
        rec = FiniteOperators.series_partial_sums(
            '\\sum_{k=0}^{\\infty} \\frac{n}{2^k}')
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertIn('\\lim_{m \\to \\infty}', rec['result'])

    def test_infinite_lower_bound_refused_without_circular_steering(self):
        # the generic infinite-bounds refusal steers to this very tactic;
        # from inside it that advice would be circular (agent-livelock
        # class) and must be reworded
        rec = FiniteOperators.series_partial_sums(
            '\\sum_{k=-\\infty}^{0} 2^k')
        self.assertFalse(rec['ok'])
        self.assertNotIn('series_partial_sums', rec['error'])
        self.assertIn('infinite lower bound', rec['error'])

    def test_finite_tactic_refusal_names_this_door(self):
        # the shipped steering once pointed at a rewrite no tactic could
        # perform (unfollowable-advice class); it must name the door now
        rec = FiniteOperators.sum_telescope('\\sum_{k=0}^{\\infty} r^k',
                                            '\\frac{r^k}{1-r}')
        self.assertFalse(rec['ok'])
        self.assertIn('series_partial_sums', rec['error'])


class TestClosedForm(unittest.TestCase):
    def test_wallis_product_equals_factorial_ratio(self):
        # the gen-33 follow-up probe: both sides parse, general equal? is
        # honestly unknown (real-valued sampling); the NAMED integer-
        # domain tactic closes it with the assumption recorded
        rec = FiniteOperators.prod_closed_form(
            WALLIS_PROD_FORM, '\\frac{(2n)!}{2^{2n}(n!)^2}')
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['check']['status'], 'agree')
        self.assertIn('integer', rec['check']['method'])
        self.assertEqual(len(rec['assumptions']), 1)
        self.assertIn('\\mathbb{Z}', rec['assumptions'][0]['text'])

    def test_product_of_indices_is_factorial(self):
        rec = FiniteOperators.prod_closed_form('\\prod_{k=1}^{n} k', 'n!')
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['result'], 'n!')

    def test_faulhaber_sum(self):
        rec = FiniteOperators.sum_closed_form(
            '\\sum_{k=1}^{n} k^2', '\\frac{n(n+1)(2n+1)}{6}')
        self.assertTrue(rec['ok'], rec.get('error'))

    def test_geometric_sum_with_free_ratio(self):
        rec = FiniteOperators.sum_closed_form(
            '\\sum_{k=0}^{n} r^k', '\\frac{1-r^{n+1}}{1-r}')
        self.assertTrue(rec['ok'], rec.get('error'))

    def test_wrong_form_refused_with_witness(self):
        rec = FiniteOperators.sum_closed_form(
            '\\sum_{k=1}^{n} k^2', '\\frac{n(n+1)}{2}')
        self.assertFalse(rec['ok'])
        self.assertIn('disagree at', rec['error'])

    def test_bound_variable_refused(self):
        rec = FiniteOperators.sum_closed_form('\\sum_{k=1}^{n} k^2', 'k n')
        self.assertFalse(rec['ok'])
        self.assertIn('bound variable', rec['error'])

    def test_stray_variable_refused(self):
        rec = FiniteOperators.sum_closed_form(
            '\\sum_{k=1}^{n} k^2', '\\frac{m(m+1)}{2}')
        self.assertFalse(rec['ok'])
        self.assertIn('new free variable', rec['error'])

    def test_vacuous_proposal_refused(self):
        rec = FiniteOperators.sum_closed_form(
            '\\sum_{k=1}^{n} k^2', '\\sum_{k=1}^{n} k^2')
        self.assertFalse(rec['ok'])
        self.assertIn('itself', rec['error'])

    def test_limit_binder_carried_through(self):
        rec = FiniteOperators.sum_closed_form(
            '\\lim_{n \\to \\infty} \\sum_{k=0}^{n} (\\frac{1}{2})^k',
            '2 - (\\frac{1}{2})^{n}')
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertIn('\\lim', rec['result'])

    def test_lim_wrapped_proposal_accepted_when_binder_matches(self):
        rec = FiniteOperators.sum_closed_form(
            '\\lim_{n \\to \\infty} \\sum_{k=0}^{n} (\\frac{1}{2})^k',
            '\\lim_{n \\to \\infty} (2 - (\\frac{1}{2})^{n})')
        self.assertTrue(rec['ok'], rec.get('error'))

    def test_lim_wrapped_proposal_refused_for_bare_expression(self):
        rec = FiniteOperators.sum_closed_form(
            '\\sum_{k=0}^{n} (\\frac{1}{2})^k',
            '\\lim_{m \\to \\infty} (2 - (\\frac{1}{2})^{m})')
        self.assertFalse(rec['ok'])

    def test_ratio_product(self):
        rec = FiniteOperators.prod_closed_form(
            '\\prod_{k=1}^{n} \\frac{k}{k+1}', '\\frac{1}{n+1}')
        self.assertTrue(rec['ok'], rec.get('error'))

    def test_shifted_sum_proposal(self):
        # index shifts are reachable as proposals: the literal loop
        # evaluates the nested shifted sum at each integer bound
        rec = FiniteOperators.sum_closed_form(
            '\\sum_{k=1}^{n} k^2', '\\sum_{j=0}^{n-1} (j+1)^2')
        self.assertTrue(rec['ok'], rec.get('error'))

    def test_infinite_bounds_refused(self):
        rec = FiniteOperators.sum_closed_form(
            '\\sum_{k=0}^{\\infty} r^k', '\\frac{1}{1-r}')
        self.assertFalse(rec['ok'])
        self.assertIn('series_partial_sums', rec['error'])


class TestBigopExpand(unittest.TestCase):
    def test_literal_sum_folds_to_value(self):
        rec = FiniteOperators.sum_expand('\\sum_{k=1}^{5} k^2')
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['result'], '55')
        self.assertEqual(len(rec['terms']), 5)

    def test_literal_product_folds_to_value(self):
        rec = FiniteOperators.prod_expand('\\prod_{k=1}^{4} k')
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['result'], '24')

    def test_empty_conventions_are_exact(self):
        rec = FiniteOperators.sum_expand('\\sum_{k=3}^{2} k')
        self.assertEqual(rec['result'], '0')
        self.assertEqual(rec['check']['status'], 'exact')
        rec = FiniteOperators.prod_expand('\\prod_{k=3}^{2} k')
        self.assertEqual(rec['result'], '1')
        self.assertEqual(rec['check']['status'], 'exact')

    def test_term_cap_refused_with_steering(self):
        rec = FiniteOperators.sum_expand('\\sum_{k=1}^{100} k')
        self.assertFalse(rec['ok'])
        self.assertIn('sum_closed_form', rec['error'])

    def test_symbolic_bound_refused_with_steering(self):
        rec = FiniteOperators.sum_expand('\\sum_{k=1}^{n} k')
        self.assertFalse(rec['ok'])
        self.assertIn('sum_closed_form', rec['error'])

    def test_fractional_terms(self):
        rec = FiniteOperators.sum_expand('\\sum_{k=1}^{3} \\frac{1}{k}')
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(
            Core.equal_exprs(rec['result'], '\\frac{11}{6}')['verdict'],
            'yes')


class TestGeometricSeriesEndToEnd(unittest.TestCase):
    def test_full_chain_closes_at_two_and_replays(self):
        # the backlog's promise: with the definitional door the geometric
        # series closes end to end with EXISTING machinery
        ledger = Ledger()
        ledger.record(FiniteOperators.series_partial_sums(GEOMETRIC_SERIES))
        ledger.record(FiniteOperators.sum_closed_form(
            ledger.last_result(), '2 - (\\frac{1}{2})^{n}'))
        split = Limits.limit_linearity(ledger.last_result())
        ledger.record(split)
        values = []
        for piece in split['limits']:
            rec = Limits.limit_table(piece)
            ledger.record(rec)
            values.append(rec['result'])
        assembled = Limits.limit_assemble(split['input'], values)
        self.assertTrue(assembled['ok'], assembled.get('error'))
        self.assertEqual(assembled['result'], '2')
        # the assemble step needs do!-recorded sources to replay; the
        # chain up to the pieces must replay standalone
        self.assertEqual(ledger.replay()['status'], 'verified')


SIGNED_SERIES = '\\sum_{n=1}^{\\infty} \\frac{(-1)^{n(n-1)/2}}{2^n}'


class TestSeriesConverges(unittest.TestCase):
    def test_signed_series_dominated_by_geometric(self):
        rec = FiniteOperators.series_converges(SIGNED_SERIES,
                                               '\\frac{1}{2^n}')
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertIn('\\le 1', rec['result'])
        self.assertIn('\\left|', rec['result'])
        self.assertEqual(rec['convergence'], 'absolute')
        self.assertIn('geometric', rec['family'])
        self.assertEqual(len(rec['assumptions']), 2)
        self.assertIn('for every', rec['assumptions'][0]['text'].replace(
            '\\text{', '').replace('}', ' ').replace('  ', ' '))

    def test_result_is_a_recordable_replayable_step(self):
        ledger = Ledger()
        rec = ledger.record(FiniteOperators.series_converges(
            SIGNED_SERIES, '\\frac{1}{2^n}'))
        self.assertIn('id', rec)
        self.assertEqual(ledger.replay()['status'], 'verified')

    def test_p_series_and_rational_p(self):
        rec = FiniteOperators.series_converges(
            '\\sum_{n=1}^{\\infty} \\frac{\\sin n}{n^2}',
            '\\frac{1}{n^2}')
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertIn('p-series', rec['family'])
        rec = FiniteOperators.series_converges(
            '\\sum_{n=1}^{\\infty} \\frac{1}{n^{3/2}}',
            '\\frac{1}{n^{3/2}}')
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['tail_bound'], '3')

    def test_geometric_spellings_and_exponent_shift(self):
        for dom in ('(\\frac{1}{2})^n', '\\frac{1}{2^n}'):
            rec = FiniteOperators.series_converges(
                '\\sum_{n=0}^{\\infty} (\\frac{1}{2})^n', dom)
            self.assertTrue(rec['ok'], rec.get('error'))
        rec = FiniteOperators.series_converges(
            '\\sum_{n=1}^{\\infty} \\frac{1}{n!}', '\\frac{1}{2^{n-1}}')
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['tail_bound'], '2')

    def test_full_sum_proposal_accepted(self):
        rec = FiniteOperators.series_converges(
            SIGNED_SERIES, '\\sum_{n=1}^{\\infty} \\frac{1}{2^n}')
        self.assertTrue(rec['ok'], rec.get('error'))

    def test_harmonic_and_growing_families_refused(self):
        rec = FiniteOperators.series_converges(
            '\\sum_{n=1}^{\\infty} \\frac{1}{n}', '\\frac{1}{n}')
        self.assertFalse(rec['ok'])
        self.assertIn('not a convergent family', rec['error'])
        rec = FiniteOperators.series_converges(
            '\\sum_{n=1}^{\\infty} 2^n', '2^n')
        self.assertFalse(rec['ok'])

    def test_domination_violation_refused_with_witness(self):
        rec = FiniteOperators.series_converges(
            '\\sum_{n=1}^{\\infty} \\frac{1}{n}', '\\frac{1}{2^n}')
        self.assertFalse(rec['ok'])
        self.assertIn('domination fails at n=1', rec['error'])

    def test_parametric_series_refused(self):
        rec = FiniteOperators.series_converges(
            '\\sum_{n=0}^{\\infty} r^n', '(\\frac{1}{2})^n')
        self.assertFalse(rec['ok'])
        self.assertIn('parametric', rec['error'])

    def test_finite_sum_refused(self):
        rec = FiniteOperators.series_converges(
            '\\sum_{n=1}^{10} \\frac{1}{2^n}', '\\frac{1}{2^n}')
        self.assertFalse(rec['ok'])
        self.assertIn('trivially', rec['error'])

    def test_unrecognized_factor_refused_with_families_named(self):
        rec = FiniteOperators.series_converges(
            '\\sum_{n=1}^{\\infty} \\frac{1}{n!}', '\\frac{1}{n!}')
        self.assertFalse(rec['ok'])
        self.assertIn('geometric', rec['error'])
        self.assertIn('p > 1', rec['error'])

    def test_refusal_is_never_divergence_evidence(self):
        # the record shape must not smuggle a verdict on refusal
        rec = FiniteOperators.series_converges(
            '\\sum_{n=1}^{\\infty} \\frac{1}{n}', '\\frac{1}{n}')
        self.assertFalse(rec['ok'])
        self.assertNotIn('convergence', rec)
        self.assertNotIn('diverge', rec['error'])
    def test_inverse_sqrt_closes(self):
        rec = Limits.limit_table('\\lim_{n \\to \\infty} ' + WALLIS_UPPER)
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['result'], '0')
        self.assertEqual(rec['rule'], 'root-power decay at infinity')
        self.assertEqual(rec['check']['status'], 'agree')

    def test_rational_powers_and_roots(self):
        for latex in ('\\lim_{n \\to \\infty} \\frac{1}{n^{3/2}}',
                      '\\lim_{n \\to \\infty} \\frac{c}{\\sqrt[3]{n^2+1}}',
                      '\\lim_{n \\to \\infty} \\frac{5}{2\\sqrt{n+3}}'):
            rec = Limits.limit_table(latex)
            self.assertTrue(rec['ok'], (latex, rec.get('error')))
            self.assertEqual(rec['result'], '0', latex)
            self.assertEqual(rec['rule'], 'root-power decay at infinity')

    def test_stays_narrow(self):
        for latex in (
                # growth, not decay
                '\\lim_{n \\to \\infty} \\sqrt{n}',
                # variable-bearing numerator
                '\\lim_{n \\to \\infty} \\frac{n}{\\sqrt{n}}',
                # negative leading coefficient (base heads to -infinity)
                '\\lim_{n \\to \\infty} \\frac{1}{\\sqrt{1-n}}',
                # negative exponent grows
                '\\lim_{n \\to \\infty} \\frac{1}{n^{-2}}',
                # only +infinity
                '\\lim_{n \\to -\\infty} \\frac{1}{\\sqrt{2n+1}}'):
            rec = Limits.limit_table(latex)
            self.assertFalse(rec['ok'], latex)


class TestLimitSqueeze(unittest.TestCase):
    def test_closes_bounded_product(self):
        rec = Limits.limit_squeeze(WALLIS_EXPLICIT, '0', WALLIS_UPPER, '0')
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['result'], '0')
        self.assertEqual(rec['check']['status'], 'agree')
        self.assertEqual(len(rec['assumptions']), 1)
        self.assertIn('\\le', rec['assumptions'][0]['text'])

    def test_closes_oscillating_decay(self):
        rec = Limits.limit_squeeze('\\lim_{n \\to \\infty} \\frac{\\sin n}{n}',
                              '-\\frac{1}{n}', '\\frac{1}{n}', '0')
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['result'], '0')

    def test_vacuous_bound_refused(self):
        rec = Limits.limit_squeeze(WALLIS_EXPLICIT, WALLIS_PROD_FORM,
                              WALLIS_UPPER, '0')
        self.assertFalse(rec['ok'])
        self.assertIn('vacuous', rec['error'])

    def test_equal_bounds_refused(self):
        rec = Limits.limit_squeeze(WALLIS_EXPLICIT, '\\frac{1}{n}',
                              '\\frac{1}{n}', '0')
        self.assertFalse(rec['ok'])

    def test_bad_ordering_refused(self):
        rec = Limits.limit_squeeze(WALLIS_EXPLICIT, WALLIS_UPPER,
                              '\\frac{1}{n}', '0')
        self.assertFalse(rec['ok'])
        self.assertIn('ordering', rec['error'])

    def test_unconfirmed_bound_limit_refused(self):
        rec = Limits.limit_squeeze(WALLIS_EXPLICIT, '0', '\\frac{n}{n+1}', '0')
        self.assertFalse(rec['ok'])
        self.assertIn('not confirmed', rec['error'])

    def test_stray_variable_refused(self):
        rec = Limits.limit_squeeze(WALLIS_EXPLICIT, '0',
                              '\\frac{c}{\\sqrt{n}}', '0')
        self.assertFalse(rec['ok'])

    def test_value_with_bound_variable_refused(self):
        rec = Limits.limit_squeeze(WALLIS_EXPLICIT, '0', WALLIS_UPPER, 'n')
        self.assertFalse(rec['ok'])


class TestEllipsisClaim(unittest.TestCase):
    def test_ellipsis_claim_records_and_closes_conditionally(self):
        ledger = Ledger()
        claim = ledger.record_claim(WALLIS_LIMIT + ' = 0')
        self.assertEqual(claim['verdict'], 'open')
        first = FiniteOperators.prod_from_ellipsis(WALLIS_LIMIT, WALLIS_PROD_FORM)
        s1 = ledger.record(first, goal=claim['id'])
        # the recorded bound-limit steps + sources mirror the do! adapter,
        # so the whole session (not just the conclude call) stays replayable
        lower = ledger.record(Limits.limit_table('\\lim_{n \\to \\infty} 0'),
                              goal=claim['id'])
        upper = ledger.record(
            Limits.limit_table('\\lim_{n \\to \\infty} ' + WALLIS_UPPER),
            goal=claim['id'])
        squeeze = Limits.limit_squeeze(first['result'], '0', WALLIS_UPPER, '0')
        squeeze['sources'] = {'lower': lower['id'], 'upper': upper['id']}
        s2 = ledger.record(squeeze, goal=claim['id'])
        closed = ledger.conclude(claim['id'], [s1['id'], s2['id']])
        self.assertEqual(closed['verdict'], 'conditional')
        self.assertEqual(closed['conclusion']['closure'], 'left-to-right')
        # both the reading of the ellipsis and the ordering are recorded
        self.assertEqual(len(closed['conclusion']['assumptions']), 2)
        # replay validates claim SHAPE only, so the ellipsis statement
        # must not be rejected there (it was until gen 32)
        rep = ledger.replay()
        self.assertEqual(rep['status'], 'verified', rep.get('reason'))

    def test_open_ellipsis_claim_session_replays(self):
        ledger = Ledger()
        ledger.record_claim(WALLIS_LIMIT + ' = 0')
        rep = ledger.replay()
        self.assertEqual(rep['status'], 'verified', rep.get('reason'))
        self.assertEqual(rep['open_claims'], 1)

    def test_retyped_dots_variant_reuses_open_claim(self):
        # live failure: the agent re-typed the goal with \cdots and
        # compressed fractions, minting a SECOND claim that it then closed
        # while the root claim stayed open. The dots family is typography:
        # the re-typed spelling must resolve to the same claim id.
        ledger = Ledger()
        root = ledger.record_claim(WALLIS_LIMIT + ' = 0')
        retyped = ledger.record_claim(
            '\\lim_{n\\to\\infty}\\left(\\frac12\\cdot\\frac34\\cdots'
            '\\frac{2n-1}{2n}\\right)=0')
        self.assertEqual(root['id'], retyped['id'])

    def test_dots_family_is_one_ellipsis(self):
        self.assertTrue(P.same_expression('a + b + \\ldots + z',
                                          'a + b + \\cdots + z'))
        self.assertTrue(P.same_expression('a \\cdot b \\dots z',
                                          'a \\cdot b \\ldots z'))

    def test_non_relation_claim_still_rejected(self):
        ledger = Ledger()
        with self.assertRaises(ValueError):
            ledger.record_claim(
                '\\frac{1}{2} \\cdot \\frac{3}{4} \\ldots \\frac{2n-1}{2n}')


class TestGoalCoverage(unittest.TestCase):
    def test_same_expression_ignores_grouping(self):
        self.assertTrue(P.same_expression('x^{2}', 'x^2'))
        self.assertFalse(P.same_expression('x^2', '2x'))

    def test_same_expression_compares_ellipsis_spellings(self):
        # agents normalize \rightarrow to \to and drop spaces when they
        # restate the goal; the linkage must still recognize it
        self.assertTrue(P.same_expression(
            TELESCOPING_LIMIT,
            '\\lim_{n \\to \\infty}\\left[\\frac{1}{1 \\cdot 2}'
            '+\\frac{1}{2 \\cdot 3}+\\ldots+\\frac{1}{n(n+1)}\\right]'))

    def test_covers_goal_accepts_operator_wrappers(self):
        self.assertTrue(P.covers_goal('\\int x^3 \\, dx', 'x^3'))
        self.assertTrue(P.covers_goal('x^3', '\\int x^3 \\, dx'))
        self.assertTrue(P.covers_goal(
            '\\lim_{x \\to 0} \\frac{\\sin x}{x}',
            '\\frac{\\sin x}{x}'))
        self.assertFalse(P.covers_goal(
            '\\lim_{n \\to \\infty} \\frac{1}{n(n+1)}', TELESCOPING_LIMIT))

    def test_same_expression_unifies_fraction_spelling(self):
        # agents retype \frac exponents as 1/2 when they restate a goal;
        # both division spellings must share one normal form
        self.assertTrue(P.same_expression('x^{\\frac 1 2}', 'x^{1/2}'))
        self.assertTrue(P.same_expression('\\frac{1}{2}', '{1}/{2}'))
        self.assertTrue(P.same_expression('\\frac{x+1}{2}', '{x+1}/{2}'))
        self.assertFalse(P.same_expression('x^{1/2}', 'x^{1/3}'))

    def test_covers_goal_unifies_textbook_integral(self):
        # the textbook differential-in-numerator goal and the canonical
        # first-step input are one integral; a goal typed either with a
        # braced or a parenthesized denominator must be covered
        canon = '\\int \\frac{1}{x^{1/2}+x^{1/3}} \\, d x'
        braces = '\\int\\frac {dx} {x^{\\frac 1 2}+x^{\\frac 1 3}}'
        parens = '\\int\\frac {dx} (x^{\\frac 1 2}+x^{\\frac 1 3})'
        self.assertTrue(P.covers_goal(canon, braces))
        self.assertTrue(P.covers_goal(canon, parens))
        self.assertTrue(P.covers_goal(braces, canon))

    def test_covers_goal_bare_integrand_of_textbook_goal(self):
        # agents legitimately restate a textbook-form goal by its bare
        # integrand (integrate_rewrite accepts one); the phantom dx in
        # the goal's fraction must not break the body comparison
        goal = '\\int\\frac {x^2 dx} {(1-x^2)^3}'
        self.assertTrue(P.covers_goal(
            '\\frac{x^{2}}{(1-x^{2})^{3}}', goal))
        self.assertTrue(P.covers_goal(
            goal, '\\frac{x^{2}}{(1-x^{2})^{3}}'))
        self.assertFalse(P.covers_goal(
            '\\frac{x^{3}}{(1-x^{2})^{3}}', goal))

    def test_covers_goal_integral_negatives(self):
        goal = '\\int\\frac{dx}{x^{1/2}+x^{1/3}}'
        self.assertFalse(P.covers_goal(
            '\\lim_{x \\to 0} \\frac{1}{x^{1/2}+x^{1/3}}', goal))
        self.assertFalse(P.covers_goal(
            '\\int \\frac{1}{x^{1/2}} \\, d x', goal))
        self.assertFalse(P.covers_goal(
            '\\int \\frac{1}{t^{1/2}+t^{1/3}} \\, d t', goal))
        # |...| is semantic, never transparent grouping
        self.assertFalse(P.covers_goal('\\int |x| \\, d x',
                                       '\\int x \\, d x'))


class TestFracArgumentWriting(unittest.TestCase):
    def test_paren_group_frac_argument_is_braced(self):
        # \frac {dx} (g) is ToyMath-dialect: a standard LaTeX reader binds
        # the denominator to "(", so emitted instructions must brace it
        from LatexWriter import LaTexWriter
        sym, notation = P.parse_latex(
            '\\frac {dx} (x^{\\frac 1 2} + x^{\\frac 1 3})')
        out = LaTexWriter(notation)(sym)
        self.assertIn('{(', out.replace(' ', ''))
        self.assertTrue(P.same_expression(
            out, '\\frac {dx} {(x^{\\frac 1 2} + x^{\\frac 1 3})}'))

    def test_plain_frac_arguments_unchanged(self):
        from LatexWriter import LaTexWriter
        for latex in ('\\frac{a}{b}', '\\frac{1}{2}', '\\frac{x+1}{2}',
                      '\\frac{dx}{\\left(x+1\\right)}'):
            sym, notation = P.parse_latex(latex)
            out = LaTexWriter(notation)(sym)
            self.assertTrue(P.same_expression(latex, out), out)


class TestFractionalPowerSteering(unittest.TestCase):
    def test_power_rule_closes_rational_exponent(self):
        rec = Integration.integrate_power_rule('\\int x^{1/2} \\, d x', 'x')
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['check']['status'], 'agree')
        self.assertIn('\\frac{2}{3}x^{\\frac{3}{2}}', rec['result'])
        self.assertEqual([a['text'] for a in rec['assumptions']], ['x > 0'])

    def test_power_rule_closes_root_sum_termwise(self):
        rec = Integration.integrate_power_rule(
            '\\int (x^{\\frac 1 2} + x^{\\frac 1 3}) \\, d x', 'x')
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['check']['status'], 'agree')
        self.assertIn('x^{\\frac{3}{2}}', rec['result'])
        self.assertIn('x^{\\frac{4}{3}}', rec['result'])

    def test_power_rule_negative_rational_exponent(self):
        rec = Integration.integrate_power_rule('\\int x^{-1/2} \\, d x', 'x')
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertIn('2x^{\\frac{1}{2}}', rec['result'])

    def test_power_rule_still_refuses_log_case(self):
        rec = Integration.integrate_power_rule('\\int \\frac{1}{x} \\, d x', 'x')
        self.assertFalse(rec['ok'])
        self.assertIn('logarithm', rec['error'])

    def test_mixed_root_fraction_steers_to_substitution(self):
        rec = Integration.integrate_power_rule(
            '\\int \\frac{1}{x^{1/2}+x^{1/3}} \\, d x', 'x')
        self.assertFalse(rec['ok'])
        self.assertIn('u = x^{1/n}', rec['error'])
        self.assertNotRegex(rec['error'], r'_n\d')

    def test_table_refusal_names_the_real_moves(self):
        rec = Integration.integrate_table('\\int x^{1/2} \\, d x', 'x')
        self.assertFalse(rec['ok'])
        # steering must be truthful: the power rule really does handle this
        self.assertIn('rational literal exponents', rec['error'])

    def test_symbolic_exponent_keeps_its_name(self):
        rec = Integration.integrate_power_rule('\\int x^{n} \\, d x', 'x')
        self.assertFalse(rec['ok'])
        self.assertIn('non-integer exponent n', rec['error'])


class TestSubstituteZeroFolding(unittest.TestCase):
    def test_pinned_constant_leaves_no_zero_residue(self):
        for expr, want in (('2u^3 + C', '2u^{3}'),
                           ('C + x^2', 'x^{2}'),
                           ('x - C', 'x'),
                           ('x + C - y', 'x-y')):
            rec = Core.substitute(expr, 'C', '0')
            self.assertTrue(rec['ok'], rec.get('error'))
            self.assertEqual(rec['result'], want)
            self.assertEqual(rec['check']['status'], 'agree')

    def test_whole_expression_zero_and_total_collapse(self):
        self.assertEqual(Core.substitute('C', 'C', '0')['result'], '(0)')
        self.assertEqual(Core.substitute('C + C', 'C', '0')['result'], '0')

    def test_nonzero_values_keep_parens(self):
        rec = Core.substitute('x + x_{1}', 'x', '2')
        self.assertEqual(rec['result'], '(2)+x_{1}')


A_LIT = r'\pmatrix{1 & 2 \cr 3 & 4}'
B_LIT = r'\pmatrix{5 & 6 \cr 7 & 8}'
A_SYM = r'\pmatrix{a & b \cr c & d}'
B_SYM = r'\pmatrix{e & f \cr g & h}'


class TestMatrixTactics(unittest.TestCase):
    def _ok(self, rec):
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['check']['status'], 'agree', rec['check'])
        return rec

    def test_add_literal_and_symbolic(self):
        rec = self._ok(Matrices.mat_add(f'{A_LIT} + {B_LIT}'))
        self.assertEqual(rec['result'], r'\pmatrix{6 & 8 \cr 10 & 12}')
        rec = self._ok(Matrices.mat_add(f'{A_SYM} + {B_SYM}'))
        self.assertEqual(rec['result'],
                         r'\pmatrix{a+e & b+f \cr c+g & d+h}')

    def test_add_subtraction_and_leading_minus(self):
        rec = self._ok(Matrices.mat_add(f'{A_LIT} - {B_LIT}'))
        self.assertEqual(rec['result'], r'\pmatrix{-4 & -4 \cr -4 & -4}')
        rec = self._ok(Matrices.mat_add(f'-{A_LIT} + {B_LIT}'))
        self.assertEqual(rec['result'], r'\pmatrix{4 & 4 \cr 4 & 4}')

    def test_add_result_takes_first_family(self):
        rec = self._ok(Matrices.mat_add(
            r'\bmatrix{1 & 2 \cr 3 & 4} + ' + B_LIT))
        self.assertIn('bmatrix', rec['result'])

    def test_add_refusals_steer(self):
        rec = Matrices.mat_add(r'\pmatrix{1 & 2} + ' + A_LIT)
        self.assertFalse(rec['ok'])
        self.assertIn('shape mismatch', rec['error'])
        rec = Matrices.mat_add(f'{A_LIT} + 3')
        self.assertFalse(rec['ok'])
        self.assertIn('scalar term', rec['error'])
        rec = Matrices.mat_add(f'2{A_LIT} + {B_LIT}')
        self.assertFalse(rec['ok'])
        self.assertIn('mat_scale', rec['error'])
        rec = Matrices.mat_add(A_LIT)
        self.assertFalse(rec['ok'])
        self.assertIn('sum of matrix literals', rec['error'])

    def test_scale_literal_symbolic_and_fraction(self):
        rec = self._ok(Matrices.mat_scale(f'2 {A_LIT}'))
        self.assertEqual(rec['result'], r'\pmatrix{2 & 4 \cr 6 & 8}')
        rec = self._ok(Matrices.mat_scale(f'x {A_SYM}'))
        self.assertEqual(rec['result'], r'\pmatrix{ax & bx \cr cx & dx}')
        rec = self._ok(Matrices.mat_scale(r'\frac{1}{2} ' + A_LIT))
        self.assertIn(r'\frac {1} {2}', rec['result'])

    def test_scale_negation_and_refusals(self):
        rec = self._ok(Matrices.mat_scale(f'-{A_LIT}'))
        self.assertEqual(rec['result'], r'\pmatrix{-1 & -2 \cr -3 & -4}')
        rec = Matrices.mat_scale(A_LIT)
        self.assertFalse(rec['ok'])
        self.assertIn('no scalar factor', rec['error'])
        rec = Matrices.mat_scale(f'{A_LIT} {B_LIT}')
        self.assertFalse(rec['ok'])
        self.assertIn('mat_mul', rec['error'])

    def test_mul_literal_symbolic_and_rectangular(self):
        rec = self._ok(Matrices.mat_mul(f'{A_LIT} {B_LIT}'))
        self.assertEqual(rec['result'], r'\pmatrix{19 & 22 \cr 43 & 50}')
        rec = self._ok(Matrices.mat_mul(f'{A_SYM} {B_SYM}'))
        self.assertEqual(rec['result'],
                         r'\pmatrix{ae+bg & af+bh \cr ce+dg & cf+dh}')
        rec = self._ok(Matrices.mat_mul(
            r'\pmatrix{1 & 2 & 3 \cr 4 & 5 & 6} '
            r'\pmatrix{1 & 2 \cr 3 & 4 \cr 5 & 6}'))
        self.assertEqual(rec['result'], r'\pmatrix{22 & 28 \cr 49 & 64}')

    def test_mul_keeps_order(self):
        upper = r'\pmatrix{0 & 1 \cr 0 & 0}'
        lower = r'\pmatrix{0 & 0 \cr 1 & 0}'
        ab = self._ok(Matrices.mat_mul(f'{upper} {lower}'))['result']
        ba = self._ok(Matrices.mat_mul(f'{lower} {upper}'))['result']
        self.assertEqual(ab, r'\pmatrix{1 & 0 \cr 0 & 0}')
        self.assertEqual(ba, r'\pmatrix{0 & 0 \cr 0 & 1}')

    def test_mul_refusals_steer(self):
        rec = Matrices.mat_mul(r'\pmatrix{1 & 2} \pmatrix{1 & 2}')
        self.assertFalse(rec['ok'])
        self.assertIn('shape mismatch', rec['error'])
        rec = Matrices.mat_mul(f'2 {A_LIT} {B_LIT}')
        self.assertFalse(rec['ok'])
        self.assertIn('mat_scale', rec['error'])
        rec = Matrices.mat_mul(f'{A_LIT}^2')
        self.assertFalse(rec['ok'])
        self.assertIn('explicit two-factor product', rec['error'])

    def test_transpose_spellings_and_shapes(self):
        want = r'\pmatrix{1 & 3 \cr 2 & 4}'
        for expr in (A_LIT, f'{A_LIT}^T', A_LIT + '^{T}'):
            rec = self._ok(Matrices.transpose(expr))
            self.assertEqual(rec['result'], want)
        rec = self._ok(Matrices.transpose(
            r'\pmatrix{1 & 2 & 3 \cr 4 & 5 & 6}'))
        self.assertEqual(rec['result'],
                         r'\pmatrix{1 & 4 \cr 2 & 5 \cr 3 & 6}')
        rec = self._ok(Matrices.transpose(f'{A_SYM}^T'))
        self.assertEqual(rec['result'], r'\pmatrix{a & c \cr b & d}')

    def test_transpose_refusals(self):
        rec = Matrices.transpose('x^T')
        self.assertFalse(rec['ok'])
        rec = Matrices.transpose(f'{A_LIT}^S')
        self.assertFalse(rec['ok'])
        self.assertIn('^T', rec['error'])

    def test_det_literal_symbolic_and_vmatrix(self):
        self.assertEqual(self._ok(Matrices.det_2x2(A_LIT))['result'], '-2')
        self.assertEqual(self._ok(Matrices.det_2x2(A_SYM))['result'],
                         'ad-bc')
        self.assertEqual(
            self._ok(Matrices.det_2x2(r'\vmatrix{a & b \cr c & d}'))
            ['result'], 'ad-bc')
        self.assertEqual(
            self._ok(Matrices.det_2x2(r'\pmatrix{1 & 2 \cr 2 & 4}'))
            ['result'], '0')

    def test_det_refuses_other_shapes(self):
        rec = Matrices.det_2x2(
            r'\pmatrix{1 & 2 & 3 \cr 4 & 5 & 6 \cr 7 & 8 & 9}')
        self.assertFalse(rec['ok'])
        self.assertIn('2x2', rec['error'])
        rec = Matrices.det_2x2('x + 1')
        self.assertFalse(rec['ok'])

    def test_nested_literals_refused(self):
        rec = Matrices.mat_add(
            r'\pmatrix{\pmatrix{1 & 2 \cr 3 & 4} & 2 \cr 3 & 4} + '
            + A_LIT)
        self.assertFalse(rec['ok'])
        self.assertIn('nested', rec['error'])

    def test_registry_replay_matches(self):
        import tactic_registry
        rec = Matrices.mat_mul(f'{A_SYM} {B_SYM}')
        replayed = tactic_registry.replay(rec['op'], rec['args'])
        self.assertTrue(replayed['ok'])
        self.assertEqual(replayed['result'], rec['result'])


class TestApplyMatrixArguments(unittest.TestCase):
    def test_multiply_records_invertibility_not_nonzero(self):
        rec = Core.apply_both_sides('x = y', '*', A_LIT)
        self.assertTrue(rec['ok'], rec.get('error'))
        texts = [a['text'] for a in rec['assumptions']]
        self.assertTrue(any('invertible' in t for t in texts), texts)
        self.assertFalse(any('\\ne 0' in t for t in texts), texts)

    def test_scalar_multiply_keeps_nonzero_record(self):
        rec = Core.apply_both_sides('x = y', '*', 'z')
        texts = [a['text'] for a in rec['assumptions']]
        self.assertTrue(any('\\ne 0' in t for t in texts), texts)

    def test_divide_and_power_by_matrix_refuse(self):
        rec = Core.apply_both_sides('x = y', '/', A_LIT)
        self.assertFalse(rec['ok'])
        self.assertIn('matrix-valued', rec['error'])
        rec = Core.apply_both_sides('x = y', '^', A_LIT)
        self.assertFalse(rec['ok'])
        self.assertIn('matrix-valued', rec['error'])


if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the agent-scoped verified-derivation primitives."""
import os
import tempfile
import unittest

from notation import Notation
from LatexParser import MathParser
from polyrat import (Poly, RatFunc, NotInFragment, to_ratfunc,
                     ratfunc_to_notation)
import primitives as P
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
        r = P.substitute('x^2 + 2x + 1', 'x', '3')
        self.assertTrue(r['ok'])
        self.assertEqual(P.evaluate(r['result'])['result'], '16')

    def test_symbolic(self):
        r = P.substitute('x^2 + y', 'x', 'a+b')
        self.assertTrue(r['ok'])
        self.assertEqual(P.equal_exprs(r['result'],
                                       'a^2 + 2ab + b^2 + y')['verdict'],
                         'yes')

    def test_check_agrees(self):
        r = P.substitute('x^3 - x', 'x', 'y+1')
        self.assertEqual(r['check']['status'], 'agree')

    def test_missing_variable(self):
        r = P.substitute('x + 1', 'z', '2')
        self.assertFalse(r['ok'])


class TestApplyBothSides(unittest.TestCase):
    def test_subtract(self):
        r = P.apply_both_sides('2x + 3 = 7', '-', '3')
        self.assertTrue(r['ok'])
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(P.expand(r['result'])['result'], '2x = 4')

    def test_divide_constant_no_assumption(self):
        r = P.apply_both_sides('2x = 4', '/', '2')
        self.assertTrue(r['ok'])
        self.assertEqual(r['assumptions'], [])

    def test_divide_symbol_records_assumption(self):
        r = P.apply_both_sides('x y = 1', '/', 'y')
        self.assertTrue(r['ok'])
        self.assertEqual(len(r['assumptions']), 1)
        self.assertIn('\\ne 0', r['assumptions'][0]['text'])

    def test_divide_by_zero_rejected(self):
        r = P.apply_both_sides('x = 1', '/', '0')
        self.assertFalse(r['ok'])

    def test_multiply_by_zero_rejected(self):
        r = P.apply_both_sides('x = 1', '*', '0')
        self.assertFalse(r['ok'])

    def test_not_an_equation(self):
        r = P.apply_both_sides('x + 1', '+', '2')
        self.assertFalse(r['ok'])

    def test_multiply_symbolic_records_assumption(self):
        r = P.apply_both_sides('\\frac{2x+1}{x+1} = 3', '*', 'x+1')
        self.assertTrue(r['ok'])
        self.assertEqual(len(r['assumptions']), 1)
        self.assertIn('\\ne 0', r['assumptions'][0]['text'])

    def test_multiply_parenthesizes_sums(self):
        r = P.apply_both_sides('x + 1 = y', '*', 'a + b')
        self.assertTrue(r['ok'])
        self.assertEqual(r['check']['status'], 'agree')

    def test_power(self):
        r = P.apply_both_sides('x = 2', '^', '2')
        self.assertTrue(r['ok'])
        self.assertEqual(r['check']['status'], 'agree')


class TestExpandCollectEvaluate(unittest.TestCase):
    def test_expand(self):
        r = P.expand('(x+1)(x-2)')
        self.assertEqual(r['result'], 'x^{2}-x-2')
        self.assertEqual(r['check']['status'], 'agree')

    def test_expand_equation(self):
        r = P.expand('{2}x+{3} - {3} = {7} - {3}')
        self.assertEqual(r['result'], '2x = 4')

    def test_expand_trig_via_atoms(self):
        # outside the fragment, expand canonicalizes over opaque atoms
        r = P.expand('\\sin x + 1')
        self.assertTrue(r['ok'])
        self.assertEqual(r['check']['status'], 'agree')

    def test_expand_merges_atom_terms(self):
        r = P.expand('2 \\sin x + 3 \\sin x')
        self.assertTrue(r['ok'])
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(P.equal_exprs(r['result'],
                                       '5 \\sin x')['verdict'], 'yes')

    def test_expand_cancels_atom_terms(self):
        r = P.expand('x \\sin x - x \\sin x + 1')
        self.assertEqual(r['result'], '1')

    def test_expand_finishes_by_parts_assembly(self):
        r = P.expand('x \\left(-\\cos\\left(x\\right)\\right) '
                     '- \\left(-\\sin\\left(x\\right) + C\\right)')
        self.assertTrue(r['ok'])
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(
            P.equal_exprs(r['result'],
                          '-x \\cos(x) + \\sin(x) - C')['verdict'], 'yes')

    def test_equal_via_shared_atoms(self):
        r = P.equal_exprs('2\\sin x + \\cos x - \\cos x', '\\sin(x) + \\sin x')
        self.assertEqual(r['verdict'], 'yes')
        self.assertIn('atoms', r['method'])

    def test_atom_inequality_not_trusted(self):
        # distinct atoms may still be related; canonical inequality must
        # fall through to the oracle, which decides correctly
        r = P.equal_exprs('\\sin^2 x + \\cos^2 x', '1')
        self.assertEqual(r['verdict'], 'yes')
        self.assertIn('numeric', r['method'])

    def test_collect(self):
        r = P.collect('x^2 + 2x + a x + a', 'x')
        self.assertTrue(r['ok'])
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(P.equal_exprs(r['result'],
                                       'x^2 + (a+2)x + a')['verdict'], 'yes')

    def test_collect_equation(self):
        r = P.collect('a x + 2x + 1 = b x', 'x')
        self.assertTrue(r['ok'])
        self.assertEqual(r['check']['status'], 'agree')

    def test_collect_over_atoms(self):
        r = P.collect('x \\sin x + x \\cos x', 'x')
        self.assertTrue(r['ok'])
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(r['result'], '( \\sin x+ \\cos x)x')

    def test_collect_over_atoms_mixed_powers(self):
        r = P.collect('2 e^x + x e^x + x^2', 'x')
        self.assertTrue(r['ok'])
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(P.equal_exprs(
            r['result'], 'x^2 + e^x x + 2 e^x')['verdict'], 'yes')

    def test_collect_equation_over_atoms(self):
        r = P.collect('x \\sin x + x \\cos x = 0', 'x')
        self.assertTrue(r['ok'])
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(r['result'], '( \\sin x+ \\cos x)x = 0')

    def test_collect_var_only_inside_atoms_rejected(self):
        r = P.collect('\\sin x + \\cos x', 'x')
        self.assertFalse(r['ok'])
        self.assertIn('does not occur', r['error'])

    def test_atom_output_drops_redundant_parens(self):
        self.assertEqual(P.expand('2 \\sin x + 3 \\sin x')['result'],
                         '5 \\sin x')
        self.assertEqual(P.expand('x \\sin x + 2 x \\sin x')['result'],
                         '3x \\sin x')
        self.assertEqual(P.expand('\\sin x + \\cos x - \\sin x')['result'],
                         '\\cos x')

    def test_atom_output_keeps_needed_parens(self):
        # a powered atom prints in standard \sin^{2}x form where the
        # position is unambiguous...
        r = P.expand('(\\sin x)^2 + 1')
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(r['result'], '\\sin^{2}x+1')
        # ...but keeps its parens as a non-trailing product factor, where
        # \sin^{2}x \cos^{2}x would rebind the argument
        r = P.expand('\\sin^2 x \\cos^2 x')
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(r['result'], '( \\sin x)^{2} \\cos^{2}x')

    def test_atom_integral_keeps_thin_space(self):
        r = P.expand('x \\sin x + \\int x \\, dx - x \\sin x')
        self.assertEqual(r['result'], '\\int x \\, dx')

    def test_evaluate_fractions(self):
        r = P.evaluate('\\frac{2}{3} + \\frac{1}{6}')
        self.assertTrue(r['exact'])
        self.assertEqual(r['result'], '\\frac {5} {6}')

    def test_evaluate_free_vars_rejected(self):
        r = P.evaluate('x + 1')
        self.assertFalse(r['ok'])

    def test_evaluate_equation_holds(self):
        r = P.evaluate('2(2) + 3 = 7')
        self.assertTrue(r['ok'])
        self.assertTrue(r['holds'])

    def test_evaluate_equation_fails(self):
        r = P.evaluate('2(3) + 3 = 7')
        self.assertTrue(r['ok'])
        self.assertFalse(r['holds'])


class TestAtomPowers(unittest.TestCase):
    # gen 8: \sin^{n} x enters the atom layer as atom(\sin x)^n, so both
    # power spellings meet in one canonical form

    def test_power_forms_share_one_atom(self):
        r = P.expand('(\\sin x)^2 - \\sin^2 x')
        self.assertTrue(r['ok'])
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(r['result'], '0')

    def test_equal_power_forms_is_canonical(self):
        r = P.equal_exprs('\\sin^2 x', '(\\sin x)^2')
        self.assertEqual(r['verdict'], 'yes')
        self.assertIn('atoms', r['method'])

    def test_powers_merge_in_products(self):
        r = P.expand('\\sin^2 x \\sin x')
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(r['result'], '\\sin^{3}x')

    def test_like_powered_terms_merge(self):
        r = P.expand('2\\sin^2 x + 3\\sin^2 x')
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(r['result'], '5 \\sin^{2}x')

    def test_collect_powered_atoms(self):
        r = P.collect('x \\sin^2 x + x \\cos^2 x', 'x')
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(r['result'], '( \\sin^{2}x+ \\cos^{2}x)x')

    def test_grouped_argument_power(self):
        r = P.expand('\\sin^2(x+1) - (\\sin(x+1))^2')
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(r['result'], '0')

    def test_distinct_powered_atoms_not_collapsed(self):
        # soundness: sin^2 and cos^2 stay distinct atoms; no fake identity
        r = P.expand('\\sin^2 x + \\cos^2 x')
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(r['result'], '\\sin^{2}x+ \\cos^{2}x')

    def test_inverse_function_power_stays_opaque(self):
        # \sin^{-1} keeps its arcsin reading: one opaque atom, no power
        r = P.expand('\\sin^{-1} x + \\sin^{-1} x')
        self.assertTrue(r['ok'])
        self.assertEqual(r['check']['status'], 'agree')
        self.assertIn('\\sin^{-', r['result'])


class TestDifferentiate(unittest.TestCase):
    def check(self, expr, var='x'):
        r = P.differentiate(expr, var)
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
        self.check('\\frac{\\sin x}{x}')

    def test_chain_exp(self):
        self.check('e^{x^2}')

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
        r = P.differentiate('x^2 y + y', 'y')
        self.assertTrue(r['ok'])
        self.assertEqual(P.equal_exprs(r['result'], 'x^2 + 1')['verdict'],
                         'yes')


class TestRewrite(unittest.TestCase):
    def test_diff_squares(self):
        r = P.rewrite('x^2 - y^2', 'diff_squares')
        self.assertTrue(r['ok'])
        self.assertEqual(r['check']['status'], 'agree')

    def test_backward(self):
        r = P.rewrite('(x + y)(x - y)', 'diff_squares',
                      direction='backward')
        self.assertTrue(r['ok'])
        self.assertEqual(P.equal_exprs(r['result'], 'x^2 - y^2')['verdict'],
                         'yes')

    def test_no_match(self):
        r = P.rewrite('x + 1', 'diff_squares')
        self.assertFalse(r['ok'])

    def test_unknown_lemma(self):
        r = P.rewrite('x^2 - y^2', 'nope')
        self.assertFalse(r['ok'])

    def test_square_of_sum(self):
        r = P.rewrite('(u + v)^2', 'square_of_sum')
        self.assertTrue(r['ok'])
        self.assertEqual(r['check']['status'], 'agree')


class TestInequalities(unittest.TestCase):
    def test_subtract_keeps_relation(self):
        r = P.apply_both_sides('2x + 3 < 7', '-', '3')
        self.assertTrue(r['ok'])
        self.assertIn('\\lt', r['result'])
        self.assertEqual(P.expand(r['result'])['result'], '2x \\lt 4')

    def test_divide_positive_keeps(self):
        r = P.apply_both_sides('2x \\lt 4', '/', '2')
        self.assertTrue(r['ok'])
        self.assertIn('\\lt', r['result'])

    def test_divide_negative_flips(self):
        r = P.apply_both_sides('-2x \\le 4', '/', '-2')
        self.assertTrue(r['ok'])
        self.assertIn('\\ge', r['result'])
        self.assertEqual(P.expand(r['result'])['result'], 'x \\ge -2')

    def test_multiply_unknown_sign_rejected(self):
        r = P.apply_both_sides('x y > 4', '*', 'y')
        self.assertFalse(r['ok'])

    def test_power_rejected(self):
        r = P.apply_both_sides('x \\ge 3', '^', '2')
        self.assertFalse(r['ok'])

    def test_ne_divide_records_assumption(self):
        r = P.apply_both_sides('x y \\ne 4', '/', 'y')
        self.assertTrue(r['ok'])
        self.assertEqual(len(r['assumptions']), 1)

    def test_evaluate_inequality_holds(self):
        self.assertTrue(P.evaluate('3 \\le 4')['holds'])
        self.assertFalse(P.evaluate('5 \\lt 4')['holds'])
        self.assertTrue(P.evaluate('2(3) \\gt 5')['holds'])


class TestSubtermRewrite(unittest.TestCase):
    def test_inside_sum(self):
        r = P.rewrite('3 + (x^2 - y^2)', 'diff_squares')
        self.assertTrue(r['ok'])
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(r['at'], 'x^{2}-y^{2}')

    def test_inside_fraction(self):
        r = P.rewrite('\\frac{x^2 - y^2}{x + y}', 'diff_squares')
        self.assertTrue(r['ok'])
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(P.equal_exprs(r['result'],
                                       'x - y')['verdict'], 'yes')

    def test_backward_inside(self):
        r = P.rewrite('1 + (a+b)(a-b)', 'diff_squares', 'backward')
        self.assertTrue(r['ok'])
        self.assertEqual(P.equal_exprs(r['result'],
                                       '1 + a^2 - b^2')['verdict'], 'yes')


class TestNumericPowerRewrite(unittest.TestCase):
    # gen 8: a^n pattern terms may bind perfect n-th power monomials

    def test_square_literal(self):
        r = P.rewrite('x^2 - 4', 'diff_squares')
        self.assertTrue(r['ok'])
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(r['result'], '(x+2)(x-2)')
        self.assertEqual(r['numeric'], {'b': '2'})

    def test_literal_on_the_left(self):
        r = P.rewrite('4 - x^2', 'diff_squares')
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(r['result'], '(2+x)(2-x)')
        self.assertEqual(r['numeric'], {'a': '2'})

    def test_monomial_roots(self):
        r = P.rewrite('4x^2 - 9', 'diff_squares')
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(r['result'], '(2x+3)(2x-3)')

    def test_even_symbolic_powers(self):
        r = P.rewrite('x^4 - y^4', 'diff_squares')
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(r['result'], '(x^{2}+y^{2})(x^{2}-y^{2})')

    def test_cube_literals(self):
        r = P.rewrite('x^3 - 8', 'diff_cubes')
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(r['numeric'], {'b': '2'})
        r = P.rewrite('x^3 + 27', 'sum_cubes')
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(r['numeric'], {'b': '3'})

    def test_fraction_square(self):
        r = P.rewrite('x^2 - \\frac{1}{4}', 'diff_squares')
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(r['numeric']['b'], '\\frac {1} {2}')

    def test_imperfect_powers_still_refused(self):
        for expr in ('x^2 - 3', 'x^2 - y', 'x^2 - 4x'):
            r = P.rewrite(expr, 'diff_squares')
            self.assertFalse(r['ok'], expr)

    def test_structural_match_keeps_priority(self):
        r = P.rewrite('x^2 - y^2', 'diff_squares')
        self.assertEqual(r['result'], '(x+y)(x-y)')
        self.assertNotIn('numeric', r)

    def test_numeric_subterm(self):
        r = P.rewrite('\\frac{x^2 - 4}{x + 2}', 'diff_squares')
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(r['at'], 'x^{2}-4')
        self.assertEqual(P.equal_exprs(r['result'], 'x - 2')['verdict'],
                         'yes')

    def test_equation_rewrite_is_checked(self):
        # relation-aware _checked: per-side oracle instead of skipped
        r = P.rewrite('x^2 - 4 = 0', 'diff_squares')
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(r['result'], '(x+2)(x-2)=0')


class TestCollectRational(unittest.TestCase):
    def test_collect_num_and_den(self):
        r = P.collect('\\frac{ax + bx + 1}{x + cx}', 'x')
        self.assertTrue(r['ok'])
        self.assertEqual(r['check']['status'], 'agree')


class TestPrettyOutput(unittest.TestCase):
    def test_no_value_braces(self):
        self.assertEqual(P.expand('(x+1)(x-2)')['result'], 'x^{2}-x-2')

    def test_index_dims_keep_braces(self):
        r = P.expand('x^{12} x')
        self.assertIn('x^{13}', r['result'])

    def test_pretty_reparses_equal(self):
        # every pretty result must parse back to an equal expression
        for expr in ['(x+1)^3', '\\frac{2}{3} + \\frac{1}{6}',
                     '2 \\cdot x \\cdot (x+1)']:
            r = P.expand(expr) if 'frac' not in expr else P.evaluate(expr)
            self.assertEqual(
                P.equal_exprs(r['result'], expr)['verdict'], 'yes', expr)


class TestFactor(unittest.TestCase):
    def test_quadratic_two_roots(self):
        r = P.factor_quadratic('x^2 - 5x + 6', 'x')
        self.assertTrue(r['ok'])
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(sorted(r['roots']), ['2', '3'])

    def test_quadratic_perfect_square(self):
        r = P.factor_quadratic('x^2 - 6x + 9', 'x')
        self.assertTrue(r['ok'])
        self.assertIn('^{2}', r['result'])

    def test_quadratic_leading_coeff(self):
        r = P.factor_quadratic('2x^2 + 5x - 3', 'x')
        self.assertTrue(r['ok'])
        self.assertEqual(r['check']['status'], 'agree')

    def test_quadratic_irrational_rejected(self):
        r = P.factor_quadratic('x^2 - 2', 'x')
        self.assertFalse(r['ok'])

    def test_quadratic_complex_rejected(self):
        r = P.factor_quadratic('x^2 + x + 1', 'x')
        self.assertFalse(r['ok'])

    def test_gcd(self):
        r = P.factor_gcd('6x^2 + 9x')
        self.assertTrue(r['ok'])
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(P.equal_exprs(r['result'],
                                       '3x(2x+3)')['verdict'], 'yes')

    def test_gcd_negative_leading(self):
        r = P.factor_gcd('-2x^2 - 4x')
        self.assertTrue(r['ok'])
        self.assertEqual(r['check']['status'], 'agree')

    def test_gcd_nothing_to_factor(self):
        r = P.factor_gcd('x + 1')
        self.assertFalse(r['ok'])


class TestParsingEdges(unittest.TestCase):
    def test_cdot_chain(self):
        r = P.expand('2 \\cdot x \\cdot (x+1)')
        self.assertTrue(r['ok'])
        self.assertEqual(r['result'], '2x^{2}+2x')

    def test_star_chain(self):
        # '*' and \cdot are grammar-level product separators (P_LIST),
        # so chains parse without preprocessing
        self.assertEqual(P.evaluate('2*3*4')['result'], '24')
        self.assertEqual(P.equal_exprs('a \\cdot b * c', 'c b a')['verdict'],
                         'yes')

    def test_substitute_into_equation(self):
        r = P.substitute('x^2 - 6x + 5 = 0', 'x', '5')
        self.assertTrue(r['ok'])
        self.assertEqual(r['check']['status'], 'agree')
        self.assertTrue(P.evaluate(r['result'])['holds'])

    def test_substitute_wrong_candidate(self):
        r = P.substitute('x^2 - 6x + 5 = 0', 'x', '4')
        self.assertTrue(r['ok'])
        self.assertFalse(P.evaluate(r['result'])['holds'])


class TestIntegration(unittest.TestCase):
    def ok(self, rec):
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['check']['status'], 'agree',
                         f"{rec.get('result')} check={rec['check']}")
        return rec

    def test_power_rule_monomial(self):
        r = self.ok(P.integrate_power_rule('x^2', 'x'))
        self.assertEqual(r['result'], '\\frac {1} {3}x^{3} + C')

    def test_power_rule_polynomial(self):
        r = self.ok(P.integrate_power_rule('3x^2 + 2x + 1', 'x'))
        self.assertEqual(r['result'], 'x^{3}+x^{2}+x + C')

    def test_power_rule_strips_integral_wrapper(self):
        r = self.ok(P.integrate_power_rule('\\int x^2 \\, dx', 'x'))
        self.assertEqual(r['result'], '\\frac {1} {3}x^{3} + C')

    def test_power_rule_negative_power(self):
        r = self.ok(P.integrate_power_rule('\\frac{1}{x^2}', 'x'))
        self.assertEqual(r['result'], '-\\frac{1}{x} + C')

    def test_power_rule_symbolic_coefficient(self):
        self.ok(P.integrate_power_rule('a x', 'x'))

    def test_power_rule_refuses_log_case(self):
        r = P.integrate_power_rule('\\frac{1}{x}', 'x')
        self.assertFalse(r['ok'])
        self.assertIn('integrate_table', r['error'])

    def test_power_rule_refuses_trig(self):
        r = P.integrate_power_rule('\\sin x', 'x')
        self.assertFalse(r['ok'])

    def test_wrong_integration_variable(self):
        r = P.integrate_power_rule('\\int x^2 \\, dx', 'y')
        self.assertFalse(r['ok'])

    def test_definite_integral_refused(self):
        r = P.integrate_power_rule('\\int_0^1 x^2 \\, dx', 'x')
        self.assertFalse(r['ok'])

    def test_table_log_records_assumption(self):
        r = self.ok(P.integrate_table('\\frac{1}{x}', 'x'))
        self.assertIn('\\ln', r['result'])
        self.assertEqual(len(r['assumptions']), 1)

    def test_table_sin(self):
        r = self.ok(P.integrate_table('\\sin x', 'x'))
        self.assertEqual(r['result'], '-\\cos\\left(x\\right) + C')

    def test_table_constant_multiple(self):
        self.ok(P.integrate_table('2 \\cos x', 'x'))

    def test_table_exp(self):
        r = self.ok(P.integrate_table('e^x', 'x'))
        self.assertEqual(r['result'], 'e^{x} + C')

    def test_table_mixed_sum(self):
        self.ok(P.integrate_table('x + \\sin x', 'x'))

    def test_table_refuses_product(self):
        r = P.integrate_table('x \\sin x', 'x')
        self.assertFalse(r['ok'])
        self.assertIn('integrate_by_parts', r['error'])

    def test_by_parts_x_sin(self):
        r = self.ok(P.integrate_by_parts('x \\sin x', 'x', 'x', '\\sin x'))
        self.assertEqual(r['v'], '-\\cos\\left(x\\right)')
        self.assertEqual(r['du'], '1')
        self.assertIn('remaining_integral', r)
        # the remaining integral must feed back into integrate_table
        r2 = self.ok(P.integrate_table(r['remaining_integral'], 'x'))
        self.assertEqual(r2['result'], '-\\sin\\left(x\\right) + C')

    def test_by_parts_rejects_wrong_split(self):
        r = P.integrate_by_parts('x \\sin x', 'x', 'x', '\\cos x')
        self.assertFalse(r['ok'])

    def test_by_parts_requires_u_with_var(self):
        r = P.integrate_by_parts('a \\sin x', 'x', 'a', '\\sin x')
        self.assertFalse(r['ok'])

    def test_fresh_constant_avoids_collision(self):
        r = self.ok(P.integrate_power_rule('C x', 'x'))
        self.assertEqual(r['constant'], 'K')
        self.assertIn('+ K', r['result'])

    def test_final_answer_verifiable_by_existing_primitives(self):
        d = P.differentiate('-x \\cos(x) + \\sin(x)', 'x')
        self.assertTrue(d['ok'])
        self.assertEqual(
            P.equal_exprs(d['result'], 'x \\sin x')['verdict'], 'yes')

    def test_substitute_tactic_chain(self):
        # ∫ 2x cos(x²) dx via u = x²
        r1 = self.ok(P.integrate_substitute('2x \\cos(x^2)', 'x',
                                            'x^2', 'u', '\\cos(u)'))
        self.assertEqual(r1['result'], '\\int \\cos(u) \\, d u')
        self.assertEqual(r1['back_substitute'],
                         {'var': 'u', 'value': 'x^2'})
        r2 = self.ok(P.integrate_table(r1['result'], 'u'))
        r3 = P.substitute(r2['result'], 'u', 'x^2')
        self.assertTrue(r3['ok'])
        d = P.differentiate('\\sin(x^2)', 'x')
        self.assertEqual(
            P.equal_exprs(d['result'], '2x \\cos(x^2)')['verdict'], 'yes')

    def test_substitute_tactic_rejects_wrong_rewrite(self):
        r = P.integrate_substitute('2x \\cos(x^2)', 'x', 'x^2', 'u',
                                   '\\sin(u)')
        self.assertFalse(r['ok'])

    def test_substitute_tactic_rejects_var_leak(self):
        r = P.integrate_substitute('2x \\cos(x^2)', 'x', 'x^2', 'u',
                                   'x \\cos(u)')
        self.assertFalse(r['ok'])

    def test_substitute_tactic_rejects_colliding_uvar(self):
        r = P.integrate_substitute('2u \\cos(u^2)', 'u', 'u^2', 'u',
                                   '\\cos(u)')
        self.assertFalse(r['ok'])

    def test_substitute_tactic_linear_inner(self):
        r1 = self.ok(P.integrate_substitute('\\int e^{3x} \\, dx', 'x',
                                            '3x', 'u', '\\frac{e^u}{3}'))
        r2 = self.ok(P.integrate_table(r1['result'], 'u'))
        self.assertEqual(
            P.equal_exprs(r2['result'].replace(' + C', ''),
                          '\\frac{e^{u}}{3}')['verdict'], 'yes')

    def test_table_constant_denominator(self):
        r = self.ok(P.integrate_table('\\frac{\\sin u}{3}', 'u'))
        self.assertIn('\\cos', r['result'])

    def test_ledger_replay_with_integration(self):
        path = os.path.join(tempfile.mkdtemp(), 'session.json')
        ledger = Ledger(path)
        ledger.record(P.integrate_by_parts('x e^x', 'x', 'x', 'e^x'))
        ledger.record(P.integrate_table('e^x', 'x'))
        ledger.save()
        self.assertEqual(Ledger(path).replay()['status'], 'verified')

    def test_markdown_render(self):
        path = os.path.join(tempfile.mkdtemp(), 'session.json')
        ledger = Ledger(path)
        ledger.record(P.integrate_table('\\frac{1}{x}', 'x'))
        md = ledger.render_markdown()
        self.assertIn('# Verified derivation', md)
        self.assertIn('assumptions', md)
        self.assertIn('\\Longrightarrow', md)


class TestEqual(unittest.TestCase):
    def test_canonical_yes(self):
        self.assertEqual(P.equal_exprs('(x+1)^2',
                                       'x^2 + 2x + 1')['verdict'], 'yes')

    def test_multivariate_cancellation_via_cross_multiply(self):
        # (x+y)(x-y)/(x+y) does not cancel (multivariate GCD is
        # monomial-level) but must still compare equal to x-y
        self.assertEqual(
            P.equal_exprs('\\frac{(x+y)(x-y)}{x+y}', 'x - y')['verdict'],
            'yes')

    def test_canonical_no(self):
        self.assertEqual(P.equal_exprs('(x+1)^2', 'x^2 + 1')['verdict'],
                         'no')

    def test_trig_identity_numeric(self):
        r = P.equal_exprs('\\sin^2 x', '1 - \\cos^2 x')
        self.assertEqual(r['verdict'], 'yes')
        self.assertIn('numeric', r['method'])

    def test_trig_non_identity(self):
        self.assertEqual(P.equal_exprs('\\sin(2x)', '2 \\sin x')['verdict'],
                         'no')

    def test_double_angle(self):
        self.assertEqual(P.equal_exprs('\\sin(2x)',
                                       '2 \\sin x \\cos x')['verdict'], 'yes')

    def test_equations(self):
        self.assertEqual(P.equal_exprs('x = 2', 'x = 2')['verdict'], 'yes')
        self.assertEqual(P.equal_exprs('x = 2', 'x = 3')['verdict'], 'no')
        self.assertEqual(P.equal_exprs('x = 2', 'x + 2')['verdict'], 'no')


class TestDomainAwareOracle(unittest.TestCase):
    # gen 9: a sample point where exactly one side is defined is a
    # definedness witness — the sides differ as real functions

    def test_log_square_is_not_two_log(self):
        r = P.equal_exprs('\\ln(x^2)', '2 \\ln x')
        self.assertEqual(r['verdict'], 'no')
        self.assertIn('domain mismatch', r['method'])
        self.assertIn('counterexample', r)
        # ...but the agent learns equality may hold on a restricted domain
        self.assertIn('restricted domain', r['note'])

    def test_log_product_is_not_sum_of_logs(self):
        r = P.equal_exprs('\\ln(xy)', '\\ln x + \\ln y')
        self.assertEqual(r['verdict'], 'no')
        self.assertIn('domain mismatch', r['method'])

    def test_sqrt_times_sqrt_is_not_x(self):
        r = P.equal_exprs('\\sqrt{x}\\sqrt{x}', 'x')
        self.assertEqual(r['verdict'], 'no')
        self.assertIn('domain mismatch', r['method'])

    def test_common_restricted_domain_is_reported(self):
        # both sides live on x > 0: yes, with the caveat recorded
        r = P.equal_exprs('\\frac{x}{\\sqrt{x}}', '\\sqrt{x}')
        self.assertEqual(r['verdict'], 'yes')
        self.assertIn('both sides are defined', r['note'])

    def test_everywhere_identities_have_no_caveat(self):
        r = P.equal_exprs('\\sin^2 x + \\cos^2 x', '1')
        self.assertEqual(r['verdict'], 'yes')
        self.assertNotIn('note', r)

    def test_value_counterexample_still_wins(self):
        # both sides defined at x < 0, values differ: plain 'no'
        r = P.equal_exprs('\\sqrt{x^2}', 'x')
        self.assertEqual(r['verdict'], 'no')
        self.assertEqual(r['method'], 'numeric-oracle')

    def test_spot_check_status_fields(self):
        c = P.numeric_spot_check('\\ln(x^2)', '2 \\ln x')
        self.assertEqual(c['status'], 'domain-differs')
        self.assertEqual(c['defined'], 'lhs')
        self.assertGreater(c['mismatches'], 0)
        self.assertGreater(c['common_samples'], 0)

    def test_merge_prefers_domain_differs(self):
        c = P._merge_checks({'status': 'agree', 'samples': 8},
                            {'status': 'domain-differs', 'mismatches': 3,
                             'common_samples': 5, 'defined': 'rhs',
                             'point': {'x': -1.0}})
        self.assertEqual(c['status'], 'domain-differs')


class TestSubtermRelax(unittest.TestCase):
    # gen 9: subterm-rewrite surgery results relax like root results

    def test_factored_subterm_prints_clean(self):
        r = P.rewrite('(x^{2}+4)(x^{2}-4)', 'diff_squares')
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(r['result'], '(x^{2}+4)(x+2)(x-2)')

    def test_rewrite_inside_function_argument(self):
        r = P.rewrite('\\sin(x^{2}-4)', 'diff_squares')
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(r['result'], '\\sin ((x+2)(x-2))')

    def test_verbatim_powers_keep_single_braces(self):
        s, n = P.parse_latex('x^{2}+x^{12}')
        self.assertEqual(P.write_latex(s, n), 'x^{2}+x^{12}')

    def test_root_and_equation_outputs_unchanged(self):
        self.assertEqual(P.rewrite('x^2 - 4', 'diff_squares')['result'],
                         '(x+2)(x-2)')
        self.assertEqual(P.rewrite('x^4 - 16 = 0', 'diff_squares')['result'],
                         '(x^{2}+4)(x^{2}-4)=0')


# classic non-commuting pair: AB = diag(1,0), BA = diag(0,1)
MAT_A = '\\pmatrix{0 & 1 \\cr 0 & 0}'
MAT_B = '\\pmatrix{0 & 0 \\cr 1 & 0}'


class TestMatrixParsing(unittest.TestCase):
    # gen 10: \begin{pmatrix}/\begin{matrix} normalize to the grammar's
    # plain-TeX commands and round-trip through the writer

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

    def test_plain_tex_form_roundtrips(self):
        s, n = P.parse_latex(MAT_A)
        out = P.write_latex(s, n)
        self.assertEqual(out, MAT_A)
        s2, n2 = P.parse_latex(out)
        self.assertEqual(P.write_latex(s2, n2), out)

    def test_substitute_into_cells(self):
        r = P.substitute('\\pmatrix{x & 2x \\cr 1 & x}', 'x', '3')
        self.assertTrue(r['ok'])
        self.assertEqual(r['check']['status'], 'agree')
        self.assertIn('\\pmatrix', r['result'])


class TestNoncommutativeAtoms(unittest.TestCase):
    # gen 10: products with >= 2 matrix-valued factors atomize as ONE
    # ordered word, so commutative polyrat can never prove AB = BA

    def test_commutator_does_not_vanish(self):
        r = P.expand(f'{MAT_A}{MAT_B} - {MAT_B}{MAT_A}')
        self.assertTrue(r['ok'])
        self.assertNotEqual(r['result'], '0')
        self.assertEqual(r['check']['status'], 'agree')

    def test_same_word_still_collects(self):
        r = P.expand(f'{MAT_A}{MAT_B} + {MAT_A}{MAT_B}')
        self.assertTrue(r['result'].startswith('2'))
        self.assertEqual(r['check']['status'], 'agree')

    def test_scalars_commute_out(self):
        r = P.expand(f'{MAT_A} x - x {MAT_A}')
        self.assertEqual(r['result'], '0')
        self.assertEqual(r['check']['status'], 'agree')

    def test_matrix_sum_power_stays_opaque(self):
        r = P.expand(f'({MAT_A} + {MAT_B})^2')
        self.assertIn('^{2}', r['result'])   # no fabricated 2AB expansion
        self.assertEqual(r['check']['status'], 'agree')

    def test_single_matrix_power_collects(self):
        r = P.expand(f'{MAT_A}^2 + {MAT_A}^2')
        self.assertTrue(r['result'].startswith('2'))
        self.assertEqual(r['check']['status'], 'agree')

    def test_scalar_division_works(self):
        r = P.expand(f'\\frac{{{MAT_A}}}{{2}} + \\frac{{{MAT_A}}}{{2}}')
        self.assertEqual(r['result'], MAT_A)
        self.assertEqual(r['check']['status'], 'agree')

    def test_division_by_matrix_stays_opaque(self):
        r = P.expand(f'\\frac{{1}}{{{MAT_A}}} {MAT_A}')
        self.assertNotEqual(r['result'], '1')   # A^{-1}A is not scalar 1

    def test_vec_scalar_collect(self):
        r = P.expand('2\\vec v + 3\\vec v')
        self.assertEqual(r['result'], '5 \\vec v')

    def test_vec_words_do_not_commute(self):
        r = P.expand('\\vec u \\vec v - \\vec v \\vec u')
        self.assertNotEqual(r['result'], '0')

    def test_collect_matrix_coefficients(self):
        r = P.collect(f'x {MAT_A} + x {MAT_B}', 'x')
        self.assertEqual(r['check']['status'], 'agree')
        self.assertTrue(r['result'].endswith('x'))

    def test_evaluate_refuses_matrices(self):
        r = P.evaluate(f'{MAT_A} + {MAT_A}')
        self.assertFalse(r['ok'])
        self.assertIn('matrix', r['error'])


class TestMatrixOracle(unittest.TestCase):
    # gen 10: the oracle evaluates literal matrices with ORDERED
    # multiplication — it can disprove commutation, not just skip

    def test_equal_disproves_commutation(self):
        r = P.equal_exprs(f'{MAT_A}{MAT_B}', f'{MAT_B}{MAT_A}')
        self.assertEqual(r['verdict'], 'no')
        self.assertEqual(r['lhs'], [[1.0, 0.0], [0.0, 0.0]])
        self.assertEqual(r['rhs'], [[0.0, 0.0], [0.0, 1.0]])

    def test_equal_confirms_product(self):
        r = P.equal_exprs(f'{MAT_A}{MAT_B}', '\\pmatrix{1 & 0 \\cr 0 & 0}')
        self.assertEqual(r['verdict'], 'yes')

    def test_equal_same_word_canonical(self):
        r = P.equal_exprs(f'{MAT_A}{MAT_B}', f'({MAT_A})({MAT_B})')
        self.assertEqual(r['verdict'], 'yes')

    def test_scaling_vs_addition(self):
        r = P.equal_exprs(f'2{MAT_A}', f'{MAT_A}+{MAT_A}')
        self.assertEqual(r['verdict'], 'yes')

    def test_shape_mismatch_is_no(self):
        r = P.equal_exprs('\\pmatrix{1 & 2 \\cr 3 & 4}', '\\pmatrix{1 & 2}')
        self.assertEqual(r['verdict'], 'no')

    def test_cancelled_matrix_agrees_with_scalar_zero(self):
        c = P.numeric_spot_check(f'{MAT_A} - {MAT_A}', '0')
        self.assertEqual(c['status'], 'agree')

    def test_symbolic_vectors_stay_skipped(self):
        r = P.equal_exprs('\\vec u \\vec v', '\\vec v \\vec u')
        self.assertEqual(r['verdict'], 'unknown')


class TestLedger(unittest.TestCase):
    def test_record_replay(self):
        path = os.path.join(tempfile.mkdtemp(), 'session.json')
        ledger = Ledger(path)
        r1 = P.apply_both_sides('2x + 3 = 7', '-', '3')
        ledger.record(r1)
        r2 = P.expand(r1['result'])
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
        ledger.record(P.apply_both_sides('x y = 1', '/', 'y'))
        ledger.save()
        self.assertEqual(len(Ledger(path).assumptions), 1)

    def test_branch_detection(self):
        path = os.path.join(tempfile.mkdtemp(), 'session.json')
        ledger = Ledger(path)
        ledger.record(P.expand('(x+1)^2'))
        ledger.record(P.expand('(y+1)^2'))
        self.assertFalse(ledger.steps[1]['continues'])


class TestOracleCatchesLies(unittest.TestCase):
    def test_disagree_on_wrong_identity(self):
        c = P.numeric_spot_check('(x+1)^2', 'x^2 + 1')
        self.assertEqual(c['status'], 'disagree')

    def test_respects_assumptions(self):
        c = P.numeric_spot_check('\\frac{x y}{y}', 'x',
                                 assumptions=[{'text': 'y \\ne 0',
                                               'nonzero': 'y'}])
        self.assertEqual(c['status'], 'agree')


if __name__ == '__main__':
    unittest.main()

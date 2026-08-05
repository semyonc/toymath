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


class TestPoweredHeadMatching(unittest.TestCase):
    # gen 58: the atomizer and the oracle already read \sin^2 x and
    # (\sin x)^2 as one object; the structural matcher used to see two.
    # \sin^2 x parses as P_LIST[INDEX(\sin, 2), x], so a pattern a^2 bound
    # a to the bare function name and stranded the argument.

    def test_powered_head_spelling_matches(self):
        r = Core.rewrite('\\sin^2 x - \\cos^2 x', 'diff_squares')
        self.assertTrue(r['ok'])
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(r['result'], '( \\sin x+ \\cos x)( \\sin x- \\cos x)')

    def test_both_spellings_give_the_same_result(self):
        for lemma, powered, parens in (
                ('diff_squares', '\\sin^2 x - \\cos^2 x',
                 '(\\sin x)^2 - (\\cos x)^2'),
                ('diff_cubes', '\\tan^3 x - 8', '(\\tan x)^3 - 8'),
                ('sum_cubes', '\\sin^3 x + \\cos^3 x',
                 '(\\sin x)^3 + (\\cos x)^3')):
            a, b = Core.rewrite(powered, lemma), Core.rewrite(parens, lemma)
            self.assertTrue(a['ok'], lemma)
            self.assertEqual(a['result'], b['result'], lemma)

    def test_compound_argument_matches(self):
        r = Core.rewrite('\\sin^2 (2x) - \\cos^2 (2x)', 'diff_squares')
        self.assertTrue(r['ok'])
        self.assertEqual(r['check']['status'], 'agree')

    def test_expand_output_is_matchable(self):
        # expand normalizes into the powered-head spelling, so its own
        # output has to be consumable by rewrite
        e = Core.expand('(\\sin x)^2 - (\\cos x)^2')
        self.assertEqual(e['result'], '\\sin^{2}x- \\cos^{2}x')
        r = Core.rewrite(e['result'], 'diff_squares')
        self.assertTrue(r['ok'])
        self.assertEqual(r['check']['status'], 'agree')

    def test_inverse_power_is_not_normalized(self):
        # \sin^{-1} must keep its own reading: no a^2 - b^2 match here
        r = Core.rewrite('\\sin^{-1} x - \\cos^2 x', 'diff_squares')
        self.assertFalse(r['ok'])
        self.assertIn('does not match', r['error'])

    def test_plain_variable_rewrites_unchanged(self):
        self.assertEqual(Core.rewrite('x^2 - y^2', 'diff_squares')['result'],
                         '(x+y)(x-y)')
        self.assertEqual(Core.rewrite('x^2 - 9', 'diff_squares')['result'],
                         '(x+3)(x-3)')

    def test_lemma_pattern_is_normalized_too(self):
        # gen 59: normalizing only the expression side left a lemma written
        # in the standard spelling matching NOTHING, which would have
        # shipped a registry whose own headline lemma silently never fired
        Core.register_lemma('g59_pyth', '\\sin^2 t', '1 - \\cos^2 t',
                            ['t'], 'pythagorean')
        try:
            for expr in ('\\sin^2 x', '(\\sin x)^2', '\\sin^2 (2x)'):
                r = Core.rewrite(expr, 'g59_pyth')
                self.assertTrue(r['ok'], expr)
                self.assertEqual(r['check']['status'], 'agree', expr)
        finally:
            Core.LEMMAS.pop('g59_pyth', None)

    def test_both_lemma_spellings_behave_alike(self):
        Core.register_lemma('g59_a', '\\sin^2 t', '1 - \\cos^2 t', ['t'])
        Core.register_lemma('g59_b', '(\\sin t)^2', '1 - (\\cos t)^2', ['t'])
        try:
            a = Core.rewrite('\\sin^2 x', 'g59_a')
            b = Core.rewrite('\\sin^2 x', 'g59_b')
            self.assertEqual(a['result'], b['result'])
        finally:
            Core.LEMMAS.pop('g59_a', None)
            Core.LEMMAS.pop('g59_b', None)


class TestProductUnitMatching(unittest.TestCase):
    # gen 62: gen 58 grouped POWERED heads at the matching boundary and left
    # bare product units open, so `2 \sin x \cos x` did not match `2ab` while
    # `2(\sin x)(\cos x)` did -- the backward direction of every trig lemma,
    # which is the direction that collapses. Spans come from the oracle's own
    # _func_arg_span, so a grouping cannot disagree with the numeric leg.

    def test_bare_product_of_applications_matches(self):
        r = Core.rewrite('2 \\sin x \\cos x', 'sin_double', 'backward')
        self.assertTrue(r['ok'])
        self.assertEqual(r['result'], '\\sin 2x')
        self.assertEqual(r['check']['status'], 'agree')

    def test_both_product_spellings_give_one_result(self):
        bare = Core.rewrite('2 \\sin x \\cos x', 'sin_double', 'backward')
        grouped = Core.rewrite('2 (\\sin x)(\\cos x)', 'sin_double',
                               'backward')
        self.assertTrue(bare['ok'])
        self.assertEqual(bare['result'], grouped['result'])

    def test_matches_a_product_inside_a_sum(self):
        r = Core.rewrite('y + 2 \\sin x \\cos x', 'sin_double', 'backward')
        self.assertTrue(r['ok'])
        self.assertEqual(r['result'], 'y+ \\sin 2x')

    def test_argument_binding_is_preserved_not_flattened(self):
        # `2 \sin x \cos x y` reads as 2*sin(x)*cos(x*y) -- the same span rule
        # the oracle uses -- so it genuinely is NOT the double-angle shape and
        # must refuse rather than flatten into a match
        r = Core.rewrite('2 \\sin x \\cos x y', 'sin_double', 'backward')
        self.assertFalse(r['ok'])
        self.assertIn('does not match', r['error'])

    def test_whole_product_application_stays_matchable(self):
        # a product that IS one application must not be wrapped: a ()-group
        # matches no pattern at all, so wrapping it would hide the root
        r = Core.rewrite('\\sin 2x', 'sin_double')
        self.assertTrue(r['ok'])
        self.assertEqual(r['result'], '2 \\sin x \\cos x')

    def test_backward_square_lemmas_close_over_bare_trig(self):
        # the round trip gen 58 opened and could only half close
        for lemma, expr in (
                ('square_of_diff',
                 '\\sin^2 x - 2\\sin x \\cos x + \\cos^2 x'),
                ('square_of_sum',
                 '\\sin^2 x + 2\\sin x \\cos x + \\cos^2 x')):
            r = Core.rewrite(expr, lemma, 'backward')
            self.assertTrue(r['ok'], lemma)
            self.assertEqual(r['check']['status'], 'agree', lemma)

    def test_plain_products_are_untouched(self):
        self.assertEqual(Core.rewrite('x^2 - y^2', 'diff_squares')['result'],
                         '(x+y)(x-y)')


class TestSubstitutedGroupIsSelfDelimiting(unittest.TestCase):
    # gen 62: Substitutor wrapped every non-symbol replacement in (), even one
    # that already was a ()-group, and the relax pass cannot reach inside an
    # INDEX base -- so the doubled layer survived into user-visible results.

    def test_substitute_does_not_double_parens(self):
        r = Core.substitute('a^2 + 1', 'a', '(x+1)')
        self.assertTrue(r['ok'])
        self.assertEqual(r['result'], '(x+1)^{2}+1')

    def test_lemma_destination_keeps_one_layer(self):
        r = Core.rewrite('(x+1)^3 + y^3', 'sum_cubes')
        self.assertTrue(r['ok'])
        self.assertEqual(r['result'],
                         '((x+1)+y)((x+1)^{2}-(x+1)y+y^{2})')

    def test_function_binding_in_an_index_base(self):
        r = Core.rewrite('\\sin^2 x - 2\\sin x \\cos x + \\cos^2 x',
                         'square_of_diff', 'backward')
        self.assertTrue(r['ok'])
        self.assertEqual(r['result'], '(( \\sin x)-( \\cos x))^{2}')


class TestTrigLemmaCatalog(unittest.TestCase):
    # gen 62: the capability behind four recorded agent asks. A lemma is an
    # EXACT structural rewrite where the same move through rewrite_as is
    # numerically sampled, so the catalog is a trust improvement, not just a
    # convenience. Registering it needed no loader and no always-on budget.

    TRIG = ('pythagorean', 'sin_squared', 'cos_squared', 'sin_double',
            'cos_double')

    def test_registered_and_discoverable(self):
        names = {l['name'] for l in Core.list_lemmas()['lemmas']}
        for name in self.TRIG:
            self.assertIn(name, names)
            self.assertTrue(Core.LEMMAS[name].description)

    def test_every_trig_lemma_applies_forward(self):
        for name in self.TRIG:
            src = Core.LEMMAS[name].lhs.replace('a', 'x')
            r = Core.rewrite(src, name)
            self.assertTrue(r['ok'], name)
            self.assertEqual(r['check']['status'], 'agree', name)

    def test_solved_forms_apply_backward(self):
        # pythagorean is excluded on purpose: its right side is the bare
        # literal 1, so backward leaves the parameter unbound
        for name in ('sin_squared', 'cos_squared', 'sin_double',
                     'cos_double'):
            dst = Core.LEMMAS[name].rhs.replace('a', 'x')
            r = Core.rewrite(dst, name, 'backward')
            self.assertTrue(r['ok'], name)
            self.assertEqual(r['check']['status'], 'agree', name)

    def test_pythagorean_backward_refuses_cleanly(self):
        r = Core.rewrite('1', 'pythagorean', 'backward')
        self.assertFalse(r['ok'])
        self.assertIn('unbound', r['error'])

    def test_gen57_critical_path_lemma(self):
        # the int1 failure needed exactly this rewrite on its critical path
        r = Core.rewrite('\\sin^2 t \\cos t', 'sin_squared')
        self.assertTrue(r['ok'])
        self.assertEqual(r['check']['status'], 'agree')
        self.assertIn('\\cos^{2}t', r['result'])

    def test_compound_arguments_bind(self):
        r = Core.rewrite('\\sin^2 (3y)', 'sin_squared')
        self.assertTrue(r['ok'])
        self.assertEqual(r['check']['status'], 'agree')

    def test_two_term_pattern_needs_the_whole_sum(self):
        # measured limit, not a bug: matching is structural, so a two-term
        # pattern does not select two terms out of a longer sum. The granular
        # route is sin_squared forward, then expand.
        self.assertTrue(Core.rewrite('\\sin^2 x + \\cos^2 x',
                                     'pythagorean')['ok'])
        self.assertFalse(Core.rewrite('y + \\sin^2 x + \\cos^2 x',
                                      'pythagorean')['ok'])
        step = Core.rewrite('y + \\sin^2 x + \\cos^2 x', 'sin_squared')
        self.assertTrue(step['ok'])
        self.assertEqual(Core.expand(step['result'])['result'], 'y+1')


class TestRewriteAs(unittest.TestCase):
    # gen 59: congruence with an agent-supplied witness. Reaches identities
    # with no registered lemma and spellings the structural matcher cannot
    # bind; equal? is the check, and its trust level is reported, not hidden.

    def test_reaches_an_identity_with_no_lemma(self):
        r = Core.rewrite_as('\\sin(2x)', '2\\sin x \\cos x')
        self.assertTrue(r['ok'])
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(r['result'], '2\\sin x \\cos x')

    def test_reaches_what_the_matcher_cannot_bind(self):
        # bare product units: structural rewrite refuses this shape
        r = Core.rewrite_as(
            '(\\sin x)^2 - 2\\sin x \\cos x + (\\cos x)^2',
            '( \\sin x - \\cos x)^2')
        self.assertTrue(r['ok'])
        self.assertEqual(r['check']['status'], 'agree')

    def test_trust_level_is_named_not_hidden(self):
        exact = Core.rewrite_as('(x+1)^2', 'x^2+2x+1')
        probabilistic = Core.rewrite_as('\\sin^2 x + \\cos^2 x', '1')
        self.assertIn('canonical', exact['check']['method'])
        self.assertIn('numeric-oracle', probabilistic['check']['method'])
        self.assertIn('samples', probabilistic['check'])

    def test_refuses_a_false_proposal_with_a_counterexample(self):
        r = Core.rewrite_as('(\\sin x - \\cos x)^2', '\\sin^2 x - \\cos^2 x')
        self.assertFalse(r['ok'])
        self.assertIn('not mechanically equal', r['error'])
        self.assertIn('counterexample', r['error'])

    def test_refuses_an_unknown_verdict(self):
        r = Core.rewrite_as('f(x)', 'g(x)')
        self.assertFalse(r['ok'])
        self.assertIn('not mechanically equal', r['error'])

    def test_refuses_a_no_op(self):
        r = Core.rewrite_as('x+1', 'x+1')
        self.assertFalse(r['ok'])
        self.assertIn('changes nothing', r['error'])

    def test_refuses_an_unparseable_proposal(self):
        r = Core.rewrite_as('x+1', '\\frac{')
        self.assertFalse(r['ok'])

    def test_accepts_an_equal_relation(self):
        r = Core.rewrite_as('x+1 = 2', '1+x = 2')
        self.assertTrue(r['ok'])
        self.assertEqual(r['check']['status'], 'agree')


class TestMatchCoefficients(unittest.TestCase):
    # gen 61: the producer for system_assemble. A live int! run spent 55% of
    # its turn budget hand-guessing partial-fraction coefficients, failing
    # `integrate_rewrite ... verdict: no` nine times, because no move derived
    # them. This derives them.

    def test_matches_like_powers(self):
        r = Equations.match_coefficients('A x^2 + B x + C = 2x^2 + 5', 'x')
        self.assertTrue(r['ok'])
        self.assertEqual(r['result'], 'A = 2, B = 0, C = 5')
        self.assertEqual(r['check']['status'], 'agree')

    def test_result_is_a_relation_system(self):
        # the gen-35 comma container, which is what system_assemble consumes
        r = Equations.match_coefficients('A x + B = 3x - 1', 'x')
        sym, notation = P.parse_latex(r['result'])
        self.assertIsNotNone(notation.getf(sym, Notation.C_LIST))

    def test_partial_fraction_shape(self):
        r = Equations.match_coefficients('x = A(x+1) + B(x-1)', 'x')
        self.assertTrue(r['ok'])
        self.assertEqual(r['result'], '1 = A+B, 0 = A-B')

    def test_impossible_ansatz_is_reported_not_hidden(self):
        # a degree that cannot balance must surface as 1 = 0, telling the
        # agent the ansatz is wrong rather than the arithmetic
        r = Equations.match_coefficients('x^2 = A(x+1) + B(x-1)', 'x')
        self.assertTrue(r['ok'])
        self.assertIn('1 = 0', r['result'])

    def test_symbolic_on_both_sides(self):
        r = Equations.match_coefficients('A x^2 + B = C x^2 + D', 'x')
        self.assertEqual(r['result'], 'A = C, B = D')
        self.assertEqual(r['check']['status'], 'agree')

    def test_check_is_independent_of_the_symbolic_extraction(self):
        # corrupting the symbolic buckets must be caught by the oracle leg
        original = Equations._coefficient_buckets

        def corrupt(poly, var):
            buckets = original(poly, var)
            if buckets:
                buckets[sorted(buckets)[0]] = Poly.const(999)
            return buckets

        Equations._coefficient_buckets = corrupt
        try:
            r = Equations.match_coefficients('A x^2 + B x + C = 2x^2 + 5',
                                             'x')
            self.assertEqual(r['check']['status'], 'disagree')
        finally:
            Equations._coefficient_buckets = original

    def test_refusals(self):
        for expr, var, msg in (
                ('A x + B', 'x', 'equality'),
                ('A x \\lt B x', 'x', 'only be matched'),
                ('A = B', 'x', 'does not occur'),
                ('A x + B = A x + B', 'x', 'nothing to equate'),
                ('\\frac{1}{x} = A', 'x', 'must be polynomials')):
            r = Equations.match_coefficients(expr, var)
            self.assertFalse(r['ok'], expr)
            self.assertIn(msg, r['error'], expr)


class TestDomainNarrowingAssumptions(unittest.TestCase):
    # gen 60: cancelling a factor drops the points where it vanished. The
    # canonical leg decides equality as rational functions and the numeric
    # legs CANNOT see the loss (a removable singularity is measure-zero, so
    # sampling never lands on it), so it has to be stated symbolically.

    def nonzero(self, rec):
        return [a.get('nonzero') for a in rec.get('assumptions') or []]

    def test_sampling_cannot_see_it(self):
        # the premise of the whole feature: both spot-checks say 'agree'
        for a, b in (('\\frac{x^2-1}{x-1}', 'x+1'),
                     ('\\frac{(x+y)(x-z)}{(x+y)(x-w)}',
                      '\\frac{x-z}{x-w}')):
            self.assertEqual(P.numeric_spot_check(a, b)['status'], 'agree')

    def test_expand_records_a_cancelled_factor(self):
        self.assertEqual(self.nonzero(Core.expand('\\frac{x^2-1}{x-1}')),
                         ['x-1'])
        self.assertEqual(self.nonzero(Core.expand('\\frac{x}{x}')), ['x'])

    def test_expand_records_a_cancelled_opaque_atom(self):
        # the trig case simplify! actually meets; needs one shared atom
        # store across both denominators or the names never line up
        r = Core.expand('\\frac{\\sin x \\cos x}{\\sin x}')
        self.assertEqual(r['result'], '\\cos x')
        self.assertEqual(self.nonzero(r), ['\\sin x'])

    def test_rewrite_as_records_a_cancelled_factor(self):
        self.assertEqual(
            self.nonzero(Core.rewrite_as('\\frac{x^2-1}{x-1}', 'x+1')),
            ['x-1'])
        self.assertEqual(
            self.nonzero(Core.rewrite_as(
                '\\frac{(x+y)(x-z)}{(x+y)(x-w)}', '\\frac{x-z}{x-w}')),
            ['x+y'])

    def test_a_sum_of_fractions_reports_every_lost_condition(self):
        # found by the FIRST live simplify! run: a top-level-only scan saw
        # no fraction here and reported nothing, losing both conditions
        r = Core.expand(
            '\\frac{\\sin x \\cos x}{\\sin x} + \\frac{x^2-1}{x-1}')
        self.assertEqual(self.nonzero(r), ['\\sin x', 'x-1'])

    def test_loss_is_attributed_to_the_written_denominators(self):
        # a partial factor reports just the cancelled part, not the whole
        # denominator it came from
        self.assertEqual(
            self.nonzero(Core.rewrite_as(
                '\\frac{(x+y)(x-z)}{(x+y)(x-w)}', '\\frac{x-z}{x-w}')),
            ['x+y'])

    def test_it_is_symmetric(self):
        # introducing a denominator excludes points just as removing one does
        self.assertEqual(
            self.nonzero(Core.rewrite_as('x+1', '\\frac{x^2-1}{x-1}')),
            ['x-1'])

    def test_assumption_uses_the_established_shape(self):
        a = (Core.expand('\\frac{x^2-1}{x-1}')['assumptions'] or [])[0]
        self.assertEqual(a['text'], 'x-1 \\ne 0')
        self.assertEqual(a['nonzero'], 'x-1')

    def test_no_assumption_when_nothing_is_lost(self):
        for expr in ('(x+1)^2', '\\frac{1}{x+1}', '2\\sin x + 3\\sin x',
                     '\\frac{x^2+1}{x-1}', '\\frac{\\cos x}{\\sin x}'):
            self.assertEqual(self.nonzero(Core.expand(expr)), [], expr)
        self.assertEqual(
            self.nonzero(Core.rewrite_as('\\sin^2 x + \\cos^2 x', '1')), [])
        self.assertEqual(
            self.nonzero(Core.rewrite_as('(x+1)^2', 'x^2+2x+1')), [])


class TestCollectNamedAtom(unittest.TestCase):
    # gen 58: the agent may name an opaque atom (\cos x) as the collection
    # variable; it resolves to the atom the atomizer already minted.

    def test_collect_by_named_atom(self):
        r = Core.collect('A\\cos^2 x + B\\cos x + C', '\\cos x')
        self.assertTrue(r['ok'])
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(r['result'], 'A \\cos^{2}x+B \\cos x+C')

    def test_coefficients_actually_combine(self):
        r = Core.collect('A\\cos x + B\\cos^2 x + D\\cos x', '\\cos x')
        self.assertTrue(r['ok'])
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(r['result'], 'B \\cos^{2}x+(A+D) \\cos x')

    def test_either_spelling_names_the_same_atom(self):
        a = Core.collect('A\\cos^2 x + B\\cos x', '\\cos x')
        b = Core.collect('A(\\cos x)^2 + B(\\cos x)', '\\cos x')
        self.assertEqual(a['result'], b['result'])

    def test_relation_sides_collect_by_atom(self):
        r = Core.collect('A\\cos^2 x + B\\cos x = 3\\cos x + 1', '\\cos x')
        self.assertTrue(r['ok'])
        self.assertEqual(r['check']['status'], 'agree')
        self.assertIn('=', r['result'])

    def test_absent_atom_is_refused(self):
        # lookup only: naming an atom the expression lacks must not mint one
        r = Core.collect('A\\cos^2 x + B\\cos x', '\\tan x')
        self.assertFalse(r['ok'])
        self.assertIn('does not occur', r['error'])

    def test_atom_is_still_usable_as_a_coefficient(self):
        r = Core.collect('y\\cos^2 x + 2y\\cos x + y', 'y')
        self.assertTrue(r['ok'])
        self.assertEqual(r['check']['status'], 'agree')
        self.assertEqual(r['result'], '( \\cos^{2}x+2 \\cos x+1)y')


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
        # the rule builder parenthesizes its chain factor for syntax
        # protection; inside the exponent's braces that wrapper is
        # delimiter-redundant, so it does not survive into the artifact
        r = self.check('e^{x^2}')
        self.assertEqual(r['result'], '2xe^{x^{2}}')

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

    def test_repeated_param_mismatch_refused(self):
        # backward square_of_sum binds 'a' twice; the false match used
        # to win with the last binding and return a disagree-checked
        # record instead of refusing
        r = Core.rewrite('x^2 + 2 x z + y^2', 'square_of_sum',
                      direction='backward')
        self.assertFalse(r['ok'])
        self.assertIn('does not match', r['error'])

    def test_repeated_param_accepts_compound_binding(self):
        # compound bindings are distinct DAG nodes per occurrence, so
        # repeated-param consistency must compare structurally
        r = Core.rewrite('(x+1)^2 + 2 (x+1) y + y^2', 'square_of_sum',
                      direction='backward')
        self.assertTrue(r['ok'], r.get('error'))
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


class TestPointsAssemble(unittest.TestCase):
    def test_assembles_the_complete_point_collection(self):
        r = Equations.points_assemble('x^3-3x', 'x', r'x=-1 \lor x=1',
                                      ['2', '-2'])
        self.assertTrue(r['ok'], r.get('error'))
        self.assertEqual(r['result'], r'\{(-1,2),(1,-2)\}')
        self.assertEqual(r['input'], r'x=-1 \lor x=1')
        self.assertEqual(r['points'], [{'root': '-1', 'value': '2'},
                                       {'root': '1', 'value': '-2'}])
        self.assertEqual(r['check']['status'], 'agree')

    def test_result_is_a_typed_collection_of_pairs(self):
        r = Equations.points_assemble('x^3-3x', 'x', r'x=-1 \lor x=1',
                                      ['2', '-2'])
        sym, notation = P.parse_latex(r['result'])
        collection = notation.getf(sym, Notation.COLLECTION)
        self.assertIsNotNone(collection)
        self.assertEqual(len(collection.args), 2)
        for item in collection.args:
            self.assertIsNotNone(notation.getf(item, Notation.PAIR))

    def test_single_repeated_root_and_fraction_roots(self):
        repeated = Equations.points_assemble('x^2-6x+9', 'x', 'x=3', ['0'])
        self.assertTrue(repeated['ok'], repeated.get('error'))
        self.assertEqual(repeated['result'], r'\{(3,0)\}')
        fraction = Equations.points_assemble(
            'x^3-3x', 'x', r'x=-\frac{1}{2} \lor x=1',
            [r'\frac{11}{8}', '-2'])
        self.assertTrue(fraction['ok'], fraction.get('error'))
        self.assertEqual(fraction['check']['status'], 'agree')
        self.assertTrue(P.same_expression(
            fraction['result'], r'\{(-\frac{1}{2},\frac{11}{8}),(1,-2)\}'))

    def test_symbolic_values_stay_associated(self):
        r = Equations.points_assemble('ax^2', 'x', r'x=-1 \lor x=2',
                                      ['a', '4a'])
        self.assertTrue(r['ok'], r.get('error'))
        self.assertEqual(r['check']['status'], 'agree')
        self.assertGreater(r['check']['samples'], 2)

    def test_swapped_values_are_refused(self):
        r = Equations.points_assemble('x^3-3x', 'x', r'x=-1 \lor x=1',
                                      ['-2', '2'])
        self.assertFalse(r['ok'])
        self.assertIn('is not the value of', r['error'])

    def test_value_count_must_match_the_recorded_solutions(self):
        r = Equations.points_assemble('x^3-3x', 'x', r'x=-1 \lor x=1', ['2'])
        self.assertFalse(r['ok'])
        self.assertIn('one recorded value step per root', r['error'])

    def test_only_recorded_solution_relations_are_accepted(self):
        inequality = Equations.points_assemble('x^3-3x', 'x', r'x \lt 1',
                                               ['2'])
        self.assertFalse(inequality['ok'])
        self.assertIn('equality', inequality['error'])
        other_var = Equations.points_assemble('x^3-3x', 'x', 'y=1', ['-2'])
        self.assertFalse(other_var['ok'])
        self.assertIn("must name 'x'", other_var['error'])
        plain = Equations.points_assemble('x^3-3x', 'x', '1', ['-2'])
        self.assertFalse(plain['ok'])
        self.assertIn('solution relation', plain['error'])

    def test_repeated_root_spellings_are_refused(self):
        structural = Equations.points_assemble(
            'x^3-3x', 'x', r'x=1 \lor x=1', ['-2', '-2'])
        self.assertFalse(structural['ok'])
        self.assertIn('the same root', structural['error'])
        respelled = Equations.points_assemble(
            'x^3-3x', 'x', r'x=1 \lor x=\frac{2}{2}', ['-2', '-2'])
        self.assertFalse(respelled['ok'])
        self.assertIn('the same root', respelled['error'])
        distinct = Equations.points_assemble(
            'ax^2', 'x', r'x=b \lor x=-b', ['ab^2', 'ab^2'])
        self.assertTrue(distinct['ok'], distinct.get('error'))

    def test_independent_check_sees_the_pairing(self):
        swapped = Equations._points_check(
            'x^3-3x', 'x', [('-1', '-2'), ('1', '2')])
        self.assertEqual(swapped['status'], 'disagree')
        self.assertEqual(swapped['root'], '-1')
        aligned = Equations._points_check(
            'x^3-3x', 'x', [('-1', '2'), ('1', '-2')])
        self.assertEqual(aligned['status'], 'agree')

    def test_undefined_value_at_a_root_is_a_domain_signal(self):
        check = Equations._points_check('\\frac{1}{x}', 'x', [('0', '0')])
        self.assertEqual(check['status'], 'domain-differs')
        self.assertEqual(check['root'], '0')


ANSATZ = r'\frac{1}{x^2-1} = \frac{A}{x-1}+\frac{B}{x+1}'


class TestSystemAssemble(unittest.TestCase):
    def test_assembles_the_several_part_answer(self):
        r = Equations.system_assemble(
            ANSATZ, [r'A = \frac{1}{2}', r'B = -\frac{1}{2}'])
        self.assertTrue(r['ok'], r.get('error'))
        self.assertEqual(r['input'], ANSATZ)
        self.assertEqual(r['unknowns'],
                         [{'unknown': 'A', 'value': r'\frac {1} {2}'},
                          {'unknown': 'B', 'value': r'- \frac {1} {2}'}])
        self.assertEqual(r['check']['status'], 'agree')
        self.assertTrue(P.same_expression(
            r['result'], r'A=\frac{1}{2},B=-\frac{1}{2}'))

    def test_result_is_a_system_of_equalities(self):
        r = Equations.system_assemble('x+y=3, x-y=1', ['x=2', 'y=1'])
        self.assertTrue(r['ok'], r.get('error'))
        self.assertEqual(r['result'], 'x=2,y=1')
        sym, notation = P.parse_latex(r['result'])
        head = notation.get(sym)
        self.assertEqual(head.sym.name, Notation.C_LIST.name)
        for item in head.args:
            self.assertIsNotNone(notation.getf(item, Notation.COMP))

    def test_symbolic_coefficients_stay_associated(self):
        r = Equations.system_assemble(
            'a = A + B, b = A - B',
            [r'A = \frac{a+b}{2}', r'B = \frac{a-b}{2}'])
        self.assertTrue(r['ok'], r.get('error'))
        self.assertEqual(r['check']['status'], 'agree')
        self.assertGreater(r['check']['samples'], 2)

    def test_swapped_values_are_refused(self):
        r = Equations.system_assemble(
            ANSATZ, [r'A = -\frac{1}{2}', r'B = \frac{1}{2}'])
        self.assertFalse(r['ok'])
        self.assertIn('do not satisfy', r['error'])

    def test_a_missing_unknown_names_what_is_still_free(self):
        r = Equations.system_assemble(ANSATZ, [r'A = \frac{1}{2}'])
        self.assertFalse(r['ok'])
        self.assertIn('still free there', r['error'])
        self.assertIn('B', r['error'])

    def test_only_resolved_assignments_are_accepted(self):
        unresolved = Equations.system_assemble(
            ANSATZ, ['A = B', r'B = -\frac{1}{2}'])
        self.assertFalse(unresolved['ok'])
        self.assertIn("still contains 'B'", unresolved['error'])
        inequality = Equations.system_assemble(
            ANSATZ, [r'A > \frac{1}{2}', r'B = -\frac{1}{2}'])
        self.assertFalse(inequality['ok'])
        self.assertIn('is not an equality', inequality['error'])
        compound = Equations.system_assemble(
            ANSATZ, [r'2A = 1', r'B = -\frac{1}{2}'])
        self.assertFalse(compound['ok'])
        self.assertIn('plain unknown on the left', compound['error'])
        twice = Equations.system_assemble(
            ANSATZ, [r'A = \frac{1}{2}', r'A = \frac{1}{2}'])
        self.assertFalse(twice['ok'])
        self.assertIn('assigned twice', twice['error'])

    def test_the_answer_is_never_its_own_target(self):
        # measured live: an agent passed the assignment list as the target,
        # which substitutes to "value = value" and certifies nothing
        r = Equations.system_assemble(
            r'A = \frac{1}{2}, B = -\frac{1}{2}',
            [r'A = \frac{1}{2}', r'B = -\frac{1}{2}'])
        self.assertFalse(r['ok'])
        self.assertIn('states the value of every unknown', r['error'])
        flipped = Equations.system_assemble(
            r'\frac{1}{2} = A', [r'A = \frac{1}{2}'])
        self.assertFalse(flipped['ok'])
        self.assertIn('never the answer itself', flipped['error'])
        defining = Equations.system_assemble('2A = 1', [r'A = \frac{1}{2}'])
        self.assertTrue(defining['ok'], defining.get('error'))
        self.assertEqual(defining['check']['status'], 'agree')

    def test_a_respelled_answer_is_still_the_answer(self):
        # the live int1 run: target in factored spelling, recorded values
        # expanded, so a structural comparison saw two different objects
        # while the substitution check passed trivially
        target = (r'A = \frac{-b}{(n-1)(a^2-b^2)}, '
                  r'B = \frac{a(2n-3)}{(n-1)(a^2-b^2)}, '
                  r'C = \frac{-(n-2)}{(n-1)(a^2-b^2)}')
        values = [r'A = \frac {-b} {a^{2}n-b^{2}n-a^{2}+b^{2}}',
                  r'B = \frac {2an-3a} {a^{2}n-b^{2}n-a^{2}+b^{2}}',
                  r'C = \frac {-n+2} {a^{2}n-b^{2}n-a^{2}+b^{2}}']
        self.assertFalse(P.same_expression(
            r'\frac{-b}{(n-1)(a^2-b^2)}', values[0].split('=', 1)[1]))
        r = Equations.system_assemble(target, values)
        self.assertFalse(r['ok'])
        self.assertIn('every unknown (A, B, C)', r['error'])

    def test_a_target_pinning_only_some_unknowns_is_real_evidence(self):
        # a system genuinely containing the row `x = 3` still constrains y
        r = Equations.system_assemble('x = 3, x+y = 5', ['x=3', 'y=2'])
        self.assertTrue(r['ok'], r.get('error'))
        self.assertEqual(r['result'], 'x=3,y=2')
        self.assertEqual(r['check']['status'], 'agree')
        constraint = Equations.system_assemble(
            'A = 2B, A+B = 3', ['A=2', 'B=1'])
        self.assertTrue(constraint['ok'], constraint.get('error'))

    def test_unknown_must_be_one_the_target_names(self):
        r = Equations.system_assemble(
            ANSATZ, [r'A = \frac{1}{2}', r'B = -\frac{1}{2}', 'C = 1'])
        self.assertFalse(r['ok'])
        self.assertIn("'C' does not occur", r['error'])

    def test_only_equality_targets_are_verifiable(self):
        r = Equations.system_assemble('x+y > 3', ['x=2', 'y=1'])
        self.assertFalse(r['ok'])
        self.assertIn('verifies equalities', r['error'])
        plain = Equations.system_assemble('x+y', ['x=2', 'y=1'])
        self.assertFalse(plain['ok'])
        self.assertIn('must be an equality', plain['error'])

    def test_independent_check_sees_the_association(self):
        relations = ['a = A + B', 'b = A - B']
        aligned = Equations._system_check(
            relations, [{'unknown': 'A', 'value': r'\frac{a+b}{2}'},
                        {'unknown': 'B', 'value': r'\frac{a-b}{2}'}])
        self.assertEqual(aligned['status'], 'agree')
        swapped = Equations._system_check(
            relations, [{'unknown': 'A', 'value': r'\frac{a-b}{2}'},
                        {'unknown': 'B', 'value': r'\frac{a+b}{2}'}])
        self.assertEqual(swapped['status'], 'disagree')
        self.assertEqual(swapped['relation'], 'b = A - B')

    def test_undefined_assignment_is_a_domain_signal(self):
        check = Equations._system_check(
            ['A = 1'], [{'unknown': 'A', 'value': r'\frac{1}{0}'}])
        self.assertEqual(check['status'], 'domain-differs')

    def test_assignment_pairs_is_the_shared_reader(self):
        pairs = Equations.assignment_pairs(['x=2', 'y=1'])
        self.assertEqual(pairs, [{'unknown': 'x', 'value': '2'},
                                 {'unknown': 'y', 'value': '1'}])
        with self.assertRaises(P.PrimitiveError):
            Equations.assignment_pairs([])


class TestNumericUnionCheck(unittest.TestCase):
    TARGET = r'\frac{1}{x} \lt 2'

    def test_true_union_agrees_with_both_sides_exercised(self):
        check = P.numeric_union_check(self.TARGET,
                                      [r'x \lt 0', r'x \gt \frac{1}{2}'])
        self.assertEqual(check['status'], 'agree')
        self.assertGreater(check['holding_points'], 0)
        self.assertLess(check['holding_points'], check['samples'])

    def test_raw_endpoint_trap_disagrees_inside_the_gap(self):
        # the case x<0 derived the endpoint x<1/2 under its hypothesis;
        # assembling that raw endpoint claims (0, 1/2) as solutions
        check = P.numeric_union_check(
            self.TARGET, [r'x \gt \frac{1}{2}', r'x \lt \frac{1}{2}'])
        self.assertEqual(check['status'], 'disagree')
        witness = check['point']['x']
        self.assertGreater(witness, 0)
        self.assertLess(witness, 0.5)
        self.assertEqual(check['holds'],
                         {'target': False, 'union': True})

    def test_too_narrow_union_disagrees(self):
        check = P.numeric_union_check(self.TARGET, [r'x \gt \frac{1}{2}'])
        self.assertEqual(check['status'], 'disagree')
        self.assertEqual(check['holds'],
                         {'target': True, 'union': False})

    def test_one_sided_sample_is_skipped_not_agreed(self):
        # a tautological target never fails, so coverage of the union is
        # never exercised in the failing direction
        check = P.numeric_union_check(r'x^2 \gt -1',
                                      [r'x \lt 1', r'x \gt 0'])
        self.assertEqual(check['status'], 'skipped')
        self.assertIn('one-sided', check['reason'])

    def test_non_relation_disjunct_is_skipped(self):
        check = P.numeric_union_check(self.TARGET, [r'x^2'])
        self.assertEqual(check['status'], 'skipped')

    def test_multi_variable_union_uses_random_sampling(self):
        check = P.numeric_union_check(r'x - y \gt 0',
                                      [r'x \gt y', r'x - y \gt 5'])
        self.assertEqual(check['status'], 'agree')
        self.assertGreater(check['holding_points'], 0)
        self.assertLess(check['holding_points'], check['samples'])

    def test_a_conjunction_disjunct_is_and_of_its_members(self):
        check = P.numeric_union_check(
            r'x^2 \lt 1', [r'-1 \lt x \lt 1'])
        self.assertEqual(check['status'], 'agree')
        self.assertGreater(check['holding_points'], 0)
        self.assertLess(check['holding_points'], check['samples'])

    def test_wrong_conjunction_member_has_a_witness(self):
        check = P.numeric_union_check(
            r'x^2 \lt 1', [r'-1 \lt x \lt 2'])
        self.assertEqual(check['status'], 'disagree')
        self.assertGreaterEqual(check['point']['x'], 1)
        self.assertEqual(check['holds'], {'target': False, 'union': True})

    def test_union_of_two_conjunctions(self):
        check = P.numeric_union_check(
            r'(x^2-1)(x^2-4) \lt 0',
            [r'-2 \lt x \lt -1', r'1 \lt x \lt 2'])
        self.assertEqual(check['status'], 'agree')

    def test_parenthesized_conjunction_disjuncts(self):
        check = P.numeric_union_check(
            r'(x^2-1)(x^2-4) \lt 0',
            [r'(-2 \lt x \lt -1)', r'\left(1 \lt x \lt 2\right)'])
        self.assertEqual(check['status'], 'agree')


class TestCasesAssemble(unittest.TestCase):
    TARGET = r'\frac{1}{x} \lt 2'
    ENDPOINTS = [r'\frac{1}{2} \lt x', r'\frac{1}{2} \gt x']
    HYPOTHESES = [r'x \gt 0', r'x \lt 0']

    def test_assembles_the_union_of_cases(self):
        # disjunct 1 restates its endpoint mirrored; disjunct 2 restates
        # its hypothesis, because the endpoint x<1/2 outgrew the case
        r = Equations.cases_assemble(
            self.TARGET, r'x \gt \frac{1}{2} \lor x \lt 0',
            self.ENDPOINTS, self.HYPOTHESES)
        self.assertTrue(r['ok'], r.get('error'))
        self.assertEqual(r['input'], self.TARGET)
        self.assertEqual(r['check']['status'], 'agree')
        sym, notation = P.parse_latex(r['result'])
        head = notation.get(sym)
        self.assertEqual(head.sym.name, Notation.O_LIST.name)
        for item in head.args:
            self.assertIsNotNone(notation.getf(item, Notation.COMP))

    def test_raw_endpoints_are_refused_with_a_witness(self):
        r = Equations.cases_assemble(
            self.TARGET, r'x \gt \frac{1}{2} \lor x \lt \frac{1}{2}',
            self.ENDPOINTS, self.HYPOTHESES)
        self.assertFalse(r['ok'])
        self.assertIn('does not hold at exactly the points', r['error'])
        self.assertIn('hypothesis instead', r['error'])

    def test_assembles_a_bounded_answer_from_one_case(self):
        r = Equations.cases_assemble(
            r'x^2 \lt 1', r'-1 \lt x \lt 1',
            [r'x \lt 1'], [r'x \gt -1'])
        self.assertTrue(r['ok'], r.get('error'))
        self.assertEqual(r['result'], r'-1 \lt x \land x \lt 1')
        self.assertEqual(r['check']['status'], 'agree')

    def test_parenthesized_conjunctions_in_a_union(self):
        r = Equations.cases_assemble(
            r'(x^2-1)(x^2-4) \lt 0',
            r'(-2 \lt x \lt -1) \lor \left(1 \lt x \lt 2\right)',
            [r'x \lt -1', r'x \lt 2'], [r'x \gt -2', r'x \gt 1'])
        self.assertTrue(r['ok'], r.get('error'))
        self.assertEqual(r['check']['status'], 'agree')
        P.parse_latex(r['result'])
        self.assertEqual(
            r['result'],
            r'-2 \lt x \land x \lt -1 \lor 1 \lt x \land x \lt 2')

    def test_every_conjunction_member_must_come_from_the_case(self):
        r = Equations.cases_assemble(
            r'x^2 \lt 1', r'-2 \lt x \lt 1',
            [r'x \lt 1'], [r'x \gt -1'])
        self.assertFalse(r['ok'])
        self.assertIn('member', r['error'])
        self.assertIn('neither the recorded endpoint', r['error'])

    def test_reordered_duplicate_conjunctions_are_refused(self):
        r = Equations.cases_assemble(
            r'x^2 \lt 1',
            r'-1 \lt x \lt 1 \lor x \lt 1 \land x \gt -1',
            [r'x \lt 1', r'x \lt 1'], [r'x \gt -1', r'x \gt -1'])
        self.assertFalse(r['ok'])
        self.assertIn('same relation(s)', r['error'])

    def test_invented_disjunct_is_refused(self):
        r = Equations.cases_assemble(
            self.TARGET, r'x \gt \frac{1}{2} \lor x \lt -1',
            self.ENDPOINTS, self.HYPOTHESES)
        self.assertFalse(r['ok'])
        self.assertIn('neither the recorded endpoint', r['error'])

    def test_non_strict_hypothesis_is_refused(self):
        r = Equations.cases_assemble(
            self.TARGET, r'x \gt \frac{1}{2} \lor x \lt 0',
            self.ENDPOINTS, [r'x \gt 0', r'x \le 0'])
        self.assertFalse(r['ok'])
        self.assertIn('not a strict relation', r['error'])

    def test_mirror_spelled_duplicate_disjuncts_are_refused(self):
        r = Equations.cases_assemble(
            self.TARGET, r'x \lt 0 \lor 0 \gt x',
            [r'\frac{1}{2} \gt x', r'\frac{1}{2} \gt x'],
            [r'x \lt 0', r'x \lt 0'])
        self.assertFalse(r['ok'])
        self.assertIn('same relation', r['error'])

    def test_foreign_variable_is_refused(self):
        r = Equations.cases_assemble(
            self.TARGET, r'x \gt \frac{1}{2} \lor y \lt 0',
            [r'\frac{1}{2} \lt x', r'y \lt 0'],
            [r'x \gt 0', r'y \lt 0'])
        self.assertFalse(r['ok'])
        self.assertIn('names y', r['error'])

    def test_single_disjunct_needs_no_assembly(self):
        r = Equations.cases_assemble(
            self.TARGET, r'x \gt \frac{1}{2}',
            [self.ENDPOINTS[0]], [self.HYPOTHESES[0]])
        self.assertFalse(r['ok'])
        self.assertIn('needs no assembly', r['error'])

    def test_case_count_must_match_the_disjuncts(self):
        r = Equations.cases_assemble(
            self.TARGET, r'x \gt \frac{1}{2} \lor x \lt 0',
            [self.ENDPOINTS[0]], [self.HYPOTHESES[0]])
        self.assertFalse(r['ok'])
        self.assertIn('one recorded case per disjunct', r['error'])

    def test_target_must_be_a_relation(self):
        r = Equations.cases_assemble(
            r'\frac{1}{x}', r'x \gt \frac{1}{2} \lor x \lt 0',
            self.ENDPOINTS, self.HYPOTHESES)
        self.assertFalse(r['ok'])
        self.assertIn('stated relation', r['error'])

    def test_sign_split_of_a_quadratic_inequality(self):
        r = Equations.cases_assemble(
            r'x^2 \gt 1', r'x \gt 1 \lor x \lt -1',
            [r'x \gt 1', r'x \lt -1'], [r'x \gt 0', r'x \lt 0'])
        self.assertTrue(r['ok'], r.get('error'))
        self.assertEqual(r['check']['status'], 'agree')

    def test_same_relation_tolerates_only_true_mirrors(self):
        self.assertTrue(Equations._same_relation(
            r'x \gt \frac{1}{2}', r'\frac{1}{2} \lt x'))
        self.assertTrue(Equations._same_relation(r'x \lt 0', r'0 \gt x'))
        self.assertFalse(Equations._same_relation(
            r'x \gt \frac{1}{2}', r'\frac{1}{2} \gt x'))
        self.assertFalse(Equations._same_relation(r'x \lt 0', r'x \gt 0'))


class TestLedgerPremises(unittest.TestCase):
    """Inputs a session states rather than derives — where checking stops."""

    def _washed(self):
        # the live int1 shape: typed answers laundered into green steps by
        # multiplying by 1 and canonicalizing
        ledger = Ledger()
        for equation in (r'A = -\frac{b}{(n-1)(a^2-b^2)}',
                         r'B = \frac{a(2n-3)}{(n-1)(a^2-b^2)}'):
            applied = ledger.record(
                Core.apply_both_sides(equation, '*', '1'))
            ledger.record(Core.expand(applied['result']))
        return ledger

    def test_a_derivation_reports_its_starting_point(self):
        ledger = Ledger()
        ledger.record(Core.expand('(x+1)^2'))
        self.assertEqual(ledger.premises(),
                         [{'step': 's1', 'input': '(x+1)^2'}])

    def test_a_continued_chain_adds_no_premise(self):
        ledger = Ledger()
        first = ledger.record(Core.apply_both_sides('2x+3 = 7', '-', '3'))
        ledger.record(Core.expand(first['result']))
        self.assertEqual([p['input'] for p in ledger.premises()],
                         ['2x+3 = 7'])

    def test_typed_assertions_surface_as_premises(self):
        ledger = self._washed()
        self.assertEqual([p['step'] for p in ledger.premises()],
                         ['s1', 's3'])
        self.assertIn('premises (stated, not derived here)', ledger.render())
        self.assertIn(r'A = -\frac{b}{(n-1)(a^2-b^2)}', ledger.render())
        self.assertIn('Rests on 2 stated premises',
                      ledger.render_markdown())

    def test_recorded_parts_of_a_result_are_derived_not_stated(self):
        # linearity splits one object into pieces it records; working on a
        # piece continues from recorded work even though no whole result
        # equals that input
        ledger = Ledger()
        linearity = Integration.integrate_linearity(
            r'\int (x^2 + x) \, dx', 'x')
        ledger.record(linearity)
        for piece in linearity['integrals']:
            ledger.record(Integration.integrate_power_rule(piece, 'x'))
        self.assertEqual([p['input'] for p in ledger.premises()],
                         [r'\int (x^2 + x) \, dx'])

    def test_the_same_given_is_stated_once(self):
        ledger = Ledger()
        ledger.record(Core.expand('(x+1)^2'))
        ledger.record(Core.substitute('(x+1)^2', 'x', '1'))
        self.assertEqual(len(ledger.premises()), 1)

    def test_premises_can_be_restricted_to_the_selected_spine(self):
        ledger = self._washed()
        self.assertEqual([p['step'] for p in ledger.premises(['s3', 's4'])],
                         ['s3'])
        topology = ledger.presentation_topology()
        self.assertEqual([p['step'] for p in topology['spine_premises']],
                         ['s1', 's3'])

    def test_query_only_and_empty_ledgers_have_no_premises(self):
        self.assertEqual(Ledger().premises(), [])
        ledger = Ledger()
        ledger.record_comment('a note is not a premise')
        self.assertEqual(ledger.premises(), [])


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

    def test_cdot_is_presentation_only_in_normal_forms(self):
        # the explicit-\cdot P_LIST prop is display marking: structural
        # identity must not split on it (live: it severed a verified
        # int! chain), while the passive round trip keeps the dots
        self.assertTrue(P.same_expression('a \\cdot b', 'a b'))
        self.assertFalse(P.same_expression('a \\cdot b', 'a + b'))
        sym, n = P.parse_latex('1 \\cdot 2')
        self.assertIn('\\cdot', P.write_latex(sym, n))

    def test_spacing_tokens_are_presentation_only_in_normal_forms(self):
        # \, \; \quad are the same class as the \cdot prop: every reader
        # convention filters them, so comparison must too (live: an int!
        # chain root spelled `\int ... \, dt` could not cover its own
        # goal spelled `\int ... dt`, and the verified FTC result was
        # refused at admission)
        self.assertTrue(P.same_expression('2 \\, x', '2 x'))
        self.assertTrue(P.same_expression('x \\; + 1', 'x + 1'))
        self.assertTrue(P.same_expression(
            '\\int x^{2} \\, dx', '\\int x^{2} dx'))
        self.assertTrue(P.same_expression(
            '\\frac{d}{dx}\\int_{0}^{x^{2}}\\sqrt{1+t^{2}}\\,dt',
            '\\frac  {d} {dx}\\int_{0}^{x^{2}}\\sqrt{{1}+t^{2}}dt'))
        # the gen-68 capture boundary must survive style-blindness
        self.assertFalse(P.same_expression('\\cos(x) y', '\\cos(x y)'))
        self.assertTrue(P.same_expression('\\cos x \\, y', '\\cos x y'))

    def test_leibniz_chain_root_covers_the_rendered_goal(self):
        # the live trace's exact admission shape: the agent fed the full
        # Leibniz spelling verbatim (with the textbook \,) and the goal
        # was the instruction's rendered spelling (without); the chain
        # must cover
        from expr_commands import _chains_to_goal
        rec = Differentiation.differentiate(
            '\\frac{d}{dx}\\int_{0}^{x^{2}}\\sqrt{1+t^{2}}\\,dt', 'x')
        self.assertTrue(rec['ok'], rec.get('error'))
        book = ledger_module.Ledger()
        step = book.record(rec)
        self.assertTrue(_chains_to_goal(
            book.steps, step['id'],
            '\\frac  {d} {dx}\\int_{0}^{x^{2}}\\sqrt{{1}+t^{2}}dt'))

    def test_display_latex_prettifies_only_structural_product_stars(self):
        cases = {
            '3*x': '3 \\cdot x',
            '2 * 3': '2 \\cdot 3',
            '\\sin x*y': '\\sin x \\cdot y',
            '\\lim_{x \\to 0} x*\\sin x':
                '\\lim_{x \\to 0} x \\cdot \\sin x',
            'a*b+\\ldots': 'a \\cdot b+\\ldots',
        }
        for source, expected in cases.items():
            self.assertEqual(P.display_latex(source), expected, source)
        self.assertEqual(P.display_latex('3x'), '3x')
        self.assertEqual(P.display_latex('3 \\cdot x'), '3 \\cdot x')
        # Invalid or non-mathematical star contexts fail closed.
        for source in ('x**2', '\\text{a*b}',
                       '\\begin{matrix*}a\\end{matrix*}', 'x\\*y'):
            self.assertEqual(P.display_latex(source), source)

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

    def test_table_arctan_unit(self):
        r = self.ok(Integration.integrate_table('\\frac{1}{1+v^2}', 'v'))
        self.assertEqual(r['result'], '\\arctan\\left(v\\right) + C')

    def test_table_arctan_scaled(self):
        # completed-square residue from the live Weierstrass run
        r = self.ok(Integration.integrate_table(
            '\\frac{1}{3 w^2 + \\frac{5}{3}}', 'w'))
        self.assertIn('\\arctan', r['result'])

    def test_table_arctan_constant_numerator_peel(self):
        # numerator outside the rational fragment, denominator in it:
        # the constant peels and the reciprocal reaches the arctan rule
        r = self.ok(Integration.integrate_table(
            '\\frac{\\sqrt{5}}{5 (z^2 + 1)}', 'z'))
        self.assertIn('\\arctan', r['result'])

    def test_table_arctan_denominator_constant_factor_split(self):
        # var-free irrational factor inside the denominator (second live
        # route): \sqrt{5}(v^2+1) splits so the variable core reaches the
        # arctan rule; the whole product spelling closes too
        r = self.ok(Integration.integrate_table(
            '\\frac{1}{\\sqrt{5} (v^2 + 1)}', 'v'))
        self.assertIn('\\arctan', r['result'])
        r2 = self.ok(Integration.integrate_table(
            '\\frac{1}{\\sqrt{5}} \\cdot \\frac{1}{v^2 + 1}', 'v'))
        self.assertIn('\\arctan', r2['result'])

    def test_table_quadratic_linear_term_steers_to_square(self):
        r = Integration.integrate_table('\\frac{1}{3 u^2 + 2 u + 2}', 'u')
        self.assertFalse(r['ok'])
        self.assertIn('complete the square', r['error'])

    def test_table_negative_shift_no_arctan(self):
        # x^2 - 1 is not the arctan family (b < 0) and must not steer
        # to completing the square either
        r = Integration.integrate_table('\\frac{1}{x^2-1}', 'x')
        self.assertFalse(r['ok'])
        self.assertNotIn('complete the square', r['error'])

    def test_table_irrational_constant_integrand(self):
        # \frac{\sqrt{5}}{5} is var-free but outside the rational
        # fragment; a constant integrates to c v regardless of spelling
        r = self.ok(Integration.integrate_table('\\frac{\\sqrt{5}}{5}', 'v'))
        self.assertEqual(r['result'], '\\frac {\\sqrt{5}} {5} v + C')

    def test_table_sum_constant_integrand(self):
        r = self.ok(Integration.integrate_table('1 + \\sqrt{2}', 'v'))
        self.assertIn('v + C', r['result'])

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

    def test_markdown_render_prettifies_star_without_mutating_ledger(self):
        ledger = Ledger()
        step = ledger.record(
            Differentiation.differentiate('x^3 - 3*x', 'x'))
        stored_hash = step['hash']
        md = ledger.render_markdown()
        self.assertIn('x^3 - 3 \\cdot x', md)
        self.assertNotIn('3*x', md)
        self.assertEqual(step['input'], 'x^3 - 3*x')
        self.assertEqual(step['hash'], stored_hash)
        self.assertEqual(ledger.replay()['status'], 'verified')


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


class TestTypedCollectionsAndPairs(unittest.TestCase):
    def test_stationary_points_have_first_class_nodes(self):
        from replicator import Replicator

        latex = '\\{(-1,2),(1,-2)\\}'
        sym, notation = P.parse_latex(latex)
        collection = notation.getf(sym, Notation.COLLECTION)
        self.assertIsNotNone(collection)
        self.assertEqual(len(collection.args), 2)
        self.assertTrue(all(notation.getf(item, Notation.PAIR) is not None
                            for item in collection.args))

        copied_notation = Notation()
        copied = Replicator(notation, copied_notation)(sym)
        written = P.write_latex(copied, copied_notation)
        self.assertEqual(written, latex)
        self.assertEqual(P._normal_form(written), P._normal_form(latex))

    def test_collection_can_hold_a_pair_relation(self):
        sym, notation = P.parse_latex('\\{(x,y)=(2,1)\\}')
        collection = notation.getf(sym, Notation.COLLECTION)
        self.assertIsNotNone(collection)
        self.assertEqual(len(collection.args), 1)
        relation = notation.getf(collection.args[0], Notation.COMP)
        self.assertIsNotNone(relation)
        self.assertIsNotNone(notation.getf(relation.args[0], Notation.PAIR))
        self.assertIsNotNone(notation.getf(relation.args[1], Notation.PAIR))

    def test_collection_variants_and_empty_collection_round_trip(self):
        cases = (
            ('\\{\\}', '\\{\\}'),
            ('\\{x\\}', '\\{x\\}'),
            ('\\{x,y\\}', '\\{x,y\\}'),
            ('\\left\\{x,y\\right\\}', '\\{x,y\\}'),
            ('\\{|x|\\}', '\\{|x|\\}'),
            ('\\{|x|,|y|\\}', '\\{|x|,|y|\\}'),
            ('\\{\\{1,2\\},3\\}', '\\{\\{1,2\\},3\\}'),
        )
        for source, canonical in cases:
            with self.subTest(source=source):
                sym, notation = P.parse_latex(source)
                self.assertIsNotNone(
                    notation.getf(sym, Notation.COLLECTION))
                self.assertEqual(P.write_latex(sym, notation), canonical)
                self.assertTrue(P.same_expression(source, canonical))

    def test_pair_is_distinct_from_comma_syntax(self):
        pair, pair_notation = P.parse_latex('(x,y)')
        self.assertIsNotNone(pair_notation.getf(pair, Notation.PAIR))
        left_pair, left_notation = P.parse_latex(
            '\\left(x,y\\right)')
        self.assertIsNotNone(left_notation.getf(
            left_pair, Notation.PAIR))

        triple, triple_notation = P.parse_latex('(x,y,z)')
        triple_group = triple_notation.getf(triple, Notation.GROUP)
        self.assertIsNotNone(triple_group)
        self.assertIsNotNone(triple_notation.getf(
            triple_group.args[0], Notation.C_LIST))

        self.assertFalse(P.same_expression('(x,y)', 'x,y'))
        self.assertFalse(P.same_expression('\\{x,y\\}', 'x,y'))

    def test_set_builder_keeps_its_existing_node(self):
        sym, notation = P.parse_latex('\\{x|x\\gt0\\}')
        self.assertIsNotNone(notation.getf(sym, Notation.S_GROUP))
        self.assertIsNone(notation.getf(sym, Notation.COLLECTION))

    def test_scalar_tactics_refuse_typed_results(self):
        rec = Core.expand('\\{x,y\\}')
        self.assertFalse(rec['ok'])
        self.assertIn('not an expression', rec['error'])


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

    def test_disagreeing_result_is_not_recorded(self):
        # admission mirrors replay: replay hard-fails a disagree check,
        # so record refuses it up front instead of poisoning the session
        ledger = Ledger(os.path.join(tempfile.mkdtemp(), 'session.json'))
        bad = dict(Core.expand('(x+1)^2'))
        bad['check'] = {'status': 'disagree'}
        with self.assertRaises(ValueError):
            ledger.record(bad)
        self.assertEqual(len(ledger.steps), 0)

    def test_sourceless_provenance_step_is_not_recorded(self):
        # the explicit-value squeeze form sets no sources while its
        # replay validator demands them; record must refuse, not defer
        # the failure to replay
        ledger = Ledger(os.path.join(tempfile.mkdtemp(), 'session.json'))
        rec = Limits.limit_squeeze(
            '\\lim_{x \\to 0} x^2 \\sin{\\frac{1}{x}}',
            '(-x^2)', 'x^2', '0')
        self.assertTrue(rec['ok'], rec.get('error'))
        with self.assertRaises(ValueError):
            ledger.record(rec)
        self.assertEqual(len(ledger.steps), 0)

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

    def _answer_chain(self, ledger, claim):
        first = ledger.record(Core.apply_both_sides('2A = 1', '/', '2'),
                              goal=claim['id'])
        second = ledger.record(Core.expand(first['result']),
                               goal=claim['id'])
        return [first['id'], second['id']]

    def test_answer_shaped_claim_closes_conditional_on_its_premise(self):
        # "A = 1/2" states what an unknown IS: no standalone check can
        # decide it, but the checked chain from 2A = 1 does establish it
        ledger = Ledger()
        claim = ledger.record_claim(r'A = \frac{1}{2}')
        closed = ledger.conclude(claim['id'],
                                 self._answer_chain(ledger, claim))
        self.assertEqual(closed['verdict'], 'conditional')
        self.assertEqual(closed['conclusion']['closure'],
                         'derived-from-premise')
        self.assertEqual(closed['conclusion']['premise'], '2A = 1')
        self.assertEqual(ledger.replay()['status'], 'verified')
        self.assertIn('given 2A = 1', ledger.render())
        self.assertIn('stated premise', ledger.render_markdown())

    def test_premise_closure_is_replay_checked(self):
        ledger = Ledger()
        claim = ledger.record_claim(r'A = \frac{1}{2}')
        ledger.conclude(claim['id'], self._answer_chain(ledger, claim))
        claim['conclusion']['premise'] = '3A = 1'
        replay = ledger.replay()
        self.assertEqual(replay['status'], 'failed')
        self.assertIn('conclusion mismatch', replay['reason'])

    def test_a_claim_is_never_derived_from_a_false_premise(self):
        ledger = Ledger()
        claim = ledger.record_claim('0 = 1')
        first = ledger.record(Core.apply_both_sides('2 = 3', '-', '2'),
                              goal=claim['id'])
        second = ledger.record(Core.expand(first['result']),
                               goal=claim['id'])
        with self.assertRaisesRegex(ValueError, 'which is false'):
            ledger.conclude(claim['id'], [first['id'], second['id']])
        self.assertEqual(claim['verdict'], 'open')

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


class TestDisagreementNeedsEvidence(unittest.TestCase):
    # A `disagree` verdict is a positive claim that two expressions differ,
    # and it bars the step from the ledger for good. Float evaluation whose
    # significant digits were consumed by cancellation cannot support that
    # claim: expand's own canonical output for (x+y)^16 landed 6% from the
    # compact form at a point where the intermediate terms exceed the
    # result by eighteen orders of magnitude, so the engine refused an
    # expansion it had produced itself and told the agent to "correct the
    # arguments". Unsupported numeric evaluation is honest ignorance, not
    # evidence against a transformation.

    CANCELLING = [
        '(x+y)^{16}', '(x+1)^{26}', '(x+1)^{28}', '(x+1)^{30}',
    ]

    def test_expand_is_not_refused_by_its_own_oracle(self):
        for expr in self.CANCELLING:
            rec = Core.expand(expr)
            self.assertTrue(rec['ok'], expr)
            self.assertNotEqual(rec['check']['status'], 'disagree', expr)

    def test_the_ledger_accepts_the_expansion(self):
        led = Ledger()
        step = led.record(Core.expand('(x+y)^{16}'))
        self.assertEqual(step['check']['status'], 'agree')

    def test_a_real_error_inside_a_cancelling_shape_still_disagrees(self):
        # the guard must not become a hiding place: each pair is WRONG and
        # each sits inside the same catastrophic-cancellation shape that
        # motivated the guard
        for lhs, rhs in [('(x+y)^{16}', '(x+y)^{16}+1'),
                         ('(x+1)^{30}', '(x+1)^{30}+x'),
                         ('(x+y)^{12}', '(x-y)^{12}')]:
            self.assertEqual(
                P.numeric_spot_check(lhs, rhs)['status'], 'disagree',
                f'{lhs} vs {rhs}')

    def test_one_corrupted_coefficient_still_disagrees(self):
        # 12871 where the binomial coefficient is 12870, buried in the
        # middle of a degree-16 expansion
        good = ('x^{16}+16x^{15}+120x^{14}+560x^{13}+1820x^{12}+4368x^{11}'
                '+8008x^{10}+11440x^9+12870x^8+11440x^7+8008x^6+4368x^5'
                '+1820x^4+560x^3+120x^2+16x+1')
        self.assertEqual(
            P.numeric_spot_check('(x+1)^{16}', good)['status'], 'agree')
        self.assertEqual(
            P.numeric_spot_check('(x+1)^{16}',
                                 good.replace('12870', '12871'))['status'],
            'disagree')

    def test_ordinary_wrong_algebra_is_untouched(self):
        for lhs, rhs in [('(x+1)^2', 'x^2+1'),
                         ('\\sin(x+y)', '\\sin x + \\sin y'),
                         ('x^2+2x+1', 'x^2+2.000001x+1'),
                         ('\\lfloor x \\rfloor', 'x'),
                         ('|x|', 'x')]:
            self.assertEqual(P.numeric_spot_check(lhs, rhs)['status'],
                             'disagree', f'{lhs} vs {rhs}')

    def test_resolution_compares_the_gap_against_the_oracles_own_noise(self):
        s1, n1 = P.parse_latex('x^2-1')
        s2, n2 = P.parse_latex('x^2+1')
        env = {'x': 2.0006072315296617}
        v1 = P.numeric_eval(s1, n1, env)
        v2 = P.numeric_eval(s2, n2, env)
        # a benign point: the gap of 2 dwarfs a one-ULP nudge
        self.assertTrue(
            P._disagreement_resolves(s1, n1, s2, n2, env, v1, v2))
        # the same comparison where one side has lost its digits
        s3, n3 = P.parse_latex('(x+y)^{16}')
        s4, n4 = P.parse_latex(Core.expand('(x+y)^{16}')['result'])
        env2 = {'x': 2.0006072315296617, 'y': -2.33288936992431}
        v3 = P.numeric_eval(s3, n3, env2)
        v4 = P.numeric_eval(s4, n4, env2)
        self.assertFalse(P._num_agree(v3, v4, 1e-6))     # the raw floats differ
        self.assertFalse(
            P._disagreement_resolves(s3, n3, s4, n4, env2, v3, v4))

    def test_a_shape_mismatch_is_structural_not_numeric(self):
        # noise can never explain away a matrix that changed shape
        s1, n1 = P.parse_latex('\\begin{pmatrix}1&2\\\\3&4\\end{pmatrix}')
        s2, n2 = P.parse_latex('\\begin{pmatrix}1&2&3\\end{pmatrix}')
        env = {}
        v1 = P.numeric_eval(s1, n1, env)
        v2 = P.numeric_eval(s2, n2, env)
        self.assertTrue(
            P._disagreement_resolves(s1, n1, s2, n2, env, v1, v2))


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


class TestBareAbsBars(unittest.TestCase):
    # Bare `|...|` used to parse only around a single scalar, so a live agent
    # proposing \sqrt{|\cos 1/x|} got a syntax error. The grammar cannot fix
    # this (| is its own opener and closer: a wider rule puts the LALR
    # machine in a shift/reduce conflict at every middle bar), so bar pairs
    # are matched by a scanner at the shared parse boundary.

    def test_composite_bodies_parse(self):
        for latex in ('|x+1|', '|2x|', '|\\cos x|', '|-x|',
                      '\\sqrt{|\\cos\\frac{1}{x}|}', '|x+1|^2'):
            with self.subTest(latex=latex):
                sym, notation = P.parse_latex(latex)
                self.assertIsNotNone(sym)

    def test_lowered_bars_build_the_same_node_as_a_bare_scalar(self):
        self.assertEqual(P.write_latex(*P.parse_latex('|x+1|')), '|x+1|')
        self.assertEqual(Core.equal_exprs('|x+1|', 'x+1')['verdict'], 'no')

    def test_adjacent_bar_pairs_stay_a_product(self):
        # |a|b|c| is |a| b |c| — bars pair left to right, the human reading
        self.assertEqual(Core.equal_exprs('|x||y|', '|x| \\cdot |y|')['verdict'],
                         'yes')
        self.assertEqual(Core.equal_exprs('|a|b|c|', '|a| \\cdot b \\cdot |c|')
                         ['verdict'], 'yes')

    def test_vert_spellings_are_the_same_operator(self):
        for latex in ('\\lvert x+1 \\rvert', '\\vert x+1 \\vert',
                      '\\left\\lvert x+1\\right\\rvert'):
            with self.subTest(latex=latex):
                self.assertEqual(
                    Core.equal_exprs(latex, '|x+1|')['verdict'], 'yes')

    def test_set_builder_separator_is_not_an_abs_bar(self):
        # the retained \{x | P\} spelling keeps its condition separator even
        # when the condition itself contains absolute values
        sym, notation = P.parse_latex('\\{x | x \\gt |a|\\}')
        self.assertIsNotNone(notation.getf(sym, Notation.S_GROUP))

    def test_collection_literals_still_parse(self):
        sym, notation = P.parse_latex('\\{(-1,2),(1,-2)\\}')
        self.assertIsNotNone(notation.getf(sym, Notation.COLLECTION))

    def test_commands_that_name_their_own_delimiters_keep_them(self):
        # \abovewithdelims names a delimiter PAIR; pairing its bars as an
        # absolute value would swallow the command's arguments
        for latex in ('a \\abovewithdelims || 2pt b',
                      'a \\atopwithdelims .| b'):
            with self.subTest(latex=latex):
                sym, notation = P.parse_latex(latex)
                self.assertIsNotNone(sym)

    def test_prose_arguments_are_not_scanned_for_bars(self):
        from LatexParser import _lower_bare_abs
        self.assertEqual(_lower_bare_abs('\\text{a|b|c}'), '\\text{a|b|c}')
        self.assertEqual(_lower_bare_abs('\\left|x\\right|'),
                         '\\left|x\\right|')


class TestFloorAndCeiling(unittest.TestCase):
    # \lfloor / \lceil used to lex as ordinary letters, so the delimiters
    # became free variables in BOTH trust legs: expand('\lfloor x+1 \rfloor')
    # returned the mangled '\lfloor x+ \rfloor' with a green oracle check.
    # They are bracket operators now, exactly like |...| since gen 12.

    def test_floor_is_not_its_argument(self):
        self.assertEqual(Core.equal_exprs('\\lfloor x \\rfloor', 'x')
                         ['verdict'], 'no')
        self.assertEqual(Core.equal_exprs('\\lceil x \\rceil', 'x')
                         ['verdict'], 'no')

    def test_expand_keeps_the_bracket_whole(self):
        for latex, expected in (
                ('\\lfloor x+1 \\rfloor', '\\lfloor x+1 \\rfloor'),
                ('\\lfloor 2x \\rfloor', '\\lfloor 2x \\rfloor'),
                ('\\lceil x+1 \\rceil', '\\lceil x+1 \\rceil')):
            with self.subTest(latex=latex):
                r = Core.expand(latex)
                self.assertTrue(r['ok'])
                self.assertEqual(r['result'], expected)
                self.assertEqual(r['check']['status'], 'agree')

    def test_like_bracket_terms_collect_over_atoms(self):
        r = Core.expand('\\lfloor x \\rfloor + 2\\lfloor x \\rfloor')
        self.assertEqual(r['result'], '3 \\lfloor x \\rfloor')
        self.assertEqual(r['check']['status'], 'agree')

    def test_floor_and_ceiling_are_distinct_atoms(self):
        r = Core.expand('\\lfloor x \\rfloor + \\lceil x \\rceil')
        self.assertTrue(r['ok'])
        self.assertEqual(Core.equal_exprs('\\lfloor x \\rfloor',
                                          '\\lceil x \\rceil')['verdict'], 'no')

    def test_oracle_computes_real_floor_and_ceiling(self):
        # true identities the independent leg must confirm
        self.assertEqual(
            Core.equal_exprs('\\lfloor x+1 \\rfloor',
                             '\\lfloor x \\rfloor + 1')['verdict'], 'yes')
        self.assertEqual(
            Core.equal_exprs('\\lceil -x \\rceil',
                             '-\\lfloor x \\rfloor')['verdict'], 'yes')
        # and a false one it must refuse
        self.assertEqual(
            Core.equal_exprs('\\lfloor x \\rfloor + \\lceil x \\rceil',
                             '2x')['verdict'], 'no')

    def test_evaluate_constant_floor_and_ceiling(self):
        self.assertEqual(float(Core.evaluate('\\lfloor 2.7 \\rfloor')['result']),
                         2.0)
        self.assertEqual(float(Core.evaluate('\\lceil 2.1 \\rceil')['result']),
                         3.0)

    def test_differentiate_refuses_by_name(self):
        r = Differentiation.differentiate('\\lfloor x \\rfloor', 'x')
        self.assertFalse(r['ok'])
        self.assertIn('floor', r['error'])
        r = Differentiation.differentiate('\\lceil x \\rceil', 'x')
        self.assertFalse(r['ok'])
        self.assertIn('ceiling', r['error'])

    def test_sized_delimiters_are_the_same_operator(self):
        self.assertEqual(
            Core.equal_exprs('\\left\\lfloor x \\right\\rfloor',
                             '\\lfloor x \\rfloor')['verdict'], 'yes')
        self.assertEqual(
            Core.equal_exprs('\\left\\lceil x \\right\\rceil', 'x')['verdict'],
            'no')

    def test_presentation_round_trip_is_stable(self):
        for latex in ('\\lfloor x \\rfloor', '\\lceil\\frac {x} {2}\\rceil',
                      '\\lfloor x \\rfloor^{2}', '|x+1|'):
            with self.subTest(latex=latex):
                once = P.write_latex(*P.parse_latex(latex))
                twice = P.write_latex(*P.parse_latex(once))
                self.assertEqual(once, twice)


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

    def test_integrand_convention_links_chain_spine_and_renderers(self):
        # live probe (substitution workflow): integrate_table consumes the
        # INTEGRAND of the previous \int result — the accepted chaining
        # convention must be linkage-visible, or a linear derivation loses
        # its first step from the spine and any upstream assumptions drop
        # from the final conditions
        ledger = Ledger()
        s1 = ledger.record(Integration.integrate_substitute(
            '2x\\cos(x^2)', 'x', 'x^2', 'u', '\\cos(u)'))
        s2 = ledger.record(Integration.integrate_table('\\cos(u)', 'u'))
        s3 = ledger.record(Core.substitute(s2['result'], 'u', 'x^2'))
        self.assertTrue(s2['continues'])
        self.assertTrue(s3['continues'])
        ledger.record_selection(s3['result'], {
            'status': 'verified', 'source': 'ledger',
            'step': s3['id'], 'method': 'exact-result',
        })
        topology = ledger.presentation_topology()
        self.assertEqual(topology['spine'], [s1['id'], s2['id'], s3['id']])
        self.assertEqual(topology['unclassified_off_spine'], [])
        self.assertEqual(topology['parents'][s2['id']], s1['id'])
        self.assertNotIn('new chain', ledger.render_markdown())
        self.assertNotIn('(branch)', ledger.render())
        self.assertEqual(ledger.replay()['status'], 'verified')

    def test_presession_discontinuity_hint_relinks_at_render_time(self):
        # sessions recorded before the convention was linkage-visible
        # persist continues=False on honest integrand chains; topology and
        # the end-of-run renderers re-derive the link while the persisted
        # hint stays untouched
        ledger = Ledger()
        s1 = ledger.record(Integration.integrate_substitute(
            '2x\\cos(x^2)', 'x', 'x^2', 'u', '\\cos(u)'))
        s2 = ledger.record(Integration.integrate_table('\\cos(u)', 'u'))
        s2['continues'] = False
        topology = ledger.presentation_topology()
        self.assertEqual(topology['parents'][s2['id']], s1['id'])
        self.assertNotIn('new chain', ledger.render_markdown())
        self.assertFalse(ledger.steps[1]['continues'])

    def test_branch_resume_accepts_operator_body_input(self):
        # gen45 live refusal class: resuming from an \int-result step with
        # an integrand-consuming tactic was structurally impossible — the
        # gate demanded the wrapped result verbatim while the tactic's
        # interface takes the integrand
        ledger = Ledger()
        s1 = ledger.record(Integration.integrate_substitute(
            '2x\\cos(x^2)', 'x', 'x^2', 'u', '\\cos(u)'))
        ledger.record(Integration.integrate_by_parts(
            '\\cos(u)', 'u', '\\cos(u)', '1'))
        marker = ledger.record_branch(
            s1['id'], 'by parts makes the remainder harder')
        resumed = ledger.record(Integration.integrate_table('\\cos(u)', 'u'))
        self.assertEqual(resumed['exploration']['marker'], marker['id'])
        self.assertEqual(resumed['exploration']['from'], s1['id'])
        # result-anchored edges stay byte-identical with older sessions
        self.assertNotIn('anchor', resumed['exploration'])
        self.assertEqual(ledger.replay()['status'], 'verified')

    def test_branch_resume_from_source_input_abandons_the_step_itself(self):
        # live probe (root-anchor gap): the FIRST checked move was the
        # dead end; with only a result-anchored resume available, a strong
        # model manufactured a skipped no-op step to satisfy the gate and
        # the dead route stayed unclassified. The source step's recorded
        # INPUT is now a legitimate resume anchor meaning "abandon that
        # step itself".
        ledger = Ledger()
        s1 = ledger.record(Integration.integrate_by_parts(
            '2x\\cos(x^2)', 'x', '\\cos(x^2)', '2x'))
        marker = ledger.record_branch(
            s1['id'], 'by parts raises the degree; restart from the input')
        resumed = ledger.record(Integration.integrate_substitute(
            '2x\\cos(x^2)', 'x', 'x^2', 'u', '\\cos(u)'))
        self.assertEqual(resumed['exploration']['from'], s1['id'])
        self.assertEqual(resumed['exploration']['anchor'], 'input')
        s4 = ledger.record(Integration.integrate_table('\\cos(u)', 'u'))
        s5 = ledger.record(Core.substitute(s4['result'], 'u', 'x^2'))
        ledger.record_selection(s5['result'], {
            'status': 'verified', 'source': 'ledger',
            'step': s5['id'], 'method': 'exact-result',
        })
        topology = ledger.presentation_topology()
        self.assertEqual(topology['spine'],
                         [resumed['id'], s4['id'], s5['id']])
        self.assertIsNone(topology['parents'][resumed['id']])
        self.assertEqual(topology['abandoned_paths'], [{
            'marker': marker['id'], 'source': s1['id'],
            'continues_at': resumed['id'],
            'reason': 'by parts raises the degree; restart from the input',
            'steps': [s1['id']],
            'anchor': 'input',
        }])
        self.assertEqual(topology['unclassified_off_spine'], [])
        md = ledger.render_markdown()
        self.assertIn('Abandoned path from <code>s1</code>', md)
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


class TestLedgerOpenOutcome(unittest.TestCase):
    def test_open_outcome_records_selects_nothing_and_replays(self):
        ledger = Ledger()
        ledger.record(Core.expand('(x+1)^2'))
        outcome = ledger.record_open(
            'no tactic covers the goal shape; a checked lower-bound '
            'relation is the missing move')
        self.assertEqual(outcome['id'], 'r1')
        self.assertIsNone(outcome['result'])
        self.assertEqual(outcome['provenance']['status'], 'open')
        self.assertEqual(outcome['provenance']['source'], 'open')
        self.assertEqual(ledger.replay()['status'], 'verified')
        with self.assertRaisesRegex(ValueError, 'needs a reason'):
            ledger.record_open('   ')

    def test_open_outcome_never_carries_authority_on_replay(self):
        ledger = Ledger()
        step = ledger.record(Core.expand('(x+1)^2'))
        outcome = ledger.record_open('stalled')
        # hash-consistent tampering must still fail the semantic checks:
        # an open outcome can neither select a value nor cite authority
        for tamper in (
                {'result': step['result']},
                {'provenance': {'status': 'open', 'source': 'open',
                                'reason': 'stalled', 'step': step['id']}},
                {'provenance': {'status': 'verified', 'source': 'open',
                                'reason': 'stalled'}},
                {'provenance': {'status': 'open', 'source': 'open',
                                'reason': ''}}):
            tampered = dict(outcome)
            tampered.update(tamper)
            tampered['hash'] = ledger_module._selection_hash(
                tampered.get('result'), tampered['provenance'],
                tampered.get('goal'))
            ledger.selections[-1] = tampered
            replay = ledger.replay()
            self.assertEqual(replay['status'], 'failed', tamper)
            self.assertIn('final selection invalid', replay['reason'])
        # plain mutation without re-hashing fails on the hash itself
        ledger.selections[-1] = dict(outcome)
        ledger.selections[-1]['provenance'] = {
            'status': 'open', 'source': 'open', 'reason': 'edited'}
        self.assertEqual(ledger.replay()['status'], 'failed')

    def test_open_ending_renders_unresolved_marker_and_banner(self):
        ledger = Ledger()
        source = ledger.record(Core.expand('(x+1)^2'))
        ledger.record(Core.substitute(source['result'], 'x', '3'))
        ledger.record_branch(source['id'], 'the detour went nowhere')
        ledger.record_open('the goal needs a tactic this session lacks')
        md = ledger.render_markdown()
        self.assertIn('**Outcome `r1` — OPEN:** no certified result', md)
        self.assertIn('*(unverified reason)*', md)
        self.assertIn('left unresolved; outcome recorded open', md)
        self.assertNotIn('awaiting a continuing step', md)
        text = ledger.render()
        self.assertIn('(unresolved; outcome open)', text)
        self.assertIn('OPEN r1#', text)
        self.assertNotIn('SELECT', text)

    def test_trailing_open_outcome_keeps_earlier_certified_spine(self):
        ledger = Ledger()
        source = ledger.record(Core.expand('(x+1)^2'))
        dead = ledger.record(Core.substitute(source['result'], 'x', '1'))
        marker = ledger.record_branch(
            source['id'], 'the numeric detour does not answer the goal')
        resumed = ledger.record(Core.factor_quadratic(
            source['result'], 'x'))
        ledger.record_selection(resumed['result'], {
            'status': 'verified', 'source': 'ledger',
            'step': resumed['id'], 'method': 'exact-result',
        })
        ledger.record_open('a later cell stalled on a new goal')
        topology = ledger.presentation_topology()
        # the session-level spine still belongs to the certified selection
        self.assertEqual(topology['spine'], [source['id'], resumed['id']])
        self.assertEqual(topology['abandoned_paths'][0]['steps'],
                         [dead['id']])
        self.assertEqual(topology['selection'], 'r2')
        # an explicitly passed open provenance stays literal: no spine
        literal = ledger.presentation_topology(
            final_provenance={'status': 'open', 'source': 'open',
                              'reason': 'stalled'},
            marker_ids=[marker['id']])
        self.assertEqual(literal['spine'], [])
        self.assertEqual(ledger.replay()['status'], 'verified')


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


OSC_ROOT = '\\lim_{x \\to 0} x \\sqrt{\\cos\\frac{1}{x}}'


class TestApproachOracleKinkedBodies(unittest.TestCase):
    # The finite-point approach oracle extrapolates over a geometric
    # h-ladder with Aitken acceleration, so |x|-kinked continuous bodies
    # converge (the smooth Richardson model alone skipped every sample).

    def test_abs_body_substitutes(self):
        rec = Limits.limit_substitute('\\lim_{x \\to 0} (|x|)')
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['check']['status'], 'agree')

    def test_negative_abs_body_substitutes(self):
        rec = Limits.limit_substitute('\\lim_{x \\to 0} (-|x|)')
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['check']['status'], 'agree')

    def test_two_sided_abs_squeeze_closes_oscillating_root(self):
        # the live-run bounds: -|x| <= x sqrt(cos(1/x)) <= |x|
        rec = Limits.limit_squeeze(OSC_ROOT, '(-|x|)', '(|x|)', '0')
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['result'], '0')

    def test_signed_bounds_still_refused_two_sided(self):
        # the flipping bounds -x <= body <= x are genuinely wrong for x < 0
        rec = Limits.limit_squeeze(OSC_ROOT, '(-x)', 'x', '0')
        self.assertFalse(rec['ok'])
        self.assertIn('ordering', rec['error'])

    def test_oscillating_body_still_refuses(self):
        rec = Limits.limit_substitute('\\lim_{x \\to 0} \\sin\\frac{1}{x}')
        self.assertFalse(rec['ok'])

    def test_leading_minus_body_names_the_repair(self):
        rec = Limits.limit_substitute('\\lim_{x \\to 0} -x')
        self.assertFalse(rec['ok'])
        self.assertIn('parenthesize', rec['error'])


class TestLimitFromSides(unittest.TestCase):
    # The core tactic validates the two-sided target and the value; the
    # recorded one-sided premises are enforced by the registry handlers
    # (session-required) and re-validated by replay provenance — a
    # source-less step fails replay, mirroring limit_squeeze.

    def test_two_sided_from_agreeing_sides(self):
        rec = Limits.limit_from_sides('\\lim_{x \\to 0} x', '0')
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['result'], '0')
        self.assertEqual(rec['check']['status'], 'agree')
        self.assertIn('^{-}', rec['left'])
        self.assertIn('^{+}', rec['right'])

    def test_one_sided_target_refused(self):
        rec = Limits.limit_from_sides('\\lim_{x \\to 0^+} x', '0')
        self.assertFalse(rec['ok'])
        self.assertIn('two-sided', rec['error'])

    def test_value_with_bound_variable_refused(self):
        rec = Limits.limit_from_sides('\\lim_{x \\to 0} x', 'x')
        self.assertFalse(rec['ok'])

    def test_contradicted_value_refused(self):
        rec = Limits.limit_from_sides('\\lim_{x \\to 0} x^2', '1')
        self.assertFalse(rec['ok'])
        self.assertIn('contradicts', rec['error'])

    def test_unconverged_oracle_defers_to_premises(self):
        # an INDEFINITE integral atom has no bounds for the quadrature
        # evaluator (definite bodies evaluate since gen 77), so the
        # record is exact-by-theorem and only the registry handlers
        # (recorded premises) can admit it
        rec = Limits.limit_from_sides(
            '\\lim_{x \\to 0} \\left(x + \\int t \\, dt\\right)',
            '\\int t \\, dt')
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['check']['status'], 'exact')

    def test_oscillating_decay_now_concurs(self):
        # the decade-spaced approach ladder reaches deep enough that the
        # x-amplitude bound itself pins x sin(1/x) below tolerance — the
        # oracle now concurs instead of deferring
        rec = Limits.limit_from_sides(
            '\\lim_{x \\to 0} x \\sin\\frac{1}{x}', '0')
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['check']['status'], 'agree')


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

    def test_covers_goal_bare_integrand_vs_textbook_integral(self):
        # live int! failure: the first step consumed the BARE integrand of
        # a textbook goal whose denominator is parenthesized; the goal
        # link must compare integrand-to-integrand, brackets stripped
        bare = '\\frac{1}{x^{\\frac{1}{2}}+x^{\\frac{1}{3}}}'
        goal = '\\int\\frac {dx} {(x^{\\frac {1} {2}}+x^{\\frac {1} {3}})}'
        self.assertTrue(P.covers_goal(bare, goal))
        self.assertTrue(P.covers_goal(goal, bare))
        self.assertFalse(P.covers_goal('\\frac{1}{x^{1/2}}', goal))

    def test_covers_goal_peels_emitter_parens_around_sum_integrand(self):
        # the \int emitter parenthesizes sum integrands as pure syntax
        # protection; the linearity/assemble input is the bare sum
        wrapped = '\\int \\left(6u^2-6u+6-\\frac{6}{u+1}\\right) \\, d u'
        plain = '6u^2-6u+6-\\frac{6}{u+1}'
        self.assertTrue(P.covers_goal(plain, wrapped))
        self.assertTrue(P.covers_goal(wrapped, plain))
        # semantic brackets are never stripped: |x| is not x
        self.assertFalse(P.covers_goal('\\int |x| \\, d x',
                                       '\\int x \\, d x'))
        self.assertFalse(P.covers_goal('|x|', '\\int x \\, d x'))

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


class TestConstrainedOracle(unittest.TestCase):
    """The oracle samples only inside a stated region."""

    POSITIVE = [{'text': 'x \\gt 0', 'constraint': 'x \\gt 0'}]

    def test_spot_check_agrees_only_under_the_hypothesis(self):
        self.assertEqual(
            P.numeric_spot_check('\\sqrt{x^2}', 'x')['status'], 'disagree')
        self.assertEqual(
            P.numeric_spot_check('\\sqrt{x^2}', 'x',
                                 assumptions=self.POSITIVE)['status'],
            'agree')

    def test_region_without_interior_is_skipped_not_agreed(self):
        both = self.POSITIVE + [{'text': 'x \\lt 0',
                                 'constraint': 'x \\lt 0'}]
        check = P.numeric_spot_check('\\sqrt{x^2}', 'x', assumptions=both)
        self.assertEqual(check['status'], 'skipped')

    def test_hypothesis_variables_are_sampled_too(self):
        # a hypothesis about a variable the compared sides never mention
        # must not reject every point
        check = P.numeric_spot_check('y + y', '2y',
                                     assumptions=self.POSITIVE)
        self.assertEqual(check['status'], 'agree')

    def test_relation_check_sees_a_wrong_flip(self):
        kept = P.numeric_relation_check(
            'a x \\lt b', '\\frac{ax}{a} \\lt \\frac{b}{a}',
            assumptions=[{'constraint': 'a \\gt 0'}])
        self.assertEqual(kept['status'], 'agree')
        flipped = P.numeric_relation_check(
            'a x \\lt b', '\\frac{ax}{a} \\gt \\frac{b}{a}',
            assumptions=[{'constraint': 'a \\gt 0'}])
        self.assertEqual(flipped['status'], 'disagree')
        # ... and the same flip is right on the other side of zero
        negative = P.numeric_relation_check(
            'a x \\lt b', '\\frac{ax}{a} \\gt \\frac{b}{a}',
            assumptions=[{'constraint': 'a \\lt 0'}])
        self.assertEqual(negative['status'], 'agree')

    def test_relation_check_needs_two_relations(self):
        self.assertEqual(
            P.numeric_relation_check('x + 1', 'x + 1')['status'], 'skipped')

    def test_exclusive_hypotheses_recognises_a_sign_split(self):
        split = [{'text': 'x \\gt 0', 'constraint': 'x \\gt 0'},
                 {'text': 'x \\lt 0', 'constraint': 'x \\lt 0'}]
        self.assertEqual(P.exclusive_hypotheses(split), [(0, 1)])
        crossed = [{'constraint': 'x \\gt 0'}, {'constraint': '0 \\gt x'}]
        self.assertEqual(P.exclusive_hypotheses(crossed), [(0, 1)])
        compatible = [{'constraint': 'x \\gt 0'}, {'constraint': 'y \\lt 0'}]
        self.assertEqual(P.exclusive_hypotheses(compatible), [])
        # a side condition is not a hypothesis
        self.assertEqual(
            P.exclusive_hypotheses([{'text': 'a \\ne 0', 'nonzero': 'a'}]),
            [])


class TestApplyUnderHypothesis(unittest.TestCase):
    """Sign case splits: the agent states the case, the tactic records it."""

    def test_unknown_sign_refusal_names_the_available_move(self):
        rec = Core.apply_both_sides('a x \\lt b', '/', 'a')
        self.assertFalse(rec['ok'])
        self.assertIn('assuming', rec['error'])
        self.assertIn('a > 0', rec['error'])

    def test_positive_case_keeps_and_negative_case_flips(self):
        positive = Core.apply_both_sides('a x \\lt b', '/', 'a',
                                         assuming='a > 0')
        self.assertTrue(positive['ok'], positive.get('error'))
        self.assertIn('\\lt', positive['result'])
        self.assertEqual(positive['check']['status'], 'agree')
        self.assertEqual([a['text'] for a in positive['assumptions']],
                         ['a \\gt 0'])
        self.assertEqual(positive['args']['assuming'], 'a > 0')
        negative = Core.apply_both_sides('a x \\lt b', '/', 'a',
                                         assuming='a < 0')
        self.assertTrue(negative['ok'], negative.get('error'))
        self.assertIn('\\gt', negative['result'])
        self.assertEqual(negative['check']['status'], 'agree')

    def test_hypothesis_about_another_expression_does_not_pin_the_sign(self):
        rec = Core.apply_both_sides('a x \\lt b', '/', 'a', assuming='x > 0')
        self.assertFalse(rec['ok'])
        self.assertIn('unknown sign', rec['error'])

    def test_rearranged_hypothesis_pins_the_factor(self):
        rec = Core.apply_both_sides('a \\lt b', '*', 'x-3', assuming='x > 3')
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertIn('\\lt', rec['result'])
        self.assertEqual(rec['check']['status'], 'agree')

    def test_hypothesis_contradicting_a_literal_is_refused(self):
        rec = Core.apply_both_sides('x \\lt b', '*', '2', assuming='2 < 0')
        self.assertFalse(rec['ok'])
        self.assertIn('contradicts', rec['error'])

    def test_unsatisfiable_hypothesis_cannot_read_as_checked(self):
        rec = Core.apply_both_sides('a x \\lt b', '*', 'x^2',
                                    assuming='x^2 < 0')
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['check']['status'], 'skipped')

    def test_only_strict_hypotheses_are_accepted(self):
        for bad, needle in ((r'a \ge 0', 'strict'), ('a = 1', 'strict'),
                            ('a', 'must be a relation')):
            rec = Core.apply_both_sides('a x \\lt b', '/', 'a', assuming=bad)
            self.assertFalse(rec['ok'], bad)
            self.assertIn(needle, rec['error'])

    def test_strict_hypothesis_replaces_the_nonzero_side_condition(self):
        rec = Core.apply_both_sides('x = y', '*', 'z', assuming='z > 0')
        texts = [a['text'] for a in rec['assumptions']]
        self.assertEqual(texts, ['z \\gt 0'])

    def test_equation_case_workflow_replays(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'cases.json')
            book = Ledger(path)
            first = Core.apply_both_sides('\\frac{1}{x} \\lt 2', '*', 'x',
                                          assuming='x > 0')
            book.record(first)
            book.record(Core.expand(first['result']))
            second = Core.apply_both_sides('\\frac{1}{x} \\lt 2', '*', 'x',
                                           assuming='x < 0')
            book.record(second)
            book.record(Core.expand(second['result']))
            book.save()
            self.assertEqual(Ledger(path).replay()['status'], 'verified')


class TestEqualUnderHypothesis(unittest.TestCase):
    def test_domain_mismatch_steers_to_the_restricted_question(self):
        rec = Core.equal_exprs('\\ln(x^2)', '2\\ln x')
        self.assertEqual(rec['verdict'], 'no')
        self.assertIn('assuming', rec['note'])

    def test_restricted_yes_carries_its_condition(self):
        rec = Core.equal_exprs('\\ln(x^2)', '2\\ln x', assuming='x > 0')
        self.assertEqual(rec['verdict'], 'yes')
        self.assertIn('under the stated assumptions', rec['method'])
        self.assertEqual([a['text'] for a in rec['assumptions']],
                         ['x \\gt 0'])

    def test_several_hypotheses_may_be_comma_separated(self):
        rec = Core.equal_exprs('\\ln(xy)', '\\ln x + \\ln y',
                               assuming='x > 0, y > 0')
        self.assertEqual(rec['verdict'], 'yes')
        self.assertEqual(len(rec['assumptions']), 2)

    def test_a_restricted_no_is_still_a_no(self):
        rec = Core.equal_exprs('\\sqrt{x^2}', 'x', assuming='x < 0')
        self.assertEqual(rec['verdict'], 'no')
        self.assertIn('counterexample', rec)

    def test_unconditional_verdicts_stay_unconditional(self):
        rec = Core.equal_exprs('x', 'x + 1', assuming='x > 0')
        self.assertEqual(rec['verdict'], 'no')
        self.assertEqual(rec['method'], 'canonical')
        self.assertNotIn('assumptions', rec)

    def test_relation_sides_inherit_the_hypothesis(self):
        rec = Core.equal_exprs('\\sqrt{x^2} = 3', 'x = 3', assuming='x > 0')
        self.assertEqual(rec['verdict'], 'yes')
        self.assertEqual([a['text'] for a in rec['assumptions']],
                         ['x \\gt 0'])


class TestAlternativeCasesInTheLedger(unittest.TestCase):
    def _two_case_ledger(self):
        # one connected chain whose two steps were recorded under opposite
        # sign hypotheses: individually checked, jointly worth nothing
        first = Core.apply_both_sides('x = x', '*', 'y', assuming='y > 0')
        second = Core.apply_both_sides(first['result'], '+', '1',
                                       assuming='y < 0')
        book = Ledger()
        claim = book.record_claim(second['result'])
        book.record(first, goal=claim['id'])
        book.record(second, goal=claim['id'])
        return book, claim

    def test_conclusion_refuses_mutually_exclusive_hypotheses(self):
        book, claim = self._two_case_ledger()
        with self.assertRaises(ValueError) as caught:
            book.conclude(claim['id'], ['s1', 's2'])
        self.assertIn('mutually exclusive', str(caught.exception))

    def test_markdown_never_conjoins_alternative_cases(self):
        book, _ = self._two_case_ledger()
        out = book.render_markdown()
        self.assertIn('Alternative case hypotheses', out)
        head = out.split('**s1**')[0]
        self.assertNotIn('Valid under the assumptions', head)

    def test_compatible_assumptions_still_render_as_one_condition(self):
        book = Ledger()
        book.record(Core.apply_both_sides('x = y', '*', 'z'))
        out = book.render_markdown()
        self.assertIn('Valid under the assumptions', out)
        self.assertNotIn('Alternative case', out)


class TestDefiniteIntegralParts(unittest.TestCase):
    """Shared structure-reading for the FTC tactic and quadrature leg."""

    def test_canonical_spelling(self):
        self.assertEqual(
            P.definite_integral_parts('\\int_0^1 x^{2} \\, dx', 'x'),
            ('x', 'x^{2}', '0', '1'))

    def test_reversed_script_order_normalizes(self):
        self.assertEqual(
            P.definite_integral_parts('\\int^1_0 x^{2} \\, dx', 'x'),
            ('x', 'x^{2}', '0', '1'))

    def test_textbook_differential_in_numerator(self):
        self.assertEqual(
            P.definite_integral_parts('\\int_1^2 \\frac{dx}{x}', 'x'),
            ('x', '\\frac {1} {x}', '1', '2'))

    def test_symbolic_bounds_read(self):
        self.assertEqual(
            P.definite_integral_parts('\\int_{a}^{b} x \\, dx', 'x'),
            ('x', 'x', 'a', 'b'))

    def test_variable_is_discovered_when_unstated(self):
        self.assertEqual(
            P.definite_integral_parts('\\int_1^2 \\frac{dt}{t}'),
            ('t', '\\frac {1} {t}', '1', '2'))

    def test_indefinite_is_none(self):
        self.assertIsNone(
            P.definite_integral_parts('\\int x^{2} \\, dx', 'x'))

    def test_single_bound_is_none(self):
        self.assertIsNone(
            P.definite_integral_parts('\\int_0 x^{2} \\, dx', 'x'))

    def test_wrong_variable_is_none(self):
        self.assertIsNone(
            P.definite_integral_parts('\\int_0^1 x^{2} \\, dx', 't'))


class TestNumericDefiniteCheck(unittest.TestCase):
    """The quadrature leg re-integrates the integrand and never touches
    the antiderivative."""

    def test_right_value_agrees(self):
        check = P.numeric_definite_check('\\int_0^1 x^{2} \\, dx', 'x',
                                         '\\frac{1}{3}')
        self.assertEqual(check['status'], 'agree')
        self.assertEqual(check['method'], 'composite-simpson quadrature')

    def test_wrong_value_disagrees(self):
        # F(2) alone for \int_1^2 — the exact live wrong-answer shape
        check = P.numeric_definite_check('\\int_1^2 x^{2} \\, dx', 'x',
                                         '\\frac{8}{3}')
        self.assertEqual(check['status'], 'disagree')

    def test_interior_pole_refuses(self):
        # the classic FTC trap: \int_{-1}^{1} x^{-2} "=" -2
        check = P.numeric_definite_check(
            '\\int_{-1}^{1} \\frac{1}{x^{2}} \\, dx', 'x', '-2')
        self.assertEqual(check['status'], 'disagree')
        self.assertIn('improper', check.get('reason', ''))

    def test_parameters_are_sampled(self):
        check = P.numeric_definite_check('\\int_0^1 c x \\, dx', 'x',
                                         '\\frac{c}{2}')
        self.assertEqual(check['status'], 'agree')
        self.assertGreater(check['samples'], 1)

    def test_symbolic_bounds_are_sampled(self):
        # a symbolic bound is a parameter: the identity in (a, b) is
        # checked at sampled bound values, no longer skipped
        check = P.numeric_definite_check('\\int_a^b x \\, dx', 'x',
                                         '\\frac{b^2-a^2}{2}')
        self.assertEqual(check['status'], 'agree')

    def test_symbolic_bound_lie_is_refused(self):
        check = P.numeric_definite_check('\\int_a^b x \\, dx', 'x',
                                         '\\frac{b^2-a^2}{3}')
        self.assertEqual(check['status'], 'disagree')

    def test_break_under_a_sampled_bound_is_a_bad_draw(self):
        # \int_0^t of the endpoint-singular integrand: draws with t > 1
        # break the domain and must be skipped, not reported as a
        # witness — the surviving draws agree
        check = P.numeric_definite_check(
            '\\int_0^t \\frac{1}{(2-x) \\sqrt{1-x}} \\, dx', 'x',
            '\\left(-2 \\arctan\\sqrt{1-t}\\right) - '
            '\\left(-2 \\arctan\\sqrt{1}\\right)')
        self.assertEqual(check['status'], 'agree')

    def test_unevaluable_result_still_hunts_the_break(self):
        # the divergent classic: F(b)-F(a) contains ln(0), which is not
        # evaluable — the check must still find the interior/endpoint
        # break instead of skipping on the result
        check = P.numeric_definite_check('\\int_0^1 \\frac{1}{x} \\, dx',
                                         'x', '\\ln(1) - \\ln(0)')
        self.assertEqual(check['status'], 'disagree')
        self.assertIn('improper', check['reason'])


class TestDerivativeOperatorParts(unittest.TestCase):
    """The Leibniz-prefix reading convention: consulted only at the
    differentiate boundary, so nothing else re-reads a fraction."""

    def test_prefix_form(self):
        self.assertEqual(
            P.derivative_operator_parts('\\frac{d}{d x} (x^2+1)'),
            ('x', 'x^{2}+1'))

    def test_adjacent_dx_spelling(self):
        var, operand = P.derivative_operator_parts('\\frac{d}{dx} x^3')
        self.assertEqual(var, 'x')
        self.assertEqual(operand, 'x^{3}')

    def test_operand_spanning_several_factors(self):
        var, operand = P.derivative_operator_parts(
            '\\frac{d}{d u} u \\sin u')
        self.assertEqual(var, 'u')
        self.assertEqual(
            Core.equal_exprs(operand, 'u \\sin u')['verdict'], 'yes')

    def test_differential_in_numerator_form(self):
        var, operand = P.derivative_operator_parts('\\frac{d y}{d x}')
        self.assertEqual((var, operand), ('x', 'y'))

    def test_numerator_operand_with_trailing_factors_is_ambiguous(self):
        self.assertIsNone(
            P.derivative_operator_parts('\\frac{d y}{d x} z'))

    def test_macro_named_variable(self):
        var, _ = P.derivative_operator_parts(
            '\\frac{d}{d \\alpha} \\alpha^2')
        self.assertEqual(var, '\\alpha')

    def test_bare_operator_without_operand_is_none(self):
        self.assertIsNone(P.derivative_operator_parts('\\frac{d}{d x}'))

    def test_ordinary_fraction_is_none(self):
        self.assertIsNone(P.derivative_operator_parts('\\frac{a}{b} x'))
        self.assertIsNone(P.derivative_operator_parts('\\frac{d}{b} x'))
        self.assertIsNone(P.derivative_operator_parts('\\frac{a}{d x} x'))

    def test_higher_order_operator_is_not_misread(self):
        self.assertIsNone(
            P.derivative_operator_parts('\\frac{d^2}{d x^2} x^3'))

    def test_fraction_not_in_prefix_position_is_none(self):
        self.assertIsNone(
            P.derivative_operator_parts('y \\frac{d}{d x} x'))


class TestDifferentiateDefiniteIntegral(unittest.TestCase):
    """The FTC bound rule as a differentiate branch: d/dx of a
    variable-bound definite integral, checked by quadrature plus a
    central difference."""

    ASK = '\\int_0^{x^2} \\sqrt{1+t^2} d t'

    def test_variable_upper_bound_closes(self):
        rec = Differentiation.differentiate(self.ASK, 'x')
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['check']['status'], 'agree')
        self.assertEqual(rec['check']['method'],
                         'quadrature central-difference')
        self.assertEqual(rec['method'], 'ftc')
        self.assertEqual(
            Core.equal_exprs(rec['result'],
                             '2 x \\sqrt{x^{4}+1}')['verdict'], 'yes')

    def test_leibniz_spelling_names_its_own_variable(self):
        rec = Differentiation.differentiate(
            '\\frac{d}{d x} ' + self.ASK, None)
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['args']['var'], 'x')
        self.assertEqual(
            Core.equal_exprs(rec['result'],
                             '2 x \\sqrt{x^{4}+1}')['verdict'], 'yes')
        # the record's input stays the typed spelling
        self.assertIn('\\frac{d}{d x}', rec['input'])

    def test_leibniz_variable_mismatch_refuses(self):
        rec = Differentiation.differentiate('\\frac{d}{d y} y^2', 'x')
        self.assertFalse(rec['ok'])
        self.assertIn('d/dy', rec['error'])

    def test_both_bounds_variable(self):
        rec = Differentiation.differentiate(
            '\\int_{x}^{x^2} t^3 d t', 'x')
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(
            Core.equal_exprs(rec['result'],
                             '2 x^{7} - x^{3}')['verdict'], 'yes')

    def test_constant_bounds_differentiate_to_zero(self):
        rec = Differentiation.differentiate(
            '\\int_0^1 \\sqrt{1+t^2} d t', 'x')
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['result'], '0')

    def test_continuity_is_recorded_not_proved(self):
        rec = Differentiation.differentiate(self.ASK, 'x')
        texts = [a['text'] for a in rec['assumptions']]
        self.assertTrue(any('continuous between 0 and x^{2}' in t
                            for t in texts), texts)

    def test_refuses_the_bound_variable(self):
        rec = Differentiation.differentiate(self.ASK, 't')
        self.assertFalse(rec['ok'])
        self.assertIn('integration variable', rec['error'])

    def test_refuses_differentiation_under_the_integral_sign(self):
        rec = Differentiation.differentiate(
            '\\int_0^{x^2} x t \\, d t', 'x')
        self.assertFalse(rec['ok'])
        self.assertIn('under the integral sign', rec['error'])

    def test_refuses_a_bound_containing_the_integration_variable(self):
        rec = Differentiation.differentiate(
            '\\int_0^{t} t^2 \\, d t', 'x')
        self.assertFalse(rec['ok'])
        self.assertIn('bound', rec['error'])

    def test_indefinite_integral_is_never_a_silent_constant(self):
        # measured before the guard: the polyrat path atomized the
        # integral and differentiated the atom to a silent, uncheckable 0
        rec = Differentiation.differentiate('\\int t^2 d t', 'x')
        self.assertFalse(rec['ok'])
        self.assertIn('indefinite', rec['error'])

    def test_rules_never_reach_across_the_integral_sign(self):
        # measured before the guard: the product rule once returned
        # `\int 2x dx + \int x^2 d` for this input
        rec = Differentiation.differentiate('\\int x^2 d x', 'x')
        self.assertFalse(rec['ok'])
        self.assertIn('indefinite', rec['error'])

    def test_embedded_definite_integral_names_the_route(self):
        rec = Differentiation.differentiate(
            'x + \\int_0^{x^2} \\sqrt{1+t^2} d t', 'x')
        self.assertFalse(rec['ok'])
        self.assertIn('its own step', rec['error'])

    def test_leibniz_prefix_on_a_plain_expression(self):
        rec = Differentiation.differentiate(
            '\\frac{d}{d x} (x^2+1)', None)
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['result'], '2x')


class TestFtcDerivativeCheck(unittest.TestCase):
    """The independent leg: quadrature evaluates the integral as a
    function, a central difference differentiates it, and a planted lie
    must be refused with evidence."""

    ASK = '\\int_0^{x^2} \\sqrt{1+t^2} d t'

    def test_correct_derivative_agrees(self):
        check = Differentiation._ftc_derivative_check(
            self.ASK, '2 x \\sqrt{x^{4}+1}', 'x')
        self.assertEqual(check['status'], 'agree')

    def test_missing_chain_factor_disagrees(self):
        check = Differentiation._ftc_derivative_check(
            self.ASK, '\\sqrt{x^{4}+1}', 'x')
        self.assertEqual(check['status'], 'disagree')

    def test_corrupted_coefficient_disagrees(self):
        check = Differentiation._ftc_derivative_check(
            self.ASK, '2.001 x \\sqrt{x^{4}+1}', 'x')
        self.assertEqual(check['status'], 'disagree')

    def test_parameterized_bounds_sample_the_parameter(self):
        check = Differentiation._ftc_derivative_check(
            '\\int_{x}^{a x} t \\, d t', 'a^{2} x - x', 'x')
        self.assertEqual(check['status'], 'agree')
        check = Differentiation._ftc_derivative_check(
            '\\int_{x}^{a x} t \\, d t', 'a^{2} x + x', 'x')
        self.assertEqual(check['status'], 'disagree')

    def test_not_an_integral_is_skipped(self):
        check = Differentiation._ftc_derivative_check('x^2', '2x', 'x')
        self.assertEqual(check['status'], 'skipped')


class TestIntegrateDefinite(unittest.TestCase):
    """FTC part 2 as a narrow move over a recorded antiderivative."""

    def test_evaluates_from_the_recorded_antiderivative(self):
        rec = Integration.integrate_definite(
            '\\int_0^1 x^{2} \\, dx', 'x', '\\frac {1} {3}x^{3} + C')
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['check']['status'], 'agree')
        self.assertEqual(rec['lower'], '0')
        self.assertEqual(rec['upper'], '1')
        # the riding + C cancels in the follow-up expand
        expanded = Core.expand(rec['result'])
        self.assertEqual(
            Core.equal_exprs(expanded['result'],
                             '\\frac{1}{3}')['verdict'], 'yes')

    def test_continuity_is_recorded_not_proved(self):
        rec = Integration.integrate_definite(
            '\\int_0^1 x^{2} \\, dx', 'x', '\\frac {1} {3}x^{3} + C')
        texts = [a['text'] for a in rec['assumptions']]
        self.assertTrue(any('continuous on [0, 1]' in t for t in texts))

    def test_refuses_an_indefinite_spelling(self):
        rec = Integration.integrate_definite(
            '\\int x^{2} \\, dx', 'x', '\\frac {1} {3}x^{3} + C')
        self.assertFalse(rec['ok'])
        self.assertIn('definite integral', rec['error'])

    def test_refuses_a_non_antiderivative(self):
        rec = Integration.integrate_definite(
            '\\int_0^1 x^{2} \\, dx', 'x', 'x^{2} + C')
        self.assertFalse(rec['ok'])
        self.assertIn('not an antiderivative', rec['error'])

    def test_interior_pole_check_refuses(self):
        # -1/x IS an antiderivative of 1/x^2 on each side of 0, so the
        # symbolic leg passes; only the quadrature leg can see that the
        # bounds straddle the pole. The record must carry that refusal.
        rec = Integration.integrate_definite(
            '\\int_{-1}^{1} \\frac{1}{x^{2}} \\, dx', 'x',
            '-\\frac{1}{x} + C')
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['check']['status'], 'disagree')
        self.assertIn('improper', rec['check'].get('reason', ''))


class TestSymbolicConstantPeel(unittest.TestCase):
    """The rational legs' multivariate refusal must not preempt the
    structural constant split (live: the a^2 sin^2 + b^2 cos^2 arctangent
    cell stopped exactly here, in every spelling the agent could reach)."""

    def test_symbolic_factor_in_the_denominator_peels(self):
        rec = Integration.integrate_table(
            '\\frac{1}{a b\\left(v^{2}+1\\right)}', 'v')
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertIn('\\arctan', rec['result'])
        self.assertEqual(rec['check']['status'], 'agree')

    def test_symbolic_factor_in_product_form_peels(self):
        rec = Integration.integrate_table(
            '\\frac{1}{a b}\\frac{1}{v^{2}+1}', 'v')
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertIn('\\arctan', rec['result'])
        self.assertEqual(rec['check']['status'], 'agree')

    def test_symbolic_numerator_over_the_literal_arctan_family(self):
        rec = Integration.integrate_table(
            '\\frac{c}{3u^{2}+\\frac{5}{3}}', 'u')
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['check']['status'], 'agree')

    def test_symbolic_quadratic_coefficients_still_refuse(self):
        # a^2 u^2 + b^2 is NOT one constant split away from the table —
        # the (a/b) substitution is a genuine agent move; the table must
        # not absorb it
        rec = Integration.integrate_table(
            '\\frac{1}{a^{2}u^{2}+b^{2}}', 'u')
        self.assertFalse(rec['ok'])
        self.assertIn('denominator must be constant', rec['error'])

    def test_completed_square_steering_survives_the_fallback(self):
        rec = Integration.integrate_table(
            '\\frac{1}{v^{2}+2v+3}', 'v')
        self.assertFalse(rec['ok'])
        self.assertIn('complete the square', rec['error'])


class TestApproachDirectionRecovery(unittest.TestCase):
    """A bare ^-/^+ marker that precedence bound to an inner factor of a
    compound point must still read as the direction — before this, the
    oracle sampled the `-` as a free variable and checked a two-sided
    limit at a corrupted point (live: every natural spelling of
    x -> pi/2^- misread)."""

    def _parts(self, expr):
        return Limits._limit_parts(expr)

    def test_slash_point_recovers_direction(self):
        p = self._parts('\\lim_{x \\to \\pi/2^-} \\sin x')
        self.assertEqual((p['point_latex'], p['direction']),
                         ('\\pi /2', 'left'))

    def test_braced_marker_recovers_direction(self):
        p = self._parts('\\lim_{x \\to \\pi/2^{-}} \\sin x')
        self.assertEqual((p['point_latex'], p['direction']),
                         ('\\pi /2', 'left'))

    def test_frac_point_recovers_direction(self):
        p = self._parts('\\lim_{x \\to \\frac{\\pi}{2}^{-}} \\sin x')
        self.assertEqual((p['point_latex'], p['direction']),
                         ('\\frac {\\pi} {2}', 'left'))

    def test_negative_point_recovers_direction(self):
        p = self._parts('\\lim_{x \\to -1^-} x')
        self.assertEqual((p['point_latex'], p['direction']),
                         ('-1', 'left'))

    def test_powered_endpoint_stays_two_sided(self):
        p = self._parts('\\lim_{x \\to a^2} x')
        self.assertEqual((p['point_latex'], p['direction']),
                         ('a^{2}', 'two-sided'))


class TestLimitEvaluate(unittest.TestCase):
    """Agent proposes a limit value; the approach oracle verifies."""

    LIM = ('\\lim_{x \\to \\pi/2^{-}} \\left(\\frac {1} {ab}\\arctan'
           '\\left (( \\frac {a} {b}( \\tan x)) \\right )+C\\right)')

    def test_certifies_the_endpoint_composite(self):
        rec = Limits.limit_evaluate(self.LIM, '\\frac{\\pi}{2|ab|}+C')
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['check']['status'], 'agree')
        self.assertIn('agent-proposed', rec['check']['method'])
        self.assertEqual(rec['direction'], 'left')

    def test_refuses_the_sign_wrong_value(self):
        rec = Limits.limit_evaluate(self.LIM, '\\frac{\\pi}{2ab}+C')
        self.assertFalse(rec['ok'])
        self.assertIn('was not confirmed', rec['error'])

    def test_refuses_a_value_containing_the_variable(self):
        rec = Limits.limit_evaluate('\\lim_{x \\to 0^+} x', 'x')
        self.assertFalse(rec['ok'])
        self.assertIn('bound variable', rec['error'])

    def test_simple_limit_still_certifies(self):
        rec = Limits.limit_evaluate('\\lim_{x \\to \\infty} \\arctan x',
                                    '\\frac{\\pi}{2}')
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['check']['status'], 'agree')


class TestIntegrateDefiniteEndpointDoor(unittest.TestCase):
    """A recorded one-sided limit stands in for substitution at a bound
    where the antiderivative's spelling is singular."""

    EXPR = ('\\int_{0}^{\\pi/2}\\frac{1}{a^{2}\\sin^{2}x'
            '+b^{2}\\cos^{2}x}\\,dx')
    F = ('\\frac {1} {ab}\\arctan\\left (( \\frac {a} {b}( \\tan x)) '
         '\\right )+C')

    def test_upper_limit_value_replaces_substitution(self):
        rec = Integration.integrate_definite(
            self.EXPR, 'x', self.F,
            upper_limit='\\frac{\\pi}{2|ab|}+C')
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertIn('\\frac{\\pi}{2|ab|}+C', rec['result'])
        self.assertEqual(rec['check']['status'], 'agree')
        texts = ' '.join(a['text'] for a in rec['assumptions'])
        self.assertIn('extends continuously', texts)

    def test_endpoint_value_with_the_variable_refused(self):
        rec = Integration.integrate_definite(
            self.EXPR, 'x', self.F, upper_limit='\\tan x')
        self.assertFalse(rec['ok'])
        self.assertIn('still contains', rec['error'])

    def test_plain_path_unchanged(self):
        rec = Integration.integrate_definite(
            '\\int_0^1 x^{2} \\, dx', 'x', '\\frac {1} {3}x^{3} + C')
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertNotIn('upper_limit', rec['args'])


class TestIntegralBearingLimitBodies(unittest.TestCase):
    """The approach oracle evaluates a variable-bound definite integral
    by graded quadrature — a capability of the LIMITS oracle leg only,
    so a limit ABOUT an integral is checkable while the integral's own
    value still closes only through the integration tactics."""

    TARGET = ('\\lim_{x \\to 0^{+}} x \\int_x^1 '
              '\\frac{\\cos t}{t^2} d t')

    def test_the_motivating_limit_certifies(self):
        rec = Limits.limit_evaluate(self.TARGET, '1')
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['check']['status'], 'agree')

    def test_a_wrong_value_is_refused_with_the_estimate(self):
        rec = Limits.limit_evaluate(self.TARGET, '2')
        self.assertFalse(rec['ok'])
        self.assertIn('was not confirmed', rec['error'])

    def test_the_two_sided_spelling_stays_honestly_open(self):
        # for x < 0 the integral crosses the non-integrable pole at 0,
        # so the left approach cannot evaluate: refuse, never certify
        rec = Limits.limit_evaluate(
            '\\lim _{x \\rightarrow 0} x \\int_x^1 '
            '\\frac{\\cos t}{t^2} d t', '1')
        self.assertFalse(rec['ok'])
        self.assertIn('did not converge', rec['error'])

    def test_lhopital_sees_the_form_through_the_integral(self):
        # the form gate samples the integral numerator; the derivative
        # of the numerator is gen 75's FTC bound rule
        rec = Limits.limit_lhopital(
            '\\lim_{x \\to 0^{+}} \\frac{\\int_x^1 '
            '\\frac{\\cos t}{t^2} d t}{\\frac{1}{x}}')
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['indeterminate_form'], 'infinity/infinity')
        self.assertIn('\\cos', rec['result'])
        self.assertEqual(rec['check']['status'], 'agree')

    def test_frac_differential_spelling_evaluates(self):
        rec = Limits.limit_evaluate(
            '\\lim_{x \\to 0^{+}} x \\int_x^1 '
            '\\frac{\\cos t \\, d t}{t^2}', '1')
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['check']['status'], 'agree')

    def test_at_infinity_integral_body(self):
        rec = Limits.limit_evaluate(
            '\\lim_{x \\to \\infty} \\int_1^x \\frac{1}{t^2} d t', '1')
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['check']['status'], 'agree')

    def test_parameters_sample_through_the_integral(self):
        rec = Limits.limit_evaluate(
            '\\lim_{x \\to 0^{+}} x \\int_x^1 \\frac{c}{t^2} d t', 'c')
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['check']['status'], 'agree')

    def test_record_replays(self):
        ledger = Ledger()
        ledger.record(Limits.limit_evaluate(self.TARGET, '1'))
        self.assertEqual(ledger.replay()['status'], 'verified')

    def test_evaluator_declines_integral_free_and_indefinite_bodies(self):
        s, n = P.parse_latex('x^2 + 1')
        self.assertIsNone(P.definite_integral_evaluator(s, n))
        s, n = P.parse_latex('x + \\int t \\, dt')
        self.assertIsNone(P.definite_integral_evaluator(s, n))

    def test_quadrature_refuses_a_domain_break(self):
        fs, fn = P.parse_latex('\\frac{1}{t^{2}}')
        with self.assertRaises(P.EvalError):
            P._graded_quadrature(fs, fn, 't', -1.0, 1.0, {})

    def test_general_equal_deliberately_does_not_gain_this(self):
        # the boundary that keeps the door meaningful: equal? must not
        # start deciding integral values numerically, or a sampled
        # proposal could close a definite integral around
        # integrate_definite/integrate_improper
        rec = Core.equal_exprs('\\int_0^1 t^2 \\, dt', '\\frac{1}{3}')
        self.assertEqual(rec['verdict'], 'unknown')


class TestNumericImproperCheck(unittest.TestCase):
    """The graded truncation-ladder quadrature leg: uniform Simpson to a
    cut, then geometric slabs toward the singular bound, iterated Aitken
    over the rung values, refusing only past its own measured residual."""

    EX = '\\int_0^1 \\frac{1}{(2-x) \\sqrt{1-x}} \\, dx'

    def test_certifies_the_definitional_value(self):
        check = P.numeric_improper_check(self.EX, 'x', 'upper',
                                         '\\frac{\\pi}{2}')
        self.assertEqual(check['status'], 'agree')
        self.assertIn('truncation quadrature', check['method'])

    def test_refuses_a_planted_lie(self):
        check = P.numeric_improper_check(self.EX, 'x', 'upper',
                                         '\\frac{\\pi}{2} + 0.01')
        self.assertEqual(check['status'], 'disagree')

    def test_refuses_a_small_corruption(self):
        # 0.05% above pi/2 — must clear the leg's own error bar
        check = P.numeric_improper_check(self.EX, 'x', 'upper',
                                         '1.5715817')
        self.assertEqual(check['status'], 'disagree')

    def test_divergent_ladder_is_never_evidence(self):
        check = P.numeric_improper_check(
            '\\int_0^1 \\frac{1}{x} \\, dx', 'x', 'lower', '5')
        self.assertEqual(check['status'], 'skipped')
        self.assertIn('never evidence of divergence', check['reason'])

    def test_interior_break_is_a_witness(self):
        # declared upper-singular, but the real break is at x = 0: the
        # oriented probe walks the declared bound LAST, so the witness
        # is never the declared singularity itself
        check = P.numeric_improper_check(
            '\\int_{-1}^{1} \\frac{1}{x^{2}} \\, dx', 'x', 'upper', '5')
        self.assertEqual(check['status'], 'disagree')
        self.assertIn('away from the declared improper bound',
                      check['reason'])

    def test_parameters_are_sampled(self):
        check = P.numeric_improper_check(
            '\\int_0^1 \\frac{c}{\\sqrt{1-x}} \\, dx', 'x', 'upper', '2c')
        self.assertEqual(check['status'], 'agree')
        self.assertGreater(check['samples'], 1)

    def test_non_singular_integrand_still_certifies(self):
        # the definitional reading is true of a proper integral too —
        # the door must not manufacture a false refusal there
        check = P.numeric_improper_check(
            '\\int_0^1 x^{2} \\, dx', 'x', 'upper', '\\frac{1}{3}')
        self.assertEqual(check['status'], 'agree')


class TestIntegrateImproper(unittest.TestCase):
    """The infinite-object door for integrals: an endpoint-singular
    integral is read as the limit of its truncated integrals, closed
    from a recorded truncated evaluation and a recorded one-sided
    limit."""

    EX = '\\int_0^1 \\frac{d x}{(2-x) \\sqrt{1-x}}'
    TR = '\\int_0^t \\frac{1}{(2-x) \\sqrt{1-x}} \\, d x'
    G = ('\\left(-2 \\arctan\\sqrt{1-t}\\right) - '
         '\\left(-2 \\arctan\\sqrt{1}\\right)')

    def test_closes_the_endpoint_singular_integral(self):
        rec = Integration.integrate_improper(
            self.EX, 'x', self.TR, self.G, '\\frac{\\pi}{2}')
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['result'], '\\frac{\\pi}{2}')
        self.assertEqual(rec['check']['status'], 'agree')
        self.assertEqual(rec['singular'], 'upper')
        self.assertEqual(rec['bound_var'], 't')

    def test_reading_and_continuity_are_recorded_not_proved(self):
        rec = Integration.integrate_improper(
            self.EX, 'x', self.TR, self.G, '\\frac{\\pi}{2}')
        texts = ' '.join(a['text'] for a in rec['assumptions'])
        self.assertIn('definitional limit', texts)
        self.assertIn('\\lim_{t \\to 1^{-}}', texts)
        self.assertIn('continuous on [0, 1)', texts)

    def test_lower_side_case(self):
        rec = Integration.integrate_improper(
            '\\int_0^1 \\frac{1}{\\sqrt{x}} \\, dx', 'x',
            '\\int_s^1 \\frac{1}{\\sqrt{x}} \\, dx',
            '2 - 2\\sqrt{s}', '2')
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['check']['status'], 'agree')
        self.assertEqual(rec['singular'], 'lower')
        texts = ' '.join(a['text'] for a in rec['assumptions'])
        self.assertIn('continuous on (0, 1]', texts)

    def test_a_wrong_value_is_refused_by_the_ladder(self):
        rec = Integration.integrate_improper(
            self.EX, 'x', self.TR, self.G, '\\frac{\\pi}{3}')
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['check']['status'], 'disagree')

    def test_refuses_a_non_fresh_truncation_variable(self):
        rec = Integration.integrate_improper(
            self.EX, 'x',
            '\\int_0^x \\frac{1}{(2-x) \\sqrt{1-x}} \\, d x',
            self.G, '\\frac{\\pi}{2}')
        self.assertFalse(rec['ok'])
        self.assertIn('must be fresh', rec['error'])

    def test_infinite_upper_bound_closes(self):
        # the infinite door (gen 78): the truncation point runs away
        # instead of approaching a finite singularity
        rec = Integration.integrate_improper(
            '\\int_1^{\\infty} \\frac{1}{x^{2}} \\, dx', 'x',
            '\\int_1^t \\frac{1}{x^{2}} \\, dx',
            '1 - \\frac{1}{t}', '1')
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['check']['status'], 'agree')
        self.assertEqual(rec['improper_kind'], 'infinite')
        texts = ' '.join(a['text'] for a in rec['assumptions'])
        self.assertIn('infinite upper bound', texts)
        self.assertIn('continuous on [1, \\infty)', texts)
        self.assertIn('\\lim_{t \\to \\infty}', texts)

    def test_plus_infinity_spelling_routes_to_the_infinite_door(self):
        # \int_0^{+\infty}: the PLUS wrapper must read as infinity — it
        # previously read as a finite symbol and the whole record
        # admitted with a skipped check
        rec = Integration.integrate_improper(
            '\\int_1^{+\\infty} \\frac{1}{x^{2}} \\, dx', 'x',
            '\\int_1^s \\frac{1}{x^{2}} \\, dx',
            '\\left(-\\frac{1}{s}\\right) - \\left(-\\frac{1}{1}\\right)',
            '1')
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['check']['status'], 'agree')
        self.assertEqual(rec['improper_kind'], 'infinite')

    def test_infinite_wrong_value_is_refused(self):
        rec = Integration.integrate_improper(
            '\\int_1^{+\\infty} \\frac{1}{x^{2}} \\, dx', 'x',
            '\\int_1^s \\frac{1}{x^{2}} \\, dx',
            '\\left(-\\frac{1}{s}\\right) - \\left(-\\frac{1}{1}\\right)',
            '2')
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['check']['status'], 'disagree')

    def test_infinite_lower_bound_closes(self):
        rec = Integration.integrate_improper(
            '\\int_{-\\infty}^0 e^{x} \\, d x', 'x',
            '\\int_s^0 e^{x} \\, d x', 'e^{0} - e^{s}', '1')
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['check']['status'], 'agree')
        texts = ' '.join(a['text'] for a in rec['assumptions'])
        self.assertIn('continuous on (-\\infty, 0]', texts)
        self.assertIn('\\lim_{s \\to -\\infty}', texts)

    def test_divergent_infinite_ladder_never_certifies(self):
        rec = Integration.integrate_improper(
            '\\int_1^{+\\infty} \\frac{1}{x} \\, dx', 'x',
            '\\int_1^s \\frac{1}{x} \\, dx', '\\ln s - \\ln 1', '5')
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['check']['status'], 'skipped')
        self.assertIn('never evidence of divergence',
                      rec['check']['reason'])

    def test_both_infinite_bounds_refused(self):
        rec = Integration.integrate_improper(
            '\\int_{-\\infty}^{+\\infty} \\frac{1}{1+x^{2}} \\, dx', 'x',
            '\\int_s^{+\\infty} \\frac{1}{1+x^{2}} \\, dx', 'g', '\\pi')
        self.assertFalse(rec['ok'])
        self.assertIn('both bounds are infinite', rec['error'])

    def test_replacing_the_finite_bound_is_refused(self):
        rec = Integration.integrate_improper(
            '\\int_1^{+\\infty} \\frac{1}{x^{2}} \\, dx', 'x',
            '\\int_s^{+\\infty} \\frac{1}{x^{2}} \\, dx',
            '\\frac{1}{s}', '1')
        self.assertFalse(rec['ok'])
        self.assertIn('must replace the infinite upper bound',
                      rec['error'])

    def test_hyperbolic_alias_spelling_closes_the_flagship(self):
        # \operatorname{ch} normalizes to \cosh at the lexer, so the
        # continental spelling and the canonical one are one expression
        self.assertTrue(P.same_expression(
            '\\operatorname{ch}^{n+1} x', '\\cosh^{n+1} x'))
        rec = Integration.integrate_substitute(
            '\\int \\frac{1}{\\operatorname{ch} x} \\, d x', 'x',
            '\\sinh x', 'u', '\\frac{1}{1+u^{2}}')
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['check']['status'], 'agree')

    def test_u_free_constant_integrand_is_checked_not_refused(self):
        rec = Integration.integrate_substitute(
            '\\int \\frac{1}{\\cosh^{2} x} \\, d x', 'x',
            '\\tanh x', 'u', '1')
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['check']['status'], 'agree')
        wrong = Integration.integrate_substitute(
            '\\int \\frac{1}{\\cosh^{2} x} \\, d x', 'x',
            '\\tanh x', 'u', '2')
        self.assertFalse(wrong['ok'])
        self.assertIn('does not equal the integrand', wrong['error'])

    def test_overflow_saturates_only_inside_the_mode(self):
        s, n = P.parse_latex('\\frac{1}{\\cosh^{3} x}')
        with P._overflow_saturation():
            self.assertEqual(P.numeric_eval(s, n, {'x': 300.0}), 0.0)
        with self.assertRaises(OverflowError):
            P.numeric_eval(s, n, {'x': 300.0})

    def test_growth_bodies_still_refuse_at_infinity(self):
        # saturation must never leak a false green: a genuinely growing
        # body raises inside the guarded evaluator and the proposal is
        # refused, exactly as before
        rec = Limits.limit_evaluate(
            '\\lim_{x \\to \\infty} e^{e^{x}}', '5')
        self.assertFalse(rec['ok'])
        rec = Limits.limit_evaluate(
            '\\lim_{T \\to \\infty} \\int_0^T '
            '\\frac{1}{\\cosh^{3} x} \\, d x', '\\frac{\\pi}{4}')
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['check']['status'], 'agree')

    def test_refuses_when_no_bound_is_replaced(self):
        rec = Integration.integrate_improper(
            self.EX, 'x',
            '\\int_0^1 \\frac{1}{(2-x) \\sqrt{1-x}} \\, dx',
            self.G, '\\frac{\\pi}{2}')
        self.assertFalse(rec['ok'])
        self.assertIn('fresh variable', rec['error'])

    def test_refuses_a_literal_truncation_bound(self):
        rec = Integration.integrate_improper(
            self.EX, 'x',
            '\\int_0^2 \\frac{1}{(2-x) \\sqrt{1-x}} \\, dx',
            self.G, '\\frac{\\pi}{2}')
        self.assertFalse(rec['ok'])
        self.assertIn('bare fresh variable', rec['error'])

    def test_refuses_a_mismatched_integrand(self):
        rec = Integration.integrate_improper(
            self.EX, 'x', '\\int_0^t \\frac{1}{\\sqrt{1-x}} \\, dx',
            self.G, '\\frac{\\pi}{2}')
        self.assertFalse(rec['ok'])
        self.assertIn('integrand differs', rec['error'])

    def test_refuses_the_truncation_variable_in_the_value(self):
        rec = Integration.integrate_improper(
            self.EX, 'x', self.TR, self.G, '\\frac{\\pi}{2} + t')
        self.assertFalse(rec['ok'])
        self.assertIn('truncation variable', rec['error'])


class TestSideConditionSplit(unittest.TestCase):
    """A trailing bracketed relation in an expr-command argument is a
    stated side condition, not a factor — as a factor it poisoned goal
    coverage (live: a perfect derivation failed designation)."""

    def test_the_live_cell_splits(self):
        import expr_commands as ec
        core, cond = ec._split_side_condition(
            '\\int_{0}^{\\pi /{2}}\\frac  {dx} '
            '{a^{2} \\sin^{2}x+b^{2} \\cos^{2}x}\\ (ab \\ne {0})')
        self.assertEqual(cond, 'ab \\ne{0}')
        self.assertTrue(P.covers_goal(
            '\\int_0^{\\pi/2} \\frac{1}{a^2 \\sin^2 x + b^2 \\cos^2 x}'
            ' \\, dx', core, establishes=True))

    def test_plain_arguments_do_not_split(self):
        import expr_commands as ec
        for arg in ['\\int_0^1 x^2 \\, dx', 'x+2 = 7', '(x+1)(x+2)',
                    '\\{(-1,2),(1,-2)\\}']:
            core, cond = ec._split_side_condition(arg)
            self.assertIsNone(cond, arg)
            self.assertEqual(core, arg)

    def test_simple_product_with_condition_splits(self):
        import expr_commands as ec
        core, cond = ec._split_side_condition('2x\\ (x \\gt 0)')
        self.assertEqual(cond, 'x \\gt 0')
        self.assertEqual(core, '2x')


class TestEstablishesGoalCoverage(unittest.TestCase):
    """covers_goal's admission question: a bare body never establishes a
    value-bearing binder."""

    def test_body_does_not_establish_a_definite_integral(self):
        goal = '\\int_0^1 x^{2} \\, dx'
        self.assertTrue(P.covers_goal('x^{2}', goal))
        self.assertFalse(P.covers_goal('x^{2}', goal, establishes=True))

    def test_body_does_not_establish_a_limit(self):
        goal = '\\lim_{x \\to 2} \\frac{x^2-4}{x-2}'
        body = '\\frac{x^2-4}{x-2}'
        self.assertTrue(P.covers_goal(body, goal))
        self.assertFalse(P.covers_goal(body, goal, establishes=True))

    def test_integrand_still_establishes_the_indefinite_integral(self):
        # the honest antiderivative chain roots at its integrand; the
        # indefinite branches are untouched by the tightening
        self.assertTrue(P.covers_goal('x^{2}', '\\int x^{2} \\, dx',
                                      establishes=True))

    def test_identity_still_establishes(self):
        goal = '\\int_0^1 x^{2} \\, dx'
        self.assertTrue(P.covers_goal(goal, goal, establishes=True))

    def test_definite_spellings_cover_each_other(self):
        # the textbook and canonical spellings of ONE definite integral —
        # an honest FTC chain was refused purely on this respelling
        textbook = ('\\int_{0}^{\\pi /{2}}\\frac  {dx} '
                    '{a^{2} \\sin^{2}x+b^{2} \\cos^{2}x}')
        canonical = ('\\int_0^{\\pi/2} \\frac{1}{a^2 \\sin^2 x '
                     '+ b^2 \\cos^2 x} \\, dx')
        self.assertTrue(P.covers_goal(canonical, textbook,
                                      establishes=True))
        self.assertTrue(P.covers_goal(textbook, canonical,
                                      establishes=True))

    def test_different_bounds_do_not_cover(self):
        self.assertFalse(P.covers_goal(
            '\\int_0^1 x^{2} \\, dx', '\\int_0^2 x^{2} \\, dx',
            establishes=True))


class TestFunctionArgumentSpanIsNotErasedByStripping(unittest.TestCase):
    # Function application is a READING CONVENTION over a flat P_LIST, not a
    # DAG node, so the argument's grouping is the ONLY thing distinguishing
    # `\cos(x) y` from `\cos x y`. Stripping it made both normal forms print
    # `\cos xy`, and every consumer of a normal form read one expression
    # where there are two -- with the numeric legs disagreeing (0.449 vs
    # 0.990). Spans come from the oracle's own _func_arg_span, so what the
    # normal form re-encodes is the reading the numeric leg checks against.

    CAPTURE_PAIRS = [
        ('\\cos\\left(x\\right) y', '\\cos x y'),
        ('\\cos{x} y', '\\cos x y'),
        ('\\ln\\left(x\\right)x', '\\ln x x'),
        ('\\sin\\left(t\\right)b', '\\sin t b'),
    ]

    RESPELLINGS = [
        ('\\cos x', '\\cos{x}'),
        ('\\cos x', '\\cos\\left(x\\right)'),
        ('\\sin x \\cos x', '\\sin\\left(x\\right)\\cos\\left(x\\right)'),
        ('2 \\sin x \\cos x', '2 \\sin(x)\\cos(x)'),
        ('\\sin 2x', '\\sin(2x)'),
    ]

    def test_capture_is_not_the_same_expression(self):
        for a, b in self.CAPTURE_PAIRS:
            self.assertFalse(P.same_expression(a, b), f'{a} vs {b}')

    def test_capture_does_not_chain_link(self):
        from ledger import _chain_links
        for a, b in self.CAPTURE_PAIRS:
            self.assertFalse(_chain_links(a, b), f'{a} vs {b}')

    def test_capture_is_not_the_same_spelling(self):
        # the provenance comparator behind the endpoint-limit door
        from tactic_registry import _same_spelling
        for a, b in self.CAPTURE_PAIRS:
            self.assertFalse(_same_spelling(a, b), f'{a} vs {b}')

    def test_the_capture_pairs_really_do_differ_numerically(self):
        # the whole point: these are not two spellings of one expression
        for a, b in self.CAPTURE_PAIRS:
            env = {'x': 0.3, 'y': 0.47, 't': 0.3, 'b': 0.47}
            sa, na = P.parse_latex(a)
            sb, nb = P.parse_latex(b)
            self.assertNotAlmostEqual(P.numeric_eval(sa, na, env),
                                      P.numeric_eval(sb, nb, env),
                                      places=6, msg=f'{a} vs {b}')

    def test_honest_respellings_still_compare_equal(self):
        # the comparator exists to tolerate agents retyping without the
        # decorative wrappers; that must keep working
        for a, b in self.RESPELLINGS:
            self.assertEqual(P._all_bracket_normal_form(a),
                             P._all_bracket_normal_form(b), f'{a} vs {b}')

    def test_sum_boundaries_were_never_affected(self):
        from ledger import _chain_links
        self.assertFalse(_chain_links('(a+b)c', 'a+bc'))
        self.assertEqual(P._all_bracket_normal_form('(a+b)c'),
                         P._all_bracket_normal_form('\\left(a+b\\right)c'))


class TestWrittenLatexReadsBackAsItsSource(unittest.TestCase):
    # write_latex used to compare its two candidate spellings only against
    # each other, which assumes at least one of them round-trips. Measured
    # false: the raw writer spells `\sin 2x` as `\sin {2}x`, whose integer
    # value repr closes the argument span, so the string re-reads as
    # `sin(2) x`. Each candidate is now compared against the SOURCE graph.

    ROUND_TRIP = ['\\sin 2x', '\\cos 3y', '\\sin 2x \\cos x',
                  '2 \\sin x \\cos x', '\\sin(2x)', '\\ln 2x']

    def test_output_reads_back_as_the_expression_written(self):
        for latex in self.ROUND_TRIP:
            sym, notation = P.parse_latex(latex)
            out = P.write_latex(sym, notation)
            self.assertEqual(P._normal_form(out),
                             P._dag_normal_form(sym, notation), latex)

    def test_numeric_agreement_after_a_write_hop(self):
        env = {'x': 0.4, 'y': 0.4}
        for latex in self.ROUND_TRIP:
            sym, notation = P.parse_latex(latex)
            out = P.write_latex(sym, notation)
            sym2, notation2 = P.parse_latex(out)
            self.assertAlmostEqual(P.numeric_eval(sym, notation, env),
                                   P.numeric_eval(sym2, notation2, env),
                                   places=12, msg=latex)

    def test_the_numeric_leading_argument_keeps_its_span(self):
        sym, notation = P.parse_latex('\\sin 2x')
        self.assertEqual(P.write_latex(sym, notation), '\\sin 2x')


class TestChainedComparison(unittest.TestCase):
    # gen 70 (contrarian): the "between" answer a solve! run needs to state
    # is spelled `-1 < x < 1` by every user and every textbook, and that was
    # a SYNTAX ERROR. The backlog item asking for conjunction solutions
    # proposed C_LIST-inside-O_LIST and never noticed that the parser
    # already speaks \land natively, nor that the user's own spelling did
    # not parse at all. A chain desugars to the A_LIST the parser has, so
    # no new node, writer, or Replicator dispatch is involved.

    def chain(self, latex):
        sym, notation = P.parse_latex(latex)
        return P.write_latex(sym, notation)

    def test_the_spelling_a_user_writes_parses(self):
        self.assertEqual(self.chain('-1 \\lt x \\lt 1'),
                         '-1 \\lt x \\land x \\lt 1')
        self.assertEqual(self.chain('-1 < x < 1'),
                         '-1 \\lt x \\land x \\lt 1')

    def test_mixed_strictness(self):
        self.assertEqual(self.chain('0 \\le x \\lt 5'),
                         '0 \\le x \\land x \\lt 5')

    def test_it_is_an_a_list_of_two_relations(self):
        sym, notation = P.parse_latex('-1 \\lt x \\lt 1')
        f = notation.getf(sym, Notation.A_LIST)
        self.assertIsNotNone(f)
        self.assertEqual(len(f.args), 2)
        for member in f.args:
            self.assertIsNotNone(notation.getf(member, Notation.COMP))

    def test_non_inequality_comparers_chain_too(self):
        self.assertEqual(self.chain('a = b = c'), 'a=b \\land b=c')

    def test_binder_arrows_are_untouched(self):
        # `\to` is a comparer, so the limit binder is the case that would
        # break if the chain rule reached inside a subscript
        r = Limits.limit_table('\\lim_{x \\to 0} \\frac{\\sin x}{x}')
        self.assertTrue(r['ok'], r.get('error'))
        self.assertEqual(r['result'], '1')

    def test_the_union_oracle_takes_the_conjunction_shape(self):
        # Gen 71 supplies the mechanism half after gen 70 supplied the
        # user's chained-comparison spelling: a disjunct is AND of members.
        r = P.numeric_union_check('x^2 \\lt 1', ['-1 \\lt x \\lt 1'])
        self.assertEqual(r['status'], 'agree')


class TestDerivativeKeepsTheDenominatorAsWritten(unittest.TestCase):
    # The quotient rule knows its denominator is a power of the one it was
    # handed; canonicalizing expanded that away, so d/dx x/(x^2+1)^3 printed
    # a degree-8 polynomial where the textbook keeps (x^2+1)^4. The
    # re-spelling runs AFTER canonicalization (so cancellation has already
    # happened) and is admitted only on the EXACT identity base**k == den.
    # Nothing is factored: the base is READ off the input, never discovered.

    def d(self, expr, var='x'):
        r = Differentiation.differentiate(expr, var)
        self.assertTrue(r['ok'], r.get('error'))
        self.assertEqual(r['check']['status'], 'agree', expr)
        return r['result']

    def test_polyrat_path_keeps_the_base(self):
        self.assertEqual(self.d('\\frac{x}{x^2+1}'),
                         '\\frac {-x^{2}+1} {\\left (x^{2}+1 \\right )^{2}}')

    def test_power_denominator_gains_exactly_one(self):
        self.assertEqual(self.d('\\frac{x}{(x^2+1)^3}'),
                         '\\frac {-5x^{2}+1} {\\left (x^{2}+1 \\right )^{4}}')
        self.assertEqual(self.d('\\frac{1}{(x+1)^2}'),
                         '\\frac {-2} {\\left (x+1 \\right )^{3}}')

    def test_rules_path_gets_it_too(self):
        # the rules path canonicalizes through core.expand, so it had the
        # same expanded denominator -- one mechanism has to serve both
        self.assertEqual(self.d('\\frac{e^x}{x^2+1}'),
                         '\\frac {x^{2}e^x-2xe^x+e^x} '
                         '{\\left (x^{2}+1 \\right )^{2}}')

    def test_symbolic_coefficients(self):
        self.assertEqual(self.d('\\frac{a}{bx+c}'),
                         '\\frac {-ab} {\\left (bx+c \\right )^{2}}')

    def test_a_monomial_denominator_is_left_alone(self):
        # x^2 is both canonical and the shorter spelling; (x)^2 is a loss
        self.assertTrue(self.d('\\frac{\\sin x}{x}').endswith('{x^{2}}'))

    def test_cancellation_still_wins(self):
        # runs after canonicalization, so a derivative that collapses to a
        # non-fraction is never dressed back up as one
        self.assertEqual(self.d('\\frac{x^2}{x}'), '1')
        self.assertEqual(self.d('\\frac{x^2-1}{x-1}'), '1')

    def test_a_non_sum_base_is_not_invented(self):
        # (x+1)(x+2) would need factor tracking to re-present; refusing to
        # guess is the invariant (no general `factor`)
        self.assertTrue(
            self.d('\\frac{x}{(x+1)(x+2)}').endswith(
                '{x^{4}+6x^{3}+13x^{2}+12x+4}'))

    def test_both_spellings_are_the_same_expression(self):
        new = self.d('\\frac{x}{x^2+1}')
        old = '\\frac {-x^{2}+1} {x^{4}+2x^{2}+1}'
        eq = Core.equal_exprs(new, old)
        self.assertEqual(eq.get('verdict'), 'yes')
        # and expand still converges them onto one canonical form, so the
        # composite glue and chain linkage are unaffected
        self.assertEqual(Core.expand(new)['result'],
                         Core.expand(old)['result'])

    def test_recorded_form_is_flagged(self):
        r = Differentiation.differentiate('\\frac{x}{x^2+1}', 'x')
        self.assertEqual(r.get('denominator'), 'as written')
        plain = Differentiation.differentiate('\\frac{\\sin x}{x}', 'x')
        self.assertIsNone(plain.get('denominator'))

    def test_the_ftc_door_still_matches_the_integrand(self):
        # integrate_definite re-differentiates the antiderivative and
        # compares by equal_exprs; that comparison must survive the new
        # spelling in BOTH directions
        for integrand in ('\\frac{-x^{2}+1}{\\left(x^{2}+1\\right)^{2}}',
                          '\\frac{-x^{2}+1}{x^{4}+2x^{2}+1}'):
            r = Integration.integrate_definite(
                f'\\int_0^1 {integrand} \\, dx', 'x', '\\frac{x}{x^{2}+1}')
            self.assertTrue(r['ok'], integrand)
            self.assertEqual(r['check']['status'], 'agree', integrand)


class TestDelimiterRedundantBracketsAreDropped(unittest.TestCase):
    # The rule builders parenthesize their factors for syntax protection.
    # Inside a {} slot -- a \sqrt argument, an INDEX dimension, a \frac or
    # \binom argument -- the braces already bound the span, so the wrapper
    # cannot be doing work and does not belong in the ledger artifact.

    def written(self, latex):
        sym, notation = P.parse_latex(latex)
        return P.write_latex(sym, notation)

    def test_sqrt_argument(self):
        self.assertEqual(self.written('\\sqrt{\\left(x^{2}+1\\right)}'),
                         '\\sqrt{x^{2}+1}')

    def test_index_dimension(self):
        self.assertEqual(self.written('e^{\\left(2x\\right)}'), 'e^{2x}')

    def test_fraction_argument(self):
        self.assertEqual(self.written('\\frac{\\left(x+1\\right)}{2}'),
                         '\\frac {x+1} {2}')

    def test_a_function_argument_is_NOT_peeled(self):
        # `\cos(x)` is the capture-prone case: a following ordinary factor
        # joins the argument span, so `\cos(x) y` is not `\cos x y`
        # (MEASURED: they evaluate to 0.449 and 0.990). Only the delimited
        # slots above are provably safe.
        self.assertEqual(self.written('\\cos\\left(x\\right) y'),
                         '\\cos\\left (x \\right )y')

    def test_semantic_brackets_survive(self):
        for latex in ('\\sqrt{\\left|x\\right|}', '\\frac{|x|}{2}'):
            sym, notation = P.parse_latex(latex)
            out = P.write_latex(sym, notation)
            self.assertIn('|', out, latex)

    def test_peeling_preserves_the_expression(self):
        env = {'x': 0.6}
        for latex in ('\\sqrt{\\left(x^{2}+1\\right)}', 'e^{\\left(2x\\right)}',
                      '\\frac{\\left(x+1\\right)}{2}', '\\sqrt{\\left|x\\right|}'):
            sym, notation = P.parse_latex(latex)
            out = P.write_latex(sym, notation)
            sym2, notation2 = P.parse_latex(out)
            self.assertAlmostEqual(P.numeric_eval(sym, notation, env),
                                   P.numeric_eval(sym2, notation2, env),
                                   places=12, msg=latex)


if __name__ == '__main__':
    unittest.main()

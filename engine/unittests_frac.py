#!/usr/bin/env python3
"""
Unit tests for fraction operations (Rules 1-5).

Tests cover both symbolic and numeric fraction operations using explicit
mul! and add! commands to verify the implementation of:
- Rule 1: Fraction + Fraction
- Rule 2: Fraction × Fraction
- Rule 3: Scalar × Fraction
- Rule 4: Fraction × Sum
- Rule 5: Scalar + Fraction
"""

import unittest
from comparer import *
from processor import MathProcessor
from preprocessor import Preprocessor
from notation import Func, Symbol

def check(expr1, expr2):
    """
    Process expr1 and compare result with expr2.
    Returns True if they match.
    """
    def unwrap_group(sym, notation):
        """Strip redundant GROUP wrappers for comparison."""
        f = notation.getf(sym, Notation.GROUP)
        if f is not None:
            return unwrap_group(f.args[0], notation)
        return sym

    def canonicalize_values(sym, notation):
        """Convert Value-based fractions into symbolic \\frac nodes for comparison."""
        f = notation.get(sym)
        if f is None:
            if isinstance(sym, FracValue):
                frac_sym = notation.setf(
                    Symbol("\\frac"),
                    (
                        notation.setf(Notation.GROUP, (IntegerValue(sym.num),), br="{}"),
                        notation.setf(Notation.GROUP, (IntegerValue(sym.denom),), br="{}"),
                    ),
                )
                return notation.setf(Notation.MINUS, (frac_sym,)) if sym.num < 0 else frac_sym
            return sym
        new_args = tuple(canonicalize_values(arg, notation) for arg in f.args)
        return notation.repf(sym, Func(f.sym, new_args, **f.props))

    notation1 = Notation()
    p1 = MathParser(notation1)
    sym1 = p1.parse(expr1)
    processor = MathProcessor()
    outsym, notation = processor(sym1, notation1, {}, {})
    norm_notation = Notation()
    norm_pre = Preprocessor(notation, norm_notation, {}, {})
    outsym = norm_pre(outsym)
    outsym = unwrap_group(outsym, norm_notation)
    outsym = canonicalize_values(outsym, norm_notation)
    notation2 = Notation()
    p2 = MathParser(notation2)
    sym2 = p2.parse(expr2)
    notation3 = Notation()
    preprocessor = Preprocessor(notation2, notation3, {}, {})
    sym3 = preprocessor(sym2)
    sym3 = unwrap_group(sym3, notation3)
    sym3 = canonicalize_values(sym3, notation3)
    cmp = NotationParametrizedComparer(sym3, notation3, [])
    return cmp.match(outsym, norm_notation) is not None


class TestFractionOperations(unittest.TestCase):
    """Test suite for fraction operations (Rules 1-5)."""

    def checkEqual(self, expr1, expr2):
        """Assert that expr1 processes to expr2."""
        return self.assertTrue(check(expr1, expr2),
                             f"\nExpected: {expr2}\nFrom: {expr1}")

    # =========================================================================
    # Rule 1: Fraction + Fraction
    # \frac{x1}{y1} + \frac{x2}{y2} → \frac{\add!{\mul!{x1·y2} + \mul!{x2·y1}}}{\mul!{y1·y2}}
    # =========================================================================

    def test_rule1_numeric_simple(self):
        """Rule 1: Simple numeric fraction addition."""
        # 1/2 + 1/3 = 5/6
        self.checkEqual("\\frac 1 2 + \\frac 1 3", "\\frac 5 6")

    def test_rule1_numeric_different_denominators(self):
        """Rule 1: Numeric fractions with different denominators."""
        # 2/3 + 1/4 = 11/12
        self.checkEqual("\\frac 2 3 + \\frac 1 4", "\\frac 11 12")

    def test_rule1_numeric_with_variable(self):
        """Rule 1: Numeric fractions multiplied by same variable."""
        # (2/3)x + (1/5)x = (13/15)x
        self.checkEqual("\\frac 2 3 x + \\frac 1 5 x", "\\frac 13 15 x")

    def test_rule1_symbolic_via_add(self):
        """Rule 1: Symbolic fraction addition using add! command."""
        # add!{a/b + c/d} should generate cross-multiplication
        self.checkEqual(
            "add! \\frac a b + \\frac c d",
            "\\frac {ad + bc} {bd}"
        )

    def test_rule1_symbolic_with_subtraction(self):
        """Rule 1: Symbolic fraction subtraction using add! command."""
        # add!{1/x - 1/y} → (y - x)/(xy)
        self.checkEqual(
            "add! \\frac 1 x - \\frac 1 y",
            "\\frac {y - x} {xy}"
        )

    def test_rule1_symbolic_same_denominator(self):
        """Rule 1: Symbolic fractions with same denominator using add!."""
        # add!{x/y + z/y} → (xy+zy)/y²
        # Note: Denominator becomes (y)² which is correct
        self.checkEqual(
            "add! \\frac x y + \\frac z y",
            "\\frac {xy+zy} {(y)^2}"
        )

    #@unittest.skip("Produces correct result (1) but comparer can't match IntegerValue vs FracValue")
    def test_rule1_multiple_fractions(self):
        """Rule 1: Multiple fraction addition (pairwise)."""
        # 1/2 + 1/3 + 1/6 = 1 (verified: produces correct LaTeX output)
        self.checkEqual("\\frac 1 2 + \\frac 1 3 + \\frac 1 6", "1")

    def test_rule1_numeric_reduction(self):
        """Rule 1: Result should auto-reduce for numeric fractions."""
        # 1/4 + 1/4 = 2/4 = 1/2
        self.checkEqual("\\frac 1 4 + \\frac 1 4", "\\frac 1 2")

    # =========================================================================
    # Rule 2: Fraction × Fraction
    # \frac{x1}{y1} · \frac{x2}{y2} → \frac{\mul!{x1·x2}}{\mul!{y1·y2}}
    # =========================================================================

    def test_rule2_numeric_simple(self):
        """Rule 2: Simple numeric fraction multiplication."""
        # 1/2 × 1/3 = 1/6
        self.checkEqual("mul! \\frac 1 2 \\frac 1 3", "\\frac 1 6")

    def test_rule2_numeric_with_reduction(self):
        """Rule 2: Numeric fraction multiplication with reduction."""
        # 2/3 × 3/4 = 6/12 = 1/2
        self.checkEqual("mul! \\frac 2 3 \\frac 3 4", "\\frac 1 2")

    def test_rule2_symbolic_via_mul(self):
        """Rule 2: Symbolic fraction multiplication using mul! command."""
        # mul!{(a/b)(c/d)} → (ac)/(bd)
        self.checkEqual(
            "mul! \\frac a b \\frac c d",
            "\\frac {ac} {bd}"
        )

    def test_rule2_symbolic_same_numerator(self):
        """Rule 2: Symbolic fractions with same numerator."""
        # mul!{(x/a)(x/b)} → x²/(ab)
        self.checkEqual(
            "mul! \\frac x a \\frac x b",
            "\\frac {(x)^2} {ab}"
        )

    def test_rule2_mixed_numeric_symbolic(self):
        """Rule 2: Mixed numeric and symbolic fraction multiplication."""
        # mul!{(2/3)(x/y)} → (2x)/(3y)
        # Current: produces \frac{2/3·x}{y} instead
        self.checkEqual(
            "mul! \\frac 2 3 \\frac x y",
            "\\frac {2x} {3y}"
        )

    # =========================================================================
    # Rule 3: Scalar × Fraction
    # a · \frac{x}{y} → \frac{\mul!{a·x}}{y}
    # =========================================================================

    def test_rule3_numeric_scalar_times_fraction(self):
        """Rule 3: Numeric scalar times fraction."""
        # 2 × (1/3) = 2/3
        # Current: produces mixed fraction 2 1/3 instead
        self.checkEqual("mul! 2 \\frac 1 3", "\\frac 2 3")

    def test_rule3_fraction_times_numeric_scalar(self):
        """Rule 3: Fraction times numeric scalar (commutative)."""
        # (1/4) × 3 = 3/4
        self.checkEqual("mul! \\frac 1 4 3", "\\frac 3 4")

    def test_rule3_symbolic_scalar_times_fraction(self):
        """Rule 3: Symbolic scalar times fraction."""
        # mul!{a(x/y)} → (ax)/y
        self.checkEqual("mul! a \\frac x y", "\\frac {ax} y")

    def test_rule3_fraction_times_symbolic_scalar(self):
        """Rule 3: Fraction times symbolic scalar (commutative)."""
        # mul!{(a/b)c} → (ac)/b
        self.checkEqual("mul! \\frac a b c", "\\frac {ac} b")

    def test_rule3_variable_times_numeric_fraction(self):
        """Rule 3: Variable times numeric fraction."""
        # mul!{x(2/3)} → (2x)/3
        # Current: produces 2/3·x (not combined)
        self.checkEqual("mul! x \\frac 2 3", "\\frac {2x} 3")

    def test_rule3_numeric_times_symbolic_fraction(self):
        """Rule 3: Numeric scalar times symbolic fraction."""
        # mul!{3(a/b)} → (3a)/b
        self.checkEqual("mul! 3 \\frac a b", "\\frac {3a} b")

    # =========================================================================
    # Rule 4: Fraction × Sum
    # \frac{x}{y} · (a + b) → \frac{\mul!{x·(a+b)}}{y}
    # =========================================================================

    def test_rule4_fraction_times_sum_symbolic(self):
        """Rule 4: Fraction times sum (symbolic) - mul! expands."""
        # mul!{(a/b)(x+y)} → (ax+ay)/b (mul! expands the product)
        self.checkEqual(
            "mul! \\frac a b (x+y)",
            "\\frac {ax+ay} b"
        )

    def test_rule4_fraction_times_sum_numeric_frac(self):
        """Rule 4: Numeric fraction times sum - mul! distributes."""
        # mul!{(1/2)(a+b)} → {1/2·a + 1/2·b} (verified: correct output with grouping)
        self.checkEqual(
            "mul! \\frac 1 2 (a+b)",
            "\\frac{1}{2} a+ \\frac{1}{2} b"
        )

    def test_rule4_sum_times_fraction(self):
        """Rule 4: Sum times fraction (commutative) - mul! expands."""
        # mul!{(x+y)(c/d)} → (xc+yc)/d (mul! expands)
        self.checkEqual(
            "mul! (x+y) \\frac c d",
            "\\frac {xc+yc} d"
        )

    def test_rule4_fraction_times_difference(self):
        """Rule 4: Fraction times difference - mul! expands."""
        # mul!{(a/b)(x-y)} → (ax-ay)/b (mul! expands)
        self.checkEqual(
            "mul! \\frac a b (x-y)",
            "\\frac {ax-ay} b"
        )

    def test_rule4_numeric_fraction_times_sum_expand(self):
        """Rule 4: Numeric fraction times sum, should expand and evaluate."""
        # (1/2)(2+4) = (1/2)×6 = 3 (verified: correct)
        self.checkEqual("mul! \\frac 1 2 (2+4)", "3")

    # =========================================================================
    # Rule 5: Scalar + Fraction
    # a + \frac{x}{y} → \frac{\add!{\mul!{a·y} + x}}{y}
    # =========================================================================

    def test_rule5_numeric_scalar_plus_fraction(self):
        """Rule 5: Numeric scalar plus fraction."""
        # 2 + 1/3 = 7/3
        self.checkEqual("2 + \\frac 1 3", "\\frac 7 3")

    def test_rule5_fraction_plus_numeric_scalar(self):
        """Rule 5: Fraction plus numeric scalar (commutative)."""
        # 1/4 + 3 = 13/4
        self.checkEqual("\\frac 1 4 + 3", "\\frac 13 4")

    def test_rule5_symbolic_scalar_plus_fraction(self):
        """Rule 5: Symbolic scalar plus fraction using add!."""
        # add!{a + (x/y)} → (ay+x)/y
        self.checkEqual(
            "add! a + \\frac x y",
            "\\frac {ay + x} y"
        )

    def test_rule5_fraction_plus_symbolic_scalar(self):
        """Rule 5: Fraction plus symbolic scalar (commutative)."""
        # add!{(a/b) + c} → (a+bc)/b
        self.checkEqual(
            "add! \\frac a b + c",
            "\\frac {a + bc} b"
        )

    def test_rule5_variable_plus_numeric_fraction(self):
        """Rule 5: Variable plus numeric fraction."""
        # add!{x + (1/2)} → (2x+1)/2
        # Current: stays as x + 1/2 (not combined)
        self.checkEqual(
            "add! x + \\frac 1 2",
            "\\frac {2x + 1} 2"
        )

    def test_rule5_numeric_plus_symbolic_fraction(self):
        """Rule 5: Numeric plus symbolic fraction."""
        # add!{2 + (a/b)} → (2b+a)/b
        self.checkEqual(
            "add! 2 + \\frac a b",
            "\\frac {2b + a} b"
        )

    # =========================================================================
    # Combined/Complex Tests
    # =========================================================================

    def test_combined_mul_and_add_fractions(self):
        """Combined: Multiply then add fractions."""
        # (1/2 × 2/3) + 1/4 = 1/3 + 1/4 = 7/12
        self.checkEqual(
            "mul! \\frac 1 2 \\frac 2 3 + \\frac 1 4",
            "\\frac 7 12"
        )

    def test_combined_add_then_mul_fractions(self):
        """Combined: Add then multiply fractions."""
        # (1/2 + 1/3) × 2 = (5/6) × 2 = 5/3
        self.checkEqual(
            "mul! (\\frac 1 2 + \\frac 1 3) 2",
            "\\frac 5 3"
        )

    def test_combined_symbolic_expression(self):
        """Combined: Complex symbolic fraction expression - mul! expands."""
        # mul! expands: (ad+cb)/bd · x → (adx+cbx)/bd
        # Multiplication is distributed over sum terms in numerator
        self.checkEqual(
            "mul! {add! \\frac a b + \\frac c d} x",
            "\\frac {adx+cbx} {bd}"
        )

    def test_nested_fractions_in_numerator(self):
        """Complex: Fraction in numerator."""
        # (a/b)/c should flatten to a/(bc)
        # normalize_frac() has the logic but may not be triggered in all cases
        self.checkEqual(
            "\\frac {\\frac a b} c",
            "\\frac a {bc}"
        )

    def test_fraction_with_sum_in_numerator(self):
        """Complex: Sum in fraction numerator."""
        # (a+b)/c should stay as is
        self.checkEqual(
            "\\frac {a+b} c",
            "\\frac {a+b} c"
        )

    def test_fraction_with_product_in_denominator(self):
        """Complex: Product in fraction denominator."""
        # a/(bc) should stay as is
        self.checkEqual(
            "\\frac a {bc}",
            "\\frac a {bc}"
        )

    def test_multiple_operations_numeric(self):
        """Complex: Multiple operations with numeric fractions."""
        # 1/2 + 1/3 + 1/4 = 6/12 + 4/12 + 3/12 = 13/12
        self.checkEqual(
            "\\frac 1 2 + \\frac 1 3 + \\frac 1 4",
            "\\frac 13 12"
        )

    def test_distributive_with_fractions(self):
        """Complex: Distributive property with fractions - mul! distributes."""
        # mul!{(1/2)(x+y)} → {x/2 + y/2} (verified: correct with grouping)
        self.checkEqual(
            "mul! \\frac 1 2 (x+y)",
            "\\frac{1}{2} x+ \\frac{1}{2} y"
        )

    # =========================================================================
    # Edge Cases
    # =========================================================================

    def test_identity_fraction(self):
        """Edge case: Fraction with denominator 1."""
        # x/1 should simplify to x
        # normalize_frac() only handles IntegerValue denominators
        self.checkEqual("\\frac x 1", "x")

    def test_zero_numerator(self):
        """Edge case: Zero numerator."""
        # 0/5 = 0 (verified: evaluates correctly)
        self.checkEqual("\\frac 0 5", "0")

    def test_negative_fractions_add(self):
        """Edge case: Adding negative fractions."""
        # 1/2 + (-1/3) = 3/6 - 2/6 = 1/6
        self.checkEqual(
            "\\frac 1 2 + (- \\frac 1 3)",
            "\\frac 1 6"
        )

    def test_negative_fractions_mul(self):
        """Edge case: Multiplying negative fractions."""
        # (-1/2) × (1/3) = {(-1/6)} (verified: correct with grouping)
        self.checkEqual(
            "mul! (- \\frac 1 2) \\frac 1 3",
            "(- \\frac{1}{6} )"
        )

    def test_fraction_times_zero(self):
        """Edge case: Fraction times zero."""
        # (a/b) × 0 = 0
        # Current: produces 0/b instead of 0
        self.checkEqual("mul! \\frac a b 0", "0")

    def test_fraction_plus_zero(self):
        """Edge case: Fraction plus zero."""
        # a/b + 0 = a/b
        self.checkEqual("\\frac a b + 0", "\\frac a b")

    # =========================================================================
    # Fraction Power Support
    # =========================================================================

    def test_power_fraction_positive_symbolic(self):
        """Power: (a/b)^2 -> a^2/b^2"""
        self.checkEqual("mul! (\\frac a b)^2", "\\frac {a^2} {b^2}")

    def test_power_fraction_positive_numeric(self):
        """Power: (2/3)^3 -> 8/27"""
        self.checkEqual("mul! (\\frac 2 3)^3", "\\frac 8 {27}")

    def test_power_fraction_negative_symbolic(self):
        """Power: (a/b)^{-2} -> b^2/a^2"""
        self.checkEqual("mul! (\\frac a b)^{-2}", "\\frac {b^2} {a^2}")

    def test_power_fraction_negative_numeric(self):
        """Power: (2/3)^{-2} -> 9/4"""
        self.checkEqual("mul! (\\frac 2 3)^{-2}", "\\frac 9 4")

    def test_power_fraction_zero(self):
        """Power: (a/b)^0 -> 1"""
        self.checkEqual("mul! (\\frac a b)^0", "1")

    def test_negative_power_symbolic(self):
        """Power: x^{-1} -> 1/x"""
        self.checkEqual("mul! x^{-1}", "\\frac 1 x")

    def test_negative_power_symbolic_squared(self):
        """Power: x^{-2} -> 1/x^2"""
        self.checkEqual("mul! x^{-2}", "\\frac 1 {x^2}")

    def test_negative_power_cancel(self):
        """Power: x^{-1} * x -> 1"""
        self.checkEqual("mul! x^{-1}x", "1")

    def test_negative_power_cancel_squared(self):
        """Power: x^{-2} * x^2 -> 1"""
        self.checkEqual("mul! x^{-2}x^2", "1")

    def test_negative_power_partial_cancel(self):
        """Power: x^{-1} * x^2 -> x"""
        self.checkEqual("mul! x^{-1}x^2", "x")


if __name__ == '__main__':
    unittest.main()

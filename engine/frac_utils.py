#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fraction utilities for symbolic fraction operations.

Provides helper functions for detecting, extracting, and normalizing
fractions in the notation graph.
"""

from notation import Symbol, Notation
from value import FracValue, IntegerValue, Value, division

# All fraction operator names
FRAC_SYMBOL_NAMES = frozenset(['\\frac', '\\dfrac', '\\tfrac', '\\cfrac'])

# For backwards compatibility
FRAC_SYMBOLS = frozenset([
    Symbol('\\frac'),
    Symbol('\\dfrac'),
    Symbol('\\tfrac'),
    Symbol('\\cfrac')
])


def is_frac(notation, sym):
    """
    Check if sym is a fraction operator.

    Args:
        notation: Notation object containing the symbol
        sym: Symbol to check

    Returns:
        True if sym represents a fraction, False otherwise
    """
    if isinstance(sym, FracValue):
        return True
    if not isinstance(sym, Symbol):
        return False
    f = notation.get(sym)
    return f is not None and f.sym.name in FRAC_SYMBOL_NAMES


def get_numerator(notation, sym):
    """
    Extract numerator from fraction, unwrapping GROUP if needed.

    Args:
        notation: Notation object containing the symbol
        sym: Fraction symbol

    Returns:
        Numerator symbol (unwrapped from GROUP), or None if not a fraction
    """
    if isinstance(sym, FracValue):
        return IntegerValue(sym.num)

    f = notation.get(sym)
    if f is None or f.sym.name not in FRAC_SYMBOL_NAMES:
        return None
    num = f.args[0]
    # Unwrap group if numerator is braced
    g = notation.getf(num, Notation.GROUP)
    return g.args[0] if g is not None else num


def get_denominator(notation, sym):
    """
    Extract denominator from fraction, unwrapping GROUP if needed.

    Args:
        notation: Notation object containing the symbol
        sym: Fraction symbol

    Returns:
        Denominator symbol (unwrapped from GROUP), or None if not a fraction
    """
    if isinstance(sym, FracValue):
        return IntegerValue(sym.denom)

    f = notation.get(sym)
    if f is None or f.sym.name not in FRAC_SYMBOL_NAMES:
        return None
    denom = f.args[1]
    # Unwrap group if denominator is braced
    g = notation.getf(denom, Notation.GROUP)
    return g.args[0] if g is not None else denom


def normalize_frac(notation, numerator_sym, denominator_sym, chainexpr_fn=None, mul_symbol=None):
    r"""
    Normalize fraction with automatic reductions. MUST be idempotent.

    Args:
        notation: Output notation for building result
        numerator_sym: Symbol representing numerator
        denominator_sym: Symbol representing denominator
        chainexpr_fn: Optional function(oper, notation, sym, negative) for wrapping commands
        mul_symbol: Optional Symbol for mul! operator (needed for flattening)

    Returns:
        Normalized fraction symbol or simplified value

    Applies:
        1. Identity: \frac{x}{1} → x
        2. Sign normalization: negative always on numerator
        3. Flatten nested: \frac{\frac{a}{b}}{c} → \frac{a}{\mul!{bc}} (if chainexpr_fn provided)
        4. Style canonicalization: always use \frac

    Note: Numeric reduction (GCD) is handled by Preprocessor during initial preprocessing.
          Fractions created during fixed-point iteration will not be automatically reduced.
    """
    from value import IntegerValue, equal_value

    # Note: Numeric reduction is handled by Preprocessor (preprocessor.py:41-46)
    # during initial preprocessing only. Fractions created during fixed-point
    # iteration will NOT be automatically reduced (future enhancement).

    # Unwrap trivial groups
    g = notation.getf(numerator_sym, Notation.GROUP)
    if g is not None and g.props.get("br") == "{}":
        numerator_sym = g.args[0]
    g = notation.getf(denominator_sym, Notation.GROUP)
    if g is not None and g.props.get("br") == "{}":
        denominator_sym = g.args[0]

    # Cancel common scalar when numerator and denominator are pure values
    if isinstance(numerator_sym, Value) and isinstance(denominator_sym, Value):
        div_res = division(numerator_sym, denominator_sym)
        if div_res is not None:
            if isinstance(div_res, FracValue):
                sign_negative = div_res.num < 0
                num_val = IntegerValue(abs(div_res.num))
                den_val = IntegerValue(div_res.denom)
                frac = notation.setf(
                    Symbol("\\frac"),
                    (
                        notation.setf(Notation.GROUP, (num_val,), br="{}"),
                        notation.setf(Notation.GROUP, (den_val,), br="{}"),
                    ),
                )
                if sign_negative:
                    frac = notation.setf(Notation.MINUS, (frac,))
                return frac
            return div_res

    # 1. Identity simplification
    if isinstance(denominator_sym, IntegerValue) and equal_value(denominator_sym, 1):
        return numerator_sym

    # 2. Sign normalization (if denominator is negative, flip both signs)
    denom_f = notation.getf(denominator_sym, Notation.MINUS)
    if denom_f is not None:
        # Flip both signs
        neg_denom = denom_f.args[0]
        num_f = notation.getf(numerator_sym, Notation.MINUS)
        if num_f is not None:
            # Both negative → both positive
            numerator_sym = num_f.args[0]
        else:
            # Denom negative, num positive → negate num
            numerator_sym = notation.setf(Notation.MINUS, (numerator_sym,))
        denominator_sym = neg_denom

    # 2b. Numeric reduction when both numerator and denominator are pure values
    if isinstance(numerator_sym, Value) and isinstance(denominator_sym, Value):
        div_res = division(numerator_sym, denominator_sym)
        if div_res is not None:
            # Preserve fraction structure for comparer consistency
            if isinstance(div_res, FracValue):
                sign_negative = div_res.num < 0
                num_val = IntegerValue(abs(div_res.num))
                den_val = IntegerValue(div_res.denom)
                frac = notation.setf(
                    Symbol("\\frac"),
                    (
                        notation.setf(Notation.GROUP, (num_val,), br="{}"),
                        notation.setf(Notation.GROUP, (den_val,), br="{}"),
                    ),
                )
                if sign_negative:
                    frac = notation.setf(Notation.MINUS, (frac,))
                return frac
            return div_res

    # NOTE: Distribution \frac{a+b}{k} -> \frac{a}{k} + \frac{b}{k} removed
    # This was causing unwanted expansion of combined fractions.
    # If needed for specific cases, should be explicit rather than automatic.

    # 3. Flatten nested fraction in numerator (only if chainexpr_fn provided)
    if chainexpr_fn is not None and mul_symbol is not None and is_frac(notation, numerator_sym):
        a = get_numerator(notation, numerator_sym)
        b = get_denominator(notation, numerator_sym)
        # \frac{\frac{a}{b}}{c} → \frac{a}{\mul!{bc}}
        new_denom_plist = notation.setf(Notation.P_LIST, (b, denominator_sym))
        denominator_sym = chainexpr_fn(mul_symbol, notation, new_denom_plist, None)
        numerator_sym = a

    # 3b. Distribute multiplication over sums in numerator: (a+b)·c → ac+bc
    # Check if numerator is a P_LIST containing a GROUP-wrapped sum
    def unwrap_trivial_group(sym):
        g = notation.getf(sym, Notation.GROUP)
        if g is not None and g.props.get("br") in ["{}", "()"]:
            return g.args[0]
        return sym

    num_unwrapped = unwrap_trivial_group(numerator_sym)
    plist = notation.getf(num_unwrapped, Notation.P_LIST)
    if plist is not None and len(plist.args) >= 2 and chainexpr_fn is not None and mul_symbol is not None:
        # Check if any element in P_LIST is a sum
        for i, elem in enumerate(plist.args):
            elem_unwrapped = unwrap_trivial_group(elem)
            slist = notation.getf(elem_unwrapped, Notation.S_LIST)
            if slist is not None:
                # Found a sum - distribute multiplication over it
                other_factors = [plist.args[j] for j in range(len(plist.args)) if j != i]

                new_terms = []
                for idx, term in enumerate(slist.args):
                    # Unwrap PLUS/MINUS
                    term_val = term
                    sign_prefix = None
                    f_sign = notation.vgetf(term, [Notation.PLUS, Notation.MINUS])
                    if f_sign is not None:
                        sign_prefix = f_sign.sym
                        term_val = f_sign.args[0]

                    # Create product with other factors
                    if other_factors:
                        if len(other_factors) == 1:
                            product_plist = notation.setf(Notation.P_LIST, (term_val, other_factors[0]))
                        else:
                            product_plist = notation.setf(Notation.P_LIST, tuple([term_val] + other_factors))
                        term_result = chainexpr_fn(mul_symbol, notation, product_plist, None)
                    else:
                        term_result = term_val

                    # Reapply sign
                    if sign_prefix == Notation.MINUS:
                        term_result = notation.setf(Notation.MINUS, (term_result,))
                    elif idx > 0:
                        term_result = notation.setf(Notation.PLUS, (term_result,))

                    new_terms.append(term_result)

                numerator_sym = notation.setf(Notation.S_LIST, tuple(new_terms))
                break  # Only distribute once

    # 4. Build canonical \frac
    num_group = notation.setf(Notation.GROUP, (numerator_sym,), br="{}")
    denom_group = notation.setf(Notation.GROUP, (denominator_sym,), br="{}")
    return notation.setf(Symbol('\\frac'), (num_group, denom_group))

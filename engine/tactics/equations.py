#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Narrow equation-solving tactics.

This module deliberately does not expose a general ``solve`` operation.
Each tactic recognizes one explicit equation class and returns a complete,
mechanically checked solution record for that class.
"""
import math
from fractions import Fraction

from notation import Notation, Symbol
from polyrat import NotInFragment, to_ratfunc
from primitives import (
    PrimitiveError, EvalError, parse_latex, numeric_eval, _num_agree,
    _result, _error,
)


def _plain_variable(var):
    sym, notation = parse_latex(var)
    if not isinstance(sym, Symbol) or notation.get(sym) is not None:
        raise PrimitiveError('variable must be a plain symbol')
    return sym.name


def _zero_polynomial(sym, notation):
    """Return the polynomial whose zeros the input requests.

    A plain expression means ``expr = 0``.  An equality means ``lhs-rhs = 0``.
    Other relations are not equation-solving inputs.
    """
    comp = notation.getf(sym, Notation.COMP)
    if comp is None:
        rf = to_ratfunc(sym, notation)
    else:
        if comp.sym.props.get('op') != '=':
            raise PrimitiveError('quadratic_roots requires an equality')
        rf = (to_ratfunc(comp.args[0], notation)
              - to_ratfunc(comp.args[1], notation))
    if not rf.is_poly():
        raise PrimitiveError('quadratic_roots supports polynomials only')
    return rf.num


def _quadratic_coefficients(poly, var):
    if poly.variables() - {var}:
        raise PrimitiveError('quadratic coefficients must be constants')
    if poly.degree(var) != 2:
        raise PrimitiveError(f'expression is not quadratic in {var!r}')
    coeffs = {0: Fraction(0), 1: Fraction(0), 2: Fraction(0)}
    for mono, coeff in poly.terms.items():
        powers = dict(mono)
        exponent = powers.pop(var, 0)
        if powers or exponent not in coeffs:
            raise PrimitiveError(f'expression is not quadratic in {var!r}')
        coeffs[exponent] += coeff
    return coeffs[2], coeffs[1], coeffs[0]


def _fraction_sqrt(value):
    if value < 0:
        return None
    numerator = math.isqrt(value.numerator)
    denominator = math.isqrt(value.denominator)
    if (numerator * numerator == value.numerator
            and denominator * denominator == value.denominator):
        return Fraction(numerator, denominator)
    return None


def _fraction_latex(value):
    if value.denominator == 1:
        return str(value.numerator)
    sign = '-' if value < 0 else ''
    value = abs(value)
    return f'{sign}\\frac{{{value.numerator}}}{{{value.denominator}}}'


def _numeric_zero_body(sym, notation, env):
    """Oracle-only numeric reading of ``expr`` or ``lhs = rhs``."""
    comp = notation.getf(sym, Notation.COMP)
    if comp is None:
        return numeric_eval(sym, notation, env)
    if comp.sym.props.get('op') != '=':
        raise EvalError('expected an equality')
    return (numeric_eval(comp.args[0], notation, env)
            - numeric_eval(comp.args[1], notation, env))


def _quadratic_roots_check(expr, var, roots):
    """Independent quadratic/root check using only the numeric evaluator.

    Reconstruct ``a*x^2+b*x+c`` from values at -1, 0, and 1, verify that
    model at additional points, then check candidate zeros and both Vieta
    identities.  This shares no polynomial conversion or quadratic-formula
    implementation with the symbolic path.
    """
    try:
        sym, notation = parse_latex(expr)

        def value_at(x):
            value = _numeric_zero_body(sym, notation, {var: float(x)})
            if isinstance(value, list) or not math.isfinite(value):
                raise EvalError('non-scalar quadratic value')
            return value

        y_minus = value_at(-1.0)
        y_zero = value_at(0.0)
        y_plus = value_at(1.0)
        a = (y_plus + y_minus) / 2.0 - y_zero
        b = (y_plus - y_minus) / 2.0
        c = y_zero
        if _num_agree(a, 0.0, 1e-10):
            return {'status': 'disagree',
                    'reason': 'numeric reconstruction is not quadratic'}

        sample_points = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
        for x in sample_points:
            observed = value_at(x)
            expected = a * x * x + b * x + c
            if _num_agree(observed, expected, 1e-8) is not True:
                return {'status': 'disagree',
                        'reason': 'numeric samples do not fit a quadratic'}

        numeric_roots = []
        for root in roots:
            root_sym, root_notation = parse_latex(root)
            value = numeric_eval(root_sym, root_notation, {})
            if isinstance(value, list) or not math.isfinite(value):
                raise EvalError('non-scalar root')
            numeric_roots.append(value)
            if _num_agree(value_at(value), 0.0, 1e-8) is not True:
                return {'status': 'disagree',
                        'reason': f'candidate {root!r} is not a zero'}

        vieta_roots = (numeric_roots if len(numeric_roots) == 2
                       else numeric_roots * 2)
        if len(vieta_roots) != 2:
            return {'status': 'disagree',
                    'reason': 'quadratic needs one repeated or two roots'}
        if (_num_agree(sum(vieta_roots), -b / a, 1e-8) is not True
                or _num_agree(vieta_roots[0] * vieta_roots[1],
                              c / a, 1e-8) is not True):
            return {'status': 'disagree',
                    'reason': 'candidate roots fail Vieta completeness'}
        return {
            'status': 'agree',
            'samples': len(sample_points) + len(numeric_roots),
            'method': 'independent numeric interpolation and Vieta',
        }
    except (PrimitiveError, EvalError, ValueError, ZeroDivisionError,
            OverflowError) as exc:
        return {'status': 'skipped', 'reason': str(exc)}


def quadratic_roots(expr, var):
    """Return all rational roots of one quadratic expression/equality."""
    args = {'expr': expr, 'var': var}
    try:
        var_name = _plain_variable(var)
        sym, notation = parse_latex(expr)
        poly = _zero_polynomial(sym, notation)
        a, b, c = _quadratic_coefficients(poly, var_name)
        discriminant = b * b - 4 * a * c
        root_disc = _fraction_sqrt(discriminant)
        if root_disc is None:
            if discriminant < 0:
                raise PrimitiveError('quadratic has no real roots')
            raise PrimitiveError('quadratic roots are not rational')
        roots = sorted({(-b - root_disc) / (2 * a),
                        (-b + root_disc) / (2 * a)})
        solutions = [_fraction_latex(root) for root in roots]
        result = ' \\lor '.join(
            f'{var_name}={solution}' for solution in solutions)
        # The final result must remain notebook-chainable even though an OR
        # list is not a top-level equality claim.
        parse_latex(result)
        check = _quadratic_roots_check(expr, var_name, solutions)
        if check.get('status') == 'skipped':
            return _error('quadratic_roots', args,
                          'independent root check unavailable: '
                          + check.get('reason', 'unknown reason'))
        return _result('quadratic_roots', args, expr, result,
                       check=check, extra={'solutions': solutions})
    except PrimitiveError as exc:
        return _error('quadratic_roots', args, str(exc))
    except (NotInFragment, ZeroDivisionError) as exc:
        return _error('quadratic_roots', args, str(exc))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Narrow equation-solving tactics.

This module deliberately does not expose a general ``solve`` operation.
Each tactic recognizes one explicit equation class and returns a complete,
mechanically checked solution record for that class.
"""
import math
import random
from fractions import Fraction

from notation import Notation, Symbol
from polyrat import NotInFragment, to_ratfunc
from primitives import (
    PrimitiveError, EvalError, parse_latex, write_latex, numeric_eval,
    free_symbols, same_expression, _num_agree, _sample_point,
    _result, _error,
)
from tactics.core import equal_exprs, substitute


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


def _root_values(roots, var_name):
    """Ordered root values written by a recorded root-solution relation.

    Accepts the ``var=a \\lor var=b`` shape produced by the root tactics and
    the single-equality shape of a repeated root.  Anything else is refused:
    this tactic completes solutions that were already found, it does not
    interpret arbitrary relations.
    """
    sym, notation = parse_latex(roots)
    head = notation.get(sym)
    if head is None:
        raise PrimitiveError(
            'roots must be a recorded solution relation such as '
            f'{var_name}=a \\lor {var_name}=b')
    items = (list(head.args) if head.sym.name == Notation.O_LIST.name
             else [sym])
    out = []
    for item in items:
        comp = notation.getf(item, Notation.COMP)
        if comp is None or comp.sym.props.get('op') != '=':
            raise PrimitiveError(
                f'every solution must be an equality {var_name}=value')
        left = comp.args[0]
        if not isinstance(left, Symbol) or left.name != var_name:
            raise PrimitiveError(
                f'every solution must name {var_name!r} on the left')
        solution = write_latex(comp.args[1], notation)
        # a complete answer lists each solution once; a repeat (in any
        # spelling) would print the same point twice and there is no
        # collection algebra to fold it back together
        for seen in out:
            if same_expression(solution, seen) or _same_value(solution, seen):
                raise PrimitiveError(
                    f'solutions {seen!r} and {solution!r} are the same root; '
                    'list each root once')
        out.append(solution)
    return out


def _same_value(left, right):
    equality = equal_exprs(left, right)
    return equality.get('ok') and equality.get('verdict') == 'yes'


def point_pairs(roots, var, values):
    """Ordered ``{root, value}`` associations named by a points record.

    Shared by the tactic and by replay provenance validation, so a recorded
    association can never drift away from the arguments it was built from.
    """
    var_name = _plain_variable(var)
    solutions = _root_values(roots, var_name)
    if not isinstance(values, (list, tuple)) or len(values) != len(solutions):
        raise PrimitiveError(
            f'{roots!r} records {len(solutions)} solution(s) but '
            f'{len(values) if isinstance(values, (list, tuple)) else "?"} '
            'value(s) were supplied; give one recorded value step per root, '
            'in the order they are written')
    return [{'root': root, 'value': value}
            for root, value in zip(solutions, values)]


def _points_check(expr, var, pairs, samples=8, seed=20260705, tol=1e-6):
    """Independent check that every pair really is ``(r, expr(r))``.

    The oracle binds ``var`` to its own numeric reading of each root and
    evaluates ``expr`` there, so it shares no substitution machinery with
    the symbolic path.  Pairing is what this check sees: a swapped or
    shifted association shows up as a numeric disagreement.
    """
    try:
        esym, enotation = parse_latex(expr)
        parsed = []
        for root, value in pairs:
            rsym, rnotation = parse_latex(root)
            vsym, vnotation = parse_latex(value)
            parsed.append((root, rsym, rnotation, vsym, vnotation))
    except PrimitiveError as exc:
        return {'status': 'skipped', 'reason': str(exc)}
    variables = free_symbols(esym, enotation) - {var}
    for _, rsym, rnotation, vsym, vnotation in parsed:
        variables |= free_symbols(rsym, rnotation)
        variables |= free_symbols(vsym, vnotation)
    closed = not variables
    target = 1 if closed else samples
    rng = random.Random(seed)
    agreed = 0
    tried = 0
    while agreed < target and tried < target * 8:
        tried += 1
        env = _sample_point(variables, rng)
        usable = True
        for root, rsym, rnotation, vsym, vnotation in parsed:
            try:
                root_value = numeric_eval(rsym, rnotation, env)
                if isinstance(root_value, list):
                    raise EvalError('non-scalar root')
                local = dict(env)
                local[var] = root_value
                at_root = numeric_eval(esym, enotation, local)
                paired = numeric_eval(vsym, vnotation, env)
            except (ValueError, ZeroDivisionError) as exc:
                # a genuine domain signal: on closed data the pair simply
                # has no value, which is evidence, not ignorance
                if closed:
                    return {'status': 'domain-differs', 'root': root,
                            'reason': str(exc)}
                usable = False
                break
            except (EvalError, OverflowError):
                usable = False
                break
            agree = _num_agree(at_root, paired, tol)
            if agree is None:
                usable = False
                break
            if not agree:
                return {'status': 'disagree', 'root': root,
                        'expected': at_root, 'paired': paired,
                        'point': env}
        if usable:
            agreed += 1
    if agreed == 0:
        return {'status': 'skipped',
                'reason': 'no evaluable sample points'}
    return {'status': 'agree', 'samples': agreed * len(parsed),
            'method': 'independent evaluation at each root'}


def points_assemble(expr, var, roots, values):
    """Assemble recorded roots and their recorded values into the complete
    point collection.

    ``values`` is ordered exactly like the solutions in ``roots``.  The do!
    tool and the CLI supply both from ledger step ids rather than letting the
    agent retype an answer: a typed tuple is a claim about where numbers came
    from, and this layer never accepts that on trust.  Each association is
    gated symbolically (substitute the root, then the independent equality
    checker) and the recorded check re-derives every pair numerically.
    """
    args = {'expr': expr, 'var': var, 'roots': roots,
            'values': list(values) if isinstance(values, (list, tuple))
            else values}
    if not isinstance(values, (list, tuple)):
        return _error('points_assemble', args,
                      'values must be an ordered list')
    try:
        var_name = _plain_variable(var)
        parse_latex(expr)
        associations = point_pairs(roots, var, values)
    except PrimitiveError as exc:
        return _error('points_assemble', args, str(exc))

    pairs = []
    for index, association in enumerate(associations, 1):
        root, value = association['root'], association['value']
        if not isinstance(value, str) or not value.strip():
            return _error('points_assemble', args,
                          f'point {index} has no recorded value')
        substituted = substitute(expr, var, root)
        if not substituted.get('ok'):
            return _error(
                'points_assemble', args,
                f'cannot substitute root {root!r}: '
                + substituted.get('error', 'unknown error'))
        equality = equal_exprs(substituted['result'], value)
        if not (equality.get('ok') and equality.get('verdict') == 'yes'):
            return _error(
                'points_assemble', args,
                f'the recorded value {value!r} is not the value of '
                f'{expr!r} at {root!r} (verdict: '
                f'{equality.get("verdict", "error")})')
        pairs.append((root, value))

    built = '\\{' + ','.join(f'({root},{value})'
                             for root, value in pairs) + '\\}'
    try:
        csym, cnotation = parse_latex(built)
        collection = cnotation.getf(csym, Notation.COLLECTION)
        if collection is None or len(collection.args) != len(pairs):
            raise PrimitiveError('assembled points are not a collection')
        for item, (root, value) in zip(collection.args, pairs):
            point = cnotation.getf(item, Notation.PAIR)
            if point is None:
                raise PrimitiveError('assembled item is not an ordered pair')
            if not (same_expression(write_latex(point.args[0], cnotation),
                                    root)
                    and same_expression(write_latex(point.args[1], cnotation),
                                        value)):
                raise PrimitiveError('assembled pair lost its association')
        result = write_latex(csym, cnotation)
    except PrimitiveError as exc:
        return _error('points_assemble', args,
                      f'internal: unusable assembly: {exc}')

    rec = _result('points_assemble', args, roots, result,
                  extra={'points': associations})
    rec['check'] = _points_check(expr, var_name, pairs)
    return rec

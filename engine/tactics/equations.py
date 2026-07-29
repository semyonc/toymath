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
from polyrat import NotInFragment, Poly, to_ratfunc, poly_to_notation
from primitives import (
    PrimitiveError, EvalError, parse_latex, write_latex, numeric_eval,
    free_symbols, same_expression, _num_agree, _sample_point,
    _result, _error,
)
from tactics.core import equal_exprs, substitute, _comp_split


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


def assignment_pairs(assignments):
    """Ordered ``{unknown, value}`` associations named by recorded relations.

    Shared by the tactic and by replay provenance validation, so a recorded
    association can never drift away from the results it was built from.
    """
    if not isinstance(assignments, (list, tuple)) or not assignments:
        raise PrimitiveError(
            'give one recorded step per unknown, each ending at '
            '"unknown = value"')
    out = []
    for index, assignment in enumerate(assignments, 1):
        if not isinstance(assignment, str) or not assignment.strip():
            raise PrimitiveError(f'assignment {index} records no relation')
        sym, notation = parse_latex(assignment)
        comp = notation.getf(sym, Notation.COMP)
        if comp is None or comp.sym.props.get('op') != '=':
            raise PrimitiveError(
                f'assignment {index} ({assignment!r}) is not an equality; '
                'each unknown needs a recorded step ending at '
                '"unknown = value"')
        left = comp.args[0]
        if not isinstance(left, Symbol) or notation.get(left) is not None:
            raise PrimitiveError(
                f'assignment {index} ({assignment!r}) must name a plain '
                'unknown on the left')
        for seen in out:
            if seen['unknown'] == left.name:
                raise PrimitiveError(
                    f'{left.name!r} is assigned twice; give one recorded '
                    'step per unknown')
        out.append({'unknown': left.name,
                    'value': write_latex(comp.args[1], notation)})
    # An answer names each unknown once, on the left. A value that still
    # mentions one of them is a half-solved system, not an answer, and the
    # order the values were substituted in would silently decide it.
    names = {pair['unknown'] for pair in out}
    for pair in out:
        vsym, vnotation = parse_latex(pair['value'])
        remaining = sorted(free_symbols(vsym, vnotation) & names)
        if remaining:
            raise PrimitiveError(
                f'the value recorded for {pair["unknown"]!r} still contains '
                f'{remaining[0]!r}; an assembled answer states each unknown '
                'in terms of the givens only')
    return out


def _target_relations(target):
    """(member relations, root sym, notation) of an equality or comma system.

    The target is what the assembled values must satisfy.  Only equalities
    are accepted: verifying an assignment against an inequality would need a
    decision procedure this layer deliberately does not have.
    """
    sym, notation = parse_latex(target)
    head = notation.get(sym)
    items = (list(head.args)
             if head is not None and head.sym.name == Notation.C_LIST.name
             else [sym])
    relations = []
    for item in items:
        comp = notation.getf(item, Notation.COMP)
        if comp is None:
            raise PrimitiveError(
                'the target must be an equality, or a comma system of '
                'equalities, that the values satisfy')
        if comp.sym.props.get('op') != '=':
            raise PrimitiveError(
                'system_assemble verifies equalities; '
                f'{write_latex(item, notation)!r} is not one')
        relations.append(write_latex(item, notation))
    return relations, sym, notation


def _system_check(relations, pairs, samples=8, seed=20260728, tol=1e-6):
    """Independent check that the assignment satisfies every target relation.

    The oracle binds each unknown to its own numeric reading of the recorded
    value and evaluates both sides there, so it shares no substitution
    machinery with the symbolic path.  Association is what this check sees: a
    swapped pair of values shows up as a numeric disagreement.
    """
    try:
        parsed = []
        for relation in relations:
            rsym, rnotation = parse_latex(relation)
            comp = rnotation.getf(rsym, Notation.COMP)
            if comp is None:
                raise PrimitiveError('target member is not a relation')
            parsed.append((relation, comp.args[0], comp.args[1], rnotation))
        values = []
        for pair in pairs:
            vsym, vnotation = parse_latex(pair['value'])
            values.append((pair['unknown'], vsym, vnotation))
    except PrimitiveError as exc:
        return {'status': 'skipped', 'reason': str(exc)}

    unknowns = {name for name, _, _ in values}
    variables = set()
    for _, lhs, rhs, rnotation in parsed:
        variables |= free_symbols(lhs, rnotation)
        variables |= free_symbols(rhs, rnotation)
    for _, vsym, vnotation in values:
        variables |= free_symbols(vsym, vnotation)
    variables -= unknowns
    closed = not variables
    wanted = 1 if closed else samples
    rng = random.Random(seed)
    agreed = 0
    tried = 0
    while agreed < wanted and tried < wanted * 8:
        tried += 1
        env = _sample_point(variables, rng)
        usable = True
        try:
            for name, vsym, vnotation in values:
                value = numeric_eval(vsym, vnotation, env)
                if isinstance(value, list):
                    raise EvalError('non-scalar assignment value')
                env[name] = value
        except (ValueError, ZeroDivisionError) as exc:
            if closed:
                return {'status': 'domain-differs', 'reason': str(exc)}
            continue
        except (EvalError, OverflowError):
            continue
        for relation, lhs, rhs, rnotation in parsed:
            try:
                left = numeric_eval(lhs, rnotation, env)
                right = numeric_eval(rhs, rnotation, env)
            except (ValueError, ZeroDivisionError) as exc:
                # on closed data the relation simply has no value there:
                # that is evidence, not ignorance
                if closed:
                    return {'status': 'domain-differs', 'relation': relation,
                            'reason': str(exc)}
                usable = False
                break
            except (EvalError, OverflowError):
                usable = False
                break
            agree = _num_agree(left, right, tol)
            if agree is None:
                usable = False
                break
            if not agree:
                return {'status': 'disagree', 'relation': relation,
                        'left': left, 'right': right, 'point': env}
        if usable:
            agreed += 1
    if agreed == 0:
        return {'status': 'skipped', 'reason': 'no evaluable sample points'}
    return {'status': 'agree', 'samples': agreed * len(parsed),
            'method': 'independent evaluation at the assigned values'}


def system_assemble(target, assignments):
    """Assemble recorded per-unknown values into the answer for one stated
    relation or comma system, verified by putting them back into it.

    This is not a solver: every value comes from a step the agent already
    recorded, and the tactic only checks that together they satisfy the
    stated target, then binds them into one answer object.  The record says
    the assignment satisfies the target — never that it is the only one that
    does.  The do! tool and the CLI supply the values from ledger step ids
    rather than letting the agent retype an answer.
    """
    args = {'target': target,
            'assignments': list(assignments)
            if isinstance(assignments, (list, tuple)) else assignments}
    if not isinstance(assignments, (list, tuple)):
        return _error('system_assemble', args,
                      'assignments must be an ordered list')
    try:
        pairs = assignment_pairs(assignments)
        relations, tsym, tnotation = _target_relations(target)
    except PrimitiveError as exc:
        return _error('system_assemble', args, str(exc))

    named = free_symbols(tsym, tnotation)
    for pair in pairs:
        if pair['unknown'] not in named:
            return _error(
                'system_assemble', args,
                f'{pair["unknown"]!r} does not occur in {target!r}; every '
                'assembled unknown must be one the target names')

    # A target that hands over every value verifies nothing: substitution
    # turns each member into "value = value". The test is SYNTACTIC — a
    # member reading "unknown = <no unknowns>" states the answer whatever
    # spelling it uses — because comparing the member against the recorded
    # value misses a respelling, which is exactly how a live run slipped an
    # answer-as-target past the first version of this guard. A target that
    # pins only SOME unknowns is still real evidence about the others (a
    # system genuinely containing the row `x = 3`), so only a target that
    # determines all of them is refused.
    named = {pair['unknown'] for pair in pairs}
    handed = {}
    for relation in relations:
        rsym, rnotation = parse_latex(relation)
        comp = rnotation.getf(rsym, Notation.COMP)
        for side, other in ((comp.args[0], comp.args[1]),
                            (comp.args[1], comp.args[0])):
            if (isinstance(side, Symbol) and rnotation.get(side) is None
                    and side.name in named
                    and not (free_symbols(other, rnotation) & named)):
                handed[side.name] = relation
    if len(handed) == len(pairs):
        return _error(
            'system_assemble', args,
            f'the target already states the value of every unknown '
            f'({", ".join(sorted(handed))}), so nothing would be verified; '
            'the target is the equality the values must SATISFY — the '
            'ansatz being matched, or the system being solved — never the '
            'answer itself')

    for relation in relations:
        current = relation
        for pair in pairs:
            csym, cnotation = parse_latex(current)
            if pair['unknown'] not in free_symbols(csym, cnotation):
                continue
            substituted = substitute(current, pair['unknown'], pair['value'])
            if not substituted.get('ok'):
                return _error(
                    'system_assemble', args,
                    f'cannot put the value of {pair["unknown"]!r} back into '
                    f'{relation!r}: '
                    + substituted.get('error', 'unknown error'))
            current = substituted['result']
        try:
            sym, notation = parse_latex(current)
            comp = notation.getf(sym, Notation.COMP)
            lhs = write_latex(comp.args[0], notation)
            rhs = write_latex(comp.args[1], notation)
        except (PrimitiveError, AttributeError):
            return _error('system_assemble', args,
                          f'internal: unreadable substitution of {relation!r}')
        equality = equal_exprs(lhs, rhs)
        if not (equality.get('ok') and equality.get('verdict') == 'yes'):
            # nothing here can tell an unknown from a genuine variable, so
            # the repair is named without guessing which symbol it is: the
            # commonest failure is one unknown left without a recorded value
            unassigned = sorted(free_symbols(*parse_latex(current))
                                - {p['unknown'] for p in pairs})
            hint = (f'; still free there: {", ".join(unassigned)} — any '
                    'unknown among them needs its own recorded value step'
                    if unassigned else '')
            return _error(
                'system_assemble', args,
                f'the recorded values do not satisfy {relation!r} '
                f'(verdict: {equality.get("verdict", "error")}){hint}')

    built = ','.join(f'{pair["unknown"]}={pair["value"]}' for pair in pairs)
    try:
        members, bsym, bnotation = _target_relations(built)
        if len(members) != len(pairs):
            raise PrimitiveError('assembled answer lost a member')
        for member, pair in zip(members, pairs):
            msym, mnotation = parse_latex(member)
            comp = mnotation.getf(msym, Notation.COMP)
            if not (write_latex(comp.args[0], mnotation) == pair['unknown']
                    and same_expression(write_latex(comp.args[1], mnotation),
                                        pair['value'])):
                raise PrimitiveError('assembled answer lost its association')
        result = write_latex(bsym, bnotation)
    except PrimitiveError as exc:
        return _error('system_assemble', args,
                      f'internal: unusable assembly: {exc}')

    rec = _result('system_assemble', args, target, result,
                  extra={'unknowns': pairs})
    rec['check'] = _system_check(relations, pairs)
    return rec


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


def _coefficient_buckets(poly, var):
    """{power of var: coefficient Poly in the remaining symbols}."""
    buckets = {}
    for mono, coeff in poly.terms.items():
        d = dict(mono)
        k = d.pop(var, 0)
        buckets.setdefault(k, {})[tuple(sorted(d.items()))] = coeff
    return {k: Poly(v) for k, v in buckets.items()}


def _vandermonde_solve(xs, ys):
    """Coefficients c with sum_j c[j] * x^j = y, by Gaussian elimination
    with partial pivoting. Deliberately hand-rolled floating point: this is
    the oracle leg's reading of the polynomial and must share nothing with
    polyrat's symbolic monomial arithmetic."""
    n = len(xs)
    m = [[xs[i] ** j for j in range(n)] + [ys[i]] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            return None
        m[col], m[piv] = m[piv], m[col]
        for r in range(n):
            if r == col:
                continue
            f = m[r][col] / m[col][col]
            for c in range(col, n + 1):
                m[r][c] -= f * m[col][c]
    return [m[i][n] / m[i][i] for i in range(n)]


def _numeric_coefficients(sym, notation, var, env, degree):
    """Coefficients of `sym` as a polynomial in `var`, recovered purely by
    evaluation — an independent reading of the same object."""
    xs, ys = [], []
    for i in range(degree + 1):
        xv = 0.5 + 0.37 * i
        point = dict(env)
        point[var] = xv
        try:
            val = numeric_eval(sym, notation, point)
        except (EvalError, PrimitiveError, ZeroDivisionError,
                ValueError, OverflowError):
            return None
        if val is None or isinstance(val, complex):
            return None
        xs.append(xv)
        ys.append(float(val))
    return _vandermonde_solve(xs, ys)


def match_coefficients(expr, var):
    """Equate like powers of `var` on the two sides of a polynomial
    identity and record the resulting system of coefficient equations.

    NOT a solver: the agent states which variable to match in, and the
    values still come from its own later steps. The step's content is that
    these ARE the coefficients — that equating them is equivalent to the
    identity is a theorem about polynomials, not a per-instance claim."""
    args = {'expr': expr, 'var': var}
    try:
        sym, notation = parse_latex(expr)
        split = _comp_split(sym, notation)
    except PrimitiveError as e:
        return _error('match_coefficients', args, str(e))
    if split is None:
        return _error('match_coefficients', args,
                      'expected an equality between two polynomials, '
                      'for example "A x + B = 2x - 1"')
    lhs, rhs, rel = split
    if rel != '=':
        return _error('match_coefficients', args,
                      f'coefficients can only be matched across "=", '
                      f'not {rel!r}')
    try:
        polys = []
        for side in (lhs, rhs):
            rf = to_ratfunc(side, notation)
            if not rf.is_poly():
                raise NotInFragment('side is not a polynomial')
            polys.append(rf.num)
    except NotInFragment as e:
        return _error('match_coefficients', args,
                      f'both sides must be polynomials in {var}: {e}')
    except ZeroDivisionError:
        return _error('match_coefficients', args,
                      'expression contains division by zero')
    left, right = (_coefficient_buckets(p, var) for p in polys)
    if not any(var in p.variables() for p in polys):
        return _error('match_coefficients', args,
                      f'{var!r} does not occur on either side')
    equations, out_n = [], Notation()
    for k in sorted(set(left) | set(right), reverse=True):
        a = left.get(k, Poly.const(0))
        b = right.get(k, Poly.const(0))
        if a == b:
            continue  # already identical: it states nothing
        equations.append((k,
                          write_latex(poly_to_notation(a, out_n), out_n),
                          write_latex(poly_to_notation(b, out_n), out_n)))
    if not equations:
        return _error('match_coefficients', args,
                      'the two sides already have identical coefficients; '
                      'there is nothing to equate')
    result = ', '.join(f'{a} = {b}' for _, a, b in equations)
    try:
        parse_latex(result)
    except PrimitiveError as e:
        return _error('match_coefficients', args,
                      f'internal: unparseable result: {e}')
    rec = _result('match_coefficients', args, expr, result,
                  extra={'matched_in': var,
                         'powers': [k for k, _, _ in equations]})
    rec['check'] = _check_coefficients(lhs, rhs, notation, var,
                                       left, right, polys)
    return rec


def _check_coefficients(lhs, rhs, notation, var, left, right, polys):
    """Independent leg: recover each side's coefficients numerically and
    confirm the symbolic extraction reports the same ones."""
    others = sorted((free_symbols(lhs, notation)
                     | free_symbols(rhs, notation)) - {var})
    degree = max([p.degree(var) or 0 for p in polys] + [0])
    rng = random.Random(0xC0EFF)
    samples = 0
    for _ in range(6):
        env = _sample_point(others, rng) if others else {}
        for side_sym, buckets in ((lhs, left), (rhs, right)):
            got = _numeric_coefficients(side_sym, notation, var, env, degree)
            if got is None:
                return {'status': 'skipped',
                        'reason': 'the oracle could not sample this shape'}
            for k in range(degree + 1):
                want = buckets.get(k)
                try:
                    if want is None:
                        expect = 0.0
                    else:
                        coef_n = Notation()
                        expect = float(numeric_eval(
                            poly_to_notation(want, coef_n), coef_n, env))
                except (EvalError, PrimitiveError, TypeError, ValueError,
                        ZeroDivisionError, OverflowError):
                    return {'status': 'skipped',
                            'reason': 'coefficient not numerically readable'}
                if not _num_agree(got[k], expect, 1e-6):
                    return {'status': 'disagree', 'power': k,
                            'symbolic': expect, 'numeric': got[k]}
        samples += 1
    return {'status': 'agree', 'samples': samples,
            'method': 'coefficients recovered by evaluation'}

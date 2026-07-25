#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Limit tactics and independent approach checks."""
import random
from fractions import Fraction

from notation import Notation, Symbol
from polyrat import NotInFragment, to_ratfunc

from primitives import (
    FRAC_NAMES, PrimitiveError, EvalError, parse_latex, write_latex,
    same_expression,
    _transparent_inner, _plain_symbol_name, _contains_free_infinity,
    free_symbols, numeric_eval,
    _sample_point, _result, _error, _int_literal, _strip_limit,
    _infinity_sign, _limit_latex, _paren, _is_sum_str,
)
from tactics.core import (
    equal_exprs, substitute, expand, _merge_checks, _q_str,
    _rational_const,
)
from tactics.differentiation import _d_frac, differentiate

def _limit_parts(expr):
    sym, notation = parse_latex(expr)
    body, var, point, direction = _strip_limit(sym, notation)
    return {
        'sym': sym,
        'notation': notation,
        'body': body,
        'body_latex': write_latex(body, notation),
        'var': var,
        'point': point,
        'point_latex': write_latex(point, notation),
        'direction': direction,
    }


def _limit_sample_envs(parsed, var, samples, seed=20260705):
    variables = set()
    for sym, notation in parsed:
        variables |= free_symbols(sym, notation)
    variables.discard(var)
    if not variables:
        return [{}]
    rng = random.Random(seed)
    return [_sample_point(variables, rng) for _ in range(samples)]


def _aitken(a, b, c):
    d = (c - b) - (b - a)
    if abs(d) < 1e-15 * max(abs(a), abs(b), abs(c), 1e-300):
        return None
    return c - (c - b) ** 2 / d


def _approach_estimate(body, notation, var, point, point_notation,
                       direction, env):
    """Independent numeric limit estimate plus a convergence-error bound."""
    inf = _infinity_sign(point, point_notation)
    if inf is not None:
        # fixed ladders: fall back to shorter/nearer ones only when a point
        # fails (e.g. 2^{10^4} overflows in a decaying denominator, or a
        # root is undefined at the small end)
        aitken = _aitken

        for ladder in ((1e2, 1e3, 1e4, 1e5), (1e3, 1e4, 1e5),
                       (10.0, 1e2, 1e3, 1e4), (10.0, 1e2, 1e3),
                       (3.0, 30.0, 300.0)):
            vals = []
            try:
                for x in ladder:
                    e = dict(env)
                    e[var] = inf * x
                    vals.append(numeric_eval(body, notation, e))
            except (OverflowError, ValueError, ZeroDivisionError,
                    EvalError):
                continue
            if any(isinstance(v, list) for v in vals):
                raise EvalError('matrix-valued limit')
            # Richardson for a leading O(1/x) error term.
            rich = (10 * vals[-1] - vals[-2]) / 9
            rich_err = abs(vals[-2] - vals[-1]) / 9
            # Aitken/Shanks over log-spaced triples additionally captures
            # power-law decay c*x^{-alpha} (any alpha > 0), which the
            # O(1/x) model extrapolates poorly. A second window (4-point
            # ladders) gives a self-consistency error estimate.
            second = aitken(*vals[-3:])
            if second is None:
                return rich, rich_err
            first = aitken(*vals[-4:-1]) if len(vals) == 4 else None
            if first is None:
                return second, max(abs(second - rich), rich_err)
            return second, max(abs(second - first), rich_err)
        raise EvalError('approach samples failed to evaluate')

    a = numeric_eval(point, point_notation, env)
    if isinstance(a, list):
        raise EvalError('matrix-valued limit point')
    h = 1e-2 * max(1.0, abs(a))

    def at(offset):
        e = dict(env)
        e[var] = a + offset
        v = numeric_eval(body, notation, e)
        if isinstance(v, list):
            raise EvalError('matrix-valued limit')
        return v

    def value_at(step):
        if direction == 'two-sided':
            return (at(step) + at(-step)) / 2
        sign = 1 if direction == 'right' else -1
        return at(sign * step)

    def richardson(coarse, fine):
        if direction == 'two-sided':
            return (4 * fine - coarse) / 3, abs(coarse - fine) / 3
        return 2 * fine - coarse, abs(coarse - fine)

    # Geometric h-ladders with Aitken acceleration, mirroring the infinity
    # branch: the plain Richardson model assumes a smooth truncation term
    # and honestly skips |x|-kinked bodies whose approach values decay like
    # c*h^alpha.  Longer ladders fall back when a point fails to evaluate
    # (e.g. a root over an oscillating argument); the final fallback is the
    # original two-point estimate.
    for ladder in ((h, h / 2, h / 4, h / 8), (h, h / 2, h / 4)):
        try:
            vals = [value_at(s) for s in ladder]
        except (OverflowError, ValueError, ZeroDivisionError, EvalError):
            continue
        rich, rich_err = richardson(vals[-2], vals[-1])
        second = _aitken(*vals[-3:])
        if second is None:
            return rich, rich_err
        first = _aitken(*vals[-4:-1]) if len(vals) == 4 else None
        if first is None:
            return second, max(abs(second - rich), rich_err)
        return second, max(abs(second - first), rich_err)
    return richardson(value_at(h), value_at(h / 2))


def _limit_check(body_latex, var, point_latex, direction, expected_latex,
                 samples=6):
    """Check a claimed limit by approach sampling, independently of tactics."""
    try:
        body, bn = parse_latex(body_latex)
        point, pn = parse_latex(point_latex)
        expected, en = parse_latex(expected_latex)
    except PrimitiveError as e:
        return {'status': 'skipped', 'reason': str(e)}
    if var in free_symbols(expected, en):
        return {'status': 'skipped',
                'reason': 'claimed limit still contains the bound variable'}
    envs = _limit_sample_envs(
        [(body, bn), (point, pn), (expected, en)], var, samples)
    agreed = 0
    for env in envs:
        try:
            estimate, error = _approach_estimate(
                body, bn, var, point, pn, direction, env)
            expected_value = numeric_eval(expected, en, env)
        except (EvalError, ZeroDivisionError, ValueError, OverflowError):
            continue
        if isinstance(expected_value, list):
            continue
        scale = max(1.0, abs(estimate), abs(expected_value))
        delta = abs(estimate - expected_value)
        if delta / scale > 1e-4:
            if delta <= 16 * error:
                continue
            return {'status': 'disagree', 'point': env,
                    'expected': expected_value, 'numeric': estimate}
        agreed += 1
    if agreed == 0:
        return {'status': 'skipped',
                'reason': 'approach samples did not converge'}
    return {'status': 'agree', 'samples': agreed,
            'method': 'approach-sampling with Richardson/Aitken '
                      'extrapolation'}


def _limit_equivalence_check(parts, new_body, samples=6):
    """Approach-sample two bodies at the same binder and compare limits."""
    try:
        old, on = parse_latex(parts['body_latex'])
        new, nn = parse_latex(new_body)
        point, pn = parse_latex(parts['point_latex'])
    except PrimitiveError as e:
        return {'status': 'skipped', 'reason': str(e)}
    envs = _limit_sample_envs(
        [(old, on), (new, nn), (point, pn)], parts['var'], samples)
    agreed = 0
    for env in envs:
        try:
            a, ea = _approach_estimate(
                old, on, parts['var'], point, pn, parts['direction'], env)
            b, eb = _approach_estimate(
                new, nn, parts['var'], point, pn, parts['direction'], env)
        except (EvalError, ZeroDivisionError, ValueError, OverflowError):
            continue
        scale = max(1.0, abs(a), abs(b))
        delta = abs(a - b)
        if delta / scale > 1e-4:
            if delta <= 16 * (ea + eb):
                continue
            return {'status': 'disagree', 'point': env,
                    'original': a, 'transformed': b}
        agreed += 1
    if agreed == 0:
        return {'status': 'skipped',
                'reason': 'approach samples did not converge'}
    return {'status': 'agree', 'samples': agreed,
            'method': 'paired approach-sampling with Richardson/Aitken '
                      'extrapolation'}


def _limit_fraction(sym, notation):
    while True:
        g = notation.vgetf(sym, [Notation.GROUP, Notation.V_GROUP,
                                 Notation.S_GROUP])
        if g is None or g.props.get('br') == '||':
            break
        sym = g.args[0]
    f = notation.get(sym)
    if f is not None and (f.sym == Notation.SLASH
                          or f.sym.name in FRAC_NAMES):
        return f.args[0], f.args[1]
    return None


def _limit_indeterminate_form(parts, numerator, denominator):
    """Spot-check whether a quotient has 0/0 or infinity/infinity form."""
    try:
        ns, nn = parse_latex(numerator)
        ds, dn = parse_latex(denominator)
        point, pn = parse_latex(parts['point_latex'])
    except PrimitiveError:
        return None
    envs = _limit_sample_envs(
        [(ns, nn), (ds, dn), (point, pn)], parts['var'], 4)
    inf = _infinity_sign(point, pn)
    for env in envs:
        if inf is None:
            try:
                a = numeric_eval(point, pn, env)
                at = dict(env)
                at[parts['var']] = a
                nv, dv = numeric_eval(ns, nn, at), numeric_eval(ds, dn, at)
                if (not isinstance(nv, list) and not isinstance(dv, list)
                        and abs(nv) < 1e-8 and abs(dv) < 1e-8):
                    return '0/0'
            except (EvalError, ZeroDivisionError, ValueError, OverflowError):
                pass
        try:
            if inf is not None:
                xs = (inf * 1e2, inf * 1e3)
            else:
                a = numeric_eval(point, pn, env)
                sign = -1 if parts['direction'] == 'left' else 1
                h = 1e-2 * max(1.0, abs(a))
                xs = (a + sign * h, a + sign * h / 2)
            nm, dm = [], []
            for x in xs:
                e = dict(env)
                e[parts['var']] = x
                nm.append(abs(numeric_eval(ns, nn, e)))
                dm.append(abs(numeric_eval(ds, dn, e)))
            if (nm[1] > 50 and dm[1] > 50
                    and nm[1] > 1.5 * nm[0]
                    and dm[1] > 1.5 * dm[0]):
                return 'infinity/infinity'
        except (EvalError, ZeroDivisionError, ValueError, OverflowError,
                TypeError):
            continue
    return None


def limit_rewrite(expr, new_body):
    """Replace a limit body by an agent-proposed mechanically equal body."""
    args = {'expr': expr, 'new_body': new_body}
    try:
        parts = _limit_parts(expr)
        parse_latex(new_body)
    except PrimitiveError as e:
        return _error('limit_rewrite', args, str(e))
    eq = equal_exprs(new_body, parts['body_latex'])
    if not (eq.get('ok') and eq.get('verdict') == 'yes'):
        return _error('limit_rewrite', args,
                      'new body is not mechanically equal to the current '
                      f'one (verdict: {eq.get("verdict", "error")})')
    result = _limit_latex(parts['var'], parts['point_latex'],
                          parts['direction'], new_body)
    try:
        parse_latex(result)
    except PrimitiveError as e:
        return _error('limit_rewrite', args,
                      f'internal: unparseable result: {e}')
    check = {'status': 'agree',
             'method': f'body equality via equal? ({eq["method"]})'}
    if 'samples' in eq:
        check['samples'] = eq['samples']
    return _result('limit_rewrite', args, expr, result, check=check,
                   extra={'body': parts['body_latex']})


def limit_substitute(expr):
    """Apply continuity: substitute the approach point into the limit body."""
    args = {'expr': expr}
    try:
        parts = _limit_parts(expr)
    except PrimitiveError as e:
        return _error('limit_substitute', args, str(e))
    ps, pn = parse_latex(parts['point_latex'])
    if _infinity_sign(ps, pn) is not None:
        return _error('limit_substitute', args,
                      'continuity substitution requires a finite point')
    sub = substitute(parts['body_latex'], parts['var'], parts['point_latex'])
    if not sub.get('ok'):
        return _error('limit_substitute', args,
                      'cannot substitute the approach point: '
                      + sub.get('error', 'unknown error'))
    result = sub['result']
    folded = expand(result)
    if folded.get('ok') and folded.get('check', {}).get('status') != 'disagree':
        result = folded['result']
    check = _limit_check(parts['body_latex'], parts['var'],
                         parts['point_latex'], parts['direction'], result)
    if check.get('status') != 'agree':
        return _error('limit_substitute', args,
                      'continuity substitution was not confirmed by the '
                      'approach oracle: ' + check.get('reason',
                                                      check.get('status',
                                                                'unknown')))
    assumption = {'text': (f'{parts["body_latex"]} is continuous at '
                           f'{parts["var"]} = {parts["point_latex"]}'),
                  'display': (f'${parts["body_latex"]}$ is continuous at '
                              f'${parts["var"]} = {parts["point_latex"]}$')}
    return _result('limit_substitute', args, expr, result,
                   assumptions=[assumption], check=check,
                   extra={'body': parts['body_latex'],
                          'var': parts['var'],
                          'point': parts['point_latex'],
                          'direction': parts['direction']})


def limit_linearity(expr):
    """Split the limit of a top-level sum, conditional on piece limits."""
    args = {'expr': expr}
    try:
        parts = _limit_parts(expr)
    except PrimitiveError as e:
        return _error('limit_linearity', args, str(e))
    sym, notation = parse_latex(parts['body_latex'])
    while True:
        g = notation.vgetf(sym, [Notation.GROUP, Notation.V_GROUP,
                                 Notation.S_GROUP])
        if g is None or g.props.get('br') == '||':
            break
        sym = g.args[0]
    f = notation.getf(sym, Notation.S_LIST)
    if f is None:
        return _error('limit_linearity', args,
                      'limit body is not a top-level sum; nothing to split')
    terms = []
    assumptions = []
    for term in f.args:
        sign = notation.vgetf(term, [Notation.PLUS, Notation.MINUS])
        neg = sign is not None and sign.sym == Notation.MINUS
        inner = sign.args[0] if sign is not None else term
        body = write_latex(inner, notation)
        piece = _limit_latex(parts['var'], parts['point_latex'],
                             parts['direction'], body)
        terms.append({'sign': -1 if neg else 1, 'limit': piece})
        assumptions.append({'text': f'{piece} exists',
                            'display': f'${piece}$ exists'})
    result = ('-' if terms[0]['sign'] < 0 else '') + terms[0]['limit']
    for term in terms[1:]:
        result += (' - ' if term['sign'] < 0 else ' + ') + term['limit']
    try:
        parse_latex(result)
    except PrimitiveError as e:
        return _error('limit_linearity', args,
                      f'internal: unparseable result: {e}')
    return _result('limit_linearity', args, expr, result,
                   assumptions=assumptions,
                   check={'status': 'exact',
                          'method': 'linearity of limits'},
                   extra={'limits': [t['limit'] for t in terms],
                          'terms': terms})


def limit_assemble(expr, values):
    """Assemble recorded results of a ``limit_linearity`` split.

    ``values`` is ordered like the split terms.  Each candidate is checked
    independently against its own limit before the signed sum is built, and
    the whole result is checked once more against the original limit.
    """
    args = {'expr': expr,
            'values': list(values) if isinstance(values, (list, tuple))
            else values}
    if not isinstance(values, (list, tuple)):
        return _error('limit_assemble', args,
                      'values must be an ordered list')
    split = limit_linearity(expr)
    if not split.get('ok'):
        return _error('limit_assemble', args,
                      'cannot assemble without a linearity split: '
                      + split.get('error', 'unknown error'))
    terms = split['terms']
    if len(values) != len(terms):
        return _error('limit_assemble', args,
                      f'expected {len(terms)} values in linearity order, '
                      f'got {len(values)}')
    rendered = []
    piece_checks = []
    for i, (term, value) in enumerate(zip(terms, values), 1):
        if not isinstance(value, str) or not value.strip():
            return _error('limit_assemble', args,
                          f'piece {i} has no limit value')
        try:
            parse_latex(value)
            piece = _limit_parts(term['limit'])
        except PrimitiveError as e:
            return _error('limit_assemble', args,
                          f'piece {i} is malformed: {e}')
        check = _limit_check(piece['body_latex'], piece['var'],
                             piece['point_latex'], piece['direction'], value)
        if check.get('status') != 'agree':
            return _error('limit_assemble', args,
                          f'piece {i} is not confirmed as the limit of '
                          f'{piece["body_latex"]!r}')
        piece_checks.append({'piece': i, 'check': check})
        rendered.append(('-' if term['sign'] < 0 else '+', _paren(value)))
    first_sign, first = rendered[0]
    assembled = ('-' if first_sign == '-' else '') + first
    for sign, value in rendered[1:]:
        assembled += f' {sign} {value}'
    folded = expand(assembled)
    if not folded.get('ok'):
        return _error('limit_assemble', args,
                      'could not canonicalize the signed assembly: '
                      + folded.get('error', 'unknown error'))
    result = folded['result']
    original = _limit_parts(expr)
    whole_check = _limit_check(
        original['body_latex'], original['var'], original['point_latex'],
        original['direction'], result)
    if whole_check.get('status') != 'agree':
        return _error('limit_assemble', args,
                      'the signed assembly was not confirmed against the '
                      'original limit')
    check = whole_check
    for item in piece_checks:
        check = _merge_checks(check, item['check'])
    if check.get('status') == 'agree':
        check['method'] = 'per-piece limits plus signed linearity'
    return _result('limit_assemble', args, expr, result,
                   assumptions=split['assumptions'], check=check,
                   extra={'pieces': piece_checks,
                          'linearity_terms': terms})


def _plain_var_power_base(sym, notation, var):
    """Rational base value when ``sym`` is INDEX(numeric base, power = the
    plain limit variable) with no other decorations, else None."""
    f = notation.getf(sym, Notation.INDEX)
    if f is None:
        return None
    base, (sub, sup_l, power, sup_r) = f.args
    if sub is not None or sup_l is not None or sup_r is not None:
        return None
    if power is None or _plain_symbol_name(power, notation) != var:
        return None
    return _rational_const(base, notation)


def _peel_signs(sym, notation):
    """Strip transparent grouping and PLUS/MINUS wrappers (sign is
    irrelevant when deciding decay to zero)."""
    while True:
        inner = _transparent_inner(sym, notation)
        signed = notation.vgetf(inner, [Notation.MINUS, Notation.PLUS])
        if signed is None:
            return inner
        sym = signed.args[0]


def _peeled_factors(sym, notation):
    """Sign-peeled factor list of a (possibly single-factor) product."""
    sym = _peel_signs(sym, notation)
    f = notation.getf(sym, Notation.P_LIST)
    return ([_peel_signs(a, notation) for a in f.args]
            if f is not None else [sym])


def _geometric_decay_zero(body, notation, var):
    """True when ``body`` is a (signed) product/quotient whose var-bearing
    factors are all numeric-base powers ``r^var`` that decay as
    var → +infinity (0 < r < 1 in a numerator, r > 1 in a denominator)
    and every other factor is a finite constant — the geometric-decay
    standard limit ``r^n → 0``."""
    def peel(sym):
        return _peel_signs(sym, notation)

    def factor_list(sym):
        return _peeled_factors(sym, notation)

    sym = peel(body)
    frac = _limit_fraction(sym, notation)
    if frac is not None:
        sides = ((factor_list(frac[0]), False), (factor_list(frac[1]), True))
    else:
        sides = ((factor_list(sym), False), ((), True))
    decaying = 0
    for side, inverted in sides:
        for factor in side:
            if var not in free_symbols(factor, notation):
                if _contains_free_infinity(factor, notation):
                    return False
                continue
            base = _plain_var_power_base(factor, notation, var)
            if base is None or base <= 0:
                return False
            if inverted:
                if base <= 1:
                    return False
            elif base >= 1:
                return False
            decaying += 1
    return decaying > 0


def _power_decay_root(sym, notation, var):
    """Positive rational exponent q when ``sym`` is ``p(var)^q`` for a
    polynomial p with positive leading coefficient (so sym → +infinity as
    var → +infinity): ``\\sqrt{p}`` / ``\\sqrt[m]{p}`` (q = 1/m), an INDEX
    power with a positive rational literal exponent, or p itself (q = 1).
    None otherwise."""
    def poly_to_infinity(s):
        try:
            rf = to_ratfunc(s, notation)
        except (NotInFragment, ZeroDivisionError):
            return False
        if rf.variables() - {var}:
            return False
        if rf.num.degree(var) < 1 or rf.den.degree(var) != 0:
            return False
        return rf.num.leading_coeff() / rf.den.leading_coeff() > 0

    f = notation.get(sym)
    if f is not None and getattr(f.sym, 'name', None) == '\\sqrt':
        if len(f.args) == 1:
            m = 2
        else:
            m = _int_literal(f.args[1], notation)
            if m is None or m < 2:
                return None
        return Fraction(1, m) if poly_to_infinity(f.args[0]) else None
    idx = notation.getf(sym, Notation.INDEX)
    if idx is not None:
        sub_l, sup_l, power, sub_r = idx.args[1]
        if (sub_l is not None or sup_l is not None or sub_r is not None
                or power is None):
            return None
        q = _rational_const(power, notation)
        if q is None or q <= 0:
            return None
        return q if poly_to_infinity(idx.args[0]) else None
    return Fraction(1) if poly_to_infinity(sym) else None


def _power_decay_zero(body, notation, var):
    """True when ``body`` is a (signed) quotient whose numerator is
    var-free and whose denominator's var-bearing factors are all growing
    root-powers ``p(var)^q`` (q > 0 rational, p a polynomial with positive
    leading coefficient) with every other factor a finite constant — the
    root/power-decay standard limit ``1/n^q → 0`` at +infinity."""
    sym = _peel_signs(body, notation)
    frac = _limit_fraction(sym, notation)
    if frac is None:
        return False
    numerator, denominator = frac
    for factor in _peeled_factors(numerator, notation):
        if var in free_symbols(factor, notation):
            return False
        if _contains_free_infinity(factor, notation):
            return False
    decaying = 0
    for factor in _peeled_factors(denominator, notation):
        if var not in free_symbols(factor, notation):
            if _contains_free_infinity(factor, notation):
                return False
            continue
        if _power_decay_root(factor, notation, var) is None:
            return False
        decaying += 1
    return decaying > 0


def limit_table(expr):
    """Apply one narrow standard-limit or rational-at-infinity table rule."""
    args = {'expr': expr}
    try:
        parts = _limit_parts(expr)
    except PrimitiveError as e:
        return _error('limit_table', args, str(e))
    body, bn = parse_latex(parts['body_latex'])
    point, pn = parse_latex(parts['point_latex'])
    rule = None
    result = None
    if parts['var'] not in free_symbols(body, bn):
        result, rule = parts['body_latex'], 'constant'
    elif _infinity_sign(point, pn) is not None:
        try:
            rf = to_ratfunc(body, bn)
            if (rf.variables() - {parts['var']}
                    or rf.num.degree(parts['var']) > rf.den.degree(parts['var'])):
                raise NotInFragment('non-finite rational limit')
            if rf.num.degree(parts['var']) < rf.den.degree(parts['var']):
                result = '0'
            else:
                result = _q_str(rf.num.leading_coeff()
                                / rf.den.leading_coeff())
            rule = 'rational leading coefficients at infinity'
        except (NotInFragment, ZeroDivisionError):
            pass
        if (result is None and _infinity_sign(point, pn) > 0
                and _geometric_decay_zero(body, bn, parts['var'])):
            result, rule = '0', 'geometric decay at infinity'
        if (result is None and _infinity_sign(point, pn) > 0
                and _power_decay_zero(body, bn, parts['var'])):
            result, rule = '0', 'root-power decay at infinity'
    else:
        zero = equal_exprs(parts['point_latex'], '0')
        if zero.get('ok') and zero.get('verdict') == 'yes':
            x = parts['var']
            table = [
                (f'\\frac{{\\sin {x}}}{{{x}}}', '1', 'sin(x)/x'),
                (f'\\frac{{{x}}}{{\\sin {x}}}', '1', 'x/sin(x)'),
                (f'\\frac{{1-\\cos {x}}}{{{x}}}', '0',
                 '(1-cos(x))/x'),
                (f'\\frac{{1-\\cos {x}}}{{{x}^2}}', '\\frac{1}{2}',
                 '(1-cos(x))/x^2'),
                (f'\\frac{{e^{{{x}}}-1}}{{{x}}}', '1', '(e^x-1)/x'),
                (f'\\frac{{\\ln(1+{x})}}{{{x}}}', '1', 'ln(1+x)/x'),
            ]
            for pattern, value, name in table:
                eq = equal_exprs(parts['body_latex'], pattern)
                if eq.get('ok') and eq.get('verdict') == 'yes':
                    result, rule = value, name
                    break
    if result is None:
        return _error('limit_table', args,
                      'no table rule for this limit; use limit_rewrite, '
                      'limit_substitute, or limit_lhopital')
    if rule == 'constant':
        check = {'status': 'exact', 'method': 'limit of a constant'}
    else:
        check = _limit_check(parts['body_latex'], parts['var'],
                             parts['point_latex'], parts['direction'], result)
    return _result('limit_table', args, expr, result, check=check,
                   extra={'rule': rule, 'body': parts['body_latex'],
                          'var': parts['var'],
                          'point': parts['point_latex'],
                          'direction': parts['direction']})


def limit_lhopital(expr):
    """One conditional l'Hopital step on a sampled 0/0 or infinity/infinity."""
    args = {'expr': expr}
    try:
        parts = _limit_parts(expr)
        frac = _limit_fraction(parts['body'], parts['notation'])
        if frac is None:
            raise PrimitiveError('limit body is not a quotient')
        numerator = write_latex(frac[0], parts['notation'])
        denominator = write_latex(frac[1], parts['notation'])
    except PrimitiveError as e:
        return _error('limit_lhopital', args, str(e))
    form = _limit_indeterminate_form(parts, numerator, denominator)
    if form is None:
        return _error('limit_lhopital', args,
                      'oracle did not confirm a 0/0 or infinity/infinity '
                      'indeterminate form')
    nd = differentiate(numerator, parts['var'])
    dd = differentiate(denominator, parts['var'])
    if not nd.get('ok') or not dd.get('ok'):
        return _error('limit_lhopital', args,
                      'could not differentiate numerator and denominator')
    new_body = _d_frac(nd['result'], dd['result'])
    result = _limit_latex(parts['var'], parts['point_latex'],
                          parts['direction'], new_body)
    try:
        parse_latex(result)
    except PrimitiveError as e:
        return _error('limit_lhopital', args,
                      f'internal: unparseable result: {e}')
    assumptions = [
        {'text': (f'{numerator} and {denominator} are differentiable in '
                  'a punctured neighborhood of the approach point'),
         'display': (f'${numerator}$ and ${denominator}$ are differentiable '
                     'in a punctured neighborhood of the approach point')},
        {'text': (f'{dd["result"]} \\ne 0 in that punctured neighborhood'),
         'display': (f'${dd["result"]} \\ne 0$ in that punctured '
                     'neighborhood')},
        {'text': 'the transformed limit exists or diverges to infinity',
         'display': 'the transformed limit exists or diverges to infinity'},
    ]
    check = _limit_equivalence_check(parts, new_body)
    check = _merge_checks(check, nd.get('check', {'status': 'skipped'}))
    check = _merge_checks(check, dd.get('check', {'status': 'skipped'}))
    if check.get('status') == 'agree':
        check['method'] = ('paired approach samples plus independently '
                           'checked derivatives')
    return _result('limit_lhopital', args, expr, result,
                   assumptions=assumptions, check=check,
                   extra={'indeterminate_form': form,
                          'numerator_derivative': nd['result'],
                          'denominator_derivative': dd['result']})


def limit_with_body(expr, body_latex):
    """The ``\\lim`` expression sharing ``expr``'s binder with a new body.
    Used by callers that must name a bound's own limit expression."""
    parts = _limit_parts(expr)
    return _limit_latex(parts['var'], parts['point_latex'],
                        parts['direction'], body_latex)


def limit_with_direction(expr, direction):
    """The ``\\lim`` expression sharing ``expr``'s body and point with the
    given approach direction.  Used by callers that must name one side's
    own limit expression."""
    parts = _limit_parts(expr)
    return _limit_latex(parts['var'], parts['point_latex'], direction,
                        parts['body_latex'])


def limit_from_sides(expr, value):
    """Close a two-sided limit from its two agreeing one-sided limits.

    The left and right limits of the same body — both equal to ``value`` —
    must already be mechanically established; in do! and the CLI they are
    resolved from recorded ledger steps and cited as provenance.  The
    combination itself is exact (a two-sided limit exists iff both
    one-sided limits exist and agree); the approach oracle additionally
    spot-checks the two-sided value wherever it can converge."""
    args = {'expr': expr, 'value': value}
    if not isinstance(value, str) or not value.strip():
        return _error('limit_from_sides', args, 'missing value')
    try:
        parts = _limit_parts(expr)
        val_sym, val_n = parse_latex(value)
    except PrimitiveError as e:
        return _error('limit_from_sides', args, str(e))
    if parts['direction'] != 'two-sided':
        return _error('limit_from_sides', args,
                      'expected a two-sided limit; the one-sided limits '
                      'are the premises, not the target')
    if parts['var'] in free_symbols(val_sym, val_n):
        return _error('limit_from_sides', args,
                      'the limit value must not contain the bound variable')
    whole = _limit_check(parts['body_latex'], parts['var'],
                         parts['point_latex'], 'two-sided', value)
    if whole.get('status') == 'disagree':
        return _error('limit_from_sides', args,
                      'the approach oracle contradicts the combined value')
    if whole.get('status') == 'agree':
        check = dict(whole)
        check['method'] = ('agreeing one-sided limits; two-sided approach '
                           'sampling concurs')
    else:
        check = {'status': 'exact',
                 'method': ('a two-sided limit equals its agreeing '
                            'one-sided limits')}
    return _result('limit_from_sides', args, expr, value, check=check,
                   extra={'body': parts['body_latex'], 'var': parts['var'],
                          'point': parts['point_latex'],
                          'left': _limit_latex(parts['var'],
                                               parts['point_latex'],
                                               'left',
                                               parts['body_latex']),
                          'right': _limit_latex(parts['var'],
                                                parts['point_latex'],
                                                'right',
                                                parts['body_latex'])})


def _squeeze_bound_check(lower_latex, body_latex, upper_latex, var,
                         point_latex, direction, samples=6):
    """Spot-check ``lower <= body <= upper`` at approach sample points —
    the numeric leg behind a squeeze step's ordering assumption."""
    try:
        lo, lo_n = parse_latex(lower_latex)
        bo, bo_n = parse_latex(body_latex)
        up, up_n = parse_latex(upper_latex)
        point, pn = parse_latex(point_latex)
    except PrimitiveError as e:
        return {'status': 'skipped', 'reason': str(e)}
    envs = _limit_sample_envs(
        [(lo, lo_n), (bo, bo_n), (up, up_n), (point, pn)], var, samples)
    inf = _infinity_sign(point, pn)
    agreed = 0
    for env in envs:
        if inf is not None:
            offsets = [inf * x for x in (10.0, 30.0, 1e2, 1e3, 1e4)]
        else:
            try:
                a = numeric_eval(point, pn, env)
            except (EvalError, ZeroDivisionError, ValueError,
                    OverflowError):
                continue
            if isinstance(a, list):
                continue
            steps = (1e-1, 1e-2, 1e-3)
            scale = max(1.0, abs(a))
            if direction == 'right':
                offsets = [a + h * scale for h in steps]
            elif direction == 'left':
                offsets = [a - h * scale for h in steps]
            else:
                offsets = [a + s * h * scale
                           for h in steps for s in (1, -1)]
        for x in offsets:
            e = dict(env)
            e[var] = x
            try:
                low = numeric_eval(lo, lo_n, e)
                mid = numeric_eval(bo, bo_n, e)
                high = numeric_eval(up, up_n, e)
            except (EvalError, ZeroDivisionError, ValueError,
                    OverflowError):
                continue
            if any(isinstance(v, list) for v in (low, mid, high)):
                continue
            tol = 1e-9 * max(1.0, abs(low), abs(mid), abs(high))
            if low - mid > tol or mid - high > tol:
                return {'status': 'disagree', 'point': e,
                        'lower': low, 'body': mid, 'upper': high}
            agreed += 1
    if agreed == 0:
        return {'status': 'skipped',
                'reason': 'no evaluable ordering sample points'}
    return {'status': 'agree', 'samples': agreed,
            'method': 'ordering spot-checked at approach samples'}


def limit_squeeze(expr, lower, upper, value):
    """Close a limit by the squeeze theorem.

    ``lower`` and ``upper`` are agent-proposed bound bodies whose limits —
    the shared ``value`` — must already be mechanically established; in
    do! they are resolved from recorded ledger steps and cited as
    provenance. The ordering ``lower <= body <= upper`` near the approach
    point is spot-checked numerically and recorded as an assumption; each
    bound's limit is re-confirmed by the independent approach oracle."""
    args = {'expr': expr, 'lower': lower, 'upper': upper, 'value': value}
    for tag, text in (('lower bound', lower), ('upper bound', upper),
                      ('value', value)):
        if not isinstance(text, str) or not text.strip():
            return _error('limit_squeeze', args, f'missing {tag}')
    try:
        parts = _limit_parts(expr)
        lo_sym, lo_n = parse_latex(lower)
        up_sym, up_n = parse_latex(upper)
        val_sym, val_n = parse_latex(value)
    except PrimitiveError as e:
        return _error('limit_squeeze', args, str(e))
    var = parts['var']
    if var in free_symbols(val_sym, val_n):
        return _error('limit_squeeze', args,
                      'the limit value must not contain the bound variable')
    allowed = free_symbols(parts['body'], parts['notation']) | {var}
    stray = ((free_symbols(lo_sym, lo_n) | free_symbols(up_sym, up_n))
             - allowed)
    if stray:
        return _error('limit_squeeze', args,
                      'bounds introduce new free variable(s): '
                      + ', '.join(sorted(stray)))
    lower_latex = write_latex(lo_sym, lo_n)
    upper_latex = write_latex(up_sym, up_n)
    if (same_expression(lower_latex, parts['body_latex'])
            or same_expression(upper_latex, parts['body_latex'])):
        return _error('limit_squeeze', args,
                      'a bound identical to the limit body makes the '
                      'squeeze vacuous; close the limit directly')
    if same_expression(lower_latex, upper_latex):
        return _error('limit_squeeze', args,
                      'equal bounds assert the body equals them '
                      'everywhere; use limit_rewrite instead')
    sandwich = _squeeze_bound_check(lower_latex, parts['body_latex'],
                                    upper_latex, var, parts['point_latex'],
                                    parts['direction'])
    if sandwich.get('status') != 'agree':
        reason = (sandwich.get('reason')
                  if sandwich.get('status') == 'skipped'
                  else f'witness point {sandwich.get("point")}')
        return _error('limit_squeeze', args,
                      'the ordering lower <= body <= upper could not be '
                      f'confirmed: {reason}')
    check = sandwich
    for tag, bound in (('lower', lower_latex), ('upper', upper_latex)):
        bound_check = _limit_check(bound, var, parts['point_latex'],
                                   parts['direction'], value)
        if bound_check.get('status') != 'agree':
            return _error('limit_squeeze', args,
                          f'the {tag} bound is not confirmed to approach '
                          f'{value!r}')
        check = _merge_checks(check, bound_check)
    whole = _limit_check(parts['body_latex'], var, parts['point_latex'],
                         parts['direction'], value)
    if whole.get('status') == 'disagree':
        return _error('limit_squeeze', args,
                      'the approach oracle contradicts the squeezed value')
    if whole.get('status') == 'agree':
        check = _merge_checks(check, whole)
    if check.get('status') == 'agree':
        check['method'] = ('bound ordering spot-checked; both bound '
                           'limits confirmed by approach sampling')
    assumption = {
        'text': (f'{lower_latex} \\le {parts["body_latex"]} '
                 f'\\le {upper_latex}'),
        'display': (f'the squeeze ordering ${lower_latex} \\le '
                    f'{parts["body_latex"]} \\le {upper_latex}$ holds for '
                    f'${var}$ near ${parts["point_latex"]}$ (spot-checked '
                    'at approach samples, assumed throughout the '
                    'approach)'),
    }
    return _result('limit_squeeze', args, expr, value,
                   assumptions=[assumption], check=check,
                   extra={'lower': lower_latex, 'upper': upper_latex,
                          'body': parts['body_latex'], 'var': var,
                          'point': parts['point_latex'],
                          'direction': parts['direction']})

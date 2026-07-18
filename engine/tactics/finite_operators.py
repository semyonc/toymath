#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Finite-sum and finite-product tactics."""
import random

from notation import Notation, Symbol

from primitives import (
    _ELLIPSIS_NAMES, PrimitiveError, EvalError, parse_latex, write_latex,
    _big_operator_name, _plain_symbol_name, _binder_info, free_symbols,
    _num_agree, numeric_eval, _sample_point, _result, _error,
    _peel_groups, _int_literal, _strip_limit, _infinity_sign,
    _limit_latex, _paren, _is_sum_str, _normal_form, same_expression,
)
from tactics.core import (
    equal_exprs, substitute, expand,
)

# ---------------------------------------------------------------------------
# finite-sum tactics (no autonomous series evaluation: the agent proposes
# the reading/reshape/telescope, toymath mechanically checks it)
# ---------------------------------------------------------------------------



def _strip_limit_maybe(sym, notation):
    """(body, limit-binder-dict-or-None): unwrap one \\lim if present."""
    try:
        body, var, point, direction = _strip_limit(sym, notation)
    except PrimitiveError:
        return sym, None
    return body, {'var': var, 'point_latex': write_latex(point, notation),
                  'direction': direction}


def _bigop_parts(expr, op, allow_ellipsis=False, allow_infinite_upper=False):
    """Parse ``\\sum_{k=a}^{b} body`` / ``\\prod_{k=a}^{b} body`` —
    optionally inside a ``\\lim`` binder — into its pieces; raises
    PrimitiveError otherwise.

    ``allow_infinite_upper`` admits a ``+\\infty`` upper bound and is for
    ``series_partial_sums`` ONLY: every numeric leg refuses infinite
    bounds (infinity must never become a sampled variable)."""
    word = 'sum' if op == '\\sum' else 'product'
    shape = f'{op}_{{k=a}}^{{b}}'
    sym, notation = parse_latex(expr, allow_ellipsis=allow_ellipsis)
    body, limit = _strip_limit_maybe(sym, notation)
    body = _peel_groups(body, notation)
    f = notation.getf(body, Notation.P_LIST)
    if f is None:
        raise PrimitiveError(f'expected a {shape} expression')
    items = [a for a in f.args if not (isinstance(a, Symbol)
                                       and notation.get(a) is None
                                       and a.name in Notation.styles)]
    if not items or _big_operator_name(items[0], notation) != op:
        raise PrimitiveError(f'expected a {shape} expression')
    info = _binder_info(items[0], notation, items[1:])
    if info['bound'] is None or len(info['parameters']) != 2:
        raise PrimitiveError(f'{word} binder must have the form {shape}')
    if not info['body']:
        raise PrimitiveError(f'{word} has no {word} body')
    summand = (info['body'][0] if len(info['body']) == 1 else
               notation.setf(Notation.P_LIST, tuple(info['body'])))
    lower, upper = info['parameters']
    bound = info['bound']
    if bound in (free_symbols(lower, notation)
                 | free_symbols(upper, notation)):
        raise PrimitiveError(
            f'{word} bounds must not contain the bound variable')
    upper_inf = _infinity_sign(upper, notation)
    if (_infinity_sign(lower, notation) is not None
            or (upper_inf is not None
                and not (allow_infinite_upper and upper_inf > 0))):
        raise PrimitiveError(
            f'infinite {word} bounds are not supported here; apply '
            'series_partial_sums to rewrite the series as the limit of its '
            f'partial {word}s (\\lim_{{n \\to \\infty}} '
            f'{op}_{{k=a}}^{{n}} ...), then drive the {word} and limit '
            'tactics')
    if isinstance(lower, Symbol):
        lower = _peel_groups(lower, notation)
    if isinstance(upper, Symbol):
        upper = _peel_groups(upper, notation)
    return {
        'limit': limit,
        'bound': bound,
        'lower': lower, 'lower_latex': write_latex(lower, notation),
        'upper': upper, 'upper_latex': write_latex(upper, notation),
        'summand': summand,
        'summand_latex': write_latex(summand, notation),
        'notation': notation,
    }


def _sum_parts(expr, allow_ellipsis=False):
    return _bigop_parts(expr, '\\sum', allow_ellipsis=allow_ellipsis)


def _bigop_latex(op, bound, lower_latex, upper_latex, summand_latex):
    body = summand_latex.strip()
    if _is_sum_str(body) or body.startswith('-'):
        body = _paren(body)
    return f'{op}_{{{bound}={lower_latex}}}^{{{upper_latex}}} {body}'


def _sum_latex(bound, lower_latex, upper_latex, summand_latex):
    return _bigop_latex('\\sum', bound, lower_latex, upper_latex,
                        summand_latex)


def _rewrap_limit(limit, body_latex):
    if limit is None:
        return body_latex
    return _limit_latex(limit['var'], limit['point_latex'],
                        limit['direction'], body_latex)


def _proposal_limit_error(proposal_limit, expr_limit, op, label):
    """Agents naturally re-type the whole limit when proposing the explicit
    operator form, so a ``\\lim`` wrapper on the proposal is accepted as a
    redundant spelling of the expression's own binder — and only then.
    Returns None to accept, or a steering message naming the exact repair."""
    if expr_limit is None:
        return (f'{label} carries a \\lim but the expression has none; '
                f'pass the bare {op} form')
    if (proposal_limit['var'] != expr_limit['var']
            or proposal_limit['direction'] != expr_limit['direction']
            or _normal_form(proposal_limit['point_latex'])
            != _normal_form(expr_limit['point_latex'])):
        return (f"{label}'s \\lim binder differs from the expression's; "
                f'keep the whole \\lim expression as expr (the binder is '
                f'carried through) and pass the bare {op} it abbreviates')
    return None


def _finite_sum_check(sum_latex, closed_latex, upper_var, lower_int,
                      samples=6, seed=20260705, word='sum'):
    """Evaluate the literal finite sum/product against a closed form at
    several integer upper bounds — the independent numeric leg for the
    finite-operator tactics."""
    try:
        ss, sn = parse_latex(sum_latex)
        cs, cn = parse_latex(closed_latex)
    except PrimitiveError as e:
        return {'status': 'skipped', 'reason': str(e)}
    variables = free_symbols(ss, sn) | free_symbols(cs, cn)
    if upper_var is not None:
        variables.discard(upper_var)
    deltas = (-1, 0, 1, 2, 3, 5, 9) if upper_var is not None \
        else tuple(range(samples))
    rng = random.Random(seed)
    agreed = 0
    for delta in deltas:
        env = _sample_point(variables, rng)
        if upper_var is not None:
            env[upper_var] = float(lower_int + delta)
        try:
            v1 = numeric_eval(ss, sn, env)
            v2 = numeric_eval(cs, cn, env)
        except (EvalError, ZeroDivisionError, ValueError, OverflowError):
            continue
        agree = _num_agree(v1, v2, 1e-6)
        if agree is None:
            continue
        if not agree:
            return {'status': 'disagree', 'point': env,
                    'sum': v1, 'closed': v2}
        agreed += 1
    if agreed == 0:
        return {'status': 'skipped',
                'reason': 'no evaluable integer sample points'}
    return {'status': 'agree', 'samples': agreed,
            'method': f'literal finite-{word} evaluation vs closed form'}


def sum_from_ellipsis(expr, sum_form):
    """Interpret an ellipsis sum as an explicit finite ``\\sum``.

    Every displayed term is mechanically checked against the proposed
    summand at its index; the continuation of the pattern under the
    ellipsis is recorded as an assumption, not proved."""
    args = {'expr': expr, 'sum_form': sum_form}
    try:
        sym, notation = parse_latex(expr, allow_ellipsis=True)
    except PrimitiveError as e:
        return _error('sum_from_ellipsis', args, str(e))
    body, limit = _strip_limit_maybe(sym, notation)
    body = _peel_groups(body, notation)
    body_latex = write_latex(body, notation)
    f = notation.getf(body, Notation.S_LIST)
    if f is None:
        return _error('sum_from_ellipsis', args,
                      'expected a sum of terms containing an ellipsis')
    terms = []
    ellipsis_at = None
    for idx, t in enumerate(f.args):
        sign = notation.vgetf(t, [Notation.PLUS, Notation.MINUS])
        if sign is not None and sign.sym == Notation.MINUS:
            return _error('sum_from_ellipsis', args,
                          'only plain + ellipsis sums are supported')
        inner = sign.args[0] if sign is not None else t
        peeled = _peel_groups(inner, notation)
        if (isinstance(peeled, Symbol) and notation.get(peeled) is None
                and peeled.name in _ELLIPSIS_NAMES):
            if ellipsis_at is not None:
                return _error('sum_from_ellipsis', args,
                              'more than one ellipsis in the sum')
            ellipsis_at = idx
            continue
        terms.append((idx, write_latex(inner, notation)))
    if ellipsis_at is None:
        return _error('sum_from_ellipsis', args,
                      'no ellipsis found; use sum_rewrite for explicit sums')
    prefix = [t for t in terms if t[0] < ellipsis_at]
    suffix = [t for t in terms if t[0] > ellipsis_at]
    if len(prefix) < 2:
        return _error('sum_from_ellipsis', args,
                      'need at least two displayed leading terms to anchor '
                      'the pattern')
    if not suffix:
        return _error('sum_from_ellipsis', args,
                      'the ellipsis must be followed by the final term(s)')
    try:
        parts = _sum_parts(sum_form)
    except PrimitiveError as e:
        return _error('sum_from_ellipsis', args, f'sum_form: {e}')
    if parts['limit'] is not None:
        err = _proposal_limit_error(parts['limit'], limit, '\\sum',
                                    'sum_form')
        if err:
            return _error('sum_from_ellipsis', args, err)
    body_free = free_symbols(body, notation) - _ELLIPSIS_NAMES
    form_sym, form_notation = parse_latex(sum_form)
    stray = free_symbols(form_sym, form_notation) - body_free
    if stray:
        return _error('sum_from_ellipsis', args,
                      'sum_form introduces new free variable(s): '
                      + ', '.join(sorted(stray)))
    k = parts['bound']
    lo = _int_literal(parts['lower'], parts['notation'])
    if lo is None:
        return _error('sum_from_ellipsis', args,
                      'sum lower bound must be an integer literal')
    matched = []
    todo = ([(term, str(lo + i)) for i, (_, term) in enumerate(prefix)]
            + [(term, parts['upper_latex'] if j == len(suffix) - 1 else
                f'{parts["upper_latex"]}-{len(suffix) - 1 - j}')
               for j, (_, term) in enumerate(suffix)])
    for term_latex, k_value in todo:
        sub = substitute(parts['summand_latex'], k, k_value)
        if not sub.get('ok'):
            return _error('sum_from_ellipsis', args,
                          f'cannot instantiate the summand at {k}={k_value}: '
                          + sub.get('error', 'unknown error'))
        eq = equal_exprs(sub['result'], term_latex)
        if not (eq.get('ok') and eq.get('verdict') == 'yes'):
            return _error('sum_from_ellipsis', args,
                          f'displayed term {term_latex!r} does not match the '
                          f'summand at {k}={k_value} (verdict: '
                          f'{eq.get("verdict", "error")})')
        matched.append({'k': k_value, 'term': term_latex,
                        'method': eq.get('method')})
    canonical = _sum_latex(k, parts['lower_latex'], parts['upper_latex'],
                           parts['summand_latex'])
    result = _rewrap_limit(limit, canonical)
    try:
        parse_latex(result)
    except PrimitiveError as e:
        return _error('sum_from_ellipsis', args,
                      f'internal: unparseable result: {e}')
    assumption = {
        'text': f'{body_latex} = {canonical}',
        'display': (f'the ellipsis in ${body_latex}$ continues the pattern '
                    f'of ${parts["summand_latex"]}$: the sum is read as '
                    f'${canonical}$'),
    }
    return _result('sum_from_ellipsis', args, expr, result,
                   assumptions=[assumption],
                   check={'status': 'exact',
                          'method': (f'all {len(matched)} displayed terms '
                                     'match the summand at their index; the '
                                     'continuation is recorded as an '
                                     'assumption')},
                   extra={'sum': canonical, 'matched': matched})


def prod_from_ellipsis(expr, prod_form):
    """Interpret an ellipsis product as an explicit finite ``\\prod``.

    Every displayed factor is mechanically checked against the proposed
    factor at its index; the continuation of the pattern under the
    ellipsis is recorded as an assumption, not proved."""
    args = {'expr': expr, 'prod_form': prod_form}
    try:
        sym, notation = parse_latex(expr, allow_ellipsis=True)
    except PrimitiveError as e:
        return _error('prod_from_ellipsis', args, str(e))
    body, limit = _strip_limit_maybe(sym, notation)
    body = _peel_groups(body, notation)
    body_latex = write_latex(body, notation)
    f = notation.getf(body, Notation.P_LIST)
    if f is None:
        return _error('prod_from_ellipsis', args,
                      'expected a product of factors containing an ellipsis')
    factors = []
    ellipsis_at = None
    idx = -1
    for t in f.args:
        if (isinstance(t, Symbol) and notation.get(t) is None
                and t.name in Notation.styles):
            continue
        idx += 1
        peeled = _peel_groups(t, notation)
        if (isinstance(peeled, Symbol) and notation.get(peeled) is None
                and peeled.name in _ELLIPSIS_NAMES):
            if ellipsis_at is not None:
                return _error('prod_from_ellipsis', args,
                              'more than one ellipsis in the product')
            ellipsis_at = idx
            continue
        factors.append((idx, write_latex(t, notation)))
    if ellipsis_at is None:
        return _error('prod_from_ellipsis', args,
                      'no ellipsis found; the product is already explicit')
    prefix = [t for t in factors if t[0] < ellipsis_at]
    suffix = [t for t in factors if t[0] > ellipsis_at]
    if len(prefix) < 2:
        return _error('prod_from_ellipsis', args,
                      'need at least two displayed leading factors to '
                      'anchor the pattern')
    if not suffix:
        return _error('prod_from_ellipsis', args,
                      'the ellipsis must be followed by the final factor(s)')
    try:
        parts = _bigop_parts(prod_form, '\\prod')
    except PrimitiveError as e:
        return _error('prod_from_ellipsis', args, f'prod_form: {e}')
    if parts['limit'] is not None:
        err = _proposal_limit_error(parts['limit'], limit, '\\prod',
                                    'prod_form')
        if err:
            return _error('prod_from_ellipsis', args, err)
    body_free = free_symbols(body, notation) - _ELLIPSIS_NAMES
    form_sym, form_notation = parse_latex(prod_form)
    stray = free_symbols(form_sym, form_notation) - body_free
    if stray:
        return _error('prod_from_ellipsis', args,
                      'prod_form introduces new free variable(s): '
                      + ', '.join(sorted(stray)))
    k = parts['bound']
    lo = _int_literal(parts['lower'], parts['notation'])
    if lo is None:
        return _error('prod_from_ellipsis', args,
                      'product lower bound must be an integer literal')
    matched = []
    todo = ([(term, str(lo + i)) for i, (_, term) in enumerate(prefix)]
            + [(term, parts['upper_latex'] if j == len(suffix) - 1 else
                f'{parts["upper_latex"]}-{len(suffix) - 1 - j}')
               for j, (_, term) in enumerate(suffix)])
    for term_latex, k_value in todo:
        sub = substitute(parts['summand_latex'], k, k_value)
        if not sub.get('ok'):
            return _error('prod_from_ellipsis', args,
                          f'cannot instantiate the factor at {k}={k_value}: '
                          + sub.get('error', 'unknown error'))
        eq = equal_exprs(sub['result'], term_latex)
        if not (eq.get('ok') and eq.get('verdict') == 'yes'):
            return _error('prod_from_ellipsis', args,
                          f'displayed factor {term_latex!r} does not match '
                          f'the proposed factor at {k}={k_value} (verdict: '
                          f'{eq.get("verdict", "error")})')
        matched.append({'k': k_value, 'factor': term_latex,
                        'method': eq.get('method')})
    canonical = _bigop_latex('\\prod', k, parts['lower_latex'],
                             parts['upper_latex'], parts['summand_latex'])
    result = _rewrap_limit(limit, canonical)
    try:
        parse_latex(result)
    except PrimitiveError as e:
        return _error('prod_from_ellipsis', args,
                      f'internal: unparseable result: {e}')
    assumption = {
        'text': f'{body_latex} = {canonical}',
        'display': (f'the ellipsis in ${body_latex}$ continues the pattern '
                    f'of ${parts["summand_latex"]}$: the product is read as '
                    f'${canonical}$'),
    }
    return _result('prod_from_ellipsis', args, expr, result,
                   assumptions=[assumption],
                   check={'status': 'exact',
                          'method': (f'all {len(matched)} displayed factors '
                                     'match the proposed factor at their '
                                     'index; the continuation is recorded '
                                     'as an assumption')},
                   extra={'prod': canonical, 'matched': matched})


def sum_rewrite(expr, new_summand):
    """Replace the summand of a finite sum by an agent-proposed
    mechanically equal one."""
    args = {'expr': expr, 'new_summand': new_summand}
    try:
        parts = _sum_parts(expr)
        parse_latex(new_summand)
    except PrimitiveError as e:
        return _error('sum_rewrite', args, str(e))
    eq = equal_exprs(new_summand, parts['summand_latex'])
    if not (eq.get('ok') and eq.get('verdict') == 'yes'):
        return _error('sum_rewrite', args,
                      'new summand is not mechanically equal to the current '
                      f'one (verdict: {eq.get("verdict", "error")})')
    result = _rewrap_limit(
        parts['limit'], _sum_latex(parts['bound'], parts['lower_latex'],
                                   parts['upper_latex'], new_summand))
    try:
        parse_latex(result)
    except PrimitiveError as e:
        return _error('sum_rewrite', args,
                      f'internal: unparseable result: {e}')
    check = {'status': 'agree',
             'method': f'summand equality via equal? ({eq["method"]})'}
    if 'samples' in eq:
        check['samples'] = eq['samples']
    return _result('sum_rewrite', args, expr, result, check=check,
                   extra={'summand': parts['summand_latex']})


def sum_telescope(expr, term):
    """Collapse ``\\sum_{k=a}^{b} (f(k) - f(k+1))`` to ``f(a) - f(b+1)``
    for the agent-proposed ``f``, written in the bound variable."""
    args = {'expr': expr, 'term': term}
    try:
        parts = _sum_parts(expr)
        parse_latex(term)
    except PrimitiveError as e:
        return _error('sum_telescope', args, str(e))
    k = parts['bound']
    lo = _int_literal(parts['lower'], parts['notation'])
    if lo is None:
        return _error('sum_telescope', args,
                      'sum lower bound must be an integer literal')
    upper_var = _plain_symbol_name(parts['upper'], parts['notation'])
    upper_int = _int_literal(parts['upper'], parts['notation'])
    if upper_var is None and upper_int is None:
        return _error('sum_telescope', args,
                      'sum upper bound must be a plain symbol or an '
                      'integer literal')
    shifted = substitute(term, k, f'{k}+1')
    if not shifted.get('ok'):
        return _error('sum_telescope', args,
                      f'cannot form the term at {k}+1: '
                      + shifted.get('error', 'unknown error'))
    head = _paren(term) if (_is_sum_str(term.strip())
                            or term.strip().startswith('-')) \
        else term.strip()
    eq = equal_exprs(parts['summand_latex'],
                     f'{head} - {_paren(shifted["result"])}')
    if not (eq.get('ok') and eq.get('verdict') == 'yes'):
        return _error('sum_telescope', args,
                      f'summand is not mechanically equal to the difference '
                      f'{term} - ({term} at {k}+1) (verdict: '
                      f'{eq.get("verdict", "error")})')
    first = substitute(term, k, parts['lower_latex'])
    last = substitute(term, k, f'{parts["upper_latex"]}+1')
    if not first.get('ok') or not last.get('ok'):
        return _error('sum_telescope', args,
                      'cannot instantiate the telescoping term at the '
                      'bounds')
    closed = (
        (_paren(first['result']) if _is_sum_str(first['result'])
         else first['result'])
        + ' - ' + _paren(last['result']))
    folded = expand(closed)
    if folded.get('ok') and folded.get('check', {}).get('status') != \
            'disagree':
        closed = folded['result']
    check = _finite_sum_check(
        _sum_latex(k, parts['lower_latex'], parts['upper_latex'],
                   parts['summand_latex']),
        closed, upper_var, lo)
    if check.get('status') != 'agree':
        return _error('sum_telescope', args,
                      'telescoped form was not confirmed by the finite-sum '
                      'oracle: ' + check.get('reason',
                                             check.get('status', 'unknown')))
    check = dict(check)
    check['method'] += ' (summand gated by equal?)'
    assumptions = [
        {'text': f'{parts["upper_latex"]} \\ge {parts["lower_latex"]} - 1',
         'display': (f'${parts["upper_latex"]} \\ge '
                     f'{parts["lower_latex"]} - 1$ '
                     '(below that the empty-sum convention applies)')},
        {'text': (f'{term}\\ \\text{{is defined for}}\\ {k} = '
                  f'{parts["lower_latex"]}, \\dots, {parts["upper_latex"]}'
                  '+1'),
         'display': (f'${term}$ is defined at every integer ${k}$ from '
                     f'${parts["lower_latex"]}$ to ${parts["upper_latex"]}'
                     '+1$')},
    ]
    result = _rewrap_limit(parts['limit'], closed)
    try:
        parse_latex(result)
    except PrimitiveError as e:
        return _error('sum_telescope', args,
                      f'internal: unparseable result: {e}')
    return _result('sum_telescope', args, expr, result,
                   assumptions=assumptions, check=check,
                   extra={'summand': parts['summand_latex'],
                          'closed_form': closed})


_FRESH_BOUND_POOL = ('n', 'm', 'N', 'M', 'j', 'p', 'q', 's')


def series_partial_sums(expr):
    """Rewrite an infinite ``\\sum``/``\\prod`` as the limit of its partial
    sums/products.

    Exact by definition: the value of an infinite series (or product) IS
    the limit of its partial sums (products), when that limit exists.
    Existence is recorded as an assumption, never assumed proved."""
    args = {'expr': expr}
    parts = op = None
    first_error = None
    for candidate in ('\\sum', '\\prod'):
        try:
            parts = _bigop_parts(expr, candidate, allow_infinite_upper=True)
            op = candidate
            break
        except PrimitiveError as e:
            # 'expected a ...' means the input is not this operator at
            # all; any other message is a defect in a recognized operator
            # and must surface instead of the other operator's mismatch
            if first_error is None and not str(e).startswith('expected a'):
                first_error = str(e)
    if parts is None:
        message = first_error or \
            'expected an infinite \\sum or \\prod expression'
        if 'series_partial_sums' in message:
            message = ('only an infinite upper bound has a partial-'
                       'sum/product reading; an infinite lower bound is '
                       'not supported')
        return _error('series_partial_sums', args, message)
    word = 'sum' if op == '\\sum' else 'product'
    if parts['limit'] is not None:
        return _error('series_partial_sums', args,
                      f'unexpected \\lim around the {word}; apply '
                      'series_partial_sums to the bare infinite '
                      f'{word} itself')
    upper_inf = _infinity_sign(parts['upper'], parts['notation'])
    if upper_inf is None or upper_inf <= 0:
        return _error('series_partial_sums', args,
                      f'the {word} upper bound is already finite; drive the '
                      f'finite {word} tactics directly')
    try:
        sym, notation = parse_latex(expr)
    except PrimitiveError as e:
        return _error('series_partial_sums', args, str(e))
    used = free_symbols(sym, notation) | {parts['bound']}
    fresh = next((name for name in _FRESH_BOUND_POOL if name not in used),
                 None)
    if fresh is None:
        return _error('series_partial_sums', args,
                      'no free name available for the partial-bound variable')
    partial = _bigop_latex(op, parts['bound'], parts['lower_latex'], fresh,
                           parts['summand_latex'])
    result = _limit_latex(fresh, '\\infty', 'two-sided', partial)
    try:
        parse_latex(result)
    except PrimitiveError as e:
        return _error('series_partial_sums', args,
                      f'internal: unparseable result: {e}')
    assumption = {
        'text': (f'\\text{{the infinite {word} denotes}}\\ {result}'
                 '\\ \\text{when that limit exists}'),
        'display': (f'by definition the infinite {word} equals the limit of '
                    f'its partial {word}s when that limit exists; if the '
                    f'limit does not exist the {word} diverges and has no '
                    'value'),
    }
    return _result('series_partial_sums', args, expr, result,
                   assumptions=[assumption],
                   check={'status': 'exact',
                          'method': (f'definition: an infinite {word} is '
                                     f'the limit of its partial {word}s')},
                   extra={'partial_form': partial, 'partial_bound': fresh})


def _bigop_closed_form(name, op, expr, closed_form):
    """Shared implementation for sum_closed_form / prod_closed_form: the
    agent proposes an equal form in the upper-bound variable; the literal
    finite-operator loop checks it at several integer bounds (including
    the empty case) and the integer domain is recorded as an assumption.

    This is deliberately a NAMED tactic and not a widening of general
    equal?: bound variables are integer-valued by the semantics of
    \\sum/\\prod, and silently integer-sampling free variables in equal?
    would hide that domain restriction."""
    word = 'sum' if op == '\\sum' else 'product'
    args = {'expr': expr, 'closed_form': closed_form}
    try:
        parts = _bigop_parts(expr, op)
    except PrimitiveError as e:
        return _error(name, args, str(e))
    k = parts['bound']
    lo = _int_literal(parts['lower'], parts['notation'])
    if lo is None:
        return _error(name, args,
                      f'{word} lower bound must be an integer literal')
    upper_var = _plain_symbol_name(parts['upper'], parts['notation'])
    upper_int = _int_literal(parts['upper'], parts['notation'])
    if upper_var is None and upper_int is None:
        return _error(name, args,
                      f'{word} upper bound must be a plain symbol or an '
                      'integer literal')
    try:
        c_sym, c_notation = parse_latex(closed_form)
    except PrimitiveError as e:
        return _error(name, args, f'closed_form: {e}')
    c_body, c_limit = _strip_limit_maybe(c_sym, c_notation)
    closed_latex = closed_form.strip()
    if c_limit is not None:
        err = _proposal_limit_error(c_limit, parts['limit'], op,
                                    'closed_form')
        if err:
            return _error(name, args, err)
        closed_latex = write_latex(c_body, c_notation)
    try:
        expr_sym, expr_notation = parse_latex(expr)
        stray_sym, stray_notation = parse_latex(closed_latex)
    except PrimitiveError as e:
        return _error(name, args, f'closed_form: {e}')
    # under a \lim wrapper the binder-aware free set excludes the upper-
    # bound variable, yet the closed form is written exactly in it
    allowed = free_symbols(expr_sym, expr_notation)
    if upper_var is not None:
        allowed = allowed | {upper_var}
    stray = free_symbols(stray_sym, stray_notation) - allowed
    if k in stray:
        return _error(name, args,
                      f'closed_form must not contain the bound variable '
                      f'{k}; propose a form in the upper-bound variable')
    if stray:
        return _error(name, args,
                      'closed_form introduces new free variable(s): '
                      + ', '.join(sorted(stray)))
    bigop = _bigop_latex(op, k, parts['lower_latex'], parts['upper_latex'],
                         parts['summand_latex'])
    if same_expression(bigop, closed_latex):
        return _error(name, args,
                      f'closed_form is the {word} itself; propose the '
                      f'closed value the {word} equals')
    check = _finite_sum_check(bigop, closed_latex,
                              upper_var, lo, word=word)
    if check.get('status') != 'agree':
        detail = check.get('reason', check.get('status', 'unknown'))
        if check.get('status') == 'disagree':
            detail = (f'disagree at {check.get("point")}: {word} = '
                      f'{check.get("sum")}, proposal = {check.get("closed")}')
        return _error(name, args,
                      'the proposed closed form was not confirmed by the '
                      f'literal finite-{word} oracle: {detail}')
    check = dict(check)
    check['method'] += ' (agent-proposed form, integer-domain identity)'
    assumptions = []
    if upper_var is not None:
        assumptions.append({
            'text': (f'{parts["upper_latex"]} \\in \\mathbb{{Z}},\\ '
                     f'{parts["upper_latex"]} \\ge {lo - 1}'),
            'display': (f'the identity is checked on the integer domain '
                        f'only: ${parts["upper_latex"]}$ is an integer with '
                        f'${parts["upper_latex"]} \\ge {lo - 1}$ (below '
                        f'{lo} the empty-{word} convention applies)'),
        })
    result = _rewrap_limit(parts['limit'], closed_latex)
    try:
        parse_latex(result)
    except PrimitiveError as e:
        return _error(name, args, f'internal: unparseable result: {e}')
    return _result(name, args, expr, result,
                   assumptions=assumptions, check=check,
                   extra={'closed_form': closed_latex,
                          'summand': parts['summand_latex']})


def sum_closed_form(expr, closed_form):
    """Replace a finite ``\\sum`` by an agent-proposed closed form in the
    upper-bound variable, checked by literal summation at several integer
    bounds; the integer domain is recorded as an assumption."""
    return _bigop_closed_form('sum_closed_form', '\\sum', expr, closed_form)


def prod_closed_form(expr, closed_form):
    """Replace a finite ``\\prod`` by an agent-proposed closed form (for
    example a factorial ratio) in the upper-bound variable, checked by
    literal multiplication at several integer bounds; the integer domain
    is recorded as an assumption."""
    return _bigop_closed_form('prod_closed_form', '\\prod', expr,
                              closed_form)


_EXPAND_TERM_CAP = 30


def _bigop_expand(name, op, expr):
    """Shared implementation for sum_expand / prod_expand: write a
    literal-bounds finite operator out term by term and fold the result;
    the literal accumulation loop is the independent check leg."""
    word = 'sum' if op == '\\sum' else 'product'
    closed_name = ('sum_closed_form' if op == '\\sum'
                   else 'prod_closed_form')
    args = {'expr': expr}
    try:
        parts = _bigop_parts(expr, op)
    except PrimitiveError as e:
        return _error(name, args, str(e))
    k = parts['bound']
    lo = _int_literal(parts['lower'], parts['notation'])
    hi = _int_literal(parts['upper'], parts['notation'])
    if lo is None or hi is None:
        return _error(name, args,
                      f'both {word} bounds must be integer literals; for a '
                      f'symbolic bound propose a form via {closed_name}'
                      + (' or sum_telescope' if op == '\\sum' else ''))
    count = hi - lo + 1
    if count > _EXPAND_TERM_CAP:
        return _error(name, args,
                      f'{count} terms exceed the expansion cap '
                      f'({_EXPAND_TERM_CAP}); propose a form via '
                      f'{closed_name} instead')
    terms = []
    if count <= 0:
        joined = '0' if op == '\\sum' else '1'
    else:
        for value in range(lo, hi + 1):
            sub = substitute(parts['summand_latex'], k, str(value))
            if not sub.get('ok'):
                return _error(name, args,
                              f'cannot instantiate the {word} term at '
                              f'{k}={value}: '
                              + sub.get('error', 'unknown error'))
            terms.append(sub['result'])
        if op == '\\sum':
            joined = ' + '.join(
                _paren(t) if (_is_sum_str(t.strip())
                              or t.strip().startswith('-')) else t.strip()
                for t in terms)
        else:
            joined = ' \\cdot '.join(
                _paren(t) if (_is_sum_str(t.strip())
                              or t.strip().startswith('-')) else t.strip()
                for t in terms)
    body = joined
    folded = expand(joined)
    if folded.get('ok') and folded.get('check', {}).get('status') != \
            'disagree':
        body = folded['result']
    bigop = _bigop_latex(op, k, parts['lower_latex'], parts['upper_latex'],
                         parts['summand_latex'])
    eq = equal_exprs(bigop, body)
    if not (eq.get('ok') and eq.get('verdict') == 'yes'):
        return _error(name, args,
                      f'the expanded form was not confirmed against the '
                      f'literal {word} (verdict: '
                      f'{eq.get("verdict", "error")})')
    result = _rewrap_limit(parts['limit'], body)
    try:
        parse_latex(result)
    except PrimitiveError as e:
        return _error(name, args, f'internal: unparseable result: {e}')
    check = {'status': 'agree',
             'method': (f'literal finite-{word} evaluation vs the '
                        f'expanded form ({eq.get("method", "equal?")})')}
    if 'samples' in eq:
        check['samples'] = eq['samples']
    if count <= 0:
        check = {'status': 'exact',
                 'method': f'empty-{word} convention '
                           f'({parts["upper_latex"]} < '
                           f'{parts["lower_latex"]})'}
    return _result(name, args, expr, result, check=check,
                   extra={'terms': terms,
                          'summand': parts['summand_latex']})


def sum_expand(expr):
    """Write a literal-bounds finite ``\\sum`` out term by term and fold
    it; the literal summation loop is the independent check leg."""
    return _bigop_expand('sum_expand', '\\sum', expr)


def prod_expand(expr):
    """Write a literal-bounds finite ``\\prod`` out factor by factor and
    fold it; the literal multiplication loop is the independent check
    leg."""
    return _bigop_expand('prod_expand', '\\prod', expr)

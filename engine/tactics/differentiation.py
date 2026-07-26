#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Differentiation tactic and its independent derivative checker."""
import random

from notation import Notation, Symbol
from value import Value
from polyrat import (RatFunc, NotInFragment, to_ratfunc,
                     ratfunc_to_notation, _index_power,
                     FUNCTION_NAMES as FUNC_NAMES)

from primitives import (
    FRAC_NAMES, PrimitiveError, EvalError, parse_latex, write_latex,
    canonical_or_same,
    free_symbols, numeric_eval, _func_power, _func_arg_span, _sample_point,
    _result, _error, _paren, _is_sum_str,
)

# ---------------------------------------------------------------------------
# primitive: differentiate
# ---------------------------------------------------------------------------

def _d_frac(a, b):
    return f'\\frac{{{a}}}{{{b}}}'


def _d_mul(*factors):
    fs = [f for f in factors if f != '1']
    if any(f == '0' for f in fs):
        return '0'
    if not fs:
        return '1'
    out = []
    for f in fs:
        if f.startswith('-') or _is_sum_str(f):
            out.append(_paren(f))
        else:
            out.append(f)
    return ' '.join(out)


def _d_add(*terms):
    ts = [t for t in terms if t != '0']
    if not ts:
        return '0'
    res = ts[0]
    for t in ts[1:]:
        res += t if t.startswith('-') else ' + ' + t
    return res


def _d_neg(a):
    if a == '0':
        return '0'
    if a.startswith('-'):
        return a[1:]
    if _is_sum_str(a):
        return '-' + _paren(a)
    return '-' + a


_DERIV_TABLE = {
    '\\sin': lambda u, du: _d_mul(f'\\cos{_paren(u)}', du),
    '\\cos': lambda u, du: _d_neg(_d_mul(f'\\sin{_paren(u)}', du)),
    '\\tan': lambda u, du: _d_frac(du, f'\\cos^{{2}}{_paren(u)}'),
    '\\cot': lambda u, du: _d_neg(_d_frac(du, f'\\sin^{{2}}{_paren(u)}')),
    '\\sinh': lambda u, du: _d_mul(f'\\cosh{_paren(u)}', du),
    '\\cosh': lambda u, du: _d_mul(f'\\sinh{_paren(u)}', du),
    '\\tanh': lambda u, du: _d_frac(du, f'\\cosh^{{2}}{_paren(u)}'),
    '\\ln': lambda u, du: _d_frac(du, u),
    '\\log': lambda u, du: _d_frac(du, _d_mul(u, '\\ln(10)')),
    '\\exp': lambda u, du: _d_mul(f'\\exp{_paren(u)}', du),
    '\\arcsin': lambda u, du: _d_frac(du, f'\\sqrt{{1 - {_paren(u)}^{{2}}}}'),
    '\\arccos': lambda u, du: _d_neg(
        _d_frac(du, f'\\sqrt{{1 - {_paren(u)}^{{2}}}}')),
    '\\arctan': lambda u, du: _d_frac(du, f'1 + {_paren(u)}^{{2}}'),
}


def _diff(sym, notation, var):
    """Return (expr_latex, derivative_latex)."""
    if isinstance(sym, Value):
        return write_latex(sym, notation), '0'
    if not isinstance(sym, Symbol):
        raise PrimitiveError(f'cannot differentiate term {sym!r}')
    f = notation.get(sym)
    if f is None:
        s = sym.name
        if s in FUNC_NAMES:
            raise PrimitiveError(
                f'{s} reached as a bare factor; cannot differentiate')
        if s == var:
            return s, '1'
        return s, '0'
    op = f.sym
    if op in (Notation.GROUP, Notation.V_GROUP):
        if Notation.is_semantic_bracket(f):
            name = Notation.BRACKET_NAMES[f.props['br']]
            raise PrimitiveError(
                f'cannot differentiate {name} (not in the rule set)')
        e, d = _diff(f.args[0], notation, var)
        return _paren(e), d
    if op == Notation.PLUS:
        return _diff(f.args[0], notation, var)
    if op == Notation.MINUS:
        e, d = _diff(f.args[0], notation, var)
        return _d_neg(e), _d_neg(d)
    if op == Notation.S_LIST:
        exprs, derivs = [], []
        for t in f.args:
            sign = notation.vgetf(t, [Notation.PLUS, Notation.MINUS])
            if sign is not None and sign.sym == Notation.MINUS:
                e, d = _diff(sign.args[0], notation, var)
                e, d = _d_neg(e), _d_neg(d)
            elif sign is not None:
                e, d = _diff(sign.args[0], notation, var)
            else:
                e, d = _diff(t, notation, var)
            exprs.append(e)
            derivs.append(d)
        return _d_add(*exprs), _d_add(*derivs)
    if op == Notation.P_LIST:
        return _diff_product(f.args, notation, var)
    if op == Notation.SLASH or op.name in FRAC_NAMES:
        u_e, u_d = _diff(f.args[0], notation, var)
        v_e, v_d = _diff(f.args[1], notation, var)
        expr = _d_frac(u_e, v_e)
        if v_d == '0':
            return expr, _d_frac(u_d, v_e)
        num = _d_add(_d_mul(u_d, v_e), _d_neg(_d_mul(u_e, v_d)))
        return expr, _d_frac(num, f'{_paren(v_e)}^{{2}}')
    if op == Notation.INDEX:
        return _diff_power(f, notation, var)
    if op == Notation.FUNC:
        fname, arg = f.args[0], f.args[1]
        if isinstance(fname, Symbol) and fname.name in _DERIV_TABLE:
            u_e, u_d = _diff(arg, notation, var)
            expr = f'{fname.name}{_paren(u_e)}'
            return expr, _DERIV_TABLE[fname.name](u_e, u_d)
        raise PrimitiveError(f'no derivative rule for {fname!r}')
    if op.name == '\\sqrt' and len(f.args) == 1:
        u_e, u_d = _diff(f.args[0], notation, var)
        expr = f'\\sqrt{{{u_e}}}'
        return expr, _d_frac(u_d, f'2 \\sqrt{{{u_e}}}')
    raise PrimitiveError(f'no derivative rule for {op.name}')


def _diff_product(args, notation, var):
    # segment the p-list into differentiable units (function applications
    # bind the factors that follow them, up to the next function name)
    units = []
    args = [a for a in args if not (isinstance(a, Symbol)
                                    and a.name in Notation.styles)]

    def is_func_head(a):
        return _is_func_symbol(a, notation) or _diff_func_power(a, notation)

    i = 0
    while i < len(args):
        a = args[i]
        head = is_func_head(a)
        if head:
            fname, power = (a.name, None) if head is True else head
            inner, j = _func_arg_span(args, i, notation,
                                      lambda t: bool(is_func_head(t)))
            if not inner:
                raise PrimitiveError(f'{fname} without argument')
            units.append(('func', fname, power, inner))
            i = j
        else:
            units.append(('expr', a))
            i += 1
    exprs, derivs = [], []
    for u in units:
        if u[0] == 'expr':
            e, d = _diff(u[1], notation, var)
        else:
            _, fname, power, inner = u
            u_e, u_d = _prod_diff(inner, notation, var)
            base_e = f'{fname}{_paren(u_e)}'
            base_d = _DERIV_TABLE[fname](u_e, u_d)
            if power is None:
                e, d = base_e, base_d
            else:
                n = _index_power(power, notation)
                e = f'{fname}^{{{n}}}{_paren(u_e)}'
                # d/dx f^n(u) = n f^{n-1}(u) (f(u))'
                fac = base_e if n == 2 else f'{fname}^{{{n - 1}}}{_paren(u_e)}'
                d = _d_mul(str(n), fac, base_d)
        exprs.append(e)
        derivs.append(d)
    # product rule over units
    total = [_d_mul(*[derivs[j] if j == i else exprs[j]
                      for j in range(len(units))])
             for i in range(len(units))]
    return _d_mul(*exprs), _d_add(*total)


def _prod_diff(syms, notation, var):
    """Product rule over a bare list of factor symbols; returns
    (expr_latex, derivative_latex)."""
    exprs, derivs = [], []
    for s in syms:
        e, d = _diff(s, notation, var)
        exprs.append(e)
        derivs.append(d)
    total = [_d_mul(*[derivs[j] if j == i else exprs[j]
                      for j in range(len(syms))])
             for i in range(len(syms))]
    return _d_mul(*exprs), _d_add(*total)


def _is_func_symbol(sym, notation):
    return (isinstance(sym, Symbol) and notation.get(sym) is None
            and sym.name in _DERIV_TABLE)


def _diff_func_power(sym, notation):
    """\\sin^{n}-style factor usable by the differentiator, or None."""
    fp = _func_power(sym, notation)
    if fp is None:
        return None
    fname, power = fp
    if fname not in _DERIV_TABLE or power is None:
        return None
    try:
        _index_power(power, notation)
    except NotInFragment:
        return None
    return fname, power


def _diff_power(f, notation, var):
    sub, sup_l, power, sup_r = f.args[1]
    if sub is not None or sup_l is not None or sup_r is not None:
        raise PrimitiveError('cannot differentiate subscripted symbol')
    base = f.args[0]
    u_e, u_d = _diff(base, notation, var)
    if power is None:
        return u_e, u_d
    p_e, p_d = _diff(power, notation, var)
    try:
        n = _index_power(power, notation)
        p_e = str(n)
    except NotInFragment:
        n = None
    base_s = f'{{{_paren(u_e) if _is_sum_str(u_e) else u_e}}}'
    expr = f'{base_s}^{{{p_e}}}'
    if p_d == '0':
        # u^p with constant p
        if u_d == '0':
            return expr, '0'
        if n is not None:
            if n - 1 == 1:
                return expr, _d_mul(p_e, base_s, u_d)
            return expr, _d_mul(p_e, f'{base_s}^{{{n - 1}}}', u_d)
        new_exp = f'{p_e} - 1'
        return expr, _d_mul(p_e, f'{base_s}^{{{_paren(new_exp)}}}', u_d)
    if u_d == '0':
        # a^p(x): a^p ln(a) p'
        if u_e == 'e':
            return expr, _d_mul(f'e^{{{p_e}}}', p_d)
        return expr, _d_mul(expr, f'\\ln{_paren(u_e)}', p_d)
    # general u^p: u^p (p' ln u + p u'/u)
    inner = _d_add(_d_mul(p_d, f'\\ln{_paren(u_e)}'),
                   _d_frac(_d_mul(p_e, u_d), u_e))
    return expr, _d_mul(expr, _paren(inner))


def differentiate(expr, var):
    """d/d(var) with ~20 mechanical rules; canonicalizes rational results."""
    args = {'expr': expr, 'var': var}
    try:
        sym, notation = parse_latex(expr)
    except PrimitiveError as e:
        return _error('differentiate', args, str(e))
    # fast exact path for the rational fragment
    try:
        rf = to_ratfunc(sym, notation)
        dnum = rf.num.derivative(var) * rf.den - rf.num * rf.den.derivative(var)
        drf = RatFunc(dnum, rf.den * rf.den)
        out_n = Notation()
        result = write_latex(ratfunc_to_notation(drf, out_n), out_n)
        rec = _result('differentiate', args, expr, result,
                      extra={'method': 'polyrat'})
        rec['check'] = _derivative_check(expr, result, var)
        return rec
    except (NotInFragment, ZeroDivisionError):
        pass
    try:
        _, deriv = _diff(sym, notation, var)
    except PrimitiveError as e:
        return _error('differentiate', args, str(e))
    result = canonical_or_same(deriv)
    assumptions = []
    cleanup_check = None
    # The rule builder deliberately favors local, obviously-correct formulas;
    # quotient/product chains can therefore contain 0/16 debris and layers of
    # transparent grouping.  Feed that formula through the same checked
    # canonical core an agent would call next.  The final derivative checker
    # still independently tests the cleaned result against the source.
    from tactics import core as core_tactics
    cleanup = core_tactics.expand(result)
    if cleanup.get('ok') and cleanup.get('check', {}).get('status') \
            in ('agree', 'domain-differs'):
        result = cleanup['result']
        assumptions = cleanup.get('assumptions', [])
        cleanup_check = cleanup['check']
    try:
        parse_latex(result)
    except PrimitiveError as e:
        return _error('differentiate', args,
                      f'internal: unparseable derivative: {e}')
    extra = {'method': 'rules'}
    if cleanup_check is not None:
        extra['cleanup'] = 'expand'
    rec = _result('differentiate', args, expr, result,
                  assumptions=assumptions, extra=extra)
    derivative_check = _derivative_check(expr, result, var)
    if derivative_check.get('status') == 'agree' and cleanup_check \
            and cleanup_check.get('status') == 'domain-differs':
        # Canonical cleanup may expose a wider written domain after dropping
        # zero-multiplied log/root atoms.  Keep that visible instead of
        # laundering the derivative into an unconditional green step.
        rec['check'] = dict(cleanup_check)
        rec['check']['derivative_samples'] = derivative_check.get('samples')
    else:
        rec['check'] = derivative_check
    return rec


def _derivative_check(expr, deriv, var, samples=8, seed=20260705):
    """Central-difference spot check: f'(x) ~ (f(x+h)-f(x-h))/2h."""
    try:
        s1, n1 = parse_latex(expr)
        s2, n2 = parse_latex(deriv)
    except PrimitiveError as e:
        return {'status': 'skipped', 'reason': str(e)}
    variables = free_symbols(s1, n1) | free_symbols(s2, n2) | {var}
    rng = random.Random(seed)
    h = 1e-5
    agreed = 0
    tried = 0
    while agreed < samples and tried < samples * 10:
        tried += 1
        env = _sample_point(variables, rng)
        try:
            d_sym = numeric_eval(s2, n2, env)
            f_vals = []
            for step in (h, -h, h / 2, -h / 2):
                env_s = dict(env)
                env_s[var] = env[var] + step
                f_vals.append(numeric_eval(s1, n1, env_s))
        except (EvalError, ZeroDivisionError, ValueError, OverflowError):
            continue
        if isinstance(d_sym, list) or any(isinstance(v, list)
                                          for v in f_vals):
            continue   # matrix-valued: central differences not supported
        f_p, f_m, f_p2, f_m2 = f_vals
        d_h = (f_p - f_m) / (2 * h)
        d_h2 = (f_p2 - f_m2) / h
        # Richardson extrapolation kills the O(h^2) truncation term;
        # |d_h - d_h2| estimates the truncation error, which explodes near
        # a singularity of f - there the central difference itself is
        # wrong, and a non-converged estimate is oracle ignorance, never
        # a counterexample
        d_num = (4 * d_h2 - d_h) / 3
        err_est = abs(d_h - d_h2) / 3
        scale = max(1.0, abs(d_sym), abs(d_num))
        if abs(d_sym - d_num) / scale > 1e-4:
            if abs(d_sym - d_num) <= 16 * err_est:
                continue   # numeric estimate has not converged: skip point
            return {'status': 'disagree', 'point': env,
                    'symbolic': d_sym, 'numeric': d_num}
        agreed += 1
    if agreed == 0:
        return {'status': 'skipped', 'reason': 'no evaluable sample points'}
    return {'status': 'agree', 'samples': agreed,
            'method': 'central-difference'}

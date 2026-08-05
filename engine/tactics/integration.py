#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Indefinite-integration tactics."""
import math
from fractions import Fraction

from notation import Notation, Symbol
from value import IntegerValue, FracValue
from polyrat import (Poly, NotInFragment, to_ratfunc, poly_to_notation,
                     FUNCTION_NAMES as FUNC_NAMES)

from primitives import (
    FRAC_NAMES, PrimitiveError, parse_latex, write_latex, free_symbols,
    _result, _error, _peel_groups, _strip_integral, _paren, _is_sum_str,
    definite_integral_parts, numeric_definite_check,
    same_expression, _infinity_sign, _limit_latex, _int_literal,
    _plain_symbol_name, numeric_improper_check,
)
from tactics.core import (
    equal_exprs, substitute, _merge_checks,
)
from tactics.differentiation import (
    _d_mul, _d_add, _d_neg, differentiate, _derivative_check,
)

# ---------------------------------------------------------------------------
# integration tactics (no autonomous `integrate` on purpose: the agent picks
# the tactic, toymath mechanically completes it)
# ---------------------------------------------------------------------------

_ANTIDERIV_TABLE = {
    '\\sin': lambda x: f'-\\cos\\left({x}\\right)',
    '\\cos': lambda x: f'\\sin\\left({x}\\right)',
    '\\exp': lambda x: f'\\exp\\left({x}\\right)',
    '\\sinh': lambda x: f'\\cosh\\left({x}\\right)',
    '\\cosh': lambda x: f'\\sinh\\left({x}\\right)',
}



def _integrand(expr, var):
    """Parse expr, stripping an optional \\int ... d<var> wrapper.
    Returns (sym, notation, integrand_latex)."""
    sym, notation = parse_latex(expr)
    inner = _strip_integral(sym, notation, var)
    if inner is not None:
        sym = inner
    return sym, notation, write_latex(sym, notation)


def _fresh_constant(taken):
    for name in ('C', 'K', 'Q'):
        if name not in taken:
            return name
    return 'C'


def _power_integrate_ratfunc(rf, var, assumptions, allow_log):
    """Antiderivative latex (no constant) for num/den where den is a
    constant or a single monomial in var. Raises PrimitiveError."""
    den = rf.den
    if den.variables() - {var}:
        raise PrimitiveError(
            'denominator must be constant or a power of the variable')
    if den.is_const():
        m, dconst = 0, den.const_value()
    else:
        if len(den.terms) != 1:
            raise PrimitiveError(
                'denominator must be a single power of the variable')
        (mono, dconst), = den.terms.items()
        m = dict(mono).get(var, 0)
    pos = {}
    neg_parts = []
    log_parts = []
    for mono, coeff in rf.num.terms.items():
        d = dict(mono)
        k = d.pop(var, 0)
        rest = tuple(sorted(d.items()))
        e = k - m
        if e == -1:
            if not allow_log:
                raise PrimitiveError(
                    'a term integrates to a logarithm (exponent -1); '
                    'use integrate_table for it')
            log_parts.append((coeff / dconst, rest))
            continue
        c2 = coeff / dconst / (e + 1)
        if e + 1 > 0:
            nm = tuple(sorted(list(rest) + [(var, e + 1)]))
            pos[nm] = pos.get(nm, Fraction(0)) + c2
        else:
            neg_parts.append((c2, rest, -(e + 1)))
    parts = []
    p = Poly(pos)
    if not p.is_zero():
        out_n = Notation()
        parts.append(write_latex(poly_to_notation(p, out_n), out_n))
    for c2, rest, pw in sorted(neg_parts, key=lambda t: (t[2], t[1])):
        out_n = Notation()
        num_s = write_latex(poly_to_notation(Poly({rest: abs(c2)}), out_n),
                            out_n)
        den_s = var if pw == 1 else f'{var}^{{{pw}}}'
        frac = f'\\frac{{{num_s}}}{{{den_s}}}'
        parts.append('-' + frac if c2 < 0 else frac)
    for c2, rest in sorted(log_parts, key=lambda t: t[1]):
        out_n = Notation()
        coeff_s = write_latex(poly_to_notation(Poly({rest: abs(c2)}), out_n),
                              out_n)
        term = _d_mul(coeff_s, f'\\ln\\left({var}\\right)')
        parts.append('-' + term if c2 < 0 else term)
        guard = {'text': f'{var} > 0', 'nonzero': var}
        if guard not in assumptions:
            assumptions.append(guard)
    if not parts:
        return '0'
    return _d_add(*parts)


def _is_bare_var(sym, notation, var):
    while True:
        g = notation.vgetf(sym, [Notation.GROUP, Notation.V_GROUP])
        if g is not None:
            sym = g.args[0]
            continue
        break
    return isinstance(sym, Symbol) and notation.get(sym) is None \
        and sym.name == var


def _sqrt_frac_latex(q):
    """LaTeX for \\sqrt{q}, q a positive Fraction, taking perfect-square
    numerator/denominator components exactly (\\sqrt{9/5} -> 3/\\sqrt{5})."""
    rn, rd = math.isqrt(q.numerator), math.isqrt(q.denominator)
    num_s = str(rn) if rn * rn == q.numerator \
        else f'\\sqrt{{{q.numerator}}}'
    den_s = str(rd) if rd * rd == q.denominator \
        else f'\\sqrt{{{q.denominator}}}'
    if den_s == '1':
        return num_s
    return f'\\frac{{{num_s}}}{{{den_s}}}'


def _arctan_integrate_ratfunc(rf, var):
    """Antiderivative latex (no constant) for c/(a var^2 + b) with rational
    literals a, b > 0 — the completed-square table family
    \\int dx/(x^2 + 1) = \\arctan x closed under positive scaling:
    \\int c/(a x^2 + b) dx = (c/\\sqrt{ab}) \\arctan(x \\sqrt{a/b}).
    Returns None when the ratfunc is not this shape."""
    if not rf.num.is_const():
        return None
    den = rf.den
    if den.variables() - {var}:
        return None
    terms = dict(den.terms)
    a = terms.pop(((var, 2),), None)
    b = terms.pop((), None)
    if a is None or b is None or terms or a <= 0 or b <= 0:
        return None
    c = rf.num.const_value()
    if c == 0:
        return '0'
    arg_s = _sqrt_frac_latex(a / b)
    arg = var if arg_s == '1' else _d_mul(arg_s, var)
    inner = f'\\arctan\\left({arg}\\right)'
    coeff_s = _sqrt_frac_latex(c * c / (a * b))
    term = inner if coeff_s == '1' else _d_mul(coeff_s, inner)
    return _d_neg(term) if c < 0 else term


def _steer_completed_square(rf, var):
    """A quadratic denominator with a linear term is one completing-the-
    square rewrite away from the arctan family — say so instead of the
    generic monomial-denominator refusal."""
    den = rf.den
    if den.variables() != {var}:
        return
    degs = {dict(mono).get(var, 0) for mono in den.terms}
    if max(degs, default=0) == 2 and 1 in degs:
        raise PrimitiveError(
            'quadratic denominator with a linear term: complete the square '
            'with integrate_rewrite to 1/(a (x+d)^2 + e), substitute the '
            'shifted variable, then integrate_table closes 1/(a u^2 + b)')


def _table_integrate(sym, notation, var, assumptions):
    """Mechanical antiderivative (no constant): power rule + logarithm +
    arctan family + basic functions of the bare variable, closed under sums
    and constant factors — symbolic var-free factors included (fully
    constant integrands integrate to c·var).
    Raises PrimitiveError with an honest reason otherwise."""
    rf = None
    try:
        rf = to_ratfunc(sym, notation)
    except NotInFragment:
        pass
    except ZeroDivisionError:
        raise PrimitiveError('integrand contains division by zero')
    rational_error = None
    if rf is not None:
        try:
            return _power_integrate_ratfunc(rf, var, assumptions,
                                            allow_log=True)
        except PrimitiveError as e:
            alt = _arctan_integrate_ratfunc(rf, var)
            if alt is not None:
                return alt
            # Do NOT raise yet: the structural branches below can peel
            # var-free SYMBOLIC factors the rational legs refuse
            # (1/(ab(v^2+1)) is multivariate to polyrat but one constant
            # split away from the arctan rule — live: the
            # a^2 sin^2 + b^2 cos^2 arctangent cell). If they cannot
            # close it either, the rational refusal below is the honest
            # message, with the completed-square steering preserved.
            rational_error = e
    try:
        return _table_structural(sym, notation, var, assumptions)
    except PrimitiveError:
        if rational_error is not None:
            _steer_completed_square(rf, var)
            raise rational_error
        raise


def _table_structural(sym, notation, var, assumptions):
    """The spelling-structural half of the table: unwraps groups and
    signs, splits sums, peels var-free (possibly symbolic) constant
    factors from products and denominators, and applies the basic
    function/power rules. Sub-terms re-enter `_table_integrate`, so a
    peeled core gets the rational legs again."""
    if var not in free_symbols(sym, notation):
        # var-free integrand outside the rational fragment (\sqrt{5}/5,
        # 1+\sqrt{2}, ...): a constant integrates to c·var regardless of
        # its spelling
        c = write_latex(sym, notation)
        return _d_mul(_paren(c) if _is_sum_str(c) else c, var)
    if not isinstance(sym, Symbol):
        raise PrimitiveError(f'cannot integrate term {sym!r}')
    f = notation.get(sym)
    if f is None:
        raise PrimitiveError(f'no rule to integrate {sym.name}')
    op = f.sym
    if op in (Notation.GROUP, Notation.V_GROUP, Notation.S_GROUP,
              Notation.PLUS):
        return _table_integrate(f.args[0], notation, var, assumptions)
    if op == Notation.MINUS:
        return _d_neg(_table_integrate(f.args[0], notation, var,
                                       assumptions))
    if op == Notation.S_LIST:
        parts = []
        for t in f.args:
            sign = notation.vgetf(t, [Notation.PLUS, Notation.MINUS])
            if sign is not None and sign.sym == Notation.MINUS:
                parts.append(_d_neg(_table_integrate(
                    sign.args[0], notation, var, assumptions)))
            elif sign is not None:
                parts.append(_table_integrate(sign.args[0], notation, var,
                                              assumptions))
            else:
                parts.append(_table_integrate(t, notation, var, assumptions))
        return _d_add(*parts)
    if op == Notation.P_LIST:
        # peel var-free constant factors off the front
        args = [a for a in f.args if not (isinstance(a, Symbol)
                                          and a.name in Notation.styles)]
        consts = []
        i = 0
        while i < len(args):
            a = args[i]
            if (isinstance(a, Symbol) and notation.get(a) is None
                    and a.name in FUNC_NAMES):
                break
            if var in free_symbols(a, notation):
                break
            consts.append(write_latex(a, notation))
            i += 1
        core = args[i:]
        if not core:
            # fully constant integrand: c dx -> c x
            return _d_mul(_d_mul(*consts) if consts else '1', var)
        if len(core) == 1:
            inner = _table_integrate(core[0], notation, var, assumptions)
        else:
            inner = _table_core_plist(core, notation, var)
        if consts:
            return _d_mul(*(consts + [_paren(inner) if _is_sum_str(inner)
                                      else inner]))
        return inner
    if op == Notation.SLASH or op.name in FRAC_NAMES:
        num, den = f.args[0], f.args[1]
        if var in free_symbols(den, notation):
            if var in free_symbols(num, notation):
                raise PrimitiveError(
                    'no table rule for a variable denominator beyond 1/x; '
                    'use integrate_power_rule or substitution')
            # var-free numerator over a variable denominator: c/g = c*(1/g).
            # Split var-free factors out of a product denominator, then the
            # reciprocal of the variable core re-enters the rational path
            # (arctan family included). Termination: the rebuilt reciprocal
            # has a literal-1 numerator and an unsplittable denominator, so
            # every recursion strictly simplifies the fraction.
            num_q = _rational_literal(num, notation)
            num_s = _frac_latex(num_q) if num_q is not None and num_q > 0 \
                else write_latex(_peel_groups(num, notation), notation)
            dcore = _peel_groups(den, notation)
            df = notation.getf(dcore, Notation.P_LIST)
            dconsts, dvars = [], []
            if df is not None and not any(
                    isinstance(a, Symbol) and notation.get(a) is None
                    and a.name in FUNC_NAMES for a in df.args):
                for a in df.args:
                    if isinstance(a, Symbol) and a.name in Notation.styles:
                        continue
                    (dconsts if var not in free_symbols(a, notation)
                     else dvars).append(a)
            if dconsts and len(dvars) == 1:
                recip_sym, recip_n = parse_latex(
                    f'\\frac{{1}}{{{write_latex(dvars[0], notation)}}}')
                inner = _table_integrate(recip_sym, recip_n, var,
                                         assumptions)
                const_s = _d_mul(*[write_latex(a, notation)
                                   for a in dconsts])
                coeff = f'\\frac{{{num_s}}}{{{const_s}}}'
                return _d_mul(coeff,
                              _paren(inner) if _is_sum_str(inner) else inner)
            if num_q != 1:
                recip_sym, recip_n = parse_latex(
                    f'\\frac{{1}}{{{write_latex(den, notation)}}}')
                inner = _table_integrate(recip_sym, recip_n, var,
                                         assumptions)
                return _d_mul(
                    _paren(num_s) if _is_sum_str(num_s) else num_s,
                    _paren(inner) if _is_sum_str(inner) else inner)
            raise PrimitiveError(
                'no table rule for a variable denominator beyond 1/x; '
                'use integrate_power_rule or substitution')
        den_s = write_latex(den, notation)
        inner = _table_integrate(num, notation, var, assumptions)
        return f'\\frac{{{inner}}}{{{den_s}}}'
    if op == Notation.INDEX:
        sub, sup_l, power, sup_r = f.args[1]
        base = f.args[0]
        if (isinstance(base, Symbol) and base.name == 'e'
                and power is not None and sub is None and sup_l is None
                and sup_r is None and _is_bare_var(power, notation, var)):
            return f'e^{{{var}}}'
        raise PrimitiveError(
            'no table rule for this power; integrate_power_rule handles '
            'rational literal exponents, or use integrate_by_parts')
    if op == Notation.FUNC:
        fname, arg = f.args[0], f.args[1]
        if isinstance(fname, Symbol) and fname.name in _ANTIDERIV_TABLE \
                and _is_bare_var(arg, notation, var):
            return _ANTIDERIV_TABLE[fname.name](var)
        raise PrimitiveError(
            'table rules require a basic function of the bare variable')
    raise PrimitiveError(f'no rule to integrate operation {op.name}')


def _table_core_plist(core, notation, var):
    """Integrate a p-list core of the form [func, arg...] where the
    argument is the bare variable."""
    head = core[0]
    if (isinstance(head, Symbol) and notation.get(head) is None
            and head.name in _ANTIDERIV_TABLE):
        rest = core[1:]
        if len(rest) == 1 and _is_bare_var(rest[0], notation, var):
            return _ANTIDERIV_TABLE[head.name](var)
        raise PrimitiveError(
            f'{head.name}: table rules require the bare variable as '
            'argument; use integrate_by_parts or substitution')
    raise PrimitiveError(
        'integrand is a product; use integrate_by_parts and choose u, dv')


def _finish_integration(op, args, expr, integrand_latex, body, var,
                        assumptions):
    taken = set()
    try:
        s, n = parse_latex(integrand_latex)
        taken = free_symbols(s, n) | {var}
    except PrimitiveError:
        pass
    const = _fresh_constant(taken)
    result = f'{body} + {const}'
    try:
        parse_latex(result)
    except PrimitiveError as e:
        return _error(op, args, f'internal: unparseable result: {e}')
    rec = _result(op, args, expr, result, assumptions=assumptions,
                  extra={'constant': const, 'integrand': integrand_latex})
    rec['check'] = _derivative_check(result, integrand_latex, var)
    return rec


def _rational_literal(sym, notation):
    """Fraction value of a rational-literal subtree (IntegerValue,
    FracValue, \\frac/SLASH of literals, +/-/group wrappers); None when
    the subtree is not an exact rational literal."""
    sign = 1
    guard = 0
    while guard < 64:
        guard += 1
        if isinstance(sym, IntegerValue):
            return Fraction(sign * sym.val)
        if isinstance(sym, FracValue):
            if sym.denom == 0:
                return None
            return Fraction(sign * sym.num, sym.denom)
        if not isinstance(sym, Symbol):
            return None
        f = notation.get(sym)
        if f is None:
            return None
        if f.sym == Notation.MINUS:
            sign, sym = -sign, f.args[0]
            continue
        if f.sym == Notation.PLUS:
            sym = f.args[0]
            continue
        if f.sym in (Notation.GROUP, Notation.V_GROUP):
            if Notation.is_semantic_bracket(f):
                return None
            sym = f.args[0]
            continue
        if f.sym == Notation.SLASH or f.sym.name in FRAC_NAMES:
            a = _rational_literal(f.args[0], notation)
            b = _rational_literal(f.args[1], notation)
            if a is None or b is None or b == 0:
                return None
            return sign * a / b
        return None
    return None


_MONOMIAL_MSG = ('integrate_power_rule integrates terms of the form '
                 'c x^{r} with rational literal c and r; for roots of the '
                 'variable inside a fraction substitute u = x^{1/n} via '
                 'integrate_substitute, otherwise use integrate_table or '
                 'integrate_by_parts')


def _monomial_parts(sym, notation, var):
    """(coeff, exponent) Fractions for a rational-literal monomial
    ``c * var^r``; raises PrimitiveError for anything else."""
    lit = _rational_literal(sym, notation)
    if lit is not None:
        return lit, Fraction(0)
    if isinstance(sym, Symbol):
        f = notation.get(sym)
        if f is None:
            if sym.name == var:
                return Fraction(1), Fraction(1)
            raise PrimitiveError(_MONOMIAL_MSG)
        if f.sym == Notation.MINUS:
            c, r = _monomial_parts(f.args[0], notation, var)
            return -c, r
        if f.sym == Notation.PLUS:
            return _monomial_parts(f.args[0], notation, var)
        if f.sym in (Notation.GROUP, Notation.V_GROUP) \
                and not Notation.is_semantic_bracket(f):
            return _monomial_parts(f.args[0], notation, var)
        if f.sym == Notation.INDEX:
            sub, sup_l, power, sup_r = f.args[1]
            base = _peel_groups(f.args[0], notation)
            if (sub is None and sup_l is None and sup_r is None
                    and power is not None and isinstance(base, Symbol)
                    and notation.get(base) is None and base.name == var):
                r = _rational_literal(power, notation)
                if r is None:
                    raise PrimitiveError(_MONOMIAL_MSG)
                return Fraction(1), r
            raise PrimitiveError(_MONOMIAL_MSG)
        if f.sym == Notation.SLASH or f.sym.name in FRAC_NAMES:
            cn, rn = _monomial_parts(f.args[0], notation, var)
            cd, rd = _monomial_parts(f.args[1], notation, var)
            if cd == 0:
                raise PrimitiveError('division by zero')
            return cn / cd, rn - rd
        if f.sym == Notation.P_LIST:
            c, r = Fraction(1), Fraction(0)
            for a in f.args:
                if isinstance(a, Symbol) and notation.get(a) is None \
                        and a.name in Notation.styles:
                    continue
                ca, ra = _monomial_parts(a, notation, var)
                c, r = c * ca, r + ra
            return c, r
    raise PrimitiveError(_MONOMIAL_MSG)


def _frac_latex(q):
    """Non-negative Fraction as a LaTeX literal."""
    if q.denominator == 1:
        return str(q.numerator)
    return f'\\frac{{{q.numerator}}}{{{q.denominator}}}'


def _rational_power_integrate(sym, notation, var, assumptions):
    """Antiderivative latex (no constant) for a sum of rational-literal
    monomials with at least one non-integer exponent. Records var > 0
    (fractional powers live on the positive axis). Raises PrimitiveError
    when a term is not such a monomial or integrates to a logarithm."""
    inner = _peel_groups(sym, notation)
    f = notation.getf(inner, Notation.S_LIST)
    terms = list(f.args) if f is not None else [inner]
    monos = []
    fractional = False
    for t in terms:
        c, r = _monomial_parts(t, notation, var)
        if r == -1:
            raise PrimitiveError(
                'a term integrates to a logarithm (exponent -1); '
                'use integrate_table for it')
        if r.denominator != 1:
            fractional = True
        if c != 0:
            monos.append((c / (r + 1), r + 1))
    if not fractional:
        raise PrimitiveError(_MONOMIAL_MSG)
    parts = []
    for c2, e in monos:
        neg = c2 < 0
        mag = -c2 if neg else c2
        if e > 0:
            pw = var if e == 1 else f'{var}^{{{_frac_latex(e)}}}'
            term = pw if mag == 1 else f'{_frac_latex(mag)}{pw}'
        else:
            pe = -e
            pw = var if pe == 1 else f'{var}^{{{_frac_latex(pe)}}}'
            term = f'\\frac{{{_frac_latex(mag)}}}{{{pw}}}'
        parts.append(('-' if neg else '+', term))
    if not parts:
        return '0'
    out = ('-' if parts[0][0] == '-' else '') + parts[0][1]
    for s, t in parts[1:]:
        out += f' {s} {t}'
    assumptions.append({'text': f'{var} > 0', 'nonzero': var})
    return out


def integrate_power_rule(expr, var):
    """Term-by-term power rule for polynomials and rational expressions
    with a constant or single-power denominator, plus rational-literal
    exponents (``x^{1/2}``; records var > 0). Refuses the exponent -1
    case (that is integrate_table's logarithm rule)."""
    args = {'expr': expr, 'var': var}
    try:
        sym, notation, integrand_latex = _integrand(expr, var)
        rf = to_ratfunc(sym, notation)
    except PrimitiveError as e:
        return _error('integrate_power_rule', args, str(e))
    except ZeroDivisionError:
        return _error('integrate_power_rule', args, 'division by zero')
    except NotInFragment as e:
        if 'non-integer exponent' in str(e):
            # rational-literal exponents: direct power rule with var > 0
            assumptions = []
            try:
                body = _rational_power_integrate(sym, notation, var,
                                                 assumptions)
            except PrimitiveError as e2:
                return _error('integrate_power_rule', args,
                              f'outside the rational fragment: {e}; {e2}')
            return _finish_integration('integrate_power_rule', args, expr,
                                       integrand_latex, body, var,
                                       assumptions)
        return _error('integrate_power_rule', args,
                      f'outside the rational fragment: {e}; '
                      'use integrate_table or integrate_by_parts')
    assumptions = []
    try:
        body = _power_integrate_ratfunc(rf, var, assumptions,
                                        allow_log=False)
    except PrimitiveError as e:
        return _error('integrate_power_rule', args, str(e))
    return _finish_integration('integrate_power_rule', args, expr,
                               integrand_latex, body, var, assumptions)


def integrate_table(expr, var):
    """Basic-function antiderivatives (sin, cos, e^x, sinh, cosh, 1/x),
    closed under sums, constant factors, and the power rule."""
    args = {'expr': expr, 'var': var}
    try:
        sym, notation, integrand_latex = _integrand(expr, var)
        assumptions = []
        body = _table_integrate(sym, notation, var, assumptions)
    except PrimitiveError as e:
        return _error('integrate_table', args, str(e))
    return _finish_integration('integrate_table', args, expr,
                               integrand_latex, body, var, assumptions)


def integrate_by_parts(expr, var, u, dv):
    """One application of integration by parts with the agent's choice of
    u and dv: returns u v - \\int v du. Requires u * dv == integrand; the
    remaining integral is left for the next step."""
    args = {'expr': expr, 'var': var, 'u': u, 'dv': dv}
    try:
        sym, notation, integrand_latex = _integrand(expr, var)
        usym, unotation = parse_latex(u)
        dvsym, dvnotation = parse_latex(dv)
    except PrimitiveError as e:
        return _error('integrate_by_parts', args, str(e))
    if var not in free_symbols(usym, unotation):
        return _error('integrate_by_parts', args,
                      f'u must depend on {var!r}')
    eq = equal_exprs(f'{_paren(u)} {_paren(dv)}', integrand_latex)
    if not (eq.get('ok') and eq.get('verdict') == 'yes'):
        return _error('integrate_by_parts', args,
                      f'u * dv must equal the integrand '
                      f'(verdict: {eq.get("verdict", "error")})')
    du_rec = differentiate(u, var)
    if not du_rec.get('ok'):
        return _error('integrate_by_parts', args,
                      f'cannot differentiate u: {du_rec.get("error")}')
    assumptions = []
    try:
        v = _table_integrate(dvsym, dvnotation, var, assumptions)
    except PrimitiveError as e:
        return _error('integrate_by_parts', args,
                      f'cannot integrate dv mechanically: {e}; '
                      'choose a simpler dv')
    du = du_rec['result']
    uv = _d_mul(_paren(u) if _is_sum_str(u) else u,
                _paren(v) if _is_sum_str(v) or v.startswith('-') else v)
    inner = _d_mul(_paren(v) if _is_sum_str(v) or v.startswith('-') else v,
                   _paren(du) if _is_sum_str(du) or du.startswith('-')
                   else du)
    result = f'{uv} - \\int {inner} \\, d {var}'
    try:
        parse_latex(result)
    except PrimitiveError as e:
        return _error('integrate_by_parts', args,
                      f'internal: unparseable result: {e}')
    rec = _result('integrate_by_parts', args, expr, result,
                  assumptions=assumptions,
                  extra={'u': u, 'du': du, 'v': v, 'dv': dv,
                         'remaining_integral': f'\\int {inner} \\, d {var}'})
    # the whole result contains an unevaluated integral, so verify the
    # pieces: v' == dv (central differences) and du (already self-checked);
    # the by-parts identity itself is mechanical
    v_check = _derivative_check(v, write_latex(dvsym, dvnotation), var)
    rec['check'] = _merge_checks(v_check,
                                 du_rec.get('check', {'status': 'skipped'}))
    if rec['check'].get('status') == 'agree':
        rec['check']['method'] = 'per-piece (v and du verified)'
    return rec


def integrate_substitute(expr, var, u_expr, u_var, new_integrand):
    """u-substitution with the agent's choice of u = u_expr and the
    integrand rewritten in u_var. Verifies mechanically that
    new_integrand[u_var := u_expr] * du/dx equals the original integrand,
    then returns \\int new_integrand d u_var. Substitute u_expr back after
    integrating the transformed integral."""
    args = {'expr': expr, 'var': var, 'u_expr': u_expr, 'u_var': u_var,
            'new_integrand': new_integrand}
    try:
        sym, notation, integrand_latex = _integrand(expr, var)
        usym, unotation = parse_latex(u_expr)
        nsym, nnotation = parse_latex(new_integrand)
        vsym, vnotation = parse_latex(u_var)
    except PrimitiveError as e:
        return _error('integrate_substitute', args, str(e))
    if not (isinstance(vsym, Symbol) and vnotation.get(vsym) is None):
        return _error('integrate_substitute', args,
                      f'u_var must be a plain symbol, got {u_var!r}')
    if u_var == var:
        return _error('integrate_substitute', args,
                      'u_var must differ from the integration variable')
    if u_var in free_symbols(sym, notation):
        return _error('integrate_substitute', args,
                      f'{u_var!r} already occurs in the integrand; '
                      'pick a fresh variable')
    if var not in free_symbols(usym, unotation):
        return _error('integrate_substitute', args,
                      f'u must depend on {var!r}')
    if var in free_symbols(nsym, nnotation):
        return _error('integrate_substitute', args,
                      f'the new integrand must not mention {var!r}; '
                      f'write it in terms of {u_var!r} only')
    if u_var not in free_symbols(nsym, nnotation):
        return _error('integrate_substitute', args,
                      f'the new integrand must be written in terms of '
                      f'{u_var!r}')
    du_rec = differentiate(u_expr, var)
    if not du_rec.get('ok'):
        return _error('integrate_substitute', args,
                      f'cannot differentiate u: {du_rec.get("error")}')
    back_rec = substitute(new_integrand, u_var, u_expr)
    if not back_rec.get('ok'):
        return _error('integrate_substitute', args,
                      f'cannot check the rewrite: {back_rec.get("error")}')
    du = du_rec['result']
    reconstructed = _d_mul(_paren(back_rec['result']), _paren(du))
    eq = equal_exprs(reconstructed, integrand_latex)
    if not (eq.get('ok') and eq.get('verdict') == 'yes'):
        return _error(
            'integrate_substitute', args,
            f'new_integrand[{u_var} := {u_expr}] * du/d{var} does not '
            f'equal the integrand (verdict: {eq.get("verdict", "error")})')
    # a leading minus would parse as `\int` minus the rest, not as a
    # negative integrand - parenthesize sums and negatives alike
    body = new_integrand.strip()
    if _is_sum_str(body) or body.startswith('-'):
        body = _paren(body)
    result = f'\\int {body} \\, d {u_var}'
    try:
        parse_latex(result)
    except PrimitiveError as e:
        return _error('integrate_substitute', args,
                      f'internal: unparseable result: {e}')
    rec = _result('integrate_substitute', args, expr, result,
                  extra={'u': f'{u_var} = {u_expr}', 'du': du,
                         'back_substitute': {'var': u_var, 'value': u_expr}})
    # the substitution identity was verified by equal?; surface it as the
    # step check
    check = {'status': 'agree',
             'method': f'substitution identity via equal? ({eq["method"]})'}
    if 'samples' in eq:
        check['samples'] = eq['samples']
    rec['check'] = _merge_checks(check,
                                 du_rec.get('check', {'status': 'skipped'}))
    if rec['check'].get('status') == 'agree':
        rec['check']['method'] = check['method']
    return rec


def integrate_rewrite(expr, var, new_integrand):
    """Congruence under the integral sign: replace the integrand with an
    agent-proposed expression that equal? confirms is the same function
    (e.g. a partial-fraction decomposition). The equality is the checked
    content; the integral wrapper is carried along unchanged."""
    args = {'expr': expr, 'var': var, 'new_integrand': new_integrand}
    try:
        sym, notation, integrand_latex = _integrand(expr, var)
        parse_latex(new_integrand)
    except PrimitiveError as e:
        return _error('integrate_rewrite', args, str(e))
    eq = equal_exprs(new_integrand, integrand_latex)
    if not (eq.get('ok') and eq.get('verdict') == 'yes'):
        return _error('integrate_rewrite', args,
                      f'new integrand is not mechanically equal to the '
                      f'current one (verdict: '
                      f'{eq.get("verdict", "error")})')
    # a leading minus would parse as `\int` minus the rest, not as a
    # negative integrand - parenthesize sums and negatives alike
    body = new_integrand.strip()
    if _is_sum_str(body) or body.startswith('-'):
        body = _paren(body)
    result = f'\\int {body} \\, d {var}'
    try:
        parse_latex(result)
    except PrimitiveError as e:
        return _error('integrate_rewrite', args,
                      f'internal: unparseable result: {e}')
    rec = _result('integrate_rewrite', args, expr, result,
                  extra={'integrand': integrand_latex})
    check = {'status': 'agree',
             'method': f'integrand equality via equal? ({eq["method"]})'}
    if 'samples' in eq:
        check['samples'] = eq['samples']
    rec['check'] = check
    return rec


def integrate_linearity(expr, var):
    """Sum rule: split the integral of a top-level sum into a signed sum
    of integrals, one per term. Purely structural, hence exact; constant
    factors stay inside their sub-integrals."""
    args = {'expr': expr, 'var': var}
    try:
        sym, notation, integrand_latex = _integrand(expr, var)
    except PrimitiveError as e:
        return _error('integrate_linearity', args, str(e))
    while True:
        g = notation.vgetf(sym, [Notation.GROUP, Notation.V_GROUP,
                                 Notation.S_GROUP])
        # bracket operators (|...|, floor, ceiling) are not grouping
        if g is None or Notation.is_semantic_bracket(g):
            break
        sym = g.args[0]
    f = notation.getf(sym, Notation.S_LIST)
    if f is None:
        return _error('integrate_linearity', args,
                      'integrand is not a top-level sum; nothing to '
                      'split')
    parts = []
    for t in f.args:
        sign = notation.vgetf(t, [Notation.PLUS, Notation.MINUS])
        if sign is not None:
            inner = sign.args[0]
            neg = sign.sym == Notation.MINUS
        else:
            inner = t
            neg = False
        body = write_latex(inner, notation)
        if _is_sum_str(body) or body.lstrip().startswith('-'):
            body = _paren(body)
        parts.append(('-' if neg else '+', f'\\int {body} \\, d {var}'))
    pieces = [p for _, p in parts]
    terms = [{'sign': -1 if sign == '-' else 1, 'integral': piece}
             for sign, piece in parts]
    result = ('-' if parts[0][0] == '-' else '') + parts[0][1]
    for sgn, piece in parts[1:]:
        result += f' {sgn} {piece}'
    try:
        parse_latex(result)
    except PrimitiveError as e:
        return _error('integrate_linearity', args,
                      f'internal: unparseable result: {e}')
    return _result('integrate_linearity', args, expr, result,
                   check={'status': 'exact',
                          'method': 'linearity of the integral'},
                   extra={'integrals': pieces, 'terms': terms})


def integrate_assemble(expr, var, antiderivatives):
    """Assemble the signed results of an ``integrate_linearity`` split.

    ``antiderivatives`` is ordered exactly like the split's ``terms``.  The
    do! tool supplies these values from ledger step ids, rather than letting
    the agent retype them.  This primitive independently differentiates each
    value against its corresponding integrand, constructs the signed sum,
    and adds one fresh constant.
    """
    args = {'expr': expr, 'var': var,
            'antiderivatives': list(antiderivatives)
            if isinstance(antiderivatives, (list, tuple))
            else antiderivatives}
    if not isinstance(antiderivatives, (list, tuple)):
        return _error('integrate_assemble', args,
                      'antiderivatives must be an ordered list')
    split = integrate_linearity(expr, var)
    if not split.get('ok'):
        return _error('integrate_assemble', args,
                      'cannot assemble without a linearity split: '
                      + split.get('error', 'unknown error'))
    terms = split['terms']
    if len(antiderivatives) != len(terms):
        return _error(
            'integrate_assemble', args,
            f'expected {len(terms)} antiderivatives in linearity order, '
            f'got {len(antiderivatives)}')

    checked = []
    rendered = []
    for i, (term, candidate) in enumerate(zip(terms, antiderivatives), 1):
        if not isinstance(candidate, str) or not candidate.strip():
            return _error('integrate_assemble', args,
                          f'piece {i} has no antiderivative')
        try:
            parse_latex(candidate)
            _, _, integrand = _integrand(term['integral'], var)
        except PrimitiveError as e:
            return _error('integrate_assemble', args,
                          f'piece {i} is malformed: {e}')
        derivative = differentiate(candidate, var)
        if not derivative.get('ok'):
            return _error('integrate_assemble', args,
                          f'cannot differentiate piece {i}: '
                          + derivative.get('error', 'unknown error'))
        equality = equal_exprs(derivative['result'], integrand)
        if not (equality.get('ok') and equality.get('verdict') == 'yes'):
            return _error(
                'integrate_assemble', args,
                f'ledger result for piece {i} is not an antiderivative of '
                f'{integrand!r} (verdict: '
                f'{equality.get("verdict", "error")})')
        checked.append({'piece': i,
                        'derivative': derivative['result'],
                        'method': equality.get('method', 'equal?')})
        rendered.append(('-' if term['sign'] < 0 else '+',
                         _paren(candidate)))

    first_sign, first = rendered[0]
    assembled = ('-' if first_sign == '-' else '') + first
    for sign, candidate in rendered[1:]:
        assembled += f' {sign} {candidate}'
    try:
        asym, anotation = parse_latex(assembled)
        const = _fresh_constant(free_symbols(asym, anotation) | {var})
        result = f'{assembled} + {const}'
        parse_latex(result)
        _, _, original_integrand = _integrand(expr, var)
    except PrimitiveError as e:
        return _error('integrate_assemble', args,
                      f'internal: unparseable assembly: {e}')

    # Defense in depth: the per-piece equalities check the provenance
    # mapping; this final symbolic check checks our signed construction.
    derivative = differentiate(result, var)
    if not derivative.get('ok'):
        return _error('integrate_assemble', args,
                      'cannot differentiate the assembled result: '
                      + derivative.get('error', 'unknown error'))
    equality = equal_exprs(derivative['result'], original_integrand)
    if not (equality.get('ok') and equality.get('verdict') == 'yes'):
        return _error('integrate_assemble', args,
                      'internal: assembled derivative does not equal the '
                      f'original integrand (verdict: '
                      f'{equality.get("verdict", "error")})')

    rec = _result('integrate_assemble', args, expr, result,
                  extra={'constant': const, 'pieces': checked,
                         'linearity_terms': terms})
    rec['check'] = _derivative_check(result, original_integrand, var)
    if rec['check'].get('status') == 'agree':
        rec['check']['method'] = (
            'per-piece derivatives + signed linearity')
    return rec


def integrate_definite(expr, var, antiderivative, upper_limit=None,
                       lower_limit=None):
    """Evaluate a definite integral from a RECORDED antiderivative
    (Fundamental Theorem of Calculus, part 2).

    ``expr`` is the definite integral ``\\int_a^b f \\, d<var>`` itself;
    ``antiderivative`` is the result of an earlier recorded integration
    step (the do! tool supplies it from a ledger step id rather than
    letting the agent retype it).  This primitive independently
    re-differentiates the antiderivative against the integrand, then
    builds ``F(b) - F(a)`` by substitution — any fresh ``+ C`` riding on
    the recorded antiderivative cancels in the following expand.

    When the antiderivative's SPELLING is singular at a bound (the
    classic ``\\arctan(c \\tan x)`` at ``\\pi/2``), substitution is the
    wrong move: the honest endpoint value is the one-sided limit of the
    antiderivative from inside the interval.  ``upper_limit`` /
    ``lower_limit`` carry that RECORDED limit value (the do! tool
    supplies each from a limit step id whose input is checked to be
    exactly ``\\lim_{var \\to bound^∓} <antiderivative>``); the endpoint
    then uses the recorded value and an assumption states the continuous
    extension.  The check leg re-integrates the integrand by quadrature
    and never touches F, so a lying endpoint value is refused
    regardless of its provenance.  Continuity of the integrand on [a,b]
    is recorded as an assumption, not proved — though the quadrature leg
    refuses outright when it lands on an interior domain break, which is
    exactly the case where F(b) - F(a) is the classic wrong answer."""
    args = {'expr': expr, 'var': var, 'antiderivative': antiderivative}
    if upper_limit is not None:
        args['upper_limit'] = upper_limit
    if lower_limit is not None:
        args['lower_limit'] = lower_limit
    parts = definite_integral_parts(expr, var)
    if parts is None:
        return _error(
            'integrate_definite', args,
            'expr must be a definite integral \\int_a^b f \\, d<var> '
            'with both bounds present (for indefinite integrals use the '
            'other integration tactics)')
    _var, integrand, lower, upper = parts
    if not isinstance(antiderivative, str) or not antiderivative.strip():
        return _error('integrate_definite', args,
                      'antiderivative must be a recorded result')
    try:
        parse_latex(antiderivative)
    except PrimitiveError as e:
        return _error('integrate_definite', args,
                      f'antiderivative is malformed: {e}')
    derivative = differentiate(antiderivative, var)
    if not derivative.get('ok'):
        return _error('integrate_definite', args,
                      'cannot differentiate the antiderivative: '
                      + derivative.get('error', 'unknown error'))
    equality = equal_exprs(derivative['result'], integrand)
    if not (equality.get('ok') and equality.get('verdict') == 'yes'):
        return _error(
            'integrate_definite', args,
            f'the recorded value is not an antiderivative of '
            f'{integrand!r} (verdict: '
            f'{equality.get("verdict", "error")})')
    assumptions = [{
        'text': f'{integrand} is continuous on '
                f'[{lower}, {upper}]',
        'display': f'${integrand}$ is continuous on '
                   f'$[{lower}, {upper}]$'}]
    endpoint_values = []
    for tag, bound, endpoint_limit in (('upper', upper, upper_limit),
                                       ('lower', lower, lower_limit)):
        if endpoint_limit is not None:
            try:
                ls, ln = parse_latex(endpoint_limit)
            except PrimitiveError as e:
                return _error('integrate_definite', args,
                              f'{tag} endpoint limit is malformed: {e}')
            if var in free_symbols(ls, ln):
                return _error('integrate_definite', args,
                              f'{tag} endpoint limit still contains '
                              f'{var}')
            endpoint_values.append(endpoint_limit)
            side = 'below' if tag == 'upper' else 'above'
            assumptions.append({
                'text': (f'the antiderivative extends continuously to '
                         f'{var} = {bound} (one-sided limit from '
                         f'{side} recorded)'),
                'display': (f'the antiderivative extends continuously '
                            f'to ${var} = {bound}$ (one-sided limit '
                            f'from {side} recorded)')})
            continue
        at_bound = substitute(antiderivative, var, bound)
        if not at_bound.get('ok'):
            return _error('integrate_definite', args,
                          f'cannot substitute the {tag} bound: '
                          + at_bound.get('error', 'unknown error'))
        endpoint_values.append(at_bound['result'])
    result = (f'{_paren(endpoint_values[0])} - '
              f'{_paren(endpoint_values[1])}')
    try:
        parse_latex(result)
    except PrimitiveError as e:
        return _error('integrate_definite', args,
                      f'internal: unparseable evaluation: {e}')
    rec = _result(
        'integrate_definite', args, expr, result,
        assumptions=assumptions,
        extra={'integrand': integrand, 'lower': lower, 'upper': upper})
    rec['check'] = numeric_definite_check(expr, var, result)
    return rec


def _improper_parts(expr, var, truncated):
    """Shared reader for the improper endpoint door: which bound of
    ``expr`` the ``truncated`` integral replaces, with what variable,
    approached from which side.  Returns a dict, or an error string.
    One reader on purpose — the tactic, the session adapter, and the
    replay validator must never disagree about this boundary."""
    parts = definite_integral_parts(expr, var)
    if parts is None:
        return ('expr must be a definite integral \\int_a^b f \\, d<var> '
                'with both bounds present')
    _var, integrand, lower, upper = parts
    for bound in (lower, upper):
        try:
            bsym, bn = parse_latex(bound)
        except PrimitiveError as e:
            return f'bound {bound!r} is malformed: {e}'
        if _infinity_sign(bsym, bn) is not None:
            return ('infinite bounds have no truncation door here; only '
                    'a singular finite endpoint does')
    tparts = definite_integral_parts(truncated, var)
    if tparts is None:
        return ('truncated must be a definite integral over the same '
                'variable')
    _tvar, t_integrand, t_lower, t_upper = tparts
    if not same_expression(integrand, t_integrand):
        return ('the truncated integrand differs from the integral '
                'being closed')
    lower_kept = same_expression(lower, t_lower)
    upper_kept = same_expression(upper, t_upper)
    if lower_kept and upper_kept:
        return ('the truncated integral must replace the singular bound '
                'with a fresh variable')
    if not (lower_kept or upper_kept):
        return ('the truncated integral must keep one bound and replace '
                'only the other')
    if lower_kept:
        side, bound, replaced, direction = 'upper', upper, t_upper, 'left'
    else:
        side, bound, replaced, direction = 'lower', lower, t_lower, 'right'
    try:
        tsym, tn = parse_latex(replaced)
    except PrimitiveError as e:
        return f'truncation bound {replaced!r} is malformed: {e}'
    bound_var = _plain_symbol_name(tsym, tn)
    if bound_var is None or _int_literal(tsym, tn) is not None:
        return ('the truncation bound must be a bare fresh variable, '
                f'not {replaced!r}')
    taken = set(free_symbols(*parse_latex(integrand)))
    for latex in (lower, upper):
        taken |= free_symbols(*parse_latex(latex))
    if bound_var == var or bound_var in taken:
        return (f'the truncation variable {bound_var!r} must be fresh: '
                'it appears in the integral itself')
    return {'integrand': integrand, 'lower': lower, 'upper': upper,
            'side': side, 'bound': bound, 'bound_var': bound_var,
            'direction': direction}


def integrate_improper(expr, var, truncated, truncated_value,
                       limit_value):
    """Close an improper endpoint integral by its definitional reading
    (the infinite-object door: value = limit of the truncated integrals
    from inside the interval).

    ``expr`` is the definite integral whose integrand is singular at one
    finite bound; ``truncated`` is the same integral with that bound
    replaced by a fresh variable, ``truncated_value`` its RECORDED
    evaluation (an earlier `integrate_definite` step — the do! tool and
    the CLI supply both from a ledger step id), and ``limit_value`` the
    RECORDED one-sided limit of that evaluation at the replaced bound.
    The step computes nothing new: it certifies that the two cited
    pieces compose into exactly the definitional limit, records the
    reading and the half-open continuity assumption, and is checked by
    an independent leg that re-integrates the integrand itself over a
    graded truncation ladder and extrapolates — never touching either
    cited piece.  A ladder that does not settle certifies nothing: a
    divergent integral has no recordable limit step to cite, and its
    check finds no finite value in evidence."""
    args = {'expr': expr, 'var': var, 'truncated': truncated,
            'truncated_value': truncated_value,
            'limit_value': limit_value}
    info = _improper_parts(expr, var, truncated)
    if isinstance(info, str):
        return _error('integrate_improper', args, info)
    for tag, value in (('truncated_value', truncated_value),
                       ('limit_value', limit_value)):
        if not isinstance(value, str) or not value.strip():
            return _error('integrate_improper', args,
                          f'{tag} must be a recorded result')
        try:
            vsym, vn = parse_latex(value)
        except PrimitiveError as e:
            return _error('integrate_improper', args,
                          f'{tag} is malformed: {e}')
        if var in free_symbols(vsym, vn):
            return _error('integrate_improper', args,
                          f'{tag} still contains {var}')
    lsym, lnot = parse_latex(limit_value)
    if info['bound_var'] in free_symbols(lsym, lnot):
        return _error('integrate_improper', args,
                      'limit_value still contains the truncation '
                      f'variable {info["bound_var"]}')
    definitional = _limit_latex(info['bound_var'], info['bound'],
                                info['direction'], truncated)
    interval = (f'[{info["lower"]}, {info["upper"]})'
                if info['side'] == 'upper'
                else f'({info["lower"]}, {info["upper"]}]')
    assumptions = [
        {'text': (f'the integral is improper at the {info["side"]} '
                  f'bound {var} = {info["bound"]}; its value is read '
                  f'as {definitional} (the definitional limit of the '
                  'truncated integrals)'),
         'display': (f'improper at the {info["side"]} bound '
                     f'${var} = {info["bound"]}$: read as '
                     f'${definitional}$ (definitional limit)')},
        {'text': (f'{info["integrand"]} is continuous on {interval}'),
         'display': (f'${info["integrand"]}$ is continuous on '
                     f'${interval}$')},
    ]
    rec = _result(
        'integrate_improper', args, expr, limit_value,
        assumptions=assumptions,
        extra={'integrand': info['integrand'], 'lower': info['lower'],
               'upper': info['upper'], 'singular': info['side'],
               'bound_var': info['bound_var']})
    rec['check'] = numeric_improper_check(expr, var, info['side'],
                                          limit_value)
    return rec

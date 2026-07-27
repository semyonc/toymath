#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Core algebra, checking, rewrite, and factoring tactics."""
import math
import random
import re
from fractions import Fraction
from itertools import combinations

from notation import Notation, Symbol, Func
from value import Value, IntegerValue, FracValue, FloatValue
from replicator import Replicator
import comparer
from comparer import NotationParam
from polyrat import (Poly, NotInFragment, to_ratfunc,
                     ratfunc_to_notation, poly_to_notation,
                     _fraction_to_notation, _index_power,
                     FUNCTION_NAMES as FUNC_NAMES)

from primitives import (
    FRAC_NAMES, _UNARY_TABLE, PrimitiveError, EvalError,
    parse_latex, write_latex,
    _write_std, _GroupStripper, _big_operator_name, _bound_symbols,
    _contains_free_infinity, _subscript_var, free_symbols,
    _num_agree, numeric_eval, _func_power, _func_arg_span, _sample_point,
    numeric_spot_check, numeric_relation_check, Substitutor, _result, _error,
    _paren, _is_sum_str, same_expression,
)

# ---------------------------------------------------------------------------
# fractional powers: Puiseux-style folding (t = x^{1/q})
# ---------------------------------------------------------------------------

# polyrat monomials have integer exponents, so x^{1/6} would atomize whole
# and (x^{1/6})^6 = x stays invisible. Folding substitutes t = x^{1/q}
# (q = lcm of the exponent denominators, per variable), canonicalizes with
# ordinary integer-power machinery, and maps t^k back to x^{k/q}. Sound
# only where the roots are defined, so every fold records `x > 0`; the
# (x^2)^{1/2} = |x| direction never folds (the base must be the plain
# variable itself).

_PUISEUX_MAX_ROOT = 24
# lexer-safe stand-ins (a raw internal name would break the validated
# writer round trip); collision-checked against the expression's symbols
_PUISEUX_POOL = ('\\vartheta', '\\varsigma', '\\varrho', '\\varpi')


def _rational_const(sym, notation):
    """Fraction value of a constant subtree, or None."""
    try:
        rf = to_ratfunc(sym, notation)
    except (NotInFragment, ZeroDivisionError):
        return None
    if not rf.is_const():
        return None
    return rf.const_value()


def _frac_power_var(sym, f, notation):
    """(var_name, Fraction) when sym is INDEX(plain variable, rational
    non-integer constant power) with empty sub/sup slots; None otherwise."""
    base = f.args[0]
    while True:
        g = notation.vgetf(base, [Notation.GROUP, Notation.V_GROUP])
        if g is None or Notation.is_semantic_bracket(g):
            break
        base = g.args[0]
    sub_l, sup_l, power, sub_r = f.args[1]
    if sub_l is not None or sup_l is not None or sub_r is not None \
            or power is None:
        return None
    if not (isinstance(base, Symbol) and notation.get(base) is None
            and '_{' not in base.name and base.name not in FUNC_NAMES
            and base.name not in Notation.p_oper):
        return None
    val = _rational_const(power, notation)
    if val is None or val.denominator == 1:
        return None
    return base.name, val


def _scan_frac_powers(root, notation):
    """{var_name: set of exponent denominators} over the whole graph."""
    dens = {}
    seen = set()
    stack = [root]
    while stack:
        s = stack.pop()
        if not isinstance(s, Symbol) or s in seen:
            continue
        seen.add(s)
        f = notation.get(s)
        if f is None:
            continue
        if f.sym == Notation.INDEX:
            hit = _frac_power_var(s, f, notation)
            if hit is not None:
                dens.setdefault(hit[0], set()).add(hit[1].denominator)
        for a in f.args:
            if isinstance(a, tuple):
                stack.extend(x for x in a if isinstance(x, Symbol)
                             or isinstance(x, Value))
            elif a is not None:
                stack.append(a)
    return dens


class _PuiseuxIn(Replicator):
    """x^{p/q'} -> t^k, bare x -> (t^q); subscripted names untouched."""

    def __init__(self, notation, output_notation, roots):
        super(_PuiseuxIn, self).__init__(notation, output_notation)
        self.roots = roots        # var_name -> (t_name, q)
        self.failed = False       # an exponent that is not a positive int

    def _tpow(self, tname, k, wrap):
        t = Symbol(tname)
        if k == 1:
            return t
        ix = self.output_notation.setf(
            Notation.INDEX, (t, (None, None, IntegerValue(k), None)))
        if not wrap:
            return ix
        return self.output_notation.setf(Notation.GROUP, (ix,), br='()')

    def _sub(self, sym):
        if isinstance(sym, Symbol) and self.notation.get(sym) is None \
                and sym.name in self.roots:
            tname, q = self.roots[sym.name]
            return self._tpow(tname, q, wrap=True)
        return None

    def enter_symbol(self, sym):
        res = self._sub(sym)
        return res if res is not None else sym

    def enter_raw_term(self, t):
        if isinstance(t, Symbol):
            res = self._sub(t)
            if res is not None:
                return res
        return t

    def enter_index(self, sym, f):
        if _subscript_var(sym, self.notation) is not None:
            # same verbatim discipline as Substitutor: x_{1} is atomic
            plain = Replicator(self.notation, self.output_notation)
            sub_l, sup_l, power, sub = f.args[1]
            dims = (None, None,
                    None if power is None else self.enter_scalar(power),
                    plain.enter_scalar(sub))
            return self.output_notation.repf(
                self.mapsym(sym),
                Func(f.sym, (plain.enter_scalar(f.args[0]), dims)))
        hit = _frac_power_var(sym, f, self.notation)
        if hit is not None and hit[0] in self.roots:
            tname, q = self.roots[hit[0]]
            k = hit[1] * q
            if k.denominator != 1 or k <= 0:
                self.failed = True
            else:
                return self._tpow(tname, int(k), wrap=False)
        return super(_PuiseuxIn, self).enter_index(sym, f)


class _PuiseuxOut(Replicator):
    """t^k -> x^{k/q} (reduced; bare x when k = q), bare t -> x^{1/q}."""

    def __init__(self, notation, output_notation, back):
        super(_PuiseuxOut, self).__init__(notation, output_notation)
        self.back = back          # t_name -> (var_name, q)

    def _xpow(self, var, frac):
        x = Symbol(var)
        if frac.denominator == 1:
            if frac.numerator == 1:
                return x
            power = IntegerValue(frac.numerator)
        else:
            power = self.output_notation.setf(
                Symbol('\\frac'), (IntegerValue(frac.numerator),
                                   IntegerValue(frac.denominator)))
        return self.output_notation.setf(
            Notation.INDEX, (x, (None, None, power, None)))

    def _sub(self, sym):
        if isinstance(sym, Symbol) and self.notation.get(sym) is None \
                and sym.name in self.back:
            var, q = self.back[sym.name]
            return self._xpow(var, Fraction(1, q))
        return None

    def enter_symbol(self, sym):
        res = self._sub(sym)
        return res if res is not None else sym

    def enter_raw_term(self, t):
        if isinstance(t, Symbol):
            res = self._sub(t)
            if res is not None:
                return res
        return t

    def enter_index(self, sym, f):
        base = f.args[0]
        while True:
            g = self.notation.vgetf(base, [Notation.GROUP, Notation.V_GROUP])
            if g is None or Notation.is_semantic_bracket(g):
                break
            base = g.args[0]
        sub_l, sup_l, power, sub_r = f.args[1]
        if isinstance(base, Symbol) and self.notation.get(base) is None \
                and base.name in self.back and sub_l is None \
                and sup_l is None and sub_r is None and power is not None:
            k = _rational_const(power, self.notation)
            if k is not None and k.denominator == 1 and k > 0:
                var, q = self.back[base.name]
                return self._xpow(var, Fraction(int(k), q))
        return super(_PuiseuxOut, self).enter_index(sym, f)


def _puiseux_fold(sym, notation):
    """(t_sym, t_notation, back_fn, assumptions) when the expression has
    rational-exponent powers of plain variables and the fold is exactly
    representable; None otherwise (callers keep today's behavior)."""
    try:
        dens = _scan_frac_powers(sym, notation)
    except Exception:
        return None
    if not dens:
        return None
    taken = free_symbols(sym, notation)
    pool = [n for n in _PUISEUX_POOL if n not in taken]
    roots = {}
    for var in sorted(dens):
        q = 1
        for d in dens[var]:
            q = q * d // math.gcd(q, d)
        if q > _PUISEUX_MAX_ROOT or not pool:
            return None
        roots[var] = (pool.pop(0), q)
    out_n = Notation()
    rep = _PuiseuxIn(notation, out_n, roots)
    new_sym = rep(sym)
    if rep.failed:
        return None
    back = {t: (v, q) for v, (t, q) in roots.items()}

    def back_fn(latex):
        s2, n2 = parse_latex(latex)
        o2 = Notation()
        r2 = _PuiseuxOut(n2, o2, back)(s2)
        return write_latex(r2, o2)

    assumptions = [{'text': f'{v} > 0', 'nonzero': v} for v in sorted(roots)]
    return new_sym, out_n, back_fn, assumptions



def _checked(rec, assumptions=None):
    # when input and result are the same relation, spot-check per side
    # (a relation itself is not numerically evaluable)
    try:
        s1, n1 = parse_latex(rec['input'])
        s2, n2 = parse_latex(rec['result'])
        sp1 = _comp_split(s1, n1)
        sp2 = _comp_split(s2, n2)
    except PrimitiveError:
        sp1 = sp2 = None
    if sp1 is not None and sp2 is not None and sp1[2] == sp2[2]:
        rec['check'] = _merge_checks(
            numeric_spot_check(write_latex(sp1[0], n1),
                               write_latex(sp2[0], n2),
                               assumptions=assumptions),
            numeric_spot_check(write_latex(sp1[1], n1),
                               write_latex(sp2[1], n2),
                               assumptions=assumptions))
        return rec
    rec['check'] = numeric_spot_check(rec['input'], rec['result'],
                                      assumptions=assumptions)
    return rec


# ---------------------------------------------------------------------------
# primitive: substitute
# ---------------------------------------------------------------------------

def _is_zero_term(term, notation):
    """True when an additive term is literally zero after unwrapping
    +/- and transparent group layers (exact zeros only; `|0|` and
    anything needing arithmetic stay untouched)."""
    guard = 0
    while guard < 64:
        guard += 1
        if isinstance(term, IntegerValue):
            return term.val == 0
        if isinstance(term, FracValue):
            return term.num == 0 and term.denom != 0
        if not isinstance(term, Symbol):
            return False
        f = notation.get(term)
        if f is None:
            return False
        if f.sym in (Notation.PLUS, Notation.MINUS):
            term = f.args[0]
            continue
        if f.sym in (Notation.GROUP, Notation.V_GROUP):
            if Notation.is_semantic_bracket(f):
                return False
            term = f.args[0]
            continue
        return False
    return False


class _ZeroTermDropper(Replicator):
    """Drop additive terms that are literally zero. Substitution pinning
    (C := 0) must not leave `+(0)` residue: live agents hand-clean it,
    and retyping a result breaks provenance linkage. Exact zeros only —
    every other simplification stays in expand/evaluate."""

    def enter_slist(self, sym, f):
        kept = [t for t in f.args if not _is_zero_term(t, self.notation)]
        if len(kept) == len(f.args):
            return super(_ZeroTermDropper, self).enter_slist(sym, f)
        if not kept:
            return self.enter_raw_term(IntegerValue(0))
        head = kept[0]
        hf = self.notation.getf(head, Notation.PLUS)
        if hf is not None:
            head = hf.args[0]  # a surviving tail term promoted to front
        if len(kept) == 1:
            return self.enter_additive_expr(head)
        args = tuple([self.enter_additive_expr(head)]
                     + [self.enter_additive_expr(t) for t in kept[1:]])
        return self.output_notation.repf(
            self.mapsym(sym), Func(Notation.S_LIST, args))


def _relation_system_items(sym, notation):
    r"""Return ``(kind, relation_symbols)`` for a relation collection.

    A comma system is a C_LIST containing only relations.  A cases system is
    a one-column ``\cases`` array whose every row is a relation; ordinary
    two-column piecewise cases deliberately remain scalar notation.
    """
    comma = notation.getf(sym, Notation.C_LIST)
    if comma is not None and len(comma.args) >= 2 \
            and all(notation.getf(item, Notation.COMP) is not None
                    for item in comma.args):
        return 'comma', list(comma.args)
    f = notation.get(sym) if isinstance(sym, Symbol) else None
    if f is not None and f.sym.name == '\\cases' and f.args \
            and all(isinstance(row, (list, tuple)) and len(row) == 1
                    and notation.getf(row[0], Notation.COMP) is not None
                    for row in f.args):
        return 'cases', [row[0] for row in f.args]
    return None


def _render_relation_system(kind, relations):
    if kind == 'comma':
        return ','.join(relations)
    return '\\cases{' + ' \\cr '.join(relations) + '}'


def substitute(expr, var, value):
    """Replace every free occurrence of `var` in `expr` by `value`.
    Additive terms that become literally zero are dropped."""
    args = {'expr': expr, 'var': var, 'value': value}
    try:
        sym, notation = parse_latex(expr)
        vsym, vnotation = parse_latex(value)
    except PrimitiveError as e:
        return _error('substitute', args, str(e))
    var_symbol = Symbol(var)
    bound = _bound_symbols(sym, notation)
    if var in bound:
        return _error('substitute', args,
                      f'cannot substitute bound variable {var!r}')
    captured = free_symbols(vsym, vnotation) & bound
    if captured:
        return _error(
            'substitute', args,
            'substitution would capture bound variable(s): '
            + ', '.join(sorted(captured)))
    if var not in free_symbols(sym, notation):
        return _error('substitute', args,
                      f'variable {var!r} does not occur in expression')
    out_n = Notation()
    try:
        out_s = Substitutor(
            notation, out_n, {var_symbol: (vsym, vnotation)})(sym)
    except PrimitiveError as e:
        return _error('substitute', args, str(e))
    clean_n = Notation()
    out_s = _ZeroTermDropper(out_n, clean_n)(out_s)
    result = write_latex(out_s, clean_n)
    rec = _result('substitute', args, expr, result)
    # check: substituting into the input must equal evaluating the output
    # at the same points; do it by fixing var := value inside the oracle
    rec['check'] = _substitute_check(expr, result, var, value)
    return rec


def _substitute_check(expr, result, var, value, samples=8, seed=20260705):
    try:
        s1, n1 = parse_latex(expr)
        s2, n2 = parse_latex(result)
        vs, vn = parse_latex(value)
        sp1 = _comp_split(s1, n1)
        sp2 = _comp_split(s2, n2)
    except PrimitiveError as e:
        return {'status': 'skipped', 'reason': str(e)}
    system1 = _relation_system_items(s1, n1)
    system2 = _relation_system_items(s2, n2)
    if system1 is not None or system2 is not None:
        if system1 is None or system2 is None \
                or len(system1[1]) != len(system2[1]):
            return {'status': 'disagree',
                    'reason': 'relation-system shape changed'}
        return _merge_check_list([
            _substitute_check(write_latex(before, n1),
                              write_latex(after, n2), var, value,
                              samples=samples, seed=seed)
            for before, after in zip(system1[1], system2[1])
        ])
    if sp1 is not None and sp2 is not None:
        return _merge_checks(
            _substitute_check(write_latex(sp1[0], n1),
                              write_latex(sp2[0], n2), var, value),
            _substitute_check(write_latex(sp1[1], n1),
                              write_latex(sp2[1], n2), var, value))
    variables = (free_symbols(s1, n1) | free_symbols(s2, n2)
                 | free_symbols(vs, vn)) - {var}
    rng = random.Random(seed)
    agreed = 0
    tried = 0
    while agreed < samples and tried < samples * 8:
        tried += 1
        env = _sample_point(variables, rng)
        try:
            env[var] = numeric_eval(vs, vn, env)
            v1 = numeric_eval(s1, n1, env)
            del env[var]
            v2 = numeric_eval(s2, n2, env)
        except (EvalError, ZeroDivisionError, ValueError, OverflowError):
            continue
        agree = _num_agree(v1, v2, 1e-6)
        if agree is None:
            continue
        if not agree:
            return {'status': 'disagree', 'point': env, 'lhs': v1, 'rhs': v2}
        agreed += 1
    if agreed == 0:
        return {'status': 'skipped', 'reason': 'no evaluable sample points'}
    return {'status': 'agree', 'samples': agreed}


# ---------------------------------------------------------------------------
# primitive: apply_both_sides
# ---------------------------------------------------------------------------

_APPLY_OPS = ('+', '-', '*', '/', '^')

# relation handling: '=' and '\ne' are sign-blind; strict/weak inequalities
# flip under multiplication/division by a negative constant
_FLIP_REL = {'<': '>', '>': '<', '\\lt': '\\gt', '\\gt': '\\lt',
             '\\le': '\\ge', '\\ge': '\\le', '\\leq': '\\geq',
             '\\geq': '\\leq'}
_SUPPORTED_REL = {'=', '\\ne', '\\neq'} | set(_FLIP_REL)


def _needs_parens_additive(sym, notation):
    if notation.vgetf(sym, [Notation.S_LIST, Notation.MINUS, Notation.PLUS]):
        return True
    if isinstance(sym, (IntegerValue, FracValue, FloatValue)):
        return False
    return False


def _needs_parens_factor(sym, notation):
    return notation.vgetf(sym, [Notation.S_LIST, Notation.MINUS,
                                Notation.PLUS]) is not None


# ---------------------------------------------------------------------------
# stated case hypotheses ("assuming")
# ---------------------------------------------------------------------------

# Only strict hypotheses are accepted. `x \ge 0` together with `x \le 0`
# would name a region with no interior, where the oracle can never sample
# and where two canonically distinct expressions may still agree — the
# assumed region must be somewhere the checks can actually live.
_CONSTRAINT_REL = {'<', '\\lt', '>', '\\gt', '\\ne', '\\neq'}
_STRICT_CONSTRAINT_REL = {'<', '\\lt', '>', '\\gt'}


def _assumption_records(assuming):
    """Turn an agent-stated hypothesis into assumption records the oracle
    samples under. Several hypotheses may be comma-separated. Returns
    (records, error); the hypothesis is recorded, never established."""
    if assuming is None:
        return [], None
    if not isinstance(assuming, str):
        return None, 'assuming must be a LaTeX relation'
    if not assuming.strip():
        return [], None
    try:
        sym, notation = parse_latex(assuming)
    except PrimitiveError as e:
        return None, f'cannot parse the assumption: {e}'
    system = _relation_system_items(sym, notation)
    if system is not None:
        items = system[1]
    elif notation.getf(sym, Notation.COMP) is not None:
        items = [sym]
    else:
        return None, ('an assumption must be a relation such as "x > 0" '
                      '(comma-separate several)')
    records = []
    for item in items:
        rel = notation.getf(item, Notation.COMP).sym.props.get('op')
        if rel not in _CONSTRAINT_REL:
            return None, (
                f'assumption relation {rel!r} is not supported: state a '
                'strict hypothesis (<, >) or \\ne, so the assumed region '
                'is one the checks can sample')
        text = write_latex(item, notation)
        record = {'text': text, 'constraint': text}
        if record not in records:
            records.append(record)
    return records, None


def _is_zero_expression(latex):
    """True when an expression canonicalizes to exactly 0, in the rational
    fragment or over opaque atoms. Symbolic only — a sign gate must never
    rest on sampling."""
    try:
        sym, notation = parse_latex(latex)
    except PrimitiveError:
        return False
    try:
        rf = to_ratfunc(sym, notation)
        return rf.is_const() and rf.const_value() == 0
    except (NotInFragment, ZeroDivisionError):
        pass
    try:
        rf = _atomized_ratfunc(sym, notation, _AtomStore())
        return rf.is_const() and rf.const_value() == 0
    except (NotInFragment, ZeroDivisionError, PrimitiveError):
        return False


def _sign_from_assumptions(records, arg_latex):
    """(sign, error): +1/-1 when a recorded strict hypothesis pins the sign
    of `arg`, None when none of them does. The hypothesis has to be about
    the factor itself — its sides must differ by exactly `arg`, decided
    canonically, so `x - 3 > 0` and `x > 3` both pin `x - 3`."""
    sign = None
    for record in records:
        constraint = record.get('constraint')
        if not constraint:
            continue
        try:
            csym, cnotation = parse_latex(constraint)
        except PrimitiveError:
            continue
        comp = cnotation.getf(csym, Notation.COMP)
        if comp is None:
            continue
        rel = comp.sym.props.get('op')
        if rel not in _STRICT_CONSTRAINT_REL:
            continue
        lhs = write_latex(comp.args[0], cnotation)
        rhs = write_latex(comp.args[1], cnotation)
        positive = rel in ('>', '\\gt')
        if _is_zero_expression(f'({lhs}) - ({rhs}) - ({arg_latex})'):
            found = 1 if positive else -1
        elif _is_zero_expression(f'({rhs}) - ({lhs}) - ({arg_latex})'):
            found = -1 if positive else 1
        else:
            continue
        if sign is not None and sign != found:
            return None, ('the stated hypotheses disagree about the sign '
                          f'of {arg_latex}')
        sign = found
    return sign, None


def _apply_one_relation(comp, notation, op, asym, anotation, arg_const,
                        assumed=(), arg_sign=None):
    rel = comp.sym.props.get('op')
    if rel not in _SUPPORTED_REL:
        return None, f'unsupported relation {rel!r}'
    is_ineq = rel in _FLIP_REL
    out_rel = rel
    lhs, rhs = comp.args[0], comp.args[1]
    lhs_s = write_latex(lhs, notation)
    rhs_s = write_latex(rhs, notation)
    arg_s = write_latex(asym, anotation)

    assumptions = []
    # the direction of an inequality moved by a factor of stated sign is
    # the claim of the step; the relation-equivalence leg checks it
    direction_claimed = False

    def additive(side):
        a = arg_s
        if _needs_parens_additive(asym, anotation):
            a = _paren(a)
        return f'{side} {op} {a}'

    def multiplicative(side_sym, side_str):
        s = side_str
        if _needs_parens_factor(side_sym, notation):
            s = _paren(s)
        return f'{s} \\cdot {_paren(arg_s)}'

    if op in ('+', '-'):
        new_lhs, new_rhs = additive(lhs_s), additive(rhs_s)
    elif op == '*':
        if arg_const == 0:
            return None, 'multiplying both sides by 0 destroys the relation'
        if is_ineq:
            sign, error = _factor_sign('multiply', arg_const, arg_sign, arg_s)
            if error is not None:
                return None, error
            direction_claimed = arg_const is None
            if sign < 0:
                out_rel = _FLIP_REL[rel]
        elif arg_const is None:
            if _is_matrix_valued(asym, anotation):
                # a nonzero matrix can still be singular; the honest
                # reversibility record is invertibility, not != 0
                assumptions.append({
                    'text': f'{arg_s} \\text{{ is invertible}}',
                    'display': f'${arg_s}$ is invertible',
                    'nonzero': arg_s})
            elif arg_sign is None:
                # if the factor can vanish, the step may introduce solutions
                # (a stated strict sign hypothesis already excludes that)
                assumptions.append({'text': f'{arg_s} \\ne 0',
                                    'nonzero': arg_s})
        new_lhs = multiplicative(lhs, lhs_s)
        new_rhs = multiplicative(rhs, rhs_s)
    elif op == '/':
        if _is_matrix_valued(asym, anotation):
            return None, ('dividing both sides by a matrix-valued '
                          'expression is not supported')
        if arg_const == 0:
            return None, 'division by zero'
        if is_ineq:
            sign, error = _factor_sign('divide', arg_const, arg_sign, arg_s)
            if error is not None:
                return None, error
            direction_claimed = arg_const is None
            if sign < 0:
                out_rel = _FLIP_REL[rel]
        elif arg_const is None and arg_sign is None:
            assumptions.append({'text': f'{arg_s} \\ne 0', 'nonzero': arg_s})
        new_lhs = f'\\frac{{{lhs_s}}}{{{arg_s}}}'
        new_rhs = f'\\frac{{{rhs_s}}}{{{arg_s}}}'
    else:  # '^'
        if rel != '=':
            return None, "op '^' is only supported for '=' relations"
        if _is_matrix_valued(asym, anotation):
            return None, 'a matrix-valued exponent is not supported'
        if arg_const is not None and arg_const < 0:
            assumptions.append({'text': f'{lhs_s} \\ne 0', 'nonzero': lhs_s})
            assumptions.append({'text': f'{rhs_s} \\ne 0', 'nonzero': rhs_s})
        if arg_const is not None and arg_const != int(arg_const) or arg_const is None:
            assumptions.append(
                {'text': f'both sides must be in the domain of x^{{{arg_s}}}',
                 'display': ('both sides must be in the domain of '
                             f'$x^{{{arg_s}}}$')})
        new_lhs = f'{_paren(lhs_s)}^{{{arg_s}}}'
        new_rhs = f'{_paren(rhs_s)}^{{{arg_s}}}'

    result = f'{new_lhs} {out_rel} {new_rhs}'
    guards = list(assumed) + assumptions
    # oracle: each new side must equal op(old side, arg) at sample points
    c1 = numeric_spot_check(new_lhs, _op_expr(lhs_s, op, arg_s),
                            assumptions=guards)
    c2 = numeric_spot_check(new_rhs, _op_expr(rhs_s, op, arg_s),
                            assumptions=guards)
    check = _merge_checks(c1, c2)
    if direction_claimed:
        # per-side values cannot see a wrong flip: check the whole relation
        check = _merge_checks(
            check,
            numeric_relation_check(f'{lhs_s} {rel} {rhs_s}', result,
                                   assumptions=guards))
    return {'result': result, 'assumptions': assumptions,
            'check': check}, None


def _factor_sign(verb, arg_const, arg_sign, arg_s):
    """(sign, error) for moving an inequality by a factor. A literal's sign
    is certain; otherwise only a stated hypothesis can supply one."""
    if arg_const is not None:
        literal = 1 if arg_const > 0 else -1
        if arg_sign is not None and arg_sign != literal:
            return None, (f'the stated hypothesis contradicts the sign of '
                          f'{arg_s}')
        return literal, None
    if arg_sign is not None:
        return arg_sign, None
    return None, (
        f'cannot {verb} an inequality by an expression of unknown sign; '
        f'state the case hypothesis as `assuming` (e.g. "{arg_s} > 0") and '
        'record the opposite case as its own step, or use a constant')


def apply_both_sides(equation, op, arg, assuming=None):
    """Apply op ∈ {+,-,*,/,^} with argument `arg` to both sides of one
    relation or every relation in a comma/one-column-cases system. Division
    records the assumption arg ≠ 0. `assuming` states a case hypothesis
    ("x > 0"): it is recorded, the checks sample only where it holds, and a
    strict hypothesis about the factor decides whether an inequality keeps
    or flips its direction."""
    args = {'equation': equation, 'op': op, 'arg': arg}
    if assuming:
        args['assuming'] = assuming
    if op not in _APPLY_OPS:
        return _error('apply_both_sides', args,
                      f'op must be one of {_APPLY_OPS}')
    assumed, error = _assumption_records(assuming)
    if error is not None:
        return _error('apply_both_sides', args, error)
    try:
        sym, notation = parse_latex(equation)
        asym, anotation = parse_latex(arg)
    except PrimitiveError as e:
        return _error('apply_both_sides', args, str(e))

    comp = notation.getf(sym, Notation.COMP)
    system = _relation_system_items(sym, notation)
    if comp is None and system is None:
        return _error('apply_both_sides', args,
                      'expression is not an equation, inequality, or '
                      'relation system')

    arg_const = None
    try:
        rf = to_ratfunc(asym, anotation)
        if rf.is_const():
            arg_const = rf.const_value()
    except (NotInFragment, ZeroDivisionError):
        pass

    arg_sign, error = _sign_from_assumptions(assumed, arg)
    if error is not None:
        return _error('apply_both_sides', args, error)

    relation_symbols = [sym] if comp is not None else system[1]
    transformed = []
    assumptions = list(assumed)
    checks = []
    for index, relation_sym in enumerate(relation_symbols):
        relation = notation.getf(relation_sym, Notation.COMP)
        outcome, error = _apply_one_relation(
            relation, notation, op, asym, anotation, arg_const,
            assumed=assumed, arg_sign=arg_sign)
        if error is not None:
            if system is not None:
                error = f'relation {index + 1}: {error}'
            return _error('apply_both_sides', args, error)
        transformed.append(outcome['result'])
        for assumption in outcome['assumptions']:
            if assumption not in assumptions:
                assumptions.append(assumption)
        checks.append(outcome['check'])

    result = (transformed[0] if system is None
              else _render_relation_system(system[0], transformed))
    try:
        parse_latex(result)
    except PrimitiveError as e:
        return _error('apply_both_sides', args,
                      f'internal: built unparseable result: {e}')
    rec = _result('apply_both_sides', args, equation, result,
                  assumptions=assumptions)
    rec['check'] = _merge_check_list(checks)
    return rec


def _op_expr(side, op, arg):
    if op == '+':
        return f'{_paren(side)} + {_paren(arg)}'
    if op == '-':
        return f'{_paren(side)} - {_paren(arg)}'
    if op == '*':
        return f'{_paren(side)} \\cdot {_paren(arg)}'
    if op == '/':
        return f'\\frac{{{side}}}{{{arg}}}'
    return f'{_paren(side)}^{{{arg}}}'


def _merge_checks(c1, c2):
    for status in ('disagree', 'domain-differs'):
        if c1['status'] == status:
            return c1
        if c2['status'] == status:
            return c2
    if c1['status'] == 'agree' and c2['status'] == 'agree':
        merged = {'status': 'agree'}
        samples = [c['samples'] for c in (c1, c2) if 'samples' in c]
        if samples:
            merged['samples'] = min(samples)
        undefined = sum(c.get('undefined_points', 0) for c in (c1, c2))
        if undefined:
            merged['undefined_points'] = undefined
        # how much of the sample actually exercised a relation direction
        holding = [c['holding_points'] for c in (c1, c2)
                   if 'holding_points' in c]
        if holding:
            merged['holding_points'] = min(holding)
        return merged
    return {'status': 'skipped',
            'reason': c1.get('reason') or c2.get('reason') or 'partial'}


def _merge_check_list(checks):
    if not checks:
        return {'status': 'skipped', 'reason': 'no checks'}
    merged = checks[0]
    for check in checks[1:]:
        merged = _merge_checks(merged, check)
    return merged


# ---------------------------------------------------------------------------
# primitives: expand / collect / evaluate  (polyrat-powered)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# opaque atoms: canonicalize expressions outside the rational fragment by
# treating maximal non-fragment subtrees (\cos x, e^x, unevaluated \int ...)
# as opaque variables, running the SAME polyrat engine, and substituting the
# subtrees back. No new rewrite rules - just the trusted canonical core.
# ---------------------------------------------------------------------------

def _canonicalize_atom_payload(sym, notation, output_notation):
    """Copy an opaque atom, canonicalizing its rational subexpressions.

    The atom itself remains opaque to the outer polynomial calculation, but
    exact rational expressions below that boundary should not retain their
    input spelling.  For example, the argument of
    ``\ln(4 + (x^2)^2)`` becomes ``x^4 + 4``.  Bracket nodes are retained:
    they carry either function-argument binding or semantics such as ``|x|``.

    This deliberately uses only the ordinary integer-power rational
    fragment.  It does not invoke opaque-atom canonicalization recursively or
    the assumption-bearing Puiseux fold.
    """
    if isinstance(sym, Symbol):
        f = notation.get(sym)
        if f is not None and f.sym in (Notation.GROUP, Notation.V_GROUP,
                                       Notation.S_GROUP):
            args = list(f.args)
            args[0] = _canonicalize_atom_payload(
                args[0], notation, output_notation)
            if len(args) > 1 and isinstance(args[1], (Symbol, Value)):
                args[1] = _canonicalize_atom_payload(
                    args[1], notation, output_notation)
            return output_notation.setf(f.sym, tuple(args), **f.props)
        if f is not None and f.sym in (Notation.PLUS, Notation.MINUS):
            # A signed term inside a mixed opaque payload must keep its
            # S_LIST separator.  Canonicalizing ``+x^2`` as a standalone
            # rational expression drops the PLUS wrapper and silently
            # changes ``opaque + x^2`` into ``opaque * x^2`` when copied
            # back into the parent list.
            inner = _canonicalize_atom_payload(
                f.args[0], notation, output_notation)
            return output_notation.setf(f.sym, (inner,), **f.props)

    try:
        rf = to_ratfunc(sym, notation)
    except NotInFragment:
        rf = None
    if rf is not None:
        return ratfunc_to_notation(rf, output_notation)

    if not isinstance(sym, Symbol):
        return sym
    f = notation.get(sym)
    if f is None:
        return sym

    def copy_arg(arg):
        if isinstance(arg, (Symbol, Value)):
            return _canonicalize_atom_payload(arg, notation,
                                               output_notation)
        if isinstance(arg, tuple):
            return tuple(copy_arg(a) for a in arg)
        if isinstance(arg, list):
            return [copy_arg(a) for a in arg]
        return arg

    return output_notation.setf(
        f.sym, tuple(copy_arg(a) for a in f.args), **f.props)


class _AtomStore(object):
    def __init__(self):
        self.by_key = {}   # normal-form key -> atom name
        self.exprs = {}    # atom name -> (sym, notation)

    def atom(self, sym, notation):
        canonical_n = Notation()
        canonical_s = _canonicalize_atom_payload(
            sym, notation, canonical_n)
        latex = _write_std(canonical_s, canonical_n)
        try:
            # atom identity ignores all transparent grouping, so \sin(x)
            # and \sin x share one atom
            s2, n2 = parse_latex(latex)
            out = Notation()
            key = _write_std(
                _GroupStripper(n2, out, all_brackets=True)(s2), out)
        except PrimitiveError:
            key = latex
        name = self.by_key.get(key)
        if name is None:
            # 'zz#' prefix: sorts after single-letter variables, so
            # canonical monomials print as x (\sin x), not (\sin x) x
            name = f'zz#a{len(self.exprs)}'
            self.by_key[key] = name
            self.exprs[name] = (canonical_s, canonical_n)
        return Symbol(name)

    def mapping(self):
        return {Symbol(name): se for name, se in self.exprs.items()}


_NON_EXPR_OPS = (Notation.COMP, Notation.C_LIST, Notation.O_LIST,
                 Notation.A_LIST, Notation.PAIR, Notation.COLLECTION)

# matrix/vector objects: array literals and \vec-marked symbols. \cases is
# excluded on purpose (piecewise scalar). Bare symbols read as scalars until
# a declaration mechanism exists.
_MATRIX_FUNCS = frozenset((
    '\\array', '\\pmatrix', '\\matrix', '\\bmatrix', '\\Bmatrix',
    '\\vmatrix', '\\Vmatrix', '\\smallmatrix', '\\vec',
))


def _is_matrix_valued(sym, notation):
    """True when the subtree contains a matrix/vector object anywhere."""
    if not isinstance(sym, Symbol):
        return False
    f = notation.get(sym)
    if f is None:
        return False
    if f.sym.name in _MATRIX_FUNCS:
        return True

    def scan(items):
        for it in items:
            if isinstance(it, (list, tuple)):
                if scan(it):
                    return True
            elif isinstance(it, Symbol) and _is_matrix_valued(it, notation):
                return True
        return False

    return scan(f.args)


def _is_matrix_object(sym, notation):
    """A single matrix/vector object under transparent grouping only.
    Powers of ONE object commute with each other, so A^n may stay in the
    polynomial layer while (AB)^n and (A+B)^n may not."""
    while isinstance(sym, Symbol):
        f = notation.get(sym)
        if f is None:
            return False
        if f.sym in (Notation.GROUP, Notation.V_GROUP, Notation.S_GROUP):
            sym = f.args[0]
            continue
        return f.sym.name in _MATRIX_FUNCS
    return False


def _atomize_walk(sym, notation, out_n, store):
    """Copy sym into out_n replacing maximal non-fragment subtrees with
    opaque atom symbols. `notation` must be a private clone (span nodes are
    added to it). Raises NotInFragment for non-expressions."""
    try:
        to_ratfunc(sym, notation)
        return Replicator(notation, out_n)(sym)
    except NotInFragment:
        pass
    if not isinstance(sym, Symbol):
        raise NotInFragment(f'cannot atomize {sym!r}')
    f = notation.get(sym)
    if f is None:
        raise NotInFragment(f'bare operator {sym.name}')
    op = f.sym
    if op in _NON_EXPR_OPS:
        raise NotInFragment(f'{op.name} is not an expression')
    if op in (Notation.GROUP, Notation.V_GROUP):
        if Notation.is_semantic_bracket(f):
            # |expr|, floor and ceiling are bracket OPERATORS, not grouping:
            # the whole bracketed term is one opaque atom (identity keeps the
            # brackets, so |x| != x and floor(x) != x).
            return store.atom(sym, notation)
        inner = _atomize_walk(f.args[0], notation, out_n, store)
        return out_n.setf(op, (inner,), **f.props)
    if op in (Notation.PLUS, Notation.MINUS):
        inner = _atomize_walk(f.args[0], notation, out_n, store)
        return out_n.setf(op, (inner,))
    if op == Notation.S_LIST:
        terms = [_atomize_walk(t, notation, out_n, store) for t in f.args]
        return out_n.setf(op, terms)
    if op == Notation.SLASH or op.name in FRAC_NAMES:
        if _is_matrix_valued(f.args[1], notation):
            # dividing BY a matrix-valued expression is not scalar algebra
            # (A/A is not 1); the whole quotient stays opaque
            return store.atom(sym, notation)
        a = _atomize_walk(f.args[0], notation, out_n, store)
        b = _atomize_walk(f.args[1], notation, out_n, store)
        return out_n.setf(f.sym, (a, b), **f.props)
    if op == Notation.INDEX:
        sub, sup_l, power, sup_r = f.args[1]
        if _subscript_var(sym, notation) is not None and power is not None:
            try:
                n = _index_power(power, notation)
            except NotInFragment:
                n = None
            if n is not None and n >= 2:
                # C_{1}^{n} is a power of the SAME atomic variable C_{1},
                # not an unrelated atom.  Put the base in the atom store and
                # the integer power in polyrat, mirroring function powers.
                base = notation.setf(
                    Notation.INDEX,
                    (f.args[0], (None, None, None, sup_r)))
                return out_n.setf(
                    Notation.INDEX,
                    (store.atom(base, notation),
                     (None, None, IntegerValue(n), None)))
        if sub is None and sup_l is None and sup_r is None \
                and power is not None:
            try:
                _index_power(power, notation)
                if _is_matrix_valued(f.args[0], notation) \
                        and not _is_matrix_object(f.args[0], notation):
                    # (AB)^n / (A+B)^n: commutative expansion would
                    # fabricate cross terms like 2AB; keep the whole
                    # power as one atom
                    return store.atom(sym, notation)
                base = _atomize_walk(f.args[0], notation, out_n, store)
                pw = Replicator(notation, out_n)(power)
                return out_n.setf(op, (base, (None, None, pw, None)))
            except NotInFragment:
                pass
        return store.atom(sym, notation)
    if op == Notation.P_LIST:
        args = [a for a in f.args if not (isinstance(a, Symbol)
                                          and a.name in Notation.styles)]

        def is_head(a):
            if (isinstance(a, Symbol) and notation.get(a) is None
                    and a.name in FUNC_NAMES):
                return True
            fp = _func_power(a, notation)
            return fp is not None and fp[0] in FUNC_NAMES

        # pass 1: group factors into multiplicative units without atomizing
        # ('int' tail span / 'head' function application / plain 'factor')
        units = []
        i = 0
        while i < len(args):
            a = args[i]
            if _big_operator_name(a, notation) is not None:
                # A big operator binds the rest of the product.  Keeping the
                # whole span as one atom prevents its body from commuting out
                # of scope in the rational core.
                # (taken from the unfiltered factors, so \, survives)
                raw = list(f.args)
                tail = raw[raw.index(a):]
                span = notation.setf(Notation.P_LIST, tail) \
                    if len(tail) > 1 else a
                units.append(('oper', span))
                break
            if is_head(a):
                inner, j = _func_arg_span(args, i, notation, is_head)
                if not inner:
                    raise NotInFragment(f'{a!r} without argument')
                units.append(('head', (a, inner)))
                i = j
            else:
                units.append(('factor', a))
                i += 1

        # pass 2: the noncommutative quote. When two or more units are
        # matrix-valued, their ordered product becomes ONE atom (a word in
        # the free algebra) — polyrat's sorted monomials would otherwise
        # prove AB = BA. Scalar units commute out and stay polynomial.
        def unit_syms(u):
            kind, payload = u
            if kind == 'head':
                return [payload[0]] + payload[1]
            return [payload]

        flags = [any(_is_matrix_valued(m, notation) for m in unit_syms(u))
                 for u in units]
        word = None
        if sum(flags) >= 2:
            parts = []
            for u, fl in zip(units, flags):
                if fl:
                    parts.extend(unit_syms(u))
            span = notation.setf(Notation.P_LIST, parts)
            word = store.atom(span, notation)
            units = [u for u, fl in zip(units, flags) if not fl]

        # pass 3: atomize the remaining scalar units as before
        out_args = []
        for kind, payload in units:
            if kind == 'oper':
                out_args.append(store.atom(payload, notation))
            elif kind == 'head':
                a, inner = payload
                n = None
                fp = _func_power(a, notation)
                if fp is not None:
                    try:
                        n = _index_power(fp[1], notation)
                    except NotInFragment:
                        n = None
                if n is not None and n >= 2:
                    # \sin^{n} arg becomes atom(\sin arg)^n: the power joins
                    # the polynomial layer, so \sin^2 x and (\sin x)^2 share
                    # one canonical form. \sin^{-1} (arcsin reading) and
                    # non-integer powers stay opaque.
                    base = notation.setf(Notation.P_LIST,
                                         [Symbol(fp[0])] + inner)
                    out_args.append(out_n.setf(
                        Notation.INDEX,
                        (store.atom(base, notation),
                         (None, None, IntegerValue(n), None))))
                else:
                    span_syms = [a] + inner
                    span = notation.setf(Notation.P_LIST, span_syms)
                    out_args.append(store.atom(span, notation))
            else:
                out_args.append(_atomize_walk(payload, notation, out_n,
                                              store))
        if word is not None:
            out_args.append(word)
        if len(out_args) == 1:
            return out_args[0]
        return out_n.setf(Notation.P_LIST, out_args)
    # anything else expression-shaped (FUNC, \sqrt, prime, ...) is an atom
    return store.atom(sym, notation)


def _atomized_ratfunc(sym, notation, store):
    """RatFunc over opaque atoms; the caller substitutes atoms back via
    store.mapping()."""
    if _contains_free_infinity(sym, notation):
        raise NotInFragment('infinity outside a big-operator bound')
    work = notation.clone()
    out_n = Notation()
    new_sym = _atomize_walk(sym, work, out_n, store)
    return to_ratfunc(new_sym, out_n)


def _paren_payload(sym, notation):
    """Inner expression of a ()-GROUP (the wrapper Substitutor adds around
    substituted atoms), or None."""
    f = notation.getf(sym, Notation.GROUP)
    if f is not None and f.props.get('br') == '()':
        return f.args[0]
    return None


def _is_signed_or_sum(sym, notation):
    return any(notation.getf(sym, op) is not None
               for op in (Notation.S_LIST, Notation.PLUS, Notation.MINUS))


def _plist_head_kind(a, notation):
    """'func' for \\sin-style factor heads (incl. \\sin^2), 'oper' for
    \\int-style big operators, None otherwise."""
    if isinstance(a, Symbol) and notation.get(a) is None:
        if a.name in FUNC_NAMES:
            return 'func'
        if a.name in Notation.p_oper:
            return 'oper'
    if _big_operator_name(a, notation) is not None:
        return 'oper'
    if _func_power(a, notation) is not None:
        return 'func'
    return None


def _powered_func_payload(sym, notation):
    """INDEX((single function application), n) with a plain positive integer
    power n >= 2 -> (head_name, arg_syms, n), or None. This is the shape the
    atomizer builds for \\sin^{n}-style factors, printable back in that
    standard form where the position is unambiguous."""
    f = notation.getf(sym, Notation.INDEX)
    if f is None:
        return None
    sub, sup_l, power, sup_r = f.args[1]
    if sub is not None or sup_l is not None or sup_r is not None \
            or power is None:
        return None
    p = power
    if isinstance(p, Symbol):
        g = notation.getf(p, Notation.GROUP)
        if g is not None:
            p = g.args[0]
    if not isinstance(p, IntegerValue) or p.val < 2:
        return None
    payload = _paren_payload(f.args[0], notation)
    if payload is None:
        return None
    pf = notation.getf(payload, Notation.P_LIST)
    if pf is None:
        return None
    head = pf.args[0]
    if not (isinstance(head, Symbol) and notation.get(head) is None
            and head.name in FUNC_NAMES and head.name in _UNARY_TABLE):
        return None
    args = list(pf.args)

    def is_head(a):
        return _plist_head_kind(a, notation) is not None

    inner, j = _func_arg_span(args, 0, notation, is_head)
    if not inner or j != len(args):
        # the payload is more than one function application; keep parens
        return None
    return head.name, inner, p.val


def _emit_powered_func(pw, notation, out_n):
    """Factors [\\sin^{n}, arg...] for a validated powered-func payload."""
    name, inner, n = pw
    head = out_n.setf(Notation.INDEX,
                      (Symbol(name), (None, None, IntegerValue(n), None)))
    return [head] + [Replicator(notation, out_n)(m) for m in inner]


def _powered_subscript_payload(sym, notation):
    """INDEX((C_{1}), n) built by atom substitution -> (C_{1}, n)."""
    f = notation.getf(sym, Notation.INDEX)
    if f is None:
        return None
    sub_l, sup_l, power, sub_r = f.args[1]
    if sub_l is not None or sup_l is not None or sub_r is not None \
            or power is None:
        return None
    try:
        n = _index_power(power, notation)
    except NotInFragment:
        return None
    if n < 2:
        return None
    payload = _paren_payload(f.args[0], notation)
    if payload is None or _subscript_var(payload, notation) is None:
        return None
    pf = notation.getf(payload, Notation.INDEX)
    if pf is None or pf.args[1][2] is not None:
        return None
    return payload, n


def _emit_powered_subscript(payload, n, notation, out_n):
    """Recombine the atom-store spelling (C_{1})^n as C_{1}^n."""
    f = notation.getf(payload, Notation.INDEX)
    base = Replicator(notation, out_n)(f.args[0])
    sub = Replicator(notation, out_n)(f.args[1][3])
    return out_n.setf(
        Notation.INDEX,
        (base, (None, None, IntegerValue(n), sub)))


def _relax_atom_parens(sym, notation):
    """Cosmetic pass over an atom-substituted result: drop the () wrappers
    Substitutor adds wherever the expression stays unambiguous — at the
    root, in additive-term position, and as the trailing factor of a
    product not captured by a preceding function head — and print powered
    atoms back as \\sin^{2}x in those same positions. The oracle check
    and the write_latex validation still guard every emitted result."""
    out_n = Notation()
    return _relax_walk(sym, notation, out_n), out_n


def _relax_walk(sym, notation, out_n):
    payload = _paren_payload(sym, notation)
    if payload is not None:
        # term/root position: parens only matter around sums and signs
        if _is_signed_or_sum(payload, notation):
            inner = _relax_walk(payload, notation, out_n)
            return out_n.setf(Notation.GROUP, (inner,), br='()')
        # recurse: instantiated lemma templates nest Substitutor wrappers
        return _relax_walk(payload, notation, out_n)
    comp = notation.getf(sym, Notation.COMP)
    if comp is not None:
        rel = comp.sym.props.get('op')
        return out_n.setf(Symbol('comp', op=rel),
                          (_relax_walk(comp.args[0], notation, out_n),
                           _relax_walk(comp.args[1], notation, out_n)))
    for op in (Notation.PLUS, Notation.MINUS):
        f = notation.getf(sym, op)
        if f is not None:
            return out_n.setf(op, (_relax_walk(f.args[0], notation, out_n),))
    f = notation.getf(sym, Notation.S_LIST)
    if f is not None:
        return out_n.setf(Notation.S_LIST,
                          [_relax_walk(t, notation, out_n) for t in f.args])
    f = notation.getf(sym, Notation.P_LIST)
    if f is not None:
        return _relax_plist(f, notation, out_n)
    if isinstance(sym, Symbol):
        ff = notation.get(sym)
        if ff is not None and ff.sym.name in FRAC_NAMES:
            def frac_arg(a):
                g = notation.getf(a, Notation.GROUP)
                if g is not None and g.props.get('br') == '{}':
                    inner = _relax_walk(g.args[0], notation, out_n)
                    return out_n.setf(Notation.GROUP, (inner,), br='{}')
                return _relax_walk(a, notation, out_n)
            return out_n.setf(ff.sym, (frac_arg(ff.args[0]),
                                       frac_arg(ff.args[1])), **ff.props)
    pw = _powered_func_payload(sym, notation)
    if pw is not None:
        # term/root position: ( \sin x)^{2} prints as \sin^{2}x
        return out_n.setf(Notation.P_LIST,
                          _emit_powered_func(pw, notation, out_n))
    ps = _powered_subscript_payload(sym, notation)
    if ps is not None:
        return _emit_powered_subscript(ps[0], ps[1], notation, out_n)
    return Replicator(notation, out_n)(sym)


def _relax_plist(f, notation, out_n):
    args = list(f.args)
    starts_head = [False] * len(args)

    def wrapped_function(a):
        if _powered_func_payload(a, notation) is not None:
            return True
        payload = _paren_payload(a, notation)
        pf = notation.getf(payload, Notation.P_LIST) \
            if payload is not None else None
        return (pf is not None and pf.args
                and _plist_head_kind(pf.args[0], notation) == 'func')

    # A parenthesized function factor can lose its wrapper before another
    # function factor because the next head terminates the current argument
    # span.  Compute that boundary right-to-left: (sin x)(cos x) may flatten,
    # while (sin x)(cos x)y must stay wrapped because cos would capture y.
    for idx in range(len(args) - 1, -1, -1):
        raw_head = _plist_head_kind(args[idx], notation) is not None
        last = idx == len(args) - 1
        boundary_after = last or starts_head[idx + 1]
        prev_raw_head = idx > 0 and _plist_head_kind(
            args[idx - 1], notation) is not None
        starts_head[idx] = raw_head or (
            wrapped_function(args[idx]) and boundary_after
            and not prev_raw_head)

    new_args = []
    for idx, a in enumerate(args):
        last = idx == len(args) - 1
        boundary_after = last or starts_head[idx + 1]
        prev_head = idx > 0 and _plist_head_kind(args[idx - 1],
                                                 notation) is not None
        payload = _paren_payload(a, notation)
        if payload is None:
            pw = _powered_func_payload(a, notation)
            if pw is not None and boundary_after and not prev_head:
                # A following function head is as safe as product-end: it
                # terminates this function's argument span.
                new_args.extend(_emit_powered_func(pw, notation, out_n))
                continue
            ps = _powered_subscript_payload(a, notation)
            if ps is not None:
                new_args.append(_emit_powered_subscript(
                    ps[0], ps[1], notation, out_n))
                continue
            new_args.append(Replicator(notation, out_n)(a))
            continue
        if _is_signed_or_sum(payload, notation):
            # collected coefficient sums keep their parens; relax inside
            inner = _relax_walk(payload, notation, out_n)
            new_args.append(out_n.setf(Notation.GROUP, (inner,), br='()'))
            continue
        pf = notation.getf(payload, Notation.P_LIST)
        if boundary_after and not prev_head:
            if pf is not None \
                    and _plist_head_kind(pf.args[0], notation) is not None:
                # function-application span: splice its factors in
                new_args.extend(Replicator(notation, out_n)(m)
                                for m in pf.args)
                continue
        if last and not prev_head:
            if pf is not None \
                    and _paren_payload(pf.args[0], notation) is not None:
                # trailing ((x+2)(x-2)) from a subterm rewrite:
                # multiplication is flat, splice the relaxed factors in
                inner = _relax_plist(pf, notation, out_n)
                inf = out_n.getf(inner, Notation.P_LIST)
                new_args.extend(inf.args if inf is not None else [inner])
                continue
            if pf is None and not isinstance(payload, Value):
                # bare values keep their parens: x(2), never x{2}
                new_args.append(_relax_walk(payload, notation, out_n))
                continue
        # kept parens still get relaxed inside: instantiated lemma
        # templates nest Substitutor wrappers arbitrarily deep
        g = notation.getf(a, Notation.GROUP)
        if 'quoted' in g.props:
            new_args.append(Replicator(notation, out_n)(a))
            continue
        inner = _relax_walk(payload, notation, out_n)
        new_args.append(out_n.setf(Notation.GROUP, (inner,), **g.props))
    if len(new_args) == 1:
        return new_args[0]
    return out_n.setf(Notation.P_LIST, new_args)


def _atomized_canonical(sym, notation):
    """Canonical latex of an expression outside the fragment.
    Returns (latex, atom_count); raises NotInFragment if impossible."""
    store = _AtomStore()
    rf = _atomized_ratfunc(sym, notation, store)
    res_n = Notation()
    res_s = ratfunc_to_notation(rf, res_n)
    if not store.exprs:
        return write_latex(res_s, res_n), 0
    final_n = Notation()
    final_s = Substitutor(res_n, final_n, store.mapping())(res_s)
    final_s, final_n = _relax_atom_parens(final_s, final_n)
    return write_latex(final_s, final_n), len(store.exprs)


def _comp_split(sym, notation):
    """Return (lhs, rhs, rel) for a supported relation, None for plain
    expressions."""
    comp = notation.getf(sym, Notation.COMP)
    if comp is None:
        return None
    rel = comp.sym.props.get('op')
    if rel not in _SUPPORTED_REL:
        raise PrimitiveError(f'unsupported relation {rel!r}')
    return comp.args[0], comp.args[1], rel


def _canonical_side(side, notation):
    try:
        rf = to_ratfunc(side, notation)
    except NotInFragment:
        return _atomized_canonical(side, notation)[0]
    out_n = Notation()
    return write_latex(ratfunc_to_notation(rf, out_n), out_n)


def expand(expr):
    """Distribute products and integer powers of sums; canonical order.
    On an equation, expands each side. Rational-exponent powers of plain
    variables fold via t = x^{1/q} (recording x > 0)."""
    args = {'expr': expr}
    fold = None
    try:
        sym, notation = parse_latex(expr)
        split = _comp_split(sym, notation)
        if split:
            lhs, rhs, rel = split
            new_l = _canonical_side(lhs, notation)
            new_r = _canonical_side(rhs, notation)
            rec = _result('expand', args, expr, f'{new_l} {rel} {new_r}')
            rec['check'] = _merge_checks(
                numeric_spot_check(write_latex(lhs, notation), new_l),
                numeric_spot_check(write_latex(rhs, notation), new_r))
            return rec
        fold = _puiseux_fold(sym, notation)
        if fold is not None:
            sym, notation = fold[0], fold[1]
        rf = to_ratfunc(sym, notation)
    except PrimitiveError as e:
        return _error('expand', args, str(e))
    except ZeroDivisionError:
        return _error('expand', args, 'expression contains division by zero')
    except NotInFragment:
        # canonicalize over opaque atoms (\cos x, e^x, unevaluated \int)
        try:
            result, n_atoms = _atomized_canonical(sym, notation)
            if fold is not None:
                result = fold[2](result)
        except NotInFragment as e:
            return _error('expand', args,
                          f'outside the rational fragment: {e}')
        except ZeroDivisionError:
            return _error('expand', args,
                          'expression contains division by zero')
        return _checked(_result(
            'expand', args, expr, result,
            assumptions=fold[3] if fold is not None else None,
            extra={'opaque_atoms': n_atoms}))
    try:
        out_n = Notation()
        result = write_latex(ratfunc_to_notation(rf, out_n), out_n)
        if fold is not None:
            result = fold[2](result)
    except PrimitiveError as e:
        return _error('expand', args, str(e))
    return _checked(_result(
        'expand', args, expr, result,
        assumptions=fold[3] if fold is not None else None))


def collect(expr, var):
    """Group a polynomial by powers of `var` (descending); outside the
    rational fragment, collects over opaque atoms. On an equation,
    collects each side; rational functions collect numerator and
    denominator separately."""
    args = {'expr': expr, 'var': var}
    try:
        sym, notation = parse_latex(expr)
        split = _comp_split(sym, notation)
        if split:
            lhs, rhs, rel = split
            sides = [_collect_side(s, notation, var, require_var=False)
                     for s in (lhs, rhs)]
            rec = _result('collect', args, expr,
                          f'{sides[0]} {rel} {sides[1]}')
            rec['check'] = _merge_checks(
                numeric_spot_check(write_latex(lhs, notation), sides[0]),
                numeric_spot_check(write_latex(rhs, notation), sides[1]))
            return rec
        result = _collect_side(sym, notation, var, require_var=True)
    except PrimitiveError as e:
        return _error('collect', args, str(e))
    except ZeroDivisionError:
        return _error('collect', args, 'expression contains division by zero')
    except NotInFragment as e:
        return _error('collect', args, f'outside the rational fragment: {e}')
    return _checked(_result('collect', args, expr, result))


def _collect_side(side, notation, var, require_var):
    """Collected latex for one side; collects over opaque atoms when the
    side leaves the rational fragment."""
    store = None
    try:
        rf = to_ratfunc(side, notation)
    except NotInFragment:
        store = _AtomStore()
        rf = _atomized_ratfunc(side, notation, store)
    if require_var and var not in rf.variables():
        note = (' (opaque subexpressions are not entered)'
                if store is not None else '')
        raise PrimitiveError(
            f'variable {var!r} does not occur in expression{note}')
    out_n = Notation()
    res = _collect_rf_sym(rf, var, out_n)
    if store is not None and store.exprs:
        fin_n = Notation()
        res = Substitutor(out_n, fin_n, store.mapping())(res)
        res, out_n = _relax_atom_parens(res, fin_n)
    result = write_latex(res, out_n)
    try:
        parse_latex(result)
    except PrimitiveError as e:
        raise PrimitiveError(f'internal: unparseable result: {e}')
    return result


def _collect_rf_sym(rf, var, notation):
    """Collected notation for a RatFunc; a side without `var` prints in
    plain canonical order."""
    def side(p):
        if var in p.variables():
            return _collect_poly_sym(p, var, notation)
        return poly_to_notation(p, notation)
    if rf.is_poly():
        return side(rf.num)
    num = notation.setf(Notation.GROUP, (side(rf.num),), br='{}')
    den = notation.setf(Notation.GROUP, (side(rf.den),), br='{}')
    return notation.setf(Symbol('\\frac'), (num, den))


def _rel_holds(rel, a, b):
    if rel == '=':
        return a == b
    if rel in ('\\ne', '\\neq'):
        return a != b
    if rel in ('<', '\\lt'):
        return a < b
    if rel in ('>', '\\gt'):
        return a > b
    if rel in ('\\le', '\\leq'):
        return a <= b
    return a >= b


def _collect_poly_sym(poly, var, notation):
    """Collected-by-var notation: descending powers of `var`, coefficient
    sums parenthesized, single-term coefficients inlined."""
    buckets = {}
    for mono, coeff in poly.terms.items():
        d = dict(mono)
        k = d.pop(var, 0)
        rest = tuple(sorted(d.items()))
        buckets.setdefault(k, {})[rest] = coeff
    if not buckets:
        return IntegerValue(0)
    terms = []
    for i, k in enumerate(sorted(buckets, reverse=True)):
        coeff_poly = Poly(buckets[k])
        if k == 0:
            term = poly_to_notation(coeff_poly, notation)
            if len(coeff_poly.terms) > 1:
                term = notation.setf(Notation.GROUP, (term,), br='()')
        elif len(coeff_poly.terms) == 1:
            # single-term coefficient: merge var^k into the monomial and
            # reuse the canonical term builder (sign handling included)
            ((mono, coeff),) = tuple(coeff_poly.terms.items())
            merged = tuple(sorted(dict(mono, **{var: k}).items()))
            term = poly_to_notation(Poly({merged: coeff}), notation)
        else:
            inner = poly_to_notation(coeff_poly, notation)
            group = notation.setf(Notation.GROUP, (inner,), br='()')
            var_f = Symbol(var) if k == 1 else notation.setf(
                Notation.INDEX,
                (Symbol(var), (None, None, IntegerValue(k), None)))
            term = notation.setf(Notation.P_LIST, (group, var_f))
        if i > 0 and notation.getf(term, Notation.MINUS) is None:
            term = notation.setf(Notation.PLUS, (term,))
        terms.append(term)
    if len(terms) == 1:
        return terms[0]
    return notation.setf(Notation.S_LIST, terms)


def evaluate(expr):
    """Exact arithmetic when no free variables remain. On an equation,
    evaluates both sides and reports whether the equality holds."""
    args = {'expr': expr}
    try:
        sym, notation = parse_latex(expr)
        split = _comp_split(sym, notation)
    except PrimitiveError as e:
        return _error('evaluate', args, str(e))
    if split:
        lhs, rhs, rel = split
        sides = []
        for side in (lhs, rhs):
            try:
                rf = to_ratfunc(side, notation)
            except (NotInFragment, ZeroDivisionError) as e:
                return _error('evaluate', args, f'cannot evaluate side: {e}')
            if not rf.is_const():
                return _error('evaluate', args,
                              'free variables remain: '
                              + ', '.join(sorted(rf.variables())))
            sides.append(rf.const_value())
        out = []
        for q in sides:
            out_n = Notation()
            if q < 0:
                s = out_n.setf(Notation.MINUS,
                               (_fraction_to_notation(-q, out_n),))
            elif q == 0:
                s = IntegerValue(0)
            else:
                s = _fraction_to_notation(q, out_n)
            out.append(write_latex(s, out_n))
        rec = _result('evaluate', args, expr, f'{out[0]} {rel} {out[1]}',
                      check={'status': 'exact'},
                      extra={'exact': True,
                             'holds': _rel_holds(rel, sides[0], sides[1])})
        return rec
    try:
        rf = to_ratfunc(sym, notation)
        if not rf.is_const():
            return _error('evaluate', args,
                          'free variables remain: '
                          + ', '.join(sorted(rf.variables())))
        q = rf.const_value()
        out_n = Notation()
        if q < 0:
            out_s = out_n.setf(Notation.MINUS,
                               (_fraction_to_notation(-q, out_n),))
        elif q == 0:
            out_s = IntegerValue(0)
        else:
            out_s = _fraction_to_notation(q, out_n)
        result = write_latex(out_s, out_n)
        return _checked(_result('evaluate', args, expr, result,
                                extra={'exact': True}))
    except ZeroDivisionError:
        return _error('evaluate', args, 'division by zero')
    except NotInFragment:
        pass
    # outside the exact fragment: fall back to a float evaluation
    if free_symbols(sym, notation):
        return _error('evaluate', args,
                      'free variables remain: '
                      + ', '.join(sorted(free_symbols(sym, notation))))
    try:
        v = numeric_eval(sym, notation, {})
    except (EvalError, ZeroDivisionError, ValueError) as e:
        return _error('evaluate', args, f'cannot evaluate: {e}')
    if isinstance(v, list):
        return _error('evaluate', args,
                      'matrix-valued expression: evaluate returns scalars; '
                      'expand collects matrix terms instead')
    return _result('evaluate', args, expr, repr(round(v, 12)),
                   extra={'exact': False})



# ---------------------------------------------------------------------------
# primitive: rewrite (lemma library)
# ---------------------------------------------------------------------------

class Lemma(object):
    def __init__(self, name, lhs, rhs, params, description=''):
        self.name = name
        self.lhs = lhs
        self.rhs = rhs
        self.params = params
        self.description = description


LEMMAS = {}


def register_lemma(name, lhs, rhs, params, description=''):
    LEMMAS[name] = Lemma(name, lhs, rhs, params, description)


register_lemma('diff_squares', 'a^2 - b^2', '(a + b)(a - b)',
               ['a', 'b'], 'difference of squares')
register_lemma('square_of_sum', '(a + b)^2', 'a^2 + 2 a b + b^2',
               ['a', 'b'], 'square of a sum')
register_lemma('square_of_diff', '(a - b)^2', 'a^2 - 2 a b + b^2',
               ['a', 'b'], 'square of a difference')
register_lemma('cube_of_sum', '(a + b)^3',
               'a^3 + 3 a^2 b + 3 a b^2 + b^3', ['a', 'b'],
               'cube of a sum')
register_lemma('diff_cubes', 'a^3 - b^3', '(a - b)(a^2 + a b + b^2)',
               ['a', 'b'], 'difference of cubes')
register_lemma('sum_cubes', 'a^3 + b^3', '(a + b)(a^2 - a b + b^2)',
               ['a', 'b'], 'sum of cubes')


def list_lemmas():
    return {'ok': True, 'op': 'lemmas',
            'lemmas': [{'name': l.name, 'lhs': l.lhs, 'rhs': l.rhs,
                        'description': l.description}
                       for l in LEMMAS.values()]}


def _int_nth_root(k, n):
    """Exact integer n-th root of k >= 0, or None."""
    if k < 0:
        return None
    if k in (0, 1):
        return k
    r = round(k ** (1.0 / n))
    for cand in (r - 1, r, r + 1):
        if cand >= 0 and cand ** n == k:
            return cand
    return None


def _perfect_power_root(sym, notation, n):
    """n-th root of a monomial binding (4 -> 2, 4x^2 -> 2x, x^4 -> x^2 for
    n=2), returned as (root_sym, root_notation); None when the binding is
    not a perfect n-th power monomial."""
    try:
        rf = to_ratfunc(sym, notation)
    except (NotInFragment, ZeroDivisionError):
        return None
    if not rf.is_poly():
        return None
    terms = list(rf.num.terms.items())
    if len(terms) != 1:
        return None
    mono, coeff = terms[0]
    if coeff <= 0:
        return None
    cn = _int_nth_root(coeff.numerator, n)
    cd = _int_nth_root(coeff.denominator, n)
    if cn is None or cd is None:
        return None
    if any(e % n for _, e in mono):
        return None
    root = Poly({tuple((v, e // n) for v, e in mono): Fraction(cn, cd)})
    out_n = Notation()
    return poly_to_notation(root, out_n), out_n


def _pattern_stats(sym, notation, counts, power_nodes):
    """Count leaf-symbol occurrences in a pattern tree; record leaves that
    appear as INDEX(leaf, n) with a plain integer power n >= 2."""
    if isinstance(sym, tuple):
        for t in sym:
            if t is not None:
                _pattern_stats(t, notation, counts, power_nodes)
        return
    if not isinstance(sym, Symbol):
        return
    f = notation.get(sym)
    if f is None:
        counts[sym.name] = counts.get(sym.name, 0) + 1
        return
    if f.sym == Notation.INDEX:
        base = f.args[0]
        sub, sup_l, power, sup_r = f.args[1]
        if (isinstance(base, Symbol) and notation.get(base) is None
                and sub is None and sup_l is None and sup_r is None):
            try:
                n = _index_power(power, notation)
            except NotInFragment:
                n = None
            if n is not None and n >= 2:
                counts[base.name] = counts.get(base.name, 0) + 1
                power_nodes.setdefault(base.name, []).append(n)
                return
    for a in f.args:
        _pattern_stats(a, notation, counts, power_nodes)


def _lemma_power_variants(src, params):
    """Variants of a lemma source pattern where a^n power terms whose
    parameter occurs nowhere else are replaced by wildcards to be bound as
    perfect n-th power monomials: 'a^2 - b^2' then also matches x^2 - 4
    (b := 2), 4x^2 - 9 (a := 2x, b := 3) or x^4 - y^4. Returns
    [(variant_src, variant_params, {wildcard: (param, n)})], fewer
    replacements first so structural bindings keep priority."""
    try:
        sym, notation = parse_latex(src)
    except PrimitiveError:
        return []
    counts, power_nodes = {}, {}
    _pattern_stats(sym, notation, counts, power_nodes)
    candidates = [(p, power_nodes[p][0]) for p in params
                  if counts.get(p) == 1
                  and len(power_nodes.get(p, ())) == 1]
    if not candidates:
        return []
    used = set(params) | set(re.findall(r'[A-Za-z]', src))
    fresh_pool = [ch for ch in 'cdefghklmnpqrstuvw' if ch not in used]
    variants = []
    for size in range(1, len(candidates) + 1):
        for combo in combinations(candidates, size):
            if len(fresh_pool) < size:
                continue
            cur = src
            powermap = {}
            v_params = [p for p in params
                        if p not in {c[0] for c in combo}]
            ok = True
            for k, (p, n) in enumerate(combo):
                fresh = fresh_pool[k]
                pat = re.compile(r'(?<![A-Za-z\\])' + re.escape(p)
                                 + r'\s*\^\s*(?:\{' + str(n) + r'\}|'
                                 + str(n) + r')(?![0-9])')
                cur, cnt = pat.subn(fresh, cur, count=1)
                if cnt != 1:
                    ok = False
                    break
                powermap[fresh] = (p, n)
                v_params.append(fresh)
            if not ok:
                continue
            # verify the string surgery on the reparsed variant
            try:
                v_sym, v_not = parse_latex(cur)
            except PrimitiveError:
                continue
            v_counts = {}
            _pattern_stats(v_sym, v_not, v_counts, {})
            if any(v_counts.get(p, 0) != 0 for p, _ in combo):
                continue
            if any(v_counts.get(w, 0) != 1 for w in powermap):
                continue
            variants.append((cur, v_params, powermap))
    return variants


def rewrite(expr, lemma_name, direction='forward', at=None):
    """Apply a registered equality lemma at the root, or at the subterm
    selected by `at` (target LaTeX or 1-based match index)."""
    args = {'expr': expr, 'lemma': lemma_name, 'direction': direction}
    if at is not None:
        args['at'] = str(at)
    lemma = LEMMAS.get(lemma_name)
    if lemma is None:
        return _error('rewrite', args,
                      f'unknown lemma {lemma_name!r}; known: '
                      + ', '.join(sorted(LEMMAS)))
    if direction not in ('forward', 'backward'):
        return _error('rewrite', args,
                      "direction must be 'forward' or 'backward'")
    src, dst = ((lemma.lhs, lemma.rhs) if direction == 'forward'
                else (lemma.rhs, lemma.lhs))
    try:
        sym, notation = parse_latex(expr)
    except PrimitiveError as e:
        return _error('rewrite', args, str(e))

    def stage_positions(pat_src, pat_params, powermap, stage, seen):
        """Matches of one pattern in position order: root first, then
        subterms in parse order (children precede parents, so an inner
        match precedes its enclosing one). Wildcards from powermap must
        bind perfect n-th power monomials, whose roots are returned bound
        to the original lemma parameter. A node already claimed by an
        earlier stage keeps that stage's binding."""
        pat = comparer.pattern(pat_src,
                               [(p, NotationParam.Any) for p in pat_params])

        def validate(s):
            bound = {}
            for wild, (orig, n) in powermap.items():
                if wild not in s:
                    return None
                root = _perfect_power_root(s[wild], notation, n)
                if root is None:
                    return None
                bound[orig] = root
            return bound

        found = []
        candidates = [sym] + [node for node in notation.rel if node != sym]
        for order, node in enumerate(candidates):
            if node in seen:
                continue
            s = pat.match(node, notation)
            if s is None:
                continue
            v = validate(s)
            if v is None:
                continue
            seen.add(node)
            found.append({'node': node, 'subst': s, 'numeric': v,
                          'stage': stage, 'order': order})
        return found

    # every match position across the base pattern and its numeric
    # variants (a^n terms binding perfect n-th power monomials); a node's
    # binding comes from the earliest stage that matches it
    seen = set()
    positions = stage_positions(src, lemma.params, {}, 0, seen)
    for v_i, (v_src, v_params, v_map) in enumerate(
            _lemma_power_variants(src, lemma.params)):
        positions += stage_positions(v_src, v_params, v_map, v_i + 1, seen)
    # a transparent wrapper and its inner content are ONE position: keep
    # the inner node (except at the root, which keeps itself)
    by_node = {p['node'] for p in positions}
    dropped = set()
    for p in positions:
        f = notation.get(p['node'])
        if (f is None or f.sym not in (Notation.GROUP, Notation.V_GROUP)
                or not f.args or f.args[0] not in by_node):
            continue
        dropped.add(f.args[0] if p['node'] == sym else p['node'])
    positions = [p for p in positions if p['node'] not in dropped]
    positions.sort(key=lambda p: p['order'])

    if not positions:
        return _error('rewrite', args,
                      f'expression does not match pattern {src!r} '
                      '(at the root or any subterm)')

    def position_menu():
        items = [f'{i}. {write_latex(p["node"], notation)}'
                 for i, p in enumerate(positions[:6], 1)]
        if len(positions) > 6:
            items.append('...')
        return '; '.join(items)

    if at is None:
        # default keeps first-match behavior: structural bindings first,
        # then variants, root before subterms within a stage
        chosen = min(positions, key=lambda p: (p['stage'], p['order']))
    elif isinstance(at, int) or at.strip().isdigit():
        index = int(at)
        if not 1 <= index <= len(positions):
            return _error(
                'rewrite', args,
                f'at index {index} is out of range; '
                f'{len(positions)} match(es): {position_menu()}')
        chosen = positions[index - 1]
    else:
        try:
            parse_latex(at)
        except PrimitiveError as e:
            return _error('rewrite', args, f'at: {e}')
        chosen = next(
            (p for p in positions
             if same_expression(write_latex(p['node'], notation), at)),
            None)
        if chosen is None:
            return _error(
                'rewrite', args,
                f'the lemma does not match at {at!r}; '
                f'{len(positions)} match(es): {position_menu()}')
    target, subst, numeric = (chosen['node'], chosen['subst'],
                              chosen['numeric'])
    matches = len(positions)
    tsym, tnotation = parse_latex(dst)
    mapping = {}
    for p in lemma.params:
        if p in numeric:
            mapping[Symbol(p)] = numeric[p]
        elif p in subst:
            mapping[Symbol(p)] = (subst[p], notation)
        else:
            return _error('rewrite', args,
                          f'pattern variable {p!r} unbound')
    extra = {}
    if target is sym:
        out_n = Notation()
        out_s = Substitutor(tnotation, out_n, mapping)(tsym)
        out_s, out_n = _relax_atom_parens(out_s, out_n)
        result = write_latex(out_s, out_n)
    else:
        # replace the matched subterm inside a clone of the graph
        at_s = write_latex(target, notation)
        work = notation.clone()
        inst = Substitutor(tnotation, work, mapping)(tsym)
        f_inst = work.get(inst)
        if f_inst is None:
            work.repf(target, Func(Notation.GROUP, (inst,), br='()'))
        else:
            work.repf(target, f_inst)
        out_s, out_n = _relax_atom_parens(sym, work)
        result = write_latex(out_s, out_n)
        extra = {'at': at_s, 'matches': matches}
    if numeric:
        extra['numeric'] = {p: write_latex(s, n)
                            for p, (s, n) in numeric.items()}
    rec = _result('rewrite', args, expr, result, extra=extra)
    return _checked(rec)


# ---------------------------------------------------------------------------
# primitives: named factorings (no general `factor` on purpose)
# ---------------------------------------------------------------------------

def _q_str(q):
    """Fraction -> latex string."""
    out_n = Notation()
    if q < 0:
        return '-' + write_latex(_fraction_to_notation(-q, out_n), out_n)
    if q == 0:
        return '0'
    return write_latex(_fraction_to_notation(q, out_n), out_n)


def _fraction_sqrt(q):
    if q < 0:
        return None
    n = math.isqrt(q.numerator)
    d = math.isqrt(q.denominator)
    if n * n == q.numerator and d * d == q.denominator:
        return Fraction(n, d)
    return None


def _as_poly_sym(sym, notation, op):
    """Convert one already-parsed expression (or relation side) to Poly."""
    rf = to_ratfunc(sym, notation)
    if not rf.is_poly():
        raise PrimitiveError(f'{op} supports polynomials only')
    return rf.num


def _factor_failure(e):
    if isinstance(e, ZeroDivisionError):
        return 'division by zero'
    if isinstance(e, NotInFragment):
        return f'outside the rational fragment: {e}'
    return str(e)


def _factor_gcd_side(sym, notation):
    poly = _as_poly_sym(sym, notation, 'factor_gcd')
    if poly.is_zero():
        raise PrimitiveError('zero polynomial')
    content = poly.content()
    if poly.leading_coeff() < 0:
        content = -content
    mono = poly.monomial_gcd()
    if content == 1 and mono == ():
        raise PrimitiveError('no common factor to pull out')
    quotient = poly.divide_monomial(mono).scale(1 / content)
    if quotient.is_const():
        raise PrimitiveError('expression is a single term; nothing to factor')
    out_n = Notation()
    factor_s = write_latex(poly_to_notation(Poly({mono: content}), out_n),
                           out_n)
    out_n2 = Notation()
    quot_s = write_latex(poly_to_notation(quotient, out_n2), out_n2)
    return f'{factor_s}{_paren(quot_s)}'


def factor_gcd(expr):
    """Pull out common factors, on a plain polynomial or relation sides."""
    args = {'expr': expr}
    try:
        sym, notation = parse_latex(expr)
        split = _comp_split(sym, notation)
        if split:
            lhs, rhs, rel = split
            outputs = []
            changed = []
            failures = []
            for label, side in (('lhs', lhs), ('rhs', rhs)):
                original = write_latex(side, notation)
                try:
                    outputs.append(_factor_gcd_side(side, notation))
                    changed.append(label)
                except (PrimitiveError, ZeroDivisionError, NotInFragment) as e:
                    outputs.append(original)
                    failures.append(f'{label}: {_factor_failure(e)}')
            if not changed:
                raise PrimitiveError('neither side can be factored ('
                                     + '; '.join(failures) + ')')
            result = f'{outputs[0]} {rel} {outputs[1]}'
            parse_latex(result)
            rec = _result('factor_gcd', args, expr, result,
                          extra={'factored_sides': changed})
            rec['check'] = _merge_checks(
                numeric_spot_check(write_latex(lhs, notation), outputs[0]),
                numeric_spot_check(write_latex(rhs, notation), outputs[1]))
            return rec
        result = _factor_gcd_side(sym, notation)
        parse_latex(result)
    except PrimitiveError as e:
        return _error('factor_gcd', args, str(e))
    except (ZeroDivisionError, NotInFragment) as e:
        return _error('factor_gcd', args, _factor_failure(e))
    return _checked(_result('factor_gcd', args, expr, result))


def _factor_quadratic_side(sym, notation, var):
    poly = _as_poly_sym(sym, notation, 'factor_quadratic')
    if poly.degree(var) != 2:
        raise PrimitiveError(f'expression is not quadratic in {var!r}')
    coeffs = {0: Fraction(0), 1: Fraction(0), 2: Fraction(0)}
    for mono, coeff in poly.terms.items():
        d = dict(mono)
        k = d.pop(var, 0)
        if d:
            raise PrimitiveError('coefficients must be constants for now')
        coeffs[k] = coeff
    a, b, c = coeffs[2], coeffs[1], coeffs[0]
    disc = b * b - 4 * a * c
    if disc < 0:
        raise PrimitiveError('negative discriminant; no real factorization')
    s = _fraction_sqrt(disc)
    if s is None:
        raise PrimitiveError('roots are not rational; not factorable over Q '
                             f'(discriminant {disc})')
    r1 = (-b + s) / (2 * a)
    r2 = (-b - s) / (2 * a)

    def root_factor(r):
        if r == 0:
            return var
        sign = '-' if r > 0 else '+'
        return f'\\left({var} {sign} {_q_str(abs(r))}\\right)'

    if r1 == r2:
        body = f'{root_factor(r1)}^{{2}}'
    else:
        body = root_factor(r1) + root_factor(r2)
    if a == 1:
        result = body
    else:
        result = f'{_q_str(a)}{body}'
    return result, [str(r1), str(r2)]


def factor_quadratic(expr, var):
    """Factor quadratics with rational roots, including relation sides."""
    args = {'expr': expr, 'var': var}
    try:
        sym, notation = parse_latex(expr)
        split = _comp_split(sym, notation)
        if split:
            lhs, rhs, rel = split
            outputs = []
            changed = []
            roots_by_side = {}
            failures = []
            for label, side in (('lhs', lhs), ('rhs', rhs)):
                original = write_latex(side, notation)
                try:
                    result, roots = _factor_quadratic_side(side, notation,
                                                           var)
                    outputs.append(result)
                    roots_by_side[label] = roots
                    changed.append(label)
                except (PrimitiveError, ZeroDivisionError, NotInFragment) as e:
                    outputs.append(original)
                    failures.append(f'{label}: {_factor_failure(e)}')
            if not changed:
                raise PrimitiveError('neither side can be factored ('
                                     + '; '.join(failures) + ')')
            result = f'{outputs[0]} {rel} {outputs[1]}'
            parse_latex(result)
            rec = _result('factor_quadratic', args, expr, result,
                          extra={'factored_sides': changed,
                                 'roots_by_side': roots_by_side})
            rec['check'] = _merge_checks(
                numeric_spot_check(write_latex(lhs, notation), outputs[0]),
                numeric_spot_check(write_latex(rhs, notation), outputs[1]))
            return rec
        result, roots = _factor_quadratic_side(sym, notation, var)
        parse_latex(result)
    except PrimitiveError as e:
        return _error('factor_quadratic', args, str(e))
    except (ZeroDivisionError, NotInFragment) as e:
        return _error('factor_quadratic', args, _factor_failure(e))
    rec = _result('factor_quadratic', args, expr, result,
                  extra={'roots': roots})
    return _checked(rec)



# ---------------------------------------------------------------------------
# checker: equal?
# ---------------------------------------------------------------------------

def equal_exprs(expr1, expr2, assuming=None):
    """yes / no / unknown. Canonical forms decide the rational fragment;
    the numeric oracle answers probabilistically outside it. Equations are
    compared side by side. `assuming` restricts the question to a stated
    region ("x > 0"): the oracle then samples only there, and a verdict
    that needed the restriction says so and carries it."""
    args = {'expr1': expr1, 'expr2': expr2}
    if assuming:
        args['assuming'] = assuming
    assumed, error = _assumption_records(assuming)
    if error is not None:
        return _error('equal', args, error)
    try:
        s1, n1 = parse_latex(expr1)
        s2, n2 = parse_latex(expr2)
        sp1 = _comp_split(s1, n1)
        sp2 = _comp_split(s2, n2)
    except PrimitiveError as e:
        return _error('equal', args, str(e))
    if (sp1 is None) != (sp2 is None):
        return {'ok': True, 'op': 'equal', 'args': args, 'verdict': 'no',
                'method': 'structural',
                'reason': 'one side is an equation, the other is not'}
    if sp1 is not None:
        if sp1[2] != sp2[2]:
            return {'ok': True, 'op': 'equal', 'args': args, 'verdict': 'no',
                    'method': 'structural',
                    'reason': f'different relations {sp1[2]!r} vs {sp2[2]!r}'}
        verdicts = []
        conditional = False
        for a, b in zip(sp1[:2], sp2[:2]):
            sub = equal_exprs(write_latex(a, n1), write_latex(b, n2),
                              assuming=assuming)
            if not sub.get('ok'):
                return sub
            verdicts.append(sub['verdict'])
            conditional = conditional or bool(sub.get('assumptions'))
        if all(v == 'yes' for v in verdicts):
            verdict = 'yes'
        elif 'no' in verdicts:
            verdict = 'no'
        else:
            verdict = 'unknown'
        rec = {'ok': True, 'op': 'equal', 'args': args, 'verdict': verdict,
               'method': 'per-side'}
        if conditional:
            rec['assumptions'] = list(assumed)
        return rec
    try:
        rf1 = to_ratfunc(s1, n1)
        rf2 = to_ratfunc(s2, n2)
        verdict = 'yes' if rf1 == rf2 else 'no'
        return {'ok': True, 'op': 'equal', 'args': args,
                'verdict': verdict, 'method': 'canonical'}
    except (NotInFragment, ZeroDivisionError):
        pass
    # canonical comparison over shared opaque atoms: equality is conclusive
    # (atoms match syntactically), inequality is NOT (distinct atoms may
    # still be related, e.g. sin^2 + cos^2), so only a 'yes' short-circuits
    try:
        store = _AtomStore()
        rf1 = _atomized_ratfunc(s1, n1, store)
        rf2 = _atomized_ratfunc(s2, n2, store)
        if rf1 == rf2:
            return {'ok': True, 'op': 'equal', 'args': args,
                    'verdict': 'yes', 'method': 'canonical (opaque atoms)'}
    except (NotInFragment, ZeroDivisionError, PrimitiveError):
        pass
    check = numeric_spot_check(expr1, expr2, samples=20, assumptions=assumed)

    def conditional(rec, method):
        """Every verdict the restricted sampling produced carries the
        restriction with it: a conditional yes must never be readable as
        an unconditional one."""
        if assumed:
            rec['method'] = f'{method} under the stated assumptions'
            rec['assumptions'] = list(assumed)
        else:
            rec['method'] = method
        return rec

    if check['status'] == 'agree':
        rec = conditional(
            {'ok': True, 'op': 'equal', 'args': args, 'verdict': 'yes',
             'samples': check['samples']},
            'numeric-oracle (probabilistic)')
        if check.get('undefined_points'):
            rec['note'] = ('compared only where both sides are defined; '
                           f"{check['undefined_points']} sample points fell "
                           'outside both domains')
        return rec
    if check['status'] == 'disagree':
        rec = conditional(
            {'ok': True, 'op': 'equal', 'args': args, 'verdict': 'no',
             'counterexample': check['point']}, 'numeric-oracle')
        if 'lhs' in check:
            # for constant inputs (e.g. literal matrices) the evaluated
            # values are the whole witness — the point alone is empty
            rec['lhs'] = check['lhs']
            rec['rhs'] = check['rhs']
        return rec
    if check['status'] == 'domain-differs':
        defined, undefined = (('expr1', 'expr2') if check['defined'] == 'lhs'
                              else ('expr2', 'expr1'))
        rec = conditional(
            {'ok': True, 'op': 'equal', 'args': args, 'verdict': 'no',
             'counterexample': check['point'],
             'reason': f'{defined} is defined at the counterexample point '
                       f'but {undefined} is not'},
            'numeric-oracle (domain mismatch)')
        if check.get('common_samples'):
            rec['note'] = (f"values agree at all {check['common_samples']} "
                           'sampled points where both sides are defined; '
                           'equality may hold on a restricted domain')
        if not assumed:
            rec['note'] = (rec.get('note', '') + ' Restrict the question '
                           'with `assuming` to ask again on that domain.'
                           ).strip()
        return rec
    return {'ok': True, 'op': 'equal', 'args': args, 'verdict': 'unknown',
            'method': 'none', 'reason': check.get('reason')}

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
primitives.py - agent-scoped verified-derivation primitives.

The agent decides strategy; each primitive executes one narrow, mechanically
checkable move and returns a JSON-able record:

    {'ok': True, 'op': 'expand', 'input': ..., 'result': ...,
     'assumptions': [...], 'check': {...}}

Primitives: substitute, apply_both_sides, expand, collect, evaluate,
differentiate, rewrite (lemma-based), and the checker equal_exprs (yes/no/
unknown). Every transforming primitive is spot-checked by an independent
numeric oracle (Schwartz-Zippel style random sampling).

Deliberately absent: solve, simplify, autonomous integrate, general factor.
"""
import math
import random
import hashlib
from fractions import Fraction

from notation import Notation, Symbol, Func
from LatexParser import MathParser
from LatexWriter import LaTexWriter
from value import Value, IntegerValue, FracValue, FloatValue
from replicator import Replicator
import comparer
from comparer import NotationParam
from polyrat import (Poly, RatFunc, NotInFragment, to_ratfunc,
                     ratfunc_to_notation, poly_to_notation,
                     _fraction_to_notation, _index_power,
                     FUNCTION_NAMES as FUNC_NAMES)

FRAC_NAMES = ('\\frac', '\\dfrac', '\\tfrac', '\\cfrac')

CONSTANT_NAMES = {'e': math.e, '\\pi': math.pi}


class PrimitiveError(Exception):
    pass


class EvalError(Exception):
    pass


# ---------------------------------------------------------------------------
# parse / write helpers
# ---------------------------------------------------------------------------

def parse_latex(latex):
    notation = Notation()
    try:
        sym = MathParser(notation).parse(latex)
    except Exception as e:
        raise PrimitiveError(f'cannot parse {latex!r}: {e}')
    if sym is None:
        raise PrimitiveError(f'cannot parse {latex!r}')
    return sym, notation


def write_latex(sym, notation):
    out = LaTexWriter(notation)(sym)
    # the writer occasionally doubles spaces; normalize for stable output
    return ' '.join(out.split())


def canonical_or_same(latex):
    """Reprint through polyrat if the expression is in the fragment."""
    try:
        sym, notation = parse_latex(latex)
        rf = to_ratfunc(sym, notation)
        out_n = Notation()
        return write_latex(ratfunc_to_notation(rf, out_n), out_n)
    except (PrimitiveError, NotInFragment, ZeroDivisionError):
        return latex


# ---------------------------------------------------------------------------
# free symbols
# ---------------------------------------------------------------------------

def free_symbols(sym, notation):
    res = set()

    def visit(s):
        if s is None or isinstance(s, Value):
            return
        if isinstance(s, (list, tuple)):
            for t in s:
                visit(t)
            return
        if isinstance(s, Symbol):
            f = notation.get(s)
            if f is None:
                name = s.name
                if (name in FUNC_NAMES or name in CONSTANT_NAMES
                        or name in Notation.styles):
                    return
                res.add(name)
                return
            visit(f.args)

    visit(sym)
    return res


# ---------------------------------------------------------------------------
# numeric oracle
# ---------------------------------------------------------------------------

_UNARY_TABLE = {
    '\\sin': math.sin, '\\cos': math.cos, '\\tan': math.tan,
    '\\sinh': math.sinh, '\\cosh': math.cosh, '\\tanh': math.tanh,
    '\\cot': lambda x: math.cos(x) / math.sin(x),
    '\\coth': lambda x: math.cosh(x) / math.sinh(x),
    '\\sec': lambda x: 1.0 / math.cos(x),
    '\\csc': lambda x: 1.0 / math.sin(x),
    '\\ln': math.log, '\\log': math.log10, '\\exp': math.exp,
    '\\arcsin': math.asin, '\\arccos': math.acos, '\\arctan': math.atan,
}


def _is_func_name(sym, notation):
    return (isinstance(sym, Symbol) and notation.get(sym) is None
            and sym.name in _UNARY_TABLE)


def numeric_eval(sym, notation, env):
    """Evaluate expression to a float. Raises EvalError outside the numeric
    fragment, ZeroDivisionError/ValueError on bad sample points."""
    if isinstance(sym, IntegerValue):
        return float(sym.val)
    if isinstance(sym, FracValue):
        if sym.denom == 0:
            raise EvalError('zero denominator literal')
        return sym.num / sym.denom
    if isinstance(sym, FloatValue):
        return float(sym.val)
    if not isinstance(sym, Symbol):
        raise EvalError(f'cannot evaluate {sym!r}')
    f = notation.get(sym)
    if f is None:
        if sym.name in env:
            return env[sym.name]
        if sym.name in CONSTANT_NAMES:
            return CONSTANT_NAMES[sym.name]
        raise EvalError(f'unbound symbol {sym.name}')
    op = f.sym
    if op in (Notation.GROUP, Notation.V_GROUP, Notation.S_GROUP,
              Notation.PLUS):
        return numeric_eval(f.args[0], notation, env)
    if op == Notation.MINUS:
        return -numeric_eval(f.args[0], notation, env)
    if op == Notation.S_LIST:
        return sum(numeric_eval(t, notation, env) for t in f.args)
    if op == Notation.P_LIST:
        return _eval_plist(f.args, notation, env)
    if op == Notation.SLASH:
        d = numeric_eval(f.args[1], notation, env)
        if d == 0:
            raise ZeroDivisionError
        return numeric_eval(f.args[0], notation, env) / d
    if op == Notation.STAR:
        return (numeric_eval(f.args[0], notation, env)
                * numeric_eval(f.args[1], notation, env))
    if op == Notation.INDEX:
        sub, sup_l, power, sup_r = f.args[1]
        if sub is not None or sup_l is not None or sup_r is not None:
            raise EvalError('subscripted symbol')
        base = f.args[0]
        if power is None:
            return numeric_eval(base, notation, env)
        b = numeric_eval(base, notation, env)
        p = numeric_eval(power, notation, env)
        if b == 0 and p < 0:
            raise ZeroDivisionError
        if b < 0 and p != int(p):
            raise ValueError('negative base, fractional power')
        return math.pow(b, p)
    if op == Notation.FUNC:
        fname, arg = f.args[0], f.args[1]
        if isinstance(fname, Symbol) and fname.name in _UNARY_TABLE:
            return _UNARY_TABLE[fname.name](numeric_eval(arg, notation, env))
        raise EvalError(f'unknown function {fname!r}')
    if op.name in FRAC_NAMES:
        d = numeric_eval(f.args[1], notation, env)
        if d == 0:
            raise ZeroDivisionError
        return numeric_eval(f.args[0], notation, env) / d
    if op.name == '\\sqrt':
        if len(f.args) == 1:
            v = numeric_eval(f.args[0], notation, env)
            if v < 0:
                raise ValueError('sqrt of negative sample')
            return math.sqrt(v)
        n = numeric_eval(f.args[1], notation, env)
        v = numeric_eval(f.args[0], notation, env)
        if v < 0:
            raise ValueError('root of negative sample')
        return math.pow(v, 1.0 / n)
    raise EvalError(f'cannot evaluate operation {op.name}')


def _func_power(sym, notation):
    """Detect \\sin^{n}-style factor: INDEX whose base is a function name.
    Returns (func_name, power_sym) or None."""
    f = notation.getf(sym, Notation.INDEX)
    if f is None:
        return None
    sub, sup_l, power, sup_r = f.args[1]
    if sub is not None or sup_l is not None or sup_r is not None:
        return None
    base = f.args[0]
    if (isinstance(base, Symbol) and notation.get(base) is None
            and base.name in _UNARY_TABLE):
        return base.name, power
    return None


def _is_group(sym, notation):
    return notation.vgetf(sym, [Notation.GROUP, Notation.V_GROUP,
                                Notation.S_GROUP]) is not None


def _func_arg_span(args, i, notation, is_head):
    """Factors bound as the argument of the function at args[i].
    A group immediately after the function is the entire argument
    (\\cos(x) y -> cos(x)*y); otherwise the run of tight factors up to the
    next function or group binds (\\sin 2x -> sin(2x)).
    Returns (arg_syms, next_index)."""
    j = i + 1
    if j < len(args) and _is_group(args[j], notation):
        return [args[j]], j + 1
    inner = []
    while (j < len(args) and not is_head(args[j])
           and not _is_group(args[j], notation)):
        inner.append(args[j])
        j += 1
    return inner, j


def _eval_plist(args, notation, env):
    result = 1.0
    i = 0
    args = [a for a in args if not (isinstance(a, Symbol)
                                    and a.name in Notation.styles)]

    def is_head(a):
        return (_is_func_name(a, notation)
                or _func_power(a, notation) is not None)

    while i < len(args):
        a = args[i]
        fname, power = (a.name, None) if _is_func_name(a, notation) else (
            _func_power(a, notation) or (None, None))
        if fname is not None:
            inner_syms, j = _func_arg_span(args, i, notation, is_head)
            if not inner_syms:
                raise EvalError(f'{fname} without argument')
            inner = 1.0
            for t in inner_syms:
                inner *= numeric_eval(t, notation, env)
            v = _UNARY_TABLE[fname](inner)
            if power is not None:
                v = math.pow(v, numeric_eval(power, notation, env))
            result *= v
            i = j
        else:
            result *= numeric_eval(a, notation, env)
            i += 1
    return result


def _sample_point(variables, rng):
    # rationals with small denominators, avoiding 0 (poles cluster there)
    env = {}
    for v in sorted(variables):
        num = rng.randint(-12, 12)
        if num == 0:
            num = 7
        den = rng.randint(1, 6)
        env[v] = num / den + rng.random() * 1e-3
    return env


def numeric_spot_check(latex1, latex2, assumptions=None, samples=12,
                       seed=20260705, tol=1e-6):
    """Independently check latex1 == latex2 at random sample points.
    Returns a dict {'status': 'agree'|'disagree'|'skipped', ...}.
    Respects recorded assumptions: sample points violating them are skipped."""
    try:
        s1, n1 = parse_latex(latex1)
        s2, n2 = parse_latex(latex2)
    except PrimitiveError as e:
        return {'status': 'skipped', 'reason': str(e)}
    variables = free_symbols(s1, n1) | free_symbols(s2, n2)
    guards = []
    for a in (assumptions or []):
        expr = a.get('nonzero')
        if expr:
            try:
                gs, gn = parse_latex(expr)
                guards.append((gs, gn))
            except PrimitiveError:
                pass
    rng = random.Random(seed)
    agreed = 0
    tried = 0
    while agreed < samples and tried < samples * 8:
        tried += 1
        env = _sample_point(variables, rng)
        try:
            if any(abs(numeric_eval(gs, gn, env)) < 1e-4 for gs, gn in guards):
                continue
            v1 = numeric_eval(s1, n1, env)
            v2 = numeric_eval(s2, n2, env)
        except (EvalError, ZeroDivisionError, ValueError, OverflowError):
            continue
        scale = max(1.0, abs(v1), abs(v2))
        if abs(v1 - v2) / scale > tol:
            return {'status': 'disagree', 'point': env,
                    'lhs': v1, 'rhs': v2}
        agreed += 1
    if agreed == 0:
        return {'status': 'skipped',
                'reason': 'no evaluable sample points'}
    return {'status': 'agree', 'samples': agreed}


# ---------------------------------------------------------------------------
# substitution machinery (also used by rewrite)
# ---------------------------------------------------------------------------

class Substitutor(Replicator):
    """Copy an expression replacing free symbols by other expressions."""

    def __init__(self, notation, output_notation, mapping):
        # mapping: {Symbol: (value_sym, value_notation)}
        super(Substitutor, self).__init__(notation, output_notation)
        self.mapping = mapping

    def _lookup(self, sym):
        entry = self.mapping.get(sym)
        if entry is None:
            return None
        value_sym, value_notation = entry
        copied = Replicator(value_notation, self.output_notation)(value_sym)
        if isinstance(copied, Symbol) and self.output_notation.get(copied) is None:
            return copied
        return self.output_notation.setf(Notation.GROUP, (copied,), br='()')

    def enter_symbol(self, sym):
        res = self._lookup(sym)
        return res if res is not None else sym

    def enter_raw_term(self, t):
        if isinstance(t, Symbol):
            res = self._lookup(t)
            if res is not None:
                return res
        return t


def _result(op, args, input_latex, result_latex, assumptions=None,
            check=None, extra=None):
    rec = {'ok': True, 'op': op, 'args': args, 'input': input_latex,
           'result': result_latex, 'assumptions': assumptions or []}
    if check is not None:
        rec['check'] = check
    if extra:
        rec.update(extra)
    return rec


def _error(op, args, message):
    return {'ok': False, 'op': op, 'args': args, 'error': message}


def _checked(rec, assumptions=None):
    rec['check'] = numeric_spot_check(rec['input'], rec['result'],
                                      assumptions=assumptions)
    return rec


# ---------------------------------------------------------------------------
# primitive: substitute
# ---------------------------------------------------------------------------

def substitute(expr, var, value):
    """Replace every free occurrence of `var` in `expr` by `value`."""
    args = {'expr': expr, 'var': var, 'value': value}
    try:
        sym, notation = parse_latex(expr)
        vsym, vnotation = parse_latex(value)
    except PrimitiveError as e:
        return _error('substitute', args, str(e))
    var_symbol = Symbol(var)
    if var not in free_symbols(sym, notation):
        return _error('substitute', args,
                      f'variable {var!r} does not occur in expression')
    out_n = Notation()
    out_s = Substitutor(notation, out_n, {var_symbol: (vsym, vnotation)})(sym)
    result = write_latex(out_s, out_n)
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
    except PrimitiveError as e:
        return {'status': 'skipped', 'reason': str(e)}
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
        scale = max(1.0, abs(v1), abs(v2))
        if abs(v1 - v2) / scale > 1e-6:
            return {'status': 'disagree', 'point': env, 'lhs': v1, 'rhs': v2}
        agreed += 1
    if agreed == 0:
        return {'status': 'skipped', 'reason': 'no evaluable sample points'}
    return {'status': 'agree', 'samples': agreed}


# ---------------------------------------------------------------------------
# primitive: apply_both_sides
# ---------------------------------------------------------------------------

_APPLY_OPS = ('+', '-', '*', '/', '^')


def _needs_parens_additive(sym, notation):
    if notation.vgetf(sym, [Notation.S_LIST, Notation.MINUS, Notation.PLUS]):
        return True
    if isinstance(sym, (IntegerValue, FracValue, FloatValue)):
        return False
    return False


def _needs_parens_factor(sym, notation):
    return notation.vgetf(sym, [Notation.S_LIST, Notation.MINUS,
                                Notation.PLUS]) is not None


def _paren(s):
    return '\\left(' + s + '\\right)'


def apply_both_sides(equation, op, arg):
    """Apply op ∈ {+,-,*,/,^} with argument `arg` to both sides of an
    equation. Division records the assumption arg ≠ 0."""
    args = {'equation': equation, 'op': op, 'arg': arg}
    if op not in _APPLY_OPS:
        return _error('apply_both_sides', args,
                      f'op must be one of {_APPLY_OPS}')
    try:
        sym, notation = parse_latex(equation)
        asym, anotation = parse_latex(arg)
    except PrimitiveError as e:
        return _error('apply_both_sides', args, str(e))
    comp = notation.getf(sym, Notation.COMP)
    if comp is None:
        return _error('apply_both_sides', args,
                      'expression is not an equation')
    if comp.sym.props.get('op') != '=':
        return _error('apply_both_sides', args,
                      "only '=' equations are supported for now")
    lhs, rhs = comp.args[0], comp.args[1]
    lhs_s = write_latex(lhs, notation)
    rhs_s = write_latex(rhs, notation)
    arg_s = write_latex(asym, anotation)

    assumptions = []
    # constant analysis of the argument
    arg_const = None
    try:
        rf = to_ratfunc(asym, anotation)
        if rf.is_const():
            arg_const = rf.const_value()
    except (NotInFragment, ZeroDivisionError):
        pass

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
            return _error('apply_both_sides', args,
                          'multiplying both sides by 0 destroys the equation')
        new_lhs = multiplicative(lhs, lhs_s)
        new_rhs = multiplicative(rhs, rhs_s)
    elif op == '/':
        if arg_const == 0:
            return _error('apply_both_sides', args, 'division by zero')
        if arg_const is None:
            assumptions.append({'text': f'{arg_s} \\ne 0', 'nonzero': arg_s})
        new_lhs = f'\\frac{{{lhs_s}}}{{{arg_s}}}'
        new_rhs = f'\\frac{{{rhs_s}}}{{{arg_s}}}'
    else:  # '^'
        if arg_const is not None and arg_const < 0:
            assumptions.append({'text': f'{lhs_s} \\ne 0', 'nonzero': lhs_s})
            assumptions.append({'text': f'{rhs_s} \\ne 0', 'nonzero': rhs_s})
        if arg_const is not None and arg_const != int(arg_const) or arg_const is None:
            assumptions.append(
                {'text': f'both sides must be in the domain of x^{{{arg_s}}}'})
        new_lhs = f'{_paren(lhs_s)}^{{{arg_s}}}'
        new_rhs = f'{_paren(rhs_s)}^{{{arg_s}}}'

    result = f'{new_lhs} = {new_rhs}'
    try:
        parse_latex(result)
    except PrimitiveError as e:
        return _error('apply_both_sides', args,
                      f'internal: built unparseable result: {e}')
    rec = _result('apply_both_sides', args, equation, result,
                  assumptions=assumptions)
    # oracle: each new side must equal op(old side, arg) at sample points
    c1 = numeric_spot_check(new_lhs, _op_expr(lhs_s, op, arg_s),
                            assumptions=assumptions)
    c2 = numeric_spot_check(new_rhs, _op_expr(rhs_s, op, arg_s),
                            assumptions=assumptions)
    rec['check'] = _merge_checks(c1, c2)
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
    if c1['status'] == 'disagree':
        return c1
    if c2['status'] == 'disagree':
        return c2
    if c1['status'] == 'agree' and c2['status'] == 'agree':
        return {'status': 'agree',
                'samples': min(c1['samples'], c2['samples'])}
    return {'status': 'skipped',
            'reason': c1.get('reason') or c2.get('reason') or 'partial'}


# ---------------------------------------------------------------------------
# primitives: expand / collect / evaluate  (polyrat-powered)
# ---------------------------------------------------------------------------

def _comp_split(sym, notation):
    """Return (lhs, rhs) for an '=' equation, None for plain expressions."""
    comp = notation.getf(sym, Notation.COMP)
    if comp is None:
        return None
    if comp.sym.props.get('op') != '=':
        raise PrimitiveError("only '=' equations are supported")
    return comp.args[0], comp.args[1]


def _canonical_side(side, notation):
    rf = to_ratfunc(side, notation)
    out_n = Notation()
    return write_latex(ratfunc_to_notation(rf, out_n), out_n)


def expand(expr):
    """Distribute products and integer powers of sums; canonical order.
    On an equation, expands each side."""
    args = {'expr': expr}
    try:
        sym, notation = parse_latex(expr)
        split = _comp_split(sym, notation)
        if split:
            lhs, rhs = split
            new_l = _canonical_side(lhs, notation)
            new_r = _canonical_side(rhs, notation)
            rec = _result('expand', args, expr, f'{new_l} = {new_r}')
            rec['check'] = _merge_checks(
                numeric_spot_check(write_latex(lhs, notation), new_l),
                numeric_spot_check(write_latex(rhs, notation), new_r))
            return rec
        rf = to_ratfunc(sym, notation)
    except PrimitiveError as e:
        return _error('expand', args, str(e))
    except ZeroDivisionError:
        return _error('expand', args, 'expression contains division by zero')
    except NotInFragment as e:
        return _error('expand', args,
                      f'outside the rational fragment: {e}')
    out_n = Notation()
    result = write_latex(ratfunc_to_notation(rf, out_n), out_n)
    return _checked(_result('expand', args, expr, result))


def collect(expr, var):
    """Group a polynomial by powers of `var` (descending). On an equation,
    collects each side."""
    args = {'expr': expr, 'var': var}
    try:
        sym, notation = parse_latex(expr)
        split = _comp_split(sym, notation)
        if split:
            lhs, rhs = split
            sides = []
            for side in split:
                rf = to_ratfunc(side, notation)
                if not rf.is_poly():
                    return _error('collect', args,
                                  'collect currently supports polynomials '
                                  'only')
                if var in rf.num.variables():
                    sides.append(_collect_poly(rf.num, var))
                else:
                    out_n = Notation()
                    sides.append(
                        write_latex(poly_to_notation(rf.num, out_n), out_n))
            rec = _result('collect', args, expr,
                          f'{sides[0]} = {sides[1]}')
            rec['check'] = _merge_checks(
                numeric_spot_check(write_latex(lhs, notation), sides[0]),
                numeric_spot_check(write_latex(rhs, notation), sides[1]))
            return rec
        rf = to_ratfunc(sym, notation)
    except PrimitiveError as e:
        return _error('collect', args, str(e))
    except ZeroDivisionError:
        return _error('collect', args, 'expression contains division by zero')
    except NotInFragment as e:
        return _error('collect', args, f'outside the rational fragment: {e}')
    if not rf.is_poly():
        return _error('collect', args,
                      'collect currently supports polynomials only')
    poly = rf.num
    if var not in poly.variables():
        return _error('collect', args,
                      f'variable {var!r} does not occur in expression')
    result = _collect_poly(poly, var)
    try:
        parse_latex(result)
    except PrimitiveError as e:
        return _error('collect', args, f'internal: unparseable result: {e}')
    return _checked(_result('collect', args, expr, result))


def _collect_poly(poly, var):
    # bucket terms by exponent of var
    buckets = {}
    for mono, coeff in poly.terms.items():
        d = dict(mono)
        k = d.pop(var, 0)
        rest = tuple(sorted(d.items()))
        buckets.setdefault(k, {})[rest] = coeff
    parts = []
    for k in sorted(buckets, reverse=True):
        coeff_poly = Poly(buckets[k])
        out_n = Notation()
        coeff_s = write_latex(poly_to_notation(coeff_poly, out_n), out_n)
        if k == 0:
            part = _paren(coeff_s) if _is_sum_str(coeff_s) else coeff_s
        else:
            var_s = var if k == 1 else f'{var}^{{{k}}}'
            if coeff_s == '1':
                part = var_s
            elif coeff_s == '-1':
                part = f'-{var_s}'
            elif _is_sum_str(coeff_s):
                part = f'{_paren(coeff_s)}{var_s}'
            else:
                part = f'{coeff_s}{var_s}'
        parts.append(part)
    result = parts[0]
    for p in parts[1:]:
        result += p if p.startswith('-') else ' + ' + p
    return result


def _is_sum_str(s):
    """True if the printed polynomial has more than one term."""
    depth = 0
    for i, ch in enumerate(s):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
        elif depth == 0 and ch in '+-' and i > 0:
            return True
    return False


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
        sides = []
        for side in split:
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
        rec = _result('evaluate', args, expr, f'{out[0]} = {out[1]}',
                      extra={'exact': True, 'holds': sides[0] == sides[1]})
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
    return _result('evaluate', args, expr, repr(round(v, 12)),
                   extra={'exact': False})


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
    if op == Notation.GROUP:
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
    try:
        parse_latex(result)
    except PrimitiveError as e:
        return _error('differentiate', args,
                      f'internal: unparseable derivative: {e}')
    rec = _result('differentiate', args, expr, result,
                  extra={'method': 'rules'})
    rec['check'] = _derivative_check(expr, result, var)
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
            env_p = dict(env)
            env_m = dict(env)
            env_p[var] = env[var] + h
            env_m[var] = env[var] - h
            d_num = (numeric_eval(s1, n1, env_p)
                     - numeric_eval(s1, n1, env_m)) / (2 * h)
        except (EvalError, ZeroDivisionError, ValueError, OverflowError):
            continue
        scale = max(1.0, abs(d_sym), abs(d_num))
        if abs(d_sym - d_num) / scale > 1e-4:
            return {'status': 'disagree', 'point': env,
                    'symbolic': d_sym, 'numeric': d_num}
        agreed += 1
    if agreed == 0:
        return {'status': 'skipped', 'reason': 'no evaluable sample points'}
    return {'status': 'agree', 'samples': agreed,
            'method': 'central-difference'}


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


def rewrite(expr, lemma_name, direction='forward'):
    """Apply a registered equality lemma at the root of the expression."""
    args = {'expr': expr, 'lemma': lemma_name, 'direction': direction}
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
    pat = comparer.pattern(src, [(p, NotationParam.Any) for p in lemma.params])
    subst = pat.match(sym, notation)
    if subst is None:
        return _error('rewrite', args,
                      f'expression does not match pattern {src!r}')
    tsym, tnotation = parse_latex(dst)
    mapping = {}
    for p in lemma.params:
        if p not in subst:
            return _error('rewrite', args,
                          f'pattern variable {p!r} unbound')
        mapping[Symbol(p)] = (subst[p], notation)
    out_n = Notation()
    out_s = Substitutor(tnotation, out_n, mapping)(tsym)
    result = write_latex(out_s, out_n)
    rec = _result('rewrite', args, expr, result)
    return _checked(rec)


# ---------------------------------------------------------------------------
# checker: equal?
# ---------------------------------------------------------------------------

def equal_exprs(expr1, expr2):
    """yes / no / unknown. Canonical forms decide the rational fragment;
    the numeric oracle answers probabilistically outside it. Equations are
    compared side by side."""
    args = {'expr1': expr1, 'expr2': expr2}
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
        verdicts = []
        for a, b in zip(sp1, sp2):
            sub = equal_exprs(write_latex(a, n1), write_latex(b, n2))
            if not sub.get('ok'):
                return sub
            verdicts.append(sub['verdict'])
        if all(v == 'yes' for v in verdicts):
            verdict = 'yes'
        elif 'no' in verdicts:
            verdict = 'no'
        else:
            verdict = 'unknown'
        return {'ok': True, 'op': 'equal', 'args': args, 'verdict': verdict,
                'method': 'per-side'}
    try:
        rf1 = to_ratfunc(s1, n1)
        rf2 = to_ratfunc(s2, n2)
        verdict = 'yes' if rf1 == rf2 else 'no'
        return {'ok': True, 'op': 'equal', 'args': args,
                'verdict': verdict, 'method': 'canonical'}
    except (NotInFragment, ZeroDivisionError):
        pass
    check = numeric_spot_check(expr1, expr2, samples=20)
    if check['status'] == 'agree':
        return {'ok': True, 'op': 'equal', 'args': args, 'verdict': 'yes',
                'method': 'numeric-oracle (probabilistic)',
                'samples': check['samples']}
    if check['status'] == 'disagree':
        return {'ok': True, 'op': 'equal', 'args': args, 'verdict': 'no',
                'method': 'numeric-oracle', 'counterexample': check['point']}
    return {'ok': True, 'op': 'equal', 'args': args, 'verdict': 'unknown',
            'method': 'none', 'reason': check.get('reason')}

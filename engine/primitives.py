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
import re
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
    # the lexer has no bare < / > tokens, only the \lt / \gt commands
    normalized = re.sub(r'(?<!\\left)<', ' \\\\lt ', latex)
    normalized = re.sub(r'(?<!\\right)>', ' \\\\gt ', normalized)
    notation = Notation()
    try:
        sym = MathParser(notation).parse(normalized)
    except Exception as e:
        raise PrimitiveError(f'cannot parse {latex!r}: {e}')
    if sym is None:
        raise PrimitiveError(f'cannot parse {latex!r}')
    return sym, notation


class PrettyWriter(LaTexWriter):
    """LaTexWriter that drops the repr braces of integer values outside
    INDEX dimensions: {2}x -> 2x, while x^{12} keeps its braces."""

    def __init__(self, notation, **kwargs):
        super(PrettyWriter, self).__init__(notation, **kwargs)
        self._keep_braces = 0

    def write_index_item(self, sym):
        self._keep_braces += 1
        try:
            super(PrettyWriter, self).write_index_item(sym)
        finally:
            self._keep_braces -= 1

    def write_raw_term(self, t):
        if self._keep_braces == 0 and isinstance(t, IntegerValue):
            self.writeString(str(abs(t.val)))
        else:
            super(PrettyWriter, self).write_raw_term(t)


def _write_std(sym, notation):
    # the writer occasionally doubles spaces; normalize for stable output
    return ' '.join(LaTexWriter(notation)(sym).split())


class _GroupStripper(Replicator):
    """Copy a graph dropping transparent {}-groups, for normal-form
    comparison only (the result may not print as valid LaTeX)."""

    def enter_group(self, sym, f):
        if f.props.get('br') == '{}' and 'quoted' not in f.props:
            return self.enter_formula(f.args[0])
        return super(_GroupStripper, self).enter_group(sym, f)


def _normal_form(latex):
    """Parse and print with {}-groups stripped: two strings with equal
    normal forms parse to the same expression."""
    sym, notation = parse_latex(latex)
    out = Notation()
    return _write_std(_GroupStripper(notation, out)(sym), out)


def write_latex(sym, notation):
    """Readable LaTeX with a safety net: the pretty form is used only if it
    parses back to the same normal form as the standard output."""
    std = _write_std(sym, notation)
    try:
        pretty = ' '.join(PrettyWriter(notation)(sym).split())
    except Exception:
        return std
    if pretty == std:
        return std
    try:
        if _normal_form(pretty) == _normal_form(std):
            return pretty
    except PrimitiveError:
        pass
    return std


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
        sp1 = _comp_split(s1, n1)
        sp2 = _comp_split(s2, n2)
    except PrimitiveError as e:
        return {'status': 'skipped', 'reason': str(e)}
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


def _paren(s):
    s = s.strip()
    if _fully_wrapped(s):
        return s
    return '\\left(' + s + '\\right)'


def _fully_wrapped(s):
    """True if s is one balanced (...) / \\left(...\\right) around the
    whole string, so another wrap adds nothing."""
    if not ((s.startswith('(') or s.startswith('\\left('))
            and s.endswith(')')):
        return False
    depth = 0
    for idx, ch in enumerate(s):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0 and idx < len(s) - 1:
                return False
    return depth == 0


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
                      'expression is not an equation or inequality')
    rel = comp.sym.props.get('op')
    if rel not in _SUPPORTED_REL:
        return _error('apply_both_sides', args,
                      f'unsupported relation {rel!r}')
    is_ineq = rel in _FLIP_REL
    out_rel = rel
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
                          'multiplying both sides by 0 destroys the relation')
        if is_ineq:
            if arg_const is None:
                return _error(
                    'apply_both_sides', args,
                    'cannot multiply an inequality by an expression of '
                    'unknown sign; use a constant or split into cases')
            if arg_const < 0:
                out_rel = _FLIP_REL[rel]
        elif arg_const is None:
            # if the factor can vanish, the step may introduce solutions
            assumptions.append({'text': f'{arg_s} \\ne 0', 'nonzero': arg_s})
        new_lhs = multiplicative(lhs, lhs_s)
        new_rhs = multiplicative(rhs, rhs_s)
    elif op == '/':
        if arg_const == 0:
            return _error('apply_both_sides', args, 'division by zero')
        if is_ineq:
            if arg_const is None:
                return _error(
                    'apply_both_sides', args,
                    'cannot divide an inequality by an expression of '
                    'unknown sign; use a constant or split into cases')
            if arg_const < 0:
                out_rel = _FLIP_REL[rel]
        elif arg_const is None:
            assumptions.append({'text': f'{arg_s} \\ne 0', 'nonzero': arg_s})
        new_lhs = f'\\frac{{{lhs_s}}}{{{arg_s}}}'
        new_rhs = f'\\frac{{{rhs_s}}}{{{arg_s}}}'
    else:  # '^'
        if rel != '=':
            return _error('apply_both_sides', args,
                          "op '^' is only supported for '=' relations")
        if arg_const is not None and arg_const < 0:
            assumptions.append({'text': f'{lhs_s} \\ne 0', 'nonzero': lhs_s})
            assumptions.append({'text': f'{rhs_s} \\ne 0', 'nonzero': rhs_s})
        if arg_const is not None and arg_const != int(arg_const) or arg_const is None:
            assumptions.append(
                {'text': f'both sides must be in the domain of x^{{{arg_s}}}'})
        new_lhs = f'{_paren(lhs_s)}^{{{arg_s}}}'
        new_rhs = f'{_paren(rhs_s)}^{{{arg_s}}}'

    result = f'{new_lhs} {out_rel} {new_rhs}'
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
            lhs, rhs, rel = split
            new_l = _canonical_side(lhs, notation)
            new_r = _canonical_side(rhs, notation)
            rec = _result('expand', args, expr, f'{new_l} {rel} {new_r}')
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
            lhs, rhs, rel = split
            sides = []
            for side in (lhs, rhs):
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
                          f'{sides[0]} {rel} {sides[1]}')
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
        # rational function: collect numerator and denominator separately
        if (var not in rf.num.variables()
                and var not in rf.den.variables()):
            return _error('collect', args,
                          f'variable {var!r} does not occur in expression')

        def side_str(p):
            if var in p.variables():
                return _collect_poly(p, var)
            out_n = Notation()
            return write_latex(poly_to_notation(p, out_n), out_n)

        result = f'\\frac{{{side_str(rf.num)}}}{{{side_str(rf.den)}}}'
        try:
            parse_latex(result)
        except PrimitiveError as e:
            return _error('collect', args,
                          f'internal: unparseable result: {e}')
        return _checked(_result('collect', args, expr, result))
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
    target = sym if subst is not None else None
    matches = 1 if subst is not None else 0
    if subst is None:
        # search subterms in parse order (children precede parents, so the
        # first hit is an innermost match); rewrite the first, count the rest
        for node in notation.rel:
            s = pat.match(node, notation)
            if s is not None:
                matches += 1
                if target is None:
                    target, subst = node, s
    if subst is None:
        return _error('rewrite', args,
                      f'expression does not match pattern {src!r} '
                      '(at the root or any subterm)')
    tsym, tnotation = parse_latex(dst)
    mapping = {}
    for p in lemma.params:
        if p not in subst:
            return _error('rewrite', args,
                          f'pattern variable {p!r} unbound')
        mapping[Symbol(p)] = (subst[p], notation)
    extra = {}
    if target is sym:
        out_n = Notation()
        out_s = Substitutor(tnotation, out_n, mapping)(tsym)
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
        result = write_latex(sym, work)
        extra = {'at': at_s, 'matches': matches}
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


def _as_poly(expr, op, args):
    """Parse expr and convert to a Poly, or return an error record."""
    sym, notation = parse_latex(expr)
    rf = to_ratfunc(sym, notation)
    if not rf.is_poly():
        raise PrimitiveError(f'{op} supports polynomials only')
    return rf.num


def factor_gcd(expr):
    """Pull out the common numeric/monomial factor: 6x^2+9x -> 3x(2x+3)."""
    args = {'expr': expr}
    try:
        poly = _as_poly(expr, 'factor_gcd', args)
    except PrimitiveError as e:
        return _error('factor_gcd', args, str(e))
    except ZeroDivisionError:
        return _error('factor_gcd', args, 'division by zero')
    except NotInFragment as e:
        return _error('factor_gcd', args,
                      f'outside the rational fragment: {e}')
    if poly.is_zero():
        return _error('factor_gcd', args, 'zero polynomial')
    content = poly.content()
    if poly.leading_coeff() < 0:
        content = -content
    mono = poly.monomial_gcd()
    if content == 1 and mono == ():
        return _error('factor_gcd', args, 'no common factor to pull out')
    quotient = poly.divide_monomial(mono).scale(1 / content)
    if quotient.is_const():
        return _error('factor_gcd', args,
                      'expression is a single term; nothing to factor')
    out_n = Notation()
    factor_s = write_latex(poly_to_notation(Poly({mono: content}), out_n),
                           out_n)
    out_n2 = Notation()
    quot_s = write_latex(poly_to_notation(quotient, out_n2), out_n2)
    result = f'{factor_s}{_paren(quot_s)}'
    try:
        parse_latex(result)
    except PrimitiveError as e:
        return _error('factor_gcd', args,
                      f'internal: unparseable result: {e}')
    return _checked(_result('factor_gcd', args, expr, result))


def factor_quadratic(expr, var):
    """Factor a quadratic in `var` with rational roots:
    x^2-5x+6 -> (x-2)(x-3); perfect squares -> (x-r)^2."""
    args = {'expr': expr, 'var': var}
    try:
        poly = _as_poly(expr, 'factor_quadratic', args)
    except PrimitiveError as e:
        return _error('factor_quadratic', args, str(e))
    except ZeroDivisionError:
        return _error('factor_quadratic', args, 'division by zero')
    except NotInFragment as e:
        return _error('factor_quadratic', args,
                      f'outside the rational fragment: {e}')
    if poly.degree(var) != 2:
        return _error('factor_quadratic', args,
                      f'expression is not quadratic in {var!r}')
    coeffs = {0: Fraction(0), 1: Fraction(0), 2: Fraction(0)}
    for mono, coeff in poly.terms.items():
        d = dict(mono)
        k = d.pop(var, 0)
        if d:
            return _error('factor_quadratic', args,
                          'coefficients must be constants for now')
        coeffs[k] = coeff
    a, b, c = coeffs[2], coeffs[1], coeffs[0]
    disc = b * b - 4 * a * c
    if disc < 0:
        return _error('factor_quadratic', args,
                      'negative discriminant; no real factorization')
    s = _fraction_sqrt(disc)
    if s is None:
        return _error('factor_quadratic', args,
                      'roots are not rational; not factorable over Q '
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
    try:
        parse_latex(result)
    except PrimitiveError as e:
        return _error('factor_quadratic', args,
                      f'internal: unparseable result: {e}')
    rec = _result('factor_quadratic', args, expr, result,
                  extra={'roots': [str(r1), str(r2)]})
    return _checked(rec)


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


def _strip_integral(sym, notation, var):
    """If sym is `\\int <integrand> [\\,] d<var>`, return the integrand sym;
    None if it is not an integral; raises on malformed/mismatched ones."""
    f = notation.getf(sym, Notation.P_LIST)
    if f is None:
        return None
    args = list(f.args)
    if not args:
        return None
    head = args[0]
    if notation.getf(head, Notation.INDEX) is not None:
        base = notation.getf(head, Notation.INDEX).args[0]
        if isinstance(base, Symbol) and base.name == '\\int':
            raise PrimitiveError('definite integrals are not supported yet')
    if not (isinstance(head, Symbol) and head.name == '\\int'):
        return None
    tail = args[1:]
    if len(tail) < 3:
        raise PrimitiveError('malformed integral')
    if not (isinstance(tail[-1], Symbol) and tail[-1].name == var):
        raise PrimitiveError(f'integral is not with respect to {var!r}')
    if not (isinstance(tail[-2], Symbol) and tail[-2].name == 'd'):
        raise PrimitiveError('malformed integral (missing d' + var + ')')
    body = tail[:-2]
    while body and isinstance(body[-1], Symbol) \
            and body[-1].name in Notation.styles:
        body.pop()
    if not body:
        raise PrimitiveError('empty integrand')
    if len(body) == 1:
        return body[0]
    return notation.setf(Notation.P_LIST, body)


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


def _table_integrate(sym, notation, var, assumptions):
    """Mechanical antiderivative (no constant): power rule + logarithm +
    basic functions of the bare variable, closed under sums and constant
    factors. Raises PrimitiveError with an honest reason otherwise."""
    try:
        rf = to_ratfunc(sym, notation)
        return _power_integrate_ratfunc(rf, var, assumptions, allow_log=True)
    except NotInFragment:
        pass
    except ZeroDivisionError:
        raise PrimitiveError('integrand contains division by zero')
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
    if op == Notation.INDEX:
        sub, sup_l, power, sup_r = f.args[1]
        base = f.args[0]
        if (isinstance(base, Symbol) and base.name == 'e'
                and power is not None and sub is None and sup_l is None
                and sup_r is None and _is_bare_var(power, notation, var)):
            return f'e^{{{var}}}'
        raise PrimitiveError(
            'no table rule for this power; integrate_power_rule handles '
            'rational powers, or use integrate_by_parts')
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


def integrate_power_rule(expr, var):
    """Term-by-term power rule for polynomials and rational expressions
    with a constant or single-power denominator. Refuses the exponent -1
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
        if sp1[2] != sp2[2]:
            return {'ok': True, 'op': 'equal', 'args': args, 'verdict': 'no',
                    'method': 'structural',
                    'reason': f'different relations {sp1[2]!r} vs {sp2[2]!r}'}
        verdicts = []
        for a, b in zip(sp1[:2], sp2[:2]):
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

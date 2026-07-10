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
from itertools import combinations

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

_MATRIX_ENV_RE = re.compile(
    r'\\begin\{(pmatrix|matrix)\}'
    r'((?:(?!\\begin\{(?:pmatrix|matrix)\}|\\end\{(?:pmatrix|matrix)\}).)*?)'
    r'\\end\{\1\}', re.DOTALL)


def _normalize_matrix_envs(latex):
    """LaTeX matrix environments -> the grammar's plain-TeX commands:
    \\begin{pmatrix} a & b \\\\ c & d \\end{pmatrix} -> \\pmatrix{a & b \\cr c & d}.
    Innermost-first so nested matrices normalize too."""
    def repl(m):
        body = m.group(2).replace('\\\\', ' \\cr ')
        return f'\\{m.group(1)}{{{body}}}'
    prev = None
    while prev != latex:
        prev = latex
        latex = _MATRIX_ENV_RE.sub(repl, latex)
    return latex


def parse_latex(latex):
    # the lexer has no bare < / > tokens, only the \lt / \gt commands
    normalized = _normalize_matrix_envs(latex)
    normalized = re.sub(r'(?<!\\left)<', ' \\\\lt ', normalized)
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
        # reparsing wraps INDEX dims in a {}-group; when it holds a bare
        # integer the value's own repr braces suffice: x^{12}, not x^{{12}}
        item = None
        if isinstance(sym, Symbol):
            g = self.notation.getf(sym, Notation.GROUP)
            if g is not None and g.props.get('br') == '{}' \
                    and 'quoted' not in g.props \
                    and isinstance(g.args[0], IntegerValue) \
                    and g.args[0].val >= 0:
                item = g.args[0]
        self._keep_braces += 1
        try:
            if item is not None:
                self.write_raw_term(item)
            else:
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
    comparison only (the result may not print as valid LaTeX).
    With all_brackets=True, ()-groups and \\left...\\right groups are
    stripped too (used for atom identity, where any grouping is noise)."""

    def __init__(self, notation, output_notation, all_brackets=False):
        super(_GroupStripper, self).__init__(notation, output_notation)
        self.all_brackets = all_brackets

    def enter_group(self, sym, f):
        transparent = f.props.get('br') == '{}' or (
            self.all_brackets and f.props.get('br') == '()')
        if transparent and 'quoted' not in f.props:
            return self.enter_formula(f.args[0])
        return super(_GroupStripper, self).enter_group(sym, f)

    def enter_vgroup(self, sym, f):
        if self.all_brackets and f.props.get('br') == '()':
            return self.enter_formula(f.args[0])
        return super(_GroupStripper, self).enter_vgroup(sym, f)


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


# ---------------------------------------------------------------------------
# matrix-aware oracle arithmetic. Matrices evaluate to lists of lists of
# floats; the helpers keep multiplication ORDERED, so the oracle can
# disprove AB = BA for literal matrices. This shares nothing with the
# symbolic path — it is the independent leg of the trust design.
# ---------------------------------------------------------------------------

_ARRAY_EVAL_NAMES = ('\\array', '\\pmatrix', '\\matrix')


def _num_shape(m):
    return (len(m), len(m[0]) if m else 0)


def _num_add(a, b):
    am, bm = isinstance(a, list), isinstance(b, list)
    if am and bm:
        if _num_shape(a) != _num_shape(b):
            raise EvalError('matrix shape mismatch in +')
        return [[x + y for x, y in zip(r1, r2)] for r1, r2 in zip(a, b)]
    if am or bm:
        raise EvalError('matrix + scalar')
    return a + b


def _num_neg(a):
    if isinstance(a, list):
        return [[-x for x in row] for row in a]
    return -a


def _num_mul(a, b):
    am, bm = isinstance(a, list), isinstance(b, list)
    if am and bm:
        ra, ca = _num_shape(a)
        rb, cb = _num_shape(b)
        if ca != rb:
            raise EvalError('matrix shape mismatch in *')
        return [[sum(a[i][k] * b[k][j] for k in range(ca))
                 for j in range(cb)] for i in range(ra)]
    if am:
        return [[x * b for x in row] for row in a]
    if bm:
        return [[a * x for x in row] for row in b]
    return a * b


def _num_pow(b, p):
    if isinstance(p, list):
        raise EvalError('matrix exponent')
    if isinstance(b, list):
        n = int(p)
        if p != n or n < 1:
            raise EvalError('matrix power must be a positive integer')
        out = b
        for _ in range(n - 1):
            out = _num_mul(out, b)
        return out
    if b == 0 and p < 0:
        raise ZeroDivisionError
    if b < 0 and p != int(p):
        raise ValueError('negative base, fractional power')
    return math.pow(b, p)


def _num_abs(v):
    if isinstance(v, list):
        return max((abs(x) for row in v for x in row), default=0.0)
    return abs(v)


def _num_agree(v1, v2, tol):
    """True/False when the values are comparable; None when a scalar meets
    a non-vanishing matrix (nothing to conclude)."""
    m1, m2 = isinstance(v1, list), isinstance(v2, list)
    if m1 and m2:
        if _num_shape(v1) != _num_shape(v2):
            return False
        scale = max(1.0, _num_abs(v1), _num_abs(v2))
        return all(abs(x - y) / scale <= tol
                   for r1, r2 in zip(v1, v2) for x, y in zip(r1, r2))
    if m1 or m2:
        if _num_abs(v1) <= tol and _num_abs(v2) <= tol:
            return True   # a fully cancelled matrix prints as scalar 0
        return None
    scale = max(1.0, abs(v1), abs(v2))
    return abs(v1 - v2) / scale <= tol


def numeric_eval(sym, notation, env):
    """Evaluate expression to a float, or to a list-of-lists of floats for
    matrix literals. Raises EvalError outside the numeric fragment,
    ZeroDivisionError/ValueError on bad sample points."""
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
        if f.props.get('br') == '||':
            # absolute value bars — the oracle computes real |·|, sharing
            # nothing with the symbolic atom path.
            v = numeric_eval(f.args[0], notation, env)
            if isinstance(v, list):
                raise EvalError('absolute value of a matrix')
            return abs(v)
        return numeric_eval(f.args[0], notation, env)
    if op == Notation.MINUS:
        return _num_neg(numeric_eval(f.args[0], notation, env))
    if op == Notation.S_LIST:
        total = None
        for t in f.args:
            v = numeric_eval(t, notation, env)
            total = v if total is None else _num_add(total, v)
        return total
    if op == Notation.P_LIST:
        return _eval_plist(f.args, notation, env)
    if op == Notation.SLASH:
        d = numeric_eval(f.args[1], notation, env)
        if isinstance(d, list):
            raise EvalError('division by a matrix')
        if d == 0:
            raise ZeroDivisionError
        return _num_mul(numeric_eval(f.args[0], notation, env), 1.0 / d)
    if op == Notation.STAR:
        return _num_mul(numeric_eval(f.args[0], notation, env),
                        numeric_eval(f.args[1], notation, env))
    if op == Notation.INDEX:
        sub, sup_l, power, sup_r = f.args[1]
        if sub is not None or sup_l is not None or sup_r is not None:
            raise EvalError('subscripted symbol')
        base = f.args[0]
        if power is None:
            return numeric_eval(base, notation, env)
        b = numeric_eval(base, notation, env)
        p = numeric_eval(power, notation, env)
        return _num_pow(b, p)
    if op == Notation.FUNC:
        fname, arg = f.args[0], f.args[1]
        if isinstance(fname, Symbol) and fname.name in _UNARY_TABLE:
            v = numeric_eval(arg, notation, env)
            if isinstance(v, list):
                raise EvalError('matrix argument to a function')
            return _UNARY_TABLE[fname.name](v)
        raise EvalError(f'unknown function {fname!r}')
    if op.name in FRAC_NAMES:
        d = numeric_eval(f.args[1], notation, env)
        if isinstance(d, list):
            raise EvalError('division by a matrix')
        if d == 0:
            raise ZeroDivisionError
        return _num_mul(numeric_eval(f.args[0], notation, env), 1.0 / d)
    if op.name == '\\sqrt':
        if len(f.args) == 1:
            v = numeric_eval(f.args[0], notation, env)
            if isinstance(v, list):
                raise EvalError('sqrt of a matrix')
            if v < 0:
                raise ValueError('sqrt of negative sample')
            return math.sqrt(v)
        n = numeric_eval(f.args[1], notation, env)
        v = numeric_eval(f.args[0], notation, env)
        if isinstance(v, list) or isinstance(n, list):
            raise EvalError('root of a matrix')
        if v < 0:
            raise ValueError('root of negative sample')
        return math.pow(v, 1.0 / n)
    if op.name in _ARRAY_EVAL_NAMES:
        rows = []
        width = None
        for row in f.args:
            vals = [numeric_eval(c, notation, env) for c in row]
            if any(isinstance(v, list) for v in vals):
                raise EvalError('nested matrix literal')
            if width is None:
                width = len(vals)
            elif len(vals) != width:
                raise EvalError('ragged matrix literal')
            rows.append(vals)
        if not rows or width == 0:
            raise EvalError('empty matrix literal')
        return rows
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
                inner = _num_mul(inner, numeric_eval(t, notation, env))
            if isinstance(inner, list):
                raise EvalError('matrix argument to a function')
            v = _UNARY_TABLE[fname](inner)
            if power is not None:
                v = math.pow(v, numeric_eval(power, notation, env))
            result = _num_mul(result, v)
            i = j
        else:
            # _num_mul keeps left-to-right order: matrix factors must
            # multiply in the order they appear
            result = _num_mul(result, numeric_eval(a, notation, env))
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


def _eval_kind(sym, notation, env):
    """numeric_eval classified: (None, value) on success, ('domain', None)
    when the point lies outside the expression's domain (log/root of a
    negative sample, a pole), ('oracle', None) when the oracle cannot
    evaluate the expression at all and the point proves nothing."""
    try:
        return None, numeric_eval(sym, notation, env)
    except (ValueError, ZeroDivisionError):
        return 'domain', None
    except (EvalError, OverflowError):
        return 'oracle', None


def numeric_spot_check(latex1, latex2, assumptions=None, samples=12,
                       seed=20260705, tol=1e-6):
    """Independently check latex1 == latex2 at random sample points.
    Returns {'status': 'agree'|'disagree'|'domain-differs'|'skipped', ...}.
    A point where exactly one side is defined is a definedness witness:
    the sides differ as real functions even if values agree elsewhere
    ('domain-differs'). Points outside both domains are skipped but
    counted ('undefined_points'), so an 'agree' restricted to a common
    domain says so. Respects recorded assumptions: sample points
    violating them are skipped."""
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
    undefined_both = 0
    mismatches = 0
    mismatch = None
    while agreed < samples and tried < samples * 8:
        tried += 1
        env = _sample_point(variables, rng)
        try:
            if any(_num_abs(numeric_eval(gs, gn, env)) < 1e-4
                   for gs, gn in guards):
                continue
        except (EvalError, ZeroDivisionError, ValueError, OverflowError):
            continue
        k1, v1 = _eval_kind(s1, n1, env)
        k2, v2 = _eval_kind(s2, n2, env)
        if 'oracle' in (k1, k2):
            continue
        if k1 == 'domain' and k2 == 'domain':
            undefined_both += 1
            continue
        if k1 == 'domain' or k2 == 'domain':
            mismatches += 1
            if mismatch is None:
                mismatch = {'point': env,
                            'defined': 'rhs' if k1 == 'domain' else 'lhs'}
            continue
        agree = _num_agree(v1, v2, tol)
        if agree is None:
            continue
        if not agree:
            return {'status': 'disagree', 'point': env,
                    'lhs': v1, 'rhs': v2}
        agreed += 1
    if mismatch is not None:
        return {'status': 'domain-differs', 'mismatches': mismatches,
                'common_samples': agreed, **mismatch}
    if agreed == 0:
        return {'status': 'skipped',
                'reason': 'no evaluable sample points'}
    result = {'status': 'agree', 'samples': agreed}
    if undefined_both:
        result['undefined_points'] = undefined_both
    return result


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
        return merged
    return {'status': 'skipped',
            'reason': c1.get('reason') or c2.get('reason') or 'partial'}


# ---------------------------------------------------------------------------
# primitives: expand / collect / evaluate  (polyrat-powered)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# opaque atoms: canonicalize expressions outside the rational fragment by
# treating maximal non-fragment subtrees (\cos x, e^x, unevaluated \int ...)
# as opaque variables, running the SAME polyrat engine, and substituting the
# subtrees back. No new rewrite rules - just the trusted canonical core.
# ---------------------------------------------------------------------------

class _AtomStore(object):
    def __init__(self):
        self.by_key = {}   # normal-form key -> atom name
        self.exprs = {}    # atom name -> (sym, notation)

    def atom(self, sym, notation):
        latex = _write_std(sym, notation)
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
            self.exprs[name] = (sym, notation)
        return Symbol(name)

    def mapping(self):
        return {Symbol(name): se for name, se in self.exprs.items()}


_NON_EXPR_OPS = (Notation.COMP, Notation.C_LIST, Notation.O_LIST,
                 Notation.A_LIST)

# matrix/vector objects: array literals and \vec-marked symbols. \cases is
# excluded on purpose (piecewise scalar). Bare symbols read as scalars until
# a declaration mechanism exists.
_MATRIX_FUNCS = frozenset(('\\array', '\\pmatrix', '\\matrix', '\\vec'))


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
        if f.props.get('br') == '||':
            # |expr| is absolute value, not grouping: the whole bar term is
            # one opaque atom (identity keeps the bars, so |x| != x).
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
            if isinstance(a, Symbol) and notation.get(a) is None \
                    and a.name == '\\int':
                # unevaluated integral: the rest of the product is one unit
                # (taken from the unfiltered factors, so \, survives)
                raw = list(f.args)
                tail = raw[raw.index(a):]
                span = notation.setf(Notation.P_LIST, tail) \
                    if len(tail) > 1 else a
                units.append(('int', span))
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
            if kind == 'int':
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
    return Replicator(notation, out_n)(sym)


def _relax_plist(f, notation, out_n):
    args = list(f.args)
    new_args = []
    for idx, a in enumerate(args):
        last = idx == len(args) - 1
        prev_head = idx > 0 and _plist_head_kind(args[idx - 1],
                                                 notation) is not None
        payload = _paren_payload(a, notation)
        if payload is None:
            pw = _powered_func_payload(a, notation)
            if pw is not None and last and not prev_head:
                # trailing ( \sin x)^{2} factor prints as \sin^{2}x
                new_args.extend(_emit_powered_func(pw, notation, out_n))
                continue
            new_args.append(Replicator(notation, out_n)(a))
            continue
        if _is_signed_or_sum(payload, notation):
            # collected coefficient sums keep their parens; relax inside
            inner = _relax_walk(payload, notation, out_n)
            new_args.append(out_n.setf(Notation.GROUP, (inner,), br='()'))
            continue
        pf = notation.getf(payload, Notation.P_LIST)
        if last and not prev_head:
            if pf is not None \
                    and _plist_head_kind(pf.args[0], notation) is not None:
                # function-application span: splice its factors in
                new_args.extend(Replicator(notation, out_n)(m)
                                for m in pf.args)
                continue
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
    except NotInFragment:
        # canonicalize over opaque atoms (\cos x, e^x, unevaluated \int)
        try:
            result, n_atoms = _atomized_canonical(sym, notation)
        except NotInFragment as e:
            return _error('expand', args,
                          f'outside the rational fragment: {e}')
        except ZeroDivisionError:
            return _error('expand', args,
                          'expression contains division by zero')
        return _checked(_result('expand', args, expr, result,
                                extra={'opaque_atoms': n_atoms}))
    out_n = Notation()
    result = write_latex(ratfunc_to_notation(rf, out_n), out_n)
    return _checked(_result('expand', args, expr, result))


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
    if isinstance(v, list):
        return _error('evaluate', args,
                      'matrix-valued expression: evaluate returns scalars; '
                      'expand collects matrix terms instead')
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
        if f.props.get('br') == '||':
            raise PrimitiveError(
                'cannot differentiate absolute value (not in the rule set)')
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
            f_p = numeric_eval(s1, n1, env_p)
            f_m = numeric_eval(s1, n1, env_m)
        except (EvalError, ZeroDivisionError, ValueError, OverflowError):
            continue
        if any(isinstance(v, list) for v in (d_sym, f_p, f_m)):
            continue   # matrix-valued: central differences not supported
        d_num = (f_p - f_m) / (2 * h)
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

    def find_match(pat_src, pat_params, powermap):
        """(target, subst, numeric_binds, matches) for one pattern; the
        root is preferred, then subterms in parse order (children precede
        parents, so the first hit is an innermost match). Wildcards from
        powermap must bind perfect n-th power monomials, whose roots are
        returned bound to the original lemma parameter."""
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

        s = pat.match(sym, notation)
        if s is not None:
            v = validate(s)
            if v is not None:
                return sym, s, v, 1
        target, best_s, best_v, matches = None, None, None, 0
        for node in notation.rel:
            s = pat.match(node, notation)
            if s is None:
                continue
            v = validate(s)
            if v is None:
                continue
            matches += 1
            if target is None:
                target, best_s, best_v = node, s, v
        return target, best_s, best_v, matches

    target, subst, numeric, matches = find_match(src, lemma.params, {})
    if subst is None:
        # numeric fallback: a^n pattern terms may bind perfect n-th power
        # monomials (x^2 - 4 matches diff_squares with b := 2)
        for v_src, v_params, v_map in _lemma_power_variants(src,
                                                            lemma.params):
            target, subst, numeric, matches = find_match(v_src, v_params,
                                                         v_map)
            if subst is not None:
                break
    if subst is None:
        return _error('rewrite', args,
                      f'expression does not match pattern {src!r} '
                      '(at the root or any subterm)')
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
    if op == Notation.SLASH or op.name in FRAC_NAMES:
        num, den = f.args[0], f.args[1]
        if var in free_symbols(den, notation):
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
    body = _paren(new_integrand) if _is_sum_str(new_integrand) \
        else new_integrand
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
    check = numeric_spot_check(expr1, expr2, samples=20)
    if check['status'] == 'agree':
        rec = {'ok': True, 'op': 'equal', 'args': args, 'verdict': 'yes',
               'method': 'numeric-oracle (probabilistic)',
               'samples': check['samples']}
        if check.get('undefined_points'):
            rec['note'] = ('compared only where both sides are defined; '
                           f"{check['undefined_points']} sample points fell "
                           'outside both domains')
        return rec
    if check['status'] == 'disagree':
        rec = {'ok': True, 'op': 'equal', 'args': args, 'verdict': 'no',
               'method': 'numeric-oracle', 'counterexample': check['point']}
        if 'lhs' in check:
            # for constant inputs (e.g. literal matrices) the evaluated
            # values are the whole witness — the point alone is empty
            rec['lhs'] = check['lhs']
            rec['rhs'] = check['rhs']
        return rec
    if check['status'] == 'domain-differs':
        defined, undefined = (('expr1', 'expr2') if check['defined'] == 'lhs'
                              else ('expr2', 'expr1'))
        rec = {'ok': True, 'op': 'equal', 'args': args, 'verdict': 'no',
               'method': 'numeric-oracle (domain mismatch)',
               'counterexample': check['point'],
               'reason': f'{defined} is defined at the counterexample point '
                         f'but {undefined} is not'}
        if check.get('common_samples'):
            rec['note'] = (f"values agree at all {check['common_samples']} "
                           'sampled points where both sides are defined; '
                           'equality may hold on a restricted domain')
        return rec
    return {'ok': True, 'op': 'equal', 'args': args, 'verdict': 'unknown',
            'method': 'none', 'reason': check.get('reason')}

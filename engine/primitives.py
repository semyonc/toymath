#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
primitives.py - common infrastructure for verified-derivation tactics.

This module owns shared LaTeX parsing/writing, notation traversal, binder and
free-symbol analysis, result records, and the independent numeric oracle.
Subject tactic implementations live in the static engine.tactics modules and
are admitted only through tactic_registry.py.

Deliberately absent: solve, simplify, autonomous integrate, general factor.
"""
import math
import random
import re

from notation import Notation, Symbol, Func
from LatexParser import MathParser
from LatexWriter import LaTexWriter
from value import Value, IntegerValue, FracValue, FloatValue
from replicator import Replicator
from polyrat import (NotInFragment, to_ratfunc, ratfunc_to_notation,
                     FUNCTION_NAMES as FUNC_NAMES)

FRAC_NAMES = ('\\frac', '\\dfrac', '\\tfrac', '\\cfrac')

CONSTANT_NAMES = {'e': math.e, '\\pi': math.pi}

_FACTORIAL_FLOAT_CAP = 170
_BINOM_EVAL_CAP = 10000


class PrimitiveError(Exception):
    pass


class EvalError(Exception):
    pass


# Ellipsis commands parse as opaque symbols with no mechanical semantics
# ("continue the pattern" is a human reading), so every primitive rejects
# them at the door except the sum_from_ellipsis / prod_from_ellipsis
# doors, which interpret them.
_ELLIPSIS_NAMES = frozenset({
    '\\ldots', '\\cdots', '\\dots', '\\vdots', '\\ddots', '\\hdots',
    '\\dotsb', '\\dotsc', '\\dotsi', '\\dotsm', '\\dotso'})
_ELLIPSIS_RE = re.compile(r'\\(?:[lcvdh])?dots[bcimo]?(?![A-Za-z])')
_DISPLAY_STAR_RE = re.compile(r'(?<!\\)[ \t]*\*[ \t]*')


def parse_latex(latex, allow_ellipsis=False, command_names=None):
    # the lexer has no bare < / > tokens, only the \lt / \gt commands
    if not allow_ellipsis and _ELLIPSIS_RE.search(latex):
        raise PrimitiveError(
            'the ellipsis in the expression has no mechanical semantics; '
            'interpret it first with sum_from_ellipsis (terms joined by +) '
            'or prod_from_ellipsis (factors joined by \\cdot or '
            'juxtaposition) by proposing the explicit \\sum_{k=a}^{b} / '
            '\\prod_{k=a}^{b} form it abbreviates')
    normalized = re.sub(r'(?<!\\left)<', ' \\\\lt ', latex)
    normalized = re.sub(r'(?<!\\right)>', ' \\\\gt ', normalized)
    notation = Notation()
    try:
        sym = MathParser(notation, command_names=command_names).parse(normalized)
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
        self._index_depth = 0

    def _index_payload(self, sym):
        """Peel redundant transparent braces around one delimited item."""
        while isinstance(sym, Symbol):
            group = self.notation.getf(sym, Notation.GROUP)
            if group is None or group.props.get('br') != '{}' \
                    or 'quoted' in group.props:
                break
            sym = group.args[0]
        return sym

    def _write_braced_item(self, sym):
        """Write exactly one standard pair of braces around ``sym``."""
        item = self._index_payload(sym)
        self.writeString('{')
        if isinstance(item, Symbol) and self.notation.get(item) is not None:
            self.write_formula(item)
        else:
            self.write_scalar(item)
        self.writeString('}')

    def write_index_item(self, sym):
        # Every parse wraps an explicitly braced dimension in one more GROUP.
        # Peel the whole transparent chain, then restore exactly the delimiter
        # the grammar needs.  Bare symbolic dimensions remain e^x/0^- so a
        # presentation pass does not introduce a semantic GROUP into later
        # tactic input; integers keep conventional x^{12} spelling.
        item = self._index_payload(sym)
        if isinstance(item, IntegerValue) and item.val >= 0:
            self._write_braced_item(item)
            return
        self._index_depth += 1
        try:
            if isinstance(item, Symbol) and self.notation.get(item) is not None:
                self._write_braced_item(item)
            else:
                super(PrettyWriter, self).write_index_item(item)
        finally:
            self._index_depth -= 1

    def write_frac_item(self, sym):
        # Fraction arguments are their own delimiter context.  Reusing the
        # exact-one-brace rule keeps x^{\frac{1}{2}} both standard and stable.
        self._write_braced_item(sym)

    def write_binom_item(self, sym):
        self._write_braced_item(sym)

    def write_scalar(self, sym):
        # Inside an already-delimited composite index, `{2}x` is safely `2x`.
        # The normal-form validation in write_latex rejects the candidate if
        # removing braces would fuse adjacent numeric tokens.
        if self._index_depth:
            item = self._index_payload(sym)
            if isinstance(item, IntegerValue) and item.val >= 0:
                self.write_raw_term(item)
                return
        super(PrettyWriter, self).write_scalar(sym)

    def write_raw_term(self, t):
        if isinstance(t, IntegerValue) and t.val >= 0:
            self.writeString(str(t.val))
        else:
            super(PrettyWriter, self).write_raw_term(t)


def _write_std(sym, notation):
    # the writer occasionally doubles spaces; normalize for stable output
    return ' '.join(LaTexWriter(notation)(sym).split())


class _GroupStripper(Replicator):
    """Copy a graph dropping transparent {}-groups, for normal-form
    comparison only (the result may not print as valid LaTeX).
    With all_brackets=True, ()-groups and \\left...\\right groups are
    stripped too (used for atom identity, where any grouping is noise).
    Division spelling is canonicalized: \\frac-family nodes become SLASH,
    so `x^{\\frac 1 2}` and `x^{1/2}` share one normal form (agents
    normalize fraction spelling when they re-type an expression, exactly
    like arrow spelling)."""

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

    def enter_oper(self, sym, f):
        if f.sym.name in FRAC_NAMES and len(f.args) == 2:
            args = tuple(self.enter_expr(expr) for expr in f.args)
            return self.output_notation.repf(
                self.mapsym(sym), Func(Notation.SLASH, args))
        return super(_GroupStripper, self).enter_oper(sym, f)

    def enter_plist(self, sym, f):
        # explicit-\cdot marking is presentation only (the structural
        # comparer ignores it); the comparison normal form must too, or
        # linkage refuses honest respellings of one product
        if 'cdot' in f.props:
            args = self.build_list(f, self.enter_expr)
            props = {k: v for k, v in f.props.items() if k != 'cdot'}
            return self.output_notation.repf(
                self.mapsym(sym), Func(Notation.P_LIST, args, **props))
        return super(_GroupStripper, self).enter_plist(sym, f)


def _normal_form(latex, allow_ellipsis=False):
    """Parse and print with {}-groups stripped: two strings with equal
    normal forms parse to the same expression."""
    sym, notation = parse_latex(latex, allow_ellipsis=allow_ellipsis)
    out = Notation()
    return _write_std(_GroupStripper(notation, out)(sym), out)


def same_expression(latex1, latex2):
    """Structural identity modulo grouping and whitespace — no oracle, no
    algebra. Used for provenance linkage, where value-equality via the
    oracle would be too permissive and parse failures must count as
    different. Ellipsis expressions compare structurally too (the guard is
    about doing MATH on an ellipsis, not about comparing spellings)."""
    if latex1 == latex2:
        return True
    try:
        return (_normal_form(latex1, allow_ellipsis=True)
                == _normal_form(latex2, allow_ellipsis=True))
    except PrimitiveError:
        return False


def display_latex(latex):
    """Presentation-only cleanup for a recorded LaTeX string.

    Agents commonly use the keyboard product separator ``*``. Keep that
    exact spelling in tactic records for hashing/replay, but show it as the
    conventional ``\\cdot`` in MathJax-facing views. This is deliberately
    not a parse/write canonicalization: re-emitting ``2*3`` or ``\\sin x*y``
    can lose the explicit boundary the agent wrote.

    Both spellings must parse to the same notation shape in both directions.
    ``same_expression`` is intentionally unsuitable here because it preserves
    the P_LIST ``cdot`` presentation prop; the structural comparer ignores
    that prop without doing algebra or consulting the numeric oracle. On any
    parse or shape mismatch, retain the source verbatim.
    """
    if not isinstance(latex, str) or '*' not in latex:
        return latex
    candidate = _DISPLAY_STAR_RE.sub(r' \\cdot ', latex)
    if candidate == latex:
        return latex
    try:
        source_sym, source_notation = parse_latex(
            latex, allow_ellipsis=True)
        display_sym, display_notation = parse_latex(
            candidate, allow_ellipsis=True)
        from comparer import s_equal
        if (s_equal(source_sym, source_notation,
                    display_sym, display_notation)
                and s_equal(display_sym, display_notation,
                            source_sym, source_notation)):
            return candidate
    except (PrimitiveError, ValueError):
        pass
    return latex


def _operator_body_latex(latex):
    """The body of a top-level big-operator expression (``\\lim``, ``\\int``
    without its differential, ``\\sum``), or None."""
    parts = _integral_parts_latex(latex)
    if parts is not None:
        # the true integrand: the textbook differential-in-numerator form
        # must not leak its phantom `dx` into the body comparison
        return parts[1]
    try:
        sym, notation = parse_latex(latex, allow_ellipsis=True)
    except PrimitiveError:
        return None
    try:
        body, _var, _point, _direction = _strip_limit(sym, notation)
        return write_latex(_peel_groups(body, notation), notation)
    except PrimitiveError:
        pass
    inner = _peel_groups(sym, notation)
    f = notation.getf(inner, Notation.P_LIST)
    if f is None:
        return None
    items = [a for a in f.args if not (isinstance(a, Symbol)
                                       and notation.get(a) is None
                                       and a.name in Notation.styles)]
    if not items:
        return None
    info = _binder_info(items[0], notation, items[1:])
    if info is None or not info['body']:
        return None
    body = (info['body'][0] if len(info['body']) == 1 else
            notation.setf(Notation.P_LIST, tuple(info['body'])))
    return write_latex(_peel_groups(body, notation), notation)


def _integral_parts_latex(latex):
    """(var, integrand_latex) for a top-level indefinite ``\\int`` in
    either canonical (``\\int f \\, dx``) or textbook differential-in-
    numerator (``\\int \\frac{f \\, dx}{g}``) form; None when the
    expression is not such an integral. The variable is discovered from
    the differential itself, so the two spellings of one integral can be
    compared integrand-to-integrand."""
    try:
        sym, notation = parse_latex(latex, allow_ellipsis=True)
    except PrimitiveError:
        return None
    inner = _peel_groups(sym, notation)
    f = notation.getf(inner, Notation.P_LIST)
    if f is None:
        return None
    items = [a for a in f.args if not (isinstance(a, Symbol)
                                       and a.name in Notation.styles)]
    if not (items and isinstance(items[0], Symbol)
            and notation.get(items[0]) is None
            and items[0].name == '\\int'):
        return None
    tail = items[1:]
    var = None
    if (len(tail) >= 2 and isinstance(tail[-1], Symbol)
            and notation.get(tail[-1]) is None
            and isinstance(tail[-2], Symbol)
            and notation.get(tail[-2]) is None
            and tail[-2].name == 'd'):
        var = tail[-1].name
    elif len(tail) == 1:
        body = tail[0]
        g = notation.vgetf(body, [Notation.GROUP, Notation.V_GROUP])
        if g is not None and not Notation.is_semantic_bracket(g):
            body = g.args[0]
        fr = notation.get(body)
        if fr is not None and (fr.sym == Notation.SLASH
                               or fr.sym.name in FRAC_NAMES):
            matched, _rest, dname = _split_trailing_differential(
                fr.args[0], notation, None)
            if matched:
                var = dname
    if var is None:
        return None
    try:
        integrand = _strip_integral(inner, notation, var)
    except PrimitiveError:
        return None
    if integrand is None:
        return None
    # the emitter parenthesizes sum/negative integrands as pure `\int`
    # syntax protection; the integrand itself is the peeled body
    return var, write_latex(_peel_groups(integrand, notation), notation)


def covers_goal(input_latex, goal_latex):
    """Whether a step input structurally restates the goal: identical
    modulo grouping, one is a big-operator wrapper whose body is the
    other (agents legitimately wrap a bare integrand/body goal in its
    ``\\int``/``\\lim`` binder, or state the body of a wrapped goal), or
    the two are one integral integrand-to-integrand — both written as
    indefinite integrals in the same variable, or one written bare (the
    textbook ``\\int \\frac{dx}{g}``, the canonical
    ``\\int \\frac{1}{g} \\, dx``, and the bare integrand
    ``\\frac{1}{g}`` all restate one another).

    Integrand comparisons use the all-bracket-stripped discipline (`||`
    and other semantic brackets are preserved): the textbook spelling
    necessarily parenthesizes its denominator, and the `\\int` emitter
    parenthesizes sum/negative bodies as pure syntax protection."""
    if same_expression(input_latex, goal_latex):
        return True

    def stripped_eq(left, right):
        try:
            return (_all_bracket_normal_form(left)
                    == _all_bracket_normal_form(right))
        except PrimitiveError:
            return False

    in_parts = _integral_parts_latex(input_latex)
    goal_parts = _integral_parts_latex(goal_latex)
    if (in_parts is not None and goal_parts is not None
            and in_parts[0] == goal_parts[0]
            and stripped_eq(in_parts[1], goal_parts[1])):
        return True
    if (in_parts is not None and goal_parts is None
            and stripped_eq(in_parts[1], goal_latex)):
        return True
    if (goal_parts is not None and in_parts is None
            and stripped_eq(goal_parts[1], input_latex)):
        return True
    body = _operator_body_latex(input_latex)
    if body is not None and same_expression(body, goal_latex):
        return True
    goal_body = _operator_body_latex(goal_latex)
    if goal_body is not None and same_expression(input_latex, goal_body):
        return True
    return False


def _all_bracket_normal_form(latex):
    sym, notation = parse_latex(latex, allow_ellipsis=True)
    out = Notation()
    return _write_std(_GroupStripper(notation, out, all_brackets=True)(sym),
                      out)


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

def _transparent_inner(sym, notation):
    """Remove ordinary grouping around binder metadata."""
    while isinstance(sym, Symbol):
        f = notation.vgetf(sym, [Notation.GROUP, Notation.V_GROUP])
        if f is None or Notation.is_semantic_bracket(f):
            break
        sym = f.args[0]
    return sym


def _big_operator_name(sym, notation):
    """Return the p_oper name for a bare or INDEX-decorated operator."""
    if isinstance(sym, Symbol) and notation.get(sym) is None:
        return sym.name if sym.name in Notation.p_oper else None
    f = notation.getf(sym, Notation.INDEX)
    if f is None:
        return None
    base = _transparent_inner(f.args[0], notation)
    if (isinstance(base, Symbol) and notation.get(base) is None
            and base.name in Notation.p_oper):
        return base.name
    return None


def _plain_symbol_name(sym, notation):
    sym = _transparent_inner(sym, notation)
    if isinstance(sym, Symbol) and notation.get(sym) is None:
        return sym.name
    return None


def _approach_point(sym, notation):
    """Return ``(point, direction)`` for a limit endpoint.

    The parser represents ``a^+`` / ``a^-`` as INDEX(a, power='+/-').
    Ordinary powered endpoints (``a^2``) remain untouched.  Direction is
    ``right`` / ``left`` / ``two-sided``.
    """
    f = notation.getf(sym, Notation.INDEX)
    if f is None:
        return sym, 'two-sided'
    sub_l, sup_l, power, sub_r = f.args[1]
    if sub_l is not None or sup_l is not None or sub_r is not None:
        return sym, 'two-sided'
    if (isinstance(power, Symbol) and notation.get(power) is None
            and power.name in ('+', '-')):
        return f.args[0], 'right' if power.name == '+' else 'left'
    return sym, 'two-sided'


def _binder_info(head, notation, tail=()):
    """Describe one big-operator binder from its head and product tail.

    The parser represents ``\\lim_{x \\to a}``, ``\\sum_{k=0}^n`` and
    bounded integrals as an INDEX-headed P_LIST.  This helper is the single
    place that interprets those dimensions.  ``parameters`` are the free
    parts of the point/bounds, while ``body`` excludes a conventional
    trailing differential for integrals.
    """
    name = _big_operator_name(head, notation)
    if name is None:
        return None
    info = {'operator': name, 'bound': None, 'parameters': [],
            'body': list(tail)}
    f = notation.getf(head, Notation.INDEX)
    if f is not None:
        _sub_l, _sup_l, upper, lower = f.args[1]
        lower_inner = _transparent_inner(lower, notation)
        rel = notation.getf(lower_inner, Notation.COMP)
        if rel is not None and rel.sym.props.get('op') in ('=', '\\to'):
            info['bound'] = _plain_symbol_name(rel.args[0], notation)
            point = rel.args[1]
            if name == '\\lim':
                point, _direction = _approach_point(point, notation)
            info['parameters'].append(point)
        elif lower is not None:
            info['parameters'].append(lower)
        if upper is not None:
            info['parameters'].append(upper)

    if name in ('\\int', '\\intop', '\\iint', '\\iiint', '\\iiiint',
                '\\idotsint', '\\oint'):
        body = list(tail)
        # Standard parser shape: [... integrand ..., \, d, x].  Styling
        # tokens are harmless separators; only d + a plain final symbol
        # establish a binder.
        significant = [(i, a) for i, a in enumerate(body)
                       if not (isinstance(a, Symbol)
                               and notation.get(a) is None
                               and a.name in Notation.styles)]
        if len(significant) >= 2:
            (di, d), (_vi, var) = significant[-2:]
            if (_plain_symbol_name(d, notation) == 'd'
                    and _plain_symbol_name(var, notation) is not None):
                info['bound'] = _plain_symbol_name(var, notation)
                info['body'] = body[:di]
    return info


def _bound_symbols(sym, notation):
    """All variables bound by a big operator anywhere below ``sym``."""
    result = set()
    seen = set()

    def visit_product(items):
        for i, item in enumerate(items):
            info = _binder_info(item, notation, items[i + 1:])
            if info is None:
                visit(item)
                continue
            if info['bound'] is not None:
                result.add(info['bound'])
            for p in info['parameters']:
                visit(p)
            visit_product(info['body'])
            return

    def visit(s):
        if not isinstance(s, Symbol) or s in seen:
            return
        seen.add(s)
        f = notation.get(s)
        if f is None:
            return
        if f.sym == Notation.P_LIST:
            visit_product(list(f.args))
            return
        info = _binder_info(s, notation)
        if info is not None:
            if info['bound'] is not None:
                result.add(info['bound'])
            for p in info['parameters']:
                visit(p)
            return
        for a in f.args:
            if isinstance(a, (tuple, list)):
                for x in a:
                    visit(x)
            else:
                visit(a)

    visit(sym)
    return result


def _contains_free_infinity(sym, notation):
    """Whether ``\\infty`` occurs outside big-operator point/bounds."""
    seen = set()

    def visit(s, allowed=False):
        if s is None or isinstance(s, Value):
            return False
        if isinstance(s, (tuple, list)):
            return any(visit(x, allowed) for x in s)
        if not isinstance(s, Symbol):
            return False
        f = notation.get(s)
        if f is None:
            return s.name == '\\infty' and not allowed
        marker = (s, allowed)
        if marker in seen:
            return False
        seen.add(marker)
        if f.sym == Notation.INDEX and _big_operator_name(s, notation):
            # Infinity denotes an endpoint only inside the operator's
            # dimensions.  Its body is visited normally by the P_LIST.
            return visit(f.args[1], True)
        return visit(f.args, allowed)

    return visit(sym)

def _subscript_var(sym, notation):
    """`x_{1}`-style INDEX used as an atomic variable: a leaf, non-function
    base carrying a pure numeral right subscript (dims slot 3). Returns the
    canonical variable key ('x_{1}') or None. Such a name is INDEPENDENT of
    its base (x_{1} is not an occurrence of x); symbolic subscripts (x_i,
    a_{n+1}) stay outside and keep raising oracle ignorance. A power on the
    node (x_{1}^{2}, dims slot 2) is allowed - the key names the variable,
    the caller handles the power."""
    f = notation.getf(sym, Notation.INDEX)
    if f is None:
        return None
    sub_l, sup_l, _power, sub = f.args[1]
    if sub is None or sub_l is not None or sup_l is not None:
        return None
    base = f.args[0]
    if not (isinstance(base, Symbol) and notation.get(base) is None):
        return None
    name = base.name
    if (name in FUNC_NAMES or name in CONSTANT_NAMES
            or name in Notation.styles or name in Notation.p_oper):
        return None
    val = sub
    while isinstance(val, Symbol):
        g = notation.vgetf(val, [Notation.GROUP, Notation.V_GROUP])
        if g is None:
            return None
        val = g.args[0]
    if not isinstance(val, IntegerValue):
        return None
    return '%s_{%d}' % (name, val.val)


def free_symbols(sym, notation):
    res = set()

    def visit_product(items, bound):
        for i, item in enumerate(items):
            info = _binder_info(item, notation, items[i + 1:])
            if info is None:
                visit(item, bound)
                continue
            for p in info['parameters']:
                visit(p, bound)
            body_bound = bound | ({info['bound']}
                                  if info['bound'] is not None else set())
            visit_product(info['body'], body_bound)
            return

    def visit(s, bound=frozenset()):
        if s is None or isinstance(s, Value):
            return
        if isinstance(s, (list, tuple)):
            for t in s:
                visit(t, bound)
            return
        if isinstance(s, Symbol):
            f = notation.get(s)
            if f is None:
                name = s.name
                if (name in FUNC_NAMES or name in CONSTANT_NAMES
                        or name in Notation.styles or name in Notation.p_oper
                        or name == '\\infty' or name in bound):
                    return
                res.add(name)
                return
            if f.sym == Notation.P_LIST:
                visit_product(list(f.args), bound)
                return
            info = _binder_info(s, notation)
            if info is not None:
                for p in info['parameters']:
                    visit(p, bound)
                return
            key = _subscript_var(s, notation)
            if key is not None:
                if key not in bound:
                    res.add(key)
                visit(f.args[1][2], bound)  # power on subscripted variable
                return
            visit(f.args, bound)

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

_ARRAY_EVAL_NAMES = (
    '\\array', '\\pmatrix', '\\matrix', '\\bmatrix', '\\Bmatrix',
    '\\vmatrix', '\\Vmatrix', '\\smallmatrix',
)


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
        if Notation.is_semantic_bracket(f):
            # bracket operators — the oracle computes the real |·|, floor
            # and ceiling, sharing nothing with the symbolic atom path.
            br = f.props['br']
            v = numeric_eval(f.args[0], notation, env)
            if isinstance(v, list):
                raise EvalError(
                    f'{Notation.BRACKET_NAMES[br]} of a matrix')
            if br == Notation.ABS_BR:
                return abs(v)
            if br == Notation.FLOOR_BR:
                return float(math.floor(v))
            return float(math.ceil(v))
        return numeric_eval(f.args[0], notation, env)
    if op in (Notation.PAIR, Notation.COLLECTION):
        raise EvalError(f'{op.name} is a typed result, not a scalar value')
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
    if op == Notation.FACTORIAL:
        v = numeric_eval(f.args[0], notation, env)
        if isinstance(v, list):
            raise EvalError('factorial of a matrix')
        if not math.isfinite(v) or v < 0 or abs(v - round(v)) > 1e-9:
            raise ValueError('factorial requires a nonnegative integer')
        n = int(round(v))
        if n > _FACTORIAL_FLOAT_CAP:
            raise OverflowError('factorial is too large for the oracle')
        return float(math.factorial(n))
    if op == Notation.BINOM:
        n = numeric_eval(f.args[0], notation, env)
        k = numeric_eval(f.args[1], notation, env)
        if isinstance(n, list) or isinstance(k, list):
            raise EvalError('binomial coefficient of a matrix')
        if (not math.isfinite(n) or not math.isfinite(k)
                or n < 0 or k < 0
                or abs(n - round(n)) > 1e-9
                or abs(k - round(k)) > 1e-9
                or round(k) > round(n)):
            raise ValueError(
                'binomial coefficient requires integers 0 <= k <= n')
        n, k = int(round(n)), int(round(k))
        if n > _BINOM_EVAL_CAP:
            raise OverflowError(
                'binomial coefficient is too large for the oracle')
        return float(math.comb(n, k))
    if op == Notation.INDEX:
        sub, sup_l, power, sup_r = f.args[1]
        if sub is not None or sup_l is not None or sup_r is not None:
            key = _subscript_var(sym, notation)
            if key is not None and key in env:
                v = env[key]
                if power is None:
                    return v
                return _num_pow(v, numeric_eval(power, notation, env))
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
    # An ordered pair immediately after a function head is one complete
    # argument object (f(x,y)), even though PAIR is semantic and never
    # transparent to normal form.
    return notation.vgetf(sym, [Notation.GROUP, Notation.V_GROUP,
                                Notation.S_GROUP, Notation.PAIR]) is not None


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


_SUM_EVAL_CAP = 100000


def _eval_finite_bigop(info, notation, env, op):
    """The oracle's independent leg for finite ``\\sum``/``\\prod``: a
    literal accumulation loop over integer bound values, sharing nothing
    with the symbolic sum/product tactics."""
    word = 'sum' if op == '\\sum' else 'product'
    if info is None or info['bound'] is None or len(info['parameters']) != 2:
        raise EvalError(f'{word} without an explicit k=a..b binder')
    if not info['body']:
        raise EvalError(f'{word} without a {word} body')
    lo = numeric_eval(info['parameters'][0], notation, env)
    hi = numeric_eval(info['parameters'][1], notation, env)
    if isinstance(lo, list) or isinstance(hi, list):
        raise EvalError(f'matrix-valued {word} bound')
    if abs(lo - round(lo)) > 1e-9 or abs(hi - round(hi)) > 1e-9:
        raise EvalError(f'non-integer {word} bound')
    lo, hi = int(round(lo)), int(round(hi))
    if hi - lo + 1 > _SUM_EVAL_CAP:
        raise EvalError(f'{word} too long to evaluate')
    # empty-range conventions: sum -> 0, product -> 1
    total = 0.0 if op == '\\sum' else 1.0
    e = dict(env)
    for k in range(lo, hi + 1):
        e[info['bound']] = float(k)
        v = _eval_plist(info['body'], notation, e)
        if isinstance(v, list):
            raise EvalError(f'matrix-valued {word} term')
        total = total + v if op == '\\sum' else total * v
    return total


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
        big = _big_operator_name(a, notation)
        if big in ('\\sum', '\\prod'):
            # the operator binds every remaining factor as its body
            info = _binder_info(a, notation, args[i + 1:])
            return _num_mul(result,
                            _eval_finite_bigop(info, notation, env, big))
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

    def __call__(self, sym):
        targets = {s.name for s in self.mapping
                   if isinstance(s, Symbol) and self.notation.get(s) is None}
        bound = _bound_symbols(sym, self.notation)
        captured_targets = targets & bound
        if captured_targets:
            names = ', '.join(sorted(captured_targets))
            raise PrimitiveError(
                f'cannot substitute bound variable(s): {names}')
        captured_values = set()
        for value_sym, value_notation in self.mapping.values():
            captured_values |= free_symbols(value_sym, value_notation) & bound
        if captured_values:
            names = ', '.join(sorted(captured_values))
            raise PrimitiveError(
                f'substitution would capture bound variable(s): {names}')
        return super(Substitutor, self).__call__(sym)

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

    def enter_index(self, sym, f):
        # a subscripted variable (x_{1}) is an atomic name, independent of
        # its base: substituting x must not capture it (x := 2 would turn
        # x_{1} into 2_{1}). Copy base and subscript verbatim; a power on
        # the node (x_{1}^{x}) still substitutes normally.
        if _subscript_var(sym, self.notation) is not None:
            plain = Replicator(self.notation, self.output_notation)
            sub_l, sup_l, power, sub = f.args[1]
            dims = (None, None,
                    None if power is None else self.enter_scalar(power),
                    plain.enter_scalar(sub))
            return self.output_notation.repf(
                self.mapsym(sym),
                Func(f.sym, (plain.enter_scalar(f.args[0]), dims)))
        return super(Substitutor, self).enter_index(sym, f)



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


def _is_sum_str(s):
    """True if a printed expression has more than one top-level term."""
    depth = 0
    for i, ch in enumerate(s):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
        elif depth == 0 and ch in '+-' and i > 0:
            return True
    return False


def _fully_wrapped(s):
    """Whether one balanced (...) / \\left(...\\right) wraps all of s."""
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


def _paren(s):
    s = s.strip()
    if _fully_wrapped(s):
        return s
    return '\\left(' + s + '\\right)'




# ---------------------------------------------------------------------------
# shared operator-structure helpers
# ---------------------------------------------------------------------------

def _strip_limit(sym, notation):
    """Return the body and binder metadata of ``\\lim_{x \\to a} body``."""
    while True:
        g = notation.vgetf(sym, [Notation.GROUP, Notation.V_GROUP,
                                 Notation.S_GROUP])
        if g is None or Notation.is_semantic_bracket(g):
            break
        sym = g.args[0]
    f = notation.getf(sym, Notation.P_LIST)
    if f is None:
        s = notation.getf(sym, Notation.S_LIST)
        if s is not None and s.args:
            head = _peel_groups(s.args[0], notation)
            hp = notation.getf(head, Notation.P_LIST)
            first = hp.args[0] if hp is not None else head
            if _big_operator_name(first, notation) == '\\lim':
                raise PrimitiveError(
                    'a leading sign splits the limit body into a sum; '
                    'parenthesize the signed body, as in '
                    '\\lim_{x \\to 0} (-x)')
        raise PrimitiveError('expected a limit expression')
    raw = list(f.args)
    significant = [a for a in raw if not (isinstance(a, Symbol)
                                           and notation.get(a) is None
                                           and a.name in Notation.styles)]
    if len(significant) < 2:
        raise PrimitiveError('malformed limit (missing body)')
    head = significant[0]
    if _big_operator_name(head, notation) != '\\lim':
        raise PrimitiveError('expected a limit expression')
    ix = notation.getf(head, Notation.INDEX)
    if ix is None:
        raise PrimitiveError('malformed limit (missing approach binder)')
    sub_l, sup_l, upper, lower = ix.args[1]
    if sub_l is not None or sup_l is not None or upper is not None:
        raise PrimitiveError('unsupported decorations on limit operator')
    lower = _transparent_inner(lower, notation)
    rel = notation.getf(lower, Notation.COMP)
    if rel is None or rel.sym.props.get('op') != '\\to':
        raise PrimitiveError('limit binder must have the form x \\to a')
    var = _plain_symbol_name(rel.args[0], notation)
    if var is None:
        raise PrimitiveError('limit variable must be a plain symbol')
    point, direction = _approach_point(rel.args[1], notation)
    if var in free_symbols(point, notation):
        raise PrimitiveError('limit point must not contain the bound variable')
    body_items = significant[1:]
    body = (body_items[0] if len(body_items) == 1 else
            notation.setf(Notation.P_LIST, tuple(body_items)))
    return body, var, point, direction


def _infinity_sign(sym, notation):
    sym = _transparent_inner(sym, notation)
    neg = notation.getf(sym, Notation.MINUS)
    if neg is not None:
        inner = _transparent_inner(neg.args[0], notation)
        if (isinstance(inner, Symbol) and notation.get(inner) is None
                and inner.name == '\\infty'):
            return -1
    if (isinstance(sym, Symbol) and notation.get(sym) is None
            and sym.name == '\\infty'):
        return 1
    return None


def _limit_latex(var, point, direction, body):
    suffix = {'two-sided': '', 'left': '^{-}', 'right': '^{+}'}[direction]
    body = body.strip()
    if _is_sum_str(body) or body.startswith('-'):
        body = _paren(body)
    return f'\\lim_{{{var} \\to {point}{suffix}}} {body}'



def _peel_groups(sym, notation):
    """Remove ordinary grouping (including \\left[...\\right]) around a
    subexpression; absolute-value bars are not grouping."""
    while True:
        g = notation.vgetf(sym, [Notation.GROUP, Notation.V_GROUP,
                                 Notation.S_GROUP])
        if g is None or Notation.is_semantic_bracket(g):
            return sym
        sym = g.args[0]


def _int_literal(sym, notation):
    """The integer value of a literal like ``1`` / ``{-2}``, else None."""
    if isinstance(sym, Symbol):
        sym = _peel_groups(sym, notation)
    neg = False
    if isinstance(sym, Symbol):
        m = notation.getf(sym, Notation.MINUS)
        if m is not None:
            neg = True
            sym = m.args[0]
            if isinstance(sym, Symbol):
                sym = _peel_groups(sym, notation)
    if isinstance(sym, IntegerValue):
        return -sym.val if neg else sym.val
    return None



def _split_trailing_differential(num, notation, var):
    """If a fraction numerator ends with the differential `d<name>`
    (textbook form `\\int \\frac{f(x) dx}{g(x)}`), return
    (True, rest_sym_or_None, name); rest is None when the numerator is
    exactly the differential. (False, None, None) when there is no
    trailing differential. A genuine variable named `d` in last-but-one
    position is indistinguishable from the differential - the human
    reading wins."""
    g = notation.vgetf(num, [Notation.GROUP, Notation.V_GROUP])
    if g is not None and not Notation.is_semantic_bracket(g):
        num = g.args[0]
    f = notation.getf(num, Notation.P_LIST)
    if f is None:
        return False, None, None
    items = [a for a in f.args if not (isinstance(a, Symbol)
                                       and a.name in Notation.styles)]
    if len(items) < 2:
        return False, None, None
    dsym, dvar = items[-2], items[-1]
    if not (isinstance(dsym, Symbol) and dsym.name == 'd'
            and notation.get(dsym) is None):
        return False, None, None
    if not (isinstance(dvar, Symbol) and notation.get(dvar) is None):
        return False, None, None
    rest = items[:-2]
    if not rest:
        return True, None, dvar.name
    if len(rest) == 1:
        return True, rest[0], dvar.name
    return True, notation.setf(Notation.P_LIST, tuple(rest)), dvar.name


def _strip_integral(sym, notation, var):
    """If sym is `\\int <integrand> [\\,] d<var>` (or the textbook form
    `\\int \\frac{f(x) d<var>}{g(x)}`), return the integrand sym;
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
    core = [t for t in tail if not (isinstance(t, Symbol)
                                    and t.name in Notation.styles)]
    if len(core) == 1:
        # textbook form: the differential lives in the fraction numerator
        inner = core[0]
        g = notation.vgetf(inner, [Notation.GROUP, Notation.V_GROUP])
        if g is not None and not Notation.is_semantic_bracket(g):
            inner = g.args[0]
        fr = notation.get(inner)
        if fr is not None and (fr.sym == Notation.SLASH
                               or fr.sym.name in FRAC_NAMES):
            matched, rest, dname = _split_trailing_differential(
                fr.args[0], notation, var)
            if matched:
                if dname != var:
                    raise PrimitiveError(
                        f'integral is not with respect to {var!r}')
                new_num = IntegerValue(1) if rest is None else rest
                return notation.setf(fr.sym, (new_num, fr.args[1]))
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

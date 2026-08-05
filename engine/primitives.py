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
import threading
from contextlib import contextmanager

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
        # explicit-\cdot marking and spacing tokens (\, \; \quad ...) are
        # presentation only — every reader convention filters them, so the
        # comparison normal form must too, or linkage refuses honest
        # respellings of one product (live: a chain root spelled
        # `\int ... \, dt` could not cover its own goal spelled
        # `\int ... dt`, and the verified result was refused at admission)
        spans = self._explicit_func_spans(list(f.args))
        if spans is not None:
            props = {k: v for k, v in f.props.items() if k != 'cdot'}
            return self.output_notation.repf(
                self.mapsym(sym), Func(Notation.P_LIST, spans, **props))
        kept = [a for a in f.args
                if not (isinstance(a, Symbol) and a.name in Notation.styles)]
        if len(kept) != len(f.args) or 'cdot' in f.props:
            props = {k: v for k, v in f.props.items() if k != 'cdot'}
            if len(kept) == 1:
                # a product reduced to one factor is that factor, exactly
                # as enter_group unwraps a transparent group
                return self.enter_formula(kept[0])
            args = tuple(self.enter_expr(a) for a in (kept or f.args))
            return self.output_notation.repf(
                self.mapsym(sym), Func(Notation.P_LIST, args, **props))
        return super(_GroupStripper, self).enter_plist(sym, f)

    def _explicit_func_spans(self, args):
        """Rebuild a product with every function-argument span explicitly
        grouped, or None when the product applies no function.

        Function application is a READING CONVENTION over a flat P_LIST,
        not a DAG node: `\\cos(x) y` and `\\cos x y` differ only by the
        group, and stripping it made both print `\\cos xy` — one string
        for two expressions that evaluate to 0.449 and 0.990 (MEASURED).
        Every consumer of a normal form therefore read a false identity:
        `same_expression` (provenance linkage), `_chain_links`, the
        endpoint-limit provenance door, and ledger duplicate detection.
        Sum/product boundaries were never affected, because the writer
        braces a composite operand itself (`(a+b)c` prints `{a+b}c`) —
        which is why the hole was function application only.

        The span comes from the ORACLE's own `_func_arg_span`, so the
        boundary this re-encodes is by construction the one the numeric
        leg checks against (the `_PoweredHeadNormalizer` precedent).
        Emitting exactly one transparent group per span keeps the honest
        respellings equal (`\\cos x`, `\\cos{x}` and `\\cos(x)` all become
        `\\cos{x}`) while separating the capture (`\\cos{x}y` from
        `\\cos{xy}`)."""
        kept = [a for a in args
                if not (isinstance(a, Symbol) and a.name in Notation.styles)]

        def is_head(a):
            return (_is_func_name(a, self.notation)
                    or _func_power(a, self.notation) is not None)

        if not any(is_head(a) for a in kept):
            return None
        out = []
        i = 0
        while i < len(kept):
            a = kept[i]
            if not is_head(a):
                out.append(self.enter_expr(a))
                i += 1
                continue
            span, j = _func_arg_span(kept, i, self.notation, is_head)
            out.append(self.enter_expr(a))
            if span:
                inner = [self.enter_expr(s) for s in span]
                payload = inner[0] if len(inner) == 1 else \
                    self.output_notation.setf(Notation.P_LIST, tuple(inner))
                out.append(self.output_notation.setf(
                    Notation.GROUP, (payload,), br='{}'))
            i = j
        return tuple(out)


class _RedundantBracketPeeler(Replicator):
    """Drop a bracket wrapper that an enclosing ``{}`` already delimits.

    A ``{}`` group is exactly the parser's delimiter: whatever slot it
    fills — a `\\sqrt` argument, an INDEX dimension, a `\\frac` or
    `\\binom` argument — its braces bound the span on both sides. So a
    `()` or `\\left(...\\right)` wrapper immediately inside one cannot be
    doing any work, and `{(X)}` is `{X}` for every X. That is what makes
    this peel safe where a bare function argument is NOT: nothing can
    capture across the braces, so the rightward-capture rule that governs
    products (a following ordinary factor joins the argument span) has
    nothing to say here.

    Semantic brackets are preserved — `\\lfloor`, `\\lceil` and `|...|`
    are bracket OPERATORS, not grouping, and peeling one would drop the
    operator in both trust legs at once."""

    def enter_group(self, sym, f):
        if f.props.get('br') == '{}' and 'quoted' not in f.props:
            inner = f.args[0]
            while True:
                g = self.notation.vgetf(inner, [Notation.GROUP,
                                                Notation.V_GROUP])
                if (g is None or Notation.is_semantic_bracket(g)
                        or 'quoted' in g.props
                        or g.props.get('br') not in ('()', '{}')):
                    break
                inner = g.args[0]
            if inner is not f.args[0]:
                return self.output_notation.repf(
                    self.mapsym(sym),
                    Func(Notation.GROUP, (self.enter_formula(inner),),
                         **f.props))
        return super(_RedundantBracketPeeler, self).enter_group(sym, f)


def _peel_redundant_brackets(sym, notation):
    """(sym, notation) with delimiter-redundant bracket wrappers dropped.

    Falls back to the input unchanged unless the peeled graph is the same
    expression modulo bracket respelling — the all-bracket normal form is
    the right question here precisely because a redundant bracket is the
    only thing being removed, and that form now keeps function-argument
    spans apart."""
    try:
        out = Notation()
        peeled = _RedundantBracketPeeler(notation, out)(sym)
        if (_dag_normal_form(peeled, out, all_brackets=True)
                == _dag_normal_form(sym, notation, all_brackets=True)):
            return peeled, out
    except Exception:
        pass
    return sym, notation


def _normal_form(latex, allow_ellipsis=False):
    """Parse and print with {}-groups stripped: two strings with equal
    normal forms parse to the same expression."""
    sym, notation = parse_latex(latex, allow_ellipsis=allow_ellipsis)
    return _dag_normal_form(sym, notation)


def _dag_normal_form(sym, notation, all_brackets=False):
    """The normal form of a graph we already hold, with no parse hop.
    Lets a candidate spelling be compared against the expression it is
    meant to spell, rather than only against another spelling of it."""
    out = Notation()
    return _write_std(
        _GroupStripper(notation, out, all_brackets=all_brackets)(sym), out)


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


def definite_integral_parts(latex, var=None):
    """(var, integrand_latex, lower_latex, upper_latex) for a top-level
    definite integral ``\\int_a^b f \\, d<var>``, else None.  With
    ``var=None`` the variable is discovered from the differential itself,
    so the canonical and textbook spellings of one definite integral can
    be compared parts-to-parts (the same discipline
    `_integral_parts_latex` gives indefinite integrals).

    Reads the notation directly: the parser normalizes both bound orders
    onto the INDEX head's (power, sup_r) slots as (upper, lower).  The
    integrand is read by re-heading the product with a bare ``\\int`` and
    reusing the indefinite readers.  Shared structure-reading only — the
    FTC tactic's symbolic leg and the quadrature check leg both read
    bounds through here, but neither leg's *computation* is shared."""
    try:
        sym, notation = parse_latex(latex)
    except PrimitiveError:
        return None
    inner = _peel_groups(sym, notation)
    f = notation.getf(inner, Notation.P_LIST)
    if f is None:
        return None
    args = list(f.args)
    if not args:
        return None
    idx = notation.getf(args[0], Notation.INDEX)
    if idx is None:
        return None
    base = idx.args[0]
    if not (isinstance(base, Symbol) and notation.get(base) is None
            and base.name == '\\int'):
        return None
    sub, sup_l, power, sup_r = idx.args[1]
    upper, lower = power, sup_r
    if (upper is None or lower is None
            or sub is not None or sup_l is not None):
        return None
    bare = notation.setf(Notation.P_LIST,
                         (Symbol('\\int'),) + tuple(args[1:]))
    if var is None:
        parts = _integral_parts_latex(write_latex(bare, notation))
        if parts is None:
            return None
        var, integrand_latex = parts
    else:
        try:
            integrand = _strip_integral(bare, notation, var)
        except PrimitiveError:
            return None
        if integrand is None:
            return None
        integrand_latex = write_latex(_peel_groups(integrand, notation),
                                      notation)
    return (var, integrand_latex,
            write_latex(_peel_groups(lower, notation), notation),
            write_latex(_peel_groups(upper, notation), notation))


class _IntegralPlaceholderer(Replicator):
    """Copy a graph replacing each definite-integral term — the ``\\int``
    INDEX head through its differential — with a fresh placeholder
    symbol, collecting (name, var, integrand, lower, upper) specs.

    This walk decides only WHERE a term starts and ends inside a
    product; what the term MEANS is delegated to
    ``definite_integral_parts`` on the sliced spelling, so the reading
    can never disagree with the other structure readers.  A slice the
    reader refuses is left in place untouched (its evaluation will
    raise, which downstream treats as oracle ignorance)."""

    def __init__(self, notation, output_notation):
        super(_IntegralPlaceholderer, self).__init__(
            notation, output_notation)
        self.specs = []

    def _int_head(self, sym):
        idx = self.notation.getf(sym, Notation.INDEX)
        if idx is None:
            return False
        base = idx.args[0]
        return (isinstance(base, Symbol) and self.notation.get(base) is None
                and base.name == '\\int')

    def _slice_end(self, args, start):
        """One past the differential of the integral starting at
        ``start``: the first bare ``d`` + plain-symbol pair, else the end
        of the product (the frac-differential spelling)."""
        k = start + 1
        while k < len(args) - 1:
            if (_is_bare_d(args[k], self.notation)
                    and isinstance(args[k + 1], Symbol)
                    and self.notation.get(args[k + 1]) is None):
                return k + 2
            k += 1
        return len(args)

    def _placeholder(self, term):
        scratch = Notation()
        rep = Replicator(self.notation, scratch)
        mapped = tuple(rep(a) for a in term)
        sub = (mapped[0] if len(mapped) == 1
               else scratch.setf(Notation.P_LIST, mapped))
        parts = definite_integral_parts(write_latex(sub, scratch))
        if parts is None:
            return None
        name = f'zz#lim{len(self.specs) + 1}'
        self.specs.append((name,) + parts)
        return Symbol(name)

    def enter_plist(self, sym, f):
        args = list(f.args)
        out = []
        i = 0
        changed = False
        while i < len(args):
            if self._int_head(args[i]):
                end = self._slice_end(args, i)
                placeholder = self._placeholder(args[i:end])
                if placeholder is not None:
                    out.append(placeholder)
                    i = end
                    changed = True
                    continue
            out.append(self.enter_expr(args[i]))
            i += 1
        if not changed:
            return super(_IntegralPlaceholderer, self).enter_plist(sym, f)
        if len(out) == 1:
            return out[0]
        return self.output_notation.repf(
            self.mapsym(sym), Func(Notation.P_LIST, tuple(out), **f.props))


def _graded_quadrature(fs, fn, var, a, b, env, panels=64):
    """The signed value of ``\\int_a^b`` under ``env`` by composite
    Simpson over slabs graded geometrically from BOTH bounds all the
    way to the midpoint, so an integrand whose scale-length is the
    distance to a nearby singularity (``1/t^2`` on ``[x, 1]`` as
    ``x -> 0``) stays resolvable in every slab — uniform panels
    measurably cannot, and grading only an outer fringe of the span
    fails the same way once the bound moves inside it.  The two-grid
    Richardson residual is its own honesty bar: raises EvalError
    whenever it cannot vouch for the value (domain break,
    non-convergence), which approach-sampling treats as ignorance,
    never as evidence."""
    if a == b:
        return 0.0
    sign = 1.0
    if b < a:
        a, b = b, a
        sign = -1.0
    span = b - a
    rel = [0.5] + [10.0 ** (-k) for k in range(1, 7)]
    cuts = ([a] + [a + r * span for r in reversed(rel)]
            + [b - r * span for r in rel[1:]] + [b])
    total = 0.0
    err = 0.0
    with _overflow_saturation():
        for lo, hi in zip(cuts, cuts[1:]):
            coarse, _bad = _simpson_panels(fs, fn, var, lo, hi, env,
                                           panels)
            fine, _bad2 = _simpson_panels(fs, fn, var, lo, hi, env,
                                          panels * 2)
            if coarse is None or fine is None:
                raise EvalError(
                    'integrand is not evaluable on the interval')
            total += fine
            err += abs(fine - coarse) / 15.0
    if not math.isfinite(total):
        raise EvalError('the integral value overflows')
    if err > 1e-6 * max(1.0, abs(total)):
        raise EvalError('quadrature over the interval did not converge')
    return sign * total


def definite_integral_evaluator(sym, notation):
    """A callable ``env -> float`` for a limit body containing
    variable-bound definite integrals: each integral is computed by
    ``_graded_quadrature`` under the ambient environment (bounds
    evaluated per call, the bound variable lexically shadowed), then the
    surrounding expression is evaluated with the values in place.
    Returns None when the body contains no definite integral, so
    callers keep plain ``numeric_eval``.

    Deliberately LIMITS-ORACLE infrastructure: general ``equal?`` does
    not gain this evaluator — a definite integral's VALUE still closes
    only through `integrate_definite`/`integrate_improper`, and letting
    the sampling comparator decide integrals numerically would open a
    proposal route around that boundary."""
    scratch = Notation()
    walker = _IntegralPlaceholderer(notation, scratch)
    skeleton = walker(sym)
    if not walker.specs:
        return None
    parsed = []
    for name, ivar, integrand, lower, upper in walker.specs:
        fs, fn = parse_latex(integrand)
        ls, ln = parse_latex(lower)
        us, un = parse_latex(upper)
        parsed.append((name, ivar, fs, fn, ls, ln, us, un))

    def evaluate(env):
        e = dict(env)
        for name, ivar, fs, fn, ls, ln, us, un in parsed:
            a = numeric_eval(ls, ln, e)
            b = numeric_eval(us, un, e)
            if (isinstance(a, list) or isinstance(b, list)
                    or not (math.isfinite(a) and math.isfinite(b))):
                raise EvalError('integral bound is not evaluable')
            e[name] = _graded_quadrature(fs, fn, ivar, a, b, e)
        return numeric_eval(skeleton, scratch, e)

    return evaluate


def _is_bare_d(sym, notation):
    return (isinstance(sym, Symbol) and notation.get(sym) is None
            and sym.name == 'd')


def _differential_denominator_var(den, notation):
    """The <var> of a ``d<var>`` fraction denominator, or None."""
    f = notation.getf(_peel_groups(den, notation), Notation.P_LIST)
    if f is None:
        return None
    items = [a for a in f.args if not (isinstance(a, Symbol)
                                       and a.name in Notation.styles)]
    if len(items) != 2 or not _is_bare_d(items[0], notation):
        return None
    v = items[1]
    if not (isinstance(v, Symbol) and notation.get(v) is None):
        return None
    return v.name


def derivative_operator_parts(latex):
    """(var, operand_latex) when ``latex`` is a Leibniz-prefixed
    derivative — ``\\frac{d}{d x} <expr>`` or the differential-in-
    numerator form ``\\frac{d <expr>}{d x}`` — else None.

    A reading convention, not a DAG node: the parser sees an ordinary
    fraction whose ``d``s are one-letter symbols, exactly as an
    integral's trailing differential is an ordinary product tail.  As
    there, a genuine variable named ``d`` is indistinguishable from the
    operator and the human reading wins (see
    ``_split_trailing_differential``).  Only the derivative-taking
    boundary — the ``differentiate`` tactic and the diff! variable
    inference — consults this reader, so nothing else re-reads a
    fraction; both consumers share this one function so the two
    readings can never disagree."""
    try:
        sym, notation = parse_latex(latex)
    except PrimitiveError:
        return None
    inner = _peel_groups(sym, notation)
    items = None
    f = notation.getf(inner, Notation.P_LIST)
    if f is not None:
        items = [a for a in f.args if not (isinstance(a, Symbol)
                                           and a.name in Notation.styles)]
        if not items:
            return None
        head = items[0]
    else:
        head = inner
    fr = notation.get(head)
    if fr is None or len(fr.args) != 2 \
            or not (fr.sym == Notation.SLASH
                    or fr.sym.name in FRAC_NAMES):
        return None
    dvar = _differential_denominator_var(fr.args[1], notation)
    if dvar is None or dvar == 'd':
        return None
    num = _peel_groups(fr.args[0], notation)
    if _is_bare_d(num, notation):
        # prefix form: the operand is everything after the fraction
        if items is None or len(items) < 2:
            return None
        rest = items[1:]
    else:
        # differential-in-numerator: \frac{d <expr>}{d x}, nothing after
        if items is not None and len(items) > 1:
            return None
        nf = notation.getf(num, Notation.P_LIST)
        if nf is None:
            return None
        nitems = [a for a in nf.args if not (isinstance(a, Symbol)
                                             and a.name in Notation.styles)]
        if len(nitems) < 2 or not _is_bare_d(nitems[0], notation):
            return None
        rest = nitems[1:]
    operand = rest[0] if len(rest) == 1 else \
        notation.setf(Notation.P_LIST, tuple(rest))
    return dvar, write_latex(_peel_groups(operand, notation), notation)


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


def covers_goal(input_latex, goal_latex, establishes=False):
    """Whether a step input structurally restates the goal: identical
    modulo grouping, one is a big-operator wrapper whose body is the
    other (agents legitimately wrap a bare integrand/body goal in its
    ``\\int``/``\\lim`` binder, or state the body of a wrapped goal), or
    the two are one integral integrand-to-integrand — both written as
    indefinite integrals in the same variable, or one written bare (the
    textbook ``\\int \\frac{dx}{g}``, the canonical
    ``\\int \\frac{1}{g} \\, dx``, and the bare integrand
    ``\\frac{1}{g}`` all restate one another).

    ``establishes=True`` asks the stricter admission question: does a
    chain rooted at this input *establish* the goal's value?  A bare body
    never establishes a value-bearing binder — the binder's own data
    (integration bounds, a limit's approach point, summation bounds) was
    then consumed by no checked step, so a chain rooted at the body can
    reach any number the body's algebra allows (live: a definite
    integral's cell value was F(upper) alone, green because the lower
    bound happened to contribute 0).  The indefinite-integral hops keep
    passing: an antiderivative chain honestly roots at its integrand,
    and the integrating step itself is derivative-checked.

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
    # one DEFINITE integral restates another: same variable, and
    # integrand plus both bounds equal modulo spelling (the textbook
    # \int_a^b dx/g and canonical \int_a^b 1/g dx forms of one
    # integral must cover each other exactly as the indefinite branch
    # above provides — live: an honest FTC chain was refused purely
    # because the goal was textbook-spelled). No body hop is involved,
    # so this holds in establishes mode too.
    in_def = definite_integral_parts(input_latex)
    goal_def = definite_integral_parts(goal_latex)
    if (in_def is not None and goal_def is not None
            and in_def[0] == goal_def[0]
            and stripped_eq(in_def[1], goal_def[1])
            and stripped_eq(in_def[2], goal_def[2])
            and stripped_eq(in_def[3], goal_def[3])):
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
    if not establishes:
        goal_body = _operator_body_latex(goal_latex)
        if goal_body is not None and same_expression(input_latex,
                                                     goal_body):
            return True
    return False


def _all_bracket_normal_form(latex):
    sym, notation = parse_latex(latex, allow_ellipsis=True)
    out = Notation()
    return _write_std(_GroupStripper(notation, out, all_brackets=True)(sym),
                      out)


def write_latex(sym, notation):
    """Readable LaTeX with a safety net: a candidate spelling is used only
    if it reads back as the expression it was asked to write.

    The two candidates used to be compared only against EACH OTHER, which
    silently assumed at least one of them round-trips. Once the normal
    form learned to keep function-argument spans apart, that assumption
    was measurably false: the raw writer spells `\\sin 2x` as
    `\\sin {2}x` — the integer value repr's braces close the argument
    span, so the string re-reads as `sin(2) x` (0.364 where the
    expression is 0.717). The pretty form was right and was being kept
    for the wrong reason. Comparing each candidate against the SOURCE
    graph instead names which one is faithful.

    Delimiter-redundant brackets are dropped first, so a rule-built
    formula's syntax-protection wrappers do not survive into the ledger
    artifact (`\\sqrt{\\left(x^{2}+1\\right)}`, `e^{\\left(2x\\right)}`)."""
    sym, notation = _peel_redundant_brackets(sym, notation)
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
    # the candidates disagree, so at most one of them re-reads correctly
    try:
        source = _dag_normal_form(sym, notation)
        if _normal_form(pretty) == source:
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


def _split_trailing_direction(sym, notation):
    """(replacement, direction) when the RIGHTMOST leaf of an approach
    expression carries the bare ``^+``/``^-`` marker that ordinary
    precedence bound to an inner factor — ``\\pi/2^-`` parses as
    ``\\pi/(2^-)`` and ``\\frac{\\pi}{2}^-`` hangs the marker on the
    denominator, so a top-level INDEX check alone reads them as
    TWO-SIDED limits at a corrupted point whose ``-`` the oracle then
    samples as a free variable.  A bare sign is never legitimate
    arithmetic (the grammar keeps it only as a direction marker), so
    stripping it structurally is sound; ``(sym, None)`` otherwise."""
    f = notation.get(sym)
    if f is None:
        return sym, None
    if f.sym == Notation.INDEX:
        sub_l, sup_l, power, sub_r = f.args[1]
        if (sub_l is None and sup_l is None and sub_r is None
                and isinstance(power, Symbol)
                and notation.get(power) is None
                and power.name in ('+', '-')):
            return (f.args[0],
                    'right' if power.name == '+' else 'left')
        return sym, None
    if f.sym.name in FRAC_NAMES:
        inner, direction = _split_trailing_direction(f.args[1], notation)
        if direction is None:
            return sym, None
        return notation.setf(f.sym, (f.args[0], inner)), direction
    if f.sym in (Notation.GROUP, Notation.V_GROUP, Notation.S_GROUP,
                 Notation.MINUS, Notation.PLUS, Notation.P_LIST,
                 Notation.S_LIST, Notation.SLASH):
        args = list(f.args)
        inner, direction = _split_trailing_direction(args[-1], notation)
        if direction is None:
            return sym, None
        args[-1] = inner
        return notation.setf(f.sym, tuple(args)), direction
    return sym, None


def _approach_point(sym, notation):
    """Return ``(point, direction)`` for a limit endpoint.

    The parser represents ``a^+`` / ``a^-`` as INDEX(a, power='+/-').
    Ordinary powered endpoints (``a^2``) remain untouched.  Direction is
    ``right`` / ``left`` / ``two-sided``.  A marker that precedence
    bound to an inner factor of a compound point (``\\pi/2^-``) is
    recovered by the trailing walk.
    """
    f = notation.getf(sym, Notation.INDEX)
    if f is None:
        stripped, direction = _split_trailing_direction(sym, notation)
        if direction is not None:
            return stripped, direction
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


_SATURATE_OVERFLOW = threading.local()


@contextmanager
def _overflow_saturation():
    """Evaluation mode in which a genuinely-huge overflow saturates to
    IEEE infinity and propagates (`1/\\cosh^{3}(300)` is 0, not an
    error), for the quadrature and approach-ladder legs whose deep
    rungs measurably die on `\\cosh`-family growth otherwise.

    Deliberately NOT the default: the comparison legs (`equal?`,
    spot checks) must keep seeing overflow as an evaluation failure —
    `inf` on both sides of `_num_agree` is nan-arithmetic, a false
    DISAGREE between identical spellings.  Every consumer of this mode
    re-raises when the final value itself is non-finite, so the mode
    never widens what a caller can observe, only what can cancel
    inside."""
    prev = getattr(_SATURATE_OVERFLOW, 'on', False)
    _SATURATE_OVERFLOW.on = True
    try:
        yield
    finally:
        _SATURATE_OVERFLOW.on = prev


def _apply_unary(fname, v):
    """One table application, saturating only the genuinely monotone
    overflow shapes (sinh/cosh/exp); `coth` saturates to ±1, never
    infinity, so its overflow stays an error."""
    try:
        return _UNARY_TABLE[fname](v)
    except OverflowError:
        if not getattr(_SATURATE_OVERFLOW, 'on', False):
            raise
        if fname == '\\sinh':
            return math.copysign(math.inf, v)
        if fname in ('\\cosh', '\\exp'):
            return math.inf
        raise


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

# How far a gap must clear the oracle's own noise before it counts as
# evidence of a difference. Measured separation is wide: a correct
# expansion whose digits were lost to cancellation sits at ratio 0.8,
# while genuinely different expressions sit above 10^12.
_RESOLUTION_MARGIN = 8.0


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
    try:
        return math.pow(b, p)
    except OverflowError:
        if not getattr(_SATURATE_OVERFLOW, 'on', False):
            raise
        if b > 0:
            return math.inf
        return math.inf if int(p) % 2 == 0 else -math.inf


def _num_abs(v):
    if isinstance(v, list):
        return max((abs(x) for row in v for x in row), default=0.0)
    return abs(v)


def _num_gap(v1, v2):
    """Distance between two oracle values, or None when the two are not
    numerically comparable at all — a shape mismatch is structural
    evidence, never a rounding artifact."""
    m1, m2 = isinstance(v1, list), isinstance(v2, list)
    if m1 != m2:
        return None
    if m1:
        if _num_shape(v1) != _num_shape(v2):
            return None
        return max((abs(x - y) for r1, r2 in zip(v1, v2)
                    for x, y in zip(r1, r2)), default=0.0)
    return abs(v1 - v2)


def _eval_noise(sym, notation, env, value):
    """How far this evaluation moves when one coordinate is nudged by a
    single ULP, or None when no nudge could be evaluated.

    Large intermediate terms amplify an input perturbation exactly as they
    amplify round-off, so this is a proxy for the oracle's own numerical
    noise at ``env`` — and it reuses the SAME evaluator, so a node type the
    oracle gains later is covered without touching this code."""
    worst = 0.0
    probed = False
    for key, coord in env.items():
        if not isinstance(coord, float):
            continue
        for toward in (math.inf, -math.inf):
            nudged = dict(env)
            nudged[key] = math.nextafter(coord, toward)
            try:
                moved = numeric_eval(sym, notation, nudged)
            except (EvalError, ZeroDivisionError, ValueError, OverflowError):
                continue
            gap = _num_gap(value, moved)
            if gap is None:
                continue
            probed = True
            worst = max(worst, gap)
    if env and not probed:
        return None
    return worst


def _disagreement_resolves(s1, n1, s2, n2, env, v1, v2):
    """Whether the values at ``env`` lie far enough apart to be evidence.

    ``disagree`` is a positive claim that two expressions differ, and it
    bars the step from the ledger for good. A float evaluation whose
    significant digits were consumed by cancellation cannot support that
    claim: ``(x+y)^{16}`` and its own canonical expansion land 6% apart at
    a sample point where the intermediate terms exceed the result by
    eighteen orders of magnitude. Unsupported numeric evaluation is honest
    ignorance, not evidence against a transformation, so a gap the oracle
    cannot resolve leaves the point undecided instead of accusing."""
    gap = _num_gap(v1, v2)
    if gap is None:
        return True
    noise1 = _eval_noise(s1, n1, env, v1)
    noise2 = _eval_noise(s2, n2, env, v2)
    if noise1 is None or noise2 is None:
        return False
    return gap > _RESOLUTION_MARGIN * max(noise1, noise2)


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
            return _apply_unary(fname.name, v)
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
            v = _apply_unary(fname, inner)
            if power is not None:
                v = _num_pow(v, numeric_eval(power, notation, env))
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


# ---------------------------------------------------------------------------
# assumption constraints: the oracle samples only inside the assumed region
# ---------------------------------------------------------------------------

# Relations the oracle can test at a sample point. The tactics accept only
# strict hypotheses (see `_CONSTRAINT_REL` in tactics/core.py); the rest are
# evaluated honestly here anyway, and a region with no interior simply
# leaves the check 'skipped' instead of silently unguarded.
_ORACLE_REL = {'=', '\\ne', '\\neq', '<', '\\lt', '>', '\\gt',
               '\\le', '\\leq', '\\ge', '\\geq'}
# distance from the boundary a sample point must keep. Points nearer than
# this prove nothing about a strict relation, so they are rejected rather
# than decided.
_STRICT_MARGIN = 1e-6
_DISTINCT_MARGIN = 1e-4


def _relation_parts(latex):
    """(lhs, rhs, rel, notation) for a relation, else None. The oracle
    re-derives relation structure itself and never routes through the
    symbolic tactic helpers — the two legs must stay independent."""
    sym, notation = parse_latex(latex)
    comp = notation.getf(sym, Notation.COMP)
    if comp is None:
        return None
    rel = comp.sym.props.get('op')
    if rel not in _ORACLE_REL:
        return None
    return comp.args[0], comp.args[1], rel, notation


def _conjunction_parts(latex):
    """Oracle-owned relation parts for one disjunct.

    A plain relation is a one-member conjunction.  An ``A_LIST`` is the
    first-class ``\\land`` shape (including a chained comparison lowered by
    the parser), and every member must independently be a supported relation.
    This parsing stays on the numeric leg; it deliberately does not reuse the
    tactic-side case/provenance helpers.
    """
    sym, notation = parse_latex(latex)
    while True:
        wrapper = notation.vgetf(sym, [Notation.GROUP, Notation.V_GROUP])
        if wrapper is None:
            break
        sym = wrapper.args[0]
    head = notation.getf(sym, Notation.A_LIST)
    members = list(head.args) if head is not None else [sym]
    parts = []
    for member in members:
        comp = notation.getf(member, Notation.COMP)
        if comp is None:
            return None
        rel = comp.sym.props.get('op')
        if rel not in _ORACLE_REL:
            return None
        parts.append((comp.args[0], comp.args[1], rel, notation))
    return parts


def hypothesis_parts(assumption):
    """(lhs, rhs, direction) for an assumption that states a strict
    hypothesis, else None. direction is -1 for '<' and +1 for '>'."""
    relation = (assumption or {}).get('constraint')
    if not relation:
        return None
    try:
        parts = _relation_parts(relation)
    except PrimitiveError:
        return None
    if parts is None:
        return None
    lhs, rhs, rel, notation = parts
    if rel in ('<', '\\lt'):
        direction = -1
    elif rel in ('>', '\\gt'):
        direction = 1
    else:
        return None
    return write_latex(lhs, notation), write_latex(rhs, notation), direction


def exclusive_hypotheses(assumptions):
    """Index pairs of recorded hypotheses that cannot hold together: the
    same two sides compared in opposite strict directions (`x > 0` with
    `x < 0`, or `x > 0` with `0 > x`).

    Structural only, and deliberately incomplete: a general contradiction
    test would be a prover, which this layer is not. It recognises exactly
    the shape a sign case-split produces, so alternative cases are never
    presented as one conjunction."""
    parsed = [hypothesis_parts(a) for a in (assumptions or [])]
    pairs = []
    for i, first in enumerate(parsed):
        if first is None:
            continue
        for j in range(i + 1, len(parsed)):
            second = parsed[j]
            if second is None:
                continue
            l1, r1, d1 = first
            l2, r2, d2 = second
            same = same_expression(l1, l2) and same_expression(r1, r2)
            crossed = same_expression(l1, r2) and same_expression(r1, l2)
            if (same and d1 == -d2) or (crossed and d1 == d2):
                pairs.append((i, j))
    return pairs


def _relation_truth(v1, v2, rel, tol):
    """True/False for a relation between two sampled values; None when the
    point is too close to the boundary (or not comparable) to decide."""
    if isinstance(v1, list) or isinstance(v2, list):
        return None
    if rel in ('=', '\\ne', '\\neq'):
        agree = _num_agree(v1, v2, tol)
        if agree is None:
            return None
        return agree if rel == '=' else not agree
    scale = max(1.0, abs(v1), abs(v2))
    d = (v1 - v2) / scale
    if abs(d) < _STRICT_MARGIN:
        return None
    if rel in ('<', '\\lt', '\\le', '\\leq'):
        return d < 0
    return d > 0


def _sample_guards(assumptions):
    """(nonzero_guards, constraint_guards) parsed out of assumption records.
    `nonzero` keeps its historical meaning; `constraint` carries a whole
    relation the sample point must satisfy."""
    nonzero = []
    constraints = []
    for a in (assumptions or []):
        expr = a.get('nonzero')
        if expr:
            try:
                nonzero.append(parse_latex(expr))
            except PrimitiveError:
                pass
        relation = a.get('constraint')
        if relation:
            try:
                parts = _relation_parts(relation)
            except PrimitiveError:
                parts = None
            if parts is not None:
                constraints.append(parts)
    return nonzero, constraints


def _admissible_point(guards, env):
    """True when the point lies inside every recorded assumption. A point
    the guards cannot be evaluated at is rejected: an unusable guard must
    never widen the sampled region."""
    nonzero, constraints = guards
    try:
        for gs, gn in nonzero:
            if _num_abs(numeric_eval(gs, gn, env)) < _DISTINCT_MARGIN:
                return False
        for lhs, rhs, rel, gn in constraints:
            v1 = numeric_eval(lhs, gn, env)
            v2 = numeric_eval(rhs, gn, env)
            if rel in ('\\ne', '\\neq'):
                if _num_agree(v1, v2, _DISTINCT_MARGIN) is not False:
                    return False
                continue
            if _relation_truth(v1, v2, rel, _STRICT_MARGIN) is not True:
                return False
    except (EvalError, ZeroDivisionError, ValueError, OverflowError):
        return False
    return True


def _guard_variables(guards):
    """Free variables the guards need. The assumed region lives in the
    joint space: a hypothesis about a variable the compared expressions do
    not mention must still be sampled, or every point is rejected."""
    nonzero, constraints = guards
    names = set()
    for gs, gn in nonzero:
        names |= free_symbols(gs, gn)
    for lhs, rhs, _rel, gn in constraints:
        names |= free_symbols(lhs, gn) | free_symbols(rhs, gn)
    return names


def _sample_budget(samples, guards):
    # rejection sampling inside a constrained region needs more tries; the
    # unconstrained budget stays exactly as it was
    return samples * (24 if guards[1] else 8)


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
    guards = _sample_guards(assumptions)
    variables = (free_symbols(s1, n1) | free_symbols(s2, n2)
                 | _guard_variables(guards))
    rng = random.Random(seed)
    agreed = 0
    tried = 0
    undefined_both = 0
    unresolved = 0
    mismatches = 0
    mismatch = None
    budget = _sample_budget(samples, guards)
    while agreed < samples and tried < budget:
        tried += 1
        env = _sample_point(variables, rng)
        if not _admissible_point(guards, env):
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
            if not _disagreement_resolves(s1, n1, s2, n2, env, v1, v2):
                unresolved += 1
                continue
            return {'status': 'disagree', 'point': env,
                    'lhs': v1, 'rhs': v2}
        agreed += 1
    if mismatch is not None:
        return {'status': 'domain-differs', 'mismatches': mismatches,
                'common_samples': agreed, **mismatch}
    if agreed == 0:
        if unresolved:
            return {'status': 'skipped', 'unresolved_points': unresolved,
                    'reason': 'every sample point lost its significant '
                              'digits to cancellation; the comparison is '
                              'undecided, not disproved'}
        return {'status': 'skipped',
                'reason': 'no evaluable sample points'}
    result = {'status': 'agree', 'samples': agreed}
    if undefined_both:
        result['undefined_points'] = undefined_both
    if unresolved:
        result['unresolved_points'] = unresolved
    return result


def numeric_relation_check(latex1, latex2, assumptions=None, samples=12,
                           seed=20260727, tol=1e-6):
    """Independently check that two relations hold at exactly the same
    sample points inside the assumed region.

    Per-side value checks see the algebra but not the DIRECTION: they pass
    just as happily on `a < b` turned into `-a < -b`. This is the oracle
    leg for a step whose claim is that the relation is preserved."""
    try:
        parts1 = _relation_parts(latex1)
        parts2 = _relation_parts(latex2)
    except PrimitiveError as e:
        return {'status': 'skipped', 'reason': str(e)}
    if parts1 is None or parts2 is None:
        return {'status': 'skipped', 'reason': 'not a supported relation'}
    l1, r1, rel1, n1 = parts1
    l2, r2, rel2, n2 = parts2
    guards = _sample_guards(assumptions)
    variables = (free_symbols(l1, n1) | free_symbols(r1, n1)
                 | free_symbols(l2, n2) | free_symbols(r2, n2)
                 | _guard_variables(guards))
    rng = random.Random(seed)
    agreed = 0
    satisfied = 0
    tried = 0
    budget = _sample_budget(samples, guards)
    while agreed < samples and tried < budget:
        tried += 1
        env = _sample_point(variables, rng)
        if not _admissible_point(guards, env):
            continue
        try:
            t1 = _relation_truth(numeric_eval(l1, n1, env),
                                 numeric_eval(r1, n1, env), rel1, tol)
            t2 = _relation_truth(numeric_eval(l2, n2, env),
                                 numeric_eval(r2, n2, env), rel2, tol)
        except (EvalError, ZeroDivisionError, ValueError, OverflowError):
            continue
        if t1 is None or t2 is None:
            continue
        if t1 != t2:
            return {'status': 'disagree', 'point': env,
                    'holds': {latex1: t1, latex2: t2}}
        agreed += 1
        if t1:
            satisfied += 1
    if agreed == 0:
        return {'status': 'skipped',
                'reason': 'no sample points inside the assumed region'}
    # a region where the relation is never true agrees vacuously; report
    # how much of the sample actually exercised the direction
    return {'status': 'agree', 'samples': agreed, 'holding_points': satisfied}


def _union_truths_at(env, tparts, dparts, tol):
    """(target_truth, union_truth) at one point, or None when any side is
    unevaluable or boundary-blurred. Every disjunct must be evaluable — a
    disjunct silently dropping out of the OR would let a junk case ride
    along unchecked."""
    tl, tr, trel, tn = tparts
    try:
        t = _relation_truth(numeric_eval(tl, tn, env),
                            numeric_eval(tr, tn, env), trel, tol)
        conjunctions = []
        for group in dparts:
            truths = [
                _relation_truth(numeric_eval(dl, dn, env),
                                numeric_eval(dr, dn, env), drel, tol)
                for dl, dr, drel, dn in group
            ]
            if any(truth is None for truth in truths):
                return None
            conjunctions.append(all(truths))
    except (EvalError, ZeroDivisionError, ValueError, OverflowError):
        return None
    if t is None:
        return None
    return t, any(conjunctions)


def _simpson_panels(fs, fn, var, a, b, env, n):
    """One composite-Simpson pass over ``fs`` for ``var`` in [a, b] under
    ``env``: (value, bad_node).  ``bad_node`` is the first node where the
    integrand leaves its domain while other nodes evaluate — an
    x-dependent domain break inside the bounds, a witness rather than
    ignorance; value None with no bad node is oracle ignorance (an
    unevaluable draw), never evidence.  Shared oracle infrastructure:
    the definite-integral evaluation check and the FTC derivative check
    both integrate through here, neither shares a computation with any
    symbolic leg."""
    h = (b - a) / n
    total = 0.0
    evaluated = 0
    domain_break = None
    unknown = False
    for i in range(n + 1):
        x = a + i * h
        node_env = dict(env)
        node_env[var] = x
        kind, value = _eval_kind(fs, fn, node_env)
        if kind is None and (isinstance(value, list)
                             or not math.isfinite(value)):
            kind = 'oracle'
        if kind == 'domain':
            if domain_break is None:
                domain_break = x
            continue
        if kind is not None:
            unknown = True     # oracle ignorance, not a witness
            continue
        evaluated += 1
        weight = 1 if i in (0, n) else (4 if i % 2 else 2)
        total += weight * value
    if domain_break is not None:
        # every node failing is a bad parameter draw, not a witness
        return None, (domain_break if evaluated else None)
    if unknown:
        return None, None
    return total * h / 3.0, None


def numeric_definite_check(expr, var, result, samples=4, seed=20260731,
                           panels=64, tol=1e-4):
    """Quadrature leg for a definite-integral evaluation: composite
    Simpson over the integrand vs the numeric value of ``result``.

    Never touches the antiderivative — the symbolic leg substitutes
    bounds into F, this leg re-integrates f itself, so the two legs stay
    independent.  Free symbols other than the integration variable are
    sampled as parameters, each sample comparing quadrature against the
    result under the same environment.

    The integrand must be evaluable at every quadrature node: a node
    where it leaves its domain while its neighbours evaluate is an
    x-dependent domain break inside [a,b] — the Fundamental Theorem's
    own hypothesis fails there, and F(b)-F(a) is exactly the classic
    wrong answer (``\\int_{-1}^{1} x^{-2} = -2``) — so that point is
    reported as a refusal, never sampled around.  The hunt for that
    witness runs even when the claimed result itself is unevaluable —
    a divergent integral's F(b)-F(a) typically contains the very
    singularity (``\\ln(0)``), and skipping on it would admit the
    nonsense the break refusal exists to bar.  A symbolic bound is
    sampled like any parameter (the identity is then claimed for the
    bound as a variable), but a domain break under a sampled bound is
    a bad draw — the claim's own continuity assumption excludes that
    region — never a witness.  A sample whose every node fails is a
    bad parameter draw and is skipped.  Convergence is estimated from
    two grids; a non-converged estimate is oracle ignorance, never a
    counterexample."""
    parts = definite_integral_parts(expr, var)
    if parts is None:
        return {'status': 'skipped', 'reason': 'not a definite integral'}
    _var, integrand, lower, upper = parts
    try:
        fs, fn = parse_latex(integrand)
        ls, ln = parse_latex(lower)
        us, un = parse_latex(upper)
        rs, rn = parse_latex(result)
    except PrimitiveError as e:
        return {'status': 'skipped', 'reason': str(e)}
    bound_syms = free_symbols(ls, ln) | free_symbols(us, un)
    if var in bound_syms:
        return {'status': 'skipped',
                'reason': 'a bound contains the integration variable'}
    if not bound_syms:
        try:
            a = numeric_eval(ls, ln, {})
            b = numeric_eval(us, un, {})
        except (EvalError, ValueError, ZeroDivisionError, OverflowError):
            return {'status': 'skipped',
                    'reason': 'bounds are not evaluable'}
        if not (math.isfinite(a) and math.isfinite(b)):
            return {'status': 'skipped', 'reason': 'bounds are not finite'}
    params = ((free_symbols(fs, fn) | free_symbols(rs, rn) | bound_syms)
              - {var})

    rng = random.Random(seed)
    rounds = samples if params else 1
    agreed = 0
    tried = 0
    while agreed < rounds and tried < rounds * 6:
        tried += 1
        env = _sample_point(params, rng) if params else {}
        if bound_syms:
            try:
                a = numeric_eval(ls, ln, env)
                b = numeric_eval(us, un, env)
            except (EvalError, ValueError, ZeroDivisionError,
                    OverflowError):
                continue
            if any(isinstance(v, list) or not math.isfinite(v)
                   for v in (a, b)):
                continue

        def simpson(env, n):
            return _simpson_panels(fs, fn, var, a, b, env, n)

        def break_refusal(bad):
            near = 1e-9 * max(1.0, abs(a), abs(b))
            if abs(bad - a) <= near or abs(bad - b) <= near:
                reason = (f'integrand is not evaluable at the bound '
                          f'{var} = {bad:.6g}; the integral is improper '
                          'there — a singular endpoint closes through '
                          'integrate_improper (evaluate the truncated '
                          'integral, then cite its recorded one-sided '
                          'limit)')
            else:
                reason = (f'integrand is not evaluable at '
                          f'{var} = {bad:.6g} inside the bounds; '
                          'the integral is improper there')
            return {'status': 'disagree', 'point': {var: bad},
                    'reason': reason}

        try:
            expected = numeric_eval(rs, rn, env)
        except (EvalError, ValueError, ZeroDivisionError, OverflowError):
            expected = None
        if (expected is None or isinstance(expected, list)
                or not math.isfinite(expected)):
            if not bound_syms:
                _value, bad = simpson(env, panels)
                if bad is not None:
                    return break_refusal(bad)
            continue
        coarse, bad = simpson(env, panels)
        if bad is not None:
            if bound_syms:
                continue
            return break_refusal(bad)
        fine, bad = simpson(env, panels * 2)
        if coarse is None or fine is None:
            continue
        err_est = abs(fine - coarse) / 15.0   # Simpson is O(h^4)
        scale = max(1.0, abs(fine), abs(expected))
        if abs(fine - expected) / scale > tol:
            if abs(fine - expected) <= 16 * err_est:
                continue   # quadrature has not converged: skip sample
            return {'status': 'disagree', 'point': env,
                    'symbolic': expected, 'numeric': fine}
        agreed += 1
    if agreed == 0:
        return {'status': 'skipped', 'reason': 'no evaluable sample points'}
    return {'status': 'agree', 'samples': agreed,
            'method': 'composite-simpson quadrature'}


def _aitken_accel(x, y, z):
    d = (z - y) - (y - x)
    if abs(d) < 1e-15 * max(abs(x), abs(y), abs(z), 1e-300):
        return None
    return z - (z - y) ** 2 / d


def _ladder_verdict(vals, quad_err, expected, tol):
    """One truncation ladder against one expected value, by iterated
    Aitken with the accumulated quadrature residual as the honesty bar:
    ('agree'|'undecided'|'disagree'|'stalled', estimate).  'stalled'
    means the rung differences stop decaying — no finite value is in
    evidence, which is the divergent shape and never a counterexample."""
    diffs = [vals[k + 1] - vals[k] for k in range(len(vals) - 1)]
    tail = [abs(d) for d in diffs[-2:]]
    if len(tail) == 2 and tail[1] > 0.95 * tail[0] \
            and tail[1] > tol * max(1.0, abs(vals[-1])):
        return 'stalled', None
    lvl1 = [_aitken_accel(*vals[k:k + 3]) for k in range(len(vals) - 2)]
    if any(v is None for v in lvl1):
        est = vals[-1]
        spread = abs(vals[-1] - vals[-2])
    else:
        lvl2 = [_aitken_accel(*lvl1[k:k + 3])
                for k in range(len(lvl1) - 2)]
        if not lvl2 or any(v is None for v in lvl2):
            est = lvl1[-1]
            spread = abs(lvl1[-1] - lvl1[-2]) if len(lvl1) > 1 else 0.0
        else:
            est = lvl2[-1]
            spread = (abs(lvl2[-1] - lvl2[-2]) if len(lvl2) > 1
                      else abs(lvl2[-1] - lvl1[-1]))
    bar = 16.0 * (quad_err + spread)
    scale = max(1.0, abs(est), abs(expected))
    delta = abs(est - expected)
    if delta / scale <= tol:
        return 'agree', est
    if delta <= bar:
        return 'undecided', est
    return 'disagree', est


def numeric_improper_check(expr, var, side, result, samples=4,
                           seed=20260805, panels=128, rungs=7, tol=1e-4,
                           kind='singular'):
    """Truncation-quadrature leg for an improper integral: ``result``
    claims the definitional value — the limit of the truncated
    integrals.  ``kind='singular'``: the integrand is singular at the
    ``side`` (``'upper'``/``'lower'``) finite bound, and the ladder's
    cut points approach it geometrically from inside, each rung adding
    one graded slab.  ``kind='infinite'``: the ``side`` bound is
    ``±\\infty``, and the ladder's truncation points grow by decades
    away from the finite bound, each rung a full graded quadrature
    (under overflow saturation, so an exponentially decaying integrand
    is 0 at deep nodes rather than an error).  Both ladders share one
    verdict: iterated Aitken against the leg's own accumulated
    residual.  Never touches any antiderivative or recorded limit — it
    shares only the structure readers with the symbolic path.

    A gap that does not clear the bar is undecided, never a
    counterexample; a ladder whose rung differences stop decaying has
    no finite value in evidence (the divergent shape) and reports
    ``skipped`` — a refusal is never evidence of divergence.  A domain
    break at a node away from the declared improper bound is a witness
    that the integrand is improper elsewhere too, and refuses — under
    sampled parameters it is a bad draw instead."""
    parts = definite_integral_parts(expr, var)
    if parts is None:
        return {'status': 'skipped', 'reason': 'not a definite integral'}
    _var, integrand, lower, upper = parts
    if side not in ('upper', 'lower'):
        return {'status': 'skipped', 'reason': 'unknown singular side'}
    if kind not in ('singular', 'infinite'):
        return {'status': 'skipped', 'reason': 'unknown improper kind'}
    try:
        fs, fn = parse_latex(integrand)
        ls, ln = parse_latex(lower)
        us, un = parse_latex(upper)
        rs, rn = parse_latex(result)
    except PrimitiveError as e:
        return {'status': 'skipped', 'reason': str(e)}
    if kind == 'infinite':
        fin_s, fin_n = (ls, ln) if side == 'upper' else (us, un)
        if free_symbols(fin_s, fin_n):
            return {'status': 'skipped',
                    'reason': 'symbolic bounds are outside this check'}
        try:
            finite = numeric_eval(fin_s, fin_n, {})
        except (EvalError, ValueError, ZeroDivisionError, OverflowError):
            return {'status': 'skipped',
                    'reason': 'bounds are not evaluable'}
        if isinstance(finite, list) or not math.isfinite(finite):
            return {'status': 'skipped',
                    'reason': 'the finite bound is not evaluable'}
        cut_scale = max(1.0, abs(finite))
        ladder_ts = [cut_scale * 10.0 ** k for k in range(1, 6)]
        a = b = finite
    else:
        if free_symbols(ls, ln) | free_symbols(us, un):
            return {'status': 'skipped',
                    'reason': 'symbolic bounds are outside this check'}
        try:
            a = numeric_eval(ls, ln, {})
            b = numeric_eval(us, un, {})
        except (EvalError, ValueError, ZeroDivisionError, OverflowError):
            return {'status': 'skipped',
                    'reason': 'bounds are not evaluable'}
        if not (math.isfinite(a) and math.isfinite(b)):
            return {'status': 'skipped', 'reason': 'bounds are not finite'}
        if not b > a:
            return {'status': 'skipped', 'reason': 'bounds are not ordered'}
        span = b - a
        deltas = [span * 1e-2 * 10.0 ** (-k) for k in range(rungs)]
    params = (free_symbols(fs, fn) | free_symbols(rs, rn)) - {var}

    def piece(lo, hi, env):
        """(fine value, richardson error, bad node) for one region."""
        with _overflow_saturation():
            coarse, bad = _simpson_panels(fs, fn, var, lo, hi, env,
                                          panels)
            if coarse is None:
                return None, None, bad
            fine, bad = _simpson_panels(fs, fn, var, lo, hi, env,
                                        panels * 2)
        if fine is None:
            return None, None, bad
        return fine, abs(fine - coarse) / 15.0, None

    def singular_ladder(env):
        """(rung values, accumulated quadrature error, bad node)."""
        if side == 'upper':
            regions = [(a, b - deltas[0])]
            regions += [(b - deltas[k], b - deltas[k + 1])
                        for k in range(rungs - 1)]
        else:
            regions = [(a + deltas[0], b)]
            regions += [(a + deltas[k + 1], a + deltas[k])
                        for k in range(rungs - 1)]
        vals = []
        total = 0.0
        err = 0.0
        for lo, hi in regions:
            value, piece_err, bad = piece(lo, hi, env)
            if value is None:
                return None, None, bad
            total += value
            err += piece_err
            vals.append(total)
        return vals, err, None

    def infinite_ladder(env):
        """(rung values, accumulated quadrature error, None) — each rung
        a full graded quadrature to a decade-farther truncation point."""
        vals = []
        err = 0.0
        for t in ladder_ts:
            lo, hi = (a, t) if side == 'upper' else (-t, b)
            try:
                value = _graded_quadrature(fs, fn, var, lo, hi, env)
            except (EvalError, ValueError, ZeroDivisionError,
                    OverflowError):
                return None, None, None
            vals.append(value)
            err += 1e-6 * max(1.0, abs(value))
        return vals, err, None

    rng = random.Random(seed)
    rounds = samples if params else 1
    agreed = 0
    tried = 0
    last_reason = 'no evaluable sample points'
    while agreed < rounds and tried < rounds * 6:
        tried += 1
        env = _sample_point(params, rng) if params else {}
        try:
            expected = numeric_eval(rs, rn, env)
        except (EvalError, ValueError, ZeroDivisionError, OverflowError):
            continue
        if isinstance(expected, list) or not math.isfinite(expected):
            continue
        # break-hunt probe walked so the declared improper bound is the
        # LAST node (singular) or not a node at all (infinite): the
        # first domain break found is then never the declared one, i.e.
        # a witness that the integrand is improper somewhere else too
        if kind == 'infinite':
            lo, hi = ((a, ladder_ts[0]) if side == 'upper'
                      else (-ladder_ts[0], b))
            with _overflow_saturation():
                _probe, bad = _simpson_panels(fs, fn, var, lo, hi, env,
                                              panels)
            witness = bad is not None
        else:
            with _overflow_saturation():
                if side == 'upper':
                    _probe, bad = _simpson_panels(fs, fn, var, a, b,
                                                  env, panels)
                    declared = b
                else:
                    _probe, bad = _simpson_panels(fs, fn, var, b, a,
                                                  env, panels)
                    declared = a
            witness = bad is not None and (
                abs(bad - declared) > 1e-9 * max(1.0, abs(declared)))
        if witness:
            if not params:
                return {'status': 'disagree', 'point': {var: bad},
                        'reason': f'integrand is not evaluable at '
                                  f'{var} = {bad:.6g}, away from the '
                                  'declared improper bound; the integral '
                                  'is improper there too'}
            continue
        if kind == 'infinite':
            vals, quad_err, bad = infinite_ladder(env)
        else:
            vals, quad_err, bad = singular_ladder(env)
        if vals is None:
            if bad is not None and not params:
                return {'status': 'disagree', 'point': {var: bad},
                        'reason': f'integrand is not evaluable at '
                                  f'{var} = {bad:.6g}, away from the '
                                  'declared improper bound; the integral '
                                  'is improper there too'}
            last_reason = 'truncated integrals were not evaluable'
            continue
        status, est = _ladder_verdict(vals, quad_err, expected, tol)
        if status == 'stalled':
            last_reason = ('the truncation ladder does not settle — no '
                           'finite value is in evidence (a refusal is '
                           'never evidence of divergence)')
            if params:
                continue
            break
        if status == 'undecided':
            last_reason = ('the truncation quadrature could not '
                           'resolve the gap within its own error bar')
            continue
        if status == 'disagree':
            return {'status': 'disagree', 'point': env,
                    'symbolic': expected, 'numeric': est}
        agreed += 1
    if agreed == 0:
        return {'status': 'skipped', 'reason': last_reason}
    return {'status': 'agree', 'samples': agreed,
            'method': 'graded truncation quadrature with iterated '
                      'Aitken extrapolation'}


def numeric_union_check(target, disjuncts, samples=12, seed=20260730,
                        tol=1e-6):
    """Independently check that a disjunction of relation conjunctions holds
    exactly where the target relation holds.

    A disjunct may be one relation or an ``A_LIST`` conjunction; its truth is
    the AND of its member truths, and the union is the OR of the disjuncts.
    An assembled union claims a biconditional: some complete disjunct is true
    at a point if and only if the target is. The mismatch region of a wrong
    union is typically a bounded interval between roots, which a handful
    of random points can miss entirely — so the one-variable case also
    walks a fine deterministic sweep (uniform steps plus pole-clustered
    reciprocal points). Agreement must exercise both truth sides: a
    sample that never saw the relation hold, or never saw it fail, is
    one-sided evidence and reports `skipped`, not `agree`."""
    try:
        tparts = _relation_parts(target)
        dparts = [_conjunction_parts(d) for d in disjuncts]
    except PrimitiveError as e:
        return {'status': 'skipped', 'reason': str(e)}
    if tparts is None or not dparts or any(p is None for p in dparts):
        return {'status': 'skipped', 'reason': 'not a supported relation'}
    tl, tr, trel, tn = tparts
    variables = free_symbols(tl, tn) | free_symbols(tr, tn)
    for group in dparts:
        for dl, dr, _drel, dn in group:
            variables |= free_symbols(dl, dn) | free_symbols(dr, dn)
    agreed = 0
    holding = 0
    if len(variables) == 1:
        var = next(iter(variables))
        sweep = [k / 20.0 for k in range(-240, 241)]
        sweep += [sign / k for sign in (1.0, -1.0) for k in range(2, 61)]
        for x in sweep:
            pair = _union_truths_at({var: x}, tparts, dparts, tol)
            if pair is None:
                continue
            t, union = pair
            if t != union:
                return {'status': 'disagree', 'point': {var: x},
                        'holds': {'target': t, 'union': union}}
            agreed += 1
            if t:
                holding += 1
    rng = random.Random(seed)
    tried = 0
    wanted = agreed + samples
    budget = _sample_budget(samples, ([], []))
    while agreed < wanted and tried < budget:
        tried += 1
        env = _sample_point(variables, rng)
        pair = _union_truths_at(env, tparts, dparts, tol)
        if pair is None:
            continue
        t, union = pair
        if t != union:
            return {'status': 'disagree', 'point': env,
                    'holds': {'target': t, 'union': union}}
        agreed += 1
        if t:
            holding += 1
    if agreed == 0:
        return {'status': 'skipped', 'reason': 'no evaluable sample points'}
    if holding == 0 or holding == agreed:
        side = 'hold' if holding == 0 else 'fail'
        return {'status': 'skipped',
                'reason': f'one-sided sample: the target was never seen to '
                          f'{side}, so coverage of the union was not '
                          f'exercised in both directions'}
    return {'status': 'agree', 'samples': agreed, 'holding_points': holding}


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
        g = self.output_notation.getf(copied, Notation.GROUP)
        if g is not None and g.props.get('br') == '()':
            # already self-delimited: a second () layer is what made
            # substitute('a^2+1','a','(x+1)') read ((x+1))^{2}+1, and it
            # survives wherever the relax pass cannot reach (an INDEX base,
            # a product factor). The wrapper exists to delimit, so a node
            # that is already delimited needs no other.
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
    pos = notation.getf(sym, Notation.PLUS)
    if pos is not None:
        # the explicit-sign spelling \int_0^{+\infty} / x \to +\infty
        sym = _transparent_inner(pos.args[0], notation)
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

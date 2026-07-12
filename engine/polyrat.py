#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
polyrat.py - canonical forms for the rational fragment.

Poly    - sparse multivariate polynomial {monomial: Fraction}
RatFunc - Poly/Poly with cancellation in the constructor

to_ratfunc(sym, notation)  - convert a notation expression iff it lies in the
                             fragment (symbols, integer powers, +, -, *, \frac,
                             numeric values); raises NotInFragment otherwise.
ratfunc_to_notation(rf, notation) - deterministic, degree-ordered output.

This module is deliberately independent from the rewrite rules in cmd_*.py:
it is the shared engine for the agent-scoped primitives (expand, collect,
evaluate, equal?) of the verified-derivation direction.
"""
import math
from fractions import Fraction

from notation import Notation, Symbol, Func
from value import IntegerValue, FracValue, FloatValue, Value


class NotInFragment(Exception):
    """Expression is outside the rational fragment."""
    pass


# LaTeX function names: these are NOT polynomial variables, and any
# expression applying them is outside the rational fragment.
FUNCTION_NAMES = set(Notation.unary_f) | {
    '\\ln', '\\log', '\\exp', '\\arcsin', '\\arccos', '\\arctan'}


# ---------------------------------------------------------------------------
# Poly: monomial is a tuple of (varname, exponent) pairs sorted by varname,
# exponents > 0. The empty tuple is the constant monomial.
# ---------------------------------------------------------------------------

def _mono_mul(m1, m2):
    d = dict(m1)
    for var, exp in m2:
        d[var] = d.get(var, 0) + exp
    return tuple(sorted((v, e) for v, e in d.items() if e != 0))


def _mono_degree(m):
    return sum(e for _, e in m)


def _mono_key(m):
    """Sort key: total degree desc, then lexicographic."""
    return (-_mono_degree(m), tuple((v, -e) for v, e in m))


class Poly(object):
    """Sparse multivariate polynomial with Fraction coefficients."""

    def __init__(self, terms=None):
        self.terms = {}
        if terms:
            for mono, coeff in terms.items():
                if coeff != 0:
                    self.terms[mono] = Fraction(coeff)

    @staticmethod
    def const(c):
        return Poly({(): Fraction(c)}) if c != 0 else Poly()

    @staticmethod
    def var(name):
        return Poly({((name, 1),): Fraction(1)})

    def is_zero(self):
        return not self.terms

    def is_const(self):
        return all(m == () for m in self.terms)

    def const_value(self):
        assert self.is_const()
        return self.terms.get((), Fraction(0))

    def variables(self):
        vs = set()
        for mono in self.terms:
            for var, _ in mono:
                vs.add(var)
        return vs

    def degree(self, var=None):
        if self.is_zero():
            return 0
        if var is None:
            return max(_mono_degree(m) for m in self.terms)
        return max((dict(m).get(var, 0) for m in self.terms), default=0)

    def __eq__(self, other):
        return isinstance(other, Poly) and self.terms == other.terms

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(frozenset(self.terms.items()))

    def __add__(self, other):
        terms = dict(self.terms)
        for mono, coeff in other.terms.items():
            terms[mono] = terms.get(mono, Fraction(0)) + coeff
        return Poly(terms)

    def __neg__(self):
        return Poly({m: -c for m, c in self.terms.items()})

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        terms = {}
        for m1, c1 in self.terms.items():
            for m2, c2 in other.terms.items():
                mono = _mono_mul(m1, m2)
                terms[mono] = terms.get(mono, Fraction(0)) + c1 * c2
        return Poly(terms)

    def __pow__(self, n):
        assert isinstance(n, int) and n >= 0
        res = Poly.const(1)
        base = self
        while n:
            if n & 1:
                res = res * base
            base = base * base
            n >>= 1
        return res

    def eval(self, point):
        """Evaluate at {varname: Fraction/float}. Returns Fraction or float."""
        total = 0
        for mono, coeff in self.terms.items():
            val = coeff
            for var, exp in mono:
                if var not in point:
                    raise KeyError(f'no value for {var}')
                val = val * (point[var] ** exp)
            total = total + val
        return total

    def derivative(self, var):
        terms = {}
        for mono, coeff in self.terms.items():
            d = dict(mono)
            exp = d.get(var, 0)
            if exp == 0:
                continue
            if exp == 1:
                del d[var]
            else:
                d[var] = exp - 1
            new_mono = tuple(sorted(d.items()))
            terms[new_mono] = terms.get(new_mono, Fraction(0)) + coeff * exp
        return Poly(terms)

    def content(self):
        """Positive rational content (gcd of coefficients); sign of the
        leading (canonically first) coefficient carried separately."""
        if self.is_zero():
            return Fraction(1)
        nums = [abs(c.numerator) for c in self.terms.values()]
        dens = [c.denominator for c in self.terms.values()]
        g = 0
        for n in nums:
            g = math.gcd(g, n)
        l = 1
        for d in dens:
            l = l * d // math.gcd(l, d)
        return Fraction(g, l)

    def leading_coeff(self):
        if self.is_zero():
            return Fraction(0)
        mono = min(self.terms, key=_mono_key)
        return self.terms[mono]

    def monomial_gcd(self):
        """Common monomial factor across all terms."""
        if self.is_zero():
            return ()
        common = None
        for mono in self.terms:
            d = dict(mono)
            if common is None:
                common = d
            else:
                common = {v: min(e, common[v]) for v, e in d.items() if v in common}
            if not common:
                return ()
        return tuple(sorted(common.items()))

    def divide_monomial(self, mono):
        if mono == ():
            return self
        terms = {}
        for m, c in self.terms.items():
            d = dict(m)
            for var, exp in mono:
                d[var] = d.get(var, 0) - exp
                assert d[var] >= 0
            terms[tuple(sorted((v, e) for v, e in d.items() if e != 0))] = c
        return Poly(terms)

    def scale(self, k):
        k = Fraction(k)
        return Poly({m: c * k for m, c in self.terms.items()})

    def div_exact(self, other):
        """Exact multivariate division: the quotient q with
        self == q * other, or None when `other` does not divide `self`.
        Leading-term elimination under lex order: when the division is
        exact, every remainder's leading term stays divisible by other's
        leading term, so a failed step proves non-divisibility."""
        if other.is_zero() or self.is_zero():
            return None
        varlist = sorted(self.variables() | other.variables())

        def key(mono):
            d = dict(mono)
            return tuple(d.get(v, 0) for v in varlist)

        lead_o = max(other.terms, key=key)
        lc_o = other.terms[lead_o]
        rem = Poly(self.terms)
        q = {}
        while not rem.is_zero():
            lead_r = max(rem.terms, key=key)
            d = dict(lead_r)
            for v, e in lead_o:
                d[v] = d.get(v, 0) - e
                if d[v] < 0:
                    return None
            m = tuple(sorted((v, e) for v, e in d.items() if e))
            c = rem.terms[lead_r] / lc_o
            q[m] = c
            rem = rem - Poly({m: c}) * other
        return Poly(q)

    def sorted_terms(self):
        return sorted(self.terms.items(), key=lambda kv: _mono_key(kv[0]))

    def __repr__(self):
        if self.is_zero():
            return '0'
        parts = []
        for mono, coeff in self.sorted_terms():
            mstr = '*'.join(f'{v}^{e}' if e > 1 else v for v, e in mono)
            parts.append(f'{coeff}{"*" + mstr if mstr else ""}')
        return ' + '.join(parts)


# ---------------------------------------------------------------------------
# Univariate helpers for GCD cancellation
# ---------------------------------------------------------------------------

def _to_univariate(poly, var):
    """Return dense coefficient list [c0, c1, ...] or None if not univariate in var."""
    coeffs = {}
    for mono, coeff in poly.terms.items():
        d = dict(mono)
        exp = d.pop(var, 0)
        if d:
            return None
        coeffs[exp] = coeff
    deg = max(coeffs, default=0)
    return [coeffs.get(i, Fraction(0)) for i in range(deg + 1)]


def _uni_normalize(c):
    while c and c[-1] == 0:
        c.pop()
    return c


def _uni_divmod(a, b):
    a = list(a)
    q = [Fraction(0)] * max(0, len(a) - len(b) + 1)
    while len(a) >= len(b) and _uni_normalize(a):
        k = a[-1] / b[-1]
        d = len(a) - len(b)
        q[d] = k
        for i, bc in enumerate(b):
            a[i + d] -= k * bc
        _uni_normalize(a)
    return q, a


def _uni_gcd(a, b):
    a = _uni_normalize(list(a))
    b = _uni_normalize(list(b))
    while b:
        _, r = _uni_divmod(a, b)
        a, b = b, _uni_normalize(r)
    if a:
        lead = a[-1]
        a = [c / lead for c in a]
    return a


def _from_univariate(coeffs, var):
    terms = {}
    for exp, c in enumerate(coeffs):
        if c == 0:
            continue
        mono = ((var, exp),) if exp > 0 else ()
        terms[mono] = c
    return Poly(terms)


# ---------------------------------------------------------------------------
# RatFunc
# ---------------------------------------------------------------------------

class RatFunc(object):
    """Rational function num/den with cancellation in the constructor."""

    def __init__(self, num, den=None):
        if den is None:
            den = Poly.const(1)
        if den.is_zero():
            raise ZeroDivisionError('zero denominator')
        num, den = self._cancel(num, den)
        self.num = num
        self.den = den

    @staticmethod
    def _cancel(num, den):
        if num.is_zero():
            return num, Poly.const(1)
        # 1. monomial gcd
        mg_n, mg_d = dict(num.monomial_gcd()), dict(den.monomial_gcd())
        common = tuple(sorted(
            (v, min(e, mg_d[v])) for v, e in mg_n.items() if v in mg_d and min(e, mg_d[v]) > 0
        ))
        if common:
            num = num.divide_monomial(common)
            den = den.divide_monomial(common)
        # 2. univariate polynomial gcd when both sides are univariate in one var
        vs = num.variables() | den.variables()
        if len(vs) == 1 and num.degree() > 0 and den.degree() > 0:
            var = next(iter(vs))
            ua, ub = _to_univariate(num, var), _to_univariate(den, var)
            if ua is not None and ub is not None:
                g = _uni_gcd(ua, ub)
                if len(g) > 1:
                    qa, _ = _uni_divmod(ua, g)
                    qb, _ = _uni_divmod(ub, g)
                    num = _from_univariate(qa, var)
                    den = _from_univariate(qb, var)
        # 2b. multivariate exact division: when one side divides the other,
        #     cancellation is complete ((x+y)x/(x+y) -> x) without needing
        #     a full multivariate GCD; partial common factors still stay
        #     uncancelled (cross-multiplying __eq__ compensates)
        if len(vs) > 1 and not den.is_const():
            q = num.div_exact(den)
            if q is not None:
                num, den = q, Poly.const(1)
            elif not num.is_const():
                q = den.div_exact(num)
                if q is not None:
                    num, den = Poly.const(1), q
        # 3. content + sign normalization: den gets positive leading coeff,
        #    den content becomes 1
        cd = den.content()
        if den.leading_coeff() < 0:
            cd = -cd
        num = num.scale(1 / cd)
        den = den.scale(1 / cd)
        return num, den

    def is_poly(self):
        return self.den.is_const() and self.den.const_value() == 1

    def is_const(self):
        return self.num.is_const() and self.den.is_const()

    def const_value(self):
        assert self.is_const()
        return self.num.const_value() / self.den.const_value()

    def variables(self):
        return self.num.variables() | self.den.variables()

    def __eq__(self, other):
        # cross-multiplication: exact even when cancellation is incomplete
        # (multivariate GCD is only monomial-level)
        return (isinstance(other, RatFunc)
                and self.num * other.den == other.num * self.den)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __add__(self, other):
        return RatFunc(self.num * other.den + other.num * self.den,
                       self.den * other.den)

    def __neg__(self):
        return RatFunc(-self.num, self.den)

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        return RatFunc(self.num * other.num, self.den * other.den)

    def __truediv__(self, other):
        if other.num.is_zero():
            raise ZeroDivisionError('division by zero expression')
        return RatFunc(self.num * other.den, self.den * other.num)

    def __pow__(self, n):
        assert isinstance(n, int)
        if n >= 0:
            return RatFunc(self.num ** n, self.den ** n)
        if self.num.is_zero():
            raise ZeroDivisionError('zero to negative power')
        return RatFunc(self.den ** (-n), self.num ** (-n))

    def eval(self, point):
        d = self.den.eval(point)
        if d == 0:
            raise ZeroDivisionError('denominator vanishes at sample point')
        return self.num.eval(point) / d

    def __repr__(self):
        if self.is_poly():
            return repr(self.num)
        return f'({self.num!r}) / ({self.den!r})'


# ---------------------------------------------------------------------------
# notation -> RatFunc
# ---------------------------------------------------------------------------

def _value_to_fraction(v):
    if isinstance(v, IntegerValue):
        return Fraction(v.val)
    if isinstance(v, FracValue):
        if v.denom == 0:
            raise NotInFragment('fraction with zero denominator')
        return Fraction(v.num, v.denom)
    if isinstance(v, FloatValue):
        raise NotInFragment('float value is outside the exact fragment')
    raise NotInFragment(f'unsupported value {v!r}')


def to_ratfunc(sym, notation):
    """Convert notation expression to RatFunc; raise NotInFragment otherwise."""
    if isinstance(sym, Value):
        return RatFunc(Poly.const(_value_to_fraction(sym)))
    if not isinstance(sym, Symbol):
        raise NotInFragment(f'unsupported term {sym!r}')
    f = notation.get(sym)
    if f is None:
        if (sym.name in FUNCTION_NAMES or sym.name in Notation.styles
                or sym.name in Notation.p_oper or sym.name == '\\infty'):
            raise NotInFragment(f'operator symbol {sym.name}')
        return RatFunc(Poly.var(sym.name))
    name = f.sym.name
    if f.sym in (Notation.GROUP, Notation.V_GROUP, Notation.S_GROUP):
        if f.props.get('br') == '||':
            # absolute value is not a transparent bracket: |x| != x. Leave
            # the fragment so the atom layer keeps it opaque.
            raise NotInFragment('absolute value')
        return to_ratfunc(f.args[0], notation)
    if f.sym == Notation.MINUS:
        return -to_ratfunc(f.args[0], notation)
    if f.sym == Notation.PLUS:
        return to_ratfunc(f.args[0], notation)
    if f.sym == Notation.S_LIST:
        res = RatFunc(Poly())
        for t in f.args:
            res = res + to_ratfunc(t, notation)
        return res
    if f.sym == Notation.P_LIST:
        res = RatFunc(Poly.const(1))
        for t in f.args:
            if isinstance(t, Symbol) and t.name in Notation.styles:
                continue
            res = res * to_ratfunc(t, notation)
        return res
    if f.sym == Notation.SLASH or f.sym == Notation.STAR:
        a = to_ratfunc(f.args[0], notation)
        b = to_ratfunc(f.args[1], notation)
        return a / b if f.sym == Notation.SLASH else a * b
    if f.sym == Notation.INDEX:
        sub, sup_l, power, sup_r = f.args[1]
        if sub is not None or sup_l is not None or sup_r is not None:
            raise NotInFragment('subscripted symbol')
        if power is None:
            return to_ratfunc(f.args[0], notation)
        n = _index_power(power, notation)
        base = to_ratfunc(f.args[0], notation)
        return base ** n
    if name in ('\\frac', '\\dfrac', '\\tfrac', '\\cfrac'):
        a = to_ratfunc(f.args[0], notation)
        b = to_ratfunc(f.args[1], notation)
        return a / b
    raise NotInFragment(f'unsupported operation {name}')


def _index_power(power, notation):
    """Extract integer exponent from an INDEX power slot."""
    neg = False
    while True:
        if isinstance(power, Symbol):
            f = notation.get(power)
            if f is not None and f.sym == Notation.GROUP:
                power = f.args[0]
                continue
            if f is not None and f.sym == Notation.MINUS:
                neg = not neg
                power = f.args[0]
                continue
        break
    if isinstance(power, IntegerValue):
        return -power.val if neg else power.val
    raise NotInFragment(f'non-integer exponent {power!r}')


# ---------------------------------------------------------------------------
# RatFunc / Poly -> notation (deterministic, degree-ordered)
# ---------------------------------------------------------------------------

def _fraction_to_notation(q, notation):
    """Positive Fraction -> IntegerValue or \\frac value structure."""
    assert q > 0
    if q.denominator == 1:
        return IntegerValue(q.numerator)
    num = notation.setf(Notation.GROUP, (IntegerValue(q.numerator),), br='{}')
    den = notation.setf(Notation.GROUP, (IntegerValue(q.denominator),), br='{}')
    return notation.setf(Symbol('\\frac'), (num, den))


def _mono_to_factors(mono, notation):
    factors = []
    for var, exp in mono:
        if exp == 1:
            factors.append(Symbol(var))
        else:
            factors.append(notation.setf(
                Notation.INDEX,
                (Symbol(var), (None, None, IntegerValue(exp), None))))
    return factors


def poly_to_notation(poly, notation):
    """Deterministic degree-ordered notation for a Poly."""
    if poly.is_zero():
        return IntegerValue(0)
    terms = []
    for i, (mono, coeff) in enumerate(poly.sorted_terms()):
        negative = coeff < 0
        q = -coeff if negative else coeff
        factors = _mono_to_factors(mono, notation)
        if q != 1 or not factors:
            factors = [_fraction_to_notation(q, notation)] + factors
        if len(factors) == 1:
            term = factors[0]
        else:
            term = notation.setf(Notation.P_LIST, factors)
        if negative:
            term = notation.setf(Notation.MINUS, (term,))
        elif i > 0:
            term = notation.setf(Notation.PLUS, (term,))
        terms.append(term)
    if len(terms) == 1:
        return terms[0]
    return notation.setf(Notation.S_LIST, terms)


def ratfunc_to_notation(rf, notation):
    if rf.is_poly():
        return poly_to_notation(rf.num, notation)
    num = notation.setf(Notation.GROUP, (poly_to_notation(rf.num, notation),), br='{}')
    den = notation.setf(Notation.GROUP, (poly_to_notation(rf.den, notation),), br='{}')
    return notation.setf(Symbol('\\frac'), (num, den))

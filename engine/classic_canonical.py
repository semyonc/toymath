"""Bounded canonical algebra for the classic ``mul!``/``add!`` commands.

The classic commands remain compatibility operations: they return notation
directly and do not create a verified-derivation record.  This adapter lets
their commutative rational fragment share ``polyrat``'s canonical forms while
leaving unsupported notation to the existing procedural implementation.
"""
from notation import Notation, Symbol
from polyrat import NotInFragment, ratfunc_to_notation, to_ratfunc
from value import IntegerValue, Value


DEFAULT_CLASSIC_MAX_TERMS = 4096


class ClassicExpansionLimitError(RuntimeError):
    """A requested classic expansion exceeds its configured term budget."""


def _sat_add(a, b, cap):
    return min(cap, a + b)


def _sat_mul(a, b, cap):
    if a == 0 or b == 0:
        return 0
    if a > cap // b:
        return cap
    return min(cap, a * b)


def _power_term_bound(terms, exponent, cap):
    """Upper-bound distinct commutative terms in ``poly ** exponent``.

    Products choose ``exponent`` input terms with repetition, so the number
    of possible monomials is at most C(exponent + terms - 1, terms - 1).
    The incremental computation saturates before large integers are built.
    """
    if exponent == 0:
        return 1
    if terms == 0:
        return 0
    if terms == 1:
        return 1
    n = exponent + terms - 1
    k = min(exponent, terms - 1)
    result = 1
    for i in range(1, k + 1):
        result = result * (n - k + i) // i
        if result >= cap:
            return cap
    return result


def _integer_power(power, notation):
    negative = False
    while isinstance(power, Symbol):
        f = notation.get(power)
        if f is not None and f.sym == Notation.GROUP:
            power = f.args[0]
            continue
        if f is not None and f.sym == Notation.MINUS:
            negative = not negative
            power = f.args[0]
            continue
        break
    if not isinstance(power, IntegerValue):
        return None
    return -power.val if negative else power.val


def _add_shapes(left, right, cap):
    ln, ld = left
    rn, rd = right
    numerator = _sat_add(
        _sat_mul(ln, rd, cap),
        _sat_mul(rn, ld, cap),
        cap,
    )
    denominator = _sat_mul(ld, rd, cap)
    return numerator, denominator


def _mul_shapes(left, right, cap):
    return (
        _sat_mul(left[0], right[0], cap),
        _sat_mul(left[1], right[1], cap),
    )


def _div_shapes(left, right, cap):
    return (
        _sat_mul(left[0], right[1], cap),
        _sat_mul(left[1], right[0], cap),
    )


def _term_shape(sym, notation, cap):
    """Conservative (numerator terms, denominator terms) expansion shape.

    Unknown operations are one opaque atom.  This keeps the guard applicable
    to the procedural fallback too, without pretending those operations are
    part of the rational fragment.
    """
    if isinstance(sym, Value) or not isinstance(sym, Symbol):
        return 1, 1
    f = notation.get(sym)
    if f is None:
        return 1, 1
    if f.sym in (Notation.GROUP, Notation.V_GROUP, Notation.S_GROUP):
        if Notation.is_semantic_bracket(f):
            return 1, 1
        return _term_shape(f.args[0], notation, cap)
    if f.sym in (Notation.PLUS, Notation.MINUS):
        return _term_shape(f.args[0], notation, cap)
    if f.sym == Notation.S_LIST:
        if not f.args:
            return 1, 1
        result = _term_shape(f.args[0], notation, cap)
        for term in f.args[1:]:
            result = _add_shapes(result, _term_shape(term, notation, cap), cap)
        return result
    if f.sym == Notation.P_LIST:
        result = (1, 1)
        for factor in f.args:
            result = _mul_shapes(
                result, _term_shape(factor, notation, cap), cap)
        return result
    if f.sym in (Notation.SLASH, Notation.STAR):
        left = _term_shape(f.args[0], notation, cap)
        right = _term_shape(f.args[1], notation, cap)
        if f.sym == Notation.SLASH:
            return _div_shapes(left, right, cap)
        return _mul_shapes(left, right, cap)
    if f.sym == Notation.INDEX:
        sub, sup_l, power, sup_r = f.args[1]
        if sub is not None or sup_l is not None or sup_r is not None:
            return 1, 1
        exponent = _integer_power(power, notation)
        if exponent is None:
            return 1, 1
        base = _term_shape(f.args[0], notation, cap)
        numerator = _power_term_bound(base[0], abs(exponent), cap)
        denominator = _power_term_bound(base[1], abs(exponent), cap)
        if exponent < 0:
            numerator, denominator = denominator, numerator
        return numerator, denominator
    if f.sym.name in ('\\frac', '\\dfrac', '\\tfrac', '\\cfrac'):
        return _div_shapes(
            _term_shape(f.args[0], notation, cap),
            _term_shape(f.args[1], notation, cap),
            cap,
        )
    return 1, 1


def _contains_preserved_product(sym, notation, seen=None):
    """Do not canonicalize an explicit ``\\cdot`` product as arithmetic."""
    if seen is None:
        seen = set()
    if not isinstance(sym, Symbol) or sym in seen:
        return False
    seen.add(sym)
    f = notation.get(sym)
    if f is None:
        return False
    if f.sym == Notation.P_LIST and f.props.get('cdot'):
        return True
    for arg in f.args:
        values = arg if isinstance(arg, (tuple, list)) else (arg,)
        if any(_contains_preserved_product(value, notation, seen)
               for value in values if value is not None):
            return True
    return False


def check_expansion_budget(sym, notation, max_terms):
    if max_terms <= 0:
        raise ValueError('classic expansion term budget must be positive')
    cap = max_terms + 1
    numerator, denominator = _term_shape(sym, notation, cap)
    estimate = max(numerator, denominator)
    if estimate > max_terms:
        raise ClassicExpansionLimitError(
            'classic expansion exceeds the term budget '
            f'({estimate}+ estimated terms, limit {max_terms})'
        )


def canonicalize_classic(sym, notation,
                         max_terms=DEFAULT_CLASSIC_MAX_TERMS):
    """Return a fresh canonical node, or ``None`` for legacy fallback.

    The budget is checked before attempting either path so unsupported opaque
    expressions cannot bypass the classic expansion guard.
    """
    check_expansion_budget(sym, notation, max_terms)
    if _contains_preserved_product(sym, notation):
        return None
    try:
        rational = to_ratfunc(sym, notation)
    except (NotInFragment, ZeroDivisionError):
        return None
    exact_terms = max(len(rational.num.terms), len(rational.den.terms))
    if exact_terms > max_terms:
        raise ClassicExpansionLimitError(
            'classic expansion exceeds the term budget '
            f'({exact_terms} terms, limit {max_terms})'
        )
    return ratfunc_to_notation(rational, notation)

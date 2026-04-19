from value import *
from replicator import Replicator
from comparer import pattern, NotationParam, s_equal
from notation import Notation, Func, Symbol
from helpers import trace_notation
from frac_utils import is_frac, get_numerator, get_denominator, normalize_frac
from processor import get_value


def chainexpr(oper, notation, sym, negative):
    if negative:
        inner = notation.setf(oper, (None, (sym,)), negative=True)
    else:
        inner = notation.setf(oper, (None, (sym,)))
    ret = notation.setf(Notation.GROUP, (inner,), br="{}")
    return ret


class Mul(object):
    arity = 1
    MUL = Symbol("mul!")
    MULEX = Symbol("mulex!")
    Pw1 = pattern("(z)^n", [("z", NotationParam.Any), ("n", NotationParam.N)])

    def __init__(self, active):
        self.active = active

    def exec(self, processor, sym, f):
        return self.run(processor, f)

    def extract(self, notation, sym):
        f = notation.vgetf(sym, [Notation.PLUS, Notation.GROUP])
        if f is not None:
            return self.extract(notation, f.args[0])
        f = notation.getf(sym, Notation.S_LIST)
        if f is not None:
            return f.args
        return [sym]

    def is_eval(self, notation, sym):
        f = notation.getf(sym, Notation.GROUP)
        if f is not None:
            f = notation.vgetf(f.args[0], [self.MUL, self.MULEX])
            if f is not None:
                return True
        return False

    def run(self, processor, f):
        sym = f.args[1][0]
        if self.active:
            outsym = processor.enter_subformula(sym)
            pass
        else:
            repl = Replicator(processor.notation, processor.output_notation)
            outsym = repl.enter_subformula(sym)
        negative = False
        if 'negative' in f.props:
            negative = True
        return self.main(processor, processor.output_notation, outsym, negative)

    def main(self, processor, notation, sym, negative):
        # Existing positive exponent pattern (keep for n > 0 non-fraction)
        subst = Mul.Pw1.match(sym, notation)
        if subst is not None:
            z, n = subst["z"], subst["n"]
            unwrapped = self.unwrap_base(notation, z)
            if is_frac(notation, unwrapped):
                result = self.power_fraction(processor, notation, unwrapped, n, negative)
                if result is not None:
                    return result
            return self.power(processor, notation, z, n)

        # Direct INDEX detection for negative/zero/any integer exponent
        power_info = self.detect_power(notation, sym)
        if power_info is not None:
            base, n = power_info
            unwrapped = self.unwrap_base(notation, base)

            if n.val == 0:
                return IntegerValue(1)

            if is_frac(notation, unwrapped):
                result = self.power_fraction(processor, notation, unwrapped, n, negative)
                if result is not None:
                    return result

            if n.val < 0:
                return self.negative_power(processor, notation, base, n, negative)

        # Fraction multiplication handling (Rules 2-4)
        result = self.multiply_fractions(processor, notation, sym, negative)
        if result is not None:
            return result

        return self.multiplay_plist(processor, notation, sym, negative)

    def power(self, processor, notation, expr, n):
        args = []
        outsym = notation.setf(Notation.GROUP, (expr,), br="()")
        for i in range(n.val):
            args.append(outsym)
        sym_plist = notation.setf(Notation.P_LIST, tuple(args))
        return notation.setf(self.MUL, (None, (sym_plist,)))

    def detect_power(self, notation, sym):
        """
        Detect (base)^exponent structure directly without patterns.
        Returns (base, exponent) or None.
        """
        # Unwrap GROUP
        f = notation.getf(sym, Notation.GROUP)
        if f is not None:
            sym = f.args[0]

        # Check for INDEX
        f = notation.getf(sym, Notation.INDEX)
        if f is None:
            return None

        base = f.args[0]
        power = f.args[1][2]  # INDEX structure: (base, (sub, sup_l, power, sup_r))

        # Get integer exponent value
        n = get_value(power, notation)
        if not isinstance(n, IntegerValue):
            return None

        return (base, n)

    def unwrap_base(self, notation, z):
        """Unwrap GROUP/MINUS from base."""
        g = notation.getf(z, Notation.GROUP)
        if g is not None:
            z = g.args[0]
        m = notation.getf(z, Notation.MINUS)
        if m is not None:
            z = m.args[0]
        return z

    def get_power_info(self, notation, sym):
        """
        Extract base and exponent from symbol.
        Returns (base, exponent_value) or (sym, IntegerValue(1)) for plain symbols.
        Exponent must be an IntegerValue.
        """
        # Unwrap GROUP
        f = notation.getf(sym, Notation.GROUP)
        if f is not None:
            sym = f.args[0]

        # Check for INDEX (power)
        f = notation.getf(sym, Notation.INDEX)
        if f is not None:
            base = f.args[0]
            power = f.args[1][2]  # INDEX: (base, (sub, sup_l, power, sup_r))
            if power is not None:
                n = get_value(power, notation)
                if isinstance(n, IntegerValue):
                    return (base, n)

        # Plain symbol treated as base^1
        return (sym, IntegerValue(1))

    def power_fraction(self, processor, notation, frac_sym, n, negative):
        """
        Expand (frac{x}{y})^n symbolically.

        n > 0: (frac{x}{y})^n -> frac{x^n}{y^n}
        n < 0: (frac{x}{y})^n -> frac{y^|n|}{x^|n|}
        n = 0: -> 1
        """
        x = get_numerator(notation, frac_sym)
        y = get_denominator(notation, frac_sym)

        # Idempotence: skip if already expanded
        if notation.getf(x, Notation.INDEX) or notation.getf(y, Notation.INDEX):
            return None

        if n.val == 0:
            return IntegerValue(1)

        if n.val > 0:
            x_pow = notation.setf(Notation.INDEX, (x, (None, None, n, None)))
            y_pow = notation.setf(Notation.INDEX, (y, (None, None, n, None)))
            result = normalize_frac(notation, x_pow, y_pow, chainexpr, self.MUL)
        else:
            pos_n = IntegerValue(-n.val)
            y_pow = notation.setf(Notation.INDEX, (y, (None, None, pos_n, None)))
            x_pow = notation.setf(Notation.INDEX, (x, (None, None, pos_n, None)))
            result = normalize_frac(notation, y_pow, x_pow, chainexpr, self.MUL)

        if negative:
            result = notation.setf(Notation.MINUS, (result,))
        return result

    def negative_power(self, processor, notation, z, n, negative):
        """x^{-n} -> frac{1}{x^n}"""
        pos_n = IntegerValue(-n.val)
        z_pow = notation.setf(Notation.INDEX, (z, (None, None, pos_n, None)))
        result = normalize_frac(notation, IntegerValue(1), z_pow, chainexpr, self.MUL)
        if negative:
            result = notation.setf(Notation.MINUS, (result,))
        return result

    def multiply_fractions(self, processor, notation, sym, negative):
        """
        Handle fraction multiplication patterns (Rules 2-4).

        Returns:
            Fraction result if pattern matches, None otherwise
        """
        # Check for P_LIST with exactly 2 factors
        f = notation.getf(sym, Notation.GROUP)
        if f is not None:
            sym = f.args[0]

        f = notation.getf(sym, Notation.P_LIST)
        if f is None or len(f.args) != 2:
            return None

        # Extract, unwrap groups, and peel leading minus signs
        def unwrap(factor):
            negative_flag = False
            g = notation.getf(factor, Notation.GROUP)
            if g is not None:
                factor = g.args[0]
            m = notation.getf(factor, Notation.MINUS)
            if m is not None:
                negative_flag = True
                factor = m.args[0]
            return factor, negative_flag

        factor1, factor2 = f.args
        unwrapped1, neg1 = unwrap(factor1)
        unwrapped2, neg2 = unwrap(factor2)
        overall_negative = negative ^ neg1 ^ neg2

        def value_to_frac_symbol(val):
            if isinstance(val, FracValue):
                sign_negative = val.num < 0
                num = IntegerValue(abs(val.num))
                den = IntegerValue(val.denom)
                frac = notation.setf(
                    Symbol("\\frac"),
                    (
                        notation.setf(Notation.GROUP, (num,), br="{}"),
                        notation.setf(Notation.GROUP, (den,), br="{}"),
                    ),
                )
                if sign_negative:
                    frac = notation.setf(Notation.MINUS, (frac,))
                return frac
            return val

        def ensure_symbolic_fraction(val):
            if isinstance(val, FracValue):
                return value_to_frac_symbol(val)
            f = notation.getf(val, Notation.MINUS)
            if f is not None:
                inner = ensure_symbolic_fraction(f.args[0])
                if inner is not f.args[0]:
                    return notation.setf(Notation.MINUS, (inner,))
                return val
            f = notation.getf(val, Notation.GROUP)
            if f is not None:
                # Unwrap trivial {} groups around fractions
                inner = ensure_symbolic_fraction(f.args[0])
                if is_frac(notation, inner) and f.props.get("br") == "{}":
                    return inner  # Remove trivial grouping
                if inner is not f.args[0]:
                    return notation.setf(Notation.GROUP, (inner,), **f.props)
            return val

        def scale_slist_terms(slist_sym, factor):
            """Distribute scalar multiplication over sum terms: (a+b)·c → ac+bc"""
            slist = notation.getf(slist_sym, Notation.S_LIST)
            out_terms = []
            for idx, term in enumerate(slist.args):
                term_sign = 1
                t = term
                fplus = notation.getf(t, Notation.PLUS)
                if fplus is not None:
                    t = fplus.args[0]
                fminus = notation.getf(t, Notation.MINUS)
                if fminus is not None:
                    term_sign = -1
                    t = fminus.args[0]

                # Create mul! command for each term
                plist = notation.setf(Notation.P_LIST, (t, factor))
                scaled = chainexpr(self.MUL, notation, plist, None)

                if term_sign < 0:
                    scaled = notation.setf(Notation.MINUS, (scaled,))
                elif idx > 0:
                    scaled = notation.setf(Notation.PLUS, (scaled,))
                out_terms.append(scaled)
            return notation.setf(Notation.S_LIST, tuple(out_terms))

        # Short-circuit zero factors
        for candidate in (unwrapped1, unwrapped2):
            val = processor.get_factor(candidate)
            if equal_value(val, 0):
                return IntegerValue(0)

        # Power cancellation: base^m * base^n -> base^{m+n}
        # Only when at least one factor has an explicit power (not implicit exponent 1)
        base1, exp1 = self.get_power_info(notation, unwrapped1)
        base2, exp2 = self.get_power_info(notation, unwrapped2)
        if (exp1.val != 1 or exp2.val != 1) and s_equal(base1, notation, base2, notation):
            total_exp = IntegerValue(exp1.val + exp2.val)
            if total_exp.val == 0:
                result = IntegerValue(1)
            elif total_exp.val == 1:
                result = base1
            elif total_exp.val < 0:
                # Negative result: base^{-n} -> \frac{1}{base^n}
                pos_exp = IntegerValue(-total_exp.val)
                if pos_exp.val == 1:
                    denom = base1
                else:
                    denom = notation.setf(Notation.INDEX, (base1, (None, None, pos_exp, None)))
                result = normalize_frac(notation, IntegerValue(1), denom, chainexpr, self.MUL)
            else:
                result = notation.setf(Notation.INDEX, (base1, (None, None, total_exp, None)))
            if overall_negative:
                result = notation.setf(Notation.MINUS, (result,))
            return result

        # Rule 2: Fraction × Fraction
        if is_frac(notation, unwrapped1) and is_frac(notation, unwrapped2):
            x1 = get_numerator(notation, unwrapped1)
            y1 = get_denominator(notation, unwrapped1)
            x2 = get_numerator(notation, unwrapped2)
            y2 = get_denominator(notation, unwrapped2)

            num_plist = notation.setf(Notation.P_LIST, (x1, x2))
            num_mul = chainexpr(self.MUL, notation, num_plist, None)

            denom_plist = notation.setf(Notation.P_LIST, (y1, y2))
            denom_mul = chainexpr(self.MUL, notation, denom_plist, None)

            res = normalize_frac(notation, num_mul, denom_mul, chainexpr, self.MUL)
            res = value_to_frac_symbol(res)
            if overall_negative and not equal_value(res, 0):
                res = notation.setf(Notation.MINUS, (res,))
                res = notation.setf(Notation.GROUP, (res,), br="()")
            return ensure_symbolic_fraction(res)

        # Rule 4: Fraction × Sum (check both orderings)
        for frac, other in [(unwrapped1, unwrapped2), (unwrapped2, unwrapped1)]:
            if is_frac(notation, frac):
                # Check if other is a sum
                sum_sym = other
                g = notation.getf(other, Notation.GROUP)
                if g is not None:
                    sum_sym = g.args[0]

                if notation.getf(sum_sym, Notation.S_LIST) is not None:
                    x = get_numerator(notation, frac)
                    y = get_denominator(notation, frac)

                    # Distribute numeric fraction over sums (e.g., (1/2)(a+b) -> 1/2 a + 1/2 b)
                    if isinstance(x, Value) and isinstance(y, Value):
                        coef = division(
                            x.get_frac() if isinstance(x, IntegerValue) else x,
                            y.get_frac() if isinstance(y, IntegerValue) else y,
                        )
                        slist = notation.getf(sum_sym, Notation.S_LIST)
                        out_terms = []
                        for idx, term in enumerate(slist.args):
                            term_sign = 1
                            t = term
                            fplus = notation.getf(t, Notation.PLUS)
                            if fplus is not None:
                                t = fplus.args[0]
                            fminus = notation.getf(t, Notation.MINUS)
                            if fminus is not None:
                                term_sign = -1
                                t = fminus.args[0]
                            factor = coef
                            if term_sign < 0:
                                factor = multiplication(factor, IntegerValue(-1))
                            neg_term = less_value(factor, 0)
                            abs_factor = factor.abs() if hasattr(factor, "abs") else factor
                            scaled = notation.setf(Notation.P_LIST, (abs_factor, t))
                            if neg_term:
                                scaled = notation.setf(Notation.MINUS, (scaled,))
                            elif idx > 0:
                                scaled = notation.setf(Notation.PLUS, (scaled,))
                            out_terms.append(scaled)
                        return notation.setf(Notation.S_LIST, tuple(out_terms))

                    sum_grouped = notation.setf(Notation.GROUP, (sum_sym,), br="()")
                    num_plist = notation.setf(Notation.P_LIST, (x, sum_grouped))
                    num_mul = chainexpr(self.MUL, notation, num_plist, negative)

                    return normalize_frac(notation, num_mul, y, chainexpr, self.MUL)

        # Rule 3a: Scalar × Fraction
        if not is_frac(notation, unwrapped1) and is_frac(notation, unwrapped2):
            a = unwrapped1
            x = get_numerator(notation, unwrapped2)
            y = get_denominator(notation, unwrapped2)

            num_plist = notation.setf(Notation.P_LIST, (a, x))
            num_mul = chainexpr(self.MUL, notation, num_plist, None)

            res = normalize_frac(notation, num_mul, y, chainexpr, self.MUL)
            res = value_to_frac_symbol(res)
            if overall_negative and not equal_value(res, 0):
                res = notation.setf(Notation.MINUS, (res,))
                res = notation.setf(Notation.GROUP, (res,), br="()")
            return ensure_symbolic_fraction(res)

        # Rule 3b: Fraction × Scalar (symmetric)
        if is_frac(notation, unwrapped1) and not is_frac(notation, unwrapped2):
            x = get_numerator(notation, unwrapped1)
            y = get_denominator(notation, unwrapped1)
            a = unwrapped2

            slist = notation.getf(x, Notation.S_LIST)
            if slist is not None:
                # Distribute: (a+b)·c → ac+bc
                num_mul = scale_slist_terms(x, a)
            else:
                num_plist = notation.setf(Notation.P_LIST, (x, a))
                num_mul = chainexpr(self.MUL, notation, num_plist, None)

            res = normalize_frac(notation, num_mul, y, chainexpr, self.MUL)
            res = value_to_frac_symbol(res)
            if overall_negative and not equal_value(res, 0):
                res = notation.setf(Notation.MINUS, (res,))
                res = notation.setf(Notation.GROUP, (res,), br="()")
            return ensure_symbolic_fraction(res)

        return None  # No pattern matched

    def multiplay_plist(self, processor, notation, sym, negative):
        entry_sym = sym
        f = notation.getf(sym, Notation.GROUP)
        if f is not None:
            sym = f.args[0]
        f = notation.getf(sym, Notation.MINUS)
        if f is not None:
            sym = f.args[0]
            negative = True
        if negative:
            f = notation.getf(sym, Notation.S_LIST)
            if f is not None:
                factor = notation.setf(Notation.GROUP, 
                    (notation.setf(Notation.MINUS, (IntegerValue(1),)), 
                ), br="()")
                expr = notation.setf(Notation.GROUP, (sym,), br="()")
                return chainexpr(self.MUL, notation, 
                                 notation.setf(Notation.P_LIST, (factor, expr)), None)
            f = notation.getf(sym, Notation.GROUP)
            if f is not None:
                mulex = notation.getf(f.args[0], self.MULEX)
                if mulex is not None:
                    mulex.props["negative"] = 'negative' not in mulex.props
                    return sym
                mul = notation.getf(f.args[0], self.MUL)
                if mul is not None:
                    mul.props["negative"] = negative
        f = notation.getf(sym, Notation.P_LIST)
        if f is None:
            return entry_sym
        if len(f.args) > 2:
            rest = notation.setf(Notation.P_LIST, tuple(f.args[1:]))
            inner = chainexpr(self.MUL, notation, rest, None)
            outer = notation.setf(Notation.P_LIST, (f.args[0], inner))
            return chainexpr(self.MULEX, notation, outer, negative)
        if self.is_eval(notation, f.args[0]) or self.is_eval(notation, f.args[1]):
            return chainexpr(self.MULEX, notation, entry_sym, negative)
        res = []
        x = self.extract(notation, f.args[0])
        y = self.extract(notation, f.args[1])
        if not self.active and (x == [f.args[0]] or y == [f.args[1]]):
            if x != [f.args[0]]:
                res.append(f.args[0])
            else:
                res.append(chainexpr(self.MUL, notation, f.args[0], None))
            if y != [f.args[1]]:
                res.append(f.args[1])
            else:
                res.append(chainexpr(self.MUL, notation, f.args[1], None))
            outer = notation.setf(Notation.P_LIST, tuple(res))
            return chainexpr(self.MULEX, notation, outer, negative)
        for a in x:
            for b in y:
                p = processor.get_factor(a)
                q = processor.get_factor(b)
                factor = multiplication(p, q)
                if equal_value(factor, 0):
                    continue
                if negative:
                    factor = multiplication(factor, IntegerValue(-1))
                pl = []
                if not_equal_value(factor, 1) and not_equal_value(factor, -1):
                    pl.append(factor.abs())
                left = processor.get_expr(a)
                if left is not None:
                    pl += processor.subst(None, left, Notation.P_LIST)
                right = processor.get_expr(b)
                if right is not None:
                    pl += processor.subst(None, right, Notation.P_LIST)
                if len(pl) == 0:
                    # Pure scalar product
                    outs = factor.abs() if less_value(factor, 0) else factor
                elif len(pl) == 1:
                    outs = pl[0]
                else:
                    outs = notation.setf(Notation.P_LIST, tuple(pl))
                if less_value(factor, 0):
                    outs = notation.setf(Notation.MINUS, (
                        processor.subst(None, outs, Notation.S_LIST),
                    ))
                    if (len(res) == 0):
                        outs = notation.setf(Notation.GROUP, (outs,), br="()")
                elif len(res) > 0:
                    outs = notation.setf(Notation.PLUS, (outs,))
                res.append(outs)
        if not res:
            return IntegerValue(0)
        return notation.setf(Notation.S_LIST, tuple(res))


def create_actions():
    return {"mul": Mul(False), "mulex": Mul(True)}

from engine.helpers import trace_notation
from value import *
from replicator import Replicator
from comparer import pattern, NotationParam
from cmd_mul import Mul, chainexpr
from frac_utils import is_frac, get_numerator, get_denominator, normalize_frac
from classic_canonical import canonicalize_classic

def is_group_inside_plist(notation, f):
    for arg in f.args:
        if notation.getf(arg, Notation.GROUP) is not None:
            return True
    return False


class Add(object):
    arity = 1
    ADD = Symbol("add!")
    ADDEX = Symbol("addex!")

    def __init__(self, active):
        self.active = active

    def exec(self, processor, sym, f):
        return self.run(processor, f.args[1][0])

    def run(self, processor, sym):
        if self.active:
            outsym = processor.enter_subformula(sym)
        else:
            repl = Replicator(processor.notation, processor.output_notation)
            outsym = repl.enter_subformula(sym)
        return self.main(processor, processor.output_notation, outsym)

    def main(self, processor, notation, sym):
        canonical = canonicalize_classic(
            sym, notation, processor.max_expansion_terms)
        if canonical is not None:
            return canonical

        out = self.add_slist([], processor, notation, sym)
        if len(out) == 1:
            sym_out = out[0]
        else:
            sym_out = notation.setf(Notation.S_LIST, tuple(out))

        # Combine fractions after flattening
        sym_out = self.add_fractions(processor, notation, sym_out)
        return sym_out
    
    def add_slist(self, out, processor, notation, sym):
        f = notation.getf(sym, Notation.GROUP)
        if f is not None:
            return self.add_slist(out, processor, notation, f.args[0])
        f_cmd = notation.get(sym)
        if f_cmd is not None and isinstance(f_cmd.sym, Symbol) and f_cmd.sym.name.endswith('!') \
                and f_cmd.sym not in (self.ADD, self.ADDEX):
            evaluated = processor.enter_subformula(sym)
            if evaluated == sym:
                out.append(sym)
                return out
            return self.add_slist(out, processor, notation, evaluated)
        f = notation.getf(sym, Notation.S_LIST)
        if f is None:
            out.append(sym)
            return out
        for arg in f.args:
            expr = arg
            negative = False
            f = notation.vgetf(expr, [Notation.PLUS, Notation.MINUS])
            if f is not None:
                if f.sym == Notation.MINUS:
                    negative = True
                expr = f.args[0]
            f = notation.getf(expr, Notation.GROUP)
            if f is not None:
                if negative:
                    evaluated = processor.enter_subformula(f.args[0])
                    _, evaluated = self._unwrap_term(notation, evaluated)
                    slist = notation.getf(evaluated, Notation.S_LIST)
                    if slist is not None:
                        for idx, term in enumerate(slist.args):
                            term_neg = False
                            t = term
                            fplus = notation.getf(t, Notation.PLUS)
                            if fplus is not None:
                                t = fplus.args[0]
                            fminus = notation.getf(t, Notation.MINUS)
                            if fminus is not None:
                                term_neg = True
                                t = fminus.args[0]
                            t = notation.setf(Notation.MINUS, (t,)) if not term_neg else t
                            if idx > 0:
                                t = notation.setf(Notation.PLUS, (t,))
                            out.append(t)
                    else:
                        mul = chainexpr(Mul.MUL, notation, expr, negative=True)
                        group = notation.setf(Notation.GROUP, (mul,), br="()")
                        # Preserve parentheses for grouped negative terms
                        out.append(group if len(out) == 0 else notation.setf(Notation.PLUS, (group,)))
                else:
                    self.add_slist(out, processor, notation, f.args[0])
                continue
            f = notation.getf(expr, Notation.P_LIST)
            if f is not None and is_group_inside_plist(notation, f):
                mul = chainexpr(Mul.MUL, notation, expr, negative)
                out.append(
                    notation.setf(Notation.PLUS, (mul,))
                )           
                continue
            if negative:
                expr = notation.setf(Notation.MINUS, (expr,))
            elif len(out) > 0:
                expr = notation.setf(Notation.PLUS, (expr,))
            out.append(expr)
        return out

    def add_fractions(self, processor, notation, sym):
        """
        Handle fraction addition patterns (Rules 1 and 5).

        Scans S_LIST for fractions and combines them.
        Returns modified expression with fractions combined.
        """
        # Unwrap GROUP if present
        f = notation.getf(sym, Notation.GROUP)
        if f is not None:
            sym = f.args[0]

        # Check if this is an S_LIST
        f = notation.getf(sym, Notation.S_LIST)
        if f is None:
            return sym  # Not a sum, nothing to combine

        terms = list(f.args)
        if len(terms) < 2:
            return sym  # Need at least 2 terms to combine

        # Pre-normalize product terms into fractions where possible
        for idx, term in enumerate(terms):
            sign, core = self._unwrap_term(notation, term)
            converted = self._product_to_fraction(notation, core, sign)
            if converted is not None:
                terms[idx] = converted

        # Scan for fractions and scalars
        # We'll process terms pairwise for simplicity (Rule 1 and Rule 5)
        modified = False
        i = 0
        while i < len(terms) - 1:
            sign1, term1 = self._unwrap_term(notation, terms[i])
            sign2, term2 = self._unwrap_term(notation, terms[i+1])

            # Try to expose fractions from products
            converted1 = self._product_to_fraction(notation, term1, sign1)
            term1 = converted1 if converted1 is not None else self._apply_sign(notation, self._eval_if_command(processor, notation, term1), sign1)
            converted2 = self._product_to_fraction(notation, term2, sign2)
            term2 = converted2 if converted2 is not None else self._apply_sign(notation, self._eval_if_command(processor, notation, term2), sign2)
            terms[i] = term1
            terms[i + 1] = term2

            # Rule 1: Fraction + Fraction
            if is_frac(notation, term1) and is_frac(notation, term2):
                combined = self._combine_two_fractions(notation, term1, term2)
                terms[i] = combined
                del terms[i+1]
                modified = True
                continue

            # Rule 5a: Scalar + Fraction
            if not is_frac(notation, term1) and is_frac(notation, term2):
                combined = self._combine_scalar_fraction(notation, term1, term2)
                terms[i] = combined
                del terms[i+1]
                modified = True
                continue

            # Rule 5b: Fraction + Scalar
            if is_frac(notation, term1) and not is_frac(notation, term2):
                combined = self._combine_scalar_fraction(notation, term2, term1)
                terms[i] = combined
                del terms[i+1]
                modified = True
                continue

            i += 1

        if not modified:
            return sym

        # Rebuild S_LIST with combined terms
        if len(terms) == 1:
            return terms[0]
        return notation.setf(Notation.S_LIST, tuple(terms))

    def _unwrap_term(self, notation, term):
        """
        Unwrap term from PLUS/MINUS/GROUP wrappers.
        Returns tuple (sign, core symbol).
        """
        sign = 1
        while True:
            f = notation.vgetf(term, [Notation.PLUS, Notation.MINUS, Notation.GROUP])
            if f is None:
                break
            if f.sym == Notation.PLUS:
                term = f.args[0]
            elif f.sym == Notation.MINUS:
                sign *= -1
                term = f.args[0]
            else:
                term = f.args[0]
        return sign, term

    def _apply_sign(self, notation, term, sign):
        """
        Apply sign to term, pushing minus into fraction numerator when needed.
        """
        if sign >= 0:
            return term
        if is_frac(notation, term):
            num = get_numerator(notation, term)
            denom = get_denominator(notation, term)
            num_signed = notation.setf(Notation.MINUS, (num,))
            return normalize_frac(notation, num_signed, denom, chainexpr, Mul.MUL)
        return notation.setf(Notation.MINUS, (term,))

    def _eval_if_command(self, processor, notation, term):
        """
        If term is a command (e.g., mul!), evaluate it once to expose fractions.
        Skips add!/addex! to avoid recursion.
        """
        f_cmd = notation.get(term)
        if f_cmd is not None and isinstance(f_cmd.sym, Symbol):
            if f_cmd.sym.name.endswith('!') and f_cmd.sym not in (self.ADD, self.ADDEX):
                return processor.enter_subformula(term)
            if f_cmd.sym == Notation.P_LIST:
                mul_sym = chainexpr(Mul.MUL, notation, term, None)
                evaluated = processor.enter_subformula(mul_sym)
                return evaluated if evaluated is not None else term
        return term

    def _product_to_fraction(self, notation, term, sign=1):
        """
        If term is a simple product containing fractions/scalars, convert it to a fraction.
        """
        f = notation.getf(term, Notation.P_LIST)
        if f is None or len(f.args) != 2:
            return None
        a, b = f.args

        def unwrap_group(x):
            g = notation.getf(x, Notation.GROUP)
            return g.args[0] if g is not None else x

        a = unwrap_group(a)
        b = unwrap_group(b)

        if is_frac(notation, a) and is_frac(notation, b):
            x1 = get_numerator(notation, a)
            y1 = get_denominator(notation, a)
            x2 = get_numerator(notation, b)
            y2 = get_denominator(notation, b)
            num_plist = notation.setf(Notation.P_LIST, (x1, x2))
            num_mul = chainexpr(Mul.MUL, notation, num_plist, None)
            denom_plist = notation.setf(Notation.P_LIST, (y1, y2))
            denom_mul = chainexpr(Mul.MUL, notation, denom_plist, None)
            res = normalize_frac(notation, num_mul, denom_mul, chainexpr, Mul.MUL)
            return self._apply_sign(notation, res, sign)

        if is_frac(notation, a) and not is_frac(notation, b):
            x = get_numerator(notation, a)
            y = get_denominator(notation, a)
            num_plist = notation.setf(Notation.P_LIST, (b, x))
            num_mul = chainexpr(Mul.MUL, notation, num_plist, None)
            res = normalize_frac(notation, num_mul, y, chainexpr, Mul.MUL)
            return self._apply_sign(notation, res, sign)

        if not is_frac(notation, a) and is_frac(notation, b):
            x = get_numerator(notation, b)
            y = get_denominator(notation, b)
            num_plist = notation.setf(Notation.P_LIST, (a, x))
            num_mul = chainexpr(Mul.MUL, notation, num_plist, None)
            res = normalize_frac(notation, num_mul, y, chainexpr, Mul.MUL)
            return self._apply_sign(notation, res, sign)

        return None

    def _combine_two_fractions(self, notation, frac1, frac2):
        """
        Rule 1: \\frac{x1}{y1} + \\frac{x2}{y2} → \\frac{\\add!{\\mul!{x1·y2} + \\mul!{x2·y1}}}{\\mul!{y1·y2}}
        """
        x1 = get_numerator(notation, frac1)
        y1 = get_denominator(notation, frac1)
        x2 = get_numerator(notation, frac2)
        y2 = get_denominator(notation, frac2)

        # Build: \mul!{x1·y2}
        x1_y2_plist = notation.setf(Notation.P_LIST, (x1, y2))
        x1_y2_mul = chainexpr(Mul.MUL, notation, x1_y2_plist, None)

        # Build: \mul!{x2·y1}
        x2_y1_plist = notation.setf(Notation.P_LIST, (x2, y1))
        x2_y1_mul = chainexpr(Mul.MUL, notation, x2_y1_plist, None)

        # Build: \add!{\mul!{x1·y2} + \mul!{x2·y1}}
        num_slist = notation.setf(Notation.S_LIST, (
            notation.setf(Notation.PLUS, (x1_y2_mul,)),
            x2_y1_mul
        ))
        num_add = chainexpr(self.ADD, notation, num_slist, None)

        # Build: \mul!{y1·y2}
        denom_plist = notation.setf(Notation.P_LIST, (y1, y2))
        denom_mul = chainexpr(Mul.MUL, notation, denom_plist, None)

        # Normalize and return
        return normalize_frac(notation, num_add, denom_mul, chainexpr, Mul.MUL)

    def _combine_scalar_fraction(self, notation, scalar, frac):
        """
        Rule 5: a + \\frac{x}{y} → \\frac{\\add!{\\mul!{a·y} + x}}{y}
        """
        x = get_numerator(notation, frac)
        y = get_denominator(notation, frac)

        # Build: \mul!{a·y}
        a_y_plist = notation.setf(Notation.P_LIST, (scalar, y))
        a_y_mul = chainexpr(Mul.MUL, notation, a_y_plist, None)

        # Build: \add!{\mul!{a·y} + x}
        num_slist = notation.setf(Notation.S_LIST, (
            notation.setf(Notation.PLUS, (a_y_mul,)),
            x
        ))
        num_add = chainexpr(self.ADD, notation, num_slist, None)

        # Normalize and return
        return normalize_frac(notation, num_add, y, chainexpr, Mul.MUL)


def create_actions():
    return {"add": Add(False), "addex": Add(True)}

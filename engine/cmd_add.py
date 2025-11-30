from engine.helpers import trace_notation
from value import *
from replicator import Replicator
from comparer import pattern, NotationParam
from cmd_mul import Mul, chainexpr
from frac_utils import is_frac, get_numerator, get_denominator, normalize_frac

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
            pass
        else:
            repl = Replicator(processor.notation, processor.output_notation)
            outsym = repl.enter_subformula(sym)
        return self.main(processor, processor.output_notation, outsym)

    def main(self, processor, notation, sym):
        # Combine fractions before flattening
        sym = self.add_fractions(processor, notation, sym)

        out = self.add_slist([], notation, sym)
        if len(out) == 1:
            return out[0]
        sym = notation.setf(Notation.S_LIST, tuple(out))
        return sym
    
    def add_slist(self, out, notation, sym):
        f = notation.getf(sym, Notation.GROUP)
        if f is not None:
            return self.add_slist(out, notation, f.args[0])
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
                    mul = chainexpr(Mul.MUL, notation, expr, negative=True)
                    group = notation.setf(Notation.GROUP, (mul,), br="()")    
                    out.append(
                        notation.setf(Notation.PLUS, (group,))
                    )                                       
                else:
                    self.add_slist(out, notation, f.args[0])
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

        # Scan for fractions and scalars
        # We'll process terms pairwise for simplicity (Rule 1 and Rule 5)
        modified = False
        i = 0
        while i < len(terms) - 1:
            term1 = self._unwrap_term(notation, terms[i])
            term2 = self._unwrap_term(notation, terms[i+1])

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
        Returns the core symbol.
        """
        # Remove PLUS wrapper
        f = notation.getf(term, Notation.PLUS)
        if f is not None:
            term = f.args[0]

        # Remove GROUP wrapper
        f = notation.getf(term, Notation.GROUP)
        if f is not None:
            term = f.args[0]

        # NOTE: Keep MINUS for now, handle negative fractions separately
        return term

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

from engine.helpers import trace_notation
from value import *
from replicator import Replicator
from comparer import pattern, NotationParam
from cmd_mul import Mul, chainexpr

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


def create_actions():
    return {"add": Add(False), "addex": Add(True)}

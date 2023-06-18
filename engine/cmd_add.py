from value import *
from replicator import Replicator
from comparer import pattern, NotationParam


def chainexpr(oper, notation, sym):
    inner = notation.setf(oper, (None, (sym,)))
    return notation.setf(Notation.GROUP, (inner,), br='{}')


def tailexpr(oper, notation, f):
    rest = f.args[1:]
    f = notation.vgetf(rest[0], [Notation.PLUS, Notation.MINUS])
    if f is not None:
        rest[0] = f.args[0]
        return notation.setf(f.sym,
                             (chainexpr(oper, notation, notation.setf(Notation.S_LIST, tuple(rest))),))
    return chainexpr(oper, notation, notation.setf(Notation.S_LIST, tuple(rest)))


class Add(object):
    arity = 1
    ADD = Symbol('add!')
    ADDEX = Symbol('addex!')

    def __init__(self, active):
        self.active = active

    def exec(self, processor, sym, f):
        return self.run(processor, f.args[1][0])

    def is_eval(self, notation, sym):
        f = notation.getf(sym, Notation.GROUP)
        if f is not None:
            f = notation.vgetf(f.args[0], [self.ADD, self.ADDEX])
            if f is not None:
                return True
        return False

    def run(self, processor, sym):
        if self.active:
            outsym = processor.enter_subformula(sym)
            pass
        else:
            repl = Replicator(processor.notation, processor.output_notation)
            outsym = repl.enter_subformula(sym)
        return self.main(processor, processor.output_notation, outsym)

    def main(self, processor, notation, sym):
        return self.add_plist(processor, notation, sym)

    def add_plist(self, processor, notation, sym):
        entry_sym = sym
        f = notation.getf(sym, Notation.S_LIST)
        if f is None:
            return sym
        if len(f.args) > 2:
            inner = tailexpr(self.ADD, notation, f)
            outer = notation.setf(Notation.S_LIST, (f.args[0], inner))
            return chainexpr(self.ADDEX, notation, outer)
        if self.is_eval(notation, f.args[0]) or self.is_eval(notation, f.args[1]):
            return chainexpr(self.ADDEX, notation, entry_sym)
        return sym


def create_actions():
    return {'add': Add(False),
            'addex': Add(True)}

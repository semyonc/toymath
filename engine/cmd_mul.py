from value import *
from replicator import Replicator
from comparer import pattern, NotationParam


class Mul(object):
    arity = 1
    MUL = Symbol('mul!')
    MULEX = Symbol('mulex!')
    Pw1 = pattern('(z)^n', [('z', NotationParam.Any),
                            ('n', NotationParam.N)])

    def __init__(self, active):
        self.active = active

    def exec(self, processor, sym, f):
        return self.run(processor, f.args[1][0])

    def extract(self, notation, sym):
        f = notation.vgetf(sym, [Notation.PLUS, Notation.GROUP])
        if f is not None:
            return self.extract(notation, f.args[0])
        f = notation.getf(sym, Notation.S_LIST)
        if f is not None:
            return f.args
        return [sym]

    def eval(self, oper, notation, sym):
        inner = notation.setf(oper, (None, (sym,)))
        return notation.setf(Notation.GROUP, (inner,), br='{}')

    def is_eval(self, notation, sym):
        f = notation.getf(sym, Notation.GROUP)
        if f is not None:
            f = notation.vgetf(f.args[0], [self.MUL, self.MULEX])
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
        subst = Mul.Pw1.match(sym, notation)
        if subst is not None:
            return self.power(processor, notation, subst['z'], subst['n'])
        return self.multiplay_plist(processor, notation, sym)

    def power(self, processor, notation, expr, n):
        args = []
        outsym = notation.setf(Notation.GROUP, (expr,), br='()')
        for i in range(n.val):
            args.append(outsym)
        sym_plist = notation.setf(Notation.P_LIST, tuple(args))
        return notation.setf(self.MUL, (None, (sym_plist,)))

    def multiplay_plist(self, processor, notation, sym):
        entry_sym = sym
        negative = False
        f = notation.getf(sym, Notation.MINUS)
        if f is not None:
            sym = f.args[0]
            negative = True
        f = notation.getf(sym, Notation.P_LIST)
        if f is None:
            return sym
        if len(f.args) > 2:
            rest = notation.setf(Notation.P_LIST, tuple(f.args[1:]))
            inner = self.eval(self.MUL, notation, rest)
            outer = notation.setf(Notation.P_LIST, (f.args[0], inner))
            return self.eval(self.MULEX, notation, outer)
        if self.is_eval(notation, f.args[0]) or self.is_eval(notation, f.args[1]):
            return self.eval(self.MULEX, notation, entry_sym)
        res = []
        x = self.extract(notation, f.args[0])
        y = self.extract(notation, f.args[1])
        if not self.active and \
                (x == [f.args[0]] or y == [f.args[1]]):
            if x != [f.args[0]]:
                res.append(f.args[0])
            else:
                res.append(self.eval(self.MUL, notation, f.args[0]))
            if y != [f.args[1]]:
                res.append(f.args[1])
            else:
                res.append(self.eval(self.MUL, notation, f.args[1]))
            outer = notation.setf(Notation.P_LIST, tuple(res))
            return self.eval(self.MULEX, notation, outer)
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
                    outs = IntegerValue(1)
                elif len(pl) == 1:
                    outs = pl[0]
                else:
                    outs = notation.setf(Notation.P_LIST, tuple(pl))
                if less_value(factor, 0):
                    outs = notation.setf(Notation.MINUS, (outs,))
                elif len(res) > 0:
                    outs = notation.setf(Notation.PLUS, (outs,))
                res.append(outs)

        return notation.setf(Notation.S_LIST, tuple(res))


def create_actions():
    return {'mul': Mul(False),
            'mulex': Mul(True)}

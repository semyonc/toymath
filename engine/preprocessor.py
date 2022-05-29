from replacer import Replacer
from replicator import Replicator
from notation import Func, issym
from value import *


class Importer(Replicator):

    def __init__(self, notation, output_notation):
        super(Importer, self).__init__(notation, output_notation)
        self.symmap = {}

    def mapsym(self, sym):
        newsym = self.symmap.get(sym)
        if newsym is None:
            newsym = Symbol()
            self.symmap[sym] = newsym
        return newsym


class Preprocessor(Replacer):

    def __init__(self, notation, output_notation, execution_history, history):
        super(Preprocessor, self).__init__(notation, output_notation)
        self.execution_history = execution_history
        self.history = history

    def enter_backref(self, sym, f):
        refs = self.execution_history[f.args[0]]
        input_notation = self.history[refs]
        repl = Importer(input_notation, self.output_notation)
        linked_sym = repl.enter_subformula(refs)
        return self.subst(sym, linked_sym, self.context())

    def enter_oper(self, sym, f):
        args = [self.enter_expr(expr) for expr in f.args]
        if f.sym.name in ['\\frac', '\\dfrac', '\\cfrac', '\\tfrac'] and \
                all(isinstance(n, IntegerValue) for n in args):
            return division(args[0].get_frac(), args[1].get_frac())
        return self.output_notation.repf(self.mapsym(sym), Func(f.sym, tuple(args)))

    def enter_group(self, sym, f):
        outs = self.enter_formula(f.args[0])
        if isinstance(outs, Value):
            return outs
        return self.output_notation.repf(self.mapsym(sym), Func(f.sym, (outs,), **f.props))

    @staticmethod
    def anyname(sym, names):
        return any(isinstance(sym, Symbol) and sym.name == t for t in names)

    def transform_plist(self, sym, args):
        res = []
        i = 0
        while i < len(args):
            if i < len(args) - 1:
                if isinstance(args[i], IntegerValue) and isinstance(args[i + 1], FracValue):
                    res.append(addition(args[i].get_frac(), args[i + 1]))
                    i += 2
                    continue
                if issym(args[i], ["d"]) and issym(args[i + 1], ['x', 'y', 'z', 't']):
                    res.append(Symbol(args[i].name + args[i + 1].name))
                    i += 2
                    continue
                pri = args[i]
                index_f = self.output_notation.getf(pri, Notation.INDEX)
                limits_f = self.output_notation.getf(pri, Notation.LIMITS)
                if index_f is not None and index_f.args[1][0] is None and index_f.args[1][1] is None:
                    pri = index_f.args[0]
                if limits_f is not None:
                    pri = limits_f.args[0]
                sec = args[i + 1]
                group_f = self.output_notation.getf(sec, Notation.GROUP)
                if group_f is not None:
                    sec = group_f.args[0]
                if self.anyname(pri, Notation.unary_f):
                    res.append(self.output_notation.setf(Notation.FUNC, (args[i], sec), fmt='unary'))
                    i += 2
                    continue
                if group_f is not None and self.anyname(pri, Notation.common_f):
                    res.append(self.output_notation.setf(Notation.FUNC, (args[i], sec)))
                    i += 2
                    continue
                if group_f is None and self.anyname(pri, Notation.p_oper):
                    res.append(self.output_notation.setf(Notation.FUNC,
                                                         (args[i], self.transform_plist(None, args[i + 1:])),
                                                         fmt='oper'))
                    break
            res.append(args[i])
            i += 1
        if len(res) == 1:
            return res[0]
        return self.output_notation.repf(sym, Func(Notation.P_LIST, res))

    def enter_plist(self, sym, f):
        args = self.build_list(f, self.enter_additive_expr)
        return self.transform_plist(self.mapsym(sym), args)

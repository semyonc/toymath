from replacer import Replacer
from replicator import Replicator
from notation import Func, issym
from value import *
from comparer import isVariable


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
        key = str(f.args[0])
        if key not in self.execution_history:
            return super(Preprocessor, self).enter_backref(sym, f)
        refs = self.execution_history[key]
        input_notation = self.notation
        if refs in self.history:
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
        return self.output_notation.repf(self.mapsym(sym), Func(f.sym, (outs,), **f.props))

    @staticmethod
    def anyname(sym, names):
        return any(isinstance(sym, Symbol) and sym.name == t for t in names)

    @staticmethod
    def funcname(sym):
        return isinstance(sym, Symbol) and sym.name.startswith('\\') and sym.name not in Notation.reserved

    def extract_args(self, args):
        ret = []
        for i, arg in enumerate(args):
            f = self.output_notation.getf(arg, Notation.GROUP)
            if f is not None:
                if i == 0:
                    ret.append(arg)
                break
            if self.anyname(arg, Notation.unary_f) or \
                    self.anyname(arg, Notation.common_f) or \
                    self.anyname(arg, Notation.p_oper) or \
                    self.funcname(arg):
                break
            ret.append(arg)
        return ret

    def transform_prefix(self, plist):
        res = []
        i = 0
        while i < len(plist):
            if i < len(plist) - 1:
                if isinstance(plist[i], IntegerValue) and isinstance(plist[i + 1], FracValue):
                    res.append(addition(plist[i].get_frac(), plist[i + 1]))
                    i += 2
                    continue
                if issym(plist[i], ["d"]) and issym(plist[i + 1], ['x', 'y', 'z', 't']):
                    res.append(Symbol(plist[i].name + plist[i + 1].name))
                    i += 2
                    continue
                pri = plist[i]
                index_f = self.output_notation.getf(pri, Notation.INDEX)
                limits_f = self.output_notation.getf(pri, Notation.LIMITS)
                if index_f is not None and index_f.args[1][0] is None and index_f.args[1][1] is None:
                    pri = index_f.args[0]
                if limits_f is not None:
                    pri = limits_f.args[0]
                sec = plist[i + 1]
                group_f = self.output_notation.getf(sec, Notation.GROUP)
                if group_f is not None:
                    sec = group_f.args[0]
                if self.anyname(pri, Notation.unary_f):
                    args = self.extract_args(plist[i + 1:])
                    if len(args) > 1:
                        sec = self.output_notation.setf(Notation.P_LIST, tuple(args))
                    if len(args) > 0:
                        res.append(self.output_notation.setf(Notation.FUNC, (plist[i], sec), fmt='unary'))
                        i += 1 + len(args)
                        continue
                if group_f is not None and self.anyname(pri, Notation.common_f):
                    res.append(self.output_notation.setf(Notation.FUNC, (plist[i], sec)))
                    i += 2
                    continue
                if group_f is None and self.anyname(pri, Notation.p_oper):
                    res.append(self.output_notation.setf(Notation.FUNC,
                                                         (plist[i], self.transform_plist(None, plist[i + 1:])),
                                                         fmt='oper'))
                    break
                if group_f is not None and self.funcname(pri):
                    res.append(self.output_notation.setf(Notation.FUNC, (Symbol(pri.name[1:]), sec),
                                                         fmt='operatorname'))
                    i += 2
                    continue
            res.append(plist[i])
            i += 1
        return res

    def transform_suffix(self, plist):
        res = []
        i = 0
        while i < len(plist):
            if i < len(plist) - 1:
                pri = plist[i]
                index1_f = self.output_notation.getf(pri, Notation.INDEX)
                if index1_f is not None:
                    pri = index1_f.args[0]
                sec = plist[i + 1]
                index2_f = self.output_notation.getf(sec, Notation.INDEX)
                if index2_f is not None and self.anyname(pri, Notation.common_f):
                    sec = index2_f.args[0]
                    group_f = self.output_notation.getf(index2_f.args[0], Notation.GROUP)
                    if group_f is not None:
                        sec = group_f.args[0]
                    res.append(self.output_notation.setf(Notation.INDEX,
                                                         (self.output_notation.setf(Notation.FUNC, (plist[i], sec)),
                                                          index2_f.args[1])))
                    i += 2
                    continue
            res.append(plist[i])
            i += 1
        return res

    def transform_plist(self, sym, args):
        res = self.transform_prefix(args)
        res = self.transform_suffix(res)
        if len(res) == 1:
            return res[0]
        return self.output_notation.repf(sym, Func(Notation.P_LIST, res))

    def enter_plist(self, sym, f):
        args = self.build_list(f, self.enter_expr)
        return self.transform_plist(self.mapsym(sym), args)

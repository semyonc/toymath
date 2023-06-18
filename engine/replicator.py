#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 17:18:23 2021
@author: semyonc
"""

from notation import Notation, Func, Symbol
from contextlib import contextmanager


# def generator(callback):
#     def closure(self, *args):
#         sym = self.enter_rel(args[0])
#         if sym is not None:
#             return sym
#         return callback(self, *args)
#
#     return closure

class Replicator(object):
    """Replicator"""

    namemap = {
        '\\above': 'enter_above',
        '\\abovewithdelims': 'enter_abovewithdelims',
        '\\atopwithdelims': 'enter_atopwithdelims',
        '\\atop': 'enter_binaryop',
        '\\brace': 'enter_binaryop',
        '\\brack': 'enter_binaryop',
        '\\color': 'enter_color',
        '\\lower': 'enter_lower',
        '\\sqrt': 'enter_sqrt',
        '\\buildrel': 'enter_buildrel',
        'func': 'enter_func',
        'group': 'enter_group',
        'vgroup': 'enter_vgroup',
        'sgroup': 'enter_sgroup',
        '\\array': 'enter_array',
        '\\cases': 'enter_array',
        '\\dashv': 'enter_binaryop'
    }

    def __init__(self, notation, output_notation):
        self.notation = notation
        self.output_notation = output_notation
        self.stack = []

    # noinspection PyMethodMayBeStatic
    def mapsym(self, sym):
        return sym

    @contextmanager
    def _enter(self, sym, f):
        try:
            self.stack.append((sym, f))
            yield sym
        finally:
            del self.stack[-1]

    def _probe(self, sym, name):
        f = self.notation.getf(sym, Symbol(name))
        if f is not None:
            with self._enter(sym, f):
                return getattr(self.__class__, Replicator.namemap[name])(self, sym, f)
        return None

    def _probe_command(self, sym):
        f = self.notation.get(sym)
        if f is not None and f.sym.name.endswith('!'):
            with self._enter(sym, f):
                return self.enter_command(sym, f)
        return None

    def context_sym(self):
        if len(self.stack) <= 1:
            return None
        return self.stack[-2][0]

    def parent_f(self):
        if len(self.stack) <= 1:
            return None
        return self.stack[-2][1]

    def context(self):
        f = self.parent_f()
        if f is None:
            return None
        return f.sym

    def __call__(self, sym):
        return self.enter_formula(sym)

    def enter_formula(self, sym):
        res = self._probe_command(sym)
        if res is not None:
            return res
        res = self._probe(sym, '\\above')
        if res is not None:
            return res
        res = self._probe(sym, '\\abovewithdelims')
        if res is not None:
            return res
        res = self._probe(sym, '\\atop')
        if res is not None:
            return res
        res = self._probe(sym, '\\atopwithdelims')
        if res is not None:
            return res
        res = self._probe(sym, '\\brace')
        if res is not None:
            return res
        res = self._probe(sym, '\\brack')
        if res is not None:
            return res
        res = self._probe(sym, '\\dashv')
        if res is not None:
            return res
        return self.enter_subformula(sym)

    def enter_subformula(self, sym):
        f = self.notation.getf(sym, Notation.COMP)
        if f is not None:
            with self._enter(sym, f):
                return self.enter_subformula_comparison(sym, f)
        else:
            return self.enter_comma_list(sym)

    def enter_comma_list(self, sym):
        f = self.notation.getf(sym, Notation.C_LIST)
        if f is not None:
            with self._enter(sym, f):
                return self.enter_clist(sym, f)
        else:
            return self.enter_additive_expr_list(sym)

    def enter_additive_expr_list(self, sym):
        f = self.notation.getf(sym, Notation.S_LIST)
        if f is not None:
            with self._enter(sym, f):
                return self.enter_slist(sym, f)
        else:
            return self.enter_additive_expr(sym)

    def enter_additive_expr(self, sym):
        f = self.notation.get(sym)
        if f is not None and f.sym.name in Notation.additive:
            with self._enter(sym, f):
                return self.enter_additive(sym, f)
        else:
            return self.enter_composite_expr(sym)

    def enter_composite_expr(self, sym):
        f = self.notation.getf(sym, Notation.P_LIST)
        if f is not None:
            with self._enter(sym, f):
                return self.enter_plist(sym, f)
        else:
            f = self.notation.getf(sym, Notation.SLASH)
            if f is not None:
                with self._enter(sym, f):
                    return self.enter_slashExpr(sym, f)
            else:
                f = self.notation.getf(sym, Notation.STAR)
                if f is not None:
                    with self._enter(sym, f):
                        return self.enter_starExpr(sym, f)
                else:
                    return self.enter_expr(sym)

    def enter_expr(self, sym):
        res = self._probe(sym, '\\color')
        if res is not None:
            return res
        res = self._probe(sym, '\\lower')
        if res is not None:
            return res
        res = self._probe(sym, '\\buildrel')
        if res is not None:
            return res
        res = self._probe(sym, '\\sqrt')
        if res is not None:
            return res
        f = self.notation.get(sym)
        if f is not None:
            if f.sym == Notation.INDEX:
                with self._enter(sym, f):
                    return self.enter_index(sym, f)
            elif f.sym == Notation.LIMITS:
                with self._enter(sym, f):
                    return self.enter_limits(sym, f)
            elif f.sym == Notation.NOLIMITS:
                with self._enter(sym, f):
                    return self.enter_nolimits(sym, f)
            elif f.sym.name in Notation.oper:
                with self._enter(sym, f):
                    return self.enter_oper(sym, f)
            else:
                return self.enter_scalar(sym)
        else:
            if isinstance(sym, Symbol):
                return self.enter_symbol(sym)
            else:
                return self.enter_term(sym)

    def enter_scalar(self, sym):
        res = self._probe(sym, "group")
        if res is not None:
            return res
        res = self._probe(sym, "vgroup")
        if res is not None:
            return res
        res = self._probe(sym, "sgroup")
        if res is not None:
            return res
        res = self._probe(sym, "\\array")
        if res is not None:
            return res
        res = self._probe(sym, "\\cases")
        if res is not None:
            return res
        res = self._probe(sym, 'func')
        if res is not None:
            return res
        f = self.notation.get(sym)
        if f is not None:
            if f.sym.name.startswith('\\text'):
                with self._enter(sym, f):
                    return self.enter_text(sym, f)
            elif f.sym == Notation.REF:
                with self._enter(sym, f):
                    return self.enter_backref(sym, f)
        return self.enter_term(sym)

    # noinspection PyMethodMayBeStatic
    def enter_symbol(self, sym):
        return sym

    def enter_term(self, sym):
        f = self.notation.getf(sym, Notation.PRIME)
        if f is not None:
            with self._enter(sym, f):
                return self.enter_prime(sym, f)
        else:
            return self.enter_raw_term(sym)

    # noinspection PyMethodMayBeStatic
    def enter_raw_term(self, t):
        return t

    def build_list(self, f, callback):
        outlist = []
        for expr in f.args:
            outsym = callback(expr)
            outf = self.output_notation.getf(outsym, f.sym)
            if outf is not None:
                outlist += outf.args
                del self.output_notation.rel[outsym]
            else:
                outlist.append(outsym)
        return outlist

    def enter_dims(self, f):
        outdims = []
        for dim in f.args[1]:
            if dim is not None:
                outdims.append(self.enter_scalar(dim))
            else:
                outdims.append(None)
        return tuple(outdims)

    def enter_command(self, sym, f):
        comma_list = f.args[0]
        if comma_list is not None:
            comma_list = self.enter_comma_list(comma_list)
        return self.output_notation.repf(self.mapsym(sym),
                                         Func(f.sym, (comma_list, tuple(self.enter_subformula(arg) for arg in f.args[1]))))

    def enter_prime(self, sym, f):
        return self.output_notation.repf(self.mapsym(sym), Func(Notation.PRIME, (self.enter_term(f.args[0]),)))

    def enter_subformula_comparison(self, sym, f):
        return self.output_notation.repf(self.mapsym(sym), Func(f.sym,
                                                                (self.enter_additive_expr(f.args[0]),
                                                                 self.enter_comma_list(f.args[1]))))

    def enter_additive(self, sym, f):
        return self.output_notation.repf(self.mapsym(sym), Func(f.sym, (self.enter_composite_expr(f.args[0]),)))

    def enter_clist(self, sym, f):
        args = self.build_list(f, self.enter_additive_expr_list)
        return self.output_notation.repf(self.mapsym(sym), Func(Notation.C_LIST, args))

    def enter_slist(self, sym, f):
        args = self.build_list(f, self.enter_additive_expr)
        return self.output_notation.repf(self.mapsym(sym), Func(Notation.S_LIST, args))

    def enter_plist(self, sym, f):
        args = self.build_list(f, self.enter_expr)
        return self.output_notation.repf(self.mapsym(sym), Func(Notation.P_LIST, args))

    def enter_slashExpr(self, sym, f):
        return self.output_notation.repf(self.mapsym(sym),
                                         Func(f.sym, (self.enter_expr(f.args[0]), self.enter_expr(f.args[1]))))

    def enter_starExpr(self, sym, f):
        return self.output_notation.repf(self.mapsym(sym),
                                         Func(f.sym, (self.enter_expr(f.args[0]), self.enter_expr(f.args[1]))))

    def enter_index(self, sym, f):
        return self.output_notation.repf(self.mapsym(sym),
                                         Func(f.sym, (self.enter_scalar(f.args[0]), self.enter_dims(f))))

    def enter_limits(self, sym, f):
        return self.output_notation.repf(self.mapsym(sym),
                                         Func(f.sym, (self.enter_scalar(f.args[0]), self.enter_dims(f))))

    def enter_nolimits(self, sym, f):
        return self.output_notation.repf(self.mapsym(sym),
                                         Func(f.sym, (self.enter_scalar(f.args[0]), self.enter_dims(f))))

    def enter_oper(self, sym, f):
        args = [self.enter_expr(expr) for expr in f.args]
        return self.output_notation.repf(self.mapsym(sym), Func(f.sym, tuple(args)))

    def enter_text(self, sym, f):
        return self.output_notation.repf(self.mapsym(sym), f)

    def enter_backref(self, sym, f):
        return self.output_notation.repf(self.mapsym(sym), f)

    def enter_above(self, sym, f):
        return self.output_notation.repf(self.mapsym(sym), Func(f.sym,
                                                                (self.enter_subformula(f.args[0]),
                                                                 self.enter_subformula(f.args[1]))))

    def enter_abovewithdelims(self, sym, f):
        return self.output_notation.repf(self.mapsym(sym), Func(f.sym,
                                                                (self.enter_subformula(f.args[0]),
                                                                 self.enter_subformula(f.args[1]))))

    def enter_atopwithdelims(self, sym, f):
        return self.output_notation.repf(self.mapsym(sym), Func(f.sym,
                                                                (self.enter_subformula(f.args[0]),
                                                                 self.enter_subformula(f.args[1]))))

    def enter_binaryop(self, sym, f):
        return self.output_notation.repf(self.mapsym(sym), Func(f.sym,
                                                                (self.enter_subformula(f.args[0]),
                                                                 self.enter_subformula(f.args[1]))))

    def enter_func(self, sym, f):
        return self.output_notation.repf(self.mapsym(sym),
                                         Func(f.sym, (self.enter_expr(f.args[0]), self.enter_formula(f.args[1])),
                                              **f.props))

    def enter_color(self, sym, f):
        return self.output_notation.repf(self.mapsym(sym), Func(f.sym, (self.enter_expr(f.args[0]),)))

    def enter_lower(self, sym, f):
        return self.output_notation.repf(self.mapsym(sym), Func(f.sym, (self.enter_expr(f.args[0]),)))

    def enter_sqrt(self, sym, f):
        return self.output_notation.repf(self.mapsym(sym), Func(f.sym, (self.enter_expr(f.args[0]),)))

    def enter_buildrel(self, sym, f):
        return self.output_notation.repf(self.mapsym(sym),
                                         Func(f.sym, (self.enter_subformula(f.args[0]), self.enter_expr(f.args[1]))))

    def enter_group(self, sym, f):
        return self.output_notation.repf(self.mapsym(sym), Func(f.sym, (self.enter_formula(f.args[0]),), **f.props))

    def enter_vgroup(self, sym, f):
        return self.output_notation.repf(self.mapsym(sym), Func(f.sym, (self.enter_formula(f.args[0]),)), **f.props)

    def enter_sgroup(self, sym, f):
        args = [self.enter_expr(f.args[0])]
        if len(f.args) > 1:
            args.append(self.enter_subformula(f.args[1]))
        return self.output_notation.repf(self.mapsym(sym), Func(f.sym, tuple(args), **f.props))

    def enter_array(self, sym, f):
        return self.output_notation.repf(self.mapsym(sym), Func(f.sym, self.enter_rowlist(f.args)))

    def enter_rowlist(self, rowlist):
        return [self.enter_collist(row) for row in rowlist]

    def enter_collist(self, collist):
        return [self.enter_subformula(col) for col in collist]

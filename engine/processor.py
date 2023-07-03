#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 19:25:05 2021

@author: semyonc
"""
import os
import glob
import sys
import importlib

import comparer
from comparer import NotationParam
from notation import Func, issym
from replacer import Replacer
from preprocessor import Preprocessor
from value import *


def iterate(x):
    if isinstance(x, list) or isinstance(x, tuple):
        for t in x:
            yield t
    else:
        yield x


def get_value(sym, notation):
    if isinstance(sym, Value):
        return sym
    f = notation.getf(sym, Notation.GROUP)
    if f is not None:
        return get_value(f.args[0], notation)
    f = notation.getf(sym, Notation.MINUS)
    if f is not None:
        return multiplication(-1, get_value(f.args[0], notation))
    return None


def mul_factor(acc, val):
    key = value_type(val)
    if key not in acc:
        acc[key] = val
    else:
        acc[key] = multiplication(acc[key], val)


def get_factor_value(acc):
    if FloatValue.typeKey in acc:
        res = 1.0
        for key in acc:
            res = multiplication(res, acc[key].get_float())
    else:
        res = None
        for key in acc:
            if res is None:
                res = acc[key]
            else:
                res = multiplication(res, acc[key])
    return res


def is_spacer(sym):
    return issym(sym, ['\\!', '\\,', '\\:', '\\;', '\\;'])


class Calculator(Replacer):
    """Calculator"""

    abbreviated_minus = comparer.pattern('(+x)', ['x'])
    abbreviated_plus = comparer.pattern('(-x)', ['x'])

    def __init__(self, notation, output_notation, actions, prologModel):
        super(Calculator, self).__init__(notation, output_notation)
        self.actions = actions
        self.prologModel = prologModel

    def enter_command(self, sym, f):
        action_name = f.sym.name[:-1]
        if action_name not in self.actions:
            raise AttributeError(f"Command {action_name} is not defined in processor")
        action = self.actions[action_name]
        if hasattr(action, 'arity') and action.arity != len(f.args[1]):
            raise AttributeError(f'Command {action_name} should have {action.arity} parameters')
        return action.exec(self, sym, f)

    def get_factor(self, sym):
        f = self.output_notation.vgetf(sym, [Notation.PLUS, Notation.GROUP])
        if f is not None:
            return self.get_factor(f.args[0])
        f = self.output_notation.getf(sym, Notation.MINUS)
        if f is not None:
            return multiplication(-1, self.get_factor(f.args[0]))
        factor = get_value(sym, self.output_notation)
        if factor is not None:
            return factor
        f = self.output_notation.getf(sym, Notation.P_LIST)
        if f is not None:
            factor = get_value(f.args[0], self.output_notation)
            if factor is not None:
                return factor
        return IntegerValue(1)

    def get_expr(self, sym):
        f = self.output_notation.getf(sym, Notation.GROUP)
        if f is not None:
            f = self.output_notation.vgetf(f.args[0], [Notation.PLUS, Notation.MINUS])
            if f is not None:
                return self.get_expr(f.args[0])
        f = self.output_notation.vgetf(sym, [Notation.PLUS, Notation.MINUS])
        if f is not None:
            return self.get_expr(f.args[0])
        if isinstance(sym, Symbol):
            if get_value(sym, self.output_notation) is None:
                f = self.output_notation.getf(sym, Notation.P_LIST)
                if f is not None:
                    if get_value(f.args[0], self.output_notation) is not None:
                        return f.args[1:]
                    else:
                        return f.args
                return [sym]
            else:
                return None
        return None

    def make_plist(self, factor, expr):
        if expr is None:
            return factor
        else:
            args = [self.subst(None, t, Notation.P_LIST) for t in iterate(expr)]
            return self.output_notation.setf(Notation.P_LIST, [factor] + args)

    def get_degree(self, sym):
        f = self.output_notation.getf(sym, Notation.INDEX)
        if f is not None:
            return f.args[1][2]
        return IntegerValue(1)

    def get_basic_expr(self, sym):
        f = self.output_notation.getf(sym, Notation.INDEX)
        if f is not None and f.args[1][0] is None \
                and f.args[1][1] is None and f.args[1][3] is None:
            return f.args[0]
        return sym

    def make_degree(self, sym, deg):
        f = self.output_notation.getf(sym, Notation.INDEX)
        if f is not None:
            return self.output_notation.repf(sym, Func(Notation.INDEX, (f.args[0],
                                                                        (f.args[1][0],
                                                                         f.args[1][1],
                                                                         self.subst(None, deg, Notation.INDEX),
                                                                         f.args[1][3]))))
        else:
            return self.output_notation.setf(Notation.INDEX,
                                             (self.subst(None, sym, Notation.INDEX),
                                              (None, None, self.subst(None, deg, Notation.INDEX), None)))

    def make_sum(self, args):
        output_args = []
        for sym in args:
            f = self.output_notation.getf(sym, Notation.S_LIST)
            if f is not None:
                output_args += f.args
            else:
                f = self.output_notation.vgetf(sym, [Notation.PLUS, Notation.MINUS])
                if f is not None:
                    output_args.append(sym)
                else:
                    output_args.append(
                        self.output_notation.setf(Notation.PLUS, (self.subst(None, sym, Notation.S_LIST),)))
        return self.output_notation.setf(Notation.S_LIST, output_args)

    func_pattern = comparer.pattern('\\{{#T x,f(x)}\\}', [('#T', NotationParam.Term), 'f', 'x'])

    def enter_formula(self, sym):
        subst = self.func_pattern.match(sym, self.notation)
        if subst is not None:
            if '#T' in subst:
                expr, nm = subst['#T']
                if isinstance(nm, Symbol) and nm.name in Notation.unary_f:
                    return self.output_notation.setf(Notation.FUNC,
                                                     (self.enter_expr(expr), self.enter_formula(subst['x'])))
            else:
                return self.output_notation.setf(Notation.FUNC, (self.enter_formula(subst['f']),
                                                                 self.escape(None, '()',
                                                                             self.enter_formula(subst['x']))))
        return super(Calculator, self).enter_formula(sym)

    def enter_index(self, sym, f):
        outdims = self.enter_dims(f)
        scalar = self.enter_scalar(f.args[0])
        n = get_value(outdims[2], self.output_notation)
        if isinstance(n, IntegerValue):
            if equal_value(n, 0):
                return IntegerValue(1)
            if equal_value(n, 1):
                return scalar
            val = get_value(scalar, self.output_notation)
            if isinstance(val, IntegerValue) or isinstance(val, FracValue):
                val = val.power(n)
                return val
        return self.output_notation.repf(self.mapsym(sym), Func(f.sym, (scalar, outdims)))

    def enter_group(self, sym, f):
        outs = self.enter_formula(f.args[0])
        if isinstance(outs, Value):
            return outs
        if f.props['br'] == '()':
            f_out = self.output_notation.vgetf(outs, [Notation.P_LIST, Notation.INDEX])
            if f_out is not None:
                ctx = self.context()
                if ctx is not None and ctx == Notation.INDEX:
                    return self.output_notation.repf(self.mapsym(sym),
                                                     Func(Notation.GROUP, (outs,), br='{}'))
                return outs
        f_out = self.output_notation.getf(outs, Notation.GROUP)
        if f_out is not None and f_out.props['br'] == f.props['br']:
            return outs
        f_out = self.output_notation.vgetf(outs, [Notation.PLUS, Notation.MINUS])
        if f_out is not None:
            ctx = self.context()
            if ctx is None or ctx == Notation.S_LIST:
                return outs
        return self.output_notation.repf(self.mapsym(sym), Func(f.sym, (outs,), **f.props))

    def enter_plist(self, sym, f):
        args = self.build_list(f, self.enter_expr)
        middle_args = []
        acc = {}
        for arg in args:
            if is_spacer(arg):
                middle_args.append(arg)
                continue
            val = self.get_factor(arg)
            if val is not None:
                mul_factor(acc, val)
            left = self.get_expr(arg)
            if left is not None:
                middle_args += left
        i = 0
        output_args = []
        while i < len(middle_args):
            left = middle_args[i]
            if is_spacer(left):
                output_args.append(left)
                i = i + 1
                continue
            deg = [self.get_degree(left)]
            j = i + 1
            while j < len(middle_args):
                right = middle_args[j]
                if comparer.s_equal(left, self.output_notation, right, self.output_notation, ctx=lambda x: x != 2):
                    deg.append(self.get_degree(right))
                    del middle_args[j]
                else:
                    j = j + 1
            if len(deg) > 1:
                left = self.make_degree(left, self.make_sum(deg))
            output_args.append(left)
            i = i + 1
        factor = get_factor_value(acc)
        if equal_value(factor, 0):
            return IntegerValue(0)
        negative = less_value(factor, 0)
        if not_equal_value(factor, 1) and not_equal_value(factor, -1):
            output_args = [factor.abs()] + [self.subst(None, arg, Notation.P_LIST) for arg in output_args]
        if len(output_args) == 0:
            return IntegerValue(1)
        elif len(output_args) == 1:
            outs = output_args[0]
        else:
            outs = self.output_notation.repf(self.mapsym(sym), Func(Notation.P_LIST, output_args))
        if negative:
            outs = self.output_notation.setf(Notation.MINUS, (outs,))
            ctx = self.context()
            if ctx is not None and ctx != Notation.S_LIST:
                outs = self.escape(None, '()', outs)
        return outs

    def enter_additive(self, sym, f):
        composite_expr = self.enter_composite_expr(f.args[0])
        if f.sym == Notation.PLUS:
            parent_f = self.parent_f()
            if parent_f is None or not (self.context() == Notation.GROUP or
                                        parent_f.sym == Notation.S_LIST and parent_f.args.index(sym)) > 0:
                return composite_expr
            subst = self.abbreviated_plus.match(composite_expr, self.output_notation)
            if subst is not None:
                return self.output_notation.repf(self.mapsym(sym),
                                                 Func(Notation.MINUS, (self.subst(None, subst['x'], self.context()),)))
        else:
            subst = self.abbreviated_plus.match(composite_expr, self.output_notation)
            if subst is not None:
                return self.output_notation.repf(self.mapsym(sym),
                                                 Func(Notation.PLUS, (self.subst(None, subst['x'], self.context()),)))
            subst = self.abbreviated_minus.match(composite_expr, self.output_notation)
            if subst is not None:
                return self.output_notation.repf(self.mapsym(sym),
                                                 Func(Notation.MINUS, (self.subst(None, subst['x'], self.context()),)))
        return self.output_notation.repf(self.mapsym(sym), Func(f.sym, (composite_expr,)))

    def enter_slist(self, sym, f):
        i = 0
        args = self.build_list(f, self.enter_additive_expr)
        output_args = []
        while i < len(args):
            left = args[i]
            factor = self.get_factor(left)
            expr1 = self.get_expr(left)
            j = i + 1
            k = factor
            while j < len(args):
                right = args[j]
                expr2 = self.get_expr(right)
                if comparer.s_equal(expr1, self.output_notation, expr2, self.output_notation, ctx=Notation.S_LIST):
                    k = addition(k, self.get_factor(right))
                    del args[j]
                else:
                    j = j + 1
            if equal_value(factor, k):
                output_args.append(left)
            elif not equal_value(k, 0):
                negative = less_value(k, 0)
                res = self.make_plist(k.abs(), expr1)
                if not negative and output_args:
                    res = self.output_notation.setf(Notation.PLUS, (res,))
                elif negative:
                    res = self.output_notation.setf(Notation.MINUS, (res,))
                output_args.append(res)
            i = i + 1
        if not output_args:
            return IntegerValue(0)
        if len(output_args) == 1:
            return output_args[0]
        return self.output_notation.repf(self.mapsym(sym), Func(Notation.S_LIST, output_args))


class MathProcessor(object):
    """MathProcessor"""

    def __init__(self, **kwargs):
        self.trace = None
        self.actions = register_actions()
        from prolog import PrologModel
        self.prologModel = PrologModel()

    # create True in Notation
    @staticmethod
    def create_true(notation):
        return notation.setf(Symbol('\\textit'), (str(True),))

    def process_rule(self, sym, notation):
        from prolog import get_operator, Term, Rule
        f = notation.getf(sym, Notation.P_LIST)
        if f is not None and len(f.args) == 3 and \
                get_operator(f.args[0], notation) is not None and \
                f.args[1] == Symbol('\\dashv'):
            goals = []
            f2 = notation.getf(f.args[2], Notation.GROUP)
            if f2 is not None:
                f2 = notation.getf(f2.args[0], Notation.C_LIST)
                if f2 is not None:
                    for g in f2.args:
                        goals.append(Term(sym=g, notation=notation))
            else:
                goals.append(Term(sym=f.args[2], notation=notation))
            self.prologModel.add_rule(Rule(Term(sym=f.args[0], notation=notation), goals=goals))
            return MathProcessor.create_true(notation)
        f = get_operator(sym, notation)
        if f is not None:
            self.prologModel.add_rule(Rule(Term(sym=sym, notation=notation)))
            return MathProcessor.create_true(notation)
        return None

    def __call__(self, sym, notation, execution_history, history):
        if self.trace is not None:
            self.trace(sym, notation, 0)
        output_notation = Notation()
        preprocessor = Preprocessor(notation, output_notation, execution_history, history)
        sym = preprocessor(sym)
        notation = output_notation
        output_notation = Notation()
        parse_res = self.process_rule(sym, notation)
        if parse_res is not None:
            return parse_res, notation
        index = 1
        while True:
            calculator = Calculator(notation, output_notation, self.actions, self.prologModel)
            outs = calculator(sym)
            if comparer.s_equal(outs, output_notation, sym, notation):
                break
            notation = output_notation
            sym = outs
            if self.trace is not None:
                self.trace(sym, notation, index)
            output_notation = Notation()
            index = index + 1
        return outs, output_notation


def register_actions(*actions):
    paths = [os.path.join(os.path.dirname(os.path.abspath(__file__)))]

    action_files = []
    for action_dir in paths:
        sys.path.append(action_dir)
        action_files.extend(glob.glob(os.path.join(action_dir, "cmd_*.py")))

    res = {}
    for action in action_files:
        basename = os.path.basename(action)
        if basename == "__init__.py":
            continue
        if len(actions) == 0 or basename in actions:
            module = importlib.import_module(os.path.splitext(basename)[0])
            if hasattr(module, "create_actions"):
                res = {**res, **module.create_actions()}
    return res

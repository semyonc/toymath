#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 18:40:00 2021

@author: semyonc
"""
from collections import defaultdict
from LatexParser import MathParser
from LatexWriter import LaTexWriter
from value import *


class Scanner(LaTexWriter):

    def __init__(self, notation, *kwargs):
        super(Scanner, self).__init__(notation)
        self._counter = 0
        self.terms = []

    def _lock(self):
        self._counter = self._counter + 1

    def _unlock(self):
        self._counter = self._counter - 1

    def write_index(self, f):
        dims = f.args[1]
        self._lock()
        if dims[0] is not None:
            self.writeString('^')
            self.write_scalar(dims[0])
        if dims[1] is not None:
            self.writeString('_')
            self.write_scalar(dims[1])
        self._unlock()
        self.write_scalar(f.args[0])
        self._lock()
        if dims[3] is not None:
            self.writeString('_')
            self.write_scalar(dims[3])
        if dims[2] is not None:
            self.writeString('^')
            self.write_scalar(dims[2])
        self._unlock()

    def write_raw_term(self, t):
        super(Scanner, self).write_raw_term(t)
        if self._counter == 0:
            self.terms.append(t)


class NotationParam(object):
    Var = 0
    Value = 1
    Term = 2
    Any = 3
    Index = 4
    N = 5
    List = 256

    def __init__(self, p):
        self.typ = NotationParam.Var
        if isinstance(p, tuple):
            name, self.typ = p
        else:
            name = p
        if isinstance(name, Symbol):
            self.sym = name
        else:
            self.sym = Symbol(name)


def create_parameter(p):
    param = NotationParam(p)
    return param.sym, param


# https://stackoverflow.com/questions/3655842/how-can-i-test-whether-a-variable-holds-a-lambda
def isalambda(v):
    LAMBDA = lambda: 0
    return v is not None and isinstance(v, type(LAMBDA)) and v.__name__ == LAMBDA.__name__


def dot3(sym, notation):
    f = notation.vgetf(sym, [Notation.PLUS, Notation.MINUS])
    if f is not None:
        return dot3(f.args[0], notation)
    return sym == Notation.DOT3


def index_variable(params):
    for sym in params:
        p = params[sym]
        if p.typ == NotationParam.Index:
            return p
    return None


class NotationComparer(object):
    """NotationComparer"""

    def __init__(self, sym, notation, params):
        self.sym = sym
        self.notation = notation
        self.params = dict(map(create_parameter, params))
        self.index = index_variable(self.params)

    def compare_index(self, f1, notation1, f2, notation2, subst, ctx):
        if not self.compare(f1.args[0], notation1, f2.args[0], notation2, subst, ctx):
            return False
        for i in range(4):
            if isalambda(ctx) and not ctx(i):
                continue
            if not self.compare(f1.args[1][i], notation1, f2.args[1][i], notation2, subst, Notation.INDEX):
                return False
        return True

    def compare(self, a, notation1, b, notation2, subst, ctx=None):
        if ctx == Notation.S_LIST:
            f1 = notation1.getf(a, Notation.PLUS)
            if f1 is not None:
                a = f1.args[0]
            f2 = notation2.getf(b, Notation.PLUS)
            if f2 is not None:
                b = f2.args[0]
        if type(a) == type(b):
            if isinstance(a, Symbol):
                f1 = notation1.get(a)
                f2 = notation2.get(b)
                if f1 is None or f2 is None:
                    return self.equal(a, notation1, b, subst)
                if f1 is not None and f2 is not None and f1.sym == f2.sym:
                    if f1.sym == Notation.INDEX:
                        return self.compare_index(f1, notation1, f2, notation2, subst, ctx)
                    else:
                        return self.compare(f1.args, notation1, f2.args, notation2, subst, f1.sym)
                else:
                    return False
            else:
                if isinstance(a, list) or isinstance(a, tuple):
                    a = list(filter(lambda t: t not in Notation.styles, a))
                    b = list(filter(lambda t: t not in Notation.styles, b))
                    if ctx in [Notation.S_LIST, Notation.P_LIST] and \
                            not any(dot3(arg, notation2) for arg in b):
                        for x in a:
                            found = False
                            for y in b:
                                if self.compare(x, notation1, y, notation2, subst, ctx):
                                    b.remove(y)
                                    found = True
                                    break
                            if not found:
                                return False
                        return True
                    else:
                        i = 0
                        j = 0
                        n = 1
                        match = 0
                        while i < len(a) and j < len(b):
                            y = b[j]
                            if match == 0 and j < len(b) - 1 and dot3(b[j + 1], notation2):
                                n = 1
                                match = 1
                            elif match == 1 and dot3(y, notation2):
                                match = 2
                                j += 1
                                continue
                            if self.index is not None:
                                subst[self.index.sym] = IntegerValue(n)
                            if not self.compare(a[i], notation1, y, notation2, subst, ctx):
                                return False
                            i += 1
                            if match > 0:
                                n += 1
                                if match == 1:
                                    j += 1
                            else:
                                n = 1
                                j += 1
                        return i == len(a) and (j == len(b) or match > 0)
                else:
                    return a == b
        return self.equal(a, notation1, b, subst)

    def equal(self, sym1, notation, sym2, subst):
        if isinstance(sym2, Symbol):
            param = self.params.get(sym2, None)
            if param is not None:
                typ = param.typ & 255
                if typ == NotationParam.Value and not isinstance(sym1, Value):
                    return False
                elif typ == NotationParam.Term:
                    scanner = Scanner(notation)
                    scanner.write_formula(sym1)
                    if len(scanner.terms) > 1:
                        return False
                    elif len(scanner.terms) == 1:
                        sym1 = scanner.terms[0]
                elif typ == NotationParam.Var and notation.get(sym1) is not None:
                    return False
                elif typ == NotationParam.N and not isinstance(sym1, IntegerValue):
                    return False
                if param.typ & NotationParam.List != 0:
                    value_list = subst.get(sym2.name, [])
                    value_list.append(sym1)
                    subst[sym2.name] = value_list
                    return True
                elif sym2 not in subst:
                    subst[sym2.name] = sym1
                    return True
                else:
                    return subst[sym2.name] != sym1
        return sym1 == sym2

    def match(self, sym, notation, ctx=None):
        subst = defaultdict()
        if self.compare(sym, notation, self.sym, self.notation, subst, ctx):
            return subst
        return None


def pattern(expr, params=[]):
    notation = Notation()
    p = MathParser(notation)
    sym = p.parse(expr)
    return NotationComparer(sym, notation, params)


def s_equal(sym1, notation1, sym2, notation2, ctx=None):
    comparer = NotationComparer(sym2, notation2, [])
    return comparer.match(sym1, notation1, ctx) is not None

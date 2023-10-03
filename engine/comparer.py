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


def isVariable(sym):
    return (isinstance(sym, Symbol) and (sym.name.startswith("#") or sym.name in Notation.variables)
            and 'quoted' not in sym.props)


def unquote(sym, notation, subst):
    while True:
        f = notation.getf(sym, Notation.GROUP)
        if f is not None and f.props['br'] == '{}':
            if 'quoted' in f.props:
                subst = None
            sym = f.args[0]
            continue
        return sym, subst


def dot3(sym, notation):
    f = notation.vgetf(sym, [Notation.PLUS, Notation.MINUS])
    if f is not None:
        return dot3(f.args[0], notation)
    return sym == Notation.DOT3


def expand_group(sym, notation):
    f = notation.getf(sym, Notation.GROUP)
    while f is not None:
        sym = f.args[0]
        f = notation.getf(sym, Notation.GROUP)
    return sym


def simplify(sym, notation):
    def traverse(tmp):
        f = notation.getf(tmp, Notation.GROUP)
        if f is not None and f.props['br'] == '{}':
            return traverse(f.args[0])
        if isVariable(tmp):
            return tmp
        return None

    vsym = traverse(sym)
    if vsym is not None:
        return vsym
    return sym


class NotationComparer(object):
    """NotationComparer"""

    def __init__(self, sym, notation):
        self.sym = sym
        self.notation = notation

    def compare_index(self, f1, notation1, subst1, f2, notation2, subst2, ctx):
        if not self.compare(f1.args[0], notation1, subst1, f2.args[0], notation2, subst2, ctx):
            return False
        for i in range(4):
            if isalambda(ctx) and not ctx(i):
                continue
            if not self.compare(f1.args[1][i], notation1, subst1, f2.args[1][i], notation2, subst2, Notation.INDEX):
                return False
        return True

    def compare(self, a, notation1, subst1, b, notation2, subst2, ctx=None):
        a, subst1 = unquote(a, notation1, subst1)
        b, subst2 = unquote(b, notation2, subst2)
        if ctx == Notation.S_LIST:
            f1 = notation1.vgetf(a, [Notation.PLUS, Notation.MINUS])
            f2 = notation2.vgetf(b, [Notation.PLUS, Notation.MINUS])
            if (f1 is not None and f1.sym == Notation.PLUS and
                    (f2 is None or f2.sym == Notation.PLUS)):
                a = f1.args[0]
            if (f2 is not None and f2.sym == Notation.PLUS and
                    (f1 is None or f1.sym == Notation.PLUS)):
                b = f2.args[0]
        if type(a) == type(b):
            if isinstance(a, Symbol):
                f1 = notation1.get(a)
                f2 = notation2.get(b)
                if f1 is None or f2 is None:
                    return self.equal(a, notation1, subst1, b, notation2, subst2, ctx)
                if f1 is not None and f2 is not None and f1.sym == f2.sym:
                    if f1.sym == Notation.INDEX:
                        return self.compare_index(f1, notation1, subst1, f2, notation2, subst2, ctx)
                    else:
                        return self.compare(f1.args, notation1, subst1, f2.args, notation2, subst2, f1.sym)
                else:
                    return False
            else:
                if isinstance(a, list) or isinstance(a, tuple):
                    a = list(filter(lambda t: t not in Notation.styles, a))
                    b = list(filter(lambda t: t not in Notation.styles, b))
                    if ctx in [Notation.S_LIST, Notation.P_LIST]:
                        if any(dot3(arg, notation2) for arg in b):
                            return self.matchdot3(a, notation1, subst1, b, notation2, subst2, ctx)
                        else:
                            if len(a) != len(b):
                                return False
                            for x in a:
                                found = False
                                for y in b:
                                    if self.compare(x, notation1, subst1, y, notation2, subst2, ctx):
                                        b.remove(y)
                                        found = True
                                        break
                                if not found:
                                    return False
                            return True
                    else:
                        if len(a) != len(b):
                            return False
                        for (x, y) in zip(a, b):
                            if not self.compare(x, notation1, subst1, y, notation2, subst2, ctx):
                                return False
                        return True
        return self.equal(a, notation1, subst1, b, notation2, subst2, ctx)

    def matchdot3(self, a, notation1, subst1, b, notation2, subst2, ctx):
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
            if not self.compare(a[i], notation1, subst1, y, notation2, subst2, ctx):
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

    def equal(self, sym1, notation1, subst1, sym2, notation2, subst2, ctx=None):
        return sym1 == sym2

    def match(self, sym, notation, ctx=None):
        subst = defaultdict()
        if self.compare(sym, notation, defaultdict(), self.sym, self.notation, subst, ctx):
            return subst
        return None


class NotationParametrizedComparer(NotationComparer):
    def __init__(self, sym, notation, params):
        super().__init__(sym, notation)
        self.params = dict(map(create_parameter, params))

    def equal(self, sym1, notation1, _, sym2, notation2, subst, ctx=None):
        if isinstance(sym2, Symbol):
            param = self.params.get(sym2, None)
            if param is not None:
                typ = param.typ & 255
                if typ == NotationParam.Value and not isinstance(sym1, Value):
                    return False
                elif typ == NotationParam.Term:
                    scanner = Scanner(notation1)
                    scanner.write_formula(sym1)
                    if len(scanner.terms) > 1:
                        return False
                    elif len(scanner.terms) == 1:
                        sym1 = scanner.terms[0]
                elif typ == NotationParam.Var and notation1.get(sym1) is not None:
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


class UnifyComparer(NotationComparer):
    def __init__(self, sym, notation, subst=None):
        super().__init__(sym, notation)
        if subst is None:
            subst = defaultdict()
        self.subst = subst

    def equal(self, sym1, notation1, subst1, sym2, notation2, subst2, ctx=None):
        var1 = isVariable(sym1) and subst1 is not None
        var2 = isVariable(sym2) and subst2 is not None
        if var1 or var2:
            value1 = sym1
            if var1:
                if sym1.name in subst1:
                    value1 = subst1[sym1.name]
                else:
                    value1 = None
            value2 = sym2
            if var2:
                if sym2.name in subst2:
                    value2 = subst2[sym2.name]
                else:
                    value2 = None
            if value1 is not None and value2 is not None:
                if value1 != value2:
                    if not self.compare(value1, notation1, None, value2, notation2, None, ctx):
                        return False
            if var1:
                if sym1.name != "##" and value2 is not None:
                    subst1[sym1.name] = value2
            if var2:
                if sym2.name != "##" and value1 is not None:
                    subst2[sym2.name] = value1
            return True
        return sym1 == sym2

    def matchdot3(self, a, notation1, subst1, b, notation2, subst2, ctx):
        i = 0
        j = 0
        match = 0
        tail = []
        while i < len(a) and j < len(b):
            y = b[j]
            if match == 0 and j < len(b) - 1 and dot3(b[j + 1], notation2):
                match = 1
            elif match == 1 and dot3(y, notation2):
                match = 2
                j += 1
                continue
            if match == 2:
                tail.append(a[i])
            elif not self.compare(a[i], notation1, subst1, y, notation2, subst2, ctx):
                return False
            i += 1
            if match > 0:
                if match == 1:
                    j += 1
            else:
                j += 1
        if match == 2:
            if not self.compare(tail, notation1, subst1, b[j], notation2, subst2, ctx):
                return False
        return i == len(a) and j == len(b) - 1

    def unify(self, sym, notation, subst=None, ctx=None):
        if subst is None:
            subst = defaultdict()
        return self.compare(sym, notation, subst, self.sym, self.notation, self.subst, ctx)


def pattern(expr, params=None):
    if params is None:
        params = []
    notation = Notation()
    p = MathParser(notation)
    sym = p.parse(expr)
    return NotationParametrizedComparer(sym, notation, params)


def s_equal(sym1, notation1, sym2, notation2, ctx=None):
    comparer = NotationComparer(sym2, notation2)
    return comparer.match(sym1, notation1, ctx) is not None

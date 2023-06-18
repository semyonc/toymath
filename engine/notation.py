#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 15:39:04 2020

@author: semyonc
"""
from typing import TypeVar
from collections import defaultdict
import json

SYMBOL = TypeVar('SYMBOL', bound='Symbol')
NOTATION = TypeVar('NOTATION', bound='Notation')


class Symbol(object):
    """ Symbol """
    autonum = 1

    def __init__(self, name=None, **kwargs):
        if name is None:
            self.name = f'_n{Symbol.autonum}'
            Symbol.autonum += 1
        else:
            self.name = name
        self.props = kwargs

    def __str__(self):
        return self.name

    def __eq__(self, o):
        return isinstance(o, Symbol) and o.name == self.name

    def __ne__(self, o):
        return not self.__eq__(o)

    def __hash__(self):
        return self.name.__hash__()

    def __repr__(self):
        return self.name


class Func(object):
    """ Func """

    def __init__(self, sym, args, **kwargs):
        self.sym = sym
        self.args = args
        self.props = kwargs

    def rank(self):
        return len(self.args)

    def _format_list(self, val):
        res = ''
        for index, t in enumerate(val):
            if index > 0:
                res += ','
            res += self._format(t)
        return res

    def _format(self, val):
        if isinstance(val, list):
            return f'[{self._format_list(val)}]'
        if isinstance(val, tuple):
            return f'({self._format_list(val)})'
        return f'{val}'

    def __repr__(self):
        props = ""
        if len(self.props) > 0:
            props = f'{json.dumps(self.props)} '
        args = ""
        for index, t in enumerate(self.args):
            if index > 0:
                args += ','
            args += self._format(t)
        if isinstance(self.args, tuple):
            return f'{self.sym.__repr__()} {props}({args})'
        else:
            return f'{self.sym.__repr__()} {props}[{args}]'


class Rule(object):
    """ Rule """

    def __init__(self, name, pattern, action, **kwargs):
        self.name = name
        self.pattern = pattern
        self.action = action
        self.props = kwargs


def issym(sym, name):
    return isinstance(sym, Symbol) and sym.name in name


class Notation(object):
    """ Notation """
    COMP = Symbol('comp')
    C_LIST = Symbol('c-list')
    S_LIST = Symbol('s-list')
    P_LIST = Symbol('p-list')
    SLASH = Symbol('/')
    STAR = Symbol('*')
    INDEX = Symbol('index')
    LIMITS = Symbol('limits')
    NOLIMITS = Symbol('nolimits')
    REF = Symbol('backref')
    PRIME = Symbol('prime')
    PLUS = Symbol('+')
    MINUS = Symbol('-')
    GROUP = Symbol('group')
    V_GROUP = Symbol('vgroup')
    S_GROUP = Symbol('sgroup')
    UNARY = Symbol('unary')
    FUNC = Symbol('func')
    NONE = Symbol('none')
    DOT3 = Symbol('...')
    DASHV = Symbol('\\dashv')

    comparer = (
        '=',
        '\\in',
        '\\to',
        '\\ge',
        '\\lt',
        '\\le',
        '\\leq',
        '\\leqq',
        '\\leqslant',
        '\\lesseqgtr',
        '\\lesseqqgtr',
        '\\lessgtr',
        '\\lesssim',
        '\\lnapprox',
        '\\lneq',
        '\\lneqq',
        '\\lnsim',
        '\\lvertneqq',
        '\\ne',
        '\\neq',
        '\\geq',
        '\\geqq',
        '\\geqslant',
        '\\gt',
        '\\gg',
        '\\ggg',
        '\\gggtr',
        '\\gtreqless',
        '\\gtreqqless',
        '\\gtrless',
        '\\gtrapprox',
        '\\gnapprox'
    )

    additive = (
        '+',
        '-'
    )

    styles = (
        '\\bf',
        '\\rm',
        '\\displaystyle',
        '\\frak',
        '\\cal',
        '\\!',
        '\\,',
        '\\:',
        '\\>',
        '\\;',
        '\\ '
    )

    oper = (
        '\\acute',
        '\\vec',
        '\\grave',
        '\\widehat',
        '\\widetilde',
        '\\partial',
        '\\phantom',
        '\\boldsymbol',
        '\\thinspace',
        '\\textstyle',
        '\\cancel',
        '\\bcancel',
        '\\boxed',
        '\\hat',
        '\\frac',
        '\\dfrac',
        '\\cfrac',
        '\\tfrac',
        '\\binom',
        '\\Bbb'
    )

    unary_f = ('\\sin', '\\sinh', '\\cos', '\\cosh', '\\cot',
               '\\coth', '\\sec', '\\csc', '\\tan', '\\tanh', '\\delta', '\\Delta', '\\varDelta')

    common_f = ('f', 'g', '\\omega', '\\Omega', '\\omicron', '\\sigma', '\\rho', '\\Re', '\\psi',
                '\\Psi', '\\phi', '\\Phi', '\\pi', '\\Pi', '\\nabla', '\\mu', '\\tau', '\\theta', '\\Theta',
                '\\upsilon',
                '\\Upsilon', '\\upsilon', '\\Gamma', '\\varGamma', '\\Xi', '\\xi', '\\kappa', '\\varkappa')
    p_oper = ('\\sum', '\\lim', '\\int', '\\prod', '\\intop', '\\coprod', '\\intop', '\\iint', '\\iiint',
              '\\iiiint', '\\idotsint', '\\oint', '\\projlim', '\\varprojlim')

    def __init__(self):
        self.rel = defaultdict()

    def clear(self):
        self.rel.clear()

    def is_empty(self):
        return len(self.rel) == 0

    def clone(self):
        res = Notation()
        for sym, f in self.rel.items():
            res.rel[sym] = f
        return res

    def join(self, notation):
        for sym, f in notation.rel.items():
            self.rel[sym] = f

    def assign(self, notation):
        self.clear()
        self.join(notation)

    def repf(self, sym, func):
        if sym is None:
            sym = Symbol()
        assert isinstance(func, Func)
        assert isinstance(sym, Symbol)
        self.rel[sym] = func
        return sym

    def setf(self, f, args, **kwargs):
        return self.repf(None, Func(f, args, **kwargs))

    def getf(self, sym, f):
        if isinstance(sym, Symbol) and sym in self.rel:
            func = self.rel[sym]
            if func is not None and func.sym == f:
                return func
        return None

    def vgetf(self, sym, listf):
        if isinstance(sym, Symbol) and sym in self.rel:
            func = self.rel[sym]
            if func is not None and func.sym in listf:
                return func
        return None

    def get(self, sym):
        if isinstance(sym, Symbol) and sym in self.rel:
            return self.rel[sym]
        return None

    def select(self, fname, arity=None):
        for sym, f in self.rel.items():
            if f.sym == fname and (arity is None or arity == len(f.args)):
                yield sym, f

    def __repr__(self):
        res = ""
        for index, sym in enumerate(iter(self.rel)):
            if index > 30:
                res += ".."
                break
            elif index > 0:
                res += "\n"
            res += f'{sym.__repr__()}: {self.rel[sym]}'
        return res
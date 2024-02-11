#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 13:41:35 2021

@author: semyonc
"""
import re
from io import StringIO
from notation import Notation, Symbol


class LaTexWriter(object):
    namemap = {
        '\\above': 'write_above',
        '\\abovewithdelims': 'write_abovewithdelims',
        '\\atopwithdelims': 'write_atopwithdelims',
        '\\atop': 'write_binaryop',
        '\\brace': 'write_binaryop',
        '\\brack': 'write_binaryop',
        '\\color': 'write_color',
        '\\lower': 'write_lower',
        '\\sqrt': 'write_sqrt',
        '\\buildrel': 'write_buildrel',
        'group': 'write_group',
        'vgroup': 'write_vgroup',
        'sgroup': 'write_sgroup',
        'func': 'write_func',
        '\\array': 'write_array',
        '\\cases': 'write_array',
        '\\text': 'write_text',
        '\\textbf': 'write_text',
        '\\textit': 'write_text',
        '\\textrm': 'write_text',
        '\\textsf': 'write_text',
        '\\texttt': 'write_text'
    }

    re_prefix = re.compile(r'\\')
    re_letter = re.compile(r'\w')
    re_digit = re.compile(r'[0-9]+(\.[0-9]+)?')

    """ LaTexWriter """

    def __init__(self, notation, show_quotes=False, *kwargs):
        self.output = StringIO()
        self.notation = notation
        self.last_value = ""
        self.head = None
        self.show_quotes = show_quotes

    def __repr__(self):
        return self.output.getvalue()

    def writeString(self, val):
        pr1 = LaTexWriter.re_prefix.match(self.last_value)
        if pr1 is not None:
            pr1 = pr1.group()
        pr2 = LaTexWriter.re_prefix.match(val)
        if pr2 is not None:
            pr2 = pr2.group()
        if self.last_value != "" and pr1 != pr2 \
                and not any([pr in ['^', '_', '{', '}'] for pr in [self.last_value, val]]):
            self.output.write(' ')
        self.output.write(val)
        self.last_value = val

    def writeDimen(self, dimen):
        self.writeString(f'{dimen[0]}{dimen[1]}')

    def _probe(self, sym, name):
        f = self.notation.getf(sym, Symbol(name))
        if f is not None:
            getattr(LaTexWriter, LaTexWriter.namemap[name])(self, f)
            return True
        return False

    def _probeCommand(self, sym):
        f = self.notation.get(sym)
        if f is not None and f.sym.name.endswith('!'):
            if f.sym.name in ['closure!', 'track!']:
                self.write_or_expr_list(f.args[1][0])
            else:
                self.writeString('\\mathop{')
                if 'negative' in f.props:
                    self.writeString('\\overline{')
                self.writeString(f.sym.name)
                self.writeString('}')
                if 'negative' in f.props:
                    self.writeString('}')
                if f.args[0] is not None:
                    self.writeString('\\limits_{')
                    self.write_comma_list(f.args[0])
                    self.writeString('}')
                if len(f.args[1]) > 0:
                    self.writeString('\\space')
                    self.write_or_expr_list(f.args[1][0])
                    if len(f.args[1]) == 2:
                        self.writeString('\\,\\Box\\,')
                        self.write_or_expr_list(f.args[1][1])
            return True
        return False

    def __call__(self, sym):
        self.output = StringIO()
        self.write_formula(sym)
        return self.output.getvalue()

    def write_formula(self, sym):
        self.head = sym
        if not self._probeCommand(sym) \
                and not self._probe(sym, '\\above') \
                and not self._probe(sym, '\\abovewithdelims') \
                and not self._probe(sym, '\\atop') \
                and not self._probe(sym, '\\atopwithdelims') \
                and not self._probe(sym, '\\brace') \
                and not self._probe(sym, '\\brack'):
            self.write_or_expr_list(sym)

    def write_or_expr_list(self, sym):
        self.head = sym
        f = self.notation.getf(sym, Notation.O_LIST)
        if f is not None:
            for i, expr in enumerate(f.args):
                if i > 0:
                    self.writeString('\\lor')
                self.write_and_expr_list(expr)
        else:
            self.write_and_expr_list(sym)

    def write_and_expr_list(self, sym):
        self.head = sym
        f = self.notation.getf(sym, Notation.A_LIST)
        if f is not None:
            for i, expr in enumerate(f.args):
                if i > 0:
                    self.writeString('\\land')
                self.write_not_expr(expr)
        else:
            self.write_not_expr(sym)

    def write_not_expr(self, sym):
        self.head = sym
        f = self.notation.getf(sym, Notation.NEG)
        if f is not None:
            self.writeString('\\neg')
            self.write_subformula(f.args[0])
        else:
            self.write_subformula(sym)

    def write_subformula(self, sym):
        self.head = sym
        f = self.notation.getf(sym, Notation.COMP)
        if f is not None:
            self.write_subformula_comparison(f)
        else:
            self.write_comma_list(sym)

    def write_subformula_comparison(self, f):
        self.write_additive_expr(f.args[0])
        self.writeString(f.sym.props['op'])
        self.write_comma_list(f.args[1])

    def write_comma_list(self, sym):
        self.head = sym
        f = self.notation.getf(sym, Notation.C_LIST)
        if f is not None:
            for i, expr in enumerate(f.args):
                if i > 0:
                    self.writeString(',')
                self.write_additive_expr_list(expr)
        else:
            self.write_additive_expr_list(sym)

    def write_additive_expr_list(self, sym):
        self.head = sym
        f = self.notation.getf(sym, Notation.S_LIST)
        if f is not None:
            for expr in f.args:
                self.write_additive_expr(expr)
        else:
            self.write_additive_expr(sym)

    def write_additive_expr(self, sym):
        self.head = sym
        f = self.notation.get(sym)
        if f is not None and f.sym.name in Notation.additive:
            self.writeString(f.sym.name)
            self.write_composite_expr(f.args[0])
        else:
            self.write_composite_expr(sym)

    def write_composite_expr(self, sym):
        self.head = sym
        f = self.notation.getf(sym, Notation.P_LIST)
        if f is not None:
            self.write_plist(f)
        else:
            f = self.notation.getf(sym, Notation.SLASH)
            if f is not None:
                self.write_slashExpr(f)
            else:
                f = self.notation.getf(sym, Notation.STAR)
                if f is not None:
                    self.write_starExpr(f)
                else:
                    self.write_expr(sym)

    def write_expr(self, sym):
        self.head = sym
        if not self._probe(sym, '\\color') \
                and not self._probe(sym, '\\lower') \
                and not self._probe(sym, '\\buildrel') \
                and not self._probe(sym, '\\sqrt'):
            f = self.notation.get(sym)
            if f is not None:
                if f.sym == Notation.INDEX:
                    self.write_index(f)
                elif f.sym == Notation.LIMITS:
                    self.write_limits(f)
                elif f.sym == Notation.NOLIMITS:
                    self.write_nolimits(f)
                elif f.sym.name in Notation.oper:
                    self.writeString(f.sym.name)
                    for expr in f.args:
                        self.write_expr(expr)
                else:
                    self.write_scalar(sym)
            else:
                self.write_term(sym)

    def write_symbol(self, sym):
        if self.show_quotes and 'quoted' in sym.props:
            self.writeString('`')
        if sym.name.startswith('#'):
            self.writeString("\\#\\text{" + sym.name[1:] + "}")
        elif sym.name.startswith('_'):
            self.writeString("\\_\\textit{" + sym.name[1:] + "}")
        else:
            self.writeString(sym.name)

    def write_scalar(self, sym):
        self.head = sym
        if not self._probe(sym, "group") \
                and not self._probe(sym, "vgroup") \
                and not self._probe(sym, "sgroup") \
                and not self._probe(sym, "func") \
                and not self._probe(sym, "\\text") \
                and not self._probe(sym, "\\textbf") \
                and not self._probe(sym, "\\textit") \
                and not self._probe(sym, "\\textrm") \
                and not self._probe(sym, "\\textsf") \
                and not self._probe(sym, "\\texttt") \
                and not self._probe(sym, "\\array") \
                and not self._probe(sym, "\\cases"):
            f = self.notation.getf(sym, Notation.REF)
            if f is not None:
                self.write_backref(f)
            else:
                self.write_term(sym)

    def write_term(self, sym):
        self.head = sym
        f = self.notation.getf(sym, Notation.PRIME)
        if f is not None:
            self.write_prime(f)
        else:
            if isinstance(sym, Symbol):
                if self.notation.get(sym) is not None:
                    self.writeString('{')
                    self.write_formula(sym)
                    self.writeString('}')
                else:
                    self.write_symbol(sym)
            else:
                self.write_raw_term(sym)

    def write_index_item(self, sym):
        f = self.notation.getf(sym, Notation.GROUP)
        escape = f is not None and f.props['br'] == '()'
        if escape:
            self.writeString('{')
            self.write_scalar(sym)
            self.writeString('}')
        else:
            self.write_scalar(sym)

    def write_index(self, f):
        dims = f.args[1]
        if dims[0] is not None:
            self.writeString('^')
            self.write_index_item(dims[0])
        if dims[1] is not None:
            self.writeString('_')
            self.write_index_item(dims[1])
        self.write_scalar(f.args[0])
        if dims[3] is not None:
            self.writeString('_')
            self.write_index_item(dims[3])
        if dims[2] is not None:
            self.writeString('^')
            self.write_index_item(dims[2])

    def write_limits(self, f):
        dims = f.args[1]
        self.write_scalar(f.args[0])
        self.writeString('\\limits')
        if dims[0] is not None:
            self.writeString('_')
            self.write_scalar(dims[0])
        if dims[1] is not None:
            self.writeString('^')
            self.write_scalar(dims[1])

    def write_nolimits(self, f):
        dims = f.args[1]
        self.write_scalar(f.args[0])
        self.writeString('\\nolimits')
        if dims[0] is not None:
            self.writeString('_')
            self.write_scalar(dims[0])
        if dims[1] is not None:
            self.writeString('^')
            self.write_scalar(dims[1])

    def write_func(self, f):
        sym_f = f.args[0]
        index_f = self.notation.getf(sym_f, Notation.INDEX)
        limits_f = self.notation.getf(sym_f, Notation.LIMITS)
        if index_f is not None:
            sym_f = index_f.args[0]
        elif limits_f is not None:
            sym_f = limits_f.args[0]
        if f.props.get('fmt', '') == 'operatorname':
            self.writeString(f'\\operatorname {{{sym_f.name}}}')
        else:
            self.writeString(sym_f.name)
        if index_f is not None:
            dims = index_f.args[1]
            if dims[3] is not None:
                self.writeString('_')
                self.write_scalar(dims[3])
            if dims[2] is not None:
                self.writeString('^')
                self.write_scalar(dims[2])
        elif limits_f is not None:
            dims = limits_f.args[1]
            self.writeString('\\limits')
            if dims[1] is not None:
                self.writeString('_')
                self.write_scalar(dims[1])
            if dims[0] is not None:
                self.writeString('^')
                self.write_scalar(dims[0])
        f1 = self.notation.get(f.args[1])
        if f.props.get('fmt', '') == 'unary' and (
                f1 is None or f1.sym not in [Notation.S_LIST, Notation.C_LIST, Notation.GROUP,
                                             Notation.FUNC, Notation.PLUS, Notation.MINUS]):
            self.write_formula(f.args[1])
        elif f.props.get('fmt', '') == 'oper' and (f1 is None or f1.sym not in [Notation.S_LIST, Notation.C_LIST,
                                                                                Notation.PLUS, Notation.MINUS]):
            self.writeString('{')
            self.write_formula(f.args[1])
            self.writeString('}')
        else:
            self.writeString('(')
            self.write_formula(f.args[1])
            self.writeString(')')

    def write_raw_term(self, t):
        self.writeString(t.__repr__())

    def write_prime(self, f):
        self.write_term(f.args[0])
        self.writeString('\'')

    def write_text(self, f):
        self.writeString(f.sym.name)
        self.writeString('{')
        self.output.write(f.args[0])
        self.writeString('}')

    def write_plist(self, f):
        for i, expr in enumerate(f.args):
            if i > 0:
                f1 = self.notation.getf(expr, Notation.GROUP)
                if f1 is not None and f1.props.get('br', '') == '{}':
                    self.writeString('\\cdot')
            self.write_expr(expr)

    def write_slashExpr(self, f):
        self.write_expr(f.args[0])
        self.writeString('/')
        self.write_expr(f.args[1])

    def write_starExpr(self, f):
        self.write_expr(f.args[0])
        self.writeString('\\cdot')
        self.write_expr(f.args[1])

    def write_backref(self, f):
        self.writeString('[[')
        self.write_subformula(f.args[0])
        self.writeString(']]')

    def write_above(self, f):
        self.write_subformula(f.args[0])
        self.writeString('\\above')
        self.writeDimen(f.sym.props['dimen'])
        self.write_subformula(f.args[1])

    def write_abovewithdelims(self, f):
        self.write_subformula(f.args[0])
        self.writeString('\\abovewithdelims')
        self.writeString(f.sym.props['delim1'])
        self.writeString(f.sym.props['delim2'])
        self.writeDimen(f.sym.props['dimen'])
        self.write_subformula(f.args[1])

    def write_atopwithdelims(self, f):
        self.write_subformula(f.args[0])
        self.writeString('\\atopwithdelims')
        self.writeString(f.sym.props['delim1'])
        self.writeString(f.sym.props['delim2'])
        self.write_subformula(f.args[1])

    def write_binaryop(self, f):
        self.write_subformula(f.args[0])
        self.writeString(f.sym.name)
        self.write_subformula(f.args[1])

    def write_color(self, f):
        self.writeString('\\color')
        self.writeString('{')
        self.output.write(f.sym.props['c'])
        self.writeString('}')
        self.write_expr(f.args[0])

    def write_lower(self, f):
        self.writeString('\\lower')
        self.writeDimen(f.sym.props['dimen'])
        self.write_expr(f.args[0])

    def write_sqrt(self, f):
        self.writeString('\\sqrt')
        if len(f.args) > 1:
            self.writeString('[')
            self.write_subformula(f.args[1])
            self.writeString(']')
        self.write_expr(f.args[0])

    def write_buildrel(self, f):
        self.writeString('\\buildrel')
        self.write_subformula(f.args[0])
        self.writeString('\\over')
        self.write_expr(f.args[1])

    def write_group(self, f):
        br = f.props['br']
        if self.show_quotes:
            if 'quoted' in f.props:
                self.writeString('`')
            if br == '{}':
                self.writeString('\\{')
        self.writeString(br[0])
        self.write_formula(f.args[0])
        self.writeString(br[1])
        if self.show_quotes and br == '{}':
            self.writeString('\\}')

    def write_vgroup(self, f):
        br = f.props['br']
        self.writeString('\\left')
        self.writeString(br[0])
        self.write_formula(f.args[0])
        self.writeString('\\right')
        self.writeString(br[1])

    def write_sgroup(self, f):
        self.writeString('\\{')
        self.write_expr(f.args[0])
        if len(f.args) > 1:
            self.writeString('|')
            self.write_subformula(f.args[1])
        self.writeString('\\}')

    def write_array(self, f):
        self.writeString(f.sym.name)
        self.writeString('{')
        self.write_row_list(f.args)
        self.writeString('}')

    def write_row_list(self, rows):
        for i, row in enumerate(rows):
            if i > 0:
                self.writeString('\\cr')
            self.write_col_list(row)

    def write_col_list(self, row):
        for i, col in enumerate(row):
            if i > 0:
                self.writeString(' & ')
            self.write_subformula(col)

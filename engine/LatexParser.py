#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 19:15:16 2020

@author: semyonc
"""
import re

import ply.yacc as yacc
from lexer import MathLexer
from notation import Notation, Symbol, Func


# These environments carry only their delimiter/presentation choice; unlike
# array/matrix* they have no alignment preamble to preserve.  Normalize them
# at the shared parser boundary so the kernel/legacy engine and the verified
# tactic layer build the same dedicated array nodes.
_PLAIN_ARRAY_ENVS = (
    'pmatrix', 'matrix', 'bmatrix', 'Bmatrix', 'vmatrix', 'Vmatrix',
    'smallmatrix', 'cases',
)
_PLAIN_ARRAY_ENV_ALT = '|'.join(re.escape(name) for name in _PLAIN_ARRAY_ENVS)
_PLAIN_ARRAY_ENV_RE = re.compile(
    rf'\\begin\{{({_PLAIN_ARRAY_ENV_ALT})\}}'
    rf'((?:(?!\\begin\{{(?:{_PLAIN_ARRAY_ENV_ALT})\}}|'
    rf'\\end\{{(?:{_PLAIN_ARRAY_ENV_ALT})\}}).)*?)'
    rf'\\end\{{\1\}}',
    re.DOTALL,
)

_ESCAPED_SET_RE = re.compile(
    r'(?<!\\left)\\\{((?:(?!\\\{|\\\}).)*?)(?<!\\right)\\\}',
    re.DOTALL,
)
_SET_BUILDER_OPEN = '\x01'
_SET_BUILDER_CLOSE = '\x02'


def _normalize_plain_array_envs(latex):
     """Turn non-alignment array environments into grammar commands.

     Innermost-first replacement supports mixed nested matrix families.  The
     alignment-bearing ``array`` and starred matrix variants are deliberately
     excluded: discarding their preambles would make round-tripping lossy.
     """
     def repl(match):
          body = match.group(2).replace('\\\\', ' \\cr ')
          return f'\\{match.group(1)}{{{body}}}'

     previous = None
     while previous != latex:
          previous = latex
          latex = _PLAIN_ARRAY_ENV_RE.sub(repl, latex)
     return latex


def _top_level_bar_count(body):
     """Count vertical bars outside ordinary delimiter nesting.

     Escaped braces are overloaded in TeX: an even number of top-level bars
     denotes a finite collection (including absolute-value elements), while
     an odd extra bar retains ToyMath's existing set-builder spelling.
     """
     stack = []
     pairs = {')': '(', ']': '[', '}': '{',
              _SET_BUILDER_CLOSE: _SET_BUILDER_OPEN}
     i = 0
     count = 0
     while i < len(body):
          ch = body[i]
          if ch == '\\':
               # Control words do not contribute delimiter characters.
               # A following raw delimiter from \left( / \right) is handled
               # on its next iteration.
               j = i + 1
               while j < len(body) and body[j].isalpha():
                    j += 1
               if j > i + 1:
                    i = j
                    continue
          if ch in ('(', '[', '{', _SET_BUILDER_OPEN):
               stack.append(ch)
          elif ch in pairs:
               if stack and stack[-1] == pairs[ch]:
                    stack.pop()
          elif ch == '|' and not stack:
               count += 1
          i += 1
     return count


def _normalize_collection_literals(latex):
     r"""Lower standard ``\{...\}`` collection syntax to an unambiguous
     internal grammar command.

     Collection commas and the existing bare/set-builder vertical-bar grammar
     otherwise create LALR conflicts. Innermost escaped braces are classified
     first; private sentinels protect retained set-builders while an enclosing
     collection is classified, then restore their original spelling.
     """
     def repl(match):
          body = match.group(1)
          if _top_level_bar_count(body) % 2:
               return _SET_BUILDER_OPEN + body + _SET_BUILDER_CLOSE
          return '\\collection{' + body + '}'

     while True:
          latex, count = _ESCAPED_SET_RE.subn(repl, latex)
          if count == 0:
               break
     return latex.replace(_SET_BUILDER_OPEN, '\\{').replace(
         _SET_BUILDER_CLOSE, '\\}')


# Control words that own a following bar, so the scanner must not read it as
# a bare delimiter: \left| ... \right|, \middle|, and the \vert spellings.
_BAR_OWNERS = ('\\left', '\\right', '\\middle', '\\bigl', '\\bigr', '\\big',
               '\\Bigl', '\\Bigr', '\\Big', '\\biggl', '\\biggr', '\\bigg',
               '\\Biggl', '\\Biggr', '\\Bigg')
_OPEN_DELIMS = {'(': ')', '[': ']', '{': '}'}
# plain-TeX fractions that name their own delimiter pair, `|` included
_DELIM_PAIR_OWNERS = ('\\abovewithdelims', '\\atopwithdelims')
# \lvert / \rvert / \vert are the same bar; agents re-type delimiter
# spellings exactly as they re-type \rightarrow for \to.  \| (norm) is a
# different operator and is deliberately left alone.
_VERT_RE = re.compile(r'\\[lr]?vert(?![A-Za-z])')


def _lower_bare_abs(latex):
     r"""Pair bare vertical bars and lower each pair to ``\abs{...}``.

     The grammar cannot do this: `|` is its own opener and closer, so a rule
     wide enough for `|x+1|` puts the LALR machine in a shift/reduce conflict
     at every middle bar (measured: it resolves as shift and then rejects
     even `|x|`).  Pairing is a scanning decision — first bar of a scope
     opens, next bar in the SAME delimiter scope closes — which is the
     reading humans use (`|a|b|c|` is `|a| b |c|`).  Unpaired bars are left
     verbatim, so the retained set-builder spelling `\{x|P\}` and matrix
     preambles pass through untouched.
     """
     latex = _VERT_RE.sub('|', latex)
     if '|' not in latex:
          return latex
     out = []
     # one entry per open delimiter scope: [closing char, pending bar mark,
     # bars still to skip].  mark = index in `out` just after the opening
     # bar.  Collection literals have already been lowered, so a surviving
     # `\{...\}` is a set-builder whose FIRST bar is the condition
     # separator, not an absolute value.
     scopes = [[None, None, 0]]
     i, n = 0, len(latex)
     while i < n:
          ch = latex[i]
          if ch == '\\':
               j = i + 1
               while j < n and latex[j].isalpha():
                    j += 1
               word = latex[i:j]
               if j == i + 1:
                    # escaped single character: \| \{ \} \( ...
                    nxt = latex[i + 1:i + 2]
                    if nxt in _OPEN_DELIMS:
                         scopes.append([_OPEN_DELIMS[nxt], None,
                                        1 if nxt == '{' else 0])
                    elif nxt in (')', ']', '}') and len(scopes) > 1 \
                            and scopes[-1][0] == nxt:
                         scopes.pop()
                    out.append(latex[i:i + 2])
                    i += 2
                    continue
               if word in _BAR_OWNERS:
                    # emit the sizing command together with its delimiter so
                    # the bar never reaches the pairing logic
                    k = j
                    while k < n and latex[k] in ' \t':
                         k += 1
                    if k < n and latex[k] == '|':
                         out.append(latex[i:k + 1])
                         i = k + 1
                         continue
               if word in _DELIM_PAIR_OWNERS:
                    out.append(word)
                    i = _copy_delims(latex, j, out, 2)
                    continue
               if word.startswith('\\text') or word in ('\\operatorname',
                                                        '\\color'):
                    # prose arguments are copied verbatim: a bar there is a
                    # character, not a delimiter
                    k = _brace_span(latex, j)
                    out.append(latex[i:k])
                    i = k
                    continue
               out.append(word)
               i = j
               continue
          if ch in _OPEN_DELIMS:
               scopes.append([_OPEN_DELIMS[ch], None, 0])
          elif ch in (')', ']', '}'):
               if len(scopes) > 1 and scopes[-1][0] == ch:
                    scopes.pop()
          elif ch == '|':
               scope = scopes[-1]
               if scope[2]:
                    scope[2] -= 1
               elif scope[1] is None:
                    scope[1] = len(out) + 1
                    out.append('|')
                    i += 1
                    continue
               else:
                    mark = scope[1]
                    body = ''.join(out[mark:])
                    del out[mark - 1:]
                    out.append('\\abs{' + body + '}')
                    scope[1] = None
                    i += 1
                    continue
          out.append(ch)
          i += 1
     return ''.join(out)


def _copy_delims(latex, pos, out, count):
     """Copy ``count`` delimiter tokens verbatim; return the new position."""
     n = len(latex)
     while count and pos < n:
          if latex[pos] in ' \t':
               out.append(latex[pos])
               pos += 1
               continue
          if latex[pos] == '\\' and pos + 1 < n:
               out.append(latex[pos:pos + 2])
               pos += 2
          else:
               out.append(latex[pos])
               pos += 1
          count -= 1
     return pos


def _brace_span(latex, pos):
     """Index just past the balanced ``{...}`` group starting at/after pos."""
     n = len(latex)
     while pos < n and latex[pos] in ' \t':
          pos += 1
     if pos >= n or latex[pos] != '{':
          return pos
     depth = 0
     while pos < n:
          if latex[pos] == '\\':
               pos += 2
               continue
          if latex[pos] == '{':
               depth += 1
          elif latex[pos] == '}':
               depth -= 1
               if depth == 0:
                    return pos + 1
          pos += 1
     return n


class MathParser(object):
     tokens = MathLexer.tokens
     literals = MathLexer.literals
     # A following ^/_ completes the current TeX index pair (x_a^b), rather
     # than starting a second decoration on the already-indexed expression.
     # Declaring the choice keeps the LALR table conflict-free now that
     # postfix factorials can appear on either side of one index pair.
     precedence = (
         ('right', '^', '_'),
     )

     def __init__(self, notation, command_names=None):
         self.notation = notation
         self.command_names = (MathLexer.KNOWN_COMMANDS
                               if command_names is None
                               else frozenset(command_names))
         self.yacc = yacc.yacc(module=self,start='formula')

     def parse(self, input):
         input = _normalize_plain_array_envs(input)
         input = _normalize_collection_literals(input)
         input = _lower_bare_abs(input)
         self.notation.clear()
         try:
             return self.yacc.parse(
                 input, lexer=MathLexer(command_names=self.command_names))
         except Exception:
             # TeX reads \frac12 as \frac{1}{2} (one token per unbraced
             # argument), but this dialect's lexer fuses the digit run
             # into a single number token, so that spelling arrives here
             # only as a syntax error. Retry with the TeX reading of
             # adjacent frac digits. Spellings that parse token-per-
             # argument (\frac 13 15 = 13/15) never reach the retry and
             # keep their dialect meaning.
             rewritten = re.sub(r'(\\frac)(\d)(\d)', r'\1{\2}{\3}', input)
             if rewritten == input:
                 raise
             self.notation.clear()
             return self.yacc.parse(
                 rewritten, lexer=MathLexer(command_names=self.command_names))

     def p_formula(self, p):
         'formula : logical-expr'
         p[0] = p[1]

     def p_formula_command_0(self, p):
         'formula : COMMAND'
         p[0] = self.notation.setf(Symbol(p[1], command=True), (None,()))

     def p_formula_command_1(self, p):
         'formula : COMMAND logical-expr'
         p[0] = self.notation.setf(Symbol(p[1], command=True), (None,(p[2],)))

     def p_formula_command_1_param(self, p):
         '''formula : COMMAND '[' comma-list ']' logical-expr'''
         p[0] = self.notation.setf(Symbol(p[1], command=True), (p[3],(p[5],)))

     def p_formula_command_2(self, p):
         'formula : COMMAND logical-expr Box logical-expr'
         p[0] = self.notation.setf(Symbol(p[1], command=True), (None,(p[2],p[4])))

     def p_formula_command_2_param(self, p):
         '''formula : COMMAND '[' comma-list ']' subformula Box subformula'''
         p[0] = self.notation.setf(Symbol(p[1], command=True), (p[3],(p[5],p[7],)))

     def p_formula_above(self, p):
          'formula : subformula above DIMEN subformula'
          p[0] = self.notation.setf(Symbol(p[2],dimen=p[3]),(p[1],p[4]))

     def p_formula_abovewithdelims(self, p):
         'formula : subformula abovewithdelims delim delim DIMEN subformula'
         p[0] = self.notation.setf(Symbol(p[2],delim1=p[3],delim2=p[4],dimen=p[5]),(p[1],p[6]))

     def p_formula_atop(self, p):
         'formula : subformula atop subformula'
         p[0] = self.notation.setf(Symbol(p[2]),(p[1],p[3]))

     def p_formula_atopwithdelims(self, p):
         'formula : subformula atopwithdelims delim delim subformula'
         p[0] = self.notation.setf(Symbol(p[2],delim1=p[3],delim2=p[4]),(p[1],p[5]))

     def p_formula_brace(self, p):
         'formula : subformula brace subformula'
         p[0] = self.notation.setf(Symbol(p[2]), (p[1],p[3]))

     def p_formula_brake(self, p):
         'formula : subformula brack subformula'
         p[0] = self.notation.setf(Symbol(p[2]), (p[1],p[3]))

     def p_delim(self, p):
         '''delim : '.'
                  | '|'
                  | open
                  | close'''
         p[0] = p[1]     
         
     def p_logical_expr(self, p):
         'logical-expr : and-expr'
         p[0] = p[1]
         
     def  p_logical_expr_or(self, p):
         'logical-expr : logical-expr lor and-expr'
         f = self.notation.getf(p[1], Notation.O_LIST)
         if f is None:
            p[0] = self.notation.setf(Notation.O_LIST, [p[1], p[3]])
         else:
            f.args.append(p[3])
            p[0] = p[1]
        
     def p_and_expr(self, p):
         'and-expr : not-expr'
         p[0] = p[1]
         
     def p_and_expr_not(self, p):
         'and-expr : and-expr land not-expr'
         f = self.notation.getf(p[1], Notation.A_LIST)
         if f is None:
            p[0] = self.notation.setf(Notation.A_LIST, [p[1], p[3]])
         else:
            f.args.append(p[3])
            p[0] = p[1]

         
     def p_not_expr(self, p):
         'not-expr : subformula'
         p[0] = p[1]
         
     def p_not_expr_not(self, p):
         'not-expr : neg subformula'
         p[0] = self.notation.setf(Notation.NEG, (p[2],))
         
         
     def p_subformula(self, p):
         'subformula : comma-list'
         p[0] = p[1]

     def p_comma_item_additive_expr(self, p):
         'comma-item : additive-expr'
         p[0] = p[1]

     def p_comma_item_comparer(self, p):
         'comma-item : additive-expr comparer additive-expr'
         p[0] = self.notation.setf(Symbol('comp', op=p[2]),(p[1],p[3]))

     def p_comparer(self, p):
         '''comparer : '='
                    | in
                    | to
                    | ge
                    | lt
                    | le
                    | leq
                    | leqq
                    | leqslant
                    | lesseqgtr
                    | lesseqqgtr
                    | lessgtr
                    | lesssim
                    | lnapprox
                    | lneq
                    | lneqq
                    | lnsim
                    | lvertneqq
                    | ne
                    | neq
                    | geq
                    | geqq
                    | geqslant
                    | gt
                    | gg
                    | ggg
                    | gggtr
                    | gtreqless
                    | gtreqqless
                    | gtrless
                    | gtrapprox
                    | gets
                    | gnapprox'''
         p[0] = p[1]

     def p_comma_list_additive_expr(self, p):
         'comma-list : comma-item'
         p[0] = p[1]

     def p_command_list_list(self, p):
         '''comma-list : comma-list ',' comma-item'''
         f = self.notation.getf(p[1], Notation.C_LIST)
         if f is None:
             p[0] = self.notation.setf(Notation.C_LIST, [p[1], p[3]])
         else:
             f.args.append(p[3])
             p[0] = p[1]

     def p_additive_expr(self, p):
        'additive-expr : composite-expr'
        p[0] = p[1]

     def p_additive_expr_additive(self, p):
         'additive-expr : additive composite-expr'
         p[0] = self.notation.setf(p[1],(p[2],))

     def p_additive_expr_list(self, p):
        'additive-expr : additive-expr additive composite-expr'
        operand = self.notation.setf(p[2],(p[3],))
        f = self.notation.getf(p[1], Notation.S_LIST)
        if f is None:
           p[0] = self.notation.setf(Notation.S_LIST, [p[1], operand])
        else:
           f.args.append(operand)
           p[0] = p[1]

     def p_additive(self, p):
         '''additive : '+'
                     | '-' '''
         p[0] = Symbol(p[1])

     def p_composite_expr(self, p):
         'composite-expr : expression-list'
         p[0] = p[1]

     def p_composite_expr_index_expr(self, p):
         'composite-expr : index-expr'
         p[0] = self.notation.setf(Notation.INDEX, (None, (p[1][0],p[1][1],None,None)))

     def p_composite_expr_index_expr_list(self, p):
         'composite-expr : index-expr expression-list'
         plist = self.notation.getf(p[2], Notation.P_LIST)
         if plist is not None:
             index = self.notation.getf(plist.args[0], Notation.INDEX)
             if index is not None:
                 dims = (p[1][0],p[1][1],index.args[1][2],index.args[1][3])
                 self.notation.repf(plist.args[0], Func(Notation.INDEX, (index.args[0], dims)))
             else:
                 plist.args[0] = self.notation.setf(Notation.INDEX, (plist.args[0],(p[1][0],p[1][1],None,None)))
             p[0] = p[2]
         else:
             index = self.notation.getf(p[2], Notation.INDEX)
             if index is not None:
                 dims = (p[1][0],p[1][1],index.args[1][2],index.args[1][3])
                 self.notation.repf(p[2], Func(Notation.INDEX, (index.args[0], dims)))
                 p[0] = p[2]
             else:
                 dims = (p[1][0],p[1][1], None, None)
                 p[0] = self.notation.setf(Notation.INDEX, (p[2],dims))

     def p_expr_list_expr(self, p):
         'expression-list : expression'
         p[0] = p[1]

     def p_expr_list_list(self, p):
        'expression-list : expression expression-list'
        f = self.notation.getf(p[2], Notation.P_LIST)
        if f is None:
           p[0] = self.notation.setf(Notation.P_LIST, [p[1],p[2]])
        else:
           f.args.insert(0, p[1])
           p[0] = p[2]

     def p_expr_list_list_sep(self, p):
        '''expression-list : expression '*' expression-list
                           | expression cdot expression-list'''
        # '*' and \cdot are explicit product separators: they build the
        # same P_LIST as juxtaposition, so chains work (a \cdot b \cdot c).
        # An explicit \cdot additionally marks the product as notation
        # (props['cdot']): the writer restores the dots and the legacy
        # calculator must not fold its numeric factors (1 \cdot 2 in a
        # series term stays 1 \cdot 2, never 2).
        f = self.notation.getf(p[3], Notation.P_LIST)
        if f is None:
           p[0] = self.notation.setf(Notation.P_LIST, [p[1],p[3]])
           f = self.notation.getf(p[0], Notation.P_LIST)
        else:
           f.args.insert(0, p[1])
           p[0] = p[3]
        if p.slice[2].type == 'cdot':
           f.props['cdot'] = True

     def p_composite_expr_slash(self, p):
        '''composite-expr : expression-list '/' expression'''
        p[0] = self.notation.setf(Notation.SLASH,(p[1],p[3]))


     def p_expression(self, p):
         'expression : postfix-expr'
         p[0] = p[1]

     def p_postfix_expr_prefactor(self, p):
         'postfix-expr : prefactor-expr'
         p[0] = p[1]

     def p_postfix_expr_indexed(self, p):
         'postfix-expr : indexed-expr'
         p[0] = p[1]

     def p_prefactor_expr_base(self, p):
         'prefactor-expr : postfix-base'
         p[0] = p[1]

     def p_prefactor_expr_factorial(self, p):
         'prefactor-expr : prefactor-expr FACTORIAL'
         p[0] = self.notation.setf(Notation.FACTORIAL, (p[1],))

     def p_postfix_base_scalar(self, p):
         'postfix-base : scalar'
         p[0] = p[1]

     def p_postfix_base_binom(self, p):
         'postfix-base : binom scalar scalar'
         # TeX's two binomial arguments are single tokens/groups. Keeping
         # them scalar prevents a trailing postfix/index from binding to the
         # second argument: \binom{n}{k}^2 is the square of the coefficient,
         # not \binom{n}{k^2}.
         p[0] = self.notation.setf(Notation.BINOM, (p[2], p[3]))

     def p_indexed_expr(self, p):
         'indexed-expr : prefactor-expr index-expr'
         p[0] = self.notation.setf(
             Notation.INDEX,
             (p[1], (None, None, p[2][0], p[2][1])))

     def p_indexed_expr_factorial(self, p):
         'indexed-expr : indexed-expr FACTORIAL'
         p[0] = self.notation.setf(Notation.FACTORIAL, (p[1],))

     def p_expression_dot3(self, p):
         '''expression : '.' '.' '.' '''
         p[0] = Notation.DOT3

     def p_expression_limits_expr(self, p):
         'postfix-base : scalar limits index-expr'
         p[0] = self.notation.setf(Notation.LIMITS, (p[1], (p[3][0], p[3][1])))

     def p_expression_nolimits_expr(self, p):
         'postfix-base : scalar nolimits index-expr'
         p[0] = self.notation.setf(Notation.NOLIMITS, (p[1], (p[3][0], p[3][1])))

     def p_index_expr_subscript(self, p):
        '''index-expr : '_' scalar '''
        p[0] = (None, p[2])

     def p_index_expr_superscript(self, p):
        '''index-expr : '^' scalar '''
        p[0] = (p[2], None)

     def p_index_expr_approach_direction(self, p):
        '''index-expr : '^' additive
                      | '^' '{' additive '}' '''
        # A bare +/- superscript is not an arithmetic expression: in
        # ``\lim_{x \to a^+}`` it is an approach-direction marker.  Keep
        # it as a raw symbol in the power slot so the ordinary index writer
        # round-trips both braced and unbraced input.
        p[0] = (p[2] if len(p) == 3 else p[3], None)

     def p_index_expr_superscript_subscript(self, p):
        '''index-expr : '^' scalar '_' scalar '''
        p[0] = (p[2], p[4])

     def p_index_expr_subscript_superscript(self, p):
        '''index-expr : '_' scalar '^' scalar '''
        p[0] = (p[4], p[2])

     def p_expression_style(self, p):
        '''expression : bf
                      | rm
                      | displaystyle
                      | frak
                      | cal
                      | NEGSP
                      | SP1
                      | SP2
                      | SP3
                      | SP4                      
                      | WS'''
        p[0] = Symbol(p[1])

     # LaTex operators
     def p_expression_unary(self, p):
         'expression : unary-op expression'
         p[0] = self.notation.setf(p[1], (p[2],))

     def p_expression_binary(self, p):
         'expression : binary-op expression expression'
         p[0] = self.notation.setf(p[1], (p[2], p[3]))

     def p_expression_color(self, p):
         'expression : color TEXT expression'
         p[0] = self.notation.setf(Symbol('color', c=p[2]), (p[3],))

     def p_expression_lower(self, p):
         'expression : lower DIMEN expression'
         p[0] = self.notation.setf(Symbol(p[1], dimen=p[2]), (p[3],))

     def p_expression_sqrt(self, p):
         'expression : sqrt expression'
         p[0] = self.notation.setf(Symbol(p[1]), (p[2],))

     def p_expression_sqrt_long(self, p):
        '''expression : sqrt '[' subformula ']'  expression'''
        p[0] = self.notation.setf(Symbol(p[1]), (p[5],p[3]))

     def p_expression_buildrel(self, p):
        'expression : buildrel subformula over expression'
        p[0] = self.notation.setf(Symbol(p[1]),(p[2],p[4]))


     def p_unary_operator(self, p):
         '''unary-op : acute
                     | vec
                     | grave
                     | widehat
                     | widetilde
                     | partial
                     | phantom
                     | boldsymbol
                     | thinspace
                     | textstyle
                     | cancel
                     | bcancel
                     | boxed
                     | Bbb
                     | hat'''
         p[0] = Symbol(p[1])

     def p_binary_operator(self, p):
         '''binary-op : frac
                      | dfrac
                      | cfrac
                      | tfrac'''
         p[0] = Symbol(p[1])

     def p_scalar_term(self, p):
         'scalar : term'
         p[0] = p[1]
         
     def p_scalar_digit(self, p):
         'scalar : DIGIT'

         p[0] = p[1]

     def p_scalar_ref(self, p):
         '''scalar : BREF expression EREF'''
         p[0] = self.notation.setf(Notation.REF, (p[2],))

     def p_scalar_text(self, p):
         '''scalar : text TEXT
                   | textbf TEXT
                   | textit TEXT
                   | textrm TEXT
                   | textsf TEXT
                   | texttt TEXT'''
         p[0] = self.notation.setf(Symbol(p[1]),(p[2],))

     def p_term_literal(self, p):
         'term : LITERAL'
         p[0] = Symbol(p[1])
         
     def p_term_quoted_literal(self, p):
         '''term : '`' LITERAL'''
         p[0] = Symbol(p[2], quoted=True)         

     def p_open(self, p):
         '''open : '('
                 | '['
                 | LBR'''
         p[0] = p[1]

     def p_close(self, p):
         '''close : ')'
                  | ']'
                  | RBR'''
         p[0] = p[1]

     def p_scalar_formula(self, p):
         '''scalar : '{' formula '}' '''
         p[0] = self.notation.setf(Notation.GROUP, (p[2],), br='{}')
         
     def p_scalar_quoted_formula(self, p):
         '''scalar : '`' '{' formula '}' '''
         p[0] = self.notation.setf(Notation.GROUP, (p[3],), br='{}', quoted=True)

     def p_scalar_group(self, p):
         '''scalar : '(' comma-list ')' '''
         items = self._comma_items(p[2])
         if len(items) == 2:
             p[0] = self.notation.setf(Notation.PAIR, tuple(items))
         else:
             p[0] = self.notation.setf(
                 Notation.GROUP, (p[2],), br='()')
         
     def p_scalar_quoted_group(self, p):
         '''scalar : '`' '(' comma-list ')' '''
         p[0] = self.notation.setf(Notation.GROUP,(p[3],), br='()', quoted=True)
         
     def p_scalar_collection(self, p):
         '''scalar : collection '{' comma-list '}' '''
         p[0] = self.notation.setf(
             Notation.COLLECTION, tuple(self._comma_items(p[3])))

     def p_scalar_collection_empty(self, p):
         '''scalar : collection '{' '}' '''
         p[0] = self.notation.setf(Notation.COLLECTION, ())

     def p_scalar_group_b(self, p):
         '''scalar : LBR expression RBR '''
         p[0] = self.notation.setf(Notation.S_GROUP, (p[2],), br='{}')

     def p_scalar_sgroup_b(self, p):
        '''scalar : LBR expression '|' subformula RBR'''
        p[0] = self.notation.setf(Notation.S_GROUP, (p[2],p[4]), br='{}')

     def p_scalar_group_a(self, p):
        '''scalar : '|' expression '|' '''
        p[0] = self.notation.setf(Notation.GROUP, (p[2],), br=Notation.ABS_BR)

     def p_scalar_abs(self, p):
        '''scalar : abs '{' subformula '}' '''
        # Internal lowering target for bare `|...|` around a composite body
        # (see _lower_bare_abs). Builds exactly the node the bare single-
        # scalar rule builds, so nothing downstream sees a second spelling.
        p[0] = self.notation.setf(Notation.GROUP, (p[3],), br=Notation.ABS_BR)

     def p_scalar_floor(self, p):
        '''scalar : lfloor subformula rfloor'''
        p[0] = self.notation.setf(Notation.GROUP, (p[2],),
                                  br=Notation.FLOOR_BR)

     def p_scalar_ceil(self, p):
        '''scalar : lceil subformula rceil'''
        p[0] = self.notation.setf(Notation.GROUP, (p[2],),
                                  br=Notation.CEIL_BR)

     def p_scalar_vgroup_floor(self, p):
        '''scalar : left lfloor subformula right rfloor'''
        p[0] = self.notation.setf(Notation.V_GROUP, (p[3],),
                                  br=Notation.FLOOR_BR)

     def p_scalar_vgroup_ceil(self, p):
        '''scalar : left lceil subformula right rceil'''
        p[0] = self.notation.setf(Notation.V_GROUP, (p[3],),
                                  br=Notation.CEIL_BR)

     def p_scalar_vgroup(self, p):
        '''scalar : left open subformula right close'''
        br = p[2] + p[5]
        items = self._comma_items(p[3])
        if br == '()' and len(items) == 2:
            p[0] = self.notation.setf(Notation.PAIR, tuple(items))
        elif br == '\\{\\}':
            p[0] = self.notation.setf(
                Notation.COLLECTION, tuple(items))
        else:
            p[0] = self.notation.setf(
                Notation.V_GROUP, (p[3],), br=br)

     def p_scalar_vgroup_a(self, p):
        '''scalar : left '|' subformula right '|' '''
        p[0] = self.notation.setf(Notation.V_GROUP, (p[3],), br='||')

     def p_scalar_operator(self, p):
         'scalar : operator'
         p[0] = p[1]

     def p_operator(self, p):
         '''operator : operatorname TEXT '(' comma-list ')' '''
         p[0] = self.notation.setf(Notation.FUNC, (Symbol(p[2]), p[4]), fmt='operatorname')

     def p_scalar_operator_index_expr(self, p):
         '''scalar : operatorname TEXT index-expr '(' comma-list ')' '''
         index_expr = self.notation.setf(Notation.INDEX, (Symbol(p[2]), (None, None, p[3][0], p[3][1])))
         p[0] = self.notation.setf(Notation.FUNC, (index_expr, p[5]), fmt='operatorname')

     def p_scalar_array(self, p):
         '''scalar : array '{' row-list '}'
                   | pmatrix '{' row-list '}'
                   | matrix '{' row-list '}'
                   | bmatrix '{' row-list '}'
                   | Bmatrix '{' row-list '}'
                   | vmatrix '{' row-list '}'
                   | Vmatrix '{' row-list '}'
                   | smallmatrix '{' row-list '}' '''
         p[0] = self.notation.setf(Symbol(p[1]), p[3])

     def p_scalar_cases(self, p):
         '''scalar : cases '{' row-list '}' '''
         p[0] = self.notation.setf(Symbol(p[1]), p[3])

     def p_row_list(self, p):
         'row-list : column-list'
         p[0] = [p[1]]

     def p_row_list_list(self, p):
         '''row-list : row-list cr column-list'''
         p[1].append(p[3])
         p[0] = p[1]

     def p_column_list(self, p):
         'column-list : subformula'
         p[0] = [p[1]]

     def p_column_list_list(self, p):
         '''column-list : column-list '&' subformula'''
         p[1].append(p[3])
         p[0] = p[1]

     def _comma_items(self, sym):
         """Items carried by comma syntax, without leaking C_LIST as type.

         A bare top-level comma list remains the legacy C_LIST used for
         command arguments and relation systems. Delimiter productions call
         this helper only when their delimiters provide the semantic type:
         parentheses for exactly two ordered items, escaped braces for a
         finite collection.
         """
         f = self.notation.getf(sym, Notation.C_LIST)
         return list(f.args) if f is not None else [sym]

     # Error rule for syntax errors
     def p_error(self, p):
         raise Exception('Syntax error in the input expression')
         #self.errorf = True

if __name__ == "__main__":
    n = Notation()
    m = MathParser(n)

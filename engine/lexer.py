#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 20:14:37 2020

@author: semyonc
"""
import ply.lex as lex
from value import IntegerValue, FloatValue

class MathLexer(object):
     """ MathLexer """
     # Bang words are commands only when admitted here (or explicitly passed
     # by a caller loading additional prompt commands).  Everything else is
     # ordinary LaTeX letters followed by the postfix FACTORIAL token.
     KNOWN_COMMANDS = frozenset({
        # legacy cmd_* actions
        'add', 'addex', 'clear', 'closure', 'debug', 'dump', 'echo-off',
        'echo-on', 'goal', 'mul', 'mulex', 'rules', 'track',
        # committed notebook commands + dispatcher built-ins
        'commands', 'conv', 'diff', 'do', 'expand', 'help', 'int', 'lim',
        'model', 'prove', 'solve',
     })

     def __init__(self, command_names=None, **kwargs):
        self.command_names = (MathLexer.KNOWN_COMMANDS if command_names is None
                              else frozenset(command_names))
        self.lexer = lex.lex(module=self, **kwargs)
    
     # List of token names
     tokens = (
        'LITERAL',
        'COMMAND',
        'FACTORIAL',
        'TEXT',
        'DIGIT',
        'DIMEN',
        'LBR',
        'RBR',
        'BREF',
        'EREF',
        'NEGSP',
        'SP1',
        'SP2',
        'SP3',
        'SP4',
        'WS',
        'lt',
        'le',
        'leq',
        'leqq',
        'leqslant',
        'lesseqgtr',
        'lesseqqgtr',
        'lessgtr',
        'lesssim',
        'lnapprox',
        'lneq',
        'lneqq',
        'lnsim',
        'lvertneqq',
        'ne',
        'neq',
        'geq',
        'geqq',
        'geqslant',
        'ge',
        'gt',
        'gg',
        'ggg',
        'gggtr',
        'gtreqless',
        'gtreqqless',
        'gtrless',
        'gtrapprox',
        'gnapprox',
        'above',
        'abovewithdelims',
        'atop',
        'atopwithdelims',
        'acute',
        'buildrel',
        'brace',
        'brack',
        'over',
        'vec',
        'widehat',
        'widetilde',
        'hat',
        'grave',
        'rm',
        'frac',
        'dfrac',
        'cfrac',
        'tfrac',
        'binom',
        'lower',
        'color',
        'sqrt',
        'partial',
        'phantom',
        'boldsymbol',
        'textstyle',
        'thinspace',
        'bf',
        'cancel',
        'bcancel',
        'left',
        'right',
        'text',
        'textbf',
        'textit',
        'textrm',
        'textsf',
        'texttt',
        'displaystyle',
        'frak',
        'cal',
        'boxed',
        'array',
        'pmatrix',
        'matrix',
        'bmatrix',
        'Bmatrix',
        'vmatrix',
        'Vmatrix',
        'smallmatrix',
        'cr',
        'cases',
        'in',
        'Bbb',
        'to',
        'cdot',
        'operatorname',
        'limits',
        'nolimits',
        'Box',
        'gets',
        'lor',
        'land',
        'neg'
     )
          
     # Declare the state
     states = (
       ('text','exclusive'),
     )
     
     literals = ('&','^','_','{','}','~','|','(',')',',','/','*','+','-','[',']','=', '.', ':', '`')
    
     # Regular expression rules for simple tokens
     t_LBR    = r'\\{'
     t_RBR    = r'\\}'
     
     t_NEGSP   = r'\\!'
     t_SP1     = r'\\,'
     t_SP2     = r'\\\:'
     t_SP3     = r'\\>'
     t_SP4     = r'\\;'
     t_WS      = r'\\(?=\s)'
     
     t_BREF    = r'\[\['
     t_EREF    = r'\]\]'
             
     t_text_ignore = ' {'         

     def t_DIMEN(self, t):
         r'([0-9]*[.])?[0-9]+(em|ex|pt|pc|mu|cm|mm|in|px)'
         t.value = (float(t.value[:-2]),t.value[-2:])
         return t

     def t_DIGIT(self, t):
         r'[0-9]+(\.[0-9]+)?'
         if '.' in t.value:
             t.value = FloatValue(float(t.value))
         else:
             t.value = IntegerValue(int(t.value))
         return t
     
     def t_COMMAND(self, t):
         r'[A-Za-z_][\w\-]*!'
         name = t.value[:-1]
         if name not in self.command_names:
             # Unknown bang words are implicit multiplication with factorial
             # on the last scalar (xy! = x y!), not executable commands.
             # Return the first letter and rewind so later lexer calls split
             # the rest and finally emit FACTORIAL for the bang.
             t.type = 'LITERAL'
             t.value = t.value[0]
             t.lexer.lexpos = t.lexpos + 1
         return t

             
     def t_LITERAL(self, t):
         r'\\[A-Za-z]+|[A-Za-z]|\#\w+|\#\#'
         val = t.value
         if val[0] == '\\':
           if val == '\\rightarrow':
             # normalize the synonym arrow to the canonical \to binder so
             # \lim_{n \rightarrow \infty} parses as a comparison, not a
             # product of plain symbols
             val = t.value = '\\to'
           if val in ('\\cdots', '\\dots', '\\hdots', '\\dotsb',
                      '\\dotsc', '\\dotsi', '\\dotsm', '\\dotso'):
             # the inline dots family is pure typography for the same
             # sequence continuation; one canonical name keeps expression
             # and claim comparison from splitting on the spelling
             # (\vdots/\ddots stay distinct: matrix typography)
             val = t.value = '\\ldots'
           if val[1:] in MathLexer.tokens:
             t.type = val[1:]
           else:
             t.value = val
         if val.startswith('\\text') or val == '\\color' \
                 or val.startswith('\\operatorname') :
            t.lexer.code_start = t.lexer.lexpos
            t.lexer.begin('text')                    
         return t
                   
     def t_LITERAL_CHAR(self, t):
         r'\\[\(\)\[\]\&\#\.\_\^\+]'
         t.type = 'LITERAL'
         return t
     
     def t_text_oper(self, t):
         r'\\.|[,.#]'
         pass
     
     def t_text_end(self, t):
         r'(?<!\\)}'
         start = t.lexer.lexdata.index('{', t.lexer.code_start, t.lexer.lexpos)
         val = t.lexer.lexdata[start+1:t.lexer.lexpos-1]
         t.type = "TEXT"
         t.lexer.lineno += val.count('\n')
         t.lexer.begin('INITIAL')
         t.value = val           
         return t
     
     def t_text_error(self, t):
         t.lexer.skip(1)
         
     
     def t_ASSIGN(self, t):
         r'{:=}'
         t.type = 'LITERAL'
         return t
     
     def t_excl(self, t):
         r'!'
         t.type = 'FACTORIAL'
         return t
              
     def t_error(self, t):
         raise Exception("Illegal character '%s'" % t.value[0])
         
     def input(self, s):
         self.lexer.input(s)
         
     def token(self):
         return self.lexer.token()
            
     # A string containing ignored characters (spaces and tabs)
     t_ignore  = ' \t\n'
          
     # Test it output
     def test(self, data):
         self.input(data)
         while True:
              tok = self.token()
              if not tok: 
                  break
              print(tok)
        
if __name__ == "__main__":
    m = MathLexer()

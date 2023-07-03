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
     def __init__(self, **kwargs):
        self.lexer = lex.lex(module=self, **kwargs)
    
     # List of token names
     tokens = (
        'LITERAL',
        'COMMAND',
        'TEXT',
        'DIGIT',
        'DIMEN',
        'LBR',
        'RBR',
        'REF',
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
     
     literals = ('&','^','_','{','}','~','|','(',')',',','/','*','+','-','[',']','=', '.', ':')
    
     # Regular expression rules for simple tokens
     t_LBR    = r'\\{'
     t_RBR    = r'\\}'
     
     t_NEGSP   = r'\\!'
     t_SP1     = r'\\,'
     t_SP2     = r'\\\:'
     t_SP3     = r'\\>'
     t_SP4     = r'\\;'
     t_WS      = r'\\(?=\s)'
             
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
         r'\w[\w\-]*!'
         return t

             
     def t_LITERAL(self, t):
         r'\\[A-Za-z]+|[A-Za-z]|\#\w+|\#\#' 
         val = t.value
         if val[0] == '\\':
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
         
     def t_REF(self, t):
         #r'!-?\d+'
         r'\[\[\-?\d+\]\]'
         t.value = int(t.value[2:-2])
         return t    
     
     def t_ASSIGN(self, t):
         r'{:=}'
         t.type = 'LITERAL'
         return t
     
     def t_excl(self, t):
         r'!'
         t.type = 'LITERAL'
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
         
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 19:15:16 2020

@author: semyonc
"""
import ply.yacc as yacc
from lexer import MathLexer
from notation import Notation, Symbol, Func

class MathParser(object):
     tokens = MathLexer.tokens
     literals = MathLexer.literals

     def __init__(self, notation):
         self.notation = notation
         self.yacc = yacc.yacc(module=self,start='formula')

     def parse(self, input):
         self.notation.clear()
         return self.yacc.parse(input, lexer=MathLexer())

     def p_formula(self, p):
         'formula : logical-expr'
         p[0] = p[1]

     def p_formula_command_0(self, p):
         'formula : COMMAND'
         p[0] = self.notation.setf(Symbol(p[1]), (None,()))

     def p_formula_command_1(self, p):
         'formula : COMMAND logical-expr'
         p[0] = self.notation.setf(Symbol(p[1]), (None,(p[2],)))

     def p_formula_command_1_param(self, p):
         '''formula : COMMAND '[' comma-list ']' logical-expr'''
         p[0] = self.notation.setf(Symbol(p[1]), (p[3],(p[5],)))

     def p_formula_command_2(self, p):
         'formula : COMMAND logical-expr Box logical-expr'
         p[0] = self.notation.setf(Symbol(p[1]), (None,(p[2],p[4])))

     def p_formula_command_2_param(self, p):
         '''formula : COMMAND '[' comma-list ']' subformula Box subformula'''
         p[0] = self.notation.setf(Symbol(p[1]), (p[3],(p[5],p[7],)))

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

     def p_subformula_expr_comparer(self, p):
         'subformula : additive-expr comparer comma-list'
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
         'comma-list : additive-expr'
         p[0] = p[1]

     def p_command_list_list(self, p):
         '''comma-list : comma-list ',' additive-expr'''
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

     def p_composite_expr_slash(self, p):
        '''composite-expr : expression '/' expression'''
        p[0] = self.notation.setf(Notation.SLASH,(p[1],p[3]))

     def p_composite_expr_star(self, p):
        '''composite-expr : expression '*' expression
                          | expression cdot expression'''
        p[0] = self.notation.setf(Notation.STAR,(p[1],p[3]))


     def p_expression(self, p):
        'expression : scalar'
        p[0] = p[1]

     def p_expression_dot3(self, p):
         '''expression : '.' '.' '.' '''
         p[0] = Notation.DOT3

     def p_expression_limits_expr(self, p):
         'expression : scalar limits index-expr'
         p[0] = self.notation.setf(Notation.LIMITS, (p[1], (p[3][0], p[3][1])))

     def p_expression_nolimits_expr(self, p):
         'expression : scalar nolimits index-expr'
         p[0] = self.notation.setf(Notation.NOLIMITS, (p[1], (p[3][0], p[3][1])))

     def p_expression_index_expr(self, p):
         'expression : scalar index-expr'
         p[0] = self.notation.setf(Notation.INDEX, (p[1], (None, None, p[2][0], p[2][1])))

     def p_index_expr_subscript(self, p):
        '''index-expr : '_' scalar '''
        p[0] = (None, p[2])

     def p_index_expr_superscript(self, p):
        '''index-expr : '^' scalar '''
        p[0] = (p[2], None)

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
                      | tfrac
                      | binom'''
         p[0] = Symbol(p[1])

     def p_scalar_term(self, p):
         'scalar : term'
         p[0] = p[1]

     def p_scalar_digit(self, p):
         'scalar : DIGIT'

         p[0] = p[1]

     def p_scalar_ref(self, p):
         'scalar : REF'
         p[0] = self.notation.setf(Notation.REF, (p[1],))

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

     def p_scalar_group(self, p):
         '''scalar : '(' comma-list ')' '''
         p[0] = self.notation.setf(Notation.GROUP,(p[2],), br='()')
         
     def p_scalar_group_b(self, p):
         '''scalar : LBR expression RBR '''
         p[0] = self.notation.setf(Notation.S_GROUP, (p[2],), br='{}')

     def p_scalar_sgroup_b(self, p):
        '''scalar : LBR expression '|' subformula RBR'''
        p[0] = self.notation.setf(Notation.S_GROUP, (p[2],p[4]), br='{}')

     def p_scalar_group_a(self, p):
        '''scalar : '|' expression '|' '''
        p[0] = self.notation.setf(Notation.GROUP, (p[2],), br='||')

     def p_scalar_vgroup(self, p):
        '''scalar : left open subformula right close'''
        p[0] = self.notation.setf(Notation.V_GROUP, (p[3],), br=p[2]+p[5])

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
         '''scalar : array '{' row-list '}' '''
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

     # Error rule for syntax errors
     def p_error(self, p):
         raise Exception('Syntax error in the input expression')
         #self.errorf = True

if __name__ == "__main__":
    n = Notation()
    m = MathParser(n)
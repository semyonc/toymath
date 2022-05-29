from unittest import TestCase
from comparer import *
from processor import MathProcessor
from preprocessor import Preprocessor


class TestScenario(TestCase):

    @staticmethod
    def execute_compare(expr1, expr2, params):
        notation1 = Notation()
        p1 = MathParser(notation1)
        sym1 = p1.parse(expr1)
        notation2 = Notation()
        p2 = MathParser(notation2)
        sym2 = p2.parse(expr2)
        cmp = NotationComparer(sym2, notation2, params)
        return cmp.match(sym1, notation1) is not None

    @staticmethod
    def check(expr1, expr2):
        notation1 = Notation()
        p1 = MathParser(notation1)
        sym1 = p1.parse(expr1)
        processor = MathProcessor()
        outsym, notation = processor(sym1, notation1, {}, {})
        notation2 = Notation()
        p2 = MathParser(notation2)
        sym2 = p2.parse(expr2)
        notation3 = Notation()
        preprocessor = Preprocessor(notation2, notation3, {}, {})
        sym3 = preprocessor(sym2)
        cmp = NotationComparer(sym3, notation3, [])
        return cmp.match(outsym, notation) is not None

    def assertCompare(self, expr1, expr2, params=[]):
        return self.assertTrue(self.execute_compare(expr1, expr2, params))

    def checkEqual(self, expr1, expr2):
        return self.assertTrue(self.check(expr1, expr2))

    def test_pattern1(self):
        self.assertCompare('\\frac 1 2', '\\frac a b',
                           [('a', NotationParam.Value), ('b', NotationParam.Value)])

    def test_pattern2(self):
        self.assertCompare('a + b + c', 'c + b + a')

    def test_pattern3(self):
        self.assertCompare('xy + ab + c', 'c + yx + ba')

    def test_pattern4(self):
        self.assertCompare('xyz', 'x...x', [('x', NotationParam.List)])

    def test_pattern5(self):
        self.assertCompare('x + y + z', 'x+...+x', [('x', NotationParam.List)])

    def test_2x2(self):
        self.checkEqual("2 2", "4")

    def test_frac1(self):
        self.checkEqual("\\frac 2 3 x + \\frac 1 5 x", "\\frac 13 15 x")

    def test_x1(self):
        self.checkEqual("(x+1)(x+1)", "(x+1)^2")

    def test_mul1(self):
        self.checkEqual("mul! (x+1)(x-1)", "x^2 -1")

    def test_sin1(self):
        self.checkEqual("\\sin x \\sin x", "(\\sin x)^2")

    def test_n1(self):
        self.assertCompare('100', 'n', [('n', NotationParam.N)])

    def test_n2(self):
        self.assertCompare('2x', 'nx', [('n', NotationParam.N)])

    def test_pw1(self):
        self.assertCompare('(x+y)^2', '(z)^n', [('z', NotationParam.Any),
                                                ('n', NotationParam.N)])

    def test_pw2(self):
        self.assertCompare('(x+1)^2', '(z)^n', [('z', NotationParam.Any),
                                                ('n', NotationParam.N)])

    def test_mul2(self):
        self.checkEqual("mul! (x+1)^3(x-1)", "{x^4+2x^3-2x-1}")

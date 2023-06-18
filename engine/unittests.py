from unittest import TestCase
from comparer import *
from processor import MathProcessor
from preprocessor import Preprocessor
from prolog import PrologModel, Rule, Term


def execute_compare(expr1, expr2, params):
    notation1 = Notation()
    p1 = MathParser(notation1)
    sym1 = p1.parse(expr1)
    notation2 = Notation()
    p2 = MathParser(notation2)
    sym2 = p2.parse(expr2)
    cmp = NotationParametrizedComparer(sym2, notation2, params)
    return cmp.match(sym1, notation1) is not None


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
    cmp = NotationParametrizedComparer(sym3, notation3, [])
    return cmp.match(outsym, notation) is not None


def compare(sym, notation1, value, ctx=None):
    if isinstance(sym, list) and isinstance(value, list):
        for i in range(len(sym)):
            if not compare(sym[i], notation1, value[i], ctx=Notation.S_LIST):
                return False
        return True
    else:
        notation2 = Notation()
        p2 = MathParser(notation2)
        sym2 = p2.parse(value)
        cmp = NotationComparer(sym2, notation2)
        return cmp.match(sym, notation1, ctx) is not None


def execute_unify(expr1, expr2, results1, results2):
    notation1 = Notation()
    p1 = MathParser(notation1)
    sym1 = p1.parse(expr1)
    notation2 = Notation()
    p2 = MathParser(notation2)
    sym2 = p2.parse(expr2)
    subst1 = defaultdict()
    subst2 = defaultdict()
    comparer = UnifyComparer(sym2, notation2, subst2)
    if comparer.unify(sym1, notation1, subst1):
        shared_items1 = {k: subst1[k] for k in subst1 if k in results1 and compare(subst1[k], notation2, results1[k])}
        shared_items2 = {k: subst2[k] for k in subst2 if k in results2 and compare(subst2[k], notation1, results2[k])}
        return len(shared_items1) == len(subst1) and len(shared_items2) == len(subst2)
    return False


class TestScenario(TestCase):

    def assertCompare(self, expr1, expr2, params=[]):
        return self.assertTrue(execute_compare(expr1, expr2, params))

    def assertUnify(self, expr1, results1, expr2, results2):
        return self.assertTrue(execute_unify(expr1, expr2, results1, results2))

    def checkEqual(self, expr1, expr2):
        return self.assertTrue(check(expr1, expr2))

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

    def test_unify1(self):
        self.assertUnify('x + y + z', {}, '##', {})

    def test_unify2(self):
        self.assertUnify('x + y + z', {}, '#X + #Y + #Z', {'#X': 'x', '#Y': 'y', '#Z': 'z'})

    def test_unify3(self):
        self.assertUnify('#X + y + #X', {'#X': 'x'}, 'x + y + x', {})

    def test_unify4(self):
        self.assertUnify("x + y + z", {}, "x + ... + #X", {'#X': ['y', 'z']})

    def test_unify5(self):
        self.assertUnify("xyz", {}, "x...#X", {'#X': ['y', 'z']})

    def test_unify6(self):
        self.assertUnify("\\operatorname{girl}(\\text{Alise})", {}, "\\operatorname{girl}(#X)", {'#X': '\\text{Alise}'})

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

    def test_prolog1(self):
        m = PrologModel([
            Rule(Term("\\operatorname{child}(#X)"), [Term("\\operatorname{boy}(#X)")]),
            Rule(Term("\\operatorname{child}(#X)"), [Term("\\operatorname{girl}(#X)")]),
            Rule(Term("\\operatorname{girl}(\\text{Alise})")),
            Rule(Term("\\operatorname{boy}(\\text{Alex})"))
        ])
        results = [LaTexWriter(n)(s['#Q']) for s, n in m.search(Term("\\operatorname{child}(#Q)"))]
        self.assertTrue(len(results) == 2 and '\\text{Alise}' in results and '\\text{Alex}' in results)

    def test_prolog2(self):
        m = PrologModel([
            Rule(Term("\\operatorname{power}(#Z,#X,#Y)"), [Term("#Z = #X^#Y")]),
        ])
        results = [LaTexWriter(n)(s['#Y']) for s, n in m.search(Term("\\operatorname{power}(x^2,#X,#Y)"))]
        self.assertTrue(len(results) == 1 and results[0] == '{2}')
        results = [LaTexWriter(n)(s['#Y']) for s, n in m.search(Term("\\operatorname{power}(x^(-y),#X,#Y)"))]
        self.assertTrue(len(results) == 1 and results[0] == '(-y)')

    def test_prolog3(self):
        m = PrologModel([
            Rule(Term("\\operatorname{eval}(Z,X)"), [Term("Z \\gets X")]),
        ])
        results = [LaTexWriter(n)(s['Z']) for s, n in m.search(Term("\\operatorname{eval}(Z,2+3)"))]
        self.assertTrue(len(results) == 1 and results[0] == '{5}')

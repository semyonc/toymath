import unittest
from comparer import *
from processor import MathProcessor
from preprocessor import Preprocessor
from llm_comparer import LLMComparer

import os
from dotenv import load_dotenv
load_dotenv()

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

def execute_unify_llm(expr1, expr2, results2):
    notation_p1 = Notation()
    p1 = MathParser(notation_p1)
    sym1 = p1.parse(expr1)
    notation_w1 = Notation()
    preprocessor = Preprocessor(notation_p1, notation_w1, {}, {})
    sym1 = preprocessor(sym1)

    notation_p2 = Notation()
    p2 = MathParser(notation_p2)
    sym2 = p2.parse(expr2)
    notation_w2 = Notation()
    preprocessor = Preprocessor(notation_p2, notation_w2, {}, {})
    sym2 = preprocessor(sym2)

    subst1 = defaultdict()
    subst2 = defaultdict()
    comparer = LLMComparer(sym2, notation_w2, subst2)

    if comparer.unify(sym1, notation_w1, subst1):
        print('LLM: ' + comparer.last_answer)
        shared_items2 = {
            k: subst2[k]
            for k in subst2
            if k in results2 and compare(subst2[k], notation_w1, results2[k])
        }
        return len(shared_items2) == len(results2)
    return False

class LlmTestScenario(unittest.TestCase):

    def test_openapi_key_available(self):
        return self.assertNotEqual(os.getenv('OPENAI_API_KEY'), '')
   
    def test_llm_unify1(self):
        return self.assertTrue(execute_unify_llm('x^2 + 3y', 'X + Y', {'X': 'x^2', 'Y': '3y'}))
    
    def test_llm_unify2(self):
        return self.assertTrue(execute_unify_llm('5x(x^2 + 3y)', 'XY', {'X': '5x', 'Y': 'x^2 + 3y'}))
    
    def test_llm_unify3(self):
        return self.assertTrue(execute_unify_llm('\\sin^2 x + \\cos^2 x', 'X^2 + Y^2', {'X': '\\sin x', 'Y': '\\cos x'}))
    
    def test_llm_unify4(self):
        return self.assertTrue(execute_unify_llm('\\sin^2 x + 2{\\sin x}{\\cos y} + \\cos^2 y', 'X^2 + 2XY + Y^2', {'X': '\\sin x', 'Y': '\\cos y'}))

    def test_llm_unify5(self):
        return self.assertTrue(execute_unify_llm('3x(x^2 + 3y + z)', '3XY', {'X': 'x', 'Y': 'x^2 + 3y + z'}))
    
    def test_llm_unify6(self):
        return self.assertTrue(execute_unify_llm('\\sin^2 x', 'a^N', {'a': '\\sin x', 'N': '2'}))   
    
if __name__ == '__main__':
    unittest.main()

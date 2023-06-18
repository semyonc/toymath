import copy
from collections import defaultdict
from typing import Union, Tuple, List, TypeVar, Any, Dict, Iterator

from engine.LatexWriter import LaTexWriter
from engine.LatexParser import MathParser
from engine.processor import MathProcessor
from engine.replicator import Replicator
from notation import Notation, NOTATION, Symbol, SYMBOL
from preprocessor import Preprocessor
from comparer import UnifyComparer, isVariable
from replacer import Replacer

RULE = TypeVar('RULE', bound='Rule')
TERM = TypeVar('TERM', bound='Term')
CALLBACK = TypeVar('CALLBACK', bound='Callback')

GOAL = Union[TERM, CALLBACK]
GOALS = List[GOAL]

goalid = 0


def get_operator(sym: SYMBOL, notation: NOTATION):
    f = notation.getf(sym, Notation.FUNC)
    if f is not None and f.props.get('fmt', '') == 'operatorname':
        return f
    return None


def unify(term1: TERM, subst1: Dict[str, Any], term2: TERM, subst2: Dict[str, Any]) -> bool:
    if term1.pred != term2.pred or term1.arity != term2.arity:
        return False
    comparer = UnifyComparer(term1.sym, term1.notation, subst1)
    if not comparer.unify(term2.sym, term2.notation, subst2):
        return False
    return True


def termeval(term: TERM, notation: NOTATION, env) -> bool:
    sym = term.sym
    f = term.notation.getf(sym, Notation.GROUP)
    if f is not None:
        sym = f.args[0]
    f = term.notation.getf(sym, Notation.COMP)
    if f is not None:
        output_notation = term.notation.clone()
        replacer = SymbolReplacer(term.notation, output_notation, notation, env)
        sym1 = replacer(f.args[0])
        sym2 = replacer(f.args[1])
        if f.sym.props['op'] == '=':
            comparer = UnifyComparer(sym1, output_notation, env)
            if comparer.unify(sym2, output_notation, env):
                return True
        elif f.sym.props['op'] == "\\gets" and isVariable(sym1):
            processor = MathProcessor()
            outsym, output_notation2 = processor(sym2, output_notation, {}, {})
            notation.join(output_notation2)
            env[sym1.name] = outsym
            return True

    return False


class SymbolWalker(Replicator):
    """SymbolWalker"""

    def __init__(self, notation: NOTATION):
        super().__init__(notation, Notation())
        self.variables = None

    def __call__(self, sym):
        self.variables = []
        self.enter_formula(sym)
        return self.variables

    def enter_symbol(self, sym):
        if isVariable(sym) and sym.name != "##" and sym not in self.variables:
            self.variables.append(sym.name)
        return super().enter_symbol(sym)


class SymbolReplacer(Replacer):
    """SymbolReplacer"""

    def __init__(self, notation: NOTATION, output_notation: NOTATION,
                 outer_notation: NOTATION, mapping: Dict[str, Any]):
        super().__init__(notation, output_notation)
        self.outer_notation = outer_notation
        self.mapping = mapping

    def enter_symbol(self, sym):
        if isVariable(sym) and sym.name in self.mapping:
            replacer = self.mapping[sym.name]
            if isinstance(replacer, Symbol) and self.output_notation.get(replacer) is None:
                replacer = Replicator(self.outer_notation, self.output_notation)(replacer)
            return self.subst(sym, replacer, self.context_sym())
        return super().enter_symbol(sym)


class Term(object):
    """Term"""

    def __init__(self, expr: str = None, sym: SYMBOL = None, notation: NOTATION = None):
        self.notation = Notation()
        if expr is not None:
            temp_notation = Notation()
            p = MathParser(temp_notation)
            sym = p.parse(expr)
            preprocessor = Preprocessor(temp_notation, self.notation, {}, {})
            self.expr = expr
            self.sym = preprocessor(sym)
        else:
            writer = LaTexWriter(notation)
            self.expr = writer(sym)
            replicator = Replicator(notation, self.notation)
            self.sym = replicator(sym)
        self.pred = None
        self.arity = -1
        f = get_operator(self.sym, self.notation)
        if f is not None:
            self.pred = f.args[0]
            self.arity = len(f.args) - 1
        walker = SymbolWalker(self.notation)
        self.variables = walker(self.sym)

    def __repr__(self):
        return self.expr

    def __deepcopy__(self, memodict={}):
        return self


class Rule(object):
    """Rule"""

    def __init__(self, head: TERM, goals: GOALS = None):
        self.head = head
        if goals is None:
            goals = []
        self.goals = goals

    def __repr__(self):
        if len(self.goals) == 0:
            return repr(self.head)
        return repr(self.head) + ' \\dashv ' + ','.join([repr(e) for e in self.goals])


class Callback(object):
    """Callback"""

    def __init__(self, cb):
        self.cb = cb


class Goal(object):
    """Goal"""

    def __init__(self, rule, notation, parent=None, env=None):
        global goalid
        goalid += 1
        self.id = goalid
        self.rule = rule
        self.parent = parent
        if env is None:
            self.env = defaultdict()
        else:
            self.env = copy.deepcopy(env)
        self.notation = notation.clone()
        self.inx = 0

    def repr_env(self):
        writer = LaTexWriter(self.notation)
        return "{" + ','.join([f'\'{k}\'={writer(v)}' for k, v in self.env.items()]) + "}"

    def __repr__(self):
        return "Goal %d rule=%s inx=%d env=%s" % \
            (self.id, self.rule, self.inx, self.repr_env())


class PrologModel(object):
    """PrologModel"""

    def __init__(self, rules: List[Rule] = None):
        if rules is None:
            rules = []
        self.rules = rules

    def add_rule(self, rule: Rule):
        self.rules.append(rule)

    def clear(self):
        self.rules.clear()

    def __repr__(self):
        return '\n'.join([repr(t) for t in self.rules])

    # Adoptaition of https://www.openbookproject.net/py4fun/prolog/prolog1.py
    # https://www.openbookproject.net/py4fun/prolog/prolog1.html
    def search(self, term: TERM, trace: bool = False) -> Iterator[Tuple[Dict[str, Any], NOTATION]]:
        global goalid
        goalid = 0
        if trace: print("\nsearch %s" % term.expr)
        rule = Rule(Term('\\operatorname{got}(#Goal)'), [term])  # Anything-just get a rule object
        goal = Goal(rule, term.notation.clone())  # target is the single goal
        if trace: print("stack %s" % goal)
        stack = [goal]
        while stack:
            c = stack.pop()  # Next state to consider
            if trace: print("  pop %s" % c)
            if c.inx >= len(c.rule.goals):  # Is this one finished?
                if c.parent is None:  # Yes. Our original goal?
                    if trace: print("  find solution %s" % c.repr_env())
                    yield c.env, c.notation
                    continue
                parent = copy.deepcopy(c.parent)  # Otherwise resume parent goal
                unify(c.rule.head,
                      c.env, parent.rule.goals[parent.inx], parent.env)
                parent.notation.join(c.notation)
                parent.inx = parent.inx + 1  # advance to next goal in body
                if trace: print("stack %s" % parent)
                stack.append(parent)  # let it wait its turn
                continue
            # No. more to do with this goal.
            term = c.rule.goals[c.inx]  # What we want to solve
            if term.pred is None:  # evaluate expression
                ans = termeval(term, c.notation, c.env)
                if ans:
                    if trace: print("  eval %s" % term.expr)
                    c.inx = c.inx + 1
                    stack.append(c)
            else:
                for rule in self.rules:  # Walk down the rule database
                    if rule.head.pred != term.pred or rule.head.arity != term.arity:
                        continue
                    child = Goal(rule, c.notation, parent=c)  # A possible subgoal
                    ans = unify(term, c.env, rule.head, child.env)
                    if ans:  # if unifies, stack it up
                        if trace: print("stack %s" % child)
                        child.notation.join(rule.head.notation)
                        stack.append(child)

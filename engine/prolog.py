import copy
from collections import defaultdict
from typing import Union, Tuple, List, TypeVar, Any, Dict, Iterator, Callable

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


def unify(term1: TERM, subst1: Dict[str, Any], input_notation: NOTATION,
          term2: TERM, subst2: Dict[str, Any], output_notation: NOTATION) -> bool:
    if term1.pred != term2.pred or term1.arity != term2.arity:
        return False

    notation = input_notation.concate(term1.notation)
    comparer = UnifyComparer(term1.sym, notation, copy.deepcopy(subst1))
    if not comparer.unify(term2.sym, term2.notation, subst2):
        return False

    for key in subst2:
        value = subst2[key]
        if notation.get(value) is not None and output_notation.get(value) is None:
            Replicator(notation, output_notation)(value)

    return True


def termeval(term: TERM, notation: NOTATION, env) -> bool:
    sym = term.sym
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
        self.negated = False
        self.parse_operator()
        walker = SymbolWalker(self.notation)
        self.variables = walker(self.sym)

    def parse_operator(self):
        f = self.notation.getf(self.sym, Notation.GROUP)
        if f is not None:
            self.sym = f.args[0]
            self.parse_operator()
        else:
            f = self.notation.getf(self.sym, Notation.NEG)
            if f is not None:
                self.sym = f.args[0]
                self.negated = True
                self.parse_operator()
            else:
                f = get_operator(self.sym, self.notation)
                if f is not None:
                    self.pred = f.args[0]
                    self.arity = len(f.args) - 1

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
        return repr(self.head) + ' \\dashv ' + '\\land'.join([repr(e) for e in self.goals])


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

    def __init__(self, rules: List[Rule] = None, callbacks: Dict[SYMBOL, Callable] = None):
        if rules is None:
            rules = []
        self.rules = rules
        if callbacks is None:
            callbacks = {}
        self.callbacks = callbacks
        self.add_default_callbacks()

    def add_rule(self, rule: Rule):
        self.rules.append(rule)

    def add_callback(self, sym: Union[SYMBOL, str], cb: Callable):
        if isinstance(sym, str):
            sym = Symbol(sym)
        self.callbacks[sym] = cb

    def clear(self):
        self.rules.clear()

    def __repr__(self):
        return '\n'.join([repr(t) for t in self.rules])

    def parse_rule(self, sym, notation):
        f = notation.getf(sym, Notation.P_LIST)
        if f is not None and len(f.args) == 3 and \
                get_operator(f.args[0], notation) is not None and \
                f.args[1] == Symbol('\\dashv'):
            goals = []
            f2 = notation.getf(f.args[2], Notation.GROUP)
            if f2 is not None:
                f2 = notation.getf(f2.args[0], Notation.A_LIST)
                if f2 is not None:
                    for g in f2.args:
                        goals.append(Term(sym=g, notation=notation))
            if not goals:
                goals.append(Term(sym=f.args[2], notation=notation))
            self.add_rule(Rule(Term(sym=f.args[0], notation=notation), goals=goals))
            return MathProcessor.create_true(notation)
        f = get_operator(sym, notation)
        if f is not None:
            self.add_rule(Rule(Term(sym=sym, notation=notation)))
            return MathProcessor.create_true(notation)
        return None

    def add_default_callbacks(self):
        self.add_callback('val', self.callback_val)

    @staticmethod
    def callback_val(term: Term, notation: Notation, env: Dict[str, Any]) -> bool:
        return all(name in env for name in term.variables)

    #@staticmethod
    #def callback_unary_function(term: Term, notation: Notation, env: Dict[str, Any]) -> bool:


    # Adoptaition of https://www.openbookproject.net/py4fun/prolog/prolog1.py
    # https://www.openbookproject.net/py4fun/prolog/prolog1.html
    def search(self,
               term: TERM, notation: NOTATION = None,
               trace: bool = False,
               maxiters: int = 100,
               env: Dict[str, Any] = None,
               exlusions: List[Rule] = None) -> Iterator[Tuple[Dict[str, Any], NOTATION]]:
        global goalid
        if trace: print("\nsearch %s" % term.expr)
        rule = Rule(Term('\\operatorname{got}(#Goal)'), [term])  # Anything-just get a rule object
        if notation is None:
            notation = term.notation
        goal = Goal(rule, notation, env=env)  # target is the single goal
        stack = [goal]
        if exlusions is None:
            exlusions = []
        if trace: print("stack %s" % goal)
        iters = 0
        while stack and iters < maxiters:
            iters += 1
            c = stack.pop()  # Next state to consider
            if trace: print("  pop %s" % c)
            if c.inx >= len(c.rule.goals):  # Is this one finished?
                if c.parent is None:  # Yes. Our original goal?
                    if trace: print("  find solution %s" % c.repr_env())
                    yield c.env, c.notation
                    continue
                parent = copy.deepcopy(c.parent)  # Otherwise resume parent goal
                unify(c.rule.head, c.env, c.notation,
                      parent.rule.goals[parent.inx], parent.env, parent.notation)
                parent.inx = parent.inx + 1  # advance to next goal in body
                if trace: print("stack %s" % parent)
                stack.append(parent)  # let it wait its turn
                continue
            # No. more to do with this goal.
            term = c.rule.goals[c.inx]  # What we want to solve
            if term.negated: # negation: recursive search
                term = Term(sym=term.sym, notation=term.notation)
                notation = c.notation.concate(term.notation)
                ans = not any(self.search(term, notation, trace, maxiters - iters, c.env, exlusions + [c.rule]))
                if ans:
                    if trace: print("  negation %s" % c)
                    c.inx = c.inx + 1
                    stack.append(c)
            elif term.pred is None:  # evaluate expression
                if term.sym == Notation.EXCL_MARK:  # cut
                    if trace: print("  cut %s" % c)
                    stack = [goal for goal in stack if goal.rule != c.rule]
                    exlusions = exlusions + [c.rule]
                    c.inx = c.inx + 1
                    stack.append(c)
                else:
                    ans = termeval(term, c.notation, c.env)
                    if ans:
                        if trace: print("  eval %s" % term.expr)
                        c.inx = c.inx + 1
                        stack.append(c)
            elif term.pred in self.callbacks:  # call callback
                ans = self.callbacks[term.pred](term, c.notation, c.env)
                if ans:
                    if trace: print("  callback %s" % term.expr)
                    c.inx = c.inx + 1
                    stack.append(c)
            else:
                for rule in self.rules:  # Walk down the rule database
                    if rule.head.pred != term.pred or rule.head.arity != term.arity:
                        continue
                    if rule in exlusions:
                        continue
                    child = Goal(rule, c.notation, parent=c)  # A possible subgoal
                    ans = unify(term, c.env, c.notation, rule.head, child.env, child.notation)
                    if ans:  # if unifies, stack it up
                        if trace: print("stack %s" % child)
                        stack.append(child)
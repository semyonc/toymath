import copy
from collections import defaultdict
from typing import Union, Tuple, List, TypeVar, Any, Dict, Iterator, Callable

from engine.LatexWriter import LaTexWriter
from engine.LatexParser import MathParser
from engine.helpers import trace_notation
from engine.processor import MathProcessor
from engine.replicator import Replicator
from notation import Notation, NOTATION, Symbol, SYMBOL, Func
from preprocessor import Preprocessor
from comparer import UnifyComparer, isVariable, expand_group
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


def replicate(notation: NOTATION, output_notation: NOTATION, subst2: Dict[str, Any]) -> None:
    for key in subst2:
        value = subst2[key]
        if notation.get(value) is not None and output_notation.get(value) is None:
            subst2[key] = PrologReplicator(notation, output_notation)(value)


def unify(term1: TERM, subst1: Dict[str, Any], input_notation: NOTATION,
          term2: TERM, subst2: Dict[str, Any], output_notation: NOTATION,
          backprop: bool = False) -> bool:
    notation = input_notation.concate(term1.notation)
    if term2.pred is not None:
        if term1.pred != term2.pred or term1.arity != term2.arity:
            return False
        if not backprop:
            comparer = UnifyComparer(term2.sym, term2.notation, subst2)
            if not comparer.unify(term1.sym, notation, copy.deepcopy(subst1)):
                return False
        else:
            for s1, s2 in zip(term1.args, term2.args):
                comparer = UnifyComparer(s2, term2.notation, subst2)
                comparer.unify(s1, notation, copy.deepcopy(subst1))
            replicate(term2.notation, output_notation, subst2)

    if Notation.RESULT.name in subst1:
        comparer = UnifyComparer(term1.sym, notation, copy.deepcopy(subst1))
        if term2.pred == Notation.SETQ:  # setq: assign result
            if not comparer.equal(Notation.RESULT, notation, subst1, term2.args[0],
                                  term2.notation, subst2, ctx=None):
                return False
        else:  # other: use result
            if not comparer.equal(Notation.RESULT, notation, subst1, Notation.RESULT, term2.notation, subst2, ctx=None):
                return False

    replicate(notation, output_notation, subst2)
    return True


def setvar(name, env: Dict[str, Any], sym: Any, notation: NOTATION) -> bool:
    if name not in env or \
            UnifyComparer(sym, notation).unify(env[name], notation) is not None:
        env[name] = sym
        return True
    return False


def transform(t_sym: SYMBOL, t_notation: NOTATION, parent: GOAL, targets: List[SYMBOL]) -> RULE:
    placeholders = []
    output_notation = t_notation.clone()
    for n, oper in enumerate(targets):
        subst = Symbol('#R' + str(n + 1))
        placeholders.append(subst)
        output_notation.repf(oper, Func(Notation.GROUP, (subst,), br='()'))
    goals = []
    processor = MathProcessor()
    for n, oper in enumerate(targets):
        oper, notation = processor(oper, t_notation, {}, {})
        c_list = notation.setf(Notation.C_LIST, (placeholders[n], oper))
        setq = notation.setf(Notation.FUNC, (Notation.SETQ, c_list), fmt='operatorname')
        term = Term(sym=setq, notation=notation)
        goals.append(term)
    notation = Notation()
    term = Term(sym=t_sym, notation=output_notation)
    goals.append(term)
    c_list = notation.setf(Notation.C_LIST, tuple(placeholders))
    oper = notation.setf(Notation.FUNC, (parent.rule.head.pred, c_list), fmt='operatorname')
    return Rule(Term(sym=oper, notation=notation), goals=goals)


# Run a term without operator
def run(term: TERM, env, notation: NOTATION, parent: GOAL, stack, trace) -> None:
    output_notation = term.notation.clone()
    replacer = SymbolReplacer(term.notation, output_notation, notation, env)
    f = term.notation.getf(term.sym, Notation.COMP)
    if f is not None:  # comparing
        sym1 = f.args[0]
        sym2 = f.args[1]
        if f.sym.props['op'] == '=':
            comparer = UnifyComparer(replacer(sym1), output_notation, env)
            if comparer.unify(replacer(sym2), output_notation, env):
                if trace: print("  unify %s" % term.expr)
                parent.inx = parent.inx + 1
                stack.append(parent)
    else:  # evaluating
        outsym = replacer(term.sym)
        walker = OperatorWalker(output_notation)
        targets = walker.resolve(outsym)
        if not targets:  # expression has no operators
            if trace: print("  process %s" % LaTexWriter(output_notation)(outsym))
            processor = MathProcessor()
            outsym, output_notation2 = processor(outsym, output_notation, {}, {})
            comparer = UnifyComparer(Notation.RESULT, notation, env)
            if comparer.equal(Notation.RESULT, notation, env, outsym, output_notation2, None):
                replicate(output_notation2, notation, env)
                if trace: print("  eval %s" % term.expr)
                parent.inx = parent.inx + 1
                stack.append(parent)
        else:  # expression has operators: create subgoals and reevaluate
            rule = transform(outsym, output_notation, parent, targets)
            child = Goal(rule, parent.notation, parent=parent)
            if trace: print("  stack %s" % child)
            stack.append(child)


class PrologReplicator(Replicator):
    def __init__(self, notation: NOTATION, output_notation: NOTATION):
        super().__init__(notation, output_notation)

    def mapsym(self, sym):
        return Symbol()


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
            return self.subst(sym, replacer, self.context())
        return super().enter_symbol(sym)


class OperatorWalker(Replicator):
    """OperatorWalker"""

    def __init__(self, notation: NOTATION):
        super().__init__(notation, Notation())
        self._stack = []
        self.relations = {}

    def enter_func(self, sym, f):
        clear = False
        if f.props.get('fmt', '') == 'operatorname':
            parent = self._stack[-1] if len(self._stack) > 0 else None
            if parent in self.relations:
                self.relations[parent].append(sym)
            else:
                self.relations[parent] = [sym]
            self._stack.append(sym)
            clear = True
        ret = super().enter_func(sym, f)
        if clear:
            self._stack.pop()
        return ret

    def resolve(self, sym):
        self.enter_formula(sym)
        ret = []

        def traverse(parent):
            if parent in self.relations:
                for child in self.relations[parent]:
                    if traverse(child):
                        ret.append(child)
                return False
            else:
                return True

        traverse(None)
        return ret


class Term(object):
    """Term"""

    def __init__(self, expr: str = None, sym: SYMBOL = None, notation: NOTATION = None):
        self.notation = Notation()
        if expr is not None:
            temp_notation = Notation()
            p = MathParser(temp_notation)
            sym = p.parse(expr)
            preprocessor = Preprocessor(temp_notation, self.notation, {}, {})
            self.sym = preprocessor(sym)
            writer = LaTexWriter(self.notation)
            self.expr = writer(self.sym)
        else:
            writer = LaTexWriter(notation, show_quotes=True)
            self.expr = writer(sym)
            replicator = Replicator(notation, self.notation)
            self.sym = replicator(sym)
        self.pred = None
        self.arity = -1
        self.args = None
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
                    f_args = self.notation.getf(f.args[1], Notation.C_LIST)
                    if f_args is not None:
                        self.args = f_args.args
                    else:
                        self.args = [f.args[1]]
                    self.arity = len(self.args)

    def __repr__(self):
        return self.expr

    def __deepcopy__(self, memodict={}):
        return self


class Rule(object):
    """Rule"""

    def __init__(self, head: TERM, goals: GOALS = None, **kwargs):
        self.head = head
        if goals is None:
            goals = []
        self.goals = goals
        self.props = kwargs

    def __repr__(self):
        if len(self.goals) == 0:
            return repr(self.head)
        return repr(self.head) + ' \\dashv ' + ',\\,'.join([repr(e) for e in self.goals])


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
        writer = LaTexWriter(self.notation, show_quotes=True)
        return "{" + ','.join([f'\'{k}\'={writer(v)}' for k, v in self.env.items()]) + "}"

    def __repr__(self):
        return "Goal %d rule=%s inx=%d env=%s" % \
            (self.id, self.rule, self.inx, self.repr_env())


class PrologModel(object):
    """PrologModel"""

    def __init__(self, rules: List[Rule] = None, callbacks: Dict[SYMBOL, Callable] = None):
        self.root = Term('\\operatorname{got}(#Goal)')
        self.eval = Term('\\operatorname{eval}(##)')
        self.setq = Term('\\operatorname{setq}(##,##)')
        if callbacks is None:
            callbacks = {}
        self.callbacks = callbacks
        self.rules = {}
        self.add_default_callbacks()
        if rules is not None:
            for r in rules:
                self.add_rule(r)

    def add_rule(self, rule: Rule):
        sym = rule.head.pred
        rules = self.rules.get(sym)
        if rules is None:
            rules = []
            self.rules[sym] = rules
        rules.insert(0, rule)  # for FIFO order of rules resolution

    def add_callback(self, name: str, arity: int, cb: Callable):
        self.add_rule(Rule(Term(f'\\operatorname{{{name}}}({",".join([f"#S{i}" for i in range(arity)])})')))
        self.callbacks[name] = cb

    def clear(self):
        self.rules.clear()
        self.add_default_callbacks()

    def __repr__(self):
        return '\n'.join([repr(t) for t in self.rules])

    @staticmethod
    def parse_goals(sym, notation):
        goals = []
        f2 = notation.getf(sym, Notation.GROUP)
        if f2 is not None:
            f2 = notation.getf(f2.args[0], Notation.C_LIST)
            if f2 is not None:
                for g in f2.args:
                    goals.append(Term(sym=g, notation=notation))
        if not goals:
            goals.append(Term(sym=sym, notation=notation))
        return goals

    @staticmethod
    def parse_rule(sym, notation):
        f = notation.getf(sym, Notation.P_LIST)
        if f is not None and len(f.args) == 3 and \
                get_operator(f.args[0], notation) is not None and \
                f.args[1] == Symbol('\\dashv'):
            goals = PrologModel.parse_goals(f.args[2], notation)
            return Rule(Term(sym=f.args[0], notation=notation), goals=goals)
        f = get_operator(sym, notation)
        if f is not None:
            return Rule(Term(sym=sym, notation=notation))
        return None

    def parse_and_add_rule(self, sym, notation):
        rule = self.parse_rule(sym, notation)
        if rule is not None:
            self.add_rule(rule)
            return MathProcessor.create_true(notation)
        return None

    def add_default_callbacks(self):
        self.add_callback('val', 1, self.callback_val)
        self.add_callback('br', 1, self.callback_br)
        self.add_callback('anyf', 3, self.callback_anyf)
        self.add_callback('index', 3, self.callback_index)

    @staticmethod
    def callback_val(notation: NOTATION, env: Dict[str, Any]) -> Iterator[Tuple[Dict[str, Any], NOTATION]]:
        if '#S0' in env:
            yield env, notation

    @staticmethod
    def callback_anyf(notation: NOTATION, env: Dict[str, Any]) -> Iterator[Tuple[Dict[str, Any], NOTATION]]:
        if '#S0' in env:
            notation = notation.clone()
            env = copy.deepcopy(env)
            f = notation.getf(env['#S0'], Notation.FUNC)
            if f is not None:
                if setvar('#S1', env, f.args[0], notation) and setvar('#S2', env, f.args[1], notation):
                    yield env, notation
        elif '#S1' in env and '#S2' in env:
            env['#S0'] = notation.setf(Notation.FUNC, (env['#S1'], env['#S2']))
            yield env, notation

    @staticmethod
    def callback_index(notation: NOTATION, env: Dict[str, Any]) -> Iterator[Tuple[Dict[str, Any], NOTATION]]:
        if '#S0' in env:
            notation = notation.clone()
            env = copy.deepcopy(env)
            f = notation.getf(env['#S0'], Notation.INDEX)
            if f is not None:
                if setvar('#S1', env, f.args[0], notation):
                    sym = notation.setf(Notation.INDEX, (Symbol('a'), f.args[1]))
                    if setvar('#S2', env, sym, notation):
                        yield env, notation
        elif '#S1' in env and '#S2' in env:
            f = notation.getf(env['#S2'], Notation.INDEX)
            if f is not None:
                env['#S0'] = notation.setf(Notation.INDEX, (env['#S1'], f.args[1]))
                yield env, notation

    @staticmethod
    def callback_br(notation: NOTATION, env: Dict[str, Any]) -> Iterator[Tuple[Dict[str, Any], NOTATION]]:
        if '#S0' in env:
            sym = env['#S0']
            while True:
                f = notation.getf(sym, Notation.QUOTE)
                if f is not None:
                    sym = f.args[0]
                    continue
                break
            f = notation.getf(sym, Notation.GROUP)
            if f is not None and setvar('#RESULT', env, f.args[0], notation):
                yield env, notation

    # Adoptaition of https://www.openbookproject.net/py4fun/prolog/prolog1.py
    # https://www.openbookproject.net/py4fun/prolog/prolog1.html
    def search(self,
               goals: List[Term], notation: NOTATION = None,
               trace: bool = False,
               maxiters: int = 1000,
               env: Dict[str, Any] = None,
               exlusions: List[Rule] = None) -> Iterator[Tuple[Dict[str, Any], NOTATION]]:
        global goalid
        if trace: print("\nsearch %s" % [repr(g) for g in goals])
        rule = Rule(self.root, goals)  # Anything-just get a rule object
        if notation is None:
            notation = Notation()
        goal = Goal(rule, notation, env=env)  # target is the single goal
        stack = [goal]
        if exlusions is None:
            exlusions = []
        if trace: print("stack %s" % goal)
        iters = 0
        while stack:
            if iters == maxiters:
                print("Max iterations reached")
                break
            iters += 1
            c = stack.pop()  # Next state to consider
            if trace: print("  pop %s" % c)
            if c.inx >= len(c.rule.goals):  # Is this one finished?
                if c.parent is None:  # Yes. Our original goal?
                    if trace: print("  find solution %s" % c.repr_env())
                    yield c.env, c.notation
                    continue
                elif c.rule.head.pred.name in self.callbacks:
                    for env, notation in self.callbacks[c.rule.head.pred.name](c.notation, c.env):
                        parent = copy.deepcopy(c.parent)  # Generate callback goals
                        unify(c.rule.head, env, notation, parent.rule.goals[parent.inx],
                              parent.env, parent.notation, backprop=True)
                        parent.inx = parent.inx + 1
                        if trace: print("stack %s" % parent)
                        stack.append(parent)  # let it wait its turn
                else:
                    parent = copy.deepcopy(c.parent)  # Otherwise resume parent goal
                    unify(c.rule.head, c.env, c.notation, parent.rule.goals[parent.inx],
                          parent.env, parent.notation, backprop=True)
                    parent.inx = parent.inx + 1  # advance to next goal in body
                    if trace: print("stack %s" % parent)
                    stack.append(parent)  # let it wait its turn
                continue
            # No more to do with this goal.
            term = c.rule.goals[c.inx]  # What we want to solve
            if term.negated:  # negation: recursive search
                output_notation = Notation()
                replacer = SymbolReplacer(term.notation, output_notation, c.notation, c.env)
                term = Term(sym=replacer(term.sym), notation=output_notation)
                ans = not any(self.search([term],
                                          output_notation, trace, maxiters - iters, c.env, exlusions + [c.rule]))
                if ans:
                    if trace: print("  negation %s" % c)
                    c.inx = c.inx + 1
                    stack.append(c)
            elif term.pred is None:  # no operator
                if term.sym == Notation.EXCL_MARK:  # cut
                    if trace: print("  cut %s" % c)
                    stack = [goal for goal in stack if goal.rule.head.pred != c.rule.head.pred or
                             goal.rule.head.arity != c.rule.head.arity]
                    # exlusions = exlusions + [c.rule]
                    c.inx = c.inx + 1
                    stack.append(c)
                else:  # evaluate expression
                    run(term, c.env, c.notation, c, stack, trace)
            elif term.pred == Notation.SETQ:  # assignment
                rule = Rule(self.setq, [Term(sym=term.args[1], notation=term.notation)])
                child = Goal(rule, c.notation, parent=c, env=c.env)
                if trace: print("stack %s (assignment)" % child)
                stack.append(child)
            else:
                rules = self.rules.get(term.pred)
                if rules is not None:
                    for rule in rules:  # Walk down the rule database
                        if rule.head.arity != term.arity:
                            continue
                        if rule in exlusions:
                            continue
                        child = Goal(rule, c.notation, parent=c)  # A possible subgoal
                        ans = unify(term, c.env, c.notation, rule.head, child.env, child.notation)
                        if ans:  # if unifies, stack it up
                            if trace: print("stack %s" % child)
                            stack.append(child)

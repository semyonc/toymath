# -*- coding: utf-8 -*-
"""
expr_commands.py - inline composition of verified do!-style commands.

An `expr: true` notebook command may appear INSIDE an expression, e.g.
`{diff! {int! x^3}}` or `{diff! x^2} + {int! x}`. `ExprResolver` walks the
parsed tree and replaces every command node with the agent's verified
result, inner-to-outer (nested commands resolve first, because resolving a
command's argument re-enters this same override).

The bridge stays honest: each command result is an oracle-checked do!
sub-derivation, and the arithmetic *glue* between results is verified
SEPARATELY by the `expand` primitive (see `MathShell.exec_composite`) - the
numeric oracle proves the composition, not another LLM call. Only `expr`
commands compose; a legacy/procedural or unknown `name!` inside a composite
cell is refused, so everything in the cell is either agent-verified or
expand-verified.

A command that MINTS symbols (frontmatter `fresh:`, e.g. int!'s integration
constant C) is effectful, not a pure function: every splice freshens the
declared symbols on collision (C -> C_{1}), so `{int! f} - {int! f}` yields
the honest C - C_{1} instead of silently collapsing to 0.

The zero-token tier: a command whose frontmatter carries
`direct: <primitive>` IS one verified primitive - no agent run, no tokens.
The argument goes straight to the primitive; its oracle-checked record is a
normal ledger step. Three tiers fall out: direct procedural primitive
(expand!, diff!) < LLM tactic (int! picks an integration tactic) <
whole-cell derivation (solve!).
"""
from LatexWriter import LaTexWriter
from notation import Notation, Symbol
from replicator import Replicator


class ExprCommandError(Exception):
    """A composite-cell failure: a non-expr command, the call cap, or a
    failed/empty agent sub-run. Surfaced to the cell as a do! error."""


def _direct(primitive, needs_var=False):
    """Adapter for one primitive as a direct command. `needs_var` primitives
    take (expr, var); the variable is inferred by the resolver (single plain
    free variable, minted constants excluded), never guessed."""
    def run(arg_latex, var):
        import primitives
        fn = getattr(primitives, primitive)
        if needs_var:
            return fn(arg_latex, var)
        return fn(arg_latex)
    run.needs_var = needs_var
    return run


# the whitelist of primitives that may back a `direct:` command. Deliberately
# single-argument shapes only (rewrite needs a lemma name, integrate_* need
# u/dv - those stay agent-driven or wait for two-arg command syntax).
DIRECT_PRIMITIVES = {
    'expand': _direct('expand'),
    'collect': _direct('collect', needs_var=True),
    'differentiate': _direct('differentiate', needs_var=True),
    'factor_gcd': _direct('factor_gcd'),
    'factor_quadratic': _direct('factor_quadratic', needs_var=True),
    'evaluate': _direct('evaluate'),
}


class ExprResolver(Replicator):
    """Copy a parsed expression, replacing `expr` command nodes with the
    verified result of running their do! template. Non-command nodes are
    copied verbatim by the base Replicator."""

    def __init__(self, notation, output_notation, commands, ledger, on_step,
                 run_instruction, max_calls=8):
        super(ExprResolver, self).__init__(notation, output_notation)
        self.commands = commands
        self.ledger = ledger
        self.on_step = on_step
        self.run_instruction = run_instruction
        self.max_calls = max_calls
        self.cache = {}        # (name, arg_latex) -> result_latex
        self.calls = 0
        self.subruns = []      # the raw run_instruction records, in order
        self.direct_records = []  # primitive records from direct commands
        self.used_names = set()  # every name seen in the cell so far
        self.minted_names = set()  # constants MINTED by fresh: commands

    def __call__(self, sym):
        import primitives
        # seed collision tracking with every name already in the cell, so a
        # minted constant can never capture a symbol the user wrote
        self.used_names |= primitives.free_symbols(sym, self.notation)
        return super(ExprResolver, self).__call__(sym)

    def enter_command(self, sym, f):
        name = f.sym.name[:-1]
        cmd = self.commands.get(name)
        if cmd is None or not getattr(cmd, 'expr', False):
            raise ExprCommandError(
                f'{name}! is not an inline expr command '
                '(only expr:true commands compose inside an expression)')
        var = self._param_var(cmd, f.args[0])
        args = f.args[1]
        if len(args) != 1:
            raise ExprCommandError(f'{name}! takes a single argument')
        # resolve the argument first; nested commands hit this override too,
        # so evaluation is naturally inner-to-outer
        arg_sym = self.enter_or_expr_list(args[0])
        arg_latex = LaTexWriter(self.output_notation)(arg_sym).strip()
        return self._splice(cmd, arg_sym, self._run(cmd, arg_latex, var))

    def _param_var(self, cmd, params):
        """`name! [x] <expr>` chooses the variable explicitly. Only
        var-taking direct commands accept it; the parameter must be one
        plain symbol."""
        if params is None:
            return None
        direct = getattr(cmd, 'direct', None)
        adapter = DIRECT_PRIMITIVES.get(direct) if direct else None
        if adapter is None or not adapter.needs_var:
            raise ExprCommandError(
                f'{cmd.name}! does not take a [var] parameter')
        if isinstance(params, Symbol) and self.notation.get(params) is None \
                and '_{' not in params.name:
            return params.name
        shown = LaTexWriter(self.notation)(params).strip()
        raise ExprCommandError(
            f'{cmd.name}!: the [var] parameter must be a single plain '
            f'variable name, got [{shown}]')

    def _run(self, cmd, arg_latex, var=None):
        key = (cmd.name, var, arg_latex)
        if key in self.cache:
            return self.cache[key]        # identical sub-expression: no re-call
        if getattr(cmd, 'direct', None):
            result = self._run_direct(cmd, arg_latex, var)
            self.cache[key] = result
            return result
        if self.calls >= self.max_calls:
            raise ExprCommandError(
                f'too many command evaluations in one cell (cap {self.max_calls})')
        self.calls += 1
        import prompt_commands
        instruction = prompt_commands.render(cmd, arg_latex)
        res = self.run_instruction(instruction, ledger=self.ledger,
                                   on_step=self.on_step)
        self.subruns.append(res)
        if not res.get('ok'):
            raise ExprCommandError(res.get('error', f'{cmd.name}! failed'))
        result = res.get('final_result')
        if not result:
            raise ExprCommandError(
                self._with_summary(f'{cmd.name}! produced no result', res))
        provenance = res.get('final_provenance') or {}
        if provenance.get('status') != 'verified':
            raise ExprCommandError(
                f'{cmd.name}! produced an unverified final value; inline '
                'commands must return a result established by a ledger step')
        if (provenance.get('source') != 'claim'
                and not _chains_to_goal(res.get('steps') or [],
                                        provenance.get('step'), arg_latex)):
            raise ExprCommandError(self._with_summary(
                f'{cmd.name}! did not close: its final value comes from a '
                'verified step that is not connected to the requested '
                'expression by a checked chain', res))
        self.cache[key] = result
        return result

    @staticmethod
    def _with_summary(message, res):
        summary = ' '.join((res.get('summary') or '').split())
        if not summary:
            return message
        if len(summary) > 600:
            summary = summary[:600] + '…'
        return f'{message} — agent notes: {summary}'

    def _run_direct(self, cmd, arg_latex, var=None):
        """The zero-token tier: the command IS one verified primitive. No
        agent run and no call-cap charge - the primitive's oracle-checked
        record becomes a normal, replayable ledger step, exactly like an
        agent sub-derivation's."""
        adapter = DIRECT_PRIMITIVES[cmd.direct]
        if var is None and adapter.needs_var:
            var = self._infer_var(cmd, arg_latex)
        rec = adapter(arg_latex, var)
        if not rec.get('ok'):
            raise ExprCommandError(
                f"{cmd.name}! ({cmd.direct}): {rec.get('error', 'failed')}")
        self.direct_records.append(rec)
        if self.ledger is not None:
            step = self.ledger.record(rec)
            if self.on_step is not None:
                self.on_step(step)
        return rec['result']

    def _infer_var(self, cmd, arg_latex):
        """The variable for a var-taking direct primitive: the argument's
        single plain free variable. Subscripted names are atomic constants
        the primitives take no var-string for, and names MINTED by fresh:
        commands in this cell are constants by construction (so
        {diff! {int! x^3}} infers x, not C). A remaining tie breaks only
        against constants RECORDED as minted by integration steps in the
        shared session ledger ([[n]]-chained antiderivatives carry their
        C across cells); otherwise ambiguity is refused, never guessed -
        `name! [x] <expr>` chooses explicitly."""
        import primitives
        try:
            sym, notation = primitives.parse_latex(arg_latex)
        except primitives.PrimitiveError as e:
            raise ExprCommandError(f'{cmd.name}! ({cmd.direct}): {e}')
        names = sorted(n for n in primitives.free_symbols(sym, notation)
                       if '_{' not in n and n not in self.minted_names)
        if len(names) == 1:
            return names[0]
        if not names:
            raise ExprCommandError(
                f'{cmd.name}! ({cmd.direct}): no variable to operate on '
                f'in {arg_latex}')
        session_constants = set()
        if self.ledger is not None:
            session_constants = {s.get('constant') for s in self.ledger.steps
                                 if s.get('constant')}
        remaining = [n for n in names if n not in session_constants]
        if len(remaining) == 1:
            return remaining[0]
        raise ExprCommandError(
            f'{cmd.name}! ({cmd.direct}): ambiguous variable in {arg_latex} '
            f"(candidates: {', '.join(names)}) - a direct command infers "
            f'only a single free variable; write {cmd.name}! [x] ... to '
            'choose one explicitly')

    def _splice(self, cmd, arg_sym, result_latex):
        import primitives
        rsym, rnot = primitives.parse_latex(result_latex)
        mapping = self._freshen(cmd, arg_sym, rsym, rnot)
        if mapping:
            copied = primitives.Substitutor(rnot, self.output_notation,
                                            mapping)(rsym)
        else:
            copied = Replicator(rnot, self.output_notation)(rsym)
        # always parenthesize a spliced result: a sum result in a product or
        # subtraction position would otherwise rebind (2 * (x^2/2 + C) is not
        # 2*x^2/2 + C). expand drops the redundant parens afterwards.
        return self.output_notation.setf(Notation.GROUP, (copied,), br='()')

    def _freshen(self, cmd, arg_sym, rsym, rnot):
        """A command that MINTS symbols (frontmatter `fresh:`, e.g. the
        integration constant C) is not a pure function: two splices must not
        share one constant, or {int! f} - {int! f} silently collapses to 0.
        Rename a declared fresh symbol on collision with any name already
        used in the cell (C -> C_{1}, C_{2}, ...). A fresh name that also
        occurs in the command's own ARGUMENT is bound to the argument, not
        minted - it keeps its name. The memo cache stays per-run; freshening
        is per-splice. Returns a Substitutor mapping."""
        import primitives
        rnames = primitives.free_symbols(rsym, rnot)
        argnames = (primitives.free_symbols(arg_sym, self.output_notation)
                    if arg_sym is not None else set())
        mapping = {}
        renamed = set()
        for name in getattr(cmd, 'fresh', ()) or ():
            if name not in rnames or name in argnames:
                continue
            if name not in self.used_names:
                self.used_names.add(name)  # first mint keeps the plain name
                self.minted_names.add(name)
                continue
            k = 1
            while ('%s_{%d}' % (name, k) in self.used_names
                   or '%s_{%d}' % (name, k) in rnames):
                k += 1
            fresh = '%s_{%d}' % (name, k)
            vs, vn = primitives.parse_latex(fresh)
            mapping[Symbol(name)] = (vs, vn)
            renamed.add(name)
            self.used_names.add(fresh)
            self.minted_names.add(fresh)
        self.used_names |= (rnames - renamed)
        return mapping


def _chains_to_goal(steps, final_id, goal_latex):
    """True when the run's final step is connected to the goal expression
    by an unbroken input==result chain of this run's recorded steps.

    Guards the last-transform fallback in `run_instruction`: a sub-run that
    verified steps about *pieces* of the goal (or something else entirely)
    must not have its last intermediate spliced in as the command's value.
    Linkage is structural identity (no oracle): agents pass recorded
    strings verbatim, and value-equality would be too permissive here."""
    import primitives
    transforming = [s for s in steps if s.get('result') is not None]
    if not transforming:
        return False
    by_id = {s['id']: s for s in transforming}
    cur = by_id.get(final_id, transforming[-1])
    seen = set()
    while cur is not None and cur['id'] not in seen:
        seen.add(cur['id'])
        cur_input = cur.get('input') or ''
        if primitives.covers_goal(cur_input, goal_latex):
            return True
        prev = None
        for s in transforming:
            if s['id'] == cur['id']:
                break
            if primitives.same_expression(s['result'], cur_input):
                prev = s
        cur = prev
    return False


def command_names(sym, notation):
    """Every command name (without the trailing !) that appears anywhere in
    the parsed tree. Used to route a cell to the composite evaluator."""
    names = set()
    seen = set()

    def walk(s):
        if not hasattr(s, 'name') or s.name in seen:
            return
        seen.add(s.name)
        f = notation.get(s)
        if f is None:
            return
        if f.sym.name.endswith('!'):
            names.add(f.sym.name[:-1])
        for a in _flatten(f.args):
            walk(a)

    walk(sym)
    return names


def _flatten(args):
    for a in args:
        if isinstance(a, (list, tuple)):
            yield from _flatten(a)
        elif a is not None:
            yield a

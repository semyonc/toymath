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
"""
from LatexWriter import LaTexWriter
from notation import Notation, Symbol
from replicator import Replicator


class ExprCommandError(Exception):
    """A composite-cell failure: a non-expr command, the call cap, or a
    failed/empty agent sub-run. Surfaced to the cell as a do! error."""


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
        self.used_names = set()  # every name seen in the cell so far

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
        args = f.args[1]
        if len(args) != 1:
            raise ExprCommandError(f'{name}! takes a single argument')
        # resolve the argument first; nested commands hit this override too,
        # so evaluation is naturally inner-to-outer
        arg_sym = self.enter_or_expr_list(args[0])
        arg_latex = LaTexWriter(self.output_notation)(arg_sym).strip()
        return self._splice(cmd, arg_sym, self._run(cmd, arg_latex))

    def _run(self, cmd, arg_latex):
        key = (cmd.name, arg_latex)
        if key in self.cache:
            return self.cache[key]        # identical sub-expression: no re-call
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
            raise ExprCommandError(f'{cmd.name}! produced no result')
        self.cache[key] = result
        return result

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
        self.used_names |= (rnames - renamed)
        return mapping


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

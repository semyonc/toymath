#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Registry and invocation boundary for agent-scoped math tactics.

The primitive implementation, CLI, do! agent, and ledger replay all consume
this allowlist.  Markdown skills explain *when* to choose a tactic; this
module defines *how* it is invoked and replayed.
"""
from dataclasses import dataclass
from typing import Any, Callable

import primitives


_MISSING = object()


@dataclass(frozen=True)
class Argument:
    """One ordered tactic argument shared by CLI and agent interfaces."""

    name: str
    metavar: str
    help: str
    nargs: str | None = None
    choices: tuple[str, ...] = ()
    default: Any = _MISSING
    option: str | None = None

    @property
    def required(self):
        return self.default is _MISSING and self.nargs not in ('*', '?')


@dataclass(frozen=True)
class TacticSpec:
    """Static metadata for one allowlisted tactic."""

    name: str
    op: str
    skill: str
    summary: str
    function: Callable[..., dict]
    arguments: tuple[Argument, ...] = ()
    transforming: bool = True
    agent_arguments: tuple[Argument, ...] | None = None
    agent_handler: Callable[[Any, dict], dict] | None = None
    cli_arguments: tuple[Argument, ...] | None = None
    cli_handler: Callable[[Any, dict], dict] | None = None
    provenance_validator: Callable[[dict, dict], str | None] | None = None

    @property
    def agent_args(self):
        return self.arguments if self.agent_arguments is None \
            else self.agent_arguments

    @property
    def cli_args(self):
        return self.arguments if self.cli_arguments is None \
            else self.cli_arguments


def _arg(name, metavar=None, help='', **kwargs):
    return Argument(name=name, metavar=metavar or name.upper(), help=help,
                    **kwargs)


def _error(op, message):
    return {'ok': False, 'op': op, 'error': message}


def _steps(context):
    """Snapshot a DoSession or CLI context without coupling to either."""
    if context is None or getattr(context, 'ledger', None) is None:
        return None
    lock = getattr(context, '_lock', None)
    if lock is None:
        return list(context.ledger.steps)
    with lock:
        return list(context.ledger.steps)


def _integrate_assemble_from_steps(context, args):
    steps = _steps(context)
    if steps is None:
        return _error('integrate_assemble',
                      'integrate_assemble requires --session')
    by_id = {step['id']: step for step in steps}
    linearity_id = args['linearity_step']
    linearity = by_id.get(linearity_id)
    if linearity is None:
        return _error('integrate_assemble',
                      f'unknown linearity step {linearity_id!r}')
    if linearity.get('op') != 'integrate_linearity':
        return _error(
            'integrate_assemble',
            f'{linearity_id!r} is {linearity.get("op")!r}, not '
            'integrate_linearity')
    source_ids = args['antiderivative_steps']
    values = []
    for source_id in source_ids:
        source = by_id.get(source_id)
        if source is None or source.get('result') is None:
            return _error('integrate_assemble',
                          f'unknown transforming step {source_id!r}')
        values.append(source['result'])
    result = primitives.integrate_assemble(
        linearity['input'], linearity['args']['var'], values)
    if result.get('ok'):
        result['sources'] = {
            'linearity': linearity_id,
            'antiderivatives': list(source_ids),
        }
    return result


def _limit_assemble_from_steps(context, args):
    steps = _steps(context)
    if steps is None:
        return _error('limit_assemble', 'limit_assemble requires a session')
    by_id = {step['id']: step for step in steps}
    linearity_id = args['linearity_step']
    linearity = by_id.get(linearity_id)
    if linearity is None or linearity.get('op') != 'limit_linearity':
        return _error(
            'limit_assemble',
            f'{linearity_id!r} is not a recorded limit_linearity step')
    source_ids = args['value_steps']
    values = []
    for source_id in source_ids:
        source = by_id.get(source_id)
        if source is None or source.get('result') is None:
            return _error('limit_assemble',
                          f'unknown transforming step {source_id!r}')
        values.append(source['result'])
    result = primitives.limit_assemble(linearity['input'], values)
    if result.get('ok'):
        result['sources'] = {
            'linearity': linearity_id,
            'values': list(source_ids),
        }
    return result


def _limit_squeeze_from_steps(context, args):
    steps = _steps(context)
    if steps is None:
        return _error('limit_squeeze', 'limit_squeeze requires a session')
    by_id = {step['id']: step for step in steps}
    resolved = []
    for tag in ('lower', 'upper'):
        bound = args[tag]
        source_id = args[f'{tag}_step']
        source = by_id.get(source_id)
        if source is None or source.get('result') is None:
            return _error('limit_squeeze',
                          f'unknown transforming step {source_id!r}')
        try:
            expected = primitives.limit_with_body(args['expr'], bound)
        except primitives.PrimitiveError as exc:
            return _error('limit_squeeze', str(exc))
        if not primitives.same_expression(source.get('input') or '',
                                          expected):
            return _error(
                'limit_squeeze',
                f'{source_id!r} does not record the {tag} bound limit '
                f'{expected!r}')
        resolved.append(source['result'])
    low_value, up_value = resolved
    if not primitives.same_expression(low_value, up_value):
        equal = primitives.equal_exprs(low_value, up_value)
        if not (equal.get('ok') and equal.get('verdict') == 'yes'):
            return _error(
                'limit_squeeze',
                'the recorded bound limits are not the same value: '
                f'{low_value!r} vs {up_value!r}')
    result = primitives.limit_squeeze(
        args['expr'], args['lower'], args['upper'], low_value)
    if result.get('ok'):
        result['sources'] = {
            'lower': args['lower_step'],
            'upper': args['upper_step'],
        }
    return result


def _validate_integrate_assemble(step, seen):
    sources = step.get('sources') or {}
    linearity = seen.get(sources.get('linearity'))
    if linearity is None or linearity.get('op') != 'integrate_linearity':
        return 'missing linearity-step provenance'
    args = step.get('args', {})
    if (linearity.get('input') != args.get('expr')
            or linearity.get('args', {}).get('var') != args.get('var')):
        return 'linearity-step provenance mismatch'
    source_ids = sources.get('antiderivatives') or []
    values = args.get('antiderivatives') or []
    if len(source_ids) != len(values):
        return 'antiderivative provenance mismatch'
    for source_id, value in zip(source_ids, values):
        source = seen.get(source_id)
        if source is None or source.get('result') != value:
            return f'antiderivative provenance mismatch at {source_id}'
    return None


def _validate_limit_assemble(step, seen):
    sources = step.get('sources') or {}
    linearity = seen.get(sources.get('linearity'))
    if linearity is None or linearity.get('op') != 'limit_linearity':
        return 'missing limit-linearity provenance'
    args = step.get('args', {})
    if linearity.get('input') != args.get('expr'):
        return 'limit-linearity provenance mismatch'
    source_ids = sources.get('values') or []
    values = args.get('values') or []
    if len(source_ids) != len(values):
        return 'limit-value provenance mismatch'
    for source_id, value in zip(source_ids, values):
        source = seen.get(source_id)
        if source is None or source.get('result') != value:
            return f'limit-value provenance mismatch at {source_id}'
    return None


def _validate_limit_squeeze(step, seen):
    sources = step.get('sources') or {}
    args = step.get('args', {})
    for tag in ('lower', 'upper'):
        source = seen.get(sources.get(tag))
        if source is None or source.get('result') is None:
            return f'missing {tag}-bound provenance'
        try:
            expected = primitives.limit_with_body(
                args.get('expr', ''), args.get(tag, ''))
        except primitives.PrimitiveError:
            return f'malformed {tag}-bound limit'
        if not primitives.same_expression(source.get('input') or '',
                                          expected):
            return f'{tag}-bound provenance mismatch'
        if not primitives.same_expression(source['result'],
                                          args.get('value', '')):
            return f'{tag}-bound limit value mismatch'
    return None


E = _arg('expr', 'EXPR', 'LaTeX expression or relation')
V = _arg('var', 'VAR', 'variable name')


TACTICS = (
    TacticSpec('substitute', 'substitute', 'core',
               'replace a free variable with a LaTeX value',
               primitives.substitute,
               (E, V, _arg('value', 'VALUE', 'replacement value'))),
    TacticSpec('apply', 'apply_both_sides', 'core',
               'apply +, -, *, /, or ^ to both sides of a relation',
               primitives.apply_both_sides,
               (_arg('equation', 'EQUATION', 'LaTeX relation'),
                _arg('op', 'OP', 'operation',
                     choices=('+', '-', '*', '/', '^')),
                _arg('arg', 'ARG', 'operand'))),
    TacticSpec('expand', 'expand', 'core',
               'canonicalize rational algebra and combine opaque atoms',
               primitives.expand, (E,)),
    TacticSpec('collect', 'collect', 'core',
               'group an expression by powers of a variable',
               primitives.collect, (E, V)),
    TacticSpec('evaluate', 'evaluate', 'core',
               'evaluate closed arithmetic or a closed relation',
               primitives.evaluate, (E,)),
    TacticSpec('rewrite', 'rewrite', 'core',
               'apply a registered equality lemma', primitives.rewrite,
               (E, _arg('lemma', 'LEMMA', 'registered lemma name'),
                _arg('direction', 'DIRECTION', 'rewrite direction',
                     choices=('forward', 'backward'), default='forward',
                     option='--direction'))),
    TacticSpec('factor_gcd', 'factor_gcd', 'core',
               'pull out a common factor', primitives.factor_gcd, (E,)),
    TacticSpec('factor_quadratic', 'factor_quadratic', 'core',
               'factor a quadratic with rational roots',
               primitives.factor_quadratic, (E, V)),
    TacticSpec('equal', 'equal', 'core',
               'check whether two expressions are equal',
               primitives.equal_exprs,
               (_arg('expr1', 'EXPR1', 'first expression'),
                _arg('expr2', 'EXPR2', 'second expression')),
               transforming=False),
    TacticSpec('lemmas', 'lemmas', 'core',
               'list registered rewrite lemmas', primitives.list_lemmas,
               transforming=False),

    TacticSpec('diff', 'differentiate', 'differentiation',
               'differentiate with respect to a variable',
               primitives.differentiate, (E, V)),

    TacticSpec('integrate_power_rule', 'integrate_power_rule',
               'integration', 'apply the termwise power rule',
               primitives.integrate_power_rule, (E, V)),
    TacticSpec('integrate_table', 'integrate_table', 'integration',
               'apply a basic antiderivative table rule',
               primitives.integrate_table, (E, V)),
    TacticSpec('integrate_by_parts', 'integrate_by_parts', 'integration',
               'apply one checked integration-by-parts split',
               primitives.integrate_by_parts,
               (E, V, _arg('u', 'U', 'chosen u'),
                _arg('dv', 'DV', 'chosen dv'))),
    TacticSpec('integrate_substitute', 'integrate_substitute',
               'integration', 'apply a checked u-substitution',
               primitives.integrate_substitute,
               (E, V, _arg('u_expr', 'U_EXPR', 'u as an expression'),
                _arg('u_var', 'U_VAR', 'new variable name'),
                _arg('new_integrand', 'NEW_INTEGRAND',
                     'integrand expressed in the new variable'))),
    TacticSpec('integrate_rewrite', 'integrate_rewrite', 'integration',
               'replace an integrand by a mechanically equal proposal',
               primitives.integrate_rewrite,
               (E, V, _arg('new_integrand', 'NEW_INTEGRAND',
                           'proposed equal integrand'))),
    TacticSpec('integrate_linearity', 'integrate_linearity', 'integration',
               'split an integral over a top-level sum',
               primitives.integrate_linearity, (E, V)),
    TacticSpec(
        'integrate_assemble', 'integrate_assemble', 'integration',
        'assemble recorded antiderivative pieces with provenance',
        primitives.integrate_assemble,
        (E, V, _arg('antiderivatives', 'ANTIDERIVATIVES',
                    'recorded antiderivative values', nargs='+')),
        agent_arguments=(
            _arg('linearity_step', 'LINEARITY_STEP',
                 'integrate_linearity ledger step id'),
            _arg('antiderivative_steps', 'STEP',
                 'ordered antiderivative step ids', nargs='+')),
        agent_handler=_integrate_assemble_from_steps,
        cli_arguments=(
            _arg('linearity_step', 'LINEARITY_STEP',
                 'integrate_linearity ledger step id'),
            _arg('antiderivative_steps', 'STEP',
                 'ordered antiderivative step ids', nargs='+')),
        cli_handler=_integrate_assemble_from_steps,
        provenance_validator=_validate_integrate_assemble),

    TacticSpec('limit_rewrite', 'limit_rewrite', 'limits',
               'replace a limit body by a mechanically equal proposal',
               primitives.limit_rewrite,
               (E, _arg('new_body', 'NEW_BODY', 'proposed equal body'))),
    TacticSpec('limit_substitute', 'limit_substitute', 'limits',
               'evaluate a finite limit by continuity substitution',
               primitives.limit_substitute, (E,)),
    TacticSpec('limit_linearity', 'limit_linearity', 'limits',
               'split a limit over a top-level sum',
               primitives.limit_linearity, (E,)),
    TacticSpec('limit_table', 'limit_table', 'limits',
               'apply a named standard limit rule', primitives.limit_table,
               (E,)),
    TacticSpec('limit_lhopital', 'limit_lhopital', 'limits',
               "apply one checked l'Hopital step",
               primitives.limit_lhopital, (E,)),
    TacticSpec(
        'limit_assemble', 'limit_assemble', 'limits',
        'assemble recorded limit pieces with provenance',
        primitives.limit_assemble,
        (E, _arg('values', 'VALUE', 'ordered piece values', nargs='+')),
        agent_arguments=(
            _arg('linearity_step', 'LINEARITY_STEP',
                 'limit_linearity ledger step id'),
            _arg('value_steps', 'STEP', 'ordered value step ids', nargs='+')),
        agent_handler=_limit_assemble_from_steps,
        provenance_validator=_validate_limit_assemble),
    TacticSpec(
        'limit_squeeze', 'limit_squeeze', 'limits',
        'close a limit between two recorded bounds with one value',
        primitives.limit_squeeze,
        (E, _arg('lower', 'LOWER', 'lower-bound body'),
         _arg('upper', 'UPPER', 'upper-bound body'),
         _arg('value', 'VALUE', 'common bound-limit value')),
        agent_arguments=(
            E, _arg('lower', 'LOWER', 'lower-bound body'),
            _arg('upper', 'UPPER', 'upper-bound body'),
            _arg('lower_step', 'LOWER_STEP', 'lower-bound limit step id'),
            _arg('upper_step', 'UPPER_STEP', 'upper-bound limit step id')),
        agent_handler=_limit_squeeze_from_steps,
        provenance_validator=_validate_limit_squeeze),

    TacticSpec('sum_from_ellipsis', 'sum_from_ellipsis',
               'finite_operators',
               'interpret an ellipsis sum as an explicit finite sum',
               primitives.sum_from_ellipsis,
               (E, _arg('sum_form', 'SUM_FORM', 'proposed finite sum'))),
    TacticSpec('prod_from_ellipsis', 'prod_from_ellipsis',
               'finite_operators',
               'interpret an ellipsis product as an explicit finite product',
               primitives.prod_from_ellipsis,
               (E, _arg('prod_form', 'PROD_FORM',
                        'proposed finite product'))),
    TacticSpec('sum_rewrite', 'sum_rewrite', 'finite_operators',
               'replace a summand by a mechanically equal proposal',
               primitives.sum_rewrite,
               (E, _arg('new_summand', 'NEW_SUMMAND',
                        'proposed equal summand'))),
    TacticSpec('sum_telescope', 'sum_telescope', 'finite_operators',
               'collapse a checked telescoping finite sum',
               primitives.sum_telescope,
               (E, _arg('term', 'TERM', 'proposed telescoping f(k)'))),
)


BY_NAME = {spec.name: spec for spec in TACTICS}
BY_OP = {spec.op: spec for spec in TACTICS}
if len(BY_NAME) != len(TACTICS) or len(BY_OP) != len(TACTICS):
    raise RuntimeError('duplicate tactic name or ledger operation')

TRANSFORMING_OPS = frozenset(
    spec.op for spec in TACTICS if spec.transforming)


def tactics(skill=None):
    """Return registry entries, optionally restricted to one skill."""
    return tuple(spec for spec in TACTICS
                 if skill is None or spec.skill == skill)


def describe(name):
    spec = BY_NAME.get(name)
    if spec is None:
        raise ValueError(f'unknown tactic {name!r}')
    return spec


def _parse_ordered(spec, argv, surface):
    argspecs = spec.agent_args if surface == 'agent' else spec.cli_args
    argv = list(argv)
    values = {}
    offset = 0
    for index, arg in enumerate(argspecs):
        if arg.nargs in ('+', '*'):
            if index != len(argspecs) - 1:
                raise RuntimeError(f'variadic argument {arg.name} is not last')
            rest = argv[offset:]
            if arg.nargs == '+' and not rest:
                raise ValueError(f'missing {arg.metavar}')
            values[arg.name] = rest
            offset = len(argv)
            break
        if offset >= len(argv):
            if arg.default is not _MISSING:
                values[arg.name] = arg.default
                continue
            if arg.nargs in ('?', '*'):
                values[arg.name] = None if arg.nargs == '?' else []
                continue
            raise ValueError(f'missing {arg.metavar}')
        value = argv[offset]
        offset += 1
        if not isinstance(value, str):
            raise ValueError(f'{arg.metavar} must be a string')
        if arg.choices and value not in arg.choices:
            raise ValueError(
                f'{arg.metavar} must be one of {", ".join(arg.choices)}')
        values[arg.name] = value
    if offset != len(argv):
        raise ValueError(f'too many arguments (expected {usage(spec, surface)})')
    return values


def _call(spec, values):
    ordered = [values[arg.name] for arg in spec.arguments]
    return spec.function(*ordered)


def invoke_agent(name, argv, context, require_loaded=True):
    """Invoke one allowlisted tactic from the stable do! dispatcher."""
    spec = BY_NAME.get(name)
    if spec is None:
        return _error('run_tactic', f'unknown tactic {name!r}; load a skill '
                      'or inspect the skill catalog')
    if require_loaded and spec.skill not in context.loaded_skills:
        return _error(
            'run_tactic',
            f'tactic {name!r} belongs to unloaded skill {spec.skill!r}; '
            f'call load_skill({spec.skill!r}) first')
    if not isinstance(argv, list):
        return _error(name, 'arguments must be an ordered list of strings')
    try:
        values = _parse_ordered(spec, argv, 'agent')
    except ValueError as exc:
        return _error(name, f'{exc}; usage: {usage(spec, "agent")}')
    result = (spec.agent_handler(context, values)
              if spec.agent_handler is not None else _call(spec, values))
    return context.record(result)


def invoke_cli(name, values, context=None):
    """Invoke one tactic from parsed CLI values."""
    spec = describe(name)
    result = (spec.cli_handler(context, values)
              if spec.cli_handler is not None else _call(spec, values))
    return result


def replay(op, args):
    """Replay a recorded operation through the same allowlisted registry."""
    spec = BY_OP.get(op)
    if spec is None:
        return _error(op, f'unknown op {op}')
    try:
        values = {}
        for arg in spec.arguments:
            if arg.name in args:
                values[arg.name] = args[arg.name]
            elif arg.default is not _MISSING:
                values[arg.name] = arg.default
            else:
                raise KeyError(arg.name)
        return _call(spec, values)
    except (KeyError, TypeError) as exc:
        return _error(op, f'malformed recorded arguments: {exc}')


def validate_provenance(step, seen):
    """Return a replay failure reason for bad source provenance, if any."""
    spec = BY_OP.get(step.get('op'))
    if spec is None or spec.provenance_validator is None:
        return None
    return spec.provenance_validator(step, seen)


def usage(spec, surface='cli'):
    argspecs = spec.agent_args if surface == 'agent' else spec.cli_args
    parts = [spec.name]
    for arg in argspecs:
        label = arg.metavar
        if arg.nargs == '+':
            label += '...'
        if arg.default is not _MISSING or arg.nargs in ('?', '*'):
            label = f'[{label}]'
        parts.append(label)
    return ' '.join(parts)


def public_record(spec):
    """JSON-friendly discovery record used by CLI help commands."""
    return {
        'name': spec.name,
        'op': spec.op,
        'skill': spec.skill,
        'summary': spec.summary,
        'usage': usage(spec),
        'agent_usage': usage(spec, 'agent'),
        'transforming': spec.transforming,
    }


def add_cli_parser(subparsers, common, spec):
    """Generate one backward-compatible argparse subcommand."""
    parser = subparsers.add_parser(
        spec.name, parents=[common], help=spec.summary,
        description=spec.summary)
    for arg in spec.cli_args:
        kwargs = {'help': arg.help}
        if arg.metavar:
            kwargs['metavar'] = arg.metavar
        if arg.nargs is not None:
            kwargs['nargs'] = arg.nargs
        if arg.choices:
            kwargs['choices'] = arg.choices
        if arg.default is not _MISSING:
            kwargs['default'] = arg.default
        if arg.option:
            parser.add_argument(arg.option, dest=arg.name, **kwargs)
        else:
            parser.add_argument(arg.name, **kwargs)
    return parser


def parsed_cli_values(namespace, spec):
    return {arg.name: getattr(namespace, arg.name) for arg in spec.cli_args}


def legacy_agent_arguments(spec, args):
    """Flatten old direct make_api calls for internal compatibility tests."""
    out = []
    for index, value in enumerate(args):
        if index >= len(spec.agent_args):
            # Old callers occasionally flattened a final variadic list into
            # positional values. Preserve that shape for internal adapters;
            # the model-visible dispatcher always passes one explicit list.
            if spec.agent_args and spec.agent_args[-1].nargs in ('+', '*'):
                out.append(value)
                continue
            raise TypeError(f'too many arguments for {spec.name}')
        arg = spec.agent_args[index]
        if arg.nargs in ('+', '*') and isinstance(value, (list, tuple)):
            out.extend(value)
        else:
            out.append(value)
    return out

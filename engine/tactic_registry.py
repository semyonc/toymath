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
from tactics import core
from tactics import differentiation
from tactics import equations
from tactics import finite_operators
from tactics import integration
from tactics import limits
from tactics import matrices


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
    result = integration.integrate_assemble(
        linearity['input'], linearity['args']['var'], values)
    if result.get('ok'):
        result['sources'] = {
            'linearity': linearity_id,
            'antiderivatives': list(source_ids),
        }
    return result


def _same_spelling(a, b):
    """Strict spelling identity for provenance: identical modulo bracket
    respelling only.  Deliberately NOT `covers_goal` — its body hops
    would let a step about the bare antiderivative stand in for the
    recorded LIMIT of it, which is the move being certified."""
    if primitives.same_expression(a, b):
        return True
    try:
        return (primitives._all_bracket_normal_form(a)
                == primitives._all_bracket_normal_form(b))
    except Exception:
        return False


def _integrate_definite_from_steps(context, args):
    steps = _steps(context)
    if steps is None:
        return _error('integrate_definite',
                      'integrate_definite requires a session')
    by_id = {step['id']: step for step in steps}
    source_id = args['antiderivative_step']
    source = by_id.get(source_id)
    if source is None or source.get('result') is None:
        return _error('integrate_definite',
                      f'unknown transforming step {source_id!r}')
    antiderivative = source['result']
    parts = primitives.definite_integral_parts(args['expr'], args['var'])
    limit_values = {}
    sources = {'antiderivative': source_id}
    for tag, direction in (('upper', 'left'), ('lower', 'right')):
        step_id = args.get(f'{tag}_limit_step')
        if step_id is None:
            continue
        if parts is None:
            return _error('integrate_definite',
                          'endpoint limits need a definite integral')
        limit_step = by_id.get(step_id)
        if limit_step is None or limit_step.get('result') is None:
            return _error('integrate_definite',
                          f'unknown transforming step {step_id!r}')
        bound = parts[3] if tag == 'upper' else parts[2]
        expected = primitives._limit_latex(args['var'], bound, direction,
                                           antiderivative)
        if not _same_spelling(limit_step.get('input') or '', expected):
            return _error(
                'integrate_definite',
                f'{step_id!r} does not record the one-sided limit '
                f'{expected!r} of the cited antiderivative')
        limit_values[f'{tag}_limit'] = limit_step['result']
        sources[f'{tag}_limit'] = step_id
    result = integration.integrate_definite(
        args['expr'], args['var'], antiderivative,
        upper_limit=limit_values.get('upper_limit'),
        lower_limit=limit_values.get('lower_limit'))
    if result.get('ok'):
        result['sources'] = sources
    return result


def _integrate_improper_from_steps(context, args):
    steps = _steps(context)
    if steps is None:
        return _error('integrate_improper',
                      'integrate_improper requires a session')
    by_id = {step['id']: step for step in steps}
    truncated_id = args['truncated_step']
    truncated = by_id.get(truncated_id)
    if truncated is None or truncated.get('result') is None:
        return _error('integrate_improper',
                      f'unknown transforming step {truncated_id!r}')
    if truncated.get('op') != 'integrate_definite':
        return _error(
            'integrate_improper',
            f'{truncated_id!r} is {truncated.get("op")!r}, not the '
            'integrate_definite evaluation of the truncated integral')
    info = integration._improper_parts(args['expr'], args['var'],
                                       truncated.get('input') or '')
    if isinstance(info, str):
        return _error('integrate_improper', info)
    limit_id = args['limit_step']
    limit_step = by_id.get(limit_id)
    if limit_step is None or limit_step.get('result') is None:
        return _error('integrate_improper',
                      f'unknown transforming step {limit_id!r}')
    expected = primitives._limit_latex(info['bound_var'], info['bound'],
                                       info['direction'],
                                       truncated['result'])
    if not _same_spelling(limit_step.get('input') or '', expected):
        return _error(
            'integrate_improper',
            f'{limit_id!r} does not record the one-sided limit '
            f'{expected!r} of the cited truncated evaluation')
    result = integration.integrate_improper(
        args['expr'], args['var'], truncated.get('input') or '',
        truncated['result'], limit_step['result'])
    if result.get('ok'):
        result['sources'] = {'truncated': truncated_id, 'limit': limit_id}
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
    result = limits.limit_assemble(linearity['input'], values)
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
            expected = limits.limit_with_body(args['expr'], bound)
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
        equal = core.equal_exprs(low_value, up_value)
        if not (equal.get('ok') and equal.get('verdict') == 'yes'):
            return _error(
                'limit_squeeze',
                'the recorded bound limits are not the same value: '
                f'{low_value!r} vs {up_value!r}')
    result = limits.limit_squeeze(
        args['expr'], args['lower'], args['upper'], low_value)
    if result.get('ok'):
        result['sources'] = {
            'lower': args['lower_step'],
            'upper': args['upper_step'],
        }
    return result


def _limit_from_sides_from_steps(context, args):
    steps = _steps(context)
    if steps is None:
        return _error('limit_from_sides',
                      'limit_from_sides requires a session')
    by_id = {step['id']: step for step in steps}
    resolved = {}
    for tag in ('left', 'right'):
        source_id = args[f'{tag}_step']
        source = by_id.get(source_id)
        if source is None or source.get('result') is None:
            return _error('limit_from_sides',
                          f'unknown transforming step {source_id!r}')
        try:
            expected = limits.limit_with_direction(args['expr'], tag)
        except primitives.PrimitiveError as exc:
            return _error('limit_from_sides', str(exc))
        if not primitives.same_expression(source.get('input') or '',
                                          expected):
            return _error(
                'limit_from_sides',
                f'{source_id!r} does not record the {tag} one-sided '
                f'limit {expected!r}')
        resolved[tag] = source['result']
    if not primitives.same_expression(resolved['left'], resolved['right']):
        equal = core.equal_exprs(resolved['left'], resolved['right'])
        if not (equal.get('ok') and equal.get('verdict') == 'yes'):
            return _error(
                'limit_from_sides',
                'the recorded one-sided limits are not the same value: '
                f'{resolved["left"]!r} vs {resolved["right"]!r}')
    result = limits.limit_from_sides(args['expr'], resolved['left'])
    if result.get('ok'):
        result['sources'] = {'left': args['left_step'],
                             'right': args['right_step']}
    return result


def _points_assemble_from_steps(context, args):
    steps = _steps(context)
    if steps is None:
        return _error('points_assemble',
                      'points_assemble requires a session')
    by_id = {step['id']: step for step in steps}
    roots_id = args['roots_step']
    roots = by_id.get(roots_id)
    if roots is None or roots.get('result') is None:
        return _error('points_assemble',
                      f'unknown transforming step {roots_id!r}')
    source_ids = args['value_steps']
    values = []
    for source_id in source_ids:
        source = by_id.get(source_id)
        if source is None or source.get('result') is None:
            return _error('points_assemble',
                          f'unknown transforming step {source_id!r}')
        values.append(source['result'])
    result = equations.points_assemble(
        args['expr'], args['var'], roots['result'], values)
    if result.get('ok'):
        result['sources'] = {
            'roots': roots_id,
            'values': list(source_ids),
        }
    return result


def _system_assemble_from_steps(context, args):
    steps = _steps(context)
    if steps is None:
        return _error('system_assemble',
                      'system_assemble requires a session')
    by_id = {step['id']: step for step in steps}
    source_ids = args['value_steps']
    values = []
    for source_id in source_ids:
        source = by_id.get(source_id)
        if source is None or source.get('result') is None:
            return _error('system_assemble',
                          f'unknown transforming step {source_id!r}')
        values.append(source['result'])
    result = equations.system_assemble(args['target'], values)
    if result.get('ok'):
        result['sources'] = {'assignments': list(source_ids)}
    return result


def _case_chain_hypotheses(steps, endpoint_id, target):
    """(hypotheses, error) walking the recorded chain backward from a case
    endpoint step to the stated target, collecting `assuming` constraints.

    The hypothesis of a case is usually recorded on an earlier step of its
    chain (the `apply --assuming` move), not on the endpoint the agent
    names — so the case is the whole chain segment rooted at the target,
    and the walk reuses the ledger's own chaining comparator so linkage
    can never disagree with topology or replay."""
    from ledger import _chain_links
    by_id = {step['id']: step for step in steps}
    order = {step['id']: position for position, step in enumerate(steps)}
    current = by_id.get(endpoint_id)
    if current is None or current.get('result') is None:
        return None, f'unknown transforming step {endpoint_id!r}'
    constraints = []
    visited = set()
    while True:
        if current['id'] in visited:
            return None, f'circular chain at {current["id"]}'
        visited.add(current['id'])
        for assumption in current.get('assumptions') or []:
            constraint = assumption.get('constraint')
            if constraint:
                constraints.append(constraint)
        cur_input = current.get('input') or ''
        if (primitives.same_expression(cur_input, target)
                or _chain_links(target, cur_input)):
            distinct = []
            for constraint in constraints:
                if not any(equations._same_relation(constraint, kept)
                           for kept in distinct):
                    distinct.append(constraint)
            if not distinct:
                return None, (
                    f'no case hypothesis is recorded on the chain of '
                    f'{endpoint_id}; a case is stated with assuming')
            if len(distinct) > 1:
                return None, (
                    f'the chain of {endpoint_id} mixes several distinct '
                    f'hypotheses ({", ".join(repr(c) for c in distinct)}); '
                    'assemble one case per stated hypothesis')
            return distinct[0], None
        producer = None
        for step in steps[:order[current['id']]]:
            if step.get('result') is None:
                continue
            if (step['result'] == cur_input
                    or _chain_links(step['result'], cur_input)):
                producer = step
        if producer is None:
            return None, (
                f'step {current["id"]} does not chain back to the stated '
                f'target {target!r}; every case must derive from it')
        current = producer


def _cases_assemble_from_steps(context, args):
    steps = _steps(context)
    if steps is None:
        return _error('cases_assemble',
                      'cases_assemble requires a session')
    by_id = {step['id']: step for step in steps}
    endpoints = []
    hypotheses = []
    for source_id in args['case_steps']:
        source = by_id.get(source_id)
        if source is None or source.get('result') is None:
            return _error('cases_assemble',
                          f'unknown transforming step {source_id!r}')
        hypothesis, walk_error = _case_chain_hypotheses(
            steps, source_id, args['target'])
        if walk_error:
            return _error('cases_assemble', walk_error)
        endpoints.append(source['result'])
        hypotheses.append(hypothesis)
    result = equations.cases_assemble(
        args['target'], args['union'], endpoints, hypotheses)
    if result.get('ok'):
        result['sources'] = {'cases': list(args['case_steps'])}
    return result


def _validate_cases_assemble(step, seen):
    sources = step.get('sources') or {}
    args = step.get('args', {})
    source_ids = sources.get('cases') or []
    endpoints = args.get('endpoints') or []
    hypotheses = args.get('hypotheses') or []
    if not (len(source_ids) == len(endpoints) == len(hypotheses)):
        return 'case provenance mismatch'
    steps = list(seen.values())
    for source_id, endpoint, hypothesis in zip(source_ids, endpoints,
                                               hypotheses):
        source = seen.get(source_id)
        if source is None or source.get('result') != endpoint:
            return f'case endpoint provenance mismatch at {source_id}'
        walked, walk_error = _case_chain_hypotheses(
            steps, source_id, args.get('target', ''))
        if walk_error:
            return f'case chain invalid at {source_id}: {walk_error}'
        if not equations._same_relation(walked, hypothesis):
            return f'case hypothesis provenance mismatch at {source_id}'
    return None


def _validate_limit_from_sides(step, seen):
    sources = step.get('sources') or {}
    args = step.get('args', {})
    for tag in ('left', 'right'):
        source = seen.get(sources.get(tag))
        if source is None or source.get('result') is None:
            return f'missing {tag}-limit provenance'
        try:
            expected = limits.limit_with_direction(args.get('expr', ''),
                                                   tag)
        except primitives.PrimitiveError:
            return f'malformed {tag} one-sided limit'
        if not primitives.same_expression(source.get('input') or '',
                                          expected):
            return f'{tag}-limit provenance mismatch'
        if not primitives.same_expression(source['result'],
                                          args.get('value', '')):
            return f'{tag}-limit value mismatch'
    return None


def _validate_points_assemble(step, seen):
    sources = step.get('sources') or {}
    args = step.get('args', {})
    roots = seen.get(sources.get('roots'))
    if roots is None or roots.get('result') is None:
        return 'missing root-step provenance'
    if roots.get('result') != args.get('roots'):
        return 'root-step provenance mismatch'
    source_ids = sources.get('values') or []
    values = args.get('values') or []
    if len(source_ids) != len(values):
        return 'point-value provenance mismatch'
    for source_id, value in zip(source_ids, values):
        source = seen.get(source_id)
        if source is None or source.get('result') != value:
            return f'point-value provenance mismatch at {source_id}'
    recorded = step.get('points')
    if recorded is not None:
        try:
            expected = equations.point_pairs(
                args.get('roots', ''), args.get('var', ''), values)
        except primitives.PrimitiveError:
            return 'unreadable point association'
        if recorded != expected:
            return 'point association mismatch'
    return None


def _validate_system_assemble(step, seen):
    sources = step.get('sources') or {}
    args = step.get('args', {})
    source_ids = sources.get('assignments') or []
    values = args.get('assignments') or []
    if len(source_ids) != len(values):
        return 'assignment provenance mismatch'
    for source_id, value in zip(source_ids, values):
        source = seen.get(source_id)
        if source is None or source.get('result') != value:
            return f'assignment provenance mismatch at {source_id}'
    recorded = step.get('unknowns')
    if recorded is not None:
        try:
            expected = equations.assignment_pairs(values)
        except primitives.PrimitiveError:
            return 'unreadable assignment'
        if recorded != expected:
            return 'assignment association mismatch'
    return None


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


def _validate_integrate_definite(step, seen):
    sources = step.get('sources') or {}
    source = seen.get(sources.get('antiderivative'))
    if source is None or source.get('result') is None:
        return 'missing antiderivative provenance'
    args = step.get('args', {})
    antiderivative = args.get('antiderivative')
    if source.get('result') != antiderivative:
        return 'antiderivative provenance mismatch'
    parts = primitives.definite_integral_parts(args.get('expr', ''),
                                               args.get('var', ''))
    for tag, direction in (('upper', 'left'), ('lower', 'right')):
        recorded = args.get(f'{tag}_limit')
        limit_id = sources.get(f'{tag}_limit')
        if recorded is None and limit_id is None:
            continue
        if recorded is None or limit_id is None:
            return f'{tag} endpoint-limit provenance mismatch'
        limit_step = seen.get(limit_id)
        if limit_step is None or limit_step.get('result') != recorded:
            return f'{tag} endpoint-limit provenance mismatch'
        if parts is None:
            return 'endpoint limits need a definite integral'
        bound = parts[3] if tag == 'upper' else parts[2]
        expected = primitives._limit_latex(args.get('var', ''), bound,
                                           direction, antiderivative)
        if not _same_spelling(limit_step.get('input') or '', expected):
            return f'{tag} endpoint-limit input mismatch'
    return None


def _validate_integrate_improper(step, seen):
    sources = step.get('sources') or {}
    truncated = seen.get(sources.get('truncated'))
    if truncated is None or truncated.get('op') != 'integrate_definite':
        return 'missing truncated-evaluation provenance'
    args = step.get('args', {})
    if (truncated.get('input') != args.get('truncated')
            or truncated.get('result') != args.get('truncated_value')):
        return 'truncated-evaluation provenance mismatch'
    limit_step = seen.get(sources.get('limit'))
    if (limit_step is None
            or limit_step.get('result') != args.get('limit_value')):
        return 'limit provenance mismatch'
    info = integration._improper_parts(
        args.get('expr', ''), args.get('var', ''),
        args.get('truncated', ''))
    if isinstance(info, str):
        return info
    expected = primitives._limit_latex(info['bound_var'], info['bound'],
                                       info['direction'],
                                       args.get('truncated_value', ''))
    if not _same_spelling(limit_step.get('input') or '', expected):
        return 'limit input mismatch'
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
            expected = limits.limit_with_body(
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
               core.substitute,
               (E, V, _arg('value', 'VALUE', 'replacement value'))),
    TacticSpec('apply', 'apply_both_sides', 'core',
               'apply +, -, *, /, or ^ to both sides of a relation',
               core.apply_both_sides,
               (_arg('equation', 'EQUATION', 'LaTeX relation'),
                _arg('op', 'OP', 'operation',
                     choices=('+', '-', '*', '/', '^')),
                _arg('arg', 'ARG', 'operand'),
                _arg('assuming', 'ASSUMING',
                     'case hypothesis to record, e.g. "x > 0"; a strict one '
                     'about the factor decides an inequality direction',
                     default=None, option='--assuming'))),
    TacticSpec('expand', 'expand', 'core',
               'canonicalize rational algebra and combine opaque atoms',
               core.expand, (E,)),
    TacticSpec('collect', 'collect', 'core',
               'group an expression by powers of a variable '
               'or of a function application like \\cos x',
               core.collect, (E, V)),
    TacticSpec('evaluate', 'evaluate', 'core',
               'evaluate closed arithmetic or a closed relation',
               core.evaluate, (E,)),
    TacticSpec('rewrite', 'rewrite', 'core',
               'apply a registered equality lemma', core.rewrite,
               (E, _arg('lemma', 'LEMMA', 'registered lemma name'),
                _arg('direction', 'DIRECTION', 'rewrite direction',
                     choices=('forward', 'backward'), default='forward',
                     option='--direction'),
                _arg('at', 'AT', 'target subterm LaTeX or 1-based match '
                     'index when several subterms match', default=None,
                     option='--at'))),
    TacticSpec('rewrite_as', 'rewrite_as', 'core',
               'replace an expression by a mechanically equal proposal',
               core.rewrite_as,
               (E, _arg('new_expr', 'NEW_EXPR',
                        'the proposed equal expression'))),
    TacticSpec('factor_gcd', 'factor_gcd', 'core',
               'pull out a common factor', core.factor_gcd, (E,)),
    TacticSpec('factor_quadratic', 'factor_quadratic', 'core',
               'factor a quadratic with rational roots',
               core.factor_quadratic, (E, V)),
    TacticSpec('equal', 'equal', 'core',
               'check whether two expressions are equal',
               core.equal_exprs,
               (_arg('expr1', 'EXPR1', 'first expression'),
                _arg('expr2', 'EXPR2', 'second expression'),
                _arg('assuming', 'ASSUMING',
                     'restrict the question to a stated region, '
                     'e.g. "x > 0"', default=None, option='--assuming')),
               transforming=False),
    TacticSpec('lemmas', 'lemmas', 'core',
               'list registered rewrite lemmas', core.list_lemmas,
               transforming=False),

    TacticSpec('diff', 'differentiate', 'differentiation',
               'differentiate with respect to a variable (a '
               'variable-bound definite integral closes via the FTC '
               'bound rule)',
               differentiation.differentiate, (E, V)),

    TacticSpec('quadratic_roots', 'quadratic_roots', 'equations',
               'find every rational root of a quadratic expression or '
               'equality', equations.quadratic_roots, (E, V)),
    TacticSpec('match_coefficients', 'match_coefficients', 'equations',
               'equate like powers of a variable on both sides of a '
               'polynomial identity, giving the coefficient system',
               equations.match_coefficients, (E, V)),
    TacticSpec(
        'points_assemble', 'points_assemble', 'equations',
        'assemble recorded roots and their recorded values into the '
        'complete point collection',
        equations.points_assemble,
        (E, V, _arg('roots', 'ROOTS', 'recorded solution relation'),
         _arg('values', 'VALUE', 'recorded values, one per root',
              nargs='+')),
        agent_arguments=(
            E, V, _arg('roots_step', 'ROOTS_STEP',
                       'ledger step id of the recorded solutions'),
            _arg('value_steps', 'STEP',
                 'ordered value step ids, one per root', nargs='+')),
        agent_handler=_points_assemble_from_steps,
        cli_arguments=(
            E, V, _arg('roots_step', 'ROOTS_STEP',
                       'ledger step id of the recorded solutions'),
            _arg('value_steps', 'STEP',
                 'ordered value step ids, one per root', nargs='+')),
        cli_handler=_points_assemble_from_steps,
        provenance_validator=_validate_points_assemble),
    TacticSpec(
        'system_assemble', 'system_assemble', 'equations',
        'assemble recorded per-unknown values into the checked answer for '
        'a stated equality or comma system',
        equations.system_assemble,
        (_arg('target', 'TARGET',
              'equality (or comma system) the values must satisfy — '
              'the problem, never the answer'),
         _arg('assignments', 'ASSIGNMENT',
              'recorded "unknown = value" relations', nargs='+')),
        agent_arguments=(
            _arg('target', 'TARGET',
                 'equality (or comma system) the values must satisfy — '
                 'the problem, never the answer'),
            _arg('value_steps', 'STEP',
                 'ledger step ids, one per unknown, each recording '
                 '"unknown = value"', nargs='+')),
        agent_handler=_system_assemble_from_steps,
        cli_arguments=(
            _arg('target', 'TARGET',
                 'equality (or comma system) the values must satisfy — '
                 'the problem, never the answer'),
            _arg('value_steps', 'STEP',
                 'ledger step ids, one per unknown, each recording '
                 '"unknown = value"', nargs='+')),
        cli_handler=_system_assemble_from_steps,
        provenance_validator=_validate_system_assemble),
    TacticSpec(
        'cases_assemble', 'cases_assemble', 'equations',
        'assemble recorded case endpoints under their stated hypotheses '
        'into the checked union of solutions for a stated relation',
        equations.cases_assemble,
        (_arg('target', 'TARGET',
              'relation the cases solve — the problem, never the answer'),
         _arg('union', 'UNION',
              'proposed \\lor disjunction; each disjunct restates its '
              "case's recorded endpoint or stated hypothesis, in order"),
         _arg('endpoints', 'ENDPOINT',
              'recorded case endpoint relations, one per disjunct',
              nargs='+'),
         _arg('hypotheses', 'HYPOTHESIS',
              'stated case hypotheses, one per disjunct', nargs='+')),
        agent_arguments=(
            _arg('target', 'TARGET',
                 'relation the cases solve — the problem, never the '
                 'answer'),
            _arg('union', 'UNION',
                 'proposed \\lor disjunction; each disjunct restates its '
                 "case's recorded endpoint or stated hypothesis, in order"),
            _arg('case_steps', 'STEP',
                 'ledger step ids of the case endpoints, one per '
                 'disjunct, in the order the union writes them',
                 nargs='+')),
        agent_handler=_cases_assemble_from_steps,
        cli_arguments=(
            _arg('target', 'TARGET',
                 'relation the cases solve — the problem, never the '
                 'answer'),
            _arg('union', 'UNION',
                 'proposed \\lor disjunction; each disjunct restates its '
                 "case's recorded endpoint or stated hypothesis, in order"),
            _arg('case_steps', 'STEP',
                 'ledger step ids of the case endpoints, one per '
                 'disjunct, in the order the union writes them',
                 nargs='+')),
        cli_handler=_cases_assemble_from_steps,
        provenance_validator=_validate_cases_assemble),

    TacticSpec('integrate_power_rule', 'integrate_power_rule',
               'integration', 'apply the termwise power rule',
               integration.integrate_power_rule, (E, V)),
    TacticSpec('integrate_table', 'integrate_table', 'integration',
               'apply a basic antiderivative table rule',
               integration.integrate_table, (E, V)),
    TacticSpec('integrate_by_parts', 'integrate_by_parts', 'integration',
               'apply one checked integration-by-parts split',
               integration.integrate_by_parts,
               (E, V, _arg('u', 'U', 'chosen u'),
                _arg('dv', 'DV', 'chosen dv'))),
    TacticSpec('integrate_substitute', 'integrate_substitute',
               'integration', 'apply a checked u-substitution',
               integration.integrate_substitute,
               (E, V, _arg('u_expr', 'U_EXPR', 'u as an expression'),
                _arg('u_var', 'U_VAR', 'new variable name'),
                _arg('new_integrand', 'NEW_INTEGRAND',
                     'integrand expressed in the new variable'))),
    TacticSpec('integrate_rewrite', 'integrate_rewrite', 'integration',
               'replace an integrand by a mechanically equal proposal',
               integration.integrate_rewrite,
               (E, V, _arg('new_integrand', 'NEW_INTEGRAND',
                           'proposed equal integrand'))),
    TacticSpec('integrate_linearity', 'integrate_linearity', 'integration',
               'split an integral over a top-level sum',
               integration.integrate_linearity, (E, V)),
    TacticSpec(
        'integrate_assemble', 'integrate_assemble', 'integration',
        'assemble recorded antiderivative pieces with provenance',
        integration.integrate_assemble,
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
    TacticSpec(
        'integrate_definite', 'integrate_definite', 'integration',
        'evaluate a definite integral from a recorded antiderivative '
        '(FTC), endpoint limits accepted where substitution is singular',
        integration.integrate_definite,
        (E, V, _arg('antiderivative', 'ANTIDERIVATIVE',
                    'recorded antiderivative value'),
         _arg('upper_limit', 'UPPER_LIMIT',
              'recorded one-sided limit value at the upper bound',
              default=None, option='--upper-limit'),
         _arg('lower_limit', 'LOWER_LIMIT',
              'recorded one-sided limit value at the lower bound',
              default=None, option='--lower-limit')),
        agent_arguments=(
            E, V,
            _arg('antiderivative_step', 'STEP',
                 'antiderivative ledger step id'),
            _arg('upper_limit_step', 'STEP',
                 'limit step id for a singular upper bound',
                 default=None, option='--upper-limit-step'),
            _arg('lower_limit_step', 'STEP',
                 'limit step id for a singular lower bound',
                 default=None, option='--lower-limit-step')),
        agent_handler=_integrate_definite_from_steps,
        cli_arguments=(
            E, V,
            _arg('antiderivative_step', 'STEP',
                 'antiderivative ledger step id'),
            _arg('upper_limit_step', 'STEP',
                 'limit step id for a singular upper bound',
                 default=None, option='--upper-limit-step'),
            _arg('lower_limit_step', 'STEP',
                 'limit step id for a singular lower bound',
                 default=None, option='--lower-limit-step')),
        cli_handler=_integrate_definite_from_steps,
        provenance_validator=_validate_integrate_definite),
    TacticSpec(
        'integrate_improper', 'integrate_improper', 'integration',
        'close an improper endpoint integral from a recorded truncated '
        'evaluation and its recorded one-sided limit (definitional '
        'door)',
        integration.integrate_improper,
        (E, V,
         _arg('truncated', 'TRUNCATED',
              'the truncated integral (singular bound replaced by a '
              'fresh variable)'),
         _arg('truncated_value', 'TRUNCATED_VALUE',
              'recorded evaluation of the truncated integral'),
         _arg('limit_value', 'LIMIT_VALUE',
              'recorded one-sided limit of that evaluation')),
        agent_arguments=(
            E, V,
            _arg('truncated_step', 'STEP',
                 'integrate_definite step id evaluating the truncated '
                 'integral'),
            _arg('limit_step', 'STEP',
                 'limit step id at the singular bound')),
        agent_handler=_integrate_improper_from_steps,
        cli_arguments=(
            E, V,
            _arg('truncated_step', 'STEP',
                 'integrate_definite step id evaluating the truncated '
                 'integral'),
            _arg('limit_step', 'STEP',
                 'limit step id at the singular bound')),
        cli_handler=_integrate_improper_from_steps,
        provenance_validator=_validate_integrate_improper),
    TacticSpec(
        'integrate_reduction', 'integrate_reduction', 'integration',
        'certify a proposed reduction formula: an integral family '
        'related to itself at a shifted parameter, both sides '
        'quadrature-checked at sampled parameters',
        integration.integrate_reduction,
        (_arg('relation', 'RELATION',
              'the proposed reduction equality'),
         V,
         _arg('param', 'PARAM', 'the reduction parameter'),
         _arg('shift', 'SHIFT', 'nonzero integer parameter shift'),
         _arg('assuming', 'ASSUMING',
              'parameter-domain relation the check samples inside, '
              'e.g. "n > 1"', default=None, option='--assuming'))),

    TacticSpec('limit_rewrite', 'limit_rewrite', 'limits',
               'replace a limit body by a mechanically equal proposal',
               limits.limit_rewrite,
               (E, _arg('new_body', 'NEW_BODY', 'proposed equal body'))),
    TacticSpec('limit_substitute', 'limit_substitute', 'limits',
               'evaluate a finite limit by continuity substitution',
               limits.limit_substitute, (E,)),
    TacticSpec('limit_linearity', 'limit_linearity', 'limits',
               'split a limit over a top-level sum',
               limits.limit_linearity, (E,)),
    TacticSpec('limit_table', 'limit_table', 'limits',
               'apply a named standard limit rule', limits.limit_table,
               (E,)),
    TacticSpec('limit_evaluate', 'limit_evaluate', 'limits',
               'certify an agent-proposed limit value by the approach '
               'oracle',
               limits.limit_evaluate,
               (E, _arg('value', 'VALUE', 'proposed limit value'))),
    TacticSpec('limit_lhopital', 'limit_lhopital', 'limits',
               "apply one checked l'Hopital step",
               limits.limit_lhopital, (E,)),
    TacticSpec(
        'limit_assemble', 'limit_assemble', 'limits',
        'assemble recorded limit pieces with provenance',
        limits.limit_assemble,
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
        limits.limit_squeeze,
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
    TacticSpec(
        'limit_from_sides', 'limit_from_sides', 'limits',
        'close a two-sided limit from its recorded agreeing one-sided '
        'limits',
        limits.limit_from_sides,
        (E, _arg('value', 'VALUE', 'common one-sided limit value')),
        agent_arguments=(
            E, _arg('left_step', 'LEFT_STEP',
                    'left one-sided limit step id'),
            _arg('right_step', 'RIGHT_STEP',
                 'right one-sided limit step id')),
        agent_handler=_limit_from_sides_from_steps,
        cli_arguments=(
            E, _arg('left_step', 'LEFT_STEP',
                    'left one-sided limit step id'),
            _arg('right_step', 'RIGHT_STEP',
                 'right one-sided limit step id')),
        cli_handler=_limit_from_sides_from_steps,
        provenance_validator=_validate_limit_from_sides),

    TacticSpec('sum_from_ellipsis', 'sum_from_ellipsis',
               'finite_operators',
               'interpret an ellipsis sum as an explicit finite sum',
               finite_operators.sum_from_ellipsis,
               (E, _arg('sum_form', 'SUM_FORM', 'proposed finite sum'))),
    TacticSpec('prod_from_ellipsis', 'prod_from_ellipsis',
               'finite_operators',
               'interpret an ellipsis product as an explicit finite product',
               finite_operators.prod_from_ellipsis,
               (E, _arg('prod_form', 'PROD_FORM',
                        'proposed finite product'))),
    TacticSpec('sum_rewrite', 'sum_rewrite', 'finite_operators',
               'replace a summand by a mechanically equal proposal',
               finite_operators.sum_rewrite,
               (E, _arg('new_summand', 'NEW_SUMMAND',
                        'proposed equal summand'))),
    TacticSpec('sum_telescope', 'sum_telescope', 'finite_operators',
               'collapse a checked telescoping finite sum',
               finite_operators.sum_telescope,
               (E, _arg('term', 'TERM', 'proposed telescoping f(k)'))),
    TacticSpec('series_partial_sums', 'series_partial_sums',
               'finite_operators',
               'rewrite an infinite sum/product as the limit of its '
               'partial sums/products',
               finite_operators.series_partial_sums, (E,)),
    TacticSpec('sum_closed_form', 'sum_closed_form', 'finite_operators',
               'replace a finite sum by a checked closed form in the '
               'upper bound', finite_operators.sum_closed_form,
               (E, _arg('closed_form', 'CLOSED_FORM',
                        'proposed closed form in the upper-bound '
                        'variable'))),
    TacticSpec('prod_closed_form', 'prod_closed_form', 'finite_operators',
               'replace a finite product by a checked closed form in the '
               'upper bound', finite_operators.prod_closed_form,
               (E, _arg('closed_form', 'CLOSED_FORM',
                        'proposed closed form in the upper-bound '
                        'variable'))),
    TacticSpec('series_converges', 'series_converges', 'finite_operators',
               'certify absolute convergence by comparison with a '
               'geometric or p-series bound',
               finite_operators.series_converges,
               (E, _arg('dominating', 'DOMINATING',
                        'proposed dominating summand (c r^k or c/k^p)'))),
    TacticSpec('sum_expand', 'sum_expand', 'finite_operators',
               'write a literal-bounds finite sum out term by term',
               finite_operators.sum_expand, (E,)),
    TacticSpec('prod_expand', 'prod_expand', 'finite_operators',
               'write a literal-bounds finite product out factor by '
               'factor', finite_operators.prod_expand, (E,)),
    TacticSpec('mat_add', 'mat_add', 'matrices',
               'add/subtract same-shape matrix literals cell by cell',
               matrices.mat_add, (E,)),
    TacticSpec('mat_scale', 'mat_scale', 'matrices',
               'distribute scalar factors into a matrix literal cell '
               'by cell', matrices.mat_scale, (E,)),
    TacticSpec('mat_mul', 'mat_mul', 'matrices',
               'multiply exactly two matrix literals, keeping factor '
               'order', matrices.mat_mul, (E,)),
    TacticSpec('transpose', 'transpose', 'matrices',
               'transpose a matrix literal (accepts the ^T spelling)',
               matrices.transpose, (E,)),
    TacticSpec('det_2x2', 'det_2x2', 'matrices',
               'determinant of a 2x2 matrix literal as a checked scalar',
               matrices.det_2x2, (E,)),
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
        if value is None and arg.default is not _MISSING:
            # an explicit null for an optional argument means "omitted"
            values[arg.name] = arg.default
            continue
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

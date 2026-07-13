#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
toymath_cli.py - agent-facing CLI for the verified-derivation primitives.

Every command prints one deterministic JSON object to stdout. With --session,
successful transforming steps are appended to a JSON ledger the agent can
replay and audit.

Examples:
    python toymath_cli.py apply "2x + 3 = 7" - 3 --session d.json
    python toymath_cli.py expand "(x+1)(x-2)"
    python toymath_cli.py equal "(x+1)^2" "x^2+2x+1"
    python toymath_cli.py show --session d.json
    python toymath_cli.py replay --session d.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'engine'))

import primitives  # noqa: E402
from ledger import Ledger, TRANSFORMING_OPS  # noqa: E402


def emit(obj, pretty=False):
    if pretty:
        print(json.dumps(obj, indent=1, ensure_ascii=False, default=str))
    else:
        print(json.dumps(obj, ensure_ascii=False, default=str))
    return 0 if obj.get('ok', True) else 1


def with_session(result, session_path, goal=None):
    if session_path and result.get('ok') and result['op'] in TRANSFORMING_OPS:
        ledger = Ledger(session_path)
        step = ledger.record(result, goal=goal)
        ledger.save()
        result['step'] = {'id': step['id'], 'hash': step['hash']}
    return result


def main(argv=None):
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--session', help='ledger JSON file to append to')
    common.add_argument('--goal', help='claim id this step serves (e.g. c1)')
    common.add_argument('--pretty', action='store_true',
                        help='indented JSON output')
    parser = argparse.ArgumentParser(
        prog='toymath',
        description='Agent-scoped verified-derivation primitives '
                    '(LaTeX in, LaTeX out, JSON records).')
    sub = parser.add_subparsers(dest='cmd', required=True)

    def add_parser(name, **kw):
        return sub.add_parser(name, parents=[common], **kw)

    p = add_parser('substitute', help='replace a variable by a value')
    p.add_argument('expr')
    p.add_argument('var')
    p.add_argument('value')

    p = add_parser('apply', help='apply op to both sides of an equation')
    p.add_argument('equation')
    p.add_argument('op', choices=['+', '-', '*', '/', '^'])
    p.add_argument('arg')

    p = add_parser('expand', help='distribute products and powers')
    p.add_argument('expr')

    p = add_parser('collect', help='group by powers of a variable')
    p.add_argument('expr')
    p.add_argument('var')

    p = add_parser('evaluate', help='exact arithmetic, no free variables')
    p.add_argument('expr')

    p = add_parser('diff', help='differentiate with respect to var')
    p.add_argument('expr')
    p.add_argument('var')

    p = add_parser('rewrite', help='apply a registered equality lemma')
    p.add_argument('expr')
    p.add_argument('lemma')
    p.add_argument('--direction', choices=['forward', 'backward'],
                   default='forward')

    p = add_parser('integrate_power_rule',
                   help='term-by-term power rule antiderivative')
    p.add_argument('expr')
    p.add_argument('var')

    p = add_parser('integrate_table',
                   help='basic-function antiderivatives (sin, cos, e^x, 1/x)')
    p.add_argument('expr')
    p.add_argument('var')

    p = add_parser('integrate_by_parts',
                   help='one application of integration by parts')
    p.add_argument('expr')
    p.add_argument('var')
    p.add_argument('u')
    p.add_argument('dv')

    p = add_parser('integrate_substitute',
                   help='u-substitution: verify the rewrite, transform '
                        'the integral')
    p.add_argument('expr')
    p.add_argument('var')
    p.add_argument('u_expr')
    p.add_argument('u_var')
    p.add_argument('new_integrand')

    p = add_parser('integrate_rewrite',
                   help='replace the integrand by a verified-equal '
                        'expression (e.g. partial fractions)')
    p.add_argument('expr')
    p.add_argument('var')
    p.add_argument('new_integrand')

    p = add_parser('integrate_linearity',
                   help='split the integral of a sum into a sum of '
                        'integrals (exact)')
    p.add_argument('expr')
    p.add_argument('var')

    p = add_parser('limit_rewrite',
                   help='replace a limit body by a verified-equal body')
    p.add_argument('expr')
    p.add_argument('new_body')

    p = add_parser('limit_substitute',
                   help='evaluate a finite limit by continuity substitution')
    p.add_argument('expr')

    p = add_parser('limit_linearity',
                   help='split the limit of a sum (conditional on pieces)')
    p.add_argument('expr')

    p = add_parser('limit_table',
                   help='apply a standard limit table rule')
    p.add_argument('expr')

    p = add_parser('limit_lhopital',
                   help='apply one checked conditional lHopital step')
    p.add_argument('expr')

    p = add_parser('limit_assemble',
                   help='assemble ordered values after limit_linearity')
    p.add_argument('expr')
    p.add_argument('values', nargs='+')

    p = add_parser('limit_squeeze',
                   help='close a limit between two bounds sharing a '
                        'confirmed limit value')
    p.add_argument('expr')
    p.add_argument('lower')
    p.add_argument('upper')
    p.add_argument('value')

    p = add_parser('sum_from_ellipsis',
                   help='interpret an ellipsis sum as an explicit finite sum')
    p.add_argument('expr')
    p.add_argument('sum_form')

    p = add_parser('prod_from_ellipsis',
                   help='interpret an ellipsis product as an explicit '
                        'finite product')
    p.add_argument('expr')
    p.add_argument('prod_form')

    p = add_parser('sum_rewrite',
                   help='replace a summand by a verified-equal one')
    p.add_argument('expr')
    p.add_argument('new_summand')

    p = add_parser('sum_telescope',
                   help='collapse a telescoping sum to its closed form')
    p.add_argument('expr')
    p.add_argument('term')

    p = add_parser('factor_gcd', help='pull out the common factor')
    p.add_argument('expr')

    p = add_parser('factor_quadratic',
                   help='factor a quadratic with rational roots')
    p.add_argument('expr')
    p.add_argument('var')

    p = add_parser('equal', help='check two expressions: yes/no/unknown')
    p.add_argument('expr1')
    p.add_argument('expr2')

    add_parser('lemmas', help='list registered rewrite lemmas')
    p = add_parser('claim', help='record a root claim or subclaim')
    p.add_argument('statement')
    p.add_argument('--parent', help='parent claim id for a subclaim')
    p = add_parser('conclude', help='close a claim from checked step ids')
    p.add_argument('claim_id')
    p.add_argument('step_ids', nargs='+')
    p = add_parser('show', help='render the session ledger')
    p.add_argument('--format', choices=['text', 'md'], default='text')
    add_parser('replay', help='re-verify every step in the session')

    args = parser.parse_args(argv)

    if args.cmd == 'substitute':
        res = primitives.substitute(args.expr, args.var, args.value)
    elif args.cmd == 'apply':
        res = primitives.apply_both_sides(args.equation, args.op, args.arg)
    elif args.cmd == 'expand':
        res = primitives.expand(args.expr)
    elif args.cmd == 'collect':
        res = primitives.collect(args.expr, args.var)
    elif args.cmd == 'evaluate':
        res = primitives.evaluate(args.expr)
    elif args.cmd == 'diff':
        res = primitives.differentiate(args.expr, args.var)
    elif args.cmd == 'rewrite':
        res = primitives.rewrite(args.expr, args.lemma, args.direction)
    elif args.cmd == 'integrate_power_rule':
        res = primitives.integrate_power_rule(args.expr, args.var)
    elif args.cmd == 'integrate_table':
        res = primitives.integrate_table(args.expr, args.var)
    elif args.cmd == 'integrate_by_parts':
        res = primitives.integrate_by_parts(args.expr, args.var,
                                            args.u, args.dv)
    elif args.cmd == 'integrate_substitute':
        res = primitives.integrate_substitute(args.expr, args.var,
                                              args.u_expr, args.u_var,
                                              args.new_integrand)
    elif args.cmd == 'integrate_rewrite':
        res = primitives.integrate_rewrite(args.expr, args.var,
                                           args.new_integrand)
    elif args.cmd == 'integrate_linearity':
        res = primitives.integrate_linearity(args.expr, args.var)
    elif args.cmd == 'limit_rewrite':
        res = primitives.limit_rewrite(args.expr, args.new_body)
    elif args.cmd == 'limit_substitute':
        res = primitives.limit_substitute(args.expr)
    elif args.cmd == 'limit_linearity':
        res = primitives.limit_linearity(args.expr)
    elif args.cmd == 'limit_table':
        res = primitives.limit_table(args.expr)
    elif args.cmd == 'limit_lhopital':
        res = primitives.limit_lhopital(args.expr)
    elif args.cmd == 'limit_assemble':
        res = primitives.limit_assemble(args.expr, args.values)
    elif args.cmd == 'limit_squeeze':
        res = primitives.limit_squeeze(args.expr, args.lower, args.upper,
                                       args.value)
    elif args.cmd == 'sum_from_ellipsis':
        res = primitives.sum_from_ellipsis(args.expr, args.sum_form)
    elif args.cmd == 'prod_from_ellipsis':
        res = primitives.prod_from_ellipsis(args.expr, args.prod_form)
    elif args.cmd == 'sum_rewrite':
        res = primitives.sum_rewrite(args.expr, args.new_summand)
    elif args.cmd == 'sum_telescope':
        res = primitives.sum_telescope(args.expr, args.term)
    elif args.cmd == 'factor_gcd':
        res = primitives.factor_gcd(args.expr)
    elif args.cmd == 'factor_quadratic':
        res = primitives.factor_quadratic(args.expr, args.var)
    elif args.cmd == 'equal':
        res = primitives.equal_exprs(args.expr1, args.expr2)
    elif args.cmd == 'lemmas':
        res = primitives.list_lemmas()
    elif args.cmd == 'claim':
        if not args.session:
            return emit({'ok': False, 'error': '--session required'})
        ledger = Ledger(args.session)
        try:
            claim = ledger.record_claim(args.statement, parent=args.parent)
        except ValueError as e:
            return emit({'ok': False, 'op': 'claim', 'error': str(e)},
                        args.pretty)
        ledger.save()
        return emit({'ok': True, 'op': 'claim', **claim}, args.pretty)
    elif args.cmd == 'conclude':
        if not args.session:
            return emit({'ok': False, 'error': '--session required'})
        ledger = Ledger(args.session)
        try:
            claim = ledger.conclude(args.claim_id, args.step_ids)
        except ValueError as e:
            return emit({'ok': False, 'op': 'conclude',
                         'claim': args.claim_id, 'error': str(e)},
                        args.pretty)
        ledger.save()
        return emit({'ok': True, 'op': 'conclude', 'claim': claim},
                    args.pretty)
    elif args.cmd == 'show':
        if not args.session:
            return emit({'ok': False, 'error': '--session required'})
        ledger = Ledger(args.session)
        if args.format == 'md':
            print(ledger.render_markdown())
        else:
            print(ledger.render())
        return 0
    elif args.cmd == 'replay':
        if not args.session:
            return emit({'ok': False, 'error': '--session required'})
        ledger = Ledger(args.session)
        rep = ledger.replay()
        rep['ok'] = rep['status'] == 'verified'
        return emit(rep, args.pretty)
    else:  # unreachable
        return emit({'ok': False, 'error': f'unknown command {args.cmd}'})

    try:
        res = with_session(res, args.session, goal=args.goal)
    except ValueError as e:
        return emit({'ok': False, 'op': res.get('op'), 'error': str(e)},
                    args.pretty)
    return emit(res, args.pretty)


if __name__ == '__main__':
    sys.exit(main())

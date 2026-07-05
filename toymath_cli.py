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
from ledger import Ledger  # noqa: E402

TRANSFORMING_OPS = {'substitute', 'apply_both_sides', 'expand', 'collect',
                    'evaluate', 'differentiate', 'rewrite'}


def emit(obj, pretty=False):
    if pretty:
        print(json.dumps(obj, indent=1, ensure_ascii=False, default=str))
    else:
        print(json.dumps(obj, ensure_ascii=False, default=str))
    return 0 if obj.get('ok', True) else 1


def with_session(result, session_path):
    if session_path and result.get('ok') and result['op'] in TRANSFORMING_OPS:
        ledger = Ledger(session_path)
        step = ledger.record(result)
        ledger.save()
        result['step'] = {'id': step['id'], 'hash': step['hash']}
    return result


def main(argv=None):
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--session', help='ledger JSON file to append to')
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

    p = add_parser('equal', help='check two expressions: yes/no/unknown')
    p.add_argument('expr1')
    p.add_argument('expr2')

    add_parser('lemmas', help='list registered rewrite lemmas')
    add_parser('show', help='render the session ledger')
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
    elif args.cmd == 'equal':
        res = primitives.equal_exprs(args.expr1, args.expr2)
    elif args.cmd == 'lemmas':
        res = primitives.list_lemmas()
    elif args.cmd == 'show':
        if not args.session:
            return emit({'ok': False, 'error': '--session required'})
        ledger = Ledger(args.session)
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

    res = with_session(res, args.session)
    return emit(res, args.pretty)


if __name__ == '__main__':
    sys.exit(main())

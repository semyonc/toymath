#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic JSON CLI for the verified-derivation tactic registry."""
import argparse
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'engine'))

import tactic_registry  # noqa: E402
import tactic_skills  # noqa: E402
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


def build_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--session', help='ledger JSON file to append to')
    common.add_argument('--goal', help='claim id this step serves (e.g. c1)')
    common.add_argument('--pretty', action='store_true',
                        help='indented JSON output')
    parser = argparse.ArgumentParser(
        prog='toymath',
        description='Agent-scoped verified-derivation tactics '
                    '(LaTeX in, LaTeX out, JSON records).')
    sub = parser.add_subparsers(dest='cmd', required=True)

    for spec in tactic_registry.TACTICS:
        tactic_registry.add_cli_parser(sub, common, spec)

    p = sub.add_parser('skills', parents=[common],
                       help='list progressively loadable tactic skills')
    p = sub.add_parser('tactics', parents=[common],
                       help='list registered tactics')
    p.add_argument('--skill', help='restrict to one skill')
    p = sub.add_parser('describe', parents=[common],
                       help='describe one registered tactic')
    p.add_argument('tactic')

    p = sub.add_parser('claim', parents=[common],
                       help='record a root claim or subclaim')
    p.add_argument('statement')
    p.add_argument('--parent', help='parent claim id for a subclaim')
    p = sub.add_parser('conclude', parents=[common],
                       help='close a claim from checked step ids')
    p.add_argument('claim_id')
    p.add_argument('step_ids', nargs='+')
    p = sub.add_parser('branch', parents=[common],
                       help='record an exploration resume marker')
    p.add_argument('from_step', help='earlier transforming step id')
    p.add_argument('reason', help='short reason for abandoning the path')
    p = sub.add_parser('open', parents=[common],
                       help='record a run-level open outcome '
                            '(no certified result)')
    p.add_argument('reason', help='the exact missing move; never a '
                                  'nonexistence claim')
    p = sub.add_parser('show', parents=[common],
                       help='render the session ledger')
    p.add_argument('--format', choices=['text', 'md'], default='text')
    sub.add_parser('replay', parents=[common],
                   help='re-verify every step in the session')
    return parser


def _require_session(args, op):
    if args.session:
        return None
    return {'ok': False, 'op': op, 'error': '--session required'}


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.cmd in tactic_registry.BY_NAME:
        spec = tactic_registry.BY_NAME[args.cmd]
        values = tactic_registry.parsed_cli_values(args, spec)
        context = None
        if spec.cli_handler is not None and args.session:
            context = SimpleNamespace(ledger=Ledger(args.session))
        result = tactic_registry.invoke_cli(args.cmd, values, context)
        try:
            result = with_session(result, args.session, goal=args.goal)
        except ValueError as exc:
            result = {'ok': False, 'op': result.get('op'),
                      'error': str(exc)}
        return emit(result, args.pretty)

    if args.cmd == 'skills':
        errors = tactic_skills.validate()
        return emit({'ok': not errors,
                     'skills': tactic_skills.catalog_records(),
                     **({'errors': errors} if errors else {})}, args.pretty)

    if args.cmd == 'tactics':
        if args.skill and args.skill not in tactic_skills.discover():
            return emit({'ok': False, 'op': 'tactics',
                         'error': f'unknown skill {args.skill!r}'},
                        args.pretty)
        records = [tactic_registry.public_record(spec)
                   for spec in tactic_registry.tactics(args.skill)]
        return emit({'ok': True, 'tactics': records}, args.pretty)

    if args.cmd == 'describe':
        try:
            spec = tactic_registry.describe(args.tactic)
        except ValueError as exc:
            return emit({'ok': False, 'op': 'describe', 'error': str(exc)},
                        args.pretty)
        return emit({'ok': True, 'tactic': tactic_registry.public_record(spec)},
                    args.pretty)

    if args.cmd == 'claim':
        error = _require_session(args, 'claim')
        if error:
            return emit(error, args.pretty)
        ledger = Ledger(args.session)
        try:
            claim = ledger.record_claim(args.statement, parent=args.parent)
        except ValueError as exc:
            return emit({'ok': False, 'op': 'claim', 'error': str(exc)},
                        args.pretty)
        ledger.save()
        return emit({'ok': True, 'op': 'claim', **claim}, args.pretty)

    if args.cmd == 'conclude':
        error = _require_session(args, 'conclude')
        if error:
            return emit(error, args.pretty)
        ledger = Ledger(args.session)
        try:
            claim = ledger.conclude(args.claim_id, args.step_ids)
        except ValueError as exc:
            return emit({'ok': False, 'op': 'conclude',
                         'claim': args.claim_id, 'error': str(exc)},
                        args.pretty)
        ledger.save()
        return emit({'ok': True, 'op': 'conclude', 'claim': claim},
                    args.pretty)

    if args.cmd == 'branch':
        error = _require_session(args, 'branch')
        if error:
            return emit(error, args.pretty)
        ledger = Ledger(args.session)
        try:
            marker = ledger.record_branch(
                args.from_step, args.reason, goal=args.goal)
        except ValueError as exc:
            return emit({'ok': False, 'op': 'branch', 'error': str(exc)},
                        args.pretty)
        ledger.save()
        return emit({'ok': True, 'op': 'branch', 'id': marker['id'],
                     'from': marker['args']['from'],
                     'hash': marker['hash']}, args.pretty)

    if args.cmd == 'open':
        error = _require_session(args, 'open')
        if error:
            return emit(error, args.pretty)
        ledger = Ledger(args.session)
        try:
            selection = ledger.record_open(args.reason, goal=args.goal)
        except ValueError as exc:
            return emit({'ok': False, 'op': 'open', 'error': str(exc)},
                        args.pretty)
        ledger.save()
        return emit({'ok': True, 'op': 'open', 'id': selection['id'],
                     'hash': selection['hash']}, args.pretty)

    if args.cmd == 'show':
        error = _require_session(args, 'show')
        if error:
            return emit(error, args.pretty)
        ledger = Ledger(args.session)
        print(ledger.render_markdown() if args.format == 'md'
              else ledger.render())
        return 0

    if args.cmd == 'replay':
        error = _require_session(args, 'replay')
        if error:
            return emit(error, args.pretty)
        report = Ledger(args.session).replay()
        report['ok'] = report['status'] == 'verified'
        return emit(report, args.pretty)

    return emit({'ok': False, 'error': f'unknown command {args.cmd}'})


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ledger.py - the step ledger: a machine-checked derivation record.

A derivation is trusted if every step is a call to a trusted primitive with
valid arguments plus an independent numeric spot-check. The ledger is the
artifact: auditable, reproducible (replayable), honestly conditional through
its accumulated assumptions.

Persistence is a plain JSON file so an agent can keep a session across turns.
"""
import json
import os
import hashlib

LEDGER_VERSION = 1


def _step_hash(op, input_latex, result_latex):
    h = hashlib.sha1(f'{op}|{input_latex}|{result_latex}'.encode('utf-8'))
    return h.hexdigest()[:7]


class Ledger(object):
    def __init__(self, path=None):
        self.path = path
        self.data = {'version': LEDGER_VERSION, 'steps': [],
                     'assumptions': []}
        if path and os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as fh:
                self.data = json.load(fh)
            if self.data.get('version') != LEDGER_VERSION:
                raise ValueError(
                    f'session file version {self.data.get("version")} '
                    f'not supported')

    @property
    def steps(self):
        return self.data['steps']

    @property
    def assumptions(self):
        return self.data['assumptions']

    def record(self, result):
        """Append a successful primitive result; returns the step record."""
        if not result.get('ok'):
            raise ValueError('only successful results are recorded')
        n = len(self.steps) + 1
        continues = None
        if self.steps:
            prev = self.steps[-1]['result']
            cur = result.get('input')
            if prev is not None and cur is not None:
                if prev == cur:
                    continues = True
                else:
                    import primitives
                    eq = primitives.equal_exprs(prev, cur)
                    continues = (eq.get('verdict') == 'yes'
                                 if eq.get('ok') else None)
        step = {
            'id': f's{n}',
            'continues': continues,
            'hash': _step_hash(result['op'], result.get('input', ''),
                               result.get('result', '')),
            'op': result['op'],
            'args': result.get('args', {}),
            'input': result.get('input'),
            'result': result.get('result'),
            'assumptions': result.get('assumptions', []),
            'check': result.get('check', {'status': 'skipped'}),
        }
        self.steps.append(step)
        for a in step['assumptions']:
            if a not in self.assumptions:
                self.assumptions.append(a)
        return step

    def save(self, path=None):
        path = path or self.path
        if not path:
            raise ValueError('no session path')
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(self.data, fh, indent=1, ensure_ascii=False)
        self.path = path

    def last_result(self):
        if not self.steps:
            return None
        return self.steps[-1]['result']

    def replay(self):
        """Re-run every step through its primitive and confirm the recorded
        result. Returns {'status': 'verified'|'failed', ...}."""
        import primitives
        dispatch = {
            'substitute': lambda a: primitives.substitute(
                a['expr'], a['var'], a['value']),
            'apply_both_sides': lambda a: primitives.apply_both_sides(
                a['equation'], a['op'], a['arg']),
            'expand': lambda a: primitives.expand(a['expr']),
            'collect': lambda a: primitives.collect(a['expr'], a['var']),
            'evaluate': lambda a: primitives.evaluate(a['expr']),
            'differentiate': lambda a: primitives.differentiate(
                a['expr'], a['var']),
            'rewrite': lambda a: primitives.rewrite(
                a['expr'], a['lemma'], a.get('direction', 'forward')),
            'factor_gcd': lambda a: primitives.factor_gcd(a['expr']),
            'factor_quadratic': lambda a: primitives.factor_quadratic(
                a['expr'], a['var']),
        }
        for step in self.steps:
            fn = dispatch.get(step['op'])
            if fn is None:
                return {'status': 'failed', 'step': step['id'],
                        'reason': f'unknown op {step["op"]}'}
            res = fn(step['args'])
            if not res.get('ok'):
                return {'status': 'failed', 'step': step['id'],
                        'reason': res.get('error')}
            if res.get('result') != step['result']:
                return {'status': 'failed', 'step': step['id'],
                        'reason': 'result mismatch',
                        'recorded': step['result'],
                        'replayed': res.get('result')}
            if res.get('check', {}).get('status') == 'disagree':
                return {'status': 'failed', 'step': step['id'],
                        'reason': 'numeric oracle disagrees on replay'}
        return {'status': 'verified', 'steps': len(self.steps),
                'assumptions': self.assumptions}

    def render(self):
        """Terse human/agent-readable summary of the derivation."""
        lines = []
        for step in self.steps:
            check = step['check'].get('status', '?')
            mark = {'agree': 'ok', 'skipped': '??', 'disagree': 'XX'}.get(
                check, '?')
            branch = '' if step.get('continues') in (True, None) else ' (branch)'
            lines.append(f"{step['id']}#{step['hash']} [{mark}]{branch} "
                         f"{step['op']}: {step['input']}  ==>  "
                         f"{step['result']}")
            for a in step['assumptions']:
                lines.append(f"      assumes {a['text']}")
        if self.assumptions:
            lines.append('assumptions: '
                         + '; '.join(a['text'] for a in self.assumptions))
        return '\n'.join(lines)

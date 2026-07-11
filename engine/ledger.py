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

# primitives whose successful results become ledger steps (everything that
# transforms an expression; equal/lemmas are queries, not steps)
TRANSFORMING_OPS = frozenset({
    'substitute', 'apply_both_sides', 'expand', 'collect', 'evaluate',
    'differentiate', 'rewrite', 'factor_gcd', 'factor_quadratic',
    'integrate_power_rule', 'integrate_table', 'integrate_by_parts',
    'integrate_substitute', 'integrate_rewrite', 'integrate_linearity'})


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
        prev = self.last_result()
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

    def record_comment(self, text):
        """Append a narrative note. Notes are unverified prose: they carry
        no input/result, are skipped by replay, and never count as
        provenance for a final result."""
        text = (text or '').strip()
        if not text:
            raise ValueError('empty comment')
        n = len(self.steps) + 1
        step = {
            'id': f's{n}',
            'continues': None,
            'hash': _step_hash('comment', '', text),
            'op': 'comment',
            'args': {'text': text},
            'input': None,
            'result': None,
            'assumptions': [],
            'check': {'status': 'note'},
        }
        self.steps.append(step)
        return step

    def save(self, path=None):
        path = path or self.path
        if not path:
            raise ValueError('no session path')
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(self.data, fh, indent=1, ensure_ascii=False)
        self.path = path

    def last_result(self):
        for step in reversed(self.steps):
            if step.get('result') is not None:
                return step['result']
        return None

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
            'integrate_power_rule': lambda a: primitives.integrate_power_rule(
                a['expr'], a['var']),
            'integrate_table': lambda a: primitives.integrate_table(
                a['expr'], a['var']),
            'integrate_by_parts': lambda a: primitives.integrate_by_parts(
                a['expr'], a['var'], a['u'], a['dv']),
            'integrate_substitute':
                lambda a: primitives.integrate_substitute(
                    a['expr'], a['var'], a['u_expr'], a['u_var'],
                    a['new_integrand']),
            'integrate_rewrite': lambda a: primitives.integrate_rewrite(
                a['expr'], a['var'], a['new_integrand']),
            'integrate_linearity': lambda a: primitives.integrate_linearity(
                a['expr'], a['var']),
        }
        for step in self.steps:
            if step['op'] == 'comment':
                continue
            fn = dispatch.get(step['op'])
            if fn is None:
                return {'status': 'failed', 'step': step['id'],
                        'reason': f'unknown op {step["op"]}'}
            res = fn(step['args'])
            if not res.get('ok'):
                return {'status': 'failed', 'step': step['id'],
                        'reason': res.get('error')}
            if res.get('result') != step['result']:
                # tolerate formatting drift between versions, but only if
                # the results are semantically equal
                eq = primitives.equal_exprs(res.get('result'),
                                            step['result'])
                if not (eq.get('ok') and eq.get('verdict') == 'yes'):
                    return {'status': 'failed', 'step': step['id'],
                            'reason': 'result mismatch',
                            'recorded': step['result'],
                            'replayed': res.get('result')}
            if res.get('check', {}).get('status') == 'disagree':
                return {'status': 'failed', 'step': step['id'],
                        'reason': 'numeric oracle disagrees on replay'}
        return {'status': 'verified', 'steps': len(self.steps),
                'assumptions': self.assumptions}

    _MARKS = {'agree': 'verified', 'exact': 'exact',
              'skipped': 'unchecked', 'disagree': 'FAILED',
              'domain-differs': 'DOMAIN DIFFERS'}

    def render_markdown(self):
        """Render the derivation as Markdown with LaTeX math blocks."""
        lines = ['# Verified derivation', '']
        if self.assumptions:
            lines.append('**Valid under the assumptions:** '
                         + ', '.join(f'${a["text"]}$'
                                     for a in self.assumptions))
            lines.append('')
        for step in self.steps:
            if step['op'] == 'comment':
                # notes are prose, never math: keep \ and $ literal so a
                # Markdown/MathJax renderer cannot typeset them
                text = (step['args']['text']
                        .replace('\\', '\\\\').replace('$', '\\$'))
                lines.append(f"**{step['id']}** *note* — {text}")
                lines.append('')
                continue
            check = step['check'].get('status', '?')
            mark = self._MARKS.get(check, check)
            branch = ('' if step.get('continues') in (True, None)
                      else ' *(new chain)*')
            arg_note = ''
            if step['op'] == 'apply_both_sides':
                a = step['args']
                arg_note = f" — `{a['op']} {a['arg']}` on both sides"
            elif step['op'] == 'substitute':
                a = step['args']
                arg_note = f" — ${a['var']} := {a['value']}$"
            elif step['op'] == 'integrate_by_parts':
                a = step['args']
                arg_note = f" — $u = {a['u']}$, $dv = {a['dv']}$"
            lines.append(f"**{step['id']}** `{step['op']}`{arg_note} "
                         f"— *{mark}*{branch}")
            lines.append('')
            lines.append(f"$${step['input']} \\;\\Longrightarrow\\; "
                         f"{step['result']}$$")
            for a in step['assumptions']:
                lines.append(f"- assumes ${a['text']}$")
            lines.append('')
        lines.append(f'*{len(self.steps)} steps; replay with '
                     f'`toymath replay --session <file>`.*')
        return '\n'.join(lines)

    def render(self):
        """Terse human/agent-readable summary of the derivation."""
        lines = []
        for step in self.steps:
            if step['op'] == 'comment':
                lines.append(f"{step['id']}#{step['hash']} [--] note: "
                             f"{step['args']['text']}")
                continue
            check = step['check'].get('status', '?')
            mark = {'agree': 'ok', 'exact': 'ok', 'skipped': '??',
                    'disagree': 'XX', 'domain-differs': 'D!'}.get(check, '?')
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

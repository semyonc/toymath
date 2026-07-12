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

LEDGER_VERSION = 2

# primitives whose successful results become ledger steps (everything that
# transforms an expression; equal/lemmas are queries, not steps)
TRANSFORMING_OPS = frozenset({
    'substitute', 'apply_both_sides', 'expand', 'collect', 'evaluate',
    'differentiate', 'rewrite', 'factor_gcd', 'factor_quadratic',
    'limit_rewrite', 'limit_substitute', 'limit_linearity', 'limit_table',
    'limit_lhopital', 'limit_assemble',
    'sum_from_ellipsis', 'sum_rewrite', 'sum_telescope',
    'integrate_power_rule', 'integrate_table', 'integrate_by_parts',
    'integrate_substitute', 'integrate_rewrite', 'integrate_linearity',
    'integrate_assemble'})


def _step_hash(op, input_latex, result_latex):
    h = hashlib.sha1(f'{op}|{input_latex}|{result_latex}'.encode('utf-8'))
    return h.hexdigest()[:7]


def _claim_hash(statement, parent):
    h = hashlib.sha1(f'claim|{parent or ""}|{statement}'.encode('utf-8'))
    return h.hexdigest()[:7]


def assumption_markdown(assumption):
    """Markdown/MathJax form of one assumption record. A record carrying a
    `display` field renders prose as prose with inline `$...$` math spans;
    a bare `text` (pure math, and every pre-`display` record) keeps the
    historical whole-line math wrapping."""
    display = assumption.get('display')
    if display is not None:
        return display
    return f"${assumption['text']}$"


def _presentation_free(conclusion):
    """Conclusion record with presentation-only assumption fields removed,
    so replay compares mechanical content across renderer versions."""
    out = dict(conclusion or {})
    out['assumptions'] = [
        {k: v for k, v in a.items() if k != 'display'}
        for a in (out.get('assumptions') or [])]
    return out


def _eq_yes(left, right):
    """Semantic equality used only for ledger connectivity/closure."""
    import primitives
    try:
        rec = primitives.equal_exprs(left, right)
    except Exception:
        return False
    return rec.get('ok') and rec.get('verdict') == 'yes'


def _relation_parts(statement):
    """Return (lhs, rhs, relation) for a parsed relation, else None."""
    import primitives
    from notation import Notation
    sym, notation = primitives.parse_latex(statement)
    comp = notation.getf(sym, Notation.COMP)
    if comp is None:
        return None
    return (primitives.write_latex(comp.args[0], notation),
            primitives.write_latex(comp.args[1], notation),
            comp.sym.props.get('op'))


def _relation_holds(statement):
    """Mechanically decide a relation endpoint when possible."""
    import primitives
    parts = _relation_parts(statement)
    if parts is None:
        return False
    lhs, rhs, relation = parts
    if relation == '=':
        return _eq_yes(lhs, rhs)
    rec = primitives.evaluate(statement)
    return rec.get('ok') and rec.get('holds') is True


def _source_ids(step):
    """Flatten provenance ids from an assembly-style sources mapping."""
    out = []
    for value in (step.get('sources') or {}).values():
        if isinstance(value, str):
            out.append(value)
        elif isinstance(value, list):
            out.extend(v for v in value if isinstance(v, str))
    return out


class Ledger(object):
    def __init__(self, path=None):
        self.path = path
        self.data = {'version': LEDGER_VERSION, 'steps': [],
                     'assumptions': [], 'claims': []}
        if path and os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as fh:
                self.data = json.load(fh)
            version = self.data.get('version')
            if version == 1:
                # v1 steps are valid v2 steps. Upgrade in memory; the next
                # save writes the v2 envelope without changing old records.
                self.data['version'] = LEDGER_VERSION
                self.data.setdefault('claims', [])
            elif version != LEDGER_VERSION:
                raise ValueError(
                    f'session file version {version} '
                    f'not supported')
            self.data.setdefault('claims', [])

    @property
    def steps(self):
        return self.data['steps']

    @property
    def assumptions(self):
        return self.data['assumptions']

    @property
    def claims(self):
        return self.data['claims']

    def get_claim(self, claim_id):
        return next((c for c in self.claims if c['id'] == claim_id), None)

    def record_claim(self, statement, parent=None):
        """Record a parseable root claim or subclaim, initially open."""
        import primitives
        statement = (statement or '').strip()
        if not statement:
            raise ValueError('empty claim')
        try:
            primitives.parse_latex(statement)
        except primitives.PrimitiveError as e:
            raise ValueError(str(e))
        if _relation_parts(statement) is None:
            raise ValueError('claim must be a top-level relation')
        if parent is not None and self.get_claim(parent) is None:
            raise ValueError(f'unknown parent claim {parent!r}')
        claim = {
            'id': f'c{len(self.claims) + 1}',
            'hash': _claim_hash(statement, parent),
            'statement': statement,
            'parent': parent,
            'verdict': 'open',
            'conclusion': None,
        }
        self.claims.append(claim)
        return claim

    def record(self, result, goal=None):
        """Append a successful primitive result; returns the step record."""
        if not result.get('ok'):
            raise ValueError('only successful results are recorded')
        if goal is not None and self.get_claim(goal) is None:
            raise ValueError(f'unknown goal {goal!r}')
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
        # Integration constants and assembly source ids stay in the
        # replayable artifact; later cells can distinguish session constants
        # from user variables and replay can audit provenance.
        if result.get('constant'):
            step['constant'] = result['constant']
        if result.get('terms'):
            step['terms'] = result['terms']
        if result.get('sources'):
            step['sources'] = result['sources']
        if goal is not None:
            step['goal'] = goal
        self.steps.append(step)
        for a in step['assumptions']:
            if a not in self.assumptions:
                self.assumptions.append(a)
        return step

    def record_comment(self, text, goal=None):
        """Append a narrative note. Notes are unverified prose: they carry
        no input/result, are skipped by replay, and never count as
        provenance for a final result."""
        text = (text or '').strip()
        if not text:
            raise ValueError('empty comment')
        if goal is not None and self.get_claim(goal) is None:
            raise ValueError(f'unknown goal {goal!r}')
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
        if goal is not None:
            step['goal'] = goal
        self.steps.append(step)
        return step

    def _validate_conclusion(self, claim_id, step_ids, steps=None):
        claim = self.get_claim(claim_id)
        if claim is None:
            raise ValueError(f'unknown claim {claim_id!r}')
        if not isinstance(step_ids, list) or not step_ids:
            raise ValueError('conclusion needs at least one step id')
        if len(set(step_ids)) != len(step_ids):
            raise ValueError('conclusion step ids must be unique')

        by_id = {s['id']: s for s in (steps if steps is not None
                                      else self.steps)}
        selected = []
        for step_id in step_ids:
            step = by_id.get(step_id)
            if step is None:
                raise ValueError(f'unknown conclusion step {step_id!r}')
            if step.get('result') is None:
                raise ValueError(f'{step_id} is not a transforming step')
            if step.get('goal') != claim_id:
                raise ValueError(
                    f'{step_id} belongs to goal {step.get("goal")!r}, '
                    f'not {claim_id}')
            status = step.get('check', {}).get('status')
            if status not in ('agree', 'exact'):
                raise ValueError(
                    f'{step_id} is not mechanically checked ({status})')
            selected.append(step)

        # The named records must form one chain. Assembly records may join
        # branches explicitly through their persisted source ids.
        for previous, current in zip(selected, selected[1:]):
            connected = _eq_yes(previous['result'], current.get('input'))
            if not connected and previous['id'] not in _source_ids(current):
                raise ValueError(
                    f'{current["id"]} does not continue from '
                    f'{previous["id"]} or cite it as provenance')

        first, last = selected[0], selected[-1]
        endpoint = last['result']
        closure = None
        if (_eq_yes(endpoint, claim['statement'])
                and _relation_holds(endpoint)):
            closure = 'true-relation-endpoint'
        else:
            parts = _relation_parts(claim['statement'])
            if parts[2] == '=':
                lhs, rhs, _ = parts
                if (_eq_yes(first.get('input'), lhs)
                        and _eq_yes(endpoint, rhs)):
                    closure = 'left-to-right'
                elif (_eq_yes(first.get('input'), rhs)
                        and _eq_yes(endpoint, lhs)):
                    closure = 'right-to-left'
        if closure is None:
            raise ValueError(
                f'chain endpoint {endpoint!r} does not close claim '
                f'{claim["statement"]!r}')

        assumptions = []
        for step in selected:
            for assumption in step.get('assumptions', []):
                if assumption not in assumptions:
                    assumptions.append(assumption)
        verdict = 'conditional' if assumptions else 'established'
        return {
            'steps': list(step_ids),
            'endpoint': endpoint,
            'assumptions': assumptions,
            'closure': closure,
            'verdict': verdict,
        }

    def conclude(self, claim_id, step_ids):
        """Mechanically close a claim from goal-owned, checked steps."""
        conclusion = self._validate_conclusion(claim_id, step_ids)
        claim = self.get_claim(claim_id)
        claim['verdict'] = conclusion.pop('verdict')
        claim['conclusion'] = conclusion
        return claim

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
            'limit_rewrite': lambda a: primitives.limit_rewrite(
                a['expr'], a['new_body']),
            'limit_substitute': lambda a: primitives.limit_substitute(
                a['expr']),
            'limit_linearity': lambda a: primitives.limit_linearity(
                a['expr']),
            'limit_table': lambda a: primitives.limit_table(a['expr']),
            'limit_lhopital': lambda a: primitives.limit_lhopital(a['expr']),
            'limit_assemble': lambda a: primitives.limit_assemble(
                a['expr'], a['values']),
            'sum_from_ellipsis': lambda a: primitives.sum_from_ellipsis(
                a['expr'], a['sum_form']),
            'sum_rewrite': lambda a: primitives.sum_rewrite(
                a['expr'], a['new_summand']),
            'sum_telescope': lambda a: primitives.sum_telescope(
                a['expr'], a['term']),
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
            'integrate_assemble': lambda a: primitives.integrate_assemble(
                a['expr'], a['var'], a['antiderivatives']),
        }
        seen = {}
        replayed_steps = []
        for step in self.steps:
            if step['op'] == 'comment':
                seen[step['id']] = step
                replayed_steps.append(step)
                continue
            if step['op'] == 'integrate_assemble':
                sources = step.get('sources') or {}
                linearity = seen.get(sources.get('linearity'))
                if linearity is None or linearity.get('op') != \
                        'integrate_linearity':
                    return {'status': 'failed', 'step': step['id'],
                            'reason': 'missing linearity-step provenance'}
                args = step.get('args', {})
                if (linearity.get('input') != args.get('expr')
                        or linearity.get('args', {}).get('var')
                        != args.get('var')):
                    return {'status': 'failed', 'step': step['id'],
                            'reason': 'linearity-step provenance mismatch'}
                source_ids = sources.get('antiderivatives') or []
                values = args.get('antiderivatives') or []
                if len(source_ids) != len(values):
                    return {'status': 'failed', 'step': step['id'],
                            'reason': 'antiderivative provenance mismatch'}
                for source_id, value in zip(source_ids, values):
                    source = seen.get(source_id)
                    if source is None or source.get('result') != value:
                        return {
                            'status': 'failed', 'step': step['id'],
                            'reason': ('antiderivative provenance mismatch '
                                       f'at {source_id}')}
            if step['op'] == 'limit_assemble':
                sources = step.get('sources') or {}
                linearity = seen.get(sources.get('linearity'))
                if linearity is None or linearity.get('op') != \
                        'limit_linearity':
                    return {'status': 'failed', 'step': step['id'],
                            'reason': 'missing limit-linearity provenance'}
                args = step.get('args', {})
                if linearity.get('input') != args.get('expr'):
                    return {'status': 'failed', 'step': step['id'],
                            'reason': 'limit-linearity provenance mismatch'}
                source_ids = sources.get('values') or []
                values = args.get('values') or []
                if len(source_ids) != len(values):
                    return {'status': 'failed', 'step': step['id'],
                            'reason': 'limit-value provenance mismatch'}
                for source_id, value in zip(source_ids, values):
                    source = seen.get(source_id)
                    if source is None or source.get('result') != value:
                        return {
                            'status': 'failed', 'step': step['id'],
                            'reason': ('limit-value provenance mismatch at '
                                       f'{source_id}')}
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
            seen[step['id']] = step
            replayed = dict(step)
            replayed['assumptions'] = res.get('assumptions', [])
            replayed['check'] = res.get('check', {'status': 'skipped'})
            replayed_steps.append(replayed)
        for claim in self.claims:
            try:
                import primitives
                primitives.parse_latex(claim.get('statement', ''))
                if claim.get('hash') != _claim_hash(
                        claim.get('statement', ''), claim.get('parent')):
                    raise ValueError('claim hash mismatch')
                if claim.get('verdict') == 'open':
                    if claim.get('conclusion') is not None:
                        raise ValueError('open claim carries a conclusion')
                    continue
                recorded = claim.get('conclusion') or {}
                checked = self._validate_conclusion(
                    claim['id'], recorded.get('steps'),
                    steps=replayed_steps)
                if (claim.get('verdict') != checked.pop('verdict')
                        or _presentation_free(recorded)
                        != _presentation_free(checked)):
                    raise ValueError('claim conclusion mismatch')
            except (KeyError, ValueError, primitives.PrimitiveError) as e:
                return {'status': 'failed',
                        'claim': claim.get('id', '?'),
                        'reason': f'claim replay failed: {e}'}
        return {'status': 'verified', 'steps': len(self.steps),
                'claims': len(self.claims),
                'open_claims': sum(c.get('verdict') == 'open'
                                   for c in self.claims),
                'assumptions': self.assumptions}

    _MARKS = {'agree': 'verified', 'exact': 'exact',
              'skipped': 'unchecked', 'disagree': 'FAILED',
              'domain-differs': 'DOMAIN DIFFERS'}

    def render_markdown(self):
        """Render the derivation as Markdown with LaTeX math blocks."""
        title = '# Derivation ledger' if self.claims else '# Verified derivation'
        lines = [title, '']
        for claim in self.claims:
            verdict = claim.get('verdict', 'open').upper()
            conclusion = claim.get('conclusion') or {}
            count = len(conclusion.get('steps') or [])
            assumptions = len(conclusion.get('assumptions') or [])
            detail = ''
            if verdict != 'OPEN':
                detail = f' ({count} steps, {assumptions} assumptions)'
            lines.append(
                f'**CLAIM {claim["id"]} — {verdict}{detail}:** '
                f'${claim["statement"]}$')
            if verdict == 'OPEN':
                lines.append('')
                lines.append('*No mechanically checked closing chain has '
                             'been recorded.*')
            lines.append('')
        if self.assumptions:
            lines.append('**Valid under the assumptions:** '
                         + ', '.join(assumption_markdown(a)
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
            elif step['op'] == 'integrate_assemble':
                src = step.get('sources', {})
                ids = ', '.join(src.get('antiderivatives', []))
                arg_note = (f" — sources `{src.get('linearity', '?')}` "
                            f"→ `{ids}`")
            elif step['op'] == 'limit_assemble':
                src = step.get('sources', {})
                ids = ', '.join(src.get('values', []))
                arg_note = (f" — sources `{src.get('linearity', '?')}` "
                            f"→ `{ids}`")
            goal = (f" → `{step['goal']}`" if step.get('goal') else '')
            lines.append(f"**{step['id']}**{goal} `{step['op']}`{arg_note} "
                         f"— *{mark}*{branch}")
            lines.append('')
            lines.append(f"$${step['input']} \\;\\Longrightarrow\\; "
                         f"{step['result']}$$")
            for a in step['assumptions']:
                lines.append(f"- assumes {assumption_markdown(a)}")
            lines.append('')
        lines.append(f'*{len(self.steps)} steps; replay with '
                     f'`toymath replay --session <file>`.*')
        return '\n'.join(lines)

    def render(self):
        """Terse human/agent-readable summary of the derivation."""
        lines = []
        for claim in self.claims:
            verdict = claim.get('verdict', 'open').upper()
            conclusion = claim.get('conclusion') or {}
            detail = ''
            if verdict != 'OPEN':
                detail = ('; steps ' + ','.join(conclusion.get('steps', []))
                          + f'; endpoint {conclusion.get("endpoint")}')
            lines.append(f"CLAIM {claim['id']}#{claim['hash']} [{verdict}] "
                         f"{claim['statement']}{detail}")
        for step in self.steps:
            if step['op'] == 'comment':
                lines.append(f"{step['id']}#{step['hash']} [--] note: "
                             f"{step['args']['text']}")
                continue
            check = step['check'].get('status', '?')
            mark = {'agree': 'ok', 'exact': 'ok', 'skipped': '??',
                    'disagree': 'XX', 'domain-differs': 'D!'}.get(check, '?')
            branch = '' if step.get('continues') in (True, None) else ' (branch)'
            goal = f" -> {step['goal']}" if step.get('goal') else ''
            lines.append(f"{step['id']}#{step['hash']} [{mark}]{branch}{goal} "
                         f"{step['op']}: {step['input']}  ==>  "
                         f"{step['result']}")
            if step['op'] == 'integrate_assemble':
                src = step.get('sources', {})
                lines.append(
                    '      sources: linearity '
                    + src.get('linearity', '?') + '; pieces '
                    + ', '.join(src.get('antiderivatives', [])))
            elif step['op'] == 'limit_assemble':
                src = step.get('sources', {})
                lines.append(
                    '      sources: linearity '
                    + src.get('linearity', '?') + '; values '
                    + ', '.join(src.get('values', [])))
            for a in step['assumptions']:
                lines.append(f"      assumes {a['text']}")
        if self.assumptions:
            lines.append('assumptions: '
                         + '; '.join(a['text'] for a in self.assumptions))
        return '\n'.join(lines)

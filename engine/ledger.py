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
import html as _html

from tactic_registry import TRANSFORMING_OPS
from tactics import core as core_tactics

LEDGER_VERSION = 2


def _display_latex(latex):
    """Derived rich-view spelling; persisted ledger text stays untouched."""
    import primitives
    return primitives.display_latex(latex)


def _display_math_spans(text):
    """Apply formula cleanup only inside the $...$ spans of mixed prose."""
    parts = text.split('$')
    return ''.join(f'${_display_latex(part)}$' if i % 2 else part
                   for i, part in enumerate(parts))


def _step_hash(op, input_latex, result_latex):
    h = hashlib.sha1(f'{op}|{input_latex}|{result_latex}'.encode('utf-8'))
    return h.hexdigest()[:7]


def _claim_hash(statement, parent):
    h = hashlib.sha1(f'claim|{parent or ""}|{statement}'.encode('utf-8'))
    return h.hexdigest()[:7]


def _legacy_branch_hash(from_step, reason):
    """Legacy marker hash, retained so saved sessions keep replaying."""
    return _step_hash('branch', from_step, reason)


def _branch_hash(from_step, reason):
    """Current marker hash. Its distinct namespace makes the promised
    marker->next-transform edge mandatory without invalidating legacy files."""
    return _step_hash('branch-next', from_step, reason)


def _branch_edge_hash(marker_id, from_step, to_step, reason):
    return _step_hash(
        'branch-edge', f'{marker_id}|{from_step}|{to_step}', reason)


def _selection_hash(result, provenance, goal):
    payload = json.dumps(provenance, sort_keys=True, ensure_ascii=False,
                         separators=(',', ':'))
    return _step_hash('selection', result,
                      f'{goal or ""}|{payload}')


def assumption_markdown(assumption):
    """Markdown/MathJax form of one assumption record. A record carrying a
    `display` field renders prose as prose with inline `$...$` math spans;
    a bare `text` (pure math, and every pre-`display` record) keeps the
    historical whole-line math wrapping."""
    display = assumption.get('display')
    if display is not None:
        return _display_math_spans(display)
    return f"${_display_latex(assumption['text'])}$"


def _markdown_prose(text):
    """Keep annotation prose inert in Markdown/MathJax/embedded HTML."""
    return (_html.escape(str(text), quote=False)
            .replace('\\', '\\\\').replace('$', '\\$'))


def _presentation_free(conclusion):
    """Conclusion record with presentation-only assumption fields removed,
    so replay compares mechanical content across renderer versions."""
    out = dict(conclusion or {})
    out['assumptions'] = [
        {k: v for k, v in a.items() if k != 'display'}
        for a in (out.get('assumptions') or [])]
    return out


def _eq_yes(left, right):
    """Semantic equality used only for ledger connectivity/closure.
    Structural identity (which tolerates ellipsis spellings) is decided
    first; equal_exprs does math and therefore refuses ellipsis input."""
    import primitives
    try:
        if primitives.same_expression(left, right):
            return True
    except Exception:
        pass
    try:
        rec = core_tactics.equal_exprs(left, right)
    except Exception:
        return False
    return rec.get('ok') and rec.get('verdict') == 'yes'


def _same_expression(left, right):
    """Shape-level linkage for exploration edges (formatting tolerant).

    An edge says which recorded expression the agent resumed from; numeric or
    algebraic value equality is too permissive for that topological claim.
    """
    import primitives
    try:
        return primitives.same_expression(left, right)
    except Exception:
        return False


def _chain_links(prev_result, cur_input):
    """Structural step-chaining linkage (wrapper/body/respelling tolerant).

    A step continues an earlier result when its input restates it modulo
    grouping, when either expression is the big-operator wrapper whose
    body is the other — the integrand/body convention the primitives
    already accept in goal gating: `integrate_table` legitimately consumes
    the integrand of the previous `\\int` result — or when the two are
    bracket respellings of one structure (live: agents retype the
    assemble result without its decorative per-piece `\\left(...\\right)`
    wrappers). The all-bracket normal form re-encodes child boundaries,
    so `(a+b)c` never links `a+bc`, and semantic brackets like `|...|`
    are preserved. Structural only; value equality stays too permissive
    for a topological claim.
    """
    import primitives
    try:
        if (primitives.covers_goal(cur_input, prev_result)
                or primitives.covers_goal(prev_result, cur_input)):
            return True
        return (primitives._all_bracket_normal_form(prev_result)
                == primitives._all_bracket_normal_form(cur_input))
    except Exception:
        return False


def _relation_parts(statement):
    """Return (lhs, rhs, relation) for a parsed relation, else None.
    Parses with allow_ellipsis: this only splits a statement into sides
    (comparison class, no math); an ellipsis claim is pinned down by the
    *_from_ellipsis step whose recorded assumption interprets it."""
    import primitives
    from notation import Notation
    sym, notation = primitives.parse_latex(statement, allow_ellipsis=True)
    comp = notation.getf(sym, Notation.COMP)
    if comp is None:
        return None
    return (primitives.write_latex(comp.args[0], notation),
            primitives.write_latex(comp.args[1], notation),
            comp.sym.props.get('op'))


def _relation_holds(statement):
    """Mechanically decide a relation endpoint when possible."""
    parts = _relation_parts(statement)
    if parts is None:
        return False
    lhs, rhs, relation = parts
    if relation == '=':
        return _eq_yes(lhs, rhs)
    rec = core_tactics.evaluate(statement)
    return rec.get('ok') and rec.get('holds') is True


def _all_bracket_normal_form_equal(left, right):
    """Structural sameness used only to dedupe presentation lists."""
    import primitives
    try:
        return (primitives._all_bracket_normal_form(left)
                == primitives._all_bracket_normal_form(right))
    except Exception:
        return False


def _recorded_parts(step):
    """Values a step recorded as named parts of its own result.

    Linearity-style steps split one object into pieces and persist them as
    `terms`; a later step working on a piece continued from recorded work
    even though no whole result equals its input.
    """
    parts = []
    for term in step.get('terms') or []:
        if isinstance(term, str):
            parts.append(term)
        elif isinstance(term, dict):
            parts.extend(value for key, value in term.items()
                         if key != 'sign' and isinstance(value, str))
    return parts


def _is_derived(step, earlier):
    """True when a step's input came from recorded work, not from typing.

    Everything else is a PREMISE: an input this ledger never produced. That
    is not an error — a derivation has to start somewhere, and a stated
    given is legitimate — but it is the boundary of what the session
    checked, so presentation must be able to name it.
    """
    current = step.get('input')
    if current is None:
        return True
    if step.get('continues') is True:
        return True
    for previous in earlier:
        result = previous.get('result')
        if result is not None and (result == current
                                   or _chain_links(result, current)):
            return True
        for part in _recorded_parts(previous):
            if part == current or _chain_links(part, current):
                return True
    return False


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
                     'assumptions': [], 'claims': [], 'selections': []}
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
            self.data.setdefault('selections', [])

    @property
    def steps(self):
        return self.data['steps']

    @property
    def assumptions(self):
        return self.data['assumptions']

    @property
    def claims(self):
        return self.data['claims']

    @property
    def selections(self):
        return self.data['selections']

    @staticmethod
    def _pending_branches_in(steps, goal):
        """Return unresolved markers for ``goal`` in ledger order.

        This is derived solely from ledger order: the first later
        transforming step in the same goal resolves a marker.  No hidden
        mutable branch cursor participates in recording or replay.
        """
        pending = []
        for step in reversed(steps):
            if step.get('goal') != goal:
                continue
            if (step.get('op') in TRANSFORMING_OPS
                    and step.get('result') is not None):
                break
            if step.get('op') == 'branch':
                pending.append(step)
        return list(reversed(pending))

    @classmethod
    def _pending_branch_in(cls, steps, goal):
        pending = cls._pending_branches_in(steps, goal)
        return pending[-1] if pending else None

    def _pending_branch(self, goal):
        return self._pending_branch_in(self.steps, goal)

    def _branch_edge(self, marker, target):
        args = marker.get('args') or {}
        edge = {
            'marker': marker.get('id'),
            'from': args.get('from'),
            'reason': args.get('reason'),
            'hash': _branch_edge_hash(
                marker.get('id'), args.get('from'), target.get('id'),
                args.get('reason')),
        }
        # Result-anchored resumes stay byte-identical with older sessions.
        # Only the abandon-the-source-itself form (the target restarts from
        # the source step's recorded INPUT) marks its anchor; the result
        # anchor wins when both match, so replay re-derives deterministically.
        source = next((s for s in self.steps
                       if s.get('id') == args.get('from')), None)
        if (source is not None
                and not _chain_links(source.get('result'),
                                     target.get('input'))
                and _chain_links(source.get('input'),
                                 target.get('input'))):
            edge['anchor'] = 'input'
        return edge

    def get_claim(self, claim_id):
        return next((c for c in self.claims if c['id'] == claim_id), None)

    def record_claim(self, statement, parent=None):
        """Record a parseable root claim or subclaim, initially open.

        Ellipsis statements are accepted: recording only validates the
        relation shape, and a claim containing an ellipsis can only ever
        close through a *_from_ellipsis step whose recorded assumption
        pins down the reading."""
        import primitives
        statement = (statement or '').strip()
        if not statement:
            raise ValueError('empty claim')
        try:
            primitives.parse_latex(statement, allow_ellipsis=True)
        except primitives.PrimitiveError as e:
            raise ValueError(str(e))
        if _relation_parts(statement) is None:
            raise ValueError('claim must be a top-level relation')
        if parent is not None and self.get_claim(parent) is None:
            raise ValueError(f'unknown parent claim {parent!r}')
        # Repeating a formatting variant of the same claim should focus
        # the existing goal, not mint another notebook-global claim id.
        # Concluded claims are deliberately reused too: the statement is
        # the same proposition, so a repeated conclude may strengthen its
        # closing chain — a duplicate id would strand every later step
        # under a goal that can never close the original claim.
        for claim in self.claims:
            if claim.get('parent') == parent:
                try:
                    same = (claim.get('statement') == statement
                            or primitives.same_expression(
                                claim.get('statement', ''), statement))
                except (primitives.PrimitiveError, ValueError):
                    same = False
                if same:
                    return claim
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
        pending_branch = self._pending_branch(goal)
        if pending_branch is not None:
            source_id = pending_branch['args']['from']
            source = next((s for s in self.steps
                           if s.get('id') == source_id), None)
            if (source is None
                    or not (_chain_links(source.get('result'),
                                         result.get('input'))
                            or _chain_links(source.get('input'),
                                            result.get('input')))):
                raise ValueError(
                    f'{result.get("op", "step")} input does not resume '
                    f'branch marker {pending_branch["id"]} from '
                    f'{source_id}; consume that source result (or its '
                    f'operator body) verbatim, or — to abandon that step '
                    f'itself — restart from its recorded input')
        continues = None
        prev = self.last_result()
        cur = result.get('input')
        if prev is not None and cur is not None:
            if prev == cur or _chain_links(prev, cur):
                continues = True
            else:
                eq = core_tactics.equal_exprs(prev, cur)
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
        if result.get('solutions') is not None:
            step['solutions'] = list(result['solutions'])
        if result.get('unknowns') is not None:
            step['unknowns'] = result['unknowns']
        if goal is not None:
            step['goal'] = goal
        if pending_branch is not None:
            # Presentation metadata only.  Mathematical authority still
            # comes exclusively from this step's registered tactic/check.
            step['exploration'] = self._branch_edge(pending_branch, step)
        # Admission mirrors replay: a step replay would reject must never
        # be recorded, or the session silently stops being a replayable
        # artifact.
        if step['check'].get('status') == 'disagree':
            raise ValueError(
                'the independent check disagrees with this result; a '
                'disagreeing step is not recorded — correct the arguments '
                'or take a different route')
        import tactic_registry
        provenance_error = tactic_registry.validate_provenance(
            step, {s['id']: s for s in self.steps})
        if provenance_error:
            raise ValueError(
                f'{provenance_error}; a step that would fail replay is '
                'not recorded — resolve the cited sources from recorded '
                'steps, or run without a session for an unrecorded check')
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

    def record_branch(self, from_step, reason, goal=None):
        """Record an exploration-only resume marker.

        The marker names an earlier transforming step but carries no result,
        check evidence, or provenance authority.  It makes an agent's chosen
        resume point auditable without deleting the abandoned checked steps.
        """
        from_step = (from_step or '').strip()
        reason = (reason or '').strip()
        if not from_step:
            raise ValueError('branch marker needs a source step id')
        if not reason:
            raise ValueError('branch marker needs a reason')
        if goal is not None and self.get_claim(goal) is None:
            raise ValueError(f'unknown goal {goal!r}')
        pending = self._pending_branch(goal)
        if pending is not None:
            raise ValueError(
                f'branch marker {pending["id"]} still needs its continuing '
                'transforming step; markers do not stack — record a plain '
                'comment (no from_step) for a note, or run the continuing '
                'tactic first')
        source = next((s for s in self.steps if s.get('id') == from_step),
                      None)
        if source is None:
            raise ValueError(f'unknown branch source {from_step!r}')
        if (source.get('op') not in TRANSFORMING_OPS
                or source.get('result') is None):
            raise ValueError(
                f'branch source {from_step!r} is not a transforming step')
        if source.get('goal') != goal:
            raise ValueError(
                f'branch source {from_step!r} belongs to goal '
                f'{source.get("goal")!r}, not {goal!r}')
        n = len(self.steps) + 1
        step = {
            'id': f's{n}',
            'continues': None,
            'hash': _branch_hash(from_step, reason),
            'op': 'branch',
            'args': {'from': from_step, 'reason': reason},
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
                    f'not {claim_id}; only steps recorded while '
                    f'{claim_id} is the focused goal can close it — '
                    f're-state the claim to focus it, then re-run the '
                    f'closing tactics')
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
        premise = None
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
        if closure is None and _eq_yes(endpoint, claim['statement']):
            # An answer-shaped claim ("A = 1/2") states what an unknown IS,
            # so _relation_holds can never decide it: asking whether A
            # equals 1/2 IS the open question. What the checked chain does
            # establish is one-directional — from its own first input the
            # endpoint follows — so the claim closes CONDITIONAL on that
            # premise, which travels with the verdict.
            candidate = first.get('input')
            # deriving the claim from itself establishes nothing
            if (candidate and _relation_parts(candidate) is not None
                    and not _eq_yes(candidate, endpoint)):
                decided = core_tactics.evaluate(candidate)
                if decided.get('ok') and decided.get('holds') is False:
                    raise ValueError(
                        f'the chain starts from {candidate!r}, which is '
                        'false; a claim derived from it holds vacuously')
                closure, premise = 'derived-from-premise', candidate
        if closure is None:
            raise ValueError(
                f'chain endpoint {endpoint!r} does not close claim '
                f'{claim["statement"]!r}')

        assumptions = []
        for step in selected:
            for assumption in step.get('assumptions', []):
                if assumption not in assumptions:
                    assumptions.append(assumption)
        # a chain that borrows from two alternative cases proves nothing:
        # its stated condition could never hold
        import primitives
        exclusive = primitives.exclusive_hypotheses(assumptions)
        if exclusive:
            first, second = exclusive[0]
            raise ValueError(
                f'the chain rests on mutually exclusive hypotheses '
                f'{assumptions[first]["text"]!r} and '
                f'{assumptions[second]["text"]!r}; close each case as its '
                f'own claim')
        verdict = ('conditional' if assumptions or premise is not None
                   else 'established')
        conclusion = {
            'steps': list(step_ids),
            'endpoint': endpoint,
            'assumptions': assumptions,
            'closure': closure,
            'verdict': verdict,
        }
        if premise is not None:
            conclusion['premise'] = premise
        return conclusion

    def conclude(self, claim_id, step_ids):
        """Mechanically close a claim from goal-owned, checked steps."""
        conclusion = self._validate_conclusion(claim_id, step_ids)
        claim = self.get_claim(claim_id)
        claim['verdict'] = conclusion.pop('verdict')
        claim['conclusion'] = conclusion
        return claim

    def _selection_error(self, selection, steps=None):
        """Return a replay/recording error for a final-result selection.

        A selection records presentation provenance only.  It can point at
        checked mathematical authority, but can never create that authority.
        """
        import primitives
        steps = self.steps if steps is None else steps
        result = selection.get('result')
        provenance = selection.get('provenance')
        if not isinstance(provenance, dict):
            return 'selection provenance must be an object'
        goal = selection.get('goal')
        if goal is not None and self.get_claim(goal) is None:
            return f'unknown selection goal {goal!r}'
        source = provenance.get('source')
        status = provenance.get('status')
        if source == 'open':
            # A run-level open outcome selects nothing: it records only
            # that the session exhibited no certified result when it
            # ended. It never asserts a mathematical nonexistence and can
            # neither cite nor create checked authority.
            if result is not None:
                return 'open outcome cannot carry a selected result'
            if status != 'open':
                return 'open outcome must have open status'
            reason = provenance.get('reason')
            if not isinstance(reason, str) or not reason.strip():
                return 'open outcome needs a reason'
            if len(reason) > 400:
                return 'open outcome reason exceeds the narrative cap'
            if provenance.get('step') or provenance.get('claim'):
                return 'open outcome cannot cite checked authority'
            if selection.get('hash') != _selection_hash(
                    result, provenance, goal):
                return 'selection hash mismatch'
            return None
        if not isinstance(result, str) or not result.strip():
            return 'missing selected result'
        try:
            primitives.parse_latex(result)
        except primitives.PrimitiveError as exc:
            return f'invalid selected result: {exc}'
        if selection.get('hash') != _selection_hash(
                result, provenance, goal):
            return 'selection hash mismatch'

        if source == 'ledger':
            step_id = provenance.get('step')
            step = next((s for s in steps if s.get('id') == step_id), None)
            if (step is None or step.get('op') not in TRANSFORMING_OPS
                    or step.get('result') is None):
                return f'unknown transforming source {step_id!r}'
            if status != 'verified':
                return 'ledger selection must have verified status'
            if not _eq_yes(result, step['result']):
                return f'selected result does not match {step_id}'
            return None
        if source == 'claim':
            claim_id = provenance.get('claim')
            claim = self.get_claim(claim_id)
            if claim is None or claim.get('verdict') == 'open':
                return f'unknown or open claim source {claim_id!r}'
            conclusion = claim.get('conclusion') or {}
            if status != claim.get('verdict'):
                return 'claim selection status does not match its verdict'
            if provenance.get('steps') != conclusion.get('steps'):
                return 'claim selection steps do not match its conclusion'
            if provenance.get('method') != conclusion.get('closure'):
                return 'claim selection method does not match its conclusion'
            if not _eq_yes(result, conclusion.get('endpoint')):
                return f'selected result does not match claim {claim_id}'
            return None
        if source == 'query-only':
            if status != 'unverified':
                return 'query-only selection must be unverified'
            if provenance.get('step') or provenance.get('claim'):
                return 'query-only selection cannot cite checked authority'
            return None
        return f'unknown selection source {source!r}'

    def record_selection(self, result, provenance, goal=None):
        """Append the final-result provenance chosen by ``set_result``.

        This separate control record lets a later CLI renderer recover the
        selected spine even when the chosen result came from an earlier step.
        It is deliberately not a ledger step and cannot close a claim or feed
        a provenance-aware tactic.
        """
        provenance = dict(provenance or {})
        selection = {
            'id': f'r{len(self.selections) + 1}',
            'result': result,
            'provenance': provenance,
        }
        if goal is not None:
            selection['goal'] = goal
        selection['hash'] = _selection_hash(
            result, provenance, selection.get('goal'))
        error = self._selection_error(selection)
        if error is not None:
            raise ValueError(error)
        self.selections.append(selection)
        return selection

    def record_open(self, reason, goal=None):
        """Append a run-level OPEN outcome: as of this record the session
        exhibits no certified result.

        The record claims nothing about the mathematics — the vocabulary
        is the claim layer's "open", never "no solution exists" — so its
        validity is ledger-decidable on replay.  A later certified
        selection supersedes it for display; the abandoned reasons stay
        in the append-only record."""
        provenance = {'status': 'open', 'source': 'open',
                      'reason': (reason or '').strip()}
        return self.record_selection(None, provenance, goal=goal)

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
        import tactic_registry
        seen = {}
        replayed_steps = []
        for step in self.steps:
            if step['op'] == 'branch':
                args = step.get('args') or {}
                from_step = args.get('from')
                reason = args.get('reason')
                source = seen.get(from_step)
                error = None
                if not isinstance(from_step, str) or not from_step:
                    error = 'missing source step id'
                elif not isinstance(reason, str) or not reason.strip():
                    error = 'missing reason'
                elif source is None:
                    error = f'unknown or forward source {from_step!r}'
                elif (source.get('op') not in TRANSFORMING_OPS
                      or source.get('result') is None):
                    error = f'source {from_step!r} is not transforming'
                elif source.get('goal') != step.get('goal'):
                    error = (f'source {from_step!r} belongs to goal '
                             f'{source.get("goal")!r}, not '
                             f'{step.get("goal")!r}')
                else:
                    pending = self._pending_branches_in(
                        replayed_steps, step.get('goal'))
                    current_is_legacy = (step.get('hash')
                                         == _legacy_branch_hash(
                                             from_step, reason))
                    pending_has_new = any(
                        p.get('hash') != _legacy_branch_hash(
                            (p.get('args') or {}).get('from'),
                            (p.get('args') or {}).get('reason'))
                        for p in pending)
                    if pending and (not current_is_legacy
                                    or pending_has_new):
                        error = 'previous marker has no continuing step'
                if error is None and (step.get('input') is not None
                      or step.get('result') is not None
                      or step.get('assumptions') != []
                      or step.get('check') != {'status': 'note'}
                      or step.get('continues') is not None):
                    error = 'marker carries transforming fields'
                elif error is None and step.get('hash') not in (
                        _branch_hash(from_step, reason),
                        _legacy_branch_hash(from_step, reason)):
                    error = 'marker hash mismatch'
                if error is not None:
                    return {'status': 'failed', 'step': step.get('id', '?'),
                            'reason': f'branch marker invalid: {error}'}
                seen[step['id']] = step
                replayed_steps.append(step)
                continue
            if step['op'] == 'comment':
                seen[step['id']] = step
                replayed_steps.append(step)
                continue
            pending_branches = self._pending_branches_in(
                replayed_steps, step.get('goal'))
            if pending_branches:
                for pending_branch in pending_branches:
                    source_id = pending_branch['args']['from']
                    source = seen.get(source_id)
                    # mirror of record()'s resume gate: identical comparator
                    if (source is None
                            or not (_chain_links(source.get('result'),
                                                 step.get('input'))
                                    or _chain_links(source.get('input'),
                                                    step.get('input')))):
                        return {
                            'status': 'failed', 'step': step.get('id', '?'),
                            'reason': (f'branch edge invalid: '
                                       f'{step.get("id")} does not resume '
                                       f'{source_id}')}
                pending_branch = pending_branches[-1]
                source_id = pending_branch['args']['from']
                expected_edge = self._branch_edge(pending_branch, step)
                all_legacy = all(
                    p.get('hash') == _legacy_branch_hash(
                        p['args'].get('from'), p['args'].get('reason'))
                    for p in pending_branches)
                if step.get('exploration') != expected_edge:
                    # Legacy ledgers did not persist the target half. Their
                    # old marker hashes opt into deterministic derivation;
                    # new markers require the recorded edge.
                    if not (all_legacy and step.get('exploration') is None):
                        return {
                            'status': 'failed', 'step': step.get('id', '?'),
                            'reason': 'branch edge metadata mismatch'}
            elif step.get('exploration') is not None:
                return {'status': 'failed', 'step': step.get('id', '?'),
                        'reason': 'branch edge has no preceding marker'}
            provenance_error = tactic_registry.validate_provenance(step,
                                                                   seen)
            if provenance_error:
                return {'status': 'failed', 'step': step['id'],
                        'reason': provenance_error}
            res = tactic_registry.replay(step['op'], step['args'])
            if not res.get('ok'):
                return {'status': 'failed', 'step': step['id'],
                        'reason': res.get('error')}
            if res.get('result') != step['result']:
                # tolerate formatting drift between versions, but only if
                # the results are semantically equal
                eq = core_tactics.equal_exprs(res.get('result'),
                                              step['result'])
                if not (eq.get('ok') and eq.get('verdict') == 'yes'):
                    return {'status': 'failed', 'step': step['id'],
                            'reason': 'result mismatch',
                            'recorded': step['result'],
                            'replayed': res.get('result')}
            if ('solutions' in step
                    and res.get('solutions') != step['solutions']):
                return {'status': 'failed', 'step': step['id'],
                        'reason': 'solution metadata mismatch'}
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
                # shape-only validation, mirroring record_claim: an ellipsis
                # claim is recordable and closes only through an
                # interpretation step, so replay must not reject it here
                primitives.parse_latex(claim.get('statement', ''),
                                       allow_ellipsis=True)
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
        for index, selection in enumerate(self.selections, 1):
            if not isinstance(selection, dict):
                return {'status': 'failed', 'selection': f'r{index}',
                        'reason': ('final selection invalid: record must be '
                                   'an object')}
            if selection.get('id') != f'r{index}':
                return {'status': 'failed',
                        'selection': selection.get('id', '?'),
                        'reason': 'final selection invalid: id mismatch'}
            error = self._selection_error(selection, steps=replayed_steps)
            if error is not None:
                return {'status': 'failed',
                        'selection': selection.get('id', '?'),
                        'reason': f'final selection invalid: {error}'}
        return {'status': 'verified', 'steps': len(self.steps),
                'claims': len(self.claims),
                'selections': len(self.selections),
                'open_claims': sum(c.get('verdict') == 'open'
                                   for c in self.claims),
                'assumptions': self.assumptions}

    def branch_edges(self):
        """Return the exploration edges encoded by marker/target order.

        New ledgers persist the target half on the transforming step.  The
        scan remains deterministic so legacy files can derive the same edge.
        Unresolved end-of-session markers have ``to=None`` and are not errors.
        """
        edges = []
        for index, marker in enumerate(self.steps):
            if marker.get('op') != 'branch':
                continue
            goal = marker.get('goal')
            target = next((step for step in self.steps[index + 1:]
                           if (step.get('goal') == goal
                               and step.get('op') in TRANSFORMING_OPS
                               and step.get('result') is not None)), None)
            args = marker.get('args') or {}
            edge = {
                'marker': marker.get('id'),
                'from': args.get('from'),
                'to': target.get('id') if target is not None else None,
                'reason': args.get('reason'),
                'goal': goal,
                'kind': 'exploration',
                'authority': 'annotation',
                'persisted': bool(target and target.get('exploration')),
            }
            if (target is not None and (target.get('exploration')
                                        or {}).get('anchor') == 'input'):
                edge['anchor'] = 'input'
            edges.append(edge)
        return edges

    def premises(self, step_ids=None):
        """Inputs this session never derived, in ledger order.

        A derivation must start somewhere, so a premise is not a fault — but
        it is exactly where the checking stops, and a reader who cannot see
        the premises cannot tell a derivation from a restatement. Returns
        `[{step, input}]`, deduplicated by expression: the same given used
        twice was stated once.
        """
        wanted = None if step_ids is None else set(step_ids)
        premises = []
        for index, step in enumerate(self.steps):
            if wanted is not None and step['id'] not in wanted:
                continue
            if step.get('result') is None or _is_derived(step,
                                                         self.steps[:index]):
                continue
            current = step['input']
            if any(seen['input'] == current
                   or _all_bracket_normal_form_equal(seen['input'], current)
                   for seen in premises):
                continue
            premises.append({'step': step['id'], 'input': current})
        return premises

    def presentation_topology(self, final_provenance=None, marker_ids=None):
        """Derive the selected spine and annotation-only abandoned paths.

        The returned structure is presentation data.  It never participates
        in claim closure, tactic provenance, or replay authority.
        """
        transforms = [s for s in self.steps
                      if (s.get('op') in TRANSFORMING_OPS
                          and s.get('result') is not None)]
        by_id = {s['id']: s for s in self.steps}
        order = {s['id']: i for i, s in enumerate(self.steps)}
        edges = self.branch_edges()
        edge_by_target = {e['to']: e for e in edges if e.get('to')}

        previous_transform = None
        parents = {}
        for step in transforms:
            edge = edge_by_target.get(step['id'])
            if edge is not None and edge.get('anchor') == 'input':
                # the target restarts from the state BEFORE the source
                # step, so it inherits the source's parent (the source
                # itself is the abandoned route)
                parents[step['id']] = parents.get(edge['from'])
            elif edge is not None:
                parents[step['id']] = edge['from']
            elif previous_transform is not None and (
                    step.get('continues') is True
                    # sessions recorded before the chaining convention was
                    # linkage-visible persist continues=False/None on honest
                    # body-convention chains; re-derive structurally
                    or _chain_links(previous_transform.get('result'),
                                    step.get('input'))):
                parents[step['id']] = previous_transform.get('id')
            else:
                parents[step['id']] = None
            previous_transform = step

        provenance = final_provenance
        selection = None
        if provenance is None and self.selections:
            selection = self.selections[-1]
            provenance = selection.get('provenance') or {}
            if provenance.get('source') == 'open':
                # an open outcome selects nothing; an earlier certified
                # selection (or a concluded claim below) still owns the
                # displayed spine. An explicitly passed open provenance
                # stays literal: that run's own render has no spine.
                provenance = next(
                    ((s.get('provenance') or {})
                     for s in reversed(self.selections)
                     if (s.get('provenance') or {}).get('source') != 'open'),
                    None)
        targets = []
        selected_goal = None
        if provenance:
            if provenance.get('source') == 'ledger':
                step_id = provenance.get('step')
                if step_id in by_id:
                    targets.append(step_id)
                    selected_goal = by_id[step_id].get('goal')
            elif provenance.get('source') == 'claim':
                claim = self.get_claim(provenance.get('claim'))
                if claim is not None:
                    targets.extend((claim.get('conclusion') or {}).get(
                        'steps') or [])
                    selected_goal = claim.get('id')
        elif self.claims:
            # A conclusion is itself persisted final-spine evidence. Prefer
            # the newest closed root, then the newest closed subclaim.
            closed = [c for c in self.claims
                      if c.get('verdict') != 'open']
            roots = [c for c in closed if c.get('parent') is None]
            claim = (roots or closed or [None])[-1]
            if claim is not None:
                targets.extend((claim.get('conclusion') or {}).get(
                    'steps') or [])
                selected_goal = claim.get('id')

        spine = set()

        def include(step_id):
            if step_id in spine or step_id not in by_id:
                return
            step = by_id[step_id]
            if step.get('result') is None:
                return
            spine.add(step_id)
            parent = parents.get(step_id)
            if parent:
                include(parent)
            for source_id in _source_ids(step):
                include(source_id)

        for target in targets:
            include(target)
        spine_ids = [s['id'] for s in transforms if s['id'] in spine]

        allowed_markers = (None if marker_ids is None else set(marker_ids))
        assigned = set()
        abandoned = []

        def descends_from(step_id, source_id):
            seen = set()
            current = step_id
            while current and current not in seen:
                if current == source_id:
                    return True
                seen.add(current)
                current = parents.get(current)
            return False

        for edge in edges:
            marker_id = edge['marker']
            if (allowed_markers is not None
                    and marker_id not in allowed_markers):
                continue
            source_id = edge.get('from')
            if not spine or source_id not in order or marker_id not in order:
                continue
            # an input-anchored resume abandons the source step itself,
            # so the source joins its own dead route
            first = (order[source_id] if edge.get('anchor') == 'input'
                     else order[source_id] + 1)
            candidates = [
                step['id'] for step in transforms
                if (first <= order[step['id']] < order[marker_id]
                    and step.get('goal') == edge.get('goal')
                    and step['id'] not in spine
                    and step['id'] not in assigned
                    and descends_from(step['id'], source_id))]
            if not candidates:
                continue
            assigned.update(candidates)
            path = {
                'marker': marker_id,
                'source': source_id,
                'continues_at': edge.get('to'),
                'reason': edge.get('reason'),
                'steps': candidates,
            }
            if edge.get('anchor') == 'input':
                path['anchor'] = 'input'
            abandoned.append(path)

        off_spine = [s['id'] for s in transforms if s['id'] not in spine]
        unclassified = [sid for sid in off_spine if sid not in assigned]
        spine_assumptions = []
        for sid in spine_ids:
            for assumption in by_id[sid].get('assumptions', []):
                if assumption not in spine_assumptions:
                    spine_assumptions.append(assumption)
        return {
            'spine_premises': self.premises(spine_ids or None),
            'selection': selection.get('id') if selection else None,
            'selected_goal': selected_goal,
            'edges': edges,
            'spine': spine_ids,
            'spine_assumptions': spine_assumptions,
            'parents': parents,
            'abandoned_paths': abandoned,
            'unclassified_off_spine': unclassified,
            'unresolved_markers': [e['marker'] for e in edges
                                   if e.get('to') is None],
        }

    _MARKS = {'agree': 'verified', 'exact': 'exact',
              'skipped': 'unchecked', 'disagree': 'FAILED',
              'domain-differs': 'DOMAIN DIFFERS'}

    def render_markdown(self):
        """Render Markdown, folding marker-classified off-spine work."""
        title = '# Derivation ledger' if self.claims else '# Verified derivation'
        lines = [title, '']
        topology = self.presentation_topology()
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
                f'${_display_latex(claim["statement"])}$')
            if verdict == 'OPEN':
                lines.append('')
                lines.append('*No mechanically checked closing chain has '
                             'been recorded.*')
            elif conclusion.get('premise'):
                # the premise is the whole content of a conditional answer
                # claim: it must never be one click away from the verdict
                lines.append('')
                lines.append(
                    '*Derived from the stated premise '
                    f'${_display_latex(conclusion["premise"])}$.*')
            lines.append('')
        ended_open = bool(self.selections and (self.selections[-1].get(
            'provenance') or {}).get('source') == 'open')
        if self.selections:
            selected = self.selections[-1]
            provenance = selected.get('provenance') or {}
            if ended_open:
                lines.append(
                    f'**Outcome `{selected["id"]}` — OPEN:** no certified '
                    'result in this session. '
                    f'{_markdown_prose(provenance.get("reason", ""))} '
                    '*(unverified reason)*')
                lines.append('')
            else:
                source = provenance.get('step') or provenance.get('claim')
                source_note = f' from `{source}`' if source else ''
                status = provenance.get('status', 'unknown').upper()
                lines.append(
                    f'**Selected final result `{selected["id"]}`'
                    f'{source_note} — {status}:** '
                    f'${_display_latex(selected["result"])}$')
                lines.append('')

        final_premises = (topology['spine_premises'] if topology['spine']
                          else self.premises())
        if final_premises:
            # where the checking starts. A reader who cannot see this cannot
            # tell a derivation from a restatement of its own answer.
            stated = ', '.join(f'${_display_latex(p["input"])}$'
                               for p in final_premises)
            lines.append(
                f'*Rests on {len(final_premises)} stated premise'
                f'{"s" if len(final_premises) != 1 else ""}, not derived '
                f'here: {stated}.*')
            lines.append('')

        final_assumptions = (topology['spine_assumptions']
                             if topology['spine'] else self.assumptions)
        if final_assumptions:
            # hypotheses from alternative cases must never read as one
            # conjunction: nothing holds under `x > 0` AND `x < 0`
            import primitives
            split = {i for pair in primitives.exclusive_hypotheses(
                final_assumptions) for i in pair}
            shared = [a for i, a in enumerate(final_assumptions)
                      if i not in split]
            alternatives = [a for i, a in enumerate(final_assumptions)
                            if i in split]
            if shared:
                label = ('**Selected spine is valid under the assumptions:** '
                         if topology['spine'] else
                         '**Valid under the assumptions:** ')
                lines.append(label + ', '.join(
                    assumption_markdown(a) for a in shared))
                lines.append('')
            if alternatives:
                lines.append(
                    '**Alternative case hypotheses** (each step holds under '
                    'the one it records, not under all of them): '
                    + ' | '.join(assumption_markdown(a)
                                 for a in alternatives))
                lines.append('')

        def render_step(step):
            out = []
            if step['op'] == 'comment':
                out.append(f"**{step['id']}** *note* — "
                           f"{_markdown_prose(step['args']['text'])}")
                out.append('')
                return out
            if step['op'] == 'branch':
                edge = next((e for e in topology['edges']
                             if e['marker'] == step['id']), None)
                target = (f" to `{edge['to']}`" if edge and edge.get('to')
                          else '')
                pending = ('' if target else
                           (' *(left unresolved; outcome recorded open)*'
                            if ended_open else
                            ' *(awaiting a continuing step)*'))
                out.append(
                    f"**{step['id']}** *branch from "
                    f"`{step['args']['from']}`{target}*{pending} — "
                    f"{_markdown_prose(step['args']['reason'])}")
                out.append('')
                return out
            check = step['check'].get('status', '?')
            mark = self._MARKS.get(check, check)
            edge = step.get('exploration')
            if edge:
                branch = (f' *(resumes from `{edge["from"]}` via '
                          f'`{edge["marker"]}`)*')
            else:
                linked = (step.get('continues') in (True, None)
                          or topology['parents'].get(step['id']))
                branch = ('' if linked
                          else ' *(new chain; no exploration edge)*')
            arg_note = ''
            if step['op'] == 'apply_both_sides':
                a = step['args']
                arg_note = f" — `{a['op']} {a['arg']}` on both sides"
            elif step['op'] == 'substitute':
                a = step['args']
                arg_note = (f" — ${_display_latex(a['var'])} := "
                            f"{_display_latex(a['value'])}$")
            elif step['op'] == 'integrate_by_parts':
                a = step['args']
                arg_note = (f" — $u = {_display_latex(a['u'])}$, "
                            f"$dv = {_display_latex(a['dv'])}$")
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
            elif step['op'] == 'points_assemble':
                src = step.get('sources', {})
                ids = ', '.join(src.get('values', []))
                # naming the paired function keeps the reader from assuming
                # it is whatever the chain last mentioned
                expr = _display_latex(step['args'].get('expr', ''))
                arg_note = (f" — values of ${expr}$ "
                            f"from `{src.get('roots', '?')}` → `{ids}`")
            elif step['op'] == 'system_assemble':
                src = step.get('sources', {})
                ids = ', '.join(src.get('assignments', []))
                unknowns = ', '.join(u['unknown']
                                     for u in step.get('unknowns') or [])
                arg_note = (f" — values for ${unknowns}$ from `{ids}`"
                            if ids else f" — values for ${unknowns}$")
            goal = (f" → `{step['goal']}`" if step.get('goal') else '')
            out.append(f"**{step['id']}**{goal} `{step['op']}`{arg_note} "
                       f"— *{mark}*{branch}")
            out.append('')
            input_latex = _display_latex(step['input'])
            result_latex = _display_latex(step['result'])
            out.append(f"$${input_latex} \\;\\Longrightarrow\\; "
                       f"{result_latex}$$")
            for a in step['assumptions']:
                out.append(f"- assumes {assumption_markdown(a)}")
            out.append('')
            return out

        paths_by_first = {}
        folded_steps = set()
        folded_markers = set()
        by_id = {s['id']: s for s in self.steps}
        for path in topology['abandoned_paths']:
            if not path['steps']:
                continue
            paths_by_first[path['steps'][0]] = path
            folded_steps.update(path['steps'])
            folded_markers.add(path['marker'])

        for step in self.steps:
            path = paths_by_first.get(step['id'])
            if path is not None:
                count = len(path['steps'])
                continuing = (f"; resumed as <code>{path['continues_at']}"
                              f"</code>" if path.get('continues_at') else '')
                lines.append('<details>')
                lines.append(
                    '<summary><strong>Abandoned path from '
                    f'<code>{path["source"]}</code></strong> — '
                    f'{_markdown_prose(path["reason"])} '
                    f'({count} checked step{"s" if count != 1 else ""}'
                    f'{continuing})</summary>')
                lines.append('')
                for path_step_id in path['steps']:
                    lines.extend(render_step(by_id[path_step_id]))
                lines.append('</details>')
                lines.append('')
                continue
            if step['id'] in folded_steps or step['id'] in folded_markers:
                continue
            lines.extend(render_step(step))
        lines.append(f'*{len(self.steps)} steps; replay with '
                     f'`toymath replay --session <file>`.*')
        return '\n'.join(lines)

    def render(self):
        """Terse human/agent-readable summary of the derivation."""
        lines = []
        topology = self.presentation_topology()
        edge_by_marker = {e['marker']: e for e in topology['edges']}
        ended_open = bool(self.selections and (self.selections[-1].get(
            'provenance') or {}).get('source') == 'open')
        for claim in self.claims:
            verdict = claim.get('verdict', 'open').upper()
            conclusion = claim.get('conclusion') or {}
            detail = ''
            if verdict != 'OPEN':
                detail = ('; steps ' + ','.join(conclusion.get('steps', []))
                          + f'; endpoint {conclusion.get("endpoint")}')
                if conclusion.get('premise'):
                    detail += f'; given {conclusion["premise"]}'
            lines.append(f"CLAIM {claim['id']}#{claim['hash']} [{verdict}] "
                         f"{claim['statement']}{detail}")
        for step in self.steps:
            if step['op'] == 'comment':
                lines.append(f"{step['id']}#{step['hash']} [--] note: "
                             f"{step['args']['text']}")
                continue
            if step['op'] == 'branch':
                edge = edge_by_marker.get(step['id']) or {}
                target = (f" to {edge['to']}" if edge.get('to')
                          else (' (unresolved; outcome open)' if ended_open
                                else ' (awaiting continuation)'))
                lines.append(
                    f"{step['id']}#{step['hash']} [--] branch from "
                    f"{step['args']['from']}{target}: "
                    f"{step['args']['reason']}")
                continue
            check = step['check'].get('status', '?')
            mark = {'agree': 'ok', 'exact': 'ok', 'skipped': '??',
                    'disagree': 'XX', 'domain-differs': 'D!'}.get(check, '?')
            linked = (step.get('continues') in (True, None)
                      or topology['parents'].get(step['id']))
            branch = '' if linked else ' (branch)'
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
            elif step['op'] == 'points_assemble':
                src = step.get('sources', {})
                lines.append(
                    '      sources: roots ' + src.get('roots', '?')
                    + '; values of ' + step['args'].get('expr', '?')
                    + ' from ' + ', '.join(src.get('values', [])))
            elif step['op'] == 'system_assemble':
                src = step.get('sources', {})
                lines.append(
                    '      sources: values for '
                    + ', '.join(u['unknown']
                                for u in step.get('unknowns') or [])
                    + ' from ' + ', '.join(src.get('assignments', [])))
            for a in step['assumptions']:
                lines.append(f"      assumes {a['text']}")
        visible_premises = (topology['spine_premises'] if topology['spine']
                            else self.premises())
        if visible_premises:
            lines.append('premises (stated, not derived here): '
                         + '; '.join(p['input'] for p in visible_premises))
        visible_assumptions = (topology['spine_assumptions']
                               if topology['spine'] else self.assumptions)
        if visible_assumptions:
            label = ('selected assumptions: ' if topology['spine']
                     else 'assumptions: ')
            lines.append(label + '; '.join(
                a['text'] for a in visible_assumptions))
        if self.selections:
            selected = self.selections[-1]
            provenance = selected.get('provenance') or {}
            if ended_open:
                lines.append(
                    f"OPEN {selected['id']}#{selected['hash']} no certified "
                    f"result: {provenance.get('reason', '')}")
            else:
                source = (provenance.get('step') or provenance.get('claim')
                          or '?')
                status = provenance.get('status', 'unknown').upper()
                lines.append(
                    f"SELECT {selected['id']}#{selected['hash']} [{status}] "
                    f"from {source}: "
                    f"{selected['result']}")
        return '\n'.join(lines)

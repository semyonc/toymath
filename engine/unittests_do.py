#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the do! agent endpoint (agent_do.py + MathShell integration).

Everything here runs offline: the agent loop is exercised with a scripted
fake Model. Set TOYMATH_LIVE_TESTS=1 to also run one real OpenRouter
round-trip (requires OPEN_ROUTER in the environment/.env).
"""
import json
import os
import unittest
from unittest import mock

import agent_do
import observability
import plot_sandbox
import tactic_registry
from tactics import core as core_tactics
from tactics import integration as integration_tactics
from agent_do import DoSession, make_api, run_instruction
from ledger import Ledger

from agents.models.interface import Model
from agents.items import ModelResponse
from agents.usage import Usage
from openai.types.responses import (ResponseFunctionToolCall,
                                    ResponseOutputMessage,
                                    ResponseOutputText)


class ScriptedModel(Model):
    """Deterministic Model: replays a fixed list of turns."""

    def __init__(self, turns):
        self.turns = list(turns)

    async def get_response(self, system_instructions, input, model_settings,
                           tools, output_schema, handoffs, tracing, *,
                           previous_response_id=None, conversation_id=None,
                           prompt=None):
        if not self.turns:
            raise AssertionError('scripted model ran out of turns')
        return ModelResponse(output=self.turns.pop(0), usage=Usage(),
                             response_id=None)

    def stream_response(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError


def tool_call(name, args, cid):
    if name in tactic_registry.BY_NAME:
        spec = tactic_registry.BY_NAME[name]
        ordered = []
        for arg in spec.agent_args:
            value = args.get(arg.name, arg.default)
            if value is tactic_registry._MISSING:
                raise AssertionError(f'missing scripted argument {arg.name}')
            if arg.nargs in ('+', '*'):
                ordered.extend(value)
            else:
                ordered.append(value)
        name = 'run_tactic'
        args = {'tactic': spec.name, 'arguments': ordered}
    return ResponseFunctionToolCall(type='function_call', name=name,
                                    arguments=json.dumps(args),
                                    call_id=cid, id=cid)


def message(text):
    return ResponseOutputMessage(
        id='m1', type='message', role='assistant', status='completed',
        content=[ResponseOutputText(type='output_text', text=text,
                                    annotations=[])])


SOLVE_SCRIPT = [
    [tool_call('apply', {'equation': '2x + 3 = 7', 'op': '-', 'arg': '3'},
               'c1')],
    [tool_call('expand', {'expr': '2x+3 - 3 = 7 - 3'}, 'c2')],
    [message('Subtracted 3 from both sides.')],
]


class TestPromptBuilder(unittest.TestCase):
    def test_skill_derived_prompt(self):
        p = agent_do.build_prompt()
        # frontmatter stripped, bash invocation replaced, rules appended
        self.assertFalse(p.startswith('---'))
        self.assertNotIn('python toymath_cli.py <command>', p)
        self.assertIn('Invocation inside do!', p)
        self.assertIn('## do! mode', p)
        # core signatures and the small domain catalog are present; detailed
        # subject tactics are progressively loaded
        self.assertIn('factor_quadratic', p)
        self.assertIn('`integration`', p)
        self.assertIn('`equations`', p)
        self.assertNotIn('integrate_by_parts EXPR', p)
        self.assertIn('never for\n  an equation whose solutions', p)
        self.assertIn('mechanically checked', p)

    def test_missing_skill_falls_back(self):
        p = agent_do.build_prompt(skill_path='/nonexistent/SKILL.md')
        self.assertIn('mechanical checker', p)
        self.assertIn('## do! mode', p)

    def test_prompt_forbids_restating_the_ledger(self):
        # live defect: an agent hand-retyped the step chain as a markdown
        # table (raw pipes in the em line, broken attachment: image link).
        # The notebook renders the verified chain from the ledger; the
        # prompt must keep final answers to plain commentary.
        p = agent_do.build_prompt()
        self.assertIn('Never restate ledger steps', p)
        self.assertIn('write image links', p)

    def test_prompt_names_structured_exploration_markers(self):
        p = agent_do.build_prompt()
        self.assertIn('exact step id as from_step', p)
        self.assertIn('source result verbatim to the\n  next tactic', p)
        self.assertIn('never mathematical case data', p)
        # the abandon-the-step-itself form (root-anchor gap) is steered too
        self.assertIn('abandon a recorded step ITSELF', p)
        self.assertIn("from that step's recorded input", p)

    def test_prove_prompt_names_actual_shared_ledger_claim(self):
        p = agent_do.build_prompt(prove_mode=True, proof_claim_id='c7')
        self.assertIn('root claim\n`c7`', p)
        self.assertIn('`c7` and the ordered step ids', p)
        self.assertNotIn('ROOT_CLAIM', p)


class TestDoSessionApi(unittest.TestCase):
    def test_transforming_call_recorded_and_streamed(self):
        seen = []
        session = DoSession(on_step=seen.append)
        api = make_api(session)
        rec = json.loads(api['apply']('2x + 3 = 7', '-', '3'))
        self.assertTrue(rec['ok'])
        self.assertEqual(rec['step']['id'], 's1')
        self.assertEqual(len(session.new_steps()), 1)
        self.assertEqual(seen[0]['id'], 's1')

    def test_query_calls_not_recorded(self):
        session = DoSession()
        api = make_api(session)
        rec = json.loads(api['equal']('(x+1)^2', 'x^2+2x+1'))
        self.assertEqual(rec['verdict'], 'yes')
        self.assertEqual(json.loads(api['lemmas']())['ok'], True)
        self.assertEqual(session.new_steps(), [])

    def test_failed_call_not_recorded(self):
        session = DoSession()
        api = make_api(session)
        rec = json.loads(api['apply']('2x + 3 = 7', '/', '0'))
        self.assertFalse(rec['ok'])
        self.assertEqual(session.new_steps(), [])

    def test_set_result_validates_query_only_as_unverified(self):
        session = DoSession()
        api = make_api(session)
        rec = json.loads(api['set_result']('x = 2'))
        self.assertTrue(rec['ok'])
        self.assertEqual(rec['provenance']['status'], 'unverified')
        self.assertEqual(rec['selection'], 'r1')
        self.assertEqual(session.ledger.selections[0]['result'], 'x = 2')
        self.assertEqual(session.result_override, 'x = 2')
        self.assertFalse(json.loads(api['set_result']('\\frac{'))['ok'])
        self.assertEqual(session.result_override, 'x = 2')  # kept
        self.assertEqual(session.new_steps(), [])  # never a ledger step

    def test_comment_streamed_but_not_transforming(self):
        seen = []
        session = DoSession(on_step=seen.append)
        api = make_api(session)
        rec = json.loads(api['comment']('strategy: partial fractions'))
        self.assertTrue(rec['ok'])
        self.assertEqual(rec['id'], 's1')
        self.assertEqual(seen[0]['op'], 'comment')
        # a note is not provenance: the run is still query-only
        res = json.loads(api['set_result']('x = 2'))
        self.assertEqual(res['provenance']['status'], 'unverified')
        self.assertFalse(json.loads(api['comment']('  '))['ok'])

    def test_comment_from_step_records_a_structured_branch_marker(self):
        seen = []
        session = DoSession(on_step=seen.append)
        api = make_api(session)
        source = json.loads(api['expand']('(x+1)^2'))['step']['id']
        rec = json.loads(api['comment'](
            'the substitution route stalled', source))
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['op'], 'branch')
        self.assertEqual(rec['from'], 's1')
        marker = session.new_steps()[-1]
        self.assertEqual(marker['args']['reason'],
                         'the substitution route stalled')
        self.assertIsNone(marker['result'])
        self.assertEqual(seen[-1]['op'], 'branch')
        self.assertEqual(session.ledger.replay()['status'], 'verified')
        bad = json.loads(api['comment']('resume', 's999'))
        self.assertFalse(bad['ok'])
        self.assertEqual(bad['op'], 'branch')

    def test_branch_marker_attaches_to_next_transform(self):
        session = DoSession()
        api = make_api(session)
        first = json.loads(api['expand']('(x+1)^2'))
        json.loads(api['substitute'](first['result'], 'x', '1'))
        marker = json.loads(api['comment'](
            'the numeric detour is not the symbolic answer', 's1'))
        refused = json.loads(api['expand']('(y+1)^2'))
        self.assertFalse(refused['ok'])
        self.assertIn('does not resume branch marker', refused['error'])
        self.assertEqual(len(session.new_steps()), 3)
        resumed = json.loads(api['factor_quadratic'](
            first['result'], 'x'))
        self.assertEqual(resumed['step']['id'], 's4')
        target = session.ledger.steps[-1]
        self.assertEqual(target['exploration']['marker'], marker['id'])
        self.assertEqual(target['exploration']['from'], 's1')
        selected = json.loads(api['set_result'](resumed['result']))
        self.assertEqual(selected['selection'], 'r1')
        topology = session.ledger.presentation_topology()
        self.assertEqual(topology['spine'], ['s1', 's4'])
        self.assertEqual(topology['abandoned_paths'][0]['steps'], ['s2'])
        self.assertEqual(session.ledger.replay()['status'], 'verified')

    def test_set_result_rejects_detached_conclusion(self):
        session = DoSession()
        api = make_api(session)
        api['factor_quadratic']('x^2 - 5x + 6', 'x')
        api['substitute']('x^2 - 5x + 6 = 0', 'x', '3')
        rec = json.loads(api['set_result']('(x-3)(x-2)=0'))
        self.assertFalse(rec['ok'])
        self.assertIn('shared ledger', rec['error'])
        self.assertIsNone(session.result_override)

    def test_set_result_can_select_earlier_semantic_result(self):
        session = DoSession()
        api = make_api(session)
        api['expand']('(x+1)^2')
        api['substitute']('x^2+2x+1=4', 'x', '1')
        rec = json.loads(api['set_result']('(x+1)^2'))
        self.assertTrue(rec['ok'])
        self.assertEqual(rec['provenance']['status'], 'verified')
        self.assertEqual(rec['provenance']['step'], 's1')
        self.assertNotEqual(rec['provenance']['method'], 'exact-result')
        self.assertEqual(session.ledger.selections[-1]['provenance']['step'],
                         's1')

    def test_set_result_can_select_shared_ledger_result(self):
        ledger = Ledger()
        first = DoSession(ledger=ledger)
        first_api = make_api(first)
        established = json.loads(first_api['expand']('(x+1)^2'))['result']
        session = DoSession(ledger=ledger)
        rec = json.loads(make_api(session)['set_result'](established))
        self.assertTrue(rec['ok'])
        self.assertEqual(rec['provenance']['step'], 's1')
        self.assertEqual(session.new_steps(), [])

    def test_plain_do_claim_does_not_become_governing_proof(self):
        session = DoSession()
        claim = session.claim('x^2-1=0')
        self.assertIsNone(session.proof_claim_id)
        step = json.loads(make_api(session)['expand']('x^2-1'))
        selected = json.loads(make_api(session)['set_result'](
            step['result']))
        self.assertTrue(selected['ok'], selected.get('error'))
        self.assertEqual(selected['provenance']['step'], 's1')
        self.assertEqual(session.ledger.get_claim(claim['id'])['verdict'],
                         'open')

    def test_plain_do_reuses_equivalent_open_claim(self):
        session = DoSession()
        first = session.claim('3x^{2}-3=0')
        again = session.claim('3x^2-3=0')
        self.assertEqual(again['id'], first['id'])
        self.assertEqual(len(session.ledger.claims), 1)

    def test_open_proof_rejects_unrelated_verified_value(self):
        session = DoSession()
        root = session.claim(
            r'\lim_{n \to \infty} \frac{n}{2^n} = 0', root=True)
        api = make_api(session)
        step = json.loads(api['limit_table'](
            r'\lim_{n \to \infty} \frac{1}{n}'))
        self.assertEqual(session.new_steps()[0]['goal'], root['id'])
        close = json.loads(api['conclude'](root['id'],
                                           [step['step']['id']]))
        self.assertFalse(close['ok'])
        self.assertIn('does not close claim', close['error'])
        selected = json.loads(api['set_result']('0'))
        self.assertFalse(selected['ok'])
        self.assertIn('still open', selected['error'])

    def test_concluded_proof_result_has_claim_provenance(self):
        session = DoSession()
        root = session.claim(r'\lim_{n \to \infty} \frac{1}{n} = 0',
                             root=True)
        api = make_api(session)
        step = json.loads(api['limit_table'](
            r'\lim_{n \to \infty} \frac{1}{n}'))
        close = json.loads(api['conclude'](root['id'],
                                           [step['step']['id']]))
        self.assertTrue(close['ok'], close.get('error'))
        selected = json.loads(api['set_result']('0'))
        self.assertTrue(selected['ok'])
        self.assertEqual(selected['provenance']['source'], 'claim')
        self.assertEqual(selected['provenance']['claim'], 'c1')

    def test_integrate_assemble_uses_ordered_ledger_results(self):
        session = DoSession()
        api = make_api(session)
        linearity = json.loads(api['integrate_linearity'](
            '\\int (x+x^2) \\, d x', 'x'))
        first = json.loads(api['integrate_power_rule'](
            linearity['integrals'][0], 'x'))
        second = json.loads(api['integrate_power_rule'](
            linearity['integrals'][1], 'x'))
        rec = json.loads(api['integrate_assemble'](
            linearity['step']['id'],
            [first['step']['id'], second['step']['id']]))
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['sources'], {
            'linearity': 's1', 'antiderivatives': ['s2', 's3']})
        self.assertEqual(rec['step']['id'], 's4')
        selected = json.loads(api['set_result'](rec['result']))
        self.assertTrue(selected['ok'])
        self.assertEqual(selected['provenance']['step'], 's4')
        self.assertEqual(session.ledger.replay()['status'], 'verified')

    def test_limit_assemble_uses_ordered_ledger_results(self):
        session = DoSession()
        api = make_api(session)
        linearity = json.loads(api['limit_linearity'](
            '\\lim_{x \\to 0} (\\frac{\\sin x}{x}+x^2)'))
        first = json.loads(api['limit_table'](
            linearity['limits'][0]))
        second = json.loads(api['limit_substitute'](
            linearity['limits'][1]))
        rec = json.loads(api['limit_assemble'](
            linearity['step']['id'],
            [first['step']['id'], second['step']['id']]))
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['result'], '1')
        self.assertEqual(rec['sources'], {
            'linearity': 's1', 'values': ['s2', 's3']})
        self.assertEqual(session.ledger.replay()['status'], 'verified')

    def test_limit_assemble_rejects_wrong_source_order(self):
        session = DoSession()
        api = make_api(session)
        linearity = json.loads(api['limit_linearity'](
            '\\lim_{x \\to 0} (\\frac{\\sin x}{x}+x^2)'))
        first = json.loads(api['limit_table'](linearity['limits'][0]))
        second = json.loads(api['limit_substitute'](linearity['limits'][1]))
        rec = json.loads(api['limit_assemble'](
            's1', [second['step']['id'], first['step']['id']]))
        self.assertFalse(rec['ok'])
        self.assertIn('piece 1', rec['error'])

    def test_replay_rejects_tampered_limit_assembly_provenance(self):
        session = DoSession()
        api = make_api(session)
        linearity = json.loads(api['limit_linearity'](
            '\\lim_{x \\to 0} (\\frac{\\sin x}{x}+x^2)'))
        first = json.loads(api['limit_table'](linearity['limits'][0]))
        second = json.loads(api['limit_substitute'](linearity['limits'][1]))
        json.loads(api['limit_assemble'](
            's1', [first['step']['id'], second['step']['id']]))
        session.ledger.steps[-1]['sources']['values'][0] = 's3'
        replay = session.ledger.replay()
        self.assertEqual(replay['status'], 'failed')
        self.assertIn('provenance mismatch', replay['reason'])

    WALLIS_EXPLICIT = '\\lim_{n \\to \\infty} \\prod_{k=1}^{n} ' \
                      '\\frac{2k-1}{2k}'
    WALLIS_UPPER = '\\frac{1}{\\sqrt{2n+1}}'

    def _squeeze_setup(self, api):
        upper = json.loads(api['limit_table'](
            '\\lim_{n \\to \\infty} ' + self.WALLIS_UPPER))
        lower = json.loads(api['limit_table']('\\lim_{n \\to \\infty} 0'))
        return lower['step']['id'], upper['step']['id']

    def test_limit_squeeze_uses_recorded_bound_limits(self):
        session = DoSession()
        api = make_api(session)
        lower_id, upper_id = self._squeeze_setup(api)
        rec = json.loads(api['limit_squeeze'](
            self.WALLIS_EXPLICIT, '0', self.WALLIS_UPPER,
            lower_id, upper_id))
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['result'], '0')
        self.assertEqual(rec['sources'],
                         {'lower': lower_id, 'upper': upper_id})
        self.assertEqual(session.ledger.replay()['status'], 'verified')

    def test_limit_squeeze_rejects_unknown_bound_step(self):
        session = DoSession()
        api = make_api(session)
        rec = json.loads(api['limit_squeeze'](
            self.WALLIS_EXPLICIT, '0', self.WALLIS_UPPER, 's7', 's8'))
        self.assertFalse(rec['ok'])
        self.assertIn('unknown transforming step', rec['error'])

    def test_limit_squeeze_rejects_mismatched_bound_step(self):
        session = DoSession()
        api = make_api(session)
        other = json.loads(api['limit_table'](
            '\\lim_{n \\to \\infty} \\frac{1}{n}'))
        lower = json.loads(api['limit_table']('\\lim_{n \\to \\infty} 0'))
        rec = json.loads(api['limit_squeeze'](
            self.WALLIS_EXPLICIT, '0', self.WALLIS_UPPER,
            lower['step']['id'], other['step']['id']))
        self.assertFalse(rec['ok'])
        self.assertIn('does not record', rec['error'])

    def test_replay_rejects_tampered_squeeze_provenance(self):
        session = DoSession()
        api = make_api(session)
        lower_id, upper_id = self._squeeze_setup(api)
        json.loads(api['limit_squeeze'](
            self.WALLIS_EXPLICIT, '0', self.WALLIS_UPPER,
            lower_id, upper_id))
        session.ledger.steps[-1]['sources']['upper'] = lower_id
        replay = session.ledger.replay()
        self.assertEqual(replay['status'], 'failed')
        self.assertIn('provenance', replay['reason'])

    def test_integrate_assemble_rejects_wrong_source_order(self):
        session = DoSession()
        api = make_api(session)
        linearity = json.loads(api['integrate_linearity'](
            '\\int (x+x^2) \\, d x', 'x'))
        first = json.loads(api['integrate_power_rule'](
            linearity['integrals'][0], 'x'))
        second = json.loads(api['integrate_power_rule'](
            linearity['integrals'][1], 'x'))
        rec = json.loads(api['integrate_assemble'](
            's1', [second['step']['id'], first['step']['id']]))
        self.assertFalse(rec['ok'])
        self.assertIn('piece 1', rec['error'])
        self.assertEqual(len(session.ledger.steps), 3)

    def test_integrate_assemble_requires_linearity_step(self):
        session = DoSession()
        api = make_api(session)
        step = json.loads(api['expand']('(x+1)^2'))['step']['id']
        rec = json.loads(api['integrate_assemble'](step, [step]))
        self.assertFalse(rec['ok'])
        self.assertIn('not integrate_linearity', rec['error'])

    def test_replay_rejects_tampered_assembly_provenance(self):
        session = DoSession()
        api = make_api(session)
        linearity = json.loads(api['integrate_linearity'](
            '\\int (x+x^2) \\, d x', 'x'))
        first = json.loads(api['integrate_power_rule'](
            linearity['integrals'][0], 'x'))
        second = json.loads(api['integrate_power_rule'](
            linearity['integrals'][1], 'x'))
        json.loads(api['integrate_assemble']('s1', [
            first['step']['id'], second['step']['id']]))
        session.ledger.steps[-1]['sources']['antiderivatives'][0] = 's3'
        replay = session.ledger.replay()
        self.assertEqual(replay['status'], 'failed')
        self.assertIn('provenance mismatch', replay['reason'])

    def test_concurrent_recording_is_thread_safe(self):
        # the Agents SDK runs sync tools on a thread pool; parallel tool
        # calls must not produce duplicate step ids
        import threading
        session = DoSession()
        api = make_api(session)

        def work():
            for _ in range(10):
                api['expand']('x + x')

        threads = [threading.Thread(target=work) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        ids = [s['id'] for s in session.new_steps()]
        self.assertEqual(len(ids), 40)
        self.assertEqual(len(set(ids)), 40)

    def test_existing_ledger_range(self):
        ledger = Ledger()
        ledger.record({'ok': True, 'op': 'expand', 'args': {'expr': 'x+x'},
                       'input': 'x+x', 'result': '2x'})
        session = DoSession(ledger=ledger)
        api = make_api(session)
        api['expand']('3x + 3x')
        self.assertEqual(len(ledger.steps), 2)
        self.assertEqual(len(session.new_steps()), 1)


LIM_SUM_EXPR = ('\\lim _{n \\rightarrow \\infty}\\left['
                '\\frac{1}{1 \\cdot 2}+\\frac{1}{2 \\cdot 3}'
                '+\\ldots+\\frac{1}{n(n+1)}\\right]')

LIM_SUM_SCRIPT = [
    [tool_call('load_skill', {'skill': 'finite_operators'}, 'sk1')],
    [tool_call('sum_from_ellipsis',
               {'expr': LIM_SUM_EXPR,
                'sum_form': '\\sum_{k=1}^{n} \\frac{1}{k(k+1)}'}, 'c1')],
    [tool_call('sum_telescope',
               {'expr': ('\\lim_{n \\to \\infty} \\sum_{k=1}^{n} '
                         '\\frac{1}{k(k+1)}'),
                'term': '\\frac{1}{k}'}, 'c2')],
    [tool_call('load_skill', {'skill': 'limits'}, 'sk2')],
    [tool_call('limit_table',
               {'expr': '\\lim_{n \\to \\infty} \\frac{n}{n+1}'}, 'c3')],
    [message('Interpreted the ellipsis, telescoped, closed at 1.')],
]


class TestScriptedAgent(unittest.TestCase):
    def test_derivative_roots_and_plot_finishes_with_solution_result(self):
        shown = []
        script = [
            [tool_call('load_skill', {'skill': 'differentiation'}, 'sk1')],
            [tool_call('diff', {'expr': 'x^3-3x', 'var': 'x'}, 'd1')],
            [tool_call('load_skill', {'skill': 'roots'}, 'sk2')],
            [tool_call('quadratic_roots', {
                'expr': '3x^{2}-3', 'var': 'x'}, 'r1')],
            [tool_call('substitute', {
                'expr': 'x^3-3x', 'var': 'x', 'value': '-1'}, 'p1')],
            [tool_call('evaluate', {
                'expr': '(-1)^3-3(-1)'}, 'p2')],
            [tool_call('substitute', {
                'expr': 'x^3-3x', 'var': 'x', 'value': '1'}, 'p3')],
            [tool_call('evaluate', {'expr': '(1)^3-3(1)'}, 'p4')],
            [tool_call('plot', {
                'code': 'import matplotlib.pyplot as plt; plt.plot([0])',
                'caption': 'stationary points'}, 'plot1')],
            [tool_call('set_result', {
                'expr': r'x=-1 \lor x=1'}, 'done')],
            [message('Mechanically checked the derivative and roots.')],
        ]
        ledger = Ledger()
        res = run_instruction(
            'differentiate x^3-3x, find where the derivative is zero, '
            'plot with those points marked',
            model=ScriptedModel(script), ledger=ledger,
            plot_backend=FakePlotBackend(),
            on_plot=lambda caption, images: shown.append(caption))
        self.assertTrue(res['ok'], res.get('error'))
        self.assertEqual(res['final_result'], r'x=-1 \lor x=1')
        self.assertEqual(res['final_provenance']['step'], 's2')
        self.assertEqual([step['op'] for step in res['steps']], [
            'differentiate', 'quadratic_roots', 'substitute', 'evaluate',
            'substitute', 'evaluate'])
        self.assertTrue(res['steps'][1]['continues'])
        self.assertEqual(shown, ['stationary points'])
        self.assertEqual(ledger.replay()['status'], 'verified')

    def test_ellipsis_series_limit_closes_via_sum_tactics(self):
        res = run_instruction('evaluate ' + LIM_SUM_EXPR,
                              model=ScriptedModel(LIM_SUM_SCRIPT))
        self.assertTrue(res['ok'])
        self.assertEqual([s['op'] for s in res['steps']],
                         ['sum_from_ellipsis', 'sum_telescope',
                          'limit_table'])
        self.assertEqual(res['final_result'], '1')
        self.assertEqual(res['final_provenance']['status'], 'verified')
        # the pattern continuation is honestly conditional
        self.assertTrue(any('\\ldots' in a['text']
                            for a in res['assumptions']))

    def test_full_run(self):
        seen = []
        res = run_instruction('solve 2x + 3 = 7 for x',
                              model=ScriptedModel(SOLVE_SCRIPT),
                              on_step=lambda s: seen.append(s['id']))
        self.assertTrue(res['ok'])
        self.assertEqual([s['op'] for s in res['steps']],
                         ['apply_both_sides', 'expand'])
        self.assertEqual(res['final_result'], '2x = 4')
        self.assertEqual(res['final_provenance']['step'], 's2')
        self.assertEqual(res['summary'], 'Subtracted 3 from both sides.')
        self.assertEqual(seen, ['s1', 's2'])

    def test_set_result_overrides_final(self):
        script = [
            [tool_call('factor_quadratic', {'expr': 'x^2 - 5x + 6',
                                            'var': 'x'}, 'c1')],
            [tool_call('substitute', {'expr': 'x^2 - 5x + 6 = 0',
                                      'var': 'x', 'value': '3'}, 'c2')],
            [tool_call('set_result', {'expr': '(x - 3)(x - 2)'}, 'c3'),
             ],
            [message('factored; root 3 checked')],
        ]
        res = run_instruction('factor and verify',
                              model=ScriptedModel(script))
        self.assertTrue(res['ok'])
        # the designated value wins over the last step's result
        self.assertEqual(res['final_result'], '(x - 3)(x - 2)')
        self.assertEqual(res['final_provenance']['step'], 's1')
        self.assertEqual(len(res['steps']), 2)

    def test_branch_run_returns_spine_and_abandoned_path_summary(self):
        script = [
            [tool_call('expand', {'expr': '(x+1)^2'}, 'c1')],
            [tool_call('substitute', {
                'expr': 'x^{2}+2x+1', 'var': 'x', 'value': '1'}, 'c2')],
            [tool_call('comment', {
                'text': 'the numeric detour does not answer the symbolic goal',
                'from_step': 's1'}, 'c3')],
            [tool_call('factor_quadratic', {
                'expr': 'x^{2}+2x+1', 'var': 'x'}, 'c4')],
            [tool_call('set_result', {'expr': '(x+1)^{2}'}, 'c5')],
            [message('Used the factored symbolic route.')],
        ]
        ledger = Ledger()
        res = run_instruction('explore then factor',
                              model=ScriptedModel(script), ledger=ledger)
        self.assertTrue(res['ok'], res.get('error'))
        self.assertEqual(res['final_provenance']['step'], 's4')
        self.assertEqual(res['branch_topology']['spine'], ['s1', 's4'])
        self.assertEqual(res['abandoned_paths'], [{
            'marker': 's3', 'source': 's1', 'continues_at': 's4',
            'reason': 'the numeric detour does not answer the symbolic goal',
            'steps': ['s2'],
        }])
        self.assertEqual(ledger.selections[-1]['id'], 'r1')
        self.assertEqual(ledger.replay()['status'], 'verified')

    def test_dead_branch_assumption_is_not_a_final_assumption(self):
        script = [
            [tool_call('expand', {'expr': 'xy=1'}, 'c1')],
            [tool_call('apply', {
                'equation': 'xy=1', 'op': '/', 'arg': 'y'}, 'c2')],
            [tool_call('comment', {
                'text': 'division is unnecessary', 'from_step': 's1'}, 'c3')],
            [tool_call('apply', {
                'equation': 'xy=1', 'op': '-', 'arg': '1'}, 'c4')],
            [tool_call('set_result', {'expr': 'xy-1=0'}, 'c5')],
            [message('Kept the assumption-free route.')],
        ]
        ledger = Ledger()
        res = run_instruction('rearrange without division',
                              model=ScriptedModel(script), ledger=ledger)
        self.assertTrue(res['ok'], res.get('error'))
        self.assertTrue(ledger.assumptions)
        self.assertEqual(res['assumptions'], [])
        self.assertEqual(res['abandoned_paths'][0]['steps'], ['s2'])

    def test_set_result_cannot_override_with_detached_value(self):
        script = [
            [tool_call('expand', {'expr': '(x+1)^2'}, 'c1')],
            [tool_call('set_result', {'expr': 'x=999'}, 'c2')],
            [message('done')],
        ]
        res = run_instruction('try to invent a result',
                              model=ScriptedModel(script))
        self.assertEqual(res['final_result'], 'x^{2}+2x+1')
        self.assertEqual(res['final_provenance']['step'], 's1')

    def test_trailing_comment_does_not_eat_the_final(self):
        script = [
            [tool_call('expand', {'expr': '(x+1)^2'}, 'c1')],
            [tool_call('comment', {'text': 'done expanding'}, 'c2')],
            [message('Expanded the square.')],
        ]
        res = run_instruction('expand and annotate',
                              model=ScriptedModel(script))
        self.assertTrue(res['ok'])
        # the note is streamed as a step but the chainable value is the
        # last transforming result
        self.assertEqual([s['op'] for s in res['steps']],
                         ['expand', 'comment'])
        self.assertEqual(res['final_result'], 'x^{2}+2x+1')
        self.assertEqual(res['final_provenance']['step'], 's1')

    def test_max_turns_returns_partial(self):
        endless = [[tool_call('expand', {'expr': 'x + x'}, f'c{i}')]
                   for i in range(10)]
        res = run_instruction('loop forever', model=ScriptedModel(endless),
                              max_turns=2)
        self.assertFalse(res['ok'])
        self.assertIn('turns', res['error'])
        self.assertEqual(len(res['steps']), 2)  # partial work is kept

    def test_prove_mode_closes_root_claim(self):
        script = [
            [tool_call('load_skill', {'skill': 'limits'}, 'sk1')],
            [tool_call('limit_table', {
                'expr': r'\lim_{n \to \infty} \frac{1}{n}'}, 'c1')],
            [tool_call('conclude', {'claim_id': 'c1',
                                    'step_ids': ['s1']}, 'c2')],
            [tool_call('set_result', {'expr': '0'}, 'c3')],
            [message('Mechanically checked.\nThe table closes it.\n'
                     'This extra prose must not dominate.')],
        ]
        res = run_instruction(
            'prove the limit', model=ScriptedModel(script),
            proof_goal=r'\lim_{n \to \infty} \frac{1}{n} = 0')
        self.assertTrue(res['ok'])
        self.assertEqual(res['claims'][0]['verdict'], 'established')
        self.assertEqual(res['steps'][0]['goal'], 'c1')
        self.assertEqual(res['final_result'], '0')
        self.assertEqual(res['final_provenance']['source'], 'claim')
        self.assertTrue(res['summary_unverified'])
        self.assertIn('narrative truncated', res['summary'])

    def test_prove_mode_closes_ellipsis_product_claim(self):
        # the reported cell: prove! \lim (1/2 * 3/4 ... (2n-1)/2n) = 0.
        # The ellipsis claim records, the product door interprets it, the
        # bound limits close by table rules, and the squeeze concludes.
        goal = ('\\lim _{n \\rightarrow \\infty}\\left(\\frac{1}{2} \\cdot '
                '\\frac{3}{4} \\ldots \\frac{2 n-1}{2 n}\\right) = 0')
        expr = ('\\lim _{n \\rightarrow \\infty}\\left(\\frac{1}{2} \\cdot '
                '\\frac{3}{4} \\ldots \\frac{2 n-1}{2 n}\\right)')
        explicit = '\\lim_{n \\to \\infty} \\prod_{k=1}^{n} \\frac{2k-1}{2k}'
        upper = '\\frac{1}{\\sqrt{2n+1}}'
        script = [
            [tool_call('load_skill', {'skill': 'finite_operators'}, 'sk1')],
            [tool_call('prod_from_ellipsis', {
                'expr': expr,
                'prod_form': '\\prod_{k=1}^{n} \\frac{2k-1}{2k}'}, 'c1')],
            [tool_call('load_skill', {'skill': 'limits'}, 'sk2')],
            [tool_call('limit_table', {
                'expr': '\\lim_{n \\to \\infty} 0'}, 'c2')],
            [tool_call('limit_table', {
                'expr': '\\lim_{n \\to \\infty} ' + upper}, 'c3')],
            [tool_call('limit_squeeze', {
                'expr': explicit, 'lower': '0', 'upper': upper,
                'lower_step': 's2', 'upper_step': 's3'}, 'c4')],
            [tool_call('conclude', {'claim_id': 'c1',
                                    'step_ids': ['s1', 's4']}, 'c5')],
            [tool_call('set_result', {'expr': '0'}, 'c6')],
            [message('Interpreted the product, squeezed it to 0.')],
        ]
        ledger = Ledger()
        res = run_instruction('prove the product limit', ledger=ledger,
                              model=ScriptedModel(script), proof_goal=goal)
        self.assertTrue(res['ok'], res.get('error'))
        self.assertEqual([s['op'] for s in res['steps']],
                         ['prod_from_ellipsis', 'limit_table',
                          'limit_table', 'limit_squeeze'])
        # conditional: the ellipsis reading and the ordering are assumptions
        self.assertEqual(res['claims'][0]['verdict'], 'conditional')
        self.assertEqual(res['final_result'], '0')
        self.assertEqual(res['final_provenance']['source'], 'claim')
        # the full session, ellipsis root claim included, must replay
        rep = ledger.replay()
        self.assertEqual(rep['status'], 'verified', rep.get('reason'))

    def test_prove_mode_reclaim_refocuses_root_and_statement_sets_result(
            self):
        # The live n/2^n trap: after a first conditional closure the agent
        # re-stated the root claim (minting c2, which stole the goal focus
        # of every later step, so conclude(c1) could never succeed again)
        # and then passed the claim STATEMENT to set_result (refused; the
        # selection wants the endpoint). Both moves must now work: the
        # re-claim focuses c1, a repeated conclude replaces the chain, and
        # the statement maps to the concluded endpoint.
        goal = r'\lim_{n \to \infty} \frac{n}{2^n} = 0'
        start = r'\lim_{n \to \infty} \frac{n}{2^n}'
        transformed = (r'\lim_{n \to \infty} \frac{1}'
                       r'{({2}^n) \ln\left (2 \right )}')
        script = [
            [tool_call('load_skill', {'skill': 'limits'}, 'sk1')],
            [tool_call('limit_lhopital', {'expr': start}, 'c1')],
            [tool_call('limit_table', {'expr': transformed}, 'c2')],
            [tool_call('conclude', {'claim_id': 'c1',
                                    'step_ids': ['s1', 's2']}, 'c3')],
            # the agent re-states the claim hunting for a better chain
            [tool_call('claim', {'statement': goal}, 'c4')],
            [tool_call('limit_lhopital', {'expr': start}, 'c5')],
            [tool_call('limit_table', {'expr': transformed}, 'c6')],
            [tool_call('conclude', {'claim_id': 'c1',
                                    'step_ids': ['s3', 's4']}, 'c7')],
            [tool_call('set_result', {'expr': goal}, 'c8')],
            [message('Mechanically checked via l\'Hopital.')],
        ]
        ledger = Ledger()
        res = run_instruction('prove the limit', ledger=ledger,
                              model=ScriptedModel(script), proof_goal=goal)
        self.assertTrue(res['ok'], res.get('error'))
        # the re-claim reused c1: one claim, and the late steps kept its goal
        self.assertEqual(len(res['claims']), 1)
        self.assertEqual([s.get('goal') for s in res['steps']
                          if s.get('result') is not None],
                         ['c1', 'c1', 'c1', 'c1'])
        self.assertEqual(res['claims'][0]['verdict'], 'conditional')
        self.assertEqual(res['claims'][0]['conclusion']['steps'],
                         ['s3', 's4'])
        # the statement spelling designated the concluded endpoint
        self.assertEqual(res['final_result'], '0')
        self.assertEqual(res['final_provenance']['source'], 'claim')
        rep = ledger.replay()
        self.assertEqual(rep['status'], 'verified', rep.get('reason'))

    def test_prove_mode_leaves_failed_target_open(self):
        script = [
            [tool_call('load_skill', {'skill': 'limits'}, 'sk1')],
            [tool_call('limit_table', {
                'expr': r'\lim_{n \to \infty} \frac{1}{n}'}, 'c1')],
            [tool_call('set_result', {'expr': '0'}, 'c2')],
            [message('A prose squeeze argument would go here.')],
        ]
        res = run_instruction(
            'try the proof', model=ScriptedModel(script),
            proof_goal=r'\lim_{n \to \infty} \frac{n}{2^n} = 0')
        self.assertTrue(res['ok'])
        self.assertEqual(res['claims'][0]['verdict'], 'open')
        self.assertIsNone(res['final_result'])
        self.assertEqual(res['final_provenance']['status'], 'open')

    def test_open_claim_caps_plain_do_summary(self):
        # even outside prove!, prose must not dominate an open claim
        script = [
            [tool_call('claim', {
                'statement': r'\lim_{n \to \infty} \frac{n}{2^n} = 0'},
                'c1')],
            [message('An informal argument follows.\n'
                     '\\[\n2^n \\ge n^2 \\text{ for } n \\ge 4\n\\]\n'
                     'Therefore the sequence is squeezed to zero, which '
                     'this prose must not be allowed to assert.')],
        ]
        res = run_instruction('prove informally',
                              model=ScriptedModel(script))
        self.assertTrue(res['ok'])
        self.assertEqual(res['claims'][0]['verdict'], 'open')
        self.assertTrue(res['summary_unverified'])
        self.assertIn('narrative truncated', res['summary'])
        self.assertNotIn('\\[', res['summary'])

    def test_cap_prefers_sentence_boundary_and_strips_math(self):
        from agent_do import _cap_prove_summary
        capped = _cap_prove_summary(
            'The claim remains open. Mechanically checked, the chain '
            'reached\n\\[\n\\lim_{n \\to \\infty} e^{-n}\n\\]')
        self.assertIn('narrative truncated', capped)
        self.assertNotIn('\\[', capped)
        long_prose = ('This sentence is padded out to be long enough to '
                      'trigger the character cap on its own. ' * 5)
        capped = _cap_prove_summary(long_prose)
        self.assertIn('narrative truncated', capped)
        self.assertTrue(capped.split(' … ')[0].endswith('.'))

    def test_notebook_ledger_accumulates(self):
        ledger = Ledger()
        run_instruction('solve', model=ScriptedModel(SOLVE_SCRIPT),
                        ledger=ledger)
        res = run_instruction(
            'again',
            model=ScriptedModel([
                [tool_call('expand', {'expr': '2x = 4'}, 'c9')],
                [message('done')]]),
            ledger=ledger)
        self.assertEqual(len(ledger.steps), 3)
        self.assertEqual(len(res['steps']), 1)  # only this run's slice


class TestObservability(unittest.TestCase):
    """Langfuse tracing is opt-in, non-fatal, and never touches the ledger.

    Runs fully offline: a Langfuse client backed by an in-memory OTEL
    exporter captures spans without any network or credentials.
    """

    def test_env_toggle_parsing(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(observability.ENABLE_VAR, None)
            self.assertFalse(observability.is_enabled())
        for on in ('1', 'true', 'on', 'yes', 'ON', 'Enabled'):
            with mock.patch.dict(os.environ,
                                 {observability.ENABLE_VAR: on}):
                self.assertTrue(observability.is_enabled(), on)
        for off in ('0', 'false', 'off', 'no', '', 'maybe'):
            with mock.patch.dict(os.environ,
                                 {observability.ENABLE_VAR: off}):
                self.assertFalse(observability.is_enabled(), off)

    def test_disabled_is_a_noop_and_run_is_unaffected(self):
        observability._reset_for_tests()
        self.addCleanup(observability._reset_for_tests)
        with mock.patch.dict(os.environ, {observability.ENABLE_VAR: 'off'}):
            self.assertFalse(observability.setup())
        self.assertFalse(observability.active())
        with observability.trace_run('anything') as span:
            self.assertIsNone(span)  # a no-op context manager
        observability.set_output(None, 'ignored')  # must not raise
        observability.flush()  # must not raise
        res = run_instruction(
            'solve 2x + 3 = 7 for x',
            model=ScriptedModel([list(t) for t in SOLVE_SCRIPT]))
        self.assertTrue(res['ok'])
        self.assertEqual(res['final_result'], '2x = 4')

    def test_build_model_tracing_toggle_follows_observability(self):
        # THE landmine: the OpenInference instrumentor rides the Agents-SDK
        # tracing pipeline, so an active observability must leave that
        # pipeline ENABLED; an inactive one disables it (no OpenAI upload).
        for is_active, expected_disabled in [(True, False), (False, True)]:
            calls = []
            with mock.patch.dict(os.environ, {'OPEN_ROUTER': 'sk-test'}), \
                 mock.patch('agents.set_tracing_disabled',
                            side_effect=calls.append), \
                 mock.patch.object(observability, 'active',
                                   return_value=is_active):
                agent_do.build_model()
            self.assertEqual(calls, [expected_disabled], is_active)

    def test_build_model_accepts_notebook_override(self):
        with mock.patch.dict(os.environ, {'OPEN_ROUTER': 'sk-test'}), \
             mock.patch.object(observability, 'active', return_value=False):
            built = agent_do.build_model('z-ai/glm-5.2')
        self.assertEqual(built.model, 'z-ai/glm-5.2')

    def test_provider_order_reaches_model_settings_extra_body(self):
        settings = []

        class RecordingModel(ScriptedModel):
            async def get_response(self, system_instructions, input,
                                   model_settings, *args, **kwargs):
                settings.append(model_settings)
                return await super().get_response(
                    system_instructions, input, model_settings,
                    *args, **kwargs)

        res = run_instruction(
            'say hello', model=RecordingModel([[message('hello')]]),
            providers=('Cerebras', 'Fireworks'))
        self.assertTrue(res['ok'])
        self.assertEqual(settings[0].extra_body, {
            'provider': {
                'order': ['Cerebras', 'Fireworks'],
                'allow_fallbacks': False,
            },
        })

    def test_active_run_emits_one_nested_langfuse_trace(self):
        from agents import set_tracing_disabled
        from openinference.instrumentation.openai_agents import (
            OpenAIAgentsInstrumentor)
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter)
        from langfuse import Langfuse

        exporter = InMemorySpanExporter()
        lf = Langfuse(public_key='pk-lf-test', secret_key='sk-lf-test',
                      tracing_enabled=True, span_exporter=exporter)
        set_tracing_disabled(False)
        observability._reset_for_tests()

        def cleanup():
            OpenAIAgentsInstrumentor().uninstrument()
            observability._reset_for_tests()
            set_tracing_disabled(True)  # leave the process quiet again
            lf.shutdown()
        self.addCleanup(cleanup)

        self.assertTrue(observability.setup(client=lf))
        self.assertTrue(observability.active())

        ledger = Ledger()
        res = run_instruction(
            'solve 2x + 3 = 7 for x',
            model=ScriptedModel([list(t) for t in SOLVE_SCRIPT]),
            ledger=ledger)
        self.assertTrue(res['ok'], res.get('error'))
        lf.flush()

        spans = exporter.get_finished_spans()
        self.assertTrue(spans, 'no spans were exported')
        # one trace for the whole run, rooted at our wrapping observation
        self.assertEqual(len({s.context.trace_id for s in spans}), 1)
        root = next(s for s in spans if s.parent is None)
        self.assertEqual(root.name, 'do!')
        attrs = dict(root.attributes)
        self.assertEqual(attrs.get('langfuse.observation.input'),
                         'solve 2x + 3 = 7 for x')
        self.assertEqual(attrs.get('langfuse.observation.metadata.mode'),
                         'do')
        self.assertEqual(attrs.get('langfuse.observation.type'), 'agent')
        # the trusted-primitive calls appear as nested tool spans
        kinds = {s.attributes.get('openinference.span.kind') for s in spans}
        self.assertIn('TOOL', kinds)
        # tracing is observability only: the ledger is exactly what it would
        # be without it
        self.assertEqual([s['op'] for s in res['steps']],
                         ['apply_both_sides', 'expand'])
        self.assertEqual(ledger.replay()['status'], 'verified')


FAKE_PNG_B64 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAAAAAA6fptVAAAACklEQVR4nGNiAAAABgADNjd8qAAAAABJRU5ErkJggg=='


FAKE_SVG = '<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0"/></svg>'


class FakePlotBackend(object):
    name = 'fake'

    def __init__(self, result=None):
        self.calls = []
        # the default deliberately speaks the older images-only dialect,
        # so the typed-figure fallback stays covered
        self.result = result or {'ok': True, 'stdout': 'drew it\n',
                                 'stderr': '', 'images': [FAKE_PNG_B64]}

    def run_plot(self, code, timeout=None):
        self.calls.append(code)
        return dict(self.result)


class FakeTikzBackend(object):
    name = 'fake-tikz'

    def __init__(self, result=None):
        self.calls = []
        self.result = result or {'ok': True, 'svg': FAKE_SVG}

    def render(self, code, timeout=None):
        self.calls.append(code)
        return dict(self.result)


class TestPlotTool(unittest.TestCase):
    def test_registered_only_with_backend(self):
        self.assertNotIn('plot', make_api(DoSession()))
        session = DoSession(plot_backend=FakePlotBackend())
        self.assertIn('plot', make_api(session))

    def test_prompt_mentions_plotting_only_when_available(self):
        self.assertNotIn('## Plotting', agent_do.build_prompt())
        self.assertIn('## Plotting', agent_do.build_prompt(plotting=True))

    def test_plot_streams_images_not_tokens(self):
        shown = []
        backend = FakePlotBackend()
        session = DoSession(plot_backend=backend,
                            on_plot=lambda cap, figs: shown.append(
                                (cap, figs)))
        api = make_api(session)
        reply = json.loads(api['plot']('plt.plot([1])', 'a parabola'))
        # the model sees counts and stdout, never image bytes
        self.assertTrue(reply['ok'])
        self.assertEqual(reply['plots'], 1)
        self.assertNotIn(FAKE_PNG_B64[:24], json.dumps(reply))
        # the user sees the figure with its caption; an images-only
        # backend is adapted up into the typed shape
        self.assertEqual(shown[0][0], 'a parabola')
        self.assertEqual(shown[0][1], [{'kind': 'png', 'data': FAKE_PNG_B64}])
        # never a ledger step
        self.assertEqual(session.new_steps(), [])

    def test_plot_streams_typed_figures(self):
        shown = []
        backend = FakePlotBackend({
            'ok': True, 'stdout': '', 'stderr': '',
            'figures': [{'kind': 'html', 'data': '<html>fig</html>',
                         'height': 400}]})
        session = DoSession(plot_backend=backend,
                            on_plot=lambda cap, figs: shown.append(figs))
        reply = json.loads(make_api(session)['plot']('fig = go.Figure()',
                                                     'interactive'))
        self.assertTrue(reply['ok'])
        self.assertEqual(reply['plots'], 1)
        self.assertEqual(shown[0][0]['kind'], 'html')
        self.assertEqual(shown[0][0]['height'], 400)
        # figure bytes never reach the model
        self.assertNotIn('<html>', json.dumps(reply))

    def test_plot_failure_reported_in_band(self):
        backend = FakePlotBackend({'ok': False, 'images': [],
                                   'error': 'NameError: nope'})
        session = DoSession(plot_backend=backend)
        reply = json.loads(make_api(session)['plot']('bad', 'cap'))
        self.assertFalse(reply['ok'])
        self.assertIn('NameError', reply['error'])

    def test_no_figure_is_an_error(self):
        backend = FakePlotBackend({'ok': True, 'stdout': '', 'stderr': '',
                                   'images': []})
        session = DoSession(plot_backend=backend)
        reply = json.loads(make_api(session)['plot']('print(1)', 'cap'))
        self.assertFalse(reply['ok'])
        self.assertIn('no figure', reply['error'])

    def test_scripted_agent_with_plot(self):
        shown = []
        script = [
            [tool_call('expand', {'expr': '(x-2)(x-3)'}, 'c1')],
            [tool_call('plot', {'code': 'plt.plot([1])',
                                'caption': 'the parabola'}, 'c2')],
            [message('done')],
        ]
        res = run_instruction('expand and plot',
                              model=ScriptedModel(script),
                              plot_backend=FakePlotBackend(),
                              on_plot=lambda cap, imgs: shown.append(cap))
        self.assertTrue(res['ok'])
        self.assertEqual(len(res['steps']), 1)  # plot is not a step
        self.assertEqual(shown, ['the parabola'])

    def test_get_backend_off(self):
        with mock.patch.dict(os.environ, {'TOYMATH_SANDBOX': 'off'}):
            self.assertIsNone(plot_sandbox.get_backend())
            self.assertIsNone(plot_sandbox.get_tikz_backend())

    def test_parse_runner_output(self):
        good = plot_sandbox._parse_runner_output(
            'noise\n{"ok": true, "images": []}\n', '')
        self.assertTrue(good['ok'])
        bad = plot_sandbox._parse_runner_output('garbage only', 'boom')
        self.assertFalse(bad['ok'])
        self.assertIn('boom', bad['stderr'])

    def test_child_env_scrubs_secrets(self):
        with mock.patch.dict(os.environ, {'OPEN_ROUTER': 'sk-secret',
                                          'AWS_SECRET_ACCESS_KEY': 'nope',
                                          'PATH': '/usr/bin'}):
            env = plot_sandbox._child_env()
        self.assertNotIn('OPEN_ROUTER', env)
        self.assertNotIn('AWS_SECRET_ACCESS_KEY', env)
        self.assertEqual(env['PATH'], '/usr/bin')


class TestTikzTool(unittest.TestCase):
    def test_registered_only_with_backend(self):
        self.assertNotIn('tikz', make_api(DoSession()))
        session = DoSession(tikz_backend=FakeTikzBackend())
        self.assertIn('tikz', make_api(session))

    def test_prompt_mentions_tikz_only_when_available(self):
        self.assertNotIn('## TikZ', agent_do.build_prompt())
        self.assertIn('## TikZ', agent_do.build_prompt(tikz=True))

    def test_tikz_streams_svg_not_tokens(self):
        shown = []
        session = DoSession(tikz_backend=FakeTikzBackend(),
                            on_plot=lambda cap, figs: shown.append(
                                (cap, figs)))
        reply = json.loads(make_api(session)['tikz'](
            r'\begin{document}\end{document}', 'a diagram'))
        self.assertTrue(reply['ok'])
        self.assertEqual(reply['plots'], 1)
        self.assertNotIn('<svg', json.dumps(reply))
        self.assertEqual(shown[0][0], 'a diagram')
        self.assertEqual(shown[0][1], [{'kind': 'svg', 'data': FAKE_SVG}])
        # illustrations are never evidence
        self.assertEqual(session.new_steps(), [])

    def test_tikz_failure_returns_tex_log(self):
        backend = FakeTikzBackend({
            'ok': False,
            'error': 'TikZ render failed: TeX engine render failed.\n'
                     'TeX log:\n! Undefined control sequence.\n'
                     'l.4 \\draw (0,0) circle (\\nosuchmacro)'})
        session = DoSession(tikz_backend=backend)
        reply = json.loads(make_api(session)['tikz']('bad', 'cap'))
        self.assertFalse(reply['ok'])
        # the agent can only fix its source if the TeX error survives
        self.assertIn('Undefined control sequence', reply['error'])


@unittest.skipUnless(os.environ.get('TOYMATH_PLOT_TESTS') == '1',
                     'set TOYMATH_PLOT_TESTS=1 for a live deno/pyodide '
                     'sandbox test')
class TestLivePlotSandbox(unittest.TestCase):
    def test_matplotlib_figure_roundtrip(self):
        import base64
        backend = plot_sandbox.get_backend()
        self.assertIsNotNone(backend, 'deno not available')
        r = backend.run_plot(
            'import numpy as np\nimport matplotlib.pyplot as plt\n'
            'x = np.linspace(0, 1, 50)\nplt.plot(x, x*x)\n')
        self.assertTrue(r['ok'], r.get('error'))
        self.assertEqual(len(r['images']), 1)
        png = base64.b64decode(r['images'][0])
        self.assertEqual(png[:8], b'\x89PNG\r\n\x1a\n')

    def test_escape_probes_denied(self):
        backend = plot_sandbox.get_backend()
        for probe in ("import js\njs.Deno.env.get('OPEN_ROUTER')",
                      "import js\njs.Deno.readTextFileSync('/etc/hosts')"):
            r = backend.run_plot(probe)
            self.assertFalse(r['ok'])
            self.assertTrue('NotCapable' in (r.get('error') or '')
                            or 'PermissionDenied' in (r.get('error') or ''))

    def test_submodule_import_of_uninstalled_package(self):
        """find_imports reports 'seaborn.objects' alongside 'seaborn', and
        find_spec raises on the dotted name while the parent is absent.
        That used to kill the sandbox before the user code ever ran."""
        backend = plot_sandbox.get_backend()
        r = backend.run_plot(
            'import seaborn.objects as so\nimport matplotlib.pyplot as plt\n'
            'plt.plot([1, 2])\n')
        self.assertTrue(r['ok'], r.get('error'))
        self.assertEqual([f['kind'] for f in r['figures']], ['png'])

    def test_bundled_submodule_still_resolves(self):
        backend = plot_sandbox.get_backend()
        r = backend.run_plot(
            'from scipy.integrate import quad\n'
            'import matplotlib.pyplot as plt\n'
            'v, _ = quad(lambda t: t**2, 0, 3)\nprint(round(v, 6))\n'
            'plt.plot([0, v])\n')
        self.assertTrue(r['ok'], r.get('error'))
        self.assertIn('9.0', r['stdout'])

    def test_plotly_figure_becomes_html(self):
        backend = plot_sandbox.get_backend()
        r = backend.run_plot(
            'import plotly.graph_objects as go\n'
            'fig = go.Figure(data=[go.Scatter(x=[1, 2], y=[1, 4])])\n'
            'fig.update_layout(height=400)\n')
        self.assertTrue(r['ok'], r.get('error'))
        self.assertEqual([f['kind'] for f in r['figures']], ['html'])
        self.assertEqual(r['images'], [])  # plotly cannot rasterise here
        self.assertEqual(r['figures'][0]['height'], 440)


@unittest.skipUnless(os.environ.get('TOYMATH_PLOT_TESTS') == '1',
                     'set TOYMATH_PLOT_TESTS=1 for a live deno/tikzjax '
                     'test')
class TestLiveTikzSandbox(unittest.TestCase):
    def test_tikz_renders_self_contained_svg(self):
        backend = plot_sandbox.get_tikz_backend()
        self.assertIsNotNone(backend, 'deno not available')
        r = backend.render(
            r'\usepackage{pgfplots}'
            '\n\\begin{document}\n\\begin{tikzpicture}\n'
            r'\begin{axis}[width=6cm]\addplot[domain=-2:2]{x^2};\end{axis}'
            '\n\\end{tikzpicture}\n\\end{document}')
        self.assertTrue(r['ok'], r.get('error'))
        svg = r['svg']
        self.assertTrue(svg.lstrip().startswith('<svg'))
        # fonts are inlined, so the figure still renders with no network
        self.assertIn('@font-face', svg)
        self.assertIn('data:font/ttf;base64', svg)
        self.assertNotIn('<script', svg.lower())

    def test_tikz_error_carries_the_tex_log(self):
        backend = plot_sandbox.get_tikz_backend()
        r = backend.render(r'\begin{document}\begin{tikzpicture}'
                           r'\draw (0,0) circle (\nosuchmacro);'
                           r'\end{tikzpicture}\end{document}')
        self.assertFalse(r['ok'])
        self.assertIn('Undefined control sequence', r['error'])


class TestFigureHtml(unittest.TestCase):
    def setUp(self):
        from mathShell import MathShell
        self.render = MathShell._figure_html

    def test_png_inlined_as_data_uri(self):
        html = self.render({'kind': 'png', 'data': FAKE_PNG_B64})
        self.assertIn(f'src="data:image/png;base64,{FAKE_PNG_B64}"', html)

    def test_svg_dropped_in_and_hidden_from_mathjax(self):
        html = self.render({'kind': 'svg', 'data': FAKE_SVG})
        self.assertIn(FAKE_SVG, html)       # inert markup, no iframe needed
        self.assertIn('tex2jax_ignore', html)

    def test_html_iframed_and_escaped(self):
        # JupyterLab strips <script> from cell output, so plotly only runs
        # inside an iframe - and srcdoc must be attribute-escaped or the
        # figure's own quotes break out of it
        html = self.render({'kind': 'html', 'height': 400,
                            'data': '<p class="x">hi</p>'})
        self.assertIn('<iframe', html)
        self.assertIn('height:400px', html)
        self.assertIn('sandbox="allow-scripts"', html)
        self.assertIn('&lt;p class=&quot;x&quot;&gt;', html)
        self.assertNotIn('<p class="x">', html)

    def test_unknown_kind_falls_back_to_png(self):
        self.assertIn('<img', self.render({'data': FAKE_PNG_B64}))


class TestMathShellDo(unittest.TestCase):
    def setUp(self):
        import engine
        self.displays = []
        engine.setHandler(lambda *objs, **kw: self.displays.extend(objs))
        from mathShell import MathShell
        self.shell = MathShell()

    def tearDown(self):
        import engine
        import IPython.display
        engine.setHandler(IPython.display.display)

    def _html(self):
        return ''.join(getattr(d, 'data', str(d)) for d in self.displays)

    def test_assumption_html_prose_and_math(self):
        mixed = self.shell._assumption_html(
            {'text': 'x \\ne 0 in that neighborhood',
             'display': '$x \\ne 0$ in that <neighborhood>'})
        self.assertEqual(mixed,
                         '$x \\ne 0$ in that &lt;neighborhood&gt;')
        bare = self.shell._assumption_html({'text': 'x > 0'})
        self.assertEqual(bare, '$x > 0$')

    def test_resolve_backrefs(self):
        self.shell.exec('2 + 3', 1, add_to_history=True)
        text = self.shell.resolve_backrefs('check [[1]] please')
        self.assertIn('5', text)
        with self.assertRaises(ValueError):
            self.shell.resolve_backrefs('[[99]]')

    def test_backref_survives_later_parses(self):
        # parser.parse() clears parsedNotation in place; the stored history
        # snapshot must not be affected (regression: [[n]] printed _nNN)
        sym = self.shell.parser.parse('x^{2}+1')
        self.shell.output(sym, self.shell.parsedNotation, 7, True)
        self.shell.parser.parse('y')
        text = self.shell.resolve_backrefs('[[7]]')
        self.assertNotIn('textit', text)
        self.assertIn('x^', text)

    def test_backref_of_fractional_power_reparses(self):
        # \int dx/(x^{1/2}+x^{1/3}) stored, then referenced from a
        # composite cell: the re-rendered LaTeX must parse (regression:
        # the writer emitted x^\frac{1}{2}, syntax error on int! [[n]])
        import primitives
        self.shell.exec('\\int \\frac {dx} {(x^{\\frac 1 2} '
                        '+ x^{\\frac 1 3})}', 3, add_to_history=True)
        resolved = self.shell.resolve_backrefs('[[3]]')
        primitives.parse_latex(resolved)

    def test_do_cell_streams_and_chains(self):
        with mock.patch.object(agent_do, 'build_model',
                               lambda model_name=None: ScriptedModel(
                                   SOLVE_SCRIPT)):
            self.shell.exec('do! solve 2x + 3 = 7 for x', 2,
                            add_to_history=True)
        out = self._html()
        self.assertIn('s1#', out)
        self.assertIn('s2#', out)
        self.assertIn('Subtracted 3', out)
        # the harness renders the end-of-run chain table from the ledger
        self.assertIn('verified chain', out)
        # the final result is chainable from later cells
        self.assertIn('2', self.shell.resolve_backrefs('[[2]]'))
        self.assertEqual(len(self.shell.ledger.steps), 2)

    def test_do_missing_backref_fails_fast(self):
        called = []
        with mock.patch.object(agent_do, 'run_instruction',
                               lambda *a, **k: called.append(1)):
            self.shell.exec('do! solve [[42]]', 3, add_to_history=True)
        self.assertEqual(called, [])  # no agent run, no tokens
        self.assertIn('do! error', self._html())

    def test_do_missing_key_reports_cleanly(self):
        env = {k: v for k, v in os.environ.items()
               if k != agent_do.API_KEY_VAR}
        with mock.patch.dict(os.environ, env, clear=True):
            self.shell.exec('do! solve 2x = 4', 4, add_to_history=True)
        self.assertIn(agent_do.API_KEY_VAR, self._html())

    def _chain_step(self, sid, op, result, check='agree', continues=True,
                    assumptions=()):
        return {'id': sid, 'hash': 'h' + sid, 'op': op,
                'args': {}, 'input': 'x', 'result': result,
                'check': {'status': check}, 'continues': continues,
                'assumptions': list(assumptions)}

    def test_chain_table_renders_from_step_records(self):
        html = self.shell.render_do_chain([
            self._chain_step('s1', 'expand', 'x^{2}+2x+1'),
            self._chain_step('s2', 'factor_quadratic', '(x+1)^{2}',
                             assumptions=[{'text': 'x \\ne 0'}]),
            self._chain_step('s3', 'sum_telescope', '\\frac{n}{n+1}',
                             check='skipped', continues=False),
        ])
        self.assertIn('verified chain', html)
        for frag in ('<code>s1</code>', '<code>expand</code>',
                     '$x^{2}+2x+1$', 'new chain; no marker',
                     '+1 assum.'):
            self.assertIn(frag, html)
        self.assertNotIn('(branch)', html)
        # check-status colors: agree green, skipped grey
        self.assertIn('#176b2c', html)
        self.assertIn('#888', html)

    def test_chain_table_skips_comments_and_short_runs(self):
        comment = {'id': 's1', 'hash': 'h1', 'op': 'comment',
                   'args': {'text': 'strategy note'}, 'input': None,
                   'result': None, 'check': {'status': 'note'},
                   'assumptions': []}
        branch = {'id': 's2', 'hash': 'h2', 'op': 'branch',
                  'args': {'from': 's1', 'reason': 'try another route'},
                  'input': None, 'result': None,
                  'check': {'status': 'note'}, 'assumptions': []}
        self.assertIsNone(self.shell.render_do_chain([]))
        self.assertIsNone(self.shell.render_do_chain(
            [comment, branch, self._chain_step('s3', 'expand', '5')]))

    def test_branch_marker_stream_render_is_plain_annotation(self):
        html = self.shell.render_do_step({
            'id': 's3', 'hash': 'h3', 'op': 'branch',
            'args': {'from': 's1',
                     'reason': 'substitution < increased complexity'},
            'input': None, 'result': None, 'continues': None,
            'check': {'status': 'note'}, 'assumptions': [],
        })
        self.assertIn('tex2jax_ignore', html)
        self.assertIn('branch from s1', html)
        self.assertIn('substitution &lt; increased complexity', html)
        self.assertNotIn('$', html)

    def test_chain_table_folds_abandoned_path_with_marker_reason(self):
        first = self._chain_step('s1', 'expand', 'x^{2}+2x+1',
                                 continues=None)
        dead = self._chain_step('s2', 'substitute', '4')
        resumed = self._chain_step('s4', 'factor_quadratic', '(x+1)^{2}',
                                   continues=False)
        resumed['exploration'] = {
            'marker': 's3', 'from': 's1',
            'reason': 'numeric < detour', 'hash': 'edge',
        }
        topology = {'abandoned_paths': [{
            'marker': 's3', 'source': 's1', 'continues_at': 's4',
            'reason': 'numeric < detour', 'steps': ['s2'],
        }]}
        html = self.shell.render_do_chain([first, dead, resumed], topology)
        self.assertIn('<details', html)
        self.assertIn('abandoned path from s1', html)
        self.assertIn('numeric &lt; detour', html)
        self.assertIn('<code>s2</code>', html)  # expandable body retained
        self.assertIn('resumed from s1 via s3', html)
        self.assertNotIn('(branch)', html)

    def test_chain_table_can_expand_prior_run_steps_named_by_new_marker(self):
        first = self._chain_step('s1', 'expand', 'x^{2}+2x+1',
                                 continues=None)
        dead = self._chain_step('s2', 'substitute', '4')
        resumed = self._chain_step('s4', 'factor_quadratic', '(x+1)^{2}',
                                   continues=False)
        resumed['exploration'] = {
            'marker': 's3', 'from': 's1', 'reason': 'old detour',
            'hash': 'edge',
        }
        topology = {'abandoned_paths': [{
            'marker': 's3', 'source': 's1', 'continues_at': 's4',
            'reason': 'old detour', 'steps': ['s2'],
        }]}
        html = self.shell.render_do_chain(
            [resumed], topology, all_steps=[first, dead, resumed])
        self.assertIn('old detour', html)
        self.assertIn('<code>s2</code>', html)
        self.assertIn('<code>s4</code>', html)

    def test_query_only_final_is_rendered_unverified(self):
        result = {'ok': True, 'steps': [], 'assumptions': [],
                  'final_result': 'x=999', 'summary': None,
                  'final_provenance': {
                      'status': 'unverified', 'source': 'query-only',
                      'reason': 'no transforming step was recorded'}}
        with mock.patch.object(agent_do, 'run_instruction',
                               lambda *a, **k: result):
            self.shell.exec('do! query only', 5, add_to_history=True)
        self.assertIn('unverified final value', self._html())
        self.assertIn('999', self.shell.resolve_backrefs('[[5]]'))


class TestPromptCommandModel(unittest.TestCase):
    """The discovery/parse model in prompt_commands.py (no shell, no agent)."""

    def test_repo_commands_load(self):
        import prompt_commands as pc
        reg = pc.load_commands()
        for name in ('int', 'diff', 'solve', 'expand', 'prove', 'conv'):
            self.assertIn(name, reg)
        self.assertIsNone(reg['int'].direct)      # LLM tactic tier
        self.assertEqual(reg['diff'].direct, 'differentiate')
        self.assertEqual(reg['expand'].direct, 'expand')
        self.assertTrue(reg['diff'].expr)         # direct implies expr
        self.assertEqual(reg['prove'].mode, 'prove')
        # conv! is a plain agent command: the verdict must live in the
        # recorded series_converges step, never in an expr splice
        self.assertFalse(reg['conv'].expr)
        self.assertIsNone(reg['conv'].direct)

    def test_repo_model_endpoint_config(self):
        import model_config
        endpoints = model_config.load_model_config()
        glm = model_config.find_model(endpoints, 'z-ai/glm-5.2')
        self.assertIsNotNone(glm)
        self.assertEqual(glm.providers, ('Cerebras', 'Fireworks'))

    def test_model_endpoint_config_validation(self):
        import model_config
        with self.assertRaises(model_config.ModelConfigError):
            model_config.parse_model_config('models: []')
        with self.assertRaises(model_config.ModelConfigError):
            model_config.parse_model_config(
                'models:\n  - model: a\n    providers: Cerebras')
        with self.assertRaises(model_config.ModelConfigError):
            model_config.parse_model_config(
                'models:\n  - model: a\n  - model: a')

    def test_model_command_completion_replaces_only_current_token(self):
        import model_config
        endpoints = (
            model_config.ModelEndpoint('alpha/one', ()),
            model_config.ModelEndpoint('beta/two', ('Fast', 'Safe')),
        )
        code = 'model! be'
        reply = model_config.complete_model_command(
            code, len(code), endpoints=endpoints)
        self.assertEqual(reply['matches'], ['alpha/one', 'beta/two'])
        self.assertEqual(code[reply['cursor_start']:reply['cursor_end']],
                         'be')
        self.assertEqual(
            [item['type'] for item in
             reply['metadata']['_jupyter_types_experimental']],
            ['model', 'model'])

    def test_provider_completion_uses_selected_model_and_skips_used(self):
        import model_config
        endpoints = (
            model_config.ModelEndpoint('beta/two', ('Fast', 'Safe')),
        )
        code = 'model! beta/two, Fast, S'
        reply = model_config.complete_model_command(
            code, len(code), endpoints=endpoints)
        self.assertEqual(reply['matches'], ['Safe'])
        self.assertEqual(code[reply['cursor_start']:reply['cursor_end']],
                         'S')
        self.assertEqual(
            reply['metadata']['_jupyter_types_experimental'][0]['type'],
            'provider')

    def test_non_model_command_has_no_model_completions(self):
        import model_config
        self.assertIsNone(model_config.complete_model_command(
            'solve! x = 2', len('solve! x = 2'), endpoints=()))

    def test_kernel_do_complete_routes_to_model_configuration(self):
        import asyncio
        from toymathkernel import MathKernel
        code = 'model! z-ai'
        reply = asyncio.run(MathKernel.do_complete(None, code, len(code)))
        self.assertIn('z-ai/glm-5.2', reply['matches'])
        self.assertEqual(reply['status'], 'ok')

    def test_static_lexer_command_table_matches_committed_registries(self):
        import prompt_commands as pc
        from lexer import MathLexer
        from processor import register_actions
        expected = (set(register_actions()) | set(pc.load_commands())
                    | set(pc.RESERVED))
        self.assertEqual(MathLexer.KNOWN_COMMANDS, expected)

    def test_render_substitutes_arguments(self):
        import prompt_commands as pc
        cmd = pc.parse_command(
            '---\nname: int\ndescription: d\n---\nIntegrate $ARGUMENTS now',
            'int')
        self.assertEqual(pc.render(cmd, 'x^2'), 'Integrate x^2 now')

    def test_fallback_name_from_stem(self):
        import prompt_commands as pc
        cmd = pc.parse_command('---\ndescription: d\n---\n$ARGUMENTS', 'foo')
        self.assertEqual(cmd.name, 'foo')

    def test_expr_flag_parsed(self):
        import prompt_commands as pc
        plain = pc.parse_command('---\nname: a\ndescription: d\n---\n$ARGUMENTS',
                                 'a')
        self.assertFalse(plain.expr)
        inline = pc.parse_command(
            '---\nname: b\ndescription: d\nexpr: true\n---\n$ARGUMENTS', 'b')
        self.assertTrue(inline.expr)

    def test_prove_mode_parsed_and_cannot_compose(self):
        import prompt_commands as pc
        cmd = pc.parse_command(
            '---\nname: p\ndescription: d\nmode: prove\n---\n$ARGUMENTS',
            'p')
        self.assertEqual(cmd.mode, 'prove')
        with self.assertRaises(ValueError):
            pc.parse_command(
                '---\nname: p\ndescription: d\nmode: prove\nexpr: true\n'
                '---\n$ARGUMENTS', 'p')

    def test_fresh_field_parsed(self):
        import prompt_commands as pc
        plain = pc.parse_command('---\nname: a\ndescription: d\n---\n$ARGUMENTS',
                                 'a')
        self.assertEqual(plain.fresh, ())
        one = pc.parse_command(
            '---\nname: b\ndescription: d\nfresh: C\n---\n$ARGUMENTS', 'b')
        self.assertEqual(one.fresh, ('C',))
        many = pc.parse_command(
            '---\nname: c\ndescription: d\nfresh: [C, K]\n---\n$ARGUMENTS', 'c')
        self.assertEqual(many.fresh, ('C', 'K'))
        with self.assertRaises(ValueError):
            pc.parse_command(
                '---\nname: e\ndescription: d\nfresh: "\\\\bad"\n---\n$ARGUMENTS',
                'e')

    def test_rejects_missing_frontmatter(self):
        import prompt_commands as pc
        with self.assertRaises(ValueError):
            pc.parse_command('bare body $ARGUMENTS', 'x')

    def test_rejects_missing_description(self):
        import prompt_commands as pc
        with self.assertRaises(ValueError):
            pc.parse_command('---\nname: x\n---\n$ARGUMENTS', 'x')

    def test_rejects_missing_placeholder(self):
        import prompt_commands as pc
        with self.assertRaises(ValueError):
            pc.parse_command('---\nname: x\ndescription: d\n---\nno slot', 'x')

    def test_rejects_reserved_name(self):
        import prompt_commands as pc
        with self.assertRaises(ValueError):
            pc.parse_command('---\nname: do\ndescription: d\n---\n$ARGUMENTS',
                             'do')

    def test_bad_file_is_skipped_not_fatal(self):
        import tempfile
        import prompt_commands as pc
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, 'good.md'), 'w') as fh:
                fh.write('---\nname: good\ndescription: g\n---\n$ARGUMENTS')
            with open(os.path.join(d, 'bad.md'), 'w') as fh:
                fh.write('no frontmatter, no placeholder')
            reg = pc.load_commands(d)
        self.assertIn('good', reg)
        self.assertNotIn('bad', reg)


class TestPromptCommandDispatch(unittest.TestCase):
    """Kernel dispatch of `name!` cells (mocked agent — offline)."""

    def setUp(self):
        import engine
        self.displays = []
        engine.setHandler(lambda *objs, **kw: self.displays.extend(objs))
        from mathShell import MathShell
        self.shell = MathShell()

    def tearDown(self):
        import engine
        import IPython.display
        engine.setHandler(IPython.display.display)

    def _html(self):
        return ''.join(getattr(d, 'data', str(d)) for d in self.displays)

    def _capture_instruction(self):
        box = {}

        def fake_run(instruction, **kw):
            box['instruction'] = instruction
            box['kwargs'] = kw
            return {'ok': True, 'steps': [], 'assumptions': [],
                    'claims': [], 'final_result': None, 'summary': None}
        return box, fake_run

    def test_command_prefix_renders_template(self):
        # solve! is a plain (non-expr) command: whole-cell prefix -> exec_do
        box, fake = self._capture_instruction()
        with mock.patch.object(agent_do, 'run_instruction', fake):
            self.shell.exec('solve! 2x = 4', 1, add_to_history=True)
        self.assertIn('Solve', box['instruction'])
        self.assertIn('2x = 4', box['instruction'])

    def test_prove_command_passes_raw_claim_to_harness(self):
        box, fake = self._capture_instruction()
        claim = r'\lim_{n \to \infty} \frac{1}{n} = 0'
        with mock.patch.object(agent_do, 'run_instruction', fake):
            self.shell.exec('prove! ' + claim, 1, add_to_history=True)
        self.assertEqual(box['kwargs']['proof_goal'], claim)
        self.assertIn('Prove ' + claim, box['instruction'])

    def test_prove_command_resolves_backref_in_harness_claim(self):
        self.shell.exec('2 = 2', 1, add_to_history=True)
        box, fake = self._capture_instruction()
        with mock.patch.object(agent_do, 'run_instruction', fake):
            self.shell.exec('prove! [[1]]', 2, add_to_history=True)
        self.assertNotIn('[[1]]', box['kwargs']['proof_goal'])
        self.assertIn('=', box['kwargs']['proof_goal'])

    def test_backref_resolves_inside_command_args(self):
        self.shell.exec('2 + 3', 1, add_to_history=True)  # result 5
        box, fake = self._capture_instruction()
        with mock.patch.object(agent_do, 'run_instruction', fake):
            self.shell.exec('solve! [[1]]', 2, add_to_history=True)
        self.assertIn('5', box['instruction'])
        self.assertNotIn('[[1]]', box['instruction'])

    def test_unregistered_prefix_is_not_a_command(self):
        # `n!` (factorial) must fall through to the math path
        called = []
        with mock.patch.object(agent_do, 'run_instruction',
                               lambda *a, **k: called.append(1)):
            handled = self.shell.dispatch_command('n', '+ 1', 1, False)
        self.assertFalse(handled)
        self.assertEqual(called, [])

    def test_empty_argument_reports_error(self):
        with mock.patch.object(agent_do, 'run_instruction',
                               lambda *a, **k: self.fail('must not run')):
            self.shell.exec('solve!', 1, add_to_history=True)
        self.assertIn('needs an argument', self._html())

    def test_commands_listing_makes_no_api_call(self):
        with mock.patch.object(agent_do, 'run_instruction',
                               lambda *a, **k: self.fail('no agent for list')):
            self.shell.exec('commands!', 1)
        out = self._html()
        self.assertIn('notebook commands', out)
        self.assertIn('int', out)
        self.assertIn('model!', out)

    def test_model_command_sets_configured_routing_for_later_runs(self):
        box, fake = self._capture_instruction()
        with mock.patch.object(agent_do, 'run_instruction', fake):
            self.shell.exec('model! z-ai/glm-5.2', 1)
            self.shell.exec('do! Hello', 2)
        self.assertEqual(box['kwargs']['model_name'], 'z-ai/glm-5.2')
        self.assertEqual(box['kwargs']['providers'],
                         ('Cerebras', 'Fireworks'))
        self.assertIn('fallbacks disabled', self._html())

    def test_model_command_provider_arguments_override_config(self):
        self.shell.exec('model! z-ai/glm-5.2, Fireworks, Cerebras, Fireworks',
                        1)
        self.assertEqual(self.shell.model_name, 'z-ai/glm-5.2')
        self.assertEqual(self.shell.model_providers,
                         ('Fireworks', 'Cerebras'))

    def test_model_command_allows_unlisted_model(self):
        self.shell.exec('model! vendor/custom-model', 1)
        self.assertEqual(self.shell.model_name, 'vendor/custom-model')
        self.assertEqual(self.shell.model_providers, ())
        self.assertIn('default provider routing', self._html())

    def test_model_command_without_arguments_guides_to_completion(self):
        original = self.shell.model_name
        self.shell.exec('model!', 1)
        self.assertEqual(self.shell.model_name, original)
        self.assertIn('press <kbd>Tab</kbd>', self._html())

    def test_model_change_notifies_frontend_handler(self):
        changes = []
        self.shell.model_change_handler = (
            lambda model, providers: changes.append((model, providers)))
        self.shell.exec('model! z-ai/glm-5.2', 1)
        self.assertEqual(changes, [
            ('z-ai/glm-5.2', ('Cerebras', 'Fireworks')),
        ])

    def test_model_command_rejects_empty_provider(self):
        original = self.shell.model_name
        self.shell.exec('model! z-ai/glm-5.2,', 1)
        self.assertEqual(self.shell.model_name, original)
        self.assertIn('model! error', self._html())

    def test_command_steps_land_in_shared_ledger(self):
        with mock.patch.object(agent_do, 'build_model',
                               lambda model_name=None: ScriptedModel(
                                   SOLVE_SCRIPT)):
            self.shell.exec('solve! 2x + 3 = 7 for x', 2, add_to_history=True)
        self.assertEqual(len(self.shell.ledger.steps), 2)
        self.assertIn('2', self.shell.resolve_backrefs('[[2]]'))


def _ok(result, goal=None):
    """A well-behaved sub-run: when `goal` is given, the run carries one
    transforming step from the goal to the result (the chain an inline
    expr command now requires); without it the run is step-less."""
    steps = [] if goal is None else [
        {'id': 's1', 'op': 'scripted', 'input': goal, 'result': result}]
    return {'ok': True, 'final_result': result, 'assumptions': [],
            'steps': steps, 'summary': None,
            'final_provenance': {
                'status': 'verified', 'source': 'ledger', 'step': 's1',
                'method': 'test-fixture'}}


def _arg_of(instruction):
    """The argument embedded in a rendered int! instruction (first line:
    'Apply symbolic integration for <ARG>.')."""
    first = instruction.split('\n', 1)[0]
    prefix = 'Apply symbolic integration for '
    assert first.startswith(prefix) and first.endswith('.'), first
    return first[len(prefix):-1]


def _fail(error='no scripted answer'):
    return {'ok': False, 'error': error, 'assumptions': [], 'steps': [],
            'final_result': None}


def _never(*args, **kwargs):
    raise AssertionError('the agent must not be called for direct commands')


class TestExprComposite(unittest.TestCase):
    """Inline composition of expr commands ({diff! {int! x^3}}) — the LLM /
    procedural bridge. Agent mocked; the glue is really oracle-checked."""

    def setUp(self):
        import engine
        self.displays = []
        engine.setHandler(lambda *objs, **kw: self.displays.extend(objs))
        from mathShell import MathShell
        self.shell = MathShell()

    def tearDown(self):
        import engine
        import IPython.display
        engine.setHandler(IPython.display.display)

    def _html(self):
        return ''.join(getattr(d, 'data', str(d)) for d in self.displays)

    def test_nested_resolves_inner_to_outer(self):
        # int! is an LLM tactic (mocked); diff! is a DIRECT command since
        # gen 16, so the outer differentiation costs no agent call and its
        # variable is inferred with the minted C excluded
        calls = []

        def fake(instruction, ledger=None, on_step=None, **kw):
            calls.append(instruction)
            if instruction.startswith('Apply symbolic integration'):
                return _ok('\\frac{x^4}{4} + C', _arg_of(instruction))
            return _fail()
        with mock.patch.object(agent_do, 'run_instruction', fake):
            self.shell.exec('{diff! {int! x^3}}', 1, add_to_history=True)
        self.assertEqual(len(calls), 1)          # int! only; diff! is direct
        self.assertTrue(calls[0].startswith('Apply symbolic integration'))
        chained = self.shell.resolve_backrefs('[[1]]')
        self.assertEqual(core_tactics.equal_exprs(chained, 'x^3')['verdict'],
                         'yes')
        # ledger: the direct differentiate step + the oracle-checked glue
        ops = [s['op'] for s in self.shell.ledger.steps]
        self.assertEqual(ops, ['differentiate', 'expand'])
        for s in self.shell.ledger.steps:
            self.assertEqual(s['check']['status'], 'agree')

    def test_history_backrefs_do_not_accumulate_index_braces(self):
        # Final do!/composite output is future [[n]] input.  Every old plain
        # writer hop added one transparent group around powers/subscripts.
        source = 'x^{3}+C_{1}'
        for execution_count in range(1, 5):
            sym = self.shell.parser.parse(source)
            shown = self.shell.output(
                sym, self.shell.parsedNotation, execution_count, True)
            self.assertEqual(shown, 'x^{3}+C_{1}')
            source = self.shell.resolve_backrefs(
                f'[[{execution_count}]]')
            self.assertEqual(source, 'x^{3}+C_{1}')

    def test_duplicate_subexpression_memoized(self):
        calls = []

        def fake(instruction, ledger=None, on_step=None, **kw):
            calls.append(instruction)
            return _ok('\\frac{x^3}{3} + C', _arg_of(instruction))
        with mock.patch.object(agent_do, 'run_instruction', fake):
            self.shell.exec('{int! x^2} + {int! x^2}', 1, add_to_history=True)
        self.assertEqual(len(calls), 1)          # identical arg -> one call
        # glue combined and oracle-checked; the second splice mints its own
        # constant (C + C_{1}), NOT the dishonest 2C of one shared C
        step = self.shell.ledger.steps[-1]
        self.assertEqual(step['op'], 'expand')
        flat = step['result'].replace(' ', '')
        self.assertIn('C', flat)
        self.assertIn('C_{', flat)
        self.assertNotIn('2C', flat)
        self.assertEqual(step['check']['status'], 'agree')

    def test_independent_constants_do_not_cancel(self):
        # {int! f} - {int! f} is an arbitrary constant, not 0: the memoised
        # result is spliced twice but each splice mints its own C
        def fake(instruction, ledger=None, on_step=None, **kw):
            return _ok('\\frac{x^3}{3} + C', _arg_of(instruction))
        with mock.patch.object(agent_do, 'run_instruction', fake):
            self.shell.exec('{int! x^2} - {int! x^2}', 1, add_to_history=True)
        step = self.shell.ledger.steps[-1]
        flat = step['result'].replace(' ', '')
        self.assertNotEqual(flat, '0')
        self.assertIn('C', flat)
        self.assertIn('C_{', flat)
        self.assertEqual(step['check']['status'], 'agree')

    def test_single_command_keeps_plain_constant(self):
        def fake(instruction, ledger=None, on_step=None, **kw):
            return _ok('\\frac{x^3}{3} + C', _arg_of(instruction))
        with mock.patch.object(agent_do, 'run_instruction', fake):
            self.shell.exec('{int! x^2}', 1, add_to_history=True)
        step = self.shell.ledger.steps[-1]
        self.assertIn('C', step['result'])
        self.assertNotIn('C_{', step['result'])  # no gratuitous renaming
        self.assertEqual(step['check']['status'], 'agree')

    def test_user_constant_never_captured(self):
        # a C the user wrote in the cell must stay distinct from the minted one
        def fake(instruction, ledger=None, on_step=None, **kw):
            return _ok('\\frac{x^3}{3} + C', _arg_of(instruction))
        with mock.patch.object(agent_do, 'run_instruction', fake):
            self.shell.exec('{int! x^2} + C', 1, add_to_history=True)
        step = self.shell.ledger.steps[-1]
        flat = step['result'].replace(' ', '')
        self.assertIn('C_{', flat)               # the minted C was renamed
        self.assertNotIn('2C', flat)
        self.assertEqual(step['check']['status'], 'agree')

    def test_argument_bound_constant_not_renamed(self):
        # nested {int! {int! x}}: the outer result's C refers to the C in its
        # own argument (the inner result) - bound, not minted, so not renamed
        def fake(instruction, ledger=None, on_step=None, **kw):
            if 'frac' in instruction:  # outer call sees the spliced inner result
                return _ok('\\frac{x^3}{6} + Cx + K', _arg_of(instruction))
            return _ok('\\frac{x^2}{2} + C', _arg_of(instruction))
        with mock.patch.object(agent_do, 'run_instruction', fake):
            self.shell.exec('{int! {int! x}}', 1, add_to_history=True)
        step = self.shell.ledger.steps[-1]
        flat = step['result'].replace(' ', '')
        self.assertNotIn('C_{', flat)            # arg-bound C keeps its name
        self.assertIn('K', flat)                 # undeclared names untouched
        self.assertEqual(step['check']['status'], 'agree')

    def test_whole_cell_expr_routes_to_composite(self):
        calls = []

        def fake(instruction, ledger=None, on_step=None, **kw):
            calls.append(instruction)
            return _ok('\\frac{x^4}{4} + C', _arg_of(instruction))
        with mock.patch.object(agent_do, 'run_instruction', fake):
            self.shell.exec('int! x^3', 1, add_to_history=True)  # no braces
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.shell.ledger.steps[-1]['op'], 'expand')

    def test_composite_agent_run_uses_notebook_model_routing(self):
        calls = []

        def fake(instruction, **kwargs):
            calls.append(kwargs)
            return _ok('\\frac{x^3}{3} + C', _arg_of(instruction))

        self.shell.exec('model! z-ai/glm-5.2', 1)
        with mock.patch.object(agent_do, 'run_instruction', fake):
            self.shell.exec('int! x^2', 2, add_to_history=True)
        self.assertEqual(calls[0]['model_name'], 'z-ai/glm-5.2')
        self.assertEqual(calls[0]['providers'],
                         ('Cerebras', 'Fireworks'))

    def test_non_expr_command_in_composite_refused(self):
        # solve! is not expr; mixed into a composite it must be refused
        with mock.patch.object(agent_do, 'run_instruction',
                               lambda *a, **k: _ok('x')):
            self.shell.exec('{diff! x} + {solve! x}', 1, add_to_history=True)
        self.assertIn('not an inline expr command', self._html())

    def test_factorial_is_not_a_composite(self):
        # the router must not hijack `n!` (n is not a registered command);
        # only a real expr command routes to the composite evaluator
        self.assertFalse(self.shell.has_expr_command('n! + 1'))
        self.assertTrue(self.shell.has_expr_command('{diff! x}'))

    def test_backref_resolves_in_composite_argument(self):
        self.shell.exec('x^3', 1, add_to_history=True)
        # diff! is direct: the backref resolves before the primitive runs
        # and the whole cell costs zero agent calls
        with mock.patch.object(agent_do, 'run_instruction', _never):
            self.shell.exec('{diff! [[1]]}', 2, add_to_history=True)
        chained = self.shell.resolve_backrefs('[[2]]')
        self.assertEqual(core_tactics.equal_exprs(chained, '3x^2')['verdict'],
                         'yes')

    def test_agent_failure_surfaces(self):
        with mock.patch.object(agent_do, 'run_instruction',
                               lambda *a, **k: _fail('boom')):
            self.shell.exec('{int! x^3}', 1, add_to_history=True)
        self.assertIn('boom', self._html())

    def test_unverified_agent_result_cannot_enter_composite(self):
        result = _ok('x=999')
        result['final_provenance'] = {
            'status': 'unverified', 'source': 'query-only',
            'reason': 'no transforming step was recorded'}
        with mock.patch.object(agent_do, 'run_instruction',
                               lambda *a, **k: result):
            self.shell.exec('{int! x^3}', 1, add_to_history=True)
        self.assertIn('unverified final value', self._html())
        self.assertEqual(self.shell.ledger.steps, [])

    def test_resolver_call_cap(self):
        # unit-level: a low cap trips on distinct sub-expressions
        import primitives
        import expr_commands
        from notation import Notation
        cmds = self.shell.commands
        sym, notation = primitives.parse_latex('{int! x} + {int! x^2}')
        r = expr_commands.ExprResolver(
            notation, Notation(), cmds, self.shell.ledger, None,
            lambda instruction, **k: _ok('F', _arg_of(instruction)),
            max_calls=1)
        with self.assertRaises(expr_commands.ExprCommandError):
            r(sym)

    def test_non_closing_subrun_refused_and_summary_surfaced(self):
        # the run's only verified step answers a DIFFERENT question than
        # the cell asked; its result must not become the cell's value
        run = {'ok': True, 'final_result': '0', 'assumptions': [],
               'steps': [{'id': 's1', 'op': 'limit_table',
                          'input': '\\lim_{n \\to \\infty} \\frac{1}{n}',
                          'result': '0'}],
               'summary': 'the tactics do not close this; a telescoping '
                          'move is missing',
               'final_provenance': {'status': 'verified', 'source': 'ledger',
                                    'step': 's1', 'method': 'last-step'}}
        with mock.patch.object(agent_do, 'run_instruction',
                               lambda *a, **k: run):
            self.shell.exec('{int! x^3}', 1, add_to_history=True)
        html = self._html()
        self.assertIn('did not close', html)
        self.assertIn('telescoping move is missing', html)
        self.assertEqual(self.shell.ledger.steps, [])

    def test_operator_wrapped_root_step_accepted(self):
        # agents legitimately restate a bare integrand goal inside \int
        def fake(instruction, ledger=None, on_step=None, **kw):
            arg = _arg_of(instruction)
            run = _ok('\\frac{x^4}{4} + C',
                      goal=f'\\int {arg} \\, dx')
            run['final_provenance']['method'] = 'last-step'
            return run
        with mock.patch.object(agent_do, 'run_instruction', fake):
            self.shell.exec('{int! x^3}', 1, add_to_history=True)
        self.assertEqual(self.shell.ledger.steps[-1]['op'], 'expand')

    def test_whole_cell_lim_ellipsis_closes_via_sum_tactics(self):
        # the original failing notebook cell, end to end through the shell
        # composite path with a scripted agent (offline)
        with mock.patch.object(agent_do, 'build_model',
                               lambda model_name=None: ScriptedModel(
                                   list(LIM_SUM_SCRIPT))):
            self.shell.exec('lim! ' + LIM_SUM_EXPR, 1, add_to_history=True)
        html = self._html()
        self.assertNotIn('do! error', html)
        ops = [s['op'] for s in self.shell.ledger.steps]
        self.assertEqual(ops, ['sum_from_ellipsis', 'sum_telescope',
                               'limit_table', 'expand'])
        chained = self.shell.resolve_backrefs('[[1]]')
        self.assertEqual(core_tactics.equal_exprs(chained, '1')['verdict'],
                         'yes')
        # the pattern-continuation assumption is surfaced on the cell
        self.assertIn('continues the pattern', html)


class TestChainsToGoal(unittest.TestCase):
    """The goal-coverage gate on inline expr sub-runs."""

    def _steps(self, *triples):
        return [{'id': f's{i}', 'op': 'scripted', 'input': a, 'result': b}
                for i, (a, b) in enumerate(triples, 1)]

    def test_connected_chain_accepted(self):
        import expr_commands as ec
        steps = self._steps(('x^{3}', '3x^{2}'), ('3x^2', '6x'))
        self.assertTrue(ec._chains_to_goal(steps, 's2', 'x^3'))

    def test_disconnected_final_step_rejected(self):
        import expr_commands as ec
        steps = self._steps(('x^{3}', '3x^{2}'), ('y', '1'))
        self.assertFalse(ec._chains_to_goal(steps, 's2', 'x^3'))

    def test_branch_piece_rejected(self):
        # the lim! failure mode: a split produced pieces; a piece result
        # is not an answer to the whole
        import expr_commands as ec
        steps = self._steps(
            ('\\lim_{x \\to 0}(\\frac{\\sin x}{x} + x^2)',
             '\\lim_{x \\to 0} \\frac{\\sin x}{x} + \\lim_{x \\to 0} x^2'),
            ('\\lim_{x \\to 0} x^2', '0'))
        self.assertFalse(ec._chains_to_goal(
            steps, 's2', '\\lim_{x \\to 0}(\\frac{\\sin x}{x} + x^2)'))

    def test_no_steps_rejected(self):
        import expr_commands as ec
        self.assertFalse(ec._chains_to_goal([], 's1', 'x'))


class TestDirectCommands(unittest.TestCase):
    """The zero-token tier (gen 16): a `direct: <primitive>` command IS one
    verified primitive - no agent run, ledger step + oracle check for free."""

    def setUp(self):
        import engine
        self.displays = []
        engine.setHandler(lambda *objs, **kw: self.displays.extend(objs))
        from mathShell import MathShell
        self.shell = MathShell()

    def tearDown(self):
        import engine
        import IPython.display
        engine.setHandler(IPython.display.display)

    def _html(self):
        return ''.join(getattr(d, 'data', str(d)) for d in self.displays)

    def test_direct_expand_costs_no_tokens(self):
        with mock.patch.object(agent_do, 'run_instruction', _never):
            self.shell.exec('{expand! (1+x)(1-x)} + 2x^2', 1,
                            add_to_history=True)
        ops = [s['op'] for s in self.shell.ledger.steps]
        self.assertEqual(ops, ['expand', 'expand'])  # direct step + glue
        final = self.shell.ledger.steps[-1]
        self.assertEqual(final['result'].replace(' ', ''), 'x^{2}+1')
        for s in self.shell.ledger.steps:
            self.assertEqual(s['check']['status'], 'agree')

    def test_direct_diff_whole_cell(self):
        with mock.patch.object(agent_do, 'run_instruction', _never):
            self.shell.exec('diff! x^2', 1, add_to_history=True)  # no braces
        ops = [s['op'] for s in self.shell.ledger.steps]
        self.assertEqual(ops, ['differentiate', 'expand'])
        self.assertEqual(
            self.shell.ledger.steps[0]['result'].replace(' ', ''), '2x')

    def test_direct_inside_direct(self):
        with mock.patch.object(agent_do, 'run_instruction', _never):
            self.shell.exec('{expand! {diff! x^3} x}', 1, add_to_history=True)
        final = self.shell.ledger.steps[-1]
        self.assertEqual(final['result'].replace(' ', ''), '3x^{3}')

    def test_minted_constant_excluded_from_var_inference(self):
        # {diff! {int! x^3}}: the inner splice contains the minted C; the
        # direct differentiate must infer x, not refuse as ambiguous
        def fake(instruction, ledger=None, on_step=None, **kw):
            return _ok('\\frac{x^4}{4} + C', _arg_of(instruction))
        with mock.patch.object(agent_do, 'run_instruction', fake):
            self.shell.exec('{diff! {int! x^3}}', 1, add_to_history=True)
        step = self.shell.ledger.steps[0]
        self.assertEqual(step['op'], 'differentiate')
        self.assertEqual(step['args'].get('var'), 'x')

    def test_user_constant_still_ambiguous(self):
        # a plain user-written second symbol is NOT minted: refuse, don't guess
        with mock.patch.object(agent_do, 'run_instruction', _never):
            self.shell.exec('{diff! x^2 + C}', 1, add_to_history=True)
        self.assertIn('ambiguous variable', self._html())
        # the refusal teaches the explicit-parameter syntax
        self.assertIn('[x]', self._html())

    def test_explicit_var_parameter_whole_cell(self):
        # diff! [x] <expr> chooses the variable; no inference needed
        with mock.patch.object(agent_do, 'run_instruction', _never):
            self.shell.exec('diff! [x] Cx^{2} + C', 1, add_to_history=True)
        step = self.shell.ledger.steps[0]
        self.assertEqual(step['op'], 'differentiate')
        self.assertEqual(step['args'].get('var'), 'x')

    def test_explicit_var_parameter_inline(self):
        with mock.patch.object(agent_do, 'run_instruction', _never):
            self.shell.exec('{diff! [C] x C} + 1', 1, add_to_history=True)
        self.assertEqual(self.shell.ledger.steps[0]['args'].get('var'), 'C')

    def test_ledger_constant_breaks_inference_tie(self):
        # an antiderivative chained via [[n]] carries its minted C into a
        # later cell; the ledger's constant provenance disambiguates
        self.shell.ledger.record(
            integration_tactics.integrate_power_rule('x^2', 'x'))
        self.assertEqual(self.shell.ledger.steps[0].get('constant'), 'C')
        with mock.patch.object(agent_do, 'run_instruction', _never):
            self.shell.exec('diff! \\frac{x^{3}}{3} + C', 2,
                            add_to_history=True)
        diff_step = self.shell.ledger.steps[1]
        self.assertEqual(diff_step['op'], 'differentiate')
        self.assertEqual(diff_step['args'].get('var'), 'x')

    def test_var_parameter_on_non_var_command_refused(self):
        with mock.patch.object(agent_do, 'run_instruction', _never):
            self.shell.exec('{expand! [x] (x+1)^2}', 3, add_to_history=True)
        self.assertIn('does not take a [var]', self._html())

    def test_var_parameter_must_be_plain_symbol(self):
        with mock.patch.object(agent_do, 'run_instruction', _never):
            self.shell.exec('{diff! [x^2] x^3}', 4, add_to_history=True)
        self.assertIn('single plain variable name', self._html())

    def test_domain_differs_renders_amber_not_red(self):
        # collapsing d/dx of a ln-carrying antiderivative to its rational
        # form is a domain extension: conditional, not a failure
        step = {'id': 's1', 'hash': 'h', 'op': 'expand', 'args': {},
                'input': 'i', 'result': 'r', 'assumptions': [],
                'continues': None, 'check': {'status': 'domain-differs'}}
        out = self.shell.render_do_step(step)
        self.assertIn('[D!]', out)
        self.assertIn('#b65c00', out)
        self.assertNotIn('#c00"', out)

    def test_no_variable_refused(self):
        with mock.patch.object(agent_do, 'run_instruction', _never):
            self.shell.exec('{diff! 2 + 3}', 1, add_to_history=True)
        self.assertIn('no variable', self._html())

    def test_primitive_failure_surfaces(self):
        # unit-level: a direct command whose primitive refuses reports the
        # primitive's error through the composite error path
        import primitives
        import expr_commands
        import prompt_commands as pc
        from notation import Notation
        cmds = {'fq': pc.PromptCommand('fq', 'd', '', True, (),
                                       'factor_quadratic')}
        sym, notation = primitives.parse_latex(
            '{fq! x^3 + 1}', command_names=cmds)
        r = expr_commands.ExprResolver(
            notation, Notation(), cmds, None, None, _never)
        with self.assertRaises(expr_commands.ExprCommandError) as ctx:
            r(sym)
        self.assertIn('factor_quadratic', str(ctx.exception))

    def test_direct_evaluate_adapter(self):
        # unit-level: evaluate as a direct command splices a numeric value
        import primitives
        import expr_commands
        import prompt_commands as pc
        from notation import Notation
        cmds = {'ev': pc.PromptCommand('ev', 'd', '', True, (), 'evaluate')}
        sym, notation = primitives.parse_latex(
            '{ev! 2^{10}} + 1', command_names=cmds)
        out = Notation()
        r = expr_commands.ExprResolver(notation, out, cmds, None, None,
                                       _never)
        root = r(sym)
        from LatexWriter import LaTexWriter
        rec = core_tactics.expand(LaTexWriter(out)(root))
        self.assertEqual(rec['result'].replace(' ', ''), '1025')

    def test_unknown_direct_primitive_rejected_at_parse(self):
        import prompt_commands as pc
        with self.assertRaises(ValueError):
            pc.parse_command(
                '---\nname: z\ndescription: d\ndirect: solve\n---\n', 'z')

    def test_direct_body_needs_no_placeholder(self):
        import prompt_commands as pc
        cmd = pc.parse_command(
            '---\nname: z\ndescription: d\ndirect: expand\n---\njust docs',
            'z')
        self.assertEqual(cmd.direct, 'expand')
        self.assertTrue(cmd.expr)

    def test_direct_assumptions_reach_the_cell(self):
        # a direct primitive's recorded assumptions must surface like an
        # agent sub-run's (factor_quadratic records none; use collect on a
        # rational form? expand of 1/x keeps x != 0 out of scope - assert
        # the plumbing instead: direct_records land on the resolver)
        import primitives
        import expr_commands
        import prompt_commands as pc
        from notation import Notation
        cmds = {'ex': pc.PromptCommand('ex', 'd', '', True, (), 'expand')}
        sym, notation = primitives.parse_latex(
            '{ex! (1+x)^2}', command_names=cmds)
        r = expr_commands.ExprResolver(notation, Notation(), cmds, None,
                                       None, _never)
        r(sym)
        self.assertEqual(len(r.direct_records), 1)
        self.assertIn('assumptions', r.direct_records[0])


class TestUnknownToolSteering(unittest.TestCase):
    def test_hallucinated_tool_names_steer_instead_of_killing_the_run(self):
        # live probe: a strong model called the TACTIC name integrate_table
        # as a raw tool and the SDK's ModelBehaviorError aborted a run that
        # already held three green steps. Unknown tool calls must return to
        # the model as one-turn steering; a tactic-shaped name must point
        # at run_tactic and the owning skill.
        captured = []

        class RecordingModel(ScriptedModel):
            async def get_response(self, system_instructions, input,
                                   *args, **kwargs):
                captured.append(input)
                return await super().get_response(
                    system_instructions, input, *args, **kwargs)

        script = [
            [ResponseFunctionToolCall(
                type='function_call', name='integrate_table',
                arguments=json.dumps({'expr': 'x', 'var': 'x'}),
                call_id='b1', id='b1')],
            [ResponseFunctionToolCall(
                type='function_call', name='frobnicate',
                arguments='{}', call_id='b2', id='b2')],
            [tool_call('expand', {'expr': '(x+1)^2'}, 'c3')],
            [message('recovered')],
        ]
        res = run_instruction('misname some tools',
                              model=RecordingModel(script))
        self.assertTrue(res['ok'], res.get('error'))
        self.assertIsNone(res.get('error'))
        self.assertEqual([s['op'] for s in res['steps']], ['expand'])

        second = json.dumps(captured[1], default=str)
        self.assertIn('is a tactic, not a tool', second)
        self.assertIn('run_tactic', second)
        self.assertIn('integration', second)
        third = json.dumps(captured[2], default=str)
        self.assertIn('does not exist', third)
        self.assertIn('load_skill, run_tactic, comment', third)


@unittest.skipUnless(os.environ.get('TOYMATH_LIVE_TESTS') == '1',
                     'set TOYMATH_LIVE_TESTS=1 for a live OpenRouter test')
class TestLiveOpenRouter(unittest.TestCase):
    def test_solve_linear_equation(self):
        ledger = Ledger()
        res = run_instruction('Solve 2x + 3 = 7 for x, step by step.',
                              ledger=ledger)
        self.assertTrue(res['ok'], res.get('error'))
        self.assertGreaterEqual(len(res['steps']), 2)
        self.assertEqual(ledger.replay()['status'], 'verified')


if __name__ == '__main__':
    unittest.main()

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
        self.assertIn('Jupyter notebook cell', p)
        self.assertIn('## do! mode', p)
        # the command table survives verbatim
        self.assertIn('factor_quadratic', p)
        self.assertIn('mechanically checked', p)

    def test_missing_skill_falls_back(self):
        p = agent_do.build_prompt(skill_path='/nonexistent/SKILL.md')
        self.assertIn('mechanical checker', p)
        self.assertIn('## do! mode', p)


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

    def test_set_result_validates_and_overrides(self):
        session = DoSession()
        api = make_api(session)
        self.assertTrue(json.loads(api['set_result']('x = 2'))['ok'])
        self.assertEqual(session.result_override, 'x = 2')
        self.assertFalse(json.loads(api['set_result']('\\frac{'))['ok'])
        self.assertEqual(session.result_override, 'x = 2')  # kept
        self.assertEqual(session.new_steps(), [])  # never a ledger step

    def test_existing_ledger_range(self):
        ledger = Ledger()
        ledger.record({'ok': True, 'op': 'expand', 'args': {'expr': 'x+x'},
                       'input': 'x+x', 'result': '2x'})
        session = DoSession(ledger=ledger)
        api = make_api(session)
        api['expand']('3x + 3x')
        self.assertEqual(len(ledger.steps), 2)
        self.assertEqual(len(session.new_steps()), 1)


class TestScriptedAgent(unittest.TestCase):
    def test_full_run(self):
        seen = []
        res = run_instruction('solve 2x + 3 = 7 for x',
                              model=ScriptedModel(SOLVE_SCRIPT),
                              on_step=lambda s: seen.append(s['id']))
        self.assertTrue(res['ok'])
        self.assertEqual([s['op'] for s in res['steps']],
                         ['apply_both_sides', 'expand'])
        self.assertEqual(res['final_result'], '2x = 4')
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
        self.assertEqual(len(res['steps']), 2)

    def test_max_turns_returns_partial(self):
        endless = [[tool_call('expand', {'expr': 'x + x'}, f'c{i}')]
                   for i in range(10)]
        res = run_instruction('loop forever', model=ScriptedModel(endless),
                              max_turns=2)
        self.assertFalse(res['ok'])
        self.assertIn('turns', res['error'])
        self.assertEqual(len(res['steps']), 2)  # partial work is kept

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

    def test_resolve_backrefs(self):
        self.shell.exec('2 + 3', 1, add_to_history=True)
        text = self.shell.resolve_backrefs('check [[1]] please')
        self.assertIn('5', text)
        with self.assertRaises(ValueError):
            self.shell.resolve_backrefs('[[99]]')

    def test_do_cell_streams_and_chains(self):
        with mock.patch.object(agent_do, 'build_model',
                               lambda: ScriptedModel(SOLVE_SCRIPT)):
            self.shell.exec('do! solve 2x + 3 = 7 for x', 2,
                            add_to_history=True)
        out = self._html()
        self.assertIn('s1#', out)
        self.assertIn('s2#', out)
        self.assertIn('Subtracted 3', out)
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

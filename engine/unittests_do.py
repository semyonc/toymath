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
import plot_sandbox
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

    def test_set_result_validates_query_only_as_unverified(self):
        session = DoSession()
        api = make_api(session)
        rec = json.loads(api['set_result']('x = 2'))
        self.assertTrue(rec['ok'])
        self.assertEqual(rec['provenance']['status'], 'unverified')
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


FAKE_PNG_B64 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAAAAAA6fptVAAAACklEQVR4nGNiAAAABgADNjd8qAAAAABJRU5ErkJggg=='


class FakePlotBackend(object):
    name = 'fake'

    def __init__(self, result=None):
        self.calls = []
        self.result = result or {'ok': True, 'stdout': 'drew it\n',
                                 'stderr': '', 'images': [FAKE_PNG_B64]}

    def run_plot(self, code, timeout=None):
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
                            on_plot=lambda cap, imgs: shown.append(
                                (cap, imgs)))
        api = make_api(session)
        reply = json.loads(api['plot']('plt.plot([1])', 'a parabola'))
        # the model sees counts and stdout, never image bytes
        self.assertTrue(reply['ok'])
        self.assertEqual(reply['plots'], 1)
        self.assertNotIn(FAKE_PNG_B64[:24], json.dumps(reply))
        # the user sees the figure with its caption
        self.assertEqual(shown[0][0], 'a parabola')
        self.assertEqual(shown[0][1], [FAKE_PNG_B64])
        # never a ledger step
        self.assertEqual(session.new_steps(), [])

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

    def test_parse_runner_output(self):
        good = plot_sandbox._parse_runner_output(
            'noise\n{"ok": true, "images": []}\n', '')
        self.assertTrue(good['ok'])
        bad = plot_sandbox._parse_runner_output('garbage only', 'boom')
        self.assertFalse(bad['ok'])
        self.assertIn('boom', bad['stderr'])


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
        for name in ('int', 'diff', 'solve', 'expand'):
            self.assertIn(name, reg)
        self.assertIsNone(reg['int'].direct)      # LLM tactic tier
        self.assertEqual(reg['diff'].direct, 'differentiate')
        self.assertEqual(reg['expand'].direct, 'expand')
        self.assertTrue(reg['diff'].expr)         # direct implies expr

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
            return {'ok': True, 'steps': [], 'assumptions': [],
                    'final_result': None, 'summary': None}
        return box, fake_run

    def test_command_prefix_renders_template(self):
        # solve! is a plain (non-expr) command: whole-cell prefix -> exec_do
        box, fake = self._capture_instruction()
        with mock.patch.object(agent_do, 'run_instruction', fake):
            self.shell.exec('solve! 2x = 4', 1, add_to_history=True)
        self.assertIn('Solve', box['instruction'])
        self.assertIn('2x = 4', box['instruction'])

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

    def test_command_steps_land_in_shared_ledger(self):
        with mock.patch.object(agent_do, 'build_model',
                               lambda: ScriptedModel(SOLVE_SCRIPT)):
            self.shell.exec('solve! 2x + 3 = 7 for x', 2, add_to_history=True)
        self.assertEqual(len(self.shell.ledger.steps), 2)
        self.assertIn('2', self.shell.resolve_backrefs('[[2]]'))


def _ok(result):
    return {'ok': True, 'final_result': result, 'assumptions': [],
            'steps': [], 'summary': None,
            'final_provenance': {
                'status': 'verified', 'source': 'ledger', 'step': 's1',
                'method': 'test-fixture'}}


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
                return _ok('\\frac{x^4}{4} + C')
            return _fail()
        with mock.patch.object(agent_do, 'run_instruction', fake):
            self.shell.exec('{diff! {int! x^3}}', 1, add_to_history=True)
        self.assertEqual(len(calls), 1)          # int! only; diff! is direct
        self.assertTrue(calls[0].startswith('Apply symbolic integration'))
        import primitives
        chained = self.shell.resolve_backrefs('[[1]]')
        self.assertEqual(primitives.equal_exprs(chained, 'x^3')['verdict'],
                         'yes')
        # ledger: the direct differentiate step + the oracle-checked glue
        ops = [s['op'] for s in self.shell.ledger.steps]
        self.assertEqual(ops, ['differentiate', 'expand'])
        for s in self.shell.ledger.steps:
            self.assertEqual(s['check']['status'], 'agree')

    def test_duplicate_subexpression_memoized(self):
        calls = []

        def fake(instruction, ledger=None, on_step=None, **kw):
            calls.append(instruction)
            return _ok('\\frac{x^3}{3} + C')
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
            return _ok('\\frac{x^3}{3} + C')
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
            return _ok('\\frac{x^3}{3} + C')
        with mock.patch.object(agent_do, 'run_instruction', fake):
            self.shell.exec('{int! x^2}', 1, add_to_history=True)
        step = self.shell.ledger.steps[-1]
        self.assertIn('C', step['result'])
        self.assertNotIn('C_{', step['result'])  # no gratuitous renaming
        self.assertEqual(step['check']['status'], 'agree')

    def test_user_constant_never_captured(self):
        # a C the user wrote in the cell must stay distinct from the minted one
        def fake(instruction, ledger=None, on_step=None, **kw):
            return _ok('\\frac{x^3}{3} + C')
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
                return _ok('\\frac{x^3}{6} + Cx + K')
            return _ok('\\frac{x^2}{2} + C')
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
            return _ok('\\frac{x^4}{4} + C')
        with mock.patch.object(agent_do, 'run_instruction', fake):
            self.shell.exec('int! x^3', 1, add_to_history=True)  # no braces
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.shell.ledger.steps[-1]['op'], 'expand')

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
        import primitives
        chained = self.shell.resolve_backrefs('[[2]]')
        self.assertEqual(primitives.equal_exprs(chained, '3x^2')['verdict'],
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
            lambda *a, **k: _ok('F'), max_calls=1)
        with self.assertRaises(expr_commands.ExprCommandError):
            r(sym)


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
            return _ok('\\frac{x^4}{4} + C')
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
        import primitives
        self.shell.ledger.record(primitives.integrate_power_rule('x^2', 'x'))
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
        sym, notation = primitives.parse_latex('{fq! x^3 + 1}')
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
        sym, notation = primitives.parse_latex('{ev! 2^{10}} + 1')
        out = Notation()
        r = expr_commands.ExprResolver(notation, out, cmds, None, None,
                                       _never)
        root = r(sym)
        from LatexWriter import LaTexWriter
        rec = primitives.expand(LaTexWriter(out)(root))
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
        sym, notation = primitives.parse_latex('{ex! (1+x)^2}')
        r = expr_commands.ExprResolver(notation, Notation(), cmds, None,
                                       None, _never)
        r(sym)
        self.assertEqual(len(r.direct_records), 1)
        self.assertIn('assumptions', r.direct_records[0])


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

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

    def test_set_result_validates_and_overrides(self):
        session = DoSession()
        api = make_api(session)
        self.assertTrue(json.loads(api['set_result']('x = 2'))['ok'])
        self.assertEqual(session.result_override, 'x = 2')
        self.assertFalse(json.loads(api['set_result']('\\frac{'))['ok'])
        self.assertEqual(session.result_override, 'x = 2')  # kept
        self.assertEqual(session.new_steps(), [])  # never a ledger step

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


class TestPromptCommandModel(unittest.TestCase):
    """The discovery/parse model in prompt_commands.py (no shell, no agent)."""

    def test_repo_commands_load(self):
        import prompt_commands as pc
        reg = pc.load_commands()
        for name in ('int', 'diff', 'solve'):
            self.assertIn(name, reg)
            self.assertIn('$ARGUMENTS', reg[name].template)

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
            'steps': [], 'summary': None}


def _fail(error='no scripted answer'):
    return {'ok': False, 'error': error, 'assumptions': [], 'steps': [],
            'final_result': None}


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
        calls = []

        def fake(instruction, ledger=None, on_step=None, **kw):
            calls.append(instruction)
            if instruction.startswith('Apply symbolic integration'):
                return _ok('\\frac{x^4}{4} + C')
            if instruction.startswith('Differentiate'):
                return _ok('x^3')
            return _fail()
        with mock.patch.object(agent_do, 'run_instruction', fake):
            self.shell.exec('{diff! {int! x^3}}', 1, add_to_history=True)
        self.assertEqual(len(calls), 2)          # int! then diff!
        self.assertTrue(calls[0].startswith('Apply symbolic integration'))
        self.assertTrue(calls[1].startswith('Differentiate'))
        import primitives
        chained = self.shell.resolve_backrefs('[[1]]')
        self.assertEqual(primitives.equal_exprs(chained, 'x^3')['verdict'],
                         'yes')
        # the verified glue is one recorded, oracle-checked expand step
        self.assertEqual(len(self.shell.ledger.steps), 1)
        self.assertEqual(self.shell.ledger.steps[-1]['op'], 'expand')
        self.assertEqual(self.shell.ledger.steps[-1]['check']['status'],
                         'agree')

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
        calls = []

        def fake(instruction, ledger=None, on_step=None, **kw):
            calls.append(instruction)
            return _ok('3x^2')
        with mock.patch.object(agent_do, 'run_instruction', fake):
            self.shell.exec('{diff! [[1]]}', 2, add_to_history=True)
        self.assertNotIn('[[1]]', calls[0])      # backref resolved before agent
        self.assertIn('x^{', calls[0])           # ...to the x^3 expression

    def test_agent_failure_surfaces(self):
        with mock.patch.object(agent_do, 'run_instruction',
                               lambda *a, **k: _fail('boom')):
            self.shell.exec('{int! x^3}', 1, add_to_history=True)
        self.assertIn('boom', self._html())

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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Registry, progressive-skill, and CLI compatibility tests."""
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout

import agent_do
import primitives
import tactic_registry
import tactic_skills
import toymath_cli
from ledger import TRANSFORMING_OPS


class TestTacticRegistry(unittest.TestCase):
    def test_tactics_live_in_their_static_subject_modules(self):
        expected = {
            'core': 'tactics.core',
            'differentiation': 'tactics.differentiation',
            'equations': 'tactics.equations',
            'integration': 'tactics.integration',
            'limits': 'tactics.limits',
            'finite_operators': 'tactics.finite_operators',
            'matrices': 'tactics.matrices',
        }
        for spec in tactic_registry.TACTICS:
            self.assertEqual(spec.function.__module__, expected[spec.skill])
            self.assertFalse(hasattr(primitives, spec.function.__name__))

    def test_registry_is_unique_and_every_skill_exists(self):
        self.assertEqual(len(tactic_registry.BY_NAME),
                         len(tactic_registry.TACTICS))
        self.assertEqual(len(tactic_registry.BY_OP),
                         len(tactic_registry.TACTICS))
        self.assertEqual(tactic_skills.validate(), [])

    def test_transforming_ops_are_derived_from_registry(self):
        expected = {spec.op for spec in tactic_registry.TACTICS
                    if spec.transforming}
        self.assertEqual(TRANSFORMING_OPS, expected)

    def test_registry_invocation_and_replay_share_one_spec(self):
        session = agent_do.DoSession()
        record = tactic_registry.invoke_agent(
            'expand', ['(x+1)^2'], session)
        self.assertTrue(record['ok'])
        self.assertEqual(record['step']['id'], 's1')
        replayed = tactic_registry.replay(record['op'], record['args'])
        self.assertTrue(replayed['ok'])
        self.assertEqual(replayed['result'], record['result'])

    def test_subject_tactic_requires_loaded_skill(self):
        session = agent_do.DoSession()
        refused = tactic_registry.invoke_agent(
            'limit_table', [r'\lim_{n \to \infty} \frac{1}{n}'], session)
        self.assertFalse(refused['ok'])
        self.assertIn('load_skill', refused['error'])
        loaded = agent_do.make_api(session)['load_skill']('limits')
        self.assertIn("Loaded skill 'limits'", loaded)
        accepted = tactic_registry.invoke_agent(
            'limit_table', [r'\lim_{n \to \infty} \frac{1}{n}'], session)
        self.assertTrue(accepted['ok'], accepted.get('error'))

    def test_virtual_loader_resolves_subject_alias_and_tactic_owner(self):
        session = agent_do.DoSession()
        api = agent_do.make_api(session)
        by_subject = api['load_skill']('roots')
        self.assertIn("Loaded skill 'equations'", by_subject)
        self.assertIn("resolved from 'roots'", by_subject)
        self.assertIn('quadratic_roots EXPR VAR', by_subject)
        self.assertIn('equations', session.loaded_skills)

        another = agent_do.DoSession()
        by_tactic = agent_do.make_api(another)['load_skill'](
            'quadratic_roots')
        self.assertIn("Loaded skill 'equations'", by_tactic)
        self.assertIn('equations', another.loaded_skills)

    def test_virtual_loader_unknown_subject_lists_canonical_choices(self):
        reply = agent_do.make_api(agent_do.DoSession())['load_skill'](
            'mystery mathematics')
        self.assertIn('Cannot load skill', reply)
        self.assertIn('available subjects', reply)
        self.assertIn('equations', reply)

    def test_runtime_tool_surface_is_constant_and_small(self):
        tools = agent_do.make_tools(agent_do.DoSession())
        # seven fixed non-plot tools: six since gen 27 plus the run-level
        # open-outcome control. Tactic growth must never grow this list.
        self.assertEqual([tool.name for tool in tools], [
            'load_skill', 'run_tactic', 'comment', 'claim', 'conclude',
            'set_result', 'set_open'])
        prompt = agent_do.build_prompt()
        payload_chars = len(prompt)
        payload_chars += sum(len(json.dumps(tool.params_json_schema,
                                            sort_keys=True))
                             + len(tool.description or '')
                             for tool in tools)
        # Gen 26 baseline was 36,001 characters / 33 tools. Keep enough
        # margin for core guidance without silently returning to that shape.
        self.assertLess(payload_chars, 10000)

    def test_loaded_skill_contains_only_its_generated_interface(self):
        limits = tactic_skills.render('limits')
        self.assertIn('limit_squeeze EXPR LOWER UPPER LOWER_STEP UPPER_STEP',
                      limits)
        self.assertIn('limit_from_sides EXPR LEFT_STEP RIGHT_STEP', limits)
        self.assertNotIn('integrate_by_parts EXPR', limits)
        equations = tactic_skills.render('equations')
        self.assertIn('quadratic_roots EXPR VAR', equations)
        self.assertIn('points_assemble EXPR VAR ROOTS_STEP STEP...',
                      equations)
        self.assertIn('system_assemble TARGET STEP...', equations)
        self.assertIn('cases_assemble TARGET UNION STEP...', equations)
        self.assertNotIn('diff EXPR VAR', equations)

    def test_existing_cli_shapes_are_preserved(self):
        parser = toymath_cli.build_parser()
        apply = parser.parse_args(['apply', '2x+3=7', '-', '3'])
        self.assertEqual((apply.cmd, apply.equation, apply.op, apply.arg),
                         ('apply', '2x+3=7', '-', '3'))
        rewrite = parser.parse_args([
            'rewrite', 'x^2-y^2', 'diff_squares',
            '--direction', 'backward'])
        self.assertEqual(rewrite.direction, 'backward')
        self.assertIsNone(rewrite.at)
        at = parser.parse_args([
            'rewrite', '(x^2-1)(x^2-4)', 'diff_squares', '--at', 'x^2-4'])
        self.assertEqual(at.at, 'x^2-4')
        assemble = parser.parse_args([
            'limit_assemble', 'L', '1', '0'])
        self.assertEqual(assemble.values, ['1', '0'])
        roots = parser.parse_args(['quadratic_roots', 'x^2-1', 'x'])
        self.assertEqual((roots.cmd, roots.expr, roots.var),
                         ('quadratic_roots', 'x^2-1', 'x'))
        points = parser.parse_args([
            'points_assemble', 'x^3-3x', 'x', 's2', 's5', 's6'])
        self.assertEqual((points.cmd, points.roots_step, points.value_steps),
                         ('points_assemble', 's2', ['s5', 's6']))
        sides = parser.parse_args([
            'limit_from_sides', 'L', 's6', 's3'])
        self.assertEqual((sides.cmd, sides.left_step, sides.right_step),
                         ('limit_from_sides', 's6', 's3'))
        branch = parser.parse_args([
            'branch', 's2', 'try another route', '--session', 'work.json'])
        self.assertEqual((branch.cmd, branch.from_step, branch.reason),
                         ('branch', 's2', 'try another route'))

    def test_cli_points_assemble_reads_recorded_steps(self):
        from ledger import Ledger
        from tactics import core, differentiation, equations

        path = os.path.join(tempfile.mkdtemp(), 'points.json')
        ledger = Ledger(path)
        ledger.record(differentiation.differentiate('x^3-3x', 'x'))
        ledger.record(equations.quadratic_roots('3x^{2}-3', 'x'))
        for value in ('(-1)^{3}-3(-1)', '(1)^{3}-3(1)'):
            ledger.record(core.evaluate(value))
        ledger.save()
        output = io.StringIO()
        with redirect_stdout(output):
            code = toymath_cli.main([
                'points_assemble', 'x^3-3x', 'x', 's2', 's3', 's4',
                '--session', path])
        self.assertEqual(code, 0)
        rec = json.loads(output.getvalue())
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['result'], r'\{(-1,2),(1,-2)\}')
        self.assertEqual(rec['sources'],
                         {'roots': 's2', 'values': ['s3', 's4']})
        self.assertEqual(Ledger(path).replay()['status'], 'verified')

    def test_cli_integrate_improper_reads_recorded_steps(self):
        from ledger import Ledger
        from tactics import core, integration, limits

        path = os.path.join(tempfile.mkdtemp(), 'improper.json')
        ledger = Ledger(path)
        ledger.record(integration.integrate_substitute(
            '\\int \\frac{1}{(2-x) \\sqrt{1-x}} \\, d x', 'x',
            '\\sqrt{1-x}', 'u', '-\\frac{2}{1+u^{2}}'))
        ledger.record(integration.integrate_table(
            ledger.last_result(), 'u'))
        ledger.record(core.substitute(
            ledger.last_result(), 'u', '\\sqrt{1-x}'))
        truncated = integration.integrate_definite(
            '\\int_0^t \\frac{1}{(2-x) \\sqrt{1-x}} \\, d x', 'x',
            ledger.last_result())
        truncated['sources'] = {'antiderivative': 's3'}
        ledger.record(truncated)
        ledger.record(limits.limit_evaluate(
            primitives._limit_latex('t', '1', 'left',
                                    ledger.last_result()),
            '\\frac{\\pi}{2}'))
        ledger.save()
        output = io.StringIO()
        with redirect_stdout(output):
            code = toymath_cli.main([
                'integrate_improper',
                '\\int_0^1 \\frac{d x}{(2-x) \\sqrt{1-x}}', 'x',
                's4', 's5', '--session', path])
        self.assertEqual(code, 0)
        rec = json.loads(output.getvalue())
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['result'], '\\frac{\\pi}{2}')
        self.assertEqual(rec['sources'], {'truncated': 's4', 'limit': 's5'})
        self.assertEqual(rec['check']['status'], 'agree')
        saved = Ledger(path)
        self.assertEqual(saved.replay()['status'], 'verified')
        # a forged limit source must fail replay with the named reason
        saved.steps[-1]['sources']['limit'] = 's3'
        report = saved.replay()
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['reason'], 'limit provenance mismatch')

    def test_cli_integrate_improper_requires_the_definite_evaluation(self):
        from ledger import Ledger
        from tactics import core

        path = os.path.join(tempfile.mkdtemp(), 'improper-op.json')
        ledger = Ledger(path)
        ledger.record(core.expand('(x+1)^2'))
        ledger.save()
        output = io.StringIO()
        with redirect_stdout(output):
            code = toymath_cli.main([
                'integrate_improper',
                '\\int_0^1 \\frac{d x}{(2-x) \\sqrt{1-x}}', 'x',
                's1', 's1', '--session', path])
        self.assertEqual(code, 1)
        rec = json.loads(output.getvalue())
        self.assertFalse(rec['ok'])
        self.assertIn('integrate_definite', rec['error'])

    def test_cli_system_assemble_reads_recorded_steps(self):
        from ledger import Ledger
        from tactics import core

        path = os.path.join(tempfile.mkdtemp(), 'system.json')
        ledger = Ledger(path)
        ledger.record(core.expand('x+2-2 = 6-2'))
        ledger.record(core.expand('y+1-1 = 3-1'))
        ledger.save()
        output = io.StringIO()
        with redirect_stdout(output):
            code = toymath_cli.main([
                'system_assemble', 'x+y=6, x-y=2', 's1', 's2',
                '--session', path])
        self.assertEqual(code, 0)
        rec = json.loads(output.getvalue())
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['result'], 'x=4,y=2')
        self.assertEqual(rec['sources'], {'assignments': ['s1', 's2']})
        self.assertEqual(rec['check']['status'], 'agree')
        self.assertEqual(Ledger(path).replay()['status'], 'verified')

    def test_cli_cases_assemble_reads_recorded_steps(self):
        from ledger import Ledger
        from tactics import core

        path = os.path.join(tempfile.mkdtemp(), 'cases.json')
        ledger = Ledger(path)
        target = r'\frac{1}{x} \lt 2'
        for hypothesis in (r'x \gt 0', r'x \lt 0'):
            applied = core.apply_both_sides(target, '*', 'x',
                                            assuming=hypothesis)
            ledger.record(applied)
            cleared = core.expand(applied['result'])
            ledger.record(cleared)
            halved = core.apply_both_sides(cleared['result'], '/', '2')
            ledger.record(halved)
            ledger.record(core.expand(halved['result']))
        ledger.save()
        output = io.StringIO()
        with redirect_stdout(output):
            code = toymath_cli.main([
                'cases_assemble', target,
                r'x \gt \frac{1}{2} \lor x \lt 0', 's4', 's8',
                '--session', path])
        self.assertEqual(code, 0)
        rec = json.loads(output.getvalue())
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['sources'], {'cases': ['s4', 's8']})
        self.assertEqual(rec['check']['status'], 'agree')
        self.assertEqual(Ledger(path).replay()['status'], 'verified')

    def test_cli_cases_assemble_accepts_one_conjunction_case(self):
        from ledger import Ledger
        from tactics import core

        path = os.path.join(tempfile.mkdtemp(), 'between.json')
        ledger = Ledger(path)
        target = r'x^2 \lt 1'
        moved = ledger.record(core.apply_both_sides(target, '-', '1'))
        expanded = ledger.record(core.expand(moved['result']))
        factored = ledger.record(core.factor_quadratic(
            expanded['result'], 'x'))
        divided = ledger.record(core.apply_both_sides(
            factored['result'], '/', 'x+1', assuming=r'x \gt -1'))
        cancelled = ledger.record(core.expand(divided['result']))
        shifted = ledger.record(core.apply_both_sides(
            cancelled['result'], '+', '1'))
        endpoint = ledger.record(core.expand(shifted['result']))
        ledger.save()

        output = io.StringIO()
        with redirect_stdout(output):
            code = toymath_cli.main([
                'cases_assemble', target, r'-1 \lt x \lt 1',
                endpoint['id'], '--session', path])
        self.assertEqual(code, 0)
        rec = json.loads(output.getvalue())
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['sources'], {'cases': [endpoint['id']]})
        self.assertEqual(rec['result'], r'-1 \lt x \land x \lt 1')
        self.assertEqual(Ledger(path).replay()['status'], 'verified')

    def test_cli_cases_assemble_without_session_refuses(self):
        output = io.StringIO()
        with redirect_stdout(output):
            code = toymath_cli.main([
                'cases_assemble', r'\frac{1}{x} \lt 2',
                r'x \gt \frac{1}{2} \lor x \lt 0', 's4', 's8'])
        rec = json.loads(output.getvalue())
        self.assertFalse(rec['ok'])
        self.assertIn('session', rec['error'])
        self.assertNotEqual(code, 0)

    def test_cli_system_assemble_without_session_refuses(self):
        output = io.StringIO()
        with redirect_stdout(output):
            code = toymath_cli.main(['system_assemble', 'x+y=6', 's1', 's2'])
        rec = json.loads(output.getvalue())
        self.assertFalse(rec['ok'])
        self.assertIn('session', rec['error'])
        self.assertNotEqual(code, 0)

    def test_cli_points_assemble_without_session_refuses(self):
        output = io.StringIO()
        with redirect_stdout(output):
            code = toymath_cli.main([
                'points_assemble', 'x^3-3x', 'x', 's2', 's3', 's4'])
        rec = json.loads(output.getvalue())
        self.assertFalse(rec['ok'])
        self.assertIn('session', rec['error'])
        self.assertNotEqual(code, 0)

    def test_cli_branch_records_and_replays_marker(self):
        from ledger import Ledger
        from tactics import core

        path = os.path.join(tempfile.mkdtemp(), 'branch.json')
        ledger = Ledger(path)
        ledger.record(core.expand('(x+1)^2'))
        ledger.save()
        output = io.StringIO()
        with redirect_stdout(output):
            code = toymath_cli.main([
                'branch', 's1', 'the substitution route stalled',
                '--session', path])
        self.assertEqual(code, 0)
        rec = json.loads(output.getvalue())
        self.assertEqual((rec['op'], rec['from']), ('branch', 's1'))
        loaded = Ledger(path)
        self.assertEqual(loaded.steps[-1]['op'], 'branch')
        self.assertEqual(loaded.replay()['status'], 'verified')

    def test_cli_open_records_replayable_open_outcome(self):
        from ledger import Ledger
        from tactics import core

        path = os.path.join(tempfile.mkdtemp(), 'open.json')
        ledger = Ledger(path)
        ledger.record(core.expand('(x+1)^2'))
        ledger.save()
        output = io.StringIO()
        with redirect_stdout(output):
            code = toymath_cli.main([
                'open', 'a checked lower-bound tactic is missing',
                '--session', path])
        self.assertEqual(code, 0)
        rec = json.loads(output.getvalue())
        self.assertEqual((rec['op'], rec['id']), ('open', 'r1'))
        loaded = Ledger(path)
        self.assertIsNone(loaded.selections[-1]['result'])
        self.assertEqual(loaded.selections[-1]['provenance']['status'],
                         'open')
        self.assertEqual(loaded.replay()['status'], 'verified')
        shown = io.StringIO()
        with redirect_stdout(shown):
            toymath_cli.main(['show', '--session', path])
        self.assertIn('OPEN r1#', shown.getvalue())

    def test_cli_squeeze_session_refuses_sourceless_record(self):
        # the CLI explicit-value squeeze cannot carry step provenance;
        # recording it would produce a session that fails replay, so the
        # ledger refuses at admission and no session file is written
        path = os.path.join(tempfile.mkdtemp(), 'squeeze.json')
        output = io.StringIO()
        with redirect_stdout(output):
            code = toymath_cli.main([
                'limit_squeeze',
                '\\lim_{x \\to 0} x^2 \\sin{\\frac{1}{x}}',
                '(-x^2)', 'x^2', '0', '--session', path])
        self.assertEqual(code, 1)
        rec = json.loads(output.getvalue())
        self.assertFalse(rec['ok'])
        self.assertIn('provenance', rec['error'])
        self.assertFalse(os.path.exists(path))

    def test_cli_markdown_show_uses_persisted_selection_to_fold_path(self):
        from ledger import Ledger
        from tactics import core

        path = os.path.join(tempfile.mkdtemp(), 'topology.json')
        ledger = Ledger(path)
        source = ledger.record(core.expand('(x+1)^2'))
        ledger.record(core.substitute(source['result'], 'x', '1'))
        ledger.record_branch(source['id'], 'numeric detour was not the goal')
        resumed = ledger.record(core.factor_quadratic(
            source['result'], 'x'))
        ledger.record_selection(resumed['result'], {
            'status': 'verified', 'source': 'ledger',
            'step': resumed['id'], 'method': 'exact-result',
        })
        ledger.save()

        output = io.StringIO()
        with redirect_stdout(output):
            code = toymath_cli.main([
                'show', '--format', 'md', '--session', path])
        self.assertEqual(code, 0)
        rendered = output.getvalue()
        self.assertIn('Selected final result `r1` from `s4`', rendered)
        self.assertIn('<details>', rendered)
        self.assertIn('numeric detour was not the goal', rendered)
        self.assertIn('**s2**', rendered)
        self.assertEqual(Ledger(path).replay()['status'], 'verified')

    def test_rewrite_at_flows_through_agent_and_replay(self):
        session = agent_do.DoSession()
        record = tactic_registry.invoke_agent(
            'rewrite',
            ['(x^2-1)(x^2-4)', 'diff_squares', 'forward', 'x^2-4'],
            session)
        self.assertTrue(record['ok'], record.get('error'))
        self.assertEqual(record['at'], 'x^{2}-4')
        self.assertEqual(record['args']['at'], 'x^2-4')
        replayed = tactic_registry.replay(record['op'], record['args'])
        self.assertEqual(replayed['result'], record['result'])
        # records from before the selector replay through the default
        legacy = tactic_registry.replay('rewrite', {
            'expr': 'x^2 - 4', 'lemma': 'diff_squares',
            'direction': 'forward'})
        self.assertTrue(legacy['ok'])
        self.assertEqual(legacy['result'], '(x+2)(x-2)')

    def test_case_hypothesis_flows_through_agent_cli_and_replay(self):
        session = agent_do.DoSession()
        record = tactic_registry.invoke_agent(
            'apply', ['a x \\lt b', '/', 'a', 'a > 0'], session)
        self.assertTrue(record['ok'], record.get('error'))
        self.assertIn('\\lt', record['result'])
        replayed = tactic_registry.replay(record['op'], record['args'])
        self.assertEqual(replayed['result'], record['result'])
        # records from before the hypothesis argument replay unchanged
        legacy = tactic_registry.replay('apply_both_sides', {
            'equation': '2x = 4', 'op': '/', 'arg': '2'})
        self.assertTrue(legacy['ok'])
        self.assertEqual(legacy['result'], '\\frac{2x}{2} = \\frac{4}{2}')

    def test_omitted_optional_argument_may_arrive_as_a_null(self):
        session = agent_do.DoSession()
        record = tactic_registry.invoke_agent(
            'apply', ['2x = 4', '/', '2', None], session)
        self.assertTrue(record['ok'], record.get('error'))
        self.assertNotIn('assuming', record['args'])

    def test_cli_accepts_the_hypothesis_option(self):
        from ledger import Ledger

        def run(*argv):
            output = io.StringIO()
            with redirect_stdout(output):
                toymath_cli.main(list(argv))
            return json.loads(output.getvalue())

        path = os.path.join(tempfile.mkdtemp(), 'cases.json')
        rec = run('apply', 'a x \\lt b', '/', 'a',
                  '--assuming', 'a < 0', '--session', path)
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertIn('\\gt', rec['result'])
        self.assertEqual([a['text'] for a in rec['assumptions']],
                         ['a \\lt 0'])
        self.assertEqual(Ledger(path).replay()['status'], 'verified')
        verdict = run('equal', '\\sqrt{x^2}', 'x', '--assuming', 'x > 0')
        self.assertEqual(verdict['verdict'], 'yes')
        self.assertIn('under the stated assumptions', verdict['method'])


if __name__ == '__main__':
    unittest.main()
